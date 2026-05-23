import FXDataEngine
import Foundation

public struct TreeCatboostCPUModel: Sendable {
    private static let classCount = 3
    private static let maxTrees = 320
    private static let maxDepth = 6
    private static let maxLeaves = 1 << maxDepth
    private static let bins = 64
    private static let maxBuffer = 6_144
    private static let minBuffer = 256
    private static let buildEvery = 48
    private static let minData = 12
    private static let minChildHessian = 0.08
    private static let gamma = 0.015
    private static let ctrBase = 10
    private static let ctrFeaturesPerBase = 3
    private static let ctrPairCount = 6
    private static let ctrPairHash = 257
    private static let ctrFeatures = ctrBase * ctrFeaturesPerBase + ctrPairCount
    private static let extWeights = FXDataEngineConstants.aiWeights + ctrFeatures
    private static let leafNewtonSteps = 5

    private struct LevelSplit: Sendable {
        var feature = -1
        var threshold = 0.0
        var defaultLeft = true
    }

    private struct Tree: Sendable {
        var depth = 0
        var levels = Array(repeating: LevelSplit(), count: TreeCatboostCPUModel.maxDepth)
        var leafValues = Array(repeating: Array(repeating: 0.0, count: TreeCatboostCPUModel.classCount), count: TreeCatboostCPUModel.maxLeaves)
        var leafMoveMean = Array(repeating: 0.0, count: TreeCatboostCPUModel.maxLeaves)
        var leafCount = Array(repeating: 0, count: TreeCatboostCPUModel.maxLeaves)
    }

    private struct Sample: Sendable {
        var x: [Double]
        var label: LabelClass
        var movePoints: Double
        var sampleWeight: Double
    }

    private var steps: Int
    private var bias: [Double]
    private var trees: [Tree]
    private var buffer: [Sample]
    private var classEMA: [Double]
    private var calibrator: PluginTernaryCalibrator
    private var moveHead: PluginMoveHead
    private var qualityBank: PluginQualityBank
    private var moveEMAAbs: Double
    private var moveReady: Bool
    private var ctrReady: Bool
    private var ctrGlobalClass: [Double]
    private var ctrBinTotal: [[Double]]
    private var ctrBinClass: [[[Double]]]
    private var ctrPairTotal: [[Double]]
    private var ctrPairBuy: [[Double]]
    private var baseBorders: [[Double]]
    private var featureUse: [Int]
    private var lossReady: Bool
    private var lossFast: Double
    private var lossSlow: Double
    private var driftCooldown: Int
    private var qualityAlarm: Int

    public init() {
        self.steps = 0
        self.bias = [0.0, 0.0, 0.10]
        self.trees = []
        self.buffer = []
        self.classEMA = Array(repeating: 1.0, count: Self.classCount)
        self.calibrator = PluginTernaryCalibrator()
        self.moveHead = PluginMoveHead()
        self.qualityBank = PluginQualityBank()
        self.moveEMAAbs = 0.0
        self.moveReady = false
        self.ctrReady = false
        self.ctrGlobalClass = Array(repeating: 1.0 / 3.0, count: Self.classCount)
        self.ctrBinTotal = Self.makeMatrix(Self.ctrBase, Self.bins)
        self.ctrBinClass = Self.makeCube(Self.ctrBase, Self.bins, Self.classCount)
        self.ctrPairTotal = Self.makeMatrix(Self.ctrPairCount, Self.ctrPairHash)
        self.ctrPairBuy = Self.makeMatrix(Self.ctrPairCount, Self.ctrPairHash)
        self.baseBorders = Array(repeating: [], count: FXDataEngineConstants.aiWeights)
        self.featureUse = Array(repeating: 0, count: Self.extWeights)
        self.lossReady = false
        self.lossFast = 0.0
        self.lossSlow = 0.0
        self.driftCooldown = 0
        self.qualityAlarm = 0
    }

    public mutating func reset() {
        self = TreeCatboostCPUModel()
    }

    public mutating func train(_ request: TrainRequestV4, hyperParameters: HyperParameters) {
        steps += 1
        let x = Self.preparedFeatures(request.x, dataHasVolume: request.context.dataHasVolume)
        let label = PluginContextRuntimeTools.normalizeClassLabel(
            rawLabel: request.labelClass.rawValue,
            x: x,
            movePoints: request.movePoints,
            priceCostPoints: request.context.priceCostPoints
        )

        for classIndex in 0..<Self.classCount {
            classEMA[classIndex] = 0.997 * classEMA[classIndex] + (classIndex == label.rawValue ? 0.003 : 0.0)
        }
        let classMean = classEMA.reduce(0.0, +) / Double(Self.classCount)
        let classBalance = fxClamp(classMean / max(classEMA[label.rawValue], 0.005), 0.60, 2.50)
        let scaledHyperParameters = PluginSupportTools.scaleHyperParametersForMove(
            hyperParameters,
            movePoints: request.movePoints
        )
        let priceCost = max(0.0, fxSafeFinite(request.context.priceCostPoints))
        let edge = max(0.0, abs(request.movePoints) - priceCost)
        var eventWeight = fxClamp(0.35 + edge / max(priceCost, 0.50), 0.10, 6.00)
        if label == .skip {
            eventWeight *= 0.85
        }
        let sampleWeight = fxClamp(request.sampleWeight * eventWeight * classBalance, 0.10, 6.00)

        let raw = rawProbabilities(extendedFeaturesForInference(x))
        let calibrated = calibrator.calibrated(raw)
        let ce = -log(fxClamp(calibrated[label.rawValue], 1.0e-6, 1.0))
        updateLossDrift(ce)
        let calibratorLearningRate = fxClamp(0.01 + 0.12 * fxClamp(scaledHyperParameters.xgbLearningRate, 0.0005, 0.3000), 0.0005, 0.0300)
        calibrator.update(
            rawProbabilities: raw,
            labelClass: label,
            sampleWeight: sampleWeight,
            learningRate: calibratorLearningRate
        )

        pushSample(Sample(x: x, label: label, movePoints: request.movePoints, sampleWeight: sampleWeight))
        qualityBank.update(request: request, sampleWeight: sampleWeight)
        updateMoveEMA(request.movePoints)
        moveHead.update(
            x: x,
            movePoints: request.movePoints,
            hyperParameters: scaledHyperParameters,
            sampleWeight: sampleWeight
        )

        var buildEvery = Self.buildEvery
        if driftCooldown > 0 {
            buildEvery = Self.buildEvery / 2
        }
        if qualityAlarm > 8 {
            buildEvery = max(24, buildEvery / 2)
        }
        buildEvery = max(buildEvery, 16)
        if buffer.count >= Self.minBuffer, steps % buildEvery == 0 {
            _ = buildOneTree(hyperParameters: scaledHyperParameters)
        }
    }

    public func predict(_ request: PredictRequestV4, hyperParameters _: HyperParameters) -> PredictionV4 {
        let x = Self.preparedFeatures(request.x, dataHasVolume: request.context.dataHasVolume)
        let window = Self.preparedWindow(request.xWindow, dataHasVolume: request.context.dataHasVolume)
        let xExt = extendedFeaturesForInference(x)
        let probabilities = PluginContextRuntimeTools.normalizeClassDistribution(calibrator.calibrated(rawProbabilities(xExt)))
        let expectedMove = max(0.0, expectedMovePoints(x: x, xExt: xExt))
        let sigma = max(0.10, 0.30 * expectedMove + 0.25 * (moveReady ? moveEMAAbs : 0.0))
        let confidence = fxClamp(max(probabilities[LabelClass.buy.rawValue], probabilities[LabelClass.sell.rawValue]), 0.0, 1.0)
        let reliability = fxClamp(
            0.45 + 0.25 * (moveReady ? 1.0 : 0.0) + 0.30 * min(Double(trees.count) / 32.0, 1.0),
            0.0,
            1.0
        )
        let q25 = max(0.0, expectedMove - 0.55 * sigma)
        let q50 = max(q25, expectedMove)
        let q75 = max(q50, expectedMove + 0.55 * sigma)
        let baseOutput = PluginModelOutputV4(
            classProbabilities: probabilities,
            moveMeanPoints: expectedMove,
            moveQ25Points: q25,
            moveQ50Points: q50,
            moveQ75Points: q75,
            mfeMeanPoints: max(q75, expectedMove),
            maeMeanPoints: max(0.0, 0.35 * expectedMove),
            hitTimeFraction: fxClamp(0.62 - 0.20 * confidence + 0.12 * probabilities[LabelClass.skip.rawValue], 0.0, 1.0),
            pathRisk: fxClamp(0.32 * probabilities[LabelClass.skip.rawValue] + 0.25 * (1.0 - reliability), 0.0, 1.0),
            fillRisk: fxClamp(request.context.priceCostPoints / max(expectedMove + request.context.minMovePoints, 0.25), 0.0, 1.0),
            confidence: confidence,
            reliability: reliability,
            hasQuantiles: true,
            hasConfidence: true,
            hasPathQuality: true
        )
        let output = PluginPathQualityTools.populatedOutput(
            baseOutput,
            x: x,
            window: window,
            context: request.context,
            family: .tree,
            activityGate: 1.0 - probabilities[LabelClass.skip.rawValue],
            structuralQuality: reliability,
            qualityPriors: qualityBank.priors(context: request.context),
            declaredWindowSize: request.windowSize
        )
        return PluginContextRuntimeTools.fillPrediction(
            modelOutput: output,
            calibratedMoveMeanPoints: expectedMove,
            context: request.context
        )
    }

    private mutating func pushSample(_ sample: Sample) {
        buffer.append(sample)
        if buffer.count > Self.maxBuffer {
            buffer.removeFirst(buffer.count - Self.maxBuffer)
        }
    }

    private mutating func updateMoveEMA(_ movePoints: Double) {
        let absMove = abs(fxSafeFinite(movePoints))
        if !moveReady {
            moveEMAAbs = absMove
            moveReady = true
        } else {
            moveEMAAbs = 0.95 * moveEMAAbs + 0.05 * absMove
        }
    }

    private mutating func updateLossDrift(_ ceLoss: Double) {
        if !lossReady {
            lossFast = ceLoss
            lossSlow = ceLoss
            lossReady = true
            return
        }
        lossFast = 0.90 * lossFast + 0.10 * ceLoss
        lossSlow = 0.995 * lossSlow + 0.005 * ceLoss
        if driftCooldown > 0 {
            driftCooldown -= 1
        }
        guard steps >= 256, driftCooldown == 0 else { return }
        if lossFast > 1.70 * max(lossSlow, 0.10) {
            driftCooldown = 96
            qualityAlarm = min(16, qualityAlarm + 1)
        } else if qualityAlarm > 0 {
            qualityAlarm -= 1
        }
    }

    private func rawProbabilities(_ xExt: [Double]) -> [Double] {
        Self.softmax(modelMargins(xExt))
    }

    private func modelMargins(_ xExt: [Double]) -> [Double] {
        var margins = bias
        for tree in trees {
            let leaf = traverseLeafIndex(tree, xExt: xExt)
            guard leaf >= 0, leaf < tree.leafValues.count else { continue }
            for classIndex in 0..<Self.classCount {
                margins[classIndex] += tree.leafValues[leaf][classIndex]
            }
        }
        return margins.map { PluginSupportTools.clipSymmetric($0, limit: 35.0) }
    }

    private func traverseLeafIndex(_ tree: Tree, xExt: [Double]) -> Int {
        guard tree.depth > 0 else { return 0 }
        var leaf = 0
        for depth in 0..<tree.depth {
            let split = tree.levels[depth]
            guard split.feature > 0, split.feature < xExt.count else { break }
            let value = xExt[split.feature]
            let goLeft = value.isFinite ? value <= split.threshold : split.defaultLeft
            leaf = (leaf << 1) | (goLeft ? 0 : 1)
            if leaf < 0 || leaf >= Self.maxLeaves {
                return 0
            }
        }
        return leaf
    }

    private func expectedMovePoints(x: [Double], xExt: [Double]) -> Double {
        var sum = 0.0
        var weightSum = 0.0
        for tree in trees {
            let leaf = traverseLeafIndex(tree, xExt: xExt)
            guard leaf >= 0, leaf < tree.leafMoveMean.count else { continue }
            let move = tree.leafMoveMean[leaf]
            guard move > 0.0 else { continue }
            let confidence = abs(tree.leafValues[leaf][LabelClass.buy.rawValue] - tree.leafValues[leaf][LabelClass.sell.rawValue]) + 0.15
            sum += confidence * move
            weightSum += confidence
        }
        let treeEstimate = weightSum > 0.0 ? sum / weightSum : -1.0
        if treeEstimate > 0.0, moveReady, moveEMAAbs > 0.0 {
            return 0.70 * treeEstimate + 0.30 * moveEMAAbs
        }
        if treeEstimate > 0.0 {
            return treeEstimate
        }
        return moveHead.ready ? max(moveHead.predictRaw(x), moveEMAAbs) : (moveReady ? moveEMAAbs : 0.0)
    }

    private mutating func buildOneTree(hyperParameters: HyperParameters) -> Bool {
        let n = buffer.count
        guard n >= Self.minBuffer else { return false }
        let xAll = buffer.map(\.x)
        let yAll = buffer.map(\.label)
        let moveAll = buffer.map(\.movePoints)
        let weights = buffer.map { fxClamp(abs($0.sampleWeight), 0.05, 10.0) }

        var etaBase = fxClamp(hyperParameters.xgbLearningRate, 0.0200, 0.0500)
        var lambda = fxClamp(hyperParameters.xgbL2, 3.0000, 8.0000)
        let schedule = pow(0.985, Double(trees.count) / 16.0)
        etaBase = fxClamp(etaBase * schedule, 0.0020, 0.0500)
        if driftCooldown > 0 {
            etaBase = fxClamp(etaBase * 0.75, 0.0020, 0.0500)
            lambda = fxClamp(lambda * 1.20, 2.0, 12.0)
        }

        let extendedTraining = buildExtendedTraining(xAll: xAll, yAll: yAll)
        let xExt = extendedTraining.xExt
        let orderedBias = extendedTraining.orderedBias
        let splitCandidates = buildSplitCandidates(xExt)
        var baseMargins = Array(repeating: Array(repeating: 0.0, count: Self.classCount), count: n)
        var gradients = baseMargins
        var hessians = baseMargins

        for index in 0..<n {
            let margins = modelMargins(xExt[index])
            baseMargins[index] = margins
            let probabilities = Self.softmax((0..<Self.classCount).map { margins[$0] + orderedBias[index][$0] })
            let weight = fxClamp(weights[index], 0.0, 12.0)
            for classIndex in 0..<Self.classCount {
                let target = classIndex == yAll[index].rawValue ? 1.0 : 0.0
                gradients[index][classIndex] = (target - probabilities[classIndex]) * weight
                hessians[index][classIndex] = fxClamp(probabilities[classIndex] * (1.0 - probabilities[classIndex]) * weight, 0.001, 8.0)
            }
        }

        for classIndex in 0..<Self.classCount {
            var gradient = 0.0
            var hessian = 0.0
            for index in 0..<n {
                gradient += gradients[index][classIndex]
                hessian += hessians[index][classIndex]
            }
            if hessian > 1.0e-9 {
                bias[classIndex] += 0.15 * etaBase * PluginSupportTools.clipSymmetric(gradient / (hessian + lambda), limit: 3.0)
            }
        }
        let meanBias = bias.reduce(0.0, +) / Double(Self.classCount)
        for classIndex in 0..<Self.classCount {
            bias[classIndex] = PluginSupportTools.clipSymmetric(bias[classIndex] - meanBias, limit: 8.0)
        }

        var tree = Tree()
        var leafIndexes = Array(repeating: 0, count: n)
        var depthUsed = 0
        for depth in 0..<Self.maxDepth {
            var best: (feature: Int, threshold: Double, defaultLeft: Bool, gain: Double)?
            for feature in 1..<Self.extWeights {
                guard let borders = splitCandidates[feature], !borders.isEmpty else { continue }
                for threshold in borders {
                    for defaultLeft in [true, false] {
                        guard let gain = evaluateSplit(
                            depth: depth,
                            leafIndexes: leafIndexes,
                            xExt: xExt,
                            gradients: gradients,
                            hessians: hessians,
                            sampleWeights: weights,
                            feature: feature,
                            threshold: threshold,
                            defaultLeft: defaultLeft,
                            lambda: lambda
                        ) else { continue }
                        if gain > (best?.gain ?? 0.0) {
                            best = (feature, threshold, defaultLeft, gain)
                        }
                    }
                }
            }
            guard let split = best, split.feature > 0 else { break }
            tree.levels[depth] = LevelSplit(feature: split.feature, threshold: split.threshold, defaultLeft: split.defaultLeft)
            featureUse[split.feature] += 1
            for index in 0..<n {
                let value = xExt[index][split.feature]
                let goLeft = value.isFinite ? value <= split.threshold : split.defaultLeft
                leafIndexes[index] = (leafIndexes[index] << 1) | (goLeft ? 0 : 1)
            }
            depthUsed = depth + 1
        }

        tree.depth = depthUsed
        buildLeafValues(
            tree: &tree,
            leafIndexes: leafIndexes,
            yAll: yAll,
            moveAll: moveAll,
            sampleWeights: weights,
            baseMargins: baseMargins,
            eta: etaBase,
            lambda: lambda
        )
        applyModelShrinkage(fxClamp(0.0015 + 0.0100 * etaBase, 0.0010, 0.0100))
        if trees.count < Self.maxTrees {
            trees.append(tree)
        } else {
            trees.removeFirst()
            trees.append(tree)
        }
        return true
    }

    private func evaluateSplit(
        depth: Int,
        leafIndexes: [Int],
        xExt: [[Double]],
        gradients: [[Double]],
        hessians: [[Double]],
        sampleWeights: [Double],
        feature: Int,
        threshold: Double,
        defaultLeft: Bool,
        lambda: Double
    ) -> Double? {
        guard depth >= 0, depth < Self.maxDepth, feature > 0, feature < Self.extWeights else { return nil }
        let oldLeafCount = 1 << depth
        let newLeafCount = 1 << (depth + 1)
        var parentG = Self.makeMatrix(Self.maxLeaves, Self.classCount)
        var parentH = Self.makeMatrix(Self.maxLeaves, Self.classCount)
        var parentW = Array(repeating: 0.0, count: Self.maxLeaves)
        var childG = Self.makeMatrix(Self.maxLeaves, Self.classCount)
        var childH = Self.makeMatrix(Self.maxLeaves, Self.classCount)
        var childW = Array(repeating: 0.0, count: Self.maxLeaves)

        for index in xExt.indices {
            let weight = sampleWeights[index]
            guard weight > 0.0 else { continue }
            let oldLeaf = leafIndexes[index]
            guard oldLeaf >= 0, oldLeaf < oldLeafCount else { continue }
            parentW[oldLeaf] += weight
            for classIndex in 0..<Self.classCount {
                parentG[oldLeaf][classIndex] += gradients[index][classIndex]
                parentH[oldLeaf][classIndex] += hessians[index][classIndex]
            }
            let value = xExt[index][feature]
            let goLeft = value.isFinite ? value <= threshold : defaultLeft
            let childLeaf = (oldLeaf << 1) | (goLeft ? 0 : 1)
            guard childLeaf >= 0, childLeaf < newLeafCount else { continue }
            childW[childLeaf] += weight
            for classIndex in 0..<Self.classCount {
                childG[childLeaf][classIndex] += gradients[index][classIndex]
                childH[childLeaf][classIndex] += hessians[index][classIndex]
            }
        }

        var gain = 0.0
        for oldLeaf in 0..<oldLeafCount {
            let leftLeaf = oldLeaf << 1
            let rightLeaf = leftLeaf + 1
            guard rightLeaf < newLeafCount else { return nil }
            if parentW[oldLeaf] < Double(Self.minData) {
                continue
            }
            guard childW[leftLeaf] >= Double(Self.minData),
                  childW[rightLeaf] >= Double(Self.minData) else {
                return nil
            }
            let leftHessian = childH[leftLeaf].reduce(0.0, +)
            let rightHessian = childH[rightLeaf].reduce(0.0, +)
            guard leftHessian >= Self.minChildHessian, rightHessian >= Self.minChildHessian else {
                return nil
            }
            let parentScore = coupledNewtonScore(gradients: parentG[oldLeaf], hessians: parentH[oldLeaf], lambda: lambda)
            let leftScore = coupledNewtonScore(gradients: childG[leftLeaf], hessians: childH[leftLeaf], lambda: lambda)
            let rightScore = coupledNewtonScore(gradients: childG[rightLeaf], hessians: childH[rightLeaf], lambda: lambda)
            gain += 0.5 * (leftScore + rightScore - parentScore)
        }
        gain -= Self.gamma
        gain -= 0.005 * Double(depth + 1)
        gain -= 0.002 * Double(featureUse[feature])
        return gain > 1.0e-7 ? gain : nil
    }

    private func coupledNewtonScore(gradients: [Double], hessians: [Double], lambda: Double) -> Double {
        var score = 0.0
        var gradientSum = 0.0
        var hessianSum = 0.0
        for classIndex in 0..<Self.classCount {
            let gradient = gradients[classIndex]
            let hessian = hessians[classIndex]
            score += gradient * gradient / (hessian + lambda + 1.0e-9)
            gradientSum += gradient
            hessianSum += hessian
        }
        score -= 0.25 * gradientSum * gradientSum / (hessianSum + 3.0 * lambda + 1.0e-9)
        return score
    }

    private func buildLeafValues(
        tree: inout Tree,
        leafIndexes: [Int],
        yAll: [LabelClass],
        moveAll: [Double],
        sampleWeights: [Double],
        baseMargins: [[Double]],
        eta: Double,
        lambda: Double
    ) {
        let leafCount = min(max(tree.depth > 0 ? (1 << tree.depth) : 1, 1), Self.maxLeaves)
        tree.leafCount = Array(repeating: 0, count: Self.maxLeaves)
        tree.leafMoveMean = Array(repeating: 0.0, count: Self.maxLeaves)
        tree.leafValues = Array(repeating: Array(repeating: 0.0, count: Self.classCount), count: Self.maxLeaves)
        var moveSum = Array(repeating: 0.0, count: Self.maxLeaves)
        for index in leafIndexes.indices {
            let leaf = min(max(tree.depth > 0 ? leafIndexes[index] : 0, 0), leafCount - 1)
            tree.leafCount[leaf] += 1
            moveSum[leaf] += abs(moveAll[index])
        }
        for leaf in 0..<leafCount where tree.leafCount[leaf] > 0 {
            tree.leafMoveMean[leaf] = moveSum[leaf] / Double(tree.leafCount[leaf])
        }

        for _ in 0..<Self.leafNewtonSteps {
            var leafG = Self.makeMatrix(Self.maxLeaves, Self.classCount)
            var leafH = Self.makeMatrix(Self.maxLeaves, Self.classCount)
            for index in leafIndexes.indices {
                let weight = sampleWeights[index]
                guard weight > 0.0 else { continue }
                let leaf = min(max(tree.depth > 0 ? leafIndexes[index] : 0, 0), leafCount - 1)
                var logits = baseMargins[index]
                for classIndex in 0..<Self.classCount {
                    logits[classIndex] += tree.leafValues[leaf][classIndex]
                }
                let probabilities = Self.softmax(logits)
                for classIndex in 0..<Self.classCount {
                    let target = classIndex == yAll[index].rawValue ? 1.0 : 0.0
                    leafG[leaf][classIndex] += (target - probabilities[classIndex]) * weight
                    leafH[leaf][classIndex] += fxClamp(probabilities[classIndex] * (1.0 - probabilities[classIndex]) * weight, 0.005, 8.0)
                }
            }

            for leaf in 0..<leafCount where tree.leafCount[leaf] > 0 {
                var updated = tree.leafValues[leaf]
                var mean = 0.0
                for classIndex in 0..<Self.classCount {
                    let step = eta * PluginSupportTools.clipSymmetric(
                        leafG[leaf][classIndex] / (leafH[leaf][classIndex] + lambda),
                        limit: 5.0
                    )
                    updated[classIndex] += step
                    mean += updated[classIndex]
                }
                mean /= Double(Self.classCount)
                for classIndex in 0..<Self.classCount {
                    tree.leafValues[leaf][classIndex] = PluginSupportTools.clipSymmetric(updated[classIndex] - mean, limit: 8.0)
                }
            }
        }
    }

    private mutating func applyModelShrinkage(_ shrink: Double) {
        let scale = fxClamp(1.0 - shrink, 0.90, 1.00)
        for classIndex in 0..<Self.classCount {
            bias[classIndex] *= scale
        }
        for treeIndex in trees.indices {
            let leafCount = min(max(trees[treeIndex].depth > 0 ? (1 << trees[treeIndex].depth) : 1, 1), Self.maxLeaves)
            for leaf in 0..<leafCount {
                for classIndex in 0..<Self.classCount {
                    trees[treeIndex].leafValues[leaf][classIndex] *= scale
                }
            }
        }
    }

    private mutating func buildExtendedTraining(xAll: [[Double]], yAll: [LabelClass]) -> (xExt: [[Double]], orderedBias: [[Double]]) {
        buildBaseBorders(xAll)
        let qbins = xAll.map { x in Self.ctrBaseFeatures.map { quantizeBaseFeature(feature: $0, value: x[$0]) } }
        buildGlobalCTRStats(yAll: yAll, qbins: qbins)
        let ordered = buildOrderedCTRFeatures(yAll: yAll, qbins: qbins)
        var xExt = Array(repeating: Array(repeating: 0.0, count: Self.extWeights), count: xAll.count)
        for index in xAll.indices {
            for feature in 0..<FXDataEngineConstants.aiWeights {
                xExt[index][feature] = xAll[index][feature]
            }
            for ctrIndex in 0..<Self.ctrFeatures {
                xExt[index][FXDataEngineConstants.aiWeights + ctrIndex] = ordered.ctr[index][ctrIndex]
            }
        }
        return (xExt, ordered.bias)
    }

    private mutating func buildBaseBorders(_ xAll: [[Double]]) {
        for feature in Self.ctrBaseFeatures {
            var values = xAll.compactMap { row -> Double? in
                let value = feature < row.count ? row[feature] : 0.0
                return value.isFinite ? value : nil
            }.sorted()
            guard values.count >= 2 else {
                baseBorders[feature] = []
                continue
            }
            let desired = min(Self.bins - 1, values.count - 1)
            var borders: [Double] = []
            var previous = -Double.greatestFiniteMagnitude
            for bin in 1...desired {
                let quantileIndex = min(max(Int(floor(Double(bin) / Double(desired + 1) * Double(values.count - 1))), 0), values.count - 1)
                let threshold = values[quantileIndex]
                if borders.isEmpty || abs(threshold - previous) > 1.0e-12 {
                    borders.append(threshold)
                    previous = threshold
                }
            }
            values.removeAll(keepingCapacity: false)
            baseBorders[feature] = borders
        }
    }

    private func quantizeBaseFeature(feature: Int, value: Double) -> Int {
        guard feature > 0, feature < FXDataEngineConstants.aiWeights, value.isFinite else { return -1 }
        let borders = baseBorders[feature]
        guard !borders.isEmpty else { return 0 }
        var low = 0
        var high = borders.count - 1
        var position = borders.count
        while low <= high {
            let mid = (low + high) >> 1
            if value <= borders[mid] {
                position = mid
                high = mid - 1
            } else {
                low = mid + 1
            }
        }
        return min(max(position, 0), Self.bins - 1)
    }

    private mutating func buildGlobalCTRStats(yAll: [LabelClass], qbins: [[Int]]) {
        ctrReady = false
        ctrBinTotal = Self.makeMatrix(Self.ctrBase, Self.bins)
        ctrBinClass = Self.makeCube(Self.ctrBase, Self.bins, Self.classCount)
        ctrPairTotal = Self.makeMatrix(Self.ctrPairCount, Self.ctrPairHash)
        ctrPairBuy = Self.makeMatrix(Self.ctrPairCount, Self.ctrPairHash)
        var classCounts = Array(repeating: 0.0, count: Self.classCount)
        for label in yAll {
            classCounts[label.rawValue] += 1.0
        }
        let total = max(classCounts.reduce(0.0, +), 1.0)
        ctrGlobalClass = classCounts.map { fxClamp($0 / total, 0.001, 0.998) }

        for index in yAll.indices {
            let label = yAll[index]
            for base in 0..<Self.ctrBase {
                let bin = qbins[index][base]
                guard bin >= 0, bin < Self.bins else { continue }
                ctrBinTotal[base][bin] += 1.0
                ctrBinClass[base][bin][label.rawValue] += 1.0
            }
            for pair in 0..<Self.ctrPairCount {
                let first = qbins[index][Self.ctrPairA[pair]]
                let second = qbins[index][Self.ctrPairB[pair]]
                guard first >= 0, second >= 0 else { continue }
                let hash = pairHash(first, second, pair)
                ctrPairTotal[pair][hash] += 1.0
                ctrPairBuy[pair][hash] += label == .buy ? 1.0 : (label == .skip ? 0.5 : 0.0)
            }
        }
        ctrReady = true
    }

    private func buildOrderedCTRFeatures(yAll: [LabelClass], qbins: [[Int]]) -> (ctr: [[Double]], bias: [[Double]]) {
        var ctr = Array(repeating: Array(repeating: 0.0, count: Self.ctrFeatures), count: yAll.count)
        var orderedBias = Array(repeating: Array(repeating: 0.0, count: Self.classCount), count: yAll.count)
        let globalPrior = ctrGlobalClass
        let globalDirection = globalPrior[LabelClass.buy.rawValue] + 0.5 * globalPrior[LabelClass.skip.rawValue]
        let prior = 2.0
        var classPrefix = Array(repeating: 0.0, count: Self.classCount)
        var baseCount = Self.makeMatrix(Self.ctrBase, Self.bins)
        var baseClass = Self.makeCube(Self.ctrBase, Self.bins, Self.classCount)
        var pairCount = Self.makeMatrix(Self.ctrPairCount, Self.ctrPairHash)
        var pairBuy = Self.makeMatrix(Self.ctrPairCount, Self.ctrPairHash)

        for index in yAll.indices {
            let prefixTotal = classPrefix.reduce(0.0, +)
            for classIndex in 0..<Self.classCount {
                let probability = fxClamp((classPrefix[classIndex] + prior * globalPrior[classIndex]) / (prefixTotal + prior), 0.001, 0.999)
                orderedBias[index][classIndex] = log(probability)
            }
            let meanBias = orderedBias[index].reduce(0.0, +) / Double(Self.classCount)
            for classIndex in 0..<Self.classCount {
                orderedBias[index][classIndex] -= meanBias
            }

            var offset = 0
            for base in 0..<Self.ctrBase {
                let bin = qbins[index][base]
                if bin < 0 || bin >= Self.bins {
                    for classIndex in 0..<Self.classCount {
                        ctr[index][offset + classIndex] = 2.0 * globalPrior[classIndex] - 1.0
                    }
                } else {
                    let total = baseCount[base][bin]
                    for classIndex in 0..<Self.classCount {
                        let probability = (baseClass[base][bin][classIndex] + prior * globalPrior[classIndex]) / (total + prior)
                        ctr[index][offset + classIndex] = fxClamp(2.0 * probability - 1.0, -1.0, 1.0)
                    }
                }
                offset += Self.ctrFeaturesPerBase
            }
            for pair in 0..<Self.ctrPairCount {
                let first = qbins[index][Self.ctrPairA[pair]]
                let second = qbins[index][Self.ctrPairB[pair]]
                var probability = globalDirection
                if first >= 0, second >= 0 {
                    let hash = pairHash(first, second, pair)
                    probability = (pairBuy[pair][hash] + prior * globalDirection) / (pairCount[pair][hash] + prior)
                }
                ctr[index][offset + pair] = fxClamp(2.0 * probability - 1.0, -1.0, 1.0)
            }

            let label = yAll[index]
            classPrefix[label.rawValue] += 1.0
            for base in 0..<Self.ctrBase {
                let bin = qbins[index][base]
                guard bin >= 0, bin < Self.bins else { continue }
                baseCount[base][bin] += 1.0
                baseClass[base][bin][label.rawValue] += 1.0
            }
            for pair in 0..<Self.ctrPairCount {
                let first = qbins[index][Self.ctrPairA[pair]]
                let second = qbins[index][Self.ctrPairB[pair]]
                guard first >= 0, second >= 0 else { continue }
                let hash = pairHash(first, second, pair)
                pairCount[pair][hash] += 1.0
                pairBuy[pair][hash] += label == .buy ? 1.0 : (label == .skip ? 0.5 : 0.0)
            }
        }
        return (ctr, orderedBias)
    }

    private func extendedFeaturesForInference(_ x: [Double]) -> [Double] {
        var output = Array(repeating: 0.0, count: Self.extWeights)
        for feature in 0..<FXDataEngineConstants.aiWeights {
            output[feature] = feature < x.count ? x[feature] : 0.0
        }
        guard ctrReady else { return output }
        let qbins = Self.ctrBaseFeatures.map { quantizeBaseFeature(feature: $0, value: x[$0]) }
        let prior = 2.0
        var offset = FXDataEngineConstants.aiWeights
        for base in 0..<Self.ctrBase {
            let bin = qbins[base]
            if bin < 0 || bin >= Self.bins {
                for classIndex in 0..<Self.classCount {
                    output[offset + classIndex] = 2.0 * ctrGlobalClass[classIndex] - 1.0
                }
            } else {
                let total = ctrBinTotal[base][bin]
                for classIndex in 0..<Self.classCount {
                    let probability = (ctrBinClass[base][bin][classIndex] + prior * ctrGlobalClass[classIndex]) / (total + prior)
                    output[offset + classIndex] = fxClamp(2.0 * probability - 1.0, -1.0, 1.0)
                }
            }
            offset += Self.ctrFeaturesPerBase
        }
        let globalDirection = ctrGlobalClass[LabelClass.buy.rawValue] + 0.5 * ctrGlobalClass[LabelClass.skip.rawValue]
        for pair in 0..<Self.ctrPairCount {
            let first = qbins[Self.ctrPairA[pair]]
            let second = qbins[Self.ctrPairB[pair]]
            var probability = globalDirection
            if first >= 0, second >= 0 {
                let hash = pairHash(first, second, pair)
                probability = (ctrPairBuy[pair][hash] + prior * globalDirection) / (ctrPairTotal[pair][hash] + prior)
            }
            output[offset + pair] = fxClamp(2.0 * probability - 1.0, -1.0, 1.0)
        }
        return output
    }

    private func buildSplitCandidates(_ xExt: [[Double]]) -> [Int: [Double]] {
        var output: [Int: [Double]] = [:]
        for feature in 1..<Self.extWeights {
            var values = xExt.compactMap { row -> Double? in
                let value = row[feature]
                return value.isFinite ? value : nil
            }.sorted()
            guard values.count >= 2 * Self.minData else { continue }
            let desired = min(Self.bins - 1, values.count - 1)
            var borders: [Double] = []
            var previous = -Double.greatestFiniteMagnitude
            for bin in 1...desired {
                let quantileIndex = min(max(Int(floor(Double(bin) / Double(desired + 1) * Double(values.count - 1))), 0), values.count - 1)
                let threshold = values[quantileIndex]
                if borders.isEmpty || abs(threshold - previous) > 1.0e-12 {
                    borders.append(threshold)
                    previous = threshold
                }
            }
            values.removeAll(keepingCapacity: false)
            if !borders.isEmpty {
                output[feature] = borders
            }
        }
        return output
    }

    private func pairHash(_ first: Int, _ second: Int, _ pair: Int) -> Int {
        var hash = (max(first, 0) * 131 + max(second, 0) * 17 + pair * 53 + 97) % Self.ctrPairHash
        if hash < 0 {
            hash += Self.ctrPairHash
        }
        return hash
    }

    private static let ctrBaseFeatures = (0..<ctrBase).map { min(max(1 + $0, 1), FXDataEngineConstants.aiWeights - 1) }
    private static let ctrPairA = [0, 0, 1, 2, 5, 7]
    private static let ctrPairB = [1, 2, 3, 4, 6, 8]
    private static let volumeFeatureIndexes = [6, 68, 69, 70, 71, 74, 75, 76, 77, 78, 80, 81, 82, 83]

    private static func preparedFeatures(_ x: [Double], dataHasVolume: Bool) -> [Double] {
        var output = Array(repeating: 0.0, count: FXDataEngineConstants.aiWeights)
        for index in 0..<FXDataEngineConstants.aiWeights {
            output[index] = fxClamp(index < x.count ? fxSafeFinite(x[index]) : 0.0, -50.0, 50.0)
        }
        if !dataHasVolume {
            for index in volumeFeatureIndexes where index < output.count {
                output[index] = 0.0
            }
        }
        return output
    }

    private static func preparedWindow(_ window: [[Double]], dataHasVolume: Bool) -> [[Double]] {
        window.map { preparedFeatures($0, dataHasVolume: dataHasVolume) }
    }

    private static func softmax(_ logits: [Double]) -> [Double] {
        let values = (0..<classCount).map { index -> Double in
            index < logits.count ? fxSafeFinite(logits[index]) : 0.0
        }
        let maximum = values.max() ?? 0.0
        let exponentials = values.map { exp(PluginSupportTools.clipSymmetric($0 - maximum, limit: 30.0)) }
        let sum = exponentials.reduce(0.0, +)
        guard sum > 0.0 else {
            return [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]
        }
        return exponentials.map { $0 / sum }
    }

    private static func makeMatrix(_ rows: Int, _ columns: Int) -> [[Double]] {
        Array(repeating: Array(repeating: 0.0, count: columns), count: rows)
    }

    private static func makeCube(_ rows: Int, _ columns: Int, _ depth: Int) -> [[[Double]]] {
        Array(repeating: Array(repeating: Array(repeating: 0.0, count: depth), count: columns), count: rows)
    }
}
