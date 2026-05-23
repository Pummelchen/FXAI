import FXDataEngine
import Foundation

public struct TreeLgbmCPUModel: Sendable {
    private static let classCount = 3
    private static let bins = 80
    private static let maxLeaves = 63
    private static let maxDepth = 10
    private static let maxNodes = 125
    private static let maxTrees = 192
    private static let maxBuffer = 4_096
    private static let minData = 20
    private static let minChildHessian = 0.20
    private static let gamma = 0.02
    private static let buildEvery = 16
    private static let minBuffer = 256
    private static let eceBins = 12
    private static let gossBins = 64

    private struct Node: Sendable {
        var isLeaf = true
        var feature = -1
        var threshold = 0.0
        var defaultLeft = true
        var left = -1
        var right = -1
        var depth = 0
        var leafValue = 0.0
        var moveMean = 0.0
        var moveVariance = 0.0
        var moveQ10 = 0.0
        var moveQ50 = 0.0
        var moveQ90 = 0.0
        var sampleCount = 0
    }

    private struct Tree: Sendable {
        var nodes: [Node] = [Node()]
    }

    private struct Sample: Sendable {
        var x: [Double]
        var label: LabelClass
        var movePoints: Double
        var priceCostPoints: Double
        var sampleWeight: Double
    }

    private struct Candidate: Sendable {
        var sampleIndex: Int
        var gradient: Double
        var hessian: Double
        var absGradient: Double
    }

    private var steps: Int
    private var bias: [Double]
    private var trees: [[Tree]]
    private var buffer: [Sample]
    private var calibrator: PluginTernaryCalibrator
    private var binaryCalibrator: PluginBinaryCalibrator
    private var moveHead: PluginMoveHead
    private var qualityBank: PluginQualityBank
    private var moveEMAAbs: Double
    private var moveReady: Bool
    private var validationReady: Bool
    private var validationSteps: Int
    private var nllFast: Double
    private var nllSlow: Double
    private var brierFast: Double
    private var brierSlow: Double
    private var eceFast: Double
    private var eceSlow: Double
    private var evFast: Double
    private var evSlow: Double
    private var eceMass: [Double]
    private var eceAccuracy: [Double]
    private var eceConfidence: [Double]
    private var qualityDegraded: Bool

    public init() {
        self.steps = 0
        self.bias = Array(repeating: 0.0, count: Self.classCount)
        self.trees = Array(repeating: [], count: Self.classCount)
        self.buffer = []
        self.calibrator = PluginTernaryCalibrator()
        self.binaryCalibrator = PluginBinaryCalibrator()
        self.moveHead = PluginMoveHead()
        self.qualityBank = PluginQualityBank()
        self.moveEMAAbs = 0.0
        self.moveReady = false
        self.validationReady = false
        self.validationSteps = 0
        self.nllFast = 0.0
        self.nllSlow = 0.0
        self.brierFast = 0.0
        self.brierSlow = 0.0
        self.eceFast = 0.0
        self.eceSlow = 0.0
        self.evFast = 0.0
        self.evSlow = 0.0
        self.eceMass = Array(repeating: 0.0, count: Self.eceBins)
        self.eceAccuracy = Array(repeating: 0.0, count: Self.eceBins)
        self.eceConfidence = Array(repeating: 0.0, count: Self.eceBins)
        self.qualityDegraded = false
    }

    public mutating func reset() {
        self = TreeLgbmCPUModel()
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
        let scaledHyperParameters = PluginSupportTools.scaleHyperParametersForMove(
            hyperParameters,
            movePoints: request.movePoints
        )
        let sampleWeight = fxClamp(
            request.sampleWeight *
                PluginSupportTools.moveSampleWeight(
                    x: x,
                    movePoints: request.movePoints,
                    priceCostPoints: request.context.priceCostPoints,
                    minMovePoints: request.context.minMovePoints,
                    qualityTargets: PluginQualityTargets(request: request)
                ),
            0.10,
            6.00
        )
        let priceCost = max(0.0, fxSafeFinite(request.context.priceCostPoints))

        let raw = rawClassProbabilities(x)
        let calibrated = PluginContextRuntimeTools.normalizeClassDistribution(calibrator.calibrated(raw))
        let buyMove = classExpectedMove(classIndex: LabelClass.buy.rawValue, x: x)
        let sellMove = classExpectedMove(classIndex: LabelClass.sell.rawValue, x: x)
        let evNow = calibrated[LabelClass.buy.rawValue] * buyMove + calibrated[LabelClass.sell.rawValue] * sellMove
        updateValidationMetrics(label: label, probabilities: calibrated, expectedMovePoints: evNow, costPoints: priceCost)

        var calibratorLearningRate = fxClamp(0.30 * scaledHyperParameters.xgbLearningRate, 0.0002, 0.0200)
        if qualityDegraded {
            calibratorLearningRate *= 0.80
        }
        calibrator.update(
            rawProbabilities: raw,
            labelClass: label,
            sampleWeight: sampleWeight,
            learningRate: calibratorLearningRate
        )
        let directionalDenominator = max(raw[LabelClass.buy.rawValue] + raw[LabelClass.sell.rawValue], 1.0e-9)
        let rawDirectionalBuy = raw[LabelClass.buy.rawValue] / directionalDenominator
        if label == .buy {
            binaryCalibrator.update(rawProbability: rawDirectionalBuy, target: true, sampleWeight: sampleWeight)
        } else if label == .sell {
            binaryCalibrator.update(rawProbability: rawDirectionalBuy, target: false, sampleWeight: sampleWeight)
        }

        pushSample(
            Sample(
                x: x,
                label: label,
                movePoints: request.movePoints,
                priceCostPoints: priceCost,
                sampleWeight: sampleWeight
            )
        )
        qualityBank.update(request: request, sampleWeight: sampleWeight)
        updateMoveEMA(request.movePoints)
        moveHead.update(
            x: x,
            movePoints: request.movePoints,
            hyperParameters: scaledHyperParameters,
            sampleWeight: sampleWeight
        )

        if buffer.count >= Self.minBuffer, steps % Self.buildEvery == 0 {
            _ = buildOneTreeClass(classIndex: LabelClass.sell.rawValue, hyperParameters: scaledHyperParameters)
            _ = buildOneTreeClass(classIndex: LabelClass.buy.rawValue, hyperParameters: scaledHyperParameters)
            _ = buildOneTreeClass(classIndex: LabelClass.skip.rawValue, hyperParameters: scaledHyperParameters)
        }
    }

    public func predict(_ request: PredictRequestV4, hyperParameters _: HyperParameters) -> PredictionV4 {
        let x = Self.preparedFeatures(request.x, dataHasVolume: request.context.dataHasVolume)
        let window = Self.preparedWindow(request.xWindow, dataHasVolume: request.context.dataHasVolume)
        let probabilities = PluginContextRuntimeTools.normalizeClassDistribution(calibrator.calibrated(rawClassProbabilities(x)))

        let buyStats = classMoveStats(classIndex: LabelClass.buy.rawValue, x: x)
        let sellStats = classMoveStats(classIndex: LabelClass.sell.rawValue, x: x)
        let evBuy = buyStats.mean > 0.0 ? buyStats.mean : classExpectedMove(classIndex: LabelClass.buy.rawValue, x: x)
        let evSell = sellStats.mean > 0.0 ? sellStats.mean : classExpectedMove(classIndex: LabelClass.sell.rawValue, x: x)
        var expectedMove = probabilities[LabelClass.buy.rawValue] * evBuy + probabilities[LabelClass.sell.rawValue] * evSell
        expectedMove = max(0.0, expectedMove - 0.35 * max(0.0, request.context.priceCostPoints))
        if expectedMove > 0.0, moveReady, moveEMAAbs > 0.0 {
            expectedMove = 0.75 * expectedMove + 0.25 * moveEMAAbs
        } else if expectedMove <= 0.0 {
            expectedMove = moveHead.ready ? max(0.0, moveHead.predictRaw(x)) : (moveReady ? moveEMAAbs : 0.0)
        }

        let mixQ10 = probabilities[LabelClass.buy.rawValue] * buyStats.q10 + probabilities[LabelClass.sell.rawValue] * sellStats.q10
        let mixQ50 = probabilities[LabelClass.buy.rawValue] * buyStats.q50 + probabilities[LabelClass.sell.rawValue] * sellStats.q50
        let mixQ90 = probabilities[LabelClass.buy.rawValue] * buyStats.q90 + probabilities[LabelClass.sell.rawValue] * sellStats.q90
        let sigma = max(0.10, 0.25 * expectedMove + 0.20 * (moveReady ? moveEMAAbs : 0.0))
        let confidence = fxClamp(max(probabilities[LabelClass.buy.rawValue], probabilities[LabelClass.sell.rawValue]), 0.0, 1.0)
        let supportReliability = fxClamp((buyStats.support + sellStats.support) / 240.0, 0.0, 1.0)
        let reliability = fxClamp(
            0.40 +
                0.20 * (moveReady ? 1.0 : 0.0) +
                0.20 * min(Double(trees[LabelClass.buy.rawValue].count) / 32.0, 1.0) +
                0.20 * supportReliability,
            0.0,
            1.0
        )

        let q25 = max(0.0, mixQ10 > 0.0 ? mixQ10 : expectedMove - 0.55 * sigma)
        let q50 = max(q25, mixQ50 > 0.0 ? mixQ50 : expectedMove)
        let q75 = max(q50, mixQ90 > 0.0 ? mixQ90 : expectedMove + 0.55 * sigma)
        let baseOutput = PluginModelOutputV4(
            classProbabilities: probabilities,
            moveMeanPoints: max(0.0, expectedMove),
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
            calibratedMoveMeanPoints: max(0.0, expectedMove),
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
        let value = abs(fxSafeFinite(movePoints))
        if moveReady {
            moveEMAAbs = 0.95 * moveEMAAbs + 0.05 * value
        } else {
            moveEMAAbs = value
            moveReady = true
        }
    }

    private func rawClassProbabilities(_ x: [Double]) -> [Double] {
        Self.softmax(modelRawLogits(x))
    }

    private func modelRawLogits(_ x: [Double]) -> [Double] {
        (0..<Self.classCount).map { classIndex in
            classMargin(classIndex: classIndex, x: x)
        }
    }

    private func modelRawLogitsDroppingClass(
        _ x: [Double],
        dropClass: Int,
        keepMask: [Bool],
        keepScale: Double
    ) -> [Double] {
        (0..<Self.classCount).map { classIndex in
            var value = bias[classIndex]
            for treeIndex in trees[classIndex].indices {
                if classIndex == dropClass, treeIndex < keepMask.count, !keepMask[treeIndex] {
                    continue
                }
                let scale = classIndex == dropClass ? keepScale : 1.0
                value += scale * treeOutput(trees[classIndex][treeIndex], x: x)
            }
            return PluginSupportTools.clipSymmetric(value, limit: 35.0)
        }
    }

    private func classMargin(classIndex: Int, x: [Double]) -> Double {
        var value = bias[classIndex]
        for tree in trees[classIndex] {
            value += treeOutput(tree, x: x)
        }
        return PluginSupportTools.clipSymmetric(value, limit: 35.0)
    }

    private func treeOutput(_ tree: Tree, x: [Double]) -> Double {
        let leaf = traverseLeafIndex(tree, x: x)
        guard leaf >= 0, leaf < tree.nodes.count else { return 0.0 }
        return tree.nodes[leaf].leafValue
    }

    private func traverseLeafIndex(_ tree: Tree, x: [Double]) -> Int {
        var nodeIndex = 0
        var guardCount = 0
        while nodeIndex >= 0, nodeIndex < tree.nodes.count, guardCount < Self.maxNodes {
            let node = tree.nodes[nodeIndex]
            if node.isLeaf {
                return nodeIndex
            }
            let feature = node.feature
            guard feature > 0, feature < x.count else { return nodeIndex }
            let value = x[feature]
            let goLeft = value.isFinite ? value <= node.threshold : node.defaultLeft
            let next = goLeft ? node.left : node.right
            guard next >= 0, next < tree.nodes.count else { return nodeIndex }
            nodeIndex = next
            guardCount += 1
        }
        return 0
    }

    private mutating func buildOneTreeClass(classIndex: Int, hyperParameters: HyperParameters) -> Bool {
        let n = buffer.count
        guard n >= Self.minBuffer, classIndex >= 0, classIndex < Self.classCount else { return false }

        var classCounts = Array(repeating: 0, count: Self.classCount)
        for sample in buffer {
            classCounts[sample.label.rawValue] += 1
        }
        let meanCount = max(1.0, Double(classCounts.reduce(0, +)) / Double(Self.classCount))
        let keepMask = dartKeepMask(classIndex: classIndex)
        let keepScale = 1.0 / 0.95
        var candidates: [Candidate] = []
        candidates.reserveCapacity(n)

        for index in 0..<n {
            guard deterministicFraction(a: index, b: classIndex, salt: 17) <= 0.80 else { continue }
            let sample = buffer[index]
            let probabilities = Self.softmax(
                modelRawLogitsDroppingClass(sample.x, dropClass: classIndex, keepMask: keepMask, keepScale: keepScale)
            )
            let target = Self.targetDistribution(
                label: sample.label,
                movePoints: sample.movePoints,
                costPoints: sample.priceCostPoints
            )
            let classBalance = fxClamp(meanCount / max(1.0, Double(classCounts[sample.label.rawValue])), 0.50, 2.50)
            let decay = Self.sampleTimeDecay(age: n - 1 - index)
            let baseWeight = sample.sampleWeight * decay * classBalance
            let weight = Self.evWeight(
                label: sample.label,
                movePoints: sample.movePoints,
                costPoints: sample.priceCostPoints,
                baseWeight: baseWeight
            )
            let probability = fxClamp(probabilities[classIndex], 0.001, 0.999)
            let gradient = (target[classIndex] - probability) * weight
            let hessian = fxClamp(probability * (1.0 - probability) * weight, 0.02, 6.0)
            candidates.append(
                Candidate(
                    sampleIndex: index,
                    gradient: gradient,
                    hessian: hessian,
                    absGradient: abs(gradient)
                )
            )
        }
        guard candidates.count >= Self.minBuffer / 2 else { return false }

        let selected = gossSelectedCandidates(candidates, classIndex: classIndex)
        guard selected.count >= Self.minBuffer / 3 else { return false }

        var xUse: [[Double]] = []
        var gradients: [Double] = []
        var hessians: [Double] = []
        var moves: [Double] = []
        xUse.reserveCapacity(selected.count)
        gradients.reserveCapacity(selected.count)
        hessians.reserveCapacity(selected.count)
        moves.reserveCapacity(selected.count)
        for candidate in selected {
            let sample = buffer[candidate.sampleIndex]
            xUse.append(sample.x)
            gradients.append(candidate.gradient)
            hessians.append(candidate.hessian)
            moves.append(sample.movePoints)
        }

        let lambda = fxClamp(hyperParameters.xgbL2, 0.0001, 10.0000)
        var eta = fxClamp(hyperParameters.xgbLearningRate, 0.0001, 0.5000)
        if qualityDegraded {
            eta *= 0.80
        }

        let rootGH = sumGradients((0..<selected.count).map { $0 }, gradients: gradients, hessians: hessians)
        if rootGH.hessian > 1.0e-9 {
            let delta = 0.15 * eta * PluginSupportTools.clipSymmetric(
                rootGH.gradient / (rootGH.hessian + lambda),
                limit: 5.0
            )
            bias[classIndex] = PluginSupportTools.clipSymmetric(bias[classIndex] + delta, limit: 8.0)
        }

        var tree = Tree()
        var assign = Array(repeating: 0, count: selected.count)
        setLeaf(
            tree: &tree,
            nodeIndex: 0,
            assign: assign,
            tag: 0,
            gradients: gradients,
            hessians: hessians,
            moves: moves,
            eta: eta,
            lambda: lambda
        )

        let featureActive = featureSubsampleMask(classIndex: classIndex)
        var leaves = 1
        while leaves < Self.maxLeaves {
            var best: (leaf: Int, feature: Int, threshold: Double, defaultLeft: Bool, gain: Double)?
            for nodeIndex in tree.nodes.indices {
                let node = tree.nodes[nodeIndex]
                guard node.isLeaf,
                      node.depth < Self.maxDepth,
                      node.sampleCount >= 2 * Self.minData else {
                    continue
                }
                guard let split = findBestSplitForLeaf(
                    assign: assign,
                    leafTag: nodeIndex,
                    depth: node.depth,
                    xAll: xUse,
                    gradients: gradients,
                    hessians: hessians,
                    lambda: lambda,
                    featureActive: featureActive
                ) else {
                    continue
                }
                if split.gain > (best?.gain ?? 0.0) {
                    best = (nodeIndex, split.feature, split.threshold, split.defaultLeft, split.gain)
                }
            }

            guard let split = best,
                  split.feature > 0,
                  split.gain > 0.0,
                  tree.nodes.count + 2 <= Self.maxNodes else {
                break
            }

            let leftNode = tree.nodes.count
            let rightNode = tree.nodes.count + 1
            let childDepth = tree.nodes[split.leaf].depth + 1
            tree.nodes.append(Node(depth: childDepth))
            tree.nodes.append(Node(depth: childDepth))
            tree.nodes[split.leaf].isLeaf = false
            tree.nodes[split.leaf].feature = split.feature
            tree.nodes[split.leaf].threshold = split.threshold
            tree.nodes[split.leaf].defaultLeft = split.defaultLeft
            tree.nodes[split.leaf].left = leftNode
            tree.nodes[split.leaf].right = rightNode

            for index in assign.indices where assign[index] == split.leaf {
                let value = xUse[index][split.feature]
                let goLeft = value.isFinite ? value <= split.threshold : split.defaultLeft
                assign[index] = goLeft ? leftNode : rightNode
            }
            setLeaf(
                tree: &tree,
                nodeIndex: leftNode,
                assign: assign,
                tag: leftNode,
                gradients: gradients,
                hessians: hessians,
                moves: moves,
                eta: eta,
                lambda: lambda
            )
            setLeaf(
                tree: &tree,
                nodeIndex: rightNode,
                assign: assign,
                tag: rightNode,
                gradients: gradients,
                hessians: hessians,
                moves: moves,
                eta: eta,
                lambda: lambda
            )
            leaves += 1
        }

        if trees[classIndex].count < Self.maxTrees {
            trees[classIndex].append(tree)
        } else {
            trees[classIndex].removeFirst()
            trees[classIndex].append(tree)
        }
        return true
    }

    private func dartKeepMask(classIndex: Int) -> [Bool] {
        let count = trees[classIndex].count
        guard count > 0 else { return [] }
        var keepMask = Array(repeating: true, count: count)
        var kept = 0
        for treeIndex in 0..<count {
            let keep: Bool
            if treeIndex < 4 {
                keep = true
            } else {
                keep = deterministicFraction(a: treeIndex, b: classIndex, salt: 31) >= 0.05
            }
            keepMask[treeIndex] = keep
            if keep {
                kept += 1
            }
        }
        if kept == 0 {
            keepMask[count - 1] = true
        }
        return keepMask
    }

    private func gossSelectedCandidates(_ candidates: [Candidate], classIndex: Int) -> [Candidate] {
        let topRate = 0.20
        let otherRate = 0.10
        let threshold = gossThreshold(candidates.map(\.absGradient), topRate: topRate)
        let smallScale = (1.0 - topRate) / otherRate
        var selected: [Candidate] = []
        selected.reserveCapacity(max(Self.minBuffer / 3, candidates.count / 4))
        var selectedSampleIndexes = Set<Int>()

        for index in candidates.indices {
            var candidate = candidates[index]
            var keep = candidate.absGradient >= threshold
            if !keep, deterministicFraction(a: candidate.sampleIndex, b: classIndex, salt: 47) <= otherRate {
                keep = true
                candidate.gradient *= smallScale
                candidate.hessian *= smallScale
            }
            if keep {
                selected.append(candidate)
                selectedSampleIndexes.insert(candidate.sampleIndex)
            }
        }

        if selected.count < Self.minBuffer / 3 {
            for candidate in candidates.sorted(by: { $0.absGradient > $1.absGradient }) {
                guard !selectedSampleIndexes.contains(candidate.sampleIndex) else { continue }
                selected.append(candidate)
                selectedSampleIndexes.insert(candidate.sampleIndex)
                if selected.count >= Self.minBuffer / 3 {
                    break
                }
            }
        }
        for label in LabelClass.allCases {
            var labelCount = selected.reduce(0) { partial, candidate in
                partial + (buffer[candidate.sampleIndex].label == label ? 1 : 0)
            }
            guard labelCount < Self.minData else { continue }
            for candidate in candidates
                .filter({ buffer[$0.sampleIndex].label == label })
                .sorted(by: { $0.absGradient > $1.absGradient }) {
                guard !selectedSampleIndexes.contains(candidate.sampleIndex) else { continue }
                selected.append(candidate)
                selectedSampleIndexes.insert(candidate.sampleIndex)
                labelCount += 1
                if labelCount >= Self.minData {
                    break
                }
            }
        }
        return selected
    }

    private func gossThreshold(_ absoluteGradients: [Double], topRate: Double) -> Double {
        guard let first = absoluteGradients.first else { return Double.greatestFiniteMagnitude }
        var minimum = first
        var maximum = first
        for value in absoluteGradients.dropFirst() {
            minimum = min(minimum, value)
            maximum = max(maximum, value)
        }
        guard maximum - minimum > 1.0e-12 else { return minimum }
        var histogram = Array(repeating: 0, count: Self.gossBins)
        for value in absoluteGradients {
            let q = (value - minimum) / (maximum - minimum)
            let bin = min(max(Int(floor(q * Double(Self.gossBins))), 0), Self.gossBins - 1)
            histogram[bin] += 1
        }
        let needed = max(1, Int(ceil(fxClamp(topRate, 0.01, 0.95) * Double(absoluteGradients.count))))
        var cumulative = 0
        for bin in stride(from: Self.gossBins - 1, through: 0, by: -1) {
            cumulative += histogram[bin]
            if cumulative >= needed {
                return minimum + (maximum - minimum) * Double(bin) / Double(Self.gossBins)
            }
        }
        return minimum
    }

    private func featureSubsampleMask(classIndex: Int) -> [Bool] {
        var active = Array(repeating: false, count: FXDataEngineConstants.aiWeights)
        var kept = 0
        for feature in 1..<FXDataEngineConstants.aiWeights {
            let useFeature = Self.alwaysActiveFeatureIndexes.contains(feature) ||
                Self.volumeFeatureIndexes.contains(feature) ||
                deterministicFraction(a: feature, b: classIndex, salt: 59) <= 0.70
            active[feature] = useFeature
            if useFeature {
                kept += 1
            }
        }
        if kept == 0, FXDataEngineConstants.aiWeights > 1 {
            active[1] = true
        }
        return active
    }

    private func findBestSplitForLeaf(
        assign: [Int],
        leafTag: Int,
        depth: Int,
        xAll: [[Double]],
        gradients: [Double],
        hessians: [Double],
        lambda: Double,
        featureActive: [Bool]
    ) -> (feature: Int, threshold: Double, defaultLeft: Bool, gain: Double)? {
        guard depth < Self.maxDepth else { return nil }
        var totalGradient = 0.0
        var totalHessian = 0.0
        var totalCount = 0
        for index in assign.indices where assign[index] == leafTag {
            totalGradient += gradients[index]
            totalHessian += hessians[index]
            totalCount += 1
        }
        guard totalCount >= 2 * Self.minData,
              totalHessian >= 2.0 * Self.minChildHessian else {
            return nil
        }
        let parentScore = totalGradient * totalGradient / (totalHessian + lambda + 1.0e-9)
        var best: (feature: Int, threshold: Double, defaultLeft: Bool, gain: Double)?

        for feature in 1..<FXDataEngineConstants.aiWeights {
            guard feature < featureActive.count, featureActive[feature] else { continue }
            var minimum = Double.greatestFiniteMagnitude
            var maximum = -Double.greatestFiniteMagnitude
            var validCount = 0
            var missingCount = 0
            var missingGradient = 0.0
            var missingHessian = 0.0

            for index in assign.indices where assign[index] == leafTag {
                let value = xAll[index][feature]
                if !value.isFinite {
                    missingCount += 1
                    missingGradient += gradients[index]
                    missingHessian += hessians[index]
                    continue
                }
                minimum = min(minimum, value)
                maximum = max(maximum, value)
                validCount += 1
            }
            guard validCount >= 2 * Self.minData, maximum - minimum >= 1.0e-9 else { continue }

            var binGradient = Array(repeating: 0.0, count: Self.bins)
            var binHessian = Array(repeating: 0.0, count: Self.bins)
            var binCount = Array(repeating: 0, count: Self.bins)
            for index in assign.indices where assign[index] == leafTag {
                let bin = Self.binByRange(xAll[index][feature], minimum: minimum, maximum: maximum)
                guard bin >= 0, bin < Self.bins else { continue }
                binGradient[bin] += gradients[index]
                binHessian[bin] += hessians[index]
                binCount[bin] += 1
            }

            var leftGradient = 0.0
            var leftHessian = 0.0
            var leftCount = 0
            for splitBin in 0..<(Self.bins - 1) {
                leftGradient += binGradient[splitBin]
                leftHessian += binHessian[splitBin]
                leftCount += binCount[splitBin]
                let threshold = minimum + (maximum - minimum) * Double(splitBin + 1) / Double(Self.bins)
                evaluateSplitCandidate(
                    totalGradient: totalGradient,
                    totalHessian: totalHessian,
                    totalCount: totalCount,
                    parentScore: parentScore,
                    leftGradient: leftGradient + missingGradient,
                    leftHessian: leftHessian + missingHessian,
                    leftCount: leftCount + missingCount,
                    feature: feature,
                    threshold: threshold,
                    defaultLeft: true,
                    lambda: lambda,
                    best: &best
                )
                evaluateSplitCandidate(
                    totalGradient: totalGradient,
                    totalHessian: totalHessian,
                    totalCount: totalCount,
                    parentScore: parentScore,
                    leftGradient: leftGradient,
                    leftHessian: leftHessian,
                    leftCount: leftCount,
                    feature: feature,
                    threshold: threshold,
                    defaultLeft: false,
                    lambda: lambda,
                    best: &best
                )
            }
        }
        guard let best, best.gain > 0.0 else { return nil }
        return best
    }

    private func evaluateSplitCandidate(
        totalGradient: Double,
        totalHessian: Double,
        totalCount: Int,
        parentScore: Double,
        leftGradient: Double,
        leftHessian: Double,
        leftCount: Int,
        feature: Int,
        threshold: Double,
        defaultLeft: Bool,
        lambda: Double,
        best: inout (feature: Int, threshold: Double, defaultLeft: Bool, gain: Double)?
    ) {
        let rightGradient = totalGradient - leftGradient
        let rightHessian = totalHessian - leftHessian
        let rightCount = totalCount - leftCount
        guard leftCount >= Self.minData,
              rightCount >= Self.minData,
              leftHessian >= Self.minChildHessian,
              rightHessian >= Self.minChildHessian else {
            return
        }
        let gain = 0.5 * (
            leftGradient * leftGradient / (leftHessian + lambda + 1.0e-9) +
                rightGradient * rightGradient / (rightHessian + lambda + 1.0e-9) -
                parentScore
        ) - Self.gamma
        if gain > (best?.gain ?? 0.0) {
            best = (feature, threshold, defaultLeft, gain)
        }
    }

    private mutating func setLeaf(
        tree: inout Tree,
        nodeIndex: Int,
        assign: [Int],
        tag: Int,
        gradients: [Double],
        hessians: [Double],
        moves: [Double],
        eta: Double,
        lambda: Double
    ) {
        var gradient = 0.0
        var hessian = 0.0
        var sumMove = 0.0
        var sumMove2 = 0.0
        var absoluteMoves: [Double] = []
        absoluteMoves.reserveCapacity(assign.count)
        for index in assign.indices where assign[index] == tag {
            gradient += gradients[index]
            hessian += hessians[index]
            let absoluteMove = abs(moves[index])
            sumMove += absoluteMove
            sumMove2 += absoluteMove * absoluteMove
            absoluteMoves.append(absoluteMove)
        }

        let count = absoluteMoves.count
        let mean = count > 0 ? sumMove / Double(count) : 0.0
        let variance = count > 0 ? max(0.0, sumMove2 / Double(count) - mean * mean) : 0.0
        absoluteMoves.sort()
        let q10 = Self.quantile(absoluteMoves, fraction: 0.10, fallback: mean)
        let q50 = Self.quantile(absoluteMoves, fraction: 0.50, fallback: mean)
        let q90 = Self.quantile(absoluteMoves, fraction: 0.90, fallback: mean)
        let leafValue = hessian > 1.0e-9 ? eta * PluginSupportTools.clipSymmetric(gradient / (hessian + lambda), limit: 6.0) : 0.0
        let depth = nodeIndex < tree.nodes.count ? tree.nodes[nodeIndex].depth : 0
        tree.nodes[nodeIndex] = Node(
            isLeaf: true,
            feature: -1,
            threshold: 0.0,
            defaultLeft: true,
            left: -1,
            right: -1,
            depth: depth,
            leafValue: leafValue,
            moveMean: mean,
            moveVariance: variance,
            moveQ10: q10,
            moveQ50: q50,
            moveQ90: q90,
            sampleCount: count
        )
    }

    private func classMoveStats(classIndex: Int, x: [Double]) -> (mean: Double, q10: Double, q50: Double, q90: Double, support: Double) {
        var mean = 0.0
        var q10 = 0.0
        var q50 = 0.0
        var q90 = 0.0
        var support = 0.0
        var weightSum = 0.0
        for tree in trees[classIndex] {
            let leaf = traverseLeafIndex(tree, x: x)
            guard leaf >= 0, leaf < tree.nodes.count else { continue }
            let node = tree.nodes[leaf]
            guard node.sampleCount > 0 else { continue }
            let weight = abs(node.leafValue) + 0.10
            mean += weight * max(0.0, node.moveMean)
            q10 += weight * max(0.0, node.moveQ10)
            q50 += weight * max(0.0, node.moveQ50)
            q90 += weight * max(0.0, node.moveQ90)
            support += weight * Double(node.sampleCount)
            weightSum += weight
        }
        guard weightSum > 0.0 else { return (0.0, 0.0, 0.0, 0.0, 0.0) }
        return (mean / weightSum, q10 / weightSum, q50 / weightSum, q90 / weightSum, support)
    }

    private func classExpectedMove(classIndex: Int, x: [Double]) -> Double {
        var sum = 0.0
        var weightSum = 0.0
        for tree in trees[classIndex] {
            let leaf = traverseLeafIndex(tree, x: x)
            guard leaf >= 0, leaf < tree.nodes.count else { continue }
            let node = tree.nodes[leaf]
            guard node.sampleCount > 0 else { continue }
            let sigma = sqrt(max(0.0, node.moveVariance))
            let move = 0.50 * node.moveQ50 + 0.30 * node.moveMean + 0.15 * node.moveQ90 + 0.05 * sigma
            guard move > 0.0 else { continue }
            let weight = abs(node.leafValue) + 0.10
            sum += weight * move
            weightSum += weight
        }
        return weightSum > 0.0 ? sum / weightSum : 0.0
    }

    private mutating func updateValidationMetrics(
        label: LabelClass,
        probabilities: [Double],
        expectedMovePoints: Double,
        costPoints: Double
    ) {
        let y = label.rawValue
        let ce = -log(fxClamp(probabilities[y], 1.0e-6, 1.0))
        var brier = 0.0
        for classIndex in 0..<Self.classCount {
            let target = classIndex == y ? 1.0 : 0.0
            let delta = probabilities[classIndex] - target
            brier += delta * delta
        }
        brier /= Double(Self.classCount)

        var predicted = 0
        var confidence = probabilities[0]
        for classIndex in 1..<Self.classCount where probabilities[classIndex] > confidence {
            confidence = probabilities[classIndex]
            predicted = classIndex
        }
        let accuracy = predicted == y ? 1.0 : 0.0
        let bin = min(max(Int(floor(fxClamp(confidence, 0.0, 0.999999) * Double(Self.eceBins))), 0), Self.eceBins - 1)
        for index in 0..<Self.eceBins {
            eceMass[index] *= 0.997
            eceAccuracy[index] *= 0.997
            eceConfidence[index] *= 0.997
        }
        eceMass[bin] += 1.0
        eceAccuracy[bin] += accuracy
        eceConfidence[bin] += confidence

        var eceNumerator = 0.0
        var eceDenominator = 0.0
        for index in 0..<Self.eceBins where eceMass[index] > 1.0e-9 {
            let binAccuracy = eceAccuracy[index] / eceMass[index]
            let binConfidence = eceConfidence[index] / eceMass[index]
            eceNumerator += eceMass[index] * abs(binAccuracy - binConfidence)
            eceDenominator += eceMass[index]
        }
        let ece = eceDenominator > 0.0 ? eceNumerator / eceDenominator : 0.0
        let evAfterCost = expectedMovePoints - max(0.0, costPoints)

        if validationReady {
            nllFast = 0.92 * nllFast + 0.08 * ce
            nllSlow = 0.995 * nllSlow + 0.005 * ce
            brierFast = 0.92 * brierFast + 0.08 * brier
            brierSlow = 0.995 * brierSlow + 0.005 * brier
            eceFast = 0.92 * eceFast + 0.08 * ece
            eceSlow = 0.995 * eceSlow + 0.005 * ece
            evFast = 0.92 * evFast + 0.08 * evAfterCost
            evSlow = 0.995 * evSlow + 0.005 * evAfterCost
        } else {
            nllFast = ce
            nllSlow = ce
            brierFast = brier
            brierSlow = brier
            eceFast = ece
            eceSlow = ece
            evFast = evAfterCost
            evSlow = evAfterCost
            validationReady = true
        }

        validationSteps += 1
        qualityDegraded = false
        if validationSteps > 128 {
            qualityDegraded = nllFast > 1.15 * max(0.05, nllSlow) ||
                brierFast > 1.20 * max(0.03, brierSlow) ||
                eceFast > 1.25 * max(0.02, eceSlow) ||
                evFast < 0.85 * evSlow
        }
    }

    private func sumGradients(_ indices: [Int], gradients: [Double], hessians: [Double]) -> (gradient: Double, hessian: Double) {
        var gradient = 0.0
        var hessian = 0.0
        for index in indices {
            gradient += gradients[index]
            hessian += hessians[index]
        }
        return (gradient, hessian)
    }

    private static func targetDistribution(label: LabelClass, movePoints: Double, costPoints: Double) -> [Double] {
        var target = Array(repeating: 0.0, count: classCount)
        let edge = abs(movePoints) - max(costPoints, 0.0)
        if label == .skip {
            target[LabelClass.skip.rawValue] = 1.0
            return target
        }
        let direction = label == .buy ? LabelClass.buy.rawValue : LabelClass.sell.rawValue
        if edge <= 0.0 {
            target[direction] = 0.35
            target[LabelClass.skip.rawValue] = 0.65
            return target
        }
        let directionalProbability = fxClamp(0.75 + 0.10 * edge / max(costPoints, 1.0), 0.75, 0.95)
        target[direction] = directionalProbability
        target[LabelClass.skip.rawValue] = 1.0 - directionalProbability
        return target
    }

    private static func sampleTimeDecay(age: Int) -> Double {
        exp(-0.693_147_180_56 * Double(max(age, 0)) / 512.0)
    }

    private static func evWeight(label: LabelClass, movePoints: Double, costPoints: Double, baseWeight: Double) -> Double {
        let weight = fxClamp(baseWeight, 0.10, 6.00)
        let edge = abs(movePoints) - max(costPoints, 0.0)
        if label == .skip {
            return edge <= 0.0 ? fxClamp(1.50 * weight, 0.10, 8.0) : fxClamp(0.80 * weight, 0.10, 8.0)
        }
        if edge <= 0.0 {
            return fxClamp(0.55 * weight, 0.10, 8.0)
        }
        return fxClamp(weight * (1.0 + 0.08 * min(edge, 30.0)), 0.10, 8.0)
    }

    private static func binByRange(_ value: Double, minimum: Double, maximum: Double) -> Int {
        guard value.isFinite else { return -1 }
        let range = maximum - minimum
        guard range > 1.0e-12 else { return 0 }
        let q = (value - minimum) / range
        if q <= 0.0 {
            return 0
        }
        if q >= 1.0 {
            return bins - 1
        }
        return min(max(Int(floor(q * Double(bins))), 0), bins - 1)
    }

    private static func quantile(_ values: [Double], fraction: Double, fallback: Double) -> Double {
        guard !values.isEmpty else { return fallback }
        let index = min(max(Int(floor(fxClamp(fraction, 0.0, 1.0) * Double(values.count - 1))), 0), values.count - 1)
        return values[index]
    }

    private func deterministicFraction(a: Int, b: Int, salt: Int) -> Double {
        var hash = UInt64(truncatingIfNeeded: a) &* 0x9E37_79B9_7F4A_7C15
        hash ^= UInt64(truncatingIfNeeded: b + 0x9E37) &* 0xBF58_476D_1CE4_E5B9
        hash ^= UInt64(truncatingIfNeeded: steps + salt) &* 0x94D0_49BB_1331_11EB
        hash ^= hash >> 30
        hash &*= 0xBF58_476D_1CE4_E5B9
        hash ^= hash >> 27
        hash &*= 0x94D0_49BB_1331_11EB
        hash ^= hash >> 31
        return Double(hash & 0xFFFF_FFFF) / Double(UInt32.max)
    }

    private static func preparedFeatures(_ x: [Double], dataHasVolume: Bool) -> [Double] {
        var output = Array(repeating: 0.0, count: FXDataEngineConstants.aiWeights)
        for index in 0..<FXDataEngineConstants.aiWeights {
            output[index] = fxClamp(index < x.count ? fxSafeFinite(x[index]) : 0.0, -50.0, 50.0)
        }
        if !dataHasVolume {
            zeroVolumeFeatures(&output)
        }
        return output
    }

    private static func preparedWindow(_ window: [[Double]], dataHasVolume: Bool) -> [[Double]] {
        window.map { preparedFeatures($0, dataHasVolume: dataHasVolume) }
    }

    private static func zeroVolumeFeatures(_ features: inout [Double]) {
        for index in volumeFeatureIndexes where index < features.count {
            features[index] = 0.0
        }
    }

    private static let volumeFeatureIndexes = [6, 68, 69, 70, 71, 74, 75, 76, 77, 78, 80, 81, 82, 83]
    private static let alwaysActiveFeatureIndexes = [1, 2, 3, 7, 8, 12, 20]

    private static func softmax(_ logits: [Double]) -> [Double] {
        let maximum = logits.max() ?? 0.0
        let values = logits.map { exp(fxClamp($0 - maximum, -30.0, 30.0)) }
        let sum = values.reduce(0.0, +)
        guard sum > 0.0 else {
            return [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]
        }
        return values.map { $0 / sum }
    }
}
