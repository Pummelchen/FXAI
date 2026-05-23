import FXDataEngine
import Foundation

public struct TreeXGBFastCPUModel: Sendable {
    private static let classCount = 3
    private static let maxTrees = 192
    private static let maxDepth = 6
    private static let maxNodes = 127
    private static let maxBins = 48
    private static let maxBuffer = 1_024

    private struct Node: Sendable {
        var isLeaf = true
        var feature = -1
        var threshold = 0.0
        var defaultLeft = true
        var left = -1
        var right = -1
        var leafValue = 0.0
        var moveMean = 0.0
        var moveVariance = 0.0
        var moveQ50 = 0.0
        var sampleCount = 0
        var gain = 0.0
    }

    private struct Tree: Sendable {
        var nodes: [Node] = [Node()]
        var age = 0
        var weight = 1.0
    }

    private struct Sample: Sendable {
        var x: [Double]
        var label: LabelClass
        var movePoints: Double
        var priceCostPoints: Double
        var sampleWeight: Double
    }

    private struct RuntimeConfig: Sendable {
        var depth = 3
        var bins = 20
        var maxTrees = 96
        var buildEvery = 32
        var minBuffer = 64
        var minSplitSamples = 8
        var subsample = 0.85
        var colsample = 0.75
        var lambda = 0.20
        var alpha = 0.05
        var gamma = 0.01
        var minChildWeight = 0.08
        var maxDelta = 1.5
        var leafClip = 4.0
        var treeDecay = 0.997
        var refreshEvery = 128
        var refreshLearningRate = 0.06
    }

    private var steps: Int
    private var bias: [Double]
    private var trees: [[Tree]]
    private var buffer: [Sample]
    private var config: RuntimeConfig
    private var calibrator: PluginBinaryCalibrator
    private var moveHead: PluginMoveHead
    private var qualityBank: PluginQualityBank

    public init() {
        self.steps = 0
        self.bias = Array(repeating: 0.0, count: Self.classCount)
        self.trees = Array(repeating: [], count: Self.classCount)
        self.buffer = []
        self.config = RuntimeConfig()
        self.calibrator = PluginBinaryCalibrator()
        self.moveHead = PluginMoveHead()
        self.qualityBank = PluginQualityBank()
    }

    public mutating func reset() {
        self = TreeXGBFastCPUModel()
    }

    public mutating func train(_ request: TrainRequestV4, hyperParameters: HyperParameters) {
        configureRuntime(hyperParameters)
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
            0.25,
            4.00
        )

        let raw = rawClassProbabilities(x)
        let directionalDenominator = max(raw[LabelClass.buy.rawValue] + raw[LabelClass.sell.rawValue], 1.0e-9)
        let rawDirectionalBuy = raw[LabelClass.buy.rawValue] / directionalDenominator
        if label != .skip {
            calibrator.update(rawProbability: rawDirectionalBuy, target: label == .buy, sampleWeight: sampleWeight)
        }

        pushSample(
            Sample(
                x: x,
                label: label,
                movePoints: request.movePoints,
                priceCostPoints: request.context.priceCostPoints,
                sampleWeight: sampleWeight
            )
        )
        qualityBank.update(request: request, sampleWeight: sampleWeight)
        moveHead.update(
            x: x,
            movePoints: request.movePoints,
            hyperParameters: scaledHyperParameters,
            sampleWeight: sampleWeight
        )
        applyTreeDecay()

        if buffer.count >= config.minBuffer, steps % config.buildEvery == 0 {
            for classIndex in 0..<Self.classCount {
                if let tree = buildOneClassTree(classIndex: classIndex, hyperParameters: scaledHyperParameters) {
                    appendTree(tree, classIndex: classIndex)
                }
            }
        }
        if config.refreshEvery > 0, steps % config.refreshEvery == 0 {
            refreshLeaves()
        }
    }

    public func predict(_ request: PredictRequestV4, hyperParameters _: HyperParameters) -> PredictionV4 {
        let x = Self.preparedFeatures(request.x, dataHasVolume: request.context.dataHasVolume)
        let window = Self.preparedWindow(request.xWindow, dataHasVolume: request.context.dataHasVolume)
        let raw = rawClassProbabilities(x)
        let directionalDenominator = max(raw[LabelClass.buy.rawValue] + raw[LabelClass.sell.rawValue], 1.0e-9)
        let calibratedDirectionalBuy = calibrator.calibrated(raw[LabelClass.buy.rawValue] / directionalDenominator)
        let active = fxClamp(1.0 - raw[LabelClass.skip.rawValue], 0.0, 1.0)
        let probabilities = PluginContextRuntimeTools.normalizeClassDistribution([
            (1.0 - calibratedDirectionalBuy) * active,
            calibratedDirectionalBuy * active,
            1.0 - active
        ])

        let buyStats = classMoveStats(classIndex: LabelClass.buy.rawValue, x: x)
        let sellStats = classMoveStats(classIndex: LabelClass.sell.rawValue, x: x)
        let buyAbs = max(buyStats.mean, 0.0) > 0.0 ? max(buyStats.mean, 0.0) : buyStats.q50
        let sellAbs = max(-sellStats.mean, 0.0) > 0.0 ? max(-sellStats.mean, 0.0) : sellStats.q50
        let uncertainty = sqrt(max(0.0, raw[LabelClass.buy.rawValue] * buyStats.variance + raw[LabelClass.sell.rawValue] * sellStats.variance))
        let qMix = raw[LabelClass.buy.rawValue] * buyStats.q50 + raw[LabelClass.sell.rawValue] * sellStats.q50
        var expectedMove = raw[LabelClass.buy.rawValue] * buyAbs + raw[LabelClass.sell.rawValue] * sellAbs
        expectedMove += 0.15 * uncertainty + 0.10 * qMix
        if expectedMove <= 0.0 {
            expectedMove = moveHead.ready ? moveHead.predictRaw(x) : 0.0
        }
        expectedMove = max(0.0, expectedMove)

        let confidence = fxClamp(max(probabilities[LabelClass.buy.rawValue], probabilities[LabelClass.sell.rawValue]), 0.0, 1.0)
        let reliability = fxClamp(
            0.45 +
                0.25 * (moveHead.ready ? 1.0 : 0.0) +
                0.30 * min(Double(trees[LabelClass.buy.rawValue].count) / 32.0, 1.0),
            0.0,
            1.0
        )
        let q25 = max(0.0, 0.50 * qMix)
        let q50 = max(q25, expectedMove)
        let q75 = max(q50, expectedMove + 0.55 * uncertainty)
        let baseOutput = PluginModelOutputV4(
            classProbabilities: probabilities,
            moveMeanPoints: expectedMove,
            moveQ25Points: q25,
            moveQ50Points: q50,
            moveQ75Points: q75,
            mfeMeanPoints: max(q75, expectedMove),
            maeMeanPoints: max(0.0, 0.35 * expectedMove),
            hitTimeFraction: fxClamp(0.62 - 0.22 * confidence + 0.15 * probabilities[LabelClass.skip.rawValue], 0.0, 1.0),
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

    private mutating func configureRuntime(_ hyperParameters: HyperParameters) {
        let complexity = fxClamp(abs(hyperParameters.xgbSplit), 0.0, 1.0)
        config.depth = Self.clampInt(3 + Int(round(2.0 * complexity)), 3, Self.maxDepth)
        config.bins = Self.clampInt(16 + Int(round(18.0 * complexity)), 16, Self.maxBins)
        config.maxTrees = Self.clampInt(64 + Int(round(64.0 * complexity)), 48, Self.maxTrees)
        config.buildEvery = Self.clampInt(32 - Int(round(12.0 * complexity)), 16, 64)
        config.minBuffer = Self.clampInt(64 + Int(round(96.0 * complexity)), 64, 192)
        config.subsample = fxClamp(0.72 + 0.20 * (1.0 - complexity) + 0.10 * hyperParameters.xgbLearningRate, 0.55, 1.0)
        config.colsample = fxClamp(0.60 + 0.35 * complexity, 0.50, 1.0)
        config.lambda = fxClamp(hyperParameters.xgbL2, 0.0001, 20.0)
        config.alpha = fxClamp(0.02 + 0.80 * hyperParameters.l2 + 0.08 * complexity, 0.0, 2.0)
        config.gamma = fxClamp(0.005 + 0.035 * complexity + 0.02 * hyperParameters.l2, 0.0, 0.25)
        config.minChildWeight = fxClamp(0.06 + 0.35 * hyperParameters.l2, 0.05, 1.50)
        config.minSplitSamples = Self.clampInt(8 + Int(round(10.0 * hyperParameters.l2)), 8, 32)
        config.maxDelta = fxClamp(0.80 + 2.40 * complexity, 0.25, 4.0)
        config.leafClip = fxClamp(3.00 + 3.00 * complexity, 2.00, 8.00)
        config.treeDecay = fxClamp(0.9985 - 0.0020 * complexity, 0.9900, 0.9999)
        config.refreshEvery = Self.clampInt(192 - Int(round(96.0 * complexity)), 64, 512)
        config.refreshLearningRate = fxClamp(0.03 + 0.08 * complexity, 0.01, 0.25)
    }

    private mutating func pushSample(_ sample: Sample) {
        buffer.append(sample)
        if buffer.count > Self.maxBuffer {
            buffer.removeFirst(buffer.count - Self.maxBuffer)
        }
    }

    private func rawClassProbabilities(_ x: [Double]) -> [Double] {
        var margins = Array(repeating: 0.0, count: Self.classCount)
        for classIndex in 0..<Self.classCount {
            margins[classIndex] = classMargin(classIndex: classIndex, x: x)
        }
        return Self.softmax(margins)
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
        return tree.weight * tree.nodes[leaf].leafValue
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

    private mutating func buildOneClassTree(classIndex: Int, hyperParameters: HyperParameters) -> Tree? {
        let n = buffer.count
        guard n >= config.minBuffer else { return nil }
        var rootIndices: [Int] = []
        rootIndices.reserveCapacity(n)
        for index in 0..<n {
            if deterministicFraction(index: index, classIndex: classIndex) <= config.subsample {
                rootIndices.append(index)
            }
        }
        if rootIndices.count < config.minBuffer {
            rootIndices = Array(max(0, n - config.minBuffer)..<n)
        }
        guard rootIndices.count >= config.minSplitSamples else { return nil }

        var gradients = Array(repeating: 0.0, count: n)
        var hessians = Array(repeating: 0.0, count: n)
        for index in 0..<n {
            let sample = buffer[index]
            let target = sample.label.rawValue == classIndex ? 1.0 : 0.0
            let probability = PluginSupportTools.sigmoid(classMargin(classIndex: classIndex, x: sample.x))
            var edgeWeight = PluginSupportTools.moveEdgeWeight(
                movePoints: sample.movePoints,
                priceCostPoints: sample.priceCostPoints
            )
            let edge = abs(sample.movePoints) - sample.priceCostPoints
            if classIndex == LabelClass.skip.rawValue {
                edgeWeight *= edge <= 0.0 ? 1.50 : 0.70
            } else {
                edgeWeight *= edge <= 0.0 ? 0.55 : fxClamp(1.0 + 0.05 * min(edge, 20.0), 1.0, 2.0)
            }
            let weight = fxClamp(sample.sampleWeight * edgeWeight, 0.10, 8.00)
            gradients[index] = (target - probability) * weight
            hessians[index] = fxClamp(probability * (1.0 - probability) * weight, 0.01, 8.0)
        }

        let rootGH = sumGradients(rootIndices, gradients: gradients, hessians: hessians)
        if rootGH.hessian > 1.0e-9 {
            let step = 0.20 * fxClamp(hyperParameters.xgbLearningRate, 0.0001, 1.0) *
                leafFromGH(rootGH.gradient, rootGH.hessian)
            bias[classIndex] = fxClamp(bias[classIndex] + PluginSupportTools.clipSymmetric(step, limit: config.maxDelta), -8.0, 8.0)
        }

        let cuts = buildCuts(indices: rootIndices)
        var tree = Tree()
        buildNode(
            tree: &tree,
            nodeIndex: 0,
            depth: 0,
            indices: rootIndices,
            gradients: gradients,
            hessians: hessians,
            cuts: cuts,
            eta: fxClamp(hyperParameters.xgbLearningRate, 0.0001, 1.0)
        )
        return tree.nodes.isEmpty ? nil : tree
    }

    private mutating func appendTree(_ tree: Tree, classIndex: Int) {
        if trees[classIndex].count < config.maxTrees {
            trees[classIndex].append(tree)
            return
        }
        var dropIndex = 0
        var worst = Double.greatestFiniteMagnitude
        for index in trees[classIndex].indices {
            let candidate = 0.70 * trees[classIndex][index].weight + 0.30 / Double(trees[classIndex][index].age + 1)
            if candidate < worst {
                worst = candidate
                dropIndex = index
            }
        }
        trees[classIndex].remove(at: dropIndex)
        trees[classIndex].append(tree)
    }

    private mutating func applyTreeDecay() {
        for classIndex in 0..<Self.classCount {
            for index in trees[classIndex].indices {
                trees[classIndex][index].weight *= config.treeDecay
                trees[classIndex][index].age += 1
            }
            trees[classIndex].removeAll { $0.weight < 0.02 }
            if trees[classIndex].count > config.maxTrees {
                trees[classIndex].removeFirst(trees[classIndex].count - config.maxTrees)
            }
        }
    }

    private func buildCuts(indices: [Int]) -> [[Double]] {
        var cuts = Array(repeating: [Double](), count: FXDataEngineConstants.aiWeights)
        guard !indices.isEmpty else { return cuts }
        for feature in 1..<FXDataEngineConstants.aiWeights {
            var minimum = Double.greatestFiniteMagnitude
            var maximum = -Double.greatestFiniteMagnitude
            var valid = 0
            for index in indices {
                let value = buffer[index].x[feature]
                guard value.isFinite else { continue }
                minimum = min(minimum, value)
                maximum = max(maximum, value)
                valid += 1
            }
            guard valid >= 2, maximum - minimum > 1.0e-12 else { continue }
            var featureCuts: [Double] = []
            for bin in 0..<(config.bins - 1) {
                featureCuts.append(minimum + (maximum - minimum) * Double(bin + 1) / Double(config.bins))
            }
            cuts[feature] = featureCuts
        }
        return cuts
    }

    private mutating func buildNode(
        tree: inout Tree,
        nodeIndex: Int,
        depth: Int,
        indices: [Int],
        gradients: [Double],
        hessians: [Double],
        cuts: [[Double]],
        eta: Double
    ) {
        guard nodeIndex >= 0, nodeIndex < Self.maxNodes else { return }
        if indices.count < config.minSplitSamples || depth >= config.depth {
            setLeaf(tree: &tree, nodeIndex: nodeIndex, indices: indices, gradients: gradients, hessians: hessians, eta: eta)
            return
        }
        guard let split = findBestSplit(
            nodeIndex: nodeIndex,
            depth: depth,
            indices: indices,
            gradients: gradients,
            hessians: hessians,
            cuts: cuts
        ) else {
            setLeaf(tree: &tree, nodeIndex: nodeIndex, indices: indices, gradients: gradients, hessians: hessians, eta: eta)
            return
        }

        var leftIndices: [Int] = []
        var rightIndices: [Int] = []
        leftIndices.reserveCapacity(indices.count)
        rightIndices.reserveCapacity(indices.count)
        for index in indices {
            let value = buffer[index].x[split.feature]
            let goLeft = value.isFinite ? value <= split.threshold : split.defaultLeft
            if goLeft {
                leftIndices.append(index)
            } else {
                rightIndices.append(index)
            }
        }
        if leftIndices.count < config.minSplitSamples ||
            rightIndices.count < config.minSplitSamples ||
            tree.nodes.count + 2 > Self.maxNodes {
            setLeaf(tree: &tree, nodeIndex: nodeIndex, indices: indices, gradients: gradients, hessians: hessians, eta: eta)
            return
        }

        let leftNode = tree.nodes.count
        let rightNode = tree.nodes.count + 1
        tree.nodes.append(Node())
        tree.nodes.append(Node())
        tree.nodes[nodeIndex] = Node(
            isLeaf: false,
            feature: split.feature,
            threshold: split.threshold,
            defaultLeft: split.defaultLeft,
            left: leftNode,
            right: rightNode,
            leafValue: 0.0,
            moveMean: 0.0,
            moveVariance: 0.0,
            moveQ50: 0.0,
            sampleCount: indices.count,
            gain: split.gain
        )
        buildNode(tree: &tree, nodeIndex: leftNode, depth: depth + 1, indices: leftIndices, gradients: gradients, hessians: hessians, cuts: cuts, eta: eta)
        buildNode(tree: &tree, nodeIndex: rightNode, depth: depth + 1, indices: rightIndices, gradients: gradients, hessians: hessians, cuts: cuts, eta: eta)
    }

    private func findBestSplit(
        nodeIndex: Int,
        depth: Int,
        indices: [Int],
        gradients: [Double],
        hessians: [Double],
        cuts: [[Double]]
    ) -> (feature: Int, threshold: Double, defaultLeft: Bool, gain: Double)? {
        guard indices.count >= 2 * config.minSplitSamples else { return nil }
        let total = sumGradients(indices, gradients: gradients, hessians: hessians)
        guard total.hessian > 2.0 * config.minChildWeight else { return nil }
        let parentScore = scoreGH(total.gradient, total.hessian)
        guard parentScore > 0.0 else { return nil }
        var best: (feature: Int, threshold: Double, defaultLeft: Bool, gain: Double)?

        for feature in 1..<FXDataEngineConstants.aiWeights {
            guard featureSelected(feature: feature, nodeSeed: nodeIndex + depth * 131) else { continue }
            guard !cuts[feature].isEmpty else { continue }
            for threshold in cuts[feature] {
                var leftGradient = 0.0
                var leftHessian = 0.0
                var leftCount = 0
                for index in indices {
                    let value = buffer[index].x[feature]
                    if value.isFinite, value <= threshold {
                        leftGradient += gradients[index]
                        leftHessian += hessians[index]
                        leftCount += 1
                    }
                }
                let rightGradient = total.gradient - leftGradient
                let rightHessian = total.hessian - leftHessian
                let rightCount = indices.count - leftCount
                guard leftCount >= config.minSplitSamples,
                      rightCount >= config.minSplitSamples,
                      leftHessian >= config.minChildWeight,
                      rightHessian >= config.minChildWeight else {
                    continue
                }
                let gain = 0.5 * (scoreGH(leftGradient, leftHessian) + scoreGH(rightGradient, rightHessian) - parentScore) - config.gamma
                if gain > (best?.gain ?? 0.0) {
                    best = (feature, threshold, true, gain)
                }
            }
        }
        guard let best, best.gain > 0.0 else { return nil }
        return best
    }

    private mutating func setLeaf(
        tree: inout Tree,
        nodeIndex: Int,
        indices: [Int],
        gradients: [Double],
        hessians: [Double],
        eta: Double
    ) {
        let gh = sumGradients(indices, gradients: gradients, hessians: hessians)
        var moveSum = 0.0
        var moveSquareSum = 0.0
        var absMoves: [Double] = []
        absMoves.reserveCapacity(indices.count)
        for index in indices {
            let move = buffer[index].movePoints
            moveSum += move
            moveSquareSum += move * move
            absMoves.append(abs(move))
        }
        let count = max(indices.count, 1)
        let mean = moveSum / Double(count)
        let variance = max(0.0, moveSquareSum / Double(count) - mean * mean)
        absMoves.sort()
        let q50 = absMoves.isEmpty ? 0.0 : absMoves[absMoves.count / 2]
        let leaf = eta * leafFromGH(gh.gradient, gh.hessian)
        tree.nodes[nodeIndex] = Node(
            isLeaf: true,
            feature: -1,
            threshold: 0.0,
            defaultLeft: true,
            left: -1,
            right: -1,
            leafValue: PluginSupportTools.clipSymmetric(PluginSupportTools.clipSymmetric(leaf, limit: config.maxDelta), limit: config.leafClip),
            moveMean: mean,
            moveVariance: variance,
            moveQ50: q50,
            sampleCount: indices.count,
            gain: 0.0
        )
    }

    private mutating func refreshLeaves() {
        guard !buffer.isEmpty else { return }
        let start = max(0, buffer.count - min(buffer.count, config.minBuffer * 2))
        for classIndex in 0..<Self.classCount {
            for treeIndex in trees[classIndex].indices {
                var counts = Array(repeating: 0.0, count: trees[classIndex][treeIndex].nodes.count)
                var positives = Array(repeating: 0.0, count: trees[classIndex][treeIndex].nodes.count)
                var moveSum = Array(repeating: 0.0, count: trees[classIndex][treeIndex].nodes.count)
                var moveSquareSum = Array(repeating: 0.0, count: trees[classIndex][treeIndex].nodes.count)
                var absMoveSum = Array(repeating: 0.0, count: trees[classIndex][treeIndex].nodes.count)
                for sampleIndex in start..<buffer.count {
                    let sample = buffer[sampleIndex]
                    let leaf = traverseLeafIndex(trees[classIndex][treeIndex], x: sample.x)
                    guard leaf >= 0, leaf < counts.count else { continue }
                    let move = sample.movePoints
                    counts[leaf] += 1.0
                    moveSum[leaf] += move
                    moveSquareSum[leaf] += move * move
                    absMoveSum[leaf] += abs(move)
                    if sample.label.rawValue == classIndex {
                        positives[leaf] += 1.0
                    }
                }
                for nodeIndex in trees[classIndex][treeIndex].nodes.indices {
                    guard trees[classIndex][treeIndex].nodes[nodeIndex].isLeaf,
                          counts[nodeIndex] > 0.0 else { continue }
                    let count = counts[nodeIndex]
                    let mean = moveSum[nodeIndex] / count
                    let variance = max(0.0, moveSquareSum[nodeIndex] / count - mean * mean)
                    let q50 = 0.80 * absMoveSum[nodeIndex] / count
                    trees[classIndex][treeIndex].nodes[nodeIndex].moveMean =
                        0.70 * trees[classIndex][treeIndex].nodes[nodeIndex].moveMean + 0.30 * mean
                    trees[classIndex][treeIndex].nodes[nodeIndex].moveVariance =
                        0.70 * trees[classIndex][treeIndex].nodes[nodeIndex].moveVariance + 0.30 * variance
                    trees[classIndex][treeIndex].nodes[nodeIndex].moveQ50 =
                        0.70 * trees[classIndex][treeIndex].nodes[nodeIndex].moveQ50 + 0.30 * q50
                    trees[classIndex][treeIndex].nodes[nodeIndex].sampleCount = Int(count)
                    let classProbability = (positives[nodeIndex] + 1.0) / (count + 2.0)
                    let targetLogit = PluginSupportTools.logit(fxClamp(classProbability, 0.01, 0.99))
                    let delta = PluginSupportTools.clipSymmetric(
                        targetLogit - trees[classIndex][treeIndex].nodes[nodeIndex].leafValue,
                        limit: config.maxDelta
                    )
                    trees[classIndex][treeIndex].nodes[nodeIndex].leafValue = PluginSupportTools.clipSymmetric(
                        trees[classIndex][treeIndex].nodes[nodeIndex].leafValue + config.refreshLearningRate * delta,
                        limit: config.leafClip
                    )
                }
            }
        }
    }

    private func classMoveStats(classIndex: Int, x: [Double]) -> (mean: Double, variance: Double, q50: Double) {
        var mean = 0.0
        var variance = 0.0
        var q50 = 0.0
        var weightSum = 0.0
        for tree in trees[classIndex] {
            let leaf = traverseLeafIndex(tree, x: x)
            guard leaf >= 0, leaf < tree.nodes.count else { continue }
            let node = tree.nodes[leaf]
            let weight = tree.weight * (abs(node.leafValue) + 0.05)
            guard weight > 0.0 else { continue }
            mean += weight * node.moveMean
            variance += weight * max(node.moveVariance, 0.0)
            q50 += weight * max(node.moveQ50, 0.0)
            weightSum += weight
        }
        guard weightSum > 0.0 else { return (0.0, 0.0, 0.0) }
        return (mean / weightSum, variance / weightSum, q50 / weightSum)
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

    private func scoreGH(_ gradient: Double, _ hessian: Double) -> Double {
        guard hessian > 1.0e-12 else { return 0.0 }
        let soft = abs(gradient) - config.alpha
        guard soft > 0.0 else { return 0.0 }
        return (soft * soft) / (hessian + config.lambda + 1.0e-9)
    }

    private func leafFromGH(_ gradient: Double, _ hessian: Double) -> Double {
        guard hessian > 1.0e-12 else { return 0.0 }
        let absoluteGradient = abs(gradient)
        guard absoluteGradient > config.alpha else { return 0.0 }
        let sign = gradient >= 0.0 ? 1.0 : -1.0
        return sign * (absoluteGradient - config.alpha) / (hessian + config.lambda + 1.0e-9)
    }

    private func featureSelected(feature: Int, nodeSeed: Int) -> Bool {
        if Self.volumeFeatureIndexes.contains(feature) {
            return true
        }
        guard config.colsample < 0.999 else { return true }
        var hash = UInt32(truncatingIfNeeded: feature) &* 2_654_435_761
        hash ^= UInt32(truncatingIfNeeded: nodeSeed) &* 2_246_822_519
        hash ^= UInt32(truncatingIfNeeded: steps) &* 3_266_489_917
        let value = Double(hash & 0xFFFF) / 65_535.0
        return value <= config.colsample
    }

    private func deterministicFraction(index: Int, classIndex: Int) -> Double {
        var hash = UInt32(truncatingIfNeeded: index) &* 2_654_435_761
        hash ^= UInt32(truncatingIfNeeded: steps) &* 2_246_822_519
        hash ^= UInt32(truncatingIfNeeded: classIndex) &* 3_266_489_917
        return Double(hash & 0xFFFF) / 65_535.0
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

    private static func softmax(_ logits: [Double]) -> [Double] {
        let maximum = logits.max() ?? 0.0
        let values = logits.map { exp(fxClamp($0 - maximum, -30.0, 30.0)) }
        let sum = values.reduce(0.0, +)
        guard sum > 0.0 else {
            return [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]
        }
        return values.map { $0 / sum }
    }

    private static func clampInt(_ value: Int, _ lower: Int, _ upper: Int) -> Int {
        min(max(value, lower), upper)
    }
}
