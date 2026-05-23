import FXDataEngine
import Foundation

public struct TreeXGBCPUModel: Sendable {
    private static let classCount = 3
    private static let maxTrees = 64
    private static let maxDepth = 3
    private static let maxNodes = 31
    private static let bins = 16
    private static let maxBuffer = 2_048
    private static let minSplitSamples = 12
    private static let minChildWeight = 0.10
    private static let gamma = 0.01
    private static let buildEvery = 64
    private static let minBuffer = 128

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
        var classMass = [0.5, 0.5, 1.0]
        var sampleCount = 0
    }

    private struct Tree: Sendable {
        var nodes: [Node] = [Node()]
    }

    private struct Sample: Sendable {
        var x: [Double]
        var yDirection: Double
        var label: LabelClass
        var movePoints: Double
        var sampleWeight: Double
    }

    private var steps: Int
    private var bias: Double
    private var trees: [Tree]
    private var buffer: [Sample]
    private var calibrator: PluginBinaryCalibrator
    private var moveHead: PluginMoveHead
    private var qualityBank: PluginQualityBank

    public init() {
        self.steps = 0
        self.bias = 0.0
        self.trees = []
        self.buffer = []
        self.calibrator = PluginBinaryCalibrator()
        self.moveHead = PluginMoveHead()
        self.qualityBank = PluginQualityBank()
    }

    public mutating func reset() {
        self = TreeXGBCPUModel()
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
        let yDirection = label == .buy ? 1.0 : 0.0
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
            4.00
        )

        let rawProbability = PluginSupportTools.sigmoid(modelMargin(x))
        if label != .skip {
            let gradientNow = (yDirection - rawProbability) * sampleWeight
            bias += 0.01 * PluginSupportTools.clipSymmetric(gradientNow, limit: 2.0)
            bias = fxClamp(bias, -8.0, 8.0)
        }

        pushSample(
            Sample(
                x: x,
                yDirection: yDirection,
                label: label,
                movePoints: request.movePoints,
                sampleWeight: sampleWeight
            )
        )
        calibrator.update(rawProbability: rawProbability, target: yDirection, sampleWeight: sampleWeight)
        qualityBank.update(request: request, sampleWeight: sampleWeight)
        moveHead.update(
            x: x,
            movePoints: request.movePoints,
            hyperParameters: scaledHyperParameters,
            sampleWeight: sampleWeight
        )

        if buffer.count >= Self.minBuffer, steps % Self.buildEvery == 0 {
            _ = buildOneTree(hyperParameters: scaledHyperParameters)
        }
    }

    public func predict(_ request: PredictRequestV4, hyperParameters _: HyperParameters) -> PredictionV4 {
        let x = Self.preparedFeatures(request.x, dataHasVolume: request.context.dataHasVolume)
        let window = Self.preparedWindow(request.xWindow, dataHasVolume: request.context.dataHasVolume)
        let directionalProbability = calibrator.calibrated(PluginSupportTools.sigmoid(modelMargin(x)))

        var classMass = Array(repeating: 0.001, count: Self.classCount)
        var moveSum = 0.0
        var moveSquareSum = 0.0
        var weightSum = 0.0
        for tree in trees {
            let leaf = traverseLeafIndex(tree, x: x)
            guard leaf >= 0, leaf < tree.nodes.count else { continue }
            let node = tree.nodes[leaf]
            let leafWeight = abs(node.leafValue) + 0.20
            for classIndex in 0..<Self.classCount {
                classMass[classIndex] += leafWeight * node.classMass[classIndex]
            }
            moveSum += leafWeight * node.moveMean
            moveSquareSum += leafWeight * (node.moveVariance + node.moveMean * node.moveMean)
            weightSum += leafWeight
        }

        let massSum = max(classMass.reduce(0.0, +), 1.0e-9)
        let leafBuy = classMass[LabelClass.buy.rawValue] / massSum
        let leafSell = classMass[LabelClass.sell.rawValue] / massSum
        let leafSkip = classMass[LabelClass.skip.rawValue] / massSum
        let active = fxClamp(1.0 - leafSkip, 0.0, 1.0)
        let directionMix = leafBuy + leafSell > 1.0e-9 ? leafBuy / (leafBuy + leafSell) : directionalProbability
        let buyProbability = fxClamp(0.55 * directionalProbability + 0.45 * directionMix, 0.001, 0.999)
        let probabilities = PluginContextRuntimeTools.normalizeClassDistribution([
            active * (1.0 - buyProbability),
            active * buyProbability,
            1.0 - active
        ])

        let meanMove: Double
        let variance: Double
        if weightSum > 0.0 {
            let mean = max(0.0, moveSum / weightSum)
            meanMove = mean + 0.10 * sqrt(max(0.0, moveSquareSum / weightSum - mean * mean) + 1.0e-6)
            variance = max(0.0, moveSquareSum / weightSum - mean * mean)
        } else {
            meanMove = moveHead.ready ? moveHead.predictRaw(x) : 0.0
            variance = 0.25 * meanMove * meanMove
        }

        let sigma = sqrt(max(0.0, variance) + 1.0e-6)
        let confidence = fxClamp(max(probabilities[LabelClass.buy.rawValue], probabilities[LabelClass.sell.rawValue]), 0.0, 1.0)
        let reliability = fxClamp(1.0 - probabilities[LabelClass.skip.rawValue], 0.0, 1.0)
        let q25 = max(0.0, meanMove - 0.674 * sigma)
        let q50 = max(q25, meanMove)
        let q75 = max(q50, meanMove + 0.674 * sigma)
        let baseOutput = PluginModelOutputV4(
            classProbabilities: probabilities,
            moveMeanPoints: max(0.0, meanMove),
            moveQ25Points: q25,
            moveQ50Points: q50,
            moveQ75Points: q75,
            mfeMeanPoints: max(q75, meanMove),
            maeMeanPoints: max(0.0, 0.35 * meanMove),
            hitTimeFraction: fxClamp(0.60 - 0.20 * confidence + 0.15 * probabilities[LabelClass.skip.rawValue], 0.0, 1.0),
            pathRisk: fxClamp(0.35 * probabilities[LabelClass.skip.rawValue] + 0.25 * (1.0 - reliability), 0.0, 1.0),
            fillRisk: fxClamp(request.context.priceCostPoints / max(meanMove + request.context.minMovePoints, 0.25), 0.0, 1.0),
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
            calibratedMoveMeanPoints: max(0.0, meanMove),
            context: request.context
        )
    }

    private mutating func pushSample(_ sample: Sample) {
        buffer.append(sample)
        if buffer.count > Self.maxBuffer {
            buffer.removeFirst(buffer.count - Self.maxBuffer)
        }
    }

    private func modelMargin(_ x: [Double]) -> Double {
        var value = bias
        for tree in trees {
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

    private mutating func buildOneTree(hyperParameters: HyperParameters) -> Bool {
        let n = buffer.count
        guard n >= Self.minBuffer else { return false }
        let lambda = fxClamp(hyperParameters.xgbL2, 0.0001, 10.0000)
        let eta = fxClamp(hyperParameters.xgbLearningRate, 0.0001, 1.0000)
        var gradients = Array(repeating: 0.0, count: n)
        var hessians = Array(repeating: 0.0, count: n)

        for index in 0..<n {
            let sample = buffer[index]
            let probability = PluginSupportTools.sigmoid(modelMargin(sample.x))
            var weight = fxClamp(sample.sampleWeight, 0.25, 4.00)
            if sample.label == .skip {
                weight *= 0.05
            }
            gradients[index] = (sample.yDirection - probability) * weight
            hessians[index] = fxClamp(probability * (1.0 - probability) * weight, 0.02, 4.00)
        }

        let rootIndices = Array(0..<n)
        let root = sumGradients(rootIndices, gradients: gradients, hessians: hessians)
        if root.hessian > 1.0e-9 {
            bias += 0.20 * eta * PluginSupportTools.clipSymmetric(root.gradient / (root.hessian + lambda), limit: 5.0)
            bias = fxClamp(bias, -8.0, 8.0)
        }

        var tree = Tree()
        buildNode(
            tree: &tree,
            nodeIndex: 0,
            depth: 0,
            indices: rootIndices,
            gradients: gradients,
            hessians: hessians,
            lambda: lambda,
            eta: eta
        )
        guard !tree.nodes.isEmpty else { return false }
        if trees.count < Self.maxTrees {
            trees.append(tree)
        } else {
            trees.removeFirst()
            trees.append(tree)
        }
        return true
    }

    private mutating func buildNode(
        tree: inout Tree,
        nodeIndex: Int,
        depth: Int,
        indices: [Int],
        gradients: [Double],
        hessians: [Double],
        lambda: Double,
        eta: Double
    ) {
        guard nodeIndex >= 0, nodeIndex < Self.maxNodes else { return }
        if indices.count < Self.minSplitSamples || depth >= Self.maxDepth {
            setLeaf(tree: &tree, nodeIndex: nodeIndex, indices: indices, gradients: gradients, hessians: hessians, lambda: lambda, eta: eta)
            return
        }
        guard let split = findBestSplit(indices: indices, gradients: gradients, hessians: hessians, lambda: lambda) else {
            setLeaf(tree: &tree, nodeIndex: nodeIndex, indices: indices, gradients: gradients, hessians: hessians, lambda: lambda, eta: eta)
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
        if leftIndices.count < Self.minSplitSamples ||
            rightIndices.count < Self.minSplitSamples ||
            tree.nodes.count + 2 > Self.maxNodes {
            setLeaf(tree: &tree, nodeIndex: nodeIndex, indices: indices, gradients: gradients, hessians: hessians, lambda: lambda, eta: eta)
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
            classMass: Array(repeating: 0.0, count: Self.classCount),
            sampleCount: indices.count
        )
        buildNode(tree: &tree, nodeIndex: leftNode, depth: depth + 1, indices: leftIndices, gradients: gradients, hessians: hessians, lambda: lambda, eta: eta)
        buildNode(tree: &tree, nodeIndex: rightNode, depth: depth + 1, indices: rightIndices, gradients: gradients, hessians: hessians, lambda: lambda, eta: eta)
    }

    private func findBestSplit(
        indices: [Int],
        gradients: [Double],
        hessians: [Double],
        lambda: Double
    ) -> (feature: Int, threshold: Double, defaultLeft: Bool, gain: Double)? {
        guard indices.count >= 2 * Self.minSplitSamples else { return nil }
        let total = sumGradients(indices, gradients: gradients, hessians: hessians)
        guard total.hessian > 2.0 * Self.minChildWeight else { return nil }
        let parentScore = total.gradient * total.gradient / (total.hessian + lambda + 1.0e-9)
        var best: (feature: Int, threshold: Double, defaultLeft: Bool, gain: Double)?

        for feature in 1..<FXDataEngineConstants.aiWeights {
            var minimum = Double.greatestFiniteMagnitude
            var maximum = -Double.greatestFiniteMagnitude
            var missingGradient = 0.0
            var missingHessian = 0.0
            var missingCount = 0
            var validCount = 0

            for index in indices {
                let value = buffer[index].x[feature]
                if !value.isFinite {
                    missingGradient += gradients[index]
                    missingHessian += hessians[index]
                    missingCount += 1
                    continue
                }
                minimum = min(minimum, value)
                maximum = max(maximum, value)
                validCount += 1
            }
            guard validCount >= 2 * Self.minSplitSamples, maximum - minimum > 1.0e-9 else { continue }

            for bin in 1..<Self.bins {
                let threshold = minimum + (maximum - minimum) * Double(bin) / Double(Self.bins)
                var leftGradient = 0.0
                var leftHessian = 0.0
                var leftCount = 0
                var rightGradient = 0.0
                var rightHessian = 0.0
                var rightCount = 0

                for index in indices {
                    let value = buffer[index].x[feature]
                    guard value.isFinite else { continue }
                    if value <= threshold {
                        leftGradient += gradients[index]
                        leftHessian += hessians[index]
                        leftCount += 1
                    } else {
                        rightGradient += gradients[index]
                        rightHessian += hessians[index]
                        rightCount += 1
                    }
                }

                evaluateSplitCandidate(
                    feature: feature,
                    threshold: threshold,
                    defaultLeft: true,
                    leftGradient: leftGradient + missingGradient,
                    leftHessian: leftHessian + missingHessian,
                    leftCount: leftCount + missingCount,
                    rightGradient: rightGradient,
                    rightHessian: rightHessian,
                    rightCount: rightCount,
                    parentScore: parentScore,
                    lambda: lambda,
                    best: &best
                )
                evaluateSplitCandidate(
                    feature: feature,
                    threshold: threshold,
                    defaultLeft: false,
                    leftGradient: leftGradient,
                    leftHessian: leftHessian,
                    leftCount: leftCount,
                    rightGradient: rightGradient + missingGradient,
                    rightHessian: rightHessian + missingHessian,
                    rightCount: rightCount + missingCount,
                    parentScore: parentScore,
                    lambda: lambda,
                    best: &best
                )
            }
        }
        guard let best, best.gain > 0.0 else { return nil }
        return best
    }

    private func evaluateSplitCandidate(
        feature: Int,
        threshold: Double,
        defaultLeft: Bool,
        leftGradient: Double,
        leftHessian: Double,
        leftCount: Int,
        rightGradient: Double,
        rightHessian: Double,
        rightCount: Int,
        parentScore: Double,
        lambda: Double,
        best: inout (feature: Int, threshold: Double, defaultLeft: Bool, gain: Double)?
    ) {
        guard leftCount >= Self.minSplitSamples,
              rightCount >= Self.minSplitSamples,
              leftHessian >= Self.minChildWeight,
              rightHessian >= Self.minChildWeight else {
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
        indices: [Int],
        gradients: [Double],
        hessians: [Double],
        lambda: Double,
        eta: Double
    ) {
        let gh = sumGradients(indices, gradients: gradients, hessians: hessians)
        var moveSum = 0.0
        var moveSquareSum = 0.0
        var classMass = Array(repeating: 0.001, count: Self.classCount)

        for index in indices {
            let sample = buffer[index]
            let absMove = abs(sample.movePoints)
            moveSum += absMove
            moveSquareSum += absMove * absMove
            classMass[sample.label.rawValue] += max(0.05, sample.sampleWeight)
        }
        let count = max(indices.count, 1)
        let moveMean = moveSum / Double(count)
        let moveVariance = max(0.0, moveSquareSum / Double(count) - moveMean * moveMean)
        let leafValue = gh.hessian > 1.0e-9 ?
            eta * PluginSupportTools.clipSymmetric(gh.gradient / (gh.hessian + lambda), limit: 5.0) :
            0.0

        tree.nodes[nodeIndex] = Node(
            isLeaf: true,
            feature: -1,
            threshold: 0.0,
            defaultLeft: true,
            left: -1,
            right: -1,
            leafValue: leafValue,
            moveMean: moveMean,
            moveVariance: moveVariance,
            classMass: classMass,
            sampleCount: indices.count
        )
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
}
