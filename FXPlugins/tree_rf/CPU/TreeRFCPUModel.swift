import FXDataEngine
import Foundation

public struct TreeRFCPUModel: Sendable {
    private static let stateDimension = 16
    private static let treeCount = 13
    private static let leafCount = 8
    private static let depth = 3

    private var steps: Int
    private var leafMass: [[[Double]]]
    private var leafMove: [[Double]]
    private var splitFeature: [[Int]]
    private var splitThreshold: [[Double]]
    private var classMass: [Double]
    private var moveEMA: Double
    private var moveReady: Bool
    private var qualityBank: PluginQualityBank

    public init() {
        self.steps = 0
        self.leafMass = Array(
            repeating: Array(
                repeating: Array(repeating: 1.0, count: LabelClass.allCases.count),
                count: Self.leafCount
            ),
            count: Self.treeCount
        )
        self.leafMove = Array(repeating: Array(repeating: 0.0, count: Self.leafCount), count: Self.treeCount)
        self.splitFeature = Array(repeating: Array(repeating: 0, count: Self.depth), count: Self.treeCount)
        self.splitThreshold = Array(repeating: Array(repeating: 0.0, count: Self.depth), count: Self.treeCount)
        self.classMass = Array(repeating: 1.0, count: LabelClass.allCases.count)
        self.moveEMA = 0.0
        self.moveReady = false
        self.qualityBank = PluginQualityBank()

        for tree in 0..<Self.treeCount {
            for depth in 0..<Self.depth {
                splitFeature[tree][depth] = 1 + ((tree * 7 + depth * 11 + AIModelID.treeRF.rawValue) % (Self.stateDimension - 1))
                splitThreshold[tree][depth] = Self.seedWeight(tree + 3, depth + 5) * 4.0
            }
        }
    }

    public mutating func reset() {
        self = TreeRFCPUModel()
    }

    public mutating func train(_ request: TrainRequestV4, hyperParameters _: HyperParameters) {
        let x = Self.preparedFeatures(request.x, dataHasVolume: request.context.dataHasVolume)
        let window = Self.preparedWindow(request.xWindow, dataHasVolume: request.context.dataHasVolume)
        let label = PluginContextRuntimeTools.normalizeClassLabel(
            rawLabel: request.labelClass.rawValue,
            x: x,
            movePoints: request.movePoints,
            priceCostPoints: request.context.priceCostPoints
        )
        let z = buildState(x: x, window: window, horizonMinutes: request.context.horizonMinutes)
        let sampleWeight = fxClamp(
            request.sampleWeight *
                PluginSupportTools.moveEdgeWeight(
                    movePoints: request.movePoints,
                    priceCostPoints: request.context.priceCostPoints
                ),
            0.15,
            6.00
        )

        updateRandomForest(label: label, z: z, movePoints: request.movePoints, sampleWeight: sampleWeight)
        qualityBank.update(request: request, sampleWeight: sampleWeight)
        classMass[label.rawValue] += sampleWeight
        updateMoveEMA(request.movePoints)
        steps += 1
    }

    public func predict(_ request: PredictRequestV4, hyperParameters _: HyperParameters) -> PredictionV4 {
        let x = Self.preparedFeatures(request.x, dataHasVolume: request.context.dataHasVolume)
        let window = Self.preparedWindow(request.xWindow, dataHasVolume: request.context.dataHasVolume)
        let z = buildState(x: x, window: window, horizonMinutes: request.context.horizonMinutes)
        var forecastVolatility = windowVolatility(z: z)
        var modelConfidence = 0.0
        let margin = randomForestMargin(z: z, confidence: &modelConfidence)
        forecastVolatility = max(forecastVolatility, 0.0)

        let cost = max(0.0, fxSafeFinite(request.context.priceCostPoints))
        let minMove = max(fxSafeFinite(request.context.minMovePoints), 0.10)
        let scale = max(max(forecastVolatility, minMove * 0.20), 0.05)
        let directionalBuy = PluginSupportTools.sigmoid(margin / scale)
        let baseMove = max(moveReady ? moveEMA : minMove, minMove)
        let edge = abs(margin) * baseMove
        let active = PluginSupportTools.sigmoid((edge - cost) / max(minMove, 0.10))
        let weak = fxClamp(1.0 - modelConfidence, 0.0, 1.0)
        let skip = fxClamp(0.12 + 0.58 * (1.0 - active) + 0.25 * weak, 0.05, 0.92)
        let probabilities = PluginContextRuntimeTools.normalizeClassDistribution([
            (1.0 - skip) * (1.0 - directionalBuy),
            (1.0 - skip) * directionalBuy,
            skip
        ])

        let expectedMove = max(0.0, max(edge, moveReady ? moveEMA : 0.0))
        let sigma = max(0.10, forecastVolatility + 0.25 * expectedMove + 0.25 * minMove)
        let confidence = fxClamp(
            0.50 * max(probabilities[LabelClass.sell.rawValue], probabilities[LabelClass.buy.rawValue]) +
                0.35 * modelConfidence +
                0.15 * active,
            0.0,
            1.0
        )
        let reliability = fxClamp(
            0.30 +
                0.30 * min(Double(steps) / 128.0, 1.0) +
                0.25 * modelConfidence +
                0.15 * (moveReady ? 1.0 : 0.0),
            0.0,
            1.0
        )
        let q25 = max(0.0, expectedMove - 0.55 * sigma)
        let q50 = max(q25, expectedMove)
        let q75 = max(q50, expectedMove + 0.55 * sigma)
        let mfe = max(q75, expectedMove * (1.05 + 0.35 * confidence))
        let mae = max(0.0, expectedMove * (0.30 + 0.35 * skip + 0.15 * weak))
        let baseOutput = PluginModelOutputV4(
            classProbabilities: probabilities,
            moveMeanPoints: expectedMove,
            moveQ25Points: q25,
            moveQ50Points: q50,
            moveQ75Points: q75,
            mfeMeanPoints: mfe,
            maeMeanPoints: mae,
            hitTimeFraction: fxClamp(0.70 - 0.35 * active + 0.25 * skip, 0.0, 1.0),
            pathRisk: fxClamp(0.35 * skip + 0.30 * weak + 0.35 * mae / max(mfe, 0.10), 0.0, 1.0),
            fillRisk: fxClamp(cost / max(expectedMove + minMove, 0.10), 0.0, 1.0),
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

    private mutating func updateRandomForest(
        label: LabelClass,
        z: [Double],
        movePoints: Double,
        sampleWeight: Double
    ) {
        for tree in 0..<Self.treeCount {
            let hash = UInt32(truncatingIfNeeded: steps) &* 1_103_515_245 &+
                UInt32(truncatingIfNeeded: tree) &* 2_654_435_761 &+
                UInt32(truncatingIfNeeded: AIModelID.treeRF.rawValue) &* 97
            if hash % 5 == 0 {
                continue
            }
            let leaf = rfLeaf(tree: tree, z: z)
            let weight = max(0.05, sampleWeight)
            leafMass[tree][leaf][label.rawValue] += weight
            leafMove[tree][leaf] = 0.96 * leafMove[tree][leaf] + 0.04 * abs(movePoints)
            for depth in 0..<Self.depth {
                let feature = rfFeatureIndex(tree: tree, depth: depth)
                splitThreshold[tree][depth] = 0.995 * splitThreshold[tree][depth] + 0.005 * z[feature]
            }
        }
    }

    private mutating func updateMoveEMA(_ movePoints: Double) {
        let absoluteMove = abs(fxSafeFinite(movePoints))
        if moveReady {
            moveEMA = 0.97 * moveEMA + 0.03 * absoluteMove
        } else {
            moveEMA = absoluteMove
            moveReady = true
        }
    }

    private func randomForestMargin(z: [Double], confidence: inout Double) -> Double {
        var vote = Array(repeating: 1.0e-3, count: LabelClass.allCases.count)
        for tree in 0..<Self.treeCount {
            let leaf = rfLeaf(tree: tree, z: z)
            let total = max(leafMass[tree][leaf].reduce(0.0, +), 1.0)
            for classIndex in 0..<LabelClass.allCases.count {
                vote[classIndex] += leafMass[tree][leaf][classIndex] / total
            }
        }
        let denominator = max(vote.reduce(0.0, +), 1.0)
        confidence = fxClamp(max(vote[LabelClass.sell.rawValue], vote[LabelClass.buy.rawValue]) / denominator, 0.0, 1.0)
        return PluginSupportTools.clipSymmetric(
            (vote[LabelClass.buy.rawValue] - vote[LabelClass.sell.rawValue]) / denominator * 5.0,
            limit: 8.0
        )
    }

    private func rfLeaf(tree: Int, z: [Double]) -> Int {
        var leaf = 0
        for depth in 0..<Self.depth {
            let feature = rfFeatureIndex(tree: tree, depth: depth)
            if z[feature] > splitThreshold[tree][depth] {
                leaf |= 1 << depth
            }
        }
        return min(max(leaf, 0), Self.leafCount - 1)
    }

    private func rfFeatureIndex(tree: Int, depth: Int) -> Int {
        min(max(splitFeature[tree][depth], 0), Self.stateDimension - 1)
    }

    private func buildState(x: [Double], window: [[Double]], horizonMinutes: Int) -> [Double] {
        var z = Array(repeating: 0.0, count: Self.stateDimension)
        z[0] = 1.0
        z[1] = Self.safeFeature(x, 1)
        z[2] = Self.safeFeature(x, 2)
        z[3] = Self.safeFeature(x, 3)
        z[4] = Self.safeFeature(x, 4)
        z[5] = Self.safeFeature(x, 7)
        z[6] = Self.safeFeature(x, 12)
        z[7] = fxClamp(0.65 * Self.safeFeature(x, 40) + 0.35 * Self.safeFeature(x, 6), -8.0, 8.0)
        z[8] = Self.windowSlope(window, feature: 0)
        z[9] = Self.windowStd(window, feature: 0)
        z[10] = Self.windowRange(window, feature: 0, count: 16)
        z[11] = Self.windowRecentDelta(window, feature: 0, count: 8)
        z[12] = Self.windowEMAMean(window, feature: 1, alpha: 0.70)
        z[13] = Self.safeFeature(x, FXDataEngineConstants.macroEventFeatureOffset + 14)
        z[14] = Self.safeFeature(x, FXDataEngineConstants.macroEventFeatureOffset + 19)
        z[15] = fxClamp(Double(horizonMinutes) / 60.0, 0.0, 2.0)
        for index in 1..<Self.stateDimension {
            z[index] = fxClamp(z[index], -8.0, 8.0)
        }
        return z
    }

    private func windowVolatility(z: [Double]) -> Double {
        var volatility = max(z[9], 0.0)
        if volatility <= 1.0e-6 {
            volatility = max(abs(z[11]), 0.0)
        }
        if volatility <= 1.0e-6 {
            volatility = max(moveReady ? 0.01 * moveEMA : 0.05, 0.01)
        }
        return volatility
    }

    private static func seedWeight(_ a: Int, _ b: Int) -> Double {
        let raw = sin(Double((AIModelID.treeRF.rawValue + 17) * (a + 3) * 37 + (b + 11) * 101))
        return 0.035 * raw
    }

    private static func safeFeature(_ x: [Double], _ index: Int) -> Double {
        guard index >= 0, index < x.count else { return 0.0 }
        return fxClamp(fxSafeFinite(x[index]), -50.0, 50.0)
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

    private static func windowSlope(_ window: [[Double]], feature: Int) -> Double {
        guard window.count >= 2 else { return 0.0 }
        let values = windowValues(window, feature: feature, count: min(window.count, 16))
        guard values.count >= 2 else { return 0.0 }
        let n = Double(values.count)
        let meanX = (n - 1.0) * 0.5
        let meanY = values.reduce(0.0, +) / n
        var numerator = 0.0
        var denominator = 0.0
        for index in values.indices {
            let x = Double(index) - meanX
            numerator += x * (values[index] - meanY)
            denominator += x * x
        }
        return denominator > 1.0e-12 ? fxClamp(-numerator / denominator, -8.0, 8.0) : 0.0
    }

    private static func windowStd(_ window: [[Double]], feature: Int) -> Double {
        let values = windowValues(window, feature: feature, count: min(window.count, 16))
        guard !values.isEmpty else { return 0.0 }
        let mean = values.reduce(0.0, +) / Double(values.count)
        let variance = values.reduce(0.0) { $0 + ($1 - mean) * ($1 - mean) } / Double(values.count)
        return sqrt(max(0.0, variance))
    }

    private static func windowRange(_ window: [[Double]], feature: Int, count: Int) -> Double {
        let values = windowValues(window, feature: feature, count: count)
        guard let minimum = values.min(), let maximum = values.max() else { return 0.0 }
        return maximum - minimum
    }

    private static func windowRecentDelta(_ window: [[Double]], feature: Int, count: Int) -> Double {
        let values = windowValues(window, feature: feature, count: count)
        guard values.count >= 2 else { return 0.0 }
        return values[0] - values[values.count - 1]
    }

    private static func windowEMAMean(_ window: [[Double]], feature: Int, alpha: Double) -> Double {
        let values = windowValues(window, feature: feature, count: min(window.count, 16))
        guard var ema = values.last else { return 0.0 }
        let clampedAlpha = fxClamp(alpha, 0.0, 1.0)
        for value in values.dropLast().reversed() {
            ema = clampedAlpha * value + (1.0 - clampedAlpha) * ema
        }
        return ema
    }

    private static func windowValues(_ window: [[Double]], feature: Int, count: Int) -> [Double] {
        guard feature >= 0, count > 0 else { return [] }
        return window.prefix(count).map { row in
            feature < row.count ? fxSafeFinite(row[feature]) : 0.0
        }
    }
}
