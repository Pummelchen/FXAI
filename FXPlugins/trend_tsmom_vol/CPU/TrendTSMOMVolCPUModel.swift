import FXDataEngine
import Foundation

public struct TrendTSMOMVolCPUModel: Sendable {
    private static let stateDimension = 16

    private var steps: Int
    private var policyWeights: [[Double]]
    private var moveEMA: Double
    private var moveReady: Bool
    private var classMass: [Double]
    private var qualityBank: PluginQualityBank

    public init() {
        self.steps = 0
        self.policyWeights = Array(
            repeating: Array(repeating: 0.0, count: Self.stateDimension),
            count: LabelClass.allCases.count
        )
        self.moveEMA = 0.0
        self.moveReady = false
        self.classMass = Array(repeating: 1.0, count: LabelClass.allCases.count)
        self.qualityBank = PluginQualityBank()

        for classIndex in 0..<LabelClass.allCases.count {
            for feature in 0..<Self.stateDimension {
                policyWeights[classIndex][feature] = Self.seedWeight(classIndex + 1, feature + 1)
            }
        }
    }

    public mutating func reset() {
        self = TrendTSMOMVolCPUModel()
    }

    public mutating func train(_ request: TrainRequestV4, hyperParameters: HyperParameters) {
        let x = Self.preparedFeatures(request.x, dataHasVolume: request.context.dataHasVolume)
        let window = Self.preparedWindow(request.xWindow, dataHasVolume: request.context.dataHasVolume)
        let label = PluginContextRuntimeTools.normalizeClassLabel(
            rawLabel: request.labelClass.rawValue,
            x: x,
            movePoints: request.movePoints,
            priceCostPoints: request.context.priceCostPoints
        )
        let z = buildState(x: x, window: window, horizonMinutes: request.context.horizonMinutes)
        let volumeMultiplier = request.context.dataHasVolume ? (1.0 + 0.05 * abs(z[7])) : 1.0
        let sampleWeight = fxClamp(
            request.sampleWeight *
                PluginSupportTools.moveEdgeWeight(
                    movePoints: request.movePoints,
                    priceCostPoints: request.context.priceCostPoints
                ) *
                volumeMultiplier,
            0.15,
            6.00
        )
        let signedMove = label == .buy ? abs(request.movePoints) : (label == .sell ? -abs(request.movePoints) : 0.0)
        updateLinearPolicy(
            label: label,
            z: z,
            signedMove: signedMove,
            hyperParameters: hyperParameters,
            costPoints: request.context.priceCostPoints,
            sampleWeight: sampleWeight
        )
        classMass[label.rawValue] += sampleWeight
        qualityBank.update(request: request, sampleWeight: sampleWeight)
        updateMoveEMA(request.movePoints)
        steps += 1
    }

    public func predict(_ request: PredictRequestV4, hyperParameters _: HyperParameters) -> PredictionV4 {
        let x = Self.preparedFeatures(request.x, dataHasVolume: request.context.dataHasVolume)
        let window = Self.preparedWindow(request.xWindow, dataHasVolume: request.context.dataHasVolume)
        let z = buildState(x: x, window: window, horizonMinutes: request.context.horizonMinutes)
        let forecastVolatility = windowVolatility(z: z)
        let margin = tsmomVolMargin(z: z, window: window, forecastVolatility: forecastVolatility)
        let modelConfidence = tsmomConfidence(z: z, window: window, forecastVolatility: forecastVolatility, dataHasVolume: request.context.dataHasVolume)

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
        let classTotal = max(classMass.reduce(0.0, +), 1.0)
        let classCoverage = fxClamp((classTotal - Double(LabelClass.allCases.count)) / 128.0, 0.0, 1.0)
        let reliability = fxClamp(
            0.30 +
                0.30 * min(Double(steps) / 128.0, 1.0) +
                0.25 * modelConfidence +
                0.15 * (moveReady ? 1.0 : 0.0) +
                0.05 * classCoverage,
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
            family: .other,
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

    private func tsmomVolMargin(z: [Double], window: [[Double]], forecastVolatility: Double) -> Double {
        let recentDelta = window.isEmpty ? z[11] : Self.windowRecentDelta(window, feature: 0, count: 16)
        return PluginSupportTools.clipSymmetric(
            (0.45 * recentDelta + 0.35 * z[8] + 0.20 * linearMargin(z: z)) / max(forecastVolatility, 0.05),
            limit: 8.0
        )
    }

    private func tsmomConfidence(z: [Double], window: [[Double]], forecastVolatility: Double, dataHasVolume: Bool) -> Double {
        let recentDelta = abs(window.isEmpty ? z[11] : Self.windowRecentDelta(window, feature: 0, count: 16))
        let momentumStrength = fxClamp((recentDelta + abs(z[8])) / max(forecastVolatility, 0.05), 0.0, 2.0) / 2.0
        let volatilityQuality = fxClamp(1.0 / (1.0 + max(forecastVolatility, 0.0)), 0.0, 1.0)
        let volumeBoost = dataHasVolume ? 0.06 * abs(z[7]) : 0.0
        return fxClamp(0.38 + 0.34 * momentumStrength + 0.16 * volatilityQuality + volumeBoost, 0.0, 1.0)
    }

    private func linearMargin(z: [Double]) -> Double {
        var buy = 0.0
        var sell = 0.0
        for index in 0..<Self.stateDimension {
            buy += policyWeights[LabelClass.buy.rawValue][index] * z[index]
            sell += policyWeights[LabelClass.sell.rawValue][index] * z[index]
        }
        return PluginSupportTools.clipSymmetric(buy - sell, limit: 12.0)
    }

    private mutating func updateLinearPolicy(
        label: LabelClass,
        z: [Double],
        signedMove: Double,
        hyperParameters: HyperParameters,
        costPoints: Double,
        sampleWeight: Double
    ) {
        var logits = Array(repeating: 0.0, count: LabelClass.allCases.count)
        for classIndex in 0..<LabelClass.allCases.count {
            for index in 0..<Self.stateDimension {
                logits[classIndex] += policyWeights[classIndex][index] * z[index]
            }
        }
        let probabilities = Self.softmax(logits)
        var learningRate = fxClamp(hyperParameters.learningRate, 0.0002, 0.08) * fxClamp(sampleWeight, 0.2, 5.0)
        learningRate *= fxClamp(abs(signedMove) / max(costPoints, 0.10), 0.5, 4.0)
        let l2 = fxClamp(hyperParameters.l2, 0.0, 0.20)
        for classIndex in 0..<LabelClass.allCases.count {
            let target = classIndex == label.rawValue ? 1.0 : 0.0
            let gradient = target - probabilities[classIndex]
            for index in 0..<Self.stateDimension {
                let shrink = l2 * policyWeights[classIndex][index]
                policyWeights[classIndex][index] = fxClamp(
                    policyWeights[classIndex][index] + learningRate * (gradient * z[index] - shrink),
                    -8.0,
                    8.0
                )
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
        z[8] = window.isEmpty ? Self.safeFeature(x, 1) : Self.windowSlope(window, feature: 0)
        z[9] = Self.windowStd(window, feature: 0)
        z[10] = Self.windowRange(window, feature: 0, count: 16)
        z[11] = window.isEmpty ? Self.safeFeature(x, 7) : Self.windowRecentDelta(window, feature: 0, count: 8)
        z[12] = window.isEmpty ? Self.safeFeature(x, 12) : Self.windowEMAMean(window, feature: 1, alpha: 0.70)
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
            volatility = max(0.01 + 0.05 * abs(z[7]), moveReady ? 0.01 * moveEMA : 0.05)
        }
        return max(volatility, 0.01)
    }

    private static func safeFeature(_ x: [Double], _ index: Int) -> Double {
        guard index >= 0, index < x.count else { return 0.0 }
        return fxClamp(fxSafeFinite(x[index]), -50.0, 50.0)
    }

    private static func seedWeight(_ a: Int, _ b: Int) -> Double {
        let raw = sin(Double((AIModelID.trendTSMOMVol.rawValue + 17) * (a + 3) * 37 + (b + 11) * 101))
        return 0.035 * raw
    }

    private static func softmax(_ logits: [Double]) -> [Double] {
        let maximum = logits.max() ?? 0.0
        let values = logits.map { exp(fxClamp($0 - maximum, -35.0, 35.0)) }
        let sum = values.reduce(0.0, +)
        guard sum > 0.0 else {
            return [0.10, 0.10, 0.80]
        }
        return values.map { $0 / sum }
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
