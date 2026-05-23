import FXDataEngine
import Foundation

public struct AIMLPCPUModel: Sendable {
    private static let featureCount = 24
    private static let hiddenCount = 16
    private static let architectureID = 9
    private static let volumeFeatureIndexes = [6, 68, 69, 70, 71, 74, 75, 76, 77, 78, 80, 81, 82, 83]

    private var steps: Int
    private var hiddenWeights: [[Double]]
    private var hiddenBias: [Double]
    private var classWeights: [[Double]]
    private var moveWeights: [Double]
    private var classMass: [Double]
    private var moveEMA: Double
    private var moveReady: Bool
    private var calibrator: PluginTernaryCalibrator
    private var qualityBank: PluginQualityBank

    public init() {
        self.steps = 0
        self.hiddenWeights = Array(repeating: Array(repeating: 0.0, count: Self.featureCount), count: Self.hiddenCount)
        self.hiddenBias = Array(repeating: 0.0, count: Self.hiddenCount)
        self.classWeights = Array(repeating: Array(repeating: 0.0, count: Self.hiddenCount + 1), count: LabelClass.allCases.count)
        self.moveWeights = Array(repeating: 0.0, count: Self.hiddenCount + 1)
        self.classMass = Array(repeating: 1.0, count: LabelClass.allCases.count)
        self.moveEMA = 0.0
        self.moveReady = false
        self.calibrator = PluginTernaryCalibrator()
        self.qualityBank = PluginQualityBank()
        seedWeights()
    }

    public mutating func reset() {
        self = AIMLPCPUModel()
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
        let features = buildFeatures(x: x, window: window, context: request.context)
        let hidden = hiddenActivations(features: features, window: window)
        let raw = rawProbabilities(hidden: hidden)
        let moveTarget = max(0.0, abs(request.movePoints) - max(0.0, request.context.priceCostPoints))
        let volumeMultiplier = request.context.dataHasVolume ? 1.0 + 0.04 * abs(Self.safeFeature(x, 6)) : 1.0
        let sampleWeight = fxClamp(
            request.sampleWeight * volumeMultiplier *
                PluginSupportTools.moveEdgeWeight(
                    movePoints: request.movePoints,
                    priceCostPoints: request.context.priceCostPoints
                ),
            0.15,
            6.0
        )
        updateClassHead(label: label, raw: raw, hidden: hidden, sampleWeight: sampleWeight, hyperParameters: hyperParameters)
        updateMoveHead(targetMove: moveTarget, hidden: hidden, sampleWeight: sampleWeight, hyperParameters: hyperParameters)
        calibrator.update(rawProbabilities: raw, labelClass: label, sampleWeight: sampleWeight, learningRate: hyperParameters.learningRate)
        classMass[label.rawValue] += sampleWeight
        updateMoveEMA(abs(request.movePoints))
        qualityBank.update(request: request, sampleWeight: sampleWeight)
        steps += 1
    }

    public func predict(_ request: PredictRequestV4, hyperParameters _: HyperParameters) -> PredictionV4 {
        let x = Self.preparedFeatures(request.x, dataHasVolume: request.context.dataHasVolume)
        let window = Self.preparedWindow(request.xWindow, dataHasVolume: request.context.dataHasVolume)
        let features = buildFeatures(x: x, window: window, context: request.context)
        let hidden = hiddenActivations(features: features, window: window)
        let raw = rawProbabilities(hidden: hidden)
        let probabilities = PluginContextRuntimeTools.normalizeClassDistribution(calibrator.calibrated(raw))

        let predictedMove = max(0.0, Self.dot(moveWeights, hiddenWithBias(hidden)))
        let active = fxClamp(1.0 - probabilities[LabelClass.skip.rawValue], 0.0, 1.0)
        let baseMove = max(predictedMove, moveReady ? moveEMA : request.context.minMovePoints)
        let expectedMove = max(0.0, baseMove * max(active, 0.15))
        let disagreement = abs(probabilities[LabelClass.buy.rawValue] - probabilities[LabelClass.sell.rawValue])
        let sigma = max(0.10, 0.35 * expectedMove + windowVolatility(features: features) + 0.20 * request.context.minMovePoints)
        let q25 = max(0.0, expectedMove - 0.55 * sigma)
        let q50 = max(q25, expectedMove)
        let q75 = max(q50, expectedMove + 0.55 * sigma)
        let confidence = fxClamp(
            0.55 * max(probabilities[LabelClass.buy.rawValue], probabilities[LabelClass.sell.rawValue]) +
                0.25 * active +
                0.20 * disagreement,
            0.0,
            1.0
        )
        let classTotal = max(classMass.reduce(0.0, +), 1.0)
        let reliability = fxClamp(
            0.28 +
                0.30 * min(Double(steps) / 160.0, 1.0) +
                0.22 * confidence +
                0.12 * (moveReady ? 1.0 : 0.0) +
                0.08 * fxClamp((classTotal - Double(LabelClass.allCases.count)) / 256.0, 0.0, 1.0),
            0.0,
            1.0
        )
        let baseOutput = PluginModelOutputV4(
            classProbabilities: probabilities,
            moveMeanPoints: expectedMove,
            moveQ25Points: q25,
            moveQ50Points: q50,
            moveQ75Points: q75,
            mfeMeanPoints: max(q75, expectedMove * (1.0 + 0.25 * confidence)),
            maeMeanPoints: max(0.0, expectedMove * (0.30 + 0.30 * probabilities[LabelClass.skip.rawValue])),
            hitTimeFraction: fxClamp(0.68 - 0.28 * active + 0.12 * probabilities[LabelClass.skip.rawValue], 0.0, 1.0),
            pathRisk: fxClamp(0.35 * probabilities[LabelClass.skip.rawValue] + 0.25 * (1.0 - reliability) + 0.15 * windowVolatility(features: features), 0.0, 1.0),
            fillRisk: fxClamp(request.context.priceCostPoints / max(expectedMove + request.context.minMovePoints, 0.10), 0.0, 1.0),
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
            family: .convolutional,
            activityGate: active,
            structuralQuality: reliability,
            qualityPriors: qualityBank.priors(context: request.context),
            declaredWindowSize: request.windowSize
        )
        return PluginContextRuntimeTools.fillPrediction(modelOutput: output, calibratedMoveMeanPoints: expectedMove, context: request.context)
    }

    private mutating func seedWeights() {
        for h in 0..<Self.hiddenCount {
            hiddenBias[h] = Self.seed(11, h) * 0.08
            for i in 0..<Self.featureCount {
                hiddenWeights[h][i] = Self.seed(h + 1, i + 3) * 0.12
            }
        }
        for c in 0..<LabelClass.allCases.count {
            for h in 0..<(Self.hiddenCount + 1) {
                classWeights[c][h] = Self.seed(c + 23, h + 5) * 0.05
            }
        }
        for h in 0..<(Self.hiddenCount + 1) {
            moveWeights[h] = 0.04 * abs(Self.seed(41, h + 7))
        }
    }

    private func buildFeatures(x: [Double], window: [[Double]], context: PluginContextV4) -> [Double] {
        let slope = window.isEmpty ? Self.safeFeature(x, 1) : Self.windowSlope(window, feature: 1)
        let std = Self.windowStd(window, feature: 1)
        let range = Self.windowRange(window, feature: 1, count: min(32, max(1, window.count)))
        let delta = window.isEmpty ? Self.safeFeature(x, 7) : Self.windowRecentDelta(window, feature: 1, count: min(16, max(2, window.count)))
        let emaFast = window.isEmpty ? Self.safeFeature(x, 1) : Self.windowEMAMean(window, feature: 1, alpha: 0.55)
        let emaSlow = window.isEmpty ? Self.safeFeature(x, 2) : Self.windowEMAMean(window, feature: 2, alpha: 0.82)
        let arch = architectureSignal(x: x, window: window, slope: slope, delta: delta, emaFast: emaFast, emaSlow: emaSlow)
        return [
            1.0,
            Self.safeFeature(x, 1),
            Self.safeFeature(x, 2),
            Self.safeFeature(x, 3),
            Self.safeFeature(x, 4),
            Self.safeFeature(x, 7),
            Self.safeFeature(x, 12),
            fxClamp(0.65 * Self.safeFeature(x, 40) + 0.35 * Self.safeFeature(x, 6), -8.0, 8.0),
            slope,
            std,
            range,
            delta,
            emaFast,
            emaSlow,
            Self.safeFeature(x, FXDataEngineConstants.macroEventFeatureOffset + 14),
            Self.safeFeature(x, FXDataEngineConstants.macroEventFeatureOffset + 19),
            fxClamp(Double(context.horizonMinutes) / 60.0, 0.0, 2.0),
            fxClamp(Double(context.sessionBucket) / 5.0, 0.0, 1.0),
                fxClamp(Double(context.sequenceBars) / 96.0, 0.0, 2.0),
            arch.0,
            arch.1,
            arch.2,
            fxClamp(Self.safeFeature(x, 1) - Self.safeFeature(x, 2), -8.0, 8.0),
            fxClamp(Self.safeFeature(x, 2) - Self.safeFeature(x, 3), -8.0, 8.0)
        ].map { fxClamp(fxSafeFinite($0), -8.0, 8.0) }
    }

    private func architectureSignal(x: [Double], window: [[Double]], slope: Double, delta: Double, emaFast: Double, emaSlow: Double) -> (Double, Double, Double) {
        let mode = Self.architectureID % 6
        let energy = windowEnergy(window)
        switch mode {
        case 0:
            return (fxClamp(0.60 * emaFast + 0.40 * delta, -8.0, 8.0), fxClamp(energy, -8.0, 8.0), fxClamp(slope - emaSlow, -8.0, 8.0))
        case 1:
            return (fxClamp(0.55 * slope + 0.45 * Self.safeFeature(x, 12), -8.0, 8.0), fxClamp(emaFast - emaSlow, -8.0, 8.0), fxClamp(energy * slope, -8.0, 8.0))
        case 2:
            return (fxClamp(Self.windowAttention(window, fallback: slope), -8.0, 8.0), fxClamp(delta, -8.0, 8.0), fxClamp(emaSlow, -8.0, 8.0))
        case 3:
            return (fxClamp(0.50 * emaFast + 0.25 * emaSlow + 0.25 * slope, -8.0, 8.0), fxClamp(energy, -8.0, 8.0), fxClamp(Self.safeFeature(x, 4), -8.0, 8.0))
        case 4:
            return (fxClamp(0.70 * delta - 0.20 * energy, -8.0, 8.0), fxClamp(Self.safeFeature(x, 6), -8.0, 8.0), fxClamp(emaFast, -8.0, 8.0))
        default:
            return (fxClamp(0.35 * slope + 0.35 * delta + 0.30 * emaFast, -8.0, 8.0), fxClamp(emaFast - emaSlow, -8.0, 8.0), fxClamp(energy, -8.0, 8.0))
        }
    }

    private func hiddenActivations(features: [Double], window: [[Double]]) -> [Double] {
        var hidden = Array(repeating: 0.0, count: Self.hiddenCount)
        let recurrence = windowEnergy(window)
        for h in 0..<Self.hiddenCount {
            var value = hiddenBias[h]
            for i in 0..<Self.featureCount {
                value += hiddenWeights[h][i] * features[i]
            }
            value += 0.03 * recurrence * Self.seed(h + 17, Self.architectureID + 5)
            hidden[h] = tanh(value)
        }
        return hidden
    }

    private func rawProbabilities(hidden: [Double]) -> [Double] {
        var logits = Array(repeating: 0.0, count: LabelClass.allCases.count)
        let hb = hiddenWithBias(hidden)
        for c in 0..<LabelClass.allCases.count {
            logits[c] = Self.dot(classWeights[c], hb)
        }
        let maximum = logits.max() ?? 0.0
        let expValues = logits.map { exp(fxClamp($0 - maximum, -35.0, 35.0)) }
        let total = expValues.reduce(0.0, +)
        guard total > 0.0 else { return [0.10, 0.10, 0.80] }
        return expValues.map { $0 / total }
    }

    private mutating func updateClassHead(label: LabelClass, raw: [Double], hidden: [Double], sampleWeight: Double, hyperParameters: HyperParameters) {
        let hb = hiddenWithBias(hidden)
        let learningRate = fxClamp(hyperParameters.learningRate, 0.0002, 0.06) * fxClamp(sampleWeight, 0.1, 4.0)
        let l2 = fxClamp(hyperParameters.l2, 0.0, 0.10)
        for c in 0..<LabelClass.allCases.count {
            let target = c == label.rawValue ? 1.0 : 0.0
            let gradient = target - raw[c]
            for h in 0..<hb.count {
                classWeights[c][h] = fxClamp(classWeights[c][h] + learningRate * (gradient * hb[h] - l2 * classWeights[c][h]), -8.0, 8.0)
            }
        }
    }

    private mutating func updateMoveHead(targetMove: Double, hidden: [Double], sampleWeight: Double, hyperParameters: HyperParameters) {
        let hb = hiddenWithBias(hidden)
        let predicted = max(0.0, Self.dot(moveWeights, hb))
        let error = fxClamp(targetMove - predicted, -1000.0, 1000.0)
        let learningRate = fxClamp(hyperParameters.learningRate, 0.0002, 0.05) * 0.25 * fxClamp(sampleWeight, 0.1, 4.0)
        for h in 0..<hb.count {
            moveWeights[h] = fxClamp(moveWeights[h] + learningRate * error * hb[h], -1000.0, 1000.0)
        }
    }

    private mutating func updateMoveEMA(_ movePoints: Double) {
        let value = max(0.0, fxSafeFinite(movePoints))
        if moveReady {
            moveEMA = 0.97 * moveEMA + 0.03 * value
        } else {
            moveEMA = value
            moveReady = true
        }
    }

    private func hiddenWithBias(_ hidden: [Double]) -> [Double] {
        [1.0] + hidden
    }

    private func windowVolatility(features: [Double]) -> Double {
        max(0.01, 0.50 * max(features[9], 0.0) + 0.25 * abs(features[11]) + 0.05 * abs(features[7]))
    }

    private func windowEnergy(_ window: [[Double]]) -> Double {
        guard !window.isEmpty else { return 0.0 }
        let values = window.prefix(min(32, window.count)).map { row in Self.safeFeature(row, 1) }
        let mean = values.reduce(0.0, +) / Double(values.count)
        let energy = values.reduce(0.0) { $0 + abs($1 - mean) } / Double(values.count)
        return fxClamp(energy, -8.0, 8.0)
    }

    private static func windowAttention(_ window: [[Double]], fallback: Double) -> Double {
        guard !window.isEmpty else { return fallback }
        var weighted = 0.0
        var total = 0.0
        for (index, row) in window.prefix(min(32, window.count)).enumerated() {
            let value = safeFeature(row, 1)
            let weight = exp(-Double(index) / 12.0) * (1.0 + 0.05 * abs(safeFeature(row, 6)))
            weighted += weight * value
            total += weight
        }
        return total > 0.0 ? weighted / total : fallback
    }

    private static func preparedFeatures(_ x: [Double], dataHasVolume: Bool) -> [Double] {
        var output = Array(repeating: 0.0, count: FXDataEngineConstants.aiWeights)
        for index in 0..<FXDataEngineConstants.aiWeights {
            output[index] = fxClamp(index < x.count ? fxSafeFinite(x[index]) : 0.0, -50.0, 50.0)
        }
        if !dataHasVolume {
            for index in volumeFeatureIndexes where index < output.count { output[index] = 0.0 }
        }
        return output
    }

    private static func preparedWindow(_ window: [[Double]], dataHasVolume: Bool) -> [[Double]] {
        window.map { preparedFeatures($0, dataHasVolume: dataHasVolume) }
    }

    private static func safeFeature(_ x: [Double], _ index: Int) -> Double {
        guard index >= 0, index < x.count else { return 0.0 }
        return fxClamp(fxSafeFinite(x[index]), -50.0, 50.0)
    }

    private static func windowSlope(_ window: [[Double]], feature: Int) -> Double {
        let values = windowValues(window, feature: feature, count: min(window.count, 32))
        guard values.count >= 2 else { return 0.0 }
        return fxClamp((values[0] - values[values.count - 1]) / Double(values.count - 1), -8.0, 8.0)
    }

    private static func windowStd(_ window: [[Double]], feature: Int) -> Double {
        let values = windowValues(window, feature: feature, count: min(window.count, 32))
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
        let values = windowValues(window, feature: feature, count: min(window.count, 32))
        guard var ema = values.last else { return 0.0 }
        let clampedAlpha = fxClamp(alpha, 0.0, 1.0)
        for value in values.dropLast().reversed() {
            ema = clampedAlpha * value + (1.0 - clampedAlpha) * ema
        }
        return ema
    }

    private static func windowValues(_ window: [[Double]], feature: Int, count: Int) -> [Double] {
        guard feature >= 0, count > 0 else { return [] }
        return window.prefix(count).map { row in feature < row.count ? fxSafeFinite(row[feature]) : 0.0 }
    }

    private static func dot(_ lhs: [Double], _ rhs: [Double]) -> Double {
        zip(lhs, rhs).reduce(0.0) { $0 + $1.0 * $1.1 }
    }

    private static func seed(_ a: Int, _ b: Int) -> Double {
        sin(Double((AIModelID.mlpTiny.rawValue + 31) * (a + 3) * 67 + (b + 11) * 131 + Self.architectureID * 17))
    }
}
