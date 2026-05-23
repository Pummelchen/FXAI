import FXDataEngine
import Foundation

public struct AIMythosRDTCPUModel: Sendable {
    private static let featureCount = 32
    private static let hiddenCount = 24
    private static let architectureID = 61
    private static let architectureMode = "mythosRDT"
    private static let volumeFeatureIndexes = [6, 68, 69, 70, 71, 74, 75, 76, 77, 78, 80, 81, 82, 83]

    private var steps: Int
    private var hiddenWeights: [[Double]]
    private var hiddenBias: [Double]
    private var classWeights: [[Double]]
    private var moveWeights: [Double]
    private var valueWeights: [Double]
    private var classMass: [Double]
    private var moveEMA: Double
    private var moveReady: Bool
    private var recurrentState: [Double]
    private var calibrator: PluginTernaryCalibrator
    private var qualityBank: PluginQualityBank

    public init() {
        self.steps = 0
        self.hiddenWeights = Array(repeating: Array(repeating: 0.0, count: Self.featureCount), count: Self.hiddenCount)
        self.hiddenBias = Array(repeating: 0.0, count: Self.hiddenCount)
        self.classWeights = Array(repeating: Array(repeating: 0.0, count: Self.hiddenCount + 1), count: LabelClass.allCases.count)
        self.moveWeights = Array(repeating: 0.0, count: Self.hiddenCount + 1)
        self.valueWeights = Array(repeating: 0.0, count: Self.hiddenCount + 1)
        self.classMass = Array(repeating: 1.0, count: LabelClass.allCases.count)
        self.moveEMA = 0.0
        self.moveReady = false
        self.recurrentState = Array(repeating: 0.0, count: Self.hiddenCount)
        self.calibrator = PluginTernaryCalibrator()
        self.qualityBank = PluginQualityBank()
        seedWeights()
    }

    public mutating func reset() {
        self = AIMythosRDTCPUModel()
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
        updateRecurrentState(hidden: hidden)
        let raw = rawProbabilities(hidden: hidden)
        let valuePrediction = Self.dot(valueWeights, hiddenWithBias(hidden))
        let reward = rewardTarget(label: label, movePoints: request.movePoints, context: request.context)
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
        updateValueHead(targetReward: reward, predictedReward: valuePrediction, hidden: hidden, sampleWeight: sampleWeight, hyperParameters: hyperParameters)
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
        var probabilities = PluginContextRuntimeTools.normalizeClassDistribution(calibrator.calibrated(raw))
        probabilities = applyArchitectureRiskGate(probabilities, features: features, hidden: hidden, context: request.context)

        let predictedMove = max(0.0, Self.dot(moveWeights, hiddenWithBias(hidden)))
        let active = fxClamp(1.0 - probabilities[LabelClass.skip.rawValue], 0.0, 1.0)
        let baseMove = max(predictedMove, moveReady ? moveEMA : request.context.minMovePoints)
        let expectedMove = max(0.0, baseMove * max(active, 0.15))
        let disagreement = abs(probabilities[LabelClass.buy.rawValue] - probabilities[LabelClass.sell.rawValue])
        let vol = windowVolatility(features: features)
        let sigma = max(0.10, 0.35 * expectedMove + vol + 0.20 * request.context.minMovePoints)
        let q25 = max(0.0, expectedMove - quantileSkew(features: features) * sigma)
        let q50 = max(q25, expectedMove)
        let q75 = max(q50, expectedMove + (0.55 + 0.10 * active) * sigma)
        let valueScore = PluginSupportTools.sigmoid(Self.dot(valueWeights, hiddenWithBias(hidden)))
        let confidence = fxClamp(
            0.50 * max(probabilities[LabelClass.buy.rawValue], probabilities[LabelClass.sell.rawValue]) +
                0.20 * active +
                0.18 * disagreement +
                0.12 * valueScore,
            0.0,
            1.0
        )
        let classTotal = max(classMass.reduce(0.0, +), 1.0)
        let reliability = fxClamp(
            0.26 +
                0.30 * min(Double(steps) / 180.0, 1.0) +
                0.22 * confidence +
                0.12 * (moveReady ? 1.0 : 0.0) +
                0.10 * fxClamp((classTotal - Double(LabelClass.allCases.count)) / 320.0, 0.0, 1.0),
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
            maeMeanPoints: max(0.0, expectedMove * (0.28 + 0.32 * probabilities[LabelClass.skip.rawValue] + 0.08 * vol)),
            hitTimeFraction: fxClamp(0.70 - 0.30 * active + 0.10 * probabilities[LabelClass.skip.rawValue], 0.0, 1.0),
            pathRisk: fxClamp(0.32 * probabilities[LabelClass.skip.rawValue] + 0.24 * (1.0 - reliability) + 0.14 * vol, 0.0, 1.0),
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
            family: .transformer,
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
            recurrentState[h] = 0.0
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
            valueWeights[h] = 0.03 * Self.seed(61, h + 13)
        }
    }

    private func buildFeatures(x: [Double], window: [[Double]], context: PluginContextV4) -> [Double] {
        let slope = window.isEmpty ? Self.safeFeature(x, 1) : Self.windowSlope(window, feature: 1)
        let std = Self.windowStd(window, feature: 1)
        let range = Self.windowRange(window, feature: 1, count: min(32, max(1, window.count)))
        let delta = window.isEmpty ? Self.safeFeature(x, 7) : Self.windowRecentDelta(window, feature: 1, count: min(16, max(2, window.count)))
        let emaFast = window.isEmpty ? Self.safeFeature(x, 1) : Self.windowEMAMean(window, feature: 1, alpha: 0.55)
        let emaSlow = window.isEmpty ? Self.safeFeature(x, 2) : Self.windowEMAMean(window, feature: 2, alpha: 0.82)
        let attention = Self.windowAttention(window, fallback: slope)
        let longDelta = Self.windowRecentDelta(window, feature: 1, count: min(48, max(2, window.count)))
        let volumePulse = context.dataHasVolume ? Self.volumePulse(x: x, window: window) : 0.0
        let arch = architectureSignals(x: x, window: window, slope: slope, std: std, delta: delta, emaFast: emaFast, emaSlow: emaSlow, attention: attention, longDelta: longDelta, volumePulse: volumePulse, context: context)
        return [
            1.0,
            Self.safeFeature(x, 1),
            Self.safeFeature(x, 2),
            Self.safeFeature(x, 3),
            Self.safeFeature(x, 4),
            Self.safeFeature(x, 7),
            Self.safeFeature(x, 12),
            fxClamp(0.65 * Self.safeFeature(x, 40) + 0.35 * volumePulse, -8.0, 8.0),
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
            attention,
            longDelta,
            volumePulse,
            fxClamp(Self.safeFeature(x, 1) - Self.safeFeature(x, 2), -8.0, 8.0),
            fxClamp(Self.safeFeature(x, 2) - Self.safeFeature(x, 3), -8.0, 8.0),
            arch.0,
            arch.1,
            arch.2,
            arch.3,
            arch.4,
            arch.5,
            arch.6,
            arch.7
        ].map { fxClamp(fxSafeFinite($0), -8.0, 8.0) }
    }

    private func architectureSignals(x: [Double], window: [[Double]], slope: Double, std: Double, delta: Double, emaFast: Double, emaSlow: Double, attention: Double, longDelta: Double, volumePulse: Double, context: PluginContextV4) -> (Double, Double, Double, Double, Double, Double, Double, Double) {
        let energy = windowEnergy(window)
        let curvature = Self.windowCurvature(window, feature: 1)
        let macro = 0.5 * Self.safeFeature(x, FXDataEngineConstants.macroEventFeatureOffset + 14) + 0.5 * Self.safeFeature(x, FXDataEngineConstants.macroEventFeatureOffset + 19)
        switch Self.architectureMode {
        case "recurrent":
            let gate = PluginSupportTools.sigmoid(0.8 * slope + 0.4 * delta - 0.2 * energy)
            return (gate * emaFast + (1.0 - gate) * emaSlow, delta, energy, slope - emaSlow, gate, longDelta, volumePulse, macro)
        case "gatedRecurrent":
            let forget = PluginSupportTools.sigmoid(0.6 * emaSlow - 0.4 * energy + 0.2 * volumePulse)
            let input = PluginSupportTools.sigmoid(0.7 * delta + 0.3 * attention)
            return (input * attention, forget * emaSlow, input - forget, energy, curvature, longDelta, volumePulse, macro)
        case "gru":
            let reset = PluginSupportTools.sigmoid(0.7 * emaFast - 0.3 * emaSlow)
            let update = PluginSupportTools.sigmoid(0.6 * delta + 0.2 * volumePulse)
            return (update * delta, reset * emaSlow, attention, energy, update - reset, curvature, volumePulse, macro)
        case "bidirectional":
            let reverseSlope = window.isEmpty ? -slope : -Self.windowSlope(window.reversed(), feature: 1)
            return (slope, reverseSlope, attention, emaFast - emaSlow, longDelta, energy, volumePulse, macro)
        case "lstmTCN", "cnnLSTM", "attentionCNNBiLSTM", "tcn":
            let convFast = Self.windowConvolution(window, feature: 1, kernel: [0.58, 0.27, 0.15])
            let convSlow = Self.windowConvolution(window, feature: 1, kernel: [0.34, 0.24, 0.18, 0.14, 0.10])
            let attentionMix = 0.60 * attention + 0.40 * convFast
            return (convFast, convSlow, attentionMix, convFast - convSlow, curvature, energy, volumePulse, macro)
        case "s4", "stmn", "fewc", "gha", "tensorTesseract":
            let stateFast = Self.windowStateSpace(window, feature: 1, decay: 0.62)
            let stateSlow = Self.windowStateSpace(window, feature: 2, decay: 0.88)
            let memory = 0.55 * stateFast + 0.30 * stateSlow + 0.15 * recurrentSummary()
            return (stateFast, stateSlow, memory, memory - emaSlow, curvature, energy, volumePulse, macro)
        case "transformer", "temporalFusionTransformer", "autoformer", "patchTransformer", "causalTokenForecaster", "foundationForecaster", "geodesicAttention":
            let seasonal = Self.windowSeasonal(window, feature: 1)
            let patch = Self.windowPatchContrast(window, feature: 1)
            let distance = Self.geodesicDistance(window, featureA: 1, featureB: 2)
            return (attention, patch, seasonal, attention - seasonal, distance, energy, volumePulse, macro)
        case "mythosRDT":
            let refined = mythosRefinement(slope: slope, delta: delta, attention: attention, energy: energy, macro: macro, volumePulse: volumePulse)
            return (refined.0, refined.1, refined.2, refined.3, refined.4, refined.5, refined.6, refined.7)
        case "currencyFactorWorld", "graphWorld":
            let relative = (Self.safeFeature(x, 13) + Self.safeFeature(x, 14) + Self.safeFeature(x, 15)) / 3.0
            let graphCarry = (Self.safeFeature(x, 1) - Self.safeFeature(x, 2)) + (Self.safeFeature(x, 2) - Self.safeFeature(x, 3))
            let structural = 0.45 * attention + 0.35 * relative + 0.20 * macro
            return (structural, graphCarry, slope - relative, longDelta, energy, volumePulse, macro, Double(context.regimeID) / 8.0)
        case "ppoPolicy":
            let advantage = delta - max(0.0, context.priceCostPoints) / max(context.minMovePoints, 0.10)
            let entropyProxy = 1.0 / (1.0 + abs(slope) + abs(delta))
            return (advantage, attention, entropyProxy, energy, volumePulse, macro, Double(context.sessionBucket) / 5.0, longDelta)
        case "qcew":
            let lower = emaFast - 0.67 * max(std, 0.0)
            let upper = emaFast + 0.67 * max(std, 0.0)
            return (lower, emaFast, upper, upper - lower, delta, energy, volumePulse, macro)
        case "trendReversalRecurrent":
            let reversal = -0.55 * slope + 0.45 * curvature
            let trend = 0.65 * emaFast + 0.35 * longDelta
            return (trend, reversal, trend + reversal, attention, energy, volumePulse, macro, delta)
        default:
            return (0.35 * slope + 0.35 * delta + 0.30 * emaFast, emaFast - emaSlow, energy, attention, curvature, longDelta, volumePulse, macro)
        }
    }

    private func mythosRefinement(slope: Double, delta: Double, attention: Double, energy: Double, macro: Double, volumePulse: Double) -> (Double, Double, Double, Double, Double, Double, Double, Double) {
        var latent = [slope, delta, attention, energy, macro, volumePulse, recurrentSummary(), 1.0]
        let difficulty = fxClamp(abs(delta) + 0.5 * energy + 0.25 * abs(macro), 0.0, 4.0)
        let loops = max(2, min(8, 2 + Int(difficulty * 2.0)))
        for depth in 0..<loops {
            for i in 0..<latent.count {
                let memory = recurrentState.isEmpty ? 0.0 : recurrentState[(i + depth) % recurrentState.count]
                let injected = 0.58 * latent[i] + 0.24 * memory + 0.18 * Self.seed(depth + 3, i + 5) * difficulty
                latent[i] = tanh(fxClamp(injected, -8.0, 8.0))
            }
            let router = PluginSupportTools.sigmoid(latent[0] + 0.5 * latent[2] - 0.35 * latent[3])
            latent[1] = router * latent[1] + (1.0 - router) * latent[4]
            latent[2] = 0.60 * latent[2] + 0.40 * tanh(latent[0] - latent[3])
        }
        return (latent[0], latent[1], latent[2], latent[3], latent[4], latent[5], latent[6], fxClamp(Double(loops) / 8.0, 0.0, 1.0))
    }

    private func hiddenActivations(features: [Double], window: [[Double]]) -> [Double] {
        FXAISequenceReferenceEncoders.encode(
            architectureMode: Self.architectureMode,
            features: features,
            window: window,
            recurrentState: recurrentState,
            hiddenBias: hiddenBias,
            hiddenWeights: hiddenWeights,
            hiddenCount: Self.hiddenCount,
            featureCount: Self.featureCount,
            architectureID: Self.architectureID
        )
    }

    private mutating func updateRecurrentState(hidden: [Double]) {
        let decay = stateDecay()
        for h in 0..<min(Self.hiddenCount, hidden.count) {
            recurrentState[h] = fxClamp(decay * recurrentState[h] + (1.0 - decay) * hidden[h], -4.0, 4.0)
        }
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

    private func applyArchitectureRiskGate(_ probabilities: [Double], features: [Double], hidden: [Double], context: PluginContextV4) -> [Double] {
        var p = probabilities
        let volatility = max(0.0, features[9])
        let macro = abs(features[14]) + abs(features[15])
        var skipBoost = 0.0
        if Self.architectureMode == "ppoPolicy" {
            let value = PluginSupportTools.sigmoid(Self.dot(valueWeights, hiddenWithBias(hidden)))
            skipBoost += value < 0.48 ? 0.08 : -0.03
            skipBoost += context.priceCostPoints > context.minMovePoints ? 0.06 : 0.0
        }
        if Self.architectureMode == "mythosRDT" {
            skipBoost += volatility > 1.5 ? 0.05 : 0.0
            skipBoost += abs(features[27]) < 0.05 ? 0.03 : 0.0
        }
        skipBoost += fxClamp(0.025 * volatility + 0.015 * macro, 0.0, 0.12)
        p[LabelClass.skip.rawValue] = fxClamp(p[LabelClass.skip.rawValue] + skipBoost, 0.01, 0.98)
        let directional = max(1.0 - p[LabelClass.skip.rawValue], 0.02)
        let sideTotal = max(p[LabelClass.buy.rawValue] + p[LabelClass.sell.rawValue], 1.0e-9)
        p[LabelClass.buy.rawValue] = directional * p[LabelClass.buy.rawValue] / sideTotal
        p[LabelClass.sell.rawValue] = directional * p[LabelClass.sell.rawValue] / sideTotal
        return PluginContextRuntimeTools.normalizeClassDistribution(p)
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

    private mutating func updateValueHead(targetReward: Double, predictedReward: Double, hidden: [Double], sampleWeight: Double, hyperParameters: HyperParameters) {
        let hb = hiddenWithBias(hidden)
        let error = fxClamp(targetReward - predictedReward, -4.0, 4.0)
        let learningRate = fxClamp(hyperParameters.learningRate, 0.0002, 0.04) * 0.20 * fxClamp(sampleWeight, 0.1, 4.0)
        for h in 0..<hb.count {
            valueWeights[h] = fxClamp(valueWeights[h] + learningRate * error * hb[h], -8.0, 8.0)
        }
    }

    private func rewardTarget(label: LabelClass, movePoints: Double, context: PluginContextV4) -> Double {
        let signed = label == .buy ? movePoints : (label == .sell ? -movePoints : 0.0)
        let net = abs(signed) - max(context.priceCostPoints, 0.0)
        return fxClamp(label == .skip ? -0.10 : net / max(context.minMovePoints, 0.10), -3.0, 3.0)
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
        max(0.01, 0.50 * max(features[9], 0.0) + 0.25 * abs(features[11]) + 0.05 * abs(features[21]))
    }

    private func quantileSkew(features: [Double]) -> Double {
        if Self.architectureMode == "qcew" { return 0.45 + 0.15 * fxClamp(abs(features[27]), 0.0, 1.0) }
        return 0.55
    }

    private func stateDecay() -> Double {
        switch Self.architectureMode {
        case "s4", "stmn", "fewc", "gha", "tensorTesseract": return 0.88
        case "recurrent", "gatedRecurrent", "gru", "bidirectional", "trendReversalRecurrent": return 0.78
        default: return 0.70
        }
    }

    private func recurrentSummary() -> Double {
        guard !recurrentState.isEmpty else { return 0.0 }
        return recurrentState.reduce(0.0, +) / Double(recurrentState.count)
    }

    private func windowEnergy(_ window: [[Double]]) -> Double {
        guard !window.isEmpty else { return 0.0 }
        let values = window.prefix(min(48, window.count)).map { row in Self.safeFeature(row, 1) }
        let mean = values.reduce(0.0, +) / Double(values.count)
        let energy = values.reduce(0.0) { $0 + abs($1 - mean) } / Double(values.count)
        return fxClamp(energy, -8.0, 8.0)
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
        let values = windowValues(window, feature: feature, count: min(window.count, 48))
        guard values.count >= 2 else { return 0.0 }
        return fxClamp((values[0] - values[values.count - 1]) / Double(values.count - 1), -8.0, 8.0)
    }

    private static func windowStd(_ window: [[Double]], feature: Int) -> Double {
        let values = windowValues(window, feature: feature, count: min(window.count, 48))
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
        let values = windowValues(window, feature: feature, count: min(window.count, 48))
        guard var ema = values.last else { return 0.0 }
        let clampedAlpha = fxClamp(alpha, 0.0, 1.0)
        for value in values.dropLast().reversed() {
            ema = clampedAlpha * value + (1.0 - clampedAlpha) * ema
        }
        return ema
    }

    private static func windowAttention(_ window: [[Double]], fallback: Double) -> Double {
        guard !window.isEmpty else { return fallback }
        var weighted = 0.0
        var total = 0.0
        for (index, row) in window.prefix(min(48, window.count)).enumerated() {
            let value = safeFeature(row, 1)
            let key = abs(safeFeature(row, 2)) + 0.15 * abs(safeFeature(row, 6))
            let weight = exp(-Double(index) / 14.0) * (1.0 + 0.08 * key)
            weighted += weight * value
            total += weight
        }
        return total > 0.0 ? weighted / total : fallback
    }

    private static func windowConvolution(_ window: [[Double]], feature: Int, kernel: [Double]) -> Double {
        guard !window.isEmpty, !kernel.isEmpty else { return 0.0 }
        var value = 0.0
        var total = 0.0
        for (index, weight) in kernel.enumerated() where index < window.count {
            value += weight * safeFeature(window[index], feature)
            total += abs(weight)
        }
        return total > 0.0 ? value / total : 0.0
    }

    private static func windowStateSpace(_ window: [[Double]], feature: Int, decay: Double) -> Double {
        let values = windowValues(window, feature: feature, count: min(window.count, 64)).reversed()
        var state = 0.0
        let alpha = fxClamp(1.0 - decay, 0.02, 0.95)
        for value in values {
            state = decay * state + alpha * value
        }
        return fxClamp(state, -8.0, 8.0)
    }

    private static func windowSeasonal(_ window: [[Double]], feature: Int) -> Double {
        let values = windowValues(window, feature: feature, count: min(window.count, 48))
        guard values.count >= 6 else { return 0.0 }
        var even = 0.0
        var odd = 0.0
        for (index, value) in values.enumerated() {
            if index % 2 == 0 { even += value } else { odd += value }
        }
        return fxClamp((even - odd) / Double(values.count), -8.0, 8.0)
    }

    private static func windowPatchContrast(_ window: [[Double]], feature: Int) -> Double {
        let values = windowValues(window, feature: feature, count: min(window.count, 48))
        guard values.count >= 8 else { return 0.0 }
        let patch = max(2, values.count / 4)
        let recent = values.prefix(patch).reduce(0.0, +) / Double(patch)
        let older = values.suffix(patch).reduce(0.0, +) / Double(patch)
        return fxClamp(recent - older, -8.0, 8.0)
    }

    private static func geodesicDistance(_ window: [[Double]], featureA: Int, featureB: Int) -> Double {
        let count = min(window.count, 32)
        guard count > 1 else { return 0.0 }
        var distance = 0.0
        for row in window.prefix(count) {
            let a = safeFeature(row, featureA)
            let b = safeFeature(row, featureB)
            distance += sqrt(max(0.0, a * a + b * b))
        }
        return fxClamp(distance / Double(count), -8.0, 8.0)
    }

    private static func windowCurvature(_ window: [[Double]], feature: Int) -> Double {
        let values = windowValues(window, feature: feature, count: min(window.count, 16))
        guard values.count >= 3 else { return 0.0 }
        return fxClamp(values[0] - 2.0 * values[1] + values[2], -8.0, 8.0)
    }

    private static func volumePulse(x: [Double], window: [[Double]]) -> Double {
        let current = safeFeature(x, 6)
        guard !window.isEmpty else { return current }
        let values = windowValues(window, feature: 6, count: min(window.count, 32))
        guard !values.isEmpty else { return current }
        let mean = values.reduce(0.0, +) / Double(values.count)
        return fxClamp(0.65 * current + 0.35 * (current - mean), -8.0, 8.0)
    }

    private static func windowValues(_ window: [[Double]], feature: Int, count: Int) -> [Double] {
        guard feature >= 0, count > 0 else { return [] }
        return window.prefix(count).map { row in feature < row.count ? fxSafeFinite(row[feature]) : 0.0 }
    }

    private static func dot(_ lhs: [Double], _ rhs: [Double]) -> Double {
        zip(lhs, rhs).reduce(0.0) { $0 + $1.0 * $1.1 }
    }

    private static func seed(_ a: Int, _ b: Int) -> Double {
        sin(Double((AIModelID.mythosRDT.rawValue + 31) * (a + 3) * 67 + (b + 11) * 131 + Self.architectureID * 17))
    }
}
