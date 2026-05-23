import FXDataEngine
import Foundation

public struct FX7FXDataEnginePlugin: FXAIPlannedPlugin {
    public var manifest: PluginManifestV4 {
        PluginManifestV4(
            aiID: AIModelID.demoFX7.rawValue,
            aiName: "fx7",
            family: .ruleBased,
            referenceTier: .ruleBaseline,
            capabilityMask: [.selfTest, .windowContext, .multiHorizon],
            featureSchema: .contextual,
            featureGroups: [.price, .multiTimeframe, .volatility, .time, .context, .volume, .microstructure, .filters],
            minHorizonMinutes: 1,
            maxHorizonMinutes: 240,
            minSequenceBars: 1,
            maxSequenceBars: FXDataEngineConstants.maxSequenceBars,
            requiresVolumeWhenAvailable: true
        )
    }

    public var accelerationPlan: FXPluginAccelerationPlan {
        FXPluginAccelerationPlan(
            pluginName: "fx7",
            primaryBackends: [.swiftScalar, .metal],
            candidateBackends: [.swiftSIMD, .accelerate],
            usesVolumeWhenAvailable: true,
            notes: "FX7 plugin-zoo adapter. CPU is the deterministic reference; Metal scores the signal grid from the same volume-aware momentum, regime, volatility, and cost-gate core used by the Swift path."
        )
    }

    public init() {}

    public mutating func reset() {}

    public func selfTest() -> Bool {
        do {
            try manifest.validate()
            let request = PredictRequestV4(
                valid: true,
                context: PluginContextV4(
                    horizonMinutes: 15,
                    sequenceBars: 1,
                    priceCostPoints: 0.35,
                    minMovePoints: 0.75,
                    dataHasVolume: true
                ),
                x: Self.selfTestFeatures()
            )
            let prediction = try predict(request, hyperParameters: HyperParameters())
            try prediction.validate()
            return prediction.classProbabilities[LabelClass.buy.rawValue] > prediction.classProbabilities[LabelClass.sell.rawValue]
        } catch {
            return false
        }
    }

    public mutating func train(_ request: TrainRequestV4, hyperParameters: HyperParameters) throws {
        try request.validate()
        try PluginContractTools.validateCompatibility(manifest: manifest, context: request.context)
    }

    public func predict(_ request: PredictRequestV4, hyperParameters: HyperParameters) throws -> PredictionV4 {
        try request.validate()
        try PluginContractTools.validateCompatibility(manifest: manifest, context: request.context)

        let snapshot = FX7SignalSnapshot(request: request)
        let score = FX7SignalScorer.score(snapshot: snapshot, context: request.context)
        let moveSeed = max(
            request.context.minMovePoints,
            request.context.priceCostPoints,
            abs(score.edge) * 120.0
        )

        if score.confidence < 0.30 || abs(score.edge) < 0.015 {
            return Self.directionalPrediction(
                label: .skip,
                strength: 0.0,
                moveSeed: 0.0,
                reliability: max(0.45, score.confidence)
            )
        }
        return Self.directionalPrediction(
            label: score.edge > 0.0 ? .buy : .sell,
            strength: score.confidence,
            moveSeed: moveSeed,
            reliability: score.reliability
        )
    }

    private static func selfTestFeatures() -> [Double] {
        var features = Array(repeating: 0.0, count: FXDataEngineConstants.aiWeights)
        features[0] = 0.08
        features[3] = 0.05
        features[4] = 0.18
        features[6] = 0.60
        features[7] = 0.14
        features[8] = 0.03
        features[12] = 0.20
        features[40] = 0.65
        return features
    }

    private static func directionalPrediction(
        label: LabelClass,
        strength: Double,
        moveSeed: Double,
        reliability: Double
    ) -> PredictionV4 {
        let clampedStrength = fxClamp(strength, 0.0, 1.0)
        let directional = fxClamp(0.54 + 0.40 * clampedStrength, 0.54, 0.96)
        let opposite = 0.04
        let skip = max(0.02, 1.0 - directional - opposite)
        let probabilities: [Double]
        switch label {
        case .buy:
            probabilities = normalized([opposite, directional, skip])
        case .sell:
            probabilities = normalized([directional, opposite, skip])
        case .skip:
            probabilities = [0.07, 0.07, 0.86]
        }

        let meanMove = label == .skip ? 0.0 : max(1.0, moveSeed)
        let sigma = max(0.10, 0.28 * meanMove)
        let q25 = max(0.0, meanMove - 0.50 * sigma)
        let q50 = max(q25, meanMove)
        let q75 = max(q50, meanMove + 0.50 * sigma)
        return PredictionV4(
            classProbabilities: probabilities,
            moveMeanPoints: meanMove,
            moveQ25Points: q25,
            moveQ50Points: q50,
            moveQ75Points: q75,
            mfeMeanPoints: meanMove * 1.05,
            maeMeanPoints: max(0.0, 0.32 * meanMove),
            hitTimeFraction: label == .skip ? 1.0 : 0.72,
            pathRisk: probabilities[LabelClass.skip.rawValue],
            fillRisk: 0.02 * clampedStrength,
            confidence: max(probabilities[LabelClass.buy.rawValue], probabilities[LabelClass.sell.rawValue]),
            reliability: fxClamp(reliability, 0.0, 1.0)
        )
    }

    private static func normalized(_ probabilities: [Double]) -> [Double] {
        let sum = probabilities.reduce(0.0, +)
        guard sum > 0.0 else { return [0.07, 0.07, 0.86] }
        return probabilities.map { $0 / sum }
    }
}

private struct FX7SignalSnapshot {
    let shortReturn: Double
    let mediumSlope: Double
    let fastReturn: Double
    let slowReturn: Double
    let volatility: Double
    let contextSignal: Double
    let volumeSignal: Double
    let windowMomentum: Double

    init(request: PredictRequestV4) {
        shortReturn = Self.feature(request.x, 0)
        mediumSlope = Self.feature(request.x, 3)
        fastReturn = Self.feature(request.x, 7)
        slowReturn = Self.feature(request.x, 8)
        volatility = max(abs(Self.feature(request.x, 4)) + 0.50 * abs(Self.feature(request.x, 5)), 0.01)
        contextSignal = Self.feature(request.x, 12)
        if request.context.dataHasVolume {
            volumeSignal = fxClamp(0.65 * Self.feature(request.x, 40) + 0.35 * Self.feature(request.x, 6), -1.0, 1.0)
        } else {
            volumeSignal = 0.0
        }
        windowMomentum = Self.windowMomentum(request)
    }

    private static func feature(_ features: [Double], _ index: Int) -> Double {
        guard index >= 0, index < features.count else { return 0.0 }
        return fxSafeFinite(features[index])
    }

    private static func windowMomentum(_ request: PredictRequestV4) -> Double {
        guard !request.xWindow.isEmpty else { return 0.0 }
        let rows = request.xWindow.prefix(min(16, request.xWindow.count))
        var weighted = 0.0
        var weightSum = 0.0
        for (offset, row) in rows.enumerated() {
            let weight = 1.0 / Double(offset + 1)
            weighted += weight * (0.55 * feature(row, 7) + 0.45 * feature(row, 3))
            weightSum += weight
        }
        guard weightSum > 0.0 else { return 0.0 }
        return fxClamp(weighted / weightSum, -1.0, 1.0)
    }
}

private enum FX7SignalScorer {
    struct Score {
        let edge: Double
        let confidence: Double
        let reliability: Double
    }

    static func score(snapshot: FX7SignalSnapshot, context: PluginContextV4) -> Score {
        let mtfEdge = snapshot.fastReturn - snapshot.slowReturn
        let trendInput = 0.38 * snapshot.fastReturn +
            0.30 * snapshot.slowReturn +
            0.18 * snapshot.mediumSlope +
            0.09 * snapshot.shortReturn +
            0.05 * snapshot.windowMomentum
        let trend = tanh(2.4 * trendInput)
        let breakout = tanh(4.0 * mtfEdge)
        let alignment = directionalAlignment(snapshot)
        let efficiency = fxClamp(abs(mtfEdge) / (snapshot.volatility + abs(snapshot.shortReturn) + 0.01), 0.0, 2.0)
        let volGate = 1.0 / (1.0 + max(snapshot.volatility - 1.50, 0.0) * 0.60)
        let panicPenalty = fxClamp(max(snapshot.volatility - 1.80, 0.0) * 0.25, 0.0, 0.45)
        let reversalPenalty = fxClamp(max(0.0, -trend * snapshot.shortReturn) * 0.35, 0.0, 0.35)
        let volumeBias = context.dataHasVolume ? 0.08 * snapshot.volumeSignal : 0.0
        let contextBias = 0.06 * snapshot.contextSignal
        let rawEdge = 0.56 * trend +
            0.20 * breakout +
            0.08 * alignment +
            contextBias +
            volumeBias
        let costScale = max(context.priceCostPoints, context.minMovePoints, 0.10) / 100.0
        let costGate = 1.0 / (1.0 + exp(-8.0 * (abs(rawEdge) - costScale)))
        let signedPenalty = fxSign(rawEdge) * (panicPenalty + reversalPenalty)
        let edge = fxClamp(rawEdge * volGate * costGate - signedPenalty, -1.0, 1.0)
        let confidence = fxClamp(
            0.25 +
                0.30 * abs(edge) +
                0.18 * min(efficiency, 1.0) +
                0.12 * abs(alignment) +
                0.08 * abs(snapshot.volumeSignal) +
                0.07 * costGate,
            0.0,
            1.0
        )
        let reliability = fxClamp(0.48 + 0.34 * confidence + 0.08 * volGate + 0.06 * costGate, 0.0, 1.0)
        return Score(edge: edge, confidence: confidence, reliability: reliability)
    }

    private static func directionalAlignment(_ snapshot: FX7SignalSnapshot) -> Double {
        let components = [
            snapshot.shortReturn,
            snapshot.mediumSlope,
            snapshot.fastReturn,
            snapshot.slowReturn,
            snapshot.windowMomentum
        ]
        let signs = components.map(fxSign)
        let nonZero = signs.filter { $0 != 0.0 }
        guard !nonZero.isEmpty else { return 0.0 }
        return fxClamp(nonZero.reduce(0.0, +) / Double(nonZero.count), -1.0, 1.0)
    }
}
