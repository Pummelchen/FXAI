import FXDataEngine
import Foundation

public struct FXStupidFXDataEnginePlugin: FXAIPlannedPlugin {
    public var manifest: PluginManifestV4 {
        PluginManifestV4(
            aiID: AIModelID.demoFXStupid.rawValue,
            aiName: "fxbacktest_fxstupid",
            family: .ruleBased,
            referenceTier: .ruleBaseline,
            capabilityMask: [.selfTest, .multiHorizon],
            featureGroups: [.price, .volume, .microstructure],
            minHorizonMinutes: 1,
            maxHorizonMinutes: 240,
            minSequenceBars: 1,
            maxSequenceBars: 1,
            requiresVolumeWhenAvailable: true
        )
    }

    public var accelerationPlan: FXPluginAccelerationPlan {
        FXPluginAccelerationPlan(
            pluginName: "fxbacktest_fxstupid",
            primaryBackends: [.swiftScalar],
            usesVolumeWhenAvailable: true,
            notes: "FXBacktest demo adapter for the converted FXStupid EA. Keep scalar because the original flow is stateful order control; use volume as a confidence input when FXDataEngine has volume."
        )
    }

    public init() {}

    public mutating func reset() {}

    public func selfTest() -> Bool {
        (try? manifest.validate()) != nil
    }

    public mutating func train(_ request: TrainRequestV4, hyperParameters: HyperParameters) throws {
        try request.validate()
        try PluginContractTools.validateCompatibility(manifest: manifest, context: request.context)
    }

    public func predict(_ request: PredictRequestV4, hyperParameters: HyperParameters) throws -> PredictionV4 {
        try request.validate()
        try PluginContractTools.validateCompatibility(manifest: manifest, context: request.context)

        let shortReturn = Self.feature(request, 0)
        let slope = Self.feature(request, 3)
        let volumeSignal = request.context.dataHasVolume ? Self.feature(request, 6) : 0.0
        let volumeBias = request.context.dataHasVolume ? 0.10 * fxClamp(volumeSignal, -1.0, 1.0) : 0.0
        let edge = shortReturn + 0.50 * slope + volumeBias
        let strength = fxClamp(abs(edge) * 3.0 + abs(volumeBias), 0.0, 1.0)
        let moveSeed = max(request.context.minMovePoints, request.context.priceCostPoints, abs(edge) * 80.0)

        if edge > 0.01 {
            return Self.directionalPrediction(label: .buy, strength: strength, moveSeed: moveSeed, reliability: 0.52 + abs(volumeBias))
        }
        if edge < -0.01 {
            return Self.directionalPrediction(label: .sell, strength: strength, moveSeed: moveSeed, reliability: 0.52 + abs(volumeBias))
        }
        return Self.directionalPrediction(label: .skip, strength: 0.0, moveSeed: 0.0, reliability: 0.48)
    }

    private static func feature(_ request: PredictRequestV4, _ index: Int) -> Double {
        guard index >= 0, index < request.x.count else { return 0.0 }
        return fxSafeFinite(request.x[index])
    }

    private static func directionalPrediction(
        label: LabelClass,
        strength: Double,
        moveSeed: Double,
        reliability: Double
    ) -> PredictionV4 {
        let clampedStrength = fxClamp(strength, 0.0, 1.0)
        let directional = fxClamp(0.55 + 0.40 * clampedStrength, 0.55, 0.95)
        let opposite = 0.05
        let skip = max(0.02, 1.0 - directional - opposite)
        let probabilities: [Double]
        switch label {
        case .buy:
            probabilities = normalized([opposite, directional, skip])
        case .sell:
            probabilities = normalized([directional, opposite, skip])
        case .skip:
            probabilities = [0.08, 0.08, 0.84]
        }

        let meanMove = label == .skip ? 0.0 : max(1.0, moveSeed)
        let sigma = max(0.10, 0.30 * meanMove)
        let q25 = max(0.0, meanMove - 0.50 * sigma)
        let q50 = max(q25, meanMove)
        let q75 = max(q50, meanMove + 0.50 * sigma)
        return PredictionV4(
            classProbabilities: probabilities,
            moveMeanPoints: meanMove,
            moveQ25Points: q25,
            moveQ50Points: q50,
            moveQ75Points: q75,
            mfeMeanPoints: meanMove,
            maeMeanPoints: max(0.0, 0.35 * meanMove),
            hitTimeFraction: 1.0,
            pathRisk: probabilities[LabelClass.skip.rawValue],
            fillRisk: 0.0,
            confidence: max(probabilities[LabelClass.buy.rawValue], probabilities[LabelClass.sell.rawValue]),
            reliability: fxClamp(reliability, 0.0, 1.0)
        )
    }

    private static func normalized(_ probabilities: [Double]) -> [Double] {
        let sum = probabilities.reduce(0.0, +)
        guard sum > 0.0 else { return [0.08, 0.08, 0.84] }
        return probabilities.map { $0 / sum }
    }
}
