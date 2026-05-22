import FXDataEngine
import Foundation

public struct RuleRandomPlugin: FXAIPlannedPlugin {
    public let manifest = PluginManifestV4(
        aiID: AIModelID.randomNoSkip.rawValue,
        aiName: "rule_random",
        family: .ruleBased,
        referenceTier: .ruleBaseline,
        capabilityMask: [.selfTest, .multiHorizon],
        featureGroups: .all,
        minHorizonMinutes: 1,
        maxHorizonMinutes: 240,
        minSequenceBars: 1,
        maxSequenceBars: 1,
        requiresVolumeWhenAvailable: true
    )

    public let accelerationPlan = FXPluginAccelerationPlan(
        pluginName: "rule_random",
        primaryBackends: [.swiftScalar],
        usesVolumeWhenAvailable: true,
        notes: "Deterministic no-skip random baseline. Accepts the volume-aware plugin contract while preserving legacy hash behavior; hashing is scalar and not worth GPU or tensor dispatch."
    )

    public init() {}

    public mutating func reset() {}

    public mutating func train(_ request: TrainRequestV4, hyperParameters: HyperParameters) throws {
        try request.validate()
        try PluginContractTools.validateCompatibility(manifest: manifest, context: request.context)
    }

    public func predict(_ request: PredictRequestV4, hyperParameters: HyperParameters) throws -> PredictionV4 {
        try request.validate()
        try PluginContractTools.validateCompatibility(manifest: manifest, context: request.context)

        let buySide = Self.deterministicRandomBuySide(request: request)
        return Self.fixedDirectionalPrediction(
            request: request,
            buyProbability: buySide ? 0.995 : 0.005,
            sellProbability: buySide ? 0.005 : 0.995,
            confidence: 0.995,
            sigmaScale: 0.35,
            quantileScale: 0.55,
            reliability: 0.50
        )
    }

    private static func fixedDirectionalPrediction(
        request: PredictRequestV4,
        buyProbability: Double,
        sellProbability: Double,
        confidence: Double,
        sigmaScale: Double,
        quantileScale: Double,
        reliability: Double
    ) -> PredictionV4 {
        let minimumMove = request.context.minMovePoints > 0.0
            ? request.context.minMovePoints
            : max(request.context.priceCostPoints, 0.10)
        let meanMove = max(1.0, 3.0 * minimumMove + 0.25)
        let sigma = max(0.10, sigmaScale * meanMove)
        let q25 = max(0.0, meanMove - quantileScale * sigma)
        let q50 = max(q25, meanMove)
        let q75 = max(q50, meanMove + quantileScale * sigma)

        return PredictionV4(
            classProbabilities: [sellProbability, buyProbability, 0.0],
            moveMeanPoints: meanMove,
            moveQ25Points: q25,
            moveQ50Points: q50,
            moveQ75Points: q75,
            mfeMeanPoints: meanMove,
            maeMeanPoints: max(0.0, 0.35 * meanMove),
            hitTimeFraction: 1.0,
            pathRisk: 0.0,
            fillRisk: 0.0,
            confidence: confidence,
            reliability: reliability
        )
    }

    private static func deterministicRandomBuySide(request: PredictRequestV4) -> Bool {
        var timestamp = request.context.sampleTimeUTC
        if timestamp < 0 {
            timestamp = -timestamp
        }
        var accumulator = Double(timestamp)
        let limit = min(request.x.count, 8)
        if limit > 0 {
            for index in 0..<limit {
                accumulator = accumulator * 1.618_033_988_75 + abs(request.x[index]) * 1_000.0 + 17.0 * Double(index + 1)
                if accumulator > 2_147_483_000.0 {
                    accumulator -= 2_147_483_000.0 * floor(accumulator / 2_147_483_000.0)
                }
            }
        }
        let hash = UInt64(abs(Int64(accumulator.rounded())))
        return hash % 2 == 0
    }
}
