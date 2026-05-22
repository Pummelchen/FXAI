import FXDataEngine
import Foundation

public struct RuleBuyOnlyPlugin: FXAIPlannedPlugin {
    public let manifest = PluginManifestV4(
        aiID: AIModelID.buyOnly.rawValue,
        aiName: "rule_buyonly",
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
        pluginName: "rule_buyonly",
        primaryBackends: [.swiftScalar],
        usesVolumeWhenAvailable: true,
        notes: "Constant BUY baseline. Accepts the volume-aware plugin contract while preserving legacy deterministic direction; no tensor or GPU backend is useful."
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
        return Self.fixedDirectionalPrediction(
            request: request,
            buyProbability: 0.999,
            sellProbability: 0.001,
            confidence: 0.999
        )
    }

    private static func fixedDirectionalPrediction(
        request: PredictRequestV4,
        buyProbability: Double,
        sellProbability: Double,
        confidence: Double
    ) -> PredictionV4 {
        let minimumMove = request.context.minMovePoints > 0.0
            ? request.context.minMovePoints
            : max(request.context.priceCostPoints, 0.10)
        let meanMove = max(1.0, 3.0 * minimumMove + 0.25)
        let sigma = max(0.10, 0.30 * meanMove)
        let q25 = max(0.0, meanMove - 0.50 * sigma)
        let q50 = max(q25, meanMove)
        let q75 = max(q50, meanMove + 0.50 * sigma)

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
            reliability: 0.55
        )
    }
}
