import FXDataEngine
import Foundation

public struct RuleSellOnlyPlugin: FXAIPlannedPlugin {
    public let manifest = PluginManifestV4(
        aiID: AIModelID.sellOnly.rawValue,
        aiName: "rule_sellonly",
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
        pluginName: "rule_sellonly",
        primaryBackends: [.swiftScalar],
        usesVolumeWhenAvailable: false,
        notes: "Constant SELL baseline. No tensor or GPU backend is useful; keep as deterministic scalar reference."
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
        return RulePredictionTools.fixedDirectionalPrediction(
            request: request,
            buyProbability: 0.001,
            sellProbability: 0.999,
            confidence: 0.999
        )
    }
}
