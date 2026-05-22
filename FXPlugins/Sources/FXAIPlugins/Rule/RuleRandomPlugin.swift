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
        usesVolumeWhenAvailable: false,
        notes: "Deterministic no-skip random baseline. Hashing is scalar and not worth GPU or tensor dispatch."
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

        let buySide = RulePredictionTools.deterministicRandomBuySide(request: request)
        return RulePredictionTools.fixedDirectionalPrediction(
            request: request,
            buyProbability: buySide ? 0.995 : 0.005,
            sellProbability: buySide ? 0.005 : 0.995,
            confidence: 0.995,
            sigmaScale: 0.35,
            quantileScale: 0.55,
            reliability: 0.50
        )
    }
}
