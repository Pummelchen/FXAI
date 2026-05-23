import FXDataEngine
import Foundation

public struct MixMoeConformalPlugin: FXAIPlannedPlugin {
    private static let pluginManifest = PluginManifestV4(
        aiID: AIModelID.moeConformal.rawValue,
        aiName: "mix_moe_conformal",
        family: .mixture,
        referenceTier: PluginPersistenceTools.defaultReferenceTier(aiID: AIModelID.moeConformal.rawValue),
        capabilityMask: [.selfTest, .onlineLearning, .replay, .multiHorizon],
        featureSchema: .contextual,
        featureGroups: [.price, .multiTimeframe, .volatility, .context, .volume],
        minHorizonMinutes: 1,
        maxHorizonMinutes: 240,
        minSequenceBars: 1,
        maxSequenceBars: 1,
        requiresVolumeWhenAvailable: true
    )

    private static let pluginAccelerationPlan = FXPluginAccelerationPlan(
        pluginName: "mix_moe_conformal",
        primaryBackends: [.accelerate],
        candidateBackends: [.pyTorchMPS],
        usesVolumeWhenAvailable: true,
        notes: "Plugin-local Swift CPU port of the MQL5 conformal mixture-of-experts router. PyTorch/MPS folder provides independent batched MoE inference; Metal, TensorFlow, and NLP are not suitable for this small online conformal model."
    )

    private var cpu: MixMoeConformalCPUModel

    public var manifest: PluginManifestV4 { Self.pluginManifest }
    public var accelerationPlan: FXPluginAccelerationPlan { Self.pluginAccelerationPlan }

    public init() {
        self.cpu = MixMoeConformalCPUModel()
    }

    public mutating func reset() {
        cpu.reset()
    }

    public func selfTest() -> Bool {
        (try? manifest.validate()) != nil &&
            accelerationPlan.primaryBackends.contains(.accelerate) &&
            accelerationPlan.candidateBackends.contains(.pyTorchMPS)
    }

    public mutating func train(_ request: TrainRequestV4, hyperParameters: HyperParameters) throws {
        try request.validate()
        try PluginContractTools.validateCompatibility(manifest: manifest, context: request.context)
        cpu.train(request, hyperParameters: hyperParameters)
    }

    public func predict(_ request: PredictRequestV4, hyperParameters: HyperParameters) throws -> PredictionV4 {
        try request.validate()
        try PluginContractTools.validateCompatibility(manifest: manifest, context: request.context)
        return cpu.predict(request, hyperParameters: hyperParameters)
    }
}
