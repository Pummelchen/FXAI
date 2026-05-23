import FXDataEngine
import Foundation

public struct MixLoffmPlugin: FXAIPlannedPlugin {
    private static let pluginManifest = PluginManifestV4(
        aiID: AIModelID.loffm.rawValue,
        aiName: "mix_loffm",
        family: .mixture,
        referenceTier: PluginPersistenceTools.defaultReferenceTier(aiID: AIModelID.loffm.rawValue),
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
        pluginName: "mix_loffm",
        primaryBackends: [.accelerate],
        candidateBackends: [.pyTorchMPS],
        usesVolumeWhenAvailable: true,
        notes: "Plugin-local Swift CPU implementation of the LOFFM gated expert model. PyTorch/MPS folder provides batched gating/inference research acceleration; TensorFlow, Metal, and NLP are not suitable for this online mixture model."
    )

    private var cpu: MixLoffmCPUModel

    public var manifest: PluginManifestV4 { Self.pluginManifest }
    public var accelerationPlan: FXPluginAccelerationPlan { Self.pluginAccelerationPlan }

    public init() {
        self.cpu = MixLoffmCPUModel()
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
