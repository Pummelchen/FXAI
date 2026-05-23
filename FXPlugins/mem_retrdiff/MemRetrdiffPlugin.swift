import FXDataEngine
import Foundation

public struct MemRetrdiffPlugin: FXAIPlannedPlugin {
    private static let pluginManifest = PluginManifestV4(
        aiID: AIModelID.retrDiff.rawValue,
        aiName: "mem_retrdiff",
        family: .retrieval,
        referenceTier: PluginPersistenceTools.defaultReferenceTier(aiID: AIModelID.retrDiff.rawValue),
        capabilityMask: [.selfTest, .onlineLearning, .replay, .windowContext, .multiHorizon],
        featureSchema: .contextual,
        featureGroups: [.price, .multiTimeframe, .volatility, .context, .volume],
        minHorizonMinutes: 1,
        maxHorizonMinutes: 240,
        minSequenceBars: 8,
        maxSequenceBars: 64,
        requiresVolumeWhenAvailable: true
    )

    private var cpu: MemRetrdiffCPUModel

    public var manifest: PluginManifestV4 { Self.pluginManifest }
    public var accelerationPlan: FXPluginAccelerationPlan { MemRetrdiffMetal.descriptor }

    public init() {
        self.cpu = MemRetrdiffCPUModel()
    }

    public mutating func reset() {
        cpu.reset()
    }

    public func selfTest() -> Bool {
        (try? manifest.validate()) != nil &&
            accelerationPlan.primaryBackends.contains(.accelerate) &&
            accelerationPlan.candidateBackends.contains(.metal)
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
