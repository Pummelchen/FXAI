import FXDataEngine
import Foundation

public struct LinEnhashPlugin: FXAIPlannedPlugin {
    private static let pluginManifest = PluginManifestV4(
        aiID: AIModelID.enhash.rawValue,
        aiName: "lin_enhash",
        family: .linear,
        referenceTier: PluginPersistenceTools.defaultReferenceTier(aiID: AIModelID.enhash.rawValue),
        capabilityMask: [.selfTest, .onlineLearning, .replay, .multiHorizon],
        featureSchema: .sparseStat,
        featureGroups: [.price, .multiTimeframe, .volatility, .volume, .filters],
        minHorizonMinutes: 1,
        maxHorizonMinutes: 240,
        minSequenceBars: 1,
        maxSequenceBars: 1,
        requiresVolumeWhenAvailable: true
    )

    private var cpu: LinEnhashCPUModel

    public var manifest: PluginManifestV4 { Self.pluginManifest }
    public var accelerationPlan: FXPluginAccelerationPlan { LinEnhashMetal.descriptor }

    public init() {
        self.cpu = LinEnhashCPUModel()
    }

    public mutating func reset() {
        cpu.reset()
    }

    public func selfTest() -> Bool {
        (try? manifest.validate()) != nil && accelerationPlan.candidateBackends.contains(.metal)
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
