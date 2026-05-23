import FXDataEngine
import Foundation

public struct AIMLPPlugin: FXAIPlannedPlugin {
    private static let pluginManifest = PluginManifestV4(
        aiID: AIModelID.mlpTiny.rawValue,
        aiName: "ai_mlp",
        family: .convolutional,
        referenceTier: PluginPersistenceTools.defaultReferenceTier(aiID: AIModelID.mlpTiny.rawValue),
        capabilityMask: [.selfTest, .onlineLearning, .replay, .stateful, .windowContext, .multiHorizon, .nativeDistribution],
        featureSchema: .sequence,
        featureGroups: [.price, .multiTimeframe, .volatility, .time, .context, .volume, .filters],
        minHorizonMinutes: 1,
        maxHorizonMinutes: 240,
        minSequenceBars: 2,
        maxSequenceBars: 96,
        requiresVolumeWhenAvailable: true
    )

    private var cpu: AIMLPCPUModel

    public var manifest: PluginManifestV4 { Self.pluginManifest }
    public var accelerationPlan: FXPluginAccelerationPlan { AIMLPAccelerated.descriptor }

    public init() {
        self.cpu = AIMLPCPUModel()
    }

    public mutating func reset() {
        cpu.reset()
    }

    public func selfTest() -> Bool {
        (try? manifest.validate()) != nil &&
            accelerationPlan.primaryBackends.contains(.swiftScalar) &&
            accelerationPlan.primaryBackends.contains(.pyTorchMPS) &&
            accelerationPlan.primaryBackends.contains(.tensorFlowMetal)
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
