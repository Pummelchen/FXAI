import FXDataEngine
import Foundation

public struct AIS4Plugin: FXAIPlannedPlugin {
    private static let pluginManifest = PluginManifestV4(
        aiID: AIModelID.s4.rawValue,
        aiName: "ai_s4",
        family: .stateSpace,
        referenceTier: PluginPersistenceTools.defaultReferenceTier(aiID: AIModelID.s4.rawValue),
        capabilityMask: [.selfTest, .onlineLearning, .replay, .stateful, .windowContext, .multiHorizon, .nativeDistribution],
        featureSchema: .sequence,
        featureGroups: [.price, .multiTimeframe, .volatility, .time, .context, .volume, .filters],
        minHorizonMinutes: 1,
        maxHorizonMinutes: 240,
        minSequenceBars: 1,
        maxSequenceBars: 96,
        requiresVolumeWhenAvailable: true
    )

    private var cpu: AIS4CPUModel

    public var manifest: PluginManifestV4 { Self.pluginManifest }
    public var accelerationPlan: FXPluginAccelerationPlan { AIS4Accelerated.descriptor }

    public init() {
        self.cpu = AIS4CPUModel()
    }

    public mutating func reset() {
        cpu.reset()
    }

    public func selfTest() -> Bool {
        (try? manifest.validate()) != nil &&
            accelerationPlan.primaryBackends.contains(.swiftScalar) &&
            accelerationPlan.usesVolumeWhenAvailable
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
