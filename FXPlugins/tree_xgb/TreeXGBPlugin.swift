import FXDataEngine
import Foundation

public struct TreeXGBPlugin: FXAIPlannedPlugin {
    private static let pluginManifest = PluginManifestV4(
        aiID: AIModelID.xgboost.rawValue,
        aiName: "tree_xgb",
        family: .tree,
        referenceTier: PluginPersistenceTools.defaultReferenceTier(aiID: AIModelID.xgboost.rawValue),
        capabilityMask: [.selfTest, .onlineLearning, .replay, .multiHorizon],
        featureSchema: .tree,
        featureGroups: [.price, .multiTimeframe, .volatility, .context, .volume, .filters],
        minHorizonMinutes: 1,
        maxHorizonMinutes: 240,
        minSequenceBars: 1,
        maxSequenceBars: 1,
        requiresVolumeWhenAvailable: true
    )

    private var cpu: TreeXGBCPUModel

    public var manifest: PluginManifestV4 { Self.pluginManifest }
    public var accelerationPlan: FXPluginAccelerationPlan { TreeXGBMetal.descriptor }

    public init() {
        self.cpu = TreeXGBCPUModel()
    }

    public mutating func reset() {
        cpu.reset()
    }

    public func selfTest() -> Bool {
        (try? manifest.validate()) != nil &&
            accelerationPlan.primaryBackends.contains(.metal) &&
            accelerationPlan.primaryBackends.contains(.accelerate)
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
