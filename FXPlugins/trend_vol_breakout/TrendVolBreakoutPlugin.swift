import FXDataEngine
import Foundation

public struct TrendVolBreakoutPlugin: FXAIPlannedPlugin {
    private static let pluginManifest = PluginManifestV4(
        aiID: AIModelID.trendVolBreakout.rawValue,
        aiName: "trend_vol_breakout",
        family: .other,
        referenceTier: PluginPersistenceTools.defaultReferenceTier(aiID: AIModelID.trendVolBreakout.rawValue),
        capabilityMask: [.selfTest, .onlineLearning, .replay, .windowContext, .multiHorizon, .nativeDistribution],
        featureSchema: .sparseStat,
        featureGroups: [.price, .multiTimeframe, .volatility, .volume, .filters],
        minHorizonMinutes: 1,
        maxHorizonMinutes: 240,
        minSequenceBars: 2,
        maxSequenceBars: 64,
        requiresVolumeWhenAvailable: true
    )

    private var cpu: TrendVolBreakoutCPUModel

    public var manifest: PluginManifestV4 { Self.pluginManifest }
    public var accelerationPlan: FXPluginAccelerationPlan { TrendVolBreakoutAccelerated.descriptor }

    public init() {
        self.cpu = TrendVolBreakoutCPUModel()
    }

    public mutating func reset() {
        cpu.reset()
    }

    public func selfTest() -> Bool {
        (try? manifest.validate()) != nil &&
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
