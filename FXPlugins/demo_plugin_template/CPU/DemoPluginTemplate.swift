import FXBacktestAPI
import FXDataEngine
import Foundation

public struct DemoPluginTemplate: FXAIPlannedPlugin {
    public let aiID: Int
    public let aiName: String

    public init(aiID: Int, aiName: String = "replace_with_plugin_name") {
        self.aiID = aiID
        self.aiName = aiName
    }

    public var manifest: PluginManifestV4 {
        PluginManifestV4(
            aiID: aiID,
            aiName: aiName,
            family: .other,
            referenceTier: .surrogate,
            capabilityMask: [.selfTest, .multiHorizon, .windowContext],
            featureGroups: [.price, .multiTimeframe, .volume],
            minHorizonMinutes: 1,
            maxHorizonMinutes: 240,
            minSequenceBars: 1,
            maxSequenceBars: 96,
            requiresVolumeWhenAvailable: true
        )
    }

    public var accelerationPlan: FXPluginAccelerationPlan {
        FXPluginAccelerationPlan(
            pluginName: aiName,
            primaryBackends: [.swiftScalar],
            candidateBackends: [.metal, .pyTorchMPS, .tensorFlowMetal, .foundationNLP],
            usesVolumeWhenAvailable: true,
            notes: "Template only. Replace the no-trade CPU shell with real strategy logic before adding the plugin to FXAIPluginRegistry."
        )
    }

    public static let configurationParameters: [FXBacktestConfigurationParameterDTO] = [
        FXBacktestConfigurationParameterDTO(
            key: "lookback_bars",
            displayName: "Lookback Bars",
            valueKind: .integer,
            defaultValue: 64,
            minimum: 1,
            step: 1,
            maximum: 5_000,
            unit: "bars",
            description: "Example plugin-owned window length. Replace with the plugin's real parameter set."
        ),
        FXBacktestConfigurationParameterDTO(
            key: "confidence_floor",
            displayName: "Confidence Floor",
            valueKind: .decimal,
            defaultValue: 0.55,
            minimum: 0.0,
            step: 0.01,
            maximum: 1.0,
            unit: "ratio",
            description: "Example decision threshold for plugin-level signal filtering."
        ),
        FXBacktestConfigurationParameterDTO(
            key: "use_volume_when_available",
            displayName: "Use Volume When Available",
            valueKind: .boolean,
            defaultValue: 1,
            minimum: 0,
            step: 1,
            maximum: 1,
            unit: "flag",
            description: "Template flag showing how plugins expose volume-aware behavior."
        )
    ]

    public func configurationRows() -> [FXBacktestPluginConfigurationDTO] {
        accelerationPlan.declaredBackends.map { backend in
            FXBacktestPluginConfigurationDTO(
                pluginId: aiName,
                acceleratorId: backend.rawValue,
                parameters: Self.configurationParameters
            )
        }
    }

    public mutating func reset() {}

    public func selfTest() -> Bool {
        (try? manifest.validate()) != nil &&
            (try? Self.configurationParameters.forEach { try $0.validate() }) != nil
    }

    public mutating func train(_ request: TrainRequestV4, hyperParameters: HyperParameters) throws {
        try request.validate()
        try PluginContractTools.validateCompatibility(manifest: manifest, context: request.context)
    }

    public func predict(_ request: PredictRequestV4, hyperParameters: HyperParameters) throws -> PredictionV4 {
        try request.validate()
        try PluginContractTools.validateCompatibility(manifest: manifest, context: request.context)
        return PredictionV4(
            classProbabilities: [0.0, 0.0, 1.0],
            moveMeanPoints: 0.0,
            moveQ25Points: 0.0,
            moveQ50Points: 0.0,
            moveQ75Points: 0.0,
            mfeMeanPoints: 0.0,
            maeMeanPoints: 0.0,
            hitTimeFraction: 0.0,
            pathRisk: 0.0,
            fillRisk: 0.0,
            confidence: 1.0,
            reliability: 1.0
        )
    }
}
