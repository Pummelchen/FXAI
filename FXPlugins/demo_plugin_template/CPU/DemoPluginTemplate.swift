import FXDataEngine
import Foundation

public enum DemoPluginParameterValueKind: String, Codable, Hashable, Sendable {
    case integer
    case decimal
    case boolean
}

public struct DemoPluginParameterTemplate: Codable, Hashable, Sendable {
    public let key: String
    public let displayName: String
    public let valueKind: DemoPluginParameterValueKind
    public let defaultValue: Double
    public let minimum: Double
    public let step: Double
    public let maximum: Double
    public let unit: String
    public let description: String

    public init(
        key: String,
        displayName: String,
        valueKind: DemoPluginParameterValueKind,
        defaultValue: Double,
        minimum: Double,
        step: Double,
        maximum: Double,
        unit: String,
        description: String
    ) {
        self.key = key
        self.displayName = displayName
        self.valueKind = valueKind
        self.defaultValue = defaultValue
        self.minimum = minimum
        self.step = step
        self.maximum = maximum
        self.unit = unit
        self.description = description
    }

    public func validate() throws {
        guard !key.isEmpty, key.allSatisfy({ $0.isLetter || $0.isNumber || $0 == "_" }) else {
            throw FXDataEngineError.validation("demo parameter key")
        }
        guard !displayName.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw FXDataEngineError.validation("demo parameter displayName")
        }
        guard defaultValue.isFinite, minimum.isFinite, step.isFinite, maximum.isFinite else {
            throw FXDataEngineError.validation("demo parameter finite bounds")
        }
        guard minimum <= defaultValue, defaultValue <= maximum, step > 0 else {
            throw FXDataEngineError.validation("demo parameter bounds")
        }
        if valueKind == .boolean {
            guard [0, 1].contains(defaultValue),
                  [0, 1].contains(minimum),
                  [0, 1].contains(maximum) else {
                throw FXDataEngineError.validation("demo boolean parameter bounds")
            }
        }
    }
}

public struct DemoPluginAcceleratorConfigurationTemplate: Codable, Hashable, Sendable {
    public let pluginId: String
    public let acceleratorId: String
    public let parameters: [DemoPluginParameterTemplate]

    public init(pluginId: String, acceleratorId: String, parameters: [DemoPluginParameterTemplate]) {
        self.pluginId = pluginId
        self.acceleratorId = acceleratorId
        self.parameters = parameters
    }

    public func validate() throws {
        guard !pluginId.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw FXDataEngineError.validation("demo configuration pluginId")
        }
        guard !acceleratorId.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw FXDataEngineError.validation("demo configuration acceleratorId")
        }
        guard !parameters.isEmpty else {
            throw FXDataEngineError.validation("demo configuration parameters")
        }
        for parameter in parameters {
            try parameter.validate()
        }
    }
}

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

    public static let configurationParameters: [DemoPluginParameterTemplate] = [
        DemoPluginParameterTemplate(
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
        DemoPluginParameterTemplate(
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
        DemoPluginParameterTemplate(
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

    public func configurationRows() -> [DemoPluginAcceleratorConfigurationTemplate] {
        accelerationPlan.declaredBackends.map { backend in
            DemoPluginAcceleratorConfigurationTemplate(
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
