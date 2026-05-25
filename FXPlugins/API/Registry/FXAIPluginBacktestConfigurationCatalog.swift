import FXBacktestAPI
import FXDataEngine
import Foundation

public enum FXAIPluginBacktestConfigurationCatalog {
    public static func pluginConfigurations(
        plugins: [any FXAIPluginV4] = FXAIPluginRegistry.availablePlugins()
    ) -> [FXBacktestPluginConfigurationDTO] {
        plugins.compactMap { $0 as? any FXAIPlannedPlugin }
            .flatMap { plugin in
                let parameters = parameters(for: plugin)
                return plugin.accelerationPlan.declaredBackends.map { backend in
                    FXBacktestPluginConfigurationDTO(
                        pluginId: plugin.manifest.aiName,
                        acceleratorId: backend.rawValue,
                        parameters: parameters + acceleratorParameters(for: backend)
                    )
                }
            }
            .sorted {
                if $0.pluginId != $1.pluginId {
                    return $0.pluginId < $1.pluginId
                }
                return $0.acceleratorId < $1.acceleratorId
            }
    }

    public static func registrationRequest(
        sharedParameters: [FXBacktestConfigurationParameterDTO] = [],
        plugins: [any FXAIPluginV4] = FXAIPluginRegistry.availablePlugins()
    ) -> FXBacktestConfigurationRegistrationRequest {
        FXBacktestConfigurationRegistrationRequest(
            sharedParameters: sharedParameters,
            pluginConfigurations: pluginConfigurations(plugins: plugins)
        )
    }

    private static func parameters(for plugin: any FXAIPlannedPlugin) -> [FXBacktestConfigurationParameterDTO] {
        let manifest = plugin.manifest
        let name = manifest.aiName
        var result = [
            parameter("learning_rate", "Learning Rate", .decimal, 0.01, 0.0001, 0.0001, 0.30, "ratio", "Primary online learning rate for \(name)."),
            parameter("l2", "L2 Regularization", .decimal, 0.0001, 0.0, 0.0001, 1.0, "ratio", "Weight decay or L2-style shrinkage for \(name).")
        ]

        if name.contains("ftrl") {
            result += [
                parameter("ftrl_alpha", "FTRL Alpha", .decimal, 0.05, 0.001, 0.001, 1.0, "ratio", "FTRL proximal alpha for \(name)."),
                parameter("ftrl_beta", "FTRL Beta", .decimal, 1.0, 0.0, 0.05, 5.0, "ratio", "FTRL proximal beta for \(name)."),
                parameter("ftrl_l1", "FTRL L1", .decimal, 0.0, 0.0, 0.0001, 1.0, "ratio", "FTRL L1 penalty for \(name)."),
                parameter("ftrl_l2", "FTRL L2", .decimal, 0.0001, 0.0, 0.0001, 1.0, "ratio", "FTRL L2 penalty for \(name).")
            ]
        }
        if name.contains("lin_pa") {
            result += [
                parameter("passive_aggressive_c", "Passive Aggressive C", .decimal, 0.5, 0.01, 0.01, 10.0, "ratio", "Passive-aggressive aggressiveness bound for \(name)."),
                parameter("passive_aggressive_margin", "Passive Aggressive Margin", .decimal, 0.05, 0.0, 0.01, 2.0, "ratio", "Target margin for \(name).")
            ]
        }
        if name.contains("enhash") {
            result += [
                parameter("enhash_learning_rate", "Enhash Learning Rate", .decimal, 0.01, 0.0001, 0.0001, 0.30, "ratio", "Hashed linear update rate for \(name)."),
                parameter("enhash_l1", "Enhash L1", .decimal, 0.0, 0.0, 0.0001, 1.0, "ratio", "Hashed linear L1 penalty for \(name)."),
                parameter("enhash_l2", "Enhash L2", .decimal, 0.0001, 0.0, 0.0001, 1.0, "ratio", "Hashed linear L2 penalty for \(name).")
            ]
        }
        if manifest.family == .tree || name.contains("xgb") || name.contains("lgbm") || name.contains("catboost") || name.contains("rf") {
            result += [
                parameter("xgb_learning_rate", "Tree Learning Rate", .decimal, 0.05, 0.001, 0.001, 0.30, "ratio", "Boosting or tree ensemble update rate for \(name)."),
                parameter("xgb_l2", "Tree L2", .decimal, 0.001, 0.0, 0.001, 10.0, "ratio", "Tree leaf shrinkage for \(name)."),
                parameter("xgb_split", "Tree Split Bias", .decimal, 0.0, -2.0, 0.05, 2.0, "score", "Split threshold bias for \(name).")
            ]
        }
        if manifest.family == .distributional || name.contains("quantile") {
            result += [
                parameter("quantile_learning_rate", "Quantile Learning Rate", .decimal, 0.01, 0.0001, 0.0001, 0.30, "ratio", "Pinball-loss learning rate for \(name)."),
                parameter("quantile_l2", "Quantile L2", .decimal, 0.0001, 0.0, 0.0001, 1.0, "ratio", "Quantile-head shrinkage for \(name).")
            ]
        }
        if usesDenseNeuralParameters(manifest: manifest, pluginName: name) {
            result += [
                parameter("mlp_learning_rate", "Neural Learning Rate", .decimal, 0.01, 0.0001, 0.0001, 0.30, "ratio", "Dense or sequence neural learning rate for \(name)."),
                parameter("mlp_l2", "Neural L2", .decimal, 0.0001, 0.0, 0.0001, 1.0, "ratio", "Dense or sequence neural weight decay for \(name)."),
                parameter("mlp_init", "Neural Init Scale", .decimal, 0.05, 0.001, 0.001, 0.50, "scale", "Initial parameter scale for \(name).")
            ]
        }
        if name.contains("tcn") || name.contains("cnn") {
            result += [
                parameter("tcn_layers", "TCN Layers", .integer, 2, 1, 1, 12, "layers", "Temporal convolution layer count for \(name)."),
                parameter("tcn_kernel", "TCN Kernel", .integer, 3, 2, 1, 15, "bars", "Temporal convolution kernel width for \(name)."),
                parameter("tcn_dilation_base", "TCN Dilation Base", .integer, 2, 1, 1, 8, "factor", "Temporal convolution dilation base for \(name).")
            ]
        }
        return result
    }

    private static func usesDenseNeuralParameters(manifest: PluginManifestV4, pluginName: String) -> Bool {
        switch manifest.family {
        case .recurrent, .convolutional, .transformer, .stateSpace, .mixture, .worldModel, .other:
            return true
        case .linear, .tree, .distributional, .retrieval, .ruleBased:
            return pluginName.contains("mlp")
        }
    }

    private static func acceleratorParameters(for backend: FXPluginAccelerationBackend) -> [FXBacktestConfigurationParameterDTO] {
        switch backend {
        case .swiftScalar:
            return [
                parameter("swift_scalar_enabled", "Swift Scalar Enabled", .boolean, 1, 0, 1, 1, "flag", "Enable the deterministic CPU scalar implementation.")
            ]
        case .swiftSIMD, .accelerate:
            return [
                parameter("cpu_vector_batch_size", "CPU Vector Batch Size", .integer, 256, 16, 16, 16_384, "samples", "Batch size for SIMD or Accelerate vectorized scoring.")
            ]
        case .metal:
            return [
                parameter("metal_threadgroup_size", "Metal Threadgroup Size", .integer, 128, 32, 32, 1_024, "threads", "Preferred Metal compute threadgroup size."),
                parameter("metal_max_passes_per_buffer", "Metal Passes Per Buffer", .integer, 4_096, 1, 1, 65_536, "passes", "Maximum optimization passes encoded in one Metal command buffer.")
            ]
        case .pyTorchMPS:
            return externalMLParameters(prefix: "pytorch", display: "PyTorch MPS")
        case .tensorFlowMetal:
            return externalMLParameters(prefix: "tensorflow", display: "TensorFlow Metal")
        case .foundationNLP:
            return [
                parameter("nlp_context_weight", "NLP Context Weight", .decimal, 0.25, 0.0, 0.05, 1.0, "ratio", "Weight applied to text/event context features."),
                parameter("nlp_max_events", "NLP Max Events", .integer, 32, 0, 1, 256, "events", "Maximum event records consumed per prediction.")
            ]
        case .coreMLNeuralEngine:
            return [
                parameter("coreml_batch_size", "CoreML Batch Size", .integer, 64, 1, 1, 4_096, "samples", "Batch size for CoreML/Neural Engine inference.")
            ]
        }
    }

    private static func externalMLParameters(prefix: String, display: String) -> [FXBacktestConfigurationParameterDTO] {
        [
            parameter("\(prefix)_batch_size", "\(display) Batch Size", .integer, 64, 1, 1, 4_096, "samples", "Training and inference batch size for \(display)."),
            parameter("\(prefix)_epochs", "\(display) Epochs", .integer, 3, 1, 1, 200, "epochs", "Maximum local fine-tuning epochs for \(display)."),
            parameter("\(prefix)_weight_decay", "\(display) Weight Decay", .decimal, 0.0001, 0.0, 0.0001, 1.0, "ratio", "Optimizer weight decay for \(display).")
        ]
    }

    private static func parameter(
        _ key: String,
        _ displayName: String,
        _ kind: FXBacktestConfigurationValueKind,
        _ defaultValue: Double,
        _ minimum: Double,
        _ step: Double,
        _ maximum: Double,
        _ unit: String,
        _ description: String
    ) -> FXBacktestConfigurationParameterDTO {
        FXBacktestConfigurationParameterDTO(
            key: key,
            displayName: displayName,
            valueKind: kind,
            defaultValue: defaultValue,
            minimum: minimum,
            step: step,
            maximum: maximum,
            unit: unit,
            description: description
        )
    }
}
