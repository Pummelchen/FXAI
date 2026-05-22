import FXDataEngine
import Foundation

public struct FXAIGeneratedPluginAdapter: FXAIPlannedPlugin {
    public let definition: FXAIGeneratedPluginDefinition

    public var manifest: PluginManifestV4 {
        PluginManifestV4(
            aiID: definition.aiID.rawValue,
            aiName: definition.aiName,
            family: definition.family,
            referenceTier: PluginPersistenceTools.defaultReferenceTier(aiID: definition.aiID.rawValue),
            capabilityMask: definition.capabilities,
            featureSchema: definition.featureSchema,
            featureGroups: definition.featureGroups,
            minHorizonMinutes: 1,
            maxHorizonMinutes: 240,
            minSequenceBars: 1,
            maxSequenceBars: definition.maxSequenceBars,
            requiresVolumeWhenAvailable: true
        )
    }

    public var accelerationPlan: FXPluginAccelerationPlan {
        FXPluginAccelerationPlan(
            pluginName: definition.aiName,
            primaryBackends: definition.primaryBackends,
            candidateBackends: definition.candidateBackends,
            usesVolumeWhenAvailable: true,
            notes: definition.notes
        )
    }

    public init(definition: FXAIGeneratedPluginDefinition) {
        self.definition = definition
    }

    public mutating func reset() {}

    public func selfTest() -> Bool {
        (try? manifest.validate()) != nil && definition.primaryBackends.isEmpty == false
    }

    public mutating func train(_ request: TrainRequestV4, hyperParameters: HyperParameters) throws {
        try request.validate()
        try PluginContractTools.validateCompatibility(manifest: manifest, context: request.context)
    }

    public func predict(_ request: PredictRequestV4, hyperParameters: HyperParameters) throws -> PredictionV4 {
        try request.validate()
        try PluginContractTools.validateCompatibility(manifest: manifest, context: request.context)

        let edge = definition.profile.edge(request)
        let volumeSignal = request.context.dataHasVolume ? FXAIGeneratedPluginAdapter.feature(request, 6) : 0.0
        let volumeLift = request.context.dataHasVolume ? 0.10 * fxClamp(abs(volumeSignal), 0.0, 1.0) : 0.0
        let rawStrength = abs(edge) * definition.profile.strengthScale + volumeLift
        let strength = fxClamp(rawStrength, 0.0, 1.0)

        if strength < definition.profile.skipThreshold {
            return FXAIGeneratedPluginAdapter.prediction(
                label: .skip,
                strength: 0.0,
                moveSeed: 0.0,
                reliability: definition.profile.baseReliability + 0.05 * volumeLift
            )
        }

        let label: LabelClass = edge >= 0.0 ? .buy : .sell
        let moveSeed = max(
            request.context.minMovePoints,
            request.context.priceCostPoints,
            abs(edge) * definition.profile.moveScale
        )
        return FXAIGeneratedPluginAdapter.prediction(
            label: label,
            strength: strength,
            moveSeed: moveSeed,
            reliability: definition.profile.baseReliability + 0.20 * volumeLift
        )
    }

    public static func generatedPlugins() -> [FXAIGeneratedPluginAdapter] {
        FXAIGeneratedPluginDefinition.all.map(FXAIGeneratedPluginAdapter.init(definition:))
    }

    public static func generatedAccelerationPlans() -> [FXPluginAccelerationPlan] {
        generatedPlugins().map(\.accelerationPlan)
    }

    private static func prediction(label: LabelClass, strength: Double, moveSeed: Double, reliability: Double) -> PredictionV4 {
        let clampedStrength = fxClamp(strength, 0.0, 1.0)
        let directional = fxClamp(0.54 + 0.38 * clampedStrength, 0.54, 0.92)
        let opposite = 0.06
        let skip = max(0.06, 1.0 - directional - opposite)
        let probabilities: [Double]
        switch label {
        case .buy:
            probabilities = normalize([opposite, directional, skip])
        case .sell:
            probabilities = normalize([directional, opposite, skip])
        case .skip:
            probabilities = [0.09, 0.09, 0.82]
        }

        let meanMove = label == .skip ? 0.0 : max(1.0, moveSeed)
        let sigma = max(0.10, 0.32 * meanMove)
        let q25 = max(0.0, meanMove - 0.55 * sigma)
        let q50 = max(q25, meanMove)
        let q75 = max(q50, meanMove + 0.55 * sigma)
        return PredictionV4(
            classProbabilities: probabilities,
            moveMeanPoints: meanMove,
            moveQ25Points: q25,
            moveQ50Points: q50,
            moveQ75Points: q75,
            mfeMeanPoints: meanMove,
            maeMeanPoints: max(0.0, 0.35 * meanMove),
            hitTimeFraction: 1.0,
            pathRisk: probabilities[LabelClass.skip.rawValue],
            fillRisk: 0.0,
            confidence: max(probabilities[LabelClass.buy.rawValue], probabilities[LabelClass.sell.rawValue]),
            reliability: fxClamp(reliability, 0.0, 1.0)
        )
    }

    private static func normalize(_ values: [Double]) -> [Double] {
        let sum = values.reduce(0.0, +)
        guard sum > 0.0 else { return [0.09, 0.09, 0.82] }
        return values.map { $0 / sum }
    }
}

public struct FXAIGeneratedPluginDefinition: Sendable, Hashable {
    public let aiID: AIModelID
    public let aiName: String
    public let family: AIFamily
    public let featureSchema: FeatureSchema
    public let featureGroups: FeatureGroupMask
    public let capabilities: PluginCapability
    public let maxSequenceBars: Int
    public let profile: FXAIGeneratedPluginProfile
    public let primaryBackends: [FXPluginAccelerationBackend]
    public let candidateBackends: [FXPluginAccelerationBackend]
    public let notes: String

    public init(
        aiID: AIModelID,
        aiName: String,
        family: AIFamily,
        featureSchema: FeatureSchema,
        featureGroups: FeatureGroupMask,
        capabilities: PluginCapability,
        maxSequenceBars: Int,
        profile: FXAIGeneratedPluginProfile,
        primaryBackends: [FXPluginAccelerationBackend],
        candidateBackends: [FXPluginAccelerationBackend] = [],
        notes: String
    ) {
        self.aiID = aiID
        self.aiName = aiName
        self.family = family
        self.featureSchema = featureSchema
        self.featureGroups = featureGroups
        self.capabilities = capabilities
        self.maxSequenceBars = maxSequenceBars
        self.profile = profile
        self.primaryBackends = primaryBackends
        self.candidateBackends = candidateBackends
        self.notes = notes
    }
}

public enum FXAIGeneratedPluginProfile: String, Sendable, Hashable {
    case linear
    case tree
    case sequence
    case distribution
    case statistical
    case factor
    case trend
    case mixture
    case memory
    case world
    case reinforcement

    public var strengthScale: Double {
        switch self {
        case .linear, .trend: 4.0
        case .tree, .factor: 3.6
        case .sequence, .world: 3.2
        case .distribution, .statistical, .mixture, .memory: 3.0
        case .reinforcement: 2.8
        }
    }

    public var skipThreshold: Double {
        switch self {
        case .linear, .trend: 0.08
        case .tree, .factor: 0.10
        case .sequence, .world, .reinforcement: 0.14
        case .distribution, .statistical, .mixture, .memory: 0.12
        }
    }

    public var moveScale: Double {
        switch self {
        case .linear, .trend, .factor: 90.0
        case .tree, .mixture, .memory: 100.0
        case .sequence, .world, .reinforcement: 110.0
        case .distribution, .statistical: 80.0
        }
    }

    public var baseReliability: Double {
        switch self {
        case .linear, .trend: 0.52
        case .tree, .factor: 0.54
        case .sequence, .world: 0.50
        case .distribution, .statistical, .mixture, .memory: 0.51
        case .reinforcement: 0.48
        }
    }

    public func edge(_ request: PredictRequestV4) -> Double {
        let shortReturn = FXAIGeneratedPluginAdapter.feature(request, 0)
        let mediumSlope = FXAIGeneratedPluginAdapter.feature(request, 3)
        let volatility = abs(FXAIGeneratedPluginAdapter.feature(request, 4))
        let volume = request.context.dataHasVolume ? FXAIGeneratedPluginAdapter.feature(request, 6) : 0.0
        let fastReturn = FXAIGeneratedPluginAdapter.feature(request, 7)
        let slowReturn = FXAIGeneratedPluginAdapter.feature(request, 8)
        let contextSignal = FXAIGeneratedPluginAdapter.feature(request, 12)
        let mtfEdge = fastReturn - slowReturn

        switch self {
        case .linear:
            return 0.48 * shortReturn + 0.34 * mediumSlope + 0.12 * mtfEdge + 0.06 * volume
        case .tree:
            return 0.35 * shortReturn + 0.25 * mediumSlope + 0.25 * mtfEdge + 0.10 * contextSignal + 0.05 * volume
        case .sequence:
            return 0.28 * shortReturn + 0.28 * mediumSlope + 0.30 * mtfEdge + 0.08 * contextSignal + 0.06 * volume
        case .distribution:
            return (0.40 * shortReturn + 0.25 * mediumSlope + 0.20 * mtfEdge + 0.10 * volume) / max(1.0 + volatility, 1.0)
        case .statistical:
            return 0.30 * shortReturn + 0.22 * mediumSlope + 0.22 * mtfEdge + 0.14 * contextSignal + 0.12 * volume
        case .factor:
            return 0.22 * shortReturn + 0.20 * mediumSlope + 0.18 * mtfEdge + 0.30 * contextSignal + 0.10 * volume
        case .trend:
            return 0.34 * shortReturn + 0.36 * mediumSlope + 0.20 * mtfEdge + 0.10 * volume
        case .mixture:
            return 0.25 * shortReturn + 0.25 * mediumSlope + 0.25 * mtfEdge + 0.15 * contextSignal + 0.10 * volume
        case .memory:
            return 0.20 * shortReturn + 0.25 * mediumSlope + 0.35 * mtfEdge + 0.10 * contextSignal + 0.10 * volume
        case .world:
            return 0.18 * shortReturn + 0.22 * mediumSlope + 0.24 * mtfEdge + 0.28 * contextSignal + 0.08 * volume
        case .reinforcement:
            return 0.24 * shortReturn + 0.24 * mediumSlope + 0.24 * mtfEdge + 0.16 * contextSignal + 0.12 * volume
        }
    }
}

extension FXAIGeneratedPluginAdapter {
    fileprivate static func feature(_ request: PredictRequestV4, _ index: Int) -> Double {
        guard index >= 0, index < request.x.count else { return 0.0 }
        return fxSafeFinite(request.x[index])
    }
}

public extension FXAIGeneratedPluginDefinition {
    static let all: [FXAIGeneratedPluginDefinition] = [
        sequence(.autoformer, "ai_autoformer", .transformer, [.pyTorchMPS], [.tensorFlowMetal, .coreMLNeuralEngine]),
        tree(.catboost, "tree_catboost"),
        sequence(.chronos, "ai_chronos", .transformer, [.pyTorchMPS], [.coreMLNeuralEngine]),
        linear(.enhash, "lin_enhash"),
        linear(.ftrlLogit, "lin_ftrl"),
        sequence(.geodesicAttention, "ai_geodesic", .transformer, [.pyTorchMPS], [.metal, .coreMLNeuralEngine]),
        tree(.lightgbm, "tree_lgbm"),
        sequence(.lstm, "ai_lstm", .recurrent, [.tensorFlowMetal], [.pyTorchMPS, .coreMLNeuralEngine]),
        sequence(.lstmg, "ai_lstmg", .recurrent, [.pyTorchMPS], [.tensorFlowMetal, .coreMLNeuralEngine]),
        sequence(.mlpTiny, "ai_mlp", .convolutional, [.accelerate], [.metal, .coreMLNeuralEngine]),
        linear(.paLinear, "lin_pa"),
        sequence(.patchTST, "ai_patchtst", .transformer, [.pyTorchMPS], [.coreMLNeuralEngine]),
        distribution(.quantile, "dist_quantile"),
        sequence(.s4, "ai_s4", .stateSpace, [.pyTorchMPS], [.metal]),
        linear(.sgdLogit, "lin_sgd"),
        sequence(.stmn, "ai_stmn", .stateSpace, [.pyTorchMPS], [.coreMLNeuralEngine]),
        sequence(.tcn, "ai_tcn", .convolutional, [.pyTorchMPS], [.tensorFlowMetal, .coreMLNeuralEngine]),
        sequence(.tft, "ai_tft", .transformer, [.pyTorchMPS], [.coreMLNeuralEngine]),
        sequence(.timesfm, "ai_timesfm", .transformer, [.pyTorchMPS], [.coreMLNeuralEngine]),
        sequence(.tst, "ai_tst", .transformer, [.pyTorchMPS], [.coreMLNeuralEngine]),
        tree(.xgbFast, "tree_xgb_fast"),
        tree(.xgboost, "tree_xgb"),
        world(.cfxWorld, "wm_cfx"),
        mixture(.loffm, "mix_loffm"),
        sequence(.trr, "ai_trr", .recurrent, [.pyTorchMPS], [.accelerate]),
        world(.graphWM, "wm_graph"),
        mixture(.moeConformal, "mix_moe_conformal"),
        memory(.retrDiff, "mem_retrdiff"),
        distribution(.qcew, "ai_qcew"),
        sequence(.fewc, "ai_fewc", .transformer, [.pyTorchMPS], [.accelerate]),
        sequence(.gha, "ai_gha", .transformer, [.accelerate], [.pyTorchMPS]),
        sequence(.tesseract, "ai_tesseract", .transformer, [.metal], [.pyTorchMPS]),
        statistical(.statMSGARCH, "stat_msgarch"),
        statistical(.statARIMAXGARCH, "stat_arimax_garch"),
        tree(.treeRF, "tree_rf"),
        statistical(.statCointVECM, "stat_coint_vecm"),
        statistical(.statOUSpread, "stat_ou_spread"),
        reinforcement(.rlPPO, "rl_ppo"),
        statistical(.statMicroflowProxy, "stat_microflow_proxy"),
        statistical(.statHMMRegime, "stat_hmm_regime"),
        linear(.linElasticLogit, "lin_elastic_logit"),
        linear(.linProfitLogit, "lin_profit_logit"),
        sequence(.cnnLSTM, "ai_cnn_lstm", .convolutional, [.tensorFlowMetal], [.pyTorchMPS, .coreMLNeuralEngine]),
        sequence(.attnCNNBiLSTM, "ai_attn_cnn_bilstm", .convolutional, [.tensorFlowMetal], [.pyTorchMPS, .coreMLNeuralEngine]),
        statistical(.statEMDHHT, "stat_emd_hht", [.accelerate], [.metal]),
        statistical(.statVMD, "stat_vmd", [.accelerate], [.metal]),
        statistical(.statTVPKalman, "stat_tvp_kalman"),
        factor(.factorPCAPanel, "factor_pca_panel"),
        factor(.factorPPPValue, "factor_ppp_value"),
        factor(.factorCarry, "factor_carry"),
        factor(.factorCMVPanel, "factor_cmv_panel"),
        trend(.trendTSMOMVol, "trend_tsmom_vol"),
        trend(.trendXSMOMRank, "trend_xsmom_rank"),
        trend(.trendVolBreakout, "trend_vol_breakout"),
        statistical(.statXRateConsistency, "stat_xrate_consistency"),
        sequence(.gru, "ai_gru", .recurrent, [.tensorFlowMetal], [.pyTorchMPS, .coreMLNeuralEngine]),
        sequence(.bilstm, "ai_bilstm", .recurrent, [.tensorFlowMetal], [.pyTorchMPS, .coreMLNeuralEngine]),
        sequence(.lstmTCN, "ai_lstm_tcn", .convolutional, [.pyTorchMPS], [.tensorFlowMetal, .coreMLNeuralEngine]),
        sequence(.mythosRDT, "ai_mythos_rdt", .transformer, [.pyTorchMPS], [.coreMLNeuralEngine])
    ]

    private static func linear(_ aiID: AIModelID, _ name: String) -> FXAIGeneratedPluginDefinition {
        FXAIGeneratedPluginDefinition(
            aiID: aiID,
            aiName: name,
            family: .linear,
            featureSchema: .sparseStat,
            featureGroups: [.price, .multiTimeframe, .volatility, .volume, .filters],
            capabilities: [.selfTest, .onlineLearning, .multiHorizon],
            maxSequenceBars: 1,
            profile: .linear,
            primaryBackends: [.swiftSIMD, .accelerate],
            notes: "Native Swift adapter for online linear plugin. Uses volume-aware fallback inference until full fitted-state parity is ported."
        )
    }

    private static func tree(_ aiID: AIModelID, _ name: String) -> FXAIGeneratedPluginDefinition {
        FXAIGeneratedPluginDefinition(
            aiID: aiID,
            aiName: name,
            family: .tree,
            featureSchema: .tree,
            featureGroups: [.price, .multiTimeframe, .volatility, .context, .volume, .filters],
            capabilities: [.selfTest, .multiHorizon],
            maxSequenceBars: 1,
            profile: .tree,
            primaryBackends: [.swiftScalar, .metal],
            candidateBackends: [.swiftSIMD, .accelerate],
            notes: "Tree ensemble adapter. Metal is planned for batched scoring while CPU remains the deterministic reference."
        )
    }

    private static func sequence(
        _ aiID: AIModelID,
        _ name: String,
        _ family: AIFamily,
        _ primary: [FXPluginAccelerationBackend],
        _ candidate: [FXPluginAccelerationBackend]
    ) -> FXAIGeneratedPluginDefinition {
        FXAIGeneratedPluginDefinition(
            aiID: aiID,
            aiName: name,
            family: family,
            featureSchema: .sequence,
            featureGroups: [.price, .multiTimeframe, .volatility, .time, .context, .volume],
            capabilities: [.selfTest, .windowContext, .multiHorizon],
            maxSequenceBars: 256,
            profile: .sequence,
            primaryBackends: primary,
            candidateBackends: candidate,
            notes: "Sequence-model adapter. Tensor training/inference is delegated to the declared backend; Swift fallback is volume-aware and deterministic."
        )
    }

    private static func distribution(_ aiID: AIModelID, _ name: String) -> FXAIGeneratedPluginDefinition {
        FXAIGeneratedPluginDefinition(
            aiID: aiID,
            aiName: name,
            family: .distributional,
            featureSchema: .sparseStat,
            featureGroups: [.price, .multiTimeframe, .volatility, .volume],
            capabilities: [.selfTest, .nativeDistribution, .multiHorizon],
            maxSequenceBars: 1,
            profile: .distribution,
            primaryBackends: [.accelerate],
            candidateBackends: [.metal],
            notes: "Distributional adapter using Accelerate-class statistics first, with Metal reserved for batched quantile work."
        )
    }

    private static func statistical(
        _ aiID: AIModelID,
        _ name: String,
        _ primary: [FXPluginAccelerationBackend] = [.accelerate],
        _ candidate: [FXPluginAccelerationBackend] = [.swiftSIMD]
    ) -> FXAIGeneratedPluginDefinition {
        FXAIGeneratedPluginDefinition(
            aiID: aiID,
            aiName: name,
            family: .other,
            featureSchema: .sparseStat,
            featureGroups: [.price, .multiTimeframe, .volatility, .context, .volume, .filters],
            capabilities: [.selfTest, .multiHorizon],
            maxSequenceBars: 1,
            profile: .statistical,
            primaryBackends: primary,
            candidateBackends: candidate,
            notes: "Statistical plugin adapter. Accelerate handles matrix/vector math; Python fitting can be added for complex estimators."
        )
    }

    private static func factor(_ aiID: AIModelID, _ name: String) -> FXAIGeneratedPluginDefinition {
        FXAIGeneratedPluginDefinition(
            aiID: aiID,
            aiName: name,
            family: .other,
            featureSchema: .contextual,
            featureGroups: [.price, .multiTimeframe, .context, .volume],
            capabilities: [.selfTest, .multiHorizon],
            maxSequenceBars: 1,
            profile: .factor,
            primaryBackends: [.accelerate],
            candidateBackends: [.swiftSIMD],
            notes: "Factor plugin adapter using context and volume-aware scoring. Accelerate is the primary native math path."
        )
    }

    private static func trend(_ aiID: AIModelID, _ name: String) -> FXAIGeneratedPluginDefinition {
        FXAIGeneratedPluginDefinition(
            aiID: aiID,
            aiName: name,
            family: .other,
            featureSchema: .sparseStat,
            featureGroups: [.price, .multiTimeframe, .volatility, .volume],
            capabilities: [.selfTest, .multiHorizon],
            maxSequenceBars: 1,
            profile: .trend,
            primaryBackends: [.swiftSIMD, .accelerate],
            candidateBackends: [.metal],
            notes: "Trend plugin adapter with SIMD/Accelerate rolling-window path and optional Metal for large parameter sweeps."
        )
    }

    private static func mixture(_ aiID: AIModelID, _ name: String) -> FXAIGeneratedPluginDefinition {
        FXAIGeneratedPluginDefinition(
            aiID: aiID,
            aiName: name,
            family: .mixture,
            featureSchema: .contextual,
            featureGroups: [.price, .multiTimeframe, .volatility, .context, .volume],
            capabilities: [.selfTest, .multiHorizon],
            maxSequenceBars: 1,
            profile: .mixture,
            primaryBackends: [.accelerate],
            candidateBackends: [.pyTorchMPS],
            notes: "Mixture adapter with native gating fallback and optional PyTorch MPS for learned expert routing."
        )
    }

    private static func memory(_ aiID: AIModelID, _ name: String) -> FXAIGeneratedPluginDefinition {
        FXAIGeneratedPluginDefinition(
            aiID: aiID,
            aiName: name,
            family: .retrieval,
            featureSchema: .contextual,
            featureGroups: [.price, .multiTimeframe, .volatility, .context, .volume],
            capabilities: [.selfTest, .stateful, .multiHorizon],
            maxSequenceBars: 128,
            profile: .memory,
            primaryBackends: [.accelerate],
            candidateBackends: [.metal],
            notes: "Retrieval-memory adapter. Accelerate distance search is primary; Metal top-k is reserved for large replay banks."
        )
    }

    private static func world(_ aiID: AIModelID, _ name: String) -> FXAIGeneratedPluginDefinition {
        FXAIGeneratedPluginDefinition(
            aiID: aiID,
            aiName: name,
            family: .worldModel,
            featureSchema: .contextual,
            featureGroups: [.price, .multiTimeframe, .volatility, .time, .context, .volume],
            capabilities: [.selfTest, .windowContext, .multiHorizon],
            maxSequenceBars: 128,
            profile: .world,
            primaryBackends: [.accelerate],
            candidateBackends: [.pyTorchMPS, .coreMLNeuralEngine],
            notes: "World-model adapter. Native graph/context scoring remains the fallback; learned world models can move to PyTorch MPS and Core ML inference."
        )
    }

    private static func reinforcement(_ aiID: AIModelID, _ name: String) -> FXAIGeneratedPluginDefinition {
        FXAIGeneratedPluginDefinition(
            aiID: aiID,
            aiName: name,
            family: .other,
            featureSchema: .contextual,
            featureGroups: [.price, .multiTimeframe, .volatility, .context, .volume, .microstructure],
            capabilities: [.selfTest, .windowContext, .multiHorizon],
            maxSequenceBars: 128,
            profile: .reinforcement,
            primaryBackends: [.pyTorchMPS],
            candidateBackends: [.coreMLNeuralEngine],
            notes: "RL policy adapter. PPO training belongs in PyTorch MPS; Core ML is an inference-only candidate after policy stabilization."
        )
    }
}
