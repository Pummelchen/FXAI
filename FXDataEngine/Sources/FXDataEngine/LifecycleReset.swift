import Foundation

public enum LifecycleResetAction: String, Codable, Sendable, CaseIterable {
    case resetModelHyperParams = "RESET_MODEL_HYPER_PARAMS"
    case resetReliabilityPending = "RESET_RELIABILITY_PENDING"
    case resetHorizonPolicyPending = "RESET_HORIZON_POLICY_PENDING"
    case resetStackPending = "RESET_STACK_PENDING"
    case resetPolicyState = "RESET_POLICY_STATE"
    case resetConformalState = "RESET_CONFORMAL_STATE"
    case resetAdaptiveRoutingState = "RESET_ADAPTIVE_ROUTING_STATE"
    case resetRegimeCalibration = "RESET_REGIME_CALIBRATION"
    case resetReplayReservoir = "RESET_REPLAY_RESERVOIR"
    case resetFeatureDriftDiagnostics = "RESET_FEATURE_DRIFT_DIAGNOSTICS"
    case resetGlobalSharedTransferBackbone = "RESET_GLOBAL_SHARED_TRANSFER_BACKBONE"
    case resetAnalogMemory = "RESET_ANALOG_MEMORY"
    case resetRegimeGraph = "RESET_REGIME_GRAPH"
    case resetBrokerExecutionReplayStats = "RESET_BROKER_EXECUTION_REPLAY_STATS"
    case resetMacroEventStore = "RESET_MACRO_EVENT_STORE"
    case resetAllPlugins = "RESET_ALL_PLUGINS"
    case resetModelAuxState = "RESET_MODEL_AUX_STATE"
    case loadMetaArtifacts = "LOAD_META_ARTIFACTS"
    case loadRuntimeArtifacts = "LOAD_RUNTIME_ARTIFACTS"
}

public struct LifecycleContextUtilityReset: Codable, Hashable, Sendable {
    public var utility: Double
    public var stability: Double
    public var lead: Double
    public var coverage: Double
    public var ready: Bool

    public init(
        utility: Double = 0.0,
        stability: Double = 0.0,
        lead: Double = 0.0,
        coverage: Double = 0.0,
        ready: Bool = false
    ) {
        self.utility = fxSafeFinite(utility)
        self.stability = fxSafeFinite(stability)
        self.lead = fxSafeFinite(lead)
        self.coverage = fxSafeFinite(coverage)
        self.ready = ready
    }
}

public struct LifecycleAIResetState: Codable, Hashable, Sendable {
    public var aiID: Int
    public var trained: Bool
    public var lastTrainBarUTC: Int64
    public var auxResetRequired: Bool

    public init(
        aiID: Int,
        trained: Bool = false,
        lastTrainBarUTC: Int64 = 0,
        auxResetRequired: Bool = true
    ) {
        self.aiID = max(0, aiID)
        self.trained = trained
        self.lastTrainBarUTC = max(0, lastTrainBarUTC)
        self.auxResetRequired = auxResetRequired
    }
}

public struct LifecycleResetPlan: Codable, Hashable, Sendable {
    public var symbol: String
    public var lastSignalBarUTC: Int64
    public var lastSignal: Int
    public var lastSignalKey: Int
    public var lastReason: String
    public var signalCache: RuntimeSignalCache
    public var warmupDone: Bool
    public var configuredHorizons: [Int]
    public var actions: [LifecycleResetAction]
    public var resetPlugins: Bool
    public var contextUtility: [LifecycleContextUtilityReset]
    public var normalizationDefaultWindow: Int
    public var normalizationFeatureWindows: [Int]
    public var aiStates: [LifecycleAIResetState]

    public init(
        symbol: String = "",
        lastSignalBarUTC: Int64 = 0,
        lastSignal: Int = -1,
        lastSignalKey: Int = -1,
        lastReason: String = "reset",
        signalCache: RuntimeSignalCache = .reset,
        warmupDone: Bool = true,
        configuredHorizons: [Int] = HorizonTools.defaultConfiguredHorizons,
        actions: [LifecycleResetAction] = [],
        resetPlugins: Bool = false,
        contextUtility: [LifecycleContextUtilityReset] = [],
        normalizationDefaultWindow: Int = NormalizationWindowTools.defaultWindow,
        normalizationFeatureWindows: [Int] = [],
        aiStates: [LifecycleAIResetState] = []
    ) {
        self.symbol = symbol
        self.lastSignalBarUTC = max(0, lastSignalBarUTC)
        self.lastSignal = lastSignal
        self.lastSignalKey = lastSignalKey
        self.lastReason = lastReason.isEmpty ? "reset" : lastReason
        self.signalCache = signalCache
        self.warmupDone = warmupDone
        self.configuredHorizons = configuredHorizons.map(HorizonTools.clampHorizon)
        self.actions = actions
        self.resetPlugins = resetPlugins
        self.contextUtility = contextUtility
        self.normalizationDefaultWindow = NormalizationWindowTools.clamp(normalizationDefaultWindow)
        self.normalizationFeatureWindows = NormalizationWindowTools.normalizedFeatureWindows(
            normalizationFeatureWindows,
            defaultWindow: self.normalizationDefaultWindow
        )
        self.aiStates = aiStates
    }
}

public enum LifecycleResetTools {
    public static let baseResetActions: [LifecycleResetAction] = [
        .resetModelHyperParams,
        .resetReliabilityPending,
        .resetHorizonPolicyPending,
        .resetStackPending,
        .resetPolicyState,
        .resetConformalState,
        .resetAdaptiveRoutingState,
        .resetRegimeCalibration,
        .resetReplayReservoir,
        .resetFeatureDriftDiagnostics,
        .resetGlobalSharedTransferBackbone,
        .resetAnalogMemory,
        .resetRegimeGraph,
        .resetBrokerExecutionReplayStats,
        .resetMacroEventStore
    ]

    public static func buildResetPlan(
        symbol: String,
        aiWarmupEnabled: Bool,
        horizonListRaw: String,
        predictionTargetMinutes: Int,
        pluginsReady: Bool = false,
        normalizationWindowsReady: Bool = false,
        existingNormalizationFeatureWindows: [Int] = [],
        existingNormalizationDefaultWindow: Int = NormalizationWindowTools.defaultWindow,
        maxContextSymbols: Int = FXDataEngineConstants.maxContextSymbols,
        aiCount: Int = FXDataEngineConstants.aiCount
    ) -> LifecycleResetPlan {
        var actions = baseResetActions
        if pluginsReady {
            actions.append(.resetAllPlugins)
        }
        actions.append(.resetModelAuxState)
        actions.append(.loadMetaArtifacts)
        actions.append(.loadRuntimeArtifacts)

        let defaultWindow: Int
        let featureWindows: [Int]
        if normalizationWindowsReady {
            defaultWindow = NormalizationWindowTools.clamp(existingNormalizationDefaultWindow)
            featureWindows = NormalizationWindowTools.normalizedFeatureWindows(
                existingNormalizationFeatureWindows,
                defaultWindow: defaultWindow
            )
        } else {
            defaultWindow = NormalizationWindowTools.defaultWindow(predictionTargetMinutes: predictionTargetMinutes)
            featureWindows = NormalizationWindowTools.buildGroupWindows(
                fast: defaultWindow,
                mid: defaultWindow,
                slow: defaultWindow,
                regime: defaultWindow
            )
        }

        let contextCount = max(0, min(maxContextSymbols, FXDataEngineConstants.maxContextSymbols))
        let modelCount = max(0, min(aiCount, FXDataEngineConstants.aiCount))

        return LifecycleResetPlan(
            symbol: symbol,
            signalCache: RuntimeSignalCache.reset,
            warmupDone: !aiWarmupEnabled,
            configuredHorizons: HorizonTools.parseHorizonList(
                horizonListRaw,
                baseHorizon: predictionTargetMinutes
            ),
            actions: actions,
            resetPlugins: pluginsReady,
            contextUtility: Array(repeating: LifecycleContextUtilityReset(), count: contextCount),
            normalizationDefaultWindow: defaultWindow,
            normalizationFeatureWindows: featureWindows,
            aiStates: (0..<modelCount).map { LifecycleAIResetState(aiID: $0) }
        )
    }
}
