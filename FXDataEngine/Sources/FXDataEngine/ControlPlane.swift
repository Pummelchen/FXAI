import Foundation

public enum PolicyLifecycleAction: Int, Codable, Sendable, CaseIterable {
    case noTrade = 0
    case enter
    case hold
    case exit
    case add
    case reduce
    case tighten
    case timeout
}

public enum ControlPlaneConstants {
    public static let directory = "FXAI/ControlPlane"
    public static let ttlSeconds = 7_200
    public static let promotionsDirectory = "FXAI/Offline/Promotions"
    public static let portfolioSupervisorFile = "FXAI/Offline/Promotions/fxai_portfolio_supervisor.tsv"
    public static let supervisorServiceGlobalFile = "FXAI/Offline/Promotions/fxai_supervisor_service_global.tsv"
    public static let supervisorCommandGlobalFile = "FXAI/Offline/Promotions/fxai_supervisor_command_global.tsv"
    public static let adaptiveRouterRegimeLabels = [
        "TREND_PERSISTENT",
        "RANGE_MEAN_REVERTING",
        "BREAKOUT_TRANSITION",
        "HIGH_VOL_EVENT",
        "RISK_ON_OFF_MACRO",
        "LIQUIDITY_STRESS",
        "SESSION_FLOW"
    ]
    public static let adaptiveRouterSessionLabels = [
        "ASIA",
        "LONDON",
        "NEWYORK",
        "LONDON_NY_OVERLAP",
        "ROLLOVER"
    ]
}

public extension AIFamily {
    var controlPlaneName: String {
        switch self {
        case .linear: "linear"
        case .tree: "tree"
        case .recurrent: "recurrent"
        case .convolutional: "convolutional"
        case .transformer: "transformer"
        case .stateSpace: "state_space"
        case .distributional: "distribution"
        case .mixture: "mixture"
        case .retrieval: "memory"
        case .worldModel: "world"
        case .ruleBased: "rule"
        case .other: "other"
        }
    }
}

public enum ControlPlanePaths {
    public static func safeToken(_ raw: String, defaultValue: String = "default") -> String {
        var clean = raw.isEmpty ? defaultValue : raw
        for character in ["\\", "/", ":", "*", "?", "\"", "<", ">", "|", " "] {
            clean = clean.replacingOccurrences(of: character, with: "_")
        }
        return clean
    }

    public static func supervisorServiceSymbolFile(symbol: String) -> String {
        "\(ControlPlaneConstants.promotionsDirectory)/fxai_supervisor_service_\(safeToken(symbol)).tsv"
    }

    public static func supervisorCommandSymbolFile(symbol: String) -> String {
        "\(ControlPlaneConstants.promotionsDirectory)/fxai_supervisor_command_\(safeToken(symbol)).tsv"
    }

    public static func studentRouterProfileFile(symbol: String) -> String {
        "\(ControlPlaneConstants.promotionsDirectory)/fxai_student_router_\(safeToken(symbol)).tsv"
    }

    public static func adaptiveRouterProfileFile(symbol: String) -> String {
        "\(ControlPlaneConstants.promotionsDirectory)/fxai_adaptive_router_\(safeToken(symbol)).tsv"
    }

    public static func liveDeploymentProfileFile(symbol: String) -> String {
        "\(ControlPlaneConstants.promotionsDirectory)/fxai_live_deploy_\(safeToken(symbol)).tsv"
    }

    public static func snapshotFile(symbol: String, login: Int64, magic: UInt64, chartID: Int64) -> String {
        let chartModulo = chartID % 2_147_483_647
        return "\(ControlPlaneConstants.directory)/cp_\(login)_\(magic)_\(safeToken(symbol))_\(chartModulo).tsv"
    }
}

public struct ControlPlaneKeyValueDocument: Codable, Hashable, Sendable {
    public let values: [String: String]

    public init(values: [String: String]) {
        self.values = values
    }

    public init(tsv: String) {
        var parsed: [String: String] = [:]
        for line in tsv.components(separatedBy: .newlines) {
            guard !line.isEmpty else { continue }
            let parts = line.split(separator: "\t", omittingEmptySubsequences: false)
            guard parts.count >= 2 else { continue }
            parsed[String(parts[0])] = String(parts[1])
        }
        self.values = parsed
    }

    public func string(_ key: String, default defaultValue: String = "") -> String {
        values[key] ?? defaultValue
    }

    public func double(_ key: String, default defaultValue: Double = 0.0) -> Double {
        guard let raw = values[key] else { return defaultValue }
        return Double(raw.trimmingCharacters(in: .whitespacesAndNewlines)) ?? 0.0
    }

    public func int(_ key: String, default defaultValue: Int = 0) -> Int {
        guard let raw = values[key] else { return defaultValue }
        return Int(raw.trimmingCharacters(in: .whitespacesAndNewlines)) ?? 0
    }

    public func int64(_ key: String, default defaultValue: Int64 = 0) -> Int64 {
        guard let raw = values[key] else { return defaultValue }
        return Int64(raw.trimmingCharacters(in: .whitespacesAndNewlines)) ?? 0
    }

    public func bool(_ key: String, default defaultValue: Bool = false) -> Bool {
        guard values[key] != nil else { return defaultValue }
        return int(key) != 0
    }
}

public enum ControlPlaneCSV {
    public static func containsToken(_ csv: String, token: String) -> Bool {
        let cleanToken = token.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !cleanToken.isEmpty else { return false }
        return csv.split(separator: ",").contains { item in
            item.trimmingCharacters(in: .whitespacesAndNewlines) == cleanToken
        }
    }

    public static func mapWeight(_ csv: String, key: String, default defaultValue: Double = 1.0) -> Double {
        let cleanKey = key.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !cleanKey.isEmpty else { return defaultValue }
        for item in csv.split(separator: ",") {
            let parts = item.split(separator: "=", maxSplits: 1, omittingEmptySubsequences: false)
            guard parts.count >= 2 else { continue }
            let itemKey = parts[0].trimmingCharacters(in: .whitespacesAndNewlines)
            guard itemKey == cleanKey else { continue }
            return Double(parts[1].trimmingCharacters(in: .whitespacesAndNewlines)) ?? 0.0
        }
        return defaultValue
    }
}

public struct LiveDeploymentProfile: Codable, Hashable, Sendable {
    public var ready = false
    public var profileName = ""
    public var symbol = ""
    public var teacherWeight = 0.58
    public var studentWeight = 0.42
    public var analogWeight = 0.18
    public var foundationWeight = 0.24
    public var policyTradeFloor = 0.52
    public var policySizeBias = 1.0
    public var portfolioBudgetBias = 1.0
    public var challengerPromoteMargin = 1.0
    public var regimeTransitionWeight = 0.35
    public var macroQualityFloor = 0.24
    public var policyNoTradeCap = 0.62
    public var capitalEfficiencyBias = 1.0
    public var supervisorBlend = 0.45
    public var teacherSignalGain = 1.0
    public var studentSignalGain = 1.0
    public var foundationQualityGain = 1.0
    public var macroStateGain = 1.0
    public var policyLifecycleGain = 1.0
    public var policyHoldFloor = 0.48
    public var policyExitFloor = 0.58
    public var policyAddFloor = 0.68
    public var policyReduceFloor = 0.56
    public var policyTimeoutFloor = 0.72
    public var maxAddFraction = 0.50
    public var reduceFraction = 0.35
    public var softTimeoutBars = 8
    public var hardTimeoutBars = 18
    public var runtimeMode = "research"
    public var telemetryLevel = "full"
    public var performanceBudgetMS = 12.0
    public var shadowEnabled = true
    public var snapshotDetail = "full"
    public var maxRuntimeModels = 12
    public var promotionTier = "experimental"
    public var loadedAtUTC: Int64 = 0

    public init(symbol: String = "") {
        self.symbol = symbol
    }

    public static func parse(symbol: String, tsv: String, loadedAtUTC: Int64 = 0) -> LiveDeploymentProfile {
        let doc = ControlPlaneKeyValueDocument(tsv: tsv)
        var profile = LiveDeploymentProfile(symbol: symbol)
        profile.profileName = doc.string("profile_name", default: profile.profileName)
        profile.symbol = doc.string("symbol", default: profile.symbol)
        profile.teacherWeight = doc.double("teacher_weight", default: profile.teacherWeight)
        profile.studentWeight = doc.double("student_weight", default: profile.studentWeight)
        profile.analogWeight = doc.double("analog_weight", default: profile.analogWeight)
        profile.foundationWeight = doc.double("foundation_weight", default: profile.foundationWeight)
        profile.policyTradeFloor = doc.double("policy_trade_floor", default: profile.policyTradeFloor)
        profile.policySizeBias = doc.double("policy_size_bias", default: profile.policySizeBias)
        profile.portfolioBudgetBias = doc.double("portfolio_budget_bias", default: profile.portfolioBudgetBias)
        profile.challengerPromoteMargin = doc.double("challenger_promote_margin", default: profile.challengerPromoteMargin)
        profile.regimeTransitionWeight = doc.double("regime_transition_weight", default: profile.regimeTransitionWeight)
        profile.macroQualityFloor = doc.double("macro_quality_floor", default: profile.macroQualityFloor)
        profile.policyNoTradeCap = doc.double("policy_no_trade_cap", default: profile.policyNoTradeCap)
        profile.capitalEfficiencyBias = doc.double("capital_efficiency_bias", default: profile.capitalEfficiencyBias)
        profile.supervisorBlend = doc.double("supervisor_blend", default: profile.supervisorBlend)
        profile.teacherSignalGain = doc.double("teacher_signal_gain", default: profile.teacherSignalGain)
        profile.studentSignalGain = doc.double("student_signal_gain", default: profile.studentSignalGain)
        profile.foundationQualityGain = doc.double("foundation_quality_gain", default: profile.foundationQualityGain)
        profile.macroStateGain = doc.double("macro_state_gain", default: profile.macroStateGain)
        profile.policyLifecycleGain = doc.double("policy_lifecycle_gain", default: profile.policyLifecycleGain)
        profile.policyHoldFloor = doc.double("policy_hold_floor", default: profile.policyHoldFloor)
        profile.policyExitFloor = doc.double("policy_exit_floor", default: profile.policyExitFloor)
        profile.policyAddFloor = doc.double("policy_add_floor", default: profile.policyAddFloor)
        profile.policyReduceFloor = doc.double("policy_reduce_floor", default: profile.policyReduceFloor)
        profile.policyTimeoutFloor = doc.double("policy_timeout_floor", default: profile.policyTimeoutFloor)
        profile.maxAddFraction = doc.double("max_add_fraction", default: profile.maxAddFraction)
        profile.reduceFraction = doc.double("reduce_fraction", default: profile.reduceFraction)
        profile.softTimeoutBars = doc.int("soft_timeout_bars", default: profile.softTimeoutBars)
        profile.hardTimeoutBars = doc.int("hard_timeout_bars", default: profile.hardTimeoutBars)
        profile.runtimeMode = doc.string("runtime_mode", default: profile.runtimeMode)
        profile.telemetryLevel = doc.string("telemetry_level", default: profile.telemetryLevel)
        profile.performanceBudgetMS = doc.double("performance_budget_ms", default: profile.performanceBudgetMS)
        profile.shadowEnabled = doc.bool("shadow_enabled", default: profile.shadowEnabled)
        profile.snapshotDetail = doc.string("snapshot_detail", default: profile.snapshotDetail)
        profile.maxRuntimeModels = doc.int("max_runtime_models", default: profile.maxRuntimeModels)
        profile.promotionTier = doc.string("promotion_tier", default: profile.promotionTier)
        return profile.normalized(loadedAtUTC: loadedAtUTC)
    }

    public func normalized(loadedAtUTC: Int64? = nil) -> LiveDeploymentProfile {
        var profile = self
        profile.teacherWeight = fxClamp(profile.teacherWeight, 0.05, 0.95)
        profile.studentWeight = fxClamp(profile.studentWeight, 0.05, 0.95)
        profile.analogWeight = fxClamp(profile.analogWeight, 0.0, 0.80)
        profile.foundationWeight = fxClamp(profile.foundationWeight, 0.0, 0.90)
        profile.policyTradeFloor = fxClamp(profile.policyTradeFloor, 0.20, 0.90)
        profile.policySizeBias = fxClamp(profile.policySizeBias, 0.40, 1.60)
        profile.portfolioBudgetBias = fxClamp(profile.portfolioBudgetBias, 0.40, 1.60)
        profile.challengerPromoteMargin = fxClamp(profile.challengerPromoteMargin, 0.50, 3.00)
        profile.regimeTransitionWeight = fxClamp(profile.regimeTransitionWeight, 0.0, 1.0)
        profile.macroQualityFloor = fxClamp(profile.macroQualityFloor, 0.0, 1.0)
        profile.policyNoTradeCap = fxClamp(profile.policyNoTradeCap, 0.25, 0.95)
        profile.capitalEfficiencyBias = fxClamp(profile.capitalEfficiencyBias, 0.40, 1.80)
        profile.supervisorBlend = fxClamp(profile.supervisorBlend, 0.0, 1.0)
        profile.teacherSignalGain = fxClamp(profile.teacherSignalGain, 0.40, 1.80)
        profile.studentSignalGain = fxClamp(profile.studentSignalGain, 0.40, 1.80)
        profile.foundationQualityGain = fxClamp(profile.foundationQualityGain, 0.40, 1.80)
        profile.macroStateGain = fxClamp(profile.macroStateGain, 0.40, 1.80)
        profile.policyLifecycleGain = fxClamp(profile.policyLifecycleGain, 0.40, 1.80)
        profile.policyHoldFloor = fxClamp(profile.policyHoldFloor, 0.20, 0.95)
        profile.policyExitFloor = fxClamp(profile.policyExitFloor, 0.20, 0.99)
        profile.policyAddFloor = fxClamp(profile.policyAddFloor, 0.20, 0.99)
        profile.policyReduceFloor = fxClamp(profile.policyReduceFloor, 0.20, 0.99)
        profile.policyTimeoutFloor = fxClamp(profile.policyTimeoutFloor, 0.20, 0.99)
        profile.maxAddFraction = fxClamp(profile.maxAddFraction, 0.05, 1.00)
        profile.reduceFraction = fxClamp(profile.reduceFraction, 0.05, 0.95)
        profile.softTimeoutBars = Int(fxClamp(Double(profile.softTimeoutBars), 1.0, 10_000.0))
        profile.hardTimeoutBars = Int(fxClamp(Double(profile.hardTimeoutBars), Double(profile.softTimeoutBars + 1), 20_000.0))
        if profile.runtimeMode != "production" {
            profile.runtimeMode = "research"
        }
        if profile.telemetryLevel != "lean" {
            profile.telemetryLevel = "full"
        }
        if profile.snapshotDetail != "lean" {
            profile.snapshotDetail = "full"
        }
        profile.performanceBudgetMS = fxClamp(profile.performanceBudgetMS, 2.0, 100.0)
        profile.maxRuntimeModels = Int(fxClamp(Double(profile.maxRuntimeModels), 1.0, Double(FXDataEngineConstants.aiCount)))
        if !["production-approved", "audit-approved", "research-approved"].contains(profile.promotionTier) {
            profile.promotionTier = "experimental"
        }
        profile.ready = true
        if let loadedAtUTC {
            profile.loadedAtUTC = loadedAtUTC
        }
        return profile
    }
}

public struct StudentRouterProfile: Codable, Hashable, Sendable {
    public var ready = false
    public var profileName = ""
    public var symbol = ""
    public var championOnly = false
    public var maxActiveModels = 12
    public var minMetaWeight = 0.0
    public var allowPluginsCSV = ""
    public var pluginWeightsCSV = ""
    public var familyWeights: [Double]
    public var loadedAtUTC: Int64 = 0

    public init(symbol: String = "") {
        self.symbol = symbol
        self.familyWeights = Array(repeating: 1.0, count: AIFamily.allCases.count)
    }

    public static func parse(symbol: String, tsv: String, loadedAtUTC: Int64 = 0) -> StudentRouterProfile {
        let doc = ControlPlaneKeyValueDocument(tsv: tsv)
        var profile = StudentRouterProfile(symbol: symbol)
        profile.profileName = doc.string("profile_name", default: profile.profileName)
        profile.symbol = doc.string("symbol", default: profile.symbol)
        profile.championOnly = doc.bool("champion_only", default: profile.championOnly)
        profile.maxActiveModels = doc.int("max_active_models", default: profile.maxActiveModels)
        profile.minMetaWeight = doc.double("min_meta_weight", default: profile.minMetaWeight)
        profile.allowPluginsCSV = doc.string("allow_plugins_csv", default: profile.allowPluginsCSV)
        profile.pluginWeightsCSV = doc.string("plugin_weights_csv", default: profile.pluginWeightsCSV)
        for family in AIFamily.allCases {
            let key = "family_weight_\(family.controlPlaneName)"
            profile.familyWeights[family.rawValue] = doc.double(key, default: profile.familyWeights[family.rawValue])
        }
        return profile.normalized(loadedAtUTC: loadedAtUTC)
    }

    public func normalized(loadedAtUTC: Int64? = nil) -> StudentRouterProfile {
        var profile = self
        profile.maxActiveModels = Int(fxClamp(Double(profile.maxActiveModels), 1.0, Double(FXDataEngineConstants.aiCount)))
        profile.minMetaWeight = fxClamp(profile.minMetaWeight, 0.0, 0.25)
        profile.familyWeights = profile.familyWeights.enumerated().map { index, value in
            guard index < AIFamily.allCases.count else { return value }
            return fxClamp(value, 0.05, 1.50)
        }
        profile.ready = true
        if let loadedAtUTC {
            profile.loadedAtUTC = loadedAtUTC
        }
        return profile
    }

    public func familyWeight(_ family: AIFamily) -> Double {
        guard ready, family.rawValue < familyWeights.count else { return 1.0 }
        return fxClamp(familyWeights[family.rawValue], 0.05, 1.50)
    }

    public func pluginWeight(pluginName: String, family: AIFamily) -> Double {
        let familyWeight = familyWeight(family)
        guard ready else { return familyWeight }
        let pluginWeight = ControlPlaneCSV.mapWeight(pluginWeightsCSV, key: pluginName, default: familyWeight)
        return fxClamp(pluginWeight, 0.01, 1.60)
    }

    public func allowsPlugin(pluginName: String, family: AIFamily) -> Bool {
        guard ready else { return true }
        if familyWeight(family) <= 0.051 {
            return false
        }
        if pluginWeight(pluginName: pluginName, family: family) <= 0.021 {
            return false
        }
        guard championOnly else { return true }
        guard !allowPluginsCSV.isEmpty else { return true }
        return ControlPlaneCSV.containsToken(allowPluginsCSV, token: pluginName)
    }
}

public struct AdaptiveRouterProfile: Codable, Hashable, Sendable {
    public var ready = false
    public var enabled = false
    public var profileName = ""
    public var symbol = ""
    public var routerMode = "WEIGHTED_ENSEMBLE"
    public var fallbackToStudentRouterOnly = true
    public var pairTagsCSV = ""
    public var cautionThreshold = 0.55
    public var abstainThreshold = 0.35
    public var blockThreshold = 0.16
    public var confidenceFloor = 0.12
    public var suppressionThreshold = 0.34
    public var downweightThreshold = 0.78
    public var staleNewsAbstainBias = 0.24
    public var staleNewsForceCaution = true
    public var minPluginWeight = 0.05
    public var maxPluginWeight = 1.80
    public var maxActiveWeightShare = 0.72
    public var pluginGlobalWeightsCSV = ""
    public var pluginNewsCompatibilityCSV = ""
    public var pluginLiquidityRobustnessCSV = ""
    public var pluginRegimeWeightsCSV: [String]
    public var pluginSessionWeightsCSV: [String]
    public var loadedAtUTC: Int64 = 0

    public init(symbol: String = "") {
        self.symbol = symbol
        self.pluginRegimeWeightsCSV = Array(repeating: "", count: ControlPlaneConstants.adaptiveRouterRegimeLabels.count)
        self.pluginSessionWeightsCSV = Array(repeating: "", count: ControlPlaneConstants.adaptiveRouterSessionLabels.count)
    }

    public static func parse(symbol: String, tsv: String, loadedAtUTC: Int64 = 0) -> AdaptiveRouterProfile {
        let doc = ControlPlaneKeyValueDocument(tsv: tsv)
        var profile = AdaptiveRouterProfile(symbol: symbol)
        profile.profileName = doc.string("profile_name", default: profile.profileName)
        profile.symbol = doc.string("symbol", default: profile.symbol)
        profile.enabled = doc.bool("enabled", default: profile.enabled)
        profile.routerMode = doc.string("router_mode", default: profile.routerMode)
        profile.fallbackToStudentRouterOnly = doc.bool("fallback_to_student_router_only", default: profile.fallbackToStudentRouterOnly)
        profile.pairTagsCSV = doc.string("pair_tags_csv", default: profile.pairTagsCSV)
        profile.cautionThreshold = doc.double("caution_threshold", default: profile.cautionThreshold)
        profile.abstainThreshold = doc.double("abstain_threshold", default: profile.abstainThreshold)
        profile.blockThreshold = doc.double("block_threshold", default: profile.blockThreshold)
        profile.confidenceFloor = doc.double("confidence_floor", default: profile.confidenceFloor)
        profile.suppressionThreshold = doc.double("suppression_threshold", default: profile.suppressionThreshold)
        profile.downweightThreshold = doc.double("downweight_threshold", default: profile.downweightThreshold)
        profile.staleNewsAbstainBias = doc.double("stale_news_abstain_bias", default: profile.staleNewsAbstainBias)
        profile.staleNewsForceCaution = doc.bool("stale_news_force_caution", default: profile.staleNewsForceCaution)
        profile.minPluginWeight = doc.double("min_plugin_weight", default: profile.minPluginWeight)
        profile.maxPluginWeight = doc.double("max_plugin_weight", default: profile.maxPluginWeight)
        profile.maxActiveWeightShare = doc.double("max_active_weight_share", default: profile.maxActiveWeightShare)
        profile.pluginGlobalWeightsCSV = doc.string("plugin_global_weights_csv", default: profile.pluginGlobalWeightsCSV)
        profile.pluginNewsCompatibilityCSV = doc.string("plugin_news_compatibility_csv", default: profile.pluginNewsCompatibilityCSV)
        profile.pluginLiquidityRobustnessCSV = doc.string("plugin_liquidity_robustness_csv", default: profile.pluginLiquidityRobustnessCSV)
        for (index, label) in ControlPlaneConstants.adaptiveRouterRegimeLabels.enumerated() {
            profile.pluginRegimeWeightsCSV[index] = doc.string("plugin_regime_\(label)_csv", default: "")
        }
        for (index, label) in ControlPlaneConstants.adaptiveRouterSessionLabels.enumerated() {
            profile.pluginSessionWeightsCSV[index] = doc.string("plugin_session_\(label)_csv", default: "")
        }
        return profile.normalized(loadedAtUTC: loadedAtUTC)
    }

    public func normalized(loadedAtUTC: Int64? = nil) -> AdaptiveRouterProfile {
        var profile = self
        profile.cautionThreshold = fxClamp(profile.cautionThreshold, 0.10, 1.50)
        profile.abstainThreshold = fxClamp(profile.abstainThreshold, 0.05, profile.cautionThreshold)
        profile.blockThreshold = fxClamp(profile.blockThreshold, 0.01, profile.abstainThreshold)
        profile.confidenceFloor = fxClamp(profile.confidenceFloor, 0.0, 1.0)
        profile.suppressionThreshold = fxClamp(profile.suppressionThreshold, 0.05, 2.50)
        profile.downweightThreshold = fxClamp(profile.downweightThreshold, 0.05, 2.50)
        profile.staleNewsAbstainBias = fxClamp(profile.staleNewsAbstainBias, 0.0, 1.0)
        profile.minPluginWeight = fxClamp(profile.minPluginWeight, 0.01, 1.0)
        profile.maxPluginWeight = fxClamp(profile.maxPluginWeight, profile.minPluginWeight, 3.0)
        profile.maxActiveWeightShare = fxClamp(profile.maxActiveWeightShare, 0.10, 0.99)
        if profile.routerMode != "WEIGHTED_ENSEMBLE" {
            profile.routerMode = "WEIGHTED_ENSEMBLE"
        }
        profile.ready = true
        if let loadedAtUTC {
            profile.loadedAtUTC = loadedAtUTC
        }
        return profile
    }

    public func globalWeight(pluginName: String) -> Double {
        guard ready, enabled else { return 1.0 }
        let value = ControlPlaneCSV.mapWeight(pluginGlobalWeightsCSV, key: pluginName, default: 1.0)
        return fxClamp(value, minPluginWeight, maxPluginWeight)
    }

    public func newsCompatibility(pluginName: String) -> Double {
        guard ready, enabled else { return 1.0 }
        let value = ControlPlaneCSV.mapWeight(pluginNewsCompatibilityCSV, key: pluginName, default: 1.0)
        return fxClamp(value, 0.05, 2.50)
    }

    public func liquidityRobustness(pluginName: String) -> Double {
        guard ready, enabled else { return 1.0 }
        let value = ControlPlaneCSV.mapWeight(pluginLiquidityRobustnessCSV, key: pluginName, default: 1.0)
        return fxClamp(value, 0.05, 2.50)
    }

    public func regimeWeight(pluginName: String, regimeLabel: String) -> Double {
        guard ready, enabled,
              let index = ControlPlaneConstants.adaptiveRouterRegimeLabels.firstIndex(of: regimeLabel),
              index < pluginRegimeWeightsCSV.count else {
            return 1.0
        }
        let value = ControlPlaneCSV.mapWeight(pluginRegimeWeightsCSV[index], key: pluginName, default: 1.0)
        return fxClamp(value, 0.05, 2.50)
    }

    public func sessionWeight(pluginName: String, sessionLabel: String) -> Double {
        guard ready, enabled,
              let index = ControlPlaneConstants.adaptiveRouterSessionLabels.firstIndex(of: sessionLabel),
              index < pluginSessionWeightsCSV.count else {
            return 1.0
        }
        let value = ControlPlaneCSV.mapWeight(pluginSessionWeightsCSV[index], key: pluginName, default: 1.0)
        return fxClamp(value, 0.05, 2.50)
    }
}

public struct ControlPlaneSnapshot: Codable, Hashable, Sendable {
    public var valid = false
    public var login: Int64 = 0
    public var magic: UInt64 = 0
    public var chartID: Int64 = 0
    public var symbol = ""
    public var barTimeUTC: Int64 = 0
    public var direction = -1
    public var signalIntensity = 0.0
    public var confidence = 0.0
    public var reliability = 0.0
    public var tradeGate = 0.0
    public var hierarchyScore = 0.0
    public var macroQuality = 0.0
    public var tradeEdgeNorm = 0.0
    public var expectedMoveNorm = 0.0
    public var policyTradeProb = 0.0
    public var policyNoTradeProb = 1.0
    public var policyEnterProb = 0.0
    public var policyExitProb = 0.0
    public var policyAddProb = 0.0
    public var policyReduceProb = 0.0
    public var policyTightenProb = 0.0
    public var policyTimeoutProb = 0.0
    public var policySizeMultiplier = 0.0
    public var policyPortfolioFit = 0.0
    public var policyCapitalEfficiency = 0.0
    public var policyLifecycleAction: PolicyLifecycleAction = .noTrade
    public var grossExposureLots = 0.0
    public var correlatedExposureLots = 0.0
    public var directionalClusterLots = 0.0
    public var capitalRiskPct = 0.0
    public var portfolioPressure = 0.0

    public init() {}

    public static func parse(tsv: String) -> ControlPlaneSnapshot {
        let doc = ControlPlaneKeyValueDocument(tsv: tsv)
        var snapshot = ControlPlaneSnapshot()
        snapshot.login = doc.int64("login")
        snapshot.magic = UInt64(max(doc.int64("magic"), 0))
        snapshot.chartID = doc.int64("chart_id")
        snapshot.symbol = doc.string("symbol")
        snapshot.barTimeUTC = doc.int64("bar_time")
        snapshot.direction = doc.int("direction", default: snapshot.direction)
        snapshot.signalIntensity = doc.double("signal_intensity", default: snapshot.signalIntensity)
        snapshot.confidence = doc.double("confidence", default: snapshot.confidence)
        snapshot.reliability = doc.double("reliability", default: snapshot.reliability)
        snapshot.tradeGate = doc.double("trade_gate", default: snapshot.tradeGate)
        snapshot.hierarchyScore = doc.double("hierarchy_score", default: snapshot.hierarchyScore)
        snapshot.macroQuality = doc.double("macro_quality", default: snapshot.macroQuality)
        snapshot.tradeEdgeNorm = doc.double("trade_edge_norm", default: snapshot.tradeEdgeNorm)
        snapshot.expectedMoveNorm = doc.double("expected_move_norm", default: snapshot.expectedMoveNorm)
        snapshot.policyTradeProb = doc.double("policy_trade_prob", default: snapshot.policyTradeProb)
        snapshot.policyNoTradeProb = doc.double("policy_no_trade_prob", default: snapshot.policyNoTradeProb)
        snapshot.policyEnterProb = doc.double("policy_enter_prob", default: snapshot.policyEnterProb)
        snapshot.policyExitProb = doc.double("policy_exit_prob", default: snapshot.policyExitProb)
        snapshot.policyAddProb = doc.double("policy_add_prob", default: snapshot.policyAddProb)
        snapshot.policyReduceProb = doc.double("policy_reduce_prob", default: snapshot.policyReduceProb)
        snapshot.policyTightenProb = doc.double("policy_tighten_prob", default: snapshot.policyTightenProb)
        snapshot.policyTimeoutProb = doc.double("policy_timeout_prob", default: snapshot.policyTimeoutProb)
        snapshot.policySizeMultiplier = doc.double("policy_size_mult", default: snapshot.policySizeMultiplier)
        snapshot.policyPortfolioFit = doc.double("policy_portfolio_fit", default: snapshot.policyPortfolioFit)
        snapshot.policyCapitalEfficiency = doc.double("policy_capital_efficiency", default: snapshot.policyCapitalEfficiency)
        let actionRaw = Int(fxClamp(
            Double(doc.int("policy_lifecycle_action")),
            Double(PolicyLifecycleAction.noTrade.rawValue),
            Double(PolicyLifecycleAction.timeout.rawValue)
        ))
        snapshot.policyLifecycleAction = PolicyLifecycleAction(rawValue: actionRaw) ?? .noTrade
        snapshot.grossExposureLots = doc.double("gross_exposure_lots", default: snapshot.grossExposureLots)
        snapshot.correlatedExposureLots = doc.double("correlated_exposure_lots", default: snapshot.correlatedExposureLots)
        snapshot.directionalClusterLots = doc.double("directional_cluster_lots", default: snapshot.directionalClusterLots)
        snapshot.capitalRiskPct = doc.double("capital_risk_pct", default: snapshot.capitalRiskPct)
        snapshot.portfolioPressure = doc.double("portfolio_pressure", default: snapshot.portfolioPressure)
        return snapshot.normalized()
    }

    public func normalized() -> ControlPlaneSnapshot {
        var snapshot = self
        snapshot.signalIntensity = fxClamp(snapshot.signalIntensity, 0.0, 4.0)
        snapshot.confidence = fxClamp(snapshot.confidence, 0.0, 1.0)
        snapshot.reliability = fxClamp(snapshot.reliability, 0.0, 1.0)
        snapshot.tradeGate = fxClamp(snapshot.tradeGate, 0.0, 1.0)
        snapshot.hierarchyScore = fxClamp(snapshot.hierarchyScore, 0.0, 1.0)
        snapshot.macroQuality = fxClamp(snapshot.macroQuality, 0.0, 1.0)
        snapshot.tradeEdgeNorm = fxClamp(snapshot.tradeEdgeNorm, -1.0, 1.0)
        snapshot.expectedMoveNorm = fxClamp(snapshot.expectedMoveNorm, 0.0, 4.0)
        snapshot.policyTradeProb = fxClamp(snapshot.policyTradeProb, 0.0, 1.0)
        snapshot.policyNoTradeProb = fxClamp(snapshot.policyNoTradeProb, 0.0, 1.0)
        snapshot.policyEnterProb = fxClamp(snapshot.policyEnterProb, 0.0, 1.0)
        snapshot.policyExitProb = fxClamp(snapshot.policyExitProb, 0.0, 1.0)
        snapshot.policyAddProb = fxClamp(snapshot.policyAddProb, 0.0, 1.0)
        snapshot.policyReduceProb = fxClamp(snapshot.policyReduceProb, 0.0, 1.0)
        snapshot.policyTightenProb = fxClamp(snapshot.policyTightenProb, 0.0, 1.0)
        snapshot.policyTimeoutProb = fxClamp(snapshot.policyTimeoutProb, 0.0, 1.0)
        snapshot.policySizeMultiplier = fxClamp(snapshot.policySizeMultiplier, 0.0, 2.0)
        snapshot.policyPortfolioFit = fxClamp(snapshot.policyPortfolioFit, 0.0, 1.0)
        snapshot.policyCapitalEfficiency = fxClamp(snapshot.policyCapitalEfficiency, 0.0, 1.0)
        snapshot.grossExposureLots = fxClamp(snapshot.grossExposureLots, 0.0, 1_000.0)
        snapshot.correlatedExposureLots = fxClamp(snapshot.correlatedExposureLots, 0.0, 1_000.0)
        snapshot.directionalClusterLots = fxClamp(snapshot.directionalClusterLots, 0.0, 1_000.0)
        snapshot.capitalRiskPct = fxClamp(snapshot.capitalRiskPct, 0.0, 100.0)
        snapshot.portfolioPressure = fxClamp(snapshot.portfolioPressure, 0.0, 2.0)
        snapshot.valid = snapshot.login > 0 &&
            snapshot.magic > 0 &&
            !snapshot.symbol.isEmpty &&
            snapshot.barTimeUTC > 0
        return snapshot
    }
}

public struct ControlPlaneAggregate: Codable, Hashable, Sendable {
    public var peerCount = 0
    public var grossIntensity = 0.0
    public var correlatedIntensity = 0.0
    public var directionalIntensity = 0.0
    public var macroOverlap = 0.0
    public var qualityOverlap = 0.0
    public var diversityBonus = 0.0
    public var concentrationPenalty = 0.0
    public var maxCapitalRiskPct = 0.0
    public var meanTradeProb = 0.0
    public var meanNoTradeProb = 0.0
    public var meanCapitalEfficiency = 0.0
    public var meanPortfolioFit = 0.0
    public var supervisorScore = 0.0
    public var score = 0.0

    public init() {}
}

public enum ControlPlaneFreshness {
    public static func artifactFresh(
        generatedAtUTC: Int64,
        expiresAtUTC: Int64,
        fallbackTTLSeconds: Int,
        nowUTC: Int64
    ) -> Bool {
        guard nowUTC > 0 else { return true }
        if expiresAtUTC > 0, nowUTC > expiresAtUTC {
            return false
        }
        if generatedAtUTC > 0,
           fallbackTTLSeconds > 0,
           nowUTC > generatedAtUTC,
           nowUTC - generatedAtUTC > Int64(fallbackTTLSeconds) {
            return false
        }
        return true
    }
}

public struct PortfolioSupervisorProfile: Codable, Hashable, Sendable {
    public var ready = false
    public var profileName = ""
    public var grossBudgetBias = 1.0
    public var correlatedBudgetBias = 1.0
    public var directionalBudgetBias = 1.0
    public var capitalRiskCapPct = 1.20
    public var macroOverlapCap = 0.92
    public var concentrationCap = 0.82
    public var supervisorWeight = 0.45
    public var hardBlockScore = 1.08
    public var policyEnterFloor = 0.42
    public var policyNoTradeCeiling = 0.74
    public var loadedAtUTC: Int64 = 0

    public init() {}

    public static func parse(tsv: String, loadedAtUTC: Int64 = 0) -> PortfolioSupervisorProfile {
        let doc = ControlPlaneKeyValueDocument(tsv: tsv)
        var profile = PortfolioSupervisorProfile()
        profile.profileName = doc.string("profile_name", default: profile.profileName)
        profile.grossBudgetBias = doc.double("gross_budget_bias", default: profile.grossBudgetBias)
        profile.correlatedBudgetBias = doc.double("correlated_budget_bias", default: profile.correlatedBudgetBias)
        profile.directionalBudgetBias = doc.double("directional_budget_bias", default: profile.directionalBudgetBias)
        profile.capitalRiskCapPct = doc.double("capital_risk_cap_pct", default: profile.capitalRiskCapPct)
        profile.macroOverlapCap = doc.double("macro_overlap_cap", default: profile.macroOverlapCap)
        profile.concentrationCap = doc.double("concentration_cap", default: profile.concentrationCap)
        profile.supervisorWeight = doc.double("supervisor_weight", default: profile.supervisorWeight)
        profile.hardBlockScore = doc.double("hard_block_score", default: profile.hardBlockScore)
        profile.policyEnterFloor = doc.double("policy_enter_floor", default: profile.policyEnterFloor)
        profile.policyNoTradeCeiling = doc.double("policy_no_trade_ceiling", default: profile.policyNoTradeCeiling)
        return profile.normalized(loadedAtUTC: loadedAtUTC)
    }

    public func normalized(loadedAtUTC: Int64? = nil) -> PortfolioSupervisorProfile {
        var profile = self
        profile.grossBudgetBias = fxClamp(profile.grossBudgetBias, 0.40, 1.60)
        profile.correlatedBudgetBias = fxClamp(profile.correlatedBudgetBias, 0.40, 1.60)
        profile.directionalBudgetBias = fxClamp(profile.directionalBudgetBias, 0.40, 1.60)
        profile.capitalRiskCapPct = fxClamp(profile.capitalRiskCapPct, 0.10, 10.0)
        profile.macroOverlapCap = fxClamp(profile.macroOverlapCap, 0.10, 2.0)
        profile.concentrationCap = fxClamp(profile.concentrationCap, 0.10, 2.0)
        profile.supervisorWeight = fxClamp(profile.supervisorWeight, 0.0, 1.0)
        profile.hardBlockScore = fxClamp(profile.hardBlockScore, 0.20, 3.0)
        profile.policyEnterFloor = fxClamp(profile.policyEnterFloor, 0.10, 0.95)
        profile.policyNoTradeCeiling = fxClamp(profile.policyNoTradeCeiling, 0.10, 0.99)
        profile.ready = true
        if let loadedAtUTC {
            profile.loadedAtUTC = loadedAtUTC
        }
        return profile
    }
}

public struct SupervisorServiceState: Codable, Hashable, Sendable {
    public var ready = false
    public var profileName = ""
    public var symbol = ""
    public var generatedAtUTC: Int64 = 0
    public var expiresAtUTC: Int64 = 0
    public var snapshotCount = 0
    public var grossPressure = 0.0
    public var directionalLongPressure = 0.0
    public var directionalShortPressure = 0.0
    public var macroPressure = 0.0
    public var concentrationPressure = 0.0
    public var freshnessPenalty = 0.0
    public var pressureVelocity = 0.0
    public var grossVelocity = 0.0
    public var longEntryBudgetMultiplier = 1.0
    public var shortEntryBudgetMultiplier = 1.0
    public var budgetMultiplier = 1.0
    public var addMultiplier = 1.0
    public var reduceBias = 0.0
    public var exitBias = 0.0
    public var entryFloor = 0.42
    public var blockScore = 1.10
    public var supervisorScore = 0.0
    public var loadedAtUTC: Int64 = 0

    public init(symbol: String = "") {
        self.symbol = symbol
    }

    public static func parse(tsv: String, nowUTC: Int64, loadedAtUTC: Int64? = nil) -> SupervisorServiceState {
        let doc = ControlPlaneKeyValueDocument(tsv: tsv)
        var state = SupervisorServiceState()
        state.profileName = doc.string("profile_name", default: state.profileName)
        state.symbol = doc.string("symbol", default: state.symbol)
        state.generatedAtUTC = doc.int64("generated_at", default: state.generatedAtUTC)
        state.expiresAtUTC = doc.int64("expires_at", default: state.expiresAtUTC)
        state.snapshotCount = doc.int("snapshot_count", default: state.snapshotCount)
        state.grossPressure = doc.double("gross_pressure", default: state.grossPressure)
        state.directionalLongPressure = doc.double("directional_long_pressure", default: state.directionalLongPressure)
        state.directionalShortPressure = doc.double("directional_short_pressure", default: state.directionalShortPressure)
        state.macroPressure = doc.double("macro_pressure", default: state.macroPressure)
        state.concentrationPressure = doc.double("concentration_pressure", default: state.concentrationPressure)
        state.freshnessPenalty = doc.double("freshness_penalty", default: state.freshnessPenalty)
        state.pressureVelocity = doc.double("pressure_velocity", default: state.pressureVelocity)
        state.grossVelocity = doc.double("gross_velocity", default: state.grossVelocity)
        state.longEntryBudgetMultiplier = doc.double("long_entry_budget_mult", default: state.longEntryBudgetMultiplier)
        state.shortEntryBudgetMultiplier = doc.double("short_entry_budget_mult", default: state.shortEntryBudgetMultiplier)
        state.budgetMultiplier = doc.double("budget_multiplier", default: state.budgetMultiplier)
        state.addMultiplier = doc.double("add_multiplier", default: state.addMultiplier)
        state.reduceBias = doc.double("reduce_bias", default: state.reduceBias)
        state.exitBias = doc.double("exit_bias", default: state.exitBias)
        state.entryFloor = doc.double("entry_floor", default: state.entryFloor)
        state.blockScore = doc.double("block_score", default: state.blockScore)
        state.supervisorScore = doc.double("supervisor_score", default: state.supervisorScore)
        return state.normalized(nowUTC: nowUTC, loadedAtUTC: loadedAtUTC ?? nowUTC)
    }

    public func normalized(nowUTC: Int64, loadedAtUTC: Int64? = nil) -> SupervisorServiceState {
        var state = self
        state.snapshotCount = Int(fxClamp(Double(state.snapshotCount), 0.0, 10_000.0))
        state.grossPressure = fxClamp(state.grossPressure, 0.0, 2.0)
        state.directionalLongPressure = fxClamp(state.directionalLongPressure, 0.0, 2.0)
        state.directionalShortPressure = fxClamp(state.directionalShortPressure, 0.0, 2.0)
        state.macroPressure = fxClamp(state.macroPressure, 0.0, 1.5)
        state.concentrationPressure = fxClamp(state.concentrationPressure, 0.0, 1.0)
        state.freshnessPenalty = fxClamp(state.freshnessPenalty, 0.0, 1.0)
        state.pressureVelocity = fxClamp(state.pressureVelocity, -1.0, 1.0)
        state.grossVelocity = fxClamp(state.grossVelocity, -1.0, 1.0)
        state.longEntryBudgetMultiplier = fxClamp(state.longEntryBudgetMultiplier, 0.10, 1.20)
        state.shortEntryBudgetMultiplier = fxClamp(state.shortEntryBudgetMultiplier, 0.10, 1.20)
        state.budgetMultiplier = fxClamp(state.budgetMultiplier, 0.10, 1.20)
        if abs(state.longEntryBudgetMultiplier - 1.0) < 1e-6,
           abs(state.shortEntryBudgetMultiplier - 1.0) < 1e-6,
           abs(state.budgetMultiplier - 1.0) > 1e-6 {
            state.longEntryBudgetMultiplier = state.budgetMultiplier
            state.shortEntryBudgetMultiplier = state.budgetMultiplier
        }
        state.addMultiplier = fxClamp(state.addMultiplier, 0.10, 1.40)
        state.reduceBias = fxClamp(state.reduceBias, 0.0, 1.0)
        state.exitBias = fxClamp(state.exitBias, 0.0, 1.0)
        state.entryFloor = fxClamp(state.entryFloor, 0.10, 0.95)
        state.blockScore = fxClamp(state.blockScore, 0.20, 3.0)
        state.supervisorScore = fxClamp(state.supervisorScore, 0.0, 3.0)
        state.ready = !state.symbol.isEmpty && ControlPlaneFreshness.artifactFresh(
            generatedAtUTC: state.generatedAtUTC,
            expiresAtUTC: state.expiresAtUTC,
            fallbackTTLSeconds: 240,
            nowUTC: nowUTC
        )
        if let loadedAtUTC {
            state.loadedAtUTC = loadedAtUTC
        }
        return state
    }

    public func resolvedSymbol(_ symbol: String) -> SupervisorServiceState {
        var state = self
        if state.symbol.isEmpty || state.symbol == "__GLOBAL__" {
            state.symbol = symbol
        }
        return state
    }
}

public struct SupervisorCommandState: Codable, Hashable, Sendable {
    public var ready = false
    public var profileName = ""
    public var symbol = ""
    public var generatedAtUTC: Int64 = 0
    public var expiresAtUTC: Int64 = 0
    public var entryBudgetMultiplier = 1.0
    public var longEntryBudgetMultiplier = 1.0
    public var shortEntryBudgetMultiplier = 1.0
    public var holdBudgetMultiplier = 1.0
    public var addCapMultiplier = 1.0
    public var reduceBias = 0.0
    public var exitBias = 0.0
    public var tightenBias = 0.0
    public var timeoutBias = 0.0
    public var longBlock = false
    public var shortBlock = false
    public var blockScore = 1.10
    public var maxActiveModels = 12
    public var championOnly = false
    public var loadedAtUTC: Int64 = 0

    public init(symbol: String = "") {
        self.symbol = symbol
    }

    public static func parse(symbol: String, tsv: String, nowUTC: Int64, loadedAtUTC: Int64? = nil) -> SupervisorCommandState {
        let doc = ControlPlaneKeyValueDocument(tsv: tsv)
        var state = SupervisorCommandState(symbol: symbol)
        state.profileName = doc.string("profile_name", default: state.profileName)
        state.symbol = doc.string("symbol", default: state.symbol)
        state.generatedAtUTC = doc.int64("generated_at", default: state.generatedAtUTC)
        state.expiresAtUTC = doc.int64("expires_at", default: state.expiresAtUTC)
        state.entryBudgetMultiplier = doc.double("entry_budget_mult", default: state.entryBudgetMultiplier)
        state.longEntryBudgetMultiplier = doc.double("long_entry_budget_mult", default: state.longEntryBudgetMultiplier)
        state.shortEntryBudgetMultiplier = doc.double("short_entry_budget_mult", default: state.shortEntryBudgetMultiplier)
        state.holdBudgetMultiplier = doc.double("hold_budget_mult", default: state.holdBudgetMultiplier)
        state.addCapMultiplier = doc.double("add_cap_mult", default: state.addCapMultiplier)
        state.reduceBias = doc.double("reduce_bias", default: state.reduceBias)
        state.exitBias = doc.double("exit_bias", default: state.exitBias)
        state.tightenBias = doc.double("tighten_bias", default: state.tightenBias)
        state.timeoutBias = doc.double("timeout_bias", default: state.timeoutBias)
        state.longBlock = doc.bool("long_block", default: state.longBlock)
        state.shortBlock = doc.bool("short_block", default: state.shortBlock)
        state.blockScore = doc.double("block_score", default: state.blockScore)
        state.maxActiveModels = doc.int("max_active_models", default: state.maxActiveModels)
        state.championOnly = doc.bool("champion_only", default: state.championOnly)
        return state.normalized(nowUTC: nowUTC, loadedAtUTC: loadedAtUTC ?? nowUTC)
    }

    public func normalized(nowUTC: Int64, loadedAtUTC: Int64? = nil) -> SupervisorCommandState {
        var state = self
        state.entryBudgetMultiplier = fxClamp(state.entryBudgetMultiplier, 0.10, 1.20)
        state.longEntryBudgetMultiplier = fxClamp(state.longEntryBudgetMultiplier, 0.10, 1.20)
        state.shortEntryBudgetMultiplier = fxClamp(state.shortEntryBudgetMultiplier, 0.10, 1.20)
        if abs(state.longEntryBudgetMultiplier - 1.0) < 1e-6,
           abs(state.shortEntryBudgetMultiplier - 1.0) < 1e-6,
           abs(state.entryBudgetMultiplier - 1.0) > 1e-6 {
            state.longEntryBudgetMultiplier = state.entryBudgetMultiplier
            state.shortEntryBudgetMultiplier = state.entryBudgetMultiplier
        }
        state.holdBudgetMultiplier = fxClamp(state.holdBudgetMultiplier, 0.10, 1.20)
        state.addCapMultiplier = fxClamp(state.addCapMultiplier, 0.05, 1.20)
        state.reduceBias = fxClamp(state.reduceBias, 0.0, 1.0)
        state.exitBias = fxClamp(state.exitBias, 0.0, 1.0)
        state.tightenBias = fxClamp(state.tightenBias, 0.0, 1.0)
        state.timeoutBias = fxClamp(state.timeoutBias, 0.0, 1.0)
        state.blockScore = fxClamp(state.blockScore, 0.20, 3.0)
        state.maxActiveModels = Int(fxClamp(Double(state.maxActiveModels), 1.0, Double(FXDataEngineConstants.aiCount)))
        state.ready = ControlPlaneFreshness.artifactFresh(
            generatedAtUTC: state.generatedAtUTC,
            expiresAtUTC: state.expiresAtUTC,
            fallbackTTLSeconds: 240,
            nowUTC: nowUTC
        )
        if let loadedAtUTC {
            state.loadedAtUTC = loadedAtUTC
        }
        return state
    }

    public func resolvedSymbol(_ symbol: String) -> SupervisorCommandState {
        var state = self
        if state.symbol == "__GLOBAL__" {
            state.symbol = symbol
        }
        return state
    }

    public func blocksDirection(_ direction: Int) -> Bool {
        guard ready else { return false }
        if direction == 1 {
            return longBlock
        }
        if direction == 0 {
            return shortBlock
        }
        return longBlock && shortBlock
    }

    public func entryBudgetMultiplier(for direction: Int) -> Double {
        guard ready else { return 1.0 }
        if direction == 1 {
            return fxClamp(longEntryBudgetMultiplier, 0.10, 1.20)
        }
        if direction == 0 {
            return fxClamp(shortEntryBudgetMultiplier, 0.10, 1.20)
        }
        return fxClamp(entryBudgetMultiplier, 0.10, 1.20)
    }
}
