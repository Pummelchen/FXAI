import Foundation

public enum FXAIExecutionProfile: String, CaseIterable, Codable, Identifiable, Sendable {
    case `default`
    case primeECN = "prime-ecn"
    case retailFX = "retail-fx"
    case stress
    case tightFX = "tight-fx"

    public var id: String { rawValue }

    public var title: String {
        switch self {
        case .default: "Default"
        case .primeECN: "Prime ECN"
        case .retailFX: "Retail FX"
        case .stress: "Stress"
        case .tightFX: "Tight FX"
        }
    }
}

public enum FXAISymbolPack: String, CaseIterable, Codable, Identifiable, Sendable {
    case none = ""
    case commodityFX = "commodity-fx"
    case dollarCore = "dollar-core"
    case majors
    case metalsRisk = "metals-risk"
    case yenCross = "yen-cross"

    public var id: String { rawValue.isEmpty ? "none" : rawValue }

    public var title: String {
        switch self {
        case .none: "None"
        case .commodityFX: "Commodity FX"
        case .dollarCore: "Dollar Core"
        case .majors: "Majors"
        case .metalsRisk: "Metals Risk"
        case .yenCross: "Yen Cross"
        }
    }
}

public enum AuditScenarioPreset: String, CaseIterable, Codable, Identifiable, Sendable {
    case smoke
    case walkForward = "walkforward"
    case macro
    case portfolio
    case adversarial

    public var id: String { rawValue }

    public var title: String {
        switch self {
        case .smoke: "Smoke"
        case .walkForward: "Walk-Forward"
        case .macro: "Macro Event"
        case .portfolio: "Portfolio Pack"
        case .adversarial: "Adversarial"
        }
    }

    public var cliList: String {
        switch self {
        case .smoke:
            "{market_recent}"
        case .walkForward:
            "{market_recent,market_walkforward}"
        case .macro:
            "{market_recent,market_walkforward,market_macro_event}"
        case .portfolio:
            "{market_recent,market_walkforward,market_macro_event,market_adversarial}"
        case .adversarial:
            "{market_recent,market_walkforward,market_adversarial}"
        }
    }
}

public enum OfflineWorkflowPreset: String, CaseIterable, Codable, Identifiable, Sendable {
    case smoke
    case continuous
    case promotion
    case governance

    public var id: String { rawValue }

    public var title: String {
        switch self {
        case .smoke: "Smoke"
        case .continuous: "Continuous"
        case .promotion: "Promotion"
        case .governance: "Governance"
        }
    }
}

public struct AuditLabDraft: Codable, Hashable, Sendable {
    public var pluginName: String
    public var allPlugins: Bool
    public var scenarioPreset: AuditScenarioPreset
    public var symbol: String
    public var symbolPack: FXAISymbolPack
    public var executionProfile: FXAIExecutionProfile
    public var bars: Int
    public var horizon: Int
    public var sequenceBars: Int
    public var normalization: String
    public var schemaID: String
    public var seed: Int

    public init(
        pluginName: String = "ai_mlp",
        allPlugins: Bool = false,
        scenarioPreset: AuditScenarioPreset = .walkForward,
        symbol: String = "EURUSD",
        symbolPack: FXAISymbolPack = .none,
        executionProfile: FXAIExecutionProfile = .default,
        bars: Int = 1500,
        horizon: Int = 5,
        sequenceBars: Int = 64,
        normalization: String = "auto",
        schemaID: String = "default",
        seed: Int = 42
    ) {
        self.pluginName = pluginName
        self.allPlugins = allPlugins
        self.scenarioPreset = scenarioPreset
        self.symbol = symbol
        self.symbolPack = symbolPack
        self.executionProfile = executionProfile
        self.bars = bars
        self.horizon = horizon
        self.sequenceBars = sequenceBars
        self.normalization = normalization
        self.schemaID = schemaID
        self.seed = seed
    }
}

public struct BacktestBuilderDraft: Codable, Hashable, Sendable {
    public var pluginName: String
    public var symbol: String
    public var scenarioPreset: AuditScenarioPreset
    public var executionProfile: FXAIExecutionProfile
    public var baselineName: String
    public var bars: Int
    public var sequenceBars: Int

    public init(
        pluginName: String = "ai_mlp",
        symbol: String = "EURUSD",
        scenarioPreset: AuditScenarioPreset = .adversarial,
        executionProfile: FXAIExecutionProfile = .default,
        baselineName: String = "eurusd_phase2_smoke",
        bars: Int = 2000,
        sequenceBars: Int = 64
    ) {
        self.pluginName = pluginName
        self.symbol = symbol
        self.scenarioPreset = scenarioPreset
        self.executionProfile = executionProfile
        self.baselineName = baselineName
        self.bars = bars
        self.sequenceBars = sequenceBars
    }
}

public struct OfflineLabDraft: Codable, Hashable, Sendable {
    public var workflowPreset: OfflineWorkflowPreset
    public var profileName: String
    public var symbol: String
    public var symbolPack: FXAISymbolPack
    public var monthsList: String
    public var autoExport: Bool
    public var includeBootstrap: Bool
    public var includeBestParams: Bool
    public var includeDeployProfiles: Bool
    public var includeLineage: Bool
    public var includeMinimalBundle: Bool
    public var runtimeMode: String
    public var topPlugins: Int
    public var limitExperiments: Int
    public var limitRuns: Int

    public init(
        workflowPreset: OfflineWorkflowPreset = .continuous,
        profileName: String = "continuous",
        symbol: String = "EURUSD",
        symbolPack: FXAISymbolPack = .majors,
        monthsList: String = "3,6,12",
        autoExport: Bool = true,
        includeBootstrap: Bool = true,
        includeBestParams: Bool = true,
        includeDeployProfiles: Bool = true,
        includeLineage: Bool = true,
        includeMinimalBundle: Bool = false,
        runtimeMode: String = "research",
        topPlugins: Int = 8,
        limitExperiments: Int = 32,
        limitRuns: Int = 96
    ) {
        self.workflowPreset = workflowPreset
        self.profileName = profileName
        self.symbol = symbol
        self.symbolPack = symbolPack
        self.monthsList = monthsList
        self.autoExport = autoExport
        self.includeBootstrap = includeBootstrap
        self.includeBestParams = includeBestParams
        self.includeDeployProfiles = includeDeployProfiles
        self.includeLineage = includeLineage
        self.includeMinimalBundle = includeMinimalBundle
        self.runtimeMode = runtimeMode
        self.topPlugins = topPlugins
        self.limitExperiments = limitExperiments
        self.limitRuns = limitRuns
    }
}
