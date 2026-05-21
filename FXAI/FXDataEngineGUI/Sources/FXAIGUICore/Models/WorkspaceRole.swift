import Foundation

public enum WorkspaceRole: String, CaseIterable, Codable, Identifiable, Sendable {
    case liveTrader = "live_trader"
    case demoTrader = "demo_trader"
    case backtester = "backtester"
    case researcher = "researcher"
    case architect = "architect"

    public var id: String { rawValue }

    public var title: String {
        switch self {
        case .liveTrader: "Live Trader"
        case .demoTrader: "Demo Trader"
        case .backtester: "Backtester"
        case .researcher: "EA Researcher"
        case .architect: "System Architect"
        }
    }

    public var subtitle: String {
        switch self {
        case .liveTrader: "Understand the promoted live state quickly."
        case .demoTrader: "Observe FXAI safely under real-time market flow."
        case .backtester: "Launch realistic evaluations and compare outcomes."
        case .researcher: "Tune, inspect, compare, and promote the model zoo."
        case .architect: "Operate the research OS, governance, and recovery surface."
        }
    }

    public var symbolName: String {
        switch self {
        case .liveTrader: "bolt.horizontal.circle.fill"
        case .demoTrader: "waveform.path.ecg.rectangle"
        case .backtester: "chart.xyaxis.line"
        case .researcher: "brain.head.profile"
        case .architect: "shippingbox.circle.fill"
        }
    }

    public var focusAreas: [String] {
        switch self {
        case .liveTrader:
            [
                "Promoted deployment profiles",
                "Artifact freshness",
                "Runtime health",
                "Command-safe inspection"
            ]
        case .demoTrader:
            [
                "Audit expectation versus live behavior",
                "Trade cadence",
                "Session-edge behavior",
                "Safe observation"
            ]
        case .backtester:
            [
                "Focused audit packs",
                "Tester realism",
                "Baseline comparison",
                "Scenario coverage"
            ]
        case .researcher:
            [
                "Plugin-zoo exploration",
                "Offline Lab flows",
                "Promotion lineage",
                "Profile generation"
            ]
        case .architect:
            [
                "Governance health",
                "Turso state",
                "Recovery paths",
                "Platform verification"
            ]
        }
    }

    public var ignoreAtFirst: String {
        switch self {
        case .liveTrader: "Ignore plugin internals and deep Offline Lab governance until live state is stable."
        case .demoTrader: "Ignore foundation and branch workflows until you have seen stable demo behavior."
        case .backtester: "Ignore Turso administration and distillation details at first."
        case .researcher: "Ignore live chart operation details while tuning and comparing candidates."
        case .architect: "Ignore one-off manual plugin testing as the primary control surface."
        }
    }
}
