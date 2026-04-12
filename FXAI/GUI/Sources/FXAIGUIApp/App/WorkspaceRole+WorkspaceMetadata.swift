import FXAIGUICore

extension WorkspaceRole {
    var defaultDestination: SidebarDestination {
        switch self {
        case .liveTrader:
            .liveOverview
        case .demoTrader:
            .demoOverview
        case .backtester:
            .backtestBuilder
        case .researcher:
            .researchWorkspace
        case .architect:
            .platformControl
        }
    }

    var workspaceTitle: String {
        switch self {
        case .liveTrader:
            "Live Overview"
        case .demoTrader:
            "Demo Overview"
        case .backtester:
            "Backtest Builder"
        case .researcher:
            "Research Workspace"
        case .architect:
            "Platform Control"
        }
    }

    var workspaceSummary: String {
        switch self {
        case .liveTrader:
            "Inspect live posture, artifact freshness, and runtime blockers before trusting a trade."
        case .demoTrader:
            "Study live behavior safely and compare it against audit expectation before risking real capital."
        case .backtester:
            "Generate realistic compile, audit, and baseline flows before handing off into MT5 Strategy Tester."
        case .researcher:
            "Move between plugin discovery, Offline Lab flows, promotion state, and visual analysis without context switching."
        case .architect:
            "Operate Research OS health, recovery, and governance from one coherent control surface."
        }
    }

    var workspaceBenefits: [String] {
        switch self {
        case .liveTrader:
            [
                "Catch stale NewsPulse, rates, cross-asset, or microstructure state before a live mistake.",
                "Read why FXAI is allowing, cautioning, blocking, or abstaining on the selected symbol.",
                "Move directly into runtime, incidents, and execution layers when a live posture needs explanation."
            ]
        case .demoTrader:
            [
                "See how the same runtime safeguards behave without live capital pressure.",
                "Compare session changes, hostile execution, and abstention against audit-led expectations.",
                "Jump from observed behavior into audit and reports without rebuilding the workflow from memory."
            ]
        case .backtester:
            [
                "Start from the exact pre-tester flow FXAI expects instead of an ad-hoc MT5 session.",
                "Keep scenario realism, execution assumptions, and baselines aligned across repeated tests.",
                "Hand the same certified setup from GUI into terminal and Strategy Tester."
            ]
        case .researcher:
            [
                "Work the full decision stack: plugin zoo, routing, calibration, execution, and promotion.",
                "Keep promotion, lineage, and artifact review attached to the tuning loop.",
                "Use quick links and command recipes instead of mentally stitching long Offline Lab chains."
            ]
        case .architect:
            [
                "See branch state, audit events, incidents, and recovery commands in one operator frame.",
                "Recover from drift or stale artifacts without loosening live safety controls.",
                "Keep documentation, packaging, and platform validation close to the operational controls."
            ]
        }
    }

    var workspaceScenarioExamples: [String] {
        switch self {
        case .liveTrader:
            [
                "Central-bank day: confirm NewsPulse and Rates Engine before trusting a strong directional score.",
                "Thin-liquidity session: inspect Microstructure and Execution Quality before accepting a nominal edge."
            ]
        case .demoTrader:
            [
                "Asia session gap study: watch whether hostile liquidity triggers abstention rather than forced trades.",
                "Compare a demo runtime block against the latest audit report before changing any configuration."
            ]
        case .backtester:
            [
                "Prepare a USDJPY macro-week versus quiet-week comparison with the same compile and audit assumptions.",
                "Save a repeatable baseline workflow before broadening a Strategy Tester campaign."
            ]
        case .researcher:
            [
                "Review a challenger that improves raw direction but worsens post-cost tradability before promotion.",
                "Trace a winning family across plugin zoo, deployment profiles, and advanced attribution views."
            ]
        case .architect:
            [
                "After a restart blocks live runtime, identify the stale upstream artifact and recover it safely.",
                "Verify branch, audit-log, and recovery state before handing the system back to traders."
            ]
        }
    }

    var workspaceQuickDestinations: [SidebarDestination] {
        switch self {
        case .liveTrader:
            [.runtimeMonitor, .newsPulse, .ratesEngine, .crossAsset, .microstructure, .incidents]
        case .demoTrader:
            [.runtimeMonitor, .auditLab, .reports, .microstructure, .executionQuality, .commands]
        case .backtester:
            [.backtestBuilder, .auditLab, .commands, .reports]
        case .researcher:
            [.plugins, .offlineLab, .promotionCenter, .advancedVisuals, .reports, .commands]
        case .architect:
            [.researchControl, .incidents, .settings, .commands]
        }
    }
}
