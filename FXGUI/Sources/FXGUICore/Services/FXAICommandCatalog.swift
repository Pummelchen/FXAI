import Foundation

public enum FXAICommandCatalog {
    public static func snapshot(projectRoot: URL) -> FXGUICommandCatalogSnapshot {
        let root = shellQuoted(projectRoot.path)
        let commands = [
            FXGUICommandDefinition(
                id: "fxai.certify.all",
                ownerProject: "FXTools",
                role: .architect,
                title: "Certify Full FXAI Stack",
                summary: "Build, test, boundary-scan, and emit certification evidence for all packages.",
                apiVersion: "fxdatabase.fxbacktest.v1",
                executionPath: .terminalFallback,
                riskLevel: .localBuild,
                terminalEquivalent: "cd \(root) && ./fxai certify --all",
                apiPath: "/v1/certification/evidence",
                expectedResultType: "FXAICertificationEvidenceRequest",
                logStreamKey: "certification"
            ),
            FXGUICommandDefinition(
                id: "fxdatabase.status",
                ownerProject: "FXDatabase",
                role: .architect,
                title: "Check FXDatabase API",
                summary: "Read FXDatabase service status without touching ClickHouse directly.",
                apiVersion: "fxdatabase.fxbacktest.v1",
                executionPath: .versionedAPI,
                riskLevel: .readOnly,
                terminalEquivalent: "curl http://127.0.0.1:5066/v1/status",
                apiPath: "/v1/status",
                expectedResultType: "FXBacktestAPIStatusResponse",
                logStreamKey: "fxdatabase"
            ),
            FXGUICommandDefinition(
                id: "fxbacktest.plugin.sinetest",
                ownerProject: "FXBacktest",
                role: .backtester,
                title: "Run Plugin SineTest",
                summary: "Run a selected root FXPlugins plugin through the unified FXDataEngine payload path.",
                apiVersion: "fxbacktest.plugin-api.v1",
                executionPath: .terminalFallback,
                riskLevel: .backtestExecution,
                parameters: [
                    FXGUICommandParameter(key: "plugin_id", displayName: "Plugin", required: true),
                    FXGUICommandParameter(key: "backend", displayName: "Backend", required: true, defaultValue: "swiftScalar")
                ],
                terminalEquivalent: "cd \(root) && swift test --package-path FXBacktest --filter FXBacktestPluginZooBridgeTests",
                apiPath: nil,
                expectedResultType: "BacktestPassResult",
                logStreamKey: "plugin-test"
            ),
            FXGUICommandDefinition(
                id: "fxbacktest.agent.selfcheck",
                ownerProject: "FXBacktestAgent",
                role: .architect,
                title: "Check Backtest Agent",
                summary: "Verify that a LAN worker fails closed until local certification and SineTest are available.",
                apiVersion: "fxbacktest.agent.tcp.v1",
                executionPath: .terminalFallback,
                riskLevel: .readOnly,
                terminalEquivalent: "cd \(root) && swift run --package-path FXBacktestAgent FXBacktestAgent --self-check",
                apiPath: nil,
                expectedResultType: "FXBacktestAgentEnvelope",
                logStreamKey: "agent"
            ),
            FXGUICommandDefinition(
                id: "fxexecution.kill_switch.status",
                ownerProject: "FXExecutionContracts",
                role: .liveTrader,
                title: "Review Kill Switch",
                summary: "Inspect account, symbol, and global execution switch state before order-capable workflows.",
                apiVersion: "fxexecution.contracts.v1",
                executionPath: .disabledUntilAPIExists,
                riskLevel: .liveExecution,
                terminalEquivalent: "FXGUI API-only action pending FXDatabase execution-safety endpoint",
                apiPath: "/v1/execution/kill-switch/status",
                expectedResultType: "FXExecutionKillSwitchState",
                logStreamKey: "execution-safety"
            )
        ]
        return FXGUICommandCatalogSnapshot(commands: commands)
    }

    private static func shellQuoted(_ value: String) -> String {
        "'" + value.replacingOccurrences(of: "'", with: "'\"'\"'") + "'"
    }
}
