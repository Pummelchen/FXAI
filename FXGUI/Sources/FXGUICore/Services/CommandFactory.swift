import Foundation

public enum CommandFactory {
    public static func recipes(projectRoot: URL) -> [CommandRecipe] {
        let root = projectRoot.path
        let toolRoot = toolProjectRoot(projectRoot).path

        return [
            CommandRecipe(
                role: .liveTrader,
                title: "Verify Swift Runtime Readiness",
                summary: "Check Swift package health and current promoted runtime artifacts.",
                command: """
                cd \(shellQuoted(root))
                swift test --package-path FXDataEngine
                swift test --package-path FXPlugins
                swift test --package-path FXBacktest
                """,
                commandKind: "Verification"
            ),
            CommandRecipe(
                role: .liveTrader,
                title: "Refresh NewsPulse",
                summary: "Rebuild the shared news-risk snapshot before reviewing active trade gates.",
                command: """
                cd \(shellQuoted(toolRoot))
                python3 Tools/fxai_offline_lab.py newspulse-once
                python3 Tools/fxai_offline_lab.py newspulse-health
                """,
                commandKind: "NewsPulse"
            ),
            CommandRecipe(
                role: .demoTrader,
                title: "Run Focused Audit",
                summary: "Validate one candidate setup before watching it on a demo chart.",
                command: """
                cd \(shellQuoted(toolRoot))
                python3 Tools/fxai_testlab.py run-audit --plugin-list "{ai_mlp}" --scenario-list "{market_recent,market_walkforward,market_macro_event}" --symbol EURUSD
                """,
                commandKind: "Audit"
            ),
            CommandRecipe(
                role: .backtester,
                title: "Run Swift Backtest Checks",
                summary: "Build and test the Swift data engine, plugin package, and backtest runtime before a campaign.",
                command: """
                cd \(shellQuoted(root))
                swift test --package-path FXDataEngine
                swift test --package-path FXPlugins
                swift test --package-path FXBacktest
                """,
                commandKind: "Build"
            ),
            CommandRecipe(
                role: .researcher,
                title: "Run Continuous Tuning",
                summary: "Export current windows and tune the model zoo through Offline Lab.",
                command: """
                cd \(shellQuoted(toolRoot))
                python3 Tools/fxai_offline_lab.py tune-zoo --profile continuous --auto-export --symbol-pack majors --months-list 3,6,12
                """,
                commandKind: "Offline Lab"
            ),
            CommandRecipe(
                role: .researcher,
                title: "Promote Best Params",
                summary: "Emit promoted parameter packs and deployment artifacts.",
                command: """
                cd \(shellQuoted(toolRoot))
                python3 Tools/fxai_offline_lab.py best-params --profile continuous
                python3 Tools/fxai_offline_lab.py deploy-profiles --profile continuous
                """,
                commandKind: "Promotion"
            ),
            CommandRecipe(
                role: .architect,
                title: "Bootstrap Research OS",
                summary: "Validate the environment and seed the research operating system.",
                command: """
                cd \(shellQuoted(toolRoot))
                python3 Tools/fxai_offline_lab.py validate-env
                python3 Tools/fxai_offline_lab.py bootstrap --seed-demo
                """,
                commandKind: "Bootstrap"
            ),
            CommandRecipe(
                role: .architect,
                title: "Validate NewsPulse",
                summary: "Check NewsPulse config, the operator policy, and current daemon/source health.",
                command: """
                cd \(shellQuoted(toolRoot))
                python3 Tools/fxai_offline_lab.py newspulse-validate
                python3 Tools/fxai_offline_lab.py newspulse-health
                """,
                commandKind: "NewsPulse"
            ),
            CommandRecipe(
                role: .architect,
                title: "Run Autonomous Governance",
                summary: "Refresh research outputs, lineage, and promoted deployment state.",
                command: """
                cd \(shellQuoted(toolRoot))
                python3 Tools/fxai_offline_lab.py autonomous-governance --profile continuous
                python3 Tools/fxai_offline_lab.py dashboard --profile continuous
                """,
                commandKind: "Governance"
            )
        ]
    }

    private static func toolProjectRoot(_ projectRoot: URL) -> URL {
        let dataEngineRoot = projectRoot.appendingPathComponent("FXDataEngine", isDirectory: true)
        let dataEngineTools = dataEngineRoot.appendingPathComponent("Tools", isDirectory: true)
        if FileManager.default.fileExists(atPath: dataEngineTools.path) {
            return dataEngineRoot
        }
        let rootTools = projectRoot.appendingPathComponent("Tools", isDirectory: true)
        if FileManager.default.fileExists(atPath: rootTools.path) {
            return projectRoot
        }
        return dataEngineRoot
    }

    private static func shellQuoted(_ value: String) -> String {
        FXAICommandSecurityPolicy.shellQuoted(value)
    }
}
