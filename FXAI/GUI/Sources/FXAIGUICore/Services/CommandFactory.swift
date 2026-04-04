import Foundation

public enum CommandFactory {
    public static func recipes(projectRoot: URL) -> [CommandRecipe] {
        let root = projectRoot.path

        return [
            CommandRecipe(
                role: .liveTrader,
                title: "Verify Live Readiness",
                summary: "Check environment, compile state, and current promoted runtime health.",
                command: """
                cd \(shellQuoted(root))
                python3 Tools/fxai_testlab.py verify-all
                """,
                commandKind: "Verification"
            ),
            CommandRecipe(
                role: .demoTrader,
                title: "Run Focused Audit",
                summary: "Validate one candidate setup before watching it on a demo chart.",
                command: """
                cd \(shellQuoted(root))
                python3 Tools/fxai_testlab.py run-audit --plugin-list "{ai_mlp}" --scenario-list "{market_recent,market_walkforward,market_macro_event}" --symbol EURUSD
                """,
                commandKind: "Audit"
            ),
            CommandRecipe(
                role: .backtester,
                title: "Compile Both MT5 Targets",
                summary: "Build the EA and Audit Runner before a Strategy Tester session.",
                command: """
                cd \(shellQuoted(root))
                python3 Tools/fxai_testlab.py compile-main
                python3 Tools/fxai_testlab.py compile-audit
                """,
                commandKind: "Build"
            ),
            CommandRecipe(
                role: .researcher,
                title: "Run Continuous Tuning",
                summary: "Export current windows and tune the model zoo through Offline Lab.",
                command: """
                cd \(shellQuoted(root))
                python3 Tools/fxai_offline_lab.py tune-zoo --profile continuous --auto-export --symbol-pack majors --months-list 3,6,12
                """,
                commandKind: "Offline Lab"
            ),
            CommandRecipe(
                role: .researcher,
                title: "Promote Best Params",
                summary: "Emit promoted parameter packs and deployment artifacts.",
                command: """
                cd \(shellQuoted(root))
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
                cd \(shellQuoted(root))
                python3 Tools/fxai_offline_lab.py validate-env
                python3 Tools/fxai_offline_lab.py bootstrap --seed-demo
                """,
                commandKind: "Bootstrap"
            ),
            CommandRecipe(
                role: .architect,
                title: "Run Autonomous Governance",
                summary: "Refresh research outputs, lineage, and promoted deployment state.",
                command: """
                cd \(shellQuoted(root))
                python3 Tools/fxai_offline_lab.py autonomous-governance --profile continuous
                python3 Tools/fxai_offline_lab.py dashboard --profile continuous
                """,
                commandKind: "Governance"
            )
        ]
    }

    private static func shellQuoted(_ value: String) -> String {
        "'" + value.replacingOccurrences(of: "'", with: "'\"'\"'") + "'"
    }
}
