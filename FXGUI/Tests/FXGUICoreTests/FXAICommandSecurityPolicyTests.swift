import FXGUICore
import Foundation
import Testing

@Suite("FXAI command security policy")
struct FXAICommandSecurityPolicyTests {
    private let root = URL(fileURLWithPath: "/tmp/fxai", isDirectory: true)

    @Test
    func approvesCurrentGeneratedCommandWorkflows() throws {
        let auditCommand = RunBuilderCommandFactory.auditCommand(
            projectRoot: root,
            draft: AuditLabDraft(
                pluginName: "ai_mlp",
                allPlugins: false,
                scenarioPreset: .portfolio,
                symbol: "EURUSD",
                symbolPack: .none,
                executionProfile: .primeECN,
                bars: 2400,
                horizon: 7,
                sequenceBars: 96,
                normalization: "auto",
                schemaID: "default",
                seed: 99
            )
        )
        let backtestCommand = RunBuilderCommandFactory.backtestWorkflow(
            projectRoot: root,
            draft: BacktestBuilderDraft(pluginName: "tree_lgbm", symbol: "EURUSD")
        )
        let offlineCommand = RunBuilderCommandFactory.offlineWorkflow(
            projectRoot: root,
            draft: OfflineLabDraft(
                workflowPreset: .governance,
                profileName: "bestparams",
                symbol: "EURUSD",
                symbolPack: .majors,
                monthsList: "3,6,12",
                autoExport: true,
                includeBootstrap: true,
                includeBestParams: true,
                includeDeployProfiles: true,
                includeLineage: true,
                includeMinimalBundle: true,
                runtimeMode: "production",
                topPlugins: 8,
                limitExperiments: 24,
                limitRuns: 64
            )
        )
        let researchCommand = ResearchOSCommandFactory.branchCommand(
            projectRoot: root,
            draft: ResearchOSBranchDraft(
                action: .pitrRestore,
                profileName: "continuous",
                sourceDatabase: "fxai-prod",
                targetDatabase: "fxai-restore-1",
                timestamp: "2026-04-05T10:11:12Z",
                groupName: "trading",
                locationName: "fra",
                tokenExpiration: "3d",
                readOnlyToken: true
            )
        )
        let packageCommand = [
            "cd \(FXAICommandSecurityPolicy.shellQuoted(root.appendingPathComponent("FXGUI").path))",
            "./Tools/package_gui_release.sh"
        ].joined(separator: "\n")

        for command in [auditCommand, backtestCommand, offlineCommand, researchCommand, packageCommand] {
            #expect(FXAICommandSecurityPolicy.assess(command: command, projectRoot: root).approved)
            #expect(throws: Never.self) {
                try FXAICommandSecurityPolicy.approvedCommandForTerminalHandoff(command, projectRoot: root)
            }
        }
    }

    @Test
    func approvesQuotedUserValuesWithoutTreatingThemAsShellSyntax() {
        let command = RunBuilderCommandFactory.offlineWorkflow(
            projectRoot: root,
            draft: OfflineLabDraft(
                workflowPreset: .smoke,
                profileName: "profile'; rm -rf ~ #",
                symbol: "EURUSD; shutdown now",
                runtimeMode: "research"
            )
        )

        #expect(FXAICommandSecurityPolicy.assess(command: command, projectRoot: root).approved)
    }

    @Test
    func approvesTerminalFallbackCatalogCommandsThatRemainShellSafe() {
        let snapshot = FXAICommandCatalog.snapshot(projectRoot: root)
        let terminalCommands = snapshot.commands
            .filter { $0.executionPath == .terminalFallback }
            .map(\.terminalEquivalent)

        #expect(!terminalCommands.isEmpty)
        for command in terminalCommands {
            #expect(FXAICommandSecurityPolicy.assess(command: command, projectRoot: root).approved)
        }
    }

    @Test
    func rejectsUnsupportedShellTextAndInjectionOperators() {
        let rejectedCommands = [
            "cd \(FXAICommandSecurityPolicy.shellQuoted(root.path))\nrm -rf ~",
            "cd \(FXAICommandSecurityPolicy.shellQuoted(root.path)); rm -rf ~",
            "cd \(FXAICommandSecurityPolicy.shellQuoted(root.path))\npython3 -c 'print(1)'",
            "cd \(FXAICommandSecurityPolicy.shellQuoted(root.path))\npython3 Tools/fxai_offline_lab.py dashboard --profile continuous && rm -rf ~",
            "cd \(FXAICommandSecurityPolicy.shellQuoted(root.path))\npython3 Tools/fxai_offline_lab.py dashboard --profile continuous # hidden trailing text",
            "cd \(FXAICommandSecurityPolicy.shellQuoted(root.path))\npython3 Tools/fxai_offline_lab.py dashboard --profile ~/fxai",
            "cd \"$(touch /tmp/fxai-policy-pwn)\""
        ]

        for command in rejectedCommands {
            #expect(!FXAICommandSecurityPolicy.assess(command: command, projectRoot: root).approved)
            #expect(throws: FXAICommandSecurityError.self) {
                try FXAICommandSecurityPolicy.approvedCommandForTerminalHandoff(command, projectRoot: root)
            }
        }
    }

    @Test
    func rejectsDirectoryChangesOutsideTheProjectRoot() {
        let command = [
            "cd \(FXAICommandSecurityPolicy.shellQuoted("/tmp"))",
            "python3 Tools/fxai_offline_lab.py dashboard --profile continuous"
        ].joined(separator: "\n")

        let assessment = FXAICommandSecurityPolicy.assess(command: command, projectRoot: root)

        #expect(!assessment.approved)
        #expect(assessment.reason?.contains("outside the FXAI project root") == true)
    }
}
