import Foundation
import Testing
@testable import FXAIGUICore

struct RunBuilderCommandFactoryTests {
    @Test
    func buildsAuditCommandWithPluginAndScenarioFlags() {
        let root = URL(fileURLWithPath: "/tmp/fxai", isDirectory: true)
        let draft = AuditLabDraft(
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

        let command = RunBuilderCommandFactory.auditCommand(projectRoot: root, draft: draft)

        #expect(command.contains("python3 Tools/fxai_testlab.py run-audit"))
        #expect(command.contains("--plugin-list '{ai_mlp}'"))
        #expect(command.contains("--scenario-list '{market_recent,market_walkforward,market_macro_event,market_adversarial}'"))
        #expect(command.contains("--execution-profile prime-ecn"))
        #expect(command.contains("--symbol 'EURUSD'"))
    }

    @Test
    func buildsBacktestWorkflowWithCompileAndBaselineSteps() {
        let root = URL(fileURLWithPath: "/tmp/fxai", isDirectory: true)
        let draft = BacktestBuilderDraft(
            pluginName: "tree_lgbm",
            symbol: "EURUSD",
            scenarioPreset: .adversarial,
            executionProfile: .stress,
            baselineName: "eurusd_tree_lgbm",
            bars: 1800,
            sequenceBars: 64
        )

        let command = RunBuilderCommandFactory.backtestWorkflow(projectRoot: root, draft: draft)

        #expect(command.contains("python3 Tools/fxai_testlab.py compile-main"))
        #expect(command.contains("python3 Tools/fxai_testlab.py compile-audit"))
        #expect(command.contains("--plugin-list '{tree_lgbm}'"))
        #expect(command.contains("--execution-profile stress"))
        #expect(command.contains("baseline-save --name 'eurusd_tree_lgbm'"))
    }

    @Test
    func buildsOfflineWorkflowWithPromotionAndDeploySteps() {
        let root = URL(fileURLWithPath: "/tmp/fxai", isDirectory: true)
        let draft = OfflineLabDraft(
            workflowPreset: .promotion,
            profileName: "bestparams",
            symbol: "EURUSD",
            symbolPack: .majors,
            monthsList: "3,6,12",
            autoExport: true,
            includeBootstrap: false,
            includeBestParams: true,
            includeDeployProfiles: true,
            includeLineage: true,
            includeMinimalBundle: true,
            runtimeMode: "production",
            topPlugins: 8,
            limitExperiments: 24,
            limitRuns: 64
        )

        let command = RunBuilderCommandFactory.offlineWorkflow(projectRoot: root, draft: draft)

        #expect(command.contains("python3 Tools/fxai_offline_lab.py tune-zoo --profile 'bestparams' --auto-export --symbol-pack majors"))
        #expect(command.contains("python3 Tools/fxai_offline_lab.py best-params --profile 'bestparams' --symbol-pack majors"))
        #expect(command.contains("python3 Tools/fxai_offline_lab.py deploy-profiles --profile 'bestparams' --runtime-mode 'production'"))
        #expect(command.contains("python3 Tools/fxai_offline_lab.py lineage-report --profile 'bestparams' --symbol 'EURUSD'"))
        #expect(command.contains("python3 Tools/fxai_offline_lab.py minimal-bundle --profile 'bestparams'"))
    }
}
