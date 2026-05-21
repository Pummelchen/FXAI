import Foundation
import Testing
@testable import FXAIGUICore

struct RuntimeArtifactReaderTests {
    @Test
    func readsDeploymentAndChampionArtifacts() throws {
        let root = try makeRuntimeFixture()
        let reader = RuntimeArtifactReader()

        let snapshot = reader.read(projectRoot: root)

        #expect(snapshot.deployments.count == 1)
        #expect(snapshot.champions.count == 1)

        let detail = try #require(snapshot.deployments.first)
        #expect(detail.symbol == "EURUSD")
        #expect(detail.pluginName == "ai_mlp")
        #expect(detail.routerSections.first?.values.contains(where: { $0.key == "plugin_weights_csv" }) == true)
        #expect(detail.supervisorSections.first?.values.contains(where: { $0.key == "budget_multiplier" }) == true)
    }

    private func makeRuntimeFixture() throws -> URL {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)

        let profileRoot = root.appendingPathComponent("Tools/OfflineLab/ResearchOS/test-runtime", isDirectory: true)
        let fileCommonRoot = root.appendingPathComponent("Promotions", isDirectory: true)

        try FileManager.default.createDirectory(at: profileRoot, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: fileCommonRoot, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: root.appendingPathComponent("Plugins", isDirectory: true), withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: root.appendingPathComponent("Tools", isDirectory: true), withIntermediateDirectories: true)
        try Data().write(to: root.appendingPathComponent("FXAI.mq5"))

        try Data(
            """
            {
              "champions": [
                {
                  "plugin_name": "ai_mlp",
                  "symbol": "EURUSD",
                  "reviewed_at": 1775259960,
                  "champion_score": 83.5,
                  "challenger_score": 81.2,
                  "status": "champion",
                  "portfolio_score": 0.72
                }
              ],
              "deployments": [
                {
                  "artifact_path": "\(fileCommonRoot.appendingPathComponent("fxai_live_deploy_EURUSD.tsv").path)",
                  "created_at": 1775259960,
                  "artifact_health": {
                    "artifact_exists": true,
                    "stale_artifact": false,
                    "missing_deployment": false,
                    "missing_router": false,
                    "missing_supervisor_service": false,
                    "missing_supervisor_command": false,
                    "missing_world_plan": false,
                    "artifact_age_sec": 0,
                    "performance_failures": [],
                    "artifact_size_failures": []
                  },
                  "live_state": {
                    "deployment_tsv": {
                      "profile_name": "test-runtime",
                      "symbol": "EURUSD",
                      "promotion_tier": "audit-approved",
                      "runtime_mode": "research"
                    }
                  },
                  "payload": {
                    "champions": [
                      { "plugin_name": "ai_mlp" }
                    ]
                  }
                }
              ]
            }
            """.utf8
        ).write(to: profileRoot.appendingPathComponent("operator_dashboard.json"))

        try Data(
            """
            [
              {
                "symbol": "EURUSD",
                "plugin_name": "ai_mlp",
                "status": "champion",
                "promotion_tier": "audit-approved",
                "champion_score": 83.5,
                "challenger_score": 81.2,
                "portfolio_score": 0.72,
                "reviewed_at": 1775259960,
                "champion_set_path": "\(root.appendingPathComponent("ai_mlp.set").path)",
                "profile_name": "test-runtime"
              }
            ]
            """.utf8
        ).write(to: profileRoot.appendingPathComponent("champions.json"))

        try Data(
            """
            profile_name\ttest-runtime
            symbol\tEURUSD
            promotion_tier\taudit-approved
            runtime_mode\tresearch
            """.utf8
        ).write(to: fileCommonRoot.appendingPathComponent("fxai_live_deploy_EURUSD.tsv"))

        try Data(
            """
            profile_name\ttest-runtime
            symbol\tEURUSD
            plugin_weights_csv\tai_mlp=1.20
            """.utf8
        ).write(to: fileCommonRoot.appendingPathComponent("fxai_student_router_EURUSD.tsv"))

        try Data(
            """
            profile_name\ttest-runtime
            symbol\tEURUSD
            budget_multiplier\t1.10
            """.utf8
        ).write(to: fileCommonRoot.appendingPathComponent("fxai_supervisor_service_EURUSD.tsv"))

        try Data(
            """
            profile_name\ttest-runtime
            symbol\tEURUSD
            block_score\t1.05
            """.utf8
        ).write(to: fileCommonRoot.appendingPathComponent("fxai_supervisor_command_EURUSD.tsv"))

        try Data(
            """
            profile_name\ttest-runtime
            symbol\tEURUSD
            sigma_scale\t0.80
            """.utf8
        ).write(to: fileCommonRoot.appendingPathComponent("fxai_world_plan_EURUSD.tsv"))

        try Data(
            """
            {
              "plugin_weights": { "ai_mlp": 1.2 },
              "family_weights": { "recurrent": 1.3 },
              "shadow_summary": { "mean_route_value": 0.26 },
              "pruned_plugins": []
            }
            """.utf8
        ).write(to: profileRoot.appendingPathComponent("student_router_EURUSD.json"))

        return root
    }
}
