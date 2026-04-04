import Foundation
import Testing
@testable import FXAIGUICore

struct ResearchOSPhase4Tests {
    @Test
    func readsResearchOSDashboardState() throws {
        let root = try makeResearchOSFixture()
        let reader = ResearchOSArtifactReader()

        let snapshot = reader.read(projectRoot: root)

        #expect(snapshot.profileName == "continuous")
        #expect(snapshot.environment?.backend == "turso")
        #expect(snapshot.branches.count == 2)
        #expect(snapshot.auditEvents.count == 1)
        #expect(snapshot.symbols.count == 1)

        let symbol = try #require(snapshot.symbols.first)
        #expect(symbol.symbol == "EURUSD")
        #expect(symbol.analogNeighbors.count == 1)
        #expect(symbol.analogNeighbors.first?.pluginName == "ai_mlp")
        #expect(snapshot.sourceOfTruth.contains(where: { $0.key == "turso_libsql" }))
    }

    @Test
    func buildsBranchAndRecoveryCommands() {
        let root = URL(fileURLWithPath: "/tmp/fxai", isDirectory: true)
        let branchDraft = ResearchOSBranchDraft(
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

        let branchCommand = ResearchOSCommandFactory.branchCommand(projectRoot: root, draft: branchDraft)
        #expect(branchCommand.contains("python3 Tools/fxai_offline_lab.py turso-pitr-restore"))
        #expect(branchCommand.contains("--source-database 'fxai-prod'"))
        #expect(branchCommand.contains("--target-database 'fxai-restore-1'"))
        #expect(branchCommand.contains("--timestamp '2026-04-05T10:11:12Z'"))
        #expect(branchCommand.contains("--read-only-token"))

        let recoveryDraft = ResearchOSRecoveryDraft(profileName: "continuous", runtimeMode: "production")
        let recoveryCommand = ResearchOSCommandFactory.recoveryCommand(projectRoot: root, draft: recoveryDraft)
        #expect(recoveryCommand.contains("python3 Tools/fxai_offline_lab.py recover-artifacts --profile 'continuous' --runtime-mode 'production'"))
        #expect(recoveryCommand.contains("python3 Tools/fxai_offline_lab.py minimal-bundle --profile 'continuous'"))
    }

    private func makeResearchOSFixture() throws -> URL {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)

        let profileRoot = root.appendingPathComponent("Tools/OfflineLab/ResearchOS/continuous", isDirectory: true)
        try FileManager.default.createDirectory(at: profileRoot, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: root.appendingPathComponent("Plugins", isDirectory: true), withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: root.appendingPathComponent("Tools", isDirectory: true), withIntermediateDirectories: true)
        try Data().write(to: root.appendingPathComponent("FXAI.mq5"))

        let payload = """
        {
          "profile_name": "continuous",
          "turso": {
            "environment": {
              "backend": "turso",
              "sync_mode": "embedded-replica",
              "database_path": "/tmp/fxai_offline_lab.turso.db",
              "database_name": "fxai-prod",
              "organization_slug": "openai",
              "group_name": "trading",
              "location_name": "fra",
              "cli_config_path": "/tmp/turso.toml",
              "sync_interval_seconds": 30,
              "encryption_enabled": true,
              "platform_api_enabled": true,
              "sync_enabled": true,
              "auth_token_configured": true,
              "api_token_configured": true,
              "config_error": ""
            },
            "branches": [
              {
                "source_database": "fxai-prod",
                "target_database": "fxai-campaign-1",
                "branch_kind": "campaign",
                "group_name": "trading",
                "location_name": "fra",
                "env_artifact_path": "/tmp/branch_fxai_campaign_1.env",
                "status": "active",
                "created_at": 1775259960
              },
              {
                "name": "fxai-pitr-1",
                "parent_name": "fxai-prod",
                "is_branch": true,
                "group": "trading",
                "hostname": "example.turso.io"
              }
            ],
            "recent_audit_logs": [
              {
                "organization_slug": "openai",
                "event_id": "evt_1",
                "event_type": "branch.created",
                "target_name": "fxai-campaign-1",
                "occurred_at": 1775259960,
                "observed_at": 1775259970
              }
            ]
          },
          "deployments": [
            {
              "symbol": "EURUSD",
              "artifact_path": "/tmp/fxai_live_deploy_EURUSD.tsv",
              "created_at": 1775259980,
              "analog_neighbors": [
                {
                  "source_key": "plugin:ai_mlp",
                  "plugin_name": "ai_mlp",
                  "cosine_distance": 0.09,
                  "similarity": 0.91,
                  "score": 0.84,
                  "source_type": "shadow",
                  "vector_scope": "analog_shadow",
                  "payload": {
                    "session": "london",
                    "regime": "trend"
                  }
                }
              ]
            }
          ],
          "source_of_truth": {
            "turso_libsql": "authoritative research and promotion state",
            "file_common_promotions": "authoritative MT5 runtime consumption layer"
          }
        }
        """

        try Data(payload.utf8).write(to: profileRoot.appendingPathComponent("operator_dashboard.json"))
        return root
    }
}
