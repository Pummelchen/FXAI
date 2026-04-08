import Foundation
import Testing
@testable import FXAIGUICore

struct ProjectScannerTests {
    @Test
    func resolvesProjectRootFromGUISubdirectory() throws {
        let tempRoot = try makeProjectFixture()
        let guiPath = tempRoot.appendingPathComponent("GUI", isDirectory: true)
        try FileManager.default.createDirectory(at: guiPath, withIntermediateDirectories: true)

        #expect(ProjectPathResolver.resolveProjectRoot(from: guiPath) == tempRoot)
    }

    @Test
    func scansPluginAndArtifactSurface() throws {
        let tempRoot = try makeProjectFixture()
        let scanner = ProjectScanner()

        let snapshot = try scanner.scan(projectRoot: tempRoot)

        #expect(snapshot.totalPluginCount == 2)
        #expect(snapshot.cleanBuildTargetCount == 2)
        #expect(snapshot.reportCategories.contains(where: { $0.category == "ResearchOS" && $0.fileCount == 2 }))
        #expect(snapshot.runtimeProfiles.count == 1)
        #expect(snapshot.operatorSummary.championCount == 1)
        #expect(snapshot.plugins.contains(where: { $0.name == "ai_mlp" && $0.sourceKind == .file }))
        #expect(snapshot.pluginFamilies.contains(where: { $0.family == "Sequence" && $0.pluginCount == 1 }))
    }

    private func makeProjectFixture() throws -> URL {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)

        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: root.appendingPathComponent("Plugins/Linear", isDirectory: true), withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: root.appendingPathComponent("Plugins/Sequence/ai_mlp", isDirectory: true), withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: root.appendingPathComponent("Tools/Baselines", isDirectory: true), withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: root.appendingPathComponent("Tools/OfflineLab/ResearchOS/test-profile", isDirectory: true), withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: root.appendingPathComponent("Tests", isDirectory: true), withIntermediateDirectories: true)

        try Data().write(to: root.appendingPathComponent("FXAI.mq5"))
        try Data().write(to: root.appendingPathComponent("Plugins/Linear/lin_sgd.mqh"))
        try Data().write(to: root.appendingPathComponent("Plugins/Sequence/ai_mlp.mqh"))
        try Data().write(to: root.appendingPathComponent("FXAI.ex5"))
        try Data().write(to: root.appendingPathComponent("Tests/FXAI_AuditRunner.ex5"))
        try Data("{\"plugins\":{}}".utf8).write(to: root.appendingPathComponent("Tools/Baselines/example.summary.json"))
        try Data(
            """
            {
              "symbol": "EURUSD",
              "profile_name": "test-profile",
              "promotion_tier": "audit-approved",
              "runtime_mode": "research",
              "champions": [
                { "plugin_name": "ai_mlp" }
              ]
            }
            """.utf8
        ).write(to: root.appendingPathComponent("Tools/OfflineLab/ResearchOS/test-profile/live_deploy_EURUSD.json"))
        try Data(
            """
            {
              "champions": [
                { "reviewed_at": 1775259960 }
              ],
              "deployments": [
                { "symbol": "EURUSD" }
              ]
            }
            """.utf8
        ).write(to: root.appendingPathComponent("Tools/OfflineLab/ResearchOS/test-profile/operator_dashboard.json"))

        return root
    }
}
