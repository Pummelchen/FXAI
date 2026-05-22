import Foundation
import Testing
@testable import FXAIGUICore

struct ProjectScannerTests {
    @Test
    func resolvesProjectRootFromFXDataEngineGUISubdirectory() throws {
        let tempRoot = try makeProjectFixture()
        let guiPath = tempRoot.appendingPathComponent("FXDataEngineGUI", isDirectory: true)
        try FileManager.default.createDirectory(at: guiPath, withIntermediateDirectories: true)

        #expect(ProjectPathResolver.resolveProjectRoot(from: guiPath) == tempRoot)
    }

    @Test
    func scansPluginAndArtifactSurface() throws {
        let tempRoot = try makeProjectFixture()
        let scanner = ProjectScanner()

        let snapshot = try scanner.scan(projectRoot: tempRoot)

        #expect(snapshot.totalPluginCount == 2)
        #expect(snapshot.cleanBuildTargetCount == 4)
        #expect(snapshot.reportCategories.contains(where: { $0.category == "ResearchOS" && $0.fileCount == 2 }))
        #expect(snapshot.runtimeProfiles.count == 1)
        #expect(snapshot.operatorSummary.championCount == 1)
        #expect(snapshot.plugins.contains(where: { $0.name == "ai_mlp" && $0.sourceKind == .file }))
        #expect(snapshot.pluginFamilies.contains(where: { $0.family == "Sequence" && $0.pluginCount == 1 }))
    }

    @Test
    func scanHonorsDotEnvDatabaseAndReplicaSettings() throws {
        let tempRoot = try makeProjectFixture()
        let configuredDB = tempRoot
            .appendingPathComponent("Data", isDirectory: true)
            .appendingPathComponent("custom-replica.db", isDirectory: false)
        try FileManager.default.createDirectory(
            at: configuredDB.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        try Data().write(to: configuredDB)
        try Data(
            """
            FXAI_DEFAULT_DB=Data/custom-replica.db
            TURSO_DATABASE_URL=libsql://fxai.example.turso.io
            TURSO_AUTH_TOKEN=test-token
            TURSO_ENCRYPTION_KEY=test-key
            """.utf8
        ).write(to: tempRoot.appendingPathComponent(".env"))

        let snapshot = try ProjectScanner().scan(projectRoot: tempRoot)

        #expect(snapshot.tursoSummary.localDatabasePresent)
        #expect(snapshot.tursoSummary.localDatabasePath == configuredDB)
        #expect(snapshot.tursoSummary.embeddedReplicaConfigured)
        #expect(snapshot.tursoSummary.encryptionConfigured)
    }

    private func makeProjectFixture() throws -> URL {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)

        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: root.appendingPathComponent("FXDataEngine", isDirectory: true), withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: root.appendingPathComponent("FXPlugins/Sources/FXAIPlugins/Linear", isDirectory: true), withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: root.appendingPathComponent("FXPlugins/Sources/FXAIPlugins/Sequence", isDirectory: true), withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: root.appendingPathComponent("FXBacktest", isDirectory: true), withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: root.appendingPathComponent("FXDatabase", isDirectory: true), withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: root.appendingPathComponent("Tools/Baselines", isDirectory: true), withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: root.appendingPathComponent("Tools/OfflineLab/ResearchOS/test-profile", isDirectory: true), withIntermediateDirectories: true)

        try Data("// swift-tools-version: 6.3\n".utf8).write(to: root.appendingPathComponent("FXDataEngine/Package.swift"))
        try Data("// swift-tools-version: 6.3\n".utf8).write(to: root.appendingPathComponent("FXPlugins/Package.swift"))
        try Data("// swift-tools-version: 6.3\n".utf8).write(to: root.appendingPathComponent("FXBacktest/Package.swift"))
        try Data("// swift-tools-version: 6.3\n".utf8).write(to: root.appendingPathComponent("FXDatabase/Package.swift"))
        try Data(
            """
            public let manifest = PluginManifestV4(
                aiID: AIModelID.sgdLogit.rawValue,
                aiName: "lin_sgd",
                family: .linear
            )
            """.utf8
        ).write(to: root.appendingPathComponent("FXPlugins/Sources/FXAIPlugins/Linear/LinearFixture.swift"))
        try Data(
            """
            public let manifest = PluginManifestV4(
                aiID: AIModelID.mlpTiny.rawValue,
                aiName: "ai_mlp",
                family: .transformer
            )
            """.utf8
        ).write(to: root.appendingPathComponent("FXPlugins/Sources/FXAIPlugins/Sequence/SequenceFixture.swift"))
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
