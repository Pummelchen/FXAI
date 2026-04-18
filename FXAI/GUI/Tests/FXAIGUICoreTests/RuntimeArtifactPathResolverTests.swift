import Foundation
import Testing
@testable import FXAIGUICore

struct RuntimeArtifactPathResolverTests {
    @Test
    func prefersRuntimeDirectoryFromFXAIToml() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        let runtime = root
            .appendingPathComponent("CustomCommon", isDirectory: true)
            .appendingPathComponent("FXAI", isDirectory: true)
            .appendingPathComponent("Runtime", isDirectory: true)

        try FileManager.default.createDirectory(at: runtime, withIntermediateDirectories: true)
        try Data(
            """
            [paths]
            common_files = "CustomCommon"
            """.utf8
        ).write(to: root.appendingPathComponent("fxai.toml"))

        let resolved = RuntimeArtifactPathResolver.runtimeDirectory(projectRoot: root)

        #expect(resolved == runtime)
    }

    @Test
    func prefersProfileSpecificRuntimeDirectoryFromFXAIToml() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        let sharedRuntime = root
            .appendingPathComponent("SharedCommon", isDirectory: true)
            .appendingPathComponent("FXAI", isDirectory: true)
            .appendingPathComponent("Runtime", isDirectory: true)
        let profileRuntime = root
            .appendingPathComponent("ProfileCommon", isDirectory: true)
            .appendingPathComponent("FXAI", isDirectory: true)
            .appendingPathComponent("Runtime", isDirectory: true)

        try FileManager.default.createDirectory(at: sharedRuntime, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: profileRuntime, withIntermediateDirectories: true)
        try Data(
            """
            [toolchain]
            profile = "headless_ci"

            [paths]
            common_files = "SharedCommon"

            [profiles.headless_ci.paths]
            common_files = "ProfileCommon"
            """.utf8
        ).write(to: root.appendingPathComponent("fxai.toml"))

        let resolved = RuntimeArtifactPathResolver.runtimeDirectory(projectRoot: root)

        #expect(resolved == profileRuntime)
    }
}
