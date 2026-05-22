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

    @Test
    func prefersRuntimeDirectoryFromDotEnvOverrides() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        let runtime = root
            .appendingPathComponent("EnvCommon", isDirectory: true)
            .appendingPathComponent("FXAI", isDirectory: true)
            .appendingPathComponent("Runtime", isDirectory: true)

        try FileManager.default.createDirectory(at: runtime, withIntermediateDirectories: true)
        try Data(
            """
            FXAI_COMMON_FILES=EnvCommon
            """.utf8
        ).write(to: root.appendingPathComponent(".env"))

        let resolved = RuntimeArtifactPathResolver.runtimeDirectory(projectRoot: root)

        #expect(resolved == runtime)
    }

    @Test
    func prefersConfigPathDefinedInDotEnv() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        let runtime = root
            .appendingPathComponent("EnvConfigCommon", isDirectory: true)
            .appendingPathComponent("FXAI", isDirectory: true)
            .appendingPathComponent("Runtime", isDirectory: true)
        let configDirectory = root.appendingPathComponent("Config", isDirectory: true)

        try FileManager.default.createDirectory(at: runtime, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: configDirectory, withIntermediateDirectories: true)
        try Data(
            """
            [paths]
            common_files = "EnvConfigCommon"
            """.utf8
        ).write(to: configDirectory.appendingPathComponent("fxai-toolchain.toml"))
        try Data(
            """
            FXAI_CONFIG=Config/fxai-toolchain.toml
            """.utf8
        ).write(to: root.appendingPathComponent(".env"))

        let resolved = RuntimeArtifactPathResolver.runtimeDirectory(projectRoot: root)

        #expect(resolved == runtime)
    }
}
