import Foundation
import Testing
@testable import FXGUICore

struct FXAIProjectConfigurationResolverTests {
    @Test
    func processEnvironmentProfileOverridesTomlProfileAndTrimsWhitespace() throws {
        let root = try temporaryProjectRoot()
        try Data(
            """
            [toolchain]
            profile = "macos_wine"
            """.utf8
        ).write(to: root.appendingPathComponent("fxai.toml"))

        let configuration = FXAIProjectConfigurationResolver.load(
            projectRoot: root,
            processEnvironment: [
                "FXAI_TOOLCHAIN_PROFILE": " headless_ci "
            ]
        )

        #expect(configuration.profile == "headless_ci")
    }

    @Test
    func processEnvironmentOverridesDotEnvProfile() throws {
        let root = try temporaryProjectRoot()
        try Data(
            """
            FXAI_TOOLCHAIN_PROFILE=macos_wine
            """.utf8
        ).write(to: root.appendingPathComponent(".env"))

        let configuration = FXAIProjectConfigurationResolver.load(
            projectRoot: root,
            processEnvironment: [
                "FXAI_TOOLCHAIN_PROFILE": "headless_ci"
            ]
        )

        #expect(configuration.profile == "headless_ci")
    }

    @Test
    func profilePathsOverrideGlobalPaths() throws {
        let root = try temporaryProjectRoot()
        try Data(
            """
            [toolchain]
            profile = "headless_ci"

            [paths]
            runtime_dir = "GlobalRuntime"
            default_db = "Global.db"

            [profiles.headless_ci.paths]
            runtime_dir = "ProfileRuntime"
            """.utf8
        ).write(to: root.appendingPathComponent("fxai.toml"))

        let configuration = FXAIProjectConfigurationResolver.load(
            projectRoot: root,
            processEnvironment: [:]
        )

        #expect(FXAIProjectConfigurationResolver.configuredValue(configuration: configuration, key: "runtime_dir") == "ProfileRuntime")
        #expect(FXAIProjectConfigurationResolver.configuredValue(configuration: configuration, key: "default_db") == "Global.db")
    }

    @Test
    func resolvedPathURLTrimsRelativePathValues() throws {
        let root = try temporaryProjectRoot()

        let resolved = FXAIProjectConfigurationResolver.resolvedPathURL(
            rawValue: "  Config/Runtime  ",
            baseDirectory: root,
            environment: [:]
        )

        #expect(resolved == root.appendingPathComponent("Config", isDirectory: true).appendingPathComponent("Runtime", isDirectory: true).standardizedFileURL)
    }

    private func temporaryProjectRoot() throws -> URL {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        return root
    }
}
