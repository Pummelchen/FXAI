import Foundation

public enum RuntimeArtifactPathResolver {
    public static func runtimeDirectory(projectRoot: URL) -> URL? {
        let configuration = FXAIProjectConfigurationResolver.load(projectRoot: projectRoot)
        let environment = configuration.environment
        if let explicit = nonEmpty(environment["FXAI_RUNTIME_DIR"]) {
            let candidate = FXAIProjectConfigurationResolver.resolvedPathURL(
                rawValue: explicit,
                baseDirectory: projectRoot,
                environment: environment
            ) ?? URL(fileURLWithPath: explicit, isDirectory: true)
            if fileExists(candidate) {
                return candidate
            }
        }
        if let explicitCommonFiles = nonEmpty(environment["FXAI_COMMON_FILES"]) {
            let candidate = (FXAIProjectConfigurationResolver.resolvedPathURL(
                rawValue: explicitCommonFiles,
                baseDirectory: projectRoot,
                environment: environment
            ) ?? URL(fileURLWithPath: explicitCommonFiles, isDirectory: true))
                .appendingPathComponent("FXAI", isDirectory: true)
                .appendingPathComponent("Runtime", isDirectory: true)
            if fileExists(candidate) {
                return candidate
            }
        }

        if let configured = runtimeDirectoryFromConfig(
            projectRoot: projectRoot,
            configuration: configuration
        ) {
            return configured
        }

        let fixtureRuntime = projectRoot
            .appendingPathComponent("FILE_COMMON", isDirectory: true)
            .appendingPathComponent("FXAI", isDirectory: true)
            .appendingPathComponent("Runtime", isDirectory: true)
        if fileExists(fixtureRuntime) {
            return fixtureRuntime
        }

        guard let driveRoot = driveCRoot(projectRoot: projectRoot) else {
            return nil
        }

        let usersRoot = driveRoot.appendingPathComponent("users", isDirectory: true)
        let preferredUser = FileManager.default.homeDirectoryForCurrentUser.lastPathComponent
        if let candidate = runtimeDirectory(usersRoot: usersRoot, username: preferredUser) {
            return candidate
        }

        let userDirectories = (try? FileManager.default.contentsOfDirectory(at: usersRoot, includingPropertiesForKeys: nil)) ?? []
        for directory in userDirectories where directory.hasDirectoryPath {
            if let candidate = runtimeDirectory(usersRoot: usersRoot, username: directory.lastPathComponent) {
                return candidate
            }
        }
        return nil
    }

    private static func runtimeDirectoryFromConfig(
        projectRoot: URL,
        configuration: FXAIProjectConfigurationSnapshot
    ) -> URL? {
        if let explicitRuntime = FXAIProjectConfigurationResolver.configuredValue(
            configuration: configuration,
            key: "runtime_dir"
        ) {
            let candidate = FXAIProjectConfigurationResolver.resolvedPathURL(
                rawValue: explicitRuntime,
                baseDirectory: projectRoot,
                environment: configuration.environment
            ) ?? URL(fileURLWithPath: explicitRuntime, isDirectory: true)
            if fileExists(candidate) {
                return candidate
            }
        }
        if let commonFiles = FXAIProjectConfigurationResolver.configuredValue(
            configuration: configuration,
            key: "common_files"
        ) {
            let candidate = (FXAIProjectConfigurationResolver.resolvedPathURL(
                rawValue: commonFiles,
                baseDirectory: projectRoot,
                environment: configuration.environment
            ) ?? URL(fileURLWithPath: commonFiles, isDirectory: true))
                .appendingPathComponent("FXAI", isDirectory: true)
                .appendingPathComponent("Runtime", isDirectory: true)
            if fileExists(candidate) {
                return candidate
            }
        }
        return nil
    }

    private static func fileExists(_ url: URL) -> Bool {
        FileManager.default.fileExists(atPath: url.path)
    }

    private static func driveCRoot(projectRoot: URL) -> URL? {
        let standardized = projectRoot.standardizedFileURL.path
        guard let range = standardized.range(of: "/drive_c/") else {
            return nil
        }
        let prefix = String(standardized[..<range.upperBound])
        return URL(fileURLWithPath: prefix, isDirectory: true)
    }

    private static func runtimeDirectory(usersRoot: URL, username: String) -> URL? {
        guard !username.isEmpty else { return nil }
        let candidate = usersRoot
            .appendingPathComponent(username, isDirectory: true)
            .appendingPathComponent("AppData", isDirectory: true)
            .appendingPathComponent("Roaming", isDirectory: true)
            .appendingPathComponent("MetaQuotes", isDirectory: true)
            .appendingPathComponent("Terminal", isDirectory: true)
            .appendingPathComponent("Common", isDirectory: true)
            .appendingPathComponent("Files", isDirectory: true)
            .appendingPathComponent("FXAI", isDirectory: true)
            .appendingPathComponent("Runtime", isDirectory: true)
        return fileExists(candidate) ? candidate : nil
    }

    private static func nonEmpty(_ value: String?) -> String? {
        guard let value, !value.isEmpty else { return nil }
        return value
    }
}
