import Foundation

public enum RuntimeArtifactPathResolver {
    public static func runtimeDirectory(projectRoot: URL) -> URL? {
        let environment = ProcessInfo.processInfo.environment
        if let explicit = environment["FXAI_RUNTIME_DIR"], !explicit.isEmpty {
            let candidate = URL(fileURLWithPath: explicit, isDirectory: true)
            if fileExists(candidate) {
                return candidate
            }
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
}
