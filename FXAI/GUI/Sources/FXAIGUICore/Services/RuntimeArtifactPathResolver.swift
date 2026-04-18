import Foundation

public enum RuntimeArtifactPathResolver {
    public static func runtimeDirectory(projectRoot: URL) -> URL? {
        let environment = ProcessInfo.processInfo.environment
        if let explicit = nonEmpty(environment["FXAI_RUNTIME_DIR"]) {
            let candidate = URL(fileURLWithPath: explicit, isDirectory: true)
            if fileExists(candidate) {
                return candidate
            }
        }
        if let explicitCommonFiles = nonEmpty(environment["FXAI_COMMON_FILES"]) {
            let candidate = resolveConfiguredPath(explicitCommonFiles, baseDirectory: projectRoot)
                .appendingPathComponent("FXAI", isDirectory: true)
                .appendingPathComponent("Runtime", isDirectory: true)
            if fileExists(candidate) {
                return candidate
            }
        }

        if let configured = runtimeDirectoryFromConfig(projectRoot: projectRoot, environment: environment) {
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

    private static func runtimeDirectoryFromConfig(projectRoot: URL, environment: [String: String]) -> URL? {
        let configURL = projectRoot.appendingPathComponent("fxai.toml")
        guard
            let data = try? Data(contentsOf: configURL),
            let text = String(data: data, encoding: .utf8)
        else {
            return nil
        }

        let parser = SimpleTOMLParser(text: text)
        let profile = resolveProfile(parser: parser, projectRoot: projectRoot, environment: environment)
        if let explicitRuntime = configuredValue(parser: parser, profile: profile, key: "runtime_dir") {
            let candidate = resolveConfiguredPath(explicitRuntime, baseDirectory: projectRoot)
            if fileExists(candidate) {
                return candidate
            }
        }
        if let commonFiles = configuredValue(parser: parser, profile: profile, key: "common_files") {
            let candidate = resolveConfiguredPath(commonFiles, baseDirectory: projectRoot)
                .appendingPathComponent("FXAI", isDirectory: true)
                .appendingPathComponent("Runtime", isDirectory: true)
            if fileExists(candidate) {
                return candidate
            }
        }
        return nil
    }

    private static func resolveConfiguredPath(_ rawValue: String, baseDirectory: URL) -> URL {
        if rawValue.hasPrefix("~") {
            let home = FileManager.default.homeDirectoryForCurrentUser.path
            let expanded = home + rawValue.dropFirst()
            return URL(fileURLWithPath: expanded, isDirectory: true).standardizedFileURL
        }
        if rawValue.hasPrefix("/") {
            return URL(fileURLWithPath: rawValue, isDirectory: true)
        }
        return baseDirectory
            .appendingPathComponent(rawValue, isDirectory: true)
            .standardizedFileURL
    }

    private static func configuredValue(parser: SimpleTOMLParser, profile: String, key: String) -> String? {
        if let profileValue = parser.value(in: ["profiles", profile, "paths"], key: key) {
            return profileValue
        }
        return parser.value(in: ["paths"], key: key)
    }

    private static func resolveProfile(parser: SimpleTOMLParser, projectRoot: URL, environment: [String: String]) -> String {
        if let explicit = nonEmpty(environment["FXAI_TOOLCHAIN_PROFILE"])?.lowercased(), explicit != "auto" {
            return explicit
        }
        if let configured = parser.value(in: ["toolchain"], key: "profile")?.lowercased(), configured != "auto" {
            return configured
        }
        if nonEmpty(environment["CI"]) != nil || nonEmpty(environment["GITHUB_ACTIONS"]) != nil {
            return "headless_ci"
        }
        if projectRoot.standardizedFileURL.path.contains("/drive_c/") {
            return "macos_wine"
        }
        if nonEmpty(environment["APPDATA"]) != nil {
            return "windows_native"
        }
        return "headless_ci"
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

private struct SimpleTOMLParser {
    private let lines: [Substring]

    init(text: String) {
        self.lines = text.split(whereSeparator: \.isNewline)
    }

    func value(in sectionPath: [String], key: String) -> String? {
        var currentSection: [String] = []
        for rawLine in lines {
            let line = rawLine.trimmingCharacters(in: .whitespaces)
            if line.isEmpty || line.hasPrefix("#") {
                continue
            }
            if line.hasPrefix("[") && line.hasSuffix("]") {
                let name = line.dropFirst().dropLast()
                currentSection = name.split(separator: ".").map { String($0) }
                continue
            }
            guard currentSection == sectionPath else {
                continue
            }
            guard let separator = line.firstIndex(of: "=") else {
                continue
            }
            let candidateKey = line[..<separator].trimmingCharacters(in: .whitespaces)
            guard candidateKey == key else {
                continue
            }
            let rawValue = line[line.index(after: separator)...].trimmingCharacters(in: .whitespaces)
            return rawValue.trimmingCharacters(in: CharacterSet(charactersIn: "\""))
        }
        return nil
    }
}
