import Foundation

struct FXAIProjectConfigurationSnapshot {
    let environment: [String: String]
    let parser: SimpleTOMLParser?
    let profile: String
}

enum FXAIProjectConfigurationResolver {
    static func load(
        projectRoot: URL,
        processEnvironment: [String: String] = ProcessInfo.processInfo.environment
    ) -> FXAIProjectConfigurationSnapshot {
        let envURL = resolvedPathURL(
            rawValue: processEnvironment["FXAI_ENV_FILE"],
            baseDirectory: projectRoot,
            environment: processEnvironment
        ) ?? projectRoot.appendingPathComponent(".env", isDirectory: false)

        let dotenv = parseDotEnv(at: envURL)
        var mergedEnvironment = dotenv
        for (key, value) in processEnvironment {
            mergedEnvironment[key] = value
        }

        let configURL = resolvedPathURL(
            rawValue: mergedEnvironment["FXAI_CONFIG"],
            baseDirectory: projectRoot,
            environment: mergedEnvironment
        ) ?? projectRoot.appendingPathComponent("fxai.toml", isDirectory: false)

        let parser = parser(at: configURL)
        let profile = resolveProfile(
            parser: parser,
            projectRoot: projectRoot,
            environment: mergedEnvironment
        )
        return FXAIProjectConfigurationSnapshot(
            environment: mergedEnvironment,
            parser: parser,
            profile: profile
        )
    }

    static func configuredValue(
        configuration: FXAIProjectConfigurationSnapshot,
        key: String
    ) -> String? {
        guard let parser = configuration.parser else {
            return nil
        }
        if let profileValue = parser.value(in: ["profiles", configuration.profile, "paths"], key: key) {
            return profileValue
        }
        return parser.value(in: ["paths"], key: key)
    }

    static func resolvedPathURL(
        rawValue: String?,
        baseDirectory: URL,
        environment: [String: String]
    ) -> URL? {
        guard let rawValue, !rawValue.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            return nil
        }
        let expanded = expandPathVariables(in: rawValue, environment: environment)
        let tildeExpanded = NSString(string: expanded).expandingTildeInPath
        if tildeExpanded.hasPrefix("/") {
            return URL(fileURLWithPath: tildeExpanded, isDirectory: true).standardizedFileURL
        }
        return baseDirectory
            .appendingPathComponent(tildeExpanded, isDirectory: true)
            .standardizedFileURL
    }

    private static func parser(at url: URL) -> SimpleTOMLParser? {
        guard
            let data = try? Data(contentsOf: url),
            let text = String(data: data, encoding: .utf8)
        else {
            return nil
        }
        return SimpleTOMLParser(text: text)
    }

    private static func parseDotEnv(at url: URL) -> [String: String] {
        guard
            let data = try? Data(contentsOf: url),
            let text = String(data: data, encoding: .utf8)
        else {
            return [:]
        }

        var payload: [String: String] = [:]
        for rawLine in text.split(whereSeparator: \.isNewline) {
            let line = rawLine.trimmingCharacters(in: .whitespaces)
            guard !line.isEmpty, !line.hasPrefix("#"), let separator = line.firstIndex(of: "=") else {
                continue
            }
            let key = String(line[..<separator]).trimmingCharacters(in: .whitespaces)
            guard !key.isEmpty else {
                continue
            }
            let value = String(line[line.index(after: separator)...])
                .trimmingCharacters(in: .whitespaces)
                .trimmingCharacters(in: CharacterSet(charactersIn: "\"'"))
            payload[key] = value
        }
        return payload
    }

    private static func resolveProfile(
        parser: SimpleTOMLParser?,
        projectRoot: URL,
        environment: [String: String]
    ) -> String {
        if let explicit = nonEmpty(environment["FXAI_TOOLCHAIN_PROFILE"])?.lowercased(), explicit != "auto" {
            return explicit
        }
        if let configured = parser?.value(in: ["toolchain"], key: "profile")?.lowercased(), configured != "auto" {
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

    private static func expandPathVariables(in rawValue: String, environment: [String: String]) -> String {
        let pattern = #"\$\{([A-Za-z_][A-Za-z0-9_]*)\}|\$([A-Za-z_][A-Za-z0-9_]*)"#
        guard let regex = try? NSRegularExpression(pattern: pattern) else {
            return rawValue
        }

        let nsRawValue = rawValue as NSString
        let matches = regex.matches(
            in: rawValue,
            options: [],
            range: NSRange(location: 0, length: nsRawValue.length)
        )

        var result = rawValue
        for match in matches.reversed() {
            let bracedRange = match.range(at: 1)
            let simpleRange = match.range(at: 2)
            let tokenRange = bracedRange.location != NSNotFound ? bracedRange : simpleRange
            guard tokenRange.location != NSNotFound else {
                continue
            }
            let token = nsRawValue.substring(with: tokenRange)
            let replacement = environment[token] ?? ""
            guard let range = Range(match.range, in: result) else {
                continue
            }
            result.replaceSubrange(range, with: replacement)
        }
        return result
    }

    private static func nonEmpty(_ value: String?) -> String? {
        guard let value, !value.isEmpty else { return nil }
        return value
    }
}

struct SimpleTOMLParser {
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
