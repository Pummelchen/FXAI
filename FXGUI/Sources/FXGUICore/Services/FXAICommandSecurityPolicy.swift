import Foundation

public struct FXAICommandSecurityAssessment: Equatable, Sendable {
    public let approved: Bool
    public let reason: String?

    public static let approved = FXAICommandSecurityAssessment(approved: true, reason: nil)

    public static func rejected(_ reason: String) -> FXAICommandSecurityAssessment {
        FXAICommandSecurityAssessment(approved: false, reason: reason)
    }
}

public enum FXAICommandSecurityError: Error, CustomStringConvertible, Sendable {
    case rejected(String)

    public var description: String {
        switch self {
        case .rejected(let reason):
            return "Command handoff rejected: \(reason)"
        }
    }
}

public enum FXAICommandSecurityPolicy {
    private static let allowedSwiftPackages: Set<String> = [
        "FXBacktest",
        "FXBacktestAgent",
        "FXDataEngine",
        "FXDatabase",
        "FXGUI",
        "FXPlugins"
    ]
    private static let allowedSwiftTestFilters: Set<String> = [
        "FXBacktestPluginZooBridgeTests"
    ]
    private static let allowedPythonScripts: Set<String> = [
        "Tools/fxai_offline_lab.py",
        "Tools/fxai_testlab.py",
        "FXDataEngine/Tools/fxai_offline_lab.py",
        "FXDataEngine/Tools/fxai_testlab.py"
    ]
    private static let allowedOfflineLabSubcommands: Set<String> = [
        "autonomous-governance",
        "best-params",
        "bootstrap",
        "dashboard",
        "deploy-profiles",
        "lineage-report",
        "minimal-bundle",
        "newspulse-health",
        "newspulse-once",
        "newspulse-validate",
        "recover-artifacts",
        "seed-demo",
        "tune-zoo",
        "turso-audit-sync",
        "turso-branch-create",
        "turso-branch-destroy",
        "turso-branch-inventory",
        "turso-pitr-restore",
        "turso-vector-neighbors",
        "turso-vector-reindex",
        "validate-env"
    ]
    private static let allowedTestLabSubcommands: Set<String> = [
        "baseline-save",
        "run-audit",
        "verify-all"
    ]

    public static func shellQuoted(_ value: String) -> String {
        "'" + value.replacingOccurrences(of: "'", with: "'\"'\"'") + "'"
    }

    public static func approvedCommandForTerminalHandoff(_ command: String, projectRoot: URL) throws -> String {
        let assessment = assess(command: command, projectRoot: projectRoot)
        guard assessment.approved else {
            throw FXAICommandSecurityError.rejected(assessment.reason ?? "unknown command policy violation")
        }
        return normalizedCommand(command)
    }

    public static func assess(command: String, projectRoot: URL) -> FXAICommandSecurityAssessment {
        let normalized = normalizedCommand(command)
        guard !normalized.isEmpty else {
            return .rejected("command is empty")
        }
        guard hasOnlySupportedControlCharacters(normalized) else {
            return .rejected("command contains unsupported control characters")
        }

        let lines = normalized
            .split(separator: "\n", omittingEmptySubsequences: false)
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }
        guard let firstLine = lines.first else {
            return .rejected("command has no executable lines")
        }

        guard case .success(let firstTokens) = tokens(for: firstLine),
              firstTokens.first == "cd"
        else {
            return .rejected("command must start by changing into the FXAI project or one of its subdirectories")
        }

        for line in lines {
            let assessment = validate(line: line, projectRoot: projectRoot)
            if !assessment.approved {
                return assessment
            }
        }
        return .approved
    }

    private static func validate(line: String, projectRoot: URL) -> FXAICommandSecurityAssessment {
        if let reason = shellExpansionOrChainingViolation(in: line) {
            return .rejected(reason)
        }

        let parsed = tokens(for: line)
        let lineTokens: [String]
        switch parsed {
        case .success(let tokens):
            lineTokens = tokens
        case .failure(let reason):
            return .rejected(reason)
        }

        guard let executable = lineTokens.first else {
            return .rejected("command line is empty")
        }

        switch executable {
        case "cd":
            return validateChangeDirectory(lineTokens, projectRoot: projectRoot)
        case "swift":
            return validateSwift(lineTokens)
        case "python3":
            return validatePython(lineTokens)
        case "./fxai":
            return lineTokens == ["./fxai", "certify", "--all"]
                ? .approved
                : .rejected("fxai wrapper may only run the full certification command")
        case "./Tools/package_gui_release.sh":
            return lineTokens.count == 1
                ? .approved
                : .rejected("GUI packaging command does not accept shell-supplied arguments")
        default:
            return .rejected("unsupported command executable: \(executable)")
        }
    }

    private static func validateChangeDirectory(_ tokens: [String], projectRoot: URL) -> FXAICommandSecurityAssessment {
        guard tokens.count == 2 else {
            return .rejected("cd must receive exactly one path")
        }
        let root = projectRoot.standardizedFileURL
        let destination = URL(fileURLWithPath: tokens[1], isDirectory: true).standardizedFileURL
        guard destination.path == root.path || destination.path.hasPrefix(root.path + "/") else {
            return .rejected("cd target is outside the FXAI project root")
        }
        return .approved
    }

    private static func validateSwift(_ tokens: [String]) -> FXAICommandSecurityAssessment {
        if tokens.count == 4,
           ["build", "test"].contains(tokens[1]),
           tokens[2] == "--package-path",
           allowedSwiftPackages.contains(tokens[3])
        {
            return .approved
        }
        if tokens.count == 6,
           tokens[1] == "test",
           tokens[2] == "--package-path",
           allowedSwiftPackages.contains(tokens[3]),
           tokens[4] == "--filter",
           allowedSwiftTestFilters.contains(tokens[5])
        {
            return .approved
        }
        if tokens == ["swift", "run", "--package-path", "FXBacktestAgent", "FXBacktestAgent", "--self-check"] {
            return .approved
        }
        return .rejected("swift command is outside the approved package build/test/run policy")
    }

    private static func validatePython(_ tokens: [String]) -> FXAICommandSecurityAssessment {
        guard tokens.count >= 3 else {
            return .rejected("python3 command must call an approved FXAI tool script and subcommand")
        }
        let script = tokens[1]
        let subcommand = tokens[2]
        guard allowedPythonScripts.contains(script) else {
            return .rejected("python3 may only call approved FXAI tool scripts")
        }
        if script.hasSuffix("fxai_offline_lab.py") {
            guard allowedOfflineLabSubcommands.contains(subcommand) else {
                return .rejected("unsupported Offline Lab subcommand: \(subcommand)")
            }
        } else if script.hasSuffix("fxai_testlab.py") {
            guard allowedTestLabSubcommands.contains(subcommand) else {
                return .rejected("unsupported Test Lab subcommand: \(subcommand)")
            }
        }
        return .approved
    }

    private static func normalizedCommand(_ command: String) -> String {
        command
            .replacingOccurrences(of: "\r\n", with: "\n")
            .trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private static func hasOnlySupportedControlCharacters(_ command: String) -> Bool {
        command.unicodeScalars.allSatisfy { scalar in
            scalar.value == 10 || scalar.value == 9 || scalar.value >= 32 && scalar.value != 127
        }
    }

    private enum TokenParseResult {
        case success([String])
        case failure(String)
    }

    private static func tokens(for line: String) -> TokenParseResult {
        var tokens: [String] = []
        var current = ""
        var quote: Character?

        for character in line {
            if let activeQuote = quote {
                if character == activeQuote {
                    quote = nil
                } else {
                    current.append(character)
                }
            } else if character == "'" || character == "\"" {
                quote = character
            } else if character == " " || character == "\t" {
                if !current.isEmpty {
                    tokens.append(current)
                    current = ""
                }
            } else {
                current.append(character)
            }
        }

        if let quote {
            return .failure("unterminated \(quote == "'" ? "single" : "double") quote")
        }
        if !current.isEmpty {
            tokens.append(current)
        }
        return .success(tokens)
    }

    private static func shellExpansionOrChainingViolation(in line: String) -> String? {
        var quote: Character?
        let unsafeOutsideQuotes: Set<Character> = [
            ";", "&", "|", "<", ">", "`", "$", "\\", "#", "~", "*", "?", "(", ")", "[", "]"
        ]

        for character in line {
            if let activeQuote = quote {
                if character == activeQuote {
                    quote = nil
                } else if activeQuote == "\"" && (character == "$" || character == "`") {
                    return "double-quoted command substitution is not allowed"
                }
            } else if character == "'" || character == "\"" {
                quote = character
            } else if unsafeOutsideQuotes.contains(character) {
                return "shell chaining, redirection, expansion, and escapes are not allowed"
            }
        }
        return nil
    }
}
