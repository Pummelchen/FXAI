import Foundation

public enum ResearchOSCommandFactory {
    public static func environmentDiagnostics(projectRoot: URL, profileName: String) -> String {
        let effectiveProfile = nonEmpty(profileName) ?? "continuous"
        return [
            "cd \(shellQuoted(projectRoot.path))",
            "python3 Tools/fxai_offline_lab.py validate-env",
            "python3 Tools/fxai_offline_lab.py dashboard --profile \(shellQuoted(effectiveProfile))"
        ].joined(separator: "\n")
    }

    public static func branchCommand(projectRoot: URL, draft: ResearchOSBranchDraft) -> String {
        let command: String
        switch draft.action {
        case .create:
            command = buildBranchCreateCommand(draft: draft)
        case .pitrRestore:
            command = buildPITRCommand(draft: draft)
        case .inventory:
            command = buildBranchInventoryCommand(draft: draft)
        case .destroy:
            command = buildBranchDestroyCommand(draft: draft)
        }
        return [
            "cd \(shellQuoted(projectRoot.path))",
            command
        ].joined(separator: "\n")
    }

    public static func auditSyncCommand(projectRoot: URL, draft: ResearchOSAuditDraft) -> String {
        [
            "cd \(shellQuoted(projectRoot.path))",
            "python3 Tools/fxai_offline_lab.py turso-audit-sync --limit \(max(draft.limit, 1)) --pages \(max(draft.pages, 1))",
            "python3 Tools/fxai_offline_lab.py dashboard --profile 'continuous'"
        ].joined(separator: "\n")
    }

    public static func vectorCommand(projectRoot: URL, draft: ResearchOSVectorDraft, reindexOnly: Bool) -> String {
        var command = "python3 Tools/fxai_offline_lab.py turso-vector-reindex --profile \(shellQuoted(draft.profileName))"
        if !draft.symbol.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            command += " --symbol \(shellQuoted(draft.symbol))"
        }

        var commands = [
            "cd \(shellQuoted(projectRoot.path))",
            command
        ]

        if !reindexOnly {
            commands.append(
                "python3 Tools/fxai_offline_lab.py turso-vector-neighbors --profile \(shellQuoted(draft.profileName)) --symbol \(shellQuoted(draft.symbol)) --limit \(max(draft.limit, 1))"
            )
        }

        return commands.joined(separator: "\n")
    }

    public static func recoveryCommand(projectRoot: URL, draft: ResearchOSRecoveryDraft) -> String {
        [
            "cd \(shellQuoted(projectRoot.path))",
            "python3 Tools/fxai_offline_lab.py recover-artifacts --profile \(shellQuoted(draft.profileName)) --runtime-mode \(shellQuoted(draft.runtimeMode))",
            "python3 Tools/fxai_offline_lab.py lineage-report --profile \(shellQuoted(draft.profileName))",
            "python3 Tools/fxai_offline_lab.py minimal-bundle --profile \(shellQuoted(draft.profileName))"
        ].joined(separator: "\n")
    }

    private static func buildBranchCreateCommand(draft: ResearchOSBranchDraft) -> String {
        var command = "python3 Tools/fxai_offline_lab.py turso-branch-create --profile \(shellQuoted(draft.profileName))"
        if let sourceDatabase = nonEmpty(draft.sourceDatabase) {
            command += " --source-database \(shellQuoted(sourceDatabase))"
        }
        if let targetDatabase = nonEmpty(draft.targetDatabase) {
            command += " --target-database \(shellQuoted(targetDatabase))"
        }
        if let timestamp = nonEmpty(draft.timestamp) {
            command += " --timestamp \(shellQuoted(timestamp))"
        }
        if let groupName = nonEmpty(draft.groupName) {
            command += " --group-name \(shellQuoted(groupName))"
        }
        if let locationName = nonEmpty(draft.locationName) {
            command += " --location-name \(shellQuoted(locationName))"
        }
        if let tokenExpiration = nonEmpty(draft.tokenExpiration) {
            command += " --token-expiration \(shellQuoted(tokenExpiration))"
        }
        if draft.readOnlyToken {
            command += " --read-only-token"
        }
        return command
    }

    private static func buildPITRCommand(draft: ResearchOSBranchDraft) -> String {
        var command = "python3 Tools/fxai_offline_lab.py turso-pitr-restore --profile \(shellQuoted(draft.profileName))"
        command += " --timestamp \(shellQuoted(nonEmpty(draft.timestamp) ?? "2026-04-05T00:00:00Z"))"
        if let sourceDatabase = nonEmpty(draft.sourceDatabase) {
            command += " --source-database \(shellQuoted(sourceDatabase))"
        }
        if let targetDatabase = nonEmpty(draft.targetDatabase) {
            command += " --target-database \(shellQuoted(targetDatabase))"
        }
        if let groupName = nonEmpty(draft.groupName) {
            command += " --group-name \(shellQuoted(groupName))"
        }
        if let locationName = nonEmpty(draft.locationName) {
            command += " --location-name \(shellQuoted(locationName))"
        }
        if let tokenExpiration = nonEmpty(draft.tokenExpiration) {
            command += " --token-expiration \(shellQuoted(tokenExpiration))"
        }
        if draft.readOnlyToken {
            command += " --read-only-token"
        }
        return command
    }

    private static func buildBranchInventoryCommand(draft: ResearchOSBranchDraft) -> String {
        var command = "python3 Tools/fxai_offline_lab.py turso-branch-inventory --profile \(shellQuoted(draft.profileName))"
        if let sourceDatabase = nonEmpty(draft.sourceDatabase) {
            command += " --source-database \(shellQuoted(sourceDatabase))"
        }
        return command
    }

    private static func buildBranchDestroyCommand(draft: ResearchOSBranchDraft) -> String {
        let target = nonEmpty(draft.targetDatabase) ?? "branch-to-destroy"
        return "python3 Tools/fxai_offline_lab.py turso-branch-destroy --target-database \(shellQuoted(target))"
    }

    private static func nonEmpty(_ value: String) -> String? {
        let trimmed = value.trimmingCharacters(in: .whitespacesAndNewlines)
        return trimmed.isEmpty ? nil : trimmed
    }

    private static func shellQuoted(_ value: String) -> String {
        "'" + value.replacingOccurrences(of: "'", with: "'\"'\"'") + "'"
    }
}
