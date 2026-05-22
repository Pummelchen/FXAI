import Foundation

public struct CommandRecipe: Identifiable, Hashable, Sendable {
    public let id: String
    public let role: WorkspaceRole
    public let title: String
    public let summary: String
    public let command: String
    public let commandKind: String

    public init(role: WorkspaceRole, title: String, summary: String, command: String, commandKind: String) {
        self.id = "\(role.rawValue)::\(title)"
        self.role = role
        self.title = title
        self.summary = summary
        self.command = command
        self.commandKind = commandKind
    }
}
