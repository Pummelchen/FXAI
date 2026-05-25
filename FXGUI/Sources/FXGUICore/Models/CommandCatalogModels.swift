import Foundation

public enum FXGUICommandExecutionPath: String, Codable, CaseIterable, Sendable {
    case versionedAPI
    case terminalFallback
    case disabledUntilAPIExists
}

public enum FXGUICommandRiskLevel: String, Codable, CaseIterable, Sendable {
    case readOnly
    case localBuild
    case databaseMutation
    case backtestExecution
    case demoExecution
    case liveExecution
}

public struct FXGUICommandParameter: Codable, Hashable, Identifiable, Sendable {
    public var id: String { key }
    public let key: String
    public let displayName: String
    public let required: Bool
    public let defaultValue: String
    public let validationRule: String

    public init(
        key: String,
        displayName: String,
        required: Bool,
        defaultValue: String = "",
        validationRule: String = ""
    ) {
        self.key = key
        self.displayName = displayName
        self.required = required
        self.defaultValue = defaultValue
        self.validationRule = validationRule
    }
}

public struct FXGUICommandDefinition: Codable, Hashable, Identifiable, Sendable {
    public let id: String
    public let ownerProject: String
    public let role: WorkspaceRole
    public let title: String
    public let summary: String
    public let apiVersion: String
    public let executionPath: FXGUICommandExecutionPath
    public let riskLevel: FXGUICommandRiskLevel
    public let destructive: Bool
    public let parameters: [FXGUICommandParameter]
    public let terminalEquivalent: String
    public let apiPath: String?
    public let expectedResultType: String
    public let logStreamKey: String

    public init(
        id: String,
        ownerProject: String,
        role: WorkspaceRole,
        title: String,
        summary: String,
        apiVersion: String,
        executionPath: FXGUICommandExecutionPath,
        riskLevel: FXGUICommandRiskLevel,
        destructive: Bool = false,
        parameters: [FXGUICommandParameter] = [],
        terminalEquivalent: String,
        apiPath: String? = nil,
        expectedResultType: String,
        logStreamKey: String
    ) {
        self.id = id
        self.ownerProject = ownerProject
        self.role = role
        self.title = title
        self.summary = summary
        self.apiVersion = apiVersion
        self.executionPath = executionPath
        self.riskLevel = riskLevel
        self.destructive = destructive
        self.parameters = parameters
        self.terminalEquivalent = terminalEquivalent
        self.apiPath = apiPath
        self.expectedResultType = expectedResultType
        self.logStreamKey = logStreamKey
    }
}

public struct FXGUICommandCatalogSnapshot: Codable, Hashable, Sendable {
    public let generatedAtUTC: Int64
    public let commands: [FXGUICommandDefinition]

    public init(generatedAtUTC: Int64 = Int64(Date().timeIntervalSince1970), commands: [FXGUICommandDefinition]) {
        self.generatedAtUTC = generatedAtUTC
        self.commands = commands.sorted { $0.id < $1.id }
    }

    public func commands(for role: WorkspaceRole) -> [FXGUICommandDefinition] {
        commands.filter { $0.role == role }
    }

    public func command(id: String) -> FXGUICommandDefinition? {
        commands.first { $0.id == id }
    }
}
