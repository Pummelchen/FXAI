import Foundation

public enum IncidentSeverity: String, CaseIterable, Codable, Comparable, Hashable, Sendable {
    case info
    case warning
    case critical

    public static func < (lhs: IncidentSeverity, rhs: IncidentSeverity) -> Bool {
        lhs.rank < rhs.rank
    }

    public var rank: Int {
        switch self {
        case .info: 0
        case .warning: 1
        case .critical: 2
        }
    }

    public var title: String {
        rawValue.capitalized
    }
}

public enum IncidentCategory: String, CaseIterable, Codable, Hashable, Sendable {
    case build
    case runtime
    case researchOS = "research_os"
    case promotion
    case performance

    public var title: String {
        switch self {
        case .build: "Build"
        case .runtime: "Runtime"
        case .researchOS: "Research OS"
        case .promotion: "Promotion"
        case .performance: "Performance"
        }
    }
}

public struct IncidentAction: Identifiable, Hashable, Sendable {
    public let id: String
    public let title: String
    public let summary: String
    public let command: String
    public let destinationSelection: String?

    public init(title: String, summary: String, command: String, destinationSelection: String? = nil) {
        self.id = title
        self.title = title
        self.summary = summary
        self.command = command
        self.destinationSelection = destinationSelection
    }
}

public struct RecoveryStep: Identifiable, Hashable, Sendable {
    public let id: String
    public let title: String
    public let summary: String
    public let command: String
    public let destinationSelection: String?

    public init(title: String, summary: String, command: String, destinationSelection: String? = nil) {
        self.id = title
        self.title = title
        self.summary = summary
        self.command = command
        self.destinationSelection = destinationSelection
    }
}

public struct RecoveryPlaybook: Hashable, Sendable {
    public let title: String
    public let summary: String
    public let steps: [RecoveryStep]
}

public struct FXAIIncident: Identifiable, Hashable, Sendable {
    public let id: String
    public let severity: IncidentSeverity
    public let category: IncidentCategory
    public let title: String
    public let summary: String
    public let affectedSymbol: String?
    public let detailLines: [String]
    public let actions: [IncidentAction]
    public let playbook: RecoveryPlaybook

    public init(
        id: String,
        severity: IncidentSeverity,
        category: IncidentCategory,
        title: String,
        summary: String,
        affectedSymbol: String? = nil,
        detailLines: [String] = [],
        actions: [IncidentAction],
        playbook: RecoveryPlaybook
    ) {
        self.id = id
        self.severity = severity
        self.category = category
        self.title = title
        self.summary = summary
        self.affectedSymbol = affectedSymbol
        self.detailLines = detailLines
        self.actions = actions
        self.playbook = playbook
    }
}

public struct IncidentCenterSnapshot: Sendable {
    public let generatedAt: Date
    public let incidents: [FXAIIncident]

    public init(generatedAt: Date = Date(), incidents: [FXAIIncident]) {
        self.generatedAt = generatedAt
        self.incidents = incidents
    }

    public var highestSeverity: IncidentSeverity? {
        incidents.map(\.severity).max()
    }

    public func count(for severity: IncidentSeverity) -> Int {
        incidents.filter { $0.severity == severity }.count
    }
}
