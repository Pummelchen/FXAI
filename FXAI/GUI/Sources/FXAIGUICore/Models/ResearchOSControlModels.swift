import Foundation

public struct ResearchOSEnvironmentStatus: Hashable, Sendable {
    public let backend: String
    public let syncMode: String
    public let databasePath: URL?
    public let databaseName: String
    public let organizationSlug: String
    public let groupName: String
    public let locationName: String
    public let cliConfigPath: URL?
    public let syncIntervalSeconds: Int?
    public let encryptionEnabled: Bool
    public let platformAPIEnabled: Bool
    public let syncEnabled: Bool
    public let authTokenConfigured: Bool
    public let apiTokenConfigured: Bool
    public let configError: String?

    public var isHealthy: Bool {
        configError == nil || configError?.isEmpty == true
    }
}

public struct ResearchOSBranchRecord: Identifiable, Hashable, Sendable {
    public let id: String
    public let name: String
    public let sourceDatabase: String
    public let parentName: String
    public let branchKind: String
    public let status: String
    public let groupName: String
    public let locationName: String
    public let hostname: String
    public let syncURL: String
    public let envArtifactPath: URL?
    public let isBranch: Bool
    public let createdAt: Date?
    public let sourceTimestamp: String

    public init(
        name: String,
        sourceDatabase: String,
        parentName: String,
        branchKind: String,
        status: String,
        groupName: String,
        locationName: String,
        hostname: String,
        syncURL: String,
        envArtifactPath: URL?,
        isBranch: Bool,
        createdAt: Date?,
        sourceTimestamp: String
    ) {
        self.id = [name, sourceDatabase, branchKind, sourceTimestamp].joined(separator: "::")
        self.name = name
        self.sourceDatabase = sourceDatabase
        self.parentName = parentName
        self.branchKind = branchKind
        self.status = status
        self.groupName = groupName
        self.locationName = locationName
        self.hostname = hostname
        self.syncURL = syncURL
        self.envArtifactPath = envArtifactPath
        self.isBranch = isBranch
        self.createdAt = createdAt
        self.sourceTimestamp = sourceTimestamp
    }
}

public struct ResearchOSAuditEvent: Identifiable, Hashable, Sendable {
    public let id: String
    public let organizationSlug: String
    public let eventID: String
    public let eventType: String
    public let targetName: String
    public let occurredAt: Date?
    public let observedAt: Date?

    public init(
        organizationSlug: String,
        eventID: String,
        eventType: String,
        targetName: String,
        occurredAt: Date?,
        observedAt: Date?
    ) {
        self.id = [organizationSlug, eventID, eventType].joined(separator: "::")
        self.organizationSlug = organizationSlug
        self.eventID = eventID
        self.eventType = eventType
        self.targetName = targetName
        self.occurredAt = occurredAt
        self.observedAt = observedAt
    }
}

public struct ResearchOSAnalogNeighbor: Identifiable, Hashable, Sendable {
    public let id: String
    public let sourceKey: String
    public let pluginName: String
    public let distance: Double
    public let similarity: Double
    public let score: Double
    public let sourceType: String
    public let scope: String
    public let payload: [KeyValueRecord]

    public init(
        sourceKey: String,
        pluginName: String,
        distance: Double,
        similarity: Double,
        score: Double,
        sourceType: String,
        scope: String,
        payload: [KeyValueRecord]
    ) {
        self.id = [sourceKey, pluginName, scope].joined(separator: "::")
        self.sourceKey = sourceKey
        self.pluginName = pluginName
        self.distance = distance
        self.similarity = similarity
        self.score = score
        self.sourceType = sourceType
        self.scope = scope
        self.payload = payload
    }
}

public struct ResearchOSSymbolControl: Identifiable, Hashable, Sendable {
    public let id: String
    public let symbol: String
    public let analogNeighbors: [ResearchOSAnalogNeighbor]
    public let deploymentArtifactPath: URL?
    public let deploymentCreatedAt: Date?

    public init(
        symbol: String,
        analogNeighbors: [ResearchOSAnalogNeighbor],
        deploymentArtifactPath: URL?,
        deploymentCreatedAt: Date?
    ) {
        self.id = symbol
        self.symbol = symbol
        self.analogNeighbors = analogNeighbors
        self.deploymentArtifactPath = deploymentArtifactPath
        self.deploymentCreatedAt = deploymentCreatedAt
    }
}

public struct ResearchOSControlSnapshot: Sendable {
    public let generatedAt: Date
    public let profileName: String?
    public let environment: ResearchOSEnvironmentStatus?
    public let branches: [ResearchOSBranchRecord]
    public let auditEvents: [ResearchOSAuditEvent]
    public let symbols: [ResearchOSSymbolControl]
    public let sourceOfTruth: [KeyValueRecord]

    public var branchCount: Int {
        branches.count
    }

    public var activeBranchCount: Int {
        branches.filter { $0.status.lowercased() != "destroyed" }.count
    }

    public var auditEventCount: Int {
        auditEvents.count
    }
}

public enum ResearchOSBranchAction: String, CaseIterable, Codable, Identifiable, Sendable {
    case create
    case pitrRestore = "pitr_restore"
    case inventory
    case destroy

    public var id: String { rawValue }

    public var title: String {
        switch self {
        case .create: "Create Branch"
        case .pitrRestore: "PITR Restore"
        case .inventory: "Inventory"
        case .destroy: "Destroy"
        }
    }
}

public struct ResearchOSBranchDraft: Codable, Hashable, Sendable {
    public var action: ResearchOSBranchAction
    public var profileName: String
    public var sourceDatabase: String
    public var targetDatabase: String
    public var timestamp: String
    public var groupName: String
    public var locationName: String
    public var tokenExpiration: String
    public var readOnlyToken: Bool

    public init(
        action: ResearchOSBranchAction = .inventory,
        profileName: String = "continuous",
        sourceDatabase: String = "",
        targetDatabase: String = "",
        timestamp: String = "",
        groupName: String = "",
        locationName: String = "",
        tokenExpiration: String = "7d",
        readOnlyToken: Bool = true
    ) {
        self.action = action
        self.profileName = profileName
        self.sourceDatabase = sourceDatabase
        self.targetDatabase = targetDatabase
        self.timestamp = timestamp
        self.groupName = groupName
        self.locationName = locationName
        self.tokenExpiration = tokenExpiration
        self.readOnlyToken = readOnlyToken
    }
}

public struct ResearchOSAuditDraft: Codable, Hashable, Sendable {
    public var limit: Int
    public var pages: Int

    public init(limit: Int = 50, pages: Int = 1) {
        self.limit = limit
        self.pages = pages
    }
}

public struct ResearchOSVectorDraft: Codable, Hashable, Sendable {
    public var profileName: String
    public var symbol: String
    public var limit: Int

    public init(profileName: String = "continuous", symbol: String = "EURUSD", limit: Int = 5) {
        self.profileName = profileName
        self.symbol = symbol
        self.limit = limit
    }
}

public struct ResearchOSRecoveryDraft: Codable, Hashable, Sendable {
    public var profileName: String
    public var runtimeMode: String

    public init(profileName: String = "continuous", runtimeMode: String = "research") {
        self.profileName = profileName
        self.runtimeMode = runtimeMode
    }
}
