import Foundation

public struct BuildTargetStatus: Identifiable, Hashable, Sendable {
    public let id: String
    public let name: String
    public let relativePath: String
    public let exists: Bool
    public let modifiedAt: Date?

    public init(name: String, relativePath: String, exists: Bool, modifiedAt: Date?) {
        self.id = relativePath
        self.name = name
        self.relativePath = relativePath
        self.exists = exists
        self.modifiedAt = modifiedAt
    }
}

public struct PluginFamilySummary: Identifiable, Hashable, Sendable {
    public let id: String
    public let family: String
    public let pluginCount: Int
}

public struct ReportCategorySummary: Identifiable, Hashable, Sendable {
    public let id: String
    public let category: String
    public let fileCount: Int
    public let latestModifiedAt: Date?
}

public struct ReportArtifact: Identifiable, Hashable, Sendable {
    public let id: String
    public let category: String
    public let name: String
    public let path: URL
    public let modifiedAt: Date?
    public let sizeBytes: Int64

    public init(category: String, name: String, path: URL, modifiedAt: Date?, sizeBytes: Int64) {
        self.id = path.path
        self.category = category
        self.name = name
        self.path = path
        self.modifiedAt = modifiedAt
        self.sizeBytes = sizeBytes
    }
}

public struct RuntimeProfileSummary: Identifiable, Hashable, Sendable {
    public let id: String
    public let symbol: String
    public let pluginName: String
    public let profileName: String
    public let promotionTier: String
    public let runtimeMode: String
    public let sourcePath: URL
}

public struct OperatorSummary: Hashable, Sendable {
    public let profileName: String?
    public let championCount: Int
    public let deploymentCount: Int
    public let latestReviewedAt: Date?
}

public struct TursoSummary: Hashable, Sendable {
    public let localDatabasePresent: Bool
    public let localDatabasePath: URL?
    public let embeddedReplicaConfigured: Bool
    public let encryptionConfigured: Bool
}

public struct FXAIProjectSnapshot: Sendable {
    public let projectRoot: URL
    public let generatedAt: Date
    public let buildTargets: [BuildTargetStatus]
    public let pluginFamilies: [PluginFamilySummary]
    public let plugins: [PluginDescriptor]
    public let reportCategories: [ReportCategorySummary]
    public let recentArtifacts: [ReportArtifact]
    public let runtimeProfiles: [RuntimeProfileSummary]
    public let operatorSummary: OperatorSummary
    public let tursoSummary: TursoSummary

    public var totalPluginCount: Int {
        plugins.count
    }

    public var totalReportCount: Int {
        reportCategories.reduce(into: 0) { partialResult, summary in
            partialResult += summary.fileCount
        }
    }

    public var cleanBuildTargetCount: Int {
        buildTargets.filter(\.exists).count
    }
}
