import Foundation

public struct RuntimeArtifactHealth: Hashable, Sendable {
    public let artifactExists: Bool
    public let staleArtifact: Bool
    public let missingDeployment: Bool
    public let missingRouter: Bool
    public let missingSupervisorService: Bool
    public let missingSupervisorCommand: Bool
    public let missingWorldPlan: Bool
    public let artifactAgeSeconds: Int
    public let performanceFailures: [String]
    public let artifactSizeFailures: [String]

    public var hasBlockingIssue: Bool {
        missingDeployment || missingRouter || missingSupervisorService || missingSupervisorCommand || missingWorldPlan
    }
}

public struct KeyValueRecord: Identifiable, Hashable, Sendable {
    public let id: String
    public let key: String
    public let value: String

    public init(key: String, value: String) {
        self.id = key
        self.key = key
        self.value = value
    }

    public var numericValue: Double? {
        Double(value)
    }
}

public struct RuntimeArtifactSection: Identifiable, Hashable, Sendable {
    public let id: String
    public let title: String
    public let sourcePath: URL?
    public let values: [KeyValueRecord]

    public init(title: String, sourcePath: URL?, values: [KeyValueRecord]) {
        self.id = title
        self.title = title
        self.sourcePath = sourcePath
        self.values = values
    }
}

public struct PromotionChampionRecord: Identifiable, Hashable, Sendable {
    public let id: String
    public let symbol: String
    public let pluginName: String
    public let status: String
    public let promotionTier: String
    public let championScore: Double
    public let challengerScore: Double
    public let portfolioScore: Double
    public let reviewedAt: Date?
    public let setPath: URL?
    public let profileName: String?

    public init(
        symbol: String,
        pluginName: String,
        status: String,
        promotionTier: String,
        championScore: Double,
        challengerScore: Double,
        portfolioScore: Double,
        reviewedAt: Date?,
        setPath: URL?,
        profileName: String?
    ) {
        self.id = "\(symbol)::\(pluginName)::\(status)"
        self.symbol = symbol
        self.pluginName = pluginName
        self.status = status
        self.promotionTier = promotionTier
        self.championScore = championScore
        self.challengerScore = challengerScore
        self.portfolioScore = portfolioScore
        self.reviewedAt = reviewedAt
        self.setPath = setPath
        self.profileName = profileName
    }
}

public struct RuntimeDeploymentDetail: Identifiable, Hashable, Sendable {
    public let id: String
    public let symbol: String
    public let profileName: String
    public let pluginName: String
    public let promotionTier: String
    public let runtimeMode: String
    public let createdAt: Date?
    public let reviewedAt: Date?
    public let artifactHealth: RuntimeArtifactHealth
    public let deploymentPath: URL?
    public let studentRouterPath: URL?
    public let supervisorServicePath: URL?
    public let supervisorCommandPath: URL?
    public let worldPlanPath: URL?
    public let deploymentSections: [RuntimeArtifactSection]
    public let routerSections: [RuntimeArtifactSection]
    public let supervisorSections: [RuntimeArtifactSection]
    public let commandSections: [RuntimeArtifactSection]
    public let worldSections: [RuntimeArtifactSection]
    public let attributionSections: [RuntimeArtifactSection]
    public let featureHighlights: [KeyValueRecord]
    public let studentRouterWeights: [KeyValueRecord]
    public let familyWeights: [KeyValueRecord]
    public let prunedPlugins: [String]

    public var summaryMetrics: [KeyValueRecord] {
        let preferredKeys = [
            "policy_trade_floor",
            "policy_no_trade_cap",
            "supervisor_blend",
            "teacher_signal_gain",
            "student_signal_gain",
            "budget_multiplier",
            "entry_floor",
            "reduce_bias",
            "exit_bias",
            "sigma_scale",
            "spread_scale",
            "shock_decay"
        ]

        let allValues = (deploymentSections + supervisorSections + worldSections)
            .flatMap(\.values)

        return preferredKeys.compactMap { key in
            allValues.first(where: { $0.key == key })
        }
    }
}

public struct RuntimeOperationsSnapshot: Sendable {
    public let generatedAt: Date
    public let profileName: String?
    public let deployments: [RuntimeDeploymentDetail]
    public let champions: [PromotionChampionRecord]

    public var symbols: [String] {
        deployments.map(\.symbol)
    }
}
