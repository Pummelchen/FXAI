import Foundation

public struct DriftGovernanceActionRecord: Identifiable, Hashable, Sendable {
    public let id: String
    public let pluginName: String
    public let previousState: String
    public let newState: String
    public let actionKind: String
    public let actionApplied: Bool
    public let createdAt: Date?

    public init(
        pluginName: String,
        previousState: String,
        newState: String,
        actionKind: String,
        actionApplied: Bool,
        createdAt: Date?
    ) {
        let stamp = createdAt?.ISO8601Format() ?? "unknown"
        id = "\(pluginName)::\(actionKind)::\(stamp)"
        self.pluginName = pluginName
        self.previousState = previousState
        self.newState = newState
        self.actionKind = actionKind
        self.actionApplied = actionApplied
        self.createdAt = createdAt
    }
}

public struct DriftGovernanceChallengerEvaluation: Hashable, Sendable {
    public let eligibilityState: String
    public let qualifies: Bool
    public let supportCount: Int
    public let shadowSupport: Int
    public let walkforwardScore: Double
    public let recentScore: Double
    public let adversarialScore: Double
    public let macroEventScore: Double
    public let calibrationError: Double
    public let issueCount: Double
    public let liveShadowScore: Double
    public let liveReliability: Double
    public let portfolioScore: Double
    public let promotionMargin: Double

    public init(
        eligibilityState: String,
        qualifies: Bool,
        supportCount: Int,
        shadowSupport: Int,
        walkforwardScore: Double,
        recentScore: Double,
        adversarialScore: Double,
        macroEventScore: Double,
        calibrationError: Double,
        issueCount: Double,
        liveShadowScore: Double,
        liveReliability: Double,
        portfolioScore: Double,
        promotionMargin: Double
    ) {
        self.eligibilityState = eligibilityState
        self.qualifies = qualifies
        self.supportCount = supportCount
        self.shadowSupport = shadowSupport
        self.walkforwardScore = walkforwardScore
        self.recentScore = recentScore
        self.adversarialScore = adversarialScore
        self.macroEventScore = macroEventScore
        self.calibrationError = calibrationError
        self.issueCount = issueCount
        self.liveShadowScore = liveShadowScore
        self.liveReliability = liveReliability
        self.portfolioScore = portfolioScore
        self.promotionMargin = promotionMargin
    }
}

public struct DriftGovernancePluginSnapshot: Identifiable, Hashable, Sendable {
    public let id: String
    public let pluginName: String
    public let familyID: Int
    public let familyName: String
    public let baseRegistryStatus: String
    public let healthState: String
    public let governanceState: String
    public let recommendedGovernanceState: String
    public let actionRecommendation: String
    public let actionApplied: Bool
    public let weightMultiplier: Double
    public let restrictLive: Bool
    public let shadowOnly: Bool
    public let disabled: Bool
    public let aggregateRiskScore: Double
    public let driftScores: [KeyValueRecord]
    public let support: [KeyValueRecord]
    public let reasonCodes: [String]
    public let qualityFlags: [KeyValueRecord]
    public let contextSummary: [KeyValueRecord]
    public let challengerEvaluation: DriftGovernanceChallengerEvaluation?

    public init(
        pluginName: String,
        familyID: Int,
        familyName: String,
        baseRegistryStatus: String,
        healthState: String,
        governanceState: String,
        recommendedGovernanceState: String,
        actionRecommendation: String,
        actionApplied: Bool,
        weightMultiplier: Double,
        restrictLive: Bool,
        shadowOnly: Bool,
        disabled: Bool,
        aggregateRiskScore: Double,
        driftScores: [KeyValueRecord],
        support: [KeyValueRecord],
        reasonCodes: [String],
        qualityFlags: [KeyValueRecord],
        contextSummary: [KeyValueRecord],
        challengerEvaluation: DriftGovernanceChallengerEvaluation?
    ) {
        id = pluginName
        self.pluginName = pluginName
        self.familyID = familyID
        self.familyName = familyName
        self.baseRegistryStatus = baseRegistryStatus
        self.healthState = healthState
        self.governanceState = governanceState
        self.recommendedGovernanceState = recommendedGovernanceState
        self.actionRecommendation = actionRecommendation
        self.actionApplied = actionApplied
        self.weightMultiplier = weightMultiplier
        self.restrictLive = restrictLive
        self.shadowOnly = shadowOnly
        self.disabled = disabled
        self.aggregateRiskScore = aggregateRiskScore
        self.driftScores = driftScores
        self.support = support
        self.reasonCodes = reasonCodes
        self.qualityFlags = qualityFlags
        self.contextSummary = contextSummary
        self.challengerEvaluation = challengerEvaluation
    }
}

public struct DriftGovernanceSymbolSnapshot: Identifiable, Hashable, Sendable {
    public let id: String
    public let symbol: String
    public let pluginCount: Int
    public let healthCounts: [KeyValueRecord]
    public let governanceCounts: [KeyValueRecord]
    public let actionCounts: [KeyValueRecord]
    public let latestContext: [KeyValueRecord]
    public let plugins: [DriftGovernancePluginSnapshot]
    public let recentActions: [DriftGovernanceActionRecord]

    public init(
        symbol: String,
        pluginCount: Int,
        healthCounts: [KeyValueRecord],
        governanceCounts: [KeyValueRecord],
        actionCounts: [KeyValueRecord],
        latestContext: [KeyValueRecord],
        plugins: [DriftGovernancePluginSnapshot],
        recentActions: [DriftGovernanceActionRecord]
    ) {
        id = symbol
        self.symbol = symbol
        self.pluginCount = pluginCount
        self.healthCounts = healthCounts
        self.governanceCounts = governanceCounts
        self.actionCounts = actionCounts
        self.latestContext = latestContext
        self.plugins = plugins
        self.recentActions = recentActions
    }
}

public struct DriftGovernanceSnapshot: Sendable {
    public let generatedAt: Date
    public let profileName: String
    public let policyVersion: Int
    public let actionMode: String
    public let symbolCount: Int
    public let pluginCount: Int
    public let latestActionCount: Int
    public let healthCounts: [KeyValueRecord]
    public let governanceCounts: [KeyValueRecord]
    public let actionCounts: [KeyValueRecord]
    public let statusRecords: [KeyValueRecord]
    public let artifactPaths: [KeyValueRecord]
    public let symbols: [DriftGovernanceSymbolSnapshot]

    public init(
        generatedAt: Date,
        profileName: String,
        policyVersion: Int,
        actionMode: String,
        symbolCount: Int,
        pluginCount: Int,
        latestActionCount: Int,
        healthCounts: [KeyValueRecord],
        governanceCounts: [KeyValueRecord],
        actionCounts: [KeyValueRecord],
        statusRecords: [KeyValueRecord],
        artifactPaths: [KeyValueRecord],
        symbols: [DriftGovernanceSymbolSnapshot]
    ) {
        self.generatedAt = generatedAt
        self.profileName = profileName
        self.policyVersion = policyVersion
        self.actionMode = actionMode
        self.symbolCount = symbolCount
        self.pluginCount = pluginCount
        self.latestActionCount = latestActionCount
        self.healthCounts = healthCounts
        self.governanceCounts = governanceCounts
        self.actionCounts = actionCounts
        self.statusRecords = statusRecords
        self.artifactPaths = artifactPaths
        self.symbols = symbols
    }
}
