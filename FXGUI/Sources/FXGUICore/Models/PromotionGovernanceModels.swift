import Foundation

// MARK: - Promotion Pipeline Stages

public enum PromotionStage: String, Codable, CaseIterable, Identifiable, Sendable {
    case backtest = "backtest"
    case demoCandidate = "demo_candidate"
    case shadowCandidate = "shadow_candidate"
    case liveCandidate = "live_candidate"
    case production = "production"

    public var id: String { rawValue }

    public var title: String {
        switch self {
        case .backtest: "Backtest"
        case .demoCandidate: "Demo Candidate"
        case .shadowCandidate: "Shadow Candidate"
        case .liveCandidate: "Live Candidate"
        case .production: "Production"
        }
    }

    public var symbolName: String {
        switch self {
        case .backtest: "chart.xyaxis.line"
        case .demoCandidate: "waveform.path.ecg.rectangle"
        case .shadowCandidate: "eye.slash.fill"
        case .liveCandidate: "bolt.horizontal.circle.fill"
        case .production: "checkmark.seal.fill"
        }
    }

    public var isTerminal: Bool {
        self == .production
    }

    public var nextStage: PromotionStage? {
        switch self {
        case .backtest: .demoCandidate
        case .demoCandidate: .shadowCandidate
        case .shadowCandidate: .liveCandidate
        case .liveCandidate: .production
        case .production: nil
        }
    }

    public var requiredApprovalCount: Int {
        switch self {
        case .backtest: 0
        case .demoCandidate: 1
        case .shadowCandidate: 2
        case .liveCandidate: 2
        case .production: 3
        }
    }
}

// MARK: - Approval Decision

public enum ApprovalDecision: String, Codable, CaseIterable, Sendable {
    case pending = "pending"
    case approved = "approved"
    case rejected = "rejected"
    case deferred = "deferred"

    public var title: String {
        switch self {
        case .pending: "Pending"
        case .approved: "Approved"
        case .rejected: "Rejected"
        case .deferred: "Deferred"
        }
    }

    public var symbolName: String {
        switch self {
        case .pending: "clock.fill"
        case .approved: "checkmark.circle.fill"
        case .rejected: "xmark.circle.fill"
        case .deferred: "hourglass"
        }
    }
}

// MARK: - Approval Record

public struct PromotionApprovalRecord: Identifiable, Hashable, Codable, Sendable {
    public let id: String
    public let reviewerRole: WorkspaceRole
    public let decision: ApprovalDecision
    public let note: String
    public let decidedAt: Date

    public init(
        reviewerRole: WorkspaceRole,
        decision: ApprovalDecision,
        note: String,
        decidedAt: Date = Date()
    ) {
        self.id = "\(reviewerRole.rawValue)::\(decidedAt.timeIntervalSince1970)"
        self.reviewerRole = reviewerRole
        self.decision = decision
        self.note = note
        self.decidedAt = decidedAt
    }
}

// MARK: - Audit Note

public struct PromotionAuditNote: Identifiable, Hashable, Codable, Sendable {
    public let id: String
    public let authorRole: WorkspaceRole
    public let text: String
    public let createdAt: Date
    public let stageAtCreation: PromotionStage

    public init(
        authorRole: WorkspaceRole,
        text: String,
        stageAtCreation: PromotionStage,
        createdAt: Date = Date()
    ) {
        self.id = UUID().uuidString
        self.authorRole = authorRole
        self.text = text
        self.createdAt = createdAt
        self.stageAtCreation = stageAtCreation
    }
}

// MARK: - Promotion Candidate

public struct PromotionCandidate: Identifiable, Hashable, Sendable {
    public let id: String
    public let symbol: String
    public let pluginName: String
    public let currentStage: PromotionStage
    public let championScore: Double
    public let challengerScore: Double
    public let portfolioScore: Double
    public let approvals: [PromotionApprovalRecord]
    public let auditNotes: [PromotionAuditNote]
    public let createdAt: Date
    public let updatedAt: Date
    public let setPath: URL?
    public let profileName: String?

    public init(
        symbol: String,
        pluginName: String,
        currentStage: PromotionStage,
        championScore: Double,
        challengerScore: Double,
        portfolioScore: Double,
        approvals: [PromotionApprovalRecord] = [],
        auditNotes: [PromotionAuditNote] = [],
        createdAt: Date = Date(),
        updatedAt: Date = Date(),
        setPath: URL? = nil,
        profileName: String? = nil
    ) {
        self.id = "\(symbol)::\(pluginName)"
        self.symbol = symbol
        self.pluginName = pluginName
        self.currentStage = currentStage
        self.championScore = championScore
        self.challengerScore = challengerScore
        self.portfolioScore = portfolioScore
        self.approvals = approvals
        self.auditNotes = auditNotes
        self.createdAt = createdAt
        self.updatedAt = updatedAt
        self.setPath = setPath
        self.profileName = profileName
    }

    public var approvalCount: Int {
        approvals.filter { $0.decision == .approved }.count
    }

    public var rejectionCount: Int {
        approvals.filter { $0.decision == .rejected }.count
    }

    public var canAdvance: Bool {
        guard let nextStage = currentStage.nextStage else { return false }
        return approvalCount >= nextStage.requiredApprovalCount && rejectionCount == 0
    }

    public var scoreDelta: Double {
        championScore - challengerScore
    }
}

// MARK: - Rollback Record

public struct RollbackRecord: Identifiable, Hashable, Sendable {
    public let id: String
    public let candidateID: String
    public let fromStage: PromotionStage
    public let toStage: PromotionStage
    public let reason: String
    public let initiatedBy: WorkspaceRole
    public let initiatedAt: Date

    public init(
        candidateID: String,
        fromStage: PromotionStage,
        toStage: PromotionStage,
        reason: String,
        initiatedBy: WorkspaceRole,
        initiatedAt: Date = Date()
    ) {
        self.id = UUID().uuidString
        self.candidateID = candidateID
        self.fromStage = fromStage
        self.toStage = toStage
        self.reason = reason
        self.initiatedBy = initiatedBy
        self.initiatedAt = initiatedAt
    }
}

// MARK: - Kill-Switch State

public struct KillSwitchState: Hashable, Sendable {
    public let accountEnabled: Bool
    public let symbolEnabled: Bool
    public let globalEnabled: Bool
    public let lastCheckedAt: Date
    public let symbol: String?

    public init(
        accountEnabled: Bool = true,
        symbolEnabled: Bool = true,
        globalEnabled: Bool = true,
        lastCheckedAt: Date = Date(),
        symbol: String? = nil
    ) {
        self.accountEnabled = accountEnabled
        self.symbolEnabled = symbolEnabled
        self.globalEnabled = globalEnabled
        self.lastCheckedAt = lastCheckedAt
        self.symbol = symbol
    }

    public var isArmed: Bool {
        !accountEnabled || !symbolEnabled || !globalEnabled
    }

    public var statusTitle: String {
        if !globalEnabled { return "Global Kill Active" }
        if !accountEnabled { return "Account Halted" }
        if !symbolEnabled { return "Symbol Halted" }
        return "All Clear"
    }
}

// MARK: - Evidence Pack

public struct EvidencePack: Identifiable, Hashable, Sendable {
    public let id: String
    public let candidateID: String
    public let symbol: String
    public let pluginName: String
    public let stage: PromotionStage
    public let generatedAt: Date
    public let sections: [EvidencePackSection]

    public init(
        candidateID: String,
        symbol: String,
        pluginName: String,
        stage: PromotionStage,
        generatedAt: Date = Date(),
        sections: [EvidencePackSection] = []
    ) {
        self.id = UUID().uuidString
        self.candidateID = candidateID
        self.symbol = symbol
        self.pluginName = pluginName
        self.stage = stage
        self.generatedAt = generatedAt
        self.sections = sections
    }

    public var totalArtifacts: Int {
        sections.reduce(0) { $0 + $1.artifacts.count }
    }
}

public struct EvidencePackSection: Identifiable, Hashable, Sendable {
    public let id: String
    public let title: String
    public let artifacts: [EvidencePackArtifact]

    public init(title: String, artifacts: [EvidencePackArtifact]) {
        self.id = title
        self.title = title
        self.artifacts = artifacts
    }
}

public struct EvidencePackArtifact: Identifiable, Hashable, Sendable {
    public let id: String
    public let name: String
    public let summary: String
    public let path: URL?

    public init(name: String, summary: String, path: URL? = nil) {
        self.id = name
        self.name = name
        self.summary = summary
        self.path = path
    }
}

// MARK: - Agent Fleet

public struct AgentFleetSnapshot: Sendable {
    public let generatedAt: Date
    public let agents: [AgentFleetMember]

    public init(generatedAt: Date = Date(), agents: [AgentFleetMember]) {
        self.generatedAt = generatedAt
        self.agents = agents
    }

    public var healthyCount: Int {
        agents.filter(\.isHealthy).count
    }

    public var unhealthyCount: Int {
        agents.count - healthyCount
    }
}

public struct AgentFleetMember: Identifiable, Hashable, Sendable {
    public let id: String
    public let name: String
    public let role: String
    public let status: AgentStatus
    public let lastHeartbeat: Date?
    public let assignedSymbols: [String]
    public let capabilities: [String]

    public init(
        name: String,
        role: String,
        status: AgentStatus,
        lastHeartbeat: Date? = nil,
        assignedSymbols: [String] = [],
        capabilities: [String] = []
    ) {
        self.id = name
        self.name = name
        self.role = role
        self.status = status
        self.lastHeartbeat = lastHeartbeat
        self.assignedSymbols = assignedSymbols
        self.capabilities = capabilities
    }

    public var isHealthy: Bool {
        status == .active || status == .idle
    }
}

public enum AgentStatus: String, Codable, CaseIterable, Hashable, Sendable {
    case active = "active"
    case idle = "idle"
    case failed = "failed"
    case starting = "starting"
    case stopped = "stopped"
    case unknown = "unknown"

    public var title: String {
        switch self {
        case .active: "Active"
        case .idle: "Idle"
        case .failed: "Failed"
        case .starting: "Starting"
        case .stopped: "Stopped"
        case .unknown: "Unknown"
        }
    }

    public var symbolName: String {
        switch self {
        case .active: "circle.fill"
        case .idle: "moon.fill"
        case .failed: "xmark.octagon.fill"
        case .starting: "arrow.triangle.2.circlepath"
        case .stopped: "stop.circle.fill"
        case .unknown: "questionmark.circle"
        }
    }
}

// MARK: - Backtest Campaign

public struct BacktestCampaignSnapshot: Sendable {
    public let generatedAt: Date
    public let campaigns: [BacktestCampaign]

    public init(generatedAt: Date = Date(), campaigns: [BacktestCampaign]) {
        self.generatedAt = generatedAt
        self.campaigns = campaigns
    }

    public var activeCount: Int {
        campaigns.filter { $0.status == .running }.count
    }

    public var completedCount: Int {
        campaigns.filter { $0.status == .completed }.count
    }
}

public struct BacktestCampaign: Identifiable, Hashable, Sendable {
    public let id: String
    public let name: String
    public let pluginName: String
    public let symbol: String
    public let status: CampaignStatus
    public let startedAt: Date?
    public let completedAt: Date?
    public let scenarioCount: Int
    public let passedCount: Int
    public let failedCount: Int
    public let bestSharpe: Double?
    public let worstDrawdown: Double?

    public init(
        name: String,
        pluginName: String,
        symbol: String,
        status: CampaignStatus,
        startedAt: Date? = nil,
        completedAt: Date? = nil,
        scenarioCount: Int = 0,
        passedCount: Int = 0,
        failedCount: Int = 0,
        bestSharpe: Double? = nil,
        worstDrawdown: Double? = nil
    ) {
        self.id = "\(name)::\(pluginName)::\(symbol)"
        self.name = name
        self.pluginName = pluginName
        self.symbol = symbol
        self.status = status
        self.startedAt = startedAt
        self.completedAt = completedAt
        self.scenarioCount = scenarioCount
        self.passedCount = passedCount
        self.failedCount = failedCount
        self.bestSharpe = bestSharpe
        self.worstDrawdown = worstDrawdown
    }

    public var passRate: Double {
        guard scenarioCount > 0 else { return 0 }
        return Double(passedCount) / Double(scenarioCount)
    }
}

public enum CampaignStatus: String, Codable, CaseIterable, Hashable, Sendable {
    case pending = "pending"
    case running = "running"
    case completed = "completed"
    case failed = "failed"
    case cancelled = "cancelled"

    public var title: String {
        switch self {
        case .pending: "Pending"
        case .running: "Running"
        case .completed: "Completed"
        case .failed: "Failed"
        case .cancelled: "Cancelled"
        }
    }

    public var symbolName: String {
        switch self {
        case .pending: "clock.fill"
        case .running: "arrow.triangle.2.circlepath"
        case .completed: "checkmark.circle.fill"
        case .failed: "xmark.octagon.fill"
        case .cancelled: "minus.circle.fill"
        }
    }
}

// MARK: - Log Stream

public struct LogStreamSnapshot: Sendable {
    public let generatedAt: Date
    public let streams: [LogStreamEntry]

    public init(generatedAt: Date = Date(), streams: [LogStreamEntry]) {
        self.generatedAt = generatedAt
        self.streams = streams
    }
}

public struct LogStreamEntry: Identifiable, Hashable, Sendable {
    public let id: String
    public let streamKey: String
    public let displayName: String
    public let lastEventAt: Date?
    public let eventCount: Int
    public let recentLines: [String]
    public let level: LogLevel

    public init(
        streamKey: String,
        displayName: String,
        lastEventAt: Date? = nil,
        eventCount: Int = 0,
        recentLines: [String] = [],
        level: LogLevel = .info
    ) {
        self.id = streamKey
        self.streamKey = streamKey
        self.displayName = displayName
        self.lastEventAt = lastEventAt
        self.eventCount = eventCount
        self.recentLines = recentLines
        self.level = level
    }
}

public enum LogLevel: String, Codable, CaseIterable, Hashable, Sendable {
    case debug = "debug"
    case info = "info"
    case warning = "warning"
    case error = "error"
    case critical = "critical"

    public var title: String {
        rawValue.capitalized
    }
}

// MARK: - Demo Deployment

public struct DemoDeploymentSnapshot: Sendable {
    public let generatedAt: Date
    public let deployments: [DemoDeployment]

    public init(generatedAt: Date = Date(), deployments: [DemoDeployment]) {
        self.generatedAt = generatedAt
        self.deployments = deployments
    }

    public var activeCount: Int {
        deployments.filter { $0.status == .active }.count
    }
}

public struct DemoDeployment: Identifiable, Hashable, Sendable {
    public let id: String
    public let symbol: String
    public let pluginName: String
    public let status: DemoDeploymentStatus
    public let startedAt: Date?
    public let promotionTier: String
    public let runtimeMode: String
    public let healthIndicators: [DemoHealthIndicator]

    public init(
        symbol: String,
        pluginName: String,
        status: DemoDeploymentStatus,
        startedAt: Date? = nil,
        promotionTier: String = "demo",
        runtimeMode: String = "demo",
        healthIndicators: [DemoHealthIndicator] = []
    ) {
        self.id = "\(symbol)::\(pluginName)::demo"
        self.symbol = symbol
        self.pluginName = pluginName
        self.status = status
        self.startedAt = startedAt
        self.promotionTier = promotionTier
        self.runtimeMode = runtimeMode
        self.healthIndicators = healthIndicators
    }
}

public enum DemoDeploymentStatus: String, Codable, CaseIterable, Hashable, Sendable {
    case active = "active"
    case paused = "paused"
    case stopped = "stopped"
    case error = "error"

    public var title: String {
        switch self {
        case .active: "Active"
        case .paused: "Paused"
        case .stopped: "Stopped"
        case .error: "Error"
        }
    }

    public var symbolName: String {
        switch self {
        case .active: "circle.fill"
        case .paused: "pause.circle.fill"
        case .stopped: "stop.circle.fill"
        case .error: "exclamationmark.triangle.fill"
        }
    }
}

public struct DemoHealthIndicator: Identifiable, Hashable, Sendable {
    public let id: String
    public let name: String
    public let value: String
    public let healthy: Bool

    public init(name: String, value: String, healthy: Bool) {
        self.id = name
        self.name = name
        self.value = value
        self.healthy = healthy
    }
}
