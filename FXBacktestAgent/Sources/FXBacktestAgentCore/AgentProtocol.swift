import Foundation

public enum FXBacktestAgentProtocolV1 {
    public static let latestVersion = "fxbacktest.agent.tcp.v1"
}

public enum FXBacktestAgentMessageKind: String, Codable, CaseIterable, Sendable {
    case hello
    case capabilities
    case certificationStatus
    case leaseRequest
    case leaseGrant
    case leaseAck
    case batchProgress
    case batchResult
    case batchFailure
    case heartbeat
    case shutdown
}

public struct FXBacktestAgentEnvelope<Payload: Codable & Sendable>: Codable, Sendable {
    public let apiVersion: String
    public let kind: FXBacktestAgentMessageKind
    public let agentId: String
    public let sequence: UInt64
    public let sentAtUTC: Int64
    public let payload: Payload

    public init(
        apiVersion: String = FXBacktestAgentProtocolV1.latestVersion,
        kind: FXBacktestAgentMessageKind,
        agentId: String,
        sequence: UInt64,
        sentAtUTC: Int64 = Int64(Date().timeIntervalSince1970),
        payload: Payload
    ) {
        self.apiVersion = apiVersion
        self.kind = kind
        self.agentId = agentId
        self.sequence = sequence
        self.sentAtUTC = sentAtUTC
        self.payload = payload
    }

    public func validate(expectedKind: FXBacktestAgentMessageKind) throws {
        guard apiVersion == FXBacktestAgentProtocolV1.latestVersion else {
            throw FXBacktestAgentProtocolError.unsupportedVersion(apiVersion)
        }
        guard kind == expectedKind else {
            throw FXBacktestAgentProtocolError.invalidMessage("expected \(expectedKind.rawValue), got \(kind.rawValue)")
        }
        guard !agentId.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw FXBacktestAgentProtocolError.invalidMessage("agent_id must not be empty")
        }
        guard sentAtUTC > 0 else {
            throw FXBacktestAgentProtocolError.invalidMessage("sent_at_utc must be positive")
        }
    }
}

public enum FXBacktestAgentProtocolError: Error, Equatable, CustomStringConvertible, Sendable {
    case unsupportedVersion(String)
    case invalidMessage(String)

    public var description: String {
        switch self {
        case .unsupportedVersion(let version):
            return "Unsupported FXBacktestAgent protocol \(version); expected \(FXBacktestAgentProtocolV1.latestVersion)."
        case .invalidMessage(let message):
            return "Invalid FXBacktestAgent message: \(message)"
        }
    }
}

public struct FXBacktestAgentCapabilities: Codable, Equatable, Sendable {
    public let hostName: String
    public let cpuCoreCount: Int
    public let memoryBytes: UInt64
    public let hardwareClass: String
    public let metalDeviceName: String
    public let pyTorchMPSAvailable: Bool
    public let tensorFlowMetalAvailable: Bool
    public let foundationNLPAvailable: Bool
    public let supportedPluginIds: [String]
    public let supportedAcceleratorBackends: [String]
    public let localCertificationRunId: String?
    public let currentLoad: Double

    public init(
        hostName: String,
        cpuCoreCount: Int,
        memoryBytes: UInt64,
        hardwareClass: String,
        metalDeviceName: String,
        pyTorchMPSAvailable: Bool,
        tensorFlowMetalAvailable: Bool,
        foundationNLPAvailable: Bool,
        supportedPluginIds: [String],
        supportedAcceleratorBackends: [String],
        localCertificationRunId: String?,
        currentLoad: Double
    ) {
        self.hostName = hostName
        self.cpuCoreCount = max(1, cpuCoreCount)
        self.memoryBytes = memoryBytes
        self.hardwareClass = hardwareClass
        self.metalDeviceName = metalDeviceName
        self.pyTorchMPSAvailable = pyTorchMPSAvailable
        self.tensorFlowMetalAvailable = tensorFlowMetalAvailable
        self.foundationNLPAvailable = foundationNLPAvailable
        self.supportedPluginIds = supportedPluginIds.sorted()
        self.supportedAcceleratorBackends = supportedAcceleratorBackends.sorted()
        self.localCertificationRunId = localCertificationRunId
        self.currentLoad = min(max(currentLoad, 0), 1)
    }
}

public struct FXBacktestAgentCertificationStatus: Codable, Equatable, Sendable {
    public let passed: Bool
    public let certificationRunId: String?
    public let sineTestPassed: Bool
    public let evidenceHash: String
    public let checkedAtUTC: Int64

    public init(
        passed: Bool,
        certificationRunId: String?,
        sineTestPassed: Bool,
        evidenceHash: String,
        checkedAtUTC: Int64 = Int64(Date().timeIntervalSince1970)
    ) {
        self.passed = passed
        self.certificationRunId = certificationRunId
        self.sineTestPassed = sineTestPassed
        self.evidenceHash = evidenceHash
        self.checkedAtUTC = checkedAtUTC
    }
}

public struct FXBacktestAgentLeaseRequest: Codable, Equatable, Sendable {
    public let maxBatches: Int
    public let maxEstimatedSeconds: Double
    public let capabilities: FXBacktestAgentCapabilities
    public let certificationStatus: FXBacktestAgentCertificationStatus

    public init(
        maxBatches: Int,
        maxEstimatedSeconds: Double,
        capabilities: FXBacktestAgentCapabilities,
        certificationStatus: FXBacktestAgentCertificationStatus
    ) {
        self.maxBatches = max(1, maxBatches)
        self.maxEstimatedSeconds = max(1, maxEstimatedSeconds)
        self.capabilities = capabilities
        self.certificationStatus = certificationStatus
    }

    public func validateForWork() throws {
        guard certificationStatus.passed, certificationStatus.sineTestPassed else {
            throw FXBacktestAgentProtocolError.invalidMessage("agent cannot lease work before local certification and SineTest pass")
        }
    }
}

public struct FXBacktestAgentLeaseGrant: Codable, Equatable, Sendable {
    public let leaseId: String
    public let runId: String
    public let batchId: String
    public let pluginId: String
    public let acceleratorBackend: String
    public let lineageId: String
    public let expiresAtUTC: Int64
    public let payloadJSON: String

    public init(
        leaseId: String,
        runId: String,
        batchId: String,
        pluginId: String,
        acceleratorBackend: String,
        lineageId: String,
        expiresAtUTC: Int64,
        payloadJSON: String
    ) {
        self.leaseId = leaseId
        self.runId = runId
        self.batchId = batchId
        self.pluginId = pluginId
        self.acceleratorBackend = acceleratorBackend
        self.lineageId = lineageId
        self.expiresAtUTC = expiresAtUTC
        self.payloadJSON = payloadJSON
    }
}

public struct FXBacktestAgentBatchResult: Codable, Equatable, Sendable {
    public let leaseId: String
    public let runId: String
    public let batchId: String
    public let lineageId: String
    public let certificationRunId: String
    public let resultHash: String
    public let resultsJSON: String
    public let completedAtUTC: Int64

    public init(
        leaseId: String,
        runId: String,
        batchId: String,
        lineageId: String,
        certificationRunId: String,
        resultHash: String,
        resultsJSON: String,
        completedAtUTC: Int64 = Int64(Date().timeIntervalSince1970)
    ) {
        self.leaseId = leaseId
        self.runId = runId
        self.batchId = batchId
        self.lineageId = lineageId
        self.certificationRunId = certificationRunId
        self.resultHash = resultHash
        self.resultsJSON = resultsJSON
        self.completedAtUTC = completedAtUTC
    }
}

public struct FXBacktestAgentHeartbeat: Codable, Equatable, Sendable {
    public let activeLeaseIds: [String]
    public let currentLoad: Double
    public let freeMemoryBytes: UInt64
    public let lastError: String?

    public init(activeLeaseIds: [String], currentLoad: Double, freeMemoryBytes: UInt64, lastError: String? = nil) {
        self.activeLeaseIds = activeLeaseIds.sorted()
        self.currentLoad = min(max(currentLoad, 0), 1)
        self.freeMemoryBytes = freeMemoryBytes
        self.lastError = lastError
    }
}
