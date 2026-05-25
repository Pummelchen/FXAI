import FXExecutionContracts
import Foundation

public enum FXLiveAgentProtocolV1 {
    public static let latestVersion = "fxlive.agent.v1"
}

/// Errors raised when a live execution workload fails the promoted safety contract.
public enum FXLiveAgentError: Error, Equatable, CustomStringConvertible, Sendable {
    case unsupportedVersion(String)
    case invalidRequest(String)
    case rejectedIntent(String)

    public var description: String {
        switch self {
        case .unsupportedVersion(let version):
            return "Unsupported FXLiveAgent protocol \(version); expected \(FXLiveAgentProtocolV1.latestVersion)."
        case .invalidRequest(let reason):
            return "Invalid FXLiveAgent request: \(reason)"
        case .rejectedIntent(let reason):
            return "FXLiveAgent rejected order intent: \(reason)"
        }
    }
}

/// Immutable evidence proving a live workload was promoted from certified backtest and demo runs.
public struct FXLivePromotionEvidence: Codable, Hashable, Sendable {
    public let promotionId: String
    public let sourceBacktestRunId: String
    public let demoRunId: String
    public let lineageId: String
    public let certificationRunId: String
    public let approver: String
    public let approvedAtUTC: Int64

    public init(
        promotionId: String,
        sourceBacktestRunId: String,
        demoRunId: String,
        lineageId: String,
        certificationRunId: String,
        approver: String,
        approvedAtUTC: Int64 = Int64(Date().timeIntervalSince1970)
    ) {
        self.promotionId = promotionId.trimmingCharacters(in: .whitespacesAndNewlines)
        self.sourceBacktestRunId = sourceBacktestRunId.trimmingCharacters(in: .whitespacesAndNewlines)
        self.demoRunId = demoRunId.trimmingCharacters(in: .whitespacesAndNewlines)
        self.lineageId = lineageId.trimmingCharacters(in: .whitespacesAndNewlines)
        self.certificationRunId = certificationRunId.trimmingCharacters(in: .whitespacesAndNewlines)
        self.approver = approver.trimmingCharacters(in: .whitespacesAndNewlines)
        self.approvedAtUTC = approvedAtUTC
    }

    /// Verifies that all promotion identifiers are present and the approval timestamp is usable.
    public func validate() throws {
        try require(promotionId, "promotion_id")
        try require(sourceBacktestRunId, "source_backtest_run_id")
        try require(demoRunId, "demo_run_id")
        try require(lineageId, "lineage_id")
        try require(certificationRunId, "certification_run_id")
        try require(approver, "approver")
        guard approvedAtUTC > 0 else {
            throw FXLiveAgentError.invalidRequest("approved_at_utc must be positive")
        }
    }
}

/// Versioned workload sent from FXBacktest or a release controller into the live execution boundary.
public struct FXLiveAgentWorkloadRequest: Codable, Hashable, Sendable {
    public let apiVersion: String
    public let requestId: String
    public let pluginId: String
    public let acceleratorId: String
    public let parameterSetId: String
    public let account: FXExecutionAccountScope
    public let riskLimits: FXExecutionRiskLimits
    public let killSwitch: FXExecutionKillSwitchState
    public let promotionEvidence: FXLivePromotionEvidence
    public let orderIntents: [FXExecutionOrderIntent]
    public let issuedAtUTC: Int64

    public init(
        apiVersion: String = FXLiveAgentProtocolV1.latestVersion,
        requestId: String,
        pluginId: String,
        acceleratorId: String,
        parameterSetId: String,
        account: FXExecutionAccountScope,
        riskLimits: FXExecutionRiskLimits,
        killSwitch: FXExecutionKillSwitchState,
        promotionEvidence: FXLivePromotionEvidence,
        orderIntents: [FXExecutionOrderIntent],
        issuedAtUTC: Int64 = Int64(Date().timeIntervalSince1970)
    ) {
        self.apiVersion = apiVersion
        self.requestId = requestId.trimmingCharacters(in: .whitespacesAndNewlines)
        self.pluginId = pluginId.trimmingCharacters(in: .whitespacesAndNewlines)
        self.acceleratorId = acceleratorId.trimmingCharacters(in: .whitespacesAndNewlines)
        self.parameterSetId = parameterSetId.trimmingCharacters(in: .whitespacesAndNewlines)
        self.account = account
        self.riskLimits = riskLimits
        self.killSwitch = killSwitch
        self.promotionEvidence = promotionEvidence
        self.orderIntents = orderIntents
        self.issuedAtUTC = issuedAtUTC
    }

    /// Verifies latest API version, live account scope, promotion evidence, lineage, and shared execution safety gates.
    public func validate() throws {
        guard apiVersion == FXLiveAgentProtocolV1.latestVersion else {
            throw FXLiveAgentError.unsupportedVersion(apiVersion)
        }
        try require(requestId, "request_id")
        try require(pluginId, "plugin_id")
        try require(acceleratorId, "accelerator_id")
        try require(parameterSetId, "parameter_set_id")
        guard issuedAtUTC > 0 else {
            throw FXLiveAgentError.invalidRequest("issued_at_utc must be positive")
        }
        guard account.environment == .live else {
            throw FXLiveAgentError.invalidRequest("live workloads require a live account scope")
        }
        try promotionEvidence.validate()
        guard !orderIntents.isEmpty else {
            throw FXLiveAgentError.invalidRequest("order_intents must not be empty")
        }
        for intent in orderIntents {
            guard intent.lineageId == promotionEvidence.lineageId else {
                throw FXLiveAgentError.invalidRequest("intent lineage_id must match promotion lineage_id")
            }
            guard intent.promotionId == promotionEvidence.promotionId else {
                throw FXLiveAgentError.invalidRequest("intent promotion_id must match promotion evidence")
            }
            do {
                try FXExecutionSafetyGate.validateOrderIntent(
                    intent,
                    account: account,
                    limits: riskLimits,
                    killSwitch: killSwitch
                )
            } catch {
                throw FXLiveAgentError.rejectedIntent(String(describing: error))
            }
        }
    }
}

/// Human-release execution plan produced after a promoted live workload passes validation.
public struct FXLiveAgentPlan: Codable, Hashable, Sendable {
    public let apiVersion: String
    public let requestId: String
    public let promotionId: String
    public let acceptedIntentCount: Int
    public let symbols: [String]
    public let lineageId: String
    public let requiresHumanRelease: Bool
    public let plannedAtUTC: Int64

    public init(
        apiVersion: String = FXLiveAgentProtocolV1.latestVersion,
        requestId: String,
        promotionId: String,
        acceptedIntentCount: Int,
        symbols: [String],
        lineageId: String,
        requiresHumanRelease: Bool,
        plannedAtUTC: Int64 = Int64(Date().timeIntervalSince1970)
    ) {
        self.apiVersion = apiVersion
        self.requestId = requestId
        self.promotionId = promotionId
        self.acceptedIntentCount = max(0, acceptedIntentCount)
        let normalizedSymbols = symbols
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines).uppercased() }
            .filter { !$0.isEmpty }
        self.symbols = Array(Set(normalizedSymbols)).sorted()
        self.lineageId = lineageId
        self.requiresHumanRelease = requiresHumanRelease
        self.plannedAtUTC = plannedAtUTC
    }
}

public enum FXLiveAgentRuntime {
    /// Converts a validated promoted workload into a live plan without touching broker or database APIs.
    public static func plan(
        _ request: FXLiveAgentWorkloadRequest,
        requiresHumanRelease: Bool = true
    ) throws -> FXLiveAgentPlan {
        try request.validate()
        return FXLiveAgentPlan(
            requestId: request.requestId,
            promotionId: request.promotionEvidence.promotionId,
            acceptedIntentCount: request.orderIntents.count,
            symbols: request.orderIntents.map(\.symbol),
            lineageId: request.promotionEvidence.lineageId,
            requiresHumanRelease: requiresHumanRelease
        )
    }
}

private func require(_ value: String, _ field: String) throws {
    guard !value.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
        throw FXLiveAgentError.invalidRequest("\(field) must not be empty")
    }
}
