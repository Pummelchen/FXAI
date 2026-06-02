import FXExecutionContracts
import Foundation

public enum FXDemoAgentProtocolV1 {
    public static let latestVersion = "fxdemo.agent.v1"
}

/// Errors raised when a demo execution workload fails the versioned safety contract.
public enum FXDemoAgentError: Error, Equatable, CustomStringConvertible, Sendable {
    case unsupportedVersion(String)
    case invalidRequest(String)
    case rejectedIntent(String)

    public var description: String {
        switch self {
        case .unsupportedVersion(let version):
            return "Unsupported FXDemoAgent protocol \(version); expected \(FXDemoAgentProtocolV1.latestVersion)."
        case .invalidRequest(let reason):
            return "Invalid FXDemoAgent request: \(reason)"
        case .rejectedIntent(let reason):
            return "FXDemoAgent rejected order intent: \(reason)"
        }
    }
}

/// Versioned workload sent from FXBacktest into the demo execution boundary.
public struct FXDemoAgentWorkloadRequest: Codable, Hashable, Sendable {
    public let apiVersion: String
    public let requestId: String
    public let sourceBacktestRunId: String
    public let lineageId: String
    public let pluginId: String
    public let acceleratorId: String
    public let parameterSetId: String
    public let account: FXExecutionAccountScope
    public let riskLimits: FXExecutionRiskLimits
    public let killSwitch: FXExecutionKillSwitchState
    public let orderIntents: [FXExecutionOrderIntent]
    public let issuedAtUTC: Int64

    public init(
        apiVersion: String = FXDemoAgentProtocolV1.latestVersion,
        requestId: String,
        sourceBacktestRunId: String,
        lineageId: String,
        pluginId: String,
        acceleratorId: String,
        parameterSetId: String,
        account: FXExecutionAccountScope,
        riskLimits: FXExecutionRiskLimits,
        killSwitch: FXExecutionKillSwitchState,
        orderIntents: [FXExecutionOrderIntent],
        issuedAtUTC: Int64 = Int64(Date().timeIntervalSince1970)
    ) {
        self.apiVersion = apiVersion
        self.requestId = requestId.trimmingCharacters(in: .whitespacesAndNewlines)
        self.sourceBacktestRunId = sourceBacktestRunId.trimmingCharacters(in: .whitespacesAndNewlines)
        self.lineageId = lineageId.trimmingCharacters(in: .whitespacesAndNewlines)
        self.pluginId = pluginId.trimmingCharacters(in: .whitespacesAndNewlines)
        self.acceleratorId = acceleratorId.trimmingCharacters(in: .whitespacesAndNewlines)
        self.parameterSetId = parameterSetId.trimmingCharacters(in: .whitespacesAndNewlines)
        self.account = account
        self.riskLimits = riskLimits
        self.killSwitch = killSwitch
        self.orderIntents = orderIntents
        self.issuedAtUTC = issuedAtUTC
    }

    /// Verifies that the workload is current-version, demo-scoped, lineage-bound, and allowed by shared execution safety gates.
    public func validate() throws {
        guard apiVersion == FXDemoAgentProtocolV1.latestVersion else {
            throw FXDemoAgentError.unsupportedVersion(apiVersion)
        }
        try FXExecutionValidation.requireNonEmpty(requestId, "request_id", error: FXDemoAgentError.invalidRequest)
        try FXExecutionValidation.requireNonEmpty(sourceBacktestRunId, "source_backtest_run_id", error: FXDemoAgentError.invalidRequest)
        try FXExecutionValidation.requireNonEmpty(lineageId, "lineage_id", error: FXDemoAgentError.invalidRequest)
        try FXExecutionValidation.requireNonEmpty(pluginId, "plugin_id", error: FXDemoAgentError.invalidRequest)
        try FXExecutionValidation.requireNonEmpty(acceleratorId, "accelerator_id", error: FXDemoAgentError.invalidRequest)
        try FXExecutionValidation.requireNonEmpty(parameterSetId, "parameter_set_id", error: FXDemoAgentError.invalidRequest)
        guard issuedAtUTC > 0 else {
            throw FXDemoAgentError.invalidRequest("issued_at_utc must be positive")
        }
        guard account.environment == .demo else {
            throw FXDemoAgentError.invalidRequest("demo workloads require a demo account scope")
        }
        guard !orderIntents.isEmpty else {
            throw FXDemoAgentError.invalidRequest("order_intents must not be empty")
        }
        for intent in orderIntents {
            guard intent.lineageId == lineageId else {
                throw FXDemoAgentError.invalidRequest("intent lineage_id must match workload lineage_id")
            }
            do {
                try FXExecutionSafetyGate.validateOrderIntent(
                    intent,
                    account: account,
                    limits: riskLimits,
                    killSwitch: killSwitch
                )
            } catch {
                throw FXDemoAgentError.rejectedIntent(String(describing: error))
            }
        }
    }
}

/// Dry-run-first execution plan produced after a demo workload passes validation.
public struct FXDemoAgentPlan: Codable, Hashable, Sendable {
    public let apiVersion: String
    public let requestId: String
    public let acceptedIntentCount: Int
    public let symbols: [String]
    public let lineageId: String
    public let dryRunOnly: Bool
    public let plannedAtUTC: Int64

    public init(
        apiVersion: String = FXDemoAgentProtocolV1.latestVersion,
        requestId: String,
        acceptedIntentCount: Int,
        symbols: [String],
        lineageId: String,
        dryRunOnly: Bool,
        plannedAtUTC: Int64 = Int64(Date().timeIntervalSince1970)
    ) {
        self.apiVersion = apiVersion
        self.requestId = requestId
        self.acceptedIntentCount = max(0, acceptedIntentCount)
        let normalizedSymbols = symbols
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines).uppercased() }
            .filter { !$0.isEmpty }
        self.symbols = Array(Set(normalizedSymbols)).sorted()
        self.lineageId = lineageId
        self.dryRunOnly = dryRunOnly
        self.plannedAtUTC = plannedAtUTC
    }
}

public enum FXDemoAgentRuntime {
    /// Converts a validated workload into an executable demo plan without touching broker or database APIs.
    public static func plan(_ request: FXDemoAgentWorkloadRequest, dryRunOnly: Bool = true) throws -> FXDemoAgentPlan {
        try request.validate()
        return FXDemoAgentPlan(
            requestId: request.requestId,
            acceptedIntentCount: request.orderIntents.count,
            symbols: request.orderIntents.map(\.symbol),
            lineageId: request.lineageId,
            dryRunOnly: dryRunOnly
        )
    }
}
