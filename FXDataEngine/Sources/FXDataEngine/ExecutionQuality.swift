import Foundation

public enum ExecutionQualityConstants {
    public static let maxReasons = 12
    public static let defaultFreshnessMaxSeconds: Int64 = 180
    public static let runtimeDirectory = "FXAI/Runtime"
}

public struct ExecutionQualityPairState: Codable, Hashable, Sendable {
    public var ready: Bool
    public var available: Bool
    public var stale: Bool
    public var fallbackUsed: Bool
    public var memoryStale: Bool
    public var dataStale: Bool
    public var supportUsable: Bool
    public var generatedAt: Int64
    public var method: String
    public var sessionLabel: String
    public var regimeLabel: String
    public var selectedTierKind: String
    public var selectedTierKey: String
    public var selectedSupport: Int
    public var selectedQuality: Double
    public var spreadNowPoints: Double
    public var spreadExpectedPoints: Double
    public var spreadWideningRisk: Double
    public var expectedSlippagePoints: Double
    public var slippageRisk: Double
    public var fillQualityScore: Double
    public var latencySensitivityScore: Double
    public var liquidityFragilityScore: Double
    public var executionQualityScore: Double
    public var allowedDeviationPoints: Double
    public var cautionLotScale: Double
    public var cautionEnterProbabilityBuffer: Double
    public var executionState: String
    public var reasons: [String]

    public init(
        ready: Bool = false,
        available: Bool = false,
        stale: Bool = true,
        fallbackUsed: Bool = false,
        memoryStale: Bool = true,
        dataStale: Bool = true,
        supportUsable: Bool = false,
        generatedAt: Int64 = 0,
        method: String = "SCORECARD_V1",
        sessionLabel: String = "UNKNOWN",
        regimeLabel: String = "UNKNOWN",
        selectedTierKind: String = "GLOBAL",
        selectedTierKey: String = "GLOBAL|*|*|*",
        selectedSupport: Int = 0,
        selectedQuality: Double = 0.0,
        spreadNowPoints: Double = 0.0,
        spreadExpectedPoints: Double = 0.0,
        spreadWideningRisk: Double = 0.0,
        expectedSlippagePoints: Double = 0.0,
        slippageRisk: Double = 0.0,
        fillQualityScore: Double = 0.0,
        latencySensitivityScore: Double = 0.0,
        liquidityFragilityScore: Double = 0.0,
        executionQualityScore: Double = 0.0,
        allowedDeviationPoints: Double = 0.0,
        cautionLotScale: Double = 1.0,
        cautionEnterProbabilityBuffer: Double = 0.0,
        executionState: String = "UNKNOWN",
        reasons: [String] = []
    ) {
        self.ready = ready
        self.available = available
        self.stale = stale
        self.fallbackUsed = fallbackUsed
        self.memoryStale = memoryStale
        self.dataStale = dataStale
        self.supportUsable = supportUsable
        self.generatedAt = max(0, generatedAt)
        self.method = method
        self.sessionLabel = sessionLabel
        self.regimeLabel = regimeLabel
        self.selectedTierKind = selectedTierKind
        self.selectedTierKey = selectedTierKey
        self.selectedSupport = selectedSupport
        self.selectedQuality = selectedQuality
        self.spreadNowPoints = spreadNowPoints
        self.spreadExpectedPoints = spreadExpectedPoints
        self.spreadWideningRisk = spreadWideningRisk
        self.expectedSlippagePoints = expectedSlippagePoints
        self.slippageRisk = slippageRisk
        self.fillQualityScore = fillQualityScore
        self.latencySensitivityScore = latencySensitivityScore
        self.liquidityFragilityScore = liquidityFragilityScore
        self.executionQualityScore = executionQualityScore
        self.allowedDeviationPoints = allowedDeviationPoints
        self.cautionLotScale = cautionLotScale
        self.cautionEnterProbabilityBuffer = cautionEnterProbabilityBuffer
        self.executionState = executionState
        self.reasons = Self.uniqueReasons(reasons)
    }

    public static var reset: ExecutionQualityPairState {
        ExecutionQualityPairState()
    }

    public var reasonCount: Int {
        reasons.count
    }

    public var reasonsCSV: String {
        reasons.filter { !$0.isEmpty }.joined(separator: "; ")
    }

    public var systemHealthState: SystemHealthExecutionQualityState {
        SystemHealthExecutionQualityState(ready: ready, stale: stale, dataStale: dataStale)
    }

    public mutating func appendReason(_ reason: String) {
        let value = reason.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !value.isEmpty,
              !reasons.contains(value),
              reasons.count < ExecutionQualityConstants.maxReasons else {
            return
        }
        reasons.append(value)
    }

    private static func uniqueReasons(_ input: [String]) -> [String] {
        var output: [String] = []
        output.reserveCapacity(min(input.count, ExecutionQualityConstants.maxReasons))
        for raw in input {
            let value = raw.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !value.isEmpty,
                  !output.contains(value),
                  output.count < ExecutionQualityConstants.maxReasons else {
                continue
            }
            output.append(value)
        }
        return output
    }
}

public enum ExecutionQualityTools {
    public static func runtimeStatePath(symbol: String) -> String {
        "\(ExecutionQualityConstants.runtimeDirectory)/fxai_execution_quality_\(ControlPlanePaths.safeToken(symbol)).tsv"
    }

    public static func readPairState(
        symbol _: String,
        stateTSV: String?,
        nowUTC: Int64 = 0,
        freshnessMaxSeconds: Int64 = ExecutionQualityConstants.defaultFreshnessMaxSeconds
    ) -> ExecutionQualityPairState? {
        guard let stateTSV else { return nil }
        let state = normalizedAvailableState(
            parseState(tsv: stateTSV),
            nowUTC: nowUTC,
            freshnessMaxSeconds: freshnessMaxSeconds
        )
        return state.available ? state : nil
    }

    public static func parseState(tsv: String) -> ExecutionQualityPairState {
        var state = ExecutionQualityPairState.reset
        for line in tsv.components(separatedBy: .newlines) where !line.isEmpty {
            let parts = line.split(separator: "\t", omittingEmptySubsequences: false)
            guard parts.count >= 2 else { continue }
            let key = String(parts[0])
            let value = String(parts[1])
            state.available = true
            state.ready = true

            switch key {
            case "generated_at":
                state.generatedAt = Int64(value) ?? 0
            case "method":
                state.method = value
            case "session_label":
                state.sessionLabel = value
            case "regime_label":
                state.regimeLabel = value
            case "selected_tier_kind":
                state.selectedTierKind = value
            case "selected_tier_key":
                state.selectedTierKey = value
            case "selected_support":
                state.selectedSupport = Int(value) ?? 0
            case "selected_quality":
                state.selectedQuality = Double(value) ?? 0.0
            case "fallback_used":
                state.fallbackUsed = (Int(value) ?? 0) != 0
            case "memory_stale":
                state.memoryStale = (Int(value) ?? 0) != 0
            case "data_stale":
                state.dataStale = (Int(value) ?? 0) != 0
            case "support_usable":
                state.supportUsable = (Int(value) ?? 0) != 0
            case "spread_now_points":
                state.spreadNowPoints = Double(value) ?? 0.0
            case "spread_expected_points":
                state.spreadExpectedPoints = Double(value) ?? 0.0
            case "spread_widening_risk":
                state.spreadWideningRisk = Double(value) ?? 0.0
            case "expected_slippage_points":
                state.expectedSlippagePoints = Double(value) ?? 0.0
            case "slippage_risk":
                state.slippageRisk = Double(value) ?? 0.0
            case "fill_quality_score":
                state.fillQualityScore = Double(value) ?? 0.0
            case "latency_sensitivity_score":
                state.latencySensitivityScore = Double(value) ?? 0.0
            case "liquidity_fragility_score":
                state.liquidityFragilityScore = Double(value) ?? 0.0
            case "execution_quality_score":
                state.executionQualityScore = Double(value) ?? 0.0
            case "allowed_deviation_points":
                state.allowedDeviationPoints = Double(value) ?? 0.0
            case "caution_lot_scale":
                state.cautionLotScale = Double(value) ?? 0.0
            case "caution_enter_prob_buffer":
                state.cautionEnterProbabilityBuffer = Double(value) ?? 0.0
            case "execution_state":
                state.executionState = value
            case "reasons_csv":
                for reason in value.split(separator: ";", omittingEmptySubsequences: false) {
                    state.appendReason(String(reason))
                }
            default:
                break
            }
        }
        return state
    }

    private static func normalizedAvailableState(
        _ state: ExecutionQualityPairState,
        nowUTC: Int64,
        freshnessMaxSeconds: Int64
    ) -> ExecutionQualityPairState {
        var output = state
        if output.available {
            if nowUTC > 0, output.generatedAt > 0 {
                output.stale = nowUTC - output.generatedAt > max(freshnessMaxSeconds, 30)
            } else {
                output.stale = true
            }
        }
        return output
    }
}
