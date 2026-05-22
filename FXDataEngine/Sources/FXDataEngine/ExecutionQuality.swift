import Foundation

public enum ExecutionQualityConstants {
    public static let maxReasons = 12
    public static let maxBuckets = 5
    public static let maxTiers = 128
    public static let defaultFreshnessMaxSeconds: Int64 = 180
    public static let runtimeDirectory = "FXAI/Runtime"
}

public struct ExecutionQualityConfig: Codable, Hashable, Sendable {
    public var ready: Bool
    public var enabled: Bool
    public var blockOnUnknown: Bool
    public var allowBlockState: Bool
    public var supportSoftFloor: Int
    public var supportHardFloor: Int
    public var memoryStaleAfterHours: Int
    public var thresholdNormalMin: Double
    public var thresholdCautionMin: Double
    public var thresholdStressedMin: Double
    public var lotScaleNormal: Double
    public var lotScaleCaution: Double
    public var lotScaleStressed: Double
    public var lotScaleBlocked: Double
    public var enterProbabilityBufferNormal: Double
    public var enterProbabilityBufferCaution: Double
    public var enterProbabilityBufferStressed: Double
    public var enterProbabilityBufferBlocked: Double
    public var capExpectedPriceCostMultiplier: Double
    public var capExpectedSlippagePoints: Double
    public var capAllowedDeviationPointsMin: Double
    public var capAllowedDeviationPointsMax: Double
    public var weightPriceCostZScore: Double
    public var weightNewsRisk: Double
    public var weightRatesRisk: Double
    public var weightMicroLiquidity: Double
    public var weightMicroHostile: Double
    public var weightVolatilityBurst: Double
    public var weightTickRateBurst: Double
    public var weightSessionThinness: Double
    public var weightBrokerReject: Double
    public var weightBrokerPartial: Double
    public var weightBrokerLatency: Double
    public var weightBrokerEventBurst: Double
    public var weightStaleContext: Double
    public var weightSupportShortfall: Double
    public var bucketCount: Int
    public var bucketHierarchy: [String]

    public init(
        ready: Bool = true,
        enabled: Bool = true,
        blockOnUnknown: Bool = true,
        allowBlockState: Bool = true,
        supportSoftFloor: Int = 64,
        supportHardFloor: Int = 16,
        memoryStaleAfterHours: Int = 168,
        thresholdNormalMin: Double = 0.72,
        thresholdCautionMin: Double = 0.54,
        thresholdStressedMin: Double = 0.36,
        lotScaleNormal: Double = 1.00,
        lotScaleCaution: Double = 0.82,
        lotScaleStressed: Double = 0.58,
        lotScaleBlocked: Double = 0.00,
        enterProbabilityBufferNormal: Double = 0.00,
        enterProbabilityBufferCaution: Double = 0.04,
        enterProbabilityBufferStressed: Double = 0.08,
        enterProbabilityBufferBlocked: Double = 1.00,
        capExpectedPriceCostMultiplier: Double = 4.50,
        capExpectedSlippagePoints: Double = 18.0,
        capAllowedDeviationPointsMin: Double = 2.0,
        capAllowedDeviationPointsMax: Double = 25.0,
        weightPriceCostZScore: Double = 0.22,
        weightNewsRisk: Double = 0.18,
        weightRatesRisk: Double = 0.10,
        weightMicroLiquidity: Double = 0.18,
        weightMicroHostile: Double = 0.18,
        weightVolatilityBurst: Double = 0.14,
        weightTickRateBurst: Double = 0.12,
        weightSessionThinness: Double = 0.10,
        weightBrokerReject: Double = 0.16,
        weightBrokerPartial: Double = 0.14,
        weightBrokerLatency: Double = 0.14,
        weightBrokerEventBurst: Double = 0.12,
        weightStaleContext: Double = 0.10,
        weightSupportShortfall: Double = 0.08,
        bucketCount: Int = 5,
        bucketHierarchy: [String] = [
            "PAIR_SESSION_REGIME",
            "PAIR_REGIME",
            "SESSION_REGIME",
            "REGIME",
            "GLOBAL"
        ]
    ) {
        self.ready = ready
        self.enabled = enabled
        self.blockOnUnknown = blockOnUnknown
        self.allowBlockState = allowBlockState
        self.supportSoftFloor = max(0, supportSoftFloor)
        self.supportHardFloor = max(0, supportHardFloor)
        self.memoryStaleAfterHours = max(0, memoryStaleAfterHours)
        self.thresholdNormalMin = fxClamp(thresholdNormalMin, 0.0, 1.0)
        self.thresholdCautionMin = fxClamp(thresholdCautionMin, 0.0, 1.0)
        self.thresholdStressedMin = fxClamp(thresholdStressedMin, 0.0, 1.0)
        self.lotScaleNormal = max(0.0, fxSafeFinite(lotScaleNormal))
        self.lotScaleCaution = max(0.0, fxSafeFinite(lotScaleCaution))
        self.lotScaleStressed = max(0.0, fxSafeFinite(lotScaleStressed))
        self.lotScaleBlocked = max(0.0, fxSafeFinite(lotScaleBlocked))
        self.enterProbabilityBufferNormal = max(0.0, fxSafeFinite(enterProbabilityBufferNormal))
        self.enterProbabilityBufferCaution = max(0.0, fxSafeFinite(enterProbabilityBufferCaution))
        self.enterProbabilityBufferStressed = max(0.0, fxSafeFinite(enterProbabilityBufferStressed))
        self.enterProbabilityBufferBlocked = max(0.0, fxSafeFinite(enterProbabilityBufferBlocked))
        self.capExpectedPriceCostMultiplier = max(0.0, fxSafeFinite(capExpectedPriceCostMultiplier))
        self.capExpectedSlippagePoints = max(0.0, fxSafeFinite(capExpectedSlippagePoints))
        self.capAllowedDeviationPointsMin = max(0.0, fxSafeFinite(capAllowedDeviationPointsMin))
        self.capAllowedDeviationPointsMax = max(
            self.capAllowedDeviationPointsMin,
            fxSafeFinite(capAllowedDeviationPointsMax, fallback: self.capAllowedDeviationPointsMin)
        )
        self.weightPriceCostZScore = fxSafeFinite(weightPriceCostZScore)
        self.weightNewsRisk = fxSafeFinite(weightNewsRisk)
        self.weightRatesRisk = fxSafeFinite(weightRatesRisk)
        self.weightMicroLiquidity = fxSafeFinite(weightMicroLiquidity)
        self.weightMicroHostile = fxSafeFinite(weightMicroHostile)
        self.weightVolatilityBurst = fxSafeFinite(weightVolatilityBurst)
        self.weightTickRateBurst = fxSafeFinite(weightTickRateBurst)
        self.weightSessionThinness = fxSafeFinite(weightSessionThinness)
        self.weightBrokerReject = fxSafeFinite(weightBrokerReject)
        self.weightBrokerPartial = fxSafeFinite(weightBrokerPartial)
        self.weightBrokerLatency = fxSafeFinite(weightBrokerLatency)
        self.weightBrokerEventBurst = fxSafeFinite(weightBrokerEventBurst)
        self.weightStaleContext = fxSafeFinite(weightStaleContext)
        self.weightSupportShortfall = fxSafeFinite(weightSupportShortfall)
        self.bucketCount = Int(fxClamp(Double(bucketCount), 0.0, Double(ExecutionQualityConstants.maxBuckets)))
        self.bucketHierarchy = Self.normalizedBuckets(bucketHierarchy)
    }

    public var effectiveBucketHierarchy: [String] {
        Array(bucketHierarchy.prefix(bucketCount))
    }

    private static func normalizedBuckets(_ values: [String]) -> [String] {
        var output = values
            .prefix(ExecutionQualityConstants.maxBuckets)
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines).uppercased() }
        if output.count < ExecutionQualityConstants.maxBuckets {
            output.append(contentsOf: Array(
                repeating: "",
                count: ExecutionQualityConstants.maxBuckets - output.count
            ))
        }
        return Array(output)
    }
}

public struct ExecutionQualityTier: Codable, Hashable, Sendable {
    public var ready: Bool
    public var kind: String
    public var symbol: String
    public var session: String
    public var regime: String
    public var support: Int
    public var quality: Double
    public var priceCostMultiplier: Double
    public var slippageMultiplier: Double
    public var fillQualityBias: Double
    public var latencyMultiplier: Double
    public var fragilityMultiplier: Double
    public var deviationMultiplier: Double

    public init(
        ready: Bool = false,
        kind: String = "GLOBAL",
        symbol: String = "*",
        session: String = "*",
        regime: String = "*",
        support: Int = 0,
        quality: Double = 0.34,
        priceCostMultiplier: Double = 1.08,
        slippageMultiplier: Double = 1.12,
        fillQualityBias: Double = -0.06,
        latencyMultiplier: Double = 1.08,
        fragilityMultiplier: Double = 1.10,
        deviationMultiplier: Double = 1.06
    ) {
        self.ready = ready
        self.kind = Self.normalizedToken(kind, fallback: "GLOBAL")
        self.symbol = Self.normalizedToken(symbol, fallback: "*")
        self.session = Self.normalizedToken(session, fallback: "*")
        self.regime = Self.normalizedToken(regime, fallback: "*")
        self.support = max(0, support)
        self.quality = fxSafeFinite(quality)
        self.priceCostMultiplier = fxSafeFinite(priceCostMultiplier)
        self.slippageMultiplier = fxSafeFinite(slippageMultiplier)
        self.fillQualityBias = fxSafeFinite(fillQualityBias)
        self.latencyMultiplier = fxSafeFinite(latencyMultiplier)
        self.fragilityMultiplier = fxSafeFinite(fragilityMultiplier)
        self.deviationMultiplier = fxSafeFinite(deviationMultiplier)
    }

    public var key: String {
        "\(kind)|\(symbol)|\(session)|\(regime)"
    }

    public static var fallback: ExecutionQualityTier {
        ExecutionQualityTier(ready: true)
    }

    private static func normalizedToken(_ raw: String, fallback: String) -> String {
        let value = raw.trimmingCharacters(in: .whitespacesAndNewlines).uppercased()
        return value.isEmpty ? fallback : value
    }
}

public struct ExecutionQualityMemory: Codable, Hashable, Sendable {
    public var generatedAt: Int64
    public var defaultMethod: String
    public var tiers: [ExecutionQualityTier]

    public init(
        generatedAt: Int64 = 0,
        defaultMethod: String = "SCORECARD_V1",
        tiers: [ExecutionQualityTier] = []
    ) {
        self.generatedAt = max(0, generatedAt)
        self.defaultMethod = defaultMethod.isEmpty ? "SCORECARD_V1" : defaultMethod
        self.tiers = Array(tiers.prefix(ExecutionQualityConstants.maxTiers))
    }
}

public struct ExecutionQualityTierSelection: Codable, Hashable, Sendable {
    public var tier: ExecutionQualityTier
    public var found: Bool
    public var fallbackUsed: Bool
    public var supportUsable: Bool

    public init(
        tier: ExecutionQualityTier = .fallback,
        found: Bool = false,
        fallbackUsed: Bool = true,
        supportUsable: Bool = false
    ) {
        self.tier = tier
        self.found = found
        self.fallbackUsed = fallbackUsed
        self.supportUsable = supportUsable
    }
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
    public static func configPath() -> String {
        "\(ExecutionQualityConstants.runtimeDirectory)/execution_quality_config.tsv"
    }

    public static func memoryPath() -> String {
        "\(ExecutionQualityConstants.runtimeDirectory)/execution_quality_memory.tsv"
    }

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

    public static func parseConfig(tsv: String?) -> ExecutionQualityConfig {
        var config = ExecutionQualityConfig()
        guard let tsv else { return config }
        var buckets = config.bucketHierarchy

        for line in tsv.components(separatedBy: .newlines) where !line.isEmpty {
            let parts = line.split(separator: "\t", omittingEmptySubsequences: false)
            guard parts.count >= 2 else { continue }
            let key = String(parts[0])
            let value = String(parts[1])

            switch key {
            case "enabled":
                config.enabled = (Int(value) ?? 0) != 0
            case "block_on_unknown":
                config.blockOnUnknown = (Int(value) ?? 0) != 0
            case "allow_block_state":
                config.allowBlockState = (Int(value) ?? 0) != 0
            case "support_soft_floor":
                config.supportSoftFloor = max(0, Int(value) ?? config.supportSoftFloor)
            case "support_hard_floor":
                config.supportHardFloor = max(0, Int(value) ?? config.supportHardFloor)
            case "memory_stale_after_hours":
                config.memoryStaleAfterHours = max(0, Int(value) ?? config.memoryStaleAfterHours)
            case "threshold_normal_min":
                config.thresholdNormalMin = fxClamp(Double(value) ?? config.thresholdNormalMin, 0.0, 1.0)
            case "threshold_caution_min":
                config.thresholdCautionMin = fxClamp(Double(value) ?? config.thresholdCautionMin, 0.0, 1.0)
            case "threshold_stressed_min":
                config.thresholdStressedMin = fxClamp(Double(value) ?? config.thresholdStressedMin, 0.0, 1.0)
            case "lot_scale_normal":
                config.lotScaleNormal = max(0.0, Double(value) ?? config.lotScaleNormal)
            case "lot_scale_caution":
                config.lotScaleCaution = max(0.0, Double(value) ?? config.lotScaleCaution)
            case "lot_scale_stressed":
                config.lotScaleStressed = max(0.0, Double(value) ?? config.lotScaleStressed)
            case "lot_scale_blocked":
                config.lotScaleBlocked = max(0.0, Double(value) ?? config.lotScaleBlocked)
            case "enter_prob_buffer_normal":
                config.enterProbabilityBufferNormal = max(0.0, Double(value) ?? config.enterProbabilityBufferNormal)
            case "enter_prob_buffer_caution":
                config.enterProbabilityBufferCaution = max(0.0, Double(value) ?? config.enterProbabilityBufferCaution)
            case "enter_prob_buffer_stressed":
                config.enterProbabilityBufferStressed = max(0.0, Double(value) ?? config.enterProbabilityBufferStressed)
            case "enter_prob_buffer_blocked":
                config.enterProbabilityBufferBlocked = max(0.0, Double(value) ?? config.enterProbabilityBufferBlocked)
            case "cap_spread_expected_mult":
                config.capExpectedPriceCostMultiplier = max(0.0, Double(value) ?? config.capExpectedPriceCostMultiplier)
            case "cap_expected_slippage_points":
                config.capExpectedSlippagePoints = max(0.0, Double(value) ?? config.capExpectedSlippagePoints)
            case "cap_allowed_deviation_points_min":
                config.capAllowedDeviationPointsMin = max(0.0, Double(value) ?? config.capAllowedDeviationPointsMin)
            case "cap_allowed_deviation_points_max":
                config.capAllowedDeviationPointsMax = max(config.capAllowedDeviationPointsMin, Double(value) ?? config.capAllowedDeviationPointsMax)
            case "weight_spread_zscore":
                config.weightPriceCostZScore = Double(value) ?? config.weightPriceCostZScore
            case "weight_news_risk":
                config.weightNewsRisk = Double(value) ?? config.weightNewsRisk
            case "weight_rates_risk":
                config.weightRatesRisk = Double(value) ?? config.weightRatesRisk
            case "weight_micro_liquidity":
                config.weightMicroLiquidity = Double(value) ?? config.weightMicroLiquidity
            case "weight_micro_hostile":
                config.weightMicroHostile = Double(value) ?? config.weightMicroHostile
            case "weight_volatility_burst":
                config.weightVolatilityBurst = Double(value) ?? config.weightVolatilityBurst
            case "weight_tick_rate_burst":
                config.weightTickRateBurst = Double(value) ?? config.weightTickRateBurst
            case "weight_session_thinness":
                config.weightSessionThinness = Double(value) ?? config.weightSessionThinness
            case "weight_broker_reject":
                config.weightBrokerReject = Double(value) ?? config.weightBrokerReject
            case "weight_broker_partial":
                config.weightBrokerPartial = Double(value) ?? config.weightBrokerPartial
            case "weight_broker_latency":
                config.weightBrokerLatency = Double(value) ?? config.weightBrokerLatency
            case "weight_broker_event_burst":
                config.weightBrokerEventBurst = Double(value) ?? config.weightBrokerEventBurst
            case "weight_stale_context":
                config.weightStaleContext = Double(value) ?? config.weightStaleContext
            case "weight_support_shortfall":
                config.weightSupportShortfall = Double(value) ?? config.weightSupportShortfall
            case "bucket_count":
                config.bucketCount = Int(fxClamp(Double(Int(value) ?? config.bucketCount), 0.0, Double(ExecutionQualityConstants.maxBuckets)))
            default:
                if key.hasPrefix("bucket_"),
                   let index = Int(key.dropFirst(7)),
                   index >= 0,
                   index < ExecutionQualityConstants.maxBuckets {
                    buckets[index] = value.trimmingCharacters(in: .whitespacesAndNewlines).uppercased()
                }
            }
        }

        config.bucketHierarchy = buckets
        if config.capAllowedDeviationPointsMax < config.capAllowedDeviationPointsMin {
            config.capAllowedDeviationPointsMax = config.capAllowedDeviationPointsMin
        }
        config.ready = true
        return config
    }

    public static func parseMemory(tsv: String?) -> ExecutionQualityMemory {
        guard let tsv else { return ExecutionQualityMemory() }
        var generatedAt: Int64 = 0
        var method = "SCORECARD_V1"
        var tiers: [ExecutionQualityTier] = []
        tiers.reserveCapacity(ExecutionQualityConstants.maxTiers)

        for line in tsv.components(separatedBy: .newlines) where !line.isEmpty {
            let parts = line.split(separator: "\t", omittingEmptySubsequences: false).map(String.init)
            guard parts.count >= 3 else { continue }
            let kind = parts[0]
            if kind == "meta" {
                if parts[1] == "generated_at" {
                    generatedAt = parseISO8601UTC(parts[2])
                } else if parts[1] == "default_method" {
                    method = parts[2]
                }
                continue
            }

            guard kind == "tier",
                  parts.count >= 13,
                  tiers.count < ExecutionQualityConstants.maxTiers else {
                continue
            }
            tiers.append(ExecutionQualityTier(
                ready: true,
                kind: parts[1],
                symbol: parts[2],
                session: parts[3],
                regime: parts[4],
                support: Int(parts[5]) ?? 0,
                quality: Double(parts[6]) ?? 0.0,
                priceCostMultiplier: Double(parts[7]) ?? 1.0,
                slippageMultiplier: Double(parts[8]) ?? 1.0,
                fillQualityBias: Double(parts[9]) ?? 0.0,
                latencyMultiplier: Double(parts[10]) ?? 1.0,
                fragilityMultiplier: Double(parts[11]) ?? 1.0,
                deviationMultiplier: Double(parts[12]) ?? 1.0
            ))
        }

        return ExecutionQualityMemory(generatedAt: generatedAt, defaultMethod: method, tiers: tiers)
    }

    public static func tierHierarchyIndex(kind: String) -> Int {
        switch kind.uppercased() {
        case "PAIR_SESSION_REGIME": return 0
        case "PAIR_REGIME": return 1
        case "SESSION_REGIME": return 2
        case "REGIME": return 3
        case "GLOBAL": return 4
        default: return 99
        }
    }

    public static func tierMatches(
        _ tier: ExecutionQualityTier,
        kind: String,
        symbol: String,
        session: String,
        regime: String
    ) -> Bool {
        let tierKind = tier.kind
        let targetKind = kind.uppercased()
        guard tierKind == targetKind else { return false }
        let targetSymbol = symbol.uppercased()
        let targetSession = session.uppercased()
        let targetRegime = regime.uppercased()
        switch targetKind {
        case "PAIR_SESSION_REGIME":
            return tier.symbol == targetSymbol && tier.session == targetSession && tier.regime == targetRegime
        case "PAIR_REGIME":
            return tier.symbol == targetSymbol && tier.regime == targetRegime
        case "SESSION_REGIME":
            return tier.session == targetSession && tier.regime == targetRegime
        case "REGIME":
            return tier.regime == targetRegime
        case "GLOBAL":
            return true
        default:
            return false
        }
    }

    public static func selectTier(
        symbol: String,
        session: String,
        regime: String,
        config: ExecutionQualityConfig,
        memory: ExecutionQualityMemory
    ) -> ExecutionQualityTierSelection {
        guard !memory.tiers.isEmpty else {
            return ExecutionQualityTierSelection()
        }

        for kind in config.effectiveBucketHierarchy where !kind.isEmpty {
            var bestPreferred: ExecutionQualityTier?
            var bestFallback: ExecutionQualityTier?
            var bestPreferredSupport = -1
            var bestFallbackSupport = -1
            var bestPreferredQuality = -1.0
            var bestFallbackQuality = -1.0

            for tier in memory.tiers where tierMatches(
                tier,
                kind: kind,
                symbol: symbol,
                session: session,
                regime: regime
            ) {
                if tier.support >= config.supportSoftFloor {
                    if bestPreferred == nil ||
                        tier.support > bestPreferredSupport ||
                        (tier.support == bestPreferredSupport && tier.quality > bestPreferredQuality) {
                        bestPreferred = tier
                        bestPreferredSupport = tier.support
                        bestPreferredQuality = tier.quality
                    }
                } else if tier.support >= config.supportHardFloor {
                    if bestFallback == nil ||
                        tier.support > bestFallbackSupport ||
                        (tier.support == bestFallbackSupport && tier.quality > bestFallbackQuality) {
                        bestFallback = tier
                        bestFallbackSupport = tier.support
                        bestFallbackQuality = tier.quality
                    }
                }
            }

            if let selected = bestPreferred {
                return ExecutionQualityTierSelection(
                    tier: selected,
                    found: true,
                    fallbackUsed: false,
                    supportUsable: true
                )
            }
            if let selected = bestFallback {
                return ExecutionQualityTierSelection(
                    tier: selected,
                    found: true,
                    fallbackUsed: true,
                    supportUsable: true
                )
            }
        }

        return ExecutionQualityTierSelection()
    }

    public static func sessionThinness(sessionLabel: String, handoffFlag: Bool) -> Double {
        let session = sessionLabel.uppercased()
        var thinness = 0.18
        if session.contains("ASIA") {
            thinness = 0.42
        }
        if session.contains("OVERLAP") {
            thinness = 0.22
        }
        if session.contains("ROLLOVER") || session.contains("OFF") {
            thinness = 0.60
        }
        if handoffFlag {
            thinness = max(thinness, 0.55)
        }
        return fxClamp(thinness, 0.0, 1.0)
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

    private static func parseISO8601UTC(_ raw: String) -> Int64 {
        guard raw.count >= 19 else { return 0 }
        var calendar = Calendar(identifier: .gregorian)
        calendar.timeZone = TimeZone(secondsFromGMT: 0) ?? .gmt
        let year = Int(raw.prefix(4)) ?? 0
        let monthStart = raw.index(raw.startIndex, offsetBy: 5)
        let monthEnd = raw.index(monthStart, offsetBy: 2)
        let dayStart = raw.index(raw.startIndex, offsetBy: 8)
        let dayEnd = raw.index(dayStart, offsetBy: 2)
        let hourStart = raw.index(raw.startIndex, offsetBy: 11)
        let hourEnd = raw.index(hourStart, offsetBy: 2)
        let minuteStart = raw.index(raw.startIndex, offsetBy: 14)
        let minuteEnd = raw.index(minuteStart, offsetBy: 2)
        let secondStart = raw.index(raw.startIndex, offsetBy: 17)
        let secondEnd = raw.index(secondStart, offsetBy: 2)
        let month = Int(raw[monthStart..<monthEnd]) ?? 0
        let day = Int(raw[dayStart..<dayEnd]) ?? 0
        let hour = Int(raw[hourStart..<hourEnd]) ?? 0
        let minute = Int(raw[minuteStart..<minuteEnd]) ?? 0
        let second = Int(raw[secondStart..<secondEnd]) ?? 0
        guard year >= 2000,
              (1...12).contains(month),
              (1...31).contains(day),
              (0...23).contains(hour),
              (0...59).contains(minute),
              (0...59).contains(second) else {
            return 0
        }
        let components = DateComponents(
            calendar: calendar,
            timeZone: calendar.timeZone,
            year: year,
            month: month,
            day: day,
            hour: hour,
            minute: minute,
            second: second
        )
        guard let date = calendar.date(from: components) else { return 0 }
        return Int64(date.timeIntervalSince1970)
    }
}
