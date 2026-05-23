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

public struct ExecutionQualityPolicyInputs: Codable, Hashable, Sendable {
    public var symbol: String
    public var generatedAtUTC: Int64
    public var sessionLabel: String?
    public var regimeLabel: String?
    public var priceCostPredictedPoints: Double
    public var horizonMinutes: Int
    public var upstreamDecision: Int
    public var pathRisk: Double
    public var fillRisk: Double
    public var newsState: NewsPulsePairState
    public var ratesState: RatesEnginePairState
    public var crossAssetState: CrossAssetPairState
    public var microstructureState: MicrostructurePairState
    public var brokerStats: BrokerExecutionStats

    public init(
        symbol: String,
        generatedAtUTC: Int64,
        sessionLabel: String? = nil,
        regimeLabel: String? = nil,
        priceCostPredictedPoints: Double = 0.0,
        horizonMinutes: Int = 1,
        upstreamDecision: Int = 2,
        pathRisk: Double = 0.0,
        fillRisk: Double = 0.0,
        newsState: NewsPulsePairState = .reset,
        ratesState: RatesEnginePairState = .reset,
        crossAssetState: CrossAssetPairState = .reset,
        microstructureState: MicrostructurePairState = .reset,
        brokerStats: BrokerExecutionStats = BrokerExecutionStats()
    ) {
        self.symbol = symbol.uppercased()
        self.generatedAtUTC = max(0, generatedAtUTC)
        self.sessionLabel = sessionLabel
        self.regimeLabel = regimeLabel
        self.priceCostPredictedPoints = max(0.0, fxSafeFinite(priceCostPredictedPoints))
        self.horizonMinutes = max(1, horizonMinutes)
        self.upstreamDecision = upstreamDecision
        self.pathRisk = fxClamp(pathRisk, 0.0, 1.0)
        self.fillRisk = fxClamp(fillRisk, 0.0, 1.0)
        self.newsState = newsState
        self.ratesState = ratesState
        self.crossAssetState = crossAssetState
        self.microstructureState = microstructureState
        self.brokerStats = brokerStats
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
    public var newsWindowActive: Bool
    public var ratesRepricingActive: Bool
    public var generatedAt: Int64
    public var symbol: String
    public var method: String
    public var sessionLabel: String
    public var regimeLabel: String
    public var selectedTierKind: String
    public var selectedTierKey: String
    public var selectedSupport: Int
    public var selectedQuality: Double
    public var brokerCoverage: Double
    public var brokerRejectProbability: Double
    public var brokerPartialFillProbability: Double
    public var priceCostNowPoints: Double
    public var priceCostExpectedPoints: Double
    public var priceCostWideningRisk: Double
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

    private enum CodingKeys: String, CodingKey {
        case ready
        case available
        case stale
        case fallbackUsed
        case memoryStale
        case dataStale
        case supportUsable
        case newsWindowActive
        case ratesRepricingActive
        case generatedAt
        case symbol
        case method
        case sessionLabel
        case regimeLabel
        case selectedTierKind
        case selectedTierKey
        case selectedSupport
        case selectedQuality
        case brokerCoverage
        case brokerRejectProbability
        case brokerPartialFillProbability
        case priceCostNowPoints
        case priceCostExpectedPoints
        case priceCostWideningRisk
        case legacySpreadNowPoints = "spreadNowPoints"
        case legacySpreadExpectedPoints = "spreadExpectedPoints"
        case legacySpreadWideningRisk = "spreadWideningRisk"
        case expectedSlippagePoints
        case slippageRisk
        case fillQualityScore
        case latencySensitivityScore
        case liquidityFragilityScore
        case executionQualityScore
        case allowedDeviationPoints
        case cautionLotScale
        case cautionEnterProbabilityBuffer
        case executionState
        case reasons
    }

    public init(
        ready: Bool = false,
        available: Bool = false,
        stale: Bool = true,
        fallbackUsed: Bool = false,
        memoryStale: Bool = true,
        dataStale: Bool = true,
        supportUsable: Bool = false,
        newsWindowActive: Bool = false,
        ratesRepricingActive: Bool = false,
        generatedAt: Int64 = 0,
        symbol: String = "",
        method: String = "SCORECARD_V1",
        sessionLabel: String = "UNKNOWN",
        regimeLabel: String = "UNKNOWN",
        selectedTierKind: String = "GLOBAL",
        selectedTierKey: String = "GLOBAL|*|*|*",
        selectedSupport: Int = 0,
        selectedQuality: Double = 0.0,
        brokerCoverage: Double = 0.0,
        brokerRejectProbability: Double = 0.0,
        brokerPartialFillProbability: Double = 0.0,
        priceCostNowPoints: Double = 0.0,
        priceCostExpectedPoints: Double = 0.0,
        priceCostWideningRisk: Double = 0.0,
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
        self.newsWindowActive = newsWindowActive
        self.ratesRepricingActive = ratesRepricingActive
        self.generatedAt = max(0, generatedAt)
        self.symbol = symbol.uppercased()
        self.method = method
        self.sessionLabel = sessionLabel
        self.regimeLabel = regimeLabel
        self.selectedTierKind = selectedTierKind
        self.selectedTierKey = selectedTierKey
        self.selectedSupport = selectedSupport
        self.selectedQuality = selectedQuality
        self.brokerCoverage = fxClamp(brokerCoverage, 0.0, 1.0)
        self.brokerRejectProbability = fxClamp(brokerRejectProbability, 0.0, 1.0)
        self.brokerPartialFillProbability = fxClamp(brokerPartialFillProbability, 0.0, 1.0)
        self.priceCostNowPoints = priceCostNowPoints
        self.priceCostExpectedPoints = priceCostExpectedPoints
        self.priceCostWideningRisk = priceCostWideningRisk
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

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.init(
            ready: try container.decodeIfPresent(Bool.self, forKey: .ready) ?? false,
            available: try container.decodeIfPresent(Bool.self, forKey: .available) ?? false,
            stale: try container.decodeIfPresent(Bool.self, forKey: .stale) ?? true,
            fallbackUsed: try container.decodeIfPresent(Bool.self, forKey: .fallbackUsed) ?? false,
            memoryStale: try container.decodeIfPresent(Bool.self, forKey: .memoryStale) ?? true,
            dataStale: try container.decodeIfPresent(Bool.self, forKey: .dataStale) ?? true,
            supportUsable: try container.decodeIfPresent(Bool.self, forKey: .supportUsable) ?? false,
            newsWindowActive: try container.decodeIfPresent(Bool.self, forKey: .newsWindowActive) ?? false,
            ratesRepricingActive: try container.decodeIfPresent(Bool.self, forKey: .ratesRepricingActive) ?? false,
            generatedAt: try container.decodeIfPresent(Int64.self, forKey: .generatedAt) ?? 0,
            symbol: try container.decodeIfPresent(String.self, forKey: .symbol) ?? "",
            method: try container.decodeIfPresent(String.self, forKey: .method) ?? "SCORECARD_V1",
            sessionLabel: try container.decodeIfPresent(String.self, forKey: .sessionLabel) ?? "UNKNOWN",
            regimeLabel: try container.decodeIfPresent(String.self, forKey: .regimeLabel) ?? "UNKNOWN",
            selectedTierKind: try container.decodeIfPresent(String.self, forKey: .selectedTierKind) ?? "GLOBAL",
            selectedTierKey: try container.decodeIfPresent(String.self, forKey: .selectedTierKey) ?? "GLOBAL|*|*|*",
            selectedSupport: try container.decodeIfPresent(Int.self, forKey: .selectedSupport) ?? 0,
            selectedQuality: try container.decodeIfPresent(Double.self, forKey: .selectedQuality) ?? 0.0,
            brokerCoverage: try container.decodeIfPresent(Double.self, forKey: .brokerCoverage) ?? 0.0,
            brokerRejectProbability: try container.decodeIfPresent(Double.self, forKey: .brokerRejectProbability) ?? 0.0,
            brokerPartialFillProbability: try container.decodeIfPresent(Double.self, forKey: .brokerPartialFillProbability) ?? 0.0,
            priceCostNowPoints: try container.decodeIfPresent(Double.self, forKey: .priceCostNowPoints) ??
                container.decodeIfPresent(Double.self, forKey: .legacySpreadNowPoints) ?? 0.0,
            priceCostExpectedPoints: try container.decodeIfPresent(Double.self, forKey: .priceCostExpectedPoints) ??
                container.decodeIfPresent(Double.self, forKey: .legacySpreadExpectedPoints) ?? 0.0,
            priceCostWideningRisk: try container.decodeIfPresent(Double.self, forKey: .priceCostWideningRisk) ??
                container.decodeIfPresent(Double.self, forKey: .legacySpreadWideningRisk) ?? 0.0,
            expectedSlippagePoints: try container.decodeIfPresent(Double.self, forKey: .expectedSlippagePoints) ?? 0.0,
            slippageRisk: try container.decodeIfPresent(Double.self, forKey: .slippageRisk) ?? 0.0,
            fillQualityScore: try container.decodeIfPresent(Double.self, forKey: .fillQualityScore) ?? 0.0,
            latencySensitivityScore: try container.decodeIfPresent(Double.self, forKey: .latencySensitivityScore) ?? 0.0,
            liquidityFragilityScore: try container.decodeIfPresent(Double.self, forKey: .liquidityFragilityScore) ?? 0.0,
            executionQualityScore: try container.decodeIfPresent(Double.self, forKey: .executionQualityScore) ?? 0.0,
            allowedDeviationPoints: try container.decodeIfPresent(Double.self, forKey: .allowedDeviationPoints) ?? 0.0,
            cautionLotScale: try container.decodeIfPresent(Double.self, forKey: .cautionLotScale) ?? 1.0,
            cautionEnterProbabilityBuffer: try container.decodeIfPresent(Double.self, forKey: .cautionEnterProbabilityBuffer) ?? 0.0,
            executionState: try container.decodeIfPresent(String.self, forKey: .executionState) ?? "UNKNOWN",
            reasons: try container.decodeIfPresent([String].self, forKey: .reasons) ?? []
        )
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(ready, forKey: .ready)
        try container.encode(available, forKey: .available)
        try container.encode(stale, forKey: .stale)
        try container.encode(fallbackUsed, forKey: .fallbackUsed)
        try container.encode(memoryStale, forKey: .memoryStale)
        try container.encode(dataStale, forKey: .dataStale)
        try container.encode(supportUsable, forKey: .supportUsable)
        try container.encode(newsWindowActive, forKey: .newsWindowActive)
        try container.encode(ratesRepricingActive, forKey: .ratesRepricingActive)
        try container.encode(generatedAt, forKey: .generatedAt)
        try container.encode(symbol, forKey: .symbol)
        try container.encode(method, forKey: .method)
        try container.encode(sessionLabel, forKey: .sessionLabel)
        try container.encode(regimeLabel, forKey: .regimeLabel)
        try container.encode(selectedTierKind, forKey: .selectedTierKind)
        try container.encode(selectedTierKey, forKey: .selectedTierKey)
        try container.encode(selectedSupport, forKey: .selectedSupport)
        try container.encode(selectedQuality, forKey: .selectedQuality)
        try container.encode(brokerCoverage, forKey: .brokerCoverage)
        try container.encode(brokerRejectProbability, forKey: .brokerRejectProbability)
        try container.encode(brokerPartialFillProbability, forKey: .brokerPartialFillProbability)
        try container.encode(priceCostNowPoints, forKey: .priceCostNowPoints)
        try container.encode(priceCostExpectedPoints, forKey: .priceCostExpectedPoints)
        try container.encode(priceCostWideningRisk, forKey: .priceCostWideningRisk)
        try container.encode(expectedSlippagePoints, forKey: .expectedSlippagePoints)
        try container.encode(slippageRisk, forKey: .slippageRisk)
        try container.encode(fillQualityScore, forKey: .fillQualityScore)
        try container.encode(latencySensitivityScore, forKey: .latencySensitivityScore)
        try container.encode(liquidityFragilityScore, forKey: .liquidityFragilityScore)
        try container.encode(executionQualityScore, forKey: .executionQualityScore)
        try container.encode(allowedDeviationPoints, forKey: .allowedDeviationPoints)
        try container.encode(cautionLotScale, forKey: .cautionLotScale)
        try container.encode(cautionEnterProbabilityBuffer, forKey: .cautionEnterProbabilityBuffer)
        try container.encode(executionState, forKey: .executionState)
        try container.encode(reasons, forKey: .reasons)
    }

    public static var reset: ExecutionQualityPairState {
        ExecutionQualityPairState()
    }

    @available(*, deprecated, renamed: "priceCostNowPoints")
    public var spreadNowPoints: Double {
        get { priceCostNowPoints }
        set { priceCostNowPoints = newValue }
    }

    @available(*, deprecated, renamed: "priceCostExpectedPoints")
    public var spreadExpectedPoints: Double {
        get { priceCostExpectedPoints }
        set { priceCostExpectedPoints = newValue }
    }

    @available(*, deprecated, renamed: "priceCostWideningRisk")
    public var spreadWideningRisk: Double {
        get { priceCostWideningRisk }
        set { priceCostWideningRisk = newValue }
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

    public static func runtimeHistoryPath(symbol: String) -> String {
        "\(ExecutionQualityConstants.runtimeDirectory)/fxai_execution_quality_history_\(ControlPlanePaths.safeToken(symbol)).ndjson"
    }

    public static func runtimeStateTSV(symbol: String, state: ExecutionQualityPairState) -> String? {
        guard state.ready, !symbol.isEmpty else { return nil }
        return runtimeStateRows(symbol: symbol, state: state)
            .map { key, value in
                "\(RuntimeArtifactTSV.field(key))\t\(RuntimeArtifactTSV.field(value))"
            }
            .joined(separator: "\r\n") + "\r\n"
    }

    public static func runtimeHistoryNDJSONLine(symbol: String, state: ExecutionQualityPairState) -> String? {
        guard state.ready, !symbol.isEmpty else { return nil }
        let qSymbol = jsonQuoted(symbol)
        let reasons = state.reasons
            .map(jsonQuoted)
            .joined(separator: ",")

        return "{" +
            "\"generated_at\":\"\(iso8601UTC(state.generatedAt))\"," +
            "\"symbol\":\(qSymbol)," +
            "\"state\":{" +
            "\"symbol\":\(qSymbol)," +
            "\"method\":\(jsonQuoted(state.method))," +
            "\"session_label\":\(jsonQuoted(state.sessionLabel))," +
            "\"regime_label\":\(jsonQuoted(state.regimeLabel))," +
            "\"selected_tier_kind\":\(jsonQuoted(state.selectedTierKind))," +
            "\"selected_tier_key\":\(jsonQuoted(state.selectedTierKey))," +
            "\"selected_support\":\(state.selectedSupport)," +
            "\"selected_quality\":\(RuntimeArtifactTSV.double(state.selectedQuality))," +
            "\"fallback_used\":\(state.fallbackUsed ? 1 : 0)," +
            "\"memory_stale\":\(state.memoryStale ? 1 : 0)," +
            "\"data_stale\":\(state.dataStale ? 1 : 0)," +
            "\"support_usable\":\(state.supportUsable ? 1 : 0)," +
            "\"news_window_active\":\(state.newsWindowActive ? 1 : 0)," +
            "\"rates_repricing_active\":\(state.ratesRepricingActive ? 1 : 0)," +
            "\"broker_coverage\":\(RuntimeArtifactTSV.double(state.brokerCoverage))," +
            "\"broker_reject_prob\":\(RuntimeArtifactTSV.double(state.brokerRejectProbability))," +
            "\"broker_partial_fill_prob\":\(RuntimeArtifactTSV.double(state.brokerPartialFillProbability))," +
            "\"price_cost_now_points\":\(RuntimeArtifactTSV.double(state.priceCostNowPoints))," +
            "\"price_cost_expected_points\":\(RuntimeArtifactTSV.double(state.priceCostExpectedPoints))," +
            "\"price_cost_widening_risk\":\(RuntimeArtifactTSV.double(state.priceCostWideningRisk))," +
            "\"expected_slippage_points\":\(RuntimeArtifactTSV.double(state.expectedSlippagePoints))," +
            "\"slippage_risk\":\(RuntimeArtifactTSV.double(state.slippageRisk))," +
            "\"fill_quality_score\":\(RuntimeArtifactTSV.double(state.fillQualityScore))," +
            "\"latency_sensitivity_score\":\(RuntimeArtifactTSV.double(state.latencySensitivityScore))," +
            "\"liquidity_fragility_score\":\(RuntimeArtifactTSV.double(state.liquidityFragilityScore))," +
            "\"execution_quality_score\":\(RuntimeArtifactTSV.double(state.executionQualityScore))," +
            "\"allowed_deviation_points\":\(RuntimeArtifactTSV.double(state.allowedDeviationPoints))," +
            "\"caution_lot_scale\":\(RuntimeArtifactTSV.double(state.cautionLotScale))," +
            "\"caution_enter_prob_buffer\":\(RuntimeArtifactTSV.double(state.cautionEnterProbabilityBuffer))," +
            "\"execution_state\":\(jsonQuoted(state.executionState))," +
            "\"reason_codes\":[\(reasons)]" +
            "}}"
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
            case "symbol":
                state.symbol = value
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
            case "news_window_active":
                state.newsWindowActive = (Int(value) ?? 0) != 0
            case "rates_repricing_active":
                state.ratesRepricingActive = (Int(value) ?? 0) != 0
            case "broker_coverage":
                state.brokerCoverage = Double(value) ?? 0.0
            case "broker_reject_prob":
                state.brokerRejectProbability = Double(value) ?? 0.0
            case "broker_partial_fill_prob":
                state.brokerPartialFillProbability = Double(value) ?? 0.0
            case "spread_now_points", "price_cost_now_points":
                state.priceCostNowPoints = Double(value) ?? 0.0
            case "spread_expected_points", "price_cost_expected_points":
                state.priceCostExpectedPoints = Double(value) ?? 0.0
            case "spread_widening_risk", "price_cost_widening_risk":
                state.priceCostWideningRisk = Double(value) ?? 0.0
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
            case "cap_price_cost_expected_mult", "cap_spread_expected_mult":
                config.capExpectedPriceCostMultiplier = max(0.0, Double(value) ?? config.capExpectedPriceCostMultiplier)
            case "cap_expected_slippage_points":
                config.capExpectedSlippagePoints = max(0.0, Double(value) ?? config.capExpectedSlippagePoints)
            case "cap_allowed_deviation_points_min":
                config.capAllowedDeviationPointsMin = max(0.0, Double(value) ?? config.capAllowedDeviationPointsMin)
            case "cap_allowed_deviation_points_max":
                config.capAllowedDeviationPointsMax = max(config.capAllowedDeviationPointsMin, Double(value) ?? config.capAllowedDeviationPointsMax)
            case "weight_price_cost_zscore", "weight_spread_zscore":
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

    public static func applyPolicy(
        config: ExecutionQualityConfig,
        memory: ExecutionQualityMemory,
        profile: ExecutionProfile,
        inputs: ExecutionQualityPolicyInputs
    ) -> ExecutionQualityPairState {
        var state = ExecutionQualityPairState.reset
        state.method = memory.defaultMethod
        state.symbol = inputs.symbol
        state.generatedAt = inputs.generatedAtUTC
        state.sessionLabel = resolvedSessionLabel(
            explicit: inputs.sessionLabel,
            news: inputs.newsState,
            microstructure: inputs.microstructureState
        )
        state.regimeLabel = resolvedRegimeLabel(explicit: inputs.regimeLabel)
        state.newsWindowActive = newsWindowActive(inputs.newsState)
        state.ratesRepricingActive = ratesRepricingActive(inputs.ratesState)
        guard config.enabled else { return state }

        let selection = selectTier(
            symbol: inputs.symbol,
            session: state.sessionLabel,
            regime: state.regimeLabel,
            config: config,
            memory: memory
        )
        let tier = selection.tier
        state.selectedTierKind = tier.kind
        state.selectedTierKey = tier.key
        state.selectedSupport = tier.support
        state.selectedQuality = fxClamp(tier.quality, 0.0, 1.0)
        state.fallbackUsed = selection.fallbackUsed

        let newsStale = inputs.newsState.ready && inputs.newsState.available && inputs.newsState.stale
        let ratesStale = inputs.ratesState.ready && inputs.ratesState.available && inputs.ratesState.stale
        let crossStale = inputs.crossAssetState.ready && inputs.crossAssetState.available && inputs.crossAssetState.stale
        let microStale = inputs.microstructureState.ready && inputs.microstructureState.available && inputs.microstructureState.stale
        let staleContextCount = [newsStale, ratesStale, crossStale, microStale].filter { $0 }.count

        state.memoryStale = memory.generatedAt <= 0 ||
            (config.memoryStaleAfterHours > 0 &&
                state.generatedAt > 0 &&
                (state.generatedAt - memory.generatedAt) > Int64(config.memoryStaleAfterHours * 3_600))
        state.priceCostNowPoints = max(inputs.priceCostPredictedPoints, 0.0)
        state.brokerCoverage = fxClamp(inputs.brokerStats.coverage, 0.0, 1.0)
        state.brokerRejectProbability = fxClamp(inputs.brokerStats.rejectProbability, 0.0, 1.0)
        state.brokerPartialFillProbability = fxClamp(
            max(
                inputs.brokerStats.partialFillProbability,
                1.0 - fxClamp(inputs.brokerStats.fillRatioMean, 0.0, 1.0)
            ),
            0.0,
            1.0
        )
        state.supportUsable = selection.supportUsable && state.brokerCoverage >= 0.05
        state.dataStale = state.memoryStale ||
            state.priceCostNowPoints <= 0.0 ||
            (config.blockOnUnknown && staleContextCount > 0)

        let newsRisk = inputs.newsState.ready && inputs.newsState.available
            ? fxClamp(inputs.newsState.newsRiskScore, 0.0, 1.0)
            : (newsStale ? 0.45 : 0.12)
        let ratesRisk = inputs.ratesState.ready && inputs.ratesState.available
            ? fxClamp(inputs.ratesState.ratesRiskScore, 0.0, 1.0)
            : (ratesStale ? 0.32 : 0.10)
        let crossRisk = inputs.crossAssetState.ready && inputs.crossAssetState.available
            ? fxClamp(
                max(
                    inputs.crossAssetState.pairCrossAssetRiskScore,
                    max(
                        inputs.crossAssetState.usdLiquidityStressScore,
                        inputs.crossAssetState.crossAssetDislocationScore
                    )
                ),
                0.0,
                1.0
            )
            : (crossStale ? 0.34 : 0.12)
        let microHostile = inputs.microstructureState.ready && inputs.microstructureState.available
            ? fxClamp(inputs.microstructureState.hostileExecutionScore, 0.0, 1.0)
            : (microStale ? 0.42 : 0.10)
        let microLiquidity = inputs.microstructureState.ready && inputs.microstructureState.available
            ? fxClamp(inputs.microstructureState.liquidityStressScore, 0.0, 1.0)
            : (microStale ? 0.44 : 0.12)
        let priceCostZNorm = inputs.microstructureState.ready && inputs.microstructureState.available
            ? fxClamp(inputs.microstructureState.priceCostZscore60s / 4.0, 0.0, 1.0)
            : 0.0
        let volatilityBurstNorm = inputs.microstructureState.ready && inputs.microstructureState.available
            ? fxClamp(inputs.microstructureState.volBurstScore5m / 3.0, 0.0, 1.0)
            : 0.0
        let tickRateNorm = inputs.microstructureState.ready && inputs.microstructureState.available
            ? fxClamp(inputs.microstructureState.tickRateZscore60s / 3.0, 0.0, 1.0)
            : 0.0
        let tickImbalanceNorm = inputs.microstructureState.ready && inputs.microstructureState.available
            ? fxClamp(abs(inputs.microstructureState.tickImbalance30s), 0.0, 1.0)
            : 0.0
        let thinness = sessionThinness(
            sessionLabel: state.sessionLabel,
            handoffFlag: inputs.microstructureState.ready &&
                inputs.microstructureState.available &&
                inputs.microstructureState.handoffFlag
        )
        let staleNorm = fxClamp(Double(staleContextCount) / 3.0, 0.0, 1.0)
        let supportShortfall = fxClamp(
            Double(config.supportSoftFloor - tier.support) / Double(max(config.supportSoftFloor, 1)),
            0.0,
            1.0
        )
        let brokerBurst = fxClamp(inputs.brokerStats.eventBurstPenalty, 0.0, 1.0)
        let brokerLatencyNorm = fxClamp(inputs.brokerStats.latencyPoints / 5.0, 0.0, 1.0)

        state.priceCostWideningRisk = fxClamp(
            0.10 +
                config.weightPriceCostZScore * priceCostZNorm +
                config.weightNewsRisk * newsRisk +
                config.weightRatesRisk * ratesRisk +
                0.12 * crossRisk +
                config.weightMicroLiquidity * microLiquidity +
                config.weightVolatilityBurst * volatilityBurstNorm +
                config.weightSessionThinness * thinness +
                config.weightBrokerReject * state.brokerRejectProbability * 0.45 +
                config.weightBrokerPartial * state.brokerPartialFillProbability * 0.40 +
                config.weightBrokerEventBurst * brokerBurst +
                config.weightStaleContext * staleNorm +
                (state.newsWindowActive ? 0.08 : 0.0) +
                (state.ratesRepricingActive ? 0.05 : 0.0) -
                0.10 * state.selectedQuality,
            0.0,
            1.0
        )

        let expectedCostMultiplier = fxClamp(
            0.96 +
                0.38 * tier.priceCostMultiplier +
                0.64 * state.priceCostWideningRisk +
                0.14 * priceCostZNorm +
                0.06 * thinness,
            1.0,
            config.capExpectedPriceCostMultiplier
        )
        state.priceCostExpectedPoints = max(
            state.priceCostNowPoints,
            state.priceCostNowPoints * expectedCostMultiplier +
                0.12 * max(inputs.brokerStats.slippagePoints, 0.0)
        )

        state.expectedSlippagePoints = fxClamp(
            max(inputs.brokerStats.slippagePoints, 0.0) * tier.slippageMultiplier +
                0.16 * state.priceCostExpectedPoints +
                0.55 * microHostile +
                0.38 * volatilityBurstNorm +
                0.26 * thinness +
                0.24 * newsRisk +
                0.18 * ratesRisk +
                0.22 * crossRisk +
                0.28 * brokerBurst +
                0.32 * state.brokerRejectProbability +
                0.30 * state.brokerPartialFillProbability +
                0.18 * brokerLatencyNorm * tier.latencyMultiplier,
            0.0,
            config.capExpectedSlippagePoints
        )

        state.slippageRisk = fxClamp(
            0.12 +
                0.24 * fxClamp(
                    state.expectedSlippagePoints / max(state.priceCostExpectedPoints + 0.5, 1.0),
                    0.0,
                    3.0
                ) / 3.0 +
                0.18 * microHostile +
                0.12 * volatilityBurstNorm +
                0.12 * newsRisk +
                0.08 * ratesRisk +
                0.10 * brokerBurst +
                0.12 * state.brokerRejectProbability +
                0.10 * state.brokerPartialFillProbability,
            0.0,
            1.0
        )

        state.latencySensitivityScore = fxClamp(
            0.14 +
                0.22 * tickRateNorm +
                0.18 * volatilityBurstNorm +
                0.16 * newsRisk +
                0.10 * ratesRisk +
                0.08 * crossRisk +
                0.12 * brokerLatencyNorm * tier.latencyMultiplier +
                0.08 * microHostile +
                0.08 * thinness +
                0.06 * tickImbalanceNorm,
            0.0,
            1.0
        )

        state.liquidityFragilityScore = fxClamp(
            0.10 +
                0.26 * microLiquidity * tier.fragilityMultiplier +
                0.16 * microHostile +
                0.12 * priceCostZNorm +
                0.08 * newsRisk +
                0.08 * ratesRisk +
                0.12 * crossRisk +
                0.12 * state.brokerPartialFillProbability +
                0.10 * state.brokerRejectProbability +
                0.10 * thinness +
                0.06 * brokerBurst -
                0.08 * state.selectedQuality,
            0.0,
            1.0
        )

        state.fillQualityScore = fxClamp(
            0.86 +
                tier.fillQualityBias -
                0.28 * state.slippageRisk -
                0.24 * state.latencySensitivityScore -
                0.22 * state.liquidityFragilityScore -
                0.14 * state.brokerRejectProbability -
                0.12 * state.brokerPartialFillProbability -
                0.08 * thinness,
            0.0,
            1.0
        )

        state.executionQualityScore = fxClamp(
            0.40 * state.fillQualityScore +
                0.18 * (1.0 - state.priceCostWideningRisk) +
                0.18 * (1.0 - state.slippageRisk) +
                0.12 * (1.0 - state.latencySensitivityScore) +
                0.12 * (1.0 - state.liquidityFragilityScore) -
                config.weightStaleContext * 0.80 * staleNorm -
                config.weightSupportShortfall * supportShortfall,
            0.0,
            1.0
        )

        let blockThreshold = config.thresholdStressedMin * 0.72
        if state.dataStale && config.blockOnUnknown {
            state.executionState = "BLOCKED"
        } else if config.allowBlockState &&
            (state.executionQualityScore < blockThreshold ||
                state.priceCostWideningRisk >= 0.90 ||
                state.slippageRisk >= 0.90 ||
                state.fillQualityScore <= 0.20) {
            state.executionState = "BLOCKED"
        } else if state.executionQualityScore < config.thresholdStressedMin {
            state.executionState = "STRESSED"
        } else if state.executionQualityScore < config.thresholdCautionMin {
            state.executionState = "CAUTION"
        } else {
            state.executionState = "NORMAL"
        }

        let baseDeviation = ExecutionReplayTools.allowedDeviationPoints(
            profile: profile,
            pathRisk: inputs.pathRisk,
            fillRisk: inputs.fillRisk
        )
        state.allowedDeviationPoints = fxClamp(
            baseDeviation *
                tier.deviationMultiplier *
                (1.0 +
                    0.14 * state.priceCostWideningRisk +
                    0.18 * state.slippageRisk +
                    0.10 * state.latencySensitivityScore),
            config.capAllowedDeviationPointsMin,
            config.capAllowedDeviationPointsMax
        )

        switch state.executionState {
        case "BLOCKED":
            state.cautionLotScale = config.lotScaleBlocked
            state.cautionEnterProbabilityBuffer = config.enterProbabilityBufferBlocked
        case "STRESSED":
            state.cautionLotScale = config.lotScaleStressed
            state.cautionEnterProbabilityBuffer = config.enterProbabilityBufferStressed
        case "CAUTION":
            state.cautionLotScale = config.lotScaleCaution
            state.cautionEnterProbabilityBuffer = config.enterProbabilityBufferCaution
        default:
            state.cautionLotScale = config.lotScaleNormal
            state.cautionEnterProbabilityBuffer = config.enterProbabilityBufferNormal
        }

        appendPolicyReasons(
            state: &state,
            newsRisk: newsRisk,
            ratesRisk: ratesRisk,
            crossRisk: crossRisk,
            microHostile: microHostile,
            microLiquidity: microLiquidity,
            priceCostZNorm: priceCostZNorm,
            volatilityBurstNorm: volatilityBurstNorm,
            sessionThinness: thinness
        )
        state.ready = true
        state.available = true
        state.stale = state.dataStale
        return state
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

    private static func appendPolicyReasons(
        state: inout ExecutionQualityPairState,
        newsRisk: Double,
        ratesRisk: Double,
        crossRisk: Double,
        microHostile: Double,
        microLiquidity: Double,
        priceCostZNorm: Double,
        volatilityBurstNorm: Double,
        sessionThinness: Double
    ) {
        if state.dataStale {
            state.appendReason("DATA_STALE")
        }
        if state.memoryStale {
            state.appendReason("MEMORY_STALE")
        }
        if !state.supportUsable {
            state.appendReason("SUPPORT_TOO_LOW")
        }
        if state.newsWindowActive || newsRisk >= 0.68 {
            state.appendReason("NEWS_WINDOW_ACTIVE")
        }
        if state.ratesRepricingActive || ratesRisk >= 0.68 {
            state.appendReason("RATES_REPRICING_ACTIVE")
        }
        if crossRisk >= 0.58 {
            state.appendReason("CROSS_ASSET_STRESS_ELEVATED")
        }
        if priceCostZNorm >= 0.55 {
            state.appendReason("PRICE_COST_ALREADY_ELEVATED")
        }
        if microHostile >= 0.62 {
            state.appendReason("MICROSTRUCTURE_HOSTILE")
        }
        if microLiquidity >= 0.62 {
            state.appendReason("LIQUIDITY_STRESS_ELEVATED")
        }
        if volatilityBurstNorm >= 0.58 {
            state.appendReason("VOLATILITY_BURST")
        }
        if sessionThinness >= 0.52 {
            state.appendReason("LOW_LIQUIDITY_SESSION")
        }
        if state.slippageRisk >= 0.66 {
            state.appendReason("SLIPPAGE_RISK_ELEVATED")
        }
        if state.latencySensitivityScore >= 0.66 {
            state.appendReason("LATENCY_SENSITIVITY_HIGH")
        }
        if state.brokerRejectProbability >= 0.40 {
            state.appendReason("BROKER_REJECT_RISK_ELEVATED")
        }
        if state.brokerPartialFillProbability >= 0.42 {
            state.appendReason("BROKER_PARTIAL_FILL_RISK_ELEVATED")
        }
        if state.executionState == "BLOCKED" {
            state.appendReason("EXECUTION_STATE_BLOCKED")
        } else if state.executionState == "STRESSED" {
            state.appendReason("EXECUTION_STATE_STRESSED")
        } else if state.executionState == "CAUTION" {
            state.appendReason("EXECUTION_STATE_CAUTION")
        }
    }

    private static func newsWindowActive(_ state: NewsPulsePairState) -> Bool {
        state.ready &&
            state.available &&
            !state.stale &&
            (state.tradeGate.uppercased() == "BLOCK" ||
                state.tradeGate.uppercased() == "CAUTION" ||
                (state.eventETAMinutes >= 0 && state.eventETAMinutes <= 30) ||
                state.newsRiskScore >= 0.64)
    }

    private static func ratesRepricingActive(_ state: RatesEnginePairState) -> Bool {
        state.ready &&
            state.available &&
            !state.stale &&
            (state.meetingPathRepriceNow ||
                state.tradeGate.uppercased() == "BLOCK" ||
                state.ratesRiskScore >= 0.64)
    }

    private static func resolvedSessionLabel(
        explicit: String?,
        news: NewsPulsePairState,
        microstructure: MicrostructurePairState
    ) -> String {
        if let explicit {
            let value = explicit.trimmingCharacters(in: .whitespacesAndNewlines)
            if !value.isEmpty { return value.uppercased() }
        }
        if microstructure.ready {
            let value = microstructure.sessionTag.trimmingCharacters(in: .whitespacesAndNewlines)
            if !value.isEmpty { return value.uppercased() }
        }
        if news.ready {
            let value = news.sessionProfile.trimmingCharacters(in: .whitespacesAndNewlines)
            if !value.isEmpty { return value.uppercased() }
        }
        return "UNKNOWN"
    }

    private static func resolvedRegimeLabel(explicit: String?) -> String {
        if let explicit {
            let value = explicit.trimmingCharacters(in: .whitespacesAndNewlines)
            if !value.isEmpty { return value.uppercased() }
        }
        return "UNKNOWN"
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

    private static func runtimeStateRows(
        symbol: String,
        state: ExecutionQualityPairState
    ) -> [(String, String)] {
        [
            ("symbol", symbol),
            ("generated_at", "\(state.generatedAt)"),
            ("method", state.method),
            ("session_label", state.sessionLabel),
            ("regime_label", state.regimeLabel),
            ("selected_tier_kind", state.selectedTierKind),
            ("selected_tier_key", state.selectedTierKey),
            ("selected_support", "\(state.selectedSupport)"),
            ("selected_quality", RuntimeArtifactTSV.double(state.selectedQuality)),
            ("fallback_used", RuntimeArtifactTSV.bool(state.fallbackUsed)),
            ("memory_stale", RuntimeArtifactTSV.bool(state.memoryStale)),
            ("data_stale", RuntimeArtifactTSV.bool(state.dataStale)),
            ("support_usable", RuntimeArtifactTSV.bool(state.supportUsable)),
            ("news_window_active", RuntimeArtifactTSV.bool(state.newsWindowActive)),
            ("rates_repricing_active", RuntimeArtifactTSV.bool(state.ratesRepricingActive)),
            ("broker_coverage", RuntimeArtifactTSV.double(state.brokerCoverage)),
            ("broker_reject_prob", RuntimeArtifactTSV.double(state.brokerRejectProbability)),
            ("broker_partial_fill_prob", RuntimeArtifactTSV.double(state.brokerPartialFillProbability)),
            ("price_cost_now_points", RuntimeArtifactTSV.double(state.priceCostNowPoints)),
            ("price_cost_expected_points", RuntimeArtifactTSV.double(state.priceCostExpectedPoints)),
            ("price_cost_widening_risk", RuntimeArtifactTSV.double(state.priceCostWideningRisk)),
            ("expected_slippage_points", RuntimeArtifactTSV.double(state.expectedSlippagePoints)),
            ("slippage_risk", RuntimeArtifactTSV.double(state.slippageRisk)),
            ("fill_quality_score", RuntimeArtifactTSV.double(state.fillQualityScore)),
            ("latency_sensitivity_score", RuntimeArtifactTSV.double(state.latencySensitivityScore)),
            ("liquidity_fragility_score", RuntimeArtifactTSV.double(state.liquidityFragilityScore)),
            ("execution_quality_score", RuntimeArtifactTSV.double(state.executionQualityScore)),
            ("allowed_deviation_points", RuntimeArtifactTSV.double(state.allowedDeviationPoints)),
            ("caution_lot_scale", RuntimeArtifactTSV.double(state.cautionLotScale)),
            ("caution_enter_prob_buffer", RuntimeArtifactTSV.double(state.cautionEnterProbabilityBuffer)),
            ("execution_state", state.executionState),
            ("reasons_csv", state.reasonsCSV)
        ]
    }

    private static func iso8601UTC(_ timestamp: Int64) -> String {
        guard timestamp > 0 else { return "" }
        var calendar = Calendar(identifier: .gregorian)
        calendar.timeZone = TimeZone(secondsFromGMT: 0) ?? .gmt
        let components = calendar.dateComponents(
            [.year, .month, .day, .hour, .minute, .second],
            from: Date(timeIntervalSince1970: TimeInterval(timestamp))
        )
        return String(
            format: "%04d-%02d-%02dT%02d:%02d:%02dZ",
            locale: Locale(identifier: "en_US_POSIX"),
            components.year ?? 0,
            components.month ?? 0,
            components.day ?? 0,
            components.hour ?? 0,
            components.minute ?? 0,
            components.second ?? 0
        )
    }

    private static func jsonQuoted(_ value: String) -> String {
        let escaped = value
            .replacingOccurrences(of: "\\", with: "\\\\")
            .replacingOccurrences(of: "\"", with: "\\\"")
            .replacingOccurrences(of: "\r", with: " ")
            .replacingOccurrences(of: "\n", with: " ")
        return "\"\(escaped)\""
    }
}

public extension RuntimeArtifactFileRepository {
    func writeExecutionQualityRuntimeArtifacts(
        symbol: String,
        state: ExecutionQualityPairState
    ) throws {
        guard let stateTSV = ExecutionQualityTools.runtimeStateTSV(symbol: symbol, state: state),
              let historyLine = ExecutionQualityTools.runtimeHistoryNDJSONLine(symbol: symbol, state: state) else {
            return
        }

        let stateURL = url(for: ExecutionQualityTools.runtimeStatePath(symbol: symbol))
        try fileManager.createDirectory(
            at: stateURL.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        try stateTSV.write(to: stateURL, atomically: true, encoding: .utf8)

        let historyURL = url(for: ExecutionQualityTools.runtimeHistoryPath(symbol: symbol))
        try fileManager.createDirectory(
            at: historyURL.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        let historyData = Data((historyLine + "\r\n").utf8)
        if fileManager.fileExists(atPath: historyURL.path) {
            let handle = try FileHandle(forWritingTo: historyURL)
            defer { try? handle.close() }
            try handle.seekToEnd()
            try handle.write(contentsOf: historyData)
        } else {
            try historyData.write(to: historyURL, options: .atomic)
        }
    }
}
