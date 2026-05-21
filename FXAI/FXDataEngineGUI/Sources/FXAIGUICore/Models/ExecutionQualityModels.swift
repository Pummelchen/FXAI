import Foundation

public struct ExecutionQualityTransition: Identifiable, Hashable, Sendable {
    public let id: String
    public let type: String
    public let fromValue: String
    public let toValue: String
    public let observedAt: Date?

    public init(type: String, fromValue: String, toValue: String, observedAt: Date?) {
        self.id = "\(type)::\(fromValue)::\(toValue)::\(observedAt?.ISO8601Format() ?? "unknown")"
        self.type = type
        self.fromValue = fromValue
        self.toValue = toValue
        self.observedAt = observedAt
    }
}

public struct ExecutionQualitySymbolSnapshot: Identifiable, Hashable, Sendable {
    public let id: String
    public let symbol: String
    public let generatedAt: Date?
    public let method: String
    public let sessionLabel: String
    public let regimeLabel: String
    public let tierKind: String
    public let tierKey: String
    public let support: Int
    public let quality: Double
    public let fallbackUsed: Bool
    public let memoryStale: Bool
    public let dataStale: Bool
    public let supportUsable: Bool
    public let newsWindowActive: Bool
    public let ratesRepricingActive: Bool
    public let brokerCoverage: Double
    public let brokerRejectProbability: Double
    public let brokerPartialFillProbability: Double
    public let spreadNowPoints: Double
    public let spreadExpectedPoints: Double
    public let spreadWideningRisk: Double
    public let expectedSlippagePoints: Double
    public let slippageRisk: Double
    public let fillQualityScore: Double
    public let latencySensitivityScore: Double
    public let liquidityFragilityScore: Double
    public let executionQualityScore: Double
    public let allowedDeviationPoints: Double
    public let cautionLotScale: Double
    public let cautionEnterProbBuffer: Double
    public let executionState: String
    public let reasons: [String]
    public let replayStateCounts: [KeyValueRecord]
    public let replayTierCounts: [KeyValueRecord]
    public let replayTopReasons: [KeyValueRecord]
    public let recentTransitions: [ExecutionQualityTransition]
    public let observationCount: Int
    public let maxSpreadWideningRisk: Double
    public let maxSlippageRisk: Double
    public let minExecutionQualityScore: Double

    public init(
        symbol: String,
        generatedAt: Date?,
        method: String,
        sessionLabel: String,
        regimeLabel: String,
        tierKind: String,
        tierKey: String,
        support: Int,
        quality: Double,
        fallbackUsed: Bool,
        memoryStale: Bool,
        dataStale: Bool,
        supportUsable: Bool,
        newsWindowActive: Bool,
        ratesRepricingActive: Bool,
        brokerCoverage: Double,
        brokerRejectProbability: Double,
        brokerPartialFillProbability: Double,
        spreadNowPoints: Double,
        spreadExpectedPoints: Double,
        spreadWideningRisk: Double,
        expectedSlippagePoints: Double,
        slippageRisk: Double,
        fillQualityScore: Double,
        latencySensitivityScore: Double,
        liquidityFragilityScore: Double,
        executionQualityScore: Double,
        allowedDeviationPoints: Double,
        cautionLotScale: Double,
        cautionEnterProbBuffer: Double,
        executionState: String,
        reasons: [String],
        replayStateCounts: [KeyValueRecord],
        replayTierCounts: [KeyValueRecord],
        replayTopReasons: [KeyValueRecord],
        recentTransitions: [ExecutionQualityTransition],
        observationCount: Int,
        maxSpreadWideningRisk: Double,
        maxSlippageRisk: Double,
        minExecutionQualityScore: Double
    ) {
        self.id = symbol
        self.symbol = symbol
        self.generatedAt = generatedAt
        self.method = method
        self.sessionLabel = sessionLabel
        self.regimeLabel = regimeLabel
        self.tierKind = tierKind
        self.tierKey = tierKey
        self.support = support
        self.quality = quality
        self.fallbackUsed = fallbackUsed
        self.memoryStale = memoryStale
        self.dataStale = dataStale
        self.supportUsable = supportUsable
        self.newsWindowActive = newsWindowActive
        self.ratesRepricingActive = ratesRepricingActive
        self.brokerCoverage = brokerCoverage
        self.brokerRejectProbability = brokerRejectProbability
        self.brokerPartialFillProbability = brokerPartialFillProbability
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
        self.cautionEnterProbBuffer = cautionEnterProbBuffer
        self.executionState = executionState
        self.reasons = reasons
        self.replayStateCounts = replayStateCounts
        self.replayTierCounts = replayTierCounts
        self.replayTopReasons = replayTopReasons
        self.recentTransitions = recentTransitions
        self.observationCount = observationCount
        self.maxSpreadWideningRisk = maxSpreadWideningRisk
        self.maxSlippageRisk = maxSlippageRisk
        self.minExecutionQualityScore = minExecutionQualityScore
    }
}

public struct ExecutionQualitySnapshot: Sendable {
    public let generatedAt: Date
    public let replayHoursBack: Int
    public let symbols: [ExecutionQualitySymbolSnapshot]

    public init(generatedAt: Date, replayHoursBack: Int, symbols: [ExecutionQualitySymbolSnapshot]) {
        self.generatedAt = generatedAt
        self.replayHoursBack = replayHoursBack
        self.symbols = symbols
    }
}
