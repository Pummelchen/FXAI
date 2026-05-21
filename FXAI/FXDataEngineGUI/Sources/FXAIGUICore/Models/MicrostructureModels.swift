import Foundation

public struct MicrostructureServiceStatus: Hashable, Sendable {
    public let ok: Bool
    public let stale: Bool
    public let enabled: Bool
    public let pollIntervalMS: Int
    public let symbolRefreshSec: Int
    public let snapshotStaleAfterSec: Int
    public let lastPollAt: Date?
    public let lastSuccessAt: Date?
    public let lastSymbolRefreshAt: Date?
    public let lastError: String?

    public init(
        ok: Bool,
        stale: Bool,
        enabled: Bool,
        pollIntervalMS: Int,
        symbolRefreshSec: Int,
        snapshotStaleAfterSec: Int,
        lastPollAt: Date?,
        lastSuccessAt: Date?,
        lastSymbolRefreshAt: Date?,
        lastError: String?
    ) {
        self.ok = ok
        self.stale = stale
        self.enabled = enabled
        self.pollIntervalMS = pollIntervalMS
        self.symbolRefreshSec = symbolRefreshSec
        self.snapshotStaleAfterSec = snapshotStaleAfterSec
        self.lastPollAt = lastPollAt
        self.lastSuccessAt = lastSuccessAt
        self.lastSymbolRefreshAt = lastSymbolRefreshAt
        self.lastError = lastError
    }
}

public struct MicrostructureSymbolState: Identifiable, Hashable, Sendable {
    public let id: String
    public let symbol: String
    public let brokerSymbol: String
    public let available: Bool
    public let stale: Bool
    public let generatedAt: Date?
    public let spreadCurrent: Double
    public let silentGapSecondsCurrent: Double
    public let sessionTag: String
    public let handoffFlag: Bool
    public let minutesSinceSessionOpen: Int?
    public let minutesToSessionClose: Int?
    public let sessionOpenBurstScore: Double
    public let sessionSpreadBehaviorScore: Double
    public let liquidityStressScore: Double
    public let hostileExecutionScore: Double
    public let microstructureRegime: String
    public let tradeGate: String
    public let tickImbalance30s: Double
    public let directionalEfficiency60s: Double
    public let spreadZScore60s: Double
    public let tickRate60s: Double
    public let tickRateZScore60s: Double
    public let realizedVol5m: Double
    public let volBurstScore5m: Double
    public let localExtremaBreachScore60s: Double
    public let sweepAndRejectFlag60s: Bool
    public let breakoutReversalScore60s: Double
    public let exhaustionProxy60s: Double
    public let reasons: [String]

    public init(
        symbol: String,
        brokerSymbol: String,
        available: Bool,
        stale: Bool,
        generatedAt: Date?,
        spreadCurrent: Double,
        silentGapSecondsCurrent: Double,
        sessionTag: String,
        handoffFlag: Bool,
        minutesSinceSessionOpen: Int?,
        minutesToSessionClose: Int?,
        sessionOpenBurstScore: Double,
        sessionSpreadBehaviorScore: Double,
        liquidityStressScore: Double,
        hostileExecutionScore: Double,
        microstructureRegime: String,
        tradeGate: String,
        tickImbalance30s: Double,
        directionalEfficiency60s: Double,
        spreadZScore60s: Double,
        tickRate60s: Double,
        tickRateZScore60s: Double,
        realizedVol5m: Double,
        volBurstScore5m: Double,
        localExtremaBreachScore60s: Double,
        sweepAndRejectFlag60s: Bool,
        breakoutReversalScore60s: Double,
        exhaustionProxy60s: Double,
        reasons: [String]
    ) {
        id = symbol
        self.symbol = symbol
        self.brokerSymbol = brokerSymbol
        self.available = available
        self.stale = stale
        self.generatedAt = generatedAt
        self.spreadCurrent = spreadCurrent
        self.silentGapSecondsCurrent = silentGapSecondsCurrent
        self.sessionTag = sessionTag
        self.handoffFlag = handoffFlag
        self.minutesSinceSessionOpen = minutesSinceSessionOpen
        self.minutesToSessionClose = minutesToSessionClose
        self.sessionOpenBurstScore = sessionOpenBurstScore
        self.sessionSpreadBehaviorScore = sessionSpreadBehaviorScore
        self.liquidityStressScore = liquidityStressScore
        self.hostileExecutionScore = hostileExecutionScore
        self.microstructureRegime = microstructureRegime
        self.tradeGate = tradeGate
        self.tickImbalance30s = tickImbalance30s
        self.directionalEfficiency60s = directionalEfficiency60s
        self.spreadZScore60s = spreadZScore60s
        self.tickRate60s = tickRate60s
        self.tickRateZScore60s = tickRateZScore60s
        self.realizedVol5m = realizedVol5m
        self.volBurstScore5m = volBurstScore5m
        self.localExtremaBreachScore60s = localExtremaBreachScore60s
        self.sweepAndRejectFlag60s = sweepAndRejectFlag60s
        self.breakoutReversalScore60s = breakoutReversalScore60s
        self.exhaustionProxy60s = exhaustionProxy60s
        self.reasons = reasons
    }
}

public struct MicrostructureSnapshot: Sendable {
    public let generatedAt: Date
    public let serviceStatus: MicrostructureServiceStatus
    public let symbols: [MicrostructureSymbolState]
    public let healthSummary: [KeyValueRecord]
    public let artifactPaths: [KeyValueRecord]

    public init(
        generatedAt: Date,
        serviceStatus: MicrostructureServiceStatus,
        symbols: [MicrostructureSymbolState],
        healthSummary: [KeyValueRecord],
        artifactPaths: [KeyValueRecord]
    ) {
        self.generatedAt = generatedAt
        self.serviceStatus = serviceStatus
        self.symbols = symbols
        self.healthSummary = healthSummary
        self.artifactPaths = artifactPaths
    }
}
