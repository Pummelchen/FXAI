import Foundation

public struct NewsPulseSourceStatus: Identifiable, Hashable, Sendable {
    public let id: String
    public let ok: Bool
    public let stale: Bool
    public let lastUpdateAt: Date?
    public let lastPollAt: Date?
    public let lastSuccessAt: Date?
    public let cursor: String?
    public let lastError: String?

    public init(
        id: String,
        ok: Bool,
        stale: Bool,
        lastUpdateAt: Date?,
        lastPollAt: Date?,
        lastSuccessAt: Date?,
        cursor: String?,
        lastError: String?
    ) {
        self.id = id
        self.ok = ok
        self.stale = stale
        self.lastUpdateAt = lastUpdateAt
        self.lastPollAt = lastPollAt
        self.lastSuccessAt = lastSuccessAt
        self.cursor = cursor
        self.lastError = lastError
    }
}

public struct NewsPulseCurrencyState: Identifiable, Hashable, Sendable {
    public let id: String
    public let currency: String
    public let breakingCount15m: Int
    public let intensity15m: Double
    public let toneMean15m: Double
    public let toneAbsMean15m: Double
    public let burstScore15m: Double
    public let nextHighImpactETAMin: Int?
    public let timeSinceLastHighImpactMin: Int?
    public let inPreEventWindow: Bool
    public let inPostEventWindow: Bool
    public let lastSurpriseProxy: Double?
    public let stale: Bool
    public let riskScore: Double
    public let reasons: [String]

    public init(
        currency: String,
        breakingCount15m: Int,
        intensity15m: Double,
        toneMean15m: Double,
        toneAbsMean15m: Double,
        burstScore15m: Double,
        nextHighImpactETAMin: Int?,
        timeSinceLastHighImpactMin: Int?,
        inPreEventWindow: Bool,
        inPostEventWindow: Bool,
        lastSurpriseProxy: Double?,
        stale: Bool,
        riskScore: Double,
        reasons: [String]
    ) {
        id = currency
        self.currency = currency
        self.breakingCount15m = breakingCount15m
        self.intensity15m = intensity15m
        self.toneMean15m = toneMean15m
        self.toneAbsMean15m = toneAbsMean15m
        self.burstScore15m = burstScore15m
        self.nextHighImpactETAMin = nextHighImpactETAMin
        self.timeSinceLastHighImpactMin = timeSinceLastHighImpactMin
        self.inPreEventWindow = inPreEventWindow
        self.inPostEventWindow = inPostEventWindow
        self.lastSurpriseProxy = lastSurpriseProxy
        self.stale = stale
        self.riskScore = riskScore
        self.reasons = reasons
    }
}

public struct NewsPulsePairState: Identifiable, Hashable, Sendable {
    public let id: String
    public let pair: String
    public let eventETAMin: Int?
    public let newsRiskScore: Double
    public let tradeGate: String
    public let newsPressure: Double
    public let stale: Bool
    public let reasons: [String]

    public init(
        pair: String,
        eventETAMin: Int?,
        newsRiskScore: Double,
        tradeGate: String,
        newsPressure: Double,
        stale: Bool,
        reasons: [String]
    ) {
        id = pair
        self.pair = pair
        self.eventETAMin = eventETAMin
        self.newsRiskScore = newsRiskScore
        self.tradeGate = tradeGate
        self.newsPressure = newsPressure
        self.stale = stale
        self.reasons = reasons
    }
}

public struct NewsPulseRecentItem: Identifiable, Hashable, Sendable {
    public let id: String
    public let source: String
    public let publishedAt: Date?
    public let seenAt: Date?
    public let currencyTags: [String]
    public let topicTags: [String]
    public let domain: String
    public let title: String
    public let url: URL?
    public let importance: String?
    public let tone: Double

    public init(
        id: String,
        source: String,
        publishedAt: Date?,
        seenAt: Date?,
        currencyTags: [String],
        topicTags: [String],
        domain: String,
        title: String,
        url: URL?,
        importance: String?,
        tone: Double
    ) {
        self.id = id
        self.source = source
        self.publishedAt = publishedAt
        self.seenAt = seenAt
        self.currencyTags = currencyTags
        self.topicTags = topicTags
        self.domain = domain
        self.title = title
        self.url = url
        self.importance = importance
        self.tone = tone
    }
}

public struct NewsPulseSnapshot: Sendable {
    public let generatedAt: Date?
    public let queryCount: Int
    public let sourceStatuses: [NewsPulseSourceStatus]
    public let currencies: [NewsPulseCurrencyState]
    public let pairs: [NewsPulsePairState]
    public let recentItems: [NewsPulseRecentItem]
    public let artifactPaths: [KeyValueRecord]

    public init(
        generatedAt: Date?,
        queryCount: Int,
        sourceStatuses: [NewsPulseSourceStatus],
        currencies: [NewsPulseCurrencyState],
        pairs: [NewsPulsePairState],
        recentItems: [NewsPulseRecentItem],
        artifactPaths: [KeyValueRecord]
    ) {
        self.generatedAt = generatedAt
        self.queryCount = queryCount
        self.sourceStatuses = sourceStatuses
        self.currencies = currencies
        self.pairs = pairs
        self.recentItems = recentItems
        self.artifactPaths = artifactPaths
    }

    public var hasBlockingIssue: Bool {
        sourceStatuses.contains(where: { !$0.ok || $0.stale })
    }
}
