import Foundation

public struct NewsPulseSourceStatus: Identifiable, Hashable, Sendable {
    public let id: String
    public let ok: Bool
    public let stale: Bool
    public let enabled: Bool
    public let required: Bool
    public let lastUpdateAt: Date?
    public let lastPollAt: Date?
    public let lastSuccessAt: Date?
    public let backoffUntil: Date?
    public let cursor: String?
    public let lastError: String?
    public let budgetExhausted: Bool
    public let throttled: Bool

    public init(
        id: String,
        ok: Bool,
        stale: Bool,
        enabled: Bool,
        required: Bool,
        lastUpdateAt: Date?,
        lastPollAt: Date?,
        lastSuccessAt: Date?,
        backoffUntil: Date?,
        cursor: String?,
        lastError: String?,
        budgetExhausted: Bool,
        throttled: Bool
    ) {
        self.id = id
        self.ok = ok
        self.stale = stale
        self.enabled = enabled
        self.required = required
        self.lastUpdateAt = lastUpdateAt
        self.lastPollAt = lastPollAt
        self.lastSuccessAt = lastSuccessAt
        self.backoffUntil = backoffUntil
        self.cursor = cursor
        self.lastError = lastError
        self.budgetExhausted = budgetExhausted
        self.throttled = throttled
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
    public let storyCount15m: Int
    public let storySeverity15m: Double
    public let officialCount24h: Int
    public let dominantStoryIDs: [String]
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
        storyCount15m: Int,
        storySeverity15m: Double,
        officialCount24h: Int,
        dominantStoryIDs: [String],
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
        self.storyCount15m = storyCount15m
        self.storySeverity15m = storySeverity15m
        self.officialCount24h = officialCount24h
        self.dominantStoryIDs = dominantStoryIDs
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
    public let baseCurrency: String
    public let quoteCurrency: String
    public let eventETAMin: Int?
    public let newsRiskScore: Double
    public let tradeGate: String
    public let newsPressure: Double
    public let stale: Bool
    public let reasons: [String]
    public let storyIDs: [String]
    public let watchlistTags: [String]
    public let brokerSymbols: [String]
    public let sessionProfile: String
    public let calibrationProfile: String
    public let cautionLotScale: Double?
    public let cautionEnterProbBuffer: Double?
    public let gateChangedAt: Date?

    public init(
        pair: String,
        baseCurrency: String,
        quoteCurrency: String,
        eventETAMin: Int?,
        newsRiskScore: Double,
        tradeGate: String,
        newsPressure: Double,
        stale: Bool,
        reasons: [String],
        storyIDs: [String],
        watchlistTags: [String],
        brokerSymbols: [String],
        sessionProfile: String,
        calibrationProfile: String,
        cautionLotScale: Double?,
        cautionEnterProbBuffer: Double?,
        gateChangedAt: Date?
    ) {
        id = pair
        self.pair = pair
        self.baseCurrency = baseCurrency
        self.quoteCurrency = quoteCurrency
        self.eventETAMin = eventETAMin
        self.newsRiskScore = newsRiskScore
        self.tradeGate = tradeGate
        self.newsPressure = newsPressure
        self.stale = stale
        self.reasons = reasons
        self.storyIDs = storyIDs
        self.watchlistTags = watchlistTags
        self.brokerSymbols = brokerSymbols
        self.sessionProfile = sessionProfile
        self.calibrationProfile = calibrationProfile
        self.cautionLotScale = cautionLotScale
        self.cautionEnterProbBuffer = cautionEnterProbBuffer
        self.gateChangedAt = gateChangedAt
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
    public let storyID: String?

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
        tone: Double,
        storyID: String?
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
        self.storyID = storyID
    }
}

public struct NewsPulseStory: Identifiable, Hashable, Sendable {
    public let id: String
    public let latestTitle: String
    public let firstPublishedAt: Date?
    public let lastPublishedAt: Date?
    public let currencyTags: [String]
    public let topicTags: [String]
    public let domains: [String]
    public let representativeURL: URL?
    public let itemCount: Int
    public let sourceCount: Int
    public let officialHits: Int
    public let severityScore: Double
    public let active: Bool
    public let itemIDs: [String]

    public init(
        id: String,
        latestTitle: String,
        firstPublishedAt: Date?,
        lastPublishedAt: Date?,
        currencyTags: [String],
        topicTags: [String],
        domains: [String],
        representativeURL: URL?,
        itemCount: Int,
        sourceCount: Int,
        officialHits: Int,
        severityScore: Double,
        active: Bool,
        itemIDs: [String]
    ) {
        self.id = id
        self.latestTitle = latestTitle
        self.firstPublishedAt = firstPublishedAt
        self.lastPublishedAt = lastPublishedAt
        self.currencyTags = currencyTags
        self.topicTags = topicTags
        self.domains = domains
        self.representativeURL = representativeURL
        self.itemCount = itemCount
        self.sourceCount = sourceCount
        self.officialHits = officialHits
        self.severityScore = severityScore
        self.active = active
        self.itemIDs = itemIDs
    }
}

public struct NewsPulsePairTimelinePoint: Identifiable, Hashable, Sendable {
    public let id: String
    public let observedAt: Date?
    public let tradeGate: String
    public let newsRiskScore: Double
    public let newsPressure: Double
    public let stale: Bool
    public let eventETAMin: Int?
    public let sessionProfile: String
    public let calibrationProfile: String
    public let watchlistTags: [String]
    public let storyIDs: [String]
    public let reasons: [String]

    public init(
        observedAt: Date?,
        tradeGate: String,
        newsRiskScore: Double,
        newsPressure: Double,
        stale: Bool,
        eventETAMin: Int?,
        sessionProfile: String,
        calibrationProfile: String,
        watchlistTags: [String],
        storyIDs: [String],
        reasons: [String]
    ) {
        id = [observedAt?.ISO8601Format() ?? "unknown", tradeGate, String(format: "%.4f", newsRiskScore)].joined(separator: "::")
        self.observedAt = observedAt
        self.tradeGate = tradeGate
        self.newsRiskScore = newsRiskScore
        self.newsPressure = newsPressure
        self.stale = stale
        self.eventETAMin = eventETAMin
        self.sessionProfile = sessionProfile
        self.calibrationProfile = calibrationProfile
        self.watchlistTags = watchlistTags
        self.storyIDs = storyIDs
        self.reasons = reasons
    }
}

public struct NewsPulseSourceHealthPoint: Identifiable, Hashable, Sendable {
    public let id: String
    public let observedAt: Date?
    public let calendarOK: Bool
    public let calendarStale: Bool
    public let gdeltOK: Bool
    public let gdeltStale: Bool
    public let officialOK: Bool
    public let officialStale: Bool

    public init(
        observedAt: Date?,
        calendarOK: Bool,
        calendarStale: Bool,
        gdeltOK: Bool,
        gdeltStale: Bool,
        officialOK: Bool,
        officialStale: Bool
    ) {
        id = observedAt?.ISO8601Format() ?? UUID().uuidString
        self.observedAt = observedAt
        self.calendarOK = calendarOK
        self.calendarStale = calendarStale
        self.gdeltOK = gdeltOK
        self.gdeltStale = gdeltStale
        self.officialOK = officialOK
        self.officialStale = officialStale
    }
}

public struct NewsPulseDaemonStatus: Hashable, Sendable {
    public let mode: String
    public let heartbeatAt: Date?
    public let lastCycleStartedAt: Date?
    public let lastCycleFinishedAt: Date?
    public let lastCycleDurationSec: Double
    public let intervalSeconds: Int
    public let cyclesCompleted: Int
    public let consecutiveFailures: Int
    public let degraded: Bool
    public let degradedReasons: [String]
    public let lastError: String?

    public init(
        mode: String,
        heartbeatAt: Date?,
        lastCycleStartedAt: Date?,
        lastCycleFinishedAt: Date?,
        lastCycleDurationSec: Double,
        intervalSeconds: Int,
        cyclesCompleted: Int,
        consecutiveFailures: Int,
        degraded: Bool,
        degradedReasons: [String],
        lastError: String?
    ) {
        self.mode = mode
        self.heartbeatAt = heartbeatAt
        self.lastCycleStartedAt = lastCycleStartedAt
        self.lastCycleFinishedAt = lastCycleFinishedAt
        self.lastCycleDurationSec = lastCycleDurationSec
        self.intervalSeconds = intervalSeconds
        self.cyclesCompleted = cyclesCompleted
        self.consecutiveFailures = consecutiveFailures
        self.degraded = degraded
        self.degradedReasons = degradedReasons
        self.lastError = lastError
    }
}

public struct NewsPulsePolicySummary: Hashable, Sendable {
    public let activePairs: [String]
    public let watchlists: [KeyValueListRecord]
    public let brokerSymbolMapCount: Int

    public init(
        activePairs: [String],
        watchlists: [KeyValueListRecord],
        brokerSymbolMapCount: Int
    ) {
        self.activePairs = activePairs
        self.watchlists = watchlists
        self.brokerSymbolMapCount = brokerSymbolMapCount
    }
}

public struct NewsPulseHealthSummary: Hashable, Sendable {
    public let requiredSourcesStale: Bool
    public let gdeltBackoffUntil: Date?
    public let historyRecordsLocal: Int
    public let storyCount: Int

    public init(
        requiredSourcesStale: Bool,
        gdeltBackoffUntil: Date?,
        historyRecordsLocal: Int,
        storyCount: Int
    ) {
        self.requiredSourcesStale = requiredSourcesStale
        self.gdeltBackoffUntil = gdeltBackoffUntil
        self.historyRecordsLocal = historyRecordsLocal
        self.storyCount = storyCount
    }
}

public struct NewsPulseSnapshot: Sendable {
    public let generatedAt: Date?
    public let queryCount: Int
    public let officialQueryCount: Int
    public let sourceStatuses: [NewsPulseSourceStatus]
    public let currencies: [NewsPulseCurrencyState]
    public let pairs: [NewsPulsePairState]
    public let recentItems: [NewsPulseRecentItem]
    public let stories: [NewsPulseStory]
    public let pairTimelines: [String: [NewsPulsePairTimelinePoint]]
    public let sourceHealthTimeline: [NewsPulseSourceHealthPoint]
    public let daemon: NewsPulseDaemonStatus?
    public let policySummary: NewsPulsePolicySummary?
    public let healthSummary: NewsPulseHealthSummary?
    public let artifactPaths: [KeyValueRecord]

    public init(
        generatedAt: Date?,
        queryCount: Int,
        officialQueryCount: Int,
        sourceStatuses: [NewsPulseSourceStatus],
        currencies: [NewsPulseCurrencyState],
        pairs: [NewsPulsePairState],
        recentItems: [NewsPulseRecentItem],
        stories: [NewsPulseStory],
        pairTimelines: [String: [NewsPulsePairTimelinePoint]],
        sourceHealthTimeline: [NewsPulseSourceHealthPoint],
        daemon: NewsPulseDaemonStatus?,
        policySummary: NewsPulsePolicySummary?,
        healthSummary: NewsPulseHealthSummary?,
        artifactPaths: [KeyValueRecord]
    ) {
        self.generatedAt = generatedAt
        self.queryCount = queryCount
        self.officialQueryCount = officialQueryCount
        self.sourceStatuses = sourceStatuses
        self.currencies = currencies
        self.pairs = pairs
        self.recentItems = recentItems
        self.stories = stories
        self.pairTimelines = pairTimelines
        self.sourceHealthTimeline = sourceHealthTimeline
        self.daemon = daemon
        self.policySummary = policySummary
        self.healthSummary = healthSummary
        self.artifactPaths = artifactPaths
    }

    public var hasBlockingIssue: Bool {
        sourceStatuses.contains(where: { $0.required && (!$0.ok || $0.stale) })
    }
}

public struct KeyValueListRecord: Identifiable, Hashable, Sendable {
    public let id: String
    public let key: String
    public let values: [String]

    public init(key: String, values: [String]) {
        id = key
        self.key = key
        self.values = values
    }
}
