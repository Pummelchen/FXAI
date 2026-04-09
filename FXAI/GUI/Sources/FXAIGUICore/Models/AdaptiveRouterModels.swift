import Foundation

public struct AdaptiveRouterProbabilityRecord: Identifiable, Hashable, Sendable {
    public let id: String
    public let label: String
    public let probability: Double

    public init(label: String, probability: Double) {
        self.id = label
        self.label = label
        self.probability = probability
    }
}

public struct AdaptiveRouterPluginState: Identifiable, Hashable, Sendable {
    public let id: String
    public let name: String
    public let weight: Double
    public let suitability: Double
    public let status: String
    public let reasons: [String]

    public init(
        name: String,
        weight: Double,
        suitability: Double,
        status: String,
        reasons: [String]
    ) {
        self.id = "\(name)::\(status)"
        self.name = name
        self.weight = weight
        self.suitability = suitability
        self.status = status
        self.reasons = reasons
    }
}

public struct AdaptiveRouterTransition: Identifiable, Hashable, Sendable {
    public let id: String
    public let type: String
    public let fromValue: String
    public let toValue: String
    public let observedAt: Date?

    public init(
        type: String,
        fromValue: String,
        toValue: String,
        observedAt: Date?
    ) {
        self.id = "\(type)::\(fromValue)::\(toValue)::\(observedAt?.ISO8601Format() ?? "unknown")"
        self.type = type
        self.fromValue = fromValue
        self.toValue = toValue
        self.observedAt = observedAt
    }
}

public struct AdaptiveRouterSymbolSnapshot: Identifiable, Hashable, Sendable {
    public let id: String
    public let symbol: String
    public let profileName: String?
    public let routerMode: String
    public let topRegime: String
    public let confidence: Double
    public let tradePosture: String
    public let abstainBias: Double
    public let sessionLabel: String
    public let spreadRegime: String
    public let volatilityRegime: String
    public let newsRiskScore: Double
    public let newsPressure: Double
    public let eventETAMin: Int?
    public let staleNews: Bool
    public let liquidityStress: Double
    public let breakoutPressure: Double
    public let trendStrength: Double
    public let rangePressure: Double
    public let macroPressure: Double
    public let generatedAt: Date?
    public let profileGeneratedAt: Date?
    public let reasons: [String]
    public let probabilities: [AdaptiveRouterProbabilityRecord]
    public let activePlugins: [AdaptiveRouterPluginState]
    public let downweightedPlugins: [AdaptiveRouterPluginState]
    public let suppressedPlugins: [AdaptiveRouterPluginState]
    public let pairTags: [String]
    public let topProfilePlugins: [String]
    public let thresholdMetrics: [KeyValueRecord]
    public let regimeBiasMetrics: [KeyValueRecord]
    public let replayRegimeCounts: [KeyValueRecord]
    public let replayPostureCounts: [KeyValueRecord]
    public let replayTopReasons: [KeyValueRecord]
    public let replayTopPlugins: [KeyValueRecord]
    public let recentTransitions: [AdaptiveRouterTransition]
    public let observationCount: Int

    public init(
        symbol: String,
        profileName: String?,
        routerMode: String,
        topRegime: String,
        confidence: Double,
        tradePosture: String,
        abstainBias: Double,
        sessionLabel: String,
        spreadRegime: String,
        volatilityRegime: String,
        newsRiskScore: Double,
        newsPressure: Double,
        eventETAMin: Int?,
        staleNews: Bool,
        liquidityStress: Double,
        breakoutPressure: Double,
        trendStrength: Double,
        rangePressure: Double,
        macroPressure: Double,
        generatedAt: Date?,
        profileGeneratedAt: Date?,
        reasons: [String],
        probabilities: [AdaptiveRouterProbabilityRecord],
        activePlugins: [AdaptiveRouterPluginState],
        downweightedPlugins: [AdaptiveRouterPluginState],
        suppressedPlugins: [AdaptiveRouterPluginState],
        pairTags: [String],
        topProfilePlugins: [String],
        thresholdMetrics: [KeyValueRecord],
        regimeBiasMetrics: [KeyValueRecord],
        replayRegimeCounts: [KeyValueRecord],
        replayPostureCounts: [KeyValueRecord],
        replayTopReasons: [KeyValueRecord],
        replayTopPlugins: [KeyValueRecord],
        recentTransitions: [AdaptiveRouterTransition],
        observationCount: Int
    ) {
        id = symbol
        self.symbol = symbol
        self.profileName = profileName
        self.routerMode = routerMode
        self.topRegime = topRegime
        self.confidence = confidence
        self.tradePosture = tradePosture
        self.abstainBias = abstainBias
        self.sessionLabel = sessionLabel
        self.spreadRegime = spreadRegime
        self.volatilityRegime = volatilityRegime
        self.newsRiskScore = newsRiskScore
        self.newsPressure = newsPressure
        self.eventETAMin = eventETAMin
        self.staleNews = staleNews
        self.liquidityStress = liquidityStress
        self.breakoutPressure = breakoutPressure
        self.trendStrength = trendStrength
        self.rangePressure = rangePressure
        self.macroPressure = macroPressure
        self.generatedAt = generatedAt
        self.profileGeneratedAt = profileGeneratedAt
        self.reasons = reasons
        self.probabilities = probabilities
        self.activePlugins = activePlugins
        self.downweightedPlugins = downweightedPlugins
        self.suppressedPlugins = suppressedPlugins
        self.pairTags = pairTags
        self.topProfilePlugins = topProfilePlugins
        self.thresholdMetrics = thresholdMetrics
        self.regimeBiasMetrics = regimeBiasMetrics
        self.replayRegimeCounts = replayRegimeCounts
        self.replayPostureCounts = replayPostureCounts
        self.replayTopReasons = replayTopReasons
        self.replayTopPlugins = replayTopPlugins
        self.recentTransitions = recentTransitions
        self.observationCount = observationCount
    }
}

public struct AdaptiveRouterSnapshot: Sendable {
    public let generatedAt: Date
    public let profileName: String?
    public let replayHoursBack: Int
    public let symbols: [AdaptiveRouterSymbolSnapshot]

    public init(
        generatedAt: Date,
        profileName: String?,
        replayHoursBack: Int,
        symbols: [AdaptiveRouterSymbolSnapshot]
    ) {
        self.generatedAt = generatedAt
        self.profileName = profileName
        self.replayHoursBack = replayHoursBack
        self.symbols = symbols
    }
}
