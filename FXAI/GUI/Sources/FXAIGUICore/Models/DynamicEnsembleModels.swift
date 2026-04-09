import Foundation

public struct DynamicEnsemblePluginState: Identifiable, Hashable, Sendable {
    public let id: String
    public let name: String
    public let family: String
    public let status: String
    public let signal: String
    public let weight: Double
    public let trust: Double
    public let calibrationShrink: Double
    public let reasons: [String]

    public init(
        name: String,
        family: String,
        status: String,
        signal: String,
        weight: Double,
        trust: Double,
        calibrationShrink: Double,
        reasons: [String]
    ) {
        self.id = "\(name)::\(status)"
        self.name = name
        self.family = family
        self.status = status
        self.signal = signal
        self.weight = weight
        self.trust = trust
        self.calibrationShrink = calibrationShrink
        self.reasons = reasons
    }
}

public struct DynamicEnsembleTransition: Identifiable, Hashable, Sendable {
    public let id: String
    public let type: String
    public let fromValue: String
    public let toValue: String
    public let observedAt: Date?

    public init(type: String, fromValue: String, toValue: String, observedAt: Date?) {
        id = "\(type)::\(fromValue)::\(toValue)::\(observedAt?.ISO8601Format() ?? "unknown")"
        self.type = type
        self.fromValue = fromValue
        self.toValue = toValue
        self.observedAt = observedAt
    }
}

public struct DynamicEnsembleSymbolSnapshot: Identifiable, Hashable, Sendable {
    public let id: String
    public let symbol: String
    public let generatedAt: Date?
    public let topRegime: String
    public let sessionLabel: String
    public let tradePosture: String
    public let ensembleQuality: Double
    public let abstainBias: Double
    public let agreementScore: Double
    public let contextFitScore: Double
    public let dominantPluginShare: Double
    public let buyProb: Double
    public let sellProb: Double
    public let skipProb: Double
    public let finalScore: Double
    public let finalAction: String
    public let fallbackUsed: Bool
    public let reasons: [String]
    public let activePlugins: [DynamicEnsemblePluginState]
    public let downweightedPlugins: [DynamicEnsemblePluginState]
    public let suppressedPlugins: [DynamicEnsemblePluginState]
    public let replayPostureCounts: [KeyValueRecord]
    public let replayActionCounts: [KeyValueRecord]
    public let replayStatusCounts: [KeyValueRecord]
    public let replayTopReasons: [KeyValueRecord]
    public let replayTopDominantPlugins: [KeyValueRecord]
    public let recentTransitions: [DynamicEnsembleTransition]
    public let observationCount: Int
    public let averageQuality: Double
    public let maxAbstainBias: Double

    public init(
        symbol: String,
        generatedAt: Date?,
        topRegime: String,
        sessionLabel: String,
        tradePosture: String,
        ensembleQuality: Double,
        abstainBias: Double,
        agreementScore: Double,
        contextFitScore: Double,
        dominantPluginShare: Double,
        buyProb: Double,
        sellProb: Double,
        skipProb: Double,
        finalScore: Double,
        finalAction: String,
        fallbackUsed: Bool,
        reasons: [String],
        activePlugins: [DynamicEnsemblePluginState],
        downweightedPlugins: [DynamicEnsemblePluginState],
        suppressedPlugins: [DynamicEnsemblePluginState],
        replayPostureCounts: [KeyValueRecord],
        replayActionCounts: [KeyValueRecord],
        replayStatusCounts: [KeyValueRecord],
        replayTopReasons: [KeyValueRecord],
        replayTopDominantPlugins: [KeyValueRecord],
        recentTransitions: [DynamicEnsembleTransition],
        observationCount: Int,
        averageQuality: Double,
        maxAbstainBias: Double
    ) {
        id = symbol
        self.symbol = symbol
        self.generatedAt = generatedAt
        self.topRegime = topRegime
        self.sessionLabel = sessionLabel
        self.tradePosture = tradePosture
        self.ensembleQuality = ensembleQuality
        self.abstainBias = abstainBias
        self.agreementScore = agreementScore
        self.contextFitScore = contextFitScore
        self.dominantPluginShare = dominantPluginShare
        self.buyProb = buyProb
        self.sellProb = sellProb
        self.skipProb = skipProb
        self.finalScore = finalScore
        self.finalAction = finalAction
        self.fallbackUsed = fallbackUsed
        self.reasons = reasons
        self.activePlugins = activePlugins
        self.downweightedPlugins = downweightedPlugins
        self.suppressedPlugins = suppressedPlugins
        self.replayPostureCounts = replayPostureCounts
        self.replayActionCounts = replayActionCounts
        self.replayStatusCounts = replayStatusCounts
        self.replayTopReasons = replayTopReasons
        self.replayTopDominantPlugins = replayTopDominantPlugins
        self.recentTransitions = recentTransitions
        self.observationCount = observationCount
        self.averageQuality = averageQuality
        self.maxAbstainBias = maxAbstainBias
    }
}

public struct DynamicEnsembleSnapshot: Sendable {
    public let generatedAt: Date
    public let replayHoursBack: Int
    public let symbols: [DynamicEnsembleSymbolSnapshot]

    public init(generatedAt: Date, replayHoursBack: Int, symbols: [DynamicEnsembleSymbolSnapshot]) {
        self.generatedAt = generatedAt
        self.replayHoursBack = replayHoursBack
        self.symbols = symbols
    }
}
