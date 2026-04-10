import Foundation

public struct CrossAssetSourceStatus: Identifiable, Hashable, Sendable {
    public let id: String
    public let ok: Bool
    public let stale: Bool
    public let lastUpdateAt: Date?
    public let proxySymbol: String?
    public let availableSymbols: Int?
    public let configuredSymbols: Int?

    public init(
        id: String,
        ok: Bool,
        stale: Bool,
        lastUpdateAt: Date?,
        proxySymbol: String?,
        availableSymbols: Int?,
        configuredSymbols: Int?
    ) {
        self.id = id
        self.ok = ok
        self.stale = stale
        self.lastUpdateAt = lastUpdateAt
        self.proxySymbol = proxySymbol
        self.availableSymbols = availableSymbols
        self.configuredSymbols = configuredSymbols
    }
}

public struct CrossAssetProxySelection: Identifiable, Hashable, Sendable {
    public let id: String
    public let symbol: String
    public let fallbackUsed: Bool
    public let available: Bool
    public let changePct1d: Double
    public let rangeRatio1d: Double

    public init(
        id: String,
        symbol: String,
        fallbackUsed: Bool,
        available: Bool,
        changePct1d: Double,
        rangeRatio1d: Double
    ) {
        self.id = id
        self.symbol = symbol
        self.fallbackUsed = fallbackUsed
        self.available = available
        self.changePct1d = changePct1d
        self.rangeRatio1d = rangeRatio1d
    }
}

public struct CrossAssetPairState: Identifiable, Hashable, Sendable {
    public let id: String
    public let pair: String
    public let baseCurrency: String
    public let quoteCurrency: String
    public let macroState: String
    public let riskState: String
    public let liquidityState: String
    public let pairCrossAssetRiskScore: Double
    public let pairSensitivity: Double
    public let tradeGate: String
    public let stale: Bool
    public let reasons: [String]

    public init(
        pair: String,
        baseCurrency: String,
        quoteCurrency: String,
        macroState: String,
        riskState: String,
        liquidityState: String,
        pairCrossAssetRiskScore: Double,
        pairSensitivity: Double,
        tradeGate: String,
        stale: Bool,
        reasons: [String]
    ) {
        id = pair
        self.pair = pair
        self.baseCurrency = baseCurrency
        self.quoteCurrency = quoteCurrency
        self.macroState = macroState
        self.riskState = riskState
        self.liquidityState = liquidityState
        self.pairCrossAssetRiskScore = pairCrossAssetRiskScore
        self.pairSensitivity = pairSensitivity
        self.tradeGate = tradeGate
        self.stale = stale
        self.reasons = reasons
    }
}

public struct CrossAssetTransition: Identifiable, Hashable, Sendable {
    public let id: String
    public let type: String
    public let target: String
    public let fromValue: String
    public let toValue: String
    public let observedAt: Date?

    public init(type: String, target: String, fromValue: String, toValue: String, observedAt: Date?) {
        self.id = "\(type)::\(target)::\(fromValue)::\(toValue)::\(observedAt?.ISO8601Format() ?? "unknown")"
        self.type = type
        self.target = target
        self.fromValue = fromValue
        self.toValue = toValue
        self.observedAt = observedAt
    }
}

public struct CrossAssetSnapshot: Sendable {
    public let generatedAt: Date
    public let sourceStatuses: [CrossAssetSourceStatus]
    public let features: [KeyValueRecord]
    public let stateScores: [KeyValueRecord]
    public let stateLabels: [KeyValueRecord]
    public let selectedProxies: [CrossAssetProxySelection]
    public let pairs: [CrossAssetPairState]
    public let recentTransitions: [CrossAssetTransition]
    public let reasons: [String]
    public let qualityFlags: [KeyValueRecord]
    public let healthSummary: [KeyValueRecord]
    public let artifactPaths: [KeyValueRecord]

    public init(
        generatedAt: Date,
        sourceStatuses: [CrossAssetSourceStatus],
        features: [KeyValueRecord],
        stateScores: [KeyValueRecord],
        stateLabels: [KeyValueRecord],
        selectedProxies: [CrossAssetProxySelection],
        pairs: [CrossAssetPairState],
        recentTransitions: [CrossAssetTransition],
        reasons: [String],
        qualityFlags: [KeyValueRecord],
        healthSummary: [KeyValueRecord],
        artifactPaths: [KeyValueRecord]
    ) {
        self.generatedAt = generatedAt
        self.sourceStatuses = sourceStatuses
        self.features = features
        self.stateScores = stateScores
        self.stateLabels = stateLabels
        self.selectedProxies = selectedProxies
        self.pairs = pairs
        self.recentTransitions = recentTransitions
        self.reasons = reasons
        self.qualityFlags = qualityFlags
        self.healthSummary = healthSummary
        self.artifactPaths = artifactPaths
    }
}
