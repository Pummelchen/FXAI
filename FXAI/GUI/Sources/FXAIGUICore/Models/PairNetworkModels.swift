import Foundation

public struct PairNetworkDependencyEdge: Identifiable, Hashable, Sendable {
    public let id: String
    public let sourcePair: String
    public let targetPair: String
    public let combinedScore: Double
    public let structuralScore: Double
    public let empiricalScore: Double
    public let correlation: Double
    public let support: Int
    public let relation: String
    public let sharedCurrencies: [String]

    public init(
        sourcePair: String,
        targetPair: String,
        combinedScore: Double,
        structuralScore: Double,
        empiricalScore: Double,
        correlation: Double,
        support: Int,
        relation: String,
        sharedCurrencies: [String]
    ) {
        id = "\(sourcePair)->\(targetPair)"
        self.sourcePair = sourcePair
        self.targetPair = targetPair
        self.combinedScore = combinedScore
        self.structuralScore = structuralScore
        self.empiricalScore = empiricalScore
        self.correlation = correlation
        self.support = support
        self.relation = relation
        self.sharedCurrencies = sharedCurrencies
    }
}

public struct PairNetworkPairSummary: Identifiable, Hashable, Sendable {
    public let id: String
    public let pair: String
    public let baseCurrency: String
    public let quoteCurrency: String
    public let factorSignature: [KeyValueRecord]
    public let topDependencies: [PairNetworkDependencyEdge]

    public init(
        pair: String,
        baseCurrency: String,
        quoteCurrency: String,
        factorSignature: [KeyValueRecord],
        topDependencies: [PairNetworkDependencyEdge]
    ) {
        id = pair
        self.pair = pair
        self.baseCurrency = baseCurrency
        self.quoteCurrency = quoteCurrency
        self.factorSignature = factorSignature
        self.topDependencies = topDependencies
    }
}

public struct PairNetworkSymbolSnapshot: Identifiable, Hashable, Sendable {
    public let id: String
    public let symbol: String
    public let generatedAt: Date?
    public let decision: String
    public let fallbackGraphUsed: Bool
    public let partialDependencyData: Bool
    public let graphStale: Bool
    public let conflictScore: Double
    public let redundancyScore: Double
    public let contradictionScore: Double
    public let concentrationScore: Double
    public let currencyConcentration: Double
    public let factorConcentration: Double
    public let recommendedSizeMultiplier: Double
    public let preferredExpression: String
    public let currencyExposure: [KeyValueRecord]
    public let factorExposure: [KeyValueRecord]
    public let reasons: [String]

    public init(
        symbol: String,
        generatedAt: Date?,
        decision: String,
        fallbackGraphUsed: Bool,
        partialDependencyData: Bool,
        graphStale: Bool,
        conflictScore: Double,
        redundancyScore: Double,
        contradictionScore: Double,
        concentrationScore: Double,
        currencyConcentration: Double,
        factorConcentration: Double,
        recommendedSizeMultiplier: Double,
        preferredExpression: String,
        currencyExposure: [KeyValueRecord],
        factorExposure: [KeyValueRecord],
        reasons: [String]
    ) {
        id = symbol
        self.symbol = symbol
        self.generatedAt = generatedAt
        self.decision = decision
        self.fallbackGraphUsed = fallbackGraphUsed
        self.partialDependencyData = partialDependencyData
        self.graphStale = graphStale
        self.conflictScore = conflictScore
        self.redundancyScore = redundancyScore
        self.contradictionScore = contradictionScore
        self.concentrationScore = concentrationScore
        self.currencyConcentration = currencyConcentration
        self.factorConcentration = factorConcentration
        self.recommendedSizeMultiplier = recommendedSizeMultiplier
        self.preferredExpression = preferredExpression
        self.currencyExposure = currencyExposure
        self.factorExposure = factorExposure
        self.reasons = reasons
    }
}

public struct PairNetworkSnapshot: Sendable {
    public let generatedAt: Date
    public let graphMode: String
    public let actionMode: String
    public let pairCount: Int
    public let currencyCount: Int
    public let edgeCount: Int
    public let fallbackGraphUsed: Bool
    public let partialDependencyData: Bool
    public let graphStale: Bool
    public let symbols: [PairNetworkSymbolSnapshot]
    public let topEdges: [PairNetworkDependencyEdge]
    public let pairSummaries: [PairNetworkPairSummary]
    public let reasons: [String]
    public let qualityFlags: [KeyValueRecord]
    public let statusRecords: [KeyValueRecord]
    public let artifactPaths: [KeyValueRecord]

    public init(
        generatedAt: Date,
        graphMode: String,
        actionMode: String,
        pairCount: Int,
        currencyCount: Int,
        edgeCount: Int,
        fallbackGraphUsed: Bool,
        partialDependencyData: Bool,
        graphStale: Bool,
        symbols: [PairNetworkSymbolSnapshot],
        topEdges: [PairNetworkDependencyEdge],
        pairSummaries: [PairNetworkPairSummary],
        reasons: [String],
        qualityFlags: [KeyValueRecord],
        statusRecords: [KeyValueRecord],
        artifactPaths: [KeyValueRecord]
    ) {
        self.generatedAt = generatedAt
        self.graphMode = graphMode
        self.actionMode = actionMode
        self.pairCount = pairCount
        self.currencyCount = currencyCount
        self.edgeCount = edgeCount
        self.fallbackGraphUsed = fallbackGraphUsed
        self.partialDependencyData = partialDependencyData
        self.graphStale = graphStale
        self.symbols = symbols
        self.topEdges = topEdges
        self.pairSummaries = pairSummaries
        self.reasons = reasons
        self.qualityFlags = qualityFlags
        self.statusRecords = statusRecords
        self.artifactPaths = artifactPaths
    }
}
