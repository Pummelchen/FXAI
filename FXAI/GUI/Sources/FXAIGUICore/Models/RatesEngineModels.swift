import Foundation

public struct RatesEngineSourceStatus: Identifiable, Hashable, Sendable {
    public let id: String
    public let ok: Bool
    public let stale: Bool
    public let enabled: Bool
    public let required: Bool
    public let lastUpdateAt: Date?
    public let mode: String?
    public let coverageRatio: Double?
    public let updatedCurrencies: Int?

    public init(
        id: String,
        ok: Bool,
        stale: Bool,
        enabled: Bool,
        required: Bool,
        lastUpdateAt: Date?,
        mode: String?,
        coverageRatio: Double?,
        updatedCurrencies: Int?
    ) {
        self.id = id
        self.ok = ok
        self.stale = stale
        self.enabled = enabled
        self.required = required
        self.lastUpdateAt = lastUpdateAt
        self.mode = mode
        self.coverageRatio = coverageRatio
        self.updatedCurrencies = updatedCurrencies
    }
}

public struct RatesEngineCurrencyState: Identifiable, Hashable, Sendable {
    public let id: String
    public let currency: String
    public let frontEndLevel: Double?
    public let frontEndBasis: String
    public let frontEndChange1d: Double?
    public let frontEndChange5d: Double?
    public let expectedPathLevel: Double?
    public let expectedPathBasis: String
    public let expectedPathChange1d: Double?
    public let expectedPathChange5d: Double?
    public let curveSlope2s10s: Double?
    public let curveBasis: String
    public let curveShapeRegime: String
    public let policyRepricingScore: Double
    public let policySurpriseScore: Double
    public let policyUncertaintyScore: Double
    public let policyDirectionScore: Double
    public let policyRelevanceScore: Double
    public let preCBEventWindow: Bool
    public let postCBEventWindow: Bool
    public let preMacroPolicyWindow: Bool
    public let postMacroPolicyWindow: Bool
    public let meetingPathRepriceNow: Bool
    public let macroToRatesTransmissionScore: Double
    public let stale: Bool
    public let reasons: [String]

    public init(
        currency: String,
        frontEndLevel: Double?,
        frontEndBasis: String,
        frontEndChange1d: Double?,
        frontEndChange5d: Double?,
        expectedPathLevel: Double?,
        expectedPathBasis: String,
        expectedPathChange1d: Double?,
        expectedPathChange5d: Double?,
        curveSlope2s10s: Double?,
        curveBasis: String,
        curveShapeRegime: String,
        policyRepricingScore: Double,
        policySurpriseScore: Double,
        policyUncertaintyScore: Double,
        policyDirectionScore: Double,
        policyRelevanceScore: Double,
        preCBEventWindow: Bool,
        postCBEventWindow: Bool,
        preMacroPolicyWindow: Bool,
        postMacroPolicyWindow: Bool,
        meetingPathRepriceNow: Bool,
        macroToRatesTransmissionScore: Double,
        stale: Bool,
        reasons: [String]
    ) {
        id = currency
        self.currency = currency
        self.frontEndLevel = frontEndLevel
        self.frontEndBasis = frontEndBasis
        self.frontEndChange1d = frontEndChange1d
        self.frontEndChange5d = frontEndChange5d
        self.expectedPathLevel = expectedPathLevel
        self.expectedPathBasis = expectedPathBasis
        self.expectedPathChange1d = expectedPathChange1d
        self.expectedPathChange5d = expectedPathChange5d
        self.curveSlope2s10s = curveSlope2s10s
        self.curveBasis = curveBasis
        self.curveShapeRegime = curveShapeRegime
        self.policyRepricingScore = policyRepricingScore
        self.policySurpriseScore = policySurpriseScore
        self.policyUncertaintyScore = policyUncertaintyScore
        self.policyDirectionScore = policyDirectionScore
        self.policyRelevanceScore = policyRelevanceScore
        self.preCBEventWindow = preCBEventWindow
        self.postCBEventWindow = postCBEventWindow
        self.preMacroPolicyWindow = preMacroPolicyWindow
        self.postMacroPolicyWindow = postMacroPolicyWindow
        self.meetingPathRepriceNow = meetingPathRepriceNow
        self.macroToRatesTransmissionScore = macroToRatesTransmissionScore
        self.stale = stale
        self.reasons = reasons
    }
}

public struct RatesEnginePairState: Identifiable, Hashable, Sendable {
    public let id: String
    public let pair: String
    public let baseCurrency: String
    public let quoteCurrency: String
    public let frontEndDiff: Double?
    public let expectedPathDiff: Double?
    public let curveDivergenceScore: Double
    public let policyDivergenceScore: Double
    public let ratesRegime: String
    public let ratesRiskScore: Double
    public let tradeGate: String
    public let policyAlignment: String
    public let meetingPathRepriceNow: Bool
    public let macroToRatesTransmissionScore: Double
    public let stale: Bool
    public let brokerSymbols: [String]
    public let reasons: [String]

    public init(
        pair: String,
        baseCurrency: String,
        quoteCurrency: String,
        frontEndDiff: Double?,
        expectedPathDiff: Double?,
        curveDivergenceScore: Double,
        policyDivergenceScore: Double,
        ratesRegime: String,
        ratesRiskScore: Double,
        tradeGate: String,
        policyAlignment: String,
        meetingPathRepriceNow: Bool,
        macroToRatesTransmissionScore: Double,
        stale: Bool,
        brokerSymbols: [String],
        reasons: [String]
    ) {
        id = pair
        self.pair = pair
        self.baseCurrency = baseCurrency
        self.quoteCurrency = quoteCurrency
        self.frontEndDiff = frontEndDiff
        self.expectedPathDiff = expectedPathDiff
        self.curveDivergenceScore = curveDivergenceScore
        self.policyDivergenceScore = policyDivergenceScore
        self.ratesRegime = ratesRegime
        self.ratesRiskScore = ratesRiskScore
        self.tradeGate = tradeGate
        self.policyAlignment = policyAlignment
        self.meetingPathRepriceNow = meetingPathRepriceNow
        self.macroToRatesTransmissionScore = macroToRatesTransmissionScore
        self.stale = stale
        self.brokerSymbols = brokerSymbols
        self.reasons = reasons
    }
}

public struct RatesEnginePolicyEvent: Identifiable, Hashable, Sendable {
    public let id: String
    public let currency: String
    public let source: String
    public let domain: String
    public let publishedAt: Date?
    public let title: String
    public let url: URL?
    public let policyRelevanceScore: Double
    public let direction: Double
    public let centralBankEvent: Bool
    public let macroPolicyEvent: Bool

    public init(
        id: String,
        currency: String,
        source: String,
        domain: String,
        publishedAt: Date?,
        title: String,
        url: URL?,
        policyRelevanceScore: Double,
        direction: Double,
        centralBankEvent: Bool,
        macroPolicyEvent: Bool
    ) {
        self.id = id
        self.currency = currency
        self.source = source
        self.domain = domain
        self.publishedAt = publishedAt
        self.title = title
        self.url = url
        self.policyRelevanceScore = policyRelevanceScore
        self.direction = direction
        self.centralBankEvent = centralBankEvent
        self.macroPolicyEvent = macroPolicyEvent
    }
}

public struct RatesEngineSnapshot: Sendable {
    public let generatedAt: Date
    public let sourceStatuses: [RatesEngineSourceStatus]
    public let currencies: [RatesEngineCurrencyState]
    public let pairs: [RatesEnginePairState]
    public let recentPolicyEvents: [RatesEnginePolicyEvent]
    public let healthSummary: [KeyValueRecord]
    public let artifactPaths: [KeyValueRecord]

    public init(
        generatedAt: Date,
        sourceStatuses: [RatesEngineSourceStatus],
        currencies: [RatesEngineCurrencyState],
        pairs: [RatesEnginePairState],
        recentPolicyEvents: [RatesEnginePolicyEvent],
        healthSummary: [KeyValueRecord],
        artifactPaths: [KeyValueRecord]
    ) {
        self.generatedAt = generatedAt
        self.sourceStatuses = sourceStatuses
        self.currencies = currencies
        self.pairs = pairs
        self.recentPolicyEvents = recentPolicyEvents
        self.healthSummary = healthSummary
        self.artifactPaths = artifactPaths
    }
}
