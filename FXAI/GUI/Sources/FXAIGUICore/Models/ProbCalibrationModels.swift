import Foundation

public struct ProbCalibrationTransition: Identifiable, Hashable, Sendable {
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

public struct ProbCalibrationSymbolSnapshot: Identifiable, Hashable, Sendable {
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
    public let rawAction: String
    public let rawScore: Double
    public let rawBuyProb: Double
    public let rawSellProb: Double
    public let rawSkipProb: Double
    public let calibratedBuyProb: Double
    public let calibratedSellProb: Double
    public let calibratedSkipProb: Double
    public let calibratedConfidence: Double
    public let expectedMoveMeanPoints: Double
    public let expectedMoveQ25Points: Double
    public let expectedMoveQ50Points: Double
    public let expectedMoveQ75Points: Double
    public let spreadCostPoints: Double
    public let slippageCostPoints: Double
    public let uncertaintyScore: Double
    public let uncertaintyPenaltyPoints: Double
    public let riskPenaltyPoints: Double
    public let expectedGrossEdgePoints: Double
    public let edgeAfterCostsPoints: Double
    public let finalAction: String
    public let abstain: Bool
    public let fallbackUsed: Bool
    public let calibrationStale: Bool
    public let inputStale: Bool
    public let supportUsable: Bool
    public let reasons: [String]
    public let replayActionCounts: [KeyValueRecord]
    public let replayTierCounts: [KeyValueRecord]
    public let replayTopReasons: [KeyValueRecord]
    public let recentTransitions: [ProbCalibrationTransition]
    public let observationCount: Int
    public let abstainCount: Int
    public let fallbackCount: Int
    public let averageConfidence: Double
    public let averageEdgeAfterCostsPoints: Double
    public let averageUncertaintyScore: Double
    public let minEdgeAfterCostsPoints: Double
    public let maxEdgeAfterCostsPoints: Double

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
        rawAction: String,
        rawScore: Double,
        rawBuyProb: Double,
        rawSellProb: Double,
        rawSkipProb: Double,
        calibratedBuyProb: Double,
        calibratedSellProb: Double,
        calibratedSkipProb: Double,
        calibratedConfidence: Double,
        expectedMoveMeanPoints: Double,
        expectedMoveQ25Points: Double,
        expectedMoveQ50Points: Double,
        expectedMoveQ75Points: Double,
        spreadCostPoints: Double,
        slippageCostPoints: Double,
        uncertaintyScore: Double,
        uncertaintyPenaltyPoints: Double,
        riskPenaltyPoints: Double,
        expectedGrossEdgePoints: Double,
        edgeAfterCostsPoints: Double,
        finalAction: String,
        abstain: Bool,
        fallbackUsed: Bool,
        calibrationStale: Bool,
        inputStale: Bool,
        supportUsable: Bool,
        reasons: [String],
        replayActionCounts: [KeyValueRecord],
        replayTierCounts: [KeyValueRecord],
        replayTopReasons: [KeyValueRecord],
        recentTransitions: [ProbCalibrationTransition],
        observationCount: Int,
        abstainCount: Int,
        fallbackCount: Int,
        averageConfidence: Double,
        averageEdgeAfterCostsPoints: Double,
        averageUncertaintyScore: Double,
        minEdgeAfterCostsPoints: Double,
        maxEdgeAfterCostsPoints: Double
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
        self.rawAction = rawAction
        self.rawScore = rawScore
        self.rawBuyProb = rawBuyProb
        self.rawSellProb = rawSellProb
        self.rawSkipProb = rawSkipProb
        self.calibratedBuyProb = calibratedBuyProb
        self.calibratedSellProb = calibratedSellProb
        self.calibratedSkipProb = calibratedSkipProb
        self.calibratedConfidence = calibratedConfidence
        self.expectedMoveMeanPoints = expectedMoveMeanPoints
        self.expectedMoveQ25Points = expectedMoveQ25Points
        self.expectedMoveQ50Points = expectedMoveQ50Points
        self.expectedMoveQ75Points = expectedMoveQ75Points
        self.spreadCostPoints = spreadCostPoints
        self.slippageCostPoints = slippageCostPoints
        self.uncertaintyScore = uncertaintyScore
        self.uncertaintyPenaltyPoints = uncertaintyPenaltyPoints
        self.riskPenaltyPoints = riskPenaltyPoints
        self.expectedGrossEdgePoints = expectedGrossEdgePoints
        self.edgeAfterCostsPoints = edgeAfterCostsPoints
        self.finalAction = finalAction
        self.abstain = abstain
        self.fallbackUsed = fallbackUsed
        self.calibrationStale = calibrationStale
        self.inputStale = inputStale
        self.supportUsable = supportUsable
        self.reasons = reasons
        self.replayActionCounts = replayActionCounts
        self.replayTierCounts = replayTierCounts
        self.replayTopReasons = replayTopReasons
        self.recentTransitions = recentTransitions
        self.observationCount = observationCount
        self.abstainCount = abstainCount
        self.fallbackCount = fallbackCount
        self.averageConfidence = averageConfidence
        self.averageEdgeAfterCostsPoints = averageEdgeAfterCostsPoints
        self.averageUncertaintyScore = averageUncertaintyScore
        self.minEdgeAfterCostsPoints = minEdgeAfterCostsPoints
        self.maxEdgeAfterCostsPoints = maxEdgeAfterCostsPoints
    }
}

public struct ProbCalibrationSnapshot: Sendable {
    public let generatedAt: Date
    public let replayHoursBack: Int
    public let symbols: [ProbCalibrationSymbolSnapshot]

    public init(generatedAt: Date, replayHoursBack: Int, symbols: [ProbCalibrationSymbolSnapshot]) {
        self.generatedAt = generatedAt
        self.replayHoursBack = replayHoursBack
        self.symbols = symbols
    }
}
