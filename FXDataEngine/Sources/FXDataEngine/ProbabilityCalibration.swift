import Foundation

public enum ProbabilityCalibrationConstants {
    public static let maxReasons = 10
    public static let maxBuckets = 4
    public static let maxTiers = 96
    public static let defaultFreshnessMaxSeconds: Int64 = 180
    public static let runtimeDirectory = "FXAI/Runtime"
}

public struct ProbabilityCalibrationConfig: Codable, Hashable, Sendable {
    public var ready: Bool
    public var enabled: Bool
    public var allowAbstainFlag: Bool
    public var neutralBlendGain: Double
    public var skipUncertaintyGain: Double
    public var skipCalibrationCredit: Double
    public var skipFloor: Double
    public var skipCap: Double
    public var baseUncertaintyScore: Double
    public var supportSoftFloor: Int
    public var supportHardFloor: Int
    public var memoryStaleAfterHours: Int
    public var minCalibrationQuality: Double
    public var maxUncertaintyScore: Double
    public var signalZeroBand: Double
    public var edgeFloorMultiplier: Double
    public var tradeEdgeFloorPoints: Double
    public var softProbabilityScale: Double
    public var softSkipBias: Double
    public var softMoveMeanScale: Double
    public var softMoveQ25Scale: Double
    public var softMoveQ50Scale: Double
    public var softMoveQ75Scale: Double
    public var softConfidenceCap: Double
    public var uncertaintySupportPenalty: Double
    public var uncertaintyQualityPenalty: Double
    public var uncertaintyDisagreementPenalty: Double
    public var uncertaintyDistributionWidthPenalty: Double
    public var uncertaintyNewsPenalty: Double
    public var uncertaintyRatesPenalty: Double
    public var uncertaintyMicroPenalty: Double
    public var uncertaintyDynamicAbstainPenalty: Double
    public var uncertaintyAdaptiveAbstainPenalty: Double
    public var uncertaintyStaleContextPenalty: Double
    public var riskNewsBlockMultiplier: Double
    public var riskRatesBlockMultiplier: Double
    public var riskMicroBlockMultiplier: Double
    public var riskCautionPostureMultiplier: Double
    public var riskAbstainPostureMultiplier: Double
    public var riskBlockPostureMultiplier: Double
    public var riskFillMultiplier: Double
    public var riskPathMultiplier: Double
    public var bucketCount: Int
    public var bucketHierarchy: [String]

    public init(
        ready: Bool = true,
        enabled: Bool = true,
        allowAbstainFlag: Bool = true,
        neutralBlendGain: Double = 0.65,
        skipUncertaintyGain: Double = 0.12,
        skipCalibrationCredit: Double = 0.05,
        skipFloor: Double = 0.02,
        skipCap: Double = 0.96,
        baseUncertaintyScore: Double = 0.18,
        supportSoftFloor: Int = 64,
        supportHardFloor: Int = 16,
        memoryStaleAfterHours: Int = 96,
        minCalibrationQuality: Double = 0.44,
        maxUncertaintyScore: Double = 0.92,
        signalZeroBand: Double = 0.035,
        edgeFloorMultiplier: Double = 0.08,
        tradeEdgeFloorPoints: Double = 0.05,
        softProbabilityScale: Double = 1.60,
        softSkipBias: Double = 0.08,
        softMoveMeanScale: Double = 0.78,
        softMoveQ25Scale: Double = 0.60,
        softMoveQ50Scale: Double = 0.72,
        softMoveQ75Scale: Double = 0.88,
        softConfidenceCap: Double = 0.58,
        uncertaintySupportPenalty: Double = 0.34,
        uncertaintyQualityPenalty: Double = 0.28,
        uncertaintyDisagreementPenalty: Double = 0.26,
        uncertaintyDistributionWidthPenalty: Double = 0.22,
        uncertaintyNewsPenalty: Double = 0.18,
        uncertaintyRatesPenalty: Double = 0.14,
        uncertaintyMicroPenalty: Double = 0.24,
        uncertaintyDynamicAbstainPenalty: Double = 0.20,
        uncertaintyAdaptiveAbstainPenalty: Double = 0.22,
        uncertaintyStaleContextPenalty: Double = 0.16,
        riskNewsBlockMultiplier: Double = 0.32,
        riskRatesBlockMultiplier: Double = 0.24,
        riskMicroBlockMultiplier: Double = 0.36,
        riskCautionPostureMultiplier: Double = 0.14,
        riskAbstainPostureMultiplier: Double = 0.24,
        riskBlockPostureMultiplier: Double = 0.42,
        riskFillMultiplier: Double = 0.20,
        riskPathMultiplier: Double = 0.16,
        bucketCount: Int = 4,
        bucketHierarchy: [String] = [
            "PAIR_SESSION_REGIME",
            "PAIR_REGIME",
            "REGIME",
            "GLOBAL"
        ]
    ) {
        self.ready = ready
        self.enabled = enabled
        self.allowAbstainFlag = allowAbstainFlag
        self.neutralBlendGain = fxSafeFinite(neutralBlendGain)
        self.skipUncertaintyGain = fxSafeFinite(skipUncertaintyGain)
        self.skipCalibrationCredit = fxSafeFinite(skipCalibrationCredit)
        self.skipFloor = fxClamp(skipFloor, 0.0, 1.0)
        self.skipCap = fxClamp(skipCap, self.skipFloor, 1.0)
        self.baseUncertaintyScore = fxSafeFinite(baseUncertaintyScore)
        self.supportSoftFloor = max(0, supportSoftFloor)
        self.supportHardFloor = max(0, supportHardFloor)
        self.memoryStaleAfterHours = max(0, memoryStaleAfterHours)
        self.minCalibrationQuality = fxClamp(minCalibrationQuality, 0.0, 1.0)
        self.maxUncertaintyScore = fxClamp(maxUncertaintyScore, 0.0, 2.5)
        self.signalZeroBand = max(0.0, fxSafeFinite(signalZeroBand))
        self.edgeFloorMultiplier = max(0.0, fxSafeFinite(edgeFloorMultiplier))
        self.tradeEdgeFloorPoints = max(0.0, fxSafeFinite(tradeEdgeFloorPoints))
        self.softProbabilityScale = max(1e-6, fxSafeFinite(softProbabilityScale, fallback: 1.60))
        self.softSkipBias = fxClamp(softSkipBias, 0.0, 1.0)
        self.softMoveMeanScale = max(0.0, fxSafeFinite(softMoveMeanScale))
        self.softMoveQ25Scale = max(0.0, fxSafeFinite(softMoveQ25Scale))
        self.softMoveQ50Scale = max(0.0, fxSafeFinite(softMoveQ50Scale))
        self.softMoveQ75Scale = max(0.0, fxSafeFinite(softMoveQ75Scale))
        self.softConfidenceCap = fxClamp(softConfidenceCap, 0.50, 0.95)
        self.uncertaintySupportPenalty = fxSafeFinite(uncertaintySupportPenalty)
        self.uncertaintyQualityPenalty = fxSafeFinite(uncertaintyQualityPenalty)
        self.uncertaintyDisagreementPenalty = fxSafeFinite(uncertaintyDisagreementPenalty)
        self.uncertaintyDistributionWidthPenalty = fxSafeFinite(uncertaintyDistributionWidthPenalty)
        self.uncertaintyNewsPenalty = fxSafeFinite(uncertaintyNewsPenalty)
        self.uncertaintyRatesPenalty = fxSafeFinite(uncertaintyRatesPenalty)
        self.uncertaintyMicroPenalty = fxSafeFinite(uncertaintyMicroPenalty)
        self.uncertaintyDynamicAbstainPenalty = fxSafeFinite(uncertaintyDynamicAbstainPenalty)
        self.uncertaintyAdaptiveAbstainPenalty = fxSafeFinite(uncertaintyAdaptiveAbstainPenalty)
        self.uncertaintyStaleContextPenalty = fxSafeFinite(uncertaintyStaleContextPenalty)
        self.riskNewsBlockMultiplier = fxSafeFinite(riskNewsBlockMultiplier)
        self.riskRatesBlockMultiplier = fxSafeFinite(riskRatesBlockMultiplier)
        self.riskMicroBlockMultiplier = fxSafeFinite(riskMicroBlockMultiplier)
        self.riskCautionPostureMultiplier = fxSafeFinite(riskCautionPostureMultiplier)
        self.riskAbstainPostureMultiplier = fxSafeFinite(riskAbstainPostureMultiplier)
        self.riskBlockPostureMultiplier = fxSafeFinite(riskBlockPostureMultiplier)
        self.riskFillMultiplier = fxSafeFinite(riskFillMultiplier)
        self.riskPathMultiplier = fxSafeFinite(riskPathMultiplier)
        self.bucketCount = Int(fxClamp(Double(bucketCount), 0.0, Double(ProbabilityCalibrationConstants.maxBuckets)))
        self.bucketHierarchy = Self.normalizedBuckets(bucketHierarchy)
    }

    public var effectiveBucketHierarchy: [String] {
        Array(bucketHierarchy.prefix(bucketCount))
    }

    private static func normalizedBuckets(_ values: [String]) -> [String] {
        var output = values
            .prefix(ProbabilityCalibrationConstants.maxBuckets)
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines).uppercased() }
        if output.count < ProbabilityCalibrationConstants.maxBuckets {
            output.append(contentsOf: Array(
                repeating: "",
                count: ProbabilityCalibrationConstants.maxBuckets - output.count
            ))
        }
        return Array(output)
    }
}

public struct ProbabilityCalibrationTier: Codable, Hashable, Sendable {
    public var ready: Bool
    public var kind: String
    public var symbol: String
    public var session: String
    public var regime: String
    public var support: Int
    public var probabilityScale: Double
    public var probabilityBias: Double
    public var skipBias: Double
    public var moveMeanScale: Double
    public var moveQ25Scale: Double
    public var moveQ50Scale: Double
    public var moveQ75Scale: Double
    public var calibrationQuality: Double
    public var uncertaintyMultiplier: Double
    public var confidenceCap: Double

    public init(
        ready: Bool = false,
        kind: String = "GLOBAL",
        symbol: String = "*",
        session: String = "*",
        regime: String = "*",
        support: Int = 0,
        probabilityScale: Double = 1.60,
        probabilityBias: Double = 0.0,
        skipBias: Double = 0.08,
        moveMeanScale: Double = 0.78,
        moveQ25Scale: Double = 0.60,
        moveQ50Scale: Double = 0.72,
        moveQ75Scale: Double = 0.88,
        calibrationQuality: Double = 0.34,
        uncertaintyMultiplier: Double = 1.30,
        confidenceCap: Double = 0.58
    ) {
        self.ready = ready
        self.kind = Self.normalizedToken(kind, fallback: "GLOBAL")
        self.symbol = Self.normalizedToken(symbol, fallback: "*")
        self.session = Self.normalizedToken(session, fallback: "*")
        self.regime = Self.normalizedToken(regime, fallback: "*")
        self.support = max(0, support)
        self.probabilityScale = max(1e-6, fxSafeFinite(probabilityScale, fallback: 1.60))
        self.probabilityBias = fxSafeFinite(probabilityBias)
        self.skipBias = fxSafeFinite(skipBias)
        self.moveMeanScale = max(0.0, fxSafeFinite(moveMeanScale))
        self.moveQ25Scale = max(0.0, fxSafeFinite(moveQ25Scale))
        self.moveQ50Scale = max(0.0, fxSafeFinite(moveQ50Scale))
        self.moveQ75Scale = max(0.0, fxSafeFinite(moveQ75Scale))
        self.calibrationQuality = fxSafeFinite(calibrationQuality)
        self.uncertaintyMultiplier = fxSafeFinite(uncertaintyMultiplier, fallback: 1.30)
        self.confidenceCap = fxSafeFinite(confidenceCap, fallback: 0.58)
    }

    public var key: String {
        "\(kind)|\(symbol)|\(session)|\(regime)"
    }

    public static var reset: ProbabilityCalibrationTier {
        ProbabilityCalibrationTier()
    }

    public static func fallback(config: ProbabilityCalibrationConfig) -> ProbabilityCalibrationTier {
        ProbabilityCalibrationTier(
            ready: true,
            kind: "GLOBAL",
            symbol: "*",
            session: "*",
            regime: "*",
            support: 0,
            probabilityScale: config.softProbabilityScale,
            probabilityBias: 0.0,
            skipBias: config.softSkipBias,
            moveMeanScale: config.softMoveMeanScale,
            moveQ25Scale: config.softMoveQ25Scale,
            moveQ50Scale: config.softMoveQ50Scale,
            moveQ75Scale: config.softMoveQ75Scale,
            calibrationQuality: 0.34,
            uncertaintyMultiplier: 1.30,
            confidenceCap: config.softConfidenceCap
        )
    }

    private static func normalizedToken(_ raw: String, fallback: String) -> String {
        let value = raw.trimmingCharacters(in: .whitespacesAndNewlines).uppercased()
        return value.isEmpty ? fallback : value
    }
}

public struct ProbabilityCalibrationMemory: Codable, Hashable, Sendable {
    public var generatedAt: Int64
    public var defaultMethod: String
    public var tiers: [ProbabilityCalibrationTier]

    public init(
        generatedAt: Int64 = 0,
        defaultMethod: String = "LOGISTIC_AFFINE",
        tiers: [ProbabilityCalibrationTier] = []
    ) {
        self.generatedAt = max(0, generatedAt)
        self.defaultMethod = defaultMethod.isEmpty ? "LOGISTIC_AFFINE" : defaultMethod
        self.tiers = Array(tiers.prefix(ProbabilityCalibrationConstants.maxTiers))
    }
}

public struct ProbabilityCalibrationTierSelection: Codable, Hashable, Sendable {
    public var tier: ProbabilityCalibrationTier
    public var found: Bool
    public var fallbackUsed: Bool
    public var supportUsable: Bool

    public init(
        tier: ProbabilityCalibrationTier = .reset,
        found: Bool = false,
        fallbackUsed: Bool = true,
        supportUsable: Bool = false
    ) {
        self.tier = tier
        self.found = found
        self.fallbackUsed = fallbackUsed
        self.supportUsable = supportUsable
    }
}

public struct ProbabilityCalibrationPolicyInputs: Codable, Hashable, Sendable {
    public var symbol: String
    public var generatedAtUTC: Int64
    public var sessionLabel: String?
    public var regimeLabel: String?
    public var rawBuyProbability: Double
    public var rawSellProbability: Double
    public var rawSkipProbability: Double
    public var moveMeanPoints: Double
    public var moveQ25Points: Double
    public var moveQ50Points: Double
    public var moveQ75Points: Double
    public var agreementScore: Double
    public var minMovePoints: Double
    public var priceCostPoints: Double
    public var commissionPoints: Double
    public var costBufferPoints: Double
    public var horizonMinutes: Int
    public var upstreamDecision: Int
    public var pathRisk: Double
    public var fillRisk: Double
    public var dynamicStateReady: Bool
    public var dynamicTradePosture: String
    public var dynamicAbstainBias: Double
    public var adaptiveRouterPosture: String
    public var adaptiveRouterAbstainBias: Double
    public var executionQualityEnabled: Bool
    public var newsState: NewsPulsePairState
    public var ratesState: RatesEnginePairState
    public var crossAssetState: CrossAssetPairState
    public var microstructureState: MicrostructurePairState
    public var executionQualityState: ExecutionQualityPairState

    public init(
        symbol: String,
        generatedAtUTC: Int64,
        sessionLabel: String? = nil,
        regimeLabel: String? = nil,
        rawBuyProbability: Double = 0.0,
        rawSellProbability: Double = 0.0,
        rawSkipProbability: Double = 1.0,
        moveMeanPoints: Double = 0.0,
        moveQ25Points: Double = 0.0,
        moveQ50Points: Double = 0.0,
        moveQ75Points: Double = 0.0,
        agreementScore: Double = 0.0,
        minMovePoints: Double = 0.0,
        priceCostPoints: Double = 0.0,
        commissionPoints: Double = 0.0,
        costBufferPoints: Double = 0.0,
        horizonMinutes: Int = 1,
        upstreamDecision: Int = -1,
        pathRisk: Double = 0.0,
        fillRisk: Double = 0.0,
        dynamicStateReady: Bool = false,
        dynamicTradePosture: String = "UNKNOWN",
        dynamicAbstainBias: Double = 0.0,
        adaptiveRouterPosture: String = "UNKNOWN",
        adaptiveRouterAbstainBias: Double = 0.0,
        executionQualityEnabled: Bool = true,
        newsState: NewsPulsePairState = .reset,
        ratesState: RatesEnginePairState = .reset,
        crossAssetState: CrossAssetPairState = .reset,
        microstructureState: MicrostructurePairState = .reset,
        executionQualityState: ExecutionQualityPairState = .reset
    ) {
        self.symbol = symbol.uppercased()
        self.generatedAtUTC = max(0, generatedAtUTC)
        self.sessionLabel = sessionLabel
        self.regimeLabel = regimeLabel
        self.rawBuyProbability = fxClamp(rawBuyProbability, 0.0, 1.0)
        self.rawSellProbability = fxClamp(rawSellProbability, 0.0, 1.0)
        self.rawSkipProbability = fxClamp(rawSkipProbability, 0.0, 1.0)
        self.moveMeanPoints = max(0.0, fxSafeFinite(moveMeanPoints))
        self.moveQ25Points = max(0.0, fxSafeFinite(moveQ25Points))
        self.moveQ50Points = max(0.0, fxSafeFinite(moveQ50Points))
        self.moveQ75Points = max(0.0, fxSafeFinite(moveQ75Points))
        self.agreementScore = fxClamp(agreementScore, 0.0, 1.0)
        self.minMovePoints = max(0.0, fxSafeFinite(minMovePoints))
        self.priceCostPoints = max(0.0, fxSafeFinite(priceCostPoints))
        self.commissionPoints = max(0.0, fxSafeFinite(commissionPoints))
        self.costBufferPoints = max(0.0, fxSafeFinite(costBufferPoints))
        self.horizonMinutes = max(1, horizonMinutes)
        self.upstreamDecision = upstreamDecision
        self.pathRisk = fxClamp(pathRisk, 0.0, 1.0)
        self.fillRisk = fxClamp(fillRisk, 0.0, 1.0)
        self.dynamicStateReady = dynamicStateReady
        self.dynamicTradePosture = dynamicTradePosture.isEmpty ? "UNKNOWN" : dynamicTradePosture.uppercased()
        self.dynamicAbstainBias = fxClamp(dynamicAbstainBias, 0.0, 1.0)
        self.adaptiveRouterPosture = adaptiveRouterPosture.isEmpty ? "UNKNOWN" : adaptiveRouterPosture.uppercased()
        self.adaptiveRouterAbstainBias = fxClamp(adaptiveRouterAbstainBias, 0.0, 1.0)
        self.executionQualityEnabled = executionQualityEnabled
        self.newsState = newsState
        self.ratesState = ratesState
        self.crossAssetState = crossAssetState
        self.microstructureState = microstructureState
        self.executionQualityState = executionQualityState
    }
}

public struct ProbabilityCalibrationRuntimeState: Codable, Hashable, Sendable {
    public var ready: Bool
    public var available: Bool
    public var stale: Bool
    public var fallbackUsed: Bool
    public var calibrationStale: Bool
    public var inputStale: Bool
    public var newsRiskBlock: Bool
    public var ratesRiskBlock: Bool
    public var microstructureStress: Bool
    public var supportUsable: Bool
    public var generatedAt: Int64
    public var symbol: String
    public var method: String
    public var sessionLabel: String
    public var regimeLabel: String
    public var selectedTierKind: String
    public var selectedTierKey: String
    public var selectedSupport: Int
    public var selectedQuality: Double
    public var rawBuyProbability: Double
    public var rawSellProbability: Double
    public var rawSkipProbability: Double
    public var rawScore: Double
    public var rawAction: String
    public var calibratedBuyProbability: Double
    public var calibratedSellProbability: Double
    public var calibratedSkipProbability: Double
    public var calibratedConfidence: Double
    public var expectedMoveMeanPoints: Double
    public var expectedMoveQ25Points: Double
    public var expectedMoveQ50Points: Double
    public var expectedMoveQ75Points: Double
    public var priceCostPoints: Double
    public var slippageCostPoints: Double
    public var uncertaintyScore: Double
    public var uncertaintyPenaltyPoints: Double
    public var riskPenaltyPoints: Double
    public var expectedGrossEdgePoints: Double
    public var edgeAfterCostsPoints: Double
    public var finalAction: String
    public var abstain: Bool
    public var reasons: [String]

    public init(
        ready: Bool = false,
        available: Bool = false,
        stale: Bool = true,
        fallbackUsed: Bool = false,
        calibrationStale: Bool = true,
        inputStale: Bool = true,
        newsRiskBlock: Bool = false,
        ratesRiskBlock: Bool = false,
        microstructureStress: Bool = false,
        supportUsable: Bool = false,
        generatedAt: Int64 = 0,
        symbol: String = "",
        method: String = "LOGISTIC_AFFINE",
        sessionLabel: String = "UNKNOWN",
        regimeLabel: String = "UNKNOWN",
        selectedTierKind: String = "GLOBAL",
        selectedTierKey: String = "GLOBAL|*|*|*",
        selectedSupport: Int = 0,
        selectedQuality: Double = 0.0,
        rawBuyProbability: Double = 0.0,
        rawSellProbability: Double = 0.0,
        rawSkipProbability: Double = 1.0,
        rawScore: Double = 0.0,
        rawAction: String = "SKIP",
        calibratedBuyProbability: Double = 0.0,
        calibratedSellProbability: Double = 0.0,
        calibratedSkipProbability: Double = 1.0,
        calibratedConfidence: Double = 0.0,
        expectedMoveMeanPoints: Double = 0.0,
        expectedMoveQ25Points: Double = 0.0,
        expectedMoveQ50Points: Double = 0.0,
        expectedMoveQ75Points: Double = 0.0,
        priceCostPoints: Double = 0.0,
        slippageCostPoints: Double = 0.0,
        uncertaintyScore: Double = 0.0,
        uncertaintyPenaltyPoints: Double = 0.0,
        riskPenaltyPoints: Double = 0.0,
        expectedGrossEdgePoints: Double = 0.0,
        edgeAfterCostsPoints: Double = 0.0,
        finalAction: String = "SKIP",
        abstain: Bool = false,
        reasons: [String] = []
    ) {
        self.ready = ready
        self.available = available
        self.stale = stale
        self.fallbackUsed = fallbackUsed
        self.calibrationStale = calibrationStale
        self.inputStale = inputStale
        self.newsRiskBlock = newsRiskBlock
        self.ratesRiskBlock = ratesRiskBlock
        self.microstructureStress = microstructureStress
        self.supportUsable = supportUsable
        self.generatedAt = max(0, generatedAt)
        self.symbol = symbol.uppercased()
        self.method = method.isEmpty ? "LOGISTIC_AFFINE" : method
        self.sessionLabel = sessionLabel.isEmpty ? "UNKNOWN" : sessionLabel
        self.regimeLabel = regimeLabel.isEmpty ? "UNKNOWN" : regimeLabel
        self.selectedTierKind = selectedTierKind.isEmpty ? "GLOBAL" : selectedTierKind
        self.selectedTierKey = selectedTierKey.isEmpty ? "GLOBAL|*|*|*" : selectedTierKey
        self.selectedSupport = max(0, selectedSupport)
        self.selectedQuality = fxClamp(selectedQuality, 0.0, 1.0)
        self.rawBuyProbability = fxClamp(rawBuyProbability, 0.0, 1.0)
        self.rawSellProbability = fxClamp(rawSellProbability, 0.0, 1.0)
        self.rawSkipProbability = fxClamp(rawSkipProbability, 0.0, 1.0)
        self.rawScore = fxSafeFinite(rawScore)
        self.rawAction = rawAction.isEmpty ? "SKIP" : rawAction
        self.calibratedBuyProbability = fxClamp(calibratedBuyProbability, 0.0, 1.0)
        self.calibratedSellProbability = fxClamp(calibratedSellProbability, 0.0, 1.0)
        self.calibratedSkipProbability = fxClamp(calibratedSkipProbability, 0.0, 1.0)
        self.calibratedConfidence = fxClamp(calibratedConfidence, 0.0, 1.0)
        self.expectedMoveMeanPoints = max(0.0, fxSafeFinite(expectedMoveMeanPoints))
        self.expectedMoveQ25Points = max(0.0, fxSafeFinite(expectedMoveQ25Points))
        self.expectedMoveQ50Points = max(0.0, fxSafeFinite(expectedMoveQ50Points))
        self.expectedMoveQ75Points = max(0.0, fxSafeFinite(expectedMoveQ75Points))
        self.priceCostPoints = max(0.0, fxSafeFinite(priceCostPoints))
        self.slippageCostPoints = max(0.0, fxSafeFinite(slippageCostPoints))
        self.uncertaintyScore = max(0.0, fxSafeFinite(uncertaintyScore))
        self.uncertaintyPenaltyPoints = max(0.0, fxSafeFinite(uncertaintyPenaltyPoints))
        self.riskPenaltyPoints = max(0.0, fxSafeFinite(riskPenaltyPoints))
        self.expectedGrossEdgePoints = max(0.0, fxSafeFinite(expectedGrossEdgePoints))
        self.edgeAfterCostsPoints = fxSafeFinite(edgeAfterCostsPoints)
        self.finalAction = finalAction.isEmpty ? "SKIP" : finalAction
        self.abstain = abstain
        self.reasons = Self.uniqueReasons(reasons)
    }

    public static var reset: ProbabilityCalibrationRuntimeState {
        ProbabilityCalibrationRuntimeState()
    }

    public var reasonCount: Int {
        reasons.count
    }

    public var reasonsCSV: String {
        reasons.filter { !$0.isEmpty }.joined(separator: "; ")
    }

    public mutating func appendReason(_ reason: String) {
        let value = reason.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !value.isEmpty,
              !reasons.contains(value),
              reasons.count < ProbabilityCalibrationConstants.maxReasons else {
            return
        }
        reasons.append(value)
    }

    private static func uniqueReasons(_ input: [String]) -> [String] {
        var output: [String] = []
        output.reserveCapacity(min(input.count, ProbabilityCalibrationConstants.maxReasons))
        for raw in input {
            let value = raw.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !value.isEmpty,
                  !output.contains(value),
                  output.count < ProbabilityCalibrationConstants.maxReasons else {
                continue
            }
            output.append(value)
        }
        return output
    }
}

public struct ProbabilityCalibrationPolicyOutcome: Codable, Hashable, Sendable {
    public var decision: Int
    public var state: ProbabilityCalibrationRuntimeState

    public init(decision: Int, state: ProbabilityCalibrationRuntimeState) {
        self.decision = decision
        self.state = state
    }
}

public enum ProbabilityCalibrationTools {
    public static func configPath() -> String {
        "\(ProbabilityCalibrationConstants.runtimeDirectory)/prob_calibration_config.tsv"
    }

    public static func memoryPath() -> String {
        "\(ProbabilityCalibrationConstants.runtimeDirectory)/prob_calibration_memory.tsv"
    }

    public static func runtimeStatePath(symbol: String) -> String {
        "\(ProbabilityCalibrationConstants.runtimeDirectory)/fxai_prob_calibration_\(ControlPlanePaths.safeToken(symbol)).tsv"
    }

    public static func runtimeHistoryPath(symbol: String) -> String {
        "\(ProbabilityCalibrationConstants.runtimeDirectory)/fxai_prob_calibration_history_\(ControlPlanePaths.safeToken(symbol)).ndjson"
    }

    public static func parseConfig(tsv: String?) -> ProbabilityCalibrationConfig {
        var config = ProbabilityCalibrationConfig()
        guard let tsv else { return config }
        var buckets = Array(repeating: "", count: ProbabilityCalibrationConstants.maxBuckets)
        var bucketCount = 0

        for line in tsv.components(separatedBy: .newlines) where !line.isEmpty {
            let parts = line.split(separator: "\t", omittingEmptySubsequences: false).map(String.init)
            guard parts.count >= 2 else { continue }
            let key = parts[0]
            let value = parts[1]
            let doubleValue = Double(value)

            switch key {
            case "enabled":
                config.enabled = (Int(value) ?? 0) != 0
            case "allow_abstain_flag":
                config.allowAbstainFlag = (Int(value) ?? 0) != 0
            case "neutral_blend_gain":
                config.neutralBlendGain = doubleValue ?? config.neutralBlendGain
            case "skip_uncertainty_gain":
                config.skipUncertaintyGain = doubleValue ?? config.skipUncertaintyGain
            case "skip_calibration_credit":
                config.skipCalibrationCredit = doubleValue ?? config.skipCalibrationCredit
            case "skip_floor":
                config.skipFloor = fxClamp(doubleValue ?? config.skipFloor, 0.0, 1.0)
            case "skip_cap":
                config.skipCap = fxClamp(doubleValue ?? config.skipCap, config.skipFloor, 1.0)
            case "base_uncertainty_score":
                config.baseUncertaintyScore = doubleValue ?? config.baseUncertaintyScore
            case "support_soft_floor":
                config.supportSoftFloor = max(0, Int(value) ?? config.supportSoftFloor)
            case "support_hard_floor":
                config.supportHardFloor = max(0, Int(value) ?? config.supportHardFloor)
            case "memory_stale_after_hours":
                config.memoryStaleAfterHours = max(0, Int(value) ?? config.memoryStaleAfterHours)
            case "min_calibration_quality":
                config.minCalibrationQuality = fxClamp(doubleValue ?? config.minCalibrationQuality, 0.0, 1.0)
            case "max_uncertainty_score":
                config.maxUncertaintyScore = max(0.0, doubleValue ?? config.maxUncertaintyScore)
            case "signal_zero_band":
                config.signalZeroBand = max(0.0, doubleValue ?? config.signalZeroBand)
            case "edge_floor_mult":
                config.edgeFloorMultiplier = max(0.0, doubleValue ?? config.edgeFloorMultiplier)
            case "trade_edge_floor_points":
                config.tradeEdgeFloorPoints = max(0.0, doubleValue ?? config.tradeEdgeFloorPoints)
            case "soft_prob_scale":
                config.softProbabilityScale = max(1e-6, doubleValue ?? config.softProbabilityScale)
            case "soft_skip_bias":
                config.softSkipBias = doubleValue ?? config.softSkipBias
            case "soft_move_mean_scale":
                config.softMoveMeanScale = max(0.0, doubleValue ?? config.softMoveMeanScale)
            case "soft_move_q25_scale":
                config.softMoveQ25Scale = max(0.0, doubleValue ?? config.softMoveQ25Scale)
            case "soft_move_q50_scale":
                config.softMoveQ50Scale = max(0.0, doubleValue ?? config.softMoveQ50Scale)
            case "soft_move_q75_scale":
                config.softMoveQ75Scale = max(0.0, doubleValue ?? config.softMoveQ75Scale)
            case "soft_confidence_cap":
                config.softConfidenceCap = fxClamp(doubleValue ?? config.softConfidenceCap, 0.50, 0.95)
            case "uncertainty_support":
                config.uncertaintySupportPenalty = doubleValue ?? config.uncertaintySupportPenalty
            case "uncertainty_quality":
                config.uncertaintyQualityPenalty = doubleValue ?? config.uncertaintyQualityPenalty
            case "uncertainty_disagreement":
                config.uncertaintyDisagreementPenalty = doubleValue ?? config.uncertaintyDisagreementPenalty
            case "uncertainty_distribution_width":
                config.uncertaintyDistributionWidthPenalty = doubleValue ?? config.uncertaintyDistributionWidthPenalty
            case "uncertainty_news":
                config.uncertaintyNewsPenalty = doubleValue ?? config.uncertaintyNewsPenalty
            case "uncertainty_rates":
                config.uncertaintyRatesPenalty = doubleValue ?? config.uncertaintyRatesPenalty
            case "uncertainty_micro":
                config.uncertaintyMicroPenalty = doubleValue ?? config.uncertaintyMicroPenalty
            case "uncertainty_dynamic_abstain":
                config.uncertaintyDynamicAbstainPenalty = doubleValue ?? config.uncertaintyDynamicAbstainPenalty
            case "uncertainty_adaptive_abstain":
                config.uncertaintyAdaptiveAbstainPenalty = doubleValue ?? config.uncertaintyAdaptiveAbstainPenalty
            case "uncertainty_stale_context":
                config.uncertaintyStaleContextPenalty = doubleValue ?? config.uncertaintyStaleContextPenalty
            case "risk_news_block_mult":
                config.riskNewsBlockMultiplier = doubleValue ?? config.riskNewsBlockMultiplier
            case "risk_rates_block_mult":
                config.riskRatesBlockMultiplier = doubleValue ?? config.riskRatesBlockMultiplier
            case "risk_micro_block_mult":
                config.riskMicroBlockMultiplier = doubleValue ?? config.riskMicroBlockMultiplier
            case "risk_caution_posture_mult":
                config.riskCautionPostureMultiplier = doubleValue ?? config.riskCautionPostureMultiplier
            case "risk_abstain_posture_mult":
                config.riskAbstainPostureMultiplier = doubleValue ?? config.riskAbstainPostureMultiplier
            case "risk_block_posture_mult":
                config.riskBlockPostureMultiplier = doubleValue ?? config.riskBlockPostureMultiplier
            case "risk_fill_mult":
                config.riskFillMultiplier = doubleValue ?? config.riskFillMultiplier
            case "risk_path_mult":
                config.riskPathMultiplier = doubleValue ?? config.riskPathMultiplier
            case "bucket_hierarchy":
                if bucketCount < ProbabilityCalibrationConstants.maxBuckets {
                    buckets[bucketCount] = value.trimmingCharacters(in: .whitespacesAndNewlines).uppercased()
                    bucketCount += 1
                }
            default:
                break
            }
        }

        if bucketCount > 0 {
            config.bucketCount = bucketCount
            config.bucketHierarchy = buckets
        }
        if config.skipCap < config.skipFloor {
            config.skipCap = config.skipFloor
        }
        config.ready = true
        return config
    }

    public static func parseMemory(tsv: String?) -> ProbabilityCalibrationMemory {
        guard let tsv else { return ProbabilityCalibrationMemory() }
        var generatedAt: Int64 = 0
        var method = "LOGISTIC_AFFINE"
        var tiers: [ProbabilityCalibrationTier] = []
        tiers.reserveCapacity(ProbabilityCalibrationConstants.maxTiers)

        for line in tsv.components(separatedBy: .newlines) where !line.isEmpty {
            let parts = line.split(separator: "\t", omittingEmptySubsequences: false).map(String.init)
            guard parts.count >= 2 else { continue }
            if parts[0] == "generated_at_unix" {
                generatedAt = Int64(parts[1]) ?? generatedAt
                continue
            }
            if parts[0] == "generated_at" {
                let parsed = parseISO8601UTC(parts[1])
                if parsed > 0 {
                    generatedAt = parsed
                }
                continue
            }
            if parts[0] == "default_method" {
                method = parts[1]
                continue
            }
            guard parts[0] == "tier",
                  parts.count >= 16,
                  tiers.count < ProbabilityCalibrationConstants.maxTiers else {
                continue
            }
            tiers.append(ProbabilityCalibrationTier(
                ready: true,
                kind: parts[1],
                symbol: parts[2],
                session: parts[3],
                regime: parts[4],
                support: Int(parts[5]) ?? 0,
                probabilityScale: Double(parts[6]) ?? 1.60,
                probabilityBias: Double(parts[7]) ?? 0.0,
                skipBias: Double(parts[8]) ?? 0.08,
                moveMeanScale: Double(parts[9]) ?? 0.78,
                moveQ25Scale: Double(parts[10]) ?? 0.60,
                moveQ50Scale: Double(parts[11]) ?? 0.72,
                moveQ75Scale: Double(parts[12]) ?? 0.88,
                calibrationQuality: Double(parts[13]) ?? 0.34,
                uncertaintyMultiplier: Double(parts[14]) ?? 1.30,
                confidenceCap: Double(parts[15]) ?? 0.58
            ))
        }

        return ProbabilityCalibrationMemory(generatedAt: generatedAt, defaultMethod: method, tiers: tiers)
    }

    public static func tierMatches(
        _ tier: ProbabilityCalibrationTier,
        kind: String,
        symbol: String,
        session: String,
        regime: String
    ) -> Bool {
        let targetKind = kind.uppercased()
        guard tier.ready, tier.kind == targetKind else { return false }
        let targetSymbol = symbol.uppercased()
        let targetSession = session.uppercased()
        let targetRegime = regime.uppercased()
        switch targetKind {
        case "PAIR_SESSION_REGIME":
            return tier.symbol == targetSymbol && tier.session == targetSession && tier.regime == targetRegime
        case "PAIR_REGIME":
            return tier.symbol == targetSymbol && tier.regime == targetRegime
        case "REGIME":
            return tier.regime == targetRegime
        case "GLOBAL":
            return true
        default:
            return false
        }
    }

    public static func selectTier(
        symbol: String,
        session: String,
        regime: String,
        config: ProbabilityCalibrationConfig,
        memory: ProbabilityCalibrationMemory
    ) -> ProbabilityCalibrationTierSelection {
        guard !memory.tiers.isEmpty else {
            return ProbabilityCalibrationTierSelection(tier: ProbabilityCalibrationTier.fallback(config: config))
        }

        for kind in config.effectiveBucketHierarchy where !kind.isEmpty {
            var bestPreferred: ProbabilityCalibrationTier?
            var bestFallback: ProbabilityCalibrationTier?
            var bestPreferredSupport = -1
            var bestFallbackSupport = -1
            var bestPreferredQuality = -1.0
            var bestFallbackQuality = -1.0

            for tier in memory.tiers where tierMatches(
                tier,
                kind: kind,
                symbol: symbol,
                session: session,
                regime: regime
            ) {
                if tier.support >= config.supportSoftFloor {
                    if bestPreferred == nil ||
                        tier.support > bestPreferredSupport ||
                        (tier.support == bestPreferredSupport && tier.calibrationQuality > bestPreferredQuality) {
                        bestPreferred = tier
                        bestPreferredSupport = tier.support
                        bestPreferredQuality = tier.calibrationQuality
                    }
                }
                if tier.support >= config.supportHardFloor {
                    if bestFallback == nil ||
                        tier.support > bestFallbackSupport ||
                        (tier.support == bestFallbackSupport && tier.calibrationQuality > bestFallbackQuality) {
                        bestFallback = tier
                        bestFallbackSupport = tier.support
                        bestFallbackQuality = tier.calibrationQuality
                    }
                }
            }

            if let selected = bestPreferred {
                return ProbabilityCalibrationTierSelection(
                    tier: selected,
                    found: true,
                    fallbackUsed: false,
                    supportUsable: true
                )
            }
            if let selected = bestFallback {
                return ProbabilityCalibrationTierSelection(
                    tier: selected,
                    found: true,
                    fallbackUsed: true,
                    supportUsable: true
                )
            }
        }

        return ProbabilityCalibrationTierSelection(tier: ProbabilityCalibrationTier.fallback(config: config))
    }

    public static func applyCalibration(
        config: ProbabilityCalibrationConfig,
        memory: ProbabilityCalibrationMemory,
        profile: ExecutionProfile,
        inputs: ProbabilityCalibrationPolicyInputs
    ) -> ProbabilityCalibrationPolicyOutcome {
        var state = ProbabilityCalibrationRuntimeState.reset
        state.ready = true
        state.available = true
        state.generatedAt = inputs.generatedAtUTC
        state.symbol = inputs.symbol
        state.method = memory.defaultMethod

        var finalDecision = inputs.upstreamDecision
        guard config.enabled else {
            state.stale = true
            return ProbabilityCalibrationPolicyOutcome(decision: finalDecision, state: state)
        }

        let normalized = normalizedRawProbabilities(
            buy: inputs.rawBuyProbability,
            sell: inputs.rawSellProbability,
            skip: inputs.rawSkipProbability
        )
        let rawBuy = normalized.buy
        let rawSell = normalized.sell
        let rawSkip = normalized.skip
        state.rawBuyProbability = rawBuy
        state.rawSellProbability = rawSell
        state.rawSkipProbability = rawSkip
        state.rawScore = rawBuy - rawSell
        state.rawAction = rawSkip >= rawBuy && rawSkip >= rawSell ? "SKIP" : (rawBuy >= rawSell ? "BUY" : "SELL")

        state.sessionLabel = resolvedSessionLabel(
            explicit: inputs.sessionLabel,
            news: inputs.newsState,
            microstructure: inputs.microstructureState
        )
        state.regimeLabel = resolvedRegimeLabel(explicit: inputs.regimeLabel)

        let selection = selectTier(
            symbol: inputs.symbol,
            session: state.sessionLabel,
            regime: state.regimeLabel,
            config: config,
            memory: memory
        )
        let tier = selection.tier
        state.fallbackUsed = selection.fallbackUsed
        state.supportUsable = selection.supportUsable
        state.selectedTierKind = tier.kind
        state.selectedTierKey = tier.key
        state.selectedSupport = tier.support
        state.selectedQuality = fxClamp(tier.calibrationQuality, 0.0, 1.0)

        state.calibrationStale = memory.generatedAt <= 0 ||
            (config.memoryStaleAfterHours > 0 &&
                inputs.generatedAtUTC > 0 &&
                (inputs.generatedAtUTC - memory.generatedAt) > Int64(config.memoryStaleAfterHours * 3_600))

        let newsStale = inputs.newsState.ready && inputs.newsState.available && inputs.newsState.stale
        let ratesStale = inputs.ratesState.ready && inputs.ratesState.available && inputs.ratesState.stale
        let crossStale = inputs.crossAssetState.ready && inputs.crossAssetState.available && inputs.crossAssetState.stale
        let microStale = inputs.microstructureState.ready && inputs.microstructureState.available && inputs.microstructureState.stale
        let executionQualityUnknown = inputs.executionQualityEnabled && !inputs.executionQualityState.ready
        let executionQualityStale = inputs.executionQualityState.ready && inputs.executionQualityState.dataStale
        let staleContextCount = [
            newsStale,
            ratesStale,
            crossStale,
            microStale,
            executionQualityUnknown || executionQualityStale
        ].filter { $0 }.count
        state.inputStale = staleContextCount > 0
        state.newsRiskBlock = inputs.newsState.ready &&
            inputs.newsState.available &&
            (inputs.newsState.tradeGate.uppercased() == "BLOCK" || inputs.newsState.newsRiskScore >= 0.84)
        state.ratesRiskBlock = inputs.ratesState.ready &&
            inputs.ratesState.available &&
            (inputs.ratesState.tradeGate.uppercased() == "BLOCK" ||
                inputs.ratesState.ratesRiskScore >= 0.82 ||
                inputs.ratesState.meetingPathRepriceNow)
        state.microstructureStress = inputs.microstructureState.ready &&
            inputs.microstructureState.available &&
            (inputs.microstructureState.tradeGate.uppercased() == "BLOCK" ||
                inputs.microstructureState.hostileExecutionScore >= 0.82 ||
                inputs.microstructureState.liquidityStressScore >= 0.84)

        let newsRisk = inputs.newsState.ready && inputs.newsState.available
            ? fxClamp(inputs.newsState.newsRiskScore, 0.0, 1.0)
            : (newsStale ? 0.45 : 0.15)
        let ratesRisk = inputs.ratesState.ready && inputs.ratesState.available
            ? fxClamp(inputs.ratesState.ratesRiskScore, 0.0, 1.0)
            : (ratesStale ? 0.35 : 0.12)
        let crossRisk = inputs.crossAssetState.ready && inputs.crossAssetState.available
            ? fxClamp(
                max(
                    inputs.crossAssetState.pairCrossAssetRiskScore,
                    max(
                        inputs.crossAssetState.usdLiquidityStressScore,
                        inputs.crossAssetState.crossAssetDislocationScore
                    )
                ),
                0.0,
                1.0
            )
            : (crossStale ? 0.32 : 0.10)
        let microRisk = inputs.microstructureState.ready && inputs.microstructureState.available
            ? fxClamp(max(inputs.microstructureState.hostileExecutionScore, inputs.microstructureState.liquidityStressScore), 0.0, 1.0)
            : (microStale ? 0.42 : 0.14)
        let dynamicAbstain = inputs.dynamicStateReady ? fxClamp(inputs.dynamicAbstainBias, 0.0, 1.0) : 0.0
        let adaptiveAbstain = fxClamp(inputs.adaptiveRouterAbstainBias, 0.0, 1.0)
        let agreement = fxClamp(inputs.agreementScore, 0.0, 1.0)

        let distributionWidth = max(inputs.moveQ75Points - inputs.moveQ25Points, 0.0)
        let distributionRatio = fxClamp(
            distributionWidth / max(max(inputs.moveMeanPoints, inputs.minMovePoints), 0.25),
            0.0,
            3.0
        ) / 3.0
        let supportShortfall = fxClamp(
            Double(config.supportSoftFloor - tier.support) / Double(max(config.supportSoftFloor, 1)),
            0.0,
            1.0
        )
        let qualityShortfall = fxClamp(config.minCalibrationQuality - tier.calibrationQuality, 0.0, 1.0)
        var uncertaintyScore = config.baseUncertaintyScore +
            config.uncertaintySupportPenalty * supportShortfall +
            config.uncertaintyQualityPenalty * qualityShortfall +
            config.uncertaintyDisagreementPenalty * (1.0 - agreement) +
            config.uncertaintyDistributionWidthPenalty * distributionRatio +
            config.uncertaintyNewsPenalty * newsRisk +
            config.uncertaintyRatesPenalty * ratesRisk +
            0.12 * crossRisk +
            config.uncertaintyMicroPenalty * microRisk +
            config.uncertaintyDynamicAbstainPenalty * dynamicAbstain +
            config.uncertaintyAdaptiveAbstainPenalty * adaptiveAbstain +
            config.uncertaintyStaleContextPenalty * fxClamp(Double(staleContextCount) / 4.0, 0.0, 1.0)
        uncertaintyScore *= fxClamp(tier.uncertaintyMultiplier, 0.40, 2.50)
        state.uncertaintyScore = uncertaintyScore

        let directionalMass = max(rawBuy + rawSell, 1e-6)
        let directionalShare = rawBuy / directionalMass
        var directionalBuy = sigmoid(tier.probabilityBias + tier.probabilityScale * logit(directionalShare))
        let neutralBlend = fxClamp(
            config.neutralBlendGain * (1.0 - fxClamp(tier.calibrationQuality, 0.0, 1.0)),
            0.0,
            0.85
        )
        directionalBuy = neutralBlend * 0.5 + (1.0 - neutralBlend) * directionalBuy
        let directionalDistance = directionalBuy - 0.5
        let maxDistance = max(fxClamp(tier.confidenceCap, 0.50, 0.95) - 0.5, 0.0)
        directionalBuy = 0.5 + fxClamp(directionalDistance, -maxDistance, maxDistance)

        let calibratedSkip = fxClamp(
            rawSkip +
                tier.skipBias +
                config.skipUncertaintyGain * uncertaintyScore -
                config.skipCalibrationCredit * fxClamp(tier.calibrationQuality, 0.0, 1.0),
            config.skipFloor,
            config.skipCap
        )
        let calibratedDirectionalMass = max(1.0 - calibratedSkip, 1e-6)
        state.calibratedBuyProbability = fxClamp(calibratedDirectionalMass * directionalBuy, 0.0, 1.0)
        state.calibratedSellProbability = fxClamp(calibratedDirectionalMass * (1.0 - directionalBuy), 0.0, 1.0)
        state.calibratedSkipProbability = fxClamp(
            1.0 - state.calibratedBuyProbability - state.calibratedSellProbability,
            0.0,
            1.0
        )
        state.calibratedConfidence = max(state.calibratedBuyProbability, state.calibratedSellProbability)

        let uncertaintyMeanMultiplier = fxClamp(1.0 - 0.18 * uncertaintyScore, 0.35, 1.0)
        let uncertaintyQ25Multiplier = fxClamp(1.0 - 0.24 * uncertaintyScore, 0.20, 1.0)
        let uncertaintyQ50Multiplier = fxClamp(1.0 - 0.16 * uncertaintyScore, 0.25, 1.0)
        let uncertaintyQ75Multiplier = fxClamp(1.0 - 0.10 * uncertaintyScore, 0.35, 1.0)
        state.expectedMoveQ25Points = max(inputs.moveQ25Points * tier.moveQ25Scale * uncertaintyQ25Multiplier, 0.0)
        state.expectedMoveQ50Points = max(
            inputs.moveQ50Points * tier.moveQ50Scale * uncertaintyQ50Multiplier,
            state.expectedMoveQ25Points
        )
        state.expectedMoveQ75Points = max(
            inputs.moveQ75Points * tier.moveQ75Scale * uncertaintyQ75Multiplier,
            state.expectedMoveQ50Points
        )
        state.expectedMoveMeanPoints = max(
            inputs.moveMeanPoints * tier.moveMeanScale * uncertaintyMeanMultiplier,
            state.expectedMoveQ50Points
        )

        let basePriceCost = max(inputs.priceCostPoints, 0.0) +
            max(inputs.commissionPoints, 0.0) +
            max(inputs.costBufferPoints, 0.0) +
            max(profile.costBufferPoints, 0.0)
        let liquidityStress = inputs.microstructureState.ready && inputs.microstructureState.available
            ? fxClamp(max(inputs.microstructureState.spreadZscore60s, 0.0), 0.0, 4.0)
            : 0.0
        let pathFlags = pathFlags(
            state: state,
            inputs: inputs
        )
        var priceCost = basePriceCost
        var slippageCost = ExecutionReplayTools.slippagePoints(
            profile: profile,
            roundTripCostPoints: basePriceCost,
            horizonMinutes: inputs.horizonMinutes,
            liquidityStressPoints: liquidityStress,
            pathFlags: pathFlags
        )
        let executionQualityUsable = inputs.executionQualityState.ready &&
            !inputs.executionQualityState.dataStale &&
            inputs.executionQualityState.spreadExpectedPoints >= 0.0
        if executionQualityUsable {
            priceCost = max(max(inputs.executionQualityState.spreadExpectedPoints, inputs.priceCostPoints), 0.0) +
                max(inputs.commissionPoints, 0.0) +
                max(inputs.costBufferPoints, 0.0) +
                max(profile.costBufferPoints, 0.0)
            slippageCost = max(slippageCost, max(inputs.executionQualityState.expectedSlippagePoints, 0.0))
        }
        state.priceCostPoints = priceCost
        state.slippageCostPoints = slippageCost
        state.uncertaintyPenaltyPoints = max(inputs.minMovePoints, 0.25) * uncertaintyScore

        var riskPenalty = ExecutionReplayTools.fillPenaltyPoints(
            profile: profile,
            roundTripCostPoints: priceCost,
            liquidityStressPoints: liquidityStress,
            pathFlags: pathFlags
        ) + max(inputs.minMovePoints, 0.25) * (
            config.riskFillMultiplier * fxClamp(inputs.fillRisk, 0.0, 1.0) +
                config.riskPathMultiplier * fxClamp(inputs.pathRisk, 0.0, 1.0) +
                (state.newsRiskBlock ? config.riskNewsBlockMultiplier : 0.0) +
                (state.ratesRiskBlock ? config.riskRatesBlockMultiplier : 0.0) +
                0.18 * crossRisk +
                (state.microstructureStress ? config.riskMicroBlockMultiplier : 0.0) +
                (postureMatches(inputs.adaptiveRouterPosture, inputs.dynamicTradePosture, "CAUTION")
                    ? config.riskCautionPostureMultiplier
                    : 0.0) +
                (postureMatches(inputs.adaptiveRouterPosture, inputs.dynamicTradePosture, "ABSTAIN_BIAS")
                    ? config.riskAbstainPostureMultiplier
                    : 0.0) +
                (postureMatches(inputs.adaptiveRouterPosture, inputs.dynamicTradePosture, "BLOCK")
                    ? config.riskBlockPostureMultiplier
                    : 0.0)
        )
        if executionQualityUsable {
            let executionPenaltyMultiplier = max(inputs.minMovePoints, 0.25)
            riskPenalty += executionPenaltyMultiplier * (
                0.42 * fxClamp(1.0 - inputs.executionQualityState.fillQualityScore, 0.0, 1.0) +
                    0.32 * fxClamp(inputs.executionQualityState.latencySensitivityScore, 0.0, 1.0) +
                    0.26 * fxClamp(inputs.executionQualityState.liquidityFragilityScore, 0.0, 1.0)
            )
            switch inputs.executionQualityState.executionState.uppercased() {
            case "BLOCKED":
                riskPenalty += executionPenaltyMultiplier * 0.60
            case "STRESSED":
                riskPenalty += executionPenaltyMultiplier * 0.35
            case "CAUTION":
                riskPenalty += executionPenaltyMultiplier * 0.18
            default:
                break
            }
        }
        state.riskPenaltyPoints = riskPenalty

        state.expectedGrossEdgePoints = abs(state.calibratedBuyProbability - state.calibratedSellProbability) *
            state.expectedMoveMeanPoints
        state.edgeAfterCostsPoints = state.expectedGrossEdgePoints -
            state.priceCostPoints -
            state.slippageCostPoints -
            state.uncertaintyPenaltyPoints -
            state.riskPenaltyPoints

        let edgeFloorPoints = max(
            config.tradeEdgeFloorPoints,
            config.edgeFloorMultiplier * max(inputs.minMovePoints, 0.25)
        )
        let costFloorPoints = state.priceCostPoints + state.slippageCostPoints + state.riskPenaltyPoints
        let calibratedDirection = state.calibratedBuyProbability >= state.calibratedSellProbability ? "BUY" : "SELL"
        state.finalAction = decisionLabel(finalDecision)

        if state.calibrationStale {
            state.appendReason("CALIBRATION_STALE")
        }
        if state.inputStale {
            state.appendReason("INPUT_STALE")
        }
        if executionQualityUnknown {
            state.appendReason("EXECUTION_QUALITY_UNKNOWN")
        }
        if executionQualityStale {
            state.appendReason("EXECUTION_QUALITY_STALE")
        }
        if !state.supportUsable {
            state.appendReason("SUPPORT_TOO_LOW")
        }
        if tier.calibrationQuality < config.minCalibrationQuality {
            state.appendReason("CALIBRATION_WEAK")
        }
        if abs(state.rawScore) < config.signalZeroBand {
            state.appendReason("SIGNAL_TOO_CLOSE_TO_ZERO")
        }
        if state.expectedMoveQ25Points <= costFloorPoints {
            state.appendReason("MOVE_DISTRIBUTION_TOO_WEAK")
        }
        if state.expectedGrossEdgePoints <= costFloorPoints {
            state.appendReason("COST_TOO_HIGH")
        }
        if state.uncertaintyScore >= config.maxUncertaintyScore {
            state.appendReason("UNCERTAINTY_TOO_HIGH")
        }
        if state.edgeAfterCostsPoints <= edgeFloorPoints {
            state.appendReason("EDGE_TOO_SMALL")
        }
        if state.newsRiskBlock {
            state.appendReason("NEWS_RISK_BLOCK")
        }
        if state.ratesRiskBlock {
            state.appendReason("RATES_RISK_BLOCK")
        }
        if inputs.crossAssetState.ready &&
            inputs.crossAssetState.available &&
            inputs.crossAssetState.tradeGate.uppercased() == "BLOCK" {
            state.appendReason("CROSS_ASSET_BLOCK")
        } else if inputs.crossAssetState.ready &&
            inputs.crossAssetState.available &&
            inputs.crossAssetState.tradeGate.uppercased() == "CAUTION" {
            state.appendReason("CROSS_ASSET_CAUTION")
        }
        if state.microstructureStress {
            state.appendReason("MICROSTRUCTURE_STRESS")
        }
        if executionQualityUsable {
            switch inputs.executionQualityState.executionState.uppercased() {
            case "BLOCKED":
                state.appendReason("EXECUTION_QUALITY_BLOCK")
            case "STRESSED":
                state.appendReason("EXECUTION_QUALITY_STRESSED")
            case "CAUTION":
                state.appendReason("EXECUTION_QUALITY_CAUTION")
            default:
                break
            }
        }

        state.abstain = false
        if finalDecision != -1 {
            let upstreamAction = decisionLabel(finalDecision)
            if upstreamAction != calibratedDirection &&
                abs(state.calibratedBuyProbability - state.calibratedSellProbability) >= 0.08 {
                state.appendReason("CALIBRATED_DIRECTION_CONFLICT")
                finalDecision = -1
            } else if state.reasonCount > 0 {
                finalDecision = -1
            }
        } else {
            state.abstain = true
        }

        if finalDecision == -1 {
            state.finalAction = "SKIP"
            state.abstain = true
        } else {
            state.finalAction = decisionLabel(finalDecision)
        }
        state.stale = state.calibrationStale || state.inputStale
        return ProbabilityCalibrationPolicyOutcome(decision: finalDecision, state: state)
    }

    public static func runtimeStateTSV(
        symbol: String,
        state: ProbabilityCalibrationRuntimeState
    ) -> String? {
        guard state.ready, !symbol.isEmpty else { return nil }
        return runtimeStateRows(symbol: symbol, state: state)
            .map { key, value in
                "\(RuntimeArtifactTSV.field(key))\t\(RuntimeArtifactTSV.field(value))"
            }
            .joined(separator: "\r\n") + "\r\n"
    }

    public static func runtimeHistoryNDJSONLine(
        symbol: String,
        state: ProbabilityCalibrationRuntimeState
    ) -> String? {
        guard state.ready, !symbol.isEmpty else { return nil }
        let reasons = state.reasons.map(jsonQuoted).joined(separator: ",")
        return "{" +
            "\"schema_version\":1," +
            "\"generated_at\":\(jsonQuoted(iso8601UTC(state.generatedAt)))," +
            "\"symbol\":\(jsonQuoted(symbol))," +
            "\"state\":{" +
            "\"method\":\(jsonQuoted(state.method))," +
            "\"session_label\":\(jsonQuoted(state.sessionLabel))," +
            "\"regime_label\":\(jsonQuoted(state.regimeLabel))," +
            "\"selected_tier_kind\":\(jsonQuoted(state.selectedTierKind))," +
            "\"selected_tier_key\":\(jsonQuoted(state.selectedTierKey))," +
            "\"selected_support\":\(state.selectedSupport)," +
            "\"selected_quality\":\(RuntimeArtifactTSV.double(state.selectedQuality))," +
            "\"raw_action\":\(jsonQuoted(state.rawAction))," +
            "\"raw_score\":\(RuntimeArtifactTSV.double(state.rawScore))," +
            "\"raw_buy_prob\":\(RuntimeArtifactTSV.double(state.rawBuyProbability))," +
            "\"raw_sell_prob\":\(RuntimeArtifactTSV.double(state.rawSellProbability))," +
            "\"raw_skip_prob\":\(RuntimeArtifactTSV.double(state.rawSkipProbability))," +
            "\"calibrated_buy_prob\":\(RuntimeArtifactTSV.double(state.calibratedBuyProbability))," +
            "\"calibrated_sell_prob\":\(RuntimeArtifactTSV.double(state.calibratedSellProbability))," +
            "\"calibrated_skip_prob\":\(RuntimeArtifactTSV.double(state.calibratedSkipProbability))," +
            "\"calibrated_confidence\":\(RuntimeArtifactTSV.double(state.calibratedConfidence))," +
            "\"expected_move_mean_points\":\(RuntimeArtifactTSV.double(state.expectedMoveMeanPoints))," +
            "\"expected_move_q25_points\":\(RuntimeArtifactTSV.double(state.expectedMoveQ25Points))," +
            "\"expected_move_q50_points\":\(RuntimeArtifactTSV.double(state.expectedMoveQ50Points))," +
            "\"expected_move_q75_points\":\(RuntimeArtifactTSV.double(state.expectedMoveQ75Points))," +
            "\"spread_cost_points\":\(RuntimeArtifactTSV.double(state.priceCostPoints))," +
            "\"slippage_cost_points\":\(RuntimeArtifactTSV.double(state.slippageCostPoints))," +
            "\"uncertainty_score\":\(RuntimeArtifactTSV.double(state.uncertaintyScore))," +
            "\"uncertainty_penalty_points\":\(RuntimeArtifactTSV.double(state.uncertaintyPenaltyPoints))," +
            "\"risk_penalty_points\":\(RuntimeArtifactTSV.double(state.riskPenaltyPoints))," +
            "\"expected_gross_edge_points\":\(RuntimeArtifactTSV.double(state.expectedGrossEdgePoints))," +
            "\"edge_after_costs_points\":\(RuntimeArtifactTSV.double(state.edgeAfterCostsPoints))," +
            "\"final_action\":\(jsonQuoted(state.finalAction))," +
            "\"abstain\":\(state.abstain ? "true" : "false")," +
            "\"fallback_used\":\(state.fallbackUsed ? "true" : "false")," +
            "\"calibration_stale\":\(state.calibrationStale ? "true" : "false")," +
            "\"input_stale\":\(state.inputStale ? "true" : "false")," +
            "\"support_usable\":\(state.supportUsable ? "true" : "false")," +
            "\"reason_codes\":[\(reasons)]" +
            "}}"
    }

    public static func readPairState(
        symbol _: String,
        stateTSV: String?,
        nowUTC: Int64 = 0,
        freshnessMaxSeconds: Int64 = ProbabilityCalibrationConstants.defaultFreshnessMaxSeconds
    ) -> ProbabilityCalibrationRuntimeState? {
        guard let stateTSV else { return nil }
        let state = normalizedAvailableState(
            parseState(tsv: stateTSV),
            nowUTC: nowUTC,
            freshnessMaxSeconds: freshnessMaxSeconds
        )
        return state.available ? state : nil
    }

    public static func parseState(tsv: String) -> ProbabilityCalibrationRuntimeState {
        var state = ProbabilityCalibrationRuntimeState.reset
        for line in tsv.components(separatedBy: .newlines) where !line.isEmpty {
            let parts = line.split(separator: "\t", omittingEmptySubsequences: false)
            guard parts.count >= 2 else { continue }
            let key = String(parts[0])
            let value = String(parts[1])
            state.ready = true
            state.available = true

            switch key {
            case "symbol":
                state.symbol = value
            case "generated_at":
                state.generatedAt = Int64(value) ?? 0
            case "method":
                state.method = value
            case "session_label":
                state.sessionLabel = value
            case "regime_label":
                state.regimeLabel = value
            case "selected_tier_kind":
                state.selectedTierKind = value
            case "selected_tier_key":
                state.selectedTierKey = value
            case "selected_support":
                state.selectedSupport = Int(value) ?? 0
            case "selected_quality":
                state.selectedQuality = Double(value) ?? 0.0
            case "raw_action":
                state.rawAction = value
            case "raw_score":
                state.rawScore = Double(value) ?? 0.0
            case "raw_buy_prob":
                state.rawBuyProbability = Double(value) ?? 0.0
            case "raw_sell_prob":
                state.rawSellProbability = Double(value) ?? 0.0
            case "raw_skip_prob":
                state.rawSkipProbability = Double(value) ?? 1.0
            case "calibrated_buy_prob":
                state.calibratedBuyProbability = Double(value) ?? 0.0
            case "calibrated_sell_prob":
                state.calibratedSellProbability = Double(value) ?? 0.0
            case "calibrated_skip_prob":
                state.calibratedSkipProbability = Double(value) ?? 1.0
            case "calibrated_confidence":
                state.calibratedConfidence = Double(value) ?? 0.0
            case "expected_move_mean_points":
                state.expectedMoveMeanPoints = Double(value) ?? 0.0
            case "expected_move_q25_points":
                state.expectedMoveQ25Points = Double(value) ?? 0.0
            case "expected_move_q50_points":
                state.expectedMoveQ50Points = Double(value) ?? 0.0
            case "expected_move_q75_points":
                state.expectedMoveQ75Points = Double(value) ?? 0.0
            case "spread_cost_points", "price_cost_points":
                state.priceCostPoints = Double(value) ?? 0.0
            case "slippage_cost_points":
                state.slippageCostPoints = Double(value) ?? 0.0
            case "uncertainty_score":
                state.uncertaintyScore = Double(value) ?? 0.0
            case "uncertainty_penalty_points":
                state.uncertaintyPenaltyPoints = Double(value) ?? 0.0
            case "risk_penalty_points":
                state.riskPenaltyPoints = Double(value) ?? 0.0
            case "expected_gross_edge_points":
                state.expectedGrossEdgePoints = Double(value) ?? 0.0
            case "edge_after_costs_points":
                state.edgeAfterCostsPoints = Double(value) ?? 0.0
            case "final_action":
                state.finalAction = value
            case "abstain":
                state.abstain = (Int(value) ?? 0) != 0
            case "fallback_used":
                state.fallbackUsed = (Int(value) ?? 0) != 0
            case "calibration_stale":
                state.calibrationStale = (Int(value) ?? 0) != 0
            case "input_stale":
                state.inputStale = (Int(value) ?? 0) != 0
            case "support_usable":
                state.supportUsable = (Int(value) ?? 0) != 0
            case "reasons_csv":
                for reason in value.split(separator: ";", omittingEmptySubsequences: false) {
                    state.appendReason(String(reason))
                }
            default:
                break
            }
        }
        return state
    }

    private static func normalizedRawProbabilities(
        buy: Double,
        sell: Double,
        skip: Double
    ) -> (buy: Double, sell: Double, skip: Double) {
        var rawBuy = fxClamp(buy, 0.0, 1.0)
        var rawSell = fxClamp(sell, 0.0, 1.0)
        var rawSkip = fxClamp(skip, 0.0, 1.0)
        var total = rawBuy + rawSell + rawSkip
        if total <= 0.0 {
            rawBuy = 0.0
            rawSell = 0.0
            rawSkip = 1.0
            total = 1.0
        }
        return (rawBuy / total, rawSell / total, rawSkip / total)
    }

    private static func sigmoid(_ value: Double) -> Double {
        if value >= 0.0 {
            let expNegative = exp(-value)
            return 1.0 / (1.0 + expNegative)
        }
        let expPositive = exp(value)
        return expPositive / (1.0 + expPositive)
    }

    private static func logit(_ probability: Double) -> Double {
        let p = fxClamp(probability, 1e-6, 1.0 - 1e-6)
        return log(p / (1.0 - p))
    }

    private static func decisionLabel(_ decision: Int) -> String {
        if decision == 1 { return "BUY" }
        if decision == 0 { return "SELL" }
        return "SKIP"
    }

    private static func pathFlags(
        state: ProbabilityCalibrationRuntimeState,
        inputs: ProbabilityCalibrationPolicyInputs
    ) -> SamplePathFlags {
        var flags: SamplePathFlags = []
        if state.newsRiskBlock || state.ratesRiskBlock || state.microstructureStress {
            flags.insert(.liquidityStress)
        }
        if inputs.pathRisk >= 0.72 ||
            (inputs.microstructureState.ready &&
                inputs.microstructureState.available &&
                inputs.microstructureState.sweepAndRejectFlag60s) {
            flags.insert(.dualHit)
        }
        if inputs.fillRisk >= 0.72 ||
            (inputs.microstructureState.ready &&
                inputs.microstructureState.available &&
                inputs.microstructureState.handoffFlag) {
            flags.insert(.slowHit)
        }
        return flags
    }

    private static func postureMatches(_ adaptive: String, _ dynamic: String, _ target: String) -> Bool {
        adaptive.uppercased() == target || dynamic.uppercased() == target
    }

    private static func resolvedSessionLabel(
        explicit: String?,
        news: NewsPulsePairState,
        microstructure: MicrostructurePairState
    ) -> String {
        if let explicit {
            let value = explicit.trimmingCharacters(in: .whitespacesAndNewlines)
            if !value.isEmpty { return value.uppercased() }
        }
        if microstructure.ready && microstructure.available {
            let value = microstructure.sessionTag.trimmingCharacters(in: .whitespacesAndNewlines)
            if !value.isEmpty { return value.uppercased() }
        }
        if news.ready && news.available {
            let value = news.sessionProfile.trimmingCharacters(in: .whitespacesAndNewlines)
            if !value.isEmpty { return value.uppercased() }
        }
        return "UNKNOWN"
    }

    private static func resolvedRegimeLabel(explicit: String?) -> String {
        if let explicit {
            let value = explicit.trimmingCharacters(in: .whitespacesAndNewlines)
            if !value.isEmpty { return value.uppercased() }
        }
        return "UNKNOWN"
    }

    private static func normalizedAvailableState(
        _ state: ProbabilityCalibrationRuntimeState,
        nowUTC: Int64,
        freshnessMaxSeconds: Int64
    ) -> ProbabilityCalibrationRuntimeState {
        var output = state
        if output.available {
            if nowUTC > 0, output.generatedAt > 0 {
                output.stale = nowUTC - output.generatedAt > max(freshnessMaxSeconds, 30)
            } else {
                output.stale = true
            }
        }
        return output
    }

    private static func runtimeStateRows(
        symbol: String,
        state: ProbabilityCalibrationRuntimeState
    ) -> [(String, String)] {
        [
            ("schema_version", "1"),
            ("symbol", symbol),
            ("generated_at", "\(state.generatedAt)"),
            ("method", state.method),
            ("session_label", state.sessionLabel),
            ("regime_label", state.regimeLabel),
            ("selected_tier_kind", state.selectedTierKind),
            ("selected_tier_key", state.selectedTierKey),
            ("selected_support", "\(state.selectedSupport)"),
            ("selected_quality", RuntimeArtifactTSV.double(state.selectedQuality)),
            ("raw_action", state.rawAction),
            ("raw_score", RuntimeArtifactTSV.double(state.rawScore)),
            ("raw_buy_prob", RuntimeArtifactTSV.double(state.rawBuyProbability)),
            ("raw_sell_prob", RuntimeArtifactTSV.double(state.rawSellProbability)),
            ("raw_skip_prob", RuntimeArtifactTSV.double(state.rawSkipProbability)),
            ("calibrated_buy_prob", RuntimeArtifactTSV.double(state.calibratedBuyProbability)),
            ("calibrated_sell_prob", RuntimeArtifactTSV.double(state.calibratedSellProbability)),
            ("calibrated_skip_prob", RuntimeArtifactTSV.double(state.calibratedSkipProbability)),
            ("calibrated_confidence", RuntimeArtifactTSV.double(state.calibratedConfidence)),
            ("expected_move_mean_points", RuntimeArtifactTSV.double(state.expectedMoveMeanPoints)),
            ("expected_move_q25_points", RuntimeArtifactTSV.double(state.expectedMoveQ25Points)),
            ("expected_move_q50_points", RuntimeArtifactTSV.double(state.expectedMoveQ50Points)),
            ("expected_move_q75_points", RuntimeArtifactTSV.double(state.expectedMoveQ75Points)),
            ("spread_cost_points", RuntimeArtifactTSV.double(state.priceCostPoints)),
            ("slippage_cost_points", RuntimeArtifactTSV.double(state.slippageCostPoints)),
            ("uncertainty_score", RuntimeArtifactTSV.double(state.uncertaintyScore)),
            ("uncertainty_penalty_points", RuntimeArtifactTSV.double(state.uncertaintyPenaltyPoints)),
            ("risk_penalty_points", RuntimeArtifactTSV.double(state.riskPenaltyPoints)),
            ("expected_gross_edge_points", RuntimeArtifactTSV.double(state.expectedGrossEdgePoints)),
            ("edge_after_costs_points", RuntimeArtifactTSV.double(state.edgeAfterCostsPoints)),
            ("final_action", state.finalAction),
            ("abstain", RuntimeArtifactTSV.bool(state.abstain)),
            ("fallback_used", RuntimeArtifactTSV.bool(state.fallbackUsed)),
            ("calibration_stale", RuntimeArtifactTSV.bool(state.calibrationStale)),
            ("input_stale", RuntimeArtifactTSV.bool(state.inputStale)),
            ("support_usable", RuntimeArtifactTSV.bool(state.supportUsable)),
            ("reasons_csv", state.reasonsCSV)
        ]
    }

    private static func parseISO8601UTC(_ raw: String) -> Int64 {
        guard raw.count >= 19 else { return 0 }
        var calendar = Calendar(identifier: .gregorian)
        calendar.timeZone = TimeZone(secondsFromGMT: 0) ?? .gmt
        let year = Int(raw.prefix(4)) ?? 0
        let monthStart = raw.index(raw.startIndex, offsetBy: 5)
        let monthEnd = raw.index(monthStart, offsetBy: 2)
        let dayStart = raw.index(raw.startIndex, offsetBy: 8)
        let dayEnd = raw.index(dayStart, offsetBy: 2)
        let hourStart = raw.index(raw.startIndex, offsetBy: 11)
        let hourEnd = raw.index(hourStart, offsetBy: 2)
        let minuteStart = raw.index(raw.startIndex, offsetBy: 14)
        let minuteEnd = raw.index(minuteStart, offsetBy: 2)
        let secondStart = raw.index(raw.startIndex, offsetBy: 17)
        let secondEnd = raw.index(secondStart, offsetBy: 2)
        let month = Int(raw[monthStart..<monthEnd]) ?? 0
        let day = Int(raw[dayStart..<dayEnd]) ?? 0
        let hour = Int(raw[hourStart..<hourEnd]) ?? 0
        let minute = Int(raw[minuteStart..<minuteEnd]) ?? 0
        let second = Int(raw[secondStart..<secondEnd]) ?? 0
        guard year >= 2000,
              (1...12).contains(month),
              (1...31).contains(day),
              (0...23).contains(hour),
              (0...59).contains(minute),
              (0...59).contains(second) else {
            return 0
        }
        let components = DateComponents(
            calendar: calendar,
            timeZone: calendar.timeZone,
            year: year,
            month: month,
            day: day,
            hour: hour,
            minute: minute,
            second: second
        )
        guard let date = calendar.date(from: components) else { return 0 }
        return Int64(date.timeIntervalSince1970)
    }

    private static func iso8601UTC(_ timestamp: Int64) -> String {
        guard timestamp > 0 else { return "" }
        var calendar = Calendar(identifier: .gregorian)
        calendar.timeZone = TimeZone(secondsFromGMT: 0) ?? .gmt
        let components = calendar.dateComponents(
            [.year, .month, .day, .hour, .minute, .second],
            from: Date(timeIntervalSince1970: TimeInterval(timestamp))
        )
        return String(
            format: "%04d-%02d-%02dT%02d:%02d:%02dZ",
            locale: Locale(identifier: "en_US_POSIX"),
            components.year ?? 0,
            components.month ?? 0,
            components.day ?? 0,
            components.hour ?? 0,
            components.minute ?? 0,
            components.second ?? 0
        )
    }

    private static func jsonQuoted(_ value: String) -> String {
        let escaped = value
            .replacingOccurrences(of: "\\", with: "\\\\")
            .replacingOccurrences(of: "\"", with: "\\\"")
            .replacingOccurrences(of: "\r", with: " ")
            .replacingOccurrences(of: "\n", with: " ")
        return "\"\(escaped)\""
    }
}

public extension RuntimeArtifactFileRepository {
    func writeProbabilityCalibrationRuntimeArtifacts(
        symbol: String,
        state: ProbabilityCalibrationRuntimeState
    ) throws {
        guard let stateTSV = ProbabilityCalibrationTools.runtimeStateTSV(symbol: symbol, state: state),
              let historyLine = ProbabilityCalibrationTools.runtimeHistoryNDJSONLine(symbol: symbol, state: state) else {
            return
        }

        let stateURL = url(for: ProbabilityCalibrationTools.runtimeStatePath(symbol: symbol))
        try fileManager.createDirectory(
            at: stateURL.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        try stateTSV.write(to: stateURL, atomically: true, encoding: .utf8)

        let historyURL = url(for: ProbabilityCalibrationTools.runtimeHistoryPath(symbol: symbol))
        try fileManager.createDirectory(
            at: historyURL.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        let historyData = Data((historyLine + "\r\n").utf8)
        if fileManager.fileExists(atPath: historyURL.path) {
            let handle = try FileHandle(forWritingTo: historyURL)
            defer { try? handle.close() }
            try handle.seekToEnd()
            try handle.write(contentsOf: historyData)
        } else {
            try historyData.write(to: historyURL, options: .atomic)
        }
    }
}
