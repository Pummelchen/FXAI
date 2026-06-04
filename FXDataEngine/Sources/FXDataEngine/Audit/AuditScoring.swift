import Foundation

public struct AuditFoldMetrics: Codable, Hashable, Sendable {
    public var samplesTotal: Int
    public var validPredictions: Int
    public var invalidPredictions: Int
    public var buyCount: Int
    public var sellCount: Int
    public var skipCount: Int
    public var directionalEvaluationCount: Int
    public var directionalCorrectCount: Int
    public var confidenceSum: Double
    public var reliabilitySum: Double
    public var moveSum: Double
    public var brierSum: Double
    public var calibrationAbsSum: Double
    public var pathQualityAbsSum: Double
    public var pathQualityCount: Int
    public var netSum: Double

    public init(
        samplesTotal: Int = 0,
        validPredictions: Int = 0,
        invalidPredictions: Int = 0,
        buyCount: Int = 0,
        sellCount: Int = 0,
        skipCount: Int = 0,
        directionalEvaluationCount: Int = 0,
        directionalCorrectCount: Int = 0,
        confidenceSum: Double = 0.0,
        reliabilitySum: Double = 0.0,
        moveSum: Double = 0.0,
        brierSum: Double = 0.0,
        calibrationAbsSum: Double = 0.0,
        pathQualityAbsSum: Double = 0.0,
        pathQualityCount: Int = 0,
        netSum: Double = 0.0
    ) {
        self.samplesTotal = max(0, samplesTotal)
        self.validPredictions = max(0, validPredictions)
        self.invalidPredictions = max(0, invalidPredictions)
        self.buyCount = max(0, buyCount)
        self.sellCount = max(0, sellCount)
        self.skipCount = max(0, skipCount)
        self.directionalEvaluationCount = max(0, directionalEvaluationCount)
        self.directionalCorrectCount = max(0, directionalCorrectCount)
        self.confidenceSum = fxSafeFinite(confidenceSum)
        self.reliabilitySum = fxSafeFinite(reliabilitySum)
        self.moveSum = fxSafeFinite(moveSum)
        self.brierSum = fxSafeFinite(brierSum)
        self.calibrationAbsSum = fxSafeFinite(calibrationAbsSum)
        self.pathQualityAbsSum = fxSafeFinite(pathQualityAbsSum)
        self.pathQualityCount = max(0, pathQualityCount)
        self.netSum = fxSafeFinite(netSum)
    }

    public mutating func recordInvalidPrediction() {
        samplesTotal += 1
        invalidPredictions += 1
    }

    public mutating func recordValidPrediction(
        decision: LabelClass,
        prediction: PredictionV4,
        brier: Double,
        netPoints: Double,
        directionalEvaluated: Bool,
        directionalCorrect: Bool,
        calibrationAbs: Double,
        pathQuality: Double
    ) {
        samplesTotal += 1
        validPredictions += 1
        switch decision {
        case .buy:
            buyCount += 1
        case .sell:
            sellCount += 1
        case .skip:
            skipCount += 1
        }

        confidenceSum += fxSafeFinite(prediction.confidence)
        reliabilitySum += fxSafeFinite(prediction.reliability)
        moveSum += fxSafeFinite(prediction.moveMeanPoints)
        brierSum += fxSafeFinite(brier)
        netSum += fxSafeFinite(netPoints)
        if directionalEvaluated {
            directionalEvaluationCount += 1
            if directionalCorrect {
                directionalCorrectCount += 1
            }
            calibrationAbsSum += fxSafeFinite(calibrationAbs)
        }
        if pathQuality >= 0.0 {
            pathQualityAbsSum += fxSafeFinite(pathQuality)
            pathQualityCount += 1
        }
    }
}

public struct AuditWalkForwardFoldEvidence: Codable, Hashable, Sendable {
    public var fold: Int
    public var trainSamples: Int
    public var testSamples: Int
    public var trainScore: Double
    public var testScore: Double
    public var gap: Double
    public var passed: Bool
    public var overfit: Bool

    public init(
        fold: Int,
        trainSamples: Int,
        testSamples: Int,
        trainScore: Double,
        testScore: Double,
        gap: Double,
        passed: Bool,
        overfit: Bool
    ) {
        self.fold = max(1, fold)
        self.trainSamples = max(0, trainSamples)
        self.testSamples = max(0, testSamples)
        self.trainScore = fxClamp(trainScore, 0.0, 100.0)
        self.testScore = fxClamp(testScore, 0.0, 100.0)
        self.gap = fxSafeFinite(gap)
        self.passed = passed
        self.overfit = overfit
    }
}

public struct AuditIssueFlags: OptionSet, Codable, Hashable, Sendable {
    public let rawValue: Int

    public init(rawValue: Int) {
        self.rawValue = rawValue
    }

    public static let invalidPrediction = AuditIssueFlags(rawValue: 1)
    public static let overtradesNoise = AuditIssueFlags(rawValue: 2)
    public static let missesTrend = AuditIssueFlags(rawValue: 4)
    public static let calibrationDrift = AuditIssueFlags(rawValue: 8)
    public static let resetDrift = AuditIssueFlags(rawValue: 16)
    public static let sequenceWeak = AuditIssueFlags(rawValue: 32)
    public static let deadOutput = AuditIssueFlags(rawValue: 64)
    public static let sideCollapse = AuditIssueFlags(rawValue: 128)
    public static let walkForwardOverfit = AuditIssueFlags(rawValue: 256)
    public static let walkForwardUnstable = AuditIssueFlags(rawValue: 512)
    public static let walkForwardWeakEdge = AuditIssueFlags(rawValue: 1024)
    public static let macroBlind = AuditIssueFlags(rawValue: 2048)
    public static let macroOverreact = AuditIssueFlags(rawValue: 4096)
    public static let macroDataGap = AuditIssueFlags(rawValue: 8192)
    public static let adversarialWeak = AuditIssueFlags(rawValue: 16384)
}

public struct AuditScenarioMetrics: Codable, Hashable, Sendable {
    public var aiID: Int
    public var aiName: String
    public var family: Int
    public var scenario: String
    public var barsTotal: Int
    public var samplesTotal: Int
    public var validPredictions: Int
    public var invalidPredictions: Int
    public var buyCount: Int
    public var sellCount: Int
    public var skipCount: Int
    public var trueBuyCount: Int
    public var trueSellCount: Int
    public var trueSkipCount: Int
    public var exactMatchCount: Int
    public var directionalEvaluationCount: Int
    public var directionalCorrectCount: Int
    public var trendAlignmentSum: Double
    public var trendAlignmentCount: Int
    public var confidenceSum: Double
    public var reliabilitySum: Double
    public var moveSum: Double
    public var directionalConfidenceSum: Double
    public var directionalHitSum: Double
    public var brierSum: Double
    public var calibrationAbsSum: Double
    public var pathQualityAbsSum: Double
    public var pathQualityCount: Int
    public var netSum: Double
    public var skipRatio: Double
    public var activeRatio: Double
    public var biasAbs: Double
    public var confidenceDrift: Double
    public var brierScore: Double
    public var calibrationError: Double
    public var pathQualityError: Double
    public var macroEventRate: Double
    public var macroPreRate: Double
    public var macroPostRate: Double
    public var macroImportanceMean: Double
    public var macroSurpriseAbsMean: Double
    public var macroDataCoverage: Double
    public var macroSurpriseZAbsMean: Double
    public var macroRevisionAbsMean: Double
    public var macroCurrencyRelevanceMean: Double
    public var macroProvenanceTrustMean: Double
    public var macroRatesRate: Double
    public var macroInflationRate: Double
    public var macroLaborRate: Double
    public var macroGrowthRate: Double
    public var resetDelta: Double
    public var sequenceDelta: Double
    public var walkForwardFolds: Int
    public var walkForwardTrainSamples: Int
    public var walkForwardTestSamples: Int
    public var walkForwardTrainScore: Double
    public var walkForwardTestScore: Double
    public var walkForwardTestScoreStd: Double
    public var walkForwardGap: Double
    public var walkForwardPBO: Double
    public var walkForwardDSR: Double
    public var walkForwardPassRate: Double
    public var walkForwardFoldEvidence: [AuditWalkForwardFoldEvidence]?
    public var score: Double
    public var issueFlags: AuditIssueFlags

    public init(
        aiID: Int = 0,
        aiName: String = "",
        family: Int = 0,
        scenario: String = "",
        barsTotal: Int = 0,
        samplesTotal: Int = 0,
        validPredictions: Int = 0,
        invalidPredictions: Int = 0,
        buyCount: Int = 0,
        sellCount: Int = 0,
        skipCount: Int = 0,
        trueBuyCount: Int = 0,
        trueSellCount: Int = 0,
        trueSkipCount: Int = 0,
        exactMatchCount: Int = 0,
        directionalEvaluationCount: Int = 0,
        directionalCorrectCount: Int = 0,
        trendAlignmentSum: Double = 0.0,
        trendAlignmentCount: Int = 0,
        confidenceSum: Double = 0.0,
        reliabilitySum: Double = 0.0,
        moveSum: Double = 0.0,
        directionalConfidenceSum: Double = 0.0,
        directionalHitSum: Double = 0.0,
        brierSum: Double = 0.0,
        calibrationAbsSum: Double = 0.0,
        pathQualityAbsSum: Double = 0.0,
        pathQualityCount: Int = 0,
        netSum: Double = 0.0,
        skipRatio: Double = 0.0,
        activeRatio: Double = 0.0,
        biasAbs: Double = 0.0,
        confidenceDrift: Double = 0.0,
        brierScore: Double = 0.0,
        calibrationError: Double = 0.0,
        pathQualityError: Double = 0.0,
        macroEventRate: Double = 0.0,
        macroPreRate: Double = 0.0,
        macroPostRate: Double = 0.0,
        macroImportanceMean: Double = 0.0,
        macroSurpriseAbsMean: Double = 0.0,
        macroDataCoverage: Double = 0.0,
        macroSurpriseZAbsMean: Double = 0.0,
        macroRevisionAbsMean: Double = 0.0,
        macroCurrencyRelevanceMean: Double = 0.0,
        macroProvenanceTrustMean: Double = 0.0,
        macroRatesRate: Double = 0.0,
        macroInflationRate: Double = 0.0,
        macroLaborRate: Double = 0.0,
        macroGrowthRate: Double = 0.0,
        resetDelta: Double = 0.0,
        sequenceDelta: Double = 0.0,
        walkForwardFolds: Int = 0,
        walkForwardTrainSamples: Int = 0,
        walkForwardTestSamples: Int = 0,
        walkForwardTrainScore: Double = 0.0,
        walkForwardTestScore: Double = 0.0,
        walkForwardTestScoreStd: Double = 0.0,
        walkForwardGap: Double = 0.0,
        walkForwardPBO: Double = 0.0,
        walkForwardDSR: Double = 0.0,
        walkForwardPassRate: Double = 0.0,
        walkForwardFoldEvidence: [AuditWalkForwardFoldEvidence]? = nil,
        score: Double = 0.0,
        issueFlags: AuditIssueFlags = []
    ) {
        self.aiID = max(0, aiID)
        self.aiName = aiName
        self.family = max(0, family)
        self.scenario = scenario
        self.barsTotal = max(0, barsTotal)
        self.samplesTotal = max(0, samplesTotal)
        self.validPredictions = max(0, validPredictions)
        self.invalidPredictions = max(0, invalidPredictions)
        self.buyCount = max(0, buyCount)
        self.sellCount = max(0, sellCount)
        self.skipCount = max(0, skipCount)
        self.trueBuyCount = max(0, trueBuyCount)
        self.trueSellCount = max(0, trueSellCount)
        self.trueSkipCount = max(0, trueSkipCount)
        self.exactMatchCount = max(0, exactMatchCount)
        self.directionalEvaluationCount = max(0, directionalEvaluationCount)
        self.directionalCorrectCount = max(0, directionalCorrectCount)
        self.trendAlignmentSum = fxSafeFinite(trendAlignmentSum)
        self.trendAlignmentCount = max(0, trendAlignmentCount)
        self.confidenceSum = fxSafeFinite(confidenceSum)
        self.reliabilitySum = fxSafeFinite(reliabilitySum)
        self.moveSum = fxSafeFinite(moveSum)
        self.directionalConfidenceSum = fxSafeFinite(directionalConfidenceSum)
        self.directionalHitSum = fxSafeFinite(directionalHitSum)
        self.brierSum = fxSafeFinite(brierSum)
        self.calibrationAbsSum = fxSafeFinite(calibrationAbsSum)
        self.pathQualityAbsSum = fxSafeFinite(pathQualityAbsSum)
        self.pathQualityCount = max(0, pathQualityCount)
        self.netSum = fxSafeFinite(netSum)
        self.skipRatio = fxSafeFinite(skipRatio)
        self.activeRatio = fxSafeFinite(activeRatio)
        self.biasAbs = fxSafeFinite(biasAbs)
        self.confidenceDrift = fxSafeFinite(confidenceDrift)
        self.brierScore = fxSafeFinite(brierScore)
        self.calibrationError = fxSafeFinite(calibrationError)
        self.pathQualityError = fxSafeFinite(pathQualityError)
        self.macroEventRate = fxSafeFinite(macroEventRate)
        self.macroPreRate = fxSafeFinite(macroPreRate)
        self.macroPostRate = fxSafeFinite(macroPostRate)
        self.macroImportanceMean = fxSafeFinite(macroImportanceMean)
        self.macroSurpriseAbsMean = fxSafeFinite(macroSurpriseAbsMean)
        self.macroDataCoverage = fxSafeFinite(macroDataCoverage)
        self.macroSurpriseZAbsMean = fxSafeFinite(macroSurpriseZAbsMean)
        self.macroRevisionAbsMean = fxSafeFinite(macroRevisionAbsMean)
        self.macroCurrencyRelevanceMean = fxSafeFinite(macroCurrencyRelevanceMean)
        self.macroProvenanceTrustMean = fxSafeFinite(macroProvenanceTrustMean)
        self.macroRatesRate = fxSafeFinite(macroRatesRate)
        self.macroInflationRate = fxSafeFinite(macroInflationRate)
        self.macroLaborRate = fxSafeFinite(macroLaborRate)
        self.macroGrowthRate = fxSafeFinite(macroGrowthRate)
        self.resetDelta = fxSafeFinite(resetDelta)
        self.sequenceDelta = fxSafeFinite(sequenceDelta)
        self.walkForwardFolds = max(0, walkForwardFolds)
        self.walkForwardTrainSamples = max(0, walkForwardTrainSamples)
        self.walkForwardTestSamples = max(0, walkForwardTestSamples)
        self.walkForwardTrainScore = fxSafeFinite(walkForwardTrainScore)
        self.walkForwardTestScore = fxSafeFinite(walkForwardTestScore)
        self.walkForwardTestScoreStd = fxSafeFinite(walkForwardTestScoreStd)
        self.walkForwardGap = fxSafeFinite(walkForwardGap)
        self.walkForwardPBO = fxClamp(walkForwardPBO, 0.0, 1.0)
        self.walkForwardDSR = fxClamp(walkForwardDSR, 0.0, 1.0)
        self.walkForwardPassRate = fxClamp(walkForwardPassRate, 0.0, 1.0)
        self.walkForwardFoldEvidence = walkForwardFoldEvidence
        self.score = fxClamp(score, 0.0, 100.0)
        self.issueFlags = issueFlags
    }
}

public enum AuditScoringTools {
    public static let invalidFoldScore = -1e9

    public static func approximateNormalCDF(_ x: Double) -> Double {
        1.0 / (1.0 + Foundation.exp(-1.702 * fxSafeFinite(x)))
    }

    public static func mean(_ values: [Double]) -> Double {
        guard !values.isEmpty else { return 0.0 }
        return values.reduce(0.0) { $0 + fxSafeFinite($1) } / Double(values.count)
    }

    public static func sampleStandardDeviation(_ values: [Double], mean: Double) -> Double {
        guard values.count > 1 else { return 0.0 }
        var variance = 0.0
        for value in values {
            let delta = fxSafeFinite(value) - mean
            variance += delta * delta
        }
        variance /= Double(values.count - 1)
        return Foundation.sqrt(max(variance, 0.0))
    }

    public static func comparePredictions(_ lhs: PredictionV4, _ rhs: PredictionV4) -> Double {
        var delta = 0.0
        for index in 0..<3 {
            delta += abs(probability(lhs, index: index) - probability(rhs, index: index))
        }
        delta += 0.10 * abs(lhs.moveMeanPoints - rhs.moveMeanPoints)
        delta += 0.04 * abs(lhs.mfeMeanPoints - rhs.mfeMeanPoints)
        delta += 0.04 * abs(lhs.maeMeanPoints - rhs.maeMeanPoints)
        delta += 0.05 * abs(lhs.hitTimeFraction - rhs.hitTimeFraction)
        delta += 0.05 * abs(lhs.pathRisk - rhs.pathRisk)
        delta += 0.05 * abs(lhs.fillRisk - rhs.fillRisk)
        delta += 0.05 * abs(lhs.confidence - rhs.confidence)
        delta += 0.05 * abs(lhs.reliability - rhs.reliability)
        return fxSafeFinite(delta)
    }

    public static func decision(from prediction: PredictionV4) -> LabelClass {
        var best: LabelClass = .skip
        var bestProbability = probability(prediction, index: LabelClass.skip.rawValue)
        if probability(prediction, index: LabelClass.buy.rawValue) > bestProbability {
            best = .buy
            bestProbability = probability(prediction, index: LabelClass.buy.rawValue)
        }
        if probability(prediction, index: LabelClass.sell.rawValue) > bestProbability {
            best = .sell
        }
        return best
    }

    public static func sessionEdgePressure(sampleTimeUTC: Int64) -> Double {
        let date = Date(timeIntervalSince1970: TimeInterval(sampleTimeUTC))
        let calendar = Calendar(identifier: .gregorian)
        let components = calendar.dateComponents(in: TimeZone(secondsFromGMT: 0)!, from: date)
        let hour = Double(components.hour ?? 0) + Double(components.minute ?? 0) / 60.0
        let tokyoDistance = min(abs(hour - 0.0), abs(hour - 24.0))
        let londonDistance = abs(hour - 8.0)
        let newYorkDistance = abs(hour - 16.0)
        let best = min(tokyoDistance, min(londonDistance, newYorkDistance))
        return fxClamp(1.0 - best / 4.0, 0.0, 1.0)
    }

    public static func adversarialWeaknessScore(
        labelClass: LabelClass?,
        movePoints: Double,
        minMovePoints: Double,
        mfePoints: Double,
        maePoints: Double,
        timeToHitFraction: Double,
        pathFlags: SamplePathFlags,
        fillRisk: Double,
        macroActivity: Double,
        sampleTimeUTC: Int64,
        prediction: PredictionV4
    ) -> Double {
        let resolvedLabel = labelClass ?? (movePoints >= 0.0 ? .buy : .sell)
        var target = [0.0, 0.0, 0.0]
        target[resolvedLabel.rawValue] = 1.0

        var brier = 0.0
        for index in 0..<3 {
            let delta = probability(prediction, index: index) - target[index]
            brier += delta * delta
        }

        let predictionDecision = decision(from: prediction)
        let directionalEvaluation = predictionDecision != .skip
        let directionalCorrect = predictionDecision == resolvedLabel
        let directionalConfidence = max(
            probability(prediction, index: LabelClass.buy.rawValue),
            probability(prediction, index: LabelClass.sell.rawValue)
        )
        let calibrationAbs = directionalEvaluation
            ? abs(directionalConfidence - (directionalCorrect ? 1.0 : 0.0))
            : abs(probability(prediction, index: LabelClass.skip.rawValue) - target[LabelClass.skip.rawValue])
        let moveScale = max(abs(movePoints), max(abs(prediction.moveMeanPoints), max(minMovePoints, 0.50)))
        let fillTarget = fxClamp(fillRisk + (pathFlags.contains(.dualHit) ? 0.25 : 0.0), 0.0, 1.0)
        let pathQuality =
            0.25 * fxClamp(abs(prediction.mfeMeanPoints - mfePoints) / moveScale, 0.0, 3.0) +
            0.20 * fxClamp(abs(prediction.maeMeanPoints - maePoints) / moveScale, 0.0, 3.0) +
            0.20 * abs(prediction.hitTimeFraction - timeToHitFraction) +
            0.20 * abs(prediction.pathRisk - fillRisk) +
            0.15 * abs(prediction.fillRisk - fillTarget)

        let wrongDirection = directionalEvaluation && !directionalCorrect ? 1.0 : 0.0
        let noiseOvertrade = resolvedLabel == .skip && directionalEvaluation ? directionalConfidence : 0.0
        let missedTrade = resolvedLabel != .skip && predictionDecision == .skip
            ? 1.0 - probability(prediction, index: LabelClass.skip.rawValue)
            : 0.0
        let stress =
            0.18 * fxClamp(fillRisk, 0.0, 4.0) +
            0.10 * (pathFlags.contains(.dualHit) ? 1.0 : 0.0) +
            0.08 * (pathFlags.contains(.slowHit) ? 1.0 : 0.0) +
            0.08 * fxClamp(macroActivity, 0.0, 1.0) +
            0.08 * sessionEdgePressure(sampleTimeUTC: sampleTimeUTC) +
            0.10 * fxClamp(abs(movePoints) / max(minMovePoints, 0.50), 0.0, 4.0)

        return 0.55 * brier +
            0.40 * calibrationAbs +
            0.32 * pathQuality +
            0.38 * wrongDirection * directionalConfidence +
            0.24 * noiseOvertrade +
            0.18 * missedTrade +
            stress
    }

    public static func scoreFold(_ metrics: AuditFoldMetrics) -> Double {
        if metrics.samplesTotal < 12 || metrics.validPredictions <= 0 {
            return invalidFoldScore
        }

        let invalidRate = Double(metrics.invalidPredictions) / Double(metrics.samplesTotal)
        let activeRatio = Double(metrics.buyCount + metrics.sellCount) / Double(metrics.samplesTotal)
        let skipRatio = Double(metrics.skipCount) / Double(metrics.samplesTotal)
        let hitRate = metrics.directionalEvaluationCount > 0
            ? Double(metrics.directionalCorrectCount) / Double(metrics.directionalEvaluationCount)
            : 0.50
        let brier = metrics.brierSum / Double(metrics.validPredictions)
        let calibration = metrics.directionalEvaluationCount > 0
            ? metrics.calibrationAbsSum / Double(metrics.directionalEvaluationCount)
            : 0.35
        let pathQuality = metrics.pathQualityCount > 0
            ? metrics.pathQualityAbsSum / Double(metrics.pathQualityCount)
            : 0.50
        let averageConfidence = metrics.confidenceSum / Double(metrics.validPredictions)
        let averageReliability = metrics.reliabilitySum / Double(metrics.validPredictions)
        let averageMove = metrics.moveSum / Double(metrics.validPredictions)
        let averageNet = metrics.netSum / Double(metrics.validPredictions)

        var score = 100.0
        score -= 42.0 * invalidRate
        score -= 28.0 * brier
        score -= 16.0 * calibration
        score -= 12.0 * pathQuality
        if activeRatio > 0.78 {
            score -= 18.0 * fxClamp((activeRatio - 0.78) / 0.22, 0.0, 1.0)
        }
        if activeRatio < 0.05 {
            score -= 10.0 * fxClamp((0.05 - activeRatio) / 0.05, 0.0, 1.0)
        }
        if skipRatio > 0.92 {
            score -= 6.0 * fxClamp((skipRatio - 0.92) / 0.08, 0.0, 1.0)
        }
        score += 24.0 * fxClamp(hitRate - 0.50, -0.50, 0.50)
        score += 8.0 * fxClamp(averageReliability - 0.50, -0.50, 0.50)
        score += 4.0 * fxClamp(averageConfidence - 0.50, -0.50, 0.50)
        score += 4.0 * fxClamp(averageMove / 8.0, 0.0, 1.0)
        score += 6.0 * fxClamp(averageNet / 8.0, -1.0, 1.0)
        return fxClamp(score, 0.0, 100.0)
    }

    public static func deflatedSharpeProxy(scores: [Double], pbo: Double) -> Double {
        let count = scores.count
        guard count > 1 else { return 0.0 }
        let returns = scores.map { (fxSafeFinite($0) - 60.0) / 20.0 }
        let meanReturn = mean(returns)
        let stdReturn = sampleStandardDeviation(returns, mean: meanReturn)
        if stdReturn <= 1e-9 {
            return meanReturn > 0.0 ? 1.0 : 0.0
        }

        let sharpe = meanReturn / stdReturn
        let sampleDeflator = Foundation.sqrt(Double(count - 1) / Double(count + 3))
        let selectionPenalty = 0.35 + 0.65 * fxClamp(pbo, 0.0, 1.0)
        let z = sharpe * sampleDeflator - selectionPenalty - 0.08 * Foundation.log(Double(count) + 1.0)
        return fxClamp(approximateNormalCDF(z), 0.0, 1.0)
    }

    public static func finalizeWalkForward(
        scenario: AuditScenarioMetrics = AuditScenarioMetrics(),
        trainFolds: [AuditFoldMetrics],
        testFolds: [AuditFoldMetrics]
    ) -> AuditScenarioMetrics {
        let count = min(trainFolds.count, testFolds.count)
        guard count > 0 else { return scenario }

        var output = scenario
        var trainScores: [Double] = []
        var testScores: [Double] = []
        var foldEvidence: [AuditWalkForwardFoldEvidence] = []
        var passCount = 0
        var overfitCount = 0

        for index in 0..<count {
            let train = trainFolds[index]
            let test = testFolds[index]
            output.walkForwardTrainSamples += train.samplesTotal
            output.walkForwardTestSamples += test.samplesTotal

            let trainScore = scoreFold(train)
            let testScore = scoreFold(test)
            if trainScore <= -1e8 || testScore <= -1e8 {
                continue
            }

            trainScores.append(trainScore)
            testScores.append(testScore)

            let overfit = testScore + 6.0 < trainScore
            let passed = testScore >= 68.0 && testScore + 8.0 >= trainScore
            if overfit {
                overfitCount += 1
            }
            if passed {
                passCount += 1
            }
            foldEvidence.append(
                AuditWalkForwardFoldEvidence(
                    fold: index + 1,
                    trainSamples: train.samplesTotal,
                    testSamples: test.samplesTotal,
                    trainScore: trainScore,
                    testScore: testScore,
                    gap: trainScore - testScore,
                    passed: passed,
                    overfit: overfit
                )
            )
        }

        output.walkForwardFolds = testScores.count
        guard output.walkForwardFolds > 0 else { return output }

        output.walkForwardTrainScore = mean(trainScores)
        output.walkForwardTestScore = mean(testScores)
        output.walkForwardTestScoreStd = sampleStandardDeviation(testScores, mean: output.walkForwardTestScore)
        output.walkForwardGap = output.walkForwardTrainScore - output.walkForwardTestScore
        output.walkForwardPBO = Double(overfitCount) / Double(output.walkForwardFolds)
        output.walkForwardPassRate = Double(passCount) / Double(output.walkForwardFolds)
        output.walkForwardDSR = deflatedSharpeProxy(scores: testScores, pbo: output.walkForwardPBO)
        output.walkForwardFoldEvidence = foldEvidence
        return output
    }

    public static func finalizeScenarioMetrics(
        _ metrics: AuditScenarioMetrics,
        macroDatasetActive: Bool = false,
        macroDatasetSafe: Bool = false
    ) -> AuditScenarioMetrics {
        var output = metrics
        if output.samplesTotal > 0 {
            let denominator = Double(output.samplesTotal)
            output.skipRatio = Double(output.skipCount) / denominator
            output.activeRatio = Double(output.buyCount + output.sellCount) / denominator
            output.macroEventRate /= denominator
            output.macroPreRate /= denominator
            output.macroPostRate /= denominator
            output.macroImportanceMean /= denominator
            output.macroSurpriseAbsMean /= denominator
            output.macroDataCoverage /= denominator
            output.macroSurpriseZAbsMean /= denominator
            output.macroRevisionAbsMean /= denominator
            output.macroCurrencyRelevanceMean /= denominator
            output.macroProvenanceTrustMean /= denominator
            output.macroRatesRate /= denominator
            output.macroInflationRate /= denominator
            output.macroLaborRate /= denominator
            output.macroGrowthRate /= denominator
        }

        let active = output.buyCount + output.sellCount
        if active > 0 {
            output.biasAbs = abs(Double(output.buyCount - output.sellCount)) / Double(active)
        }
        if output.directionalEvaluationCount > 0 {
            let averageConfidence = output.directionalConfidenceSum / Double(output.directionalEvaluationCount)
            let averageHit = output.directionalHitSum / Double(output.directionalEvaluationCount)
            output.confidenceDrift = abs(averageConfidence - averageHit)
            output.calibrationError = output.calibrationAbsSum / Double(output.directionalEvaluationCount)
        }
        if output.validPredictions > 0 {
            output.brierScore = output.brierSum / Double(output.validPredictions)
        }
        if output.pathQualityCount > 0 {
            output.pathQualityError = output.pathQualityAbsSum / Double(output.pathQualityCount)
        }

        let averageNet = output.validPredictions > 0 ? output.netSum / Double(output.validPredictions) : 0.0
        let hitRate = output.directionalEvaluationCount > 0
            ? Double(output.directionalCorrectCount) / Double(output.directionalEvaluationCount)
            : 0.50

        var score = 100.0
        if output.invalidPredictions > 0 {
            score -= 35.0
        }
        if noisyScenario(output.scenario) {
            if output.skipRatio < 0.45 {
                score -= 18.0
            }
            if output.activeRatio > 0.80 {
                score -= 12.0
            }
        }
        if trendScenario(output.scenario), output.trendAlignmentCount > 0 {
            let alignment = output.trendAlignmentSum / Double(output.trendAlignmentCount)
            if alignment < 0.20 {
                score -= 18.0
            }
        }
        if output.scenario == "market_session_edges", output.confidenceDrift > 0.18 {
            score -= 8.0
        }
        if output.confidenceDrift > 0.22 {
            score -= 10.0
        }
        if output.brierScore > 0.52 {
            score -= 8.0
        }
        if output.calibrationError > 0.28 {
            score -= 8.0
        }
        if output.pathQualityError > 0.55 {
            score -= 8.0
        }

        if output.scenario == "market_macro_event" {
            if macroDatasetActive {
                if !macroDatasetSafe {
                    score -= 22.0
                } else {
                    if output.macroDataCoverage < 0.08 { score -= 20.0 }
                    if output.macroEventRate < 0.06 { score -= 16.0 }
                    if output.macroImportanceMean < 0.08 { score -= 10.0 }
                    if output.macroCurrencyRelevanceMean < 0.40 { score -= 8.0 }
                    if output.macroProvenanceTrustMean < 0.45 { score -= 8.0 }
                    if output.activeRatio < 0.05, output.macroEventRate > 0.10 { score -= 8.0 }
                    if output.activeRatio > 0.88, output.macroSurpriseAbsMean < 0.20 { score -= 8.0 }
                    if averageNet < 0.0 {
                        score -= 10.0 * fxClamp(-averageNet / 4.0, 0.0, 1.0)
                    }
                }
            }
        }

        if ["market_session_edges", "market_liquidity_shock", "market_spread_shock", "market_walkforward"].contains(output.scenario),
           averageNet < 0.0 {
            score -= 8.0 * fxClamp(-averageNet / 4.0, 0.0, 1.0)
        }
        if output.scenario == "market_adversarial" {
            if hitRate < 0.53 {
                score -= 12.0 * fxClamp((0.53 - hitRate) / 0.18, 0.0, 1.0)
            }
            if output.confidenceDrift > 0.20 { score -= 8.0 }
            if output.calibrationError > 0.26 { score -= 10.0 }
            if output.pathQualityError > 0.50 { score -= 10.0 }
            if averageNet < 0.0 {
                score -= 12.0 * fxClamp(-averageNet / 4.0, 0.0, 1.0)
            }
            if output.activeRatio < 0.03 { score -= 6.0 }
            if output.activeRatio > 0.90, output.brierScore > 0.42 { score -= 8.0 }
        }
        if output.resetDelta > 0.30 {
            score -= 12.0
        }
        if output.sequenceDelta < 0.005, output.sequenceDelta >= 0.0 {
            score -= 6.0
        }
        if output.moveSum <= 0.0 {
            score -= 8.0
        }

        if output.scenario == "market_walkforward" {
            if output.walkForwardFolds < 3 { score -= 18.0 }
            if output.walkForwardGap > 12.0 { score -= 10.0 }
            if output.walkForwardPBO > 0.45 { score -= 12.0 }
            if output.walkForwardPassRate < 0.55 { score -= 12.0 }
            if output.walkForwardDSR < 0.35 { score -= 10.0 }
            if output.walkForwardTestScore > 0.0, output.walkForwardTestScore < 68.0 { score -= 10.0 }
            if output.walkForwardTestScoreStd > 10.0 { score -= 6.0 }
        }

        output.score = fxClamp(score, 0.0, 100.0)
        output.issueFlags = issueFlags(
            for: output,
            macroDatasetActive: macroDatasetActive,
            macroDatasetSafe: macroDatasetSafe,
            averageNet: averageNet,
            hitRate: hitRate,
            active: active
        )
        return output
    }

    private static func probability(_ prediction: PredictionV4, index: Int) -> Double {
        guard index >= 0, index < prediction.classProbabilities.count else { return 0.0 }
        return fxSafeFinite(prediction.classProbabilities[index])
    }

    private static func noisyScenario(_ scenario: String) -> Bool {
        ["random_walk", "market_chop", "market_liquidity_shock", "market_spread_shock"].contains(scenario)
    }

    private static func trendScenario(_ scenario: String) -> Bool {
        [
            "drift_up",
            "drift_down",
            "monotonic_up",
            "monotonic_down",
            "market_trend",
            "market_walkforward"
        ].contains(scenario)
    }

    private static func issueFlags(
        for metrics: AuditScenarioMetrics,
        macroDatasetActive: Bool,
        macroDatasetSafe: Bool,
        averageNet: Double,
        hitRate: Double,
        active: Int
    ) -> AuditIssueFlags {
        var flags: AuditIssueFlags = []
        if metrics.invalidPredictions > 0 {
            flags.insert(.invalidPrediction)
        }
        if (noisyScenario(metrics.scenario) || metrics.scenario == "market_session_edges"),
           metrics.skipRatio < 0.55 || metrics.activeRatio > 0.70 {
            flags.insert(.overtradesNoise)
        }
        if metrics.scenario == "market_macro_event", macroDatasetActive {
            if !macroDatasetSafe || metrics.macroDataCoverage < 0.05 {
                flags.insert(.macroDataGap)
            }
            if macroDatasetSafe, metrics.activeRatio < 0.05, metrics.macroEventRate > 0.10 {
                flags.insert(.macroBlind)
            }
            if macroDatasetSafe, metrics.activeRatio > 0.88, metrics.macroSurpriseAbsMean < 0.20 {
                flags.insert(.macroOverreact)
            }
        }
        if trendScenario(metrics.scenario),
           metrics.trendAlignmentCount > 0,
           metrics.trendAlignmentSum / Double(metrics.trendAlignmentCount) < 0.25 {
            flags.insert(.missesTrend)
        }
        if metrics.confidenceDrift > 0.22 {
            flags.insert(.calibrationDrift)
        }
        if metrics.resetDelta > 0.30 {
            flags.insert(.resetDrift)
        }
        if metrics.sequenceDelta >= 0.0, metrics.sequenceDelta < 0.005 {
            flags.insert(.sequenceWeak)
        }
        if metrics.moveSum <= 0.0 {
            flags.insert(.deadOutput)
        }
        if noisyScenario(metrics.scenario), metrics.biasAbs > 0.85, active > 24 {
            flags.insert(.sideCollapse)
        }
        if metrics.scenario == "market_adversarial",
           metrics.score < 68.0 ||
            averageNet < 0.0 ||
            hitRate < 0.53 ||
            metrics.calibrationError > 0.26 ||
            metrics.pathQualityError > 0.50 {
            flags.insert(.adversarialWeak)
        }
        if metrics.scenario == "market_walkforward" {
            if metrics.walkForwardPBO > 0.45 || metrics.walkForwardGap > 12.0 {
                flags.insert(.walkForwardOverfit)
            }
            if metrics.walkForwardFolds < 3 ||
                metrics.walkForwardPassRate < 0.55 ||
                metrics.walkForwardTestScoreStd > 10.0 {
                flags.insert(.walkForwardUnstable)
            }
            if metrics.walkForwardDSR < 0.35 ||
                (metrics.walkForwardTestScore > 0.0 && metrics.walkForwardTestScore < 68.0) {
                flags.insert(.walkForwardWeakEdge)
            }
        }
        return flags
    }
}
