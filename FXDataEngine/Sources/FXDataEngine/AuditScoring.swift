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

public struct AuditScenarioMetrics: Codable, Hashable, Sendable {
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

    public init(
        walkForwardFolds: Int = 0,
        walkForwardTrainSamples: Int = 0,
        walkForwardTestSamples: Int = 0,
        walkForwardTrainScore: Double = 0.0,
        walkForwardTestScore: Double = 0.0,
        walkForwardTestScoreStd: Double = 0.0,
        walkForwardGap: Double = 0.0,
        walkForwardPBO: Double = 0.0,
        walkForwardDSR: Double = 0.0,
        walkForwardPassRate: Double = 0.0
    ) {
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

            if testScore + 6.0 < trainScore {
                overfitCount += 1
            }
            if testScore >= 68.0 && testScore + 8.0 >= trainScore {
                passCount += 1
            }
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
        return output
    }

    private static func probability(_ prediction: PredictionV4, index: Int) -> Double {
        guard index >= 0, index < prediction.classProbabilities.count else { return 0.0 }
        return fxSafeFinite(prediction.classProbabilities[index])
    }
}
