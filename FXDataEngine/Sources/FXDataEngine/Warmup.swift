import Foundation

public struct WarmupBucketStats: Codable, Hashable, Sendable {
    public var trades: Int
    public var wins: Int
    public var netSum: Double
    public var grossPositive: Double
    public var grossNegative: Double
    public var equity: Double
    public var equityPeak: Double
    public var maxDrawdown: Double

    public init(
        trades: Int = 0,
        wins: Int = 0,
        netSum: Double = 0.0,
        grossPositive: Double = 0.0,
        grossNegative: Double = 0.0,
        equity: Double = 0.0,
        equityPeak: Double = 0.0,
        maxDrawdown: Double = 0.0
    ) {
        let clampedTrades = max(0, trades)
        self.trades = clampedTrades
        self.wins = min(max(0, wins), clampedTrades)
        self.netSum = fxSafeFinite(netSum)
        self.grossPositive = max(0.0, fxSafeFinite(grossPositive))
        self.grossNegative = max(0.0, fxSafeFinite(grossNegative))
        self.equity = fxSafeFinite(equity)
        self.equityPeak = fxSafeFinite(equityPeak)
        self.maxDrawdown = max(0.0, fxSafeFinite(maxDrawdown))
    }

    public mutating func update(netPoints: Double) {
        let netPoints = fxSafeFinite(netPoints)
        netSum += netPoints
        if netPoints >= 0.0 {
            grossPositive += netPoints
        } else {
            grossNegative += -netPoints
        }
        equity += netPoints
        if equity > equityPeak {
            equityPeak = equity
        }
        let drawdown = equityPeak - equity
        if drawdown > maxDrawdown {
            maxDrawdown = drawdown
        }
        trades += 1
        if netPoints > 0.0 {
            wins += 1
        }
    }
}

public struct WarmupThresholdPair: Codable, Hashable, Sendable {
    public var buy: Double
    public var sell: Double

    public init(buy: Double, sell: Double) {
        self.buy = buy
        self.sell = sell
    }
}

public struct WarmupExecutionPlan: Codable, Hashable, Sendable {
    public var samples: Int
    public var loops: Int
    public var trainEpochs: Int
    public var folds: Int
    public var minimumTrades: Int
    public var baseHorizonMinutes: Int
    public var horizons: [Int]
    public var maxHorizonMinutes: Int
    public var featureLookback: Int
    public var neededBars: Int
    public var thresholds: WarmupThresholdPair
    public var evThresholdPoints: Double
    public var costBufferPoints: Double
    public var evLookbackSamples: Int
    public var aiHint: Int
    public var transferSampleCap: Int
    public var portfolioEvaluationCap: Int

    public init(
        samples: Int,
        loops: Int,
        trainEpochs: Int,
        folds: Int,
        minimumTrades: Int,
        baseHorizonMinutes: Int,
        horizons: [Int],
        maxHorizonMinutes: Int,
        featureLookback: Int,
        neededBars: Int,
        thresholds: WarmupThresholdPair,
        evThresholdPoints: Double,
        costBufferPoints: Double,
        evLookbackSamples: Int,
        aiHint: Int,
        transferSampleCap: Int,
        portfolioEvaluationCap: Int
    ) {
        self.samples = samples
        self.loops = loops
        self.trainEpochs = trainEpochs
        self.folds = folds
        self.minimumTrades = minimumTrades
        self.baseHorizonMinutes = baseHorizonMinutes
        self.horizons = horizons
        self.maxHorizonMinutes = maxHorizonMinutes
        self.featureLookback = featureLookback
        self.neededBars = neededBars
        self.thresholds = thresholds
        self.evThresholdPoints = evThresholdPoints
        self.costBufferPoints = costBufferPoints
        self.evLookbackSamples = evLookbackSamples
        self.aiHint = aiHint
        self.transferSampleCap = transferSampleCap
        self.portfolioEvaluationCap = portfolioEvaluationCap
    }
}

public struct WarmupCandidateSplit: Codable, Hashable, Sendable {
    public var validationStart: Int
    public var validationEnd: Int
    public var trainingStart: Int
    public var trainingEnd: Int

    public init(validationStart: Int, validationEnd: Int, trainingStart: Int, trainingEnd: Int) {
        self.validationStart = validationStart
        self.validationEnd = validationEnd
        self.trainingStart = trainingStart
        self.trainingEnd = trainingEnd
    }
}

public struct WarmupAdaptiveThresholds: Codable, Hashable, Sendable {
    public var buyMinProbability: Double
    public var sellMinProbability: Double
    public var skipMinProbability: Double

    public init(buyMinProbability: Double, sellMinProbability: Double, skipMinProbability: Double) {
        self.buyMinProbability = buyMinProbability
        self.sellMinProbability = sellMinProbability
        self.skipMinProbability = skipMinProbability
    }
}

public struct WarmupPortfolioContribution: Codable, Hashable, Sendable {
    public var edge: Double
    public var weight: Double
    public var absoluteCorrelation: Double
    public var diversificationWeight: Double

    public init(edge: Double, weight: Double, absoluteCorrelation: Double = 0.0, diversificationWeight: Double = 1.0) {
        self.edge = fxSafeFinite(edge)
        self.weight = max(0.0, fxSafeFinite(weight))
        self.absoluteCorrelation = fxClamp(absoluteCorrelation, 0.0, 1.0)
        self.diversificationWeight = fxClamp(diversificationWeight, 0.0, 1.0)
    }
}

public struct WarmupPortfolioDiagnostics: Codable, Hashable, Sendable {
    public var meanEdge: Double
    public var stability: Double
    public var correlationPenalty: Double
    public var diversification: Double
    public var symbolCount: Int

    public init(
        meanEdge: Double,
        stability: Double,
        correlationPenalty: Double,
        diversification: Double,
        symbolCount: Int
    ) {
        self.meanEdge = fxSafeFinite(meanEdge)
        self.stability = fxClamp(stability, 0.0, 1.0)
        self.correlationPenalty = fxClamp(correlationPenalty, 0.0, 1.0)
        self.diversification = fxClamp(diversification, 0.0, 1.0)
        self.symbolCount = max(0, symbolCount)
    }
}

public enum WarmupTools {
    public static let missingScore = -1e9
    public static let transferSeedSymbols = [
        "EURUSD", "USDJPY", "GBPUSD", "AUDUSD", "USDCAD",
        "USDCHF", "NZDUSD", "EURJPY", "EURGBP", "EURAUD"
    ]
    public static let sgdLogitAIID = 14
    public static let ftrlLogitAIID = 4
    public static let lightGBMAIID = 6
    public static let xgbFastAIID = 20
    public static let seriousNativeAIIDs: Set<Int> = [32, 33, 34, 35, 25, lightGBMAIID, xgbFastAIID]

    public static func sanitizeThresholdPair(buyThreshold: Double, sellThreshold: Double) -> WarmupThresholdPair {
        var buy = fxClamp(buyThreshold, 0.50, 0.95)
        var sell = fxClamp(sellThreshold, 0.05, 0.50)

        if sell >= buy {
            sell = fxClamp(sell, 0.05, 0.49)
            buy = fxClamp(max(buy, sell + 0.01), 0.50, 0.95)
            if sell >= buy {
                sell = 0.49
                buy = 0.50
            }
        }

        return WarmupThresholdPair(buy: buy, sell: sell)
    }

    public static func resolveExecutionPlan(
        warmupSamples: Int,
        warmupLoops: Int,
        trainEpochs: Int,
        warmupFolds: Int,
        warmupMinimumTrades: Int,
        predictionTargetMinutes: Int,
        configuredHorizons: [Int],
        multiHorizon: Bool,
        buyThreshold: Double,
        sellThreshold: Double,
        evThresholdPoints: Double,
        costBufferPoints: Double,
        evLookbackSamples: Int,
        aiType: Int,
        ensemble: Bool,
        maxHorizons: Int = RuntimeArtifactConstants.maxHorizons,
        featureLookback: Int = 10
    ) -> WarmupExecutionPlan {
        let samples = min(max(warmupSamples, 2_000), 50_000)
        let loops = min(max(warmupLoops, 10), 500)
        let epochs = min(max(trainEpochs, 1), 5)
        let folds = min(max(warmupFolds, 2), 5)
        let minimumTrades = min(max(warmupMinimumTrades, 20), 2_000)
        let baseHorizon = HorizonTools.clampHorizon(predictionTargetMinutes)
        let horizonLimit = max(1, maxHorizons)

        var horizons: [Int] = []
        if multiHorizon, !configuredHorizons.isEmpty {
            for horizon in configuredHorizons.prefix(horizonLimit) {
                horizons.append(HorizonTools.clampHorizon(horizon))
            }
        }
        if horizons.isEmpty {
            horizons.append(baseHorizon)
        }

        var hasPrimary = false
        var maxHorizon = baseHorizon
        for horizon in horizons {
            if horizon == baseHorizon {
                hasPrimary = true
            }
            if horizon > maxHorizon {
                maxHorizon = horizon
            }
        }

        if !hasPrimary, horizons.count < horizonLimit {
            horizons.append(baseHorizon)
            if baseHorizon > maxHorizon {
                maxHorizon = baseHorizon
            }
        }

        let lookback = max(0, featureLookback)
        let neededBars = samples + maxHorizon + lookback
        let evLookback = min(max(evLookbackSamples, 20), 400)
        let aiHint = ensemble || aiType < -1 || aiType >= FXDataEngineConstants.aiCount ? -1 : aiType
        let transferCap = min(640, max(128, samples / 6))
        let portfolioCap = min(160, max(64, samples / 12))

        return WarmupExecutionPlan(
            samples: samples,
            loops: loops,
            trainEpochs: epochs,
            folds: folds,
            minimumTrades: minimumTrades,
            baseHorizonMinutes: baseHorizon,
            horizons: horizons,
            maxHorizonMinutes: maxHorizon,
            featureLookback: lookback,
            neededBars: neededBars,
            thresholds: sanitizeThresholdPair(buyThreshold: buyThreshold, sellThreshold: sellThreshold),
            evThresholdPoints: fxClamp(evThresholdPoints, 0.0, 100.0),
            costBufferPoints: max(0.0, fxSafeFinite(costBufferPoints)),
            evLookbackSamples: evLookback,
            aiHint: aiHint,
            transferSampleCap: transferCap,
            portfolioEvaluationCap: portfolioCap
        )
    }

    public static func transferUniverse(mainSymbol: String, contextSymbols: [String]) -> [String] {
        var symbols: [String] = []
        if !mainSymbol.isEmpty {
            symbols.append(mainSymbol)
        }

        for candidate in contextSymbols where !candidate.isEmpty {
            if !symbols.contains(candidate) {
                symbols.append(candidate)
            }
        }

        for candidate in transferSeedSymbols where !candidate.isEmpty {
            if !symbols.contains(candidate) {
                symbols.append(candidate)
            }
        }

        return symbols
    }

    public static func transferSampleRange(
        neededBars: Int,
        maxHorizonMinutes: Int,
        sampleCap: Int,
        featureLookback: Int = 10
    ) -> ClosedRange<Int>? {
        guard sampleCap > 0 else { return nil }
        let start = maxHorizonMinutes
        var end = start + sampleCap - 1
        let maxValid = neededBars - max(0, featureLookback) - 1
        if end > maxValid {
            end = maxValid
        }
        guard end > start else { return nil }
        return start...end
    }

    public static func cappedValidSampleIndexes(validFlags: [Bool], sampleCap: Int) -> [Int] {
        guard sampleCap > 0 else { return [] }
        let validTotal = validFlags.filter { $0 }.count
        guard validTotal > 0 else { return [] }

        let stride = validTotal > sampleCap ? max(1, validTotal / sampleCap) : 1
        var emitted: [Int] = []
        emitted.reserveCapacity(min(validTotal, sampleCap))
        var seen = 0
        for index in validFlags.indices.reversed() {
            guard validFlags[index] else { continue }
            if (seen % stride) != 0 {
                seen += 1
                continue
            }
            seen += 1
            emitted.append(index)
            if emitted.count >= sampleCap {
                break
            }
        }
        return emitted
    }

    public static func cappedValidSampleIndexes(samples: [PreparedTrainingSample], sampleCap: Int) -> [Int] {
        cappedValidSampleIndexes(validFlags: samples.map(\.valid), sampleCap: sampleCap)
    }

    public static func normalizationScoringModelIDs(primaryAI: Int, ensemble: Bool) -> [Int] {
        let primary = (0..<FXDataEngineConstants.aiCount).contains(primaryAI) ? primaryAI : sgdLogitAIID
        let anchors = [primary, ftrlLogitAIID, xgbFastAIID, lightGBMAIID]
        let maxModels = min(max(ensemble ? 4 : 1, 1), 4)
        var modelIDs: [Int] = []
        for id in anchors {
            guard !modelIDs.contains(id) else { continue }
            modelIDs.append(id)
            if modelIDs.count >= maxModels {
                break
            }
        }
        return modelIDs
    }

    public static func normalizationCandidateSplit(
        horizonMinutes: Int,
        startIndex: Int,
        endIndex: Int
    ) -> WarmupCandidateSplit? {
        let span = endIndex - startIndex + 1
        guard span >= 240 else { return nil }

        var validationLength = span / 3
        if validationLength < 80 {
            validationLength = 80
        }
        if validationLength > 240 {
            validationLength = 240
        }

        let validationStart = startIndex
        var validationEnd = validationStart + validationLength - 1
        if validationEnd >= endIndex {
            validationEnd = endIndex - 1
        }
        guard validationEnd > validationStart else { return nil }

        var purge = horizonMinutes + 240
        if purge < horizonMinutes + 40 {
            purge = horizonMinutes + 40
        }
        let trainingStart = validationEnd + purge + 1
        let trainingEnd = endIndex
        guard trainingEnd - trainingStart >= 100 else { return nil }

        return WarmupCandidateSplit(
            validationStart: validationStart,
            validationEnd: validationEnd,
            trainingStart: trainingStart,
            trainingEnd: trainingEnd
        )
    }

    public static func warmupFoldSplits(
        horizonMinutes: Int,
        startIndex: Int,
        endIndex: Int,
        folds: Int
    ) -> [WarmupCandidateSplit] {
        let sampleSpan = endIndex - startIndex + 1
        guard sampleSpan > 0, folds > 0 else { return [] }

        var foldLength = sampleSpan / (folds + 1)
        if foldLength < 40 {
            foldLength = 40
        }
        if foldLength > sampleSpan / 2 {
            foldLength = sampleSpan / 2
        }
        guard foldLength >= 20 else { return [] }

        var splits: [WarmupCandidateSplit] = []
        splits.reserveCapacity(folds)
        for fold in 0..<folds {
            var validationStart = startIndex + (fold * foldLength)
            var validationEnd = validationStart + foldLength - 1
            if validationStart < startIndex {
                validationStart = startIndex
            }
            if validationEnd >= endIndex {
                validationEnd = endIndex - 1
            }
            guard validationEnd > validationStart else { continue }

            var purge = horizonMinutes + 240
            if purge < horizonMinutes + 40 {
                purge = horizonMinutes + 40
            }
            let trainingStart = validationEnd + purge + 1
            let trainingEnd = endIndex
            guard trainingEnd - trainingStart >= 100 else { continue }

            splits.append(
                WarmupCandidateSplit(
                    validationStart: validationStart,
                    validationEnd: validationEnd,
                    trainingStart: trainingStart,
                    trainingEnd: trainingEnd
                )
            )
        }
        return splits
    }

    public static func deriveAdaptiveThresholds(
        baseBuyThreshold: Double,
        baseSellThreshold: Double,
        minMovePoints: Double,
        expectedMovePoints: Double,
        volatilityProxy: Double
    ) -> WarmupAdaptiveThresholds {
        let buyBase = fxClamp(baseBuyThreshold, 0.50, 0.95)
        let sellBase = fxClamp(1.0 - baseSellThreshold, 0.50, 0.95)
        let expectedMove = max(expectedMovePoints, minMovePoints + 0.10)
        let costRatio = fxClamp(minMovePoints / expectedMove, 0.0, 2.0)
        let volatilityRatio = fxClamp(volatilityProxy / 4.0, 0.0, 1.0)
        let tighten = fxClamp(((costRatio - 0.35) * 0.35) + (0.10 * volatilityRatio), 0.0, 0.25)

        return WarmupAdaptiveThresholds(
            buyMinProbability: fxClamp(buyBase + tighten, 0.50, 0.96),
            sellMinProbability: fxClamp(sellBase + tighten, 0.50, 0.96),
            skipMinProbability: fxClamp(0.45 + (0.20 * costRatio) + (0.10 * volatilityRatio), 0.35, 0.85)
        )
    }

    public static func classSignalFromEV(
        probabilities: [Double],
        thresholds: WarmupAdaptiveThresholds,
        expectedMovePoints: Double,
        minMovePoints: Double,
        evThresholdPoints: Double
    ) -> LabelClass? {
        guard expectedMovePoints > 0.0 else { return nil }
        let sellProbability = HorizonTools.value(in: probabilities, index: LabelClass.sell.rawValue, default: 0.0)
        let buyProbability = HorizonTools.value(in: probabilities, index: LabelClass.buy.rawValue, default: 0.0)
        let skipProbability = HorizonTools.value(in: probabilities, index: LabelClass.skip.rawValue, default: 0.0)

        if skipProbability >= thresholds.skipMinProbability {
            return nil
        }

        let buyEV = ((2.0 * buyProbability) - 1.0) * expectedMovePoints - minMovePoints
        let sellEV = ((2.0 * sellProbability) - 1.0) * expectedMovePoints - minMovePoints
        let evThreshold = fxSafeFinite(evThresholdPoints)
        if buyProbability >= thresholds.buyMinProbability, buyEV >= evThreshold, buyEV > sellEV {
            return .buy
        }
        if sellProbability >= thresholds.sellMinProbability, sellEV >= evThreshold, sellEV > buyEV {
            return .sell
        }
        return nil
    }

    public static func warmupEpochBudget(
        aiID: Int,
        horizonMinutes: Int,
        baseEpochs: Int,
        symbol: String
    ) -> Int {
        var epochs = max(baseEpochs, 1)
        guard seriousNativeAIIDs.contains(aiID) else { return epochs }

        let scale = ModelContextTools.modelCapacityScale(symbol: symbol, horizonMinutes: max(horizonMinutes, 1))
        var bonus = 1
        if scale > 1.05 {
            bonus += 1
        }
        if scale > 1.18 {
            bonus += 1
        }
        if scale > 1.28 {
            bonus += 1
        }
        if horizonMinutes >= 15 {
            bonus += 1
        }
        if horizonMinutes >= 30 {
            bonus += 1
        }
        if horizonMinutes >= 60 {
            bonus += 1
        }
        epochs += bonus
        return min(epochs, 10)
    }

    public static func estimatePortfolioSymbolCorrelation(samples: [PreparedTrainingSample]) -> Double {
        var correlationSum = 0.0
        var used = 0
        for sample in samples where sample.valid {
            correlationSum += abs(HorizonTools.value(in: sample.x, index: 53, default: 0.0))
            used += 1
        }
        guard used > 0 else { return 0.0 }
        return fxClamp(correlationSum / Double(used), 0.0, 1.0)
    }

    public static func primaryPortfolioWeight(tradeRate: Double) -> Double {
        fxClamp(0.70 + 0.30 * tradeRate, 0.35, 1.20)
    }

    public static func transferDiversificationWeight(absoluteCorrelation: Double) -> Double {
        fxClamp(1.0 - 0.60 * absoluteCorrelation, 0.25, 1.0)
    }

    public static func transferPortfolioWeight(tradeRate: Double, diversificationWeight: Double) -> Double {
        fxClamp(diversificationWeight * (0.55 + 0.45 * tradeRate), 0.15, 1.10)
    }

    public static func portfolioDiagnostics(
        contributions: [WarmupPortfolioContribution]
    ) -> WarmupPortfolioDiagnostics? {
        var scoreSum = 0.0
        var scoreSquareSum = 0.0
        var weightSum = 0.0
        var correlationSum = 0.0
        var diversificationSum = 0.0
        var symbolCount = 0

        for contribution in contributions where contribution.weight > 0.0 {
            scoreSum += contribution.weight * contribution.edge
            scoreSquareSum += contribution.weight * contribution.edge * contribution.edge
            weightSum += contribution.weight
            correlationSum += contribution.weight * contribution.absoluteCorrelation
            diversificationSum += contribution.weight * contribution.diversificationWeight
            symbolCount += 1
        }

        guard weightSum > 0.0 else { return nil }
        let meanEdge = scoreSum / weightSum
        let variance = max(scoreSquareSum / weightSum - meanEdge * meanEdge, 0.0)
        let standardDeviation = sqrt(variance)
        let scale = max(abs(meanEdge), 0.50)
        let stability = 1.0 - fxClamp(standardDeviation / scale, 0.0, 1.0)
        let correlationPenalty = correlationSum > 0.0 ? correlationSum / weightSum : 0.0
        let diversification = fxClamp(diversificationSum / weightSum, 0.0, 1.0)
        return WarmupPortfolioDiagnostics(
            meanEdge: meanEdge,
            stability: stability,
            correlationPenalty: correlationPenalty,
            diversification: diversification,
            symbolCount: symbolCount
        )
    }

    public static func scoreBucket(_ stats: WarmupBucketStats) -> Double {
        guard stats.trades > 0 else { return missingScore }
        let tradeCount = Double(stats.trades)
        let winRate = Double(stats.wins) / tradeCount
        let averageNet = stats.netSum / tradeCount
        let profitFactor = min(stats.grossPositive / max(stats.grossNegative, 1e-6), 8.0)
        let drawdownPenalty: Double
        if stats.grossPositive > 0.0 {
            drawdownPenalty = stats.maxDrawdown / stats.grossPositive
        } else if stats.maxDrawdown > 0.0 {
            drawdownPenalty = 2.0
        } else {
            drawdownPenalty = 0.0
        }
        return (averageNet * 5.0) +
            (winRate * 1.75) +
            (0.80 * profitFactor) -
            (1.50 * drawdownPenalty)
    }

    public static func portfolioObjectiveProxy(
        totalScore: Double,
        totalTrades: Int,
        regimeScores: [Double],
        regimeTrades: [Int]
    ) -> Double {
        var mean = 0.0
        var squareSum = 0.0
        var used = 0
        var covered = 0
        for index in 0..<min(regimeScores.count, regimeTrades.count) {
            guard regimeTrades[index] > 0 else { continue }
            let score = fxSafeFinite(regimeScores[index])
            mean += score
            squareSum += score * score
            used += 1
            if regimeTrades[index] >= 12 {
                covered += 1
            }
        }
        guard used > 0 else { return 0.0 }

        let usedCount = Double(used)
        mean /= usedCount
        let variance = max(squareSum / usedCount - mean * mean, 0.0)
        let standardDeviation = sqrt(variance)
        let stability = 1.0 - fxClamp(standardDeviation / max(abs(mean), 0.50), 0.0, 1.0)
        let diversification = fxClamp(Double(covered) / 4.0, 0.0, 1.0)
        let tradeCoverage = fxClamp(Double(max(totalTrades, 0)) / 64.0, 0.0, 1.0)
        let edgeNorm = fxClamp(totalScore / 100.0, -1.0, 1.0)
        let objective = 0.35 * stability +
            0.25 * diversification +
            0.20 * tradeCoverage +
            0.20 * (0.5 + 0.5 * edgeNorm)
        return fxClamp(1.20 * (objective - 0.50), -0.60, 0.60)
    }
}
