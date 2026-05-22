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

public enum WarmupTools {
    public static let missingScore = -1e9

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
