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

public enum WarmupTools {
    public static let missingScore = -1e9

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
