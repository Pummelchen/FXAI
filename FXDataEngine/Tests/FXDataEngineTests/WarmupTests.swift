import XCTest
@testable import FXDataEngine

final class WarmupTests: XCTestCase {
    func testWarmupThresholdSanitizerMatchesLegacyRules() {
        XCTAssertEqual(
            WarmupTools.sanitizeThresholdPair(buyThreshold: 0.97, sellThreshold: 0.02),
            WarmupThresholdPair(buy: 0.95, sell: 0.05)
        )
        XCTAssertEqual(
            WarmupTools.sanitizeThresholdPair(buyThreshold: 0.40, sellThreshold: 0.60),
            WarmupThresholdPair(buy: 0.50, sell: 0.49)
        )
        XCTAssertEqual(
            WarmupTools.sanitizeThresholdPair(buyThreshold: 0.72, sellThreshold: 0.31),
            WarmupThresholdPair(buy: 0.72, sell: 0.31)
        )
    }

    func testWarmupExecutionPlanClampsSingleHorizonDefaults() {
        let plan = WarmupTools.resolveExecutionPlan(
            warmupSamples: 100,
            warmupLoops: 5,
            trainEpochs: 0,
            warmupFolds: 1,
            warmupMinimumTrades: 5,
            predictionTargetMinutes: 0,
            configuredHorizons: [5, 13],
            multiHorizon: false,
            buyThreshold: 0.40,
            sellThreshold: 0.60,
            evThresholdPoints: -3.0,
            costBufferPoints: -1.0,
            evLookbackSamples: 5,
            aiType: 100,
            ensemble: false
        )

        XCTAssertEqual(plan.samples, 2_000)
        XCTAssertEqual(plan.loops, 10)
        XCTAssertEqual(plan.trainEpochs, 1)
        XCTAssertEqual(plan.folds, 2)
        XCTAssertEqual(plan.minimumTrades, 20)
        XCTAssertEqual(plan.baseHorizonMinutes, 1)
        XCTAssertEqual(plan.horizons, [1])
        XCTAssertEqual(plan.maxHorizonMinutes, 1)
        XCTAssertEqual(plan.neededBars, 2_011)
        XCTAssertEqual(plan.thresholds, WarmupThresholdPair(buy: 0.50, sell: 0.49))
        XCTAssertEqual(plan.evThresholdPoints, 0.0, accuracy: 0.0)
        XCTAssertEqual(plan.costBufferPoints, 0.0, accuracy: 0.0)
        XCTAssertEqual(plan.evLookbackSamples, 20)
        XCTAssertEqual(plan.aiHint, -1)
        XCTAssertEqual(plan.transferSampleCap, 333)
        XCTAssertEqual(plan.portfolioEvaluationCap, 160)
    }

    func testWarmupExecutionPlanPreservesConfiguredHorizonOrderAndCaps() {
        let plan = WarmupTools.resolveExecutionPlan(
            warmupSamples: 60_000,
            warmupLoops: 700,
            trainEpochs: 9,
            warmupFolds: 8,
            warmupMinimumTrades: 5_000,
            predictionTargetMinutes: 34,
            configuredHorizons: [13, 5, 999, 34, 1, 2, 3, 8, 21],
            multiHorizon: true,
            buyThreshold: 0.72,
            sellThreshold: 0.31,
            evThresholdPoints: 180.0,
            costBufferPoints: 2.5,
            evLookbackSamples: 500,
            aiType: 3,
            ensemble: true
        )

        XCTAssertEqual(plan.samples, 50_000)
        XCTAssertEqual(plan.loops, 500)
        XCTAssertEqual(plan.trainEpochs, 5)
        XCTAssertEqual(plan.folds, 5)
        XCTAssertEqual(plan.minimumTrades, 2_000)
        XCTAssertEqual(plan.horizons, [13, 5, 720, 34, 1, 2, 3, 8])
        XCTAssertEqual(plan.maxHorizonMinutes, 720)
        XCTAssertEqual(plan.neededBars, 50_730)
        XCTAssertEqual(plan.thresholds, WarmupThresholdPair(buy: 0.72, sell: 0.31))
        XCTAssertEqual(plan.evThresholdPoints, 100.0, accuracy: 0.0)
        XCTAssertEqual(plan.costBufferPoints, 2.5, accuracy: 0.0)
        XCTAssertEqual(plan.evLookbackSamples, 400)
        XCTAssertEqual(plan.aiHint, -1)
        XCTAssertEqual(plan.transferSampleCap, 640)
        XCTAssertEqual(plan.portfolioEvaluationCap, 160)

        let appended = WarmupTools.resolveExecutionPlan(
            warmupSamples: 2_000,
            warmupLoops: 10,
            trainEpochs: 1,
            warmupFolds: 2,
            warmupMinimumTrades: 20,
            predictionTargetMinutes: 34,
            configuredHorizons: [5, 13],
            multiHorizon: true,
            buyThreshold: 0.72,
            sellThreshold: 0.31,
            evThresholdPoints: 1.0,
            costBufferPoints: 0.0,
            evLookbackSamples: 20,
            aiType: 3,
            ensemble: false
        )
        XCTAssertEqual(appended.horizons, [5, 13, 34])
        XCTAssertEqual(appended.aiHint, 3)
    }

    func testWarmupBucketStatsUpdateAndScoreMatchLegacyFormula() {
        var stats = WarmupBucketStats()
        for value in [3.0, -2.0, 0.0, 5.0, -1.0] {
            stats.update(netPoints: value)
        }

        XCTAssertEqual(stats.trades, 5)
        XCTAssertEqual(stats.wins, 2)
        XCTAssertEqual(stats.netSum, 5.0, accuracy: 1e-12)
        XCTAssertEqual(stats.grossPositive, 8.0, accuracy: 1e-12)
        XCTAssertEqual(stats.grossNegative, 3.0, accuracy: 1e-12)
        XCTAssertEqual(stats.equity, 5.0, accuracy: 1e-12)
        XCTAssertEqual(stats.equityPeak, 6.0, accuracy: 1e-12)
        XCTAssertEqual(stats.maxDrawdown, 2.0, accuracy: 1e-12)
        XCTAssertEqual(WarmupTools.scoreBucket(stats), 7.458333333333333, accuracy: 1e-12)
    }

    func testWarmupBucketScoreFailsClosedWithoutTrades() {
        XCTAssertEqual(WarmupTools.scoreBucket(WarmupBucketStats()), WarmupTools.missingScore, accuracy: 0.0)
    }

    func testWarmupPortfolioObjectiveProxyMatchesLegacyBlend() {
        let objective = WarmupTools.portfolioObjectiveProxy(
            totalScore: 25.0,
            totalTrades: 40,
            regimeScores: [10.0, 20.0, -5.0, 0.0],
            regimeTrades: [12, 15, 5, 0]
        )

        XCTAssertEqual(objective, -0.15, accuracy: 1e-12)
        XCTAssertEqual(
            WarmupTools.portfolioObjectiveProxy(totalScore: 10.0, totalTrades: 20, regimeScores: [], regimeTrades: []),
            0.0,
            accuracy: 0.0
        )
    }
}
