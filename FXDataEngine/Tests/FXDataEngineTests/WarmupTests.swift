import XCTest
@testable import FXDataEngine

final class WarmupTests: XCTestCase {
    func testWarmupTransferUniverseAndRangeMatchLegacyOrdering() {
        XCTAssertEqual(
            WarmupTools.transferUniverse(mainSymbol: "EURUSD", contextSymbols: ["USDJPY", "EURUSD", "XAUUSD", ""]),
            ["EURUSD", "USDJPY", "XAUUSD", "GBPUSD", "AUDUSD", "USDCAD", "USDCHF", "NZDUSD", "EURJPY", "EURGBP", "EURAUD"]
        )
        XCTAssertEqual(
            WarmupTools.transferSampleRange(neededBars: 2_011, maxHorizonMinutes: 34, sampleCap: 100),
            34...133
        )
        XCTAssertNil(WarmupTools.transferSampleRange(neededBars: 100, maxHorizonMinutes: 90, sampleCap: 20))
    }

    func testWarmupCappedValidSampleIndexesUseLegacyReverseStride() {
        XCTAssertEqual(
            WarmupTools.cappedValidSampleIndexes(validFlags: Array(repeating: true, count: 10), sampleCap: 3),
            [9, 6, 3]
        )
        XCTAssertEqual(
            WarmupTools.cappedValidSampleIndexes(validFlags: [true, false, true], sampleCap: 10),
            [2, 0]
        )
        XCTAssertTrue(WarmupTools.cappedValidSampleIndexes(validFlags: [false, false], sampleCap: 2).isEmpty)
    }

    func testWarmupNormalizationModelListAndSplitsMatchLegacyPlanning() {
        XCTAssertEqual(WarmupTools.normalizationScoringModelIDs(primaryAI: 20, ensemble: true), [20, 4, 6])
        XCTAssertEqual(WarmupTools.normalizationScoringModelIDs(primaryAI: -4, ensemble: false), [14])

        XCTAssertEqual(
            WarmupTools.normalizationCandidateSplit(horizonMinutes: 13, startIndex: 0, endIndex: 599),
            WarmupCandidateSplit(validationStart: 400, validationEnd: 599, trainingStart: 0, trainingEnd: 146)
        )
        XCTAssertNil(WarmupTools.normalizationCandidateSplit(horizonMinutes: 13, startIndex: 0, endIndex: 239))

        XCTAssertEqual(
            WarmupTools.warmupFoldSplits(horizonMinutes: 13, startIndex: 0, endIndex: 999, folds: 3),
            [
                WarmupCandidateSplit(validationStart: 500, validationEnd: 749, trainingStart: 0, trainingEnd: 246),
                WarmupCandidateSplit(validationStart: 750, validationEnd: 999, trainingStart: 0, trainingEnd: 496)
            ]
        )
    }

    func testWarmupAdaptiveThresholdsAndEVSignalMatchLegacyRules() {
        let thresholds = WarmupTools.deriveAdaptiveThresholds(
            baseBuyThreshold: 0.60,
            baseSellThreshold: 0.40,
            minMovePoints: 2.0,
            expectedMovePoints: 4.0,
            volatilityProxy: 2.0
        )

        XCTAssertEqual(thresholds.buyMinProbability, 0.7025, accuracy: 1e-12)
        XCTAssertEqual(thresholds.sellMinProbability, 0.7025, accuracy: 1e-12)
        XCTAssertEqual(thresholds.skipMinProbability, 0.60, accuracy: 1e-12)
        XCTAssertEqual(
            WarmupTools.classSignalFromEV(
                probabilities: [0.10, 0.75, 0.15],
                thresholds: thresholds,
                expectedMovePoints: 6.0,
                minMovePoints: 2.0,
                evThresholdPoints: 0.30
            ),
            .buy
        )
        XCTAssertEqual(
            WarmupTools.classSignalFromEV(
                probabilities: [0.80, 0.10, 0.10],
                thresholds: thresholds,
                expectedMovePoints: 6.0,
                minMovePoints: 2.0,
                evThresholdPoints: 0.30
            ),
            .sell
        )
        XCTAssertNil(
            WarmupTools.classSignalFromEV(
                probabilities: [0.10, 0.80, 0.90],
                thresholds: thresholds,
                expectedMovePoints: 6.0,
                minMovePoints: 2.0,
                evThresholdPoints: 0.30
            )
        )

        let clamped = WarmupTools.deriveAdaptiveThresholds(
            baseBuyThreshold: 0.95,
            baseSellThreshold: 0.05,
            minMovePoints: 5.0,
            expectedMovePoints: 0.0,
            volatilityProxy: 9.0
        )
        XCTAssertEqual(clamped.buyMinProbability, 0.96, accuracy: 1e-12)
        XCTAssertEqual(clamped.sellMinProbability, 0.96, accuracy: 1e-12)
    }

    func testWarmupEpochBudgetAndPortfolioDiagnosticsMatchLegacyMath() {
        let expectedSeriousNativeIDs: Set<Int> = [
            AIModelID.mlpTiny.rawValue,
            AIModelID.lstm.rawValue,
            AIModelID.lstmg.rawValue,
            AIModelID.tcn.rawValue,
            AIModelID.tft.rawValue,
            AIModelID.tst.rawValue,
            AIModelID.s4.rawValue,
            AIModelID.autoformer.rawValue,
            AIModelID.patchTST.rawValue,
            AIModelID.chronos.rawValue,
            AIModelID.timesfm.rawValue,
            AIModelID.qcew.rawValue,
            AIModelID.fewc.rawValue,
            AIModelID.gha.rawValue,
            AIModelID.tesseract.rawValue,
            AIModelID.graphWM.rawValue,
            AIModelID.lightgbm.rawValue,
            AIModelID.xgbFast.rawValue
        ]
        XCTAssertEqual(WarmupTools.seriousNativeAIIDs, expectedSeriousNativeIDs)
        XCTAssertFalse(WarmupTools.isSeriousNativeAI(aiID: AIModelID.sgdLogit.rawValue))
        XCTAssertFalse(WarmupTools.isSeriousNativeAI(aiID: AIModelID.xgboost.rawValue))

        XCTAssertEqual(
            WarmupTools.warmupEpochBudget(aiID: 14, horizonMinutes: 60, baseEpochs: 0, symbol: "EURUSD"),
            1
        )
        XCTAssertEqual(
            WarmupTools.warmupEpochBudget(aiID: AIModelID.lstm.rawValue, horizonMinutes: 60, baseEpochs: 2, symbol: "EURUSD"),
            6
        )
        XCTAssertEqual(
            WarmupTools.warmupEpochBudget(aiID: 20, horizonMinutes: 60, baseEpochs: 2, symbol: "EURUSD"),
            6
        )
        XCTAssertEqual(
            WarmupTools.warmupEpochBudget(aiID: 20, horizonMinutes: 240, baseEpochs: 5, symbol: "XAUUSD"),
            10
        )
        XCTAssertEqual(WarmupTools.warmupBlockSpan(aiID: AIModelID.sgdLogit.rawValue, horizonMinutes: 60, symbol: "EURUSD"), 1)
        XCTAssertEqual(WarmupTools.warmupBlockSpan(aiID: AIModelID.lstm.rawValue, horizonMinutes: 60, symbol: "EURUSD"), 34)
        XCTAssertEqual(WarmupTools.warmupBlockSpan(aiID: AIModelID.xgbFast.rawValue, horizonMinutes: 240, symbol: "XAUUSD"), 36)

        let blockPlan = WarmupTools.makeBlockBatchPlan(
            startIndex: 10,
            currentEnd: 37,
            blockSpan: 12,
            replayEnabled: true,
            learningRateScale: 7.0
        )
        XCTAssertEqual(blockPlan.start, 26)
        XCTAssertEqual(blockPlan.end, 37)
        XCTAssertEqual(blockPlan.replayBudget, 2)
        XCTAssertEqual(blockPlan.learningRateScale, 4.0, accuracy: 0.0)

        let singlePlan = WarmupTools.makeBlockBatchPlan(
            startIndex: 10,
            currentEnd: 12,
            blockSpan: 0,
            replayEnabled: false,
            learningRateScale: 0.01
        )
        XCTAssertEqual(singlePlan.start, 12)
        XCTAssertEqual(singlePlan.end, 12)
        XCTAssertEqual(singlePlan.replayBudget, 0)
        XCTAssertEqual(singlePlan.learningRateScale, 0.10, accuracy: 0.0)

        let prioritySample = PreparedTrainingSample(
            valid: true,
            movePoints: 10.0,
            costPoints: 2.0,
            sampleWeight: 2.0,
            mfePoints: 12.0,
            maePoints: 3.0,
            timeToHitFraction: 0.25,
            fillRisk: 0.5
        )
        XCTAssertEqual(WarmupTools.curriculumPriority(sample: prioritySample), 2.939142857142857, accuracy: 1e-12)

        var sampleA = PreparedTrainingSample(valid: true)
        var xA = sampleA.x
        xA[53] = -0.25
        sampleA.x = xA
        var sampleB = PreparedTrainingSample(valid: true)
        var xB = sampleB.x
        xB[53] = 0.75
        sampleB.x = xB
        XCTAssertEqual(WarmupTools.estimatePortfolioSymbolCorrelation(samples: [sampleA, sampleB]), 0.50, accuracy: 1e-12)

        let diversity = WarmupTools.transferDiversificationWeight(absoluteCorrelation: 0.40)
        XCTAssertEqual(WarmupTools.primaryPortfolioWeight(tradeRate: 0.50), 0.85, accuracy: 1e-12)
        XCTAssertEqual(diversity, 0.76, accuracy: 1e-12)
        XCTAssertEqual(WarmupTools.transferPortfolioWeight(tradeRate: 0.50, diversificationWeight: diversity), 0.589, accuracy: 1e-12)

        let diagnostics = WarmupTools.portfolioDiagnostics(
            contributions: [
                WarmupPortfolioContribution(edge: 2.0, weight: 1.0),
                WarmupPortfolioContribution(edge: -1.0, weight: 0.5, absoluteCorrelation: 0.4, diversificationWeight: 0.76)
            ]
        )
        XCTAssertEqual(diagnostics?.meanEdge ?? 0.0, 1.0, accuracy: 1e-12)
        XCTAssertEqual(diagnostics?.stability ?? -1.0, 0.0, accuracy: 1e-12)
        XCTAssertEqual(diagnostics?.correlationPenalty ?? 0.0, 0.13333333333333333, accuracy: 1e-12)
        XCTAssertEqual(diagnostics?.diversification ?? 0.0, 0.92, accuracy: 1e-12)
        XCTAssertEqual(diagnostics?.symbolCount, 2)
        XCTAssertNil(WarmupTools.portfolioDiagnostics(contributions: []))
    }

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
