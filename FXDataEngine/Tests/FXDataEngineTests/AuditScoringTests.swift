import XCTest
@testable import FXDataEngine

final class AuditScoringTests: XCTestCase {
    func testAuditFoldRecordingAndScoringMatchLegacyFormula() {
        var recorded = AuditFoldMetrics()
        recorded.recordInvalidPrediction()
        recorded.recordValidPrediction(
            decision: .buy,
            prediction: PredictionV4(
                classProbabilities: [0.10, 0.80, 0.10],
                moveMeanPoints: 8.0,
                confidence: 0.72,
                reliability: 0.63
            ),
            brier: 0.18,
            netPoints: 2.5,
            directionalEvaluated: true,
            directionalCorrect: true,
            calibrationAbs: 0.20,
            pathQuality: 0.25
        )
        recorded.recordValidPrediction(
            decision: .skip,
            prediction: PredictionV4(
                classProbabilities: [0.12, 0.16, 0.72],
                moveMeanPoints: 2.0,
                confidence: 0.55,
                reliability: 0.58
            ),
            brier: 0.30,
            netPoints: 0.0,
            directionalEvaluated: false,
            directionalCorrect: false,
            calibrationAbs: 0.0,
            pathQuality: -1.0
        )

        XCTAssertEqual(recorded.samplesTotal, 3)
        XCTAssertEqual(recorded.validPredictions, 2)
        XCTAssertEqual(recorded.invalidPredictions, 1)
        XCTAssertEqual(recorded.buyCount, 1)
        XCTAssertEqual(recorded.skipCount, 1)
        XCTAssertEqual(recorded.directionalEvaluationCount, 1)
        XCTAssertEqual(recorded.directionalCorrectCount, 1)
        XCTAssertEqual(recorded.pathQualityCount, 1)

        let fold = AuditFoldMetrics(
            samplesTotal: 20,
            validPredictions: 17,
            invalidPredictions: 3,
            buyCount: 10,
            sellCount: 4,
            skipCount: 3,
            directionalEvaluationCount: 14,
            directionalCorrectCount: 9,
            confidenceSum: 11.9,
            reliabilitySum: 10.2,
            moveSum: 72.0,
            brierSum: 4.25,
            calibrationAbsSum: 2.8,
            pathQualityAbsSum: 3.6,
            pathQualityCount: 12,
            netSum: 34.0
        )
        XCTAssertEqual(AuditScoringTools.scoreFold(fold), 88.54621848739497, accuracy: 1e-12)
        XCTAssertEqual(AuditScoringTools.scoreFold(AuditFoldMetrics(samplesTotal: 11, validPredictions: 11)), -1e9, accuracy: 0.0)
    }

    func testAuditPredictionComparisonAndDistributionStatsMatchLegacyFormula() {
        let first = PredictionV4(
            classProbabilities: [0.20, 0.70, 0.10],
            moveMeanPoints: 8.0,
            mfeMeanPoints: 11.0,
            maeMeanPoints: 3.0,
            hitTimeFraction: 0.40,
            pathRisk: 0.20,
            fillRisk: 0.30,
            confidence: 0.75,
            reliability: 0.60
        )
        let second = PredictionV4(
            classProbabilities: [0.30, 0.40, 0.30],
            moveMeanPoints: 5.0,
            mfeMeanPoints: 8.0,
            maeMeanPoints: 4.0,
            hitTimeFraction: 0.80,
            pathRisk: 0.60,
            fillRisk: 0.10,
            confidence: 0.40,
            reliability: 0.90
        )

        XCTAssertEqual(AuditScoringTools.comparePredictions(first, second), 1.1425, accuracy: 1e-12)
        XCTAssertEqual(AuditScoringTools.mean([72.0, 69.0, 81.0, 64.0]), 71.5, accuracy: 1e-12)
        XCTAssertEqual(
            AuditScoringTools.sampleStandardDeviation([72.0, 69.0, 81.0, 64.0], mean: 71.5),
            7.14142842854285,
            accuracy: 1e-12
        )
        XCTAssertEqual(AuditScoringTools.approximateNormalCDF(0.0), 0.5, accuracy: 1e-12)
        XCTAssertEqual(
            AuditScoringTools.deflatedSharpeProxy(scores: [72.0, 69.0, 81.0, 64.0], pbo: 0.25),
            0.668816670995866,
            accuracy: 1e-12
        )
        XCTAssertEqual(AuditScoringTools.deflatedSharpeProxy(scores: [70.0, 70.0], pbo: 0.0), 1.0, accuracy: 0.0)
    }

    func testAuditWalkForwardAggregationMatchesLegacyRules() {
        let trainFolds = [
            fold(20, 17, 3, 10, 4, 3, 14, 9, 11.9, 10.2, 72.0, 4.25, 2.8, 3.6, 12, 34.0),
            fold(24, 22, 2, 8, 6, 8, 14, 11, 16.0, 15.0, 90.0, 3.6, 1.8, 2.2, 12, 55.0),
            fold(18, 16, 2, 5, 4, 9, 9, 5, 10.0, 9.0, 44.0, 5.0, 2.1, 2.7, 10, 12.0)
        ]
        let testFolds = [
            fold(20, 18, 2, 8, 3, 9, 11, 7, 12.0, 11.0, 65.0, 4.2, 2.4, 2.4, 10, 28.0),
            fold(24, 22, 2, 7, 5, 12, 12, 9, 15.0, 14.0, 76.0, 4.0, 2.0, 2.4, 12, 40.0),
            fold(18, 16, 2, 4, 3, 11, 7, 4, 9.0, 8.0, 35.0, 5.8, 2.4, 3.0, 10, 5.0)
        ]

        let metrics = AuditScoringTools.finalizeWalkForward(trainFolds: trainFolds, testFolds: testFolds)
        XCTAssertEqual(metrics.walkForwardFolds, 3)
        XCTAssertEqual(metrics.walkForwardTrainSamples, 62)
        XCTAssertEqual(metrics.walkForwardTestSamples, 62)
        XCTAssertEqual(metrics.walkForwardTrainScore, 90.8090172735761, accuracy: 1e-12)
        XCTAssertEqual(metrics.walkForwardTestScore, 89.11260251322751, accuracy: 1e-12)
        XCTAssertEqual(metrics.walkForwardTestScoreStd, 9.035438539220445, accuracy: 1e-12)
        XCTAssertEqual(metrics.walkForwardGap, 1.6964147603485884, accuracy: 1e-12)
        XCTAssertEqual(metrics.walkForwardPBO, 0.0, accuracy: 0.0)
        XCTAssertEqual(metrics.walkForwardPassRate, 1.0, accuracy: 0.0)
        XCTAssertEqual(metrics.walkForwardDSR, 0.9154201124367981, accuracy: 1e-12)
    }

    private func fold(
        _ samplesTotal: Int,
        _ validPredictions: Int,
        _ invalidPredictions: Int,
        _ buyCount: Int,
        _ sellCount: Int,
        _ skipCount: Int,
        _ directionalEvaluationCount: Int,
        _ directionalCorrectCount: Int,
        _ confidenceSum: Double,
        _ reliabilitySum: Double,
        _ moveSum: Double,
        _ brierSum: Double,
        _ calibrationAbsSum: Double,
        _ pathQualityAbsSum: Double,
        _ pathQualityCount: Int,
        _ netSum: Double
    ) -> AuditFoldMetrics {
        AuditFoldMetrics(
            samplesTotal: samplesTotal,
            validPredictions: validPredictions,
            invalidPredictions: invalidPredictions,
            buyCount: buyCount,
            sellCount: sellCount,
            skipCount: skipCount,
            directionalEvaluationCount: directionalEvaluationCount,
            directionalCorrectCount: directionalCorrectCount,
            confidenceSum: confidenceSum,
            reliabilitySum: reliabilitySum,
            moveSum: moveSum,
            brierSum: brierSum,
            calibrationAbsSum: calibrationAbsSum,
            pathQualityAbsSum: pathQualityAbsSum,
            pathQualityCount: pathQualityCount,
            netSum: netSum
        )
    }
}
