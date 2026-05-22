import XCTest
@testable import FXDataEngine

final class AuditUtilitiesTests: XCTestCase {
    func testFixedHorizonSlotUsesAuditBuckets() {
        XCTAssertEqual(AuditUtilityTools.clampHorizon(0), 1)
        XCTAssertEqual(AuditUtilityTools.clampHorizon(999), 720)
        XCTAssertEqual(AuditUtilityTools.fixedHorizonSlot(horizonMinutes: 1), 0)
        XCTAssertEqual(AuditUtilityTools.fixedHorizonSlot(horizonMinutes: 3), 0)
        XCTAssertEqual(AuditUtilityTools.fixedHorizonSlot(horizonMinutes: 4), 1)
        XCTAssertEqual(AuditUtilityTools.fixedHorizonSlot(horizonMinutes: 5), 1)
        XCTAssertEqual(AuditUtilityTools.fixedHorizonSlot(horizonMinutes: 6), 2)
        XCTAssertEqual(AuditUtilityTools.fixedHorizonSlot(horizonMinutes: 8), 2)
        XCTAssertEqual(AuditUtilityTools.fixedHorizonSlot(horizonMinutes: 9), 3)
        XCTAssertEqual(AuditUtilityTools.fixedHorizonSlot(horizonMinutes: 14), 4)
        XCTAssertEqual(AuditUtilityTools.fixedHorizonSlot(horizonMinutes: 22), 5)
        XCTAssertEqual(AuditUtilityTools.fixedHorizonSlot(horizonMinutes: 35), 6)
        XCTAssertEqual(AuditUtilityTools.fixedHorizonSlot(horizonMinutes: 56), 7)
        XCTAssertEqual(AuditUtilityTools.fixedHorizonSlot(horizonMinutes: 999), 7)
    }

    func testValueAndPositiveMeanFollowAuditFallbackRules() {
        XCTAssertEqual(AuditUtilityTools.value([1.5, 2.5], index: 1, default: -1.0), 2.5, accuracy: 0.0)
        XCTAssertEqual(AuditUtilityTools.value([1.5, 2.5], index: 2, default: -1.0), -1.0, accuracy: 0.0)
        XCTAssertEqual(
            AuditUtilityTools.positiveIntMean([0, 4, -2, 8], startIndex: 0, width: 4, fallback: 0.01),
            6.0,
            accuracy: 0.0
        )
        XCTAssertEqual(
            AuditUtilityTools.positiveIntMean([0, -2], startIndex: 0, width: 2, fallback: 0.01),
            0.10,
            accuracy: 0.0
        )
    }

    func testNoSpreadStaticRegimeUsesSessionLiquidityAndVolatility() {
        let sessionOneUTC: Int64 = 1_704_099_600
        XCTAssertEqual(
            AuditUtilityTools.noSpreadStaticRegimeID(
                timestampUTC: sessionOneUTC,
                liquidityStress: 0.30,
                liquidityStressReference: 0.10,
                volatilityProxyAbs: 0.20,
                volatilityReference: 0.05
            ),
            7
        )
        XCTAssertEqual(
            AuditUtilityTools.noSpreadStaticRegimeID(
                timestampUTC: sessionOneUTC,
                liquidityStress: 0.0,
                liquidityStressReference: 0.10,
                volatilityProxyAbs: 0.01,
                volatilityReference: 0.05
            ),
            4
        )
    }

    func testSanitizeNormalizationAndAuditDefaultHyperParameters() {
        XCTAssertEqual(AuditUtilityTools.sanitizeNormalizationMethod(-1), .existing)
        XCTAssertEqual(AuditUtilityTools.sanitizeNormalizationMethod(FeatureNormalizationMethod.dain.rawValue), .dain)

        let lightGBM = AuditUtilityTools.defaultHyperParameters(aiID: AIModelID.lightgbm.rawValue)
        XCTAssertEqual(lightGBM.xgbLearningRate, 0.0800, accuracy: 0.0)
        XCTAssertEqual(lightGBM.xgbL2, 0.0200, accuracy: 0.0)
        XCTAssertEqual(lightGBM.xgbSplit, 0.5000, accuracy: 0.0)

        let lstm = AuditUtilityTools.defaultHyperParameters(aiID: AIModelID.lstm.rawValue)
        XCTAssertEqual(lstm.learningRate, 0.0060, accuracy: 0.0)
        XCTAssertEqual(lstm.l2, 0.0020, accuracy: 0.0)

        let baseline = AuditUtilityTools.defaultHyperParameters(aiID: AIModelID.m1Sync.rawValue)
        XCTAssertEqual(baseline.learningRate, 0.0, accuracy: 0.0)
        XCTAssertEqual(baseline.l2, 0.0, accuracy: 0.0)

        let ftrl = AuditUtilityTools.defaultHyperParameters(aiID: AIModelID.ftrlLogit.rawValue)
        XCTAssertEqual(ftrl.ftrlL1, 0.0, accuracy: 0.0)
        XCTAssertEqual(ftrl.ftrlL2, 0.0100, accuracy: 0.0)
    }
}
