import XCTest
@testable import FXDataEngine

final class HorizonToolsTests: XCTestCase {
    func testSafeArrayAndPositiveIntMeanMatchLegacyHelpers() {
        XCTAssertEqual(HorizonTools.value(in: [1.0, 2.0], index: 1, default: -1.0), 2.0)
        XCTAssertEqual(HorizonTools.value(in: [1.0, 2.0], index: -1, default: -1.0), -1.0)
        XCTAssertEqual(HorizonTools.value(in: [1.0, 2.0], index: 2, default: -1.0), -1.0)

        XCTAssertEqual(
            HorizonTools.positiveIntMean([0, -1, 10, 14, 0], startIndex: 0, width: 4, fallback: 0.05),
            12.0,
            accuracy: 1e-12
        )
        XCTAssertEqual(
            HorizonTools.positiveIntMean([0, -1], startIndex: 0, width: 2, fallback: 0.05),
            0.10,
            accuracy: 1e-12
        )
        XCTAssertEqual(
            HorizonTools.positiveIntMean([], startIndex: 0, width: 2, fallback: 0.25),
            0.25,
            accuracy: 1e-12
        )
    }

    func testParseHorizonListMatchesLegacyNormalizationAndBaseAppend() {
        XCTAssertEqual(
            HorizonTools.parseHorizonList("{13; 5|bad,720,999,5}", baseHorizon: 34),
            [1, 5, 13, 34, 720]
        )
        XCTAssertEqual(
            HorizonTools.parseHorizonList("1,2,3,5,8,13,21,55", baseHorizon: 34),
            [1, 2, 3, 5, 8, 13, 21, 34, 55]
        )
        XCTAssertEqual(HorizonTools.parseHorizonList("", baseHorizon: 0), [1])
    }

    func testHorizonSlotAndMaximumUseClampedConfiguredHorizons() {
        XCTAssertEqual(HorizonTools.clampHorizon(0), 1)
        XCTAssertEqual(HorizonTools.clampHorizon(999), 720)
        XCTAssertEqual(
            HorizonTools.maxConfiguredHorizon(configuredHorizons: [0, 55, 999], fallbackHorizon: 13),
            720
        )
        XCTAssertEqual(
            HorizonTools.horizonSlot(horizonMinutes: 8, configuredHorizons: [1, 5, 13, 34]),
            1
        )
        XCTAssertEqual(
            HorizonTools.horizonSlot(horizonMinutes: 11, configuredHorizons: [1, 5, 13, 34]),
            2
        )
        XCTAssertEqual(
            TrainingSampleTools.horizonSlot(horizonMinutes: 11, configuredHorizons: [1, 5, 13, 34]),
            2
        )
    }

    func testNoSpreadStaticRegimeUsesSessionAndVolatilityOnly() {
        let earlyUTC = Int64(1_704_067_200)
        let londonUTC = earlyUTC + Int64(9 * 3_600)
        let newYorkUTC = earlyUTC + Int64(17 * 3_600)

        XCTAssertEqual(HorizonTools.sessionGroup(timestampUTC: earlyUTC), 0)
        XCTAssertEqual(HorizonTools.sessionGroup(timestampUTC: londonUTC), 1)
        XCTAssertEqual(HorizonTools.sessionGroup(timestampUTC: newYorkUTC), 2)

        XCTAssertEqual(
            HorizonTools.noSpreadStaticRegimeID(
                timestampUTC: londonUTC,
                volatilityProxyAbs: 2.0,
                volatilityRef: 1.0
            ),
            6
        )
        XCTAssertEqual(
            HorizonTools.noSpreadStaticRegimeID(
                timestampUTC: londonUTC,
                volatilityProxyAbs: 1.0,
                volatilityRef: 1.0
            ),
            4
        )
    }
}
