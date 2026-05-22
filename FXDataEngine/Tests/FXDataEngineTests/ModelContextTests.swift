import XCTest
@testable import FXDataEngine

final class ModelContextTests: XCTestCase {
    func testClampAndSymbolScalesMatchLegacyHelpers() {
        XCTAssertEqual(ModelContextTools.coreClampHorizon(0), 1)
        XCTAssertEqual(ModelContextTools.coreClampHorizon(2_000), 1_440)
        XCTAssertEqual(ModelContextTools.coreClampHorizon(60), 60)

        XCTAssertEqual(ModelContextTools.symbolModelScale("xauusd"), 1.18, accuracy: 1e-12)
        XCTAssertEqual(ModelContextTools.symbolModelScale("GOLD.cash"), 1.18, accuracy: 1e-12)
        XCTAssertEqual(ModelContextTools.symbolModelScale("NAS100"), 1.15, accuracy: 1e-12)
        XCTAssertEqual(ModelContextTools.symbolModelScale("WTI"), 1.12, accuracy: 1e-12)
        XCTAssertEqual(ModelContextTools.symbolModelScale("GBPJPY"), 1.06, accuracy: 1e-12)
        XCTAssertEqual(ModelContextTools.symbolModelScale("EURUSD"), 1.0, accuracy: 1e-12)
    }

    func testHorizonAndCapacityScalesMatchLegacyHelpers() {
        XCTAssertEqual(ModelContextTools.horizonModelScale(5), 0.92, accuracy: 1e-12)
        XCTAssertEqual(ModelContextTools.horizonModelScale(15), 0.98, accuracy: 1e-12)
        XCTAssertEqual(ModelContextTools.horizonModelScale(60), 1.00, accuracy: 1e-12)
        XCTAssertEqual(ModelContextTools.horizonModelScale(240), 1.10, accuracy: 1e-12)
        XCTAssertEqual(ModelContextTools.horizonModelScale(241), 1.18, accuracy: 1e-12)

        XCTAssertEqual(
            ModelContextTools.modelCapacityScale(symbol: "XAUUSD", horizonMinutes: 1_440),
            1.35,
            accuracy: 1e-12
        )
        XCTAssertEqual(
            ModelContextTools.modelCapacityScale(symbol: "EURUSD", horizonMinutes: 1),
            0.92,
            accuracy: 1e-12
        )
    }

    func testContextSpanHelpersMatchLegacyRoundingAndClamps() {
        XCTAssertEqual(
            ModelContextTools.contextSequenceSpan(maxCap: 24, horizonMinutes: 13, symbol: "EURUSD", baseMin: 8),
            21
        )
        XCTAssertEqual(
            ModelContextTools.contextBatchSpan(maxCap: 16, horizonMinutes: 300, symbol: "XAUUSD", baseMin: 4),
            16
        )
        XCTAssertEqual(
            ModelContextTools.contextTreeBudget(maxCap: 32, horizonMinutes: 5, symbol: "EURUSD", baseMin: 6),
            29
        )
        XCTAssertEqual(
            ModelContextTools.contextSequenceSpan(maxCap: 4, horizonMinutes: 60, symbol: "EURUSD", baseMin: 8),
            4
        )
    }

    func testMoveEMAUpdateMatchesLegacyReadyAndAlphaRules() {
        var state = MoveEMAState()
        state.update(movePoints: -8.0, alpha: 0.2)
        XCTAssertTrue(state.ready)
        XCTAssertEqual(state.emaAbsMove, 8.0, accuracy: 1e-12)

        state.update(movePoints: 2.0, alpha: 0.9)
        XCTAssertEqual(state.emaAbsMove, 5.0, accuracy: 1e-12)

        let unchanged = ModelContextTools.updatedMoveEMA(
            emaAbsMove: state.emaAbsMove,
            ready: state.ready,
            movePoints: .infinity,
            alpha: 0.2
        )
        XCTAssertEqual(unchanged, state)
    }

    func testThreeWayBranchMatchesLegacySplitBands() {
        XCTAssertEqual(ModelContextTools.threeWayBranch(9.49, split: 10.0), 0)
        XCTAssertEqual(ModelContextTools.threeWayBranch(9.5, split: 10.0), 1)
        XCTAssertEqual(ModelContextTools.threeWayBranch(10.5, split: 10.0), 1)
        XCTAssertEqual(ModelContextTools.threeWayBranch(10.51, split: 10.0), 2)
    }
}
