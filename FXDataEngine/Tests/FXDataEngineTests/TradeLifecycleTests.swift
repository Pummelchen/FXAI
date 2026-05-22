import XCTest
@testable import FXDataEngine

final class TradeLifecycleTests: XCTestCase {
    func testLifecycleProbabilitiesMatchLegacyPreparedMath() {
        let position = TradeLifecyclePositionState(
            hasPosition: true,
            direction: 1,
            openProfit: 10.0,
            profitPoints: 10.0,
            cyclePeakProfit: 20.0,
            givebackFraction: 0.20,
            heldBars: 5
        )

        let probabilities = TradeLifecycleTools.probabilities(
            position: position,
            signalDirection: 1,
            policy: TradeLifecyclePolicyState(),
            minMovePoints: 10.0
        )

        XCTAssertEqual(probabilities.add, 0.196999998, accuracy: 1e-8)
        XCTAssertEqual(probabilities.reduce, 0.024, accuracy: 1e-12)
        XCTAssertEqual(probabilities.timeout, 0.11533333333333334, accuracy: 1e-12)
        XCTAssertEqual(probabilities.tighten, 0.141, accuracy: 1e-12)
        XCTAssertEqual(probabilities.exit, 0.036, accuracy: 1e-12)
    }

    func testLifecycleRecommendationUsesLegacyActionPriority() {
        let idle = TradeLifecycleTools.recommendation(
            position: TradeLifecyclePositionState(),
            signalDirection: 1,
            policy: TradeLifecyclePolicyState(addProb: 1.0),
            minMovePoints: 10.0
        )
        XCTAssertEqual(idle.action, .noTrade)
        XCTAssertEqual(idle.reason, "lifecycle_idle")

        let position = TradeLifecyclePositionState(hasPosition: true, direction: 1, profitPoints: 1.0, heldBars: 20)
        let timeout = TradeLifecycleTools.recommendation(
            position: position,
            signalDirection: 1,
            policy: TradeLifecyclePolicyState(timeoutProb: 0.80),
            minMovePoints: 10.0
        )
        XCTAssertEqual(timeout.action, .timeout)
        XCTAssertEqual(timeout.reason, "lifecycle_timeout_exit")

        let exit = TradeLifecycleTools.recommendation(
            position: TradeLifecyclePositionState(hasPosition: true, direction: 1),
            signalDirection: 1,
            policy: TradeLifecyclePolicyState(exitProb: 0.60),
            minMovePoints: 10.0
        )
        XCTAssertEqual(exit.action, .exit)
        XCTAssertEqual(exit.reason, "lifecycle_exit")

        let reduce = TradeLifecycleTools.recommendation(
            position: TradeLifecyclePositionState(hasPosition: true, direction: 1),
            signalDirection: 1,
            policy: TradeLifecyclePolicyState(reduceProb: 0.60),
            minMovePoints: 10.0
        )
        XCTAssertEqual(reduce.action, .reduce)
        XCTAssertEqual(reduce.reason, "lifecycle_reduce")
        XCTAssertEqual(reduce.reduceFraction, 0.308, accuracy: 1e-12)
    }

    func testLifecycleRecommendationHandlesTightenAddHoldAndNoTrade() {
        let basePosition = TradeLifecyclePositionState(hasPosition: true, direction: 1)

        let tighten = TradeLifecycleTools.recommendation(
            position: basePosition,
            signalDirection: 1,
            policy: TradeLifecyclePolicyState(tightenProb: 0.53),
            minMovePoints: 10.0
        )
        XCTAssertEqual(tighten.action, .tighten)
        XCTAssertEqual(tighten.reason, "lifecycle_tighten")

        let add = TradeLifecycleTools.recommendation(
            position: basePosition,
            signalDirection: 1,
            policy: TradeLifecyclePolicyState(addProb: 0.70),
            minMovePoints: 10.0
        )
        XCTAssertEqual(add.action, .add)
        XCTAssertEqual(add.reason, "lifecycle_add")
        XCTAssertEqual(add.addSizeMultiplier, 0.50, accuracy: 1e-12)

        let hold = TradeLifecycleTools.recommendation(
            position: basePosition,
            signalDirection: 0,
            policy: TradeLifecyclePolicyState(holdQuality: 0.50),
            minMovePoints: 10.0
        )
        XCTAssertEqual(hold.action, .hold)
        XCTAssertEqual(hold.reason, "lifecycle_hold")

        let noTrade = TradeLifecycleTools.recommendation(
            position: basePosition,
            signalDirection: 0,
            policy: TradeLifecyclePolicyState(holdQuality: 0.40),
            minMovePoints: 10.0
        )
        XCTAssertEqual(noTrade.action, .noTrade)
        XCTAssertEqual(noTrade.reason, "lifecycle_hold")
    }
}
