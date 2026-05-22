import XCTest
@testable import FXDataEngine

final class TradeExecutionPlanTests: XCTestCase {
    func testMarketPlanUsesSingleM1ReferencePriceAndProtectiveLevels() {
        let signal = RiskPolicySignalState(
            tradeGate: 0.72,
            pathRisk: 0.20,
            fillRisk: 0.20,
            minMovePoints: 10.0,
            expectedMovePoints: 30.0,
            hierarchyExecution: 0.70
        )
        let market = TradeExecutionMarketContext(
            referencePrice: 1.10000,
            pointValue: 0.0001,
            digits: 5,
            minDistancePoints: 5.0,
            generatedAtUTC: 1_800_000_000
        )

        let plan = TradeExecutionPlanTools.buildPlan(
            TradeExecutionPlanningInput(
                symbol: "EURUSD",
                direction: 1,
                market: market,
                signal: signal,
                estimatedRiskPoints: 20.0
            )
        )

        XCTAssertTrue(plan.ready)
        XCTAssertFalse(plan.usePending)
        XCTAssertEqual(plan.orderType, .buy)
        XCTAssertEqual(plan.mode, "MARKET")
        XCTAssertEqual(plan.entryPrice, 1.10000, accuracy: 1e-12)
        XCTAssertEqual(plan.stopLoss, 1.09800, accuracy: 1e-12)
        XCTAssertEqual(plan.takeProfit, 1.10405, accuracy: 1e-12)
        XCTAssertEqual(plan.expiryTimeUTC, 0)
        XCTAssertEqual(TradeExecutionPlanTools.orderTypeBucket(plan.orderType), 1)
    }

    func testCautionPostureBuildsPendingLimitWithMQLDistanceAndExpiryMath() {
        let signal = RiskPolicySignalState(
            tradeGate: 0.52,
            pathRisk: 0.62,
            fillRisk: 0.30,
            minMovePoints: 10.0,
            expectedMovePoints: 30.0,
            hierarchyExecution: 0.40
        )
        let market = TradeExecutionMarketContext(
            referencePrice: 1.10000,
            pointValue: 0.0001,
            digits: 5,
            minDistancePoints: 5.0,
            generatedAtUTC: 1_800_000_000
        )
        let systemHealth = SystemHealthState(
            ready: true,
            healthScore: 0.66,
            posture: .caution
        )

        let plan = TradeExecutionPlanTools.buildPlan(
            TradeExecutionPlanningInput(
                symbol: "EURUSD",
                direction: 1,
                market: market,
                signal: signal,
                estimatedRiskPoints: 20.0,
                systemHealth: systemHealth
            )
        )

        XCTAssertTrue(plan.ready)
        XCTAssertTrue(plan.usePending)
        XCTAssertEqual(plan.orderType, .buyLimit)
        XCTAssertEqual(plan.mode, "BUY_LIMIT")
        XCTAssertEqual(plan.entryPrice, 1.09940, accuracy: 1e-12)
        XCTAssertEqual(plan.stopLoss, 1.09740, accuracy: 1e-12)
        XCTAssertEqual(plan.takeProfit, 1.10345, accuracy: 1e-12)
        XCTAssertEqual(plan.expiryTimeUTC, 1_800_001_200)
        XCTAssertEqual(TradeExecutionPlanTools.orderTypeBucket(plan.orderType), 2)
    }

    func testBreakoutPendingSellStopUsesLegacyBreakoutGate() {
        let signal = RiskPolicySignalState(
            tradeGate: 0.74,
            pathRisk: 0.30,
            fillRisk: 0.20,
            minMovePoints: 12.0,
            expectedMovePoints: 18.0,
            hierarchyExecution: 0.58
        )
        let market = TradeExecutionMarketContext(
            referencePrice: 155.250,
            pointValue: 0.001,
            digits: 3,
            minDistancePoints: 6.0,
            generatedAtUTC: 1_800_000_000
        )

        let plan = TradeExecutionPlanTools.buildPlan(
            TradeExecutionPlanningInput(
                symbol: "USDJPY",
                direction: 0,
                market: market,
                signal: signal,
                estimatedRiskPoints: 24.0,
                newsPulseTradeGate: "CAUTION"
            )
        )

        XCTAssertTrue(plan.ready)
        XCTAssertTrue(plan.usePending)
        XCTAssertEqual(plan.orderType, .sellStop)
        XCTAssertEqual(plan.mode, "SELL_STOP")
        XCTAssertEqual(plan.entryPrice, 155.243, accuracy: 1e-12)
        XCTAssertEqual(plan.stopLoss, 155.267, accuracy: 1e-12)
        XCTAssertEqual(plan.takeProfit, 155.211, accuracy: 1e-12)
        XCTAssertEqual(TradeExecutionPlanTools.orderTypeBucket(plan.orderType), 3)
    }

    func testInvalidInputsReturnResetPlanAndOrderBucketsMatchMQL() {
        let invalidDirection = TradeExecutionPlanTools.buildPlan(
            TradeExecutionPlanningInput(
                direction: -1,
                market: TradeExecutionMarketContext(referencePrice: 1.1),
                signal: RiskPolicySignalState()
            )
        )
        XCTAssertFalse(invalidDirection.ready)
        XCTAssertFalse(invalidDirection.usePending)
        XCTAssertEqual(invalidDirection.orderType, .buy)
        XCTAssertEqual(invalidDirection.mode, "MARKET")

        let fallbackMarket = TradeExecutionMarketContext(
            referencePrice: 1.20000,
            pointValue: 0.0,
            digits: -1,
            minDistancePoints: 0.0
        )
        XCTAssertEqual(fallbackMarket.pointValue, 0.0001, accuracy: 1e-12)
        XCTAssertEqual(fallbackMarket.digits, 5)
        XCTAssertEqual(fallbackMarket.minDistancePoints, 2.0, accuracy: 1e-12)

        XCTAssertEqual(TradeExecutionPlanTools.normalizeOrderPrice(1.234567, digits: 5), 1.23457, accuracy: 1e-12)
        XCTAssertEqual(TradeExecutionPlanTools.orderTypeBucket(.buyStopLimit), 4)
        XCTAssertEqual(TradeExecutionPlanTools.orderTypeBucket(.sellStopLimit), 4)
    }
}
