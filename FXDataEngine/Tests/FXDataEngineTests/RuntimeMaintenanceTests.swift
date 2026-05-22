import XCTest
@testable import FXDataEngine

final class RuntimeMaintenanceTests: XCTestCase {
    func testInvariantPlanResetsActiveCycleWhenExposureDisappears() {
        let plan = RuntimeMaintenanceTools.checkRuntimeInvariants(
            RuntimeInvariantInput(
                symbolExposureCount: 0,
                cycleActive: true,
                lastOrderRequestPending: true
            )
        )

        XCTAssertEqual(plan.reason, "cycle_reset_no_exposure")
        XCTAssertEqual(plan.actions, [.resetCycleState, .clearLastOrderRequest])
    }

    func testInvariantPlanClearsPendingRequestWithoutExposure() {
        let plan = RuntimeMaintenanceTools.checkRuntimeInvariants(
            RuntimeInvariantInput(
                symbolExposureCount: 0,
                cycleActive: false,
                lastOrderRequestPending: true
            )
        )

        XCTAssertEqual(plan.reason, "cleared_stale_order_request")
        XCTAssertEqual(plan.actions, [.clearLastOrderRequest])
    }

    func testInvariantPlanRecoversCycleAndPreservesReasonPriority() {
        let recovered = RuntimeMaintenanceTools.checkRuntimeInvariants(
            RuntimeInvariantInput(
                symbolExposureCount: 2,
                cycleActive: false,
                lastOrderRequestPending: true,
                lastOrderRequestUsesPendingOrder: true,
                lastOrderRequestSymbol: "EURUSD",
                lastOrderRequestSymbolOrderCount: 0,
                lastOrderRequestTimeUTC: 1_000,
                nowUTC: 10_000
            )
        )

        XCTAssertEqual(recovered.reason, "cycle_recovered")
        XCTAssertEqual(recovered.actions, [.recoverCycleState, .clearLastOrderRequest])

        let completed = RuntimeMaintenanceTools.checkRuntimeInvariants(
            RuntimeInvariantInput(
                symbolExposureCount: 1,
                cycleActive: true,
                lastOrderRequestPending: true,
                lastOrderRequestUsesPendingOrder: true,
                lastOrderRequestSymbol: "EURUSD",
                lastOrderRequestSymbolOrderCount: 0
            )
        )

        XCTAssertEqual(completed.reason, "cleared_completed_pending_request")
        XCTAssertEqual(completed.actions, [.clearLastOrderRequest])
    }

    func testInvariantPlanExpiresOldPendingRequestAfterExposureChecks() {
        let plan = RuntimeMaintenanceTools.checkRuntimeInvariants(
            RuntimeInvariantInput(
                symbolExposureCount: 1,
                cycleActive: false,
                lastOrderRequestPending: true,
                lastOrderRequestUsesPendingOrder: false,
                lastOrderRequestSymbol: "EURUSD",
                lastOrderRequestSymbolOrderCount: 1,
                lastOrderRequestTimeUTC: 1_000,
                nowUTC: 8_300
            )
        )

        XCTAssertEqual(plan.reason, "expired_order_request_state")
        XCTAssertEqual(plan.actions, [.recoverCycleState, .clearLastOrderRequest])

        let notExpired = RuntimeMaintenanceTools.checkRuntimeInvariants(
            RuntimeInvariantInput(
                symbolExposureCount: 1,
                cycleActive: true,
                lastOrderRequestPending: true,
                lastOrderRequestTimeUTC: 1_000,
                nowUTC: 8_200
            )
        )

        XCTAssertEqual(notExpired.reason, "ok")
        XCTAssertTrue(notExpired.actions.isEmpty)
    }

    func testRuntimeMaintenanceRunPlanCarriesHealthReadinessAndDebugMessage() {
        let timeContext = RuntimeStageTools.buildTimeContext(serverNow: 1_704_067_200)
        let health = SystemHealthState(ready: true, generatedAt: 1_704_067_200, healthScore: 1.0, posture: .healthy)
        let plan = RuntimeMaintenanceTools.buildRunPlan(
            symbol: "EURUSD",
            timeContext: timeContext,
            systemHealth: health,
            invariantInput: RuntimeInvariantInput(
                symbolExposureCount: 0,
                cycleActive: true
            ),
            emitDebug: true
        )

        XCTAssertTrue(plan.ready)
        XCTAssertEqual(plan.symbol, "EURUSD")
        XCTAssertEqual(plan.timeContext.serverNow, 1_704_067_200)
        XCTAssertEqual(plan.invariantPlan.reason, "cycle_reset_no_exposure")
        XCTAssertEqual(plan.debugMessage, "runtime maintenance action: reason=cycle_reset_no_exposure")

        let quiet = RuntimeMaintenanceTools.buildRunPlan(
            symbol: "EURUSD",
            timeContext: timeContext,
            systemHealth: health,
            invariantInput: RuntimeInvariantInput(symbolExposureCount: 1, cycleActive: true),
            emitDebug: true
        )
        XCTAssertEqual(quiet.invariantPlan.reason, "ok")
        XCTAssertEqual(quiet.debugMessage, "")
    }
}
