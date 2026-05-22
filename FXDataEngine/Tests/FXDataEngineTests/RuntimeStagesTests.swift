import XCTest
@testable import FXDataEngine

final class RuntimeStagesTests: XCTestCase {
    func testRuntimeTimeContextDefaultsMatchLegacyReset() {
        let context = RuntimeTimeContext.reset

        XCTAssertFalse(context.ready)
        XCTAssertEqual(context.serverNow, 0)
        XCTAssertEqual(context.utcNow, 0)
        XCTAssertEqual(context.localNow, 0)
        XCTAssertEqual(context.serverUTCOffsetSeconds, 0)
        XCTAssertEqual(context.localUTCOffsetSeconds, 0)
        XCTAssertEqual(context.sessionBucket, 0)
        XCTAssertEqual(context.serverDayOfWeek, -1)
        XCTAssertEqual(context.utcDayOfWeek, -1)
        XCTAssertEqual(context.localDayOfWeek, -1)
        XCTAssertEqual(context.summary, "time_context_unavailable")
    }

    func testRuntimeTimeContextBuildsOffsetsDaysAndConversions() {
        let utcMidnightMonday = Int64(1_704_067_200)
        let serverNow = utcMidnightMonday + 7_200
        let localNow = utcMidnightMonday + 25_200
        let context = RuntimeStageTools.buildTimeContext(
            serverNow: serverNow,
            utcNow: utcMidnightMonday,
            localNow: localNow
        )

        XCTAssertTrue(context.ready)
        XCTAssertEqual(context.serverNow, serverNow)
        XCTAssertEqual(context.utcNow, utcMidnightMonday)
        XCTAssertEqual(context.localNow, localNow)
        XCTAssertEqual(context.serverUTCOffsetSeconds, 7_200)
        XCTAssertEqual(context.localUTCOffsetSeconds, 25_200)
        XCTAssertEqual(context.sessionBucket, 0)
        XCTAssertEqual(context.serverDayOfWeek, 1)
        XCTAssertEqual(context.utcDayOfWeek, 1)
        XCTAssertEqual(context.localDayOfWeek, 1)
        XCTAssertEqual(RuntimeStageTools.serverToUTC(serverNow, context: context), utcMidnightMonday)
        XCTAssertEqual(RuntimeStageTools.utcToServer(utcMidnightMonday, context: context), serverNow)
        XCTAssertEqual(RuntimeStageTools.localToServer(localNow, context: context), serverNow)
        XCTAssertEqual(
            context.summary,
            "server=1704074400 utc=1704067200 local=1704092400 server_offset=7200 local_offset=25200 session=0"
        )
    }

    func testRuntimeTimeContextFallsBackToServerTimeWhenUTCAndLocalMissing() {
        let serverNow = Int64(1_704_110_400)
        let context = RuntimeStageTools.buildTimeContext(serverNow: serverNow, utcNow: 0, localNow: -1)

        XCTAssertTrue(context.ready)
        XCTAssertEqual(context.serverNow, serverNow)
        XCTAssertEqual(context.utcNow, serverNow)
        XCTAssertEqual(context.localNow, serverNow)
        XCTAssertEqual(context.serverUTCOffsetSeconds, 0)
        XCTAssertEqual(context.localUTCOffsetSeconds, 0)
        XCTAssertEqual(context.sessionBucket, 3)
    }

    func testRuntimeRouterCapPreservesOrderWithoutBudgetPressure() {
        var profile = LiveDeploymentProfile()
        profile.maxRuntimeModels = 3
        profile.performanceBudgetMS = 10.0

        let result = RuntimeStageTools.applyPerformanceModelCap(
            activeAIIDs: [7, 4, 2, 1],
            deployProfile: profile,
            performance: RuntimePerformanceState()
        )

        XCTAssertEqual(result.activeAIIDs, [7, 4, 2])
        XCTAssertEqual(result.activeModelCount, 3)
        XCTAssertEqual(result.runtimeModelCap, 3)
        XCTAssertEqual(result.budgetPressure, 0.0)
    }

    func testRuntimeRouterCapScalesDownUnderBudgetPressure() {
        var profile = LiveDeploymentProfile()
        profile.maxRuntimeModels = 10
        profile.performanceBudgetMS = 10.0
        var performance = RuntimePerformanceState()
        performance.recordStage(.total, elapsedMS: 20.0)

        let result = RuntimeStageTools.applyPerformanceModelCap(
            activeAIIDs: [0, 1, 2, 3, 4, 5, 6],
            deployProfile: profile,
            performance: performance
        )

        XCTAssertEqual(result.budgetPressure, 1.0, accuracy: 1e-12)
        XCTAssertEqual(result.runtimeModelCap, 4)
        XCTAssertEqual(result.activeAIIDs, [0, 1, 2, 3])
    }

    func testRuntimeFinalizeDecisionUsesLegacyReasonPriority() {
        XCTAssertEqual(RuntimeStageTools.finalizeDecision(.init(decision: 1)).reason, "buy")
        XCTAssertEqual(RuntimeStageTools.finalizeDecision(.init(decision: 0)).reason, "sell")
        XCTAssertEqual(
            RuntimeStageTools.finalizeDecision(
                .init(
                    decision: -1,
                    singleNoTradeReason: "single_reason",
                    probabilityCalibrationReady: true,
                    probabilityCalibrationPrimaryReason: "calibration_reason"
                )
            ).reason,
            "calibration_reason"
        )
        XCTAssertEqual(
            RuntimeStageTools.finalizeDecision(
                .init(
                    decision: -1,
                    aiType: RuntimeStageTools.m1SyncAIID,
                    singleNoTradeReason: "m1sync_no_trade"
                )
            ).reason,
            "m1sync_no_trade"
        )
        XCTAssertEqual(
            RuntimeStageTools.finalizeDecision(.init(decision: -1, ensembleMode: true, ensembleMetaTotal: 0.0)).reason,
            "no_meta_weight"
        )
        XCTAssertEqual(RuntimeStageTools.finalizeDecision(.init(decision: -1)).reason, "no_consensus_or_ev")
    }

    func testRuntimeFinalizeDecisionMatchesLegacySignalIntensityMath() {
        let finalized = RuntimeStageTools.finalizeDecision(
            .init(
                symbol: "EURUSD",
                decision: 1,
                signalBarUTC: 1_704_067_200,
                decisionKey: 42,
                macroProfileShortfall: 0.40,
                regimeTransitionPenalty: 0.25,
                tradeGate: 0.70,
                policyTradeProbability: 0.60,
                policyConfidence: 0.50,
                policySizeMultiplier: 1.20
            )
        )

        let expected = (0.55 * 0.70 + 0.25 * 0.60 + 0.20 * 0.50) *
            1.20 *
            (1.0 - 0.35 * 0.40 - 0.20 * 0.25)
        XCTAssertEqual(finalized.symbol, "EURUSD")
        XCTAssertEqual(finalized.decision, 1)
        XCTAssertEqual(finalized.signalBarUTC, 1_704_067_200)
        XCTAssertEqual(finalized.decisionKey, 42)
        XCTAssertEqual(finalized.signalIntensity, expected, accuracy: 1e-12)

        let skipped = RuntimeStageTools.finalizeDecision(
            .init(
                decision: -1,
                tradeGate: 2.0,
                policyTradeProbability: 2.0,
                policyConfidence: 2.0,
                policySizeMultiplier: 10.0
            )
        )
        XCTAssertEqual(skipped.signalIntensity, 0.0)
    }
}
