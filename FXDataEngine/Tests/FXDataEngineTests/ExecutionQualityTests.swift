import XCTest
@testable import FXDataEngine

final class ExecutionQualityTests: XCTestCase {
    func testExecutionQualityRuntimePathUsesControlPlaneSafeToken() {
        XCTAssertEqual(
            ExecutionQualityTools.runtimeStatePath(symbol: "EUR/USD live"),
            "FXAI/Runtime/fxai_execution_quality_EUR_USD_live.tsv"
        )
    }

    func testExecutionQualityParsesKeyValueStateAndReasonCSV() {
        let state = ExecutionQualityTools.readPairState(
            symbol: "EURUSD",
            stateTSV: """
            generated_at\t1000
            method\tSCORECARD_V2
            session_label\tLONDON
            regime_label\tTREND
            selected_tier_kind\tSYMBOL_SESSION
            selected_tier_key\tSYMBOL|EURUSD|LONDON|*
            selected_support\t42
            selected_quality\t0.81
            fallback_used\t1
            memory_stale\t0
            data_stale\t1
            support_usable\t1
            spread_now_points\t1.2
            spread_expected_points\t0.8
            spread_widening_risk\t1.4
            expected_slippage_points\t0.6
            slippage_risk\t0.7
            fill_quality_score\t0.75
            latency_sensitivity_score\t0.2
            liquidity_fragility_score\t0.3
            execution_quality_score\t0.64
            allowed_deviation_points\t2.5
            caution_lot_scale\t1.3
            caution_enter_prob_buffer\t0.4
            execution_state\tSTRESSED
            reasons_csv\tDATA_STALE; MEMORY_STALE ;DATA_STALE; ; SUPPORT_TOO_LOW
            """,
            nowUTC: 1_100,
            freshnessMaxSeconds: 180
        )

        XCTAssertNotNil(state)
        XCTAssertTrue(state?.ready ?? false)
        XCTAssertTrue(state?.available ?? false)
        XCTAssertFalse(state?.stale ?? true)
        XCTAssertTrue(state?.dataStale ?? false)
        XCTAssertEqual(state?.systemHealthState.healthComponent.stale, true)
        XCTAssertTrue(state?.fallbackUsed ?? false)
        XCTAssertFalse(state?.memoryStale ?? true)
        XCTAssertTrue(state?.supportUsable ?? false)
        XCTAssertEqual(state?.generatedAt, 1_000)
        XCTAssertEqual(state?.method, "SCORECARD_V2")
        XCTAssertEqual(state?.sessionLabel, "LONDON")
        XCTAssertEqual(state?.regimeLabel, "TREND")
        XCTAssertEqual(state?.selectedTierKind, "SYMBOL_SESSION")
        XCTAssertEqual(state?.selectedTierKey, "SYMBOL|EURUSD|LONDON|*")
        XCTAssertEqual(state?.selectedSupport, 42)
        XCTAssertEqual(state?.selectedQuality, 0.81)
        XCTAssertEqual(state?.spreadNowPoints, 1.2)
        XCTAssertEqual(state?.spreadExpectedPoints, 0.8)
        XCTAssertEqual(state?.spreadWideningRisk, 1.4)
        XCTAssertEqual(state?.expectedSlippagePoints, 0.6)
        XCTAssertEqual(state?.slippageRisk, 0.7)
        XCTAssertEqual(state?.fillQualityScore, 0.75)
        XCTAssertEqual(state?.latencySensitivityScore, 0.2)
        XCTAssertEqual(state?.liquidityFragilityScore, 0.3)
        XCTAssertEqual(state?.executionQualityScore, 0.64)
        XCTAssertEqual(state?.allowedDeviationPoints, 2.5)
        XCTAssertEqual(state?.cautionLotScale, 1.3)
        XCTAssertEqual(state?.cautionEnterProbabilityBuffer, 0.4)
        XCTAssertEqual(state?.executionState, "STRESSED")
        XCTAssertEqual(state?.reasonsCSV, "DATA_STALE; MEMORY_STALE; SUPPORT_TOO_LOW")
    }

    func testExecutionQualityFreshnessOverwritesStaleAndMissingClockFailsStale() {
        let staleState = ExecutionQualityTools.readPairState(
            symbol: "EURUSD",
            stateTSV: "generated_at\t1000\n",
            nowUTC: 1_181,
            freshnessMaxSeconds: 180
        )
        let missingClockState = ExecutionQualityTools.readPairState(
            symbol: "EURUSD",
            stateTSV: "generated_at\t1000\n",
            nowUTC: 0
        )
        let missingGeneratedState = ExecutionQualityTools.readPairState(
            symbol: "EURUSD",
            stateTSV: "method\tSCORECARD_V1\n",
            nowUTC: 1_000
        )

        XCTAssertTrue(staleState?.stale ?? false)
        XCTAssertTrue(missingClockState?.stale ?? false)
        XCTAssertTrue(missingGeneratedState?.stale ?? false)
    }

    func testExecutionQualityUnknownKeyStillMarksStateAvailableLikeLegacyReader() {
        let state = ExecutionQualityTools.readPairState(
            symbol: "EURUSD",
            stateTSV: "unknown\tvalue\n",
            nowUTC: 1_000
        )

        XCTAssertNotNil(state)
        XCTAssertTrue(state?.ready ?? false)
        XCTAssertTrue(state?.available ?? false)
        XCTAssertTrue(state?.stale ?? false)
        XCTAssertEqual(state?.method, "SCORECARD_V1")
        XCTAssertEqual(state?.selectedTierKey, "GLOBAL|*|*|*")
    }

    func testExecutionQualityReasonLimitMatchesLegacyReader() {
        let state = ExecutionQualityTools.readPairState(
            symbol: "EURUSD",
            stateTSV: """
            generated_at\t1000
            reasons_csv\tr1;r2;r3;r4;r5;r6;r7;r8;r9;r10;r11;r12;r13
            """,
            nowUTC: 1_001
        )

        XCTAssertEqual(state?.reasonCount, ExecutionQualityConstants.maxReasons)
        XCTAssertEqual(state?.reasonsCSV, "r1; r2; r3; r4; r5; r6; r7; r8; r9; r10; r11; r12")
    }

    func testExecutionQualityUnavailableWithoutStateFile() {
        XCTAssertNil(ExecutionQualityTools.readPairState(symbol: "EURUSD", stateTSV: nil))
        XCTAssertNil(ExecutionQualityTools.readPairState(symbol: "EURUSD", stateTSV: ""))
    }
}
