import XCTest
@testable import FXDataEngine

final class ExecutionQualityTests: XCTestCase {
    func testExecutionQualityRuntimePathUsesControlPlaneSafeToken() {
        XCTAssertEqual(
            ExecutionQualityTools.configPath(),
            "FXAI/Runtime/execution_quality_config.tsv"
        )
        XCTAssertEqual(
            ExecutionQualityTools.memoryPath(),
            "FXAI/Runtime/execution_quality_memory.tsv"
        )
        XCTAssertEqual(
            ExecutionQualityTools.runtimeStatePath(symbol: "EUR/USD live"),
            "FXAI/Runtime/fxai_execution_quality_EUR_USD_live.tsv"
        )
    }

    func testExecutionQualityConfigParsesLegacyTSVWithRenamedCostFields() {
        let config = ExecutionQualityTools.parseConfig(tsv: """
        enabled\t0
        block_on_unknown\t0
        support_soft_floor\t40
        support_hard_floor\t12
        threshold_caution_min\t0.61
        cap_spread_expected_mult\t5.50
        weight_spread_zscore\t0.33
        weight_broker_event_burst\t0.21
        bucket_count\t2
        bucket_0\tpair_regime
        bucket_1\tglobal
        cap_allowed_deviation_points_max\t1.0
        cap_allowed_deviation_points_min\t2.5
        """)

        XCTAssertTrue(config.ready)
        XCTAssertFalse(config.enabled)
        XCTAssertFalse(config.blockOnUnknown)
        XCTAssertEqual(config.supportSoftFloor, 40)
        XCTAssertEqual(config.supportHardFloor, 12)
        XCTAssertEqual(config.thresholdCautionMin, 0.61, accuracy: 0.0)
        XCTAssertEqual(config.capExpectedPriceCostMultiplier, 5.50, accuracy: 0.0)
        XCTAssertEqual(config.weightPriceCostZScore, 0.33, accuracy: 0.0)
        XCTAssertEqual(config.weightBrokerEventBurst, 0.21, accuracy: 0.0)
        XCTAssertEqual(config.capAllowedDeviationPointsMin, 2.5, accuracy: 0.0)
        XCTAssertEqual(config.capAllowedDeviationPointsMax, 2.5, accuracy: 0.0)
        XCTAssertEqual(config.effectiveBucketHierarchy, ["PAIR_REGIME", "GLOBAL"])

        let defaults = ExecutionQualityConfig()
        XCTAssertEqual(defaults.bucketHierarchy[0], "PAIR_SESSION_REGIME")
        XCTAssertEqual(defaults.lotScaleStressed, 0.58, accuracy: 0.0)
        XCTAssertEqual(defaults.enterProbabilityBufferBlocked, 1.0, accuracy: 0.0)
    }

    func testExecutionQualityMemoryTierSelectionMatchesLegacySupportPreference() {
        let config = ExecutionQualityConfig(
            supportSoftFloor: 64,
            supportHardFloor: 16,
            bucketCount: 3,
            bucketHierarchy: ["PAIR_SESSION_REGIME", "PAIR_REGIME", "GLOBAL"]
        )
        let memory = ExecutionQualityTools.parseMemory(tsv: """
        meta\tgenerated_at\t2024-01-01T00:00:00Z
        meta\tdefault_method\tSCORECARD_V2
        tier\tPAIR_SESSION_REGIME\tEURUSD\tLONDON\tTREND\t65\t0.95\t1.10\t1.20\t-0.02\t1.03\t1.04\t1.05
        tier\tPAIR_SESSION_REGIME\tEURUSD\tLONDON\tTREND\t70\t0.82\t1.12\t1.18\t-0.03\t1.06\t1.07\t1.08
        tier\tPAIR_REGIME\tEURUSD\t*\tTREND\t20\t0.90\t1.30\t1.40\t-0.10\t1.11\t1.12\t1.13
        tier\tGLOBAL\t*\t*\t*\t120\t0.70\t1.01\t1.02\t-0.01\t1.00\t1.00\t1.00
        """)

        XCTAssertEqual(memory.generatedAt, 1_704_067_200)
        XCTAssertEqual(memory.defaultMethod, "SCORECARD_V2")
        XCTAssertEqual(memory.tiers.count, 4)

        let preferred = ExecutionQualityTools.selectTier(
            symbol: "eurusd",
            session: "london",
            regime: "trend",
            config: config,
            memory: memory
        )
        XCTAssertTrue(preferred.found)
        XCTAssertFalse(preferred.fallbackUsed)
        XCTAssertTrue(preferred.supportUsable)
        XCTAssertEqual(preferred.tier.support, 70)
        XCTAssertEqual(preferred.tier.quality, 0.82, accuracy: 0.0)
        XCTAssertEqual(preferred.tier.key, "PAIR_SESSION_REGIME|EURUSD|LONDON|TREND")

        let fallback = ExecutionQualityTools.selectTier(
            symbol: "EURUSD",
            session: "TOKYO",
            regime: "TREND",
            config: config,
            memory: memory
        )
        XCTAssertTrue(fallback.found)
        XCTAssertTrue(fallback.fallbackUsed)
        XCTAssertTrue(fallback.supportUsable)
        XCTAssertEqual(fallback.tier.kind, "PAIR_REGIME")
        XCTAssertEqual(fallback.tier.support, 20)
    }

    func testExecutionQualityTierSelectionFallsBackAndSessionThinnessMatchesLegacyRules() {
        let empty = ExecutionQualityTools.selectTier(
            symbol: "EURUSD",
            session: "LONDON",
            regime: "TREND",
            config: ExecutionQualityConfig(),
            memory: ExecutionQualityMemory()
        )

        XCTAssertFalse(empty.found)
        XCTAssertTrue(empty.fallbackUsed)
        XCTAssertFalse(empty.supportUsable)
        XCTAssertTrue(empty.tier.ready)
        XCTAssertEqual(empty.tier.key, "GLOBAL|*|*|*")

        XCTAssertEqual(ExecutionQualityTools.sessionThinness(sessionLabel: "LONDON", handoffFlag: false), 0.18, accuracy: 0.0)
        XCTAssertEqual(ExecutionQualityTools.sessionThinness(sessionLabel: "ASIA", handoffFlag: false), 0.42, accuracy: 0.0)
        XCTAssertEqual(ExecutionQualityTools.sessionThinness(sessionLabel: "EU_US_OVERLAP", handoffFlag: false), 0.22, accuracy: 0.0)
        XCTAssertEqual(ExecutionQualityTools.sessionThinness(sessionLabel: "ROLLOVER", handoffFlag: false), 0.60, accuracy: 0.0)
        XCTAssertEqual(ExecutionQualityTools.sessionThinness(sessionLabel: "LONDON", handoffFlag: true), 0.55, accuracy: 0.0)
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
