import Foundation
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
        XCTAssertEqual(
            ExecutionQualityTools.runtimeHistoryPath(symbol: "EUR/USD live"),
            "FXAI/Runtime/fxai_execution_quality_history_EUR_USD_live.ndjson"
        )
    }

    func testExecutionQualityConfigParsesPriceCostTSVAndLegacyAliases() {
        let config = ExecutionQualityTools.parseConfig(tsv: """
        enabled\t0
        block_on_unknown\t0
        support_soft_floor\t40
        support_hard_floor\t12
        threshold_caution_min\t0.61
        cap_price_cost_expected_mult\t5.50
        weight_price_cost_zscore\t0.33
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

        let legacyConfig = ExecutionQualityTools.parseConfig(tsv: """
        cap_spread_expected_mult\t6.25
        weight_spread_zscore\t0.44
        """)
        XCTAssertEqual(legacyConfig.capExpectedPriceCostMultiplier, 6.25, accuracy: 0.0)
        XCTAssertEqual(legacyConfig.weightPriceCostZScore, 0.44, accuracy: 0.0)
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

    func testExecutionQualityPolicyApplyMatchesLegacyScoreFormula() {
        let memory = ExecutionQualityTools.parseMemory(tsv: """
        meta\tgenerated_at\t2024-01-01T00:00:00Z
        meta\tdefault_method\tSCORECARD_V2
        tier\tPAIR_SESSION_REGIME\tEURUSD\tEU_US_OVERLAP\tTREND\t100\t0.80\t1.10\t1.20\t-0.02\t1.05\t1.10\t1.10
        """)
        let inputs = ExecutionQualityPolicyInputs(
            symbol: "EURUSD",
            generatedAtUTC: 1_704_067_200,
            regimeLabel: "TREND",
            priceCostPredictedPoints: 1.5,
            horizonMinutes: 15,
            upstreamDecision: 1,
            pathRisk: 0.30,
            fillRisk: 0.20,
            newsState: NewsPulsePairState(
                ready: true,
                available: true,
                stale: false,
                generatedAt: 1_704_067_100,
                eventETAMinutes: 20,
                newsRiskScore: 0.70,
                tradeGate: "CAUTION",
                sessionProfile: "LONDON"
            ),
            ratesState: RatesEnginePairState(
                ready: true,
                available: true,
                stale: false,
                generatedAt: 1_704_067_100,
                ratesRiskScore: 0.20
            ),
            crossAssetState: CrossAssetPairState(
                ready: true,
                available: true,
                stale: false,
                generatedAt: 1_704_067_100,
                usdLiquidityStressScore: 0.20,
                crossAssetDislocationScore: 0.30,
                pairCrossAssetRiskScore: 0.50
            ),
            microstructureState: MicrostructurePairState(
                ready: true,
                available: true,
                stale: false,
                generatedAt: 1_704_067_100,
                tickImbalance30s: -0.40,
                priceCostZscore60s: 2.0,
                tickRateZscore60s: 1.5,
                volBurstScore5m: 1.5,
                liquidityStressScore: 0.40,
                hostileExecutionScore: 0.30,
                sessionTag: "EU_US_OVERLAP"
            ),
            brokerStats: BrokerExecutionStats(
                coverage: 0.80,
                slippagePoints: 1.20,
                latencyPoints: 2.0,
                rejectProbability: 0.25,
                partialFillProbability: 0.35,
                fillRatioMean: 0.70,
                eventBurstPenalty: 0.40
            )
        )

        let state = ExecutionQualityTools.applyPolicy(
            config: ExecutionQualityConfig(),
            memory: memory,
            profile: ExecutionProfile.preset(.defaultProfile),
            inputs: inputs
        )

        XCTAssertTrue(state.ready)
        XCTAssertTrue(state.available)
        XCTAssertFalse(state.stale)
        XCTAssertEqual(state.symbol, "EURUSD")
        XCTAssertEqual(state.method, "SCORECARD_V2")
        XCTAssertEqual(state.sessionLabel, "EU_US_OVERLAP")
        XCTAssertEqual(state.regimeLabel, "TREND")
        XCTAssertTrue(state.newsWindowActive)
        XCTAssertFalse(state.ratesRepricingActive)
        XCTAssertEqual(state.selectedTierKey, "PAIR_SESSION_REGIME|EURUSD|EU_US_OVERLAP|TREND")
        XCTAssertEqual(state.brokerCoverage, 0.80, accuracy: 0.0)
        XCTAssertEqual(state.brokerRejectProbability, 0.25, accuracy: 0.0)
        XCTAssertEqual(state.brokerPartialFillProbability, 0.35, accuracy: 0.0)
        XCTAssertEqual(state.priceCostWideningRisk, 0.6656, accuracy: 1e-12)
        XCTAssertEqual(state.priceCostExpectedPoints, 2.9747760000000003, accuracy: 1e-12)
        XCTAssertEqual(state.expectedSlippagePoints, 3.0147641600000004, accuracy: 1e-12)
        XCTAssertEqual(state.slippageRisk, 0.5084091166739958, accuracy: 1e-12)
        XCTAssertEqual(state.latencySensitivityScore, 0.628, accuracy: 1e-12)
        XCTAssertEqual(state.liquidityFragilityScore, 0.5034000000000001, accuracy: 1e-12)
        XCTAssertEqual(state.fillQualityScore, 0.34157744733128126, accuracy: 1e-12)
        XCTAssertEqual(state.executionQualityScore, 0.3895413379311932, accuracy: 1e-12)
        XCTAssertEqual(state.executionState, "CAUTION")
        XCTAssertEqual(state.allowedDeviationPoints, 4.5970288070898615, accuracy: 1e-12)
        XCTAssertEqual(state.cautionLotScale, 0.82, accuracy: 0.0)
        XCTAssertEqual(state.cautionEnterProbabilityBuffer, 0.04, accuracy: 0.0)
        XCTAssertEqual(state.reasonsCSV, "NEWS_WINDOW_ACTIVE; EXECUTION_STATE_CAUTION")
    }

    func testExecutionQualityRuntimeArtifactsUsePriceCostKeys() throws {
        let state = ExecutionQualityPairState(
            ready: true,
            available: true,
            stale: false,
            fallbackUsed: true,
            memoryStale: false,
            dataStale: true,
            supportUsable: true,
            newsWindowActive: true,
            ratesRepricingActive: false,
            generatedAt: 1_704_067_200,
            symbol: "EURUSD",
            method: "SCORECARD_V2",
            sessionLabel: "EU_US_OVERLAP",
            regimeLabel: "TREND",
            selectedTierKind: "PAIR_SESSION_REGIME",
            selectedTierKey: "PAIR_SESSION_REGIME|EURUSD|EU_US_OVERLAP|TREND",
            selectedSupport: 100,
            selectedQuality: 0.80,
            brokerCoverage: 0.90,
            brokerRejectProbability: 0.20,
            brokerPartialFillProbability: 0.30,
            priceCostNowPoints: 1.25,
            priceCostExpectedPoints: 2.50,
            priceCostWideningRisk: 0.60,
            expectedSlippagePoints: 3.75,
            slippageRisk: 0.40,
            fillQualityScore: 0.55,
            latencySensitivityScore: 0.45,
            liquidityFragilityScore: 0.35,
            executionQualityScore: 0.50,
            allowedDeviationPoints: 4.25,
            cautionLotScale: 0.82,
            cautionEnterProbabilityBuffer: 0.04,
            executionState: "CAUTION",
            reasons: ["DATA_STALE", "EXECUTION_STATE_CAUTION"]
        )

        let tsv = try XCTUnwrap(ExecutionQualityTools.runtimeStateTSV(symbol: "EUR/USD live", state: state))
        XCTAssertTrue(tsv.hasSuffix("\r\n"))
        XCTAssertTrue(tsv.contains("symbol\tEUR/USD live\r\n"))
        XCTAssertTrue(tsv.contains("selected_quality\t0.800000\r\n"))
        XCTAssertTrue(tsv.contains("fallback_used\t1\r\n"))
        XCTAssertTrue(tsv.contains("rates_repricing_active\t0\r\n"))
        XCTAssertTrue(tsv.contains("price_cost_expected_points\t2.500000\r\n"))
        XCTAssertFalse(tsv.contains("spread_expected_points"))
        XCTAssertTrue(tsv.contains("reasons_csv\tDATA_STALE; EXECUTION_STATE_CAUTION\r\n"))

        let parsed = try XCTUnwrap(ExecutionQualityTools.readPairState(symbol: "EUR/USD live", stateTSV: tsv, nowUTC: 1_704_067_201))
        XCTAssertEqual(parsed.symbol, "EUR/USD live")
        XCTAssertEqual(parsed.executionState, "CAUTION")
        XCTAssertEqual(parsed.priceCostExpectedPoints, 2.50)
        XCTAssertEqual(parsed.reasonsCSV, "DATA_STALE; EXECUTION_STATE_CAUTION")

        let line = try XCTUnwrap(ExecutionQualityTools.runtimeHistoryNDJSONLine(symbol: "EUR/USD live", state: state))
        let data = try XCTUnwrap(line.data(using: .utf8))
        let object = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
        XCTAssertEqual(object["generated_at"] as? String, "2024-01-01T00:00:00Z")
        XCTAssertEqual(object["symbol"] as? String, "EUR/USD live")
        let nested = try XCTUnwrap(object["state"] as? [String: Any])
        XCTAssertEqual(nested["execution_state"] as? String, "CAUTION")
        XCTAssertEqual(nested["fallback_used"] as? Int, 1)
        XCTAssertEqual(try XCTUnwrap(nested["selected_quality"] as? Double), 0.80, accuracy: 0.0)
        XCTAssertEqual(try XCTUnwrap(nested["price_cost_expected_points"] as? Double), 2.50, accuracy: 0.0)
        XCTAssertNil(nested["spread_expected_points"])
        XCTAssertEqual(nested["reason_codes"] as? [String], ["DATA_STALE", "EXECUTION_STATE_CAUTION"])
    }

    func testExecutionQualityRepositoryWritesStateAndAppendsHistory() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("ExecutionQualityTests-\(UUID().uuidString)", isDirectory: true)
        let repository = RuntimeArtifactFileRepository(rootURL: root)
        var state = ExecutionQualityPairState(
            ready: true,
            generatedAt: 1_704_067_200,
            method: "SCORECARD_V2",
            executionState: "NORMAL",
            reasons: ["EXECUTION_STATE_CAUTION"]
        )

        try repository.writeExecutionQualityRuntimeArtifacts(symbol: "EUR/USD live", state: state)
        state.generatedAt = 1_704_067_260
        state.executionState = "CAUTION"
        try repository.writeExecutionQualityRuntimeArtifacts(symbol: "EUR/USD live", state: state)

        let statePath = ExecutionQualityTools.runtimeStatePath(symbol: "EUR/USD live")
        let historyPath = ExecutionQualityTools.runtimeHistoryPath(symbol: "EUR/USD live")
        let stateText = try String(contentsOf: root.appendingPathComponent(statePath), encoding: .utf8)
        let historyText = try String(contentsOf: root.appendingPathComponent(historyPath), encoding: .utf8)

        XCTAssertTrue(stateText.contains("generated_at\t1704067260\r\n"))
        XCTAssertTrue(stateText.contains("execution_state\tCAUTION\r\n"))
        XCTAssertEqual(historyText.components(separatedBy: .newlines).filter { !$0.isEmpty }.count, 2)
        XCTAssertTrue(historyText.contains("\"generated_at\":\"2024-01-01T00:00:00Z\""))
        XCTAssertTrue(historyText.contains("\"generated_at\":\"2024-01-01T00:01:00Z\""))
    }

    func testExecutionQualityParsesKeyValueStateAndReasonCSV() {
        let state = ExecutionQualityTools.readPairState(
            symbol: "EURUSD",
            stateTSV: """
            symbol\tEURUSD
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
            news_window_active\t1
            rates_repricing_active\t0
            broker_coverage\t0.9
            broker_reject_prob\t0.2
            broker_partial_fill_prob\t0.3
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
        XCTAssertEqual(state?.symbol, "EURUSD")
        XCTAssertEqual(state?.method, "SCORECARD_V2")
        XCTAssertEqual(state?.sessionLabel, "LONDON")
        XCTAssertEqual(state?.regimeLabel, "TREND")
        XCTAssertEqual(state?.selectedTierKind, "SYMBOL_SESSION")
        XCTAssertEqual(state?.selectedTierKey, "SYMBOL|EURUSD|LONDON|*")
        XCTAssertEqual(state?.selectedSupport, 42)
        XCTAssertEqual(state?.selectedQuality, 0.81)
        XCTAssertTrue(state?.newsWindowActive ?? false)
        XCTAssertFalse(state?.ratesRepricingActive ?? true)
        XCTAssertEqual(state?.brokerCoverage, 0.9)
        XCTAssertEqual(state?.brokerRejectProbability, 0.2)
        XCTAssertEqual(state?.brokerPartialFillProbability, 0.3)
        XCTAssertEqual(state?.priceCostNowPoints, 1.2)
        XCTAssertEqual(state?.priceCostExpectedPoints, 0.8)
        XCTAssertEqual(state?.priceCostWideningRisk, 1.4)
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

    func testExecutionQualityParsesPriceCostStateKeys() {
        let state = ExecutionQualityTools.readPairState(
            symbol: "EURUSD",
            stateTSV: """
            symbol\tEURUSD
            generated_at\t1000
            price_cost_now_points\t1.3
            price_cost_expected_points\t2.4
            price_cost_widening_risk\t0.6
            execution_state\tNORMAL
            """,
            nowUTC: 1_100,
            freshnessMaxSeconds: 180
        )

        XCTAssertNotNil(state)
        XCTAssertEqual(state?.priceCostNowPoints, 1.3)
        XCTAssertEqual(state?.priceCostExpectedPoints, 2.4)
        XCTAssertEqual(state?.priceCostWideningRisk, 0.6)
        XCTAssertEqual(state?.executionState, "NORMAL")
    }

    func testExecutionQualityPairStateCodableDecodesLegacySpreadKeysAndEncodesPriceCostKeys() throws {
        let legacyJSON = """
        {
          "ready": true,
          "available": true,
          "stale": false,
          "generatedAt": 1704067200,
          "symbol": "eurusd",
          "spreadNowPoints": 1.1,
          "spreadExpectedPoints": 2.2,
          "spreadWideningRisk": 0.3,
          "reasons": ["DATA_STALE", "DATA_STALE", " SUPPORT_TOO_LOW "]
        }
        """
        let decoded = try JSONDecoder().decode(
            ExecutionQualityPairState.self,
            from: try XCTUnwrap(legacyJSON.data(using: .utf8))
        )

        XCTAssertEqual(decoded.symbol, "EURUSD")
        XCTAssertEqual(decoded.priceCostNowPoints, 1.1)
        XCTAssertEqual(decoded.priceCostExpectedPoints, 2.2)
        XCTAssertEqual(decoded.priceCostWideningRisk, 0.3)
        XCTAssertEqual(decoded.reasonsCSV, "DATA_STALE; SUPPORT_TOO_LOW")

        let encoded = try JSONEncoder().encode(decoded)
        let encodedText = try XCTUnwrap(String(data: encoded, encoding: .utf8))
        XCTAssertTrue(encodedText.contains("\"priceCostExpectedPoints\":2.2"))
        XCTAssertFalse(encodedText.contains("spreadExpectedPoints"))
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
