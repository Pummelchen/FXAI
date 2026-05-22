import Foundation
import XCTest
@testable import FXDataEngine

final class ProbabilityCalibrationTests: XCTestCase {
    func testProbabilityCalibrationRuntimePathUsesControlPlaneSafeToken() {
        XCTAssertEqual(
            ProbabilityCalibrationTools.configPath(),
            "FXAI/Runtime/prob_calibration_config.tsv"
        )
        XCTAssertEqual(
            ProbabilityCalibrationTools.memoryPath(),
            "FXAI/Runtime/prob_calibration_memory.tsv"
        )
        XCTAssertEqual(
            ProbabilityCalibrationTools.runtimeStatePath(symbol: "EUR/USD live"),
            "FXAI/Runtime/fxai_prob_calibration_EUR_USD_live.tsv"
        )
        XCTAssertEqual(
            ProbabilityCalibrationTools.runtimeHistoryPath(symbol: "EUR/USD live"),
            "FXAI/Runtime/fxai_prob_calibration_history_EUR_USD_live.ndjson"
        )
    }

    func testProbabilityCalibrationConfigParsesLegacyRuntimeTSV() {
        let config = ProbabilityCalibrationTools.parseConfig(tsv: """
        schema_version\t1
        enabled\t0
        allow_abstain_flag\t0
        neutral_blend_gain\t0.40
        skip_uncertainty_gain\t0.20
        support_soft_floor\t80
        support_hard_floor\t20
        memory_stale_after_hours\t12
        min_calibration_quality\t0.50
        max_uncertainty_score\t0.85
        soft_prob_scale\t1.90
        soft_skip_bias\t0.11
        uncertainty_support\t0.44
        uncertainty_distribution_width\t0.31
        risk_path_mult\t0.29
        bucket_hierarchy\tpair_regime
        bucket_hierarchy\tglobal
        """)

        XCTAssertTrue(config.ready)
        XCTAssertFalse(config.enabled)
        XCTAssertFalse(config.allowAbstainFlag)
        XCTAssertEqual(config.neutralBlendGain, 0.40, accuracy: 0.0)
        XCTAssertEqual(config.skipUncertaintyGain, 0.20, accuracy: 0.0)
        XCTAssertEqual(config.supportSoftFloor, 80)
        XCTAssertEqual(config.supportHardFloor, 20)
        XCTAssertEqual(config.memoryStaleAfterHours, 12)
        XCTAssertEqual(config.minCalibrationQuality, 0.50, accuracy: 0.0)
        XCTAssertEqual(config.maxUncertaintyScore, 0.85, accuracy: 0.0)
        XCTAssertEqual(config.softProbabilityScale, 1.90, accuracy: 0.0)
        XCTAssertEqual(config.softSkipBias, 0.11, accuracy: 0.0)
        XCTAssertEqual(config.uncertaintySupportPenalty, 0.44, accuracy: 0.0)
        XCTAssertEqual(config.uncertaintyDistributionWidthPenalty, 0.31, accuracy: 0.0)
        XCTAssertEqual(config.riskPathMultiplier, 0.29, accuracy: 0.0)
        XCTAssertEqual(config.effectiveBucketHierarchy, ["PAIR_REGIME", "GLOBAL"])

        let defaults = ProbabilityCalibrationConfig()
        XCTAssertEqual(defaults.bucketHierarchy[0], "PAIR_SESSION_REGIME")
        XCTAssertEqual(defaults.uncertaintyStaleContextPenalty, 0.16, accuracy: 0.0)
        XCTAssertEqual(defaults.riskBlockPostureMultiplier, 0.42, accuracy: 0.0)
    }

    func testProbabilityCalibrationMemoryTierSelectionMatchesLegacySupportPreference() {
        let config = ProbabilityCalibrationConfig(
            supportSoftFloor: 64,
            supportHardFloor: 16,
            bucketCount: 3,
            bucketHierarchy: ["PAIR_SESSION_REGIME", "PAIR_REGIME", "GLOBAL"]
        )
        let memory = ProbabilityCalibrationTools.parseMemory(tsv: """
        schema_version\t1
        generated_at\t2024-01-01T00:00:00Z
        generated_at_unix\t1704067200
        default_method\tLOGISTIC_AFFINE_V2
        tier\tPAIR_SESSION_REGIME\tEURUSD\tLONDON\tTREND\t65\t2.20\t0.01\t0.04\t0.92\t0.78\t0.86\t0.98\t0.66\t0.86\t0.66
        tier\tPAIR_SESSION_REGIME\tEURUSD\tLONDON\tTREND\t70\t2.10\t0.02\t0.05\t0.90\t0.76\t0.84\t0.96\t0.61\t0.90\t0.64
        tier\tPAIR_REGIME\tEURUSD\t*\tTREND\t20\t1.80\t0.00\t0.07\t0.82\t0.70\t0.78\t0.92\t0.55\t1.10\t0.60
        tier\tGLOBAL\t*\t*\t*\t120\t1.70\t0.00\t0.08\t0.78\t0.60\t0.72\t0.88\t0.50\t1.30\t0.58
        """)

        XCTAssertEqual(memory.generatedAt, 1_704_067_200)
        XCTAssertEqual(memory.defaultMethod, "LOGISTIC_AFFINE_V2")
        XCTAssertEqual(memory.tiers.count, 4)

        let preferred = ProbabilityCalibrationTools.selectTier(
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
        XCTAssertEqual(preferred.tier.calibrationQuality, 0.61, accuracy: 0.0)
        XCTAssertEqual(preferred.tier.key, "PAIR_SESSION_REGIME|EURUSD|LONDON|TREND")

        let fallback = ProbabilityCalibrationTools.selectTier(
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

        let empty = ProbabilityCalibrationTools.selectTier(
            symbol: "EURUSD",
            session: "LONDON",
            regime: "TREND",
            config: config,
            memory: ProbabilityCalibrationMemory()
        )
        XCTAssertFalse(empty.found)
        XCTAssertTrue(empty.fallbackUsed)
        XCTAssertFalse(empty.supportUsable)
        XCTAssertTrue(empty.tier.ready)
        XCTAssertEqual(empty.tier.probabilityScale, config.softProbabilityScale, accuracy: 0.0)
    }

    func testProbabilityCalibrationApplyMatchesLegacyPreparedStateFormula() {
        let memory = ProbabilityCalibrationTools.parseMemory(tsv: """
        generated_at_unix\t1704067200
        default_method\tLOGISTIC_AFFINE_V2
        tier\tPAIR_SESSION_REGIME\tEURUSD\tEU_US_OVERLAP\tTREND\t100\t2.12\t0.00\t0.05\t0.84\t0.68\t0.78\t0.92\t0.58\t1.00\t0.62
        """)
        let executionQuality = ExecutionQualityPairState(
            ready: true,
            dataStale: false,
            priceCostExpectedPoints: 2.20,
            expectedSlippagePoints: 1.40,
            fillQualityScore: 0.70,
            latencySensitivityScore: 0.20,
            liquidityFragilityScore: 0.30,
            executionState: "NORMAL"
        )
        let inputs = ProbabilityCalibrationPolicyInputs(
            symbol: "EURUSD",
            generatedAtUTC: 1_704_067_200,
            regimeLabel: "TREND",
            rawBuyProbability: 0.58,
            rawSellProbability: 0.21,
            rawSkipProbability: 0.21,
            moveMeanPoints: 12.0,
            moveQ25Points: 5.0,
            moveQ50Points: 9.0,
            moveQ75Points: 16.0,
            agreementScore: 0.82,
            minMovePoints: 3.0,
            priceCostPoints: 1.0,
            commissionPoints: 0.2,
            costBufferPoints: 0.3,
            horizonMinutes: 15,
            upstreamDecision: 1,
            pathRisk: 0.20,
            fillRisk: 0.10,
            dynamicStateReady: true,
            dynamicTradePosture: "NORMAL",
            dynamicAbstainBias: 0.04,
            adaptiveRouterPosture: "NORMAL",
            adaptiveRouterAbstainBias: 0.06,
            executionQualityEnabled: true,
            newsState: NewsPulsePairState(
                ready: true,
                available: true,
                stale: false,
                generatedAt: 1_704_067_100,
                eventETAMinutes: 120,
                newsRiskScore: 0.12,
                tradeGate: "OPEN",
                sessionProfile: "LONDON"
            ),
            ratesState: RatesEnginePairState(
                ready: true,
                available: true,
                stale: false,
                generatedAt: 1_704_067_100,
                ratesRiskScore: 0.10,
                tradeGate: "OPEN"
            ),
            crossAssetState: CrossAssetPairState(
                ready: true,
                available: true,
                stale: false,
                generatedAt: 1_704_067_100,
                usdLiquidityStressScore: 0.05,
                crossAssetDislocationScore: 0.08,
                pairCrossAssetRiskScore: 0.10,
                tradeGate: "OPEN"
            ),
            microstructureState: MicrostructurePairState(
                ready: true,
                available: true,
                stale: false,
                generatedAt: 1_704_067_100,
                spreadZscore60s: 0.4,
                liquidityStressScore: 0.14,
                hostileExecutionScore: 0.10,
                sessionTag: "EU_US_OVERLAP",
                tradeGate: "OPEN"
            ),
            executionQualityState: executionQuality
        )

        let outcome = ProbabilityCalibrationTools.applyCalibration(
            config: ProbabilityCalibrationConfig(),
            memory: memory,
            profile: ExecutionProfile.preset(.defaultProfile),
            inputs: inputs
        )
        let state = outcome.state

        XCTAssertTrue(state.ready)
        XCTAssertTrue(state.available)
        XCTAssertFalse(state.calibrationStale)
        XCTAssertFalse(state.inputStale)
        XCTAssertEqual(outcome.decision, -1)
        XCTAssertEqual(state.symbol, "EURUSD")
        XCTAssertEqual(state.method, "LOGISTIC_AFFINE_V2")
        XCTAssertEqual(state.sessionLabel, "EU_US_OVERLAP")
        XCTAssertEqual(state.regimeLabel, "TREND")
        XCTAssertEqual(state.selectedTierKey, "PAIR_SESSION_REGIME|EURUSD|EU_US_OVERLAP|TREND")
        XCTAssertEqual(state.rawAction, "BUY")
        XCTAssertEqual(state.rawScore, 0.37, accuracy: 1e-12)
        XCTAssertEqual(state.calibratedBuyProbability, 0.4472861866666667, accuracy: 1e-12)
        XCTAssertEqual(state.calibratedSellProbability, 0.27414314666666667, accuracy: 1e-12)
        XCTAssertEqual(state.calibratedSkipProbability, 0.2785706666666667, accuracy: 1e-12)
        XCTAssertEqual(state.calibratedConfidence, 0.4472861866666667, accuracy: 1e-12)
        XCTAssertEqual(state.expectedMoveQ25Points, 3.0765194666666673, accuracy: 1e-12)
        XCTAssertEqual(state.expectedMoveQ50Points, 6.57473856, accuracy: 1e-12)
        XCTAssertEqual(state.expectedMoveQ75Points, 14.136466488888889, accuracy: 1e-12)
        XCTAssertEqual(state.expectedMoveMeanPoints, 9.36073152, accuracy: 1e-12)
        XCTAssertEqual(state.priceCostPoints, 4.70, accuracy: 1e-12)
        XCTAssertEqual(state.slippageCostPoints, 1.40, accuracy: 1e-12)
        XCTAssertEqual(state.uncertaintyScore, 0.39642222222222223, accuracy: 1e-12)
        XCTAssertEqual(state.uncertaintyPenaltyPoints, 1.1892666666666667, accuracy: 1e-12)
        XCTAssertEqual(state.riskPenaltyPoints, 1.014, accuracy: 1e-12)
        XCTAssertEqual(state.finalAction, "SKIP")
        XCTAssertTrue(state.abstain)
        XCTAssertEqual(state.reasonsCSV, "MOVE_DISTRIBUTION_TOO_WEAK; COST_TOO_HIGH; EDGE_TOO_SMALL")
    }

    func testProbabilityCalibrationRuntimeArtifactsMatchLegacyTSVAndNDJSONShape() throws {
        let state = ProbabilityCalibrationRuntimeState(
            ready: true,
            available: true,
            stale: false,
            fallbackUsed: true,
            calibrationStale: false,
            inputStale: true,
            supportUsable: true,
            generatedAt: 1_704_067_200,
            symbol: "EURUSD",
            method: "LOGISTIC_AFFINE_V2",
            sessionLabel: "EU_US_OVERLAP",
            regimeLabel: "TREND",
            selectedTierKind: "PAIR_SESSION_REGIME",
            selectedTierKey: "PAIR_SESSION_REGIME|EURUSD|EU_US_OVERLAP|TREND",
            selectedSupport: 100,
            selectedQuality: 0.80,
            rawBuyProbability: 0.58,
            rawSellProbability: 0.21,
            rawSkipProbability: 0.21,
            rawScore: 0.37,
            rawAction: "BUY",
            calibratedBuyProbability: 0.45,
            calibratedSellProbability: 0.25,
            calibratedSkipProbability: 0.30,
            calibratedConfidence: 0.45,
            expectedMoveMeanPoints: 9.0,
            expectedMoveQ25Points: 3.0,
            expectedMoveQ50Points: 6.0,
            expectedMoveQ75Points: 14.0,
            priceCostPoints: 4.7,
            slippageCostPoints: 1.4,
            uncertaintyScore: 0.34,
            uncertaintyPenaltyPoints: 1.02,
            riskPenaltyPoints: 0.38,
            expectedGrossEdgePoints: 1.85,
            edgeAfterCostsPoints: -5.65,
            finalAction: "SKIP",
            abstain: true,
            reasons: ["INPUT_STALE", "EDGE_TOO_SMALL"]
        )

        let tsv = try XCTUnwrap(ProbabilityCalibrationTools.runtimeStateTSV(symbol: "EUR/USD live", state: state))
        XCTAssertTrue(tsv.hasSuffix("\r\n"))
        XCTAssertTrue(tsv.contains("schema_version\t1\r\n"))
        XCTAssertTrue(tsv.contains("symbol\tEUR/USD live\r\n"))
        XCTAssertTrue(tsv.contains("selected_quality\t0.800000\r\n"))
        XCTAssertTrue(tsv.contains("spread_cost_points\t4.700000\r\n"))
        XCTAssertTrue(tsv.contains("abstain\t1\r\n"))
        XCTAssertTrue(tsv.contains("reasons_csv\tINPUT_STALE; EDGE_TOO_SMALL\r\n"))

        let parsed = try XCTUnwrap(ProbabilityCalibrationTools.readPairState(symbol: "EUR/USD live", stateTSV: tsv, nowUTC: 1_704_067_201))
        XCTAssertEqual(parsed.symbol, "EUR/USD live")
        XCTAssertEqual(parsed.priceCostPoints, 4.7, accuracy: 0.0)
        XCTAssertEqual(parsed.reasonsCSV, "INPUT_STALE; EDGE_TOO_SMALL")

        let line = try XCTUnwrap(ProbabilityCalibrationTools.runtimeHistoryNDJSONLine(symbol: "EUR/USD live", state: state))
        let data = try XCTUnwrap(line.data(using: .utf8))
        let object = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
        XCTAssertEqual(object["schema_version"] as? Int, 1)
        XCTAssertEqual(object["generated_at"] as? String, "2024-01-01T00:00:00Z")
        XCTAssertEqual(object["symbol"] as? String, "EUR/USD live")
        let nested = try XCTUnwrap(object["state"] as? [String: Any])
        XCTAssertEqual(nested["final_action"] as? String, "SKIP")
        XCTAssertEqual(nested["abstain"] as? Bool, true)
        XCTAssertEqual(try XCTUnwrap(nested["spread_cost_points"] as? Double), 4.7, accuracy: 0.0)
        XCTAssertEqual(nested["reason_codes"] as? [String], ["INPUT_STALE", "EDGE_TOO_SMALL"])
    }

    func testProbabilityCalibrationRepositoryWritesStateAndAppendsHistory() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("ProbabilityCalibrationTests-\(UUID().uuidString)", isDirectory: true)
        let repository = RuntimeArtifactFileRepository(rootURL: root)
        var state = ProbabilityCalibrationRuntimeState(
            ready: true,
            generatedAt: 1_704_067_200,
            method: "LOGISTIC_AFFINE_V2",
            finalAction: "BUY",
            reasons: ["EDGE_TOO_SMALL"]
        )

        try repository.writeProbabilityCalibrationRuntimeArtifacts(symbol: "EUR/USD live", state: state)
        state.generatedAt = 1_704_067_260
        state.finalAction = "SKIP"
        try repository.writeProbabilityCalibrationRuntimeArtifacts(symbol: "EUR/USD live", state: state)

        let statePath = ProbabilityCalibrationTools.runtimeStatePath(symbol: "EUR/USD live")
        let historyPath = ProbabilityCalibrationTools.runtimeHistoryPath(symbol: "EUR/USD live")
        let stateText = try String(contentsOf: root.appendingPathComponent(statePath), encoding: .utf8)
        let historyText = try String(contentsOf: root.appendingPathComponent(historyPath), encoding: .utf8)

        XCTAssertTrue(stateText.contains("generated_at\t1704067260\r\n"))
        XCTAssertTrue(stateText.contains("final_action\tSKIP\r\n"))
        XCTAssertEqual(historyText.components(separatedBy: .newlines).filter { !$0.isEmpty }.count, 2)
        XCTAssertTrue(historyText.contains("\"generated_at\":\"2024-01-01T00:00:00Z\""))
        XCTAssertTrue(historyText.contains("\"generated_at\":\"2024-01-01T00:01:00Z\""))
    }
}
