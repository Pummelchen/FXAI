import Foundation
import XCTest
@testable import FXDataEngine

final class DynamicEnsembleRuntimeTests: XCTestCase {
    func testDynamicEnsemblePathsLabelsAndConfigParsingUsePriceCostKeys() {
        XCTAssertEqual(
            DynamicEnsembleRuntimeTools.runtimeStatePath(symbol: "EUR/USD live"),
            "FXAI/Runtime/fxai_dynamic_ensemble_EUR_USD_live.tsv"
        )
        XCTAssertEqual(
            DynamicEnsembleRuntimeTools.runtimeHistoryPath(symbol: "EUR/USD live"),
            "FXAI/Runtime/fxai_dynamic_ensemble_history_EUR_USD_live.ndjson"
        )
        XCTAssertEqual(DynamicEnsembleRuntimeConstants.configPath, "FXAI/Runtime/dynamic_ensemble_config.tsv")
        XCTAssertEqual(DynamicEnsembleRuntimeTools.actionLabel(1), "BUY")
        XCTAssertEqual(DynamicEnsembleRuntimeTools.actionLabel(0), "SELL")
        XCTAssertEqual(DynamicEnsembleRuntimeTools.actionLabel(-1), "SKIP")
        XCTAssertEqual(DynamicEnsembleRuntimeTools.actionCode("BUY"), 1)
        XCTAssertEqual(DynamicEnsembleRuntimeTools.familySlot("transformer"), .transformer)
        XCTAssertEqual(DynamicEnsembleRuntimeTools.familySlot("missing"), .other)

        let defaults = DynamicEnsembleConfig.defaults
        XCTAssertEqual(defaults.familyNewsCompatibility[.linear], 0.82, accuracy: 0.0)
        XCTAssertEqual(defaults.familyCostRobustness[.transformer], 0.82, accuracy: 0.0)
        XCTAssertEqual(defaults.familyConfidenceCap[.ruleBased], 0.78, accuracy: 0.0)

        let parsed = DynamicEnsembleConfig.parse(tsv: """
        enabled\t0
        fallback_to_routed_ensemble\t0
        threshold_suppress_trust_threshold\t0.25
        threshold_min_active_plugins\t2
        penalty_price_cost_penalty\t0.31
        weight_adaptive_upweight_gain\t0.07
        family_news_compat_transformer\t1.12
        family_cost_robustness_linear\t1.04
        family_confidence_cap_rule\t0.74
        """)
        XCTAssertFalse(parsed.enabled)
        XCTAssertFalse(parsed.fallbackToRoutedEnsemble)
        XCTAssertEqual(parsed.suppressTrustThreshold, 0.25, accuracy: 0.0)
        XCTAssertEqual(parsed.minActivePlugins, 2)
        XCTAssertEqual(parsed.penaltyCost, 0.31, accuracy: 0.0)
        XCTAssertEqual(parsed.weightAdaptiveUpweightGain, 0.07, accuracy: 0.0)
        XCTAssertEqual(parsed.familyNewsCompatibility[.transformer], 1.12, accuracy: 0.0)
        XCTAssertEqual(parsed.familyCostRobustness[.linear], 1.04, accuracy: 0.0)
        XCTAssertEqual(parsed.familyConfidenceCap[.ruleBased], 0.74, accuracy: 0.0)

        let legacyParsed = DynamicEnsembleConfig.parse(tsv: "penalty_spread_cost_penalty\t0.36\n")
        XCTAssertEqual(legacyParsed.penaltyCost, 0.36, accuracy: 0.0)
    }

    func testDynamicEnsembleEvaluatesPreparedPluginRecords() throws {
        let evaluation = DynamicEnsembleRuntimeTools.evaluate(inputs: makeInputs())

        XCTAssertTrue(evaluation.applied)
        XCTAssertTrue(evaluation.state.ready)
        XCTAssertEqual(evaluation.state.symbol, "EURUSD")
        XCTAssertEqual(evaluation.state.generatedAt, 1_704_114_000)
        XCTAssertEqual(evaluation.state.topRegime, "HIGH_VOL_EVENT")
        XCTAssertEqual(evaluation.state.sessionLabel, "LONDON_NY_OVERLAP")
        XCTAssertEqual(evaluation.state.tradePosture, "CAUTION")
        XCTAssertEqual(evaluation.state.ensembleQuality, 0.3610820881713318, accuracy: 1e-12)
        XCTAssertEqual(evaluation.state.abstainBias, 0.14, accuracy: 1e-12)
        XCTAssertEqual(evaluation.state.agreementScore, 1.0, accuracy: 1e-12)
        XCTAssertEqual(evaluation.state.contextFitScore, 0.5454, accuracy: 1e-12)
        XCTAssertEqual(evaluation.state.dominantPluginShare, 1.0, accuracy: 1e-12)
        XCTAssertEqual(evaluation.state.participatingCount, 1)
        XCTAssertEqual(evaluation.state.downweightedCount, 0)
        XCTAssertEqual(evaluation.state.suppressedCount, 3)
        XCTAssertEqual(evaluation.state.buySupport, 1.0, accuracy: 1e-12)
        XCTAssertEqual(evaluation.state.sellSupport, 0.0, accuracy: 1e-12)
        XCTAssertEqual(evaluation.state.skipSupport, 0.0, accuracy: 1e-12)
        XCTAssertEqual(evaluation.state.buyProbability, 0.6093597436590923, accuracy: 1e-12)
        XCTAssertEqual(evaluation.state.sellProbability, 0.15669250551233802, accuracy: 1e-12)
        XCTAssertEqual(evaluation.state.skipProbability, 0.23394775082856978, accuracy: 1e-12)
        XCTAssertEqual(evaluation.state.finalScore, 0.45266723814675425, accuracy: 1e-12)
        XCTAssertEqual(evaluation.state.finalAction, 1)
        XCTAssertEqual(
            evaluation.state.reasonsCSV,
            "ensemble_quality_caution; strong_plugin_agreement; plugin_concentration_elevated; newspulse_caution_active"
        )

        XCTAssertEqual(evaluation.records[0].status, .active)
        XCTAssertEqual(evaluation.records[0].trustScore, 0.8064834838676848, accuracy: 1e-12)
        XCTAssertEqual(evaluation.records[0].normalizedWeight, 1.0, accuracy: 1e-12)
        XCTAssertEqual(evaluation.records[0].calibrationShrink, 0.7595999999999999, accuracy: 1e-12)
        XCTAssertEqual(evaluation.records[0].reasonsCSV, "adaptive_router_upweighted; newspulse_caution_context")
        XCTAssertEqual(evaluation.records[2].status, .suppressed)
        XCTAssertEqual(evaluation.records[3].status, .suppressed)
        XCTAssertEqual(evaluation.records[3].reasonsCSV, "suppressed_by_adaptive_router")
    }

    func testDynamicEnsembleFailsClosedWhenUnavailable() {
        let evaluation = DynamicEnsembleRuntimeTools.evaluate(inputs: DynamicEnsembleInputs(
            symbol: "EURUSD",
            generatedAt: 1_704_114_000,
            records: [],
            dynamicEnsembleEnabled: false
        ))
        XCTAssertFalse(evaluation.applied)
        XCTAssertFalse(evaluation.state.ready)
        XCTAssertTrue(evaluation.state.fallbackUsed)
    }

    func testDynamicEnsembleRuntimeArtifactsMatchLegacyShape() throws {
        let evaluation = DynamicEnsembleRuntimeTools.evaluate(inputs: makeInputs())
        let tsv = try XCTUnwrap(DynamicEnsembleRuntimeTools.runtimeStateTSV(
            symbol: "EUR/USD live",
            state: evaluation.state,
            records: evaluation.records,
            finalDecision: 1
        ))
        XCTAssertTrue(tsv.hasSuffix("\r\n"))
        XCTAssertTrue(tsv.contains("schema_version\t1\r\n"))
        XCTAssertTrue(tsv.contains("symbol\tEUR/USD live\r\n"))
        XCTAssertTrue(tsv.contains("top_regime\tHIGH_VOL_EVENT\r\n"))
        XCTAssertTrue(tsv.contains("final_action\tBUY\r\n"))
        XCTAssertTrue(tsv.contains("active_plugins_csv\tai_tft:"))
        XCTAssertTrue(tsv.contains("suppressed_plugins_csv\t"))

        let parsed = try XCTUnwrap(DynamicEnsembleRuntimeTools.readRuntimeState(symbol: "EUR/USD live", stateTSV: tsv))
        XCTAssertEqual(parsed.finalAction, 1)
        XCTAssertEqual(parsed.topRegime, "HIGH_VOL_EVENT")
        XCTAssertEqual(parsed.reasonsCSV, evaluation.state.reasonsCSV)

        let line = try XCTUnwrap(DynamicEnsembleRuntimeTools.runtimeHistoryNDJSONLine(
            symbol: "EUR/USD live",
            state: evaluation.state,
            records: evaluation.records,
            finalDecision: 1
        ))
        let object = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(line.utf8)) as? [String: Any])
        XCTAssertEqual(object["schema_version"] as? Int, 1)
        XCTAssertEqual(object["generated_at"] as? String, "2024-01-01T13:00:00Z")
        let ensemble = try XCTUnwrap(object["ensemble"] as? [String: Any])
        XCTAssertEqual(ensemble["final_action"] as? String, "BUY")
        XCTAssertEqual(ensemble["top_regime"] as? String, "HIGH_VOL_EVENT")
        let plugins = try XCTUnwrap(object["plugins"] as? [[String: Any]])
        XCTAssertEqual(plugins.count, 4)
        XCTAssertEqual(plugins[3]["status"] as? String, "SUPPRESSED")
    }

    func testDynamicEnsembleRepositoryWritesStateAndAppendsHistory() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("DynamicEnsembleRuntimeTests-\(UUID().uuidString)", isDirectory: true)
        let repository = RuntimeArtifactFileRepository(rootURL: root)
        let evaluation = DynamicEnsembleRuntimeTools.evaluate(inputs: makeInputs())

        try repository.writeDynamicEnsembleRuntimeArtifacts(
            symbol: "EUR/USD live",
            state: evaluation.state,
            records: evaluation.records,
            finalDecision: 1
        )
        var secondState = evaluation.state
        secondState.generatedAt = 1_704_114_060
        secondState.tradePosture = "BLOCK"
        try repository.writeDynamicEnsembleRuntimeArtifacts(
            symbol: "EUR/USD live",
            state: secondState,
            records: evaluation.records,
            finalDecision: -1
        )

        let stateText = try String(
            contentsOf: root.appendingPathComponent(DynamicEnsembleRuntimeTools.runtimeStatePath(symbol: "EUR/USD live")),
            encoding: .utf8
        )
        let historyText = try String(
            contentsOf: root.appendingPathComponent(DynamicEnsembleRuntimeTools.runtimeHistoryPath(symbol: "EUR/USD live")),
            encoding: .utf8
        )

        XCTAssertTrue(stateText.contains("generated_at\t1704114060\r\n"))
        XCTAssertTrue(stateText.contains("trade_posture\tBLOCK\r\n"))
        XCTAssertTrue(stateText.contains("final_action\tSKIP\r\n"))
        XCTAssertEqual(historyText.components(separatedBy: .newlines).filter { !$0.isEmpty }.count, 2)
        XCTAssertTrue(historyText.contains("\"generated_at\":\"2024-01-01T13:00:00Z\""))
        XCTAssertTrue(historyText.contains("\"generated_at\":\"2024-01-01T13:01:00Z\""))
    }

    private func makeInputs() -> DynamicEnsembleInputs {
        DynamicEnsembleInputs(
            symbol: "EURUSD",
            generatedAt: 1_704_114_000,
            priceCostPoints: 3.0,
            minMovePoints: 12.0,
            driftNorm: 0.22,
            regimeState: AdaptiveRegimeState(
                valid: true,
                symbol: "EURUSD",
                generatedAt: 1_704_114_000,
                topLabel: "HIGH_VOL_EVENT",
                confidence: 0.32,
                probabilities: [0.05, 0.06, 0.12, 0.52, 0.12, 0.08, 0.05],
                sessionLabel: "LONDON_NY_OVERLAP",
                priceCostRegime: "ELEVATED",
                volatilityRegime: "HIGH",
                newsRiskScore: 0.68,
                newsPressure: -0.20,
                eventETAMinutes: 20,
                staleNews: false
            ),
            newsState: NewsPulsePairState(
                ready: true,
                available: true,
                stale: false,
                generatedAt: 1_704_113_900,
                eventETAMinutes: 20,
                newsRiskScore: 0.68,
                newsPressure: -0.20,
                tradeGate: "CAUTION",
                sessionProfile: "LONDON_NY_OVERLAP"
            ),
            ratesState: RatesEnginePairState(
                ready: true,
                available: true,
                stale: false,
                generatedAt: 1_704_113_900,
                ratesRiskScore: 0.42,
                tradeGate: "OPEN"
            ),
            crossAssetState: CrossAssetPairState(
                ready: true,
                available: true,
                stale: false,
                generatedAt: 1_704_113_900,
                usdLiquidityStressScore: 0.22,
                pairCrossAssetRiskScore: 0.31,
                tradeGate: "OPEN"
            ),
            microstructureState: MicrostructurePairState(
                ready: true,
                available: true,
                stale: false,
                generatedAt: 1_704_113_900,
                liquidityStressScore: 0.24,
                hostileExecutionScore: 0.20,
                tradeGate: "OPEN"
            ),
            records: makeRecords()
        )
    }

    private func makeRecords() -> [DynamicEnsemblePluginRecord] {
        [
            DynamicEnsemblePluginRecord(
                ready: true,
                aiIndex: 4,
                aiName: "ai_tft",
                family: .transformer,
                signal: 1,
                buyProbability: 0.70,
                sellProbability: 0.18,
                skipProbability: 0.12,
                confidence: 0.78,
                reliability: 0.66,
                baseMetaWeight: 1.20,
                adaptiveStatus: .upweighted,
                contextEdgeNorm: 0.55,
                contextRegret: 0.10,
                globalEdgeNorm: 0.30,
                portfolioEdgeNorm: 0.35,
                portfolioStability: 0.70,
                portfolioCorrelation: 0.20,
                portfolioDiversity: 0.65,
                contextTrust: 0.62
            ),
            DynamicEnsemblePluginRecord(
                ready: true,
                aiIndex: 1,
                aiName: "lin_pa",
                family: .linear,
                signal: 0,
                buyProbability: 0.34,
                sellProbability: 0.50,
                skipProbability: 0.16,
                confidence: 0.54,
                reliability: 0.62,
                baseMetaWeight: 0.75,
                adaptiveStatus: .active,
                contextEdgeNorm: 0.18,
                contextRegret: 0.18,
                globalEdgeNorm: 0.12,
                portfolioEdgeNorm: 0.10,
                portfolioStability: 0.48,
                portfolioCorrelation: 0.25,
                portfolioDiversity: 0.35,
                contextTrust: 0.40
            ),
            DynamicEnsemblePluginRecord(
                ready: true,
                aiIndex: 20,
                aiName: "rule_guard",
                family: .ruleBased,
                signal: -1,
                buyProbability: 0.22,
                sellProbability: 0.18,
                skipProbability: 0.60,
                confidence: 0.35,
                reliability: 0.44,
                baseMetaWeight: 0.25,
                adaptiveStatus: .downweighted,
                contextEdgeNorm: 0.05,
                contextRegret: 0.24,
                globalEdgeNorm: 0.02,
                portfolioEdgeNorm: 0.03,
                portfolioStability: 0.60,
                portfolioCorrelation: 0.15,
                portfolioDiversity: 0.30,
                contextTrust: 0.35
            ),
            DynamicEnsemblePluginRecord(
                ready: true,
                aiIndex: 8,
                aiName: "suppressed_plugin",
                family: .tree,
                signal: 1,
                buyProbability: 0.64,
                sellProbability: 0.20,
                skipProbability: 0.16,
                confidence: 0.70,
                reliability: 0.40,
                baseMetaWeight: 0.90,
                adaptiveStatus: .suppressed
            )
        ]
    }
}
