import Foundation
import XCTest
@testable import FXDataEngine

final class AdaptiveRouterRuntimeTests: XCTestCase {
    func testAdaptiveRouterRuntimePathsAndSessionLabelsMatchLegacyRules() {
        XCTAssertEqual(
            AdaptiveRouterRuntimeTools.runtimeStatePath(symbol: "EUR/USD live"),
            "FXAI/Runtime/fxai_regime_router_EUR_USD_live.tsv"
        )
        XCTAssertEqual(
            AdaptiveRouterRuntimeTools.runtimeHistoryPath(symbol: "EUR/USD live"),
            "FXAI/Runtime/fxai_regime_router_history_EUR_USD_live.ndjson"
        )
        XCTAssertEqual(AdaptiveRouterRuntimeTools.regimeLabel(0), "TREND_PERSISTENT")
        XCTAssertEqual(AdaptiveRouterRuntimeTools.regimeLabel(6), "SESSION_FLOW")
        XCTAssertEqual(AdaptiveRouterRuntimeTools.regimeLabel(99), "TREND_PERSISTENT")
        XCTAssertEqual(AdaptiveRouterRuntimeTools.sessionLabel(sampleTimeUTC: 1_704_067_200), "ROLLOVER")
        XCTAssertEqual(AdaptiveRouterRuntimeTools.sessionLabel(sampleTimeUTC: 1_704_078_000), "ASIA")
        XCTAssertEqual(AdaptiveRouterRuntimeTools.sessionLabel(sampleTimeUTC: 1_704_099_600), "LONDON")
        XCTAssertEqual(AdaptiveRouterRuntimeTools.sessionLabel(sampleTimeUTC: 1_704_114_000), "LONDON_NY_OVERLAP")
        XCTAssertEqual(AdaptiveRouterRuntimeTools.sessionLabel(sampleTimeUTC: 1_704_128_400), "NEWYORK")
    }

    func testAdaptiveRouterBuildsRegimeStateFromM1OHLCVAndContext() throws {
        let series = try makeSeries()
        let state = AdaptiveRouterRuntimeTools.buildRegimeState(inputs: AdaptiveRouterRegimeInputs(
            symbol: "EURUSD",
            series: series,
            index: 19,
            pointValue: 1.0,
            priceCostPoints: 3.0,
            priceCostReferencePoints: 1.8,
            volatilityProxyAbs: 2.8,
            volatilityReference: 1.6,
            minMovePoints: 3.0,
            contextStrength: 1.4,
            contextQuality: 0.70,
            regimeGraph: RegimeGraphQuery(
                persistence: 0.70,
                transitionConfidence: 0.55,
                instability: 0.35,
                edgeBias: 0.40,
                qualityBias: 0.50,
                macroAlignment: 0.60,
                predictedRegime: 2
            ),
            newsState: NewsPulsePairState(
                ready: true,
                available: true,
                stale: false,
                generatedAt: 1_704_156_900,
                eventETAMinutes: 25,
                newsRiskScore: 0.76,
                newsPressure: -0.45,
                tradeGate: "CAUTION",
                sessionProfile: "LONDON_NY_OVERLAP"
            ),
            crossAssetState: CrossAssetPairState(
                ready: true,
                available: true,
                stale: false,
                generatedAt: 1_704_156_900,
                riskOffScore: 0.55,
                usdLiquidityStressScore: 0.30,
                crossAssetDislocationScore: 0.25,
                pairCrossAssetRiskScore: 0.42,
                macroState: "RISK_OFF",
                tradeGate: "CAUTION"
            ),
            microstructureState: MicrostructurePairState(
                ready: true,
                available: true,
                stale: false,
                generatedAt: 1_704_156_900,
                tickImbalance30s: -0.40,
                directionalEfficiency60s: 0.65,
                spreadZscore60s: 2.0,
                volBurstScore5m: 1.7,
                localExtremaBreachScore60s: 0.62,
                breakoutReversalScore60s: 0.30,
                exhaustionProxy60s: 0.20,
                liquidityStressScore: 0.35,
                hostileExecutionScore: 0.25,
                sessionTag: "LONDON_NY_OVERLAP",
                sessionOpenBurstScore: 0.55,
                sessionSpreadBehaviorScore: 0.40,
                tradeGate: "OPEN"
            )
        ))

        XCTAssertTrue(state.valid)
        XCTAssertEqual(state.symbol, "EURUSD")
        XCTAssertEqual(state.generatedAt, 1_704_114_000)
        XCTAssertEqual(state.sessionLabel, "LONDON_NY_OVERLAP")
        XCTAssertEqual(state.priceCostRegime, "ELEVATED")
        XCTAssertEqual(state.volatilityRegime, "HIGH")
        XCTAssertEqual(state.topLabel, "HIGH_VOL_EVENT")
        XCTAssertEqual(state.confidence, 0.13077461436881324, accuracy: 1e-12)
        XCTAssertEqual(state.probabilities.reduce(0.0, +), 1.0, accuracy: 1e-12)
        XCTAssertEqual(state.probabilities[3], 0.20440715297486603, accuracy: 1e-12)
        XCTAssertEqual(state.breakoutPressure, 0.6557, accuracy: 1e-12)
        XCTAssertEqual(state.trendStrength, 0.7849999999999999, accuracy: 1e-12)
        XCTAssertEqual(state.rangePressure, 0.06575900000000001, accuracy: 1e-12)
        XCTAssertEqual(state.macroPressure, 0.5376, accuracy: 1e-12)
        XCTAssertEqual(state.liquidityStress, 0.41830000000000006, accuracy: 1e-12)
        XCTAssertEqual(
            state.reasonsCSV,
            "NewsPulse event window active; Spread regime elevated; Volatility expansion detected; Directional persistence elevated; Cross-asset macro regime active"
        )
    }

    func testAdaptiveRouterSuitabilityStatusAndPostureMatchLegacyRules() throws {
        let profile = AdaptiveRouterProfile.parse(symbol: "EURUSD", tsv: """
        enabled\t1
        symbol\tEURUSD
        router_mode\tWEIGHTED_ENSEMBLE
        caution_threshold\t0.55
        abstain_threshold\t0.35
        block_threshold\t0.16
        confidence_floor\t0.12
        suppression_threshold\t0.34
        downweight_threshold\t0.78
        stale_news_abstain_bias\t0.24
        stale_news_force_caution\t1
        min_plugin_weight\t0.05
        max_plugin_weight\t1.80
        plugin_global_weights_csv\tai_tft=1.20,lin_pa=0.40
        plugin_news_compatibility_csv\tai_tft=0.80,lin_pa=0.70
        plugin_liquidity_robustness_csv\tai_tft=0.90,lin_pa=0.60
        plugin_regime_HIGH_VOL_EVENT_csv\tai_tft=1.50,lin_pa=0.30
        plugin_session_LONDON_NY_OVERLAP_csv\tai_tft=1.10,lin_pa=0.80
        """)
        let state = AdaptiveRegimeState(
            valid: true,
            symbol: "EURUSD",
            generatedAt: 1_704_158_340,
            topLabel: "HIGH_VOL_EVENT",
            confidence: 0.20,
            probabilities: [0.05, 0.05, 0.10, 0.55, 0.10, 0.10, 0.05],
            sessionLabel: "LONDON_NY_OVERLAP",
            newsRiskScore: 0.76,
            newsPressure: -0.20,
            staleNews: false,
            liquidityStress: 0.42
        )

        let tftSuitability = AdaptiveRouterRuntimeTools.pluginSuitability(
            profile: profile,
            state: state,
            pluginName: "ai_tft"
        )
        let linearSuitability = AdaptiveRouterRuntimeTools.pluginSuitability(
            profile: profile,
            state: state,
            pluginName: "lin_pa"
        )

        XCTAssertEqual(tftSuitability, 1.5313953600000008, accuracy: 1e-12)
        XCTAssertEqual(linearSuitability, 0.15616473600000003, accuracy: 1e-12)
        XCTAssertEqual(
            AdaptiveRouterRuntimeTools.suitabilityStatus(profile: profile, suitability: tftSuitability),
            .upweighted
        )
        XCTAssertEqual(
            AdaptiveRouterRuntimeTools.suitabilityStatus(profile: profile, suitability: linearSuitability),
            .suppressed
        )

        let posture = AdaptiveRouterRuntimeTools.computePosture(
            profile: profile,
            state: state,
            bestSuitability: tftSuitability,
            eligibleCount: 1
        )
        XCTAssertEqual(posture, "CAUTION")
        XCTAssertEqual(
            AdaptiveRouterRuntimeTools.postureAbstainBias(profile: profile, state: state, posture: posture),
            0.12,
            accuracy: 0.0
        )

        var staleState = state
        staleState.staleNews = true
        XCTAssertEqual(
            AdaptiveRouterRuntimeTools.computePosture(
                profile: profile,
                state: staleState,
                bestSuitability: tftSuitability,
                eligibleCount: 1
            ),
            "CAUTION"
        )
        XCTAssertEqual(
            AdaptiveRouterRuntimeTools.postureAbstainBias(profile: profile, state: staleState, posture: "CAUTION"),
            0.36,
            accuracy: 0.0
        )
    }

    func testAdaptiveRouterRuntimeArtifactsMatchLegacyShape() throws {
        let state = AdaptiveRegimeState(
            valid: true,
            symbol: "EURUSD",
            generatedAt: 1_704_067_200,
            topLabel: "HIGH_VOL_EVENT",
            confidence: 0.62,
            probabilities: [0.05, 0.08, 0.12, 0.55, 0.10, 0.07, 0.05],
            sessionLabel: "LONDON_NY_OVERLAP",
            priceCostRegime: "ELEVATED",
            volatilityRegime: "HIGH",
            newsRiskScore: 0.76,
            newsPressure: -0.25,
            eventETAMinutes: 25,
            staleNews: false,
            liquidityStress: 0.42,
            breakoutPressure: 0.51,
            trendStrength: 0.49,
            rangePressure: 0.50,
            macroPressure: 0.53,
            reasons: ["NewsPulse event window active", "Spread regime elevated"]
        )
        let profile = AdaptiveRouterProfile.parse(symbol: "EURUSD", tsv: """
        enabled\t1
        router_mode\tWEIGHTED_ENSEMBLE
        """)
        let routes = [
            AdaptiveRouterPluginRoute(
                name: "ai_tft",
                eligible: true,
                routedWeight: 0.70,
                suitability: 1.20,
                status: .upweighted,
                reasons: ["Upweighted by strong regime fit"]
            ),
            AdaptiveRouterPluginRoute(
                name: "lin_pa",
                eligible: false,
                routedWeight: 0.10,
                suitability: 0.20,
                status: .suppressed
            ),
            AdaptiveRouterPluginRoute(
                name: "idle",
                eligible: false,
                routedWeight: 0.0,
                suitability: 0.0,
                status: .active
            )
        ]

        let tsv = try XCTUnwrap(AdaptiveRouterRuntimeTools.runtimeStateTSV(
            symbol: "EUR/USD live",
            state: state,
            posture: "CAUTION",
            abstainBias: 0.12,
            routes: routes
        ))
        XCTAssertTrue(tsv.hasSuffix("\r\n"))
        XCTAssertTrue(tsv.contains("schema_version\t1\r\n"))
        XCTAssertTrue(tsv.contains("symbol\tEUR/USD live\r\n"))
        XCTAssertTrue(tsv.contains("top_regime_label\tHIGH_VOL_EVENT\r\n"))
        XCTAssertTrue(tsv.contains("spread_regime\tELEVATED\r\n"))
        XCTAssertTrue(tsv.contains("active_plugins_csv\tai_tft:1.0000:1.2000\r\n"))
        XCTAssertTrue(tsv.contains("suppressed_plugins_csv\tlin_pa:0.0000:0.2000\r\n"))
        XCTAssertTrue(tsv.contains("probabilities_csv\tTREND_PERSISTENT=0.050000"))

        let parsed = try XCTUnwrap(AdaptiveRouterRuntimeTools.readRegimeState(symbol: "EUR/USD live", stateTSV: tsv))
        XCTAssertEqual(parsed.topLabel, "HIGH_VOL_EVENT")
        XCTAssertEqual(parsed.priceCostRegime, "ELEVATED")
        XCTAssertEqual(parsed.reasonsCSV, "NewsPulse event window active; Spread regime elevated")
        XCTAssertEqual(parsed.probabilities[3], 0.55, accuracy: 0.0)

        let line = try XCTUnwrap(AdaptiveRouterRuntimeTools.runtimeHistoryNDJSONLine(
            symbol: "EUR/USD live",
            profile: profile,
            state: state,
            posture: "CAUTION",
            abstainBias: 0.12,
            routes: routes
        ))
        let object = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(line.utf8)) as? [String: Any])
        XCTAssertEqual(object["schema_version"] as? Int, 1)
        XCTAssertEqual(object["generated_at"] as? String, "2024-01-01T00:00:00Z")
        let regime = try XCTUnwrap(object["regime"] as? [String: Any])
        XCTAssertEqual(regime["top_label"] as? String, "HIGH_VOL_EVENT")
        XCTAssertEqual(regime["spread_regime"] as? String, "ELEVATED")
        let router = try XCTUnwrap(object["router"] as? [String: Any])
        XCTAssertEqual(router["trade_posture"] as? String, "CAUTION")
        let plugins = try XCTUnwrap(object["plugins"] as? [[String: Any]])
        XCTAssertEqual(plugins.count, 2)
        XCTAssertEqual(plugins[0]["status"] as? String, "UPWEIGHTED")
        XCTAssertEqual(plugins[1]["reasons"] as? [String], ["Suppressed in event regime"])
    }

    func testAdaptiveRouterRepositoryWritesStateAndAppendsHistory() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("AdaptiveRouterRuntimeTests-\(UUID().uuidString)", isDirectory: true)
        let repository = RuntimeArtifactFileRepository(rootURL: root)
        let profile = AdaptiveRouterProfile.parse(symbol: "EURUSD", tsv: "enabled\t1\n")
        var state = AdaptiveRegimeState(
            valid: true,
            symbol: "EURUSD",
            generatedAt: 1_704_067_200,
            topLabel: "HIGH_VOL_EVENT",
            confidence: 0.62,
            probabilities: [0.05, 0.08, 0.12, 0.55, 0.10, 0.07, 0.05],
            reasons: ["NewsPulse event window active"]
        )

        try repository.writeAdaptiveRouterRuntimeArtifacts(
            symbol: "EUR/USD live",
            profile: profile,
            state: state,
            posture: "CAUTION",
            abstainBias: 0.12
        )
        state.generatedAt = 1_704_067_260
        state.topLabel = "LIQUIDITY_STRESS"
        try repository.writeAdaptiveRouterRuntimeArtifacts(
            symbol: "EUR/USD live",
            profile: profile,
            state: state,
            posture: "BLOCK",
            abstainBias: 0.92
        )

        let stateText = try String(
            contentsOf: root.appendingPathComponent(AdaptiveRouterRuntimeTools.runtimeStatePath(symbol: "EUR/USD live")),
            encoding: .utf8
        )
        let historyText = try String(
            contentsOf: root.appendingPathComponent(AdaptiveRouterRuntimeTools.runtimeHistoryPath(symbol: "EUR/USD live")),
            encoding: .utf8
        )

        XCTAssertTrue(stateText.contains("generated_at\t1704067260\r\n"))
        XCTAssertTrue(stateText.contains("top_regime_label\tLIQUIDITY_STRESS\r\n"))
        XCTAssertTrue(stateText.contains("trade_posture\tBLOCK\r\n"))
        XCTAssertEqual(historyText.components(separatedBy: .newlines).filter { !$0.isEmpty }.count, 2)
        XCTAssertTrue(historyText.contains("\"generated_at\":\"2024-01-01T00:00:00Z\""))
        XCTAssertTrue(historyText.contains("\"generated_at\":\"2024-01-01T00:01:00Z\""))
    }

    private func makeSeries() throws -> M1OHLCVSeries {
        let start: Int64 = 1_704_112_860
        let closes: [Int64] = [
            100_000, 100_012, 100_025, 100_038, 100_052,
            100_046, 100_060, 100_078, 100_096, 100_088,
            100_106, 100_124, 100_144, 100_166, 100_190,
            100_214, 100_238, 100_264, 100_292, 100_320
        ]
        var timestamps: [Int64] = []
        var open: [Int64] = []
        var high: [Int64] = []
        var low: [Int64] = []
        for (index, close) in closes.enumerated() {
            timestamps.append(start + Int64(index * 60))
            let openValue = index == 0 ? close - 4 : closes[index - 1]
            open.append(openValue)
            high.append(max(openValue, close) + 10)
            low.append(min(openValue, close) - 10)
        }
        return try M1OHLCVSeries(
            metadata: FXMarketMetadata(
                brokerSourceId: "fixture",
                sourceOrigin: "fixture",
                logicalSymbol: "EURUSD",
                timeframe: .m1,
                digits: 0
            ),
            utcTimestamps: ContiguousArray(timestamps),
            open: ContiguousArray(open),
            high: ContiguousArray(high),
            low: ContiguousArray(low),
            close: ContiguousArray(closes),
            volume: ContiguousArray(Array(repeating: UInt64(1_000), count: closes.count))
        )
    }
}
