import Foundation
import XCTest
@testable import FXDataEngine

final class PairNetworkTests: XCTestCase {
    func testPairNetworkDefaultsAndPathsMatchLegacyShape() {
        let config = PairNetworkConfig()

        XCTAssertTrue(config.ready)
        XCTAssertTrue(config.enabled)
        XCTAssertTrue(config.fallbackStructuralOnly)
        XCTAssertTrue(config.autoApply)
        XCTAssertTrue(config.graphStale)
        XCTAssertEqual(config.graphMode, "STRUCTURAL_ONLY")
        XCTAssertEqual(config.graphStaleAfterSeconds, 43_200)
        XCTAssertEqual(config.historyPoints, 192)
        XCTAssertEqual(config.maxEdgesPerPair, 10)
        XCTAssertEqual(config.currencyProfiles.count, 0)
        XCTAssertEqual(PairNetworkTools.runtimeStatePath(symbol: "EUR/USD live"), "FXAI/Runtime/fxai_pair_network_EUR_USD_live.tsv")
        XCTAssertEqual(PairNetworkTools.runtimeHistoryPath(symbol: "EUR/USD live"), "FXAI/Runtime/fxai_pair_network_history_EUR_USD_live.ndjson")
        XCTAssertEqual(PairNetworkTools.decisionLabel(.preferAlternative), "PREFER_ALTERNATIVE_EXPRESSION")
        XCTAssertEqual(PairNetworkTools.factorName(3), "commodity_fx")
        XCTAssertEqual(PairNetworkTools.factorName(99), "unknown")
    }

    func testPairNetworkConfigParserHandlesWeightsProfilesAndStatus() {
        let config = PairNetworkTools.parseConfig(
            configTSV: """
            enabled\t0
            graph_stale_after_sec\t3600
            history_points\t64
            max_edges_per_pair\t4
            fallback_structural_only\t0
            min_empirical_overlap\t32
            empirical_lookback_bars\t256
            structural_weight\t0.6
            empirical_weight\t0.4
            redundancy_threshold\t0.7
            contradiction_threshold\t0.8
            concentration_reduce_threshold\t0.5
            concentration_block_threshold\t0.9
            execution_overlap_threshold\t0.55
            reduced_size_multiplier_floor\t0.35
            preferred_expression_margin\t0.05
            min_incremental_edge_score\t0.2
            action_mode\tRECOMMEND_ONLY
            selection_weight\tedge_after_costs\t0.31
            selection_weight\texecution_quality\t0.21
            selection_weight\tcalibration_quality\t0.17
            selection_weight\tportfolio_fit\t0.13
            selection_weight\tdiversification\t0.11
            selection_weight\tmacro_fit\t0.07
            currency_profile\tEUR\teur_rates\t0.9
            currency_profile\tEUR\tmacro_shock\t0.4
            currency_profile\tUSD\tusd_bloc\t1.0
            """,
            statusTSV: """
            graph_mode\tEMPIRICAL_BLEND
            fallback_graph_used\t1
            partial_dependency_data\t1
            graph_stale\t0
            """
        )

        XCTAssertTrue(config.ready)
        XCTAssertFalse(config.enabled)
        XCTAssertFalse(config.fallbackStructuralOnly)
        XCTAssertFalse(config.autoApply)
        XCTAssertFalse(config.graphStale)
        XCTAssertTrue(config.fallbackGraphUsed)
        XCTAssertTrue(config.partialDependencyData)
        XCTAssertEqual(config.graphMode, "EMPIRICAL_BLEND")
        XCTAssertEqual(config.graphStaleAfterSeconds, 3_600)
        XCTAssertEqual(config.historyPoints, 64)
        XCTAssertEqual(config.maxEdgesPerPair, 4)
        XCTAssertEqual(config.minEmpiricalOverlap, 32)
        XCTAssertEqual(config.empiricalLookbackBars, 256)
        XCTAssertEqual(config.structuralWeight, 0.6)
        XCTAssertEqual(config.empiricalWeight, 0.4)
        XCTAssertEqual(config.weightEdgeAfterCosts, 0.31)
        XCTAssertEqual(config.weightExecutionQuality, 0.21)
        XCTAssertEqual(config.weightCalibrationQuality, 0.17)
        XCTAssertEqual(config.weightPortfolioFit, 0.13)
        XCTAssertEqual(config.weightDiversification, 0.11)
        XCTAssertEqual(config.weightMacroFit, 0.07)
        XCTAssertEqual(config.currencyProfiles.count, 2)
        XCTAssertEqual(PairNetworkTools.currencyFactorWeight(config, currency: "EUR", factor: .eurRates), 0.9)
        XCTAssertEqual(PairNetworkTools.currencyFactorWeight(config, currency: "EUR", factor: .macroShock), 0.4)
        XCTAssertEqual(PairNetworkTools.currencyFactorWeight(config, currency: "USD", factor: .usdBloc), 1.0)
    }

    func testPairNetworkBuildSymbolExposureUsesBaseMinusQuoteFactors() {
        let config = PairNetworkTools.parseConfig(configTSV: """
        currency_profile\tEUR\teur_rates\t0.8
        currency_profile\tEUR\tmacro_shock\t0.3
        currency_profile\tUSD\tusd_bloc\t1.0
        currency_profile\tUSD\tmacro_shock\t0.1
        """)

        let buyExposure = PairNetworkTools.buildSymbolExposure(config: config, symbol: "EURUSD", direction: 1, sizeUnits: 2.0)
        let sellExposure = PairNetworkTools.buildSymbolExposure(config: config, symbol: "EURUSD", direction: 0, sizeUnits: 2.0)

        XCTAssertEqual(buyExposure.currencyKeys, ["EUR", "USD"])
        XCTAssertEqual(buyExposure.currencyValues, [2.0, -2.0])
        XCTAssertEqual(buyExposure.factorValues[PairNetworkFactor.eurRates.rawValue], 1.6)
        XCTAssertEqual(buyExposure.factorValues[PairNetworkFactor.usdBloc.rawValue], -2.0)
        XCTAssertEqual(buyExposure.factorValues[PairNetworkFactor.macroShock.rawValue], 0.4, accuracy: 1e-12)
        XCTAssertEqual(sellExposure.currencyValues, [-2.0, 2.0])
        XCTAssertEqual(sellExposure.factorValues[PairNetworkFactor.eurRates.rawValue], -1.6)
    }

    func testPairNetworkStructuralOverlapMatchesLegacyBlend() {
        let config = PairNetworkConfig()

        XCTAssertEqual(
            PairNetworkTools.structuralOverlap(config: config, lhsSymbol: "EURUSD", rhsSymbol: "EURJPY"),
            0.6776,
            accuracy: 1e-12
        )
        XCTAssertEqual(
            PairNetworkTools.structuralOverlap(config: config, lhsSymbol: "EURUSD", rhsSymbol: "GBPUSD"),
            0.7576,
            accuracy: 1e-12
        )
        XCTAssertEqual(
            PairNetworkTools.structuralOverlap(config: config, lhsSymbol: "EURUSD", rhsSymbol: "USDEUR"),
            0.8344,
            accuracy: 1e-12
        )
    }

    func testPairNetworkQualityScoreUsesConfiguredWeightsAndClampsInputs() {
        var config = PairNetworkConfig()
        config.weightEdgeAfterCosts = 0.30
        config.weightExecutionQuality = 0.20
        config.weightCalibrationQuality = 0.15
        config.weightPortfolioFit = 0.15
        config.weightDiversification = 0.10
        config.weightMacroFit = 0.10

        let score = PairNetworkTools.qualityScore(
            config: config,
            edgeScore: 1.2,
            executionQualityScore: 0.7,
            calibrationQuality: 0.8,
            portfolioFit: 0.6,
            macroFit: -0.5,
            overlapScore: 0.25
        )

        XCTAssertEqual(score, 0.785, accuracy: 1e-12)
    }

    func testPairNetworkDecisionStateReasonsMatchLegacyLimitAndCSV() {
        var state = PairNetworkDecisionState(reasons: ["", "A", "A", "B"])

        XCTAssertFalse(state.ready)
        XCTAssertTrue(state.graphStale)
        XCTAssertEqual(state.direction, -1)
        XCTAssertEqual(state.decision, "ALLOW")
        XCTAssertEqual(state.recommendedSizeMultiplier, 1.0)
        XCTAssertEqual(state.reasons, ["A", "B"])

        for index in 1...20 {
            state.appendReason("R\(index)")
        }
        state.appendReason("R1")
        state.appendReason("")

        XCTAssertEqual(state.reasonCount, PairNetworkConstants.maxReasons)
        XCTAssertEqual(state.reasonsCSV, "A; B; R1; R2; R3; R4; R5; R6; R7; R8; R9; R10")
    }

    func testPairNetworkRuntimeArtifactsMatchLegacyTSVAndNDJSONShape() throws {
        let state = PairNetworkDecisionState(
            ready: true,
            fallbackGraphUsed: true,
            partialDependencyData: true,
            graphStale: false,
            generatedAt: 1_704_067_200,
            symbol: "EURUSD",
            direction: 1,
            decision: "ALLOW_REDUCED",
            conflictScore: 0.12,
            redundancyScore: 0.23,
            contradictionScore: 0.34,
            concentrationScore: 0.45,
            currencyConcentration: 0.56,
            factorConcentration: 0.67,
            recommendedSizeMultiplier: 0.78,
            preferredExpression: "GBPUSD",
            currencyExposureCSV: "EUR:1.0000; USD:-1.0000",
            factorExposureCSV: "eur_rates:0.5000",
            reasons: ["PAIR_NETWORK_REDUCE", "GRAPH_FALLBACK"]
        )

        let tsv = PairNetworkTools.runtimeStateTSV(symbol: "EUR/USD live", state: state)
        XCTAssertTrue(tsv.hasSuffix("\r\n"))
        XCTAssertTrue(tsv.contains("symbol\tEUR/USD live\r\n"))
        XCTAssertTrue(tsv.contains("decision\tALLOW_REDUCED\r\n"))
        XCTAssertTrue(tsv.contains("fallback_graph_used\t1\r\n"))
        XCTAssertTrue(tsv.contains("graph_stale\t0\r\n"))
        XCTAssertTrue(tsv.contains("recommended_size_multiplier\t0.780000\r\n"))
        XCTAssertTrue(tsv.contains("currency_exposure_csv\tEUR:1.0000; USD:-1.0000\r\n"))
        XCTAssertTrue(tsv.contains("reasons_csv\tPAIR_NETWORK_REDUCE; GRAPH_FALLBACK\r\n"))

        let line = PairNetworkTools.runtimeHistoryNDJSONLine(symbol: "EUR/USD live", state: state)
        let data = try XCTUnwrap(line.data(using: .utf8))
        let object = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
        XCTAssertEqual(object["generated_at"] as? String, "2024-01-01T00:00:00Z")
        XCTAssertEqual(object["symbol"] as? String, "EUR/USD live")
        XCTAssertEqual(object["decision"] as? String, "ALLOW_REDUCED")
        XCTAssertEqual(object["fallback_graph_used"] as? Int, 1)
        XCTAssertEqual(try XCTUnwrap(object["recommended_size_multiplier"] as? Double), 0.78, accuracy: 0.0)
        XCTAssertEqual(object["reason_codes"] as? [String], ["PAIR_NETWORK_REDUCE", "GRAPH_FALLBACK"])
    }

    func testPairNetworkRepositoryWritesStateAndAppendsHistory() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("PairNetworkTests-\(UUID().uuidString)", isDirectory: true)
        let repository = RuntimeArtifactFileRepository(rootURL: root)
        var state = PairNetworkDecisionState(
            ready: true,
            generatedAt: 1_704_067_200,
            decision: "ALLOW",
            reasons: ["ALLOW"]
        )

        try repository.writePairNetworkRuntimeArtifacts(symbol: "EUR/USD live", state: state)
        state.generatedAt = 1_704_067_260
        state.decision = "BLOCK_CONCENTRATION"
        try repository.writePairNetworkRuntimeArtifacts(symbol: "EUR/USD live", state: state)

        let statePath = PairNetworkTools.runtimeStatePath(symbol: "EUR/USD live")
        let historyPath = PairNetworkTools.runtimeHistoryPath(symbol: "EUR/USD live")
        let stateText = try String(contentsOf: root.appendingPathComponent(statePath), encoding: .utf8)
        let historyText = try String(contentsOf: root.appendingPathComponent(historyPath), encoding: .utf8)

        XCTAssertTrue(stateText.contains("generated_at\t1704067260\r\n"))
        XCTAssertTrue(stateText.contains("decision\tBLOCK_CONCENTRATION\r\n"))
        XCTAssertEqual(historyText.components(separatedBy: .newlines).filter { !$0.isEmpty }.count, 2)
        XCTAssertTrue(historyText.contains("\"generated_at\":\"2024-01-01T00:00:00Z\""))
        XCTAssertTrue(historyText.contains("\"generated_at\":\"2024-01-01T00:01:00Z\""))
    }

    func testPairNetworkConcentrationAndCSVHelpersMatchLegacyMath() {
        let exposure = PairNetworkExposure(
            currencyKeys: ["EUR", "USD", "JPY"],
            currencyValues: [2.0, -1.0, 0.0],
            factorValues: [0.0, 1.5, 0.0, -0.25, 0.0, 0.5, 0.0]
        )
        let other = PairNetworkExposure(currencyKeys: ["USD", "JPY"], currencyValues: [-1.0, 1.0])

        XCTAssertEqual(PairNetworkTools.currencyExposureCSV(exposure), "EUR:2.0000; USD:-1.0000")
        XCTAssertEqual(
            PairNetworkTools.factorExposureCSV(exposure.factorValues),
            "eur_rates:1.5000; commodity_fx:-0.2500; liquidity_stress:0.5000"
        )
        XCTAssertEqual(PairNetworkTools.currencyCosine(exposure, other), 1.0 / sqrt(10.0), accuracy: 1e-12)
        XCTAssertEqual(PairNetworkTools.topShareCurrency([2.0, -1.0]), 2.0 / 3.0, accuracy: 1e-12)
        XCTAssertEqual(PairNetworkTools.herfindahlCurrency([2.0, -1.0]), 5.0 / 9.0, accuracy: 1e-12)
        XCTAssertEqual(PairNetworkTools.topShareFactor(exposure.factorValues), 1.5 / 2.25, accuracy: 1e-12)
        XCTAssertEqual(
            PairNetworkTools.herfindahlFactor(exposure.factorValues),
            (1.5 / 2.25) * (1.5 / 2.25) + (0.25 / 2.25) * (0.25 / 2.25) + (0.5 / 2.25) * (0.5 / 2.25),
            accuracy: 1e-12
        )
    }

    func testPairNetworkMacroFitUsesCrossAssetAndRatesStates() {
        let factors = [0.0, 0.0, 0.7, 0.0, 0.2, 0.5, 0.0]
        let cross = CrossAssetPairState(ready: true, riskState: "RISK_OFF", liquidityState: "STRESSED")
        let rates = RatesEnginePairState(ready: true, tradeGate: "CAUTION")

        XCTAssertEqual(PairNetworkTools.macroFit(candidateFactors: factors, crossState: cross, ratesState: rates), 0.52, accuracy: 1e-12)
        XCTAssertEqual(
            PairNetworkTools.macroFit(candidateFactors: factors, crossState: .reset, ratesState: RatesEnginePairState(ready: true, tradeGate: "BLOCK")),
            0.38,
            accuracy: 1e-12
        )
    }
}
