import XCTest
@testable import FXDataEngine

final class MicrostructureTests: XCTestCase {
    func testMicrostructurePairIDUsesMapThenNewsPulseFallback() {
        let microMap = MicrostructureTools.parseSymbolMap(tsv: """
        symbol\tEURUSD.r\tEURUSD
        symbol\tbad\tEUR
        """)
        let newsMap = NewsPulseTools.parseSymbolMap(tsv: "symbol\tGBPUSD.pro\tGBPUSD\n")

        XCTAssertEqual(microMap, [MicrostructureSymbolMapEntry(symbol: "EURUSD.R", pairID: "EURUSD")])
        XCTAssertEqual(MicrostructureTools.pairID(symbol: "eurusd.r", symbolMap: microMap), "EURUSD")
        XCTAssertEqual(MicrostructureTools.pairID(symbol: "gbpusd.pro", newsPulseSymbolMap: newsMap), "GBPUSD")
        XCTAssertEqual(MicrostructureTools.pairID(symbol: "prefixAUDNZD"), "AUDNZD")
    }

    func testMicrostructureParsesSnapshotClampsAndDeduplicatesReasons() {
        let state = MicrostructureTools.readPairState(
            symbol: "EURUSD",
            snapshotTSV: """
            meta\tglobal\tgenerated_at_unix\t1000
            pair\tEURUSD\ttick_imbalance_30s\t2.0
            pair\tEURUSD\tdirectional_efficiency_60s\t-0.5
            pair\tEURUSD\tprice_cost_current\t-3.0
            pair\tEURUSD\tspread_zscore_60s\t9.0
            pair\tEURUSD\ttick_rate_60s\t-1.0
            pair\tEURUSD\ttick_rate_zscore_60s\t-9.0
            pair\tEURUSD\trealized_vol_5m\t-0.2
            pair\tEURUSD\tvol_burst_score_5m\t9.0
            pair\tEURUSD\tlocal_extrema_breach_score_60s\t1.5
            pair\tEURUSD\tsweep_and_reject_flag_60s\t1
            pair\tEURUSD\tbreakout_reversal_score_60s\t-1.0
            pair\tEURUSD\texhaustion_proxy_60s\t2.0
            pair\tEURUSD\tliquidity_stress_score\t1.2
            pair\tEURUSD\thostile_execution_score\t-0.2
            pair\tEURUSD\tmicrostructure_regime\tHOSTILE
            pair\tEURUSD\tsession_tag\tNY
            pair\tEURUSD\thandoff_flag\t1
            pair\tEURUSD\tsession_open_burst_score\t1.3
            pair\tEURUSD\tsession_price_cost_behavior_score\t-0.2
            pair\tEURUSD\ttrade_gate\tCAUTION
            pair\tEURUSD\tstale\t0
            pair\tEURUSD\tcaution_lot_scale\t1.5
            pair\tEURUSD\tcaution_enter_prob_buffer\t0.4
            pair_reason\tEURUSD\tevent\twide_spread
            pair_reason\tEURUSD\tevent\twide_spread
            pair_reason\tEURUSD\tevent\tsweep_reject
            pair\tGBPUSD\ttrade_gate\tBLOCK
            """,
            nowUTC: 1_040,
            freshnessMaxSeconds: 45
        )

        XCTAssertNotNil(state)
        XCTAssertTrue(state?.ready ?? false)
        XCTAssertTrue(state?.available ?? false)
        XCTAssertFalse(state?.stale ?? true)
        XCTAssertEqual(state?.generatedAt, 1_000)
        XCTAssertEqual(state?.tickImbalance30s, 1.0)
        XCTAssertEqual(state?.directionalEfficiency60s, 0.0)
        XCTAssertEqual(state?.priceCostCurrent, 0.0)
        XCTAssertEqual(state?.priceCostZscore60s, 8.0)
        XCTAssertEqual(state?.tickRate60s, 0.0)
        XCTAssertEqual(state?.tickRateZscore60s, -8.0)
        XCTAssertEqual(state?.realizedVol5m, 0.0)
        XCTAssertEqual(state?.volBurstScore5m, 8.0)
        XCTAssertEqual(state?.localExtremaBreachScore60s, 1.0)
        XCTAssertTrue(state?.sweepAndRejectFlag60s ?? false)
        XCTAssertEqual(state?.breakoutReversalScore60s, 0.0)
        XCTAssertEqual(state?.exhaustionProxy60s, 1.0)
        XCTAssertEqual(state?.liquidityStressScore, 1.0)
        XCTAssertEqual(state?.hostileExecutionScore, 0.0)
        XCTAssertEqual(state?.microstructureRegime, "HOSTILE")
        XCTAssertEqual(state?.sessionTag, "NY")
        XCTAssertTrue(state?.handoffFlag ?? false)
        XCTAssertEqual(state?.sessionOpenBurstScore, 1.0)
        XCTAssertEqual(state?.sessionPriceCostBehaviorScore, 0.0)
        XCTAssertEqual(state?.tradeGate, "CAUTION")
        XCTAssertEqual(state?.cautionLotScale, 1.5)
        XCTAssertEqual(state?.cautionEnterProbabilityBuffer, 0.4)
        XCTAssertEqual(state?.reasonsCSV, "wide_price_cost; sweep_reject")
    }

    func testMicrostructurePairStateCodableDecodesLegacySpreadKeysAndEncodesPriceCostKeys() throws {
        let legacyJSON = """
        {
          "ready": true,
          "available": true,
          "stale": false,
          "spreadCurrent": 1.2,
          "spreadZscore60s": 2.4,
          "sessionSpreadBehaviorScore": 0.7,
          "reasons": ["wide_spread", "Spread instability elevated"]
        }
        """.data(using: .utf8)!

        let state = try JSONDecoder().decode(MicrostructurePairState.self, from: legacyJSON)
        XCTAssertEqual(state.priceCostCurrent, 1.2, accuracy: 0.0)
        XCTAssertEqual(state.priceCostZscore60s, 2.4, accuracy: 0.0)
        XCTAssertEqual(state.sessionPriceCostBehaviorScore, 0.7, accuracy: 0.0)
        XCTAssertEqual(state.reasonsCSV, "wide_price_cost; Price-cost instability elevated")

        let encoded = try JSONEncoder().encode(state)
        let encodedText = String(data: encoded, encoding: .utf8) ?? ""
        XCTAssertTrue(encodedText.contains("priceCostCurrent"))
        XCTAssertTrue(encodedText.contains("priceCostZscore60s"))
        XCTAssertTrue(encodedText.contains("sessionPriceCostBehaviorScore"))
        XCTAssertFalse(encodedText.contains("spreadCurrent"))
        XCTAssertFalse(encodedText.contains("spreadZscore60s"))
        XCTAssertFalse(encodedText.contains("sessionSpreadBehaviorScore"))
    }

    func testMicrostructureMarksStaleAndAppliesStringDefaults() {
        let state = MicrostructureTools.readPairState(
            symbol: "EURUSD",
            snapshotTSV: """
            meta\tglobal\tgenerated_at_unix\t1000
            pair\tEURUSD\tstale\t0
            pair\tEURUSD\tmicrostructure_regime\t
            pair\tEURUSD\tsession_tag\t
            pair\tEURUSD\ttrade_gate\t
            """,
            nowUTC: 1_011,
            freshnessMaxSeconds: 0
        )

        XCTAssertNotNil(state)
        XCTAssertTrue(state?.stale ?? false)
        XCTAssertEqual(state?.microstructureRegime, "UNKNOWN")
        XCTAssertEqual(state?.sessionTag, "UNKNOWN")
        XCTAssertEqual(state?.tradeGate, "UNKNOWN")
    }

    func testMicrostructureReasonLimitAndMissingTimestampDoNotForceStaleBeyondParsedFlag() {
        let state = MicrostructureTools.readPairState(
            symbol: "EURUSD",
            snapshotTSV: """
            pair\tEURUSD\tstale\t0
            pair_reason\tEURUSD\tevent\tr1
            pair_reason\tEURUSD\tevent\tr2
            pair_reason\tEURUSD\tevent\tr3
            pair_reason\tEURUSD\tevent\tr4
            pair_reason\tEURUSD\tevent\tr5
            pair_reason\tEURUSD\tevent\tr6
            pair_reason\tEURUSD\tevent\tr7
            """,
            nowUTC: 2_000
        )

        XCTAssertNotNil(state)
        XCTAssertFalse(state?.stale ?? true)
        XCTAssertEqual(state?.reasonCount, MicrostructureConstants.maxReasons)
        XCTAssertEqual(state?.reasonsCSV, "r1; r2; r3; r4; r5; r6")
    }

    func testMicrostructureUnavailableWithoutPairState() {
        XCTAssertNil(MicrostructureTools.readPairState(symbol: "bad", snapshotTSV: nil))
        XCTAssertNil(MicrostructureTools.readPairState(symbol: "EURUSD", snapshotTSV: nil))
        XCTAssertNil(MicrostructureTools.readPairState(symbol: "EURUSD", snapshotTSV: "meta\tglobal\tgenerated_at_unix\t1000\n"))
    }
}
