import XCTest
@testable import FXDataEngine

final class RatesEngineTests: XCTestCase {
    func testRatesEnginePairIDUsesRatesMapThenNewsPulseFallback() {
        let ratesMap = RatesEngineTools.parseSymbolMap(tsv: """
        symbol\tEURUSD.r\tEURUSD
        symbol\tbad\tEUR
        """)
        let newsMap = NewsPulseTools.parseSymbolMap(tsv: "symbol\tGBPUSD.pro\tGBPUSD\n")

        XCTAssertEqual(ratesMap, [RatesEngineSymbolMapEntry(symbol: "EURUSD.R", pairID: "EURUSD")])
        XCTAssertEqual(RatesEngineTools.pairID(symbol: "eurusd.r", symbolMap: ratesMap), "EURUSD")
        XCTAssertEqual(RatesEngineTools.pairID(symbol: "gbpusd.pro", newsPulseSymbolMap: newsMap), "GBPUSD")
        XCTAssertEqual(RatesEngineTools.pairID(symbol: "prefixAUDNZD"), "AUDNZD")
    }

    func testRatesEngineParsesSnapshotClampsAndDeduplicatesReasons() {
        let state = RatesEngineTools.readPairState(
            symbol: "EURUSD",
            snapshotTSV: """
            meta\tglobal\tgenerated_at_unix\t1000
            pair\tEURUSD\tfront_end_diff\t12.5
            pair\tEURUSD\texpected_path_diff\t-12.5
            pair\tEURUSD\tcurve_divergence_score\t1.5
            pair\tEURUSD\tpolicy_divergence_score\t0.7
            pair\tEURUSD\trates_risk_score\t2.0
            pair\tEURUSD\tmacro_to_rates_transmission_score\t-1.0
            pair\tEURUSD\tmeeting_path_reprice_now\t1
            pair\tEURUSD\tstale\t0
            pair\tEURUSD\trates_regime\tHAWKISH
            pair\tEURUSD\ttrade_gate\tCAUTION
            pair\tEURUSD\tpolicy_alignment\tbase_hawkish
            pair_reason\tEURUSD\tevent\tfomc
            pair_reason\tEURUSD\tevent\tfomc
            pair_reason\tEURUSD\tevent\tecb
            pair\tGBPUSD\ttrade_gate\tBLOCK
            """,
            nowUTC: 1_100,
            freshnessMaxSeconds: 900
        )

        XCTAssertNotNil(state)
        XCTAssertTrue(state?.ready ?? false)
        XCTAssertTrue(state?.available ?? false)
        XCTAssertFalse(state?.stale ?? true)
        XCTAssertEqual(state?.generatedAt, 1_000)
        XCTAssertEqual(state?.frontEndDiff, 10.0)
        XCTAssertEqual(state?.expectedPathDiff, -10.0)
        XCTAssertEqual(state?.curveDivergenceScore, 1.0)
        XCTAssertEqual(state?.policyDivergenceScore, 0.7)
        XCTAssertEqual(state?.ratesRiskScore, 1.0)
        XCTAssertEqual(state?.macroToRatesTransmissionScore, 0.0)
        XCTAssertTrue(state?.meetingPathRepriceNow ?? false)
        XCTAssertEqual(state?.ratesRegime, "HAWKISH")
        XCTAssertEqual(state?.tradeGate, "CAUTION")
        XCTAssertEqual(state?.policyAlignment, "base_hawkish")
        XCTAssertEqual(state?.reasonsCSV, "fomc; ecb")
    }

    func testRatesEngineMarksStaleAndAppliesStringDefaults() {
        let state = RatesEngineTools.readPairState(
            symbol: "EURUSD",
            snapshotTSV: """
            meta\tglobal\tgenerated_at_unix\t1000
            pair\tEURUSD\tstale\t0
            pair\tEURUSD\trates_regime\t
            pair\tEURUSD\ttrade_gate\t
            pair\tEURUSD\tpolicy_alignment\t
            """,
            nowUTC: 2_000,
            freshnessMaxSeconds: 60
        )

        XCTAssertNotNil(state)
        XCTAssertTrue(state?.stale ?? false)
        XCTAssertEqual(state?.ratesRegime, "UNKNOWN")
        XCTAssertEqual(state?.tradeGate, "UNKNOWN")
        XCTAssertEqual(state?.policyAlignment, "balanced")
    }

    func testRatesEngineUnavailableWithoutPairState() {
        XCTAssertNil(RatesEngineTools.readPairState(symbol: "bad", snapshotTSV: nil))
        XCTAssertNil(RatesEngineTools.readPairState(symbol: "EURUSD", snapshotTSV: nil))
        XCTAssertNil(RatesEngineTools.readPairState(symbol: "EURUSD", snapshotTSV: "meta\tglobal\tgenerated_at_unix\t1000\n"))
    }
}
