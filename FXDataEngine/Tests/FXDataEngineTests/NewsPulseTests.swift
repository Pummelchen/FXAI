import XCTest
@testable import FXDataEngine

final class NewsPulseTests: XCTestCase {
    func testNewsPulsePairIDUsesMapThenHeuristic() {
        let map = NewsPulseTools.parseSymbolMap(tsv: """
        symbol\tEURUSD.r\tEURUSD
        symbol\tbad\tEUR
        ignored\tEURUSD\tGBPUSD
        """)

        XCTAssertEqual(map, [NewsPulseSymbolMapEntry(symbol: "EURUSD.R", pairID: "EURUSD")])
        XCTAssertEqual(NewsPulseTools.pairID(symbol: "eurusd.r", symbolMap: map), "EURUSD")
        XCTAssertEqual(NewsPulseTools.pairID(symbol: "xauEURUSDpro"), "EURUSD")
        XCTAssertEqual(NewsPulseTools.pairID(symbol: "USDUSD"), "")
        XCTAssertEqual(NewsPulseTools.alphaOnly("EUR/USD.r"), "EURUSDR")
    }

    func testNewsPulseParsesSnapshotClampsAndDeduplicatesReasons() {
        let state = NewsPulseTools.readPairState(
            symbol: "EURUSD",
            snapshotTSV: """
            meta\tglobal\tgenerated_at_unix\t1000
            pair\tEURUSD\tevent_eta_min\t15
            pair\tEURUSD\tnews_risk_score\t1.4
            pair\tEURUSD\tnews_pressure\t-2.0
            pair\tEURUSD\ttrade_gate\tCAUTION
            pair\tEURUSD\tstale\t0
            pair\tEURUSD\tsession_profile\tlondon
            pair\tEURUSD\tcalibration_profile\t
            pair\tEURUSD\twatchlist_tags\tECB,CPI
            pair\tEURUSD\tcaution_lot_scale\t1.5
            pair\tEURUSD\tcaution_enter_prob_buffer\t0.5
            pair_reason\tEURUSD\tevent\tcpi
            pair_reason\tEURUSD\tevent\tcpi
            pair_reason\tEURUSD\tevent\trates
            pair\tGBPUSD\ttrade_gate\tBLOCK
            """,
            nowUTC: 1_100,
            freshnessMaxSeconds: 360
        )

        XCTAssertNotNil(state)
        XCTAssertTrue(state?.ready ?? false)
        XCTAssertTrue(state?.available ?? false)
        XCTAssertFalse(state?.stale ?? true)
        XCTAssertEqual(state?.generatedAt, 1_000)
        XCTAssertEqual(state?.eventETAMinutes, 15)
        XCTAssertEqual(state?.newsRiskScore, 1.0)
        XCTAssertEqual(state?.newsPressure, -1.0)
        XCTAssertEqual(state?.tradeGate, "CAUTION")
        XCTAssertEqual(state?.sessionProfile, "london")
        XCTAssertEqual(state?.calibrationProfile, "london")
        XCTAssertEqual(state?.watchlistTagsCSV, "ECB,CPI")
        XCTAssertEqual(state?.cautionLotScale, 1.0)
        XCTAssertEqual(state?.cautionEnterProbabilityBuffer, 0.25)
        XCTAssertEqual(state?.reasonsCSV, "cpi; rates")
    }

    func testNewsPulseMarksStaleAndAppliesCalendarFallback() {
        let calendar = CalendarCachePairState(
            ready: true,
            stale: false,
            generatedAt: 2_000,
            nextEventETAMinutes: 4,
            tradeGate: .block,
            eventRiskScore: 0.80,
            cautionLotScale: 0.40,
            cautionEnterProbabilityBuffer: 0.12,
            reasons: ["central_bank"]
        )

        let state = NewsPulseTools.readPairState(
            symbol: "EURUSD",
            snapshotTSV: """
            meta\tglobal\tgenerated_at_unix\t1000
            pair\tEURUSD\tnews_risk_score\t0.2
            pair\tEURUSD\ttrade_gate\tUNKNOWN
            pair\tEURUSD\tstale\t0
            pair_reason\tEURUSD\tevent\told_news
            """,
            calendarFallback: calendar,
            nowUTC: 2_000,
            freshnessMaxSeconds: 60
        )

        XCTAssertNotNil(state)
        XCTAssertTrue(state?.ready ?? false)
        XCTAssertFalse(state?.stale ?? true)
        XCTAssertEqual(state?.generatedAt, 2_000)
        XCTAssertEqual(state?.eventETAMinutes, 4)
        XCTAssertEqual(state?.newsRiskScore, 0.80)
        XCTAssertEqual(state?.newsPressure, 0.40)
        XCTAssertEqual(state?.tradeGate, "BLOCK")
        XCTAssertEqual(state?.sessionProfile, "calendar_cache")
        XCTAssertEqual(state?.calibrationProfile, "calendar_cache")
        XCTAssertEqual(state?.watchlistTagsCSV, "mt5_calendar_cache")
        XCTAssertEqual(state?.cautionLotScale, 0.40)
        XCTAssertEqual(state?.cautionEnterProbabilityBuffer, 0.12)
        XCTAssertEqual(state?.reasonsCSV, "old_news; central_bank; calendar_cache_fallback")
    }

    func testNewsPulseUnavailableWithoutSnapshotOrFallback() {
        XCTAssertNil(NewsPulseTools.readPairState(symbol: "bad", snapshotTSV: nil))
        XCTAssertNil(NewsPulseTools.readPairState(symbol: "EURUSD", snapshotTSV: nil))
    }
}
