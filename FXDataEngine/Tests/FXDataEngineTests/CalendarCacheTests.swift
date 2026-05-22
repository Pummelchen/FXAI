import XCTest
@testable import FXDataEngine

final class CalendarCacheTests: XCTestCase {
    func testCalendarCacheParsesStateAndClassifiesEvents() throws {
        let state = CalendarCacheTools.parseStateTSV(Self.stateTSV(ok: true, stale: false, recordCount: 2))
        let expectedTime = try XCTUnwrap(MacroEventTools.parseEventTimeUTC("2024.01.01 12:00:00"))

        XCTAssertTrue(state.ready)
        XCTAssertTrue(state.ok)
        XCTAssertFalse(state.stale)
        XCTAssertEqual(state.lastUpdateTradeServer, expectedTime)
        XCTAssertEqual(state.collectorGeneratedAt, expectedTime - 60)
        XCTAssertEqual(state.tradeServerOffsetSeconds, 7_200)
        XCTAssertEqual(state.recordCount, 2)
        XCTAssertEqual(CalendarCacheTools.eventClassFromTitle("FOMC rate decision"), CalendarEventClass.rates.rawValue)
        XCTAssertEqual(CalendarCacheTools.eventClassFromTitle("CPI inflation"), CalendarEventClass.inflation.rawValue)
        XCTAssertEqual(CalendarCacheTools.eventClassFromTitle("payroll employment"), CalendarEventClass.labor.rawValue)
        XCTAssertEqual(CalendarCacheTools.eventClassFromTitle("central bank speech"), CalendarEventClass.speech.rawValue)
        XCTAssertEqual(CalendarCacheTools.importanceWeight(3), 1.0)
        XCTAssertEqual(CalendarCacheTools.importanceWeight(2), 0.60)
        XCTAssertEqual(CalendarCacheTools.importanceWeight(0), 0.10)
        XCTAssertEqual(CalendarCacheTools.symbolLegs("EUR/USD.r").base, "EUR")
        XCTAssertEqual(CalendarCacheTools.symbolLegs("EUR/USD.r").quote, "USD")
        XCTAssertTrue(CalendarCacheTools.eventAffectsSymbol(symbol: "EUR/USD.r", currency: "USD"))
    }

    func testCalendarCachePairStateBlocksAroundCentralBankEvent() throws {
        let now = try XCTUnwrap(MacroEventTools.parseEventTimeUTC("2024.01.01 12:00:00"))
        let snapshot = CalendarCacheSnapshot.parse(
            stateTSV: Self.stateTSV(ok: true, stale: false, recordCount: 1),
            feedTSV: Self.feedTSV(rows: [
                Self.feedRow(currency: "USD", title: "FOMC rate decision", eventTime: now + 20 * 60, importance: 3)
            ])
        )

        let pair = snapshot.pairState(symbol: "EURUSD", nowTradeServer: now)

        XCTAssertTrue(pair.ready)
        XCTAssertFalse(pair.stale)
        XCTAssertEqual(pair.generatedAt, now)
        XCTAssertEqual(pair.nextEventETAMinutes, 20)
        XCTAssertEqual(pair.tradeGate, .block)
        XCTAssertEqual(pair.eventRiskScore, 1.0)
        XCTAssertEqual(pair.cautionLotScale, 0.0)
        XCTAssertEqual(pair.cautionEnterProbabilityBuffer, 0.10)
        XCTAssertEqual(pair.reasons, ["calendar_blackout", "calendar_central_bank"])
        XCTAssertEqual(pair.reasonsCSV, "calendar_blackout; calendar_central_bank")
    }

    func testCalendarCachePairStateCautionsAndDeduplicatesReasons() throws {
        let now = try XCTUnwrap(MacroEventTools.parseEventTimeUTC("2024.01.01 12:00:00"))
        let snapshot = CalendarCacheSnapshot.parse(
            stateTSV: Self.stateTSV(ok: true, stale: false, recordCount: 2),
            feedTSV: Self.feedTSV(rows: [
                Self.feedRow(eventID: 11, currency: "EUR", title: "CPI inflation", eventTime: now + 60 * 60, importance: 2),
                Self.feedRow(eventID: 12, currency: "EUR", title: "PPI price index", eventTime: now + 80 * 60, importance: 2)
            ])
        )

        let pair = snapshot.pairState(symbol: "EURUSD", nowTradeServer: now)

        XCTAssertEqual(pair.tradeGate, .caution)
        XCTAssertEqual(pair.nextEventETAMinutes, 60)
        XCTAssertEqual(pair.eventRiskScore, 0.441, accuracy: 1e-12)
        XCTAssertEqual(pair.cautionLotScale, 0.55)
        XCTAssertEqual(pair.cautionEnterProbabilityBuffer, 0.05)
        XCTAssertEqual(pair.reasons, ["calendar_caution", "calendar_inflation"])
    }

    func testCalendarCachePairStateSafeStaleAndErrorReason() throws {
        let now = try XCTUnwrap(MacroEventTools.parseEventTimeUTC("2024.01.01 12:20:00"))
        let snapshot = CalendarCacheSnapshot.parse(
            stateTSV: Self.stateTSV(ok: false, stale: false, recordCount: 0, lastError: "collector down"),
            feedTSV: Self.feedTSV(rows: [])
        )

        let pair = snapshot.pairState(symbol: "EURUSD", nowTradeServer: now)

        XCTAssertTrue(pair.ready)
        XCTAssertTrue(pair.stale)
        XCTAssertEqual(pair.tradeGate, .safe)
        XCTAssertEqual(pair.cautionLotScale, 1.0)
        XCTAssertEqual(pair.cautionEnterProbabilityBuffer, 0.0)
        XCTAssertEqual(pair.eventRiskScore, 0.0)
        XCTAssertEqual(pair.reasons, ["calendar_state_error"])
    }

    func testCalendarCacheUnavailableStateFailsClosed() {
        let pair = CalendarCacheSnapshot.parse(stateTSV: "", feedTSV: Self.feedTSV(rows: []))
            .pairState(symbol: "EURUSD", nowTradeServer: 1_704_067_200)

        XCTAssertFalse(pair.ready)
        XCTAssertTrue(pair.stale)
        XCTAssertEqual(pair.tradeGate, .unknown)
        XCTAssertEqual(pair.nextEventETAMinutes, -1)
    }

    private static func stateTSV(
        ok: Bool,
        stale: Bool,
        recordCount: Int,
        lastError: String = ""
    ) -> String {
        """
schema_version\t1
ok\t\(ok ? 1 : 0)
stale\t\(stale ? 1 : 0)
time_basis\ttrade_server
trade_server_offset_sec\t7200
last_update_trade_server\t2024-01-01T12:00:00
collector_generated_at\t2024-01-01T11:59:00Z
record_count\t\(recordCount)
last_error\t\(lastError)

"""
    }

    private static func feedTSV(rows: [String]) -> String {
        let header = "event_id\tevent_key\ttitle\tcountry_code\tcountry_name\tcurrency\tevent_time_unix\timportance\tactual\tforecast\tprevious\trevised_previous\tsurprise_proxy\tcollector_seen_unix\tchange_id\tevent_time_trade_server\tcollector_seen_trade_server\tevent_time_utc_unix\tcollector_seen_utc_unix\ttrade_server_offset_sec"
        return ([header] + rows).joined(separator: "\n") + "\n"
    }

    private static func feedRow(
        eventID: UInt64 = 10,
        currency: String,
        title: String,
        eventTime: Int64,
        importance: Int
    ) -> String {
        [
            "\(eventID)",
            "\(eventID)|\(eventTime)|0",
            title,
            currency,
            currency,
            currency,
            "\(eventTime)",
            "\(importance)",
            "",
            "",
            "",
            "",
            "",
            "\(eventTime - 600)",
            "1",
            "2024-01-01T12:00:00",
            "2024-01-01T11:50:00",
            "\(eventTime)",
            "\(eventTime - 600)",
            "7200"
        ].joined(separator: "\t")
    }
}
