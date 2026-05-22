import XCTest
@testable import FXDataEngine

final class NewsPulseReplayTests: XCTestCase {
    func testReplayTimelineFiltersPairAndParsesLegacyColumns() {
        let records = NewsPulseReplayTools.parseTimeline(
            tsv: """
            pair_id\tsymbol\tobserved_at\ttrade_gate\tnews_risk_score\tnews_pressure\tstale\tevent_eta_min
            GBPUSD\tGBPUSD\t90\tBLOCK\t0.9\t-0.1\t0\t4
            EURUSD\tEURUSD\t100\tCAUTION\t1.4\t0.5\t0\t
            EURUSD\tEURUSD\t130\tBLOCK\t0.2\t-0.8\t1\t12
            bad
            """,
            symbol: "eurusd.r",
            symbolMapTSV: "symbol\teurusd.r\tEURUSD"
        )

        XCTAssertEqual(records.count, 2)
        XCTAssertEqual(records[0].pairID, "EURUSD")
        XCTAssertEqual(records[0].observedAtUTC, 100)
        XCTAssertEqual(records[0].tradeGate, "CAUTION")
        XCTAssertEqual(records[0].newsRiskScore, 1.4)
        XCTAssertEqual(records[0].newsPressure, 0.5)
        XCTAssertFalse(records[0].stale)
        XCTAssertEqual(records[0].eventETAMinutes, -1)
        XCTAssertEqual(records[1].observedAtUTC, 130)
        XCTAssertTrue(records[1].stale)
    }

    func testReplayIndexUsesLastObservedRecordAndFallsBackToFirst() {
        let records = [
            NewsPulseReplayRecord(
                pairID: "EURUSD",
                observedAtUTC: 100,
                eventETAMinutes: -1,
                newsRiskScore: 0.10,
                newsPressure: 0.0,
                stale: false,
                tradeGate: "ALLOW"
            ),
            NewsPulseReplayRecord(
                pairID: "EURUSD",
                observedAtUTC: 160,
                eventETAMinutes: -1,
                newsRiskScore: 0.30,
                newsPressure: 0.0,
                stale: false,
                tradeGate: "ALLOW"
            ),
        ]

        XCTAssertEqual(NewsPulseReplayTools.replayIndex(records: records, queryTimeUTC: 50), 0)
        XCTAssertEqual(NewsPulseReplayTools.replayIndex(records: records, queryTimeUTC: 100), 0)
        XCTAssertEqual(NewsPulseReplayTools.replayIndex(records: records, queryTimeUTC: 159), 0)
        XCTAssertEqual(NewsPulseReplayTools.replayIndex(records: records, queryTimeUTC: 160), 1)
        XCTAssertNil(NewsPulseReplayTools.replayIndex(records: [], queryTimeUTC: 160))
    }

    func testReplayWindowScoreMatchesGateStaleAndEventPriority() {
        let records = [
            NewsPulseReplayRecord(
                pairID: "EURUSD",
                observedAtUTC: 100,
                eventETAMinutes: -1,
                newsRiskScore: 0.30,
                newsPressure: 0.0,
                stale: false,
                tradeGate: "ALLOW"
            ),
            NewsPulseReplayRecord(
                pairID: "EURUSD",
                observedAtUTC: 130,
                eventETAMinutes: -1,
                newsRiskScore: 0.40,
                newsPressure: 0.0,
                stale: false,
                tradeGate: "CAUTION"
            ),
            NewsPulseReplayRecord(
                pairID: "EURUSD",
                observedAtUTC: 160,
                eventETAMinutes: 10,
                newsRiskScore: 0.20,
                newsPressure: 0.0,
                stale: false,
                tradeGate: "ALLOW"
            ),
            NewsPulseReplayRecord(
                pairID: "EURUSD",
                observedAtUTC: 190,
                eventETAMinutes: -1,
                newsRiskScore: 0.50,
                newsPressure: 0.0,
                stale: true,
                tradeGate: "BLOCK"
            ),
        ]
        let times: [Int64] = [100, 115, 130, 145, 160, 175, 190]

        XCTAssertEqual(
            NewsPulseReplayTools.windowScore(records: records, sampleTimesUTC: times, start: 0, bars: 3),
            0.72,
            accuracy: 1.0e-12
        )
        XCTAssertEqual(
            NewsPulseReplayTools.windowScore(records: records, sampleTimesUTC: times, start: 2, bars: 3),
            0.84,
            accuracy: 1.0e-12
        )
        XCTAssertEqual(
            NewsPulseReplayTools.windowScore(records: records, sampleTimesUTC: times, start: 4, bars: 3),
            0.98,
            accuracy: 1.0e-12
        )
    }

    func testReplayWindowScoreReturnsZeroForInvalidWindowsOrMissingReplay() {
        let records = [
            NewsPulseReplayRecord(
                pairID: "EURUSD",
                observedAtUTC: 100,
                eventETAMinutes: -1,
                newsRiskScore: 0.30,
                newsPressure: 0.0,
                stale: false,
                tradeGate: "ALLOW"
            ),
        ]
        let times: [Int64] = [100, 101]

        XCTAssertEqual(NewsPulseReplayTools.windowScore(records: records, sampleTimesUTC: times, start: -1, bars: 1), 0.0)
        XCTAssertEqual(NewsPulseReplayTools.windowScore(records: records, sampleTimesUTC: times, start: 0, bars: 0), 0.0)
        XCTAssertEqual(NewsPulseReplayTools.windowScore(records: records, sampleTimesUTC: times, start: 1, bars: 2), 0.0)
        XCTAssertEqual(NewsPulseReplayTools.windowScore(records: [], sampleTimesUTC: times, start: 0, bars: 2), 0.0)
    }
}
