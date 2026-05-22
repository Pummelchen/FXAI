import XCTest
@testable import FXDataEngine

final class SystemHealthTests: XCTestCase {
    func testSystemHealthAllReadyIsHealthy() {
        let health = SystemHealthTools.refresh(
            generatedAt: 1_704_067_200,
            input: SystemHealthInput(
                news: .init(ready: true, stale: false),
                rates: .init(ready: true, stale: false),
                crossAsset: .init(ready: true, stale: false),
                microstructure: .init(ready: true, stale: false),
                executionQuality: .init(ready: true, stale: false),
                calendar: CalendarCachePairState(ready: true, stale: false),
                factor: PairFactorContext(ready: true, stale: false, symbol: "EURUSD")
            )
        )

        XCTAssertTrue(health.ready)
        XCTAssertEqual(health.generatedAt, 1_704_067_200)
        XCTAssertEqual(health.healthScore, 1.0)
        XCTAssertEqual(health.degradedCount, 0)
        XCTAssertEqual(health.posture, .healthy)
        XCTAssertEqual(health.reasonsCSV, "")
        XCTAssertTrue(health.calendarReady)
        XCTAssertTrue(health.factorReady)
        XCTAssertFalse(health.factorStale)
    }

    func testSystemHealthMatchesLegacyStaleAndUnavailableScoring() {
        let health = SystemHealthTools.refresh(
            generatedAt: 1_704_067_200,
            input: SystemHealthInput(
                news: .init(ready: true, stale: true),
                rates: nil,
                crossAsset: .init(ready: true, stale: false),
                microstructure: .init(ready: true, stale: true),
                executionQuality: .init(ready: true, stale: false, dataStale: true),
                calendar: CalendarCachePairState(ready: true, stale: false),
                factor: PairFactorContext(ready: true, stale: true, symbol: "EURUSD")
            )
        )

        let expectedScore = (0.45 + 0.0 + 1.0 + 0.40 + 0.45 + 1.0 + 1.0) / 7.0
        XCTAssertEqual(health.healthScore, expectedScore, accuracy: 1e-12)
        XCTAssertEqual(health.degradedCount, 4)
        XCTAssertEqual(health.posture, .caution)
        XCTAssertTrue(health.newsStale)
        XCTAssertFalse(health.ratesReady)
        XCTAssertTrue(health.microStale)
        XCTAssertTrue(health.executionStale)
        XCTAssertTrue(health.factorStale)
        XCTAssertEqual(
            health.reasonsCSV,
            "newspulse_stale; rates_unavailable; microstructure_stale; execution_quality_stale"
        )
    }

    func testSystemHealthAllUnavailableIsDegradedAndDedupesReasons() {
        var csv = "newspulse_stale"
        SystemHealthTools.appendReason(&csv, "newspulse_stale")
        SystemHealthTools.appendReason(&csv, "rates_unavailable")
        XCTAssertEqual(csv, "newspulse_stale; rates_unavailable")

        let health = SystemHealthTools.refresh(generatedAt: 1, input: SystemHealthInput())

        XCTAssertEqual(health.healthScore, 0.0)
        XCTAssertEqual(health.degradedCount, 7)
        XCTAssertEqual(health.posture, .degraded)
        XCTAssertEqual(
            health.reasonsCSV,
            "newspulse_unavailable; rates_unavailable; cross_asset_unavailable; microstructure_unavailable; execution_quality_unavailable; calendar_cache_unavailable; factor_context_unavailable"
        )
    }

    func testFactorNotReadyScoresZeroWithoutStaleReasonLikeLegacy() {
        let health = SystemHealthTools.refresh(
            generatedAt: 1,
            input: SystemHealthInput(
                news: .init(ready: true, stale: false),
                rates: .init(ready: true, stale: false),
                crossAsset: .init(ready: true, stale: false),
                microstructure: .init(ready: true, stale: false),
                executionQuality: .init(ready: true, stale: false),
                calendar: CalendarCachePairState(ready: true, stale: false),
                factor: PairFactorContext(ready: false, stale: true, symbol: "EURUSD")
            )
        )

        XCTAssertEqual(health.healthScore, 6.0 / 7.0, accuracy: 1e-12)
        XCTAssertEqual(health.degradedCount, 0)
        XCTAssertEqual(health.posture, .healthy)
        XCTAssertEqual(health.reasonsCSV, "")
    }
}
