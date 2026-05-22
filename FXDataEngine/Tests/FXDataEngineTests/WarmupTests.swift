import XCTest
@testable import FXDataEngine

final class WarmupTests: XCTestCase {
    func testWarmupBucketStatsUpdateAndScoreMatchLegacyFormula() {
        var stats = WarmupBucketStats()
        for value in [3.0, -2.0, 0.0, 5.0, -1.0] {
            stats.update(netPoints: value)
        }

        XCTAssertEqual(stats.trades, 5)
        XCTAssertEqual(stats.wins, 2)
        XCTAssertEqual(stats.netSum, 5.0, accuracy: 1e-12)
        XCTAssertEqual(stats.grossPositive, 8.0, accuracy: 1e-12)
        XCTAssertEqual(stats.grossNegative, 3.0, accuracy: 1e-12)
        XCTAssertEqual(stats.equity, 5.0, accuracy: 1e-12)
        XCTAssertEqual(stats.equityPeak, 6.0, accuracy: 1e-12)
        XCTAssertEqual(stats.maxDrawdown, 2.0, accuracy: 1e-12)
        XCTAssertEqual(WarmupTools.scoreBucket(stats), 7.458333333333333, accuracy: 1e-12)
    }

    func testWarmupBucketScoreFailsClosedWithoutTrades() {
        XCTAssertEqual(WarmupTools.scoreBucket(WarmupBucketStats()), WarmupTools.missingScore, accuracy: 0.0)
    }

    func testWarmupPortfolioObjectiveProxyMatchesLegacyBlend() {
        let objective = WarmupTools.portfolioObjectiveProxy(
            totalScore: 25.0,
            totalTrades: 40,
            regimeScores: [10.0, 20.0, -5.0, 0.0],
            regimeTrades: [12, 15, 5, 0]
        )

        XCTAssertEqual(objective, -0.15, accuracy: 1e-12)
        XCTAssertEqual(
            WarmupTools.portfolioObjectiveProxy(totalScore: 10.0, totalTrades: 20, regimeScores: [], regimeTrades: []),
            0.0,
            accuracy: 0.0
        )
    }
}
