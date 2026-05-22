import XCTest
@testable import FXDataEngine

final class FactorContextTests: XCTestCase {
    func testFactorScoresMirrorLegacyMath() {
        let series = Self.eurusdSeries

        XCTAssertEqual(
            FactorContextTools.d1Return(series: series, shiftNow: 1, shiftThen: 126),
            0.10,
            accuracy: 1e-12
        )

        let expectedTrend = (
            0.20 * ((1.10 / 1.08) - 1.0)
                + 0.35 * ((1.10 / 1.05) - 1.0)
                + 0.45 * ((1.10 / 1.00) - 1.0)
        ) * 8.0
        XCTAssertEqual(FactorContextTools.trendScore(series: series), expectedTrend, accuracy: 1e-12)

        let mediumAnchor = 0.5 * (1.05 + 0.90)
        let expectedValue = -((1.10 - mediumAnchor) / mediumAnchor) * 6.0
        XCTAssertEqual(FactorContextTools.valueScore(series: series), expectedValue, accuracy: 1e-12)

        let carry = FactorCarrySnapshot(swapLong: 0.40, swapShort: -0.20)
        XCTAssertEqual(FactorContextTools.carryDirectional(carry, direction: 1), 0.80, accuracy: 1e-12)
        XCTAssertEqual(FactorContextTools.carryDirectional(carry, direction: 0), -0.40, accuracy: 1e-12)
        XCTAssertEqual(FactorContextTools.carryDirectional(carry, direction: -1), 1.0, accuracy: 1e-12)
    }

    func testPolicyAndCommodityScoresUsePreparedOfflineInputs() {
        let market = Self.marketSnapshot()

        XCTAssertEqual(FactorContextTools.policyPressure(currency: "EUR", market: market), 0.88, accuracy: 1e-12)
        XCTAssertEqual(FactorContextTools.policyPressure(currency: "USD", market: market), 0.88, accuracy: 1e-12)
        XCTAssertEqual(FactorContextTools.commodityScore(currency: "AUD", market: market), 0.60, accuracy: 1e-12)
        XCTAssertEqual(FactorContextTools.commodityScore(currency: "JPY", market: market), -0.40, accuracy: 1e-12)
        XCTAssertEqual(FactorContextTools.commodityScore(currency: "EUR", market: market), 0.0, accuracy: 1e-12)
    }

    func testBuildCurrencyStateSelectsAnchorsAndUSDNeutralFallback() {
        let market = Self.marketSnapshot()
        let eur = FactorContextTools.buildCurrencyState(currency: "EUR", market: market)
        let usd = FactorContextTools.buildCurrencyState(currency: "USD", market: market)
        let jpy = FactorContextTools.buildCurrencyState(currency: "JPY", market: market)

        XCTAssertTrue(eur.ready)
        XCTAssertGreaterThan(eur.trendScore, 0.50)
        XCTAssertEqual(eur.carryScore, 0.80, accuracy: 1e-12)
        XCTAssertEqual(eur.policyScore, 0.88, accuracy: 1e-12)
        XCTAssertLessThan(eur.valueScore, -0.70)
        XCTAssertGreaterThan(eur.blendedScore, 0.20)

        XCTAssertTrue(usd.ready)
        XCTAssertEqual(usd.trendScore, 0.0)
        XCTAssertEqual(usd.policyScore, 0.88, accuracy: 1e-12)
        XCTAssertEqual(usd.blendedScore, 0.1936, accuracy: 1e-12)

        XCTAssertTrue(jpy.ready)
        XCTAssertLessThan(jpy.trendScore, 0.0)
        XCTAssertGreaterThan(jpy.carryScore, 0.0)
        XCTAssertGreaterThan(jpy.valueScore, 0.0)
    }

    func testBuildPairFactorContextComposesBaseMinusQuoteAndBias() {
        let market = Self.marketSnapshot()
        let context = FactorContextTools.buildPairContext(symbol: "EUR/USD.r", market: market)

        XCTAssertTrue(context.ready)
        XCTAssertFalse(context.stale)
        XCTAssertEqual(context.symbol, "EUR/USD.R")
        XCTAssertEqual(context.generatedAt, 1_704_067_200)
        XCTAssertEqual(context.policyScore, 0.0, accuracy: 1e-12)
        XCTAssertGreaterThan(context.blendedScore, 0.08)
        XCTAssertEqual(context.biasDirection, 1)
        XCTAssertEqual(context.alignmentScore, 0.50 + 0.50 * abs(context.blendedScore), accuracy: 1e-12)
        XCTAssertTrue(context.rationale.contains("trend="))
        XCTAssertTrue(context.rationale.contains("commodity="))
    }

    func testBuildPairFactorContextFailsClosedWhenCurrencyAnchorMissing() {
        let market = FactorMarketSnapshot(
            dailySeriesBySymbol: ["EURUSD": Self.eurusdSeries],
            generatedAt: 1_704_067_200
        )

        let context = FactorContextTools.buildPairContext(symbol: "EURJPY", market: market)

        XCTAssertFalse(context.ready)
        XCTAssertTrue(context.stale)
        XCTAssertEqual(context.symbol, "EURJPY")
        XCTAssertEqual(context.biasDirection, -1)
    }

    private static let eurusdSeries = FactorDailySeries(
        symbol: "EURUSD",
        closeByShift: [1: 1.10, 21: 1.08, 63: 1.05, 126: 1.00, 252: 0.90]
    )

    private static let usdjpySeries = FactorDailySeries(
        symbol: "USDJPY",
        closeByShift: [1: 150.0, 21: 140.0, 63: 130.0, 126: 120.0, 252: 110.0]
    )

    private static let goldSeries = FactorDailySeries(
        symbol: "XAUUSD",
        closeByShift: [1: 110.0, 20: 100.0, 60: 100.0]
    )

    private static func marketSnapshot() -> FactorMarketSnapshot {
        FactorMarketSnapshot(
            dailySeriesBySymbol: [
                "EURUSD": eurusdSeries,
                "USDJPY": usdjpySeries,
                "XAUUSD": goldSeries
            ],
            carryBySymbol: [
                "EURUSD": FactorCarrySnapshot(swapLong: 0.40, swapShort: -0.20),
                "USDJPY": FactorCarrySnapshot(swapLong: 1.50, swapShort: -0.50)
            ],
            calendarPairStateBySymbol: [
                "EURUSD": CalendarCachePairState(
                    ready: true,
                    stale: false,
                    nextEventETAMinutes: 30,
                    tradeGate: .block,
                    eventRiskScore: 1.0,
                    reasons: ["calendar_blackout", "calendar_central_bank", "calendar_inflation"]
                )
            ],
            generatedAt: 1_704_067_200
        )
    }
}
