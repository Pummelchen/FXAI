import XCTest
@testable import FXDataEngine

final class CrossAssetTests: XCTestCase {
    func testCrossAssetPairIDUsesCrossAssetMapThenNewsPulseFallback() {
        let crossMap = CrossAssetTools.parseSymbolMap(tsv: """
        symbol\tEURUSD.r\tEURUSD
        symbol\tbad\tEUR
        """)
        let newsMap = NewsPulseTools.parseSymbolMap(tsv: "symbol\tGBPUSD.pro\tGBPUSD\n")

        XCTAssertEqual(crossMap, [CrossAssetSymbolMapEntry(symbol: "EURUSD.R", pairID: "EURUSD")])
        XCTAssertEqual(CrossAssetTools.pairID(symbol: "eurusd.r", symbolMap: crossMap), "EURUSD")
        XCTAssertEqual(CrossAssetTools.pairID(symbol: "gbpusd.pro", newsPulseSymbolMap: newsMap), "GBPUSD")
        XCTAssertEqual(CrossAssetTools.pairID(symbol: "prefixAUDNZD"), "AUDNZD")
    }

    func testCrossAssetParsesGlobalAndPairSnapshotWithoutClamping() {
        let state = CrossAssetTools.readPairState(
            symbol: "EURUSD",
            snapshotTSV: """
            meta\tglobal\tgenerated_at_unix\t1000
            score\tglobal\trates_repricing_score\t1.2
            score\tglobal\trisk_off_score\t0.6
            score\tglobal\tcommodity_shock_score\t-0.3
            score\tglobal\tvolatility_shock_score\t0.8
            score\tglobal\tusd_liquidity_stress_score\t1.5
            score\tglobal\tcross_asset_dislocation_score\t0.4
            pair\tEURUSD\tpair_cross_asset_risk_score\t1.7
            pair\tEURUSD\tpair_sensitivity\t-0.25
            pair\tEURUSD\tmacro_state\tTIGHTENING
            pair\tEURUSD\trisk_state\tRISK_OFF
            pair\tEURUSD\tliquidity_state\tSTRESSED
            pair\tEURUSD\ttrade_gate\tCAUTION
            pair\tEURUSD\tstale\t0
            pair_reason\tEURUSD\tevent\tusd_liquidity
            pair_reason\tEURUSD\tevent\tusd_liquidity
            pair_reason\tEURUSD\tevent\trates_reprice
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
        XCTAssertEqual(state?.ratesRepricingScore, 1.2)
        XCTAssertEqual(state?.riskOffScore, 0.6)
        XCTAssertEqual(state?.commodityShockScore, -0.3)
        XCTAssertEqual(state?.volatilityShockScore, 0.8)
        XCTAssertEqual(state?.usdLiquidityStressScore, 1.5)
        XCTAssertEqual(state?.crossAssetDislocationScore, 0.4)
        XCTAssertEqual(state?.pairCrossAssetRiskScore, 1.7)
        XCTAssertEqual(state?.pairSensitivity, -0.25)
        XCTAssertEqual(state?.macroState, "TIGHTENING")
        XCTAssertEqual(state?.riskState, "RISK_OFF")
        XCTAssertEqual(state?.liquidityState, "STRESSED")
        XCTAssertEqual(state?.tradeGate, "CAUTION")
        XCTAssertEqual(state?.reasonsCSV, "usd_liquidity; rates_reprice")
    }

    func testCrossAssetReasonLimitAndFreshnessRulesMatchLegacyReader() {
        let state = CrossAssetTools.readPairState(
            symbol: "EURUSD",
            snapshotTSV: """
            meta\tglobal\tgenerated_at_unix\t1000
            pair\tEURUSD\tstale\t0
            pair_reason\tEURUSD\tevent\tr1
            pair_reason\tEURUSD\tevent\tr2
            pair_reason\tEURUSD\tevent\tr3
            pair_reason\tEURUSD\tevent\tr4
            pair_reason\tEURUSD\tevent\tr5
            pair_reason\tEURUSD\tevent\tr6
            pair_reason\tEURUSD\tevent\tr7
            pair_reason\tEURUSD\tevent\tr8
            pair_reason\tEURUSD\tevent\tr9
            """,
            nowUTC: 2_000,
            freshnessMaxSeconds: 60
        )

        XCTAssertNotNil(state)
        XCTAssertTrue(state?.stale ?? false)
        XCTAssertEqual(state?.reasonCount, CrossAssetConstants.maxReasons)
        XCTAssertEqual(state?.reasonsCSV, "r1; r2; r3; r4; r5; r6; r7; r8")
    }

    func testCrossAssetMarksAvailableStateStaleWhenClockOrGeneratedAtMissing() {
        let stateWithoutGeneratedAt = CrossAssetTools.readPairState(
            symbol: "EURUSD",
            snapshotTSV: "pair\tEURUSD\tstale\t0\n",
            nowUTC: 2_000
        )
        let stateWithoutClock = CrossAssetTools.readPairState(
            symbol: "EURUSD",
            snapshotTSV: """
            meta\tglobal\tgenerated_at_unix\t1000
            pair\tEURUSD\tstale\t0
            """,
            nowUTC: 0
        )

        XCTAssertTrue(stateWithoutGeneratedAt?.stale ?? false)
        XCTAssertTrue(stateWithoutClock?.stale ?? false)
    }

    func testCrossAssetUnavailableWithoutPairState() {
        XCTAssertNil(CrossAssetTools.readPairState(symbol: "bad", snapshotTSV: nil))
        XCTAssertNil(CrossAssetTools.readPairState(symbol: "EURUSD", snapshotTSV: nil))
        XCTAssertNil(CrossAssetTools.readPairState(symbol: "EURUSD", snapshotTSV: "meta\tglobal\tgenerated_at_unix\t1000\n"))
    }
}
