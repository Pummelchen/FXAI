import FXAIPlugins
import XCTest

final class Wave3FactorTrendReferenceTests: XCTestCase {
    func testCarryReferenceRanksRatesAndGatesVolume() {
        let inputs = [
            FactorCarryReference.Input(symbol: "AUDJPY", baseRate: 0.042, quoteRate: 0.001, volume: 9.0),
            FactorCarryReference.Input(symbol: "EURUSD", baseRate: 0.020, quoteRate: 0.050, volume: 1.0),
            FactorCarryReference.Input(symbol: "USDCHF", baseRate: 0.050, quoteRate: 0.018, volume: 4.0)
        ]

        let withVolume = FactorCarryReference.rank(inputs, dataHasVolume: true)
        let withoutVolume = FactorCarryReference.rank(inputs, dataHasVolume: false)

        XCTAssertEqual(withVolume.first?.symbol, "AUDJPY")
        XCTAssertEqual(withoutVolume.first?.symbol, "AUDJPY")
        XCTAssertTrue(withoutVolume.allSatisfy { $0.liquidityWeight == 1.0 })
        XCTAssertGreaterThan(withVolume.first?.liquidityWeight ?? 0.0, 1.0)
    }

    func testCMVPanelReferenceNormalizesMomentumValueAndGatesVolume() {
        let rows = [
            FactorCMVPanelReference.PanelRow(symbol: "EURUSD", closes: [1.0, 1.02, 1.04], volumes: [10, 10, 30]),
            FactorCMVPanelReference.PanelRow(symbol: "GBPUSD", closes: [1.4, 1.39, 1.38], volumes: [10, 10, 10]),
            FactorCMVPanelReference.PanelRow(symbol: "USDJPY", closes: [140, 141, 142], volumes: [8, 8, 8])
        ]

        let withVolume = FactorCMVPanelReference.exposures(rows: rows, dataHasVolume: true)
        let withoutVolume = FactorCMVPanelReference.exposures(rows: rows, dataHasVolume: false)

        XCTAssertEqual(withVolume.count, rows.count)
        XCTAssertTrue(withoutVolume.allSatisfy { $0.volume == 0.0 })
        XCTAssertGreaterThan(abs(withVolume.first?.volume ?? 0.0), 0.1)
        XCTAssertEqual(withVolume.map(\.momentum).reduce(0.0, +), 0.0, accuracy: 1.0e-9)
    }

    func testPCAReferenceProducesOrthonormalFirstComponent() {
        let panel = (0..<32).map { index in
            let factor = sin(Double(index) * 0.20)
            return [factor, 0.8 * factor + 0.02 * cos(Double(index)), -0.4 * factor]
        }

        let result = FactorPCAPanelReference.firstComponent(panel: panel)
        let norm = sqrt(result.loadings.map { $0 * $0 }.reduce(0.0, +))

        XCTAssertEqual(result.loadings.count, 3)
        XCTAssertEqual(norm, 1.0, accuracy: 1.0e-8)
        XCTAssertGreaterThan(result.explainedVarianceRatio, 0.90)
        XCTAssertEqual(result.covariance[0][1], result.covariance[1][0], accuracy: 1.0e-12)
        XCTAssertEqual(result.scores.count, panel.count)
    }

    func testPPPReferenceScoresMeanReversionAndStaleness() {
        let scores = FactorPPPValueReference.scores([
            .init(symbol: "EURUSD", spot: 1.20, pppFairValue: 1.10, observationAgeDays: 10, halfLifeDays: 100),
            .init(symbol: "USDJPY", spot: 150, pppFairValue: 155, observationAgeDays: 10, halfLifeDays: 100),
            .init(symbol: "GBPUSD", spot: 1.35, pppFairValue: 1.30, observationAgeDays: 100, halfLifeDays: 100)
        ])

        let overvalued = scores.first { $0.symbol == "EURUSD" }
        let stale = scores.first { $0.symbol == "GBPUSD" }

        XCTAssertLessThan(overvalued?.valueScore ?? 0.0, 0.0)
        XCTAssertLessThan(stale?.staleDecay ?? 1.0, 0.60)
        XCTAssertEqual(scores.map(\.zScore).reduce(0.0, +), 0.0, accuracy: 1.0e-9)
    }

    func testTSMOMReferenceVolTargetsAndGatesLiquidity() {
        let up = TrendTSMOMVolReference.Series(
            symbol: "EURUSD",
            closes: (0..<40).map { 1.0 + 0.002 * Double($0) },
            volumes: Array(repeating: 10.0, count: 39) + [30.0]
        )
        let down = TrendTSMOMVolReference.Series(
            symbol: "USDJPY",
            closes: (0..<40).map { 150.0 - 0.05 * Double($0) },
            volumes: Array(repeating: 10.0, count: 40)
        )

        let withVolume = TrendTSMOMVolReference.signals(series: [up, down], lookback: 20, volatilityTarget: 0.10, maxLeverage: 2.0, dataHasVolume: true)
        let withoutVolume = TrendTSMOMVolReference.signals(series: [up, down], lookback: 20, volatilityTarget: 0.10, maxLeverage: 2.0, dataHasVolume: false)

        XCTAssertGreaterThan(withVolume[0].targetWeight, 0.0)
        XCTAssertLessThan(withVolume[1].targetWeight, 0.0)
        XCTAssertGreaterThan(withVolume[0].liquidityConfidence, withoutVolume[0].liquidityConfidence)
        XCTAssertLessThanOrEqual(abs(withVolume[0].targetWeight), 2.0)
    }

    func testXSMOMReferenceRanksNeutralizedPanelAndBalancesWeights() {
        let ranks = TrendXSMOMRankReference.ranks(inputs: [
            .init(symbol: "EURUSD", returns: [0.01, 0.02, 0.01], neutralizer: 1.0),
            .init(symbol: "GBPUSD", returns: [0.00, 0.01, 0.00], neutralizer: 0.5),
            .init(symbol: "USDJPY", returns: [-0.02, -0.01, -0.01], neutralizer: -1.0),
            .init(symbol: "AUDUSD", returns: [0.01, 0.01, 0.01], neutralizer: 0.9)
        ])

        XCTAssertEqual(ranks.count, 4)
        XCTAssertGreaterThanOrEqual(ranks[0].percentile, ranks[1].percentile)
        XCTAssertEqual(ranks.map(\.portfolioWeight).reduce(0.0, +), 0.0, accuracy: 1.0e-9)
        XCTAssertEqual(ranks.map { abs($0.portfolioWeight) }.reduce(0.0, +), 1.0, accuracy: 1.0e-9)
    }

    func testVolBreakoutReferenceComputesATRBandsStopsAndTargets() {
        let history = [
            TrendVolBreakoutReference.Bar(high: 10.1, low: 9.9, close: 10.0),
            TrendVolBreakoutReference.Bar(high: 10.3, low: 10.0, close: 10.2),
            TrendVolBreakoutReference.Bar(high: 10.5, low: 10.1, close: 10.4),
            TrendVolBreakoutReference.Bar(high: 10.7, low: 10.3, close: 10.6),
            TrendVolBreakoutReference.Bar(high: 11.1, low: 10.8, close: 11.0)
        ]

        let signal = TrendVolBreakoutReference.signal(bars: history, lookback: 3, atrPeriod: 3, bandMultiplier: 0.0)

        XCTAssertEqual(signal.direction, 1)
        XCTAssertGreaterThan(signal.atr, 0.0)
        XCTAssertLessThan(signal.stop, history.last?.close ?? 0.0)
        XCTAssertGreaterThan(signal.target, history.last?.close ?? 0.0)
    }
}
