import XCTest
@testable import FXDataEngine

final class FeatureMathTests: XCTestCase {
    func testBasicAsSeriesReturnSlopeAndMoveEstimatorsMatchLegacyShape() {
        let close = [130.0, 120.0, 110.0, 100.0, 90.0]

        XCTAssertEqual(FeatureMath.safeReturn(close, currentIndex: 0, previousIndex: 1), 10.0 / 120.0, accuracy: 1e-12)
        XCTAssertEqual(FeatureMath.safeReturn(close, currentIndex: -1, previousIndex: 1), 0.0, accuracy: 0.0)
        XCTAssertEqual(FeatureMath.normalizedSlopeAsSeries(close, startIndex: 0, width: 3), 10.0 / 130.0, accuracy: 1e-12)
        XCTAssertEqual(
            FeatureMath.estimateExpectedAbsMovePointsAsSeries(
                close: close,
                horizonM1: 1,
                sampleCount: 3,
                point: 1.0
            ),
            10.0,
            accuracy: 1e-12
        )
        XCTAssertEqual(
            FeatureMath.estimateExpectedAbsMovePointsAtIndexAsSeries(
                close: close,
                startIndex: 1,
                horizonM1: 1,
                sampleCount: 2,
                point: 1.0
            ),
            10.0,
            accuracy: 1e-12
        )
    }

    func testRollingReturnAndMovingAverageHelpersMatchLegacyMath() {
        let close = [110.0, 105.0, 100.0, 95.0]
        let expectedAbsReturn = ((5.0 / 105.0) + (5.0 / 100.0) + (5.0 / 95.0)) / 3.0
        let returns = [
            5.0 / 105.0,
            5.0 / 100.0,
            5.0 / 95.0
        ]
        let mean = returns.reduce(0.0, +) / 3.0
        let expectedStd = sqrt(returns.reduce(0.0) { $0 + ($1 - mean) * ($1 - mean) } / 3.0)

        XCTAssertEqual(FeatureMath.rollingAbsReturnAsSeries(close, startIndex: 0, width: 3), expectedAbsReturn, accuracy: 1e-12)
        XCTAssertEqual(FeatureMath.rollingReturnStdAsSeries(close, startIndex: 0, width: 3), expectedStd, accuracy: 1e-12)
        XCTAssertEqual(FeatureMath.smaAsSeries(close, startIndex: 0, period: 3), 105.0, accuracy: 1e-12)
        XCTAssertEqual(FeatureMath.emaAsSeries(close, startIndex: 0, period: 3), 106.25, accuracy: 1e-12)
        XCTAssertEqual(
            FeatureMath.movingAverageEdgeFeature(refPrice: 110.0, movingAverage: 100.0, volatilityUnit: 0.02),
            5.0,
            accuracy: 1e-12
        )
    }

    func testCandleGeometryAndMomentumIndicatorsMatchLegacyMath() {
        let candle = FeatureMath.candleGeometryNormalize(
            open: 100.0,
            high: 110.0,
            low: 90.0,
            close: 105.0,
            previousClose: 100.0
        )
        XCTAssertEqual(candle.bodyNorm, 0.05, accuracy: 1e-12)
        XCTAssertEqual(candle.upperWickNorm, 0.25, accuracy: 1e-12)
        XCTAssertEqual(candle.lowerWickNorm, 0.5, accuracy: 1e-12)
        XCTAssertEqual(candle.rangeNorm, 0.2, accuracy: 1e-12)

        let rising = [110.0, 105.0, 100.0, 96.0]
        let falling = [96.0, 100.0, 105.0, 110.0]
        XCTAssertEqual(FeatureMath.rsiAsSeries(rising, startIndex: 0, period: 3), 100.0, accuracy: 1e-12)
        XCTAssertEqual(FeatureMath.rsiAsSeries(falling, startIndex: 0, period: 3), 0.0, accuracy: 1e-12)
        XCTAssertEqual(FeatureMath.rsiAsSeries([100.0, 100.0, 100.0, 100.0], startIndex: 0, period: 3), 50.0, accuracy: 1e-12)
    }

    func testVolatilityAndRobustFilterHelpersMatchLegacyMath() {
        let open = [109.0, 105.0, 100.0, 96.0]
        let high = [112.0, 106.0, 102.0, 98.0]
        let low = [108.0, 104.0, 99.0, 95.0]
        let close = [110.0, 105.0, 100.0, 96.0]

        XCTAssertEqual(FeatureMath.atrAsSeries(high: high, low: low, close: close, startIndex: 0, period: 3), 19.0 / 3.0, accuracy: 1e-12)

        let parkinsonSum = pow(log(112.0 / 108.0), 2.0) +
            pow(log(106.0 / 104.0), 2.0) +
            pow(log(102.0 / 99.0), 2.0)
        let expectedParkinson = sqrt(parkinsonSum / (4.0 * log(2.0) * 3.0))
        XCTAssertEqual(FeatureMath.parkinsonVolAsSeries(high: high, low: low, startIndex: 0, period: 3), expectedParkinson, accuracy: 1e-12)

        var rogersSatchellSum = 0.0
        for index in 0..<3 {
            let highClose = log(high[index] / close[index])
            let highOpen = log(high[index] / open[index])
            let lowClose = log(low[index] / close[index])
            let lowOpen = log(low[index] / open[index])
            rogersSatchellSum += max((highClose * highOpen) + (lowClose * lowOpen), 0.0)
        }
        let expectedRS = sqrt(rogersSatchellSum / 3.0)
        XCTAssertEqual(
            FeatureMath.rogersSatchellVolAsSeries(open: open, high: high, low: low, close: close, startIndex: 0, period: 3),
            expectedRS,
            accuracy: 1e-12
        )

        let coefficient = (2.0 * log(2.0)) - 1.0
        var garmanKlassSum = 0.0
        for index in 0..<3 {
            let highLow = log(high[index] / low[index])
            let closeOpen = log(close[index] / open[index])
            garmanKlassSum += max(0.5 * highLow * highLow - coefficient * closeOpen * closeOpen, 0.0)
        }
        let expectedGK = sqrt(garmanKlassSum / 3.0)
        XCTAssertEqual(
            FeatureMath.garmanKlassVolAsSeries(open: open, high: high, low: low, close: close, startIndex: 0, period: 3),
            expectedGK,
            accuracy: 1e-12
        )

        XCTAssertEqual(FeatureMath.rollingMedianAsSeries([5.0, 1.0, 9.0, 3.0], startIndex: 0, period: 4), 4.0, accuracy: 1e-12)
        XCTAssertEqual(FeatureMath.rollingMADAsSeries([5.0, 1.0, 9.0, 3.0], startIndex: 0, period: 4, median: 4.0), 2.0, accuracy: 1e-12)
    }

    func testAdvancedSmoothersPreserveConstantSeries() {
        let constant = Array(repeating: 100.0, count: 40)
        XCTAssertEqual(FeatureMath.qsdemaAsSeries(constant, startIndex: 0, period: 5), 100.0, accuracy: 1e-9)
        XCTAssertEqual(FeatureMath.kalmanEstimateAsSeries(constant, startIndex: 0, period: 10), 100.0, accuracy: 1e-9)
        XCTAssertEqual(FeatureMath.ehlersSuperSmootherAsSeries(constant, startIndex: 0, period: 10), 100.0, accuracy: 1e-9)
    }
}
