import XCTest
@testable import FXDataEngine

final class AuditContextSeriesTests: XCTestCase {
    func testRollingCorrelationUsesLegacyAsSeriesReturnWindowRules() {
        let close = [6.0, 5.0, 4.0, 3.0, 2.0, 1.0]
        XCTAssertEqual(
            AuditContextSeriesTools.rollingCorrelationAsSeries(close, close, startIndex: 0, width: 4),
            1.0,
            accuracy: 1e-12
        )
        XCTAssertEqual(
            AuditContextSeriesTools.rollingCorrelationAsSeries(close, close, startIndex: 0, width: 3),
            0.0,
            accuracy: 0.0
        )
        XCTAssertEqual(
            AuditContextSeriesTools.rollingCorrelationAsSeries(close, Array(close.dropLast()), startIndex: 0, width: 4),
            0.0,
            accuracy: 0.0
        )
    }

    func testChronologicalReversalAndAggregateCloseMatchAuditSeriesShape() {
        var bars: [AuditScenarioDoubleBar] = []
        for index in 0..<5 {
            let priceBase = Double(index)
            let timestamp = Int64(60 * (index + 1))
            let bar = AuditScenarioDoubleBar(
                timestampUTC: timestamp,
                open: priceBase + 10.0,
                high: priceBase + 11.0,
                low: priceBase + 9.0,
                close: priceBase + 20.0,
                volume: Double(index + 1) * 100.0,
                fillRiskPoints: Double(index + 1)
            )
            bars.append(bar)
        }

        let series = AuditContextSeriesTools.reverseChronologicalBarsToSeries(bars)
        XCTAssertTrue(series.isConsistent)
        XCTAssertEqual(series.timeUTC, [300, 240, 180, 120, 60])
        XCTAssertEqual(series.close, [24.0, 23.0, 22.0, 21.0, 20.0])
        XCTAssertEqual(series.volume, [500.0, 400.0, 300.0, 200.0, 100.0])
        XCTAssertEqual(series.fillRiskPoints, [5.0, 4.0, 3.0, 2.0, 1.0])

        let aggregate = AuditContextSeriesTools.aggregateCloseTimeframe(chronologicalBars: bars, step: 2)
        XCTAssertEqual(aggregate.timeUTC, [240, 120])
        XCTAssertEqual(aggregate.close, [23.0, 21.0])
        XCTAssertEqual(
            AuditContextSeriesTools.aggregateCloseTimeframe(chronologicalBars: bars, step: 0).close,
            []
        )
    }

    func testDeriveContextSeriesUsesFillRiskAndVolumeInsteadOfSpread() {
        let base = AuditAsSeriesOHLCV(
            timeUTC: [180, 120, 60],
            open: [1.2050, 1.1550, 1.1050],
            high: [1.2150, 1.1650, 1.1150],
            low: [1.1950, 1.1450, 1.0950],
            close: [1.2100, 1.1600, 1.1100],
            volume: [100.0, 80.0, 60.0],
            fillRiskPoints: [2.0, 1.5, 1.0]
        )

        let momentumContext = AuditContextSeriesTools.deriveContextSeriesFromBase(
            point: 0.0001,
            base: base,
            transformID: 0
        )
        XCTAssertTrue(momentumContext.isConsistent)
        XCTAssertEqual(momentumContext.count, base.count)
        XCTAssertEqual(momentumContext.timeUTC, base.timeUTC)

        for index in 0..<momentumContext.count {
            XCTAssertGreaterThan(momentumContext.open[index], 0.0)
            XCTAssertGreaterThanOrEqual(momentumContext.high[index], max(momentumContext.open[index], momentumContext.close[index]))
            XCTAssertLessThanOrEqual(momentumContext.low[index], min(momentumContext.open[index], momentumContext.close[index]))
            XCTAssertGreaterThan(momentumContext.volume[index], 0.0)
            XCTAssertGreaterThan(momentumContext.fillRiskPoints[index], 0.0)
        }

        XCTAssertGreaterThan(momentumContext.volume[0], base.volume[0])
        XCTAssertGreaterThan(momentumContext.fillRiskPoints[0], base.fillRiskPoints[0])

        let smoothedContext = AuditContextSeriesTools.deriveContextSeriesFromBase(
            point: 0.0001,
            base: base,
            transformID: 1
        )
        XCTAssertLessThan(smoothedContext.volume[0], base.volume[0])
        XCTAssertLessThan(smoothedContext.fillRiskPoints[0], momentumContext.fillRiskPoints[0])

        let inverseContext = AuditContextSeriesTools.deriveContextSeriesFromBase(
            point: 0.0001,
            base: base,
            transformID: 2
        )
        XCTAssertNotEqual(inverseContext.close[0], momentumContext.close[0])
    }

    func testAggregatedCandleStateUsesVolumeAwareLiquidityPressure() {
        let series = makeSyntheticAsSeries(count: 70, currentVolume: 10.0, olderVolume: 100.0)
        let state = AuditContextSeriesTools.computeAggregatedCandleLiquidityState(
            index: 0,
            windowBars: 5,
            series: series,
            point: 0.0001
        )
        XCTAssertNotNil(state)
        XCTAssertGreaterThan(state?.liquidityPressure ?? 0.0, 0.0)
        XCTAssertGreaterThanOrEqual(state?.rangePressure ?? -10.0, -6.0)
        XCTAssertLessThanOrEqual(state?.rangePressure ?? 10.0, 6.0)

        XCTAssertEqual(AuditContextSeriesTools.contextMTFBarsForSlot(0), 1)
        XCTAssertEqual(AuditContextSeriesTools.contextMTFBarsForSlot(4), 60)
        XCTAssertEqual(
            AuditContextSeriesTools.contextSlotMTFExtraIndex(slot: 0, timeframeSlot: 0, metric: .volumePressure),
            FXDataEngineConstants.contextMTFOffset + MTFStateMetric.volumePressure.rawValue
        )
    }

    func testBuildContextFeaturesPopulatesSharedAndMTFExtras() {
        let base = makeSyntheticAsSeries(count: 70, currentVolume: 50.0, olderVolume: 100.0)
        let context1 = AuditContextSeriesTools.deriveContextSeriesFromBase(point: 0.0001, base: base, transformID: 0)
        let context2 = AuditContextSeriesTools.deriveContextSeriesFromBase(point: 0.0001, base: base, transformID: 1)
        let context3 = AuditContextSeriesTools.deriveContextSeriesFromBase(point: 0.0001, base: base, transformID: 2)
        let features = AuditContextSeriesTools.buildContextFeatures(
            mainClose: base.close,
            point: 0.0001,
            contexts: [context1, context2, context3]
        )

        XCTAssertEqual(features.count, base.count)
        XCTAssertEqual(features.extra.count, base.count * FXDataEngineConstants.contextExtraFeatures)
        XCTAssertNotEqual(features.mean[0], 0.0)
        XCTAssertEqual(
            features.extraValue(sampleIndex: 0, featureIndex: FXDataEngineConstants.contextSharedOffset + 3),
            1.0,
            accuracy: 0.0
        )
        let mtfLiquidityIndex = AuditContextSeriesTools.contextSlotMTFExtraIndex(
            slot: 0,
            timeframeSlot: 0,
            metric: .volumePressure
        )
        XCTAssertNotNil(mtfLiquidityIndex)
        XCTAssertNotEqual(
            features.extraValue(sampleIndex: 0, featureIndex: mtfLiquidityIndex ?? -1),
            0.0
        )
    }

    private func makeSyntheticAsSeries(
        count: Int,
        currentVolume: Double,
        olderVolume: Double
    ) -> AuditAsSeriesOHLCV {
        var timeUTC: [Int64] = []
        var open: [Double] = []
        var high: [Double] = []
        var low: [Double] = []
        var close: [Double] = []
        var volume: [Double] = []
        var fillRiskPoints: [Double] = []

        for index in 0..<count {
            let price = 1.2000 - Double(index) * 0.0007
            timeUTC.append(Int64(1_704_067_200 - index * 60))
            open.append(price - 0.0002)
            high.append(price + 0.0005)
            low.append(price - 0.0006)
            close.append(price)
            volume.append(index < 5 ? currentVolume : olderVolume)
            fillRiskPoints.append(1.0 + (index < 5 ? 0.1 : 0.0))
        }

        return AuditAsSeriesOHLCV(
            timeUTC: timeUTC,
            open: open,
            high: high,
            low: low,
            close: close,
            volume: volume,
            fillRiskPoints: fillRiskPoints
        )
    }
}
