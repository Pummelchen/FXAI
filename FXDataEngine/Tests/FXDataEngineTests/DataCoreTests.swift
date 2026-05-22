import XCTest
@testable import FXDataEngine

final class DataCoreTests: XCTestCase {
    func testDataCoreRequestNormalizesAndDeduplicatesContextSymbols() {
        var request = DataCoreRequest(
            symbol: " eurusd ",
            neededBars: 20,
            contextSymbols: ["", " usdjpy ", "USDJPY", "gbpusd", "EURUSD"]
        )

        XCTAssertEqual(request.symbol, "EURUSD")
        XCTAssertEqual(request.contextSymbols, ["USDJPY", "GBPUSD"])
        XCTAssertFalse(request.addContextSymbol("usdjpy"))
        XCTAssertFalse(request.addContextSymbol("eurusd"))
        XCTAssertTrue(request.addContextSymbol("audusd"))
        XCTAssertEqual(request.contextSymbols, ["USDJPY", "GBPUSD", "AUDUSD"])
    }

    func testDataCoreTimeframeNeedsAndLagsMatchLegacyControlHelpers() {
        XCTAssertEqual(DataCoreTimeframeNeeds.legacy(neededBars: 120), DataCoreTimeframeNeeds(m5: 220, m15: 220, m30: 220, h1: 220))
        XCTAssertEqual(DataCoreTimeframeNeeds.legacy(neededBars: 3_000), DataCoreTimeframeNeeds(m5: 680, m15: 280, m30: 220, h1: 220))

        let lags = DataCoreTimeframeLags.legacy
        XCTAssertEqual(lags.m5Seconds, 600)
        XCTAssertEqual(lags.m15Seconds, 1_800)
        XCTAssertEqual(lags.m30Seconds, 3_600)
        XCTAssertEqual(lags.h1Seconds, 7_200)
    }

    func testDataCoreAlignmentMapsAscendingM1SeriesWithLag() {
        let reference = ContiguousArray<Int64>([60, 120, 180, 240, 300])
        let target = ContiguousArray<Int64>([60, 180, 300])
        let map = DataCoreAlignment.buildAlignedIndexMap(
            referenceTimesAscending: reference,
            targetTimesAscending: target,
            maxLagSeconds: 90,
            upToIndex: 4
        )

        XCTAssertEqual(map, [0, 0, 1, 1, 2])
        XCTAssertEqual(
            DataCoreAlignment.findAlignedIndex(targetTimesAscending: target, referenceTimeUTC: 240, maxLagSeconds: 30),
            -1
        )
        XCTAssertEqual(
            DataCoreAlignment.findAlignedIndex(targetTimesAscending: target, referenceTimeUTC: 240, maxLagSeconds: 90),
            1
        )
        XCTAssertEqual(
            DataCoreAlignment.alignedFreshnessWeight(
                targetTimesAscending: target,
                targetIndex: 1,
                referenceTimeUTC: 240,
                maxLagSeconds: 120
            ),
            0.50,
            accuracy: 1e-12
        )
        XCTAssertEqual(
            DataCoreAlignment.alignedFreshnessWeight(
                targetTimesAscending: target,
                targetIndex: 1,
                referenceTimeUTC: 240,
                maxLagSeconds: 30
            ),
            0.0,
            accuracy: 0.0
        )
        XCTAssertEqual(
            DataCoreAlignment.movePoints(priceNow: 1.1000, priceFuture: 1.1015, point: 0.0001),
            15.0,
            accuracy: 1e-9
        )
    }

    func testDataCoreBuildsContextAggregatesForBundle() throws {
        let primary = try makeSeries(symbol: "EURUSD", drift: 2, count: 120)
        let risingContext = try makeSeries(symbol: "USDJPY", drift: 4, count: 120)
        let fallingContext = try makeSeries(symbol: "GBPUSD", drift: -2, count: 120)
        let universe = try MarketUniverse(primarySymbol: "EURUSD", series: [primary, risingContext, fallingContext])

        let bundle = try DataCore().buildBundle(
            request: DataCoreRequest(
                symbol: "EURUSD",
                neededBars: 80,
                contextSymbols: ["USDJPY", "GBPUSD"]
            ),
            universe: universe
        )

        let index = bundle.sampleIndex
        XCTAssertTrue(bundle.ready)
        XCTAssertEqual(bundle.contextAggregates.mean.count, primary.count)
        XCTAssertNotEqual(bundle.contextAggregates.mean[index], 0.0)
        XCTAssertGreaterThan(bundle.contextAggregates.standardDeviation[index], 0.0)
        XCTAssertGreaterThan(bundle.contextAggregates.extraValue(sampleIndex: index, featureIndex: 0), 0.0)
        XCTAssertEqual(
            bundle.contextAggregates.extraValue(
                sampleIndex: index,
                featureIndex: FXDataEngineConstants.contextSharedOffset + 3
            ),
            1.0,
            accuracy: 1e-9
        )
        XCTAssertTrue(bundle.contextAggregates.symbolReady[0])
    }

    func testDataCoreContextAggregatesDefaultClosedWhenNoContextExists() throws {
        let primary = try makeSeries(symbol: "EURUSD", drift: 2, count: 90)
        let universe = try MarketUniverse(primarySymbol: "EURUSD", series: [primary])
        let bundle = try DataCore().buildBundle(
            request: DataCoreRequest(symbol: "EURUSD", neededBars: 80),
            universe: universe
        )

        XCTAssertEqual(bundle.contextAggregates.mean[bundle.sampleIndex], 0.0)
        XCTAssertEqual(bundle.contextAggregates.upRatio[bundle.sampleIndex], 0.5)
        XCTAssertFalse(bundle.contextAggregates.symbolReady.contains(true))
    }

    private func makeSeries(symbol: String, drift: Int64, count: Int) throws -> M1OHLCVSeries {
        let start = Int64(1_704_067_200)
        var utc = ContiguousArray<Int64>()
        var open = ContiguousArray<Int64>()
        var high = ContiguousArray<Int64>()
        var low = ContiguousArray<Int64>()
        var close = ContiguousArray<Int64>()
        var volume = ContiguousArray<UInt64>()
        var price = Int64(108_000)

        for index in 0..<count {
            let wave = Int64((index % 7) - 3)
            let next = max(90_000, price + drift + wave)
            utc.append(start + Int64(index * 60))
            open.append(price)
            high.append(max(price, next) + 5)
            low.append(min(price, next) - 5)
            close.append(next)
            volume.append(UInt64(100 + (index % 13)))
            price = next
        }

        return try M1OHLCVSeries(
            metadata: FXMarketMetadata(
                brokerSourceId: "test",
                sourceOrigin: "TEST",
                logicalSymbol: symbol,
                providerSymbol: symbol,
                digits: 5,
                firstUTC: utc.first,
                lastUTC: utc.last
            ),
            utcTimestamps: utc,
            open: open,
            high: high,
            low: low,
            close: close,
            volume: volume
        )
    }
}
