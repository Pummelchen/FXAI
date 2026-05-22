import FXBacktestAPI
import XCTest
@testable import FXDataEngine

final class MarketDataGatewayTests: XCTestCase {
    func testFXDatabaseMarketHistoryRequestNormalizesProviderContract() throws {
        let request = FXDatabaseMarketHistoryRequest(
            brokerSourceId: " demo-broker ",
            sourceOrigin: "dukascopy",
            logicalSymbol: " eurusd ",
            expectedProviderSymbol: " EUR/USD ",
            expectedDigits: 5,
            utcStartInclusive: 1_704_067_200,
            utcEndExclusive: 1_704_067_800,
            maximumRows: 1_000
        )

        let apiRequest = request.apiRequest()

        XCTAssertEqual(apiRequest.brokerSourceId, "demo-broker")
        XCTAssertEqual(apiRequest.sourceOrigin, "DUKASCOPY")
        XCTAssertEqual(apiRequest.logicalSymbol, "EURUSD")
        XCTAssertEqual(apiRequest.expectedMT5Symbol, "EUR/USD")
        XCTAssertEqual(apiRequest.expectedDigits, 5)
        XCTAssertEqual(apiRequest.maximumRows, 1_000)
        try apiRequest.validate()
    }

    func testFXDatabaseMarketUniverseRequestBuildsPrimaryAndDedupedContextRequests() throws {
        let request = FXDatabaseMarketUniverseRequest(
            brokerSourceId: " demo-broker ",
            sourceOrigin: "dukascopy",
            primarySymbol: " eurusd ",
            contextSymbols: ["USDJPY", "eurusd", " gbpusd ", "USDJPY", ""],
            expectedProviderSymbolsBySymbol: [
                "eurusd": "EUR/USD",
                "usdjpy": "USD/JPY",
                " ": "ignored"
            ],
            expectedDigitsBySymbol: [
                "EURUSD": 5,
                "USDJPY": 3,
                "GBPUSD": 5
            ],
            utcStartInclusive: 1_704_067_200,
            utcEndExclusive: 1_704_067_800,
            maximumRowsPerSymbol: 2_000,
            requireAlignedTimestamps: false
        )

        let historyRequests = try request.historyRequests()

        XCTAssertEqual(request.symbols, ["EURUSD", "USDJPY", "GBPUSD"])
        XCTAssertEqual(historyRequests.map(\.logicalSymbol), ["EURUSD", "USDJPY", "GBPUSD"])
        XCTAssertEqual(historyRequests.map(\.sourceOrigin), ["DUKASCOPY", "DUKASCOPY", "DUKASCOPY"])
        XCTAssertEqual(historyRequests[0].expectedProviderSymbol, "EUR/USD")
        XCTAssertEqual(historyRequests[1].expectedProviderSymbol, "USD/JPY")
        XCTAssertNil(historyRequests[2].expectedProviderSymbol)
        XCTAssertEqual(historyRequests[1].expectedDigits, 3)
        XCTAssertEqual(historyRequests[2].maximumRows, 2_000)
        XCTAssertFalse(request.requireAlignedTimestamps)
        for apiRequest in historyRequests.map({ $0.apiRequest() }) {
            try apiRequest.validate()
        }
    }

    func testFXDatabaseMarketUniverseRequestRejectsInvalidRangeAndDigits() {
        let invalidRange = FXDatabaseMarketUniverseRequest(
            brokerSourceId: "demo",
            primarySymbol: "EURUSD",
            expectedDigitsBySymbol: ["EURUSD": 5],
            utcStartInclusive: 1_704_067_800,
            utcEndExclusive: 1_704_067_200
        )
        XCTAssertThrowsError(try invalidRange.historyRequests())

        let invalidDigits = FXDatabaseMarketUniverseRequest(
            brokerSourceId: "demo",
            primarySymbol: "EURUSD",
            expectedDigitsBySymbol: ["EURUSD": 11],
            utcStartInclusive: 1_704_067_200,
            utcEndExclusive: 1_704_067_800
        )
        XCTAssertThrowsError(try invalidDigits.historyRequests())
    }

    func testM1SeriesBuildsFromFXDatabaseResponseWithNonMT5Volume() throws {
        let response = makeResponse(sourceOrigin: "DUKASCOPY", providerSymbol: "EUR/USD", count: 4)
        let series = try M1OHLCVSeries(response: response)

        XCTAssertEqual(series.metadata.brokerSourceId, "demo")
        XCTAssertEqual(series.metadata.sourceOrigin, "DUKASCOPY")
        XCTAssertEqual(series.metadata.logicalSymbol, "EURUSD")
        XCTAssertEqual(series.metadata.providerSymbol, "EUR/USD")
        XCTAssertEqual(series.metadata.timeframe, .m1)
        XCTAssertEqual(series.count, 4)
        XCTAssertTrue(series.hasVolume)
        XCTAssertEqual(series.volume[3], 103)
    }

    func testMarketGatewayIndexesAndSlicesCanonicalAscendingBars() throws {
        let series = try M1OHLCVSeries(response: makeResponse(count: 8))
        let start = series.utcTimestamps[0]

        XCTAssertEqual(MarketDataGateway.barIndex(in: series, atOrBefore: start + 3 * 60 + 15), 3)
        XCTAssertEqual(MarketDataGateway.barIndex(in: series, atOrBefore: start + 3 * 60 + 15, exact: true), -1)
        XCTAssertEqual(MarketDataGateway.closedBarPosition(in: series, atOrBefore: start + 3 * 60, exact: true), 4)

        let timeSlice = try MarketDataGateway.bars(
            in: series,
            fromUTCInclusive: start + 2 * 60,
            toUTCExclusive: start + 5 * 60
        )
        XCTAssertEqual(Array(timeSlice.utcTimestamps), Array(series.utcTimestamps[2..<5]))
        XCTAssertEqual(timeSlice.metadata.firstUTC, series.utcTimestamps[2])
        XCTAssertEqual(timeSlice.metadata.lastUTC, series.utcTimestamps[4])
        XCTAssertEqual(timeSlice.volume[2], series.volume[4])

        let window = try MarketDataGateway.closedWindow(in: series, endingAt: 5, count: 3)
        XCTAssertEqual(Array(window.close), Array(series.close[3..<6]))
        XCTAssertEqual(window.metadata.firstUTC, series.utcTimestamps[3])
        XCTAssertEqual(window.metadata.lastUTC, series.utcTimestamps[5])
    }

    func testMarketGatewayRejectsInsufficientClosedWindow() throws {
        let series = try M1OHLCVSeries(response: makeResponse(count: 4))

        XCTAssertThrowsError(try MarketDataGateway.closedWindow(in: series, endingAt: 1, count: 3)) { error in
            XCTAssertEqual(
                error as? FXDataEngineError,
                .insufficientData("market data window ending at 1 needs 3 bars")
            )
        }
    }

    func testMarketGatewayResamplesM1ToM5WithSummedVolumeAndDropsPartialByDefault() throws {
        let series = try M1OHLCVSeries(response: makeResponse(count: 12))
        let m5 = try MarketDataGateway.resample(series, to: .m5)

        XCTAssertEqual(m5.metadata.timeframe, .m5)
        XCTAssertEqual(m5.count, 2)
        XCTAssertEqual(m5.utcTimestamps[0], series.utcTimestamps[0])
        XCTAssertEqual(m5.open[0], series.open[0])
        XCTAssertEqual(m5.close[0], series.close[4])
        XCTAssertEqual(m5.high[0], series.high[4])
        XCTAssertEqual(m5.low[0], series.low[0])
        XCTAssertEqual(m5.volume[0], 510)
        XCTAssertEqual(m5.open[1], series.open[5])
        XCTAssertEqual(m5.close[1], series.close[9])
        XCTAssertEqual(m5.volume[1], 535)
        XCTAssertTrue(m5.hasVolume)
    }

    func testMarketGatewayResamplingCanIncludePartialBucketsAndAlignToM1() throws {
        let series = try M1OHLCVSeries(response: makeResponse(count: 12))
        let m5 = try MarketDataGateway.resample(series, to: .m5, includePartialBuckets: true)
        let map = MarketDataGateway.alignedIndexMap(
            referenceM1: series,
            target: m5,
            maxLagSeconds: 299,
            upToIndex: 8
        )

        XCTAssertEqual(m5.count, 3)
        XCTAssertEqual(m5.utcTimestamps[2], series.utcTimestamps[10])
        XCTAssertEqual(m5.close[2], series.close[11])
        XCTAssertEqual(m5.volume[2], 221)
        XCTAssertEqual(Array(map[0...8]), [0, 0, 0, 0, 0, 1, 1, 1, 1])
        XCTAssertEqual(map[9], -1)
    }

    private func makeResponse(
        sourceOrigin: String = "MT5",
        providerSymbol: String = "EURUSD",
        count: Int
    ) -> FXBacktestM1HistoryResponse {
        let start = Int64(1_704_067_200)
        let timestamps = (0..<count).map { start + Int64($0 * 60) }
        let open = (0..<count).map { 108_000 + Int64($0 * 10) }
        let close = open.map { $0 + 4 }
        let high = close.map { $0 + 6 }
        let low = open.map { $0 - 6 }
        let volume = (0..<count).map { UInt64(100 + $0) }
        return FXBacktestM1HistoryResponse(
            metadata: FXBacktestM1HistoryMetadata(
                brokerSourceId: "demo",
                sourceOrigin: sourceOrigin,
                logicalSymbol: "EURUSD",
                mt5Symbol: providerSymbol,
                digits: 5,
                requestedUtcStart: start,
                requestedUtcEndExclusive: start + Int64(max(count, 1) * 60),
                firstUtc: timestamps.first,
                lastUtc: timestamps.last,
                rowCount: count
            ),
            utcTimestamps: timestamps,
            open: open,
            high: high,
            low: low,
            close: close,
            volume: volume
        )
    }
}
