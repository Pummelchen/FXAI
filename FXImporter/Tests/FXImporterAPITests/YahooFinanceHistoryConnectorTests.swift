import FXImporter
import XCTest

final class YahooFinanceHistoryConnectorTests: XCTestCase {
    func testDescriptorExposesD1HistoricalVolumeCapabilityOnly() {
        let connector = YahooFinanceHistoryConnector()

        XCTAssertEqual(connector.descriptor.kind, .yahooFinanceHistory)
        XCTAssertFalse(connector.descriptor.capabilities.supportsHistoricalM1OHLC)
        XCTAssertTrue(connector.descriptor.capabilities.supportsHistoricalD1OHLC)
        XCTAssertFalse(connector.descriptor.capabilities.supportsLiveM1OHLC)
        XCTAssertTrue(connector.descriptor.capabilities.providesVolume)
    }

    func testBuildsYahooChartURLForDailyHistory() throws {
        let request = FXImporterD1HistoryRequest(
            sourceSymbol: "BRK-B",
            fromSourceTimestamp: 1_704_067_200,
            toSourceTimestampExclusive: 1_704_326_400,
            maxBars: 100
        )

        let url = try YahooFinanceHistoryConnector.makeChartURL(
            baseURL: URL(string: "https://example.test")!,
            request: request
        )
        let text = url.absoluteString

        XCTAssertTrue(text.hasPrefix("https://example.test/v8/finance/chart/BRK-B?"))
        XCTAssertTrue(text.contains("period1=1704067200"))
        XCTAssertTrue(text.contains("period2=1704326400"))
        XCTAssertTrue(text.contains("interval=1d"))
        XCTAssertTrue(text.contains("events=history"))
        XCTAssertTrue(text.contains("includeAdjustedClose=true"))
    }

    func testBuildsYahooChartURLWithoutDoubleEncodingFXSymbols() throws {
        let request = FXImporterD1HistoryRequest(
            sourceSymbol: "EURUSD=X",
            fromSourceTimestamp: 1_704_067_200,
            toSourceTimestampExclusive: 1_704_326_400,
            maxBars: 100
        )

        let url = try YahooFinanceHistoryConnector.makeChartURL(
            baseURL: URL(string: "https://example.test")!,
            request: request
        )
        let text = url.absoluteString

        XCTAssertTrue(text.hasPrefix("https://example.test/v8/finance/chart/EURUSD%3DX?"))
        XCTAssertFalse(text.contains("EURUSD%253DX"))
    }

    func testBuildsYahooChartRequestWithAuditableHeaders() throws {
        let request = FXImporterD1HistoryRequest(
            sourceSymbol: "AAPL",
            fromSourceTimestamp: 1_704_067_200,
            toSourceTimestampExclusive: 1_704_326_400,
            maxBars: 100
        )

        let urlRequest = try YahooFinanceHistoryConnector.makeChartRequest(
            baseURL: URL(string: "https://example.test")!,
            request: request
        )

        XCTAssertEqual(urlRequest.value(forHTTPHeaderField: "Accept"), "application/json")
        XCTAssertEqual(urlRequest.value(forHTTPHeaderField: "User-Agent"), "FXAI-FXImporter/0.1")
        XCTAssertEqual(urlRequest.timeoutInterval, 30)
    }

    func testParsesDailyChartResponseIntoD1Batch() throws {
        let request = FXImporterD1HistoryRequest(
            sourceSymbol: "AAPL",
            fromSourceTimestamp: 1_704_067_200,
            toSourceTimestampExclusive: 1_704_326_400,
            maxBars: 10
        )

        let batch = try YahooFinanceHistoryConnector.makeBatch(
            request: request,
            data: Self.chartFixture.data(using: .utf8)!
        )

        XCTAssertTrue(batch.sourceComplete)
        XCTAssertEqual(batch.bars.count, 2)
        XCTAssertEqual(batch.bars[0].sourceSymbol, "AAPL")
        XCTAssertEqual(batch.bars[0].sourceTimestamp, 1_704_067_200)
        XCTAssertEqual(batch.bars[0].utcTimestamp, 1_704_067_200)
        XCTAssertEqual(batch.bars[0].open, "187.15")
        XCTAssertEqual(batch.bars[0].high, "188.44")
        XCTAssertEqual(batch.bars[0].low, "183.885")
        XCTAssertEqual(batch.bars[0].close, "185.64")
        XCTAssertEqual(batch.bars[0].adjustedClose, "184.9384155273")
        XCTAssertEqual(batch.bars[0].volume, 82_400_000)
    }

    func testRejectsMalformedDailyOHLCInvariant() throws {
        let request = FXImporterD1HistoryRequest(
            sourceSymbol: "AAPL",
            fromSourceTimestamp: 1_704_067_200,
            toSourceTimestampExclusive: 1_704_326_400,
            maxBars: 10
        )

        XCTAssertThrowsError(try YahooFinanceHistoryConnector.makeBatch(
            request: request,
            data: Self.invalidOHLCChartFixture.data(using: .utf8)!
        )) { error in
            XCTAssertTrue(String(describing: error).contains("OHLC invariant failed"))
        }
    }

    func testFetchD1HistoryUsesInjectedLoaderAndRejectsHTTPError() async throws {
        let connector = YahooFinanceHistoryConnector(
            baseURL: URL(string: "https://example.test")!,
            loadData: { request in
                XCTAssertEqual(request.url?.host, "example.test")
                XCTAssertEqual(request.value(forHTTPHeaderField: "Accept"), "application/json")
                return (Data(), 429)
            }
        )
        let request = FXImporterD1HistoryRequest(
            sourceSymbol: "AAPL",
            fromSourceTimestamp: 1_704_067_200,
            toSourceTimestampExclusive: 1_704_326_400,
            maxBars: 10
        )

        do {
            _ = try await connector.fetchD1History(request)
            XCTFail("Expected HTTP error")
        } catch let error as YahooFinanceHistoryConnectorError {
            XCTAssertEqual(error.description, "Yahoo Finance history request failed with HTTP status 429.")
        }
    }

    func testM1HistoryFailsClosedForYahooConnector() async throws {
        let connector = YahooFinanceHistoryConnector()
        let request = FXImporterM1HistoryRequest(
            sourceSymbol: "AAPL",
            fromSourceTimestamp: 1,
            toSourceTimestampExclusive: 2,
            maxBars: 1
        )

        do {
            _ = try await connector.fetchM1History(request)
            XCTFail("Expected unsupported M1 history error")
        } catch let error as YahooFinanceHistoryConnectorError {
            XCTAssertEqual(error.description, "Yahoo Finance connector supports historical D1 OHLCV only, not M1 history.")
        }
    }

    private static let chartFixture = """
    {
      "chart": {
        "result": [
          {
            "timestamp": [1704067200, 1704153600],
            "indicators": {
              "quote": [
                {
                  "open": [187.15, 184.22],
                  "high": [188.44, 185.88],
                  "low": [183.885, 183.43],
                  "close": [185.64, 185.59],
                  "volume": [82400000, 58400000]
                }
              ],
              "adjclose": [
                {
                  "adjclose": [184.9384155273, 184.8885955811]
                }
              ]
            }
          }
        ],
        "error": null
      }
    }
    """

    private static let invalidOHLCChartFixture = """
    {
      "chart": {
        "result": [
          {
            "timestamp": [1704067200],
            "indicators": {
              "quote": [
                {
                  "open": [187.15],
                  "high": [186.44],
                  "low": [183.885],
                  "close": [185.64],
                  "volume": [82400000]
                }
              ]
            }
          }
        ],
        "error": null
      }
    }
    """
}
