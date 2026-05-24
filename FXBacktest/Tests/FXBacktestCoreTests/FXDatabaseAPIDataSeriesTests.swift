import FXBacktestAPI
import FXBacktestCore
import XCTest

final class FXDatabaseAPIDataSeriesTests: XCTestCase {
    func testOhlcDataSeriesBuildsFromFXDatabaseAPIResponse() throws {
        let response = FXBacktestM1HistoryResponse(
            metadata: FXBacktestM1HistoryMetadata(
                brokerSourceId: "demo",
                sourceOrigin: "YAHOO_FINANCE_HISTORY",
                logicalSymbol: "EURUSD",
                mt5Symbol: "EURUSD",
                digits: 5,
                requestedUtcStart: 1_704_067_200,
                requestedUtcEndExclusive: 1_704_067_320,
                firstUtc: 1_704_067_200,
                lastUtc: 1_704_067_260,
                rowCount: 2
            ),
            utcTimestamps: [1_704_067_200, 1_704_067_260],
            open: [108_000, 108_010],
            high: [108_020, 108_030],
            low: [107_990, 108_000],
            close: [108_010, 108_020],
            volume: [0, 42]
        )

        let series = try OhlcDataSeries(response: response)

        XCTAssertEqual(series.count, 2)
        XCTAssertEqual(series.metadata.brokerSourceId, "demo")
        XCTAssertEqual(series.metadata.sourceOrigin, "YAHOO_FINANCE_HISTORY")
        XCTAssertEqual(series.metadata.logicalSymbol, "EURUSD")
        XCTAssertEqual(series.metadata.mt5Symbol, "EURUSD")
        XCTAssertEqual(series.metadata.digits, 5)
        XCTAssertEqual(series.open[0], 108_000)
        XCTAssertEqual(series.close[1], 108_020)
        XCTAssertEqual(series.volume[1], 42)
        XCTAssertTrue(series.hasVolume)
    }

    func testFXDatabaseHistoryRequestPropagatesSourceOriginIntoVersionedAPIRequest() {
        let request = FXDatabaseHistoryRequest(
            brokerSourceId: "vendor",
            sourceOrigin: " yahoo_finance_history ",
            logicalSymbol: " aapl ",
            expectedMT5Symbol: " AAPL ",
            expectedDigits: 2,
            utcStartInclusive: 1_704_067_200,
            utcEndExclusive: 1_704_067_260,
            maximumRows: 1
        )

        XCTAssertEqual(request.sourceOrigin, "YAHOO_FINANCE_HISTORY")
        XCTAssertEqual(request.logicalSymbol, "AAPL")

        let apiRequest = FXBacktestM1HistoryRequest(
            brokerSourceId: request.brokerSourceId,
            sourceOrigin: request.sourceOrigin,
            logicalSymbol: request.logicalSymbol,
            utcStartInclusive: request.utcStartInclusive,
            utcEndExclusive: request.utcEndExclusive,
            expectedMT5Symbol: request.expectedMT5Symbol,
            expectedDigits: request.expectedDigits,
            maximumRows: request.maximumRows
        )

        XCTAssertEqual(apiRequest.apiVersion, FXBacktestAPIV1.latestVersion)
        XCTAssertEqual(apiRequest.sourceOrigin, "YAHOO_FINANCE_HISTORY")
        XCTAssertNoThrow(try apiRequest.validate())
    }

    func testOhlcDataSeriesRejectsInvalidDigits() throws {
        XCTAssertThrowsError(try OhlcDataSeries(
            metadata: FXBacktestMarketMetadata(
                brokerSourceId: "demo",
                logicalSymbol: "EURUSD",
                digits: -1
            ),
            utcTimestamps: [1_704_067_200],
            open: [108_000],
            high: [108_010],
            low: [107_990],
            close: [108_005]
        ))
    }

    func testOhlcDataSeriesRejectsNonMinuteTimestampAndNonPositivePrices() throws {
        XCTAssertThrowsError(try OhlcDataSeries(
            metadata: FXBacktestMarketMetadata(
                brokerSourceId: "demo",
                logicalSymbol: "EURUSD",
                digits: 5
            ),
            utcTimestamps: [1_704_067_201],
            open: [108_000],
            high: [108_010],
            low: [107_990],
            close: [108_005]
        ))

        XCTAssertThrowsError(try OhlcDataSeries(
            metadata: FXBacktestMarketMetadata(
                brokerSourceId: "demo",
                logicalSymbol: "EURUSD",
                digits: 5
            ),
            utcTimestamps: [1_704_067_200],
            open: [0],
            high: [108_010],
            low: [107_990],
            close: [108_005]
        ))
    }

}
