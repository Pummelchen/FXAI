import FXImporter
import XCTest

final class MT5ImporterConnectorTests: XCTestCase {
    func testMT5RatesResponseMapsToImporterM1BatchWithZeroVolume() throws {
        let response = try JSONDecoder().decode(RatesResponseDTO.self, from: Data("""
        {
          "mt5_symbol": "EURUSD",
          "timeframe": "M1",
          "requested_from_mt5_server_ts": 60,
          "requested_to_mt5_server_ts_exclusive": 180,
          "series_synchronized": true,
          "rates": [
            {
              "mt5_server_time": 60,
              "open": "1.10000",
              "high": "1.10010",
              "low": "1.09990",
              "close": "1.10005"
            },
            {
              "mt5_server_time": 120,
              "open": "1.10005",
              "high": "1.10020",
              "low": "1.10000",
              "close": "1.10015"
            }
          ]
        }
        """.utf8))
        let request = FXImporterM1HistoryRequest(
            sourceSymbol: "EURUSD",
            fromSourceTimestamp: 60,
            toSourceTimestampExclusive: 180,
            maxBars: 2
        )

        let batch = try MT5ImporterConnector.makeBatch(request: request, response: response)

        XCTAssertEqual(batch.bars.count, 2)
        XCTAssertTrue(batch.sourceComplete)
        XCTAssertEqual(batch.bars[0].sourceTimestamp, 60)
        XCTAssertEqual(batch.bars[0].open, "1.10000")
        XCTAssertEqual(batch.bars[0].volume, 0)
        XCTAssertNil(batch.bars[0].utcTimestamp)
    }

    func testMT5ImporterRejectsMismatchedSymbol() throws {
        let response = try JSONDecoder().decode(RatesResponseDTO.self, from: Data("""
        {
          "mt5_symbol": "GBPUSD",
          "timeframe": "M1",
          "requested_from_mt5_server_ts": 60,
          "requested_to_mt5_server_ts_exclusive": 180,
          "rates": []
        }
        """.utf8))
        let request = FXImporterM1HistoryRequest(
            sourceSymbol: "EURUSD",
            fromSourceTimestamp: 60,
            toSourceTimestampExclusive: 180,
            maxBars: 2
        )

        XCTAssertThrowsError(try MT5ImporterConnector.makeBatch(request: request, response: response))
    }
}
