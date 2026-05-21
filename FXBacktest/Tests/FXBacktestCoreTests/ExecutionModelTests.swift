import FXBacktestCore
import XCTest

final class ExecutionModelTests: XCTestCase {
    func testBacktestBrokerV2UsesPureClosePricesAndLedger() throws {
        let settings = BacktestRunSettings(
            initialDeposit: 10_000,
            contractSize: 100_000,
            lotSize: 1
        )
        let context = BacktestContext(settings: settings, digits: 5)
        var broker = BacktestBrokerV2(context: context)

        let positionID = try broker.openMarket(symbol: "EURUSD", side: .buy, midPrice: 100_000, lots: 1, openedAtUtc: 1)
        XCTAssertEqual(broker.positions.first?.entryPrice, 100_000)
        XCTAssertEqual(broker.balance, 10_000, accuracy: 0.0001)

        broker.markToMarket(symbol: "EURUSD", midPrice: 100_500)
        XCTAssertEqual(broker.equity, 10_500, accuracy: 0.0001)

        try broker.closePosition(id: positionID, midPrice: 101_000, closedAtUtc: 2)

        XCTAssertEqual(broker.positions.count, 0)
        XCTAssertEqual(broker.totalTrades, 1)
        XCTAssertEqual(broker.ledger.count, 1)
        XCTAssertEqual(broker.ledger[0].exitPrice, 101_000)
        XCTAssertEqual(broker.ledger[0].grossProfit, 1_000, accuracy: 0.0001)
        XCTAssertEqual(broker.ledger[0].netProfit, 1_000, accuracy: 0.0001)
        XCTAssertEqual(broker.netProfit, 1_000, accuracy: 0.0001)
    }
}
