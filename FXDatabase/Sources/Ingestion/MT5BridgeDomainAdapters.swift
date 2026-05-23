import Domain
import MT5Bridge

public extension TerminalInfoDTO {
    func brokerServerIdentity() throws -> BrokerServerIdentity {
        try BrokerServerIdentity(company: company, server: server, accountLogin: accountLogin)
    }
}

public extension MT5RateDTO {
    func toClosedM1Bar(logicalSymbol: LogicalSymbol, mt5Symbol: MT5Symbol, digits: Digits) throws -> ClosedM1Bar {
        ClosedM1Bar(
            sourceOrigin: .mt5,
            logicalSymbol: logicalSymbol,
            mt5Symbol: mt5Symbol,
            timeframe: .m1,
            mt5ServerTime: MT5ServerSecond(rawValue: mt5ServerTime),
            open: try PriceScaled.fromDecimalString(open, digits: digits),
            high: try PriceScaled.fromDecimalString(high, digits: digits),
            low: try PriceScaled.fromDecimalString(low, digits: digits),
            close: try PriceScaled.fromDecimalString(close, digits: digits),
            volume: .zero,
            digits: digits
        )
    }
}
