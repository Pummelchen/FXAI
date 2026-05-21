import Domain
import Foundation

public struct VerificationBar: Equatable, Sendable {
    public let brokerSourceId: BrokerSourceId
    public let sourceOrigin: DataSourceOrigin
    public let logicalSymbol: LogicalSymbol
    public let mt5Symbol: MT5Symbol
    public let mt5ServerTime: MT5ServerSecond
    public let utcTime: UtcSecond
    public let open: PriceScaled
    public let high: PriceScaled
    public let low: PriceScaled
    public let close: PriceScaled
    public let volume: M1Volume
    public let digits: Digits
    public let offsetConfidence: OffsetConfidence
    public let barHash: BarHash

    public init(
        brokerSourceId: BrokerSourceId,
        sourceOrigin: DataSourceOrigin = .mt5,
        logicalSymbol: LogicalSymbol,
        mt5Symbol: MT5Symbol,
        mt5ServerTime: MT5ServerSecond,
        utcTime: UtcSecond,
        open: PriceScaled,
        high: PriceScaled,
        low: PriceScaled,
        close: PriceScaled,
        volume: M1Volume = .zero,
        digits: Digits,
        offsetConfidence: OffsetConfidence = .verified,
        barHash: BarHash
    ) {
        self.brokerSourceId = brokerSourceId
        self.sourceOrigin = sourceOrigin
        self.logicalSymbol = logicalSymbol
        self.mt5Symbol = mt5Symbol
        self.mt5ServerTime = mt5ServerTime
        self.utcTime = utcTime
        self.open = open
        self.high = high
        self.low = low
        self.close = close
        self.volume = volume
        self.digits = digits
        self.offsetConfidence = offsetConfidence
        self.barHash = barHash
    }

    public init(validatedBar: ValidatedBar) {
        self.init(
            brokerSourceId: validatedBar.brokerSourceId,
            sourceOrigin: validatedBar.sourceOrigin,
            logicalSymbol: validatedBar.logicalSymbol,
            mt5Symbol: validatedBar.mt5Symbol,
            mt5ServerTime: validatedBar.mt5ServerTime,
            utcTime: validatedBar.utcTime,
            open: validatedBar.open,
            high: validatedBar.high,
            low: validatedBar.low,
            close: validatedBar.close,
            volume: validatedBar.volume,
            digits: validatedBar.digits,
            offsetConfidence: validatedBar.offsetConfidence,
            barHash: validatedBar.barHash
        )
    }
}
