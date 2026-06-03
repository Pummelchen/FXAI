import Domain
import Foundation

public struct BarSeriesMetadata: Hashable, Sendable {
    public let brokerSourceId: BrokerSourceId
    public let sourceOrigin: DataSourceOrigin
    public let logicalSymbol: LogicalSymbol
    public let timeframe: Timeframe
    public let digits: Digits
    public let requestedUtcStart: UtcSecond?
    public let requestedUtcEndExclusive: UtcSecond?
    public let firstUtc: UtcSecond?
    public let lastUtc: UtcSecond?

    public init(
        brokerSourceId: BrokerSourceId,
        sourceOrigin: DataSourceOrigin = .mt5,
        logicalSymbol: LogicalSymbol,
        digits: Digits,
        timeframe: Timeframe = .m1,
        requestedUtcStart: UtcSecond? = nil,
        requestedUtcEndExclusive: UtcSecond? = nil,
        firstUtc: UtcSecond? = nil,
        lastUtc: UtcSecond? = nil
    ) {
        self.brokerSourceId = brokerSourceId
        self.sourceOrigin = sourceOrigin
        self.logicalSymbol = logicalSymbol
        self.timeframe = timeframe
        self.digits = digits
        self.requestedUtcStart = requestedUtcStart
        self.requestedUtcEndExclusive = requestedUtcEndExclusive
        self.firstUtc = firstUtc
        self.lastUtc = lastUtc
    }
}

public protocol BarSeries: Sendable {
    var metadata: BarSeriesMetadata { get }
    var count: Int { get }
}
