import Domain
import Foundation

public struct HistoryDataConfigFile: Codable, Sendable {
    public let brokerSourceId: BrokerSourceId
    public let sourceOrigin: DataSourceOrigin
    public let logicalSymbol: LogicalSymbol
    public let fromUtc: UtcSecond
    public let toUtc: UtcSecond
    public let useMetal: Bool

    enum CodingKeys: String, CodingKey {
        case brokerSourceId = "broker_source_id"
        case sourceOrigin = "source_origin"
        case logicalSymbol = "logical_symbol"
        case fromUtc = "from_utc"
        case toUtc = "to_utc"
        case useMetal = "use_metal"
    }

    public init(
        brokerSourceId: BrokerSourceId,
        sourceOrigin: DataSourceOrigin = .mt5,
        logicalSymbol: LogicalSymbol,
        fromUtc: UtcSecond,
        toUtc: UtcSecond,
        useMetal: Bool
    ) {
        self.brokerSourceId = brokerSourceId
        self.sourceOrigin = sourceOrigin
        self.logicalSymbol = logicalSymbol
        self.fromUtc = fromUtc
        self.toUtc = toUtc
        self.useMetal = useMetal
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.brokerSourceId = try container.decode(BrokerSourceId.self, forKey: .brokerSourceId)
        self.sourceOrigin = try container.decodeIfPresent(DataSourceOrigin.self, forKey: .sourceOrigin) ?? .mt5
        self.logicalSymbol = try container.decode(LogicalSymbol.self, forKey: .logicalSymbol)
        self.fromUtc = try container.decode(UtcSecond.self, forKey: .fromUtc)
        self.toUtc = try container.decode(UtcSecond.self, forKey: .toUtc)
        self.useMetal = try container.decode(Bool.self, forKey: .useMetal)
    }
}

@available(*, deprecated, renamed: "HistoryDataConfigFile")
public typealias BacktestConfigFile = HistoryDataConfigFile
