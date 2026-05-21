import Domain
import Foundation

public struct SymbolMapping: Codable, Hashable, Sendable {
    public let sourceOrigin: DataSourceOrigin
    public let logicalSymbol: LogicalSymbol
    public let mt5Symbol: MT5Symbol
    public let digits: Digits

    enum CodingKeys: String, CodingKey {
        case sourceOrigin = "source_origin"
        case logicalSymbol = "logical_symbol"
        case mt5Symbol = "mt5_symbol"
        case digits
    }

    public init(sourceOrigin: DataSourceOrigin = .mt5, logicalSymbol: LogicalSymbol, mt5Symbol: MT5Symbol, digits: Digits) {
        self.sourceOrigin = sourceOrigin
        self.logicalSymbol = logicalSymbol
        self.mt5Symbol = mt5Symbol
        self.digits = digits
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.sourceOrigin = try container.decodeIfPresent(DataSourceOrigin.self, forKey: .sourceOrigin) ?? .mt5
        self.logicalSymbol = try container.decode(LogicalSymbol.self, forKey: .logicalSymbol)
        self.mt5Symbol = try container.decode(MT5Symbol.self, forKey: .mt5Symbol)
        self.digits = try container.decode(Digits.self, forKey: .digits)
    }
}

public struct SymbolConfig: Codable, Sendable {
    public let symbols: [SymbolMapping]

    public init(symbols: [SymbolMapping]) {
        self.symbols = symbols
    }

    public func mapping(for logicalSymbol: LogicalSymbol) -> SymbolMapping? {
        mapping(for: logicalSymbol, sourceOrigin: .mt5)
    }

    public func mapping(for logicalSymbol: LogicalSymbol, sourceOrigin: DataSourceOrigin) -> SymbolMapping? {
        symbols.first { $0.logicalSymbol == logicalSymbol && $0.sourceOrigin == sourceOrigin }
    }
}
