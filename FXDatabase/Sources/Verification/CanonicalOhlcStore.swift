import ClickHouse
import Domain
import Foundation

public enum CanonicalOhlcStoreError: Error, CustomStringConvertible, Sendable {
    case invalidRow(String)
    case invalidHash(String)
    case invalidOffsetSource(String)
    case invalidOffsetConfidence(String)

    public var description: String {
        switch self {
        case .invalidRow(let row):
            return "Invalid canonical OHLC row: \(row)"
        case .invalidHash(let hash):
            return "Invalid canonical bar hash '\(hash)'."
        case .invalidOffsetSource(let source):
            return "Invalid canonical offset source '\(source)'."
        case .invalidOffsetConfidence(let confidence):
            return "Invalid canonical offset confidence '\(confidence)'."
        }
    }
}

public struct CanonicalOhlcStore: Sendable {
    private let clickHouse: ClickHouseClientProtocol
    private let database: String

    public init(clickHouse: ClickHouseClientProtocol, database: String) {
        self.clickHouse = clickHouse
        self.database = database
    }

    public func fetch(range: VerificationRange) async throws -> [VerificationBar] {
        let sql = """
        SELECT broker_source_id, source_origin, logical_symbol, mt5_symbol, timeframe,
               mt5_server_ts_raw, ts_utc, server_utc_offset_seconds,
               offset_source, offset_confidence,
               open_scaled, high_scaled, low_scaled, close_scaled, volume,
               digits, bar_hash
        FROM \(database).ohlc_m1_canonical
        WHERE broker_source_id = '\(Self.sqlLiteral(range.brokerSourceId.rawValue))'
          AND source_origin = '\(Self.sqlLiteral(range.sourceOrigin.rawValue))'
          AND logical_symbol = '\(Self.sqlLiteral(range.logicalSymbol.rawValue))'
          AND mt5_server_ts_raw >= \(range.mt5Start.rawValue)
          AND mt5_server_ts_raw < \(range.mt5EndExclusive.rawValue)
        ORDER BY mt5_server_ts_raw ASC
        FORMAT TabSeparated
        """
        let body = try await clickHouse.execute(.select(sql))
        return try body
            .split(separator: "\n", omittingEmptySubsequences: true)
            .map { try Self.parseRow(String($0)) }
    }

    private static func parseRow(_ row: String) throws -> VerificationBar {
        let fields = row.split(separator: "\t", omittingEmptySubsequences: false).map { unescapeTabSeparated(String($0)) }
        guard fields.count == 17,
              let sourceOrigin = DataSourceOrigin(rawValue: fields[1]),
              let timeframe = Timeframe(rawValue: fields[4]),
              timeframe == .m1,
              let mt5ServerTime = Int64(fields[5]),
              let utcTime = Int64(fields[6]),
              Int64(fields[7]) != nil,
              let open = Int64(fields[10]),
              let high = Int64(fields[11]),
              let low = Int64(fields[12]),
              let close = Int64(fields[13]),
              let volume = UInt64(fields[14]),
              let digitsValue = Int(fields[15]) else {
            throw CanonicalOhlcStoreError.invalidRow(row)
        }
        guard OffsetSource(rawValue: fields[8]) != nil else {
            throw CanonicalOhlcStoreError.invalidOffsetSource(fields[8])
        }
        guard let offsetConfidence = OffsetConfidence(rawValue: fields[9]) else {
            throw CanonicalOhlcStoreError.invalidOffsetConfidence(fields[9])
        }
        guard let hashValue = UInt64(fields[16], radix: 16) else {
            throw CanonicalOhlcStoreError.invalidHash(fields[16])
        }
        let digits = try Digits(digitsValue)
        return VerificationBar(
            brokerSourceId: try BrokerSourceId(fields[0]),
            sourceOrigin: sourceOrigin,
            logicalSymbol: try LogicalSymbol(fields[2]),
            mt5Symbol: try MT5Symbol(fields[3]),
            mt5ServerTime: MT5ServerSecond(rawValue: mt5ServerTime),
            utcTime: UtcSecond(rawValue: utcTime),
            open: PriceScaled(rawValue: open, digits: digits),
            high: PriceScaled(rawValue: high, digits: digits),
            low: PriceScaled(rawValue: low, digits: digits),
            close: PriceScaled(rawValue: close, digits: digits),
            volume: M1Volume(rawValue: volume),
            digits: digits,
            offsetConfidence: offsetConfidence,
            barHash: BarHash(rawValue: hashValue)
        )
    }

    static func sqlLiteral(_ value: String) -> String {
        value.replacingOccurrences(of: "\\", with: "\\\\").replacingOccurrences(of: "'", with: "\\'")
    }

    static func unescapeTabSeparated(_ value: String) -> String {
        var result = ""
        var escaping = false
        for character in value {
            if escaping {
                switch character {
                case "t": result.append("\t")
                case "n": result.append("\n")
                case "r": result.append("\r")
                case "\\": result.append("\\")
                default: result.append(character)
                }
                escaping = false
            } else if character == "\\" {
                escaping = true
            } else {
                result.append(character)
            }
        }
        if escaping {
            result.append("\\")
        }
        return result
    }
}
