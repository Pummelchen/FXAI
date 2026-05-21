import Domain
import Foundation

public struct HelloResponseDTO: Codable, Sendable {
    public let bridgeName: String
    public let bridgeVersion: String
    public let schemaVersion: Int

    enum CodingKeys: String, CodingKey {
        case bridgeName = "bridge_name"
        case bridgeVersion = "bridge_version"
        case schemaVersion = "schema_version"
    }
}

public struct TerminalInfoDTO: Codable, Sendable {
    public let terminalName: String
    public let company: String
    public let server: String
    public let accountLogin: Int64

    enum CodingKeys: String, CodingKey {
        case terminalName = "terminal_name"
        case company
        case server
        case accountLogin = "account_login"
    }

    public func brokerServerIdentity() throws -> BrokerServerIdentity {
        try BrokerServerIdentity(company: company, server: server, accountLogin: accountLogin)
    }
}

public struct SymbolInfoDTO: Codable, Sendable {
    public let mt5Symbol: String
    public let selected: Bool
    public let digits: Int

    enum CodingKeys: String, CodingKey {
        case mt5Symbol = "mt5_symbol"
        case selected
        case digits
    }
}

public struct HistoryStatusDTO: Codable, Sendable {
    public let mt5Symbol: String
    public let synchronized: Bool
    public let bars: Int

    enum CodingKeys: String, CodingKey {
        case mt5Symbol = "mt5_symbol"
        case synchronized
        case bars
    }
}

public enum M1MonthHistoryStatus: String, Codable, Sendable {
    case unavailable
    case future
    case loaded
    case loading
    case partial
}

public struct M1MonthHistoryStatusDTO: Codable, Sendable {
    public let mt5Symbol: String
    public let timeframe: String
    public let monthStartMT5ServerTs: Int64
    public let monthEndMT5ServerTsExclusive: Int64
    public let effectiveToMT5ServerTsExclusive: Int64
    public let serverFirstDateMT5ServerTs: Int64
    public let localFirstDateBeforeMT5ServerTs: Int64
    public let localFirstDateAfterMT5ServerTs: Int64
    public let rangeBarsBefore: Int
    public let rangeBarsAfter: Int
    public let totalBarsBefore: Int
    public let totalBarsAfter: Int
    public let seriesSynchronizedBefore: Bool
    public let seriesSynchronizedAfter: Bool
    public let historicalAvailable: Bool
    public let alreadyLoaded: Bool
    public let loadAttempted: Bool
    public let loadComplete: Bool
    public let copiedCount: Int
    public let firstMT5ServerTs: Int64
    public let lastMT5ServerTs: Int64
    public let lastError: Int
    public let status: M1MonthHistoryStatus

    enum CodingKeys: String, CodingKey {
        case mt5Symbol = "mt5_symbol"
        case timeframe
        case monthStartMT5ServerTs = "month_start_mt5_server_ts"
        case monthEndMT5ServerTsExclusive = "month_end_mt5_server_ts_exclusive"
        case effectiveToMT5ServerTsExclusive = "effective_to_mt5_server_ts_exclusive"
        case serverFirstDateMT5ServerTs = "server_first_date_mt5_server_ts"
        case localFirstDateBeforeMT5ServerTs = "local_first_date_before_mt5_server_ts"
        case localFirstDateAfterMT5ServerTs = "local_first_date_after_mt5_server_ts"
        case rangeBarsBefore = "range_bars_before"
        case rangeBarsAfter = "range_bars_after"
        case totalBarsBefore = "total_bars_before"
        case totalBarsAfter = "total_bars_after"
        case seriesSynchronizedBefore = "series_synchronized_before"
        case seriesSynchronizedAfter = "series_synchronized_after"
        case historicalAvailable = "historical_available"
        case alreadyLoaded = "already_loaded"
        case loadAttempted = "load_attempted"
        case loadComplete = "load_complete"
        case copiedCount = "copied_count"
        case firstMT5ServerTs = "first_mt5_server_ts"
        case lastMT5ServerTs = "last_mt5_server_ts"
        case lastError = "last_error"
        case status
    }
}

public struct ServerTimeSnapshotDTO: Codable, Sendable {
    public let timeTradeServer: Int64
    public let timeGMT: Int64
    public let timeLocal: Int64

    enum CodingKeys: String, CodingKey {
        case timeTradeServer = "time_trade_server"
        case timeGMT = "time_gmt"
        case timeLocal = "time_local"
    }

    public init(timeTradeServer: Int64, timeGMT: Int64, timeLocal: Int64) {
        self.timeTradeServer = timeTradeServer
        self.timeGMT = timeGMT
        self.timeLocal = timeLocal
    }
}

public struct MT5RateDTO: Codable, Sendable {
    public let mt5ServerTime: Int64
    public let open: String
    public let high: String
    public let low: String
    public let close: String

    enum CodingKeys: String, CodingKey {
        case mt5ServerTime = "mt5_server_time"
        case open
        case high
        case low
        case close
    }

    public func toClosedM1Bar(logicalSymbol: LogicalSymbol, mt5Symbol: MT5Symbol, digits: Digits) throws -> ClosedM1Bar {
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

public struct RatesResponseDTO: Codable, Sendable {
    public let mt5Symbol: String
    public let timeframe: String
    public let requestedFromMT5ServerTs: Int64?
    public let requestedToMT5ServerTsExclusive: Int64?
    public let effectiveToMT5ServerTsExclusive: Int64?
    public let latestClosedMT5ServerTs: Int64?
    public let seriesSynchronized: Bool?
    public let copiedCount: Int?
    public let emittedCount: Int?
    public let firstMT5ServerTs: Int64?
    public let lastMT5ServerTs: Int64?
    public let rates: [MT5RateDTO]

    enum CodingKeys: String, CodingKey {
        case mt5Symbol = "mt5_symbol"
        case timeframe
        case requestedFromMT5ServerTs = "requested_from_mt5_server_ts"
        case requestedToMT5ServerTsExclusive = "requested_to_mt5_server_ts_exclusive"
        case effectiveToMT5ServerTsExclusive = "effective_to_mt5_server_ts_exclusive"
        case latestClosedMT5ServerTs = "latest_closed_mt5_server_ts"
        case seriesSynchronized = "series_synchronized"
        case copiedCount = "copied_count"
        case emittedCount = "emitted_count"
        case firstMT5ServerTs = "first_mt5_server_ts"
        case lastMT5ServerTs = "last_mt5_server_ts"
        case rates
    }
}

public struct SingleTimeResponseDTO: Codable, Sendable {
    public let mt5Symbol: String
    public let mt5ServerTime: Int64

    enum CodingKeys: String, CodingKey {
        case mt5Symbol = "mt5_symbol"
        case mt5ServerTime = "mt5_server_time"
    }
}
