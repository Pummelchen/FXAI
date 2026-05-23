import Foundation

public enum FXImporterSourceKind: String, Codable, CaseIterable, Sendable {
    case metaTrader5 = "METATRADER5"
    case interactiveBrokersTWS = "INTERACTIVE_BROKERS_TWS"
    case yahooFinanceHistory = "YAHOO_FINANCE_HISTORY"
    case tradingView = "TRADINGVIEW"
    case custom = "CUSTOM"
}

public struct FXImporterConnectorDescriptor: Codable, Hashable, Sendable {
    public let id: String
    public let displayName: String
    public let kind: FXImporterSourceKind
    public let version: String
    public let capabilities: FXImporterCapabilities

    public init(
        id: String,
        displayName: String,
        kind: FXImporterSourceKind,
        version: String,
        capabilities: FXImporterCapabilities
    ) {
        self.id = id
        self.displayName = displayName
        self.kind = kind
        self.version = version
        self.capabilities = capabilities
    }
}

public struct FXImporterCapabilities: Codable, Hashable, Sendable {
    public let supportsSymbolDiscovery: Bool
    public let supportsHistoricalM1OHLC: Bool
    public let supportsHistoricalD1OHLC: Bool
    public let supportsLiveM1OHLC: Bool
    public let providesBrokerServerTime: Bool
    public let providesVolume: Bool

    public init(
        supportsSymbolDiscovery: Bool,
        supportsHistoricalM1OHLC: Bool,
        supportsHistoricalD1OHLC: Bool = false,
        supportsLiveM1OHLC: Bool,
        providesBrokerServerTime: Bool,
        providesVolume: Bool
    ) {
        self.supportsSymbolDiscovery = supportsSymbolDiscovery
        self.supportsHistoricalM1OHLC = supportsHistoricalM1OHLC
        self.supportsHistoricalD1OHLC = supportsHistoricalD1OHLC
        self.supportsLiveM1OHLC = supportsLiveM1OHLC
        self.providesBrokerServerTime = providesBrokerServerTime
        self.providesVolume = providesVolume
    }
}

public struct FXImporterSymbol: Codable, Hashable, Sendable {
    public let sourceSymbol: String
    public let displayName: String?
    public let digits: Int?

    public init(sourceSymbol: String, displayName: String? = nil, digits: Int? = nil) {
        self.sourceSymbol = sourceSymbol
        self.displayName = displayName
        self.digits = digits
    }
}

public struct FXImporterM1HistoryRequest: Codable, Hashable, Sendable {
    public let sourceSymbol: String
    public let fromSourceTimestamp: Int64
    public let toSourceTimestampExclusive: Int64
    public let maxBars: Int

    public init(
        sourceSymbol: String,
        fromSourceTimestamp: Int64,
        toSourceTimestampExclusive: Int64,
        maxBars: Int
    ) {
        self.sourceSymbol = sourceSymbol
        self.fromSourceTimestamp = fromSourceTimestamp
        self.toSourceTimestampExclusive = toSourceTimestampExclusive
        self.maxBars = maxBars
    }
}

public struct FXImporterM1Bar: Codable, Hashable, Sendable {
    public let sourceSymbol: String
    public let sourceTimestamp: Int64
    public let utcTimestamp: Int64?
    public let open: String
    public let high: String
    public let low: String
    public let close: String
    public let volume: UInt64

    public init(
        sourceSymbol: String,
        sourceTimestamp: Int64,
        utcTimestamp: Int64?,
        open: String,
        high: String,
        low: String,
        close: String,
        volume: UInt64 = 0
    ) {
        self.sourceSymbol = sourceSymbol
        self.sourceTimestamp = sourceTimestamp
        self.utcTimestamp = utcTimestamp
        self.open = open
        self.high = high
        self.low = low
        self.close = close
        self.volume = volume
    }
}

public struct FXImporterM1Batch: Codable, Hashable, Sendable {
    public let request: FXImporterM1HistoryRequest
    public let bars: [FXImporterM1Bar]
    public let sourceComplete: Bool

    public init(request: FXImporterM1HistoryRequest, bars: [FXImporterM1Bar], sourceComplete: Bool) {
        self.request = request
        self.bars = bars
        self.sourceComplete = sourceComplete
    }
}

public struct FXImporterD1HistoryRequest: Codable, Hashable, Sendable {
    public let sourceSymbol: String
    public let fromSourceTimestamp: Int64
    public let toSourceTimestampExclusive: Int64
    public let maxBars: Int
    public let includeAdjustedClose: Bool

    public init(
        sourceSymbol: String,
        fromSourceTimestamp: Int64,
        toSourceTimestampExclusive: Int64,
        maxBars: Int,
        includeAdjustedClose: Bool = true
    ) {
        self.sourceSymbol = sourceSymbol
        self.fromSourceTimestamp = fromSourceTimestamp
        self.toSourceTimestampExclusive = toSourceTimestampExclusive
        self.maxBars = maxBars
        self.includeAdjustedClose = includeAdjustedClose
    }
}

public struct FXImporterD1Bar: Codable, Hashable, Sendable {
    public let sourceSymbol: String
    public let sourceTimestamp: Int64
    public let utcTimestamp: Int64?
    public let open: String
    public let high: String
    public let low: String
    public let close: String
    public let adjustedClose: String?
    public let volume: UInt64

    public init(
        sourceSymbol: String,
        sourceTimestamp: Int64,
        utcTimestamp: Int64?,
        open: String,
        high: String,
        low: String,
        close: String,
        adjustedClose: String? = nil,
        volume: UInt64 = 0
    ) {
        self.sourceSymbol = sourceSymbol
        self.sourceTimestamp = sourceTimestamp
        self.utcTimestamp = utcTimestamp
        self.open = open
        self.high = high
        self.low = low
        self.close = close
        self.adjustedClose = adjustedClose
        self.volume = volume
    }
}

public struct FXImporterD1Batch: Codable, Hashable, Sendable {
    public let request: FXImporterD1HistoryRequest
    public let bars: [FXImporterD1Bar]
    public let sourceComplete: Bool

    public init(request: FXImporterD1HistoryRequest, bars: [FXImporterD1Bar], sourceComplete: Bool) {
        self.request = request
        self.bars = bars
        self.sourceComplete = sourceComplete
    }
}

public enum FXImporterConnectorError: Error, CustomStringConvertible, Sendable {
    case unsupportedCapability(connectorID: String, capability: String)

    public var description: String {
        switch self {
        case .unsupportedCapability(let connectorID, let capability):
            return "Connector \(connectorID) does not support \(capability)."
        }
    }
}

public protocol FXImporterConnector: Sendable {
    var descriptor: FXImporterConnectorDescriptor { get }

    func health() async throws -> FXImporterHealth
    func symbols() async throws -> [FXImporterSymbol]
    func fetchM1History(_ request: FXImporterM1HistoryRequest) async throws -> FXImporterM1Batch
    func fetchD1History(_ request: FXImporterD1HistoryRequest) async throws -> FXImporterD1Batch
}

public extension FXImporterConnector {
    func fetchD1History(_ request: FXImporterD1HistoryRequest) async throws -> FXImporterD1Batch {
        throw FXImporterConnectorError.unsupportedCapability(
            connectorID: descriptor.id,
            capability: "D1 historical OHLC"
        )
    }
}

public struct FXImporterHealth: Codable, Hashable, Sendable {
    public let isConnected: Bool
    public let sourceClockTimestamp: Int64?
    public let message: String?

    public init(isConnected: Bool, sourceClockTimestamp: Int64? = nil, message: String? = nil) {
        self.isConnected = isConnected
        self.sourceClockTimestamp = sourceClockTimestamp
        self.message = message
    }
}
