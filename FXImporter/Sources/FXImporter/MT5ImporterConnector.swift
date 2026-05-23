import Foundation
import FXImporterAPI
import MT5Bridge

public enum MT5ImporterConnectorError: Error, CustomStringConvertible, Sendable {
    case symbolDiscoveryUnsupported
    case symbolMismatch(expected: String, actual: String)
    case timeframeMismatch(String)
    case requestedRangeMismatch

    public var description: String {
        switch self {
        case .symbolDiscoveryUnsupported:
            return "MT5 bridge symbol discovery is unsupported; configure MT5 symbols in FXDatabase."
        case .symbolMismatch(let expected, let actual):
            return "MT5 response symbol mismatch. Expected \(expected), got \(actual)."
        case .timeframeMismatch(let timeframe):
            return "MT5 response timeframe mismatch. Expected M1, got \(timeframe)."
        case .requestedRangeMismatch:
            return "MT5 response range metadata does not match the importer request."
        }
    }
}

public struct MT5ImporterConnector: FXImporterConnector {
    public let descriptor: FXImporterConnectorDescriptor

    private let bridge: MT5BridgeClient

    public init(
        bridge: MT5BridgeClient,
        id: String = "mt5-local",
        displayName: String = "MetaTrader 5 Local Bridge",
        version: String = "1.0"
    ) {
        self.bridge = bridge
        self.descriptor = FXImporterConnectorDescriptor(
            id: id,
            displayName: displayName,
            kind: .metaTrader5,
            version: version,
            capabilities: FXImporterCapabilities(
                supportsSymbolDiscovery: false,
                supportsHistoricalM1OHLC: true,
                supportsHistoricalD1OHLC: false,
                supportsLiveM1OHLC: true,
                providesBrokerServerTime: true,
                providesVolume: false
            )
        )
    }

    public func health() async throws -> FXImporterHealth {
        let snapshot = try bridge.serverTimeSnapshot()
        return FXImporterHealth(isConnected: true, sourceClockTimestamp: snapshot.timeTradeServer)
    }

    public func symbols() async throws -> [FXImporterSymbol] {
        throw MT5ImporterConnectorError.symbolDiscoveryUnsupported
    }

    public func fetchM1History(_ request: FXImporterM1HistoryRequest) async throws -> FXImporterM1Batch {
        let response = try bridge.ratesRange(
            mt5Symbol: request.sourceSymbol,
            fromMT5ServerTs: request.fromSourceTimestamp,
            toMT5ServerTsExclusive: request.toSourceTimestampExclusive,
            maxBars: request.maxBars
        )
        return try Self.makeBatch(request: request, response: response)
    }

    public static func makeBatch(
        request: FXImporterM1HistoryRequest,
        response: RatesResponseDTO
    ) throws -> FXImporterM1Batch {
        guard response.mt5Symbol == request.sourceSymbol else {
            throw MT5ImporterConnectorError.symbolMismatch(expected: request.sourceSymbol, actual: response.mt5Symbol)
        }
        guard response.timeframe == "M1" else {
            throw MT5ImporterConnectorError.timeframeMismatch(response.timeframe)
        }
        if let from = response.requestedFromMT5ServerTs, from != request.fromSourceTimestamp {
            throw MT5ImporterConnectorError.requestedRangeMismatch
        }
        if let to = response.requestedToMT5ServerTsExclusive, to != request.toSourceTimestampExclusive {
            throw MT5ImporterConnectorError.requestedRangeMismatch
        }

        let bars = response.rates.map { rate in
            FXImporterM1Bar(
                sourceSymbol: response.mt5Symbol,
                sourceTimestamp: rate.mt5ServerTime,
                utcTimestamp: nil,
                open: rate.open,
                high: rate.high,
                low: rate.low,
                close: rate.close,
                volume: 0
            )
        }
        return FXImporterM1Batch(request: request, bars: bars, sourceComplete: response.seriesSynchronized ?? true)
    }
}
