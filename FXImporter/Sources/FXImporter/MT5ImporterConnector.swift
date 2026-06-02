import Foundation
import FXImporterAPI
import MT5Bridge

public enum MT5ImporterConnectorError: Error, CustomStringConvertible, Sendable {
    case invalidRequest(String)
    case malformedRate(String)
    case symbolDiscoveryUnsupported
    case symbolMismatch(expected: String, actual: String)
    case timeframeMismatch(String)
    case requestedRangeMismatch

    public var description: String {
        switch self {
        case .invalidRequest(let reason):
            return "Invalid MT5 importer request: \(reason)."
        case .malformedRate(let reason):
            return "Malformed MT5 M1 rate: \(reason)."
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
        try validateLatestAPI()
        let snapshot = try bridge.serverTimeSnapshot()
        return FXImporterHealth(isConnected: true, sourceClockTimestamp: snapshot.timeTradeServer)
    }

    public func symbols() async throws -> [FXImporterSymbol] {
        try validateLatestAPI()
        throw MT5ImporterConnectorError.symbolDiscoveryUnsupported
    }

    public func fetchM1History(_ request: FXImporterM1HistoryRequest) async throws -> FXImporterM1Batch {
        try validateLatestAPI()
        try request.validateLatestAPI()
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
        try validate(request)
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

        var bars: [FXImporterM1Bar] = []
        bars.reserveCapacity(min(response.rates.count, request.maxBars))
        var previousTimestamp: Int64?
        for rate in response.rates {
            try validate(rate: rate, request: request, previousTimestamp: previousTimestamp)
            previousTimestamp = rate.mt5ServerTime
            bars.append(FXImporterM1Bar(
                sourceSymbol: response.mt5Symbol,
                sourceTimestamp: rate.mt5ServerTime,
                utcTimestamp: nil,
                open: rate.open,
                high: rate.high,
                low: rate.low,
                close: rate.close,
                volume: 0
            ))
        }
        let sourceComplete = (response.seriesSynchronized ?? false) && bars.count <= request.maxBars
        let cappedBars = sourceComplete ? bars : Array(bars.prefix(request.maxBars))
        return FXImporterM1Batch(request: request, bars: cappedBars, sourceComplete: sourceComplete)
    }

    private static func validate(_ request: FXImporterM1HistoryRequest) throws {
        try request.validateLatestAPI()
        guard !request.sourceSymbol.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw MT5ImporterConnectorError.invalidRequest("sourceSymbol is empty")
        }
        guard request.fromSourceTimestamp < request.toSourceTimestampExclusive else {
            throw MT5ImporterConnectorError.invalidRequest("from timestamp must be earlier than to timestamp")
        }
        guard request.fromSourceTimestamp % 60 == 0, request.toSourceTimestampExclusive % 60 == 0 else {
            throw MT5ImporterConnectorError.invalidRequest("M1 range boundaries must be minute-aligned")
        }
        guard request.maxBars > 0 else {
            throw MT5ImporterConnectorError.invalidRequest("maxBars must be positive")
        }
    }

    private static func validate(
        rate: MT5RateDTO,
        request: FXImporterM1HistoryRequest,
        previousTimestamp: Int64?
    ) throws {
        guard rate.mt5ServerTime >= request.fromSourceTimestamp,
              rate.mt5ServerTime < request.toSourceTimestampExclusive else {
            throw MT5ImporterConnectorError.malformedRate("timestamp \(rate.mt5ServerTime) is outside requested range")
        }
        guard rate.mt5ServerTime % 60 == 0 else {
            throw MT5ImporterConnectorError.malformedRate("timestamp \(rate.mt5ServerTime) is not minute-aligned")
        }
        if let previousTimestamp, rate.mt5ServerTime <= previousTimestamp {
            throw MT5ImporterConnectorError.malformedRate("timestamps must be strictly increasing")
        }

        guard let open = Double(rate.open),
              let high = Double(rate.high),
              let low = Double(rate.low),
              let close = Double(rate.close),
              open.isFinite,
              high.isFinite,
              low.isFinite,
              close.isFinite else {
            throw MT5ImporterConnectorError.malformedRate("OHLC contains a non-finite or non-numeric value at \(rate.mt5ServerTime)")
        }
        guard open > 0.0, high > 0.0, low > 0.0, close > 0.0 else {
            throw MT5ImporterConnectorError.malformedRate("OHLC contains a non-positive value at \(rate.mt5ServerTime)")
        }
        guard high >= open,
              high >= close,
              high >= low,
              low <= open,
              low <= close else {
            throw MT5ImporterConnectorError.malformedRate("OHLC invariant failed at \(rate.mt5ServerTime)")
        }
    }
}
