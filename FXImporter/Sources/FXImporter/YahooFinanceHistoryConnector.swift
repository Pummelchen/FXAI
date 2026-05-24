import Foundation
import FXImporterAPI

public typealias YahooFinanceHistoryDataLoader = @Sendable (URLRequest) async throws -> (Data, Int)

public enum YahooFinanceHistoryConnectorError: Error, CustomStringConvertible, Sendable {
    case invalidRequest(String)
    case symbolDiscoveryUnsupported
    case m1HistoryUnsupported
    case invalidURL(String)
    case httpStatus(Int)
    case chartError(String)
    case malformedResponse(String)

    public var description: String {
        switch self {
        case .invalidRequest(let reason):
            return "Invalid Yahoo Finance history request: \(reason)"
        case .symbolDiscoveryUnsupported:
            return "Yahoo Finance connector does not support provider-wide symbol discovery."
        case .m1HistoryUnsupported:
            return "Yahoo Finance connector supports historical D1 OHLCV only, not M1 history."
        case .invalidURL(let reason):
            return "Could not build Yahoo Finance history URL: \(reason)"
        case .httpStatus(let statusCode):
            return "Yahoo Finance history request failed with HTTP status \(statusCode)."
        case .chartError(let message):
            return "Yahoo Finance chart API error: \(message)"
        case .malformedResponse(let reason):
            return "Malformed Yahoo Finance chart response: \(reason)"
        }
    }
}

public struct YahooFinanceHistoryConnector: FXImporterConnector {
    public let descriptor: FXImporterConnectorDescriptor

    private let baseURL: URL
    private let loadData: YahooFinanceHistoryDataLoader

    public init(
        baseURL: URL = URL(string: "https://query1.finance.yahoo.com")!,
        id: String = "yahoo-finance-history",
        displayName: String = "Yahoo Finance Historical Daily",
        version: String = "1.0",
        loadData: YahooFinanceHistoryDataLoader? = nil
    ) {
        self.baseURL = baseURL
        self.loadData = loadData ?? Self.defaultLoadData
        self.descriptor = FXImporterConnectorDescriptor(
            id: id,
            displayName: displayName,
            kind: .yahooFinanceHistory,
            version: version,
            capabilities: FXImporterCapabilities(
                supportsSymbolDiscovery: false,
                supportsHistoricalM1OHLC: false,
                supportsHistoricalD1OHLC: true,
                supportsLiveM1OHLC: false,
                providesBrokerServerTime: false,
                providesVolume: true
            )
        )
    }

    public func health() async throws -> FXImporterHealth {
        try validateLatestAPI()
        return FXImporterHealth(
            isConnected: true,
            message: "Stateless connector; network and provider status are verified during each D1 history fetch."
        )
    }

    public func symbols() async throws -> [FXImporterSymbol] {
        try validateLatestAPI()
        throw YahooFinanceHistoryConnectorError.symbolDiscoveryUnsupported
    }

    public func fetchM1History(_ request: FXImporterM1HistoryRequest) async throws -> FXImporterM1Batch {
        try validateLatestAPI()
        try request.validateLatestAPI()
        throw YahooFinanceHistoryConnectorError.m1HistoryUnsupported
    }

    public func fetchD1History(_ request: FXImporterD1HistoryRequest) async throws -> FXImporterD1Batch {
        try validateLatestAPI()
        try Self.validate(request)
        let chartRequest = try Self.makeChartRequest(baseURL: baseURL, request: request)
        let (data, statusCode) = try await loadData(chartRequest)
        guard (200..<300).contains(statusCode) else {
            throw YahooFinanceHistoryConnectorError.httpStatus(statusCode)
        }
        return try Self.makeBatch(request: request, data: data)
    }

    public static func makeChartRequest(
        baseURL: URL = URL(string: "https://query1.finance.yahoo.com")!,
        request: FXImporterD1HistoryRequest
    ) throws -> URLRequest {
        let url = try makeChartURL(baseURL: baseURL, request: request)
        var urlRequest = URLRequest(url: url, cachePolicy: .reloadIgnoringLocalCacheData, timeoutInterval: 30)
        urlRequest.setValue("application/json", forHTTPHeaderField: "Accept")
        urlRequest.setValue("FXAI-FXImporter/0.1", forHTTPHeaderField: "User-Agent")
        return urlRequest
    }

    public static func makeChartURL(
        baseURL: URL = URL(string: "https://query1.finance.yahoo.com")!,
        request: FXImporterD1HistoryRequest
    ) throws -> URL {
        try validate(request)
        guard let encodedSymbol = request.sourceSymbol.addingPercentEncoding(withAllowedCharacters: yahooPathAllowedCharacters) else {
            throw YahooFinanceHistoryConnectorError.invalidURL("symbol \(request.sourceSymbol) could not be percent encoded")
        }

        var components = URLComponents(url: baseURL, resolvingAgainstBaseURL: false)
        components?.percentEncodedPath = "/v8/finance/chart/\(encodedSymbol)"
        components?.queryItems = [
            URLQueryItem(name: "period1", value: String(request.fromSourceTimestamp)),
            URLQueryItem(name: "period2", value: String(request.toSourceTimestampExclusive)),
            URLQueryItem(name: "interval", value: "1d"),
            URLQueryItem(name: "events", value: "history"),
            URLQueryItem(name: "includeAdjustedClose", value: request.includeAdjustedClose ? "true" : "false")
        ]

        guard let url = components?.url else {
            throw YahooFinanceHistoryConnectorError.invalidURL("URLComponents returned nil")
        }
        return url
    }

    public static func makeBatch(request: FXImporterD1HistoryRequest, data: Data) throws -> FXImporterD1Batch {
        try validate(request)
        let response = try JSONDecoder().decode(YahooChartResponseDTO.self, from: data)
        if let error = response.chart.error {
            throw YahooFinanceHistoryConnectorError.chartError(error.displayMessage)
        }
        guard let result = response.chart.result?.first else {
            throw YahooFinanceHistoryConnectorError.malformedResponse("missing chart result")
        }
        guard let quote = result.indicators.quote.first else {
            throw YahooFinanceHistoryConnectorError.malformedResponse("missing quote indicators")
        }

        let timestamps = result.timestamp
        let count = timestamps.count
        guard quote.open.count == count,
              quote.high.count == count,
              quote.low.count == count,
              quote.close.count == count,
              quote.volume.count == count else {
            throw YahooFinanceHistoryConnectorError.malformedResponse("timestamp and quote array counts differ")
        }

        let adjustedClose = result.indicators.adjclose?.first?.adjclose ?? []
        if !adjustedClose.isEmpty && adjustedClose.count != count {
            throw YahooFinanceHistoryConnectorError.malformedResponse("adjusted close count differs from timestamp count")
        }

        var bars: [FXImporterD1Bar] = []
        bars.reserveCapacity(min(count, request.maxBars))
        var previousTimestamp: Int64?
        for index in timestamps.indices {
            let timestamp = timestamps[index]
            guard timestamp >= request.fromSourceTimestamp,
                  timestamp < request.toSourceTimestampExclusive,
                  let open = quote.open[index],
                  let high = quote.high[index],
                  let low = quote.low[index],
                  let close = quote.close[index] else {
                continue
            }
            try validateOHLC(
                open: open,
                high: high,
                low: low,
                close: close,
                timestamp: timestamp
            )
            if let previousTimestamp, timestamp <= previousTimestamp {
                throw YahooFinanceHistoryConnectorError.malformedResponse("timestamps must be strictly increasing")
            }
            previousTimestamp = timestamp

            let adjusted = adjustedClose.indices.contains(index) ? adjustedClose[index] : nil
            bars.append(
                FXImporterD1Bar(
                    sourceSymbol: request.sourceSymbol,
                    sourceTimestamp: timestamp,
                    utcTimestamp: timestamp,
                    open: try priceString(open),
                    high: try priceString(high),
                    low: try priceString(low),
                    close: try priceString(close),
                    adjustedClose: try adjusted.map(priceString),
                    volume: quote.volume[index] ?? 0
                )
            )
        }

        let sourceComplete = bars.count <= request.maxBars
        let cappedBars = sourceComplete ? bars : Array(bars.prefix(request.maxBars))
        return FXImporterD1Batch(request: request, bars: cappedBars, sourceComplete: sourceComplete)
    }

    private static func validate(_ request: FXImporterD1HistoryRequest) throws {
        try request.validateLatestAPI()
        guard !request.sourceSymbol.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw YahooFinanceHistoryConnectorError.invalidRequest("sourceSymbol is empty")
        }
        guard request.fromSourceTimestamp < request.toSourceTimestampExclusive else {
            throw YahooFinanceHistoryConnectorError.invalidRequest("from timestamp must be earlier than to timestamp")
        }
        guard request.maxBars > 0 else {
            throw YahooFinanceHistoryConnectorError.invalidRequest("maxBars must be positive")
        }
    }

    private static func priceString(_ value: Double) throws -> String {
        guard value.isFinite else {
            throw YahooFinanceHistoryConnectorError.malformedResponse("price contains non-finite value")
        }
        guard value > 0.0 else {
            throw YahooFinanceHistoryConnectorError.malformedResponse("price must be positive")
        }
        let raw = String(format: "%.10f", locale: Locale(identifier: "en_US_POSIX"), value)
        let trimmedZeros = raw.replacingOccurrences(of: #"0+$"#, with: "", options: .regularExpression)
        let trimmedDecimal = trimmedZeros.replacingOccurrences(of: #"\.$"#, with: "", options: .regularExpression)
        return trimmedDecimal.isEmpty ? "0" : trimmedDecimal
    }

    private static func validateOHLC(
        open: Double,
        high: Double,
        low: Double,
        close: Double,
        timestamp: Int64
    ) throws {
        guard open.isFinite, high.isFinite, low.isFinite, close.isFinite else {
            throw YahooFinanceHistoryConnectorError.malformedResponse("OHLC contains non-finite value at \(timestamp)")
        }
        guard open > 0.0, high > 0.0, low > 0.0, close > 0.0 else {
            throw YahooFinanceHistoryConnectorError.malformedResponse("OHLC contains non-positive value at \(timestamp)")
        }
        guard high >= open,
              high >= close,
              high >= low,
              low <= open,
              low <= close else {
            throw YahooFinanceHistoryConnectorError.malformedResponse("OHLC invariant failed at \(timestamp)")
        }
    }

    private static func defaultLoadData(_ request: URLRequest) async throws -> (Data, Int) {
        let (data, response) = try await URLSession.shared.data(for: request)
        guard let httpResponse = response as? HTTPURLResponse else {
            throw YahooFinanceHistoryConnectorError.malformedResponse("missing HTTP response")
        }
        return (data, httpResponse.statusCode)
    }

    private static let yahooPathAllowedCharacters: CharacterSet = {
        var allowed = CharacterSet.urlPathAllowed
        allowed.remove(charactersIn: "/?#[]@!$&'()*+,;=")
        return allowed
    }()
}

private struct YahooChartResponseDTO: Decodable {
    let chart: YahooChartDTO
}

private struct YahooChartDTO: Decodable {
    let result: [YahooChartResultDTO]?
    let error: YahooChartErrorDTO?
}

private struct YahooChartErrorDTO: Decodable {
    let code: String?
    let message: String?

    var displayMessage: String {
        [code, message].compactMap { $0 }.joined(separator: ": ")
    }

    private enum CodingKeys: String, CodingKey {
        case code
        case message = "description"
    }
}

private struct YahooChartResultDTO: Decodable {
    let timestamp: [Int64]
    let indicators: YahooIndicatorsDTO
}

private struct YahooIndicatorsDTO: Decodable {
    let quote: [YahooQuoteDTO]
    let adjclose: [YahooAdjustedCloseDTO]?
}

private struct YahooQuoteDTO: Decodable {
    let open: [Double?]
    let high: [Double?]
    let low: [Double?]
    let close: [Double?]
    let volume: [UInt64?]
}

private struct YahooAdjustedCloseDTO: Decodable {
    let adjclose: [Double?]
}
