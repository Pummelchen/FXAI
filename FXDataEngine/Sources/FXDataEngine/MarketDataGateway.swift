import Foundation
import FXBacktestAPI

public struct FXDatabaseMarketConnection: Codable, Hashable, Sendable {
    public var apiBaseURL: URL
    public var requestTimeoutSeconds: Double

    public init(
        apiBaseURL: URL = URL(string: "http://127.0.0.1:5066")!,
        requestTimeoutSeconds: Double = 120
    ) {
        self.apiBaseURL = apiBaseURL
        self.requestTimeoutSeconds = max(1.0, requestTimeoutSeconds)
    }
}

public struct FXDatabaseMarketHistoryRequest: Codable, Hashable, Sendable {
    public var brokerSourceId: String
    public var sourceOrigin: String
    public var logicalSymbol: String
    public var expectedProviderSymbol: String?
    public var expectedDigits: Int?
    public var utcStartInclusive: Int64
    public var utcEndExclusive: Int64
    public var maximumRows: Int

    public init(
        brokerSourceId: String,
        sourceOrigin: String = "MT5",
        logicalSymbol: String,
        expectedProviderSymbol: String? = nil,
        expectedDigits: Int? = nil,
        utcStartInclusive: Int64,
        utcEndExclusive: Int64,
        maximumRows: Int = FXBacktestAPIV1.maximumRowsLimit
    ) {
        self.brokerSourceId = brokerSourceId.trimmingCharacters(in: .whitespacesAndNewlines)
        self.sourceOrigin = sourceOrigin.trimmingCharacters(in: .whitespacesAndNewlines).uppercased()
        self.logicalSymbol = DataCoreRequest.normalizedSymbol(logicalSymbol)
        let provider = expectedProviderSymbol?.trimmingCharacters(in: .whitespacesAndNewlines)
        self.expectedProviderSymbol = provider?.isEmpty == true ? nil : provider
        self.expectedDigits = expectedDigits
        self.utcStartInclusive = utcStartInclusive
        self.utcEndExclusive = utcEndExclusive
        self.maximumRows = max(1, min(maximumRows, FXBacktestAPIV1.maximumRowsLimit))
    }

    public func apiRequest() -> FXBacktestM1HistoryRequest {
        FXBacktestM1HistoryRequest(
            brokerSourceId: brokerSourceId,
            sourceOrigin: sourceOrigin,
            logicalSymbol: logicalSymbol,
            utcStartInclusive: utcStartInclusive,
            utcEndExclusive: utcEndExclusive,
            expectedMT5Symbol: expectedProviderSymbol,
            expectedDigits: expectedDigits,
            maximumRows: maximumRows
        )
    }
}

public struct FXDatabaseMarketDataLoader: Sendable {
    public init() {}

    public func loadSeries(
        connection: FXDatabaseMarketConnection,
        request: FXDatabaseMarketHistoryRequest
    ) async throws -> M1OHLCVSeries {
        do {
            let response = try await FXBacktestAPIClient(
                baseURL: connection.apiBaseURL,
                requestTimeoutSeconds: connection.requestTimeoutSeconds
            )
                .loadM1History(request.apiRequest())
            return try M1OHLCVSeries(response: response)
        } catch is CancellationError {
            throw CancellationError()
        } catch {
            if Task.isCancelled {
                throw CancellationError()
            }
            throw FXDataEngineError.externalBackend(String(describing: error))
        }
    }

    public func loadUniverse(
        connection: FXDatabaseMarketConnection,
        requests: [FXDatabaseMarketHistoryRequest],
        primarySymbol: String,
        requireAlignedTimestamps: Bool = true
    ) async throws -> MarketUniverse {
        guard !requests.isEmpty else {
            throw FXDataEngineError.invalidRequest("at least one FXDatabase market history request is required")
        }

        var series: [M1OHLCVSeries] = []
        series.reserveCapacity(requests.count)
        try await withThrowingTaskGroup(of: M1OHLCVSeries.self) { group in
            for request in requests {
                group.addTask {
                    try await loadSeries(connection: connection, request: request)
                }
            }
            for try await item in group {
                series.append(item)
            }
        }
        return try MarketUniverse(
            primarySymbol: primarySymbol,
            series: series,
            requireAlignedTimestamps: requireAlignedTimestamps
        )
    }
}

public enum MarketDataGateway {
    public static func barIndex(
        in series: M1OHLCVSeries,
        atOrBefore timestampUTC: Int64,
        exact: Bool = false
    ) -> Int {
        series.index(atOrBefore: timestampUTC, exact: exact)
    }

    public static func closedBarPosition(
        in series: M1OHLCVSeries,
        atOrBefore timestampUTC: Int64,
        exact: Bool = false
    ) -> Int {
        series.closedPosition(atOrBefore: timestampUTC, exact: exact)
    }

    public static func bars(
        in series: M1OHLCVSeries,
        range: Range<Int>
    ) throws -> M1OHLCVSeries {
        try series.slice(range: range)
    }

    public static func bars(
        in series: M1OHLCVSeries,
        fromUTCInclusive: Int64,
        toUTCExclusive: Int64
    ) throws -> M1OHLCVSeries {
        try series.slice(fromUTCInclusive: fromUTCInclusive, toUTCExclusive: toUTCExclusive)
    }

    public static func closedWindow(
        in series: M1OHLCVSeries,
        endingAt index: Int,
        count: Int
    ) throws -> M1OHLCVSeries {
        try series.window(endingAt: index, count: count)
    }

    public static func closedWindow(
        in series: M1OHLCVSeries,
        endingAtOrBefore timestampUTC: Int64,
        count: Int,
        exact: Bool = false
    ) throws -> M1OHLCVSeries {
        try series.window(endingAtOrBefore: timestampUTC, count: count, exact: exact)
    }
}

public extension M1OHLCVSeries {
    func index(atOrBefore timestampUTC: Int64, exact: Bool = false) -> Int {
        guard timestampUTC > 0, !utcTimestamps.isEmpty else { return -1 }
        var low = 0
        var high = utcTimestamps.count - 1
        var answer = -1
        while low <= high {
            let mid = (low + high) / 2
            if utcTimestamps[mid] <= timestampUTC {
                answer = mid
                low = mid + 1
            } else {
                high = mid - 1
            }
        }
        guard answer >= 0 else { return -1 }
        if exact, utcTimestamps[answer] != timestampUTC {
            return -1
        }
        return answer
    }

    func closedPosition(atOrBefore timestampUTC: Int64, exact: Bool = false) -> Int {
        let index = self.index(atOrBefore: timestampUTC, exact: exact)
        guard index >= 0 else { return -1 }
        return count - 1 - index
    }

    func slice(range: Range<Int>) throws -> M1OHLCVSeries {
        guard range.lowerBound >= 0,
              range.upperBound >= range.lowerBound,
              range.upperBound <= count else {
            throw FXDataEngineError.invalidRequest("market data range \(range) is outside 0..<\(count)")
        }
        return try M1OHLCVSeries(
            metadata: metadataForSlice(
                firstUTC: range.isEmpty ? nil : utcTimestamps[range.lowerBound],
                lastUTC: range.isEmpty ? nil : utcTimestamps[range.upperBound - 1]
            ),
            utcTimestamps: ContiguousArray(utcTimestamps[range]),
            open: ContiguousArray(open[range]),
            high: ContiguousArray(high[range]),
            low: ContiguousArray(low[range]),
            close: ContiguousArray(close[range]),
            volume: ContiguousArray(volume[range])
        )
    }

    func slice(fromUTCInclusive: Int64, toUTCExclusive: Int64) throws -> M1OHLCVSeries {
        guard fromUTCInclusive < toUTCExclusive else {
            throw FXDataEngineError.invalidRequest("market data UTC range start must be before end")
        }
        let start = lowerBound(timestampUTC: fromUTCInclusive)
        let end = lowerBound(timestampUTC: toUTCExclusive)
        return try slice(range: start..<end)
    }

    func window(endingAt index: Int, count requestedCount: Int) throws -> M1OHLCVSeries {
        guard requestedCount > 0 else {
            throw FXDataEngineError.invalidRequest("market data window count must be positive")
        }
        guard index >= 0, index < count else {
            throw FXDataEngineError.invalidRequest("market data window end index \(index) is outside 0..<\(count)")
        }
        guard index + 1 >= requestedCount else {
            throw FXDataEngineError.insufficientData("market data window ending at \(index) needs \(requestedCount) bars")
        }
        return try slice(range: (index + 1 - requestedCount)..<(index + 1))
    }

    func window(endingAtOrBefore timestampUTC: Int64, count requestedCount: Int, exact: Bool = false) throws -> M1OHLCVSeries {
        let index = index(atOrBefore: timestampUTC, exact: exact)
        guard index >= 0 else {
            throw FXDataEngineError.insufficientData("no M1 bar found at or before \(timestampUTC)")
        }
        return try window(endingAt: index, count: requestedCount)
    }

    private func lowerBound(timestampUTC: Int64) -> Int {
        var low = 0
        var high = utcTimestamps.count
        while low < high {
            let mid = (low + high) / 2
            if utcTimestamps[mid] < timestampUTC {
                low = mid + 1
            } else {
                high = mid
            }
        }
        return low
    }

    private func metadataForSlice(firstUTC: Int64?, lastUTC: Int64?) -> FXMarketMetadata {
        FXMarketMetadata(
            brokerSourceId: metadata.brokerSourceId,
            sourceOrigin: metadata.sourceOrigin,
            logicalSymbol: metadata.logicalSymbol,
            providerSymbol: metadata.providerSymbol,
            timeframe: metadata.timeframe,
            digits: metadata.digits,
            firstUTC: firstUTC,
            lastUTC: lastUTC
        )
    }
}
