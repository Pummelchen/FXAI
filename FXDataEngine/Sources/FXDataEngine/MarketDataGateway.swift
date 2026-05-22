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

public struct FXDatabaseMarketUniverseRequest: Codable, Hashable, Sendable {
    public var brokerSourceId: String
    public var sourceOrigin: String
    public var primarySymbol: String
    public var contextSymbols: [String]
    public var expectedProviderSymbolsBySymbol: [String: String]
    public var expectedDigitsBySymbol: [String: Int]
    public var utcStartInclusive: Int64
    public var utcEndExclusive: Int64
    public var maximumRowsPerSymbol: Int
    public var requireAlignedTimestamps: Bool

    public init(
        brokerSourceId: String,
        sourceOrigin: String = "MT5",
        primarySymbol: String,
        contextSymbols: [String] = [],
        expectedProviderSymbolsBySymbol: [String: String] = [:],
        expectedDigitsBySymbol: [String: Int] = [:],
        utcStartInclusive: Int64,
        utcEndExclusive: Int64,
        maximumRowsPerSymbol: Int = FXBacktestAPIV1.maximumRowsLimit,
        requireAlignedTimestamps: Bool = true
    ) {
        let normalizedPrimary = DataCoreRequest.normalizedSymbol(primarySymbol)
        self.brokerSourceId = brokerSourceId.trimmingCharacters(in: .whitespacesAndNewlines)
        self.sourceOrigin = sourceOrigin.trimmingCharacters(in: .whitespacesAndNewlines).uppercased()
        self.primarySymbol = normalizedPrimary
        self.contextSymbols = DataCoreRequest.normalizedContextSymbols(contextSymbols)
            .filter { $0 != normalizedPrimary }
        self.expectedProviderSymbolsBySymbol = Self.normalizedProviderSymbols(expectedProviderSymbolsBySymbol)
        self.expectedDigitsBySymbol = Self.normalizedDigits(expectedDigitsBySymbol)
        self.utcStartInclusive = utcStartInclusive
        self.utcEndExclusive = utcEndExclusive
        self.maximumRowsPerSymbol = max(1, min(maximumRowsPerSymbol, FXBacktestAPIV1.maximumRowsLimit))
        self.requireAlignedTimestamps = requireAlignedTimestamps
    }

    public var symbols: [String] {
        [primarySymbol] + contextSymbols
    }

    public func validate() throws {
        guard !brokerSourceId.isEmpty else {
            throw FXDataEngineError.invalidRequest("brokerSourceId must not be empty")
        }
        guard !primarySymbol.isEmpty else {
            throw FXDataEngineError.invalidRequest("primarySymbol must not be empty")
        }
        guard utcStartInclusive < utcEndExclusive else {
            throw FXDataEngineError.invalidRequest("UTC range start must be before end")
        }
        guard utcStartInclusive % 60 == 0, utcEndExclusive % 60 == 0 else {
            throw FXDataEngineError.invalidRequest("UTC range boundaries must be minute-aligned")
        }
        for (symbol, digits) in expectedDigitsBySymbol {
            guard (0...10).contains(digits) else {
                throw FXDataEngineError.invalidRequest("expected digits for \(symbol) must be in 0...10")
            }
        }
    }

    public func historyRequests() throws -> [FXDatabaseMarketHistoryRequest] {
        try validate()
        return symbols.map { symbol in
            FXDatabaseMarketHistoryRequest(
                brokerSourceId: brokerSourceId,
                sourceOrigin: sourceOrigin,
                logicalSymbol: symbol,
                expectedProviderSymbol: expectedProviderSymbolsBySymbol[symbol],
                expectedDigits: expectedDigitsBySymbol[symbol],
                utcStartInclusive: utcStartInclusive,
                utcEndExclusive: utcEndExclusive,
                maximumRows: maximumRowsPerSymbol
            )
        }
    }

    private static func normalizedProviderSymbols(_ input: [String: String]) -> [String: String] {
        var output: [String: String] = [:]
        output.reserveCapacity(input.count)
        for (rawSymbol, rawProvider) in input {
            let symbol = DataCoreRequest.normalizedSymbol(rawSymbol)
            let provider = rawProvider.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !symbol.isEmpty, !provider.isEmpty else { continue }
            output[symbol] = provider
        }
        return output
    }

    private static func normalizedDigits(_ input: [String: Int]) -> [String: Int] {
        var output: [String: Int] = [:]
        output.reserveCapacity(input.count)
        for (rawSymbol, digits) in input {
            let symbol = DataCoreRequest.normalizedSymbol(rawSymbol)
            guard !symbol.isEmpty else { continue }
            output[symbol] = digits
        }
        return output
    }
}

public struct MarketOHLCVBar: Codable, Hashable, Sendable {
    public let utcTimestamp: Int64
    public let timeframe: MarketTimeframe
    public let open: Int64
    public let high: Int64
    public let low: Int64
    public let close: Int64
    public let volume: UInt64

    public init(
        utcTimestamp: Int64,
        timeframe: MarketTimeframe,
        open: Int64,
        high: Int64,
        low: Int64,
        close: Int64,
        volume: UInt64 = 0
    ) {
        self.utcTimestamp = utcTimestamp
        self.timeframe = timeframe
        self.open = open
        self.high = high
        self.low = low
        self.close = close
        self.volume = volume
    }
}

public struct MarketTimeframeOHLCVSeries: Sendable {
    public let metadata: FXMarketMetadata
    public let utcTimestamps: ContiguousArray<Int64>
    public let open: ContiguousArray<Int64>
    public let high: ContiguousArray<Int64>
    public let low: ContiguousArray<Int64>
    public let close: ContiguousArray<Int64>
    public let volume: ContiguousArray<UInt64>

    @inlinable public var count: Int { utcTimestamps.count }
    @inlinable public var isEmpty: Bool { count == 0 }
    @inlinable public var hasVolume: Bool { volume.contains { $0 > 0 } }

    public init(
        metadata: FXMarketMetadata,
        utcTimestamps: ContiguousArray<Int64>,
        open: ContiguousArray<Int64>,
        high: ContiguousArray<Int64>,
        low: ContiguousArray<Int64>,
        close: ContiguousArray<Int64>,
        volume: ContiguousArray<UInt64>
    ) throws {
        guard (0...10).contains(metadata.digits) else {
            throw FXDataEngineError.validation("digits must be in 0...10")
        }
        let rowCount = utcTimestamps.count
        guard open.count == rowCount,
              high.count == rowCount,
              low.count == rowCount,
              close.count == rowCount,
              volume.count == rowCount else {
            throw FXDataEngineError.validation("OHLCV columns must have equal length")
        }

        let stepSeconds = Int64(metadata.timeframe.minutes * 60)
        for index in 0..<rowCount {
            let timestamp = utcTimestamps[index]
            guard timestamp > 0, timestamp % stepSeconds == 0 else {
                throw FXDataEngineError.validation("\(metadata.timeframe.rawValue) timestamp at row \(index) must be positive and timeframe-aligned")
            }
            if index > 0, timestamp <= utcTimestamps[index - 1] {
                throw FXDataEngineError.validation("utc timestamps must be strictly increasing")
            }
            guard open[index] > 0, high[index] > 0, low[index] > 0, close[index] > 0 else {
                throw FXDataEngineError.validation("OHLC prices must be positive at row \(index)")
            }
            guard high[index] >= open[index],
                  high[index] >= close[index],
                  high[index] >= low[index],
                  low[index] <= open[index],
                  low[index] <= close[index] else {
                throw FXDataEngineError.validation("OHLC invariant failed at row \(index)")
            }
        }

        self.metadata = metadata
        self.utcTimestamps = utcTimestamps
        self.open = open
        self.high = high
        self.low = low
        self.close = close
        self.volume = volume
    }

    @inlinable
    public func bar(at index: Int) -> MarketOHLCVBar {
        MarketOHLCVBar(
            utcTimestamp: utcTimestamps[index],
            timeframe: metadata.timeframe,
            open: open[index],
            high: high[index],
            low: low[index],
            close: close[index],
            volume: volume[index]
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

    public func loadUniverse(
        connection: FXDatabaseMarketConnection,
        request: FXDatabaseMarketUniverseRequest
    ) async throws -> MarketUniverse {
        try await loadUniverse(
            connection: connection,
            requests: try request.historyRequests(),
            primarySymbol: request.primarySymbol,
            requireAlignedTimestamps: request.requireAlignedTimestamps
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

    public static func resample(
        _ series: M1OHLCVSeries,
        to timeframe: MarketTimeframe,
        includePartialBuckets: Bool = false
    ) throws -> MarketTimeframeOHLCVSeries {
        try series.resampled(to: timeframe, includePartialBuckets: includePartialBuckets)
    }

    public static func alignedIndexMap(
        referenceM1: M1OHLCVSeries,
        target: MarketTimeframeOHLCVSeries,
        maxLagSeconds: Int,
        upToIndex: Int? = nil
    ) -> [Int] {
        DataCoreAlignment.buildAlignedIndexMap(
            referenceTimesAscending: referenceM1.utcTimestamps,
            targetTimesAscending: target.utcTimestamps,
            maxLagSeconds: maxLagSeconds,
            upToIndex: upToIndex
        )
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

    func resampled(to timeframe: MarketTimeframe, includePartialBuckets: Bool = false) throws -> MarketTimeframeOHLCVSeries {
        if timeframe == .m1 {
            return try MarketTimeframeOHLCVSeries(
                metadata: metadata,
                utcTimestamps: utcTimestamps,
                open: open,
                high: high,
                low: low,
                close: close,
                volume: volume
            )
        }

        let stepSeconds = Int64(timeframe.minutes * 60)
        var outTime = ContiguousArray<Int64>()
        var outOpen = ContiguousArray<Int64>()
        var outHigh = ContiguousArray<Int64>()
        var outLow = ContiguousArray<Int64>()
        var outClose = ContiguousArray<Int64>()
        var outVolume = ContiguousArray<UInt64>()

        var bucketStart: Int64?
        var bucketOpen: Int64 = 0
        var bucketHigh: Int64 = 0
        var bucketLow: Int64 = 0
        var bucketClose: Int64 = 0
        var bucketVolume: UInt64 = 0
        var bucketCount = 0

        func appendBucketIfNeeded() {
            guard let bucketStart else { return }
            guard includePartialBuckets || bucketCount == timeframe.minutes else { return }
            outTime.append(bucketStart)
            outOpen.append(bucketOpen)
            outHigh.append(bucketHigh)
            outLow.append(bucketLow)
            outClose.append(bucketClose)
            outVolume.append(bucketVolume)
        }

        for index in 0..<count {
            let currentBucket = (utcTimestamps[index] / stepSeconds) * stepSeconds
            if bucketStart != currentBucket {
                appendBucketIfNeeded()
                bucketStart = currentBucket
                bucketOpen = open[index]
                bucketHigh = high[index]
                bucketLow = low[index]
                bucketClose = close[index]
                bucketVolume = volume[index]
                bucketCount = 1
            } else {
                bucketHigh = max(bucketHigh, high[index])
                bucketLow = min(bucketLow, low[index])
                bucketClose = close[index]
                bucketVolume = bucketVolume.saturatingAdd(volume[index])
                bucketCount += 1
            }
        }
        appendBucketIfNeeded()

        return try MarketTimeframeOHLCVSeries(
            metadata: FXMarketMetadata(
                brokerSourceId: metadata.brokerSourceId,
                sourceOrigin: metadata.sourceOrigin,
                logicalSymbol: metadata.logicalSymbol,
                providerSymbol: metadata.providerSymbol,
                timeframe: timeframe,
                digits: metadata.digits,
                firstUTC: outTime.first,
                lastUTC: outTime.last
            ),
            utcTimestamps: outTime,
            open: outOpen,
            high: outHigh,
            low: outLow,
            close: outClose,
            volume: outVolume
        )
    }
}

private extension UInt64 {
    func saturatingAdd(_ other: UInt64) -> UInt64 {
        let result = addingReportingOverflow(other)
        return result.overflow ? UInt64.max : result.partialValue
    }
}
