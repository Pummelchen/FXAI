import Foundation
import FXBacktestAPI

public struct FXMarketMetadata: Codable, Hashable, Sendable {
    public let brokerSourceId: String
    public let sourceOrigin: String
    public let logicalSymbol: String
    public let providerSymbol: String?
    public let timeframe: MarketTimeframe
    public let digits: Int
    public let firstUTC: Int64?
    public let lastUTC: Int64?

    public init(
        brokerSourceId: String,
        sourceOrigin: String,
        logicalSymbol: String,
        providerSymbol: String? = nil,
        timeframe: MarketTimeframe = .m1,
        digits: Int,
        firstUTC: Int64? = nil,
        lastUTC: Int64? = nil
    ) {
        self.brokerSourceId = brokerSourceId
        self.sourceOrigin = sourceOrigin
        self.logicalSymbol = logicalSymbol.uppercased()
        self.providerSymbol = providerSymbol
        self.timeframe = timeframe
        self.digits = digits
        self.firstUTC = firstUTC
        self.lastUTC = lastUTC
    }
}

public struct M1OHLCVBar: Codable, Hashable, Sendable {
    public let utcTimestamp: Int64
    public let open: Int64
    public let high: Int64
    public let low: Int64
    public let close: Int64
    public let volume: UInt64

    public init(utcTimestamp: Int64, open: Int64, high: Int64, low: Int64, close: Int64, volume: UInt64 = 0) {
        self.utcTimestamp = utcTimestamp
        self.open = open
        self.high = high
        self.low = low
        self.close = close
        self.volume = volume
    }
}

public struct M1OHLCVSeries: Sendable {
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
        guard metadata.timeframe == .m1 else {
            throw FXDataEngineError.invalidRequest("FXDataEngine consumes canonical M1 data only.")
        }
        guard (0...10).contains(metadata.digits) else {
            throw FXDataEngineError.validation("digits must be in 0...10")
        }

        let rowCount = utcTimestamps.count
        guard open.count == rowCount,
              high.count == rowCount,
              low.count == rowCount,
              close.count == rowCount,
              volume.count == rowCount else {
            throw FXDataEngineError.validation("M1 OHLCV columns must have equal length")
        }

        for index in 0..<rowCount {
            let timestamp = utcTimestamps[index]
            guard timestamp > 0, timestamp % 60 == 0 else {
                throw FXDataEngineError.validation("utc timestamp at row \(index) must be positive and minute-aligned")
            }
            if index > 0, timestamp <= utcTimestamps[index - 1] {
                throw FXDataEngineError.validation("utc timestamps must be strictly increasing")
            }
            guard open[index] > 0, high[index] > 0, low[index] > 0, close[index] > 0 else {
                throw FXDataEngineError.validation("OHLC prices must be positive at row \(index)")
            }
            guard high[index] >= open[index],
                  high[index] >= close[index],
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

    public init(response: FXBacktestM1HistoryResponse) throws {
        try response.validate()
        try self.init(
            metadata: FXMarketMetadata(
                brokerSourceId: response.metadata.brokerSourceId,
                sourceOrigin: response.metadata.sourceOrigin,
                logicalSymbol: response.metadata.logicalSymbol,
                providerSymbol: response.metadata.mt5Symbol,
                timeframe: .m1,
                digits: response.metadata.digits,
                firstUTC: response.metadata.firstUtc,
                lastUTC: response.metadata.lastUtc
            ),
            utcTimestamps: ContiguousArray(response.utcTimestamps),
            open: ContiguousArray(response.open),
            high: ContiguousArray(response.high),
            low: ContiguousArray(response.low),
            close: ContiguousArray(response.close),
            volume: ContiguousArray(response.volume)
        )
    }

    @inlinable
    public func bar(at index: Int) -> M1OHLCVBar {
        M1OHLCVBar(
            utcTimestamp: utcTimestamps[index],
            open: open[index],
            high: high[index],
            low: low[index],
            close: close[index],
            volume: volume[index]
        )
    }

    @inlinable
    public func price(_ scaled: Int64) -> Double {
        Double(scaled) / pow(10.0, Double(metadata.digits))
    }

    @inlinable
    public func normalizedReturn(from olderIndex: Int, to newerIndex: Int) -> Double {
        guard olderIndex >= 0,
              newerIndex >= 0,
              olderIndex < count,
              newerIndex < count else { return 0.0 }
        let older = Double(close[olderIndex])
        let newer = Double(close[newerIndex])
        guard older > 0 else { return 0.0 }
        return fxClampSignedUnit((newer - older) / older * 100.0)
    }
}

public struct MarketUniverse: Sendable {
    public let primarySymbol: String
    public let seriesBySymbol: [String: M1OHLCVSeries]
    public let symbols: [String]

    public init(primarySymbol: String? = nil, series: [M1OHLCVSeries], requireAlignedTimestamps: Bool = true) throws {
        guard !series.isEmpty else {
            throw FXDataEngineError.invalidRequest("market universe requires at least one M1 OHLCV series")
        }

        var bySymbol: [String: M1OHLCVSeries] = [:]
        bySymbol.reserveCapacity(series.count)
        for item in series {
            let symbol = item.metadata.logicalSymbol.uppercased()
            guard bySymbol[symbol] == nil else {
                throw FXDataEngineError.validation("duplicate market series for \(symbol)")
            }
            bySymbol[symbol] = item
        }

        let resolvedPrimary = (primarySymbol ?? series[0].metadata.logicalSymbol).uppercased()
        guard bySymbol[resolvedPrimary] != nil else {
            throw FXDataEngineError.validation("primary symbol \(resolvedPrimary) is not present")
        }

        if requireAlignedTimestamps {
            try Self.validateAligned(series)
        }

        self.primarySymbol = resolvedPrimary
        self.seriesBySymbol = bySymbol
        self.symbols = bySymbol.keys.sorted()
    }

    public var primary: M1OHLCVSeries {
        seriesBySymbol[primarySymbol]!
    }

    public subscript(symbol: String) -> M1OHLCVSeries? {
        seriesBySymbol[symbol.uppercased()]
    }

    public func contextSeries(excludingPrimary: Bool = true) -> [M1OHLCVSeries] {
        symbols
            .filter { !excludingPrimary || $0 != primarySymbol }
            .compactMap { seriesBySymbol[$0] }
    }

    private static func validateAligned(_ series: [M1OHLCVSeries]) throws {
        guard let reference = series.first else { return }
        for candidate in series.dropFirst() {
            guard candidate.count == reference.count else {
                throw FXDataEngineError.validation("\(candidate.metadata.logicalSymbol) row count does not match primary")
            }
            for index in 0..<reference.count where candidate.utcTimestamps[index] != reference.utcTimestamps[index] {
                throw FXDataEngineError.validation("\(candidate.metadata.logicalSymbol) timestamp mismatch at row \(index)")
            }
        }
    }
}

public extension M1OHLCVSeries {
    static func demoEURUSD(barCount: Int = 3_000) throws -> M1OHLCVSeries {
        let safeCount = max(300, barCount)
        let start = Int64(1_704_067_200)
        var utc = ContiguousArray<Int64>()
        var open = ContiguousArray<Int64>()
        var high = ContiguousArray<Int64>()
        var low = ContiguousArray<Int64>()
        var close = ContiguousArray<Int64>()
        var volume = ContiguousArray<UInt64>()
        utc.reserveCapacity(safeCount)
        open.reserveCapacity(safeCount)
        high.reserveCapacity(safeCount)
        low.reserveCapacity(safeCount)
        close.reserveCapacity(safeCount)
        volume.reserveCapacity(safeCount)

        var price = Int64(108_000)
        for index in 0..<safeCount {
            let cycle = Int64((index % 144) - 72)
            let drift = Int64(index / 500)
            let move = (cycle / 9) + (index % 17 == 0 ? 7 : -2) + drift
            let barOpen = price
            let barClose = max(90_000, price + move)
            let barHigh = max(barOpen, barClose) + Int64(5 + (index % 6))
            let barLow = min(barOpen, barClose) - Int64(5 + (index % 4))
            utc.append(start + Int64(index * 60))
            open.append(barOpen)
            high.append(barHigh)
            low.append(barLow)
            close.append(barClose)
            volume.append(UInt64(100 + (index % 21)))
            price = barClose
        }

        return try M1OHLCVSeries(
            metadata: FXMarketMetadata(
                brokerSourceId: "demo",
                sourceOrigin: "DEMO",
                logicalSymbol: "EURUSD",
                providerSymbol: "EURUSD",
                digits: 5,
                firstUTC: utc.first,
                lastUTC: utc.last
            ),
            utcTimestamps: utc,
            open: open,
            high: high,
            low: low,
            close: close,
            volume: volume
        )
    }
}
