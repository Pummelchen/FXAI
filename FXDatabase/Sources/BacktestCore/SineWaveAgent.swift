import Domain
import Foundation

public enum SineTestSecurity {
    public static let displayName = "SineTest"
    public static let logicalSymbolRawValue = "SINETEST"
    public static let defaultBrokerSourceId = validatedBrokerSourceId("virtual-sinetest")
    public static let logicalSymbol = validatedLogicalSymbol(logicalSymbolRawValue)
    public static let providerSymbol = validatedMT5Symbol(displayName)
    public static let sourceOrigin = DataSourceOrigin.synthetic
    public static let digits = validatedDigits(6)
    public static let genesisUtc = UtcSecond(rawValue: 946_684_800)
    public static let minimumNormalizedValue = 0.001
    public static let syncIntervalSeconds = 10

    public static func matches(_ logicalSymbol: LogicalSymbol) -> Bool {
        logicalSymbol.rawValue == logicalSymbolRawValue
    }

    public static func acceptsProviderSymbol(_ value: String) -> Bool {
        let trimmed = value.trimmingCharacters(in: .whitespacesAndNewlines)
        return trimmed == displayName || trimmed.uppercased() == logicalSymbolRawValue
    }

    private static func validatedBrokerSourceId(_ value: String) -> BrokerSourceId {
        do {
            return try BrokerSourceId(value)
        } catch {
            preconditionFailure("Invalid SineTest broker source id '\(value)': \(error)")
        }
    }

    private static func validatedLogicalSymbol(_ value: String) -> LogicalSymbol {
        do {
            return try LogicalSymbol(value)
        } catch {
            preconditionFailure("Invalid SineTest logical symbol '\(value)': \(error)")
        }
    }

    private static func validatedMT5Symbol(_ value: String) -> MT5Symbol {
        do {
            return try MT5Symbol(value)
        } catch {
            preconditionFailure("Invalid SineTest provider symbol '\(value)': \(error)")
        }
    }

    private static func validatedDigits(_ value: Int) -> Digits {
        do {
            return try Digits(value)
        } catch {
            preconditionFailure("Invalid SineTest digits '\(value)': \(error)")
        }
    }
}

public struct SineWaveAgent: HistoricalOhlcDataProviding {
    public let digits: Digits

    public init(digits: Digits = SineTestSecurity.digits) {
        self.digits = digits
    }

    public func loadM1Ohlc(_ request: HistoricalOhlcRequest) async throws -> ColumnarOhlcSeries {
        guard SineTestSecurity.matches(request.logicalSymbol) else {
            throw HistoryDataError.invalidRequest(
                "SineWaveAgent only serves \(SineTestSecurity.logicalSymbolRawValue)."
            )
        }
        if let expectedMT5Symbol = request.expectedMT5Symbol,
           !SineTestSecurity.acceptsProviderSymbol(expectedMT5Symbol.rawValue) {
            throw HistoryDataError.invalidRequest(
                "SineTest provider symbol mismatch: expected \(SineTestSecurity.displayName), got \(expectedMT5Symbol.rawValue)."
            )
        }
        if let expectedDigits = request.expectedDigits, expectedDigits != digits {
            throw HistoryDataError.invalidRequest(
                "SineTest digits mismatch: expected \(digits.rawValue), got \(expectedDigits.rawValue)."
            )
        }

        return try Self.generateM1Ohlc(
            brokerSourceId: request.brokerSourceId,
            utcStartInclusive: request.utcStartInclusive,
            utcEndExclusive: request.utcEndExclusive,
            digits: digits,
            maximumRows: request.maximumRows,
            allowEmpty: request.allowEmpty
        )
    }

    public static func request(
        brokerSourceId: BrokerSourceId = SineTestSecurity.defaultBrokerSourceId,
        utcStartInclusive: UtcSecond,
        utcEndExclusive: UtcSecond,
        maximumRows: Int? = nil,
        allowEmpty: Bool = false
    ) throws -> HistoricalOhlcRequest {
        try HistoricalOhlcRequest(
            brokerSourceId: brokerSourceId,
            sourceOrigin: SineTestSecurity.sourceOrigin,
            logicalSymbol: SineTestSecurity.logicalSymbol,
            utcStartInclusive: utcStartInclusive,
            utcEndExclusive: utcEndExclusive,
            expectedMT5Symbol: SineTestSecurity.providerSymbol,
            expectedDigits: SineTestSecurity.digits,
            maximumRows: maximumRows,
            allowEmpty: allowEmpty
        )
    }

    public static func generateM1Ohlc(
        brokerSourceIdRawValue: String = SineTestSecurity.defaultBrokerSourceId.rawValue,
        utcStartInclusive: Int64,
        utcEndExclusive: Int64,
        maximumRows: Int? = nil,
        allowEmpty: Bool = false
    ) throws -> ColumnarOhlcSeries {
        try generateM1Ohlc(
            brokerSourceId: BrokerSourceId(brokerSourceIdRawValue),
            utcStartInclusive: UtcSecond(rawValue: utcStartInclusive),
            utcEndExclusive: UtcSecond(rawValue: utcEndExclusive),
            digits: SineTestSecurity.digits,
            maximumRows: maximumRows,
            allowEmpty: allowEmpty
        )
    }

    public static func generateM1Ohlc(
        brokerSourceId: BrokerSourceId = SineTestSecurity.defaultBrokerSourceId,
        utcStartInclusive: UtcSecond,
        utcEndExclusive: UtcSecond,
        digits: Digits = SineTestSecurity.digits,
        maximumRows: Int? = nil,
        allowEmpty: Bool = false
    ) throws -> ColumnarOhlcSeries {
        guard utcStartInclusive.rawValue < utcEndExclusive.rawValue else {
            throw HistoryDataError.invalidRequest("UTC start must be before UTC end.")
        }
        guard utcStartInclusive.isMinuteAligned, utcEndExclusive.isMinuteAligned else {
            throw HistoryDataError.invalidRequest("UTC range boundaries must be minute-aligned.")
        }

        let rowCount64 = (utcEndExclusive.rawValue - utcStartInclusive.rawValue) / Timeframe.m1.seconds
        guard rowCount64 <= Int64(Int.max) else {
            throw HistoryDataError.invalidRequest("SineTest UTC range is too large.")
        }
        let rowCount = Int(rowCount64)
        if let maximumRows, rowCount > maximumRows {
            throw HistoryDataError.rowLimitExceeded(limit: maximumRows)
        }
        guard rowCount > 0 || allowEmpty else {
            throw HistoryDataError.emptyResult(
                SineTestSecurity.logicalSymbol,
                utcStartInclusive,
                utcEndExclusive
            )
        }

        let scale = pow10(digits.rawValue)
        var utcTimestamps: [Int64] = []
        var open: [Int64] = []
        var high: [Int64] = []
        var low: [Int64] = []
        var close: [Int64] = []
        var volume: [UInt64] = []
        utcTimestamps.reserveCapacity(rowCount)
        open.reserveCapacity(rowCount)
        high.reserveCapacity(rowCount)
        low.reserveCapacity(rowCount)
        close.reserveCapacity(rowCount)
        volume.reserveCapacity(rowCount)

        for row in 0..<rowCount {
            let utc = utcStartInclusive.rawValue + Int64(row) * Timeframe.m1.seconds
            let barOpen = scaledPrice(atUTC: utc, scale: scale)
            let barClose = scaledPrice(atUTC: utc + Timeframe.m1.seconds, scale: scale)
            utcTimestamps.append(utc)
            open.append(barOpen)
            high.append(max(barOpen, barClose))
            low.append(min(barOpen, barClose))
            close.append(barClose)
            volume.append(volumeFor(open: barOpen, close: barClose, scale: scale))
        }

        return try ColumnarOhlcSeries(
            metadata: BarSeriesMetadata(
                brokerSourceId: brokerSourceId,
                sourceOrigin: SineTestSecurity.sourceOrigin,
                logicalSymbol: SineTestSecurity.logicalSymbol,
                digits: digits,
                requestedUtcStart: utcStartInclusive,
                requestedUtcEndExclusive: utcEndExclusive,
                firstUtc: utcTimestamps.first.map(UtcSecond.init(rawValue:)),
                lastUtc: utcTimestamps.last.map(UtcSecond.init(rawValue:))
            ),
            utcTimestamps: utcTimestamps,
            open: open,
            high: high,
            low: low,
            close: close,
            volume: volume
        )
    }

    public static func normalizedValue(atUTC utc: Int64) -> Double {
        let minuteIndex = floorDiv(utc, Timeframe.m1.seconds)
        let minuteOfHour = positiveModulo(minuteIndex, 60)
        let radians = 2.0 * Double.pi * Double(minuteOfHour) / 60.0
        let zeroToOne = 0.5 + 0.5 * cos(radians)
        return SineTestSecurity.minimumNormalizedValue + ((1.0 - SineTestSecurity.minimumNormalizedValue) * zeroToOne)
    }

    public static func scaledPrice(atUTC utc: Int64, digits: Digits = SineTestSecurity.digits) -> Int64 {
        scaledPrice(atUTC: utc, scale: pow10(digits.rawValue))
    }

    private static func scaledPrice(atUTC utc: Int64, scale: Int64) -> Int64 {
        let scaled = Int64((normalizedValue(atUTC: utc) * Double(scale)).rounded())
        let minimumScaled = max(1, Int64((SineTestSecurity.minimumNormalizedValue * Double(scale)).rounded()))
        return min(scale, max(minimumScaled, scaled))
    }

    private static func volumeFor(open: Int64, close: Int64, scale: Int64) -> UInt64 {
        let meanLevel = Double(open + close) / (2.0 * Double(scale))
        return UInt64(1_000 + Int((meanLevel * 1_000.0).rounded()))
    }

    private static func pow10(_ exponent: Int) -> Int64 {
        var value: Int64 = 1
        for _ in 0..<exponent {
            value *= 10
        }
        return value
    }

    private static func floorDiv(_ numerator: Int64, _ denominator: Int64) -> Int64 {
        precondition(denominator > 0)
        let quotient = numerator / denominator
        let remainder = numerator % denominator
        return remainder < 0 ? quotient - 1 : quotient
    }

    private static func positiveModulo(_ value: Int64, _ modulus: Int64) -> Int64 {
        let result = value % modulus
        return result >= 0 ? result : result + modulus
    }
}
