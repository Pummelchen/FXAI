import FXDataEngine
import Foundation

enum PluginSineTestSeriesFactory {
    static func makeSeries(
        brokerSourceId: String,
        startUTC: Int64,
        dayCount: Int
    ) throws -> M1OHLCVSeries {
        try makeSeries(
            brokerSourceId: brokerSourceId,
            startUTC: startUTC,
            endUTC: startUTC + Int64(max(1, dayCount) * 24 * 60 * 60)
        )
    }

    static func makeSeries(
        brokerSourceId: String,
        startUTC: Int64,
        endUTC: Int64
    ) throws -> M1OHLCVSeries {
        guard startUTC > 0, endUTC > startUTC, startUTC % 60 == 0, endUTC % 60 == 0 else {
            throw FXDataEngineError.invalidRequest("SineTest fixture range must be positive, increasing, and minute-aligned.")
        }

        let scale: Int64 = 1_000_000
        let rowCount = Int((endUTC - startUTC) / 60)
        var utcTimestamps = ContiguousArray<Int64>()
        var open = ContiguousArray<Int64>()
        var high = ContiguousArray<Int64>()
        var low = ContiguousArray<Int64>()
        var close = ContiguousArray<Int64>()
        var volume = ContiguousArray<UInt64>()
        utcTimestamps.reserveCapacity(rowCount)
        open.reserveCapacity(rowCount)
        high.reserveCapacity(rowCount)
        low.reserveCapacity(rowCount)
        close.reserveCapacity(rowCount)
        volume.reserveCapacity(rowCount)

        for row in 0..<rowCount {
            let utc = startUTC + Int64(row) * 60
            let barOpen = sineScaledPrice(atUTC: utc, scale: scale)
            let barClose = sineScaledPrice(atUTC: utc + 60, scale: scale)
            utcTimestamps.append(utc)
            open.append(barOpen)
            high.append(max(barOpen, barClose))
            low.append(min(barOpen, barClose))
            close.append(barClose)
            let meanLevel = Double(barOpen + barClose) / (2.0 * Double(scale))
            volume.append(UInt64(1_000 + Int((meanLevel * 1_000.0).rounded())))
        }

        return try M1OHLCVSeries(
            metadata: FXMarketMetadata(
                brokerSourceId: brokerSourceId,
                sourceOrigin: "SYNTHETIC",
                logicalSymbol: "SINETEST",
                providerSymbol: "SineTest",
                digits: 6,
                firstUTC: utcTimestamps.first,
                lastUTC: utcTimestamps.last
            ),
            utcTimestamps: utcTimestamps,
            open: open,
            high: high,
            low: low,
            close: close,
            volume: volume
        )
    }

    private static func sineScaledPrice(atUTC utc: Int64, scale: Int64) -> Int64 {
        let minuteIndex = floorDiv(utc, 60)
        let minuteOfHour = positiveModulo(minuteIndex, 60)
        let radians = 2.0 * Double.pi * Double(minuteOfHour) / 60.0
        let zeroToOne = 0.5 + 0.5 * cos(radians)
        let normalized = 0.001 + ((1.0 - 0.001) * zeroToOne)
        let scaled = Int64((normalized * Double(scale)).rounded())
        return min(scale, max(1_000, scaled))
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
