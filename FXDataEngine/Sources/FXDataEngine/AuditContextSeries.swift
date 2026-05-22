import Foundation

public struct AuditAsSeriesOHLCV: Codable, Hashable, Sendable {
    public var timeUTC: [Int64]
    public var open: [Double]
    public var high: [Double]
    public var low: [Double]
    public var close: [Double]
    public var volume: [Double]
    public var fillRiskPoints: [Double]

    public var count: Int { close.count }
    public var isConsistent: Bool {
        timeUTC.count == count &&
            open.count == count &&
            high.count == count &&
            low.count == count &&
            volume.count == count &&
            fillRiskPoints.count == count
    }

    public init(
        timeUTC: [Int64] = [],
        open: [Double] = [],
        high: [Double] = [],
        low: [Double] = [],
        close: [Double] = [],
        volume: [Double] = [],
        fillRiskPoints: [Double] = []
    ) {
        self.timeUTC = timeUTC
        self.open = open.map { fxSafeFinite($0) }
        self.high = high.map { fxSafeFinite($0) }
        self.low = low.map { fxSafeFinite($0) }
        self.close = close.map { fxSafeFinite($0) }
        self.volume = volume.map { max(0.0, fxSafeFinite($0)) }
        self.fillRiskPoints = fillRiskPoints.map { max(0.0, fxSafeFinite($0)) }
    }
}

public enum AuditContextSeriesTools {
    public static func rollingCorrelationAsSeries(
        _ a: [Double],
        _ b: [Double],
        startIndex: Int,
        width: Int
    ) -> Double {
        let count = a.count
        guard count == b.count,
              width >= 4,
              startIndex >= 0,
              startIndex + width < count else {
            return 0.0
        }

        var sumA = 0.0
        var sumB = 0.0
        var sumAA = 0.0
        var sumBB = 0.0
        var sumAB = 0.0
        var used = 0

        for offset in 0..<width {
            let returnA = FeatureMath.safeReturn(
                a,
                currentIndex: startIndex + offset,
                previousIndex: startIndex + offset + 1
            )
            let returnB = FeatureMath.safeReturn(
                b,
                currentIndex: startIndex + offset,
                previousIndex: startIndex + offset + 1
            )
            sumA += returnA
            sumB += returnB
            sumAA += returnA * returnA
            sumBB += returnB * returnB
            sumAB += returnA * returnB
            used += 1
        }

        guard used >= 4 else { return 0.0 }
        let countDouble = Double(used)
        let meanA = sumA / countDouble
        let meanB = sumB / countDouble
        let varianceA = sumAA / countDouble - meanA * meanA
        let varianceB = sumBB / countDouble - meanB * meanB
        let covariance = sumAB / countDouble - meanA * meanB
        guard varianceA > 1e-12, varianceB > 1e-12 else { return 0.0 }
        return fxClamp(covariance / sqrt(varianceA * varianceB), -1.0, 1.0)
    }

    public static func reverseChronologicalBarsToSeries(_ bars: [AuditScenarioDoubleBar]) -> AuditAsSeriesOHLCV {
        let reversed = bars.reversed()
        return AuditAsSeriesOHLCV(
            timeUTC: reversed.map(\.timestampUTC),
            open: reversed.map(\.open),
            high: reversed.map(\.high),
            low: reversed.map(\.low),
            close: reversed.map(\.close),
            volume: reversed.map(\.volume),
            fillRiskPoints: reversed.map(\.fillRiskPoints)
        )
    }

    public static func reverseChronologicalCloseSeries(
        timeUTC: [Int64],
        close: [Double]
    ) -> (timeUTC: [Int64], close: [Double]) {
        let count = min(timeUTC.count, close.count)
        guard count > 0 else { return ([], []) }
        let indexes = (0..<count).reversed()
        return (
            indexes.map { timeUTC[$0] },
            indexes.map { fxSafeFinite(close[$0]) }
        )
    }

    public static func aggregateCloseTimeframe(
        chronologicalBars: [AuditScenarioDoubleBar],
        step: Int
    ) -> (timeUTC: [Int64], close: [Double]) {
        let count = chronologicalBars.count
        let bars = step > 0 ? count / step : 0
        guard bars > 0 else { return ([], []) }

        var timeUTC: [Int64] = []
        var close: [Double] = []
        timeUTC.reserveCapacity(bars)
        close.reserveCapacity(bars)
        for bucket in 0..<bars {
            let end = min(count - 1, bucket * step + step - 1)
            timeUTC.append(chronologicalBars[end].timestampUTC)
            close.append(chronologicalBars[end].close)
        }
        return reverseChronologicalCloseSeries(timeUTC: timeUTC, close: close)
    }

    public static func deriveContextSeriesFromBase(
        point: Double,
        base: AuditAsSeriesOHLCV,
        transformID: Int
    ) -> AuditAsSeriesOHLCV {
        let count = base.count
        guard count > 0, base.isConsistent else { return AuditAsSeriesOHLCV() }

        var contextOpen = Array(repeating: 0.0, count: count)
        var contextHigh = Array(repeating: 0.0, count: count)
        var contextLow = Array(repeating: 0.0, count: count)
        var contextClose = Array(repeating: 0.0, count: count)
        var contextVolume = Array(repeating: 0.0, count: count)
        var contextFillRisk = Array(repeating: 0.0, count: count)
        let pt = point > 0.0 ? point : 1e-5

        for index in stride(from: count - 1, through: 0, by: -1) {
            let closeBase = value(base.close, index: index, default: 0.0)
            let previousBase = value(base.close, index: index + 1, default: closeBase)
            let returnBase = previousBase > 0.0 ? (closeBase - previousBase) / previousBase : 0.0
            var closeValue = closeBase
            var gapScale = 0.60
            var rangeScale = 1.0
            var fillRiskScale = 1.0
            var volumeScale = 1.0

            if transformID == 0 {
                closeValue = closeBase * (1.0 + 0.60 * returnBase)
                gapScale = 0.70
                rangeScale = 1.05
                fillRiskScale = 1.05 + 0.20 * fxClamp(abs(returnBase) * 4_000.0, 0.0, 1.0)
                volumeScale = 1.05
            } else if transformID == 1 {
                closeValue = 0.65 * closeBase + 0.35 * previousBase
                gapScale = 0.45
                rangeScale = 0.90
                fillRiskScale = 0.95 + 0.10 * fxClamp(abs(returnBase) * 3_000.0, 0.0, 1.0)
                volumeScale = 0.95
            } else {
                closeValue = closeBase * (1.0 - 0.35 * returnBase)
                gapScale = -0.25
                rangeScale = 1.08
                fillRiskScale = 1.08 + 0.15 * fxClamp(abs(returnBase) * 3_500.0, 0.0, 1.0)
                volumeScale = 1.08
            }

            if closeValue <= pt {
                closeValue = max(closeBase, pt)
            }

            let previousContext = value(contextClose, index: index + 1, default: closeValue)
            let openBase = value(base.open, index: index, default: closeBase)
            let openGap = previousBase > 0.0 ? (openBase - previousBase) / previousBase : 0.0
            var openValue = previousContext * (1.0 + gapScale * openGap)
            if openValue <= pt {
                openValue = previousContext
            }
            if openValue <= pt {
                openValue = closeValue
            }

            let highBase = value(base.high, index: index, default: max(openBase, closeBase))
            let lowBase = value(base.low, index: index, default: min(openBase, closeBase))
            let baseRange = max(highBase - lowBase, pt)
            let baseBodyHigh = max(openBase, closeBase)
            let baseBodyLow = min(openBase, closeBase)
            let upperRatio = max(highBase - baseBodyHigh, 0.0) / baseRange
            let lowerRatio = max(baseBodyLow - lowBase, 0.0) / baseRange
            let scaledRange = rangeScale * baseRange * max(closeValue / max(closeBase, pt), 0.25)
            let upperWick = max(0.25 * pt, scaledRange * upperRatio)
            let lowerWick = max(0.25 * pt, scaledRange * lowerRatio)
            let bodyHigh = max(openValue, closeValue)
            let bodyLow = min(openValue, closeValue)

            contextOpen[index] = openValue
            contextHigh[index] = bodyHigh + upperWick
            contextLow[index] = max(pt, bodyLow - lowerWick)
            contextClose[index] = closeValue
            contextVolume[index] = max(0.0, value(base.volume, index: index, default: 0.0) * volumeScale)
            contextFillRisk[index] = max(0.0, value(base.fillRiskPoints, index: index, default: 0.0) * fillRiskScale)
        }

        return AuditAsSeriesOHLCV(
            timeUTC: base.timeUTC,
            open: contextOpen,
            high: contextHigh,
            low: contextLow,
            close: contextClose,
            volume: contextVolume,
            fillRiskPoints: contextFillRisk
        )
    }

    private static func value(_ values: [Double], index: Int, default defaultValue: Double) -> Double {
        guard index >= 0, index < values.count else { return defaultValue }
        return fxSafeFinite(values[index])
    }
}
