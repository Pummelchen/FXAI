import Foundation

public struct CandleGeometryFeatures: Codable, Hashable, Sendable {
    public var bodyNorm: Double
    public var upperWickNorm: Double
    public var lowerWickNorm: Double
    public var rangeNorm: Double

    public init(bodyNorm: Double, upperWickNorm: Double, lowerWickNorm: Double, rangeNorm: Double) {
        self.bodyNorm = fxSafeFinite(bodyNorm)
        self.upperWickNorm = fxClampUnit(upperWickNorm)
        self.lowerWickNorm = fxClampUnit(lowerWickNorm)
        self.rangeNorm = fxSafeFinite(rangeNorm)
    }
}

public enum FeatureMath {
    public static func safeReturn(_ values: [Double], currentIndex: Int, previousIndex: Int) -> Double {
        guard currentIndex >= 0,
              previousIndex >= 0,
              currentIndex < values.count,
              previousIndex < values.count else {
            return 0.0
        }
        let current = fxSafeFinite(values[currentIndex])
        let previous = fxSafeFinite(values[previousIndex])
        guard previous > 0.0 else { return 0.0 }
        return (current - previous) / previous
    }

    public static func normalizedSlopeAsSeries(_ values: [Double], startIndex: Int, width: Int) -> Double {
        guard width >= 2,
              startIndex >= 0,
              startIndex + width <= values.count else {
            return 0.0
        }

        var sumX = 0.0
        var sumX2 = 0.0
        var sumY = 0.0
        var sumXY = 0.0
        for offset in 0..<width {
            let x = Double(offset)
            let y = fxSafeFinite(values[startIndex + offset])
            sumX += x
            sumX2 += x * x
            sumY += y
            sumXY += x * y
        }

        let denom = Double(width) * sumX2 - sumX * sumX
        guard denom != 0.0 else { return 0.0 }
        let slope = (Double(width) * sumXY - sumX * sumY) / denom
        let reference = fxSafeFinite(values[startIndex])
        guard reference != 0.0 else { return 0.0 }
        return (-slope) / reference
    }

    public static func estimateExpectedAbsMovePointsAsSeries(
        close: [Double],
        horizonM1: Int,
        sampleCount: Int,
        point: Double
    ) -> Double {
        let count = close.count
        guard count > horizonM1 + 1, horizonM1 >= 0, point > 0.0 else { return 0.0 }
        let maxIndex = count - 1
        let stopIndex = min(maxIndex, horizonM1 + sampleCount)
        guard horizonM1 <= stopIndex else { return 0.0 }

        var sumAbs = 0.0
        var used = 0
        for index in horizonM1...stopIndex {
            let futureIndex = index - horizonM1
            guard futureIndex >= 0, futureIndex < count else { continue }
            sumAbs += abs(movePoints(priceNow: close[index], priceFuture: close[futureIndex], point: point))
            used += 1
        }
        return used > 0 ? sumAbs / Double(used) : 0.0
    }

    public static func estimateExpectedAbsMovePointsAtIndexAsSeries(
        close: [Double],
        startIndex: Int,
        horizonM1: Int,
        sampleCount: Int,
        point: Double
    ) -> Double {
        let count = close.count
        guard count > 0,
              startIndex >= 0,
              startIndex < count,
              horizonM1 >= 1,
              point > 0.0 else {
            return 0.0
        }

        let oldestNeeded = min(count - 1, startIndex + horizonM1 + sampleCount)
        let first = startIndex + horizonM1
        guard first <= oldestNeeded else { return 0.0 }

        var sumAbs = 0.0
        var used = 0
        for index in first...oldestNeeded {
            let futureIndex = index - horizonM1
            guard futureIndex >= startIndex, futureIndex < count else { continue }
            sumAbs += abs(movePoints(priceNow: close[index], priceFuture: close[futureIndex], point: point))
            used += 1
        }
        return used > 0 ? sumAbs / Double(used) : 0.0
    }

    public static func rollingAbsReturnAsSeries(_ values: [Double], startIndex: Int, width: Int) -> Double {
        guard width >= 2, startIndex >= 0, startIndex < values.count else { return 0.0 }
        let effectiveWidth = min(width, values.count - startIndex - 1)
        guard effectiveWidth >= 2 else { return 0.0 }

        var sumAbs = 0.0
        var used = 0
        for offset in 0..<effectiveWidth {
            sumAbs += abs(safeReturn(values, currentIndex: startIndex + offset, previousIndex: startIndex + offset + 1))
            used += 1
        }
        return used > 0 ? sumAbs / Double(used) : 0.0
    }

    public static func rollingReturnStdAsSeries(_ values: [Double], startIndex: Int, width: Int) -> Double {
        guard width >= 2, startIndex >= 0, startIndex < values.count else { return 0.0 }
        let effectiveWidth = min(width, values.count - startIndex - 1)
        guard effectiveWidth >= 2 else { return 0.0 }

        var sum = 0.0
        var sum2 = 0.0
        var used = 0
        for offset in 0..<effectiveWidth {
            let value = safeReturn(values, currentIndex: startIndex + offset, previousIndex: startIndex + offset + 1)
            sum += value
            sum2 += value * value
            used += 1
        }
        guard used >= 2 else { return 0.0 }
        let mean = sum / Double(used)
        return sqrt(max((sum2 / Double(used)) - mean * mean, 0.0))
    }

    public static func smaAsSeries(_ values: [Double], startIndex: Int, period: Int) -> Double {
        guard period > 0,
              startIndex >= 0,
              startIndex < values.count,
              startIndex + period <= values.count else {
            return 0.0
        }
        var sum = 0.0
        for offset in 0..<period {
            sum += fxSafeFinite(values[startIndex + offset])
        }
        return sum / Double(period)
    }

    public static func emaAsSeries(_ values: [Double], startIndex: Int, period: Int) -> Double {
        guard period > 1,
              startIndex >= 0,
              startIndex < values.count,
              startIndex + period <= values.count else {
            return 0.0
        }
        let oldest = startIndex + period - 1
        var ema = fxSafeFinite(values[oldest])
        let alpha = 2.0 / (Double(period) + 1.0)
        let oneMinusAlpha = 1.0 - alpha
        if oldest > startIndex {
            for index in stride(from: oldest - 1, through: startIndex, by: -1) {
                ema = alpha * fxSafeFinite(values[index]) + oneMinusAlpha * ema
            }
        }
        return ema
    }

    public static func movingAverageEdgeFeature(refPrice: Double, movingAverage: Double, volatilityUnit: Double) -> Double {
        guard refPrice > 0.0, movingAverage > 0.0, volatilityUnit > 0.0 else { return 0.0 }
        return ((refPrice - movingAverage) / movingAverage) / volatilityUnit
    }

    public static func candleGeometryNormalize(
        open: Double,
        high: Double,
        low: Double,
        close: Double,
        previousClose: Double,
        epsilon: Double = 1e-8
    ) -> CandleGeometryFeatures {
        let safeEpsilon = epsilon > 0.0 ? epsilon : 1e-8
        let range = abs(high - low)
        let closeDenominator = max(abs(previousClose), safeEpsilon)
        let rangeDenominator = max(range, safeEpsilon)
        let upperWick = max(high - max(open, close), 0.0)
        let lowerWick = max(min(open, close) - low, 0.0)
        return CandleGeometryFeatures(
            bodyNorm: (close - open) / closeDenominator,
            upperWickNorm: upperWick / rangeDenominator,
            lowerWickNorm: lowerWick / rangeDenominator,
            rangeNorm: range / closeDenominator
        )
    }

    public static func qsdemaAsSeries(_ values: [Double], startIndex: Int, period: Int) -> Double {
        guard period > 1, startIndex >= 0, startIndex < values.count else { return 0.0 }
        var warmup = period * 6
        if warmup < period + 20 {
            warmup = period + 20
        }
        let oldest = min(values.count - 1, startIndex + warmup - 1)
        guard oldest - startIndex + 1 >= period else { return 0.0 }

        var ema1 = Array(repeating: fxSafeFinite(values[oldest]), count: 4)
        var ema2 = ema1
        var dema = ema1
        let alpha = 2.0 / (Double(period) + 1.0)
        let oneMinusAlpha = 1.0 - alpha

        if oldest > startIndex {
            for index in stride(from: oldest - 1, through: startIndex, by: -1) {
                var value = fxSafeFinite(values[index])
                for pass in 0..<4 {
                    ema1[pass] = alpha * value + oneMinusAlpha * ema1[pass]
                    ema2[pass] = alpha * ema1[pass] + oneMinusAlpha * ema2[pass]
                    dema[pass] = 2.0 * ema1[pass] - ema2[pass]
                    value = dema[pass]
                }
            }
        }
        return dema[3]
    }

    public static func rsiAsSeries(_ values: [Double], startIndex: Int, period: Int) -> Double {
        guard period > 1,
              startIndex >= 0,
              startIndex < values.count,
              startIndex + period < values.count else {
            return 50.0
        }

        var gain = 0.0
        var loss = 0.0
        for offset in stride(from: period, through: 1, by: -1) {
            let older = startIndex + offset
            let newer = older - 1
            let delta = fxSafeFinite(values[newer]) - fxSafeFinite(values[older])
            if delta > 0.0 {
                gain += delta
            } else {
                loss -= delta
            }
        }

        let averageGain = gain / Double(period)
        let averageLoss = loss / Double(period)
        if averageLoss <= 1e-12, averageGain <= 1e-12 { return 50.0 }
        if averageLoss <= 1e-12 { return 100.0 }
        let ratio = averageGain / averageLoss
        return 100.0 - (100.0 / (1.0 + ratio))
    }

    public static func atrAsSeries(high: [Double], low: [Double], close: [Double], startIndex: Int, period: Int) -> Double {
        let count = close.count
        guard period > 1,
              startIndex >= 0,
              high.count == count,
              low.count == count,
              count > 0,
              startIndex + period < count else {
            return 0.0
        }

        var sumTrueRange = 0.0
        var used = 0
        let last = startIndex + period - 1
        for index in startIndex...last {
            let previousClose = fxSafeFinite(close[index + 1])
            let highValue = fxSafeFinite(high[index])
            let lowValue = fxSafeFinite(low[index])
            let trueRange = max(abs(highValue - lowValue), abs(highValue - previousClose), abs(lowValue - previousClose))
            sumTrueRange += trueRange
            used += 1
        }
        return used > 0 ? sumTrueRange / Double(used) : 0.0
    }

    public static func parkinsonVolAsSeries(high: [Double], low: [Double], startIndex: Int, period: Int) -> Double {
        guard period > 1,
              startIndex >= 0,
              high.count == low.count,
              !high.isEmpty,
              startIndex + period <= high.count else {
            return 0.0
        }

        var sumSquared = 0.0
        var used = 0
        let last = startIndex + period - 1
        for index in startIndex...last {
            var highValue = fxSafeFinite(high[index])
            var lowValue = fxSafeFinite(low[index])
            guard highValue > 0.0, lowValue > 0.0 else { continue }
            if highValue < lowValue {
                swap(&highValue, &lowValue)
            }
            guard highValue > lowValue else { continue }
            let logRange = log(highValue / lowValue)
            sumSquared += logRange * logRange
            used += 1
        }
        guard used > 1 else { return 0.0 }
        let denominator = 4.0 * log(2.0) * Double(used)
        guard denominator > 0.0 else { return 0.0 }
        return sqrt(sumSquared / denominator)
    }

    public static func rollingMedianAsSeries(_ values: [Double], startIndex: Int, period: Int) -> Double {
        guard period > 1,
              startIndex >= 0,
              startIndex < values.count,
              startIndex + period <= values.count else {
            return 0.0
        }
        let sorted = sortedSmall(Array(values[startIndex..<(startIndex + period)]))
        if period % 2 == 1 {
            return sorted[period / 2]
        }
        let middle = period / 2
        return 0.5 * (sorted[middle - 1] + sorted[middle])
    }

    public static func rollingMADAsSeries(_ values: [Double], startIndex: Int, period: Int, median: Double) -> Double {
        guard period > 1,
              startIndex >= 0,
              startIndex < values.count,
              startIndex + period <= values.count else {
            return 0.0
        }
        var deviations: [Double] = []
        deviations.reserveCapacity(period)
        for offset in 0..<period {
            deviations.append(abs(fxSafeFinite(values[startIndex + offset]) - median))
        }
        let sorted = sortedSmall(deviations)
        if period % 2 == 1 {
            return sorted[period / 2]
        }
        let middle = period / 2
        return 0.5 * (sorted[middle - 1] + sorted[middle])
    }

    public static func rogersSatchellVolAsSeries(
        open: [Double],
        high: [Double],
        low: [Double],
        close: [Double],
        startIndex: Int,
        period: Int
    ) -> Double {
        let count = close.count
        guard period > 1,
              startIndex >= 0,
              startIndex + period <= count,
              open.count == count,
              high.count == count,
              low.count == count else {
            return 0.0
        }

        var sum = 0.0
        var used = 0
        let last = startIndex + period - 1
        for index in startIndex...last {
            let openValue = fxSafeFinite(open[index])
            var highValue = fxSafeFinite(high[index])
            var lowValue = fxSafeFinite(low[index])
            let closeValue = fxSafeFinite(close[index])
            guard openValue > 0.0, highValue > 0.0, lowValue > 0.0, closeValue > 0.0 else { continue }
            if highValue < lowValue {
                swap(&highValue, &lowValue)
            }
            guard highValue > lowValue else { continue }
            let term = log(highValue / closeValue) * log(highValue / openValue) +
                log(lowValue / closeValue) * log(lowValue / openValue)
            sum += max(term, 0.0)
            used += 1
        }
        return used > 1 ? sqrt(sum / Double(used)) : 0.0
    }

    public static func garmanKlassVolAsSeries(
        open: [Double],
        high: [Double],
        low: [Double],
        close: [Double],
        startIndex: Int,
        period: Int
    ) -> Double {
        let count = close.count
        guard period > 1,
              startIndex >= 0,
              startIndex + period <= count,
              open.count == count,
              high.count == count,
              low.count == count else {
            return 0.0
        }

        let coefficient = (2.0 * log(2.0)) - 1.0
        var sum = 0.0
        var used = 0
        let last = startIndex + period - 1
        for index in startIndex...last {
            let openValue = fxSafeFinite(open[index])
            var highValue = fxSafeFinite(high[index])
            var lowValue = fxSafeFinite(low[index])
            let closeValue = fxSafeFinite(close[index])
            guard openValue > 0.0, highValue > 0.0, lowValue > 0.0, closeValue > 0.0 else { continue }
            if highValue < lowValue {
                swap(&highValue, &lowValue)
            }
            guard highValue > lowValue else { continue }
            let highLow = log(highValue / lowValue)
            let closeOpen = log(closeValue / openValue)
            let variance = 0.5 * highLow * highLow - coefficient * closeOpen * closeOpen
            sum += max(variance, 0.0)
            used += 1
        }
        return used > 1 ? sqrt(sum / Double(used)) : 0.0
    }

    public static func kalmanEstimateAsSeries(_ values: [Double], startIndex: Int, period: Int) -> Double {
        guard period > 2, startIndex >= 0, startIndex < values.count else { return 0.0 }
        let oldest = min(values.count - 1, startIndex + period - 1)
        guard oldest - startIndex + 1 >= 3 else { return 0.0 }

        var returnVariance = 0.0
        var returnCount = 0
        if oldest > startIndex {
            for index in stride(from: oldest, through: startIndex + 1, by: -1) {
                let previous = fxSafeFinite(values[index])
                let current = fxSafeFinite(values[index - 1])
                guard previous > 0.0 else { continue }
                let ret = (current - previous) / previous
                returnVariance += ret * ret
                returnCount += 1
            }
        }
        if returnCount <= 0 {
            returnCount = 1
        }
        returnVariance /= Double(returnCount)
        if returnVariance < 1e-10 {
            returnVariance = 1e-10
        }

        let measurementVariance = returnVariance * 4.0
        let processVariance = returnVariance
        var estimate = fxSafeFinite(values[oldest])
        var covariance = 1.0
        if oldest > startIndex {
            for index in stride(from: oldest - 1, through: startIndex, by: -1) {
                covariance += processVariance
                let gain = covariance / (covariance + measurementVariance)
                estimate += gain * (fxSafeFinite(values[index]) - estimate)
                covariance = (1.0 - gain) * covariance
            }
        }
        return estimate
    }

    public static func ehlersSuperSmootherAsSeries(_ values: [Double], startIndex: Int, period: Int) -> Double {
        guard period > 2, startIndex >= 0, startIndex < values.count else { return 0.0 }
        var warmup = period * 3
        if warmup < 12 {
            warmup = 12
        }
        let oldest = min(values.count - 1, startIndex + warmup - 1)
        guard oldest - startIndex + 1 >= 3 else { return 0.0 }

        let a1 = exp(-1.41421356237 * Double.pi / Double(period))
        let b1 = 2.0 * a1 * cos(1.41421356237 * Double.pi / Double(period))
        let c2 = b1
        let c3 = -a1 * a1
        let c1 = 1.0 - c2 - c3

        var y2 = fxSafeFinite(values[oldest])
        var y1 = fxSafeFinite(values[oldest - 1])
        if oldest - 2 >= startIndex {
            for index in stride(from: oldest - 2, through: startIndex, by: -1) {
                let x0 = fxSafeFinite(values[index])
                let x1 = fxSafeFinite(values[index + 1])
                let y = c1 * 0.5 * (x0 + x1) + c2 * y1 + c3 * y2
                y2 = y1
                y1 = y
            }
        }
        return y1
    }

    private static func movePoints(priceNow: Double, priceFuture: Double, point: Double) -> Double {
        guard point > 0.0 else { return 0.0 }
        return (fxSafeFinite(priceFuture) - fxSafeFinite(priceNow)) / point
    }

    private static func sortedSmall(_ values: [Double]) -> [Double] {
        guard values.count > 1 else { return values.map { fxSafeFinite($0) } }
        var output = values.map { fxSafeFinite($0) }
        for index in 1..<output.count {
            let key = output[index]
            var cursor = index - 1
            while cursor >= 0, output[cursor] > key {
                output[cursor + 1] = output[cursor]
                cursor -= 1
            }
            output[cursor + 1] = key
        }
        return output
    }
}
