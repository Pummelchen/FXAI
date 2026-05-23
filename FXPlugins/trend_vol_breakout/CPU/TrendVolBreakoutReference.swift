import Foundation

public enum TrendVolBreakoutReference {
    public struct Bar: Equatable, Sendable {
        public let high: Double
        public let low: Double
        public let close: Double

        public init(high: Double, low: Double, close: Double) {
            self.high = high
            self.low = low
            self.close = close
        }
    }

    public struct Signal: Equatable, Sendable {
        public let upperBand: Double
        public let lowerBand: Double
        public let atr: Double
        public let direction: Int
        public let stop: Double
        public let target: Double
    }

    public static func signal(
        bars: [Bar],
        lookback: Int,
        atrPeriod: Int,
        bandMultiplier: Double,
        targetMultiplier: Double = 2.0
    ) -> Signal {
        guard let last = bars.last, bars.count >= 2 else {
            return Signal(upperBand: 0.0, lowerBand: 0.0, atr: 0.0, direction: 0, stop: 0.0, target: 0.0)
        }
        let history = Array(bars.dropLast())
        let lb = min(max(lookback, 1), history.count)
        let rangeBars = history.suffix(lb)
        let highs = rangeBars.map(\.high)
        let lows = rangeBars.map(\.low)
        let atr = averageTrueRange(bars: bars, period: atrPeriod)
        let upper = (highs.max() ?? last.close) + bandMultiplier * atr
        let lower = (lows.min() ?? last.close) - bandMultiplier * atr
        let direction = last.close > upper ? 1 : (last.close < lower ? -1 : 0)
        let stop = direction > 0 ? last.close - atr : (direction < 0 ? last.close + atr : last.close)
        let target = direction > 0 ? last.close + targetMultiplier * atr : (direction < 0 ? last.close - targetMultiplier * atr : last.close)
        return Signal(upperBand: upper, lowerBand: lower, atr: atr, direction: direction, stop: stop, target: target)
    }

    public static func averageTrueRange(bars: [Bar], period: Int) -> Double {
        guard bars.count > 1 else { return 0.0 }
        var trueRanges: [Double] = []
        for index in 1..<bars.count {
            let current = bars[index]
            let previousClose = bars[index - 1].close
            trueRanges.append(max(current.high - current.low, abs(current.high - previousClose), abs(current.low - previousClose)))
        }
        let slice = trueRanges.suffix(max(min(period, trueRanges.count), 1))
        return slice.reduce(0.0, +) / Double(slice.count)
    }
}
