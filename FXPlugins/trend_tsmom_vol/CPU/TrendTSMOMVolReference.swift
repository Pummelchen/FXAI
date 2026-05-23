import Foundation

public enum TrendTSMOMVolReference {
    public struct Series: Equatable, Sendable {
        public let symbol: String
        public let closes: [Double]
        public let volumes: [Double]

        public init(symbol: String, closes: [Double], volumes: [Double] = []) {
            self.symbol = symbol
            self.closes = closes
            self.volumes = volumes
        }
    }

    public struct Signal: Equatable, Sendable {
        public let symbol: String
        public let momentum: Double
        public let realizedVolatility: Double
        public let targetWeight: Double
        public let liquidityConfidence: Double
    }

    public static func signals(
        series: [Series],
        lookback: Int,
        annualization: Double = sqrt(252.0),
        volatilityTarget: Double,
        maxLeverage: Double,
        dataHasVolume: Bool
    ) -> [Signal] {
        series.map { item in
            let returns = logReturns(item.closes)
            let lb = min(max(lookback, 1), max(returns.count, 1))
            let recentReturns = Array(returns.suffix(lb))
            let momentum = recentReturns.reduce(0.0, +)
            let vol = standardDeviation(recentReturns) * annualization
            let rawWeight = vol > 1.0e-12 ? (momentum >= 0.0 ? 1.0 : -1.0) * volatilityTarget / vol : 0.0
            let liquidity = liquidityConfidence(volumes: item.volumes, dataHasVolume: dataHasVolume)
            let adjustedWeight = min(max(rawWeight, -maxLeverage), maxLeverage) * liquidity
            return Signal(
                symbol: item.symbol,
                momentum: momentum,
                realizedVolatility: vol,
                targetWeight: min(max(adjustedWeight, -maxLeverage), maxLeverage),
                liquidityConfidence: liquidity
            )
        }
    }

    private static func logReturns(_ closes: [Double]) -> [Double] {
        guard closes.count > 1 else { return [] }
        return zip(closes.dropFirst(), closes.dropLast()).map { current, previous in
            guard current > 0.0, previous > 0.0 else { return 0.0 }
            return log(current / previous)
        }
    }

    private static func liquidityConfidence(volumes: [Double], dataHasVolume: Bool) -> Double {
        guard dataHasVolume, let last = volumes.last, last > 0.0 else { return 1.0 }
        let positive = volumes.filter { $0 > 0.0 }
        let ratio = last / max(mean(positive), 1.0e-12)
        return min(max(0.50 + 0.50 * ratio, 0.25), 1.50)
    }

    private static func standardDeviation(_ values: [Double]) -> Double {
        guard values.count > 1 else { return 0.0 }
        let m = mean(values)
        return sqrt(values.map { pow($0 - m, 2.0) }.reduce(0.0, +) / Double(values.count - 1))
    }

    private static func mean(_ values: [Double]) -> Double {
        values.isEmpty ? 0.0 : values.reduce(0.0, +) / Double(values.count)
    }
}
