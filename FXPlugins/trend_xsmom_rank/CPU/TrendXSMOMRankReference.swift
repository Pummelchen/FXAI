import Foundation

public enum TrendXSMOMRankReference {
    public struct Input: Equatable, Sendable {
        public let symbol: String
        public let returns: [Double]
        public let neutralizer: Double?

        public init(symbol: String, returns: [Double], neutralizer: Double? = nil) {
            self.symbol = symbol
            self.returns = returns
            self.neutralizer = neutralizer
        }
    }

    public struct Rank: Equatable, Sendable {
        public let symbol: String
        public let momentum: Double
        public let neutralizedMomentum: Double
        public let percentile: Double
        public let portfolioWeight: Double
    }

    public static func ranks(inputs: [Input], longShortGross: Double = 1.0) -> [Rank] {
        guard !inputs.isEmpty else { return [] }
        let rawMomentum = inputs.map { $0.returns.filter(\.isFinite).reduce(0.0, +) }
        let neutralized = neutralize(rawMomentum, by: inputs.map(\.neutralizer))
        let percentiles = percentileRanks(neutralized)
        let centered = percentiles.map { $0 - 0.5 }
        let normalizer = max(centered.map(abs).reduce(0.0, +), 1.0e-12)
        return inputs.indices.map { index in
            Rank(
                symbol: inputs[index].symbol,
                momentum: rawMomentum[index],
                neutralizedMomentum: neutralized[index],
                percentile: percentiles[index],
                portfolioWeight: longShortGross * centered[index] / normalizer
            )
        }
        .sorted { $0.percentile > $1.percentile }
    }

    private static func neutralize(_ values: [Double], by neutralizers: [Double?]) -> [Double] {
        let usable = neutralizers.compactMap { $0 }
        guard usable.count == values.count, values.count > 2 else { return values }
        let x = usable
        let beta = covariance(values, x) / max(variance(x), 1.0e-12)
        let alpha = mean(values) - beta * mean(x)
        return zip(values, x).map { $0 - alpha - beta * $1 }
    }

    private static func percentileRanks(_ values: [Double]) -> [Double] {
        guard values.count > 1 else { return Array(repeating: 0.5, count: values.count) }
        let sorted = values.enumerated().sorted { lhs, rhs in
            lhs.element == rhs.element ? lhs.offset < rhs.offset : lhs.element < rhs.element
        }
        var ranks = Array(repeating: 0.0, count: values.count)
        var index = 0
        while index < sorted.count {
            var end = index
            while end + 1 < sorted.count, abs(sorted[end + 1].element - sorted[index].element) < 1.0e-12 {
                end += 1
            }
            let averageRank = 0.5 * (Double(index) + Double(end))
            for cursor in index...end {
                ranks[sorted[cursor].offset] = averageRank / Double(values.count - 1)
            }
            index = end + 1
        }
        return ranks
    }

    private static func covariance(_ lhs: [Double], _ rhs: [Double]) -> Double {
        let ml = mean(lhs)
        let mr = mean(rhs)
        return zip(lhs, rhs).map { ($0 - ml) * ($1 - mr) }.reduce(0.0, +) / Double(max(lhs.count - 1, 1))
    }

    private static func variance(_ values: [Double]) -> Double {
        let m = mean(values)
        return values.map { pow($0 - m, 2.0) }.reduce(0.0, +) / Double(max(values.count - 1, 1))
    }

    private static func mean(_ values: [Double]) -> Double {
        values.isEmpty ? 0.0 : values.reduce(0.0, +) / Double(values.count)
    }
}
