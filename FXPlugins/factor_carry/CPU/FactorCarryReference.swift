import Foundation

public enum FactorCarryReference {
    public struct Input: Equatable, Sendable {
        public let symbol: String
        public let baseRate: Double
        public let quoteRate: Double
        public let forwardPoints: Double
        public let spot: Double
        public let volume: Double

        public init(symbol: String, baseRate: Double, quoteRate: Double, forwardPoints: Double = 0.0, spot: Double = 1.0, volume: Double = 0.0) {
            self.symbol = symbol
            self.baseRate = baseRate
            self.quoteRate = quoteRate
            self.forwardPoints = forwardPoints
            self.spot = spot
            self.volume = volume
        }
    }

    public struct Score: Equatable, Sendable {
        public let symbol: String
        public let rawCarry: Double
        public let normalizedCarry: Double
        public let liquidityWeight: Double
        public let finalScore: Double
    }

    public static func rank(_ inputs: [Input], dataHasVolume: Bool) -> [Score] {
        let raw = inputs.map { input -> Double in
            let forwardYield = input.spot > 0.0 ? input.forwardPoints / input.spot : 0.0
            return input.baseRate - input.quoteRate + forwardYield
        }
        let normalized = zScores(raw)
        return zip(inputs, normalized).map { input, z in
            let liquidity = dataHasVolume && input.volume > 0.0 ? sqrt(max(input.volume, 0.0)) : 1.0
            let boundedLiquidity = min(max(liquidity, 0.25), 4.0)
            return Score(
                symbol: input.symbol,
                rawCarry: input.baseRate - input.quoteRate + (input.spot > 0.0 ? input.forwardPoints / input.spot : 0.0),
                normalizedCarry: z,
                liquidityWeight: boundedLiquidity,
                finalScore: z * boundedLiquidity
            )
        }
        .sorted { $0.finalScore > $1.finalScore }
    }

    private static func zScores(_ values: [Double]) -> [Double] {
        guard values.count > 1 else { return Array(repeating: 0.0, count: values.count) }
        let m = mean(values)
        let sigma = sqrt(max(values.map { pow($0 - m, 2.0) }.reduce(0.0, +) / Double(values.count - 1), 1.0e-12))
        return values.map { ($0 - m) / sigma }
    }

    private static func mean(_ values: [Double]) -> Double {
        values.isEmpty ? 0.0 : values.reduce(0.0, +) / Double(values.count)
    }
}
