import Foundation

public enum FactorPPPValueReference {
    public struct Input: Equatable, Sendable {
        public let symbol: String
        public let spot: Double
        public let pppFairValue: Double
        public let observationAgeDays: Double
        public let halfLifeDays: Double

        public init(symbol: String, spot: Double, pppFairValue: Double, observationAgeDays: Double = 0.0, halfLifeDays: Double = 365.0) {
            self.symbol = symbol
            self.spot = spot
            self.pppFairValue = pppFairValue
            self.observationAgeDays = observationAgeDays
            self.halfLifeDays = halfLifeDays
        }
    }

    public struct Score: Equatable, Sendable {
        public let symbol: String
        public let logMisvaluation: Double
        public let zScore: Double
        public let staleDecay: Double
        public let valueScore: Double
    }

    public static func scores(_ inputs: [Input]) -> [Score] {
        let misvaluations = inputs.map { input -> Double in
            guard input.spot > 0.0, input.pppFairValue > 0.0 else { return 0.0 }
            return log(input.spot / input.pppFairValue)
        }
        let z = zScores(misvaluations)
        return inputs.indices.map { index in
            let halfLife = max(inputs[index].halfLifeDays, 1.0)
            let decay = exp(-log(2.0) * max(inputs[index].observationAgeDays, 0.0) / halfLife)
            return Score(
                symbol: inputs[index].symbol,
                logMisvaluation: misvaluations[index],
                zScore: z[index],
                staleDecay: decay,
                valueScore: -z[index] * decay
            )
        }
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
