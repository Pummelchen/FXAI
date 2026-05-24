import Foundation

public enum MixLoffmReference {
    public struct Expert: Equatable, Sendable {
        public let bias: Double
        public let linearWeights: [Double]
        public let factors: [[Double]]
        public let reliability: Double
        public let usageEMA: Double

        public init(
            bias: Double,
            linearWeights: [Double],
            factors: [[Double]],
            reliability: Double = 0.5,
            usageEMA: Double = 0.25
        ) {
            self.bias = bias
            self.linearWeights = linearWeights
            self.factors = factors
            self.reliability = reliability
            self.usageEMA = usageEMA
        }
    }

    /// Computes the standard second-order factorization-machine score.
    ///
    /// The low-rank term is 0.5 * ((sum(v_i*x_i))^2 - sum(v_i^2*x_i^2)), which
    /// captures all pair interactions without materializing the full matrix.
    public static func factorizationScore(features: [Double], expert: Expert) -> Double {
        var score = expert.bias + dot(features, expert.linearWeights)
        let rank = expert.factors.map(\.count).max() ?? 0
        guard rank > 0 else { return score }
        for factor in 0..<rank {
            var summed = 0.0
            var squared = 0.0
            for feature in features.indices {
                let value = features[feature]
                let loading = feature < expert.factors.count && factor < expert.factors[feature].count ? expert.factors[feature][factor] : 0.0
                summed += loading * value
                squared += loading * loading * value * value
            }
            score += 0.5 * (summed * summed - squared)
        }
        return score
    }

    /// Routes a sample over latent online factor experts.
    ///
    /// Reliability lifts experts with recent edge while the usage penalty keeps
    /// the mixture from collapsing into one overused expert.
    public static func gateProbabilities(
        features: [Double],
        experts: [Expert],
        loadBalanceStrength: Double = 0.35
    ) -> [Double] {
        guard !experts.isEmpty else { return [] }
        let targetUsage = 1.0 / Double(experts.count)
        let logits = experts.map { expert in
            factorizationScore(features: features, expert: expert) +
                log(max(expert.reliability, 1.0e-6)) -
                loadBalanceStrength * (expert.usageEMA - targetUsage)
        }
        return softmax(logits)
    }

    /// Returns a normalized interaction-energy diagnostic for a feature vector.
    ///
    /// The value is useful as a golden fixture because it increases when the
    /// low-rank expert captures cross-feature structure instead of only linears.
    public static func interactionEnergy(features: [Double], expert: Expert) -> Double {
        let score = factorizationScore(features: features, expert: expert)
        let linear = expert.bias + dot(features, expert.linearWeights)
        return abs(score - linear) / max(1.0, features.map(abs).reduce(0.0, +))
    }

    private static func dot(_ lhs: [Double], _ rhs: [Double]) -> Double {
        zip(lhs, rhs).map(*).reduce(0.0, +)
    }

    private static func softmax(_ values: [Double]) -> [Double] {
        guard let maxValue = values.max() else { return [] }
        let exps = values.map { exp(min(max($0 - maxValue, -60.0), 60.0)) }
        let total = max(exps.reduce(0.0, +), 1.0e-12)
        return exps.map { $0 / total }
    }
}
