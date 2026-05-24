import Foundation

public enum MixMoeConformalReference {
    public struct PredictionSet: Equatable, Sendable {
        public let classes: Set<Int>
        public let cutoff: Double
    }

    /// Computes the split-conformal finite-sample quantile cutoff.
    ///
    /// Scores are nonconformity values such as 1 - p(true class). The rank uses
    /// ceil((n + 1) * (1 - alpha)), which is the standard conservative split
    /// conformal classification rule.
    public static func splitConformalCutoff(scores: [Double], alpha: Double) -> Double {
        let clean = scores.filter(\.isFinite).sorted()
        guard !clean.isEmpty else { return 1.0 }
        let confidence = min(max(1.0 - alpha, 0.0), 1.0)
        let rank = Int(ceil(Double(clean.count + 1) * confidence)) - 1
        guard rank < clean.count else { return 1.0 }
        return clean[max(rank, 0)]
    }

    /// Builds a conformal prediction set from class probabilities.
    ///
    /// Classes whose nonconformity is below the cutoff are retained; if the set
    /// would be empty, the argmax class is kept so downstream trading code always
    /// receives at least one admissible class.
    public static func predictionSet(probabilities: [Double], cutoff: Double) -> PredictionSet {
        let normalized = normalize(probabilities)
        var classes = Set(normalized.indices.filter { 1.0 - normalized[$0] <= cutoff })
        if classes.isEmpty, let best = normalized.indices.max(by: { normalized[$0] < normalized[$1] }) {
            classes.insert(best)
        }
        return PredictionSet(classes: classes, cutoff: cutoff)
    }

    /// Measures empirical label coverage for a fixed conformal cutoff.
    ///
    /// This is a model-family fixture: properly calibrated MoE-conformal routes
    /// should keep realized coverage close to the requested confidence level.
    public static func empiricalCoverage(probabilities: [[Double]], labels: [Int], cutoff: Double) -> Double {
        guard !probabilities.isEmpty else { return 0.0 }
        var hits = 0
        for index in probabilities.indices {
            let set = predictionSet(probabilities: probabilities[index], cutoff: cutoff)
            if index < labels.count, set.classes.contains(labels[index]) {
                hits += 1
            }
        }
        return Double(hits) / Double(probabilities.count)
    }

    /// Computes load-balanced expert router probabilities.
    ///
    /// The balancing term discourages overused experts while preserving the
    /// temperature-scaled MoE softmax used by the accelerator runtime.
    public static func routerGates(
        regime: [Double],
        expertWeights: [[Double]],
        usageEMA: [Double],
        temperature: Double = 1.0,
        loadBalanceStrength: Double = 0.35
    ) -> [Double] {
        guard !expertWeights.isEmpty else { return [] }
        let targetUsage = 1.0 / Double(expertWeights.count)
        let logits = expertWeights.indices.map { expert in
            dot(regime, expertWeights[expert]) / max(temperature, 1.0e-6) -
                loadBalanceStrength * ((expert < usageEMA.count ? usageEMA[expert] : targetUsage) - targetUsage)
        }
        return softmax(logits)
    }

    private static func normalize(_ values: [Double]) -> [Double] {
        let clipped = values.map { min(max($0, 1.0e-9), 1.0) }
        let total = max(clipped.reduce(0.0, +), 1.0e-12)
        return clipped.map { $0 / total }
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
