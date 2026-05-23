import Foundation

public enum TreeXGBReference {
    public struct MulticlassGradient: Equatable, Sendable {
        public let probabilities: [Double]
        public let gradients: [Double]
        public let hessians: [Double]
    }

    public static func multiclassGradient(logits: [Double], label: Int) -> MulticlassGradient {
        let probabilities = softmax(logits)
        let gradients = probabilities.indices.map { probabilities[$0] - ($0 == label ? 1.0 : 0.0) }
        let hessians = probabilities.map { max($0 * (1.0 - $0), 1.0e-6) }
        return MulticlassGradient(probabilities: probabilities, gradients: gradients, hessians: hessians)
    }

    public static func leafWeights(gradients: [Double], hessians: [Double], lambda: Double, learningRate: Double) -> [Double] {
        zip(gradients, hessians).map { -learningRate * $0 / max($1 + lambda, 1.0e-12) }
    }

    public static func route(value: Double?, threshold: Double, missingGoesLeft: Bool) -> Bool {
        guard let value, value.isFinite else { return missingGoesLeft }
        return value <= threshold
    }

    private static func softmax(_ logits: [Double]) -> [Double] {
        guard let maximum = logits.max() else { return [] }
        let exps = logits.map { exp($0 - maximum) }
        let total = max(exps.reduce(0.0, +), 1.0e-12)
        return exps.map { $0 / total }
    }
}
