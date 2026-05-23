import Foundation

public enum LinSGDReference {
    public static func probabilities(weights: [[Double]], features: [Double]) -> [Double] {
        softmax(weights.map { dot($0, features) })
    }

    public static func trainStep(weights: inout [[Double]], features: [Double], label: Int, learningRate: Double, l2: Double = 0.0) -> [Double] {
        let probabilities = probabilities(weights: weights, features: features)
        for classIndex in weights.indices {
            let target = classIndex == label ? 1.0 : 0.0
            for featureIndex in weights[classIndex].indices {
                let feature = featureIndex < features.count ? features[featureIndex] : 0.0
                let gradient = (target - probabilities[classIndex]) * feature - l2 * weights[classIndex][featureIndex]
                weights[classIndex][featureIndex] += learningRate * gradient
            }
        }
        return probabilities
    }

    private static func softmax(_ logits: [Double]) -> [Double] {
        guard let maximum = logits.max() else { return [] }
        let exps = logits.map { exp($0 - maximum) }
        let total = max(exps.reduce(0.0, +), 1.0e-12)
        return exps.map { $0 / total }
    }

    private static func dot(_ lhs: [Double], _ rhs: [Double]) -> Double {
        zip(lhs, rhs).map(*).reduce(0.0, +)
    }
}
