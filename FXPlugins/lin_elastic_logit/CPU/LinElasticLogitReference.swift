import Foundation

public enum LinElasticLogitReference {
    public static func trainStep(
        weights: inout [Double],
        bias: inout Double,
        features: [Double],
        label: Double,
        learningRate: Double,
        l1: Double,
        l2: Double
    ) -> Double {
        let prediction = sigmoid(dot(weights, features) + bias)
        let error = prediction - label
        for index in weights.indices {
            let feature = index < features.count ? features[index] : 0.0
            let gradientStep = weights[index] - learningRate * (error * feature + l2 * weights[index])
            weights[index] = softThreshold(gradientStep, threshold: learningRate * l1)
        }
        bias -= learningRate * error
        return prediction
    }

    private static func softThreshold(_ value: Double, threshold: Double) -> Double {
        if value > threshold { return value - threshold }
        if value < -threshold { return value + threshold }
        return 0.0
    }

    private static func sigmoid(_ value: Double) -> Double {
        1.0 / (1.0 + exp(-max(min(value, 40.0), -40.0)))
    }

    private static func dot(_ lhs: [Double], _ rhs: [Double]) -> Double {
        zip(lhs, rhs).map(*).reduce(0.0, +)
    }
}
