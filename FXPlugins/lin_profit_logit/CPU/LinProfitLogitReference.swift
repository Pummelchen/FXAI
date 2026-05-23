import Foundation

public enum LinProfitLogitReference {
    public struct LossGradient: Equatable, Sendable {
        public let prediction: Double
        public let loss: Double
        public let gradient: [Double]
        public let biasGradient: Double
        public let effectiveWeight: Double
    }

    public static func lossGradient(
        weights: [Double],
        bias: Double,
        features: [Double],
        label: Double,
        movePoints: Double,
        costPoints: Double,
        buyAsymmetry: Double = 1.0,
        sellAsymmetry: Double = 1.0
    ) -> LossGradient {
        let prediction = sigmoid(dot(weights, features) + bias)
        let economicEdge = max(abs(movePoints) - max(costPoints, 0.0), 0.0)
        let asymmetry = label >= 0.5 ? buyAsymmetry : sellAsymmetry
        let effectiveWeight = max(1.0 + economicEdge * asymmetry, 1.0e-6)
        let clipped = min(max(prediction, 1.0e-12), 1.0 - 1.0e-12)
        let loss = -effectiveWeight * (label * log(clipped) + (1.0 - label) * log(1.0 - clipped))
        let biasGradient = effectiveWeight * (prediction - label)
        return LossGradient(
            prediction: prediction,
            loss: loss,
            gradient: features.map { biasGradient * $0 },
            biasGradient: biasGradient,
            effectiveWeight: effectiveWeight
        )
    }

    private static func sigmoid(_ value: Double) -> Double {
        1.0 / (1.0 + exp(-max(min(value, 40.0), -40.0)))
    }

    private static func dot(_ lhs: [Double], _ rhs: [Double]) -> Double {
        zip(lhs, rhs).map(*).reduce(0.0, +)
    }
}
