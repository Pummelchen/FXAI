import Foundation

public struct LinFTRLReferenceState: Equatable, Sendable {
    public var z: [Double]
    public var n: [Double]
    public var alpha: Double
    public var beta: Double
    public var l1: Double
    public var l2: Double

    public init(featureCount: Int, alpha: Double, beta: Double, l1: Double, l2: Double) {
        self.z = Array(repeating: 0.0, count: featureCount)
        self.n = Array(repeating: 0.0, count: featureCount)
        self.alpha = alpha
        self.beta = beta
        self.l1 = l1
        self.l2 = l2
    }
}

public enum LinFTRLReference {
    public static func weights(state: LinFTRLReferenceState) -> [Double] {
        state.z.indices.map { index in
            let z = state.z[index]
            guard abs(z) > state.l1 else { return 0.0 }
            let sign = z < 0.0 ? -1.0 : 1.0
            return -(z - sign * state.l1) / ((state.beta + sqrt(state.n[index])) / state.alpha + state.l2)
        }
    }

    public static func predict(features: [Double], state: LinFTRLReferenceState) -> Double {
        sigmoid(dot(weights(state: state), features))
    }

    public static func update(state: inout LinFTRLReferenceState, features: [Double], label: Double) -> Double {
        let currentWeights = weights(state: state)
        let prediction = sigmoid(dot(currentWeights, features))
        let gradientScale = prediction - label
        for index in state.z.indices {
            let feature = index < features.count ? features[index] : 0.0
            let gradient = gradientScale * feature
            let sigma = (sqrt(state.n[index] + gradient * gradient) - sqrt(state.n[index])) / state.alpha
            state.z[index] += gradient - sigma * currentWeights[index]
            state.n[index] += gradient * gradient
        }
        return prediction
    }

    private static func sigmoid(_ value: Double) -> Double {
        1.0 / (1.0 + exp(-max(min(value, 40.0), -40.0)))
    }

    private static func dot(_ lhs: [Double], _ rhs: [Double]) -> Double {
        zip(lhs, rhs).map(*).reduce(0.0, +)
    }
}
