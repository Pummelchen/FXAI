import Foundation

public enum LinPAReference {
    public enum Mode: Sendable {
        case pa
        case paI(c: Double)
        case paII(c: Double)
    }

    public static func update(weights: inout [[Double]], features: [Double], label: Int, mode: Mode) -> (predicted: Int, loss: Double, tau: Double) {
        let scores = weights.map { dot($0, features) }
        guard scores.indices.contains(label), weights.count > 1 else {
            return (scores.indices.max { scores[$0] < scores[$1] } ?? 0, 0.0, 0.0)
        }
        let predicted = scores.indices.max { scores[$0] < scores[$1] } ?? 0
        let impostor = scores.indices.filter { $0 != label }.max { scores[$0] < scores[$1] } ?? predicted
        let loss = max(0.0, 1.0 - scores[label] + scores[impostor])
        let norm2 = max(dot(features, features), 1.0e-12)
        let tau: Double
        switch mode {
        case .pa:
            tau = loss / (2.0 * norm2)
        case let .paI(c):
            tau = min(c, loss / (2.0 * norm2))
        case let .paII(c):
            tau = loss / (2.0 * norm2 + 1.0 / max(2.0 * c, 1.0e-12))
        }
        if loss > 0.0, weights.indices.contains(label), weights.indices.contains(impostor) {
            for index in features.indices where index < weights[label].count && index < weights[impostor].count {
                weights[label][index] += tau * features[index]
                weights[impostor][index] -= tau * features[index]
            }
        }
        return (predicted, loss, tau)
    }

    private static func dot(_ lhs: [Double], _ rhs: [Double]) -> Double {
        zip(lhs, rhs).map(*).reduce(0.0, +)
    }
}
