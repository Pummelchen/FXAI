import Foundation

public enum StatMSGARCHReference {
    public struct FilterResult: Equatable, Sendable {
        public let probabilities: [[Double]]
        public let variances: [[Double]]
        public let logLikelihood: Double
    }

    public static func filter(
        returns: [Double],
        transition: [[Double]],
        omega: [Double],
        alpha: [Double],
        beta: [Double]
    ) -> FilterResult {
        let regimes = min(omega.count, alpha.count, beta.count, transition.count)
        guard regimes > 0, !returns.isEmpty else {
            return FilterResult(probabilities: [], variances: [], logLikelihood: 0.0)
        }
        var probabilities = Array(repeating: 1.0 / Double(regimes), count: regimes)
        var previousVariances = omega.map { max($0 / max(1.0 - 0.05 - 0.85, 0.05), 1.0e-6) }
        var previousReturn = returns[0]
        var allProbabilities: [[Double]] = []
        var allVariances: [[Double]] = []
        var logLikelihood = 0.0
        for value in returns {
            var predicted = Array(repeating: 0.0, count: regimes)
            for target in 0..<regimes {
                for source in 0..<regimes {
                    predicted[target] += probabilities[source] * transition[source][target]
                }
            }
            var likelihoods = Array(repeating: 0.0, count: regimes)
            var variances = Array(repeating: 0.0, count: regimes)
            for regime in 0..<regimes {
                let variance = max(omega[regime] + alpha[regime] * previousReturn * previousReturn + beta[regime] * previousVariances[regime], 1.0e-9)
                variances[regime] = variance
                likelihoods[regime] = predicted[regime] * gaussianPDF(value, variance: variance)
            }
            let evidence = max(likelihoods.reduce(0.0, +), 1.0e-300)
            probabilities = likelihoods.map { $0 / evidence }
            logLikelihood += log(evidence)
            allProbabilities.append(probabilities)
            allVariances.append(variances)
            previousReturn = value
            previousVariances = variances
        }
        return FilterResult(probabilities: allProbabilities, variances: allVariances, logLikelihood: logLikelihood)
    }

    private static func gaussianPDF(_ value: Double, variance: Double) -> Double {
        let v = max(variance, 1.0e-9)
        return exp(-0.5 * value * value / v) / sqrt(2.0 * Double.pi * v)
    }
}
