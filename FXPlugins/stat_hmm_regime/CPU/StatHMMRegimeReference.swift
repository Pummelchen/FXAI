import Foundation

public enum StatHMMRegimeReference {
    public struct Fit: Equatable, Sendable {
        public let initial: [Double]
        public let transition: [[Double]]
        public let means: [Double]
        public let variances: [Double]
        public let posteriors: [[Double]]
        public let logLikelihoods: [Double]
    }

    public static func baumWelch(observations: [Double], regimes: Int, iterations: Int) -> Fit {
        guard regimes > 0, !observations.isEmpty else {
            return Fit(initial: [], transition: [], means: [], variances: [], posteriors: [], logLikelihoods: [])
        }
        var means = seededMeans(observations: observations, regimes: regimes)
        var variances = Array(repeating: max(variance(observations), 1.0e-6), count: regimes)
        var initial = Array(repeating: 1.0 / Double(regimes), count: regimes)
        var transition = Array(repeating: Array(repeating: 1.0 / Double(regimes), count: regimes), count: regimes)
        for i in 0..<regimes {
            for j in 0..<regimes {
                transition[i][j] = i == j ? 0.86 : 0.14 / Double(max(regimes - 1, 1))
            }
        }

        var posteriors = Array(repeating: Array(repeating: 1.0 / Double(regimes), count: regimes), count: observations.count)
        var logLikelihoods: [Double] = []
        for _ in 0..<max(iterations, 1) {
            let emissions = observations.map { observation in
                (0..<regimes).map { gaussianLogPDF(observation, mean: means[$0], variance: variances[$0]) }
            }
            let forward = forwardLog(initial: initial, transition: transition, emissions: emissions)
            let backward = backwardLog(transition: transition, emissions: emissions)
            let logLikelihood = logSumExp(forward.last ?? [])
            logLikelihoods.append(logLikelihood)
            for t in observations.indices {
                let row = (0..<regimes).map { exp(forward[t][$0] + backward[t][$0] - logLikelihood) }
                posteriors[t] = normalize(row)
            }
            initial = posteriors[0]
            var xiSum = Array(repeating: Array(repeating: 0.0, count: regimes), count: regimes)
            if observations.count > 1 {
                for t in 0..<(observations.count - 1) {
                    var normalizer = -Double.infinity
                    var values = Array(repeating: Array(repeating: 0.0, count: regimes), count: regimes)
                    for i in 0..<regimes {
                        for j in 0..<regimes {
                            let value = forward[t][i] + log(max(transition[i][j], 1.0e-300)) + emissions[t + 1][j] + backward[t + 1][j]
                            values[i][j] = value
                            normalizer = logAdd(normalizer, value)
                        }
                    }
                    for i in 0..<regimes {
                        for j in 0..<regimes {
                            xiSum[i][j] += exp(values[i][j] - normalizer)
                        }
                    }
                }
                for i in 0..<regimes {
                    transition[i] = normalize(xiSum[i])
                }
            }
            for regime in 0..<regimes {
                let weight = posteriors.map { $0[regime] }.reduce(0.0, +)
                if weight > 1.0e-9 {
                    means[regime] = zip(observations, posteriors).map { $0 * $1[regime] }.reduce(0.0, +) / weight
                    variances[regime] = max(zip(observations, posteriors).map { obs, gamma in gamma[regime] * (obs - means[regime]) * (obs - means[regime]) }.reduce(0.0, +) / weight, 1.0e-6)
                }
            }
        }
        return Fit(initial: initial, transition: transition, means: means, variances: variances, posteriors: posteriors, logLikelihoods: logLikelihoods)
    }

    public static func viterbi(observations: [Double], fit: Fit) -> [Int] {
        let regimes = fit.means.count
        guard regimes > 0, !observations.isEmpty else { return [] }
        var scores = Array(repeating: Array(repeating: -Double.infinity, count: regimes), count: observations.count)
        var back = Array(repeating: Array(repeating: 0, count: regimes), count: observations.count)
        for r in 0..<regimes {
            scores[0][r] = log(max(fit.initial[r], 1.0e-300)) + gaussianLogPDF(observations[0], mean: fit.means[r], variance: fit.variances[r])
        }
        if observations.count > 1 {
            for t in 1..<observations.count {
                for r in 0..<regimes {
                    var bestScore = -Double.infinity
                    var bestState = 0
                    for previous in 0..<regimes {
                        let score = scores[t - 1][previous] + log(max(fit.transition[previous][r], 1.0e-300))
                        if score > bestScore {
                            bestScore = score
                            bestState = previous
                        }
                    }
                    scores[t][r] = bestScore + gaussianLogPDF(observations[t], mean: fit.means[r], variance: fit.variances[r])
                    back[t][r] = bestState
                }
            }
        }
        var state = scores.last?.indices.max(by: { scores.last![$0] < scores.last![$1] }) ?? 0
        var path = Array(repeating: 0, count: observations.count)
        for t in stride(from: observations.count - 1, through: 0, by: -1) {
            path[t] = state
            state = back[t][state]
        }
        return path
    }

    private static func forwardLog(initial: [Double], transition: [[Double]], emissions: [[Double]]) -> [[Double]] {
        let regimes = initial.count
        var alpha = Array(repeating: Array(repeating: -Double.infinity, count: regimes), count: emissions.count)
        for r in 0..<regimes {
            alpha[0][r] = log(max(initial[r], 1.0e-300)) + emissions[0][r]
        }
        if emissions.count > 1 {
            for t in 1..<emissions.count {
                for r in 0..<regimes {
                    alpha[t][r] = logSumExp((0..<regimes).map { alpha[t - 1][$0] + log(max(transition[$0][r], 1.0e-300)) }) + emissions[t][r]
                }
            }
        }
        return alpha
    }

    private static func backwardLog(transition: [[Double]], emissions: [[Double]]) -> [[Double]] {
        let regimes = transition.count
        var beta = Array(repeating: Array(repeating: 0.0, count: regimes), count: emissions.count)
        guard emissions.count > 1 else { return beta }
        for t in stride(from: emissions.count - 2, through: 0, by: -1) {
            for r in 0..<regimes {
                beta[t][r] = logSumExp((0..<regimes).map { log(max(transition[r][$0], 1.0e-300)) + emissions[t + 1][$0] + beta[t + 1][$0] })
            }
        }
        return beta
    }

    private static func seededMeans(observations: [Double], regimes: Int) -> [Double] {
        let sorted = observations.sorted()
        return (0..<regimes).map { index in
            let position = Int(round(Double(index) * Double(max(sorted.count - 1, 0)) / Double(max(regimes - 1, 1))))
            return sorted[min(max(position, 0), sorted.count - 1)]
        }
    }

    private static func gaussianLogPDF(_ value: Double, mean: Double, variance: Double) -> Double {
        let v = max(variance, 1.0e-9)
        let delta = value - mean
        return -0.5 * (log(2.0 * Double.pi * v) + delta * delta / v)
    }

    private static func normalize(_ values: [Double]) -> [Double] {
        let total = values.reduce(0.0, +)
        guard total.isFinite, total > 0 else {
            return Array(repeating: 1.0 / Double(max(values.count, 1)), count: values.count)
        }
        return values.map { $0 / total }
    }

    private static func logSumExp(_ values: [Double]) -> Double {
        guard let maximum = values.max(), maximum.isFinite else { return -Double.infinity }
        return maximum + log(values.map { exp($0 - maximum) }.reduce(0.0, +))
    }

    private static func logAdd(_ lhs: Double, _ rhs: Double) -> Double {
        if lhs == -Double.infinity { return rhs }
        if rhs == -Double.infinity { return lhs }
        let maximum = max(lhs, rhs)
        return maximum + log(exp(lhs - maximum) + exp(rhs - maximum))
    }

    private static func variance(_ values: [Double]) -> Double {
        guard values.count > 1 else { return 1.0e-6 }
        let mean = values.reduce(0.0, +) / Double(values.count)
        return values.map { ($0 - mean) * ($0 - mean) }.reduce(0.0, +) / Double(values.count - 1)
    }
}
