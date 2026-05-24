import Foundation

public enum FactorPCAPanelReference {
    public struct Result: Equatable, Sendable {
        public let covariance: [[Double]]
        public let loadings: [Double]
        public let scores: [Double]
        public let eigenvalue: Double
        public let explainedVarianceRatio: Double
    }

    public struct ComponentSet: Equatable, Sendable {
        public let covariance: [[Double]]
        public let loadings: [[Double]]
        public let scores: [[Double]]
        public let eigenvalues: [Double]
        public let explainedVarianceRatios: [Double]
    }

    /// Returns the first principal component using covariance power iteration.
    ///
    /// This is the single-factor entry point used by the existing panel factor
    /// plugin while the multi-component API below provides the full reference
    /// decomposition needed for parity checks.
    public static func firstComponent(panel: [[Double]], iterations: Int = 64) -> Result {
        let set = components(panel: panel, componentCount: 1, iterations: iterations)
        guard let loading = set.loadings.first, let scores = set.scores.first, let eigenvalue = set.eigenvalues.first else {
            return Result(covariance: [], loadings: [], scores: [], eigenvalue: 0.0, explainedVarianceRatio: 0.0)
        }
        return Result(
            covariance: set.covariance,
            loadings: loading,
            scores: scores,
            eigenvalue: eigenvalue,
            explainedVarianceRatio: set.explainedVarianceRatios.first ?? 0.0
        )
    }

    /// Computes multiple orthonormal principal components by deflation.
    ///
    /// Each component is estimated from the current residual covariance, then
    /// Gram-Schmidt corrected against previous loadings and deflated by its
    /// eigenvalue. The result exposes loadings, sample scores, eigenvalues, and
    /// explained-variance ratios for golden reference tests.
    public static func components(panel: [[Double]], componentCount: Int, iterations: Int = 96) -> ComponentSet {
        guard let width = panel.first?.count, panel.count > 1, width > 0 else {
            return ComponentSet(covariance: [], loadings: [], scores: [], eigenvalues: [], explainedVarianceRatios: [])
        }
        let centered = centerColumns(panel)
        let covariance = covarianceMatrix(centered)
        let totalVariance = covariance.indices.map { covariance[$0][$0] }.reduce(0.0, +)
        var residual = covariance
        var loadings: [[Double]] = []
        var scores: [[Double]] = []
        var eigenvalues: [Double] = []

        for component in 0..<min(max(componentCount, 0), width) {
            var vector = deterministicSeedVector(width: width, component: component)
            for _ in 0..<max(iterations, 1) {
                var next = matVec(residual, vector)
                for previous in loadings {
                    next = subtract(next, previous, scale: dot(next, previous))
                }
                next = normalize(next)
                if distance(next, vector) < 1.0e-10 {
                    vector = next
                    break
                }
                vector = next
            }
            let cv = matVec(covariance, vector)
            let eigenvalue = max(dot(vector, cv), 0.0)
            guard eigenvalue > 1.0e-14 else { break }
            loadings.append(vector)
            scores.append(centered.map { dot($0, vector) })
            eigenvalues.append(eigenvalue)
            residual = deflate(residual, vector: vector, eigenvalue: eigenvalue)
        }
        return ComponentSet(
            covariance: covariance,
            loadings: loadings,
            scores: scores,
            eigenvalues: eigenvalues,
            explainedVarianceRatios: totalVariance > 0.0 ? eigenvalues.map { $0 / totalVariance } : Array(repeating: 0.0, count: eigenvalues.count)
        )
    }

    private static func centerColumns(_ panel: [[Double]]) -> [[Double]] {
        guard let width = panel.first?.count else { return [] }
        let means = (0..<width).map { column in
            panel.map { column < $0.count ? $0[column] : 0.0 }.reduce(0.0, +) / Double(panel.count)
        }
        return panel.map { row in
            (0..<width).map { column in (column < row.count ? row[column] : 0.0) - means[column] }
        }
    }

    private static func covarianceMatrix(_ centered: [[Double]]) -> [[Double]] {
        guard let width = centered.first?.count, centered.count > 1 else { return [] }
        var matrix = Array(repeating: Array(repeating: 0.0, count: width), count: width)
        for row in centered {
            for i in 0..<width {
                for j in i..<width {
                    matrix[i][j] += row[i] * row[j]
                }
            }
        }
        let divisor = Double(centered.count - 1)
        for i in 0..<width {
            for j in i..<width {
                matrix[i][j] /= divisor
                matrix[j][i] = matrix[i][j]
            }
        }
        return matrix
    }

    private static func matVec(_ matrix: [[Double]], _ vector: [Double]) -> [Double] {
        matrix.map { dot($0, vector) }
    }

    private static func normalize(_ values: [Double]) -> [Double] {
        let norm = sqrt(max(dot(values, values), 1.0e-24))
        return values.map { $0 / norm }
    }

    private static func distance(_ lhs: [Double], _ rhs: [Double]) -> Double {
        sqrt(zip(lhs, rhs).map { pow($0 - $1, 2.0) }.reduce(0.0, +))
    }

    private static func dot(_ lhs: [Double], _ rhs: [Double]) -> Double {
        zip(lhs, rhs).map(*).reduce(0.0, +)
    }

    private static func subtract(_ lhs: [Double], _ rhs: [Double], scale: Double) -> [Double] {
        zip(lhs, rhs).map { $0 - scale * $1 }
    }

    private static func deterministicSeedVector(width: Int, component: Int) -> [Double] {
        normalize((0..<width).map { index in
            sin(Double((index + 1) * (component + 3))) + 0.25 * cos(Double((index + 2) * (component + 5)))
        })
    }

    private static func deflate(_ matrix: [[Double]], vector: [Double], eigenvalue: Double) -> [[Double]] {
        matrix.indices.map { row in
            matrix[row].indices.map { column in
                matrix[row][column] - eigenvalue * vector[row] * vector[column]
            }
        }
    }
}
