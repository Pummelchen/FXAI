import Foundation

public enum FactorPCAPanelReference {
    public struct Result: Equatable, Sendable {
        public let covariance: [[Double]]
        public let loadings: [Double]
        public let scores: [Double]
        public let eigenvalue: Double
        public let explainedVarianceRatio: Double
    }

    public static func firstComponent(panel: [[Double]], iterations: Int = 64) -> Result {
        guard let width = panel.first?.count, panel.count > 1, width > 0 else {
            return Result(covariance: [], loadings: [], scores: [], eigenvalue: 0.0, explainedVarianceRatio: 0.0)
        }
        let centered = centerColumns(panel)
        let covariance = covarianceMatrix(centered)
        var vector = Array(repeating: 1.0 / sqrt(Double(width)), count: width)
        for _ in 0..<max(iterations, 1) {
            let next = normalize(matVec(covariance, vector))
            if distance(next, vector) < 1.0e-10 {
                vector = next
                break
            }
            vector = next
        }
        let cv = matVec(covariance, vector)
        let eigenvalue = dot(vector, cv)
        let totalVariance = covariance.indices.map { covariance[$0][$0] }.reduce(0.0, +)
        let scores = centered.map { dot($0, vector) }
        return Result(
            covariance: covariance,
            loadings: vector,
            scores: scores,
            eigenvalue: eigenvalue,
            explainedVarianceRatio: totalVariance > 0.0 ? eigenvalue / totalVariance : 0.0
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
}
