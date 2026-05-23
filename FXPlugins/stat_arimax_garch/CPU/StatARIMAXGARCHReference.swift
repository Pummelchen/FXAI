import Foundation

public enum StatARIMAXGARCHReference {
    public struct ARIMAXFit: Equatable, Sendable {
        public let coefficients: [Double]
        public let fitted: [Double]
        public let residuals: [Double]
        public let lagOrder: Int
    }

    public struct GARCHFit: Equatable, Sendable {
        public let omega: Double
        public let alpha: Double
        public let beta: Double
        public let variances: [Double]
        public let logLikelihood: Double
    }

    public static func fitARIMAX(
        series: [Double],
        exogenous: [[Double]] = [],
        lagOrder: Int,
        ridge: Double = 1.0e-8
    ) -> ARIMAXFit {
        let n = series.count
        guard lagOrder > 0, n > lagOrder + 1 else {
            return ARIMAXFit(coefficients: [mean(series)], fitted: series.map { _ in mean(series) }, residuals: series.map { $0 - mean(series) }, lagOrder: max(lagOrder, 0))
        }
        let exogWidth = exogenous.first?.count ?? 0
        var design: [[Double]] = []
        var target: [Double] = []
        for t in lagOrder..<n {
            var row = [1.0]
            for lag in 1...lagOrder {
                row.append(series[t - lag])
            }
            if t < exogenous.count {
                row.append(contentsOf: exogenous[t].prefix(exogWidth))
            } else {
                row.append(contentsOf: Array(repeating: 0.0, count: exogWidth))
            }
            design.append(row)
            target.append(series[t])
        }
        let coefficients = solveNormalEquations(design: design, target: target, ridge: ridge)
        let fitted = design.map { dot(coefficients, $0) }
        let residuals = zip(target, fitted).map(-)
        return ARIMAXFit(coefficients: coefficients, fitted: fitted, residuals: residuals, lagOrder: lagOrder)
    }

    public static func estimateGARCH(
        residuals: [Double],
        omegaGrid: [Double] = [0.0001, 0.0005, 0.001, 0.005],
        alphaGrid: [Double] = [0.05, 0.10, 0.15, 0.20],
        betaGrid: [Double] = [0.70, 0.80, 0.88, 0.93]
    ) -> GARCHFit {
        var best = garchFit(residuals: residuals, omega: 0.001, alpha: 0.10, beta: 0.85)
        for omega in omegaGrid where omega > 0 {
            for alpha in alphaGrid where alpha >= 0 {
                for beta in betaGrid where beta >= 0 && alpha + beta < 0.999 {
                    let candidate = garchFit(residuals: residuals, omega: omega, alpha: alpha, beta: beta)
                    if candidate.logLikelihood > best.logLikelihood {
                        best = candidate
                    }
                }
            }
        }
        return best
    }

    public static func garchFit(residuals: [Double], omega: Double, alpha: Double, beta: Double) -> GARCHFit {
        guard !residuals.isEmpty else {
            return GARCHFit(omega: omega, alpha: alpha, beta: beta, variances: [], logLikelihood: 0.0)
        }
        let unconditional = max(variance(residuals), 1.0e-8)
        var previousVariance = unconditional
        var previousResidual = residuals[0]
        var variances: [Double] = []
        variances.reserveCapacity(residuals.count)
        var logLikelihood = 0.0
        for residual in residuals {
            let nextVariance = max(omega + alpha * previousResidual * previousResidual + beta * previousVariance, 1.0e-10)
            variances.append(nextVariance)
            logLikelihood += -0.5 * (log(2.0 * Double.pi) + log(nextVariance) + residual * residual / nextVariance)
            previousResidual = residual
            previousVariance = nextVariance
        }
        return GARCHFit(omega: omega, alpha: alpha, beta: beta, variances: variances, logLikelihood: logLikelihood)
    }

    private static func solveNormalEquations(design: [[Double]], target: [Double], ridge: Double) -> [Double] {
        guard let width = design.first?.count, width > 0 else { return [] }
        var a = Array(repeating: Array(repeating: 0.0, count: width), count: width)
        var b = Array(repeating: 0.0, count: width)
        for (row, y) in zip(design, target) {
            for i in 0..<width {
                b[i] += row[i] * y
                for j in 0..<width {
                    a[i][j] += row[i] * row[j]
                }
            }
        }
        for i in 0..<width {
            a[i][i] += ridge
        }
        return gaussianSolve(a, b)
    }

    private static func gaussianSolve(_ matrix: [[Double]], _ rhs: [Double]) -> [Double] {
        let n = rhs.count
        guard matrix.count == n, n > 0 else { return [] }
        var a = matrix
        var b = rhs
        for pivot in 0..<n {
            var best = pivot
            for row in pivot..<n where abs(a[row][pivot]) > abs(a[best][pivot]) {
                best = row
            }
            if best != pivot {
                a.swapAt(best, pivot)
                b.swapAt(best, pivot)
            }
            let diagonal = abs(a[pivot][pivot]) < 1.0e-12 ? (a[pivot][pivot] < 0 ? -1.0e-12 : 1.0e-12) : a[pivot][pivot]
            for row in (pivot + 1)..<n {
                let factor = a[row][pivot] / diagonal
                guard factor.isFinite else { continue }
                for column in pivot..<n {
                    a[row][column] -= factor * a[pivot][column]
                }
                b[row] -= factor * b[pivot]
            }
        }
        var x = Array(repeating: 0.0, count: n)
        for row in stride(from: n - 1, through: 0, by: -1) {
            var value = b[row]
            if row + 1 < n {
                for column in (row + 1)..<n {
                    value -= a[row][column] * x[column]
                }
            }
            let diagonal = abs(a[row][row]) < 1.0e-12 ? (a[row][row] < 0 ? -1.0e-12 : 1.0e-12) : a[row][row]
            x[row] = value / diagonal
        }
        return x
    }

    private static func dot(_ lhs: [Double], _ rhs: [Double]) -> Double {
        zip(lhs, rhs).map(*).reduce(0.0, +)
    }

    private static func mean(_ values: [Double]) -> Double {
        values.isEmpty ? 0.0 : values.reduce(0.0, +) / Double(values.count)
    }

    private static func variance(_ values: [Double]) -> Double {
        guard values.count > 1 else { return 1.0e-8 }
        let m = mean(values)
        return values.map { ($0 - m) * ($0 - m) }.reduce(0.0, +) / Double(values.count - 1)
    }
}
