import Foundation

public enum StatCointVECMReference {
    public struct PairFit: Equatable, Sendable {
        public let intercept: Double
        public let beta: Double
        public let residuals: [Double]
        public let residualMean: Double
        public let residualVariance: Double
        public let adfTStatistic: Double
        public let rank: Int
        public let alphaY: Double
        public let alphaX: Double
        public let gammaY: Double
        public let gammaX: Double
    }

    public static func estimatePair(y: [Double], x: [Double], lagOrder: Int = 1) -> PairFit {
        let count = min(y.count, x.count)
        guard count >= max(8, lagOrder + 4) else {
            return PairFit(
                intercept: mean(y),
                beta: 0.0,
                residuals: [],
                residualMean: 0.0,
                residualVariance: 0.0,
                adfTStatistic: 0.0,
                rank: 0,
                alphaY: 0.0,
                alphaX: 0.0,
                gammaY: 0.0,
                gammaX: 0.0
            )
        }

        let yy = Array(y.prefix(count))
        let xx = Array(x.prefix(count))
        let meanY = mean(yy)
        let meanX = mean(xx)
        let beta = covariance(yy, xx) / max(variance(xx), 1.0e-12)
        let intercept = meanY - beta * meanX
        let residuals = zip(yy, xx).map { $0 - intercept - beta * $1 }
        let residualMean = mean(residuals)
        let residualVariance = variance(residuals)
        let adf = adfTStatistic(residuals: residuals, lagOrder: lagOrder)
        let deltasY = differences(yy)
        let deltasX = differences(xx)
        let laggedResidual = Array(residuals.dropLast())
        let laggedDeltaY = Array(deltasY.dropLast())
        let laggedDeltaX = Array(deltasX.dropLast())
        let responseDeltaY = Array(deltasY.dropFirst())
        let responseDeltaX = Array(deltasX.dropFirst())

        let alphaY = slope(response: responseDeltaY, predictor: Array(laggedResidual.dropFirst()))
        let alphaX = slope(response: responseDeltaX, predictor: Array(laggedResidual.dropFirst()))
        let gammaY = slope(response: responseDeltaY, predictor: laggedDeltaY)
        let gammaX = slope(response: responseDeltaX, predictor: laggedDeltaX)

        return PairFit(
            intercept: intercept,
            beta: beta,
            residuals: residuals,
            residualMean: residualMean,
            residualVariance: residualVariance,
            adfTStatistic: adf,
            rank: adf < -2.90 ? 1 : 0,
            alphaY: alphaY,
            alphaX: alphaX,
            gammaY: gammaY,
            gammaX: gammaX
        )
    }

    public static func forecastNext(yLast: Double, xLast: Double, fit: PairFit) -> (deltaY: Double, deltaX: Double, residual: Double) {
        let residual = yLast - fit.intercept - fit.beta * xLast
        return (
            deltaY: fit.alphaY * residual,
            deltaX: fit.alphaX * residual,
            residual: residual
        )
    }

    public static func spreadZScores(fit: PairFit) -> [Double] {
        let sigma = sqrt(max(fit.residualVariance, 1.0e-12))
        return fit.residuals.map { ($0 - fit.residualMean) / sigma }
    }

    private static func adfTStatistic(residuals: [Double], lagOrder: Int) -> Double {
        let deltas = differences(residuals)
        guard deltas.count > lagOrder + 2 else { return 0.0 }
        var design: [[Double]] = []
        var target: [Double] = []
        for t in lagOrder..<deltas.count {
            var row = [1.0, residuals[t]]
            if lagOrder > 0 {
                for lag in 1...lagOrder {
                    row.append(deltas[t - lag])
                }
            }
            design.append(row)
            target.append(deltas[t])
        }
        let coefficients = solveNormalEquations(design: design, target: target, ridge: 1.0e-9)
        let fitted = design.map { dot(coefficients, $0) }
        let residual = zip(target, fitted).map { $0 - $1 }
        let dof = max(Double(target.count - coefficients.count), 1.0)
        let sigma2 = residual.map { $0 * $0 }.reduce(0.0, +) / dof
        let xtxInv = inverseSymmetricNormal(design: design, ridge: 1.0e-9)
        guard coefficients.count > 1, xtxInv.count > 1 else { return 0.0 }
        let stderr = sqrt(max(sigma2 * xtxInv[1][1], 1.0e-12))
        return coefficients[1] / stderr
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

    private static func inverseSymmetricNormal(design: [[Double]], ridge: Double) -> [[Double]] {
        guard let width = design.first?.count, width > 0 else { return [] }
        var a = Array(repeating: Array(repeating: 0.0, count: width), count: width)
        for row in design {
            for i in 0..<width {
                for j in 0..<width {
                    a[i][j] += row[i] * row[j]
                }
            }
        }
        for i in 0..<width {
            a[i][i] += ridge
        }
        var inverse = Array(repeating: Array(repeating: 0.0, count: width), count: width)
        for column in 0..<width {
            var rhs = Array(repeating: 0.0, count: width)
            rhs[column] = 1.0
            let solved = gaussianSolve(a, rhs)
            for row in 0..<width {
                inverse[row][column] = row < solved.count ? solved[row] : 0.0
            }
        }
        return inverse
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
            let diagonal = abs(a[pivot][pivot]) < 1.0e-12 ? (a[pivot][pivot] < 0.0 ? -1.0e-12 : 1.0e-12) : a[pivot][pivot]
            for row in (pivot + 1)..<n {
                let factor = a[row][pivot] / diagonal
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
            let diagonal = abs(a[row][row]) < 1.0e-12 ? (a[row][row] < 0.0 ? -1.0e-12 : 1.0e-12) : a[row][row]
            x[row] = value / diagonal
        }
        return x
    }

    private static func differences(_ values: [Double]) -> [Double] {
        guard values.count > 1 else { return [] }
        return zip(values.dropFirst(), values.dropLast()).map { $0 - $1 }
    }

    private static func slope(response: [Double], predictor: [Double]) -> Double {
        let count = min(response.count, predictor.count)
        guard count > 2 else { return 0.0 }
        let y = Array(response.prefix(count))
        let x = Array(predictor.prefix(count))
        return covariance(y, x) / max(variance(x), 1.0e-12)
    }

    private static func dot(_ lhs: [Double], _ rhs: [Double]) -> Double {
        zip(lhs, rhs).map(*).reduce(0.0, +)
    }

    private static func covariance(_ lhs: [Double], _ rhs: [Double]) -> Double {
        let count = min(lhs.count, rhs.count)
        guard count > 1 else { return 0.0 }
        let left = Array(lhs.prefix(count))
        let right = Array(rhs.prefix(count))
        let ml = mean(left)
        let mr = mean(right)
        return zip(left, right).map { ($0 - ml) * ($1 - mr) }.reduce(0.0, +) / Double(count - 1)
    }

    private static func mean(_ values: [Double]) -> Double {
        values.isEmpty ? 0.0 : values.reduce(0.0, +) / Double(values.count)
    }

    private static func variance(_ values: [Double]) -> Double {
        guard values.count > 1 else { return 0.0 }
        let m = mean(values)
        return values.map { ($0 - m) * ($0 - m) }.reduce(0.0, +) / Double(values.count - 1)
    }
}
