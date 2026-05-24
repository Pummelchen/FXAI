import Foundation

public enum WMCFXReference {
    public struct PairReturn: Equatable, Sendable {
        public let base: String
        public let quote: String
        public let logReturn: Double
        public let volume: Double

        public init(base: String, quote: String, logReturn: Double, volume: Double = 0.0) {
            self.base = base.uppercased()
            self.quote = quote.uppercased()
            self.logReturn = logReturn
            self.volume = volume
        }
    }

    public struct CurrencyFactor: Equatable, Sendable {
        public let currency: String
        public let strength: Double
        public let liquidityWeight: Double
    }

    public struct FactorSnapshot: Equatable, Sendable {
        public let factors: [CurrencyFactor]
        public let residualRMSE: Double
    }

    public struct CycleImbalance: Equatable, Sendable {
        public let currencies: [String]
        public let basisPoints: Double
    }

    /// Infers zero-sum latent currency strengths from pair log returns.
    ///
    /// The reference model solves a weighted ridge least-squares system where
    /// each pair is encoded as +base and -quote, matching the standard currency
    /// factor interpretation used by FX world models.
    public static func inferCurrencyFactors(
        _ returns: [PairReturn],
        dataHasVolume: Bool,
        ridge: Double = 1.0e-6
    ) -> FactorSnapshot {
        let currencies = Array(Set(returns.flatMap { [$0.base, $0.quote] })).sorted()
        guard !currencies.isEmpty else {
            return FactorSnapshot(factors: [], residualRMSE: 0.0)
        }
        let index = Dictionary(uniqueKeysWithValues: currencies.enumerated().map { ($0.element, $0.offset) })
        let n = currencies.count
        var normal = Array(repeating: Array(repeating: 0.0, count: n), count: n)
        var rhs = Array(repeating: 0.0, count: n)
        var liquidity = Array(repeating: 0.0, count: n)

        for row in returns {
            guard let base = index[row.base], let quote = index[row.quote] else { continue }
            let rowScale = dataHasVolume && row.volume > 0.0 ? sqrt(row.volume) : 1.0
            let sampleWeight = rowScale * rowScale
            let encoded = [(base, 1.0), (quote, -1.0)]
            for (i, vi) in encoded {
                rhs[i] += sampleWeight * vi * row.logReturn
                liquidity[i] += rowScale
                for (j, vj) in encoded {
                    normal[i][j] += sampleWeight * vi * vj
                }
            }
        }

        for i in 0..<n {
            normal[i][i] += max(ridge, 1.0e-12)
        }
        var strengths = solve(normal, rhs) ?? Array(repeating: 0.0, count: n)
        let meanStrength = strengths.reduce(0.0, +) / Double(max(strengths.count, 1))
        strengths = strengths.map { $0 - meanStrength }

        let residuals = returns.map { row -> Double in
            guard let base = index[row.base], let quote = index[row.quote] else { return 0.0 }
            return (strengths[base] - strengths[quote]) - row.logReturn
        }
        let rmse = sqrt(residuals.map { $0 * $0 }.reduce(0.0, +) / Double(max(residuals.count, 1)))
        let factors = currencies.indices.map { idx in
            CurrencyFactor(
                currency: currencies[idx],
                strength: strengths[idx],
                liquidityWeight: dataHasVolume ? max(liquidity[idx] / Double(max(returns.count, 1)), 0.0) : 1.0
            )
        }
        return FactorSnapshot(factors: factors.sorted { $0.strength > $1.strength }, residualRMSE: rmse)
    }

    /// Computes triangular cross-rate consistency errors from pair log returns.
    ///
    /// A zero cycle value means pair returns are internally consistent; non-zero
    /// basis points expose graph/world-model arbitrage tension.
    public static func cycleImbalances(_ returns: [PairReturn]) -> [CycleImbalance] {
        let currencies = Array(Set(returns.flatMap { [$0.base, $0.quote] })).sorted()
        guard currencies.count >= 3 else { return [] }
        return combinations(currencies, choose: 3).compactMap { triple in
            guard
                let ab = signedReturn(triple[0], triple[1], returns),
                let bc = signedReturn(triple[1], triple[2], returns),
                let ac = signedReturn(triple[0], triple[2], returns)
            else { return nil }
            return CycleImbalance(currencies: triple, basisPoints: (ab + bc - ac) * 10_000.0)
        }
        .sorted { abs($0.basisPoints) > abs($1.basisPoints) }
    }

    private static func signedReturn(_ base: String, _ quote: String, _ returns: [PairReturn]) -> Double? {
        if let direct = returns.first(where: { $0.base == base && $0.quote == quote }) {
            return direct.logReturn
        }
        if let inverse = returns.first(where: { $0.base == quote && $0.quote == base }) {
            return -inverse.logReturn
        }
        return nil
    }

    private static func combinations(_ values: [String], choose: Int) -> [[String]] {
        guard choose > 0 else { return [[]] }
        guard values.count >= choose else { return [] }
        if choose == 1 { return values.map { [$0] } }
        var result: [[String]] = []
        for index in 0...(values.count - choose) {
            let head = values[index]
            let tail = Array(values[(index + 1)...])
            result += combinations(tail, choose: choose - 1).map { [head] + $0 }
        }
        return result
    }

    private static func solve(_ matrix: [[Double]], _ rhs: [Double]) -> [Double]? {
        let n = rhs.count
        guard matrix.count == n, matrix.allSatisfy({ $0.count == n }) else { return nil }
        var a = matrix
        var b = rhs
        for pivot in 0..<n {
            var best = pivot
            for row in pivot..<n where abs(a[row][pivot]) > abs(a[best][pivot]) {
                best = row
            }
            guard abs(a[best][pivot]) > 1.0e-14 else { return nil }
            if best != pivot {
                a.swapAt(best, pivot)
                b.swapAt(best, pivot)
            }
            let divisor = a[pivot][pivot]
            for column in pivot..<n {
                a[pivot][column] /= divisor
            }
            b[pivot] /= divisor
            for row in 0..<n where row != pivot {
                let factor = a[row][pivot]
                guard factor != 0.0 else { continue }
                for column in pivot..<n {
                    a[row][column] -= factor * a[pivot][column]
                }
                b[row] -= factor * b[pivot]
            }
        }
        return b
    }
}
