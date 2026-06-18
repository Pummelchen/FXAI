import Foundation

public enum AINbeatsReference {
    /// Evaluates a polynomial trend basis of the given degree at normalized time indices.
    ///
    /// Returns a matrix of shape `[timeSteps, degree + 1]` where column `k`
    /// contains `(t / timeSteps)^k` for `t` in `0..<timeSteps`.
    public static func polynomialBasis(timeSteps: Int, degree: Int) -> [[Double]] {
        guard timeSteps > 0, degree >= 0 else { return [] }
        let norm = max(Double(timeSteps - 1), 1.0)
        return (0..<timeSteps).map { t in
            let x = Double(t) / norm
            var row = Array(repeating: 0.0, count: degree + 1)
            row[0] = 1.0
            for k in 1...degree {
                row[k] = row[k - 1] * x
            }
            return row
        }
    }

    /// Evaluates a Fourier seasonality basis with the specified number of harmonics.
    ///
    /// Returns a matrix of shape `[timeSteps, 2 * harmonics]` where columns
    /// alternate between cosine and sine terms for each harmonic frequency.
    public static func fourierBasis(timeSteps: Int, harmonics: Int) -> [[Double]] {
        guard timeSteps > 0, harmonics > 0 else { return [] }
        return (0..<timeSteps).map { t in
            var row = Array(repeating: 0.0, count: 2 * harmonics)
            for h in 0..<harmonics {
                let freq = 2.0 * Double.pi * Double(h + 1) / Double(timeSteps)
                row[2 * h] = cos(freq * Double(t))
                row[2 * h + 1] = sin(freq * Double(t))
            }
            return row
        }
    }

    /// Computes the mean absolute error between two equal-length sequences.
    public static func mae(_ a: [Double], _ b: [Double]) -> Double {
        let n = min(a.count, b.count)
        guard n > 0 else { return 0.0 }
        return zip(a.prefix(n), b.prefix(n)).map { abs($0 - $1) }.reduce(0.0, +) / Double(n)
    }
}
