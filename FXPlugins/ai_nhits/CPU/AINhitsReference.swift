import Foundation

public enum AINhitsReference {
    /// Performs multi-rate max-pooling over a 1-D sequence with the given pool size.
    ///
    /// Returns a downsampled array of length `ceil(input.count / poolSize)`.
    /// Each output element is the maximum of its corresponding window.
    public static func maxPool1d(_ input: [Double], poolSize: Int) -> [Double] {
        guard poolSize > 0, !input.isEmpty else { return [] }
        let count = input.count
        var output: [Double] = []
        output.reserveCapacity((count + poolSize - 1) / poolSize)
        var index = 0
        while index < count {
            let end = min(index + poolSize, count)
            var maxVal = input[index]
            for j in (index + 1)..<end {
                maxVal = max(maxVal, input[j])
            }
            output.append(maxVal)
            index += poolSize
        }
        return output
    }

    /// Linearly interpolates a short sequence up to the target length.
    ///
    /// This mirrors the hierarchical interpolation strategy of N-HiTS where
    /// coarser-scale forecasts are upsampled and blended with finer scales.
    public static func linearInterpolate(_ input: [Double], targetLength: Int) -> [Double] {
        guard !input.isEmpty, targetLength > 0 else { return [] }
        if input.count == targetLength { return input }
        if input.count == 1 { return Array(repeating: input[0], count: targetLength) }
        let sourceMax = Double(input.count - 1)
        let targetMax = Double(targetLength - 1)
        return (0..<targetLength).map { i in
            let position = sourceMax * Double(i) / targetMax
            let lo = Int(position.rounded(.down))
            let hi = min(lo + 1, input.count - 1)
            let frac = position - Double(lo)
            return input[lo] * (1.0 - frac) + input[hi] * frac
        }
    }

    /// Computes the root mean squared error between two equal-length sequences.
    public static func rmse(_ a: [Double], _ b: [Double]) -> Double {
        let n = min(a.count, b.count)
        guard n > 0 else { return 0.0 }
        let mse = zip(a.prefix(n), b.prefix(n)).map { ($0 - $1) * ($0 - $1) }.reduce(0.0, +) / Double(n)
        return sqrt(mse)
    }
}
