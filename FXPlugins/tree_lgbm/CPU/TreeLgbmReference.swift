import Foundation

public enum TreeLgbmReference {
    public struct Bin: Equatable, Sendable {
        public let upperBound: Double
        public let gradient: Double
        public let hessian: Double
        public let count: Int
    }

    public struct Split: Equatable, Sendable {
        public let binIndex: Int
        public let gain: Double
        public let leftCount: Int
        public let rightCount: Int
    }

    public static func histogram(feature: [Double], gradients: [Double], hessians: [Double], binCount: Int) -> [Bin] {
        let count = min(feature.count, gradients.count, hessians.count)
        guard count > 0, binCount > 0 else { return [] }
        let values = Array(feature.prefix(count))
        let minValue = values.min() ?? 0.0
        let maxValue = values.max() ?? minValue
        let width = max((maxValue - minValue) / Double(binCount), 1.0e-12)
        var grad = Array(repeating: 0.0, count: binCount)
        var hess = Array(repeating: 0.0, count: binCount)
        var counts = Array(repeating: 0, count: binCount)
        for index in 0..<count {
            let bin = min(max(Int((values[index] - minValue) / width), 0), binCount - 1)
            grad[bin] += gradients[index]
            hess[bin] += hessians[index]
            counts[bin] += 1
        }
        return (0..<binCount).map { bin in
            Bin(upperBound: minValue + Double(bin + 1) * width, gradient: grad[bin], hessian: hess[bin], count: counts[bin])
        }
    }

    public static func bestSplit(histogram: [Bin], lambda: Double, minDataInLeaf: Int) -> Split? {
        guard histogram.count > 1 else { return nil }
        let totalGradient = histogram.map(\.gradient).reduce(0.0, +)
        let totalHessian = histogram.map(\.hessian).reduce(0.0, +)
        let totalCount = histogram.map(\.count).reduce(0, +)
        var leftGradient = 0.0
        var leftHessian = 0.0
        var leftCount = 0
        var best: Split?
        for index in 0..<(histogram.count - 1) {
            leftGradient += histogram[index].gradient
            leftHessian += histogram[index].hessian
            leftCount += histogram[index].count
            let rightCount = totalCount - leftCount
            guard leftCount >= minDataInLeaf, rightCount >= minDataInLeaf else { continue }
            let rightGradient = totalGradient - leftGradient
            let rightHessian = totalHessian - leftHessian
            let gain = score(leftGradient, leftHessian, lambda) + score(rightGradient, rightHessian, lambda) - score(totalGradient, totalHessian, lambda)
            if best == nil || gain > best!.gain {
                best = Split(binIndex: index, gain: gain, leftCount: leftCount, rightCount: rightCount)
            }
        }
        return best
    }

    public static func dartKeepMask(treeCount: Int, dropRate: Double, seed: UInt64) -> [Bool] {
        var state = seed == 0 ? 0x9E37_79B9_7F4A_7C15 : seed
        return (0..<treeCount).map { _ in
            state = state &* 6364136223846793005 &+ 1442695040888963407
            let u = Double(state >> 11) / Double(UInt64.max >> 11)
            return u >= dropRate
        }
    }

    private static func score(_ gradient: Double, _ hessian: Double, _ lambda: Double) -> Double {
        gradient * gradient / max(hessian + lambda, 1.0e-12)
    }
}
