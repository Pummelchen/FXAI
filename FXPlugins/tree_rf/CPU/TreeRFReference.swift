import Foundation

public enum TreeRFReference {
    public struct Bootstrap: Equatable, Sendable {
        public let sample: [Int]
        public let outOfBag: [Int]
    }

    public struct GiniSplit: Equatable, Sendable {
        public let threshold: Double
        public let impurity: Double
        public let leftCount: Int
        public let rightCount: Int
    }

    public static func bootstrap(rowCount: Int, sampleCount: Int, seed: UInt64) -> Bootstrap {
        guard rowCount > 0, sampleCount > 0 else { return Bootstrap(sample: [], outOfBag: Array(0..<max(rowCount, 0))) }
        var state = seed == 0 ? 0xA24B_AED4_963E_E407 : seed
        var sample: [Int] = []
        sample.reserveCapacity(sampleCount)
        for _ in 0..<sampleCount {
            state = state &* 2862933555777941757 &+ 3037000493
            sample.append(Int(state % UInt64(rowCount)))
        }
        let sampled = Set(sample)
        let oob = (0..<rowCount).filter { !sampled.contains($0) }
        return Bootstrap(sample: sample, outOfBag: oob)
    }

    public static func bestGiniSplit(feature: [Double], labels: [Int], minLeaf: Int) -> GiniSplit? {
        let count = min(feature.count, labels.count)
        guard count >= 2 * minLeaf, minLeaf > 0 else { return nil }
        let pairs = (0..<count).map { (feature[$0], labels[$0]) }.sorted { $0.0 < $1.0 }
        var best: GiniSplit?
        for index in minLeaf..<(count - minLeaf + 1) {
            guard pairs[index - 1].0 < pairs[min(index, count - 1)].0 else { continue }
            let leftLabels = pairs.prefix(index).map(\.1)
            let rightLabels = pairs.suffix(count - index).map(\.1)
            let impurity = (Double(leftLabels.count) * gini(leftLabels) + Double(rightLabels.count) * gini(rightLabels)) / Double(count)
            let threshold = 0.5 * (pairs[index - 1].0 + pairs[index].0)
            if best == nil || impurity < best!.impurity {
                best = GiniSplit(threshold: threshold, impurity: impurity, leftCount: leftLabels.count, rightCount: rightLabels.count)
            }
        }
        return best
    }

    private static func gini(_ labels: [Int]) -> Double {
        guard !labels.isEmpty else { return 0.0 }
        var counts: [Int: Int] = [:]
        for label in labels {
            counts[label, default: 0] += 1
        }
        let total = Double(labels.count)
        return 1.0 - counts.values.map { pow(Double($0) / total, 2.0) }.reduce(0.0, +)
    }
}
