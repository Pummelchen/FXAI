import Foundation

public enum TreeCatboostReference {
    public struct CTR: Equatable, Sendable {
        public let row: Int
        public let category: String
        public let value: Double
    }

    public static func orderedCTR(categories: [String], labels: [Double], permutation: [Int], prior: Double, priorWeight: Double) -> [CTR] {
        let count = min(categories.count, labels.count)
        var sums: [String: Double] = [:]
        var counts: [String: Double] = [:]
        var result = Array(repeating: CTR(row: 0, category: "", value: prior), count: count)
        for row in permutation where row >= 0 && row < count {
            let category = categories[row]
            let numerator = (sums[category] ?? 0.0) + prior * priorWeight
            let denominator = (counts[category] ?? 0.0) + priorWeight
            result[row] = CTR(row: row, category: category, value: numerator / max(denominator, 1.0e-12))
            sums[category, default: 0.0] += labels[row]
            counts[category, default: 0.0] += 1.0
        }
        return result
    }

    public static func symmetricLeafIndex(features: [Double], featureIndexes: [Int], thresholds: [Double]) -> Int {
        var leaf = 0
        for depth in 0..<min(featureIndexes.count, thresholds.count) {
            let featureIndex = featureIndexes[depth]
            let value = featureIndex < features.count ? features[featureIndex] : 0.0
            if value > thresholds[depth] {
                leaf |= 1 << depth
            }
        }
        return leaf
    }
}
