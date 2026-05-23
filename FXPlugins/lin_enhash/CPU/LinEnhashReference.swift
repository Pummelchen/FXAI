import Foundation

public enum LinEnhashReference {
    public struct Interaction: Equatable, Sendable {
        public let leftField: Int
        public let rightField: Int
        public let bucket: Int
        public let value: Double
    }

    public static func interactions(features: [Double], fieldRanges: [Range<Int>], bucketCount: Int, seed: UInt64) -> [Interaction] {
        guard bucketCount > 0 else { return [] }
        var output: [Interaction] = []
        for left in fieldRanges.indices {
            for right in fieldRanges.indices where right >= left {
                let leftValue = fieldValue(features: features, range: fieldRanges[left])
                let rightValue = fieldValue(features: features, range: fieldRanges[right])
                let bucket = hash(left: left, right: right, bucketCount: bucketCount, seed: seed)
                output.append(Interaction(leftField: left, rightField: right, bucket: bucket, value: leftValue * rightValue))
            }
        }
        return output
    }

    public static func collisionRate(interactions: [Interaction]) -> Double {
        guard !interactions.isEmpty else { return 0.0 }
        let unique = Set(interactions.map(\.bucket)).count
        return 1.0 - Double(unique) / Double(interactions.count)
    }

    private static func fieldValue(features: [Double], range: Range<Int>) -> Double {
        let values = range.compactMap { index in
            index >= 0 && index < features.count ? features[index] : nil
        }
        guard !values.isEmpty else { return 0.0 }
        return values.reduce(0.0, +) / Double(values.count)
    }

    private static func hash(left: Int, right: Int, bucketCount: Int, seed: UInt64) -> Int {
        var value = seed ^ UInt64(left &+ 0x9E37_79B9)
        value = value &* 0xBF58_476D_1CE4_E5B9
        value ^= UInt64(right &+ 0x94D0_49BB)
        value = value &* 0x94D0_49BB_1331_11EB
        value ^= value >> 31
        return Int(value % UInt64(bucketCount))
    }
}
