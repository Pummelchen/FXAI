import Foundation

public enum MemRetrdiffReference {
    public struct Item: Equatable, Sendable {
        public let id: String
        public let vector: [Double]
        public let label: Double
        public let timestamp: Int

        public init(id: String, vector: [Double], label: Double, timestamp: Int) {
            self.id = id
            self.vector = vector
            self.label = label
            self.timestamp = timestamp
        }
    }

    public struct Neighbor: Equatable, Sendable {
        public let item: Item
        public let distance: Double
    }

    public static func topK(query: [Double], memory: [Item], k: Int) -> [Neighbor] {
        memory.map { Neighbor(item: $0, distance: euclidean(query, $0.vector)) }
            .sorted {
                if abs($0.distance - $1.distance) < 1.0e-12 {
                    return $0.item.timestamp > $1.item.timestamp
                }
                return $0.distance < $1.distance
            }
            .prefix(max(k, 0))
            .map { $0 }
    }

    public static func evict(memory: [Item], capacity: Int) -> [Item] {
        guard capacity > 0 else { return [] }
        return memory.sorted { $0.timestamp > $1.timestamp }.prefix(capacity).sorted { $0.timestamp < $1.timestamp }
    }

    private static func euclidean(_ lhs: [Double], _ rhs: [Double]) -> Double {
        sqrt(zip(lhs, rhs).map { pow($0 - $1, 2.0) }.reduce(0.0, +))
    }
}
