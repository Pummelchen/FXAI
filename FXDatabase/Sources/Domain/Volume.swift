import Foundation

public struct M1Volume: RawRepresentable, Codable, Hashable, Sendable, Comparable, CustomStringConvertible {
    public let rawValue: UInt64

    public init(rawValue: UInt64) {
        self.rawValue = rawValue
    }

    public static let zero = M1Volume(rawValue: 0)

    public var description: String { String(rawValue) }

    public static func < (lhs: M1Volume, rhs: M1Volume) -> Bool {
        lhs.rawValue < rhs.rawValue
    }
}
