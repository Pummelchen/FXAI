import Foundation

public struct DataSourceOrigin: RawRepresentable, Codable, Hashable, Sendable, Comparable, CustomStringConvertible {
    public static let mt5 = DataSourceOrigin(rawValueUnchecked: "MT5")
    public static let synthetic = DataSourceOrigin(rawValueUnchecked: "SYNTHETIC")

    public let rawValue: String

    public init(_ rawValue: String) throws {
        let trimmed = rawValue.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { throw DomainError.emptyDataSourceOrigin }
        guard trimmed == trimmed.uppercased() else { throw DomainError.invalidDataSourceOrigin(rawValue) }
        guard trimmed.allSatisfy({ $0.isUppercase || $0.isNumber || $0 == "_" || $0 == "-" }) else {
            throw DomainError.invalidDataSourceOrigin(rawValue)
        }
        self.rawValue = trimmed
    }

    public init?(rawValue: String) {
        do {
            try self.init(rawValue)
        } catch {
            return nil
        }
    }

    private init(rawValueUnchecked: String) {
        self.rawValue = rawValueUnchecked
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        try self.init(try container.decode(String.self))
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        try container.encode(rawValue)
    }

    public var description: String { rawValue }

    public static func < (lhs: DataSourceOrigin, rhs: DataSourceOrigin) -> Bool {
        lhs.rawValue < rhs.rawValue
    }
}
