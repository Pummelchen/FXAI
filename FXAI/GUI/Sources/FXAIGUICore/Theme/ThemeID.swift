import Foundation

public struct ThemeID: RawRepresentable, Hashable, Codable, Sendable, ExpressibleByStringLiteral {
    public static let financialDashboardV1: ThemeID = "FinancialDashboardThemeV1"

    public let rawValue: String

    public init(rawValue: String) {
        self.rawValue = rawValue
    }

    public init(stringLiteral value: StringLiteralType) {
        self.rawValue = value
    }
}
