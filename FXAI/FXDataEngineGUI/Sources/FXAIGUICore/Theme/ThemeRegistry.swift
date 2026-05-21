import Foundation

public struct ThemeRegistry {
    private let themesByID: [ThemeID: any AppTheme]

    public init(themes: [any AppTheme]) {
        var mapping: [ThemeID: any AppTheme] = [:]
        for theme in themes {
            mapping[theme.themeID] = theme
        }
        themesByID = mapping
    }

    public var allThemes: [any AppTheme] {
        themesByID.values.sorted { $0.displayName < $1.displayName }
    }

    public var defaultTheme: (any AppTheme)? {
        theme(for: .financialDashboardV1) ?? allThemes.first
    }

    public func theme(for id: ThemeID) -> (any AppTheme)? {
        themesByID[id]
    }

    public func contains(_ id: ThemeID) -> Bool {
        themesByID[id] != nil
    }
}
