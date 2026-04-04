import Foundation
import SwiftUI

public protocol AppTheme {
    var id: String { get }
    var displayName: String { get }
    var colors: ThemeColors { get }
    var gradients: ThemeGradients { get }
    var shadows: ThemeShadows { get }
    var glows: ThemeGlows { get }
    var cornerRadii: ThemeCornerRadii { get }
    var typography: ThemeTypography { get }
    var layoutMetrics: ThemeLayoutMetrics { get }
    var materials: ThemeMaterials { get }
    var chartStyle: ThemeChartStyle { get }
}

@MainActor
public final class ThemeManager: ObservableObject {
    public static let shared = ThemeManager()

    @Published public private(set) var currentTheme: any AppTheme

    private let registry: [String: any AppTheme]

    public init(defaultTheme: any AppTheme = FinancialDashboardThemeV1()) {
        let primary = defaultTheme
        let themes: [String: any AppTheme] = [
            primary.id: primary,
            FinancialDashboardThemeV1.themeID: FinancialDashboardThemeV1()
        ]
        registry = themes
        currentTheme = primary
    }

    public var allThemes: [any AppTheme] {
        registry.values.sorted { $0.displayName < $1.displayName }
    }

    public func activateTheme(id: String) {
        guard let theme = registry[id] else { return }
        currentTheme = theme
    }
}
