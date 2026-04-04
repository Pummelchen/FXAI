import SwiftUI

@MainActor
public final class ThemeEnvironment: ObservableObject {
    public static let shared = ThemeEnvironment()

    @Published public private(set) var currentTheme: any AppTheme
    @Published public private(set) var selectedThemeID: ThemeID

    public private(set) var registry: ThemeRegistry
    private let defaultsKey = "FXAI.GUI.SelectedThemeID"

    public init(
        registry: ThemeRegistry = ThemeRegistry(themes: [FinancialDashboardThemeV1()]),
        initialThemeID: ThemeID? = nil
    ) {
        self.registry = registry

        let persisted = UserDefaults.standard.string(forKey: defaultsKey).map(ThemeID.init(rawValue:))
        let resolvedID = initialThemeID ?? persisted ?? .financialDashboardV1
        let fallbackTheme = registry.defaultTheme ?? FinancialDashboardThemeV1()
        let selectedTheme = registry.theme(for: resolvedID) ?? fallbackTheme
        selectedThemeID = selectedTheme.themeID
        currentTheme = selectedTheme
    }

    public var allThemes: [any AppTheme] {
        registry.allThemes
    }

    public func activateTheme(_ id: ThemeID) {
        guard let theme = registry.theme(for: id) else { return }
        selectedThemeID = id
        currentTheme = theme
        UserDefaults.standard.set(id.rawValue, forKey: defaultsKey)
    }

    public func configure(
        registry: ThemeRegistry,
        initialThemeID: ThemeID? = nil
    ) {
        self.registry = registry
        let persisted = UserDefaults.standard.string(forKey: defaultsKey).map(ThemeID.init(rawValue:))
        let resolvedID = initialThemeID ?? persisted ?? registry.defaultTheme?.themeID ?? .financialDashboardV1
        let fallbackTheme = registry.defaultTheme ?? FinancialDashboardThemeV1()
        let selectedTheme = registry.theme(for: resolvedID) ?? fallbackTheme
        selectedThemeID = selectedTheme.themeID
        currentTheme = selectedTheme
    }

    public static func preview(themeID: ThemeID = .financialDashboardV1) -> ThemeEnvironment {
        let environment = ThemeEnvironment.shared
        environment.configure(
            registry: ThemeRegistry(themes: [FinancialDashboardThemeV1()]),
            initialThemeID: themeID
        )
        return environment
    }
}
