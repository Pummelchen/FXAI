import FXAIGUICore
import Testing

struct ThemeRegistryTests {
    @Test
    func registryResolvesRegisteredThemeByID() throws {
        let theme = FinancialDashboardThemeV1()
        let registry = ThemeRegistry(themes: [theme])

        let resolved = try #require(registry.theme(for: .financialDashboardV1))
        #expect(resolved.themeID == .financialDashboardV1)
        #expect(resolved.displayName == "FXAI Operator Theme")
        #expect(registry.contains(.financialDashboardV1))
    }

    @MainActor
    @Test
    func themeEnvironmentActivatesThemeFromRegistry() {
        let environment = ThemeEnvironment.preview()

        #expect(environment.selectedThemeID == .financialDashboardV1)
        #expect(environment.currentTheme.themeID == .financialDashboardV1)
        #expect(environment === ThemeEnvironment.shared)

        environment.activateTheme(.financialDashboardV1)
        #expect(environment.currentTheme.displayName == "FXAI Operator Theme")
    }
}
