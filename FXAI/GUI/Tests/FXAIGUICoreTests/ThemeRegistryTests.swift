import FXAIGUICore
import Testing

struct ThemeRegistryTests {
    @Test
    func registryResolvesRegisteredThemeByID() throws {
        let theme = FinancialDashboardThemeV1()
        let registry = ThemeRegistry(themes: [theme])

        let resolved = try #require(registry.theme(for: .financialDashboardV1))
        #expect(resolved.themeID == .financialDashboardV1)
        #expect(resolved.displayName == "Financial Dashboard Theme V1")
        #expect(registry.contains(.financialDashboardV1))
    }

    @MainActor
    @Test
    func themeEnvironmentActivatesThemeFromRegistry() {
        let environment = ThemeEnvironment.preview()

        #expect(environment.selectedThemeID == .financialDashboardV1)
        #expect(environment.currentTheme.themeID == .financialDashboardV1)

        environment.activateTheme(.financialDashboardV1)
        #expect(environment.currentTheme.displayName == "Financial Dashboard Theme V1")
    }
}
