import FXAIGUICore

@MainActor
enum ThemeBootstrap {
    static func makeThemeEnvironment() -> ThemeEnvironment {
        let environment = ThemeEnvironment.shared
        environment.configure(
            registry: ThemeRegistry(themes: [
                FinancialDashboardThemeV1()
            ]),
            initialThemeID: .financialDashboardV1
        )
        return environment
    }
}
