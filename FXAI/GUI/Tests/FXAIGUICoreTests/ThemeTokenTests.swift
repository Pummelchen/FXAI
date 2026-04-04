import FXAIGUICore
import Testing

struct ThemeTokenTests {
    @Test
    func financialDashboardThemeV1ExposesSemanticTokenSets() {
        let theme = FinancialDashboardThemeV1()

        #expect(theme.themeID == .financialDashboardV1)
        #expect(theme.spacing.cardPadding == 22)
        #expect(theme.spacing.ultraWideMaxContentWidth == 5400)
        #expect(theme.components.kpiCard.ringSize == 54)
        #expect(theme.components.trendRing.lineWidth == 5.4)
        #expect(theme.renderingPolicy.policy(for: .amountOwedGlassCard).preferredTier == .coreAnimation)
        #expect(theme.renderingPolicy.chartMinimumReadableWidth == 420)
        #expect(theme.renderingPolicy.compactGlowReductionThreshold == 0.86)
    }

    @Test
    func themeTypographyAndLayoutClampsAreBounded() {
        let theme = FinancialDashboardThemeV1()
        let policy = DashboardScalePolicy.themeDefault(for: theme)

        #expect(policy.minimumScale == theme.layoutMetrics.minimumScale)
        #expect(policy.maximumScale == theme.layoutMetrics.maximumScale)
        #expect(policy.minimumTypographyScale <= policy.maximumTypographyScale)
        #expect(policy.minimumSpacingScale <= policy.maximumSpacingScale)
    }
}
