import FXAIGUICore
import Testing

struct DashboardLayoutAuditTests {
    private let theme = FinancialDashboardThemeV1()

    @Test
    func layoutAuditMatrixPassesAcrossReferenceScenarios() {
        for scenario in GUIValidationScenario.layoutAuditMatrix {
            let report = DashboardLayoutAuditor.audit(scenario: scenario, theme: theme)
            #expect(report.passed, "Scenario \(scenario.title) failed audit: \(report.issues.map { $0.summary }.joined(separator: " | "))")
            #expect(report.score >= 94, "Scenario \(scenario.title) score was \(report.score)")
        }
    }

    @Test
    func compactScenarioDeprioritizesDecorationBeforeReadability() {
        let report = DashboardLayoutAuditor.audit(scenario: .compactEdge, theme: theme)
        #expect(report.output.reducedDecorativeGlow)
        #expect(report.output.frameModel.decorativeVisibility.hideSecondaryDecorations)
        #expect(report.output.typographyScale >= DashboardScalePolicy.themeDefault(for: theme).minimumTypographyScale)
        #expect(report.output.frameModel.chartPlotFrame.height >= theme.renderingPolicy.chartMinimumReadableHeight * report.output.spacingScale * 0.82)
    }
}
