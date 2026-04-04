import CoreGraphics
import FXAIGUICore
import Testing

struct DashboardResponsivePolicyTests {
    private let theme = FinancialDashboardThemeV1()

    @Test
    func reducedEffectsSuppressAmbientDecorationsFirst() {
        let output = DashboardLayoutEngine.makeLayout(
            input: DashboardLayoutInput(
                windowSize: CGSize(width: 1400, height: 930),
                effectiveContentSize: CGSize(width: 1400, height: 930),
                backingScaleFactor: 2,
                theme: theme,
                contentPriorities: DashboardAdaptiveRules.defaultContentPriorities,
                overlapPolicy: .preserveFloatingCard,
                scalePolicy: .themeDefault(for: theme),
                reducedEffects: true
            )
        )

        #expect(output.reducedDecorativeGlow)
        #expect(output.hiddenZones.contains(.ambientDecorations))
        #expect(output.frameModel.decorativeVisibility.hideSecondaryDecorations)
    }

    @Test
    func ultrawideLayoutsStayAnchoredWithoutStretchingPastThemeBand() {
        let output = DashboardLayoutEngine.makeLayout(
            input: DashboardLayoutInput(
                windowSize: CGSize(width: 5120, height: 2160),
                effectiveContentSize: CGSize(width: 5120, height: 2160),
                backingScaleFactor: 2,
                theme: theme,
                contentPriorities: DashboardAdaptiveRules.defaultContentPriorities,
                overlapPolicy: .preserveFloatingCard,
                scalePolicy: .themeDefault(for: theme),
                reducedEffects: false
            )
        )

        #expect(output.frameModel.layoutClass == .ultraWideDesktop)
        #expect(output.frameModel.stageFrame.width <= theme.spacing.ultraWideMaxContentWidth)
        #expect(output.typographyScale <= DashboardScalePolicy.themeDefault(for: theme).maximumTypographyScale)
        #expect(output.spacingScale <= DashboardScalePolicy.themeDefault(for: theme).maximumSpacingScale)
    }
}
