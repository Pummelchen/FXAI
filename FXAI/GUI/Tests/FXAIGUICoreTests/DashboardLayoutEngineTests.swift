import CoreGraphics
import FXAIGUICore
import Testing

struct DashboardLayoutEngineTests {
    private let theme = FinancialDashboardThemeV1()

    @Test
    func standardDesktopStaysCloseToReferenceComposition() {
        let output = DashboardLayoutEngine.makeLayout(
            input: DashboardLayoutInput(
                windowSize: CGSize(width: 1728, height: 1117),
                effectiveContentSize: CGSize(width: 1728, height: 1117),
                backingScaleFactor: 2,
                theme: theme,
                contentPriorities: DashboardAdaptiveRules.defaultContentPriorities,
                overlapPolicy: .preserveFloatingCard,
                scalePolicy: .themeDefault(for: theme),
                reducedEffects: false
            )
        )

        #expect(output.frameModel.layoutClass == .standardDesktop)
        #expect(output.kpiArrangement == .singleRow)
        #expect(output.chartPlacement == .anchoredRight)
        #expect(output.hiddenZones.isEmpty)
    }

    @Test
    func compactDesktopReflowsIntoGridAndChartBelow() {
        let output = DashboardLayoutEngine.makeLayout(
            input: DashboardLayoutInput(
                windowSize: CGSize(width: 1320, height: 960),
                effectiveContentSize: CGSize(width: 1320, height: 960),
                backingScaleFactor: 2,
                theme: theme,
                contentPriorities: DashboardAdaptiveRules.defaultContentPriorities,
                overlapPolicy: .preserveFloatingCard,
                scalePolicy: .themeDefault(for: theme),
                reducedEffects: false
            )
        )

        let ready = output.frameModel.topCardFrames[.readyToAssign]!
        let pending = output.frameModel.topCardFrames[.pendingSignOffs]!
        let declined = output.frameModel.topCardFrames[.declined]!

        #expect(output.frameModel.layoutClass == .compactDesktop)
        #expect(output.kpiArrangement == .gridTwoByTwo)
        #expect(output.chartPlacement == .belowInvoices)
        #expect(ready.minY == pending.minY)
        #expect(declined.minY > ready.maxY)
    }
}
