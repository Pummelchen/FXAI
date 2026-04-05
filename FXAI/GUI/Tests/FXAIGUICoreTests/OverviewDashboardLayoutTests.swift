import FXAIGUICore
import Testing

struct OverviewDashboardLayoutTests {
    @Test
    func defaultOverviewLayoutIncludesEverySectionAndRespectsMinimumGridUnit() {
        let layout = OverviewDashboardLayoutState.default()

        #expect(layout.sections.count == OverviewDashboardSectionKind.allCases.count)
        #expect(layout.gridUnitPoints >= 40)
        #expect(layout.sections.contains(where: { $0.kind == .hero && $0.widgets.contains(where: { $0.kind == .heroSummary }) }))
    }

    @Test
    func normalizationRestoresMissingSectionsAndClampsWidgetSizes() throws {
        let oversized = OverviewDashboardWidgetLayout(
            kind: .buildTargetsMetric,
            widthUnits: 99,
            heightUnits: 99
        )
        let layout = OverviewDashboardLayoutState(
            gridUnitPoints: 8,
            sections: [
                OverviewDashboardSectionLayout(kind: .metrics, widgets: [oversized])
            ]
        ).normalized()

        let spec = OverviewDashboardLayoutState.spec(for: .buildTargetsMetric)
        let metrics = try #require(layout.sections.first(where: { $0.kind == .metrics }))
        let widget = try #require(metrics.widgets.first(where: { $0.kind == .buildTargetsMetric }))

        #expect(layout.gridUnitPoints >= 40)
        #expect(layout.sections.count == OverviewDashboardSectionKind.allCases.count)
        #expect(widget.widthUnits == spec.maximumWidthUnits)
        #expect(widget.heightUnits == spec.maximumHeightUnits)
    }

    @Test
    func gridPlannerNeverDropsBelowOneCentimeterLikeUnit() {
        let widgets = [
            OverviewDashboardWidgetLayout(kind: .buildTargetsMetric),
            OverviewDashboardWidgetLayout(kind: .pluginsMetric),
            OverviewDashboardWidgetLayout(kind: .artifactsMetric)
        ]

        let plan = OverviewDashboardGridPlanner.plan(
            availableWidth: 460,
            widgets: widgets,
            baseGridUnitPoints: 12
        )

        #expect(plan.unitPoints >= OverviewDashboardGridPlanner.minimumGridUnitPoints)
        #expect(plan.columnCount >= 1)
        #expect(plan.contentHeight > 0)
    }
}
