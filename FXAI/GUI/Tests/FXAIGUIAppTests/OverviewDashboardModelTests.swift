import FXAIGUICore
@testable import FXAIGUIApp
import Testing

@MainActor
struct OverviewDashboardModelTests {
    @Test
    func modelCanResizeMoveAndResetOverviewWidgets() throws {
        let model = FXAIGUIModel.validationFixture()
        let baseline = model.overviewLayout
        let section = try #require(model.overviewLayout.sections.first(where: { $0.kind == .metrics }))
        let widget = try #require(section.widgets.first)

        model.resizeOverviewWidget(sectionID: section.id, widgetID: widget.id, widthDelta: 1, heightDelta: 1)
        let resizedSection = try #require(model.overviewLayout.sections.first(where: { $0.id == section.id }))
        let resizedWidget = try #require(resizedSection.widgets.first(where: { $0.id == widget.id }))
        #expect(resizedWidget.widthUnits >= widget.widthUnits)
        #expect(resizedWidget.heightUnits >= widget.heightUnits)

        model.moveOverviewWidget(sectionID: section.id, widgetID: widget.id, by: 1)
        let reorderedSection = try #require(model.overviewLayout.sections.first(where: { $0.id == section.id }))
        #expect(reorderedSection.widgets.first?.id != widget.id || reorderedSection.widgets.count == 1)

        model.resetOverviewLayout()
        #expect(model.overviewLayout.sections.map(\.kind) == baseline.sections.map(\.kind))
        #expect(
            model.overviewLayout.sections.map { $0.widgets.map(\.kind) } ==
            baseline.sections.map { $0.widgets.map(\.kind) }
        )
        #expect(
            model.overviewLayout.sections.map { $0.widgets.map(\.widthUnits) } ==
            baseline.sections.map { $0.widgets.map(\.widthUnits) }
        )
        #expect(
            model.overviewLayout.sections.map { $0.widgets.map(\.heightUnits) } ==
            baseline.sections.map { $0.widgets.map(\.heightUnits) }
        )
    }
}
