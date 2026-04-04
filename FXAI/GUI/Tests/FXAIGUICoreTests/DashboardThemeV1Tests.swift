import CoreGraphics
import FXAIGUICore
import Testing

struct DashboardThemeV1Tests {
    @Test
    func svgReferenceParsesExpectedCanvas() {
        let parsed = SVGParserSupport.parseReferenceAsset()
        #expect(parsed != nil)
        #expect(parsed?.canvasSize == SVGMetrics.canvasSize)
        #expect(parsed?.containsTooltipRect == true)
        #expect(parsed?.containsAugBar == true)
        #expect(SVGParserSupport.validateReferenceAsset().isEmpty)
    }

    @Test
    func layoutEnginePreservesWideReferenceComposition() {
        let theme = FinancialDashboardThemeV1()
        let model = DashboardLayoutEngine.makeFrameModel(
            containerSize: CGSize(width: 1728, height: 1117),
            theme: theme
        )
        let scale = model.scale

        #expect(model.layoutClass == DashboardLayoutClass.standardDesktop)
        #expect(abs(model.topCardFrames[SVGKPIKind.readyToAssign]!.minX - 186 * scale) < 0.001)
        #expect(abs(model.gaugeFrame.height - 457 * scale) < 0.001)
        #expect(abs(model.tooltipFrame.minY - 534 * scale) < 0.001)
    }

    @Test
    func compactLayoutReflowsIntoTwoColumns() throws {
        let theme = FinancialDashboardThemeV1()
        let model = DashboardLayoutEngine.makeFrameModel(
            containerSize: CGSize(width: 1320, height: 960),
            theme: theme
        )

        let ready = try #require(model.topCardFrames[SVGKPIKind.readyToAssign])
        let pending = try #require(model.topCardFrames[SVGKPIKind.pendingSignOffs])
        let declined = try #require(model.topCardFrames[SVGKPIKind.declined])

        #expect(model.layoutClass == DashboardLayoutClass.compactDesktop)
        #expect(ready.minY == pending.minY)
        #expect(declined.minY > ready.maxY)
        #expect(model.chartPlotFrame.minY > model.invoiceMetricFrames[SVGInvoiceCardKind.liveFundUpdate]!.maxY)
    }
}
