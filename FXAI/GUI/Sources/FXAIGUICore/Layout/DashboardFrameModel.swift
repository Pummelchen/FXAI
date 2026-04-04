import CoreGraphics

public struct DashboardDecorativeVisibility {
    public let glowIntensity: CGFloat
    public let ambientOpacity: CGFloat
    public let hideSecondaryDecorations: Bool
    public let metalOverlayEnabled: Bool

    public init(glowIntensity: CGFloat, ambientOpacity: CGFloat, hideSecondaryDecorations: Bool, metalOverlayEnabled: Bool) {
        self.glowIntensity = glowIntensity
        self.ambientOpacity = ambientOpacity
        self.hideSecondaryDecorations = hideSecondaryDecorations
        self.metalOverlayEnabled = metalOverlayEnabled
    }
}

public struct DashboardFrameModel {
    public let containerSize: CGSize
    public let stageFrame: CGRect
    public let layoutClass: DashboardLayoutClass
    public let scale: CGFloat
    public let mainPanelFrame: CGRect
    public let footerFrame: CGRect
    public let headerDividerFrame: CGRect
    public let headerTitleOrigin: CGPoint
    public let headerSubtitleOrigin: CGPoint
    public let billsOrigin: CGPoint
    public let invoicesOrigin: CGPoint
    public let topCardFrames: [SVGKPIKind: CGRect]
    public let gaugeFrame: CGRect
    public let invoiceMetricFrames: [SVGInvoiceCardKind: CGRect]
    public let amountOwedFrame: CGRect
    public let chartPlotFrame: CGRect
    public let chartBars: [SVGChartMonth: CGRect]
    public let tooltipFrame: CGRect
    public let footerDateCenter: CGPoint
    public let footerTimeCenter: CGPoint
    public let footerDayCenter: CGPoint
    public let decorativeVisibility: DashboardDecorativeVisibility
}
