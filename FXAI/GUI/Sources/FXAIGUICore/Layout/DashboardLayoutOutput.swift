import CoreGraphics

public enum DashboardKPIArrangement: String, Sendable {
    case singleRow
    case gridTwoByTwo
}

public enum DashboardChartPlacement: String, Sendable {
    case anchoredRight
    case belowInvoices
}

public struct DashboardLayoutOutput {
    public let frameModel: DashboardFrameModel
    public let typographyScale: CGFloat
    public let spacingScale: CGFloat
    public let chartPlacement: DashboardChartPlacement
    public let kpiArrangement: DashboardKPIArrangement
    public let reducedDecorativeGlow: Bool
    public let hiddenZones: Set<DashboardZone>
}
