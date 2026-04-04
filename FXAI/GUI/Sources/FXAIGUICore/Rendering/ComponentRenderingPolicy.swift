import Foundation

public enum DashboardStyledComponent: String, CaseIterable, Sendable {
    case stage
    case sidebar
    case header
    case kpiCard
    case invoiceMetricCard
    case gaugeCard
    case amountOwedGlassCard
    case trendRing
    case barChart
    case footerStrip
    case tooltip
    case calibrationOverlay
}

public struct ComponentRenderingPolicy: Sendable {
    public let component: DashboardStyledComponent
    public let preferredTier: RenderingTier
    public let fallbackTier: RenderingTier
    public let capabilities: RenderCapability

    public init(
        component: DashboardStyledComponent,
        preferredTier: RenderingTier,
        fallbackTier: RenderingTier,
        capabilities: RenderCapability
    ) {
        self.component = component
        self.preferredTier = preferredTier
        self.fallbackTier = fallbackTier
        self.capabilities = capabilities
    }
}
