import Foundation

public struct ThemeRenderingPolicy {
    public let policies: [DashboardStyledComponent: ComponentRenderingPolicy]
    public let compactGlowReductionThreshold: CGFloat
    public let chartMinimumReadableHeight: CGFloat
    public let chartMinimumReadableWidth: CGFloat

    public init(
        policies: [DashboardStyledComponent: ComponentRenderingPolicy],
        compactGlowReductionThreshold: CGFloat,
        chartMinimumReadableHeight: CGFloat,
        chartMinimumReadableWidth: CGFloat
    ) {
        self.policies = policies
        self.compactGlowReductionThreshold = compactGlowReductionThreshold
        self.chartMinimumReadableHeight = chartMinimumReadableHeight
        self.chartMinimumReadableWidth = chartMinimumReadableWidth
    }

    public func policy(for component: DashboardStyledComponent) -> ComponentRenderingPolicy {
        policies[component] ?? ComponentRenderingPolicy(
            component: component,
            preferredTier: .swiftUI,
            fallbackTier: .swiftUI,
            capabilities: []
        )
    }
}
