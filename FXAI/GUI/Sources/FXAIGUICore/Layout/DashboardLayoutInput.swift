import CoreGraphics

public struct DashboardLayoutInput {
    public let windowSize: CGSize
    public let effectiveContentSize: CGSize
    public let backingScaleFactor: CGFloat
    public let theme: any AppTheme
    public let contentPriorities: [DashboardZone: DashboardContentPriority]
    public let overlapPolicy: DashboardOverlapPolicy
    public let scalePolicy: DashboardScalePolicy
    public let reducedEffects: Bool

    public init(
        windowSize: CGSize,
        effectiveContentSize: CGSize,
        backingScaleFactor: CGFloat,
        theme: any AppTheme,
        contentPriorities: [DashboardZone: DashboardContentPriority],
        overlapPolicy: DashboardOverlapPolicy,
        scalePolicy: DashboardScalePolicy,
        reducedEffects: Bool
    ) {
        self.windowSize = windowSize
        self.effectiveContentSize = effectiveContentSize
        self.backingScaleFactor = backingScaleFactor
        self.theme = theme
        self.contentPriorities = contentPriorities
        self.overlapPolicy = overlapPolicy
        self.scalePolicy = scalePolicy
        self.reducedEffects = reducedEffects
    }
}
