import CoreGraphics

public struct DashboardScalePolicy: Sendable {
    public let minimumScale: CGFloat
    public let maximumScale: CGFloat
    public let minimumTypographyScale: CGFloat
    public let maximumTypographyScale: CGFloat
    public let minimumSpacingScale: CGFloat
    public let maximumSpacingScale: CGFloat

    public init(
        minimumScale: CGFloat,
        maximumScale: CGFloat,
        minimumTypographyScale: CGFloat,
        maximumTypographyScale: CGFloat,
        minimumSpacingScale: CGFloat,
        maximumSpacingScale: CGFloat
    ) {
        self.minimumScale = minimumScale
        self.maximumScale = maximumScale
        self.minimumTypographyScale = minimumTypographyScale
        self.maximumTypographyScale = maximumTypographyScale
        self.minimumSpacingScale = minimumSpacingScale
        self.maximumSpacingScale = maximumSpacingScale
    }

    public static func themeDefault(for theme: any AppTheme) -> DashboardScalePolicy {
        DashboardScalePolicy(
            minimumScale: theme.layoutMetrics.minimumScale,
            maximumScale: theme.layoutMetrics.maximumScale,
            minimumTypographyScale: max(0.82, theme.layoutMetrics.minimumScale),
            maximumTypographyScale: min(3.0, theme.layoutMetrics.maximumScale),
            minimumSpacingScale: max(0.84, theme.layoutMetrics.minimumScale),
            maximumSpacingScale: min(2.4, theme.layoutMetrics.maximumScale)
        )
    }
}
