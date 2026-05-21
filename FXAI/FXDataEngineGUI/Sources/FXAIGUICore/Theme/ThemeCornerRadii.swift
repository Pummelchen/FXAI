import CoreGraphics

public struct ThemeCornerRadii {
    public let panel: CGFloat
    public let standardCard: CGFloat
    public let smallCard: CGFloat
    public let glassCard: CGFloat
    public let bar: CGFloat
    public let tooltip: CGFloat

    public init(panel: CGFloat, standardCard: CGFloat, smallCard: CGFloat, glassCard: CGFloat, bar: CGFloat, tooltip: CGFloat) {
        self.panel = panel
        self.standardCard = standardCard
        self.smallCard = smallCard
        self.glassCard = glassCard
        self.bar = bar
        self.tooltip = tooltip
    }
}
