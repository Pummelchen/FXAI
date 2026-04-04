import CoreGraphics

public struct ThemeLayoutMetrics {
    public let referenceCanvasSize: CGSize
    public let minimumScale: CGFloat
    public let maximumScale: CGFloat
    public let outerPadding: CGFloat
    public let wideMaxWidth: CGFloat
    public let compactCardGap: CGFloat
    public let standardCardPadding: CGFloat
    public let chartLabelOffset: CGFloat
    public let footerHeight: CGFloat
    public let decorativeFadeThreshold: CGFloat
    public let ultrawideBreathingRoom: CGFloat

    public init(
        referenceCanvasSize: CGSize,
        minimumScale: CGFloat,
        maximumScale: CGFloat,
        outerPadding: CGFloat,
        wideMaxWidth: CGFloat,
        compactCardGap: CGFloat,
        standardCardPadding: CGFloat,
        chartLabelOffset: CGFloat,
        footerHeight: CGFloat,
        decorativeFadeThreshold: CGFloat,
        ultrawideBreathingRoom: CGFloat
    ) {
        self.referenceCanvasSize = referenceCanvasSize
        self.minimumScale = minimumScale
        self.maximumScale = maximumScale
        self.outerPadding = outerPadding
        self.wideMaxWidth = wideMaxWidth
        self.compactCardGap = compactCardGap
        self.standardCardPadding = standardCardPadding
        self.chartLabelOffset = chartLabelOffset
        self.footerHeight = footerHeight
        self.decorativeFadeThreshold = decorativeFadeThreshold
        self.ultrawideBreathingRoom = ultrawideBreathingRoom
    }
}
