import CoreGraphics

public struct ThemeSpacing {
    public let shellInset: CGFloat
    public let cardPadding: CGFloat
    public let compactCardPadding: CGFloat
    public let stackGap: CGFloat
    public let compactVerticalGap: CGFloat
    public let footerHorizontalInset: CGFloat
    public let ultraWideMaxContentWidth: CGFloat

    public init(
        shellInset: CGFloat,
        cardPadding: CGFloat,
        compactCardPadding: CGFloat,
        stackGap: CGFloat,
        compactVerticalGap: CGFloat,
        footerHorizontalInset: CGFloat,
        ultraWideMaxContentWidth: CGFloat
    ) {
        self.shellInset = shellInset
        self.cardPadding = cardPadding
        self.compactCardPadding = compactCardPadding
        self.stackGap = stackGap
        self.compactVerticalGap = compactVerticalGap
        self.footerHorizontalInset = footerHorizontalInset
        self.ultraWideMaxContentWidth = ultraWideMaxContentWidth
    }
}
