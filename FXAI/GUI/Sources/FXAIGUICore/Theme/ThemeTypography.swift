import SwiftUI

public struct ThemeFontToken {
    public let size: CGFloat
    public let weight: Font.Weight
    public let tracking: CGFloat
    public let opacity: Double
    public let lineSpacing: CGFloat

    public init(size: CGFloat, weight: Font.Weight, tracking: CGFloat = 0, opacity: Double = 1, lineSpacing: CGFloat = 0) {
        self.size = size
        self.weight = weight
        self.tracking = tracking
        self.opacity = opacity
        self.lineSpacing = lineSpacing
    }

    public func font(scaledBy scale: CGFloat = 1) -> Font {
        .system(size: size * scale, weight: weight, design: .rounded)
    }
}

public struct ThemeTypography {
    public let headerTitle: ThemeFontToken
    public let headerSubtitle: ThemeFontToken
    public let sectionTitle: ThemeFontToken
    public let cardTitle: ThemeFontToken
    public let kpiValueMajor: ThemeFontToken
    public let kpiValueMinor: ThemeFontToken
    public let bodyValue: ThemeFontToken
    public let caption: ThemeFontToken
    public let ringValue: ThemeFontToken
    public let footer: ThemeFontToken
    public let tooltip: ThemeFontToken

    public init(
        headerTitle: ThemeFontToken,
        headerSubtitle: ThemeFontToken,
        sectionTitle: ThemeFontToken,
        cardTitle: ThemeFontToken,
        kpiValueMajor: ThemeFontToken,
        kpiValueMinor: ThemeFontToken,
        bodyValue: ThemeFontToken,
        caption: ThemeFontToken,
        ringValue: ThemeFontToken,
        footer: ThemeFontToken,
        tooltip: ThemeFontToken
    ) {
        self.headerTitle = headerTitle
        self.headerSubtitle = headerSubtitle
        self.sectionTitle = sectionTitle
        self.cardTitle = cardTitle
        self.kpiValueMajor = kpiValueMajor
        self.kpiValueMinor = kpiValueMinor
        self.bodyValue = bodyValue
        self.caption = caption
        self.ringValue = ringValue
        self.footer = footer
        self.tooltip = tooltip
    }
}
