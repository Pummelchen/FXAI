import SwiftUI

public struct ThemeColors {
    public let outerBackground: Color
    public let outerVignette: Color
    public let mainPanel: Color
    public let sidebar: Color
    public let footer: Color
    public let divider: Color
    public let cardBase: Color
    public let cardHighlight: Color
    public let cardStroke: Color
    public let glassFillStart: Color
    public let glassFillEnd: Color
    public let glassBorderStart: Color
    public let glassBorderEnd: Color
    public let textPrimary: Color
    public let textSecondary: Color
    public let textMuted: Color
    public let successGreen: Color
    public let warningGreen: Color
    public let chartOlive: Color
    public let chartGreen: Color
    public let chartForest: Color
    public let chartLime: Color
    public let tooltipBackground: Color
    public let tooltipText: Color
    public let debugOutline: Color
    public let debugGuide: Color

    public init(
        outerBackground: Color,
        outerVignette: Color,
        mainPanel: Color,
        sidebar: Color,
        footer: Color,
        divider: Color,
        cardBase: Color,
        cardHighlight: Color,
        cardStroke: Color,
        glassFillStart: Color,
        glassFillEnd: Color,
        glassBorderStart: Color,
        glassBorderEnd: Color,
        textPrimary: Color,
        textSecondary: Color,
        textMuted: Color,
        successGreen: Color,
        warningGreen: Color,
        chartOlive: Color,
        chartGreen: Color,
        chartForest: Color,
        chartLime: Color,
        tooltipBackground: Color,
        tooltipText: Color,
        debugOutline: Color,
        debugGuide: Color
    ) {
        self.outerBackground = outerBackground
        self.outerVignette = outerVignette
        self.mainPanel = mainPanel
        self.sidebar = sidebar
        self.footer = footer
        self.divider = divider
        self.cardBase = cardBase
        self.cardHighlight = cardHighlight
        self.cardStroke = cardStroke
        self.glassFillStart = glassFillStart
        self.glassFillEnd = glassFillEnd
        self.glassBorderStart = glassBorderStart
        self.glassBorderEnd = glassBorderEnd
        self.textPrimary = textPrimary
        self.textSecondary = textSecondary
        self.textMuted = textMuted
        self.successGreen = successGreen
        self.warningGreen = warningGreen
        self.chartOlive = chartOlive
        self.chartGreen = chartGreen
        self.chartForest = chartForest
        self.chartLime = chartLime
        self.tooltipBackground = tooltipBackground
        self.tooltipText = tooltipText
        self.debugOutline = debugOutline
        self.debugGuide = debugGuide
    }
}

public extension Color {
    init(hex: UInt32, opacity: Double = 1.0) {
        let red = Double((hex >> 16) & 0xFF) / 255.0
        let green = Double((hex >> 8) & 0xFF) / 255.0
        let blue = Double(hex & 0xFF) / 255.0
        self.init(.sRGB, red: red, green: green, blue: blue, opacity: opacity)
    }
}
