import SwiftUI

public struct ThemeGradients {
    public let canvas: LinearGradient
    public let standardCard: LinearGradient
    public let pendingCard: LinearGradient
    public let glassCard: LinearGradient
    public let glassBorder: LinearGradient
    public let gaugeLeft: LinearGradient
    public let gaugeRight: LinearGradient
    public let footer: LinearGradient
    public let chartTooltip: LinearGradient

    public init(
        canvas: LinearGradient,
        standardCard: LinearGradient,
        pendingCard: LinearGradient,
        glassCard: LinearGradient,
        glassBorder: LinearGradient,
        gaugeLeft: LinearGradient,
        gaugeRight: LinearGradient,
        footer: LinearGradient,
        chartTooltip: LinearGradient
    ) {
        self.canvas = canvas
        self.standardCard = standardCard
        self.pendingCard = pendingCard
        self.glassCard = glassCard
        self.glassBorder = glassBorder
        self.gaugeLeft = gaugeLeft
        self.gaugeRight = gaugeRight
        self.footer = footer
        self.chartTooltip = chartTooltip
    }
}
