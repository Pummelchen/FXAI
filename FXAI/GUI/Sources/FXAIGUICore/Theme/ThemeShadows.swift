import SwiftUI

public struct ShadowLayer: Identifiable, Hashable {
    public let id = UUID()
    public let color: Color
    public let radius: CGFloat
    public let x: CGFloat
    public let y: CGFloat
    public let opacity: Double

    public init(color: Color = .black, radius: CGFloat, x: CGFloat, y: CGFloat, opacity: Double) {
        self.color = color
        self.radius = radius
        self.x = x
        self.y = y
        self.opacity = opacity
    }
}

public struct ShadowStack {
    public let layers: [ShadowLayer]

    public init(_ layers: [ShadowLayer]) {
        self.layers = layers
    }
}

public struct ThemeShadows {
    public let kpiCard: ShadowStack
    public let pendingCard: ShadowStack
    public let smallCard: ShadowStack
    public let gaugeCard: ShadowStack
    public let amountOwed: ShadowStack
    public let chartBarPrimary: ShadowStack
    public let chartBarDefault: ShadowStack
    public let footer: ShadowStack

    public init(
        kpiCard: ShadowStack,
        pendingCard: ShadowStack,
        smallCard: ShadowStack,
        gaugeCard: ShadowStack,
        amountOwed: ShadowStack,
        chartBarPrimary: ShadowStack,
        chartBarDefault: ShadowStack,
        footer: ShadowStack
    ) {
        self.kpiCard = kpiCard
        self.pendingCard = pendingCard
        self.smallCard = smallCard
        self.gaugeCard = gaugeCard
        self.amountOwed = amountOwed
        self.chartBarPrimary = chartBarPrimary
        self.chartBarDefault = chartBarDefault
        self.footer = footer
    }
}
