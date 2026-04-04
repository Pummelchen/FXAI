import SwiftUI

public struct GlowOrb: Identifiable, Hashable {
    public let id = UUID()
    public let color: Color
    public let size: CGSize
    public let offset: CGSize
    public let blur: CGFloat
    public let opacity: Double

    public init(color: Color, size: CGSize, offset: CGSize, blur: CGFloat, opacity: Double) {
        self.color = color
        self.size = size
        self.offset = offset
        self.blur = blur
        self.opacity = opacity
    }
}

public struct GlowStack {
    public let orbs: [GlowOrb]

    public init(_ orbs: [GlowOrb]) {
        self.orbs = orbs
    }
}

public struct ThemeGlows {
    public let pendingCard: GlowStack
    public let amountOwed: GlowStack
    public let ambientStage: GlowStack

    public init(pendingCard: GlowStack, amountOwed: GlowStack, ambientStage: GlowStack) {
        self.pendingCard = pendingCard
        self.amountOwed = amountOwed
        self.ambientStage = ambientStage
    }
}
