import SwiftUI

public struct GlowRenderer: View {
    public let glow: GlowStack
    public let scale: CGFloat
    public let intensity: CGFloat

    public init(glow: GlowStack, scale: CGFloat, intensity: CGFloat) {
        self.glow = glow
        self.scale = scale
        self.intensity = intensity
    }

    public var body: some View {
        ZStack {
            ForEach(Array(glow.orbs.enumerated()), id: \.offset) { item in
                let orb = item.element
                Ellipse()
                    .fill(orb.color.opacity(orb.opacity * intensity))
                    .frame(width: orb.size.width * scale, height: orb.size.height * scale)
                    .blur(radius: orb.blur * scale)
                    .offset(x: orb.offset.width * scale, y: orb.offset.height * scale)
            }
        }
        .allowsHitTesting(false)
    }
}
