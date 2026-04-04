import SwiftUI

public struct GlassRenderer: View {
    public let size: CGSize
    public let cornerRadius: CGFloat
    public let theme: any AppTheme
    public let scale: CGFloat

    public init(size: CGSize, cornerRadius: CGFloat, theme: any AppTheme, scale: CGFloat) {
        self.size = size
        self.cornerRadius = cornerRadius
        self.theme = theme
        self.scale = scale
    }

    public var body: some View {
        ZStack {
            RoundedRectangle(cornerRadius: cornerRadius, style: .continuous)
                .fill(theme.gradients.glassCard)
                .opacity(theme.materials.glassOpacity)

            RoundedRectangle(cornerRadius: cornerRadius, style: .continuous)
                .fill(.white.opacity(0.06))
                .blur(radius: 18 * scale)
                .mask(
                    RoundedRectangle(cornerRadius: cornerRadius, style: .continuous)
                        .frame(width: size.width, height: size.height * 0.48)
                        .offset(y: -size.height * 0.22)
                )

            RoundedRectangle(cornerRadius: cornerRadius, style: .continuous)
                .strokeBorder(theme.gradients.glassBorder, lineWidth: 0.92 * scale)
                .opacity(theme.materials.glassStrokeOpacity)
        }
        .frame(width: size.width, height: size.height)
        .compositingGroup()
    }
}
