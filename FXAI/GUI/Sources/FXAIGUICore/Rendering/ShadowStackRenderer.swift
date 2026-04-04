import SwiftUI

public struct ShadowStackRenderer: View {
    public let size: CGSize
    public let cornerRadius: CGFloat
    public let shadow: ShadowStack
    public let scale: CGFloat
    public let fillColor: Color

    public init(size: CGSize, cornerRadius: CGFloat, shadow: ShadowStack, scale: CGFloat, fillColor: Color = .black) {
        self.size = size
        self.cornerRadius = cornerRadius
        self.shadow = shadow
        self.scale = scale
        self.fillColor = fillColor
    }

    public var body: some View {
        ZStack {
            ForEach(Array(shadow.layers.enumerated()), id: \.offset) { item in
                let layer = item.element
                RoundedRectangle(cornerRadius: cornerRadius, style: .continuous)
                    .fill(fillColor.opacity(layer.opacity))
                    .frame(width: size.width, height: size.height)
                    .blur(radius: layer.radius * scale)
                    .offset(x: layer.x * scale, y: layer.y * scale)
            }
        }
        .frame(width: size.width, height: size.height)
        .allowsHitTesting(false)
    }
}
