import SwiftUI

public struct ShadowStackRenderer: View {
    public let size: CGSize
    public let cornerRadius: CGFloat
    public let shadow: ShadowStack
    public let scale: CGFloat
    public let context: ShadowProjectionContext?

    public init(
        size: CGSize,
        cornerRadius: CGFloat,
        shadow: ShadowStack,
        scale: CGFloat,
        context: ShadowProjectionContext? = nil
    ) {
        self.size = size
        self.cornerRadius = cornerRadius
        self.shadow = shadow
        self.scale = scale
        self.context = context
    }

    public var body: some View {
        ZStack {
            ForEach(Array(ShadowProjector.resolve(stack: shadow, context: context, scale: scale).enumerated()), id: \.offset) { item in
                let layer = item.element
                RoundedRectangle(cornerRadius: cornerRadius, style: .continuous)
                    .fill(layer.color.opacity(layer.opacity))
                    .frame(width: size.width, height: size.height)
                    .blur(radius: layer.blurRadius)
                    .offset(x: layer.offset.width, y: layer.offset.height)
            }
        }
        .frame(width: size.width, height: size.height)
        .allowsHitTesting(false)
    }
}
