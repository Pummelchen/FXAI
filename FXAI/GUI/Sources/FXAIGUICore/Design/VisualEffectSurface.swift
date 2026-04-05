import AppKit
import SwiftUI

public enum FXAIGlassSurfaceStyle: Sendable {
    case card
    case sidebar
    case hero
    case badge

    fileprivate var material: NSVisualEffectView.Material {
        switch self {
        case .card:
            .hudWindow
        case .sidebar:
            .sidebar
        case .hero:
            .menu
        case .badge:
            .hudWindow
        }
    }

    fileprivate var baseOpacity: Double {
        switch self {
        case .card:
            0.78
        case .sidebar:
            0.64
        case .hero:
            0.70
        case .badge:
            0.52
        }
    }

    fileprivate var topHighlightOpacity: Double {
        switch self {
        case .card:
            0.16
        case .sidebar:
            0.12
        case .hero:
            0.20
        case .badge:
            0.12
        }
    }

    fileprivate var bottomShadeOpacity: Double {
        switch self {
        case .card:
            0.18
        case .sidebar:
            0.14
        case .hero:
            0.16
        case .badge:
            0.12
        }
    }

    fileprivate var borderOpacity: Double {
        switch self {
        case .card:
            0.62
        case .sidebar:
            0.48
        case .hero:
            0.74
        case .badge:
            0.42
        }
    }

    fileprivate var highlightStrokeOpacity: Double {
        switch self {
        case .card:
            0.26
        case .sidebar:
            0.18
        case .hero:
            0.34
        case .badge:
            0.18
        }
    }

    fileprivate var shadowOpacity: Double {
        switch self {
        case .card:
            0.20
        case .sidebar:
            0.14
        case .hero:
            0.18
        case .badge:
            0.10
        }
    }

    fileprivate var shadowRadius: CGFloat {
        switch self {
        case .card:
            22
        case .sidebar:
            18
        case .hero:
            28
        case .badge:
            10
        }
    }

    fileprivate var shadowOffset: CGSize {
        switch self {
        case .card:
            CGSize(width: 0, height: 14)
        case .sidebar:
            CGSize(width: 0, height: 10)
        case .hero:
            CGSize(width: 0, height: 18)
        case .badge:
            CGSize(width: 0, height: 4)
        }
    }
}

public struct FXAIGlassRoundedBackground: View {
    @Environment(\.guiRenderingProfile) private var renderingProfile

    public let cornerRadius: CGFloat
    public let style: FXAIGlassSurfaceStyle
    public let tint: Color

    public init(
        cornerRadius: CGFloat = 22,
        style: FXAIGlassSurfaceStyle = .card,
        tint: Color = .clear
    ) {
        self.cornerRadius = cornerRadius
        self.style = style
        self.tint = tint
    }

    public var body: some View {
        let shape = RoundedRectangle(cornerRadius: cornerRadius, style: .continuous)
        let baseOpacity = style.baseOpacity * renderingProfile.glassOpacityScale
        let topHighlightOpacity = style.topHighlightOpacity * renderingProfile.glassOpacityScale
        let bottomShadeOpacity = style.bottomShadeOpacity * renderingProfile.glassOpacityScale
        let borderOpacity = style.borderOpacity * renderingProfile.glassOpacityScale
        let highlightOpacity = style.highlightStrokeOpacity * renderingProfile.glassOpacityScale
        let shadowOpacity = style.shadowOpacity * renderingProfile.shadowOpacityScale
        ZStack {
            if renderingProfile.allowsGlassBackdrop {
                FXAIGlassBackdrop(material: style.material)
                    .clipShape(shape)
            }

            shape
                .fill(
                    LinearGradient(
                        colors: [
                            FXAITheme.panelStrong.opacity(baseOpacity),
                            FXAITheme.panel.opacity(baseOpacity * 0.88)
                        ],
                        startPoint: .topLeading,
                        endPoint: .bottomTrailing
                    )
                )

            shape.fill(tint.opacity(0.12 * renderingProfile.decorativeOpacityScale))

            shape
                .fill(
                    LinearGradient(
                        colors: [
                            .white.opacity(topHighlightOpacity),
                            .white.opacity(0.03),
                            .clear
                        ],
                        startPoint: .top,
                        endPoint: .center
                    )
                )

            shape
                .fill(
                    LinearGradient(
                        colors: [
                            .clear,
                            .black.opacity(bottomShadeOpacity)
                        ],
                        startPoint: .center,
                        endPoint: .bottom
                    )
                )

            shape.strokeBorder(FXAITheme.stroke.opacity(borderOpacity), lineWidth: 1)

            shape
                .strokeBorder(.white.opacity(highlightOpacity), lineWidth: 0.5)
                .blur(radius: 0.4)
                .blendMode(.screen)

            shape
                .inset(by: 1.4)
                .strokeBorder(
                    LinearGradient(
                        colors: [
                            .white.opacity(highlightOpacity * 0.72),
                            .clear,
                            .black.opacity(0.10)
                        ],
                        startPoint: .topLeading,
                        endPoint: .bottomTrailing
                    ),
                    lineWidth: 0.8
                )
        }
        .shadow(
            color: .black.opacity(shadowOpacity),
            radius: style.shadowRadius * CGFloat(renderingProfile.backgroundBlurScale),
            x: style.shadowOffset.width,
            y: style.shadowOffset.height
        )
    }
}

public struct FXAIGlassCapsuleBackground: View {
    @Environment(\.guiRenderingProfile) private var renderingProfile

    public let style: FXAIGlassSurfaceStyle
    public let tint: Color

    public init(style: FXAIGlassSurfaceStyle = .badge, tint: Color = .clear) {
        self.style = style
        self.tint = tint
    }

    public var body: some View {
        let shape = Capsule(style: .continuous)
        let baseOpacity = style.baseOpacity * renderingProfile.glassOpacityScale
        let highlightOpacity = style.highlightStrokeOpacity * renderingProfile.glassOpacityScale
        let borderOpacity = style.borderOpacity * renderingProfile.glassOpacityScale
        let shadowOpacity = style.shadowOpacity * renderingProfile.shadowOpacityScale
        ZStack {
            if renderingProfile.allowsGlassBackdrop {
                FXAIGlassBackdrop(material: style.material)
                    .clipShape(shape)
            }

            shape
                .fill(
                    LinearGradient(
                        colors: [
                            FXAITheme.panelStrong.opacity(baseOpacity * 0.92),
                            FXAITheme.panel.opacity(baseOpacity * 0.72)
                        ],
                        startPoint: .topLeading,
                        endPoint: .bottomTrailing
                    )
                )
            shape.fill(tint.opacity(0.14 * renderingProfile.decorativeOpacityScale))
            shape
                .strokeBorder(.white.opacity(highlightOpacity), lineWidth: 0.6)
                .blendMode(.screen)
            shape
                .strokeBorder(FXAITheme.stroke.opacity(borderOpacity), lineWidth: 0.8)
        }
        .shadow(
            color: .black.opacity(shadowOpacity),
            radius: style.shadowRadius * CGFloat(renderingProfile.backgroundBlurScale),
            x: style.shadowOffset.width,
            y: style.shadowOffset.height
        )
    }
}

public struct FXAIVisualEffectSurface<Content: View>: View {
    private let style: FXAIGlassSurfaceStyle
    private let cornerRadius: CGFloat
    private let contentPadding: CGFloat
    private let tint: Color
    private let content: Content

    public init(
        style: FXAIGlassSurfaceStyle = .card,
        cornerRadius: CGFloat = 22,
        contentPadding: CGFloat = 18,
        tint: Color = .clear,
        @ViewBuilder content: () -> Content
    ) {
        self.style = style
        self.cornerRadius = cornerRadius
        self.contentPadding = contentPadding
        self.tint = tint
        self.content = content()
    }

    public var body: some View {
        ZStack {
            FXAIGlassRoundedBackground(cornerRadius: cornerRadius, style: style, tint: tint)
            content
                .padding(contentPadding)
        }
    }
}

private struct FXAIGlassBackdrop: NSViewRepresentable {
    let material: NSVisualEffectView.Material

    func makeNSView(context: Context) -> NSVisualEffectView {
        let view = NSVisualEffectView()
        view.blendingMode = .withinWindow
        view.material = material
        view.state = .active
        view.isEmphasized = false
        return view
    }

    func updateNSView(_ nsView: NSVisualEffectView, context: Context) {
        nsView.material = material
        nsView.state = .active
    }
}
