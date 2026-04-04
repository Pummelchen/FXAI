import AppKit
import SwiftUI

public struct FXAIVisualEffectSurface<Content: View>: View {
    private let content: Content

    public init(@ViewBuilder content: () -> Content) {
        self.content = content()
    }

    public var body: some View {
        ZStack {
            VisualEffectView()
                .clipShape(RoundedRectangle(cornerRadius: 22, style: .continuous))

            RoundedRectangle(cornerRadius: 22, style: .continuous)
                .fill(FXAITheme.panel)

            RoundedRectangle(cornerRadius: 22, style: .continuous)
                .strokeBorder(FXAITheme.stroke, lineWidth: 1)

            content
                .padding(18)
        }
        .shadow(color: .black.opacity(0.28), radius: 18, x: 0, y: 10)
    }
}

private struct VisualEffectView: NSViewRepresentable {
    func makeNSView(context: Context) -> NSVisualEffectView {
        let view = NSVisualEffectView()
        view.blendingMode = .withinWindow
        view.material = .hudWindow
        view.state = .active
        return view
    }

    func updateNSView(_ nsView: NSVisualEffectView, context: Context) {}
}
