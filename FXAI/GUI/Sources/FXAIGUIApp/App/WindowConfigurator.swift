import AppKit
import SwiftUI

struct WindowConfigurator: NSViewRepresentable {
    func makeNSView(context: Context) -> NSView {
        let view = NSView()
        DispatchQueue.main.async {
            guard let window = view.window else { return }
            window.titleVisibility = .hidden
            window.titlebarAppearsTransparent = true
            window.isOpaque = false
            window.backgroundColor = .clear
            window.toolbarStyle = .unifiedCompact
            window.standardWindowButton(.zoomButton)?.isHidden = false
        }
        return view
    }

    func updateNSView(_ nsView: NSView, context: Context) {}
}
