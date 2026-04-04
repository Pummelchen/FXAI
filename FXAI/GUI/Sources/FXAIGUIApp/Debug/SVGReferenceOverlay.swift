import AppKit
import FXAIGUICore
import SwiftUI
import WebKit

struct SVGReferenceOverlay: NSViewRepresentable {
    let showScreenshotReference: Bool

    func makeNSView(context: Context) -> NSView {
        let container = NSView()
        container.wantsLayer = true
        container.layer?.backgroundColor = .clear

        let webView = WKWebView(frame: .zero, configuration: WKWebViewConfiguration())
        webView.setValue(false, forKey: "drawsBackground")
        webView.translatesAutoresizingMaskIntoConstraints = false
        container.addSubview(webView)

        NSLayoutConstraint.activate([
            webView.leadingAnchor.constraint(equalTo: container.leadingAnchor),
            webView.trailingAnchor.constraint(equalTo: container.trailingAnchor),
            webView.topAnchor.constraint(equalTo: container.topAnchor),
            webView.bottomAnchor.constraint(equalTo: container.bottomAnchor)
        ])

        context.coordinator.webView = webView
        loadContent(showScreenshotReference: showScreenshotReference, into: webView)
        return container
    }

    func updateNSView(_ nsView: NSView, context: Context) {
        if let webView = context.coordinator.webView {
            loadContent(showScreenshotReference: showScreenshotReference, into: webView)
        }
    }

    func makeCoordinator() -> Coordinator {
        Coordinator()
    }

    private func loadContent(showScreenshotReference: Bool, into webView: WKWebView) {
        if showScreenshotReference, let url = SVGAssetCatalog.referencePNGURL() {
            let html = """
            <html><body style="margin:0;background:transparent;overflow:hidden;">
            <img src="\(url.absoluteString)" style="width:100%;height:100%;object-fit:contain;display:block;" />
            </body></html>
            """
            webView.loadHTMLString(html, baseURL: nil)
            return
        }

        guard let svgString = SVGAssetCatalog.referenceSVGString() else { return }
        let html = """
        <html>
        <head><meta name="viewport" content="width=device-width, initial-scale=1.0"></head>
        <body style="margin:0;background:transparent;overflow:hidden;">
        <div style="width:100vw;height:100vh;display:flex;align-items:center;justify-content:center;">
        \(svgString)
        </div>
        <style>svg{width:100%;height:100%;display:block;}</style>
        </body>
        </html>
        """
        webView.loadHTMLString(html, baseURL: nil)
    }

    final class Coordinator {
        var webView: WKWebView?
    }
}
