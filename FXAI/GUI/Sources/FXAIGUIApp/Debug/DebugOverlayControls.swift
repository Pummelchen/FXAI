import FXAIGUICore
import SwiftUI

struct DashboardDebugState {
    var overlayEnabled = false
    var overlayOpacity = 0.56
    var showScreenshotReference = false
    var showLayoutGuides = false
    var showShadowGlowDebug = false
    var showFrameOutlines = false
    var reducedEffects = false
}

struct DebugOverlayControls: View {
    @Binding var debugState: DashboardDebugState
    let frameModel: DashboardFrameModel
    let theme: any AppTheme
    let connectionStatus: String

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Text("Calibration")
                    .font(.system(size: 13, weight: .semibold, design: .rounded))
                    .foregroundStyle(theme.colors.textPrimary)
                Spacer()
                Text(frameModel.layoutClass.displayName)
                    .font(.system(size: 11, weight: .medium, design: .rounded))
                    .foregroundStyle(theme.colors.textMuted)
            }

            Toggle("SVG Overlay", isOn: $debugState.overlayEnabled)
            Toggle("PNG Compare", isOn: $debugState.showScreenshotReference)
            Toggle("Layout Guides", isOn: $debugState.showLayoutGuides)
            Toggle("Frame Outlines", isOn: $debugState.showFrameOutlines)
            Toggle("Shadow / Glow Debug", isOn: $debugState.showShadowGlowDebug)
            Toggle("Reduced Effects", isOn: $debugState.reducedEffects)

            VStack(alignment: .leading, spacing: 6) {
                Text("Overlay Opacity")
                    .font(.system(size: 11, weight: .medium, design: .rounded))
                    .foregroundStyle(theme.colors.textMuted)
                Slider(value: $debugState.overlayOpacity, in: 0...1)
                    .tint(theme.colors.warningGreen)
            }

            VStack(alignment: .leading, spacing: 4) {
                statusLine("Scale", value: String(format: "%.3f", frameModel.scale))
                statusLine("Canvas", value: "\(Int(frameModel.stageFrame.width)) × \(Int(frameModel.stageFrame.height))")
                statusLine("Connection", value: connectionStatus)
            }
        }
        .toggleStyle(.switch)
        .padding(16)
        .frame(width: 260)
        .background(
            RoundedRectangle(cornerRadius: 18, style: .continuous)
                .fill(theme.colors.mainPanel.opacity(0.88))
        )
        .overlay(
            RoundedRectangle(cornerRadius: 18, style: .continuous)
                .stroke(theme.colors.cardStroke.opacity(0.9), lineWidth: 1)
        )
        .shadow(color: .black.opacity(0.28), radius: 22, x: 0, y: 16)
    }

    private func statusLine(_ title: String, value: String) -> some View {
        HStack {
            Text(title)
                .font(.system(size: 11, weight: .medium, design: .rounded))
                .foregroundStyle(theme.colors.textMuted)
            Spacer()
            Text(value)
                .font(.system(size: 11, weight: .semibold, design: .rounded))
                .foregroundStyle(theme.colors.textSecondary)
        }
    }
}
