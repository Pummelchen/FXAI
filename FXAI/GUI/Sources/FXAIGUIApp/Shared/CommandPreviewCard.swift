import FXAIGUICore
import SwiftUI

struct CommandPreviewCard: View {
    let title: String
    let summary: String
    let command: String
    let onCopy: () -> Void
    let onTerminal: () -> Void

    var body: some View {
        FXAIVisualEffectSurface {
            VStack(alignment: .leading, spacing: 14) {
                HStack(alignment: .top) {
                    VStack(alignment: .leading, spacing: 4) {
                        Text(title)
                            .font(.headline)
                            .foregroundStyle(FXAITheme.textPrimary)
                        Text(summary)
                            .font(.caption)
                            .foregroundStyle(FXAITheme.textSecondary)
                    }
                    Spacer()
                    HStack {
                        Button("Copy Command", action: onCopy)
                            .buttonStyle(.bordered)
                        Button("Open Terminal Here", action: onTerminal)
                            .buttonStyle(.borderedProminent)
                            .tint(FXAITheme.accent)
                    }
                }

                Text(command)
                    .font(.system(.caption, design: .monospaced))
                    .foregroundStyle(FXAITheme.textMuted)
                    .textSelection(.enabled)
                    .padding(14)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .background(
                        RoundedRectangle(cornerRadius: 18, style: .continuous)
                            .fill(FXAITheme.backgroundSecondary.opacity(0.7))
                    )
            }
        }
    }
}
