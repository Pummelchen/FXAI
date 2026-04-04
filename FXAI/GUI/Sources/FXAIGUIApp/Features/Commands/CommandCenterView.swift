import FXAIGUICore
import SwiftUI

struct CommandCenterView: View {
    @EnvironmentObject private var model: FXAIGUIModel

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 18) {
                SectionHeader(
                    title: "Command Center",
                    subtitle: "The terminal remains first-class. The GUI helps you discover, inspect, and copy the right FXAI commands quickly."
                )

                ForEach(WorkspaceRole.allCases) { role in
                    FXAIVisualEffectSurface {
                        VStack(alignment: .leading, spacing: 14) {
                            HStack {
                                Label(role.title, systemImage: role.symbolName)
                                    .font(.headline)
                                    .foregroundStyle(FXAITheme.textPrimary)
                                Spacer()
                            }

                            ForEach(model.commandRecipes.filter { $0.role == role }) { recipe in
                                VStack(alignment: .leading, spacing: 8) {
                                    HStack {
                                        VStack(alignment: .leading, spacing: 4) {
                                            Text(recipe.title)
                                                .font(.subheadline.weight(.semibold))
                                                .foregroundStyle(FXAITheme.textPrimary)
                                            Text(recipe.summary)
                                                .font(.caption)
                                                .foregroundStyle(FXAITheme.textSecondary)
                                        }
                                        Spacer()
                                        Button("Copy") {
                                            model.copyToPasteboard(recipe.command)
                                        }
                                        .buttonStyle(.borderedProminent)
                                        .tint(FXAITheme.accent)
                                    }

                                    Text(recipe.command)
                                        .font(.system(.caption, design: .monospaced))
                                        .foregroundStyle(FXAITheme.textMuted)
                                        .textSelection(.enabled)
                                        .padding(12)
                                        .frame(maxWidth: .infinity, alignment: .leading)
                                        .background(
                                            RoundedRectangle(cornerRadius: 16, style: .continuous)
                                                .fill(FXAITheme.backgroundSecondary.opacity(0.56))
                                        )
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
