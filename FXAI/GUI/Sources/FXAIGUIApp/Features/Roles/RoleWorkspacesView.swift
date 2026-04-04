import FXAIGUICore
import SwiftUI

struct RoleWorkspacesView: View {
    @EnvironmentObject private var model: FXAIGUIModel

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 18) {
                SectionHeader(
                    title: "Role Workspaces",
                    subtitle: "Different users need different default surfaces. FXAI GUI starts from the role, not from raw internals."
                )

                HStack(spacing: 12) {
                    ForEach(WorkspaceRole.allCases) { role in
                        Button {
                            model.selectedRole = role
                        } label: {
                            VStack(alignment: .leading, spacing: 8) {
                                Label(role.title, systemImage: role.symbolName)
                                    .font(.headline)
                                Text(role.subtitle)
                                    .font(.caption)
                                    .foregroundStyle(FXAITheme.textSecondary)
                                    .multilineTextAlignment(.leading)
                            }
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .padding(16)
                            .background(
                                RoundedRectangle(cornerRadius: 18, style: .continuous)
                                    .fill(model.selectedRole == role ? FXAITheme.panelStrong : FXAITheme.panel.opacity(0.72))
                            )
                            .overlay(
                                RoundedRectangle(cornerRadius: 18, style: .continuous)
                                    .strokeBorder(model.selectedRole == role ? FXAITheme.accent.opacity(0.35) : FXAITheme.stroke, lineWidth: 1)
                            )
                        }
                        .buttonStyle(.plain)
                    }
                }

                roleDetail(role: model.selectedRole)
            }
        }
    }

    private func roleDetail(role: WorkspaceRole) -> some View {
        FXAIVisualEffectSurface {
            VStack(alignment: .leading, spacing: 18) {
                HStack {
                    VStack(alignment: .leading, spacing: 6) {
                        Text(role.title)
                            .font(.system(size: 28, weight: .semibold, design: .rounded))
                            .foregroundStyle(FXAITheme.textPrimary)
                        Text(role.subtitle)
                            .foregroundStyle(FXAITheme.textSecondary)
                    }
                    Spacer()
                    VStack(alignment: .trailing, spacing: 10) {
                        Image(systemName: role.symbolName)
                            .font(.system(size: 28, weight: .semibold))
                            .foregroundStyle(FXAITheme.accent)
                        StatusBadge(
                            title: "Onboarding",
                            value: model.hasCompletedOnboarding(for: role) ? "Done" : "Open",
                            tint: model.hasCompletedOnboarding(for: role) ? FXAITheme.success : FXAITheme.warning
                        )
                    }
                }

                HStack(alignment: .top, spacing: 18) {
                    VStack(alignment: .leading, spacing: 10) {
                        Text("Focus Areas")
                            .font(.headline)
                            .foregroundStyle(FXAITheme.textPrimary)
                        ForEach(role.focusAreas, id: \.self) { item in
                            Label(item, systemImage: "checkmark.circle.fill")
                                .foregroundStyle(FXAITheme.textSecondary)
                        }
                    }
                    .frame(maxWidth: .infinity, alignment: .leading)

                    VStack(alignment: .leading, spacing: 10) {
                        Text("Ignore At First")
                            .font(.headline)
                            .foregroundStyle(FXAITheme.textPrimary)
                        Text(role.ignoreAtFirst)
                            .foregroundStyle(FXAITheme.textSecondary)
                    }
                    .frame(maxWidth: .infinity, alignment: .leading)
                }

                VStack(alignment: .leading, spacing: 10) {
                    Text("Recommended Commands")
                        .font(.headline)
                        .foregroundStyle(FXAITheme.textPrimary)

                    ForEach(model.commandRecipes.filter { $0.role == role }) { recipe in
                        VStack(alignment: .leading, spacing: 6) {
                            HStack {
                                Text(recipe.title)
                                    .font(.subheadline.weight(.semibold))
                                    .foregroundStyle(FXAITheme.textPrimary)
                                Spacer()
                                Text(recipe.commandKind)
                                    .font(.caption.weight(.semibold))
                                    .foregroundStyle(FXAITheme.accentSoft)
                            }
                            Text(recipe.summary)
                                .font(.caption)
                                .foregroundStyle(FXAITheme.textSecondary)
                            Text(recipe.command)
                                .font(.system(.caption, design: .monospaced))
                                .foregroundStyle(FXAITheme.textMuted)
                                .textSelection(.enabled)
                        }
                        .padding(12)
                        .background(
                            RoundedRectangle(cornerRadius: 16, style: .continuous)
                                .fill(FXAITheme.backgroundSecondary.opacity(0.5))
                        )
                    }
                }

                HStack {
                    Button("Open Role Guide") {
                        model.selectedRole = role
                        model.navigate(to: .onboarding)
                    }
                    .buttonStyle(.borderedProminent)
                    .tint(FXAITheme.accent)

                    Button(model.hasCompletedOnboarding(for: role) ? "Reset Onboarding" : "Mark Onboarding Complete") {
                        if model.hasCompletedOnboarding(for: role) {
                            model.resetOnboarding(for: role)
                        } else {
                            model.markOnboardingCompleted(for: role)
                        }
                    }
                    .buttonStyle(.bordered)
                }
            }
        }
    }
}
