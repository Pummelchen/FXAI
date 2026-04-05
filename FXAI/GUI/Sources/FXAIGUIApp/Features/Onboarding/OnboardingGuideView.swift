import FXAIGUICore
import SwiftUI

struct OnboardingGuideView: View {
    @EnvironmentObject private var model: FXAIGUIModel

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 18) {
                SectionHeader(
                    title: "Role Onboarding",
                    subtitle: "A clean first path for each FXAI user type. Learn the right screens, commands, and workflow order before diving into internals."
                )

                ViewThatFits(in: .horizontal) {
                    Picker("Role", selection: $model.selectedRole) {
                        ForEach(WorkspaceRole.allCases) { role in
                            Text(role.title).tag(role)
                        }
                    }
                    .pickerStyle(.segmented)

                    Picker("Role", selection: $model.selectedRole) {
                        ForEach(WorkspaceRole.allCases) { role in
                            Text(role.title).tag(role)
                        }
                    }
                    .pickerStyle(.menu)
                }

                if let guide = model.currentOnboardingGuide {
                    hero(guide: guide)
                    steps(guide: guide)
                    destinations(guide: guide)
                    commands(guide: guide)
                    shortcutLegend
                } else {
                    EmptyStateView(
                        title: "No onboarding guide available",
                        message: "Select a valid FXAI project root so the GUI can load role-specific guidance and command recipes.",
                        symbolName: "sparkles.rectangle.stack.fill"
                    )
                }
            }
            .padding(.bottom, 22)
        }
    }

    private func hero(guide: RoleOnboardingGuide) -> some View {
        FXAIVisualEffectSurface {
            VStack(alignment: .leading, spacing: 16) {
                ViewThatFits(in: .horizontal) {
                    HStack(alignment: .top) {
                        VStack(alignment: .leading, spacing: 8) {
                            Text(guide.headline)
                                .font(.system(size: 30, weight: .semibold, design: .rounded))
                                .foregroundStyle(FXAITheme.textPrimary)
                            Text(guide.summary)
                                .foregroundStyle(FXAITheme.textSecondary)
                        }
                        Spacer()
                        VStack(alignment: .trailing, spacing: 10) {
                            StatusBadge(
                                title: guide.role.title,
                                value: model.hasCompletedOnboarding(for: guide.role) ? "Completed" : "Open",
                                tint: model.hasCompletedOnboarding(for: guide.role) ? FXAITheme.success : FXAITheme.warning
                            )
                            Button(model.hasCompletedOnboarding(for: guide.role) ? "Reset" : "Mark Completed") {
                                if model.hasCompletedOnboarding(for: guide.role) {
                                    model.resetOnboarding(for: guide.role)
                                } else {
                                    model.markOnboardingCompleted(for: guide.role)
                                }
                            }
                            .buttonStyle(.borderedProminent)
                            .tint(model.hasCompletedOnboarding(for: guide.role) ? FXAITheme.warning : FXAITheme.accent)
                        }
                    }
                    VStack(alignment: .leading, spacing: 16) {
                        VStack(alignment: .leading, spacing: 8) {
                            Text(guide.headline)
                                .font(.system(size: 30, weight: .semibold, design: .rounded))
                                .foregroundStyle(FXAITheme.textPrimary)
                            Text(guide.summary)
                                .foregroundStyle(FXAITheme.textSecondary)
                        }
                        HStack(spacing: 12) {
                            StatusBadge(
                                title: guide.role.title,
                                value: model.hasCompletedOnboarding(for: guide.role) ? "Completed" : "Open",
                                tint: model.hasCompletedOnboarding(for: guide.role) ? FXAITheme.success : FXAITheme.warning
                            )
                            Button(model.hasCompletedOnboarding(for: guide.role) ? "Reset" : "Mark Completed") {
                                if model.hasCompletedOnboarding(for: guide.role) {
                                    model.resetOnboarding(for: guide.role)
                                } else {
                                    model.markOnboardingCompleted(for: guide.role)
                                }
                            }
                            .buttonStyle(.borderedProminent)
                            .tint(model.hasCompletedOnboarding(for: guide.role) ? FXAITheme.warning : FXAITheme.accent)
                        }
                    }
                }
            }
        }
    }

    private func steps(guide: RoleOnboardingGuide) -> some View {
        VStack(alignment: .leading, spacing: 14) {
            Text("Recommended First Steps")
                .font(.headline)
                .foregroundStyle(FXAITheme.textPrimary)

            ForEach(Array(guide.steps.enumerated()), id: \.element.id) { index, step in
                FXAIVisualEffectSurface {
                    HStack(alignment: .top, spacing: 16) {
                        ZStack {
                            Circle()
                                .fill(FXAITheme.accent.opacity(0.18))
                                .frame(width: 34, height: 34)
                            Text("\(index + 1)")
                                .font(.headline)
                                .foregroundStyle(FXAITheme.accent)
                        }

                        VStack(alignment: .leading, spacing: 8) {
                            HStack {
                                Text(step.title)
                                    .font(.headline)
                                    .foregroundStyle(FXAITheme.textPrimary)
                                Spacer()
                                if let destination = step.destination {
                                    Button(destination.title) {
                                        if let sidebarDestination = SidebarDestination(rawValue: destination.selection) {
                                            model.navigate(to: sidebarDestination)
                                        }
                                    }
                                    .buttonStyle(.bordered)
                                }
                            }

                            Text(step.summary)
                                .foregroundStyle(FXAITheme.textSecondary)
                        }
                    }
                }
            }
        }
    }

    private func destinations(guide: RoleOnboardingGuide) -> some View {
        FXAIVisualEffectSurface {
            VStack(alignment: .leading, spacing: 14) {
                Text("Recommended Screens")
                    .font(.headline)
                    .foregroundStyle(FXAITheme.textPrimary)

                LazyVGrid(
                    columns: [GridItem(.adaptive(minimum: 180), spacing: 12, alignment: .leading)],
                    spacing: 12
                ) {
                    ForEach(guide.recommendedDestinations) { destination in
                        Button {
                            if let sidebarDestination = SidebarDestination(rawValue: destination.selection) {
                                model.navigate(to: sidebarDestination)
                            }
                        } label: {
                            Text(destination.title)
                                .font(.subheadline.weight(.semibold))
                                .foregroundStyle(FXAITheme.textPrimary)
                                .padding(.horizontal, 14)
                                .padding(.vertical, 10)
                                .background(
                                    Capsule(style: .continuous)
                                        .fill(FXAITheme.backgroundSecondary.opacity(0.78))
                                )
                        }
                        .buttonStyle(.plain)
                    }
                }
            }
        }
    }

    private func commands(guide: RoleOnboardingGuide) -> some View {
        VStack(alignment: .leading, spacing: 14) {
            Text("Suggested Commands")
                .font(.headline)
                .foregroundStyle(FXAITheme.textPrimary)

            ForEach(guide.recommendedCommands) { recipe in
                CommandPreviewCard(
                    title: recipe.title,
                    summary: recipe.summary,
                    command: recipe.command,
                    onCopy: { model.copyToPasteboard(recipe.command) },
                    onTerminal: { model.handoffCommandToTerminal(recipe.command) }
                )
            }
        }
    }

    private var shortcutLegend: some View {
        FXAIVisualEffectSurface {
            VStack(alignment: .leading, spacing: 12) {
                Text("Keyboard First")
                    .font(.headline)
                    .foregroundStyle(FXAITheme.textPrimary)

                LazyVGrid(
                    columns: [
                        GridItem(.flexible(minimum: 180), spacing: 12),
                        GridItem(.flexible(minimum: 180), spacing: 12),
                        GridItem(.flexible(minimum: 180), spacing: 12)
                    ],
                    spacing: 12
                ) {
                    shortcutKey("⌘R", "Refresh FXAI state")
                    shortcutKey("⇧⌘S", "Save current view")
                    shortcutKey("⇧⌘O", "Open onboarding")
                    shortcutKey("⇧⌘I", "Open incidents")
                    shortcutKey("⌘1…⌘6", "Jump to primary screens")
                    shortcutKey("⇧⌘T", "Open command center")
                }
            }
        }
    }

    private func shortcutKey(_ key: String, _ description: String) -> some View {
        HStack(spacing: 10) {
            Text(key)
                .font(.system(.caption, design: .monospaced).weight(.semibold))
                .foregroundStyle(FXAITheme.accent)
                .padding(.horizontal, 10)
                .padding(.vertical, 6)
                .background(
                    Capsule(style: .continuous)
                        .fill(FXAITheme.backgroundSecondary.opacity(0.84))
                )
            Text(description)
                .font(.caption)
                .foregroundStyle(FXAITheme.textSecondary)
            Spacer()
        }
    }

}
