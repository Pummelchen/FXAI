import FXAIGUICore
import SwiftUI

struct SettingsView: View {
    @EnvironmentObject private var model: FXAIGUIModel
    @EnvironmentObject private var themeEnvironment: ThemeEnvironment

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 18) {
                SectionHeader(
                    title: "Settings",
                    subtitle: "Manage project connection, saved workspaces, onboarding state, environment checks, and release packaging."
                )

                FXAIVisualEffectSurface {
                    VStack(alignment: .leading, spacing: 14) {
                        Text("Theme System")
                            .font(.headline)
                            .foregroundStyle(FXAITheme.textPrimary)

                        ThemeInspectorView(layoutOutput: nil)
                            .environmentObject(themeEnvironment)

                        Text("The GUI presentation is driven by the shared FXAI theme registry, so future themes can be introduced without rewriting the operator shell.")
                            .font(.callout)
                            .foregroundStyle(FXAITheme.textSecondary)
                    }
                }

                FXAIVisualEffectSurface {
                    VStack(alignment: .leading, spacing: 14) {
                        Text("Project Connection")
                            .font(.headline)
                            .foregroundStyle(FXAITheme.textPrimary)

                        ViewThatFits(in: .horizontal) {
                            HStack(spacing: 18) {
                                connectionBadges
                            }
                            LazyVGrid(
                                columns: [GridItem(.adaptive(minimum: 180), spacing: 12, alignment: .leading)],
                                spacing: 12
                            ) {
                                connectionBadges
                            }
                        }

                        Text(model.projectPathLabel)
                            .font(.callout)
                            .foregroundStyle(FXAITheme.textSecondary)
                            .textSelection(.enabled)

                        ViewThatFits(in: .horizontal) {
                            HStack {
                                connectionButtons
                            }
                            VStack(alignment: .leading, spacing: 10) {
                                connectionButtons
                            }
                        }

                        Toggle("Enable soft auto reconnect every 10 seconds", isOn: $model.autoReconnectEnabled)
                    }
                }

                savedViewsSection
                onboardingSection
                packagingSection

                if let snapshot = model.snapshot {
                    FXAIVisualEffectSurface {
                        VStack(alignment: .leading, spacing: 14) {
                            Text("Environment")
                                .font(.headline)
                                .foregroundStyle(FXAITheme.textPrimary)

                            ViewThatFits(in: .horizontal) {
                                HStack(spacing: 18) {
                                    environmentBadges(snapshot: snapshot)
                                }
                                LazyVGrid(
                                    columns: [GridItem(.adaptive(minimum: 180), spacing: 12, alignment: .leading)],
                                    spacing: 12
                                ) {
                                    environmentBadges(snapshot: snapshot)
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    private var connectionTint: Color {
        switch model.connectionState {
        case .connected:
            FXAITheme.success
        case .waitingForProject:
            FXAITheme.warning
        case .disconnectedByUser:
            FXAITheme.textMuted
        }
    }

    private var savedViewsSection: some View {
        FXAIVisualEffectSurface {
            VStack(alignment: .leading, spacing: 14) {
                Text("Saved Views")
                    .font(.headline)
                    .foregroundStyle(FXAITheme.textPrimary)

                HStack(spacing: 12) {
                    TextField("View name (optional)", text: $model.saveViewNameDraft)
                        .textFieldStyle(.roundedBorder)
                    Button("Save Current View") {
                        model.saveCurrentView()
                    }
                    .buttonStyle(.borderedProminent)
                    .tint(FXAITheme.accent)
                }

                if model.savedViews.isEmpty {
                    Text("No saved views yet. Save the current workspace to keep a repeatable operator setup.")
                        .foregroundStyle(FXAITheme.textSecondary)
                } else {
                    ForEach(model.savedViews) { savedView in
                        ViewThatFits(in: .horizontal) {
                            HStack(alignment: .top, spacing: 12) {
                                savedViewSummary(savedView)
                                Spacer()
                                savedViewActions(savedView)
                            }
                            VStack(alignment: .leading, spacing: 12) {
                                savedViewSummary(savedView)
                                savedViewActions(savedView)
                            }
                        }
                        .padding(12)
                        .background(
                            RoundedRectangle(cornerRadius: 16, style: .continuous)
                                .fill(FXAITheme.backgroundSecondary.opacity(0.56))
                        )
                    }
                }
            }
        }
    }

    private var onboardingSection: some View {
        FXAIVisualEffectSurface {
            VStack(alignment: .leading, spacing: 14) {
                Text("Role Onboarding")
                    .font(.headline)
                    .foregroundStyle(FXAITheme.textPrimary)

                ForEach(WorkspaceRole.allCases) { role in
                    ViewThatFits(in: .horizontal) {
                        HStack {
                            Label(role.title, systemImage: role.symbolName)
                                .foregroundStyle(FXAITheme.textPrimary)
                            Spacer()
                            StatusBadge(
                                title: "Status",
                                value: model.hasCompletedOnboarding(for: role) ? "Completed" : "Open",
                                tint: model.hasCompletedOnboarding(for: role) ? FXAITheme.success : FXAITheme.warning
                            )
                            Button("Guide") {
                                model.selectedRole = role
                                model.navigate(to: .onboarding)
                            }
                            .buttonStyle(.bordered)
                            Button(model.hasCompletedOnboarding(for: role) ? "Reset" : "Complete") {
                                if model.hasCompletedOnboarding(for: role) {
                                    model.resetOnboarding(for: role)
                                } else {
                                    model.markOnboardingCompleted(for: role)
                                }
                            }
                            .buttonStyle(.bordered)
                        }
                        VStack(alignment: .leading, spacing: 10) {
                            Label(role.title, systemImage: role.symbolName)
                                .foregroundStyle(FXAITheme.textPrimary)
                            HStack(spacing: 12) {
                                StatusBadge(
                                    title: "Status",
                                    value: model.hasCompletedOnboarding(for: role) ? "Completed" : "Open",
                                    tint: model.hasCompletedOnboarding(for: role) ? FXAITheme.success : FXAITheme.warning
                                )
                                Button("Guide") {
                                    model.selectedRole = role
                                    model.navigate(to: .onboarding)
                                }
                                .buttonStyle(.bordered)
                            }
                            Button(model.hasCompletedOnboarding(for: role) ? "Reset" : "Complete") {
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
    }

    private var packagingSection: some View {
        FXAIVisualEffectSurface {
            VStack(alignment: .leading, spacing: 14) {
                Text("Release Packaging")
                    .font(.headline)
                    .foregroundStyle(FXAITheme.textPrimary)

                Text("Phase 6 includes a release-packaging path for the GUI so operators can build a polished macOS app bundle without hunting for build steps.")
                    .foregroundStyle(FXAITheme.textSecondary)

                if let projectRoot = model.projectRoot {
                    let packageCommand = [
                        "cd '\(projectRoot.appendingPathComponent("GUI").path.replacingOccurrences(of: "'", with: "'\"'\"'"))'",
                        "./Tools/package_gui_release.sh"
                    ].joined(separator: "\n")

                    CommandPreviewCard(
                        title: "Package FXAI GUI",
                        summary: "Build a release bundle and zip artifact for operator distribution.",
                        command: packageCommand,
                        onCopy: { model.copyToPasteboard(packageCommand) },
                        onTerminal: { model.handoffCommandToTerminal(packageCommand) }
                    )
                }
            }
        }
    }

    private var connectionBadges: some View {
        Group {
            StatusBadge(
                title: "State",
                value: model.connectionStatusLabel,
                tint: connectionTint
            )
            StatusBadge(
                title: "Auto Reconnect",
                value: model.autoReconnectEnabled ? "On" : "Off",
                tint: model.autoReconnectEnabled ? FXAITheme.success : FXAITheme.textMuted
            )
            StatusBadge(
                title: "Last Check",
                value: model.lastConnectionCheckAt.map(FXAIFormatting.dateTimeString(for:)) ?? "Pending",
                tint: FXAITheme.accentSoft
            )
        }
    }

    private var connectionButtons: some View {
        Group {
            Button(model.projectRoot == nil ? "Connect" : "Choose Project") {
                if model.projectRoot == nil {
                    model.reconnectProject()
                } else {
                    model.chooseProjectRoot()
                }
            }
            .buttonStyle(.borderedProminent)
            .tint(FXAITheme.accent)

            Button(model.projectRoot == nil ? "Pick Project" : "Disconnect") {
                if model.projectRoot == nil {
                    model.chooseProjectRoot()
                } else {
                    model.disconnectProject()
                }
            }
            .buttonStyle(.bordered)

            Button("Reveal in Finder") {
                model.openProjectRootInFinder()
            }
            .buttonStyle(.bordered)
            .disabled(model.projectRoot == nil)
        }
    }

    private func environmentBadges(snapshot: FXAIProjectSnapshot) -> some View {
        Group {
            StatusBadge(
                title: "Database",
                value: snapshot.tursoSummary.localDatabasePresent ? "Ready" : "Missing",
                tint: snapshot.tursoSummary.localDatabasePresent ? FXAITheme.success : FXAITheme.warning
            )
            StatusBadge(
                title: "Embedded Replica",
                value: snapshot.tursoSummary.embeddedReplicaConfigured ? "Configured" : "Local Only",
                tint: snapshot.tursoSummary.embeddedReplicaConfigured ? FXAITheme.accent : FXAITheme.accentSoft
            )
            StatusBadge(
                title: "Encryption",
                value: snapshot.tursoSummary.encryptionConfigured ? "On" : "Off",
                tint: snapshot.tursoSummary.encryptionConfigured ? FXAITheme.success : FXAITheme.warning
            )
        }
    }

    private func savedViewSummary(_ savedView: SavedWorkspaceView) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(savedView.name)
                .font(.subheadline.weight(.semibold))
                .foregroundStyle(FXAITheme.textPrimary)
            Text(savedView.titleSummary)
                .font(.caption)
                .foregroundStyle(FXAITheme.textSecondary)
            Text(savedView.projectRootPath ?? "")
                .font(.caption2)
                .foregroundStyle(FXAITheme.textMuted)
                .lineLimit(2)
        }
    }

    private func savedViewActions(_ savedView: SavedWorkspaceView) -> some View {
        HStack {
            Button("Open") {
                model.applySavedView(savedView)
            }
            .buttonStyle(.bordered)
            Button("Delete") {
                model.deleteSavedView(savedView)
            }
            .buttonStyle(.bordered)
        }
    }

}
