import FXAIGUICore
import SwiftUI

struct RoleWorkspaceDashboardView: View {
    @EnvironmentObject private var model: FXAIGUIModel

    let role: WorkspaceRole

    private var guide: RoleOnboardingGuide? {
        guard let projectRoot = model.projectRoot else { return nil }
        return OnboardingGuideFactory.guide(role: role, projectRoot: projectRoot)
    }

    private var roleCommands: [CommandRecipe] {
        model.commandRecipes.filter { $0.role == role }
    }

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 18) {
                SectionHeader(
                    title: role.workspaceTitle,
                    subtitle: role.workspaceSummary
                )

                hero
                statusSummary
                benefitsAndScenarios
                quickScreens

                if !roleCommands.isEmpty {
                    commands
                }
            }
            .padding(.bottom, 22)
            .onAppear {
                model.selectedRole = role
            }
        }
    }

    private var hero: some View {
        FXAIVisualEffectSurface(style: .hero, cornerRadius: 24, contentPadding: 18, tint: FXAITheme.accent.opacity(0.10)) {
            ViewThatFits(in: .horizontal) {
                HStack(alignment: .top, spacing: 18) {
                    heroText
                    Spacer(minLength: 12)
                    heroActions
                }
                VStack(alignment: .leading, spacing: 16) {
                    heroText
                    heroActions
                }
            }
        }
    }

    private var heroText: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(role.workspaceTitle)
                .font(.system(size: 30, weight: .semibold, design: .rounded))
                .foregroundStyle(FXAITheme.textPrimary)
            Text(role.workspaceSummary)
                .foregroundStyle(FXAITheme.textSecondary)
            Text("Primary value: \(role.focusAreas.joined(separator: " • "))")
                .font(.caption)
                .foregroundStyle(FXAITheme.textMuted)
                .fixedSize(horizontal: false, vertical: true)
        }
    }

    private var heroActions: some View {
        VStack(alignment: .trailing, spacing: 10) {
            StatusBadge(
                title: role.title,
                value: model.hasCompletedOnboarding(for: role) ? "Ready" : "Guide Open",
                tint: model.hasCompletedOnboarding(for: role) ? FXAITheme.success : FXAITheme.warning
            )
            Button("Open Role Guide") {
                model.selectedRole = role
                model.navigate(to: .onboarding)
            }
            .buttonStyle(.borderedProminent)
            .tint(FXAITheme.accent)
        }
    }

    private var statusSummary: some View {
        LazyVGrid(
            columns: [
                GridItem(.flexible(minimum: 220), spacing: 16),
                GridItem(.flexible(minimum: 220), spacing: 16),
                GridItem(.flexible(minimum: 220), spacing: 16),
                GridItem(.flexible(minimum: 220), spacing: 16)
            ],
            spacing: 16
        ) {
            ForEach(summaryCards()) { card in
                MetricCard(
                    title: card.title,
                    value: card.value,
                    footnote: card.footnote,
                    symbolName: card.symbolName,
                    tint: card.tint
                )
            }
        }
    }

    private var benefitsAndScenarios: some View {
        ViewThatFits(in: .horizontal) {
            HStack(alignment: .top, spacing: 16) {
                bulletPanel(title: "Benefits", items: role.workspaceBenefits)
                bulletPanel(title: "Example Scenarios", items: role.workspaceScenarioExamples)
            }
            VStack(alignment: .leading, spacing: 16) {
                bulletPanel(title: "Benefits", items: role.workspaceBenefits)
                bulletPanel(title: "Example Scenarios", items: role.workspaceScenarioExamples)
            }
        }
    }

    private var quickScreens: some View {
        FXAIVisualEffectSurface {
            VStack(alignment: .leading, spacing: 14) {
                Text("Quick Screens")
                    .font(.headline)
                    .foregroundStyle(FXAITheme.textPrimary)

                LazyVGrid(
                    columns: [GridItem(.adaptive(minimum: 190), spacing: 12, alignment: .leading)],
                    spacing: 12
                ) {
                    ForEach(role.workspaceQuickDestinations) { destination in
                        Button {
                            model.selectedRole = role
                            model.navigate(to: destination)
                        } label: {
                            HStack {
                                Label(destination.title, systemImage: destination.symbolName)
                                    .font(.subheadline.weight(.semibold))
                                    .foregroundStyle(FXAITheme.textPrimary)
                                Spacer()
                            }
                            .padding(.horizontal, 14)
                            .padding(.vertical, 12)
                            .background(
                                RoundedRectangle(cornerRadius: 16, style: .continuous)
                                    .fill(FXAITheme.backgroundSecondary.opacity(0.78))
                            )
                        }
                        .buttonStyle(.plain)
                    }
                }

                if let guide, !guide.steps.isEmpty {
                    VStack(alignment: .leading, spacing: 10) {
                        Text("How To Use This Workspace")
                            .font(.subheadline.weight(.semibold))
                            .foregroundStyle(FXAITheme.textPrimary)

                        ForEach(Array(guide.steps.enumerated()), id: \.element.id) { index, step in
                            HStack(alignment: .top, spacing: 12) {
                                Text("\(index + 1)")
                                    .font(.caption.weight(.semibold))
                                    .foregroundStyle(FXAITheme.accent)
                                    .frame(width: 22, alignment: .leading)
                                VStack(alignment: .leading, spacing: 4) {
                                    Text(step.title)
                                        .font(.subheadline.weight(.semibold))
                                        .foregroundStyle(FXAITheme.textPrimary)
                                    Text(step.summary)
                                        .font(.caption)
                                        .foregroundStyle(FXAITheme.textSecondary)
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    private var commands: some View {
        VStack(alignment: .leading, spacing: 14) {
            Text("Suggested Commands")
                .font(.headline)
                .foregroundStyle(FXAITheme.textPrimary)

            ForEach(roleCommands) { recipe in
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

    private func bulletPanel(title: String, items: [String]) -> some View {
        FXAIVisualEffectSurface {
            VStack(alignment: .leading, spacing: 12) {
                Text(title)
                    .font(.headline)
                    .foregroundStyle(FXAITheme.textPrimary)

                ForEach(items, id: \.self) { item in
                    Label(item, systemImage: "checkmark.circle.fill")
                        .foregroundStyle(FXAITheme.textSecondary)
                        .fixedSize(horizontal: false, vertical: true)
                }
            }
        }
        .frame(maxWidth: .infinity, alignment: .topLeading)
    }

    private func summaryCards() -> [RoleWorkspaceMetricCard] {
        switch role {
        case .liveTrader:
            return [
                RoleWorkspaceMetricCard(
                    title: "Runtime Symbols",
                    value: "\(model.runtimeSnapshot?.deployments.count ?? 0)",
                    footnote: "Deployed symbol states currently visible to the GUI.",
                    symbolName: "waveform.path.ecg.rectangle.fill",
                    tint: FXAITheme.accent
                ),
                RoleWorkspaceMetricCard(
                    title: "Incidents",
                    value: "\(model.incidentSnapshot?.incidents.count ?? 0)",
                    footnote: "Generated runtime or artifact problems that weaken live trust.",
                    symbolName: "exclamationmark.triangle.fill",
                    tint: (model.incidentSnapshot?.incidents.isEmpty == false) ? FXAITheme.warning : FXAITheme.success
                ),
                RoleWorkspaceMetricCard(
                    title: "Champions",
                    value: "\(model.snapshot?.operatorSummary.championCount ?? 0)",
                    footnote: "Promoted champion packs available in the current FXAI tree.",
                    symbolName: "rosette",
                    tint: FXAITheme.success
                ),
                RoleWorkspaceMetricCard(
                    title: "Connection",
                    value: model.connectionStatusLabel,
                    footnote: "The GUI must be attached before a live review is trustworthy.",
                    symbolName: "link.circle.fill",
                    tint: tintForConnection()
                )
            ]
        case .demoTrader:
            return [
                RoleWorkspaceMetricCard(
                    title: "Runtime Symbols",
                    value: "\(model.runtimeSnapshot?.deployments.count ?? 0)",
                    footnote: "Observed deployed states for safe study.",
                    symbolName: "waveform.path.ecg.rectangle.fill",
                    tint: FXAITheme.accent
                ),
                RoleWorkspaceMetricCard(
                    title: "Recent Reports",
                    value: "\(model.snapshot?.recentArtifacts.count ?? 0)",
                    footnote: "Artifacts you can compare against current demo behavior.",
                    symbolName: "doc.text.image.fill",
                    tint: FXAITheme.accentSoft
                ),
                RoleWorkspaceMetricCard(
                    title: "Incidents",
                    value: "\(model.incidentSnapshot?.incidents.count ?? 0)",
                    footnote: "Warnings worth understanding before trusting a demo conclusion.",
                    symbolName: "checkmark.shield.fill",
                    tint: (model.incidentSnapshot?.incidents.isEmpty == false) ? FXAITheme.warning : FXAITheme.success
                ),
                RoleWorkspaceMetricCard(
                    title: "Commands",
                    value: "\(roleCommands.count)",
                    footnote: "Curated role workflows available from the GUI command surfaces.",
                    symbolName: "terminal.fill",
                    tint: FXAITheme.success
                )
            ]
        case .backtester:
            return [
                RoleWorkspaceMetricCard(
                    title: "Plugins",
                    value: "\(model.snapshot?.totalPluginCount ?? 0)",
                    footnote: "Available plugins to target from the builder.",
                    symbolName: "shippingbox.fill",
                    tint: FXAITheme.accent
                ),
                RoleWorkspaceMetricCard(
                    title: "Baselines",
                    value: "\(model.snapshot?.reportCategories.first(where: { $0.category == "Baselines" })?.fileCount ?? 0)",
                    footnote: "Current regression baselines visible in the project tree.",
                    symbolName: "chart.xyaxis.line",
                    tint: FXAITheme.success
                ),
                RoleWorkspaceMetricCard(
                    title: "Commands",
                    value: "\(roleCommands.count)",
                    footnote: "Backtest prep workflows ready for terminal handoff.",
                    symbolName: "terminal.fill",
                    tint: FXAITheme.accentSoft
                ),
                RoleWorkspaceMetricCard(
                    title: "Connection",
                    value: model.connectionStatusLabel,
                    footnote: "Project attachment determines whether the builder can inspect real FXAI state.",
                    symbolName: "link.circle.fill",
                    tint: tintForConnection()
                )
            ]
        case .researcher:
            return [
                RoleWorkspaceMetricCard(
                    title: "Plugins",
                    value: "\(model.snapshot?.totalPluginCount ?? 0)",
                    footnote: "The size of the current model zoo available to research.",
                    symbolName: "shippingbox.fill",
                    tint: FXAITheme.accent
                ),
                RoleWorkspaceMetricCard(
                    title: "Artifacts",
                    value: "\(model.snapshot?.totalReportCount ?? 0)",
                    footnote: "Research and promotion artifacts available for inspection.",
                    symbolName: "tray.full.fill",
                    tint: FXAITheme.accentSoft
                ),
                RoleWorkspaceMetricCard(
                    title: "Champions",
                    value: "\(model.snapshot?.operatorSummary.championCount ?? 0)",
                    footnote: "Promoted winners available to compare against challengers.",
                    symbolName: "rosette",
                    tint: FXAITheme.success
                ),
                RoleWorkspaceMetricCard(
                    title: "Deployments",
                    value: "\(model.snapshot?.operatorSummary.deploymentCount ?? 0)",
                    footnote: "Current deployment profiles emitted from the research outputs.",
                    symbolName: "waveform.path.ecg",
                    tint: FXAITheme.warning
                )
            ]
        case .architect:
            return [
                RoleWorkspaceMetricCard(
                    title: "Incidents",
                    value: "\(model.incidentSnapshot?.incidents.count ?? 0)",
                    footnote: "The generated recovery queue for build, runtime, and Research OS issues.",
                    symbolName: "exclamationmark.triangle.fill",
                    tint: (model.incidentSnapshot?.incidents.isEmpty == false) ? FXAITheme.warning : FXAITheme.success
                ),
                RoleWorkspaceMetricCard(
                    title: "Branches",
                    value: "\(model.researchSnapshot?.branchCount ?? 0)",
                    footnote: "Tracked Research OS branches in the current operator dashboard.",
                    symbolName: "point.3.connected.trianglepath.dotted",
                    tint: FXAITheme.accentSoft
                ),
                RoleWorkspaceMetricCard(
                    title: "Audit Events",
                    value: "\(model.researchSnapshot?.auditEventCount ?? 0)",
                    footnote: "Recent platform audit events attached to the latest dashboard.",
                    symbolName: "checkmark.shield.fill",
                    tint: FXAITheme.success
                ),
                RoleWorkspaceMetricCard(
                    title: "Connection",
                    value: model.connectionStatusLabel,
                    footnote: "Detached mode is safe for startup, but not for platform validation.",
                    symbolName: "link.circle.fill",
                    tint: tintForConnection()
                )
            ]
        }
    }

    private func tintForConnection() -> Color {
        switch model.connectionState {
        case .connected:
            FXAITheme.success
        case .waitingForProject:
            FXAITheme.warning
        case .disconnectedByUser:
            FXAITheme.textMuted
        }
    }
}

private struct RoleWorkspaceMetricCard: Identifiable {
    let id = UUID()
    let title: String
    let value: String
    let footnote: String
    let symbolName: String
    let tint: Color
}
