import Charts
import FXAIGUICore
import SwiftUI

struct OverviewDashboardView: View {
    @EnvironmentObject private var model: FXAIGUIModel

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 18) {
                SectionHeader(
                    title: "FXAI Overview",
                    subtitle: "A project-aware dark dashboard for the current MT5 tree, research outputs, and operator workflows."
                )

                if let snapshot = model.snapshot {
                    hero(snapshot: snapshot)

                    LazyVGrid(
                        columns: [
                            GridItem(.flexible(minimum: 220), spacing: 16),
                            GridItem(.flexible(minimum: 220), spacing: 16),
                            GridItem(.flexible(minimum: 220), spacing: 16),
                            GridItem(.flexible(minimum: 220), spacing: 16)
                        ],
                        spacing: 16
                    ) {
                        MetricCard(
                            title: "Build Targets",
                            value: "\(snapshot.cleanBuildTargetCount)/\(snapshot.buildTargets.count)",
                            footnote: "MT5 outputs currently present in the project tree.",
                            symbolName: "hammer.fill",
                            tint: FXAITheme.warning
                        )
                        MetricCard(
                            title: "Plugins",
                            value: "\(snapshot.totalPluginCount)",
                            footnote: "Discovered across \(snapshot.pluginFamilies.count) families.",
                            symbolName: "shippingbox.fill",
                            tint: FXAITheme.accent
                        )
                        MetricCard(
                            title: "Artifacts",
                            value: "\(snapshot.totalReportCount)",
                            footnote: "Baselines, ResearchOS, profiles, and distillation outputs.",
                            symbolName: "tray.full.fill",
                            tint: FXAITheme.accentSoft
                        )
                        MetricCard(
                            title: "Runtime Profiles",
                            value: "\(snapshot.runtimeProfiles.count)",
                            footnote: "Current deployment payloads discovered under ResearchOS.",
                            symbolName: "waveform.path.ecg",
                            tint: FXAITheme.success
                        )
                        MetricCard(
                            title: "Incidents",
                            value: "\(model.incidentSnapshot?.incidents.count ?? 0)",
                            footnote: "Generated operator issues with guided recovery steps.",
                            symbolName: "exclamationmark.triangle.fill",
                            tint: (model.incidentSnapshot?.incidents.isEmpty == false) ? FXAITheme.warning : FXAITheme.success
                        )
                        MetricCard(
                            title: "Saved Views",
                            value: "\(model.savedViews.count)",
                            footnote: "Reusable GUI workspace states for repeatable operator flows.",
                            symbolName: "bookmark.fill",
                            tint: FXAITheme.accentSoft
                        )
                    }

                    HStack(alignment: .top, spacing: 16) {
                        pluginChart(snapshot: snapshot)
                        reportChart(snapshot: snapshot)
                    }

                    HStack(alignment: .top, spacing: 16) {
                        recentArtifacts(snapshot: snapshot)
                        runtimeProfiles(snapshot: snapshot)
                    }

                    if let guide = model.currentOnboardingGuide, !model.hasCompletedOnboarding(for: guide.role) {
                        onboardingPrompt(guide: guide)
                    }
                } else {
                    EmptyStateView(
                        title: "No FXAI project loaded",
                        message: model.lastErrorMessage ?? "Choose a project root to load plugin, report, and runtime inventory.",
                        symbolName: "folder.badge.questionmark"
                    )
                }
            }
            .padding(.bottom, 22)
        }
        .scrollContentBackground(.hidden)
    }

    private func hero(snapshot: FXAIProjectSnapshot) -> some View {
        FXAIVisualEffectSurface {
            VStack(alignment: .leading, spacing: 18) {
                HStack(alignment: .top, spacing: 18) {
                    VStack(alignment: .leading, spacing: 8) {
                        Text("FXAI Operator GUI")
                            .font(.caption.weight(.semibold))
                            .foregroundStyle(FXAITheme.textSecondary)

                        Text("Terminal-first control and visibility for the full FXAI stack.")
                            .font(.system(size: 30, weight: .semibold, design: .rounded))
                            .foregroundStyle(FXAITheme.textPrimary)

                        Text("Project root: \(snapshot.projectRoot.path)")
                            .font(.subheadline)
                            .foregroundStyle(FXAITheme.textSecondary)
                            .lineLimit(2)
                    }

                    Spacer(minLength: 12)

                    VStack(alignment: .trailing, spacing: 12) {
                        StatusBadge(
                            title: "Research DB",
                            value: snapshot.tursoSummary.embeddedReplicaConfigured ? "Embedded Replica" : "Local Turso",
                            tint: snapshot.tursoSummary.embeddedReplicaConfigured ? FXAITheme.accent : FXAITheme.accentSoft
                        )
                        StatusBadge(
                            title: "Updated",
                            value: FXAIFormatting.dateTimeString(for: snapshot.generatedAt),
                            tint: FXAITheme.warning
                        )
                    }
                }

                Rectangle()
                    .fill(FXAITheme.stroke)
                    .frame(height: 1)

                HStack(spacing: 18) {
                    StatusBadge(
                        title: "Champions",
                        value: "\(snapshot.operatorSummary.championCount)",
                        tint: FXAITheme.success
                    )
                    StatusBadge(
                        title: "Deployments",
                        value: "\(snapshot.operatorSummary.deploymentCount)",
                        tint: FXAITheme.accent
                    )
                    StatusBadge(
                        title: "Turso",
                        value: snapshot.tursoSummary.localDatabasePresent ? "Present" : "Missing",
                        tint: snapshot.tursoSummary.localDatabasePresent ? FXAITheme.success : FXAITheme.warning
                    )
                    StatusBadge(
                        title: "Encryption",
                        value: snapshot.tursoSummary.encryptionConfigured ? "Configured" : "Off",
                        tint: snapshot.tursoSummary.encryptionConfigured ? FXAITheme.success : FXAITheme.warning
                    )
                }
            }
            .padding(8)
            .background(
                RoundedRectangle(cornerRadius: 26, style: .continuous)
                    .fill(FXAITheme.heroGradient.opacity(0.55))
            )
        }
    }

    private func pluginChart(snapshot: FXAIProjectSnapshot) -> some View {
        FXAIVisualEffectSurface {
            VStack(alignment: .leading, spacing: 14) {
                Text("Plugin Family Footprint")
                    .font(.headline)
                    .foregroundStyle(FXAITheme.textPrimary)

                Chart(snapshot.pluginFamilies) { family in
                    BarMark(
                        x: .value("Family", family.family),
                        y: .value("Plugins", family.pluginCount)
                    )
                    .foregroundStyle(FXAITheme.accent.gradient)
                    .cornerRadius(6)
                }
                .chartYAxis {
                    AxisMarks(position: .leading)
                }
                .frame(minHeight: 240)
            }
        }
        .frame(maxWidth: .infinity)
    }

    private func reportChart(snapshot: FXAIProjectSnapshot) -> some View {
        FXAIVisualEffectSurface {
            VStack(alignment: .leading, spacing: 14) {
                Text("Artifact Surface")
                    .font(.headline)
                    .foregroundStyle(FXAITheme.textPrimary)

                Chart(snapshot.reportCategories) { category in
                    BarMark(
                        x: .value("Category", category.category),
                        y: .value("Files", category.fileCount)
                    )
                    .foregroundStyle(FXAITheme.accentSoft.gradient)
                    .cornerRadius(6)
                }
                .chartYAxis {
                    AxisMarks(position: .leading)
                }
                .frame(minHeight: 240)
            }
        }
        .frame(maxWidth: .infinity)
    }

    private func recentArtifacts(snapshot: FXAIProjectSnapshot) -> some View {
        FXAIVisualEffectSurface {
            VStack(alignment: .leading, spacing: 12) {
                Text("Recent Artifacts")
                    .font(.headline)
                    .foregroundStyle(FXAITheme.textPrimary)

                ForEach(snapshot.recentArtifacts.prefix(8)) { artifact in
                    HStack(alignment: .top) {
                        VStack(alignment: .leading, spacing: 4) {
                            Text(artifact.name)
                                .font(.subheadline.weight(.semibold))
                                .foregroundStyle(FXAITheme.textPrimary)

                            Text(artifact.category)
                                .font(.caption)
                                .foregroundStyle(FXAITheme.textMuted)
                        }
                        Spacer()
                        Text(FXAIFormatting.byteCountString(for: artifact.sizeBytes))
                            .font(.caption)
                            .foregroundStyle(FXAITheme.textSecondary)
                    }
                    .padding(.vertical, 4)
                }
            }
        }
        .frame(maxWidth: .infinity, alignment: .topLeading)
    }

    private func runtimeProfiles(snapshot: FXAIProjectSnapshot) -> some View {
        FXAIVisualEffectSurface {
            VStack(alignment: .leading, spacing: 12) {
                Text("Deployment Profiles")
                    .font(.headline)
                    .foregroundStyle(FXAITheme.textPrimary)

                if snapshot.runtimeProfiles.isEmpty {
                    Text("No live deployment profiles were discovered in the current ResearchOS output.")
                        .font(.subheadline)
                        .foregroundStyle(FXAITheme.textSecondary)
                } else {
                    ForEach(snapshot.runtimeProfiles.prefix(8)) { profile in
                        VStack(alignment: .leading, spacing: 4) {
                            HStack {
                                Text(profile.symbol)
                                    .font(.subheadline.weight(.semibold))
                                    .foregroundStyle(FXAITheme.textPrimary)
                                Spacer()
                                Text(profile.runtimeMode)
                                    .font(.caption.weight(.semibold))
                                    .foregroundStyle(FXAITheme.accent)
                            }

                            Text("\(profile.pluginName) • \(profile.promotionTier)")
                                .font(.caption)
                                .foregroundStyle(FXAITheme.textSecondary)
                        }
                        .padding(.vertical, 4)
                    }
                }
            }
        }
        .frame(maxWidth: .infinity, alignment: .topLeading)
    }

    private func onboardingPrompt(guide: RoleOnboardingGuide) -> some View {
        FXAIVisualEffectSurface {
            VStack(alignment: .leading, spacing: 12) {
                HStack {
                    VStack(alignment: .leading, spacing: 6) {
                        Text("Next Best Step For \(guide.role.title)")
                            .font(.headline)
                            .foregroundStyle(FXAITheme.textPrimary)
                        Text(guide.headline)
                            .foregroundStyle(FXAITheme.textSecondary)
                    }
                    Spacer()
                    Button("Open Guide") {
                        model.navigate(to: .onboarding)
                    }
                    .buttonStyle(.borderedProminent)
                    .tint(FXAITheme.accent)
                }
            }
        }
    }
}
