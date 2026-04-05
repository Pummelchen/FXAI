import Charts
import FXAIGUICore
import SwiftUI

struct RuntimeMonitorView: View {
    @EnvironmentObject private var model: FXAIGUIModel

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 18) {
                SectionHeader(
                    title: "Runtime Monitor",
                    subtitle: "Inspect deployed symbol state, promoted profiles, and runtime health without opening raw artifacts manually."
                )

                if let runtimeSnapshot = model.runtimeSnapshot, !runtimeSnapshot.deployments.isEmpty {
                    ViewThatFits(in: .horizontal) {
                        HStack {
                            Picker("Symbol", selection: $model.selectedRuntimeSymbol) {
                                ForEach(runtimeSnapshot.symbols, id: \.self) { symbol in
                                    Text(symbol).tag(symbol)
                                }
                            }
                            .pickerStyle(.segmented)

                            Spacer()
                        }
                        HStack {
                            Picker("Symbol", selection: $model.selectedRuntimeSymbol) {
                                ForEach(runtimeSnapshot.symbols, id: \.self) { symbol in
                                    Text(symbol).tag(symbol)
                                }
                            }
                            .pickerStyle(.menu)
                            Spacer()
                        }
                    }

                    if let detail = model.selectedRuntimeDetail {
                        summary(detail: detail)
                        metricChart(detail: detail)

                        LazyVGrid(
                            columns: [
                                GridItem(.flexible(minimum: 320), spacing: 16),
                                GridItem(.flexible(minimum: 320), spacing: 16)
                            ],
                            spacing: 16
                        ) {
                            ForEach(detail.deploymentSections) { section in
                                ArtifactSectionCard(section: section, onReveal: model.openInFinder)
                            }
                            ForEach(detail.routerSections) { section in
                                ArtifactSectionCard(section: section, onReveal: model.openInFinder)
                            }
                            ForEach(detail.supervisorSections) { section in
                                ArtifactSectionCard(section: section, onReveal: model.openInFinder)
                            }
                            ForEach(detail.commandSections) { section in
                                ArtifactSectionCard(section: section, onReveal: model.openInFinder)
                            }
                            ForEach(detail.worldSections) { section in
                                ArtifactSectionCard(section: section, onReveal: model.openInFinder)
                            }
                            ForEach(detail.attributionSections) { section in
                                ArtifactSectionCard(section: section, onReveal: model.openInFinder)
                            }
                        }
                    }
                } else {
                    EmptyStateView(
                        title: "No runtime artifacts found",
                        message: "Run Offline Lab promotion and deployment flows first so the GUI can inspect live deployment, router, supervisor, and world-plan state.",
                        symbolName: "waveform.path.ecg.rectangle"
                    )
                }
            }
        }
    }

    private func summary(detail: RuntimeDeploymentDetail) -> some View {
        VStack(alignment: .leading, spacing: 14) {
            ViewThatFits(in: .horizontal) {
                HStack(spacing: 12) {
                    runtimeHealthBadges(detail: detail)
                }
                LazyVGrid(
                    columns: [GridItem(.adaptive(minimum: 180), spacing: 12, alignment: .leading)],
                    spacing: 12
                ) {
                    runtimeHealthBadges(detail: detail)
                }
            }

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
                    title: "Plugin",
                    value: detail.pluginName,
                    footnote: "Champion currently associated with this deployment.",
                    symbolName: "brain.head.profile",
                    tint: FXAITheme.accent
                )
                MetricCard(
                    title: "Tier",
                    value: detail.promotionTier,
                    footnote: "Promotion level emitted by the research OS.",
                    symbolName: "rosette",
                    tint: FXAITheme.warning
                )
                MetricCard(
                    title: "Mode",
                    value: detail.runtimeMode,
                    footnote: "Current runtime mode loaded by the deployment profile.",
                    symbolName: "dial.low.fill",
                    tint: FXAITheme.accentSoft
                )
                MetricCard(
                    title: "Artifact Age",
                    value: "\(detail.artifactHealth.artifactAgeSeconds)s",
                    footnote: detail.artifactHealth.staleArtifact ? "Marked stale" : "Fresh according to dashboard health.",
                    symbolName: "clock.fill",
                    tint: detail.artifactHealth.staleArtifact ? FXAITheme.warning : FXAITheme.success
                )
            }
        }
    }

    private func metricChart(detail: RuntimeDeploymentDetail) -> some View {
        FXAIVisualEffectSurface {
            VStack(alignment: .leading, spacing: 12) {
                Text("Runtime Highlights")
                    .font(.headline)
                    .foregroundStyle(FXAITheme.textPrimary)

                if detail.summaryMetrics.isEmpty {
                    Text("No numeric runtime highlights were found for this deployment.")
                        .foregroundStyle(FXAITheme.textSecondary)
                } else {
                    Chart(detail.summaryMetrics.prefix(10)) { metric in
                        if let value = metric.numericValue {
                            BarMark(
                                x: .value("Metric", metric.key),
                                y: .value("Value", value)
                            )
                            .foregroundStyle(FXAITheme.accent.gradient)
                            .cornerRadius(5)
                        }
                    }
                    .chartYAxis {
                        AxisMarks(position: .leading)
                    }
                    .frame(height: 240)
                }
            }
        }
    }

    private func runtimeHealthBadges(detail: RuntimeDeploymentDetail) -> some View {
        Group {
            StatusBadge(
                title: "Deployment",
                value: detail.artifactHealth.missingDeployment ? "Missing" : "Ready",
                tint: detail.artifactHealth.missingDeployment ? FXAITheme.warning : FXAITheme.success
            )
            StatusBadge(
                title: "Router",
                value: detail.artifactHealth.missingRouter ? "Missing" : "Ready",
                tint: detail.artifactHealth.missingRouter ? FXAITheme.warning : FXAITheme.success
            )
            StatusBadge(
                title: "Supervisor",
                value: detail.artifactHealth.missingSupervisorService ? "Missing" : "Ready",
                tint: detail.artifactHealth.missingSupervisorService ? FXAITheme.warning : FXAITheme.success
            )
            StatusBadge(
                title: "World Plan",
                value: detail.artifactHealth.missingWorldPlan ? "Missing" : "Ready",
                tint: detail.artifactHealth.missingWorldPlan ? FXAITheme.warning : FXAITheme.success
            )
        }
    }
}
