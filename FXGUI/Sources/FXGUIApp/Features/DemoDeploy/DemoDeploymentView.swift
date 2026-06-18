import FXGUICore
import SwiftUI

struct DemoDeploymentView: View {
    @EnvironmentObject private var model: FXGUIModel

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 18) {
                SectionHeader(
                    title: "Demo Deployments",
                    subtitle: "Monitor demo-mode deployments, their health indicators, and deployment state from a single operations view."
                )

                if let snapshot = model.demoDeploymentSnapshot, !snapshot.deployments.isEmpty {
                    summary(snapshot: snapshot)
                    deploymentGrid(snapshot: snapshot)
                    selectedDeploymentDetail
                } else {
                    EmptyStateView(
                        title: "No demo deployments",
                        message: "Demo deployments are detected from runtime deployment profiles with demo mode or demo promotion tier.",
                        symbolName: "play.rectangle"
                    )
                }
            }
        }
    }

    private func summary(snapshot: DemoDeploymentSnapshot) -> some View {
        LazyVGrid(
            columns: [
                GridItem(.flexible(minimum: 200), spacing: 16),
                GridItem(.flexible(minimum: 200), spacing: 16),
                GridItem(.flexible(minimum: 200), spacing: 16)
            ],
            spacing: 16
        ) {
            MetricCard(
                title: "Total Deployments",
                value: "\(snapshot.deployments.count)",
                footnote: "Demo-mode deployments detected.",
                symbolName: "play.rectangle.fill",
                tint: FXAITheme.accent
            )
            MetricCard(
                title: "Active",
                value: "\(snapshot.activeCount)",
                footnote: "Currently active demo deployments.",
                symbolName: "circle.fill",
                tint: FXAITheme.success
            )
            MetricCard(
                title: "Issues",
                value: "\(snapshot.deployments.filter { $0.status == .error }.count)",
                footnote: "Deployments reporting errors.",
                symbolName: "exclamationmark.triangle.fill",
                tint: FXAITheme.warning
            )
        }
    }

    private func deploymentGrid(snapshot: DemoDeploymentSnapshot) -> some View {
        LazyVGrid(
            columns: [
                GridItem(.flexible(minimum: 340), spacing: 16),
                GridItem(.flexible(minimum: 340), spacing: 16)
            ],
            spacing: 16
        ) {
            ForEach(snapshot.deployments) { deployment in
                deploymentCard(deployment: deployment)
            }
        }
    }

    private func deploymentCard(deployment: DemoDeployment) -> some View {
        Button {
            model.selectedDemoDeploymentID = deployment.id
        } label: {
            FXAIVisualEffectSurface {
                VStack(alignment: .leading, spacing: 10) {
                    HStack {
                        Image(systemName: deployment.status.symbolName)
                            .foregroundStyle(deploymentStatusColor(deployment.status))
                        VStack(alignment: .leading, spacing: 2) {
                            Text(deployment.symbol)
                                .font(.headline)
                                .foregroundStyle(FXAITheme.textPrimary)
                            Text(deployment.pluginName)
                                .font(.caption)
                                .foregroundStyle(FXAITheme.textSecondary)
                        }
                        Spacer()
                        StatusBadge(
                            title: "Status",
                            value: deployment.status.title,
                            tint: deploymentStatusColor(deployment.status)
                        )
                    }

                    HStack(spacing: 8) {
                        Label(deployment.promotionTier, systemImage: "rosette")
                            .font(.caption2)
                            .foregroundStyle(FXAITheme.accentSoft)
                        Label(deployment.runtimeMode, systemImage: "dial.low.fill")
                            .font(.caption2)
                            .foregroundStyle(FXAITheme.textMuted)
                    }

                    if !deployment.healthIndicators.isEmpty {
                        LazyVGrid(
                            columns: [GridItem(.adaptive(minimum: 120), spacing: 6)],
                            spacing: 6
                        ) {
                            ForEach(deployment.healthIndicators) { indicator in
                                HStack(spacing: 4) {
                                    Circle()
                                        .fill(indicator.healthy ? FXAITheme.success : .red)
                                        .frame(width: 6, height: 6)
                                    Text("\(indicator.name): \(indicator.value)")
                                        .font(.caption2)
                                        .foregroundStyle(FXAITheme.textSecondary)
                                }
                            }
                        }
                    }
                }
            }
        }
        .buttonStyle(.plain)
    }

    @ViewBuilder
    private var selectedDeploymentDetail: some View {
        if let deployment = model.selectedDemoDeployment {
            FXAIVisualEffectSurface {
                VStack(alignment: .leading, spacing: 14) {
                    HStack {
                        Text("\(deployment.symbol) — \(deployment.pluginName)")
                            .font(.system(size: 20, weight: .semibold, design: .rounded))
                            .foregroundStyle(FXAITheme.textPrimary)
                        Spacer()
                        StatusBadge(
                            title: "Status",
                            value: deployment.status.title,
                            tint: deploymentStatusColor(deployment.status)
                        )
                    }

                    LazyVGrid(
                        columns: [
                            GridItem(.flexible(minimum: 180), spacing: 12),
                            GridItem(.flexible(minimum: 180), spacing: 12),
                            GridItem(.flexible(minimum: 180), spacing: 12)
                        ],
                        spacing: 12
                    ) {
                        MetricCard(
                            title: "Tier",
                            value: deployment.promotionTier,
                            footnote: "Promotion tier for this deployment.",
                            symbolName: "rosette",
                            tint: FXAITheme.accent
                        )
                        MetricCard(
                            title: "Mode",
                            value: deployment.runtimeMode,
                            footnote: "Runtime mode of the deployment.",
                            symbolName: "dial.low.fill",
                            tint: FXAITheme.accentSoft
                        )
                        MetricCard(
                            title: "Started",
                            value: deployment.startedAt.map { FXAIFormatting.dateTimeString(for: $0) } ?? "N/A",
                            footnote: "Deployment start time.",
                            symbolName: "clock.fill",
                            tint: FXAITheme.success
                        )
                    }

                    if !deployment.healthIndicators.isEmpty {
                        Text("Health Indicators")
                            .font(.subheadline.weight(.semibold))
                            .foregroundStyle(FXAITheme.textPrimary)

                        ForEach(deployment.healthIndicators) { indicator in
                            HStack(spacing: 8) {
                                Image(systemName: indicator.healthy ? "checkmark.circle.fill" : "xmark.circle.fill")
                                    .foregroundStyle(indicator.healthy ? FXAITheme.success : .red)
                                Text(indicator.name)
                                    .font(.caption.weight(.semibold))
                                    .foregroundStyle(FXAITheme.textPrimary)
                                Spacer()
                                Text(indicator.value)
                                    .font(.caption)
                                    .foregroundStyle(indicator.healthy ? FXAITheme.success : .red)
                            }
                            .padding(.vertical, 4)
                        }
                    }
                }
            }
        }
    }

    private func deploymentStatusColor(_ status: DemoDeploymentStatus) -> Color {
        switch status {
        case .active: return FXAITheme.success
        case .paused: return FXAITheme.warning
        case .stopped: return FXAITheme.textMuted
        case .error: return .red
        }
    }
}
