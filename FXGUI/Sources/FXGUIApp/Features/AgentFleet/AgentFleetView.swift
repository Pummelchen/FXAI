import FXGUICore
import SwiftUI

struct AgentFleetView: View {
    @EnvironmentObject private var model: FXGUIModel

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 18) {
                SectionHeader(
                    title: "Agent Fleet Status",
                    subtitle: "Monitor the health, heartbeat, and capabilities of all FXAI agents from a single operations console."
                )

                if let fleet = model.agentFleetSnapshot {
                    summary(fleet: fleet)
                    agentGrid(fleet: fleet)
                } else {
                    EmptyStateView(
                        title: "No agent fleet data",
                        message: "Connect to an FXAI project to see agent fleet status.",
                        symbolName: "cpu"
                    )
                }
            }
        }
    }

    private func summary(fleet: AgentFleetSnapshot) -> some View {
        LazyVGrid(
            columns: [
                GridItem(.flexible(minimum: 200), spacing: 16),
                GridItem(.flexible(minimum: 200), spacing: 16),
                GridItem(.flexible(minimum: 200), spacing: 16),
                GridItem(.flexible(minimum: 200), spacing: 16)
            ],
            spacing: 16
        ) {
            MetricCard(
                title: "Total Agents",
                value: "\(fleet.agents.count)",
                footnote: "Agents registered in the FXAI fleet.",
                symbolName: "cpu.fill",
                tint: FXAITheme.accent
            )
            MetricCard(
                title: "Healthy",
                value: "\(fleet.healthyCount)",
                footnote: "Agents with active or idle status.",
                symbolName: "checkmark.circle.fill",
                tint: FXAITheme.success
            )
            MetricCard(
                title: "Unhealthy",
                value: "\(fleet.unhealthyCount)",
                footnote: "Agents with failed, stopped, or unknown status.",
                symbolName: "xmark.octagon.fill",
                tint: fleet.unhealthyCount > 0 ? FXAITheme.warning : FXAITheme.success
            )
            MetricCard(
                title: "Fleet Health",
                value: fleet.agents.isEmpty ? "N/A" : "\(Int(Double(fleet.healthyCount) / Double(fleet.agents.count) * 100))%",
                footnote: "Percentage of agents in healthy state.",
                symbolName: "heart.fill",
                tint: fleet.healthyCount == fleet.agents.count ? FXAITheme.success : FXAITheme.warning
            )
        }
    }

    private func agentGrid(fleet: AgentFleetSnapshot) -> some View {
        LazyVGrid(
            columns: [
                GridItem(.flexible(minimum: 340), spacing: 16),
                GridItem(.flexible(minimum: 340), spacing: 16)
            ],
            spacing: 16
        ) {
            ForEach(fleet.agents) { agent in
                agentCard(agent: agent)
            }
        }
    }

    private func agentCard(agent: AgentFleetMember) -> some View {
        FXAIVisualEffectSurface {
            VStack(alignment: .leading, spacing: 12) {
                HStack {
                    Image(systemName: agent.status.symbolName)
                        .font(.title2)
                        .foregroundStyle(statusColor(agent.status))
                    VStack(alignment: .leading, spacing: 2) {
                        Text(agent.name)
                            .font(.headline)
                            .foregroundStyle(FXAITheme.textPrimary)
                        Text(agent.role)
                            .font(.caption)
                            .foregroundStyle(FXAITheme.textSecondary)
                    }
                    Spacer()
                    StatusBadge(
                        title: "Status",
                        value: agent.status.title,
                        tint: statusColor(agent.status)
                    )
                }

                if let heartbeat = agent.lastHeartbeat {
                    HStack(spacing: 6) {
                        Image(systemName: "heart.fill")
                            .font(.caption2)
                            .foregroundStyle(FXAITheme.success)
                        Text("Last heartbeat: \(FXAIFormatting.dateTimeString(for: heartbeat))")
                            .font(.caption2)
                            .foregroundStyle(FXAITheme.textMuted)
                    }
                } else {
                    HStack(spacing: 6) {
                        Image(systemName: "heart.slash")
                            .font(.caption2)
                            .foregroundStyle(FXAITheme.textMuted)
                        Text("No heartbeat recorded")
                            .font(.caption2)
                            .foregroundStyle(FXAITheme.textMuted)
                    }
                }

                if !agent.assignedSymbols.isEmpty {
                    VStack(alignment: .leading, spacing: 4) {
                        Text("Assigned Symbols")
                            .font(.caption.weight(.semibold))
                            .foregroundStyle(FXAITheme.textSecondary)
                        HStack(spacing: 4) {
                            ForEach(agent.assignedSymbols.prefix(8), id: \.self) { symbol in
                                Text(symbol)
                                    .font(.caption2.weight(.semibold))
                                    .foregroundStyle(FXAITheme.accent)
                                    .padding(.horizontal, 6)
                                    .padding(.vertical, 2)
                                    .background(Capsule().fill(FXAITheme.accent.opacity(0.12)))
                            }
                            if agent.assignedSymbols.count > 8 {
                                Text("+\(agent.assignedSymbols.count - 8)")
                                    .font(.caption2)
                                    .foregroundStyle(FXAITheme.textMuted)
                            }
                        }
                    }
                }

                if !agent.capabilities.isEmpty {
                    VStack(alignment: .leading, spacing: 4) {
                        Text("Capabilities")
                            .font(.caption.weight(.semibold))
                            .foregroundStyle(FXAITheme.textSecondary)
                        ForEach(agent.capabilities, id: \.self) { capability in
                            Label(capability, systemImage: "checkmark.circle.fill")
                                .font(.caption2)
                                .foregroundStyle(FXAITheme.textMuted)
                        }
                    }
                }
            }
        }
    }

    private func statusColor(_ status: AgentStatus) -> Color {
        switch status {
        case .active:
            return FXAITheme.success
        case .idle:
            return FXAITheme.accentSoft
        case .failed:
            return .red
        case .starting:
            return FXAITheme.warning
        case .stopped:
            return FXAITheme.textMuted
        case .unknown:
            return FXAITheme.textMuted
        }
    }
}
