import FXGUICore
import SwiftUI

struct KillSwitchView: View {
    @EnvironmentObject private var model: FXGUIModel

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 18) {
                SectionHeader(
                    title: "Kill Switch Controls",
                    subtitle: "Inspect and activate execution safety switches before order-capable workflows. All switches are operator-controlled and auditable."
                )

                overallStatus

                switchGrid

                safetyGuidance

                auditTrail
            }
        }
    }

    private var overallStatus: some View {
        FXAIVisualEffectSurface {
            HStack(spacing: 16) {
                ZStack {
                    Circle()
                        .fill(model.killSwitchState.isArmed ? Color.red.opacity(0.18) : FXAITheme.success.opacity(0.18))
                        .frame(width: 64, height: 64)
                    Image(systemName: model.killSwitchState.isArmed ? "power.circle.fill" : "checkmark.seal.fill")
                        .font(.system(size: 28))
                        .foregroundStyle(model.killSwitchState.isArmed ? Color.red : FXAITheme.success)
                }

                VStack(alignment: .leading, spacing: 4) {
                    Text(model.killSwitchState.statusTitle)
                        .font(.system(size: 22, weight: .semibold, design: .rounded))
                        .foregroundStyle(model.killSwitchState.isArmed ? Color.red : FXAITheme.success)
                    Text(model.killSwitchState.isArmed
                         ? "One or more execution kill switches are active. Live order flow is blocked."
                         : "All execution paths are clear. No kill switches are armed.")
                        .font(.caption)
                        .foregroundStyle(FXAITheme.textSecondary)
                }

                Spacer()

                Text("Checked \(FXAIFormatting.dateTimeString(for: model.killSwitchState.lastCheckedAt))")
                    .font(.caption2)
                    .foregroundStyle(FXAITheme.textMuted)
            }
        }
    }

    private var switchGrid: some View {
        LazyVGrid(
            columns: [
                GridItem(.flexible(minimum: 280), spacing: 16),
                GridItem(.flexible(minimum: 280), spacing: 16),
                GridItem(.flexible(minimum: 280), spacing: 16)
            ],
            spacing: 16
        ) {
            killSwitchCard(
                title: "Global Kill Switch",
                description: "Blocks all execution across all accounts and symbols.",
                isActive: !model.killSwitchState.globalEnabled,
                component: .global,
                symbolName: "globe",
                riskLabel: "Highest Risk"
            )

            killSwitchCard(
                title: "Account Kill Switch",
                description: "Blocks execution for the connected account only.",
                isActive: !model.killSwitchState.accountEnabled,
                component: .account,
                symbolName: "person.crop.circle.badge.xmark",
                riskLabel: "High Risk"
            )

            killSwitchCard(
                title: "Symbol Kill Switch",
                description: "Blocks execution for the selected symbol only.",
                isActive: !model.killSwitchState.symbolEnabled,
                component: .symbol,
                symbolName: "chart.bar.xaxis",
                riskLabel: "Moderate Risk"
            )
        }
    }

    private func killSwitchCard(
        title: String,
        description: String,
        isActive: Bool,
        component: FXGUIModel.KillSwitchComponent,
        symbolName: String,
        riskLabel: String
    ) -> some View {
        FXAIVisualEffectSurface {
            VStack(alignment: .leading, spacing: 12) {
                HStack {
                    Image(systemName: symbolName)
                        .font(.title2)
                        .foregroundStyle(isActive ? Color.red : FXAITheme.success)
                    VStack(alignment: .leading, spacing: 2) {
                        Text(title)
                            .font(.headline)
                            .foregroundStyle(FXAITheme.textPrimary)
                        Text(riskLabel)
                            .font(.caption2.weight(.semibold))
                            .foregroundStyle(isActive ? Color.red : FXAITheme.textMuted)
                    }
                    Spacer()
                    StatusBadge(
                        title: "State",
                        value: isActive ? "ARMED" : "CLEAR",
                        tint: isActive ? Color.red : FXAITheme.success
                    )
                }

                Text(description)
                    .font(.caption)
                    .foregroundStyle(FXAITheme.textSecondary)

                HStack {
                    Button {
                        model.toggleKillSwitch(component: component)
                    } label: {
                        Label(
                            isActive ? "Disarm" : "Arm Kill Switch",
                            systemImage: isActive ? "checkmark.shield" : "power.circle"
                        )
                    }
                    .buttonStyle(.borderedProminent)
                    .tint(isActive ? FXAITheme.success : Color.red)

                    if isActive {
                        Text("Active — execution blocked")
                            .font(.caption.weight(.semibold))
                            .foregroundStyle(Color.red)
                    }
                }
            }
        }
    }

    private var safetyGuidance: some View {
        FXAIVisualEffectSurface {
            VStack(alignment: .leading, spacing: 10) {
                Text("Safety Guidance")
                    .font(.headline)
                    .foregroundStyle(FXAITheme.textPrimary)
                Label("Always verify kill-switch state before initiating any live order-capable workflow.", systemImage: "exclamationmark.shield.fill")
                    .font(.caption)
                    .foregroundStyle(FXAITheme.textSecondary)
                Label("Kill-switch arming is immediate and blocks execution until explicitly disarmed.", systemImage: "clock.badge.checkmark.fill")
                    .font(.caption)
                    .foregroundStyle(FXAITheme.textSecondary)
                Label("All kill-switch state changes are recorded in the audit trail below.", systemImage: "doc.text.fill")
                    .font(.caption)
                    .foregroundStyle(FXAITheme.textSecondary)
            }
        }
    }

    private var auditTrail: some View {
        FXAIVisualEffectSurface {
            VStack(alignment: .leading, spacing: 10) {
                Text("Current State Audit")
                    .font(.headline)
                    .foregroundStyle(FXAITheme.textPrimary)

                HStack(spacing: 16) {
                    auditLine(label: "Global", active: !model.killSwitchState.globalEnabled)
                    auditLine(label: "Account", active: !model.killSwitchState.accountEnabled)
                    auditLine(label: "Symbol", active: !model.killSwitchState.symbolEnabled)
                }

                Text("The FXExecutionContracts kill-switch API at /v1/execution/kill-switch/status is defined but pending a live FXDatabase endpoint. State is currently managed locally in the GUI session.")
                    .font(.caption2)
                    .foregroundStyle(FXAITheme.textMuted)
            }
        }
    }

    private func auditLine(label: String, active: Bool) -> some View {
        HStack(spacing: 6) {
            Circle()
                .fill(active ? Color.red : FXAITheme.success)
                .frame(width: 8, height: 8)
            Text("\(label): \(active ? "ARMED" : "Clear")")
                .font(.caption.weight(.semibold))
                .foregroundStyle(active ? Color.red : FXAITheme.textSecondary)
        }
    }
}
