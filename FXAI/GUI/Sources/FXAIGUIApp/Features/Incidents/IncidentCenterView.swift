import FXAIGUICore
import SwiftUI

struct IncidentCenterView: View {
    @EnvironmentObject private var model: FXAIGUIModel

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 18) {
                SectionHeader(
                    title: "Incident Center",
                    subtitle: "Generated operator incidents and recovery playbooks built from the actual FXAI build, runtime, promotion, and Research OS state."
                )

                if let incidentSnapshot = model.incidentSnapshot, !incidentSnapshot.incidents.isEmpty {
                    summary(snapshot: incidentSnapshot)

                    HStack(alignment: .top, spacing: 16) {
                        incidentList(snapshot: incidentSnapshot)
                        incidentDetail
                    }
                } else {
                    FXAIVisualEffectSurface {
                        VStack(alignment: .leading, spacing: 14) {
                            Label("No current incidents", systemImage: "checkmark.seal.fill")
                                .font(.headline)
                                .foregroundStyle(FXAITheme.success)
                            Text("The GUI did not detect any build, runtime, promotion, or Research OS incidents from the current FXAI state. Keep using verify-all before release changes.")
                                .foregroundStyle(FXAITheme.textSecondary)
                            CommandPreviewCard(
                                title: "Full Platform Verification",
                                summary: "Run the standard full-stack verification path before you trust the platform as production-ready.",
                                command: verifyCommand,
                                onCopy: { model.copyToPasteboard(verifyCommand) },
                                onTerminal: { model.handoffCommandToTerminal(verifyCommand) }
                            )
                        }
                    }
                }
            }
            .padding(.bottom, 22)
        }
    }

    private var verifyCommand: String {
        guard let projectRoot = model.projectRoot else { return "" }
        return [
            "cd '\(projectRoot.path.replacingOccurrences(of: "'", with: "'\"'\"'"))'",
            "python3 Tools/fxai_testlab.py verify-all"
        ].joined(separator: "\n")
    }

    private var incidentDetail: some View {
        Group {
            if let incident = model.selectedIncident {
                VStack(alignment: .leading, spacing: 16) {
                    FXAIVisualEffectSurface {
                        VStack(alignment: .leading, spacing: 14) {
                            HStack(alignment: .top) {
                                VStack(alignment: .leading, spacing: 6) {
                                    Text(incident.title)
                                        .font(.system(size: 24, weight: .semibold, design: .rounded))
                                        .foregroundStyle(FXAITheme.textPrimary)
                                    Text(incident.summary)
                                        .foregroundStyle(FXAITheme.textSecondary)
                                }
                                Spacer()
                                IncidentSeverityBadge(severity: incident.severity)
                            }

                            if !incident.detailLines.isEmpty {
                                VStack(alignment: .leading, spacing: 8) {
                                    Text("Detected Issues")
                                        .font(.headline)
                                        .foregroundStyle(FXAITheme.textPrimary)
                                    ForEach(incident.detailLines, id: \.self) { line in
                                        Label(line, systemImage: "exclamationmark.circle.fill")
                                            .foregroundStyle(FXAITheme.textSecondary)
                                    }
                                }
                            }

                            if !incident.actions.isEmpty {
                                VStack(alignment: .leading, spacing: 12) {
                                    Text("Immediate Actions")
                                        .font(.headline)
                                        .foregroundStyle(FXAITheme.textPrimary)

                                    ForEach(incident.actions) { action in
                                        CommandPreviewCard(
                                            title: action.title,
                                            summary: action.summary,
                                            command: action.command,
                                            onCopy: { model.copyToPasteboard(action.command) },
                                            onTerminal: {
                                                if let destination = action.destinationSelection,
                                                   let sidebarDestination = SidebarDestination(rawValue: destination) {
                                                    model.navigate(to: sidebarDestination)
                                                }
                                                model.handoffCommandToTerminal(action.command)
                                            }
                                        )
                                    }
                                }
                            }
                        }
                    }

                    RecoveryWizardView(playbook: incident.playbook)
                }
            } else {
                EmptyStateView(
                    title: "No incident selected",
                    message: "Pick an incident to inspect the detailed reasoning and generated recovery steps.",
                    symbolName: "exclamationmark.triangle.fill"
                )
            }
        }
        .frame(maxWidth: .infinity, alignment: .topLeading)
    }

    private func summary(snapshot: IncidentCenterSnapshot) -> some View {
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
                title: "Critical",
                value: "\(snapshot.count(for: .critical))",
                footnote: "Needs operator attention before trusting runtime state.",
                symbolName: "bolt.trianglebadge.exclamationmark.fill",
                tint: FXAITheme.warning
            )
            MetricCard(
                title: "Warnings",
                value: "\(snapshot.count(for: .warning))",
                footnote: "Non-fatal issues that still weaken trust in current FXAI state.",
                symbolName: "exclamationmark.circle.fill",
                tint: FXAITheme.accentSoft
            )
            MetricCard(
                title: "Info",
                value: "\(snapshot.count(for: .info))",
                footnote: "Low-severity guidance and operator context.",
                symbolName: "info.circle.fill",
                tint: FXAITheme.accent
            )
            MetricCard(
                title: "Updated",
                value: FXAIFormatting.dateTimeString(for: snapshot.generatedAt),
                footnote: "Derived from the most recent project refresh.",
                symbolName: "clock.fill",
                tint: FXAITheme.success
            )
        }
    }

    private func incidentList(snapshot: IncidentCenterSnapshot) -> some View {
        FXAIVisualEffectSurface {
            VStack(alignment: .leading, spacing: 12) {
                Text("Detected Incidents")
                    .font(.headline)
                    .foregroundStyle(FXAITheme.textPrimary)

                ForEach(snapshot.incidents) { incident in
                    Button {
                        model.selectedIncidentID = incident.id
                    } label: {
                        HStack(alignment: .top, spacing: 12) {
                            IncidentSeverityDot(severity: incident.severity)
                            VStack(alignment: .leading, spacing: 6) {
                                HStack {
                                    Text(incident.title)
                                        .font(.subheadline.weight(.semibold))
                                        .foregroundStyle(FXAITheme.textPrimary)
                                    Spacer()
                                    Text(incident.category.title)
                                        .font(.caption.weight(.semibold))
                                        .foregroundStyle(FXAITheme.textMuted)
                                }
                                Text(incident.summary)
                                    .font(.caption)
                                    .foregroundStyle(FXAITheme.textSecondary)
                                    .lineLimit(3)

                                if let symbol = incident.affectedSymbol {
                                    Text(symbol)
                                        .font(.caption.weight(.semibold))
                                        .foregroundStyle(FXAITheme.accent)
                                }
                            }
                        }
                        .padding(14)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .background(
                            RoundedRectangle(cornerRadius: 18, style: .continuous)
                                .fill(model.selectedIncidentID == incident.id ? FXAITheme.panelStrong : FXAITheme.backgroundSecondary.opacity(0.62))
                        )
                        .overlay(
                            RoundedRectangle(cornerRadius: 18, style: .continuous)
                                .strokeBorder(model.selectedIncidentID == incident.id ? FXAITheme.accent.opacity(0.32) : FXAITheme.stroke, lineWidth: 1)
                        )
                    }
                    .buttonStyle(.plain)
                }
            }
        }
        .frame(maxWidth: 420, alignment: .topLeading)
    }
}

private struct RecoveryWizardView: View {
    @EnvironmentObject private var model: FXAIGUIModel

    let playbook: RecoveryPlaybook

    var body: some View {
        FXAIVisualEffectSurface {
            VStack(alignment: .leading, spacing: 14) {
                HStack {
                    VStack(alignment: .leading, spacing: 6) {
                        Text(playbook.title)
                            .font(.headline)
                            .foregroundStyle(FXAITheme.textPrimary)
                        Text(playbook.summary)
                            .foregroundStyle(FXAITheme.textSecondary)
                    }
                    Spacer()
                    Label("Recovery Wizard", systemImage: "wand.and.stars.inverse")
                        .font(.caption.weight(.semibold))
                        .foregroundStyle(FXAITheme.accent)
                }

                ForEach(Array(playbook.steps.enumerated()), id: \.element.id) { index, step in
                    VStack(alignment: .leading, spacing: 10) {
                        HStack(alignment: .top) {
                            ZStack {
                                Circle()
                                    .fill(FXAITheme.accentSoft.opacity(0.18))
                                    .frame(width: 30, height: 30)
                                Text("\(index + 1)")
                                    .font(.subheadline.weight(.semibold))
                                    .foregroundStyle(FXAITheme.accentSoft)
                            }
                            VStack(alignment: .leading, spacing: 6) {
                                HStack {
                                    Text(step.title)
                                        .font(.subheadline.weight(.semibold))
                                        .foregroundStyle(FXAITheme.textPrimary)
                                    Spacer()
                                    if let destinationSelection = step.destinationSelection,
                                       let destination = SidebarDestination(rawValue: destinationSelection) {
                                        Button(destination.title) {
                                            model.navigate(to: destination)
                                        }
                                        .buttonStyle(.bordered)
                                    }
                                }

                                Text(step.summary)
                                    .font(.caption)
                                    .foregroundStyle(FXAITheme.textSecondary)

                                HStack {
                                    Button("Copy") {
                                        model.copyToPasteboard(step.command)
                                    }
                                    .buttonStyle(.bordered)

                                    Button("Open In Terminal") {
                                        model.handoffCommandToTerminal(step.command)
                                    }
                                    .buttonStyle(.borderedProminent)
                                    .tint(FXAITheme.accent)
                                }

                                Text(step.command)
                                    .font(.system(.caption, design: .monospaced))
                                    .foregroundStyle(FXAITheme.textMuted)
                                    .textSelection(.enabled)
                                    .padding(12)
                                    .frame(maxWidth: .infinity, alignment: .leading)
                                    .background(
                                        RoundedRectangle(cornerRadius: 16, style: .continuous)
                                            .fill(FXAITheme.backgroundSecondary.opacity(0.72))
                                    )
                            }
                        }
                    }
                    .padding(14)
                    .background(
                        RoundedRectangle(cornerRadius: 18, style: .continuous)
                            .fill(FXAITheme.backgroundSecondary.opacity(0.48))
                    )
                }
            }
        }
    }
}

private struct IncidentSeverityBadge: View {
    let severity: IncidentSeverity

    var body: some View {
        Text(severity.title)
            .font(.caption.weight(.semibold))
            .foregroundStyle(severityColor)
            .padding(.horizontal, 12)
            .padding(.vertical, 8)
            .background(
                Capsule(style: .continuous)
                    .fill(severityColor.opacity(0.14))
            )
    }

    private var severityColor: Color {
        switch severity {
        case .info:
            FXAITheme.accentSoft
        case .warning:
            FXAITheme.warning
        case .critical:
            Color.red.opacity(0.92)
        }
    }
}

private struct IncidentSeverityDot: View {
    let severity: IncidentSeverity

    var body: some View {
        Circle()
            .fill(color)
            .frame(width: 10, height: 10)
            .padding(.top, 6)
    }

    private var color: Color {
        switch severity {
        case .info:
            FXAITheme.accentSoft
        case .warning:
            FXAITheme.warning
        case .critical:
            Color.red.opacity(0.92)
        }
    }
}
