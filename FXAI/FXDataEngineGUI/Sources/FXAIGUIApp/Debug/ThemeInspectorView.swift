import FXAIGUICore
import SwiftUI

struct ThemeInspectorView: View {
    @EnvironmentObject private var themeEnvironment: ThemeEnvironment

    let layoutOutput: DashboardLayoutOutput?

    var body: some View {
        let theme = themeEnvironment.currentTheme

        VStack(alignment: .leading, spacing: 12) {
            Text("Theme Inspector")
                .font(.system(size: 13, weight: .semibold, design: .rounded))
                .foregroundStyle(theme.colors.textPrimary)

            Picker("Theme", selection: Binding(
                get: { themeEnvironment.selectedThemeID },
                set: { themeEnvironment.activateTheme($0) }
            )) {
                ForEach(themeEnvironment.allThemes, id: \.themeID) { candidate in
                    Text(candidate.displayName).tag(candidate.themeID)
                }
            }
            .pickerStyle(.menu)

            HStack(spacing: 8) {
                swatch(theme.colors.mainPanel, label: "Panel")
                swatch(theme.colors.warningGreen, label: "Accent")
                swatch(theme.colors.successGreen, label: "Success")
                swatch(theme.colors.textPrimary, label: "Text")
            }

            if let layoutOutput {
                VStack(alignment: .leading, spacing: 4) {
                    inspectorLine("Layout", value: layoutOutput.frameModel.layoutClass.displayName)
                    inspectorLine("KPI Arrangement", value: layoutOutput.kpiArrangement.rawValue)
                    inspectorLine("Chart Placement", value: layoutOutput.chartPlacement.rawValue)
                    inspectorLine("Typography Scale", value: String(format: "%.3f", layoutOutput.typographyScale))
                    inspectorLine("Spacing Scale", value: String(format: "%.3f", layoutOutput.spacingScale))
                }
            }

            VStack(alignment: .leading, spacing: 4) {
                inspectorLine("Stage", value: renderLine(theme.renderingPolicy.policy(for: .stage)))
                inspectorLine("KPI", value: renderLine(theme.renderingPolicy.policy(for: .kpiCard)))
                inspectorLine("Gauge", value: renderLine(theme.renderingPolicy.policy(for: .gaugeCard)))
                inspectorLine("Chart", value: renderLine(theme.renderingPolicy.policy(for: .barChart)))
                inspectorLine("Glass", value: renderLine(theme.renderingPolicy.policy(for: .amountOwedGlassCard)))
            }
        }
        .padding(16)
        .background(
            RoundedRectangle(cornerRadius: 18, style: .continuous)
                .fill(theme.colors.mainPanel.opacity(0.88))
        )
        .overlay(
            RoundedRectangle(cornerRadius: 18, style: .continuous)
                .stroke(theme.colors.cardStroke.opacity(0.88), lineWidth: 1)
        )
    }

    private func swatch(_ color: Color, label: String) -> some View {
        VStack(spacing: 5) {
            RoundedRectangle(cornerRadius: 8, style: .continuous)
                .fill(color)
                .frame(width: 28, height: 28)
            Text(label)
                .font(.system(size: 10, weight: .medium, design: .rounded))
                .foregroundStyle(themeEnvironment.currentTheme.colors.textMuted)
        }
    }

    private func renderLine(_ policy: ComponentRenderingPolicy) -> String {
        "\(policy.preferredTier.rawValue) → \(policy.fallbackTier.rawValue)"
    }

    private func inspectorLine(_ title: String, value: String) -> some View {
        HStack {
            Text(title)
                .font(.system(size: 11, weight: .medium, design: .rounded))
                .foregroundStyle(themeEnvironment.currentTheme.colors.textMuted)
            Spacer()
            Text(value)
                .font(.system(size: 11, weight: .semibold, design: .rounded))
                .foregroundStyle(themeEnvironment.currentTheme.colors.textSecondary)
        }
    }
}
