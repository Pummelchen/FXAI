import Charts
import FXAIGUICore
import SwiftUI

struct DynamicEnsembleView: View {
    @EnvironmentObject private var model: FXAIGUIModel

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 18) {
                SectionHeader(
                    title: "Dynamic Ensemble",
                    subtitle: "Inspect the post-inference meta-layer that now reweights, downweights, or suppresses plugin outputs based on live context, calibration, and recent usefulness."
                )

                if let snapshot = model.dynamicEnsembleSnapshot, !snapshot.symbols.isEmpty {
                    symbolPicker(snapshot: snapshot)
                    if let detail = model.selectedDynamicEnsembleDetail {
                        statusPanel(detail: detail)
                        distributionPanel(detail: detail)
                        pluginPanel(detail: detail)
                        replayPanel(detail: detail, replayHoursBack: snapshot.replayHoursBack)
                    }
                } else {
                    EmptyStateView(
                        title: "Dynamic Ensemble artifacts not found",
                        message: "Run dynamic-ensemble-validate so the runtime config is exported, let the live runtime publish state, and rebuild the replay report to inspect the current ensemble controller.",
                        symbolName: "dial.high.fill"
                    )
                }
            }
        }
    }

    private func symbolPicker(snapshot: DynamicEnsembleSnapshot) -> some View {
        ViewThatFits(in: .horizontal) {
            HStack {
                Picker("Symbol", selection: $model.selectedDynamicEnsembleSymbol) {
                    ForEach(snapshot.symbols, id: \.symbol) { detail in
                        Text(detail.symbol).tag(detail.symbol)
                    }
                }
                .pickerStyle(.segmented)
                Spacer()
            }

            HStack {
                Picker("Symbol", selection: $model.selectedDynamicEnsembleSymbol) {
                    ForEach(snapshot.symbols, id: \.symbol) { detail in
                        Text(detail.symbol).tag(detail.symbol)
                    }
                }
                .pickerStyle(.menu)
                Spacer()
            }
        }
    }

    private func statusPanel(detail: DynamicEnsembleSymbolSnapshot) -> some View {
        LazyVGrid(columns: [GridItem(.adaptive(minimum: 180), spacing: 16)], spacing: 16) {
            MetricCard(
                title: "Posture",
                value: detail.tradePosture,
                footnote: detail.generatedAt.map { "Live state \($0.formatted(date: .omitted, time: .shortened))" } ?? "Runtime timestamp unavailable",
                symbolName: postureSymbol(detail.tradePosture),
                tint: postureTint(detail.tradePosture)
            )
            MetricCard(
                title: "Quality",
                value: percentString(detail.ensembleQuality),
                footnote: "Agreement \(percentString(detail.agreementScore)) • Context fit \(percentString(detail.contextFitScore))",
                symbolName: "dial.high.fill",
                tint: detail.ensembleQuality >= 0.58 ? FXAITheme.success : FXAITheme.warning
            )
            MetricCard(
                title: "Abstain Bias",
                value: percentString(detail.abstainBias),
                footnote: detail.fallbackUsed ? "Fallback to routed ensemble was used." : "Lower is more permissive.",
                symbolName: "pause.circle.fill",
                tint: detail.fallbackUsed ? FXAITheme.warning : FXAITheme.accentSoft
            )
            MetricCard(
                title: "Final Action",
                value: detail.finalAction,
                footnote: "\(detail.topRegime.replacingOccurrences(of: "_", with: " ")) • \(detail.sessionLabel.replacingOccurrences(of: "_", with: " "))",
                symbolName: finalActionSymbol(detail.finalAction),
                tint: finalActionTint(detail.finalAction)
            )
            MetricCard(
                title: "Distribution",
                value: "B \(percentString(detail.buyProb)) • S \(percentString(detail.sellProb))",
                footnote: "Skip \(percentString(detail.skipProb)) • Score \(String(format: "%.2f", detail.finalScore))",
                symbolName: "chart.bar.fill",
                tint: FXAITheme.accent
            )
            MetricCard(
                title: "Dominance",
                value: percentString(detail.dominantPluginShare),
                footnote: "\(detail.activePlugins.count) active • \(detail.downweightedPlugins.count) reduced • \(detail.suppressedPlugins.count) suppressed",
                symbolName: "person.2.slash.fill",
                tint: detail.dominantPluginShare >= 0.58 ? FXAITheme.warning : FXAITheme.accentSoft
            )
        }
    }

    private func distributionPanel(detail: DynamicEnsembleSymbolSnapshot) -> some View {
        let distribution = [
            KeyValueRecord(key: "BUY", value: String(format: "%.4f", detail.buyProb)),
            KeyValueRecord(key: "SELL", value: String(format: "%.4f", detail.sellProb)),
            KeyValueRecord(key: "SKIP", value: String(format: "%.4f", detail.skipProb)),
        ]

        return FXAIVisualEffectSurface {
            VStack(alignment: .leading, spacing: 14) {
                Text("Decision Shape")
                    .font(.headline)
                    .foregroundStyle(FXAITheme.textPrimary)

                Chart(distribution) { item in
                    BarMark(
                        x: .value("Side", item.key),
                        y: .value("Probability", Double(item.value) ?? 0)
                    )
                    .foregroundStyle(barTint(item.key).gradient)
                    .cornerRadius(6)
                }
                .chartYAxis { AxisMarks(position: .leading) }
                .frame(height: 220)

                VStack(alignment: .leading, spacing: 8) {
                    Text("Reasons")
                        .font(.subheadline.weight(.semibold))
                        .foregroundStyle(FXAITheme.textPrimary)
                    ForEach(detail.reasons, id: \.self) { reason in
                        HStack(alignment: .top, spacing: 8) {
                            Image(systemName: "circle.fill")
                                .font(.system(size: 5))
                                .foregroundStyle(postureTint(detail.tradePosture))
                                .padding(.top, 6)
                            Text(reason)
                                .font(.caption)
                                .foregroundStyle(FXAITheme.textMuted)
                        }
                    }
                }
            }
        }
    }

    private func pluginPanel(detail: DynamicEnsembleSymbolSnapshot) -> some View {
        FXAIVisualEffectSurface {
            VStack(alignment: .leading, spacing: 14) {
                Text("Plugin Participation")
                    .font(.headline)
                    .foregroundStyle(FXAITheme.textPrimary)

                ViewThatFits(in: .horizontal) {
                    HStack(alignment: .top, spacing: 16) {
                        pluginColumn(title: "Active", plugins: detail.activePlugins, tint: FXAITheme.success)
                        pluginColumn(title: "Downweighted", plugins: detail.downweightedPlugins, tint: FXAITheme.warning)
                        pluginColumn(title: "Suppressed", plugins: detail.suppressedPlugins, tint: FXAITheme.textMuted)
                    }

                    VStack(alignment: .leading, spacing: 16) {
                        pluginColumn(title: "Active", plugins: detail.activePlugins, tint: FXAITheme.success)
                        pluginColumn(title: "Downweighted", plugins: detail.downweightedPlugins, tint: FXAITheme.warning)
                        pluginColumn(title: "Suppressed", plugins: detail.suppressedPlugins, tint: FXAITheme.textMuted)
                    }
                }
            }
        }
    }

    private func replayPanel(detail: DynamicEnsembleSymbolSnapshot, replayHoursBack: Int) -> some View {
        FXAIVisualEffectSurface {
            VStack(alignment: .leading, spacing: 14) {
                Text("Replay & Drift")
                    .font(.headline)
                    .foregroundStyle(FXAITheme.textPrimary)

                Text("Last \(replayHoursBack)h • \(detail.observationCount) observations • Avg quality \(percentString(detail.averageQuality))")
                    .font(.caption)
                    .foregroundStyle(FXAITheme.textMuted)

                LazyVGrid(columns: [GridItem(.adaptive(minimum: 220), spacing: 14)], spacing: 14) {
                    metricListCard(title: "Posture Counts", values: detail.replayPostureCounts)
                    metricListCard(title: "Action Counts", values: detail.replayActionCounts)
                    metricListCard(title: "Status Counts", values: detail.replayStatusCounts)
                    metricListCard(title: "Top Reasons", values: detail.replayTopReasons)
                    metricListCard(title: "Dominant Plugins", values: detail.replayTopDominantPlugins)
                    metricListCard(
                        title: "Replay Summary",
                        values: [
                            KeyValueRecord(key: "max_abstain_bias", value: String(format: "%.2f", detail.maxAbstainBias)),
                            KeyValueRecord(key: "final_action", value: detail.finalAction),
                        ]
                    )
                }

                if !detail.recentTransitions.isEmpty {
                    VStack(alignment: .leading, spacing: 8) {
                        Text("Recent Transitions")
                            .font(.subheadline.weight(.semibold))
                            .foregroundStyle(FXAITheme.textPrimary)
                        ForEach(detail.recentTransitions.prefix(10)) { transition in
                            HStack {
                                Text(transition.type.replacingOccurrences(of: "_", with: " ").capitalized)
                                    .font(.caption.weight(.semibold))
                                    .foregroundStyle(FXAITheme.textPrimary)
                                Spacer()
                                Text("\(transition.fromValue) → \(transition.toValue)")
                                    .font(.caption)
                                    .foregroundStyle(FXAITheme.textMuted)
                            }
                        }
                    }
                }
            }
        }
    }

    private func pluginColumn(title: String, plugins: [DynamicEnsemblePluginState], tint: Color) -> some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                Text(title)
                    .font(.subheadline.weight(.semibold))
                    .foregroundStyle(FXAITheme.textPrimary)
                Spacer()
                Text("\(plugins.count)")
                    .font(.caption.weight(.semibold))
                    .foregroundStyle(tint)
            }
            .padding(.bottom, 2)

            if plugins.isEmpty {
                Text("No plugins in this state.")
                    .font(.caption)
                    .foregroundStyle(FXAITheme.textMuted)
            } else {
                ForEach(plugins) { plugin in
                    VStack(alignment: .leading, spacing: 4) {
                        HStack {
                            Text(plugin.name)
                                .font(.caption.weight(.semibold))
                                .foregroundStyle(FXAITheme.textPrimary)
                            Spacer()
                            Text(percentString(plugin.weight))
                                .font(.caption.monospacedDigit())
                                .foregroundStyle(tint)
                        }
                        Text("\(plugin.family) • \(plugin.signal) • trust \(String(format: "%.2f", plugin.trust)) • cal \(String(format: "%.2f", plugin.calibrationShrink))")
                            .font(.caption2)
                            .foregroundStyle(FXAITheme.textMuted)
                        ForEach(plugin.reasons.prefix(2), id: \.self) { reason in
                            Text(reason)
                                .font(.caption2)
                                .foregroundStyle(FXAITheme.textMuted)
                        }
                    }
                    .padding(10)
                    .background(
                        RoundedRectangle(cornerRadius: 12, style: .continuous)
                            .fill(tint.opacity(0.08))
                    )
                }
            }
        }
        .frame(maxWidth: .infinity, alignment: .topLeading)
    }

    private func metricListCard(title: String, values: [KeyValueRecord]) -> some View {
        FXAIVisualEffectSurface {
            VStack(alignment: .leading, spacing: 10) {
                Text(title)
                    .font(.subheadline.weight(.semibold))
                    .foregroundStyle(FXAITheme.textPrimary)

                if values.isEmpty {
                    Text("No data available.")
                        .font(.caption)
                        .foregroundStyle(FXAITheme.textMuted)
                } else {
                    ForEach(values.prefix(8)) { record in
                        HStack {
                            Text(record.key)
                                .font(.caption)
                                .foregroundStyle(FXAITheme.textMuted)
                            Spacer()
                            Text(record.value)
                                .font(.caption.monospacedDigit())
                                .foregroundStyle(FXAITheme.textPrimary)
                        }
                    }
                }
            }
        }
    }

    private func percentString(_ value: Double) -> String {
        String(format: "%.0f%%", value * 100)
    }

    private func postureTint(_ posture: String) -> Color {
        switch posture.uppercased() {
        case "BLOCK": criticalColor
        case "ABSTAIN_BIAS": FXAITheme.warning
        case "CAUTION": FXAITheme.warning
        default: FXAITheme.success
        }
    }

    private func postureSymbol(_ posture: String) -> String {
        switch posture.uppercased() {
        case "BLOCK": "hand.raised.fill"
        case "ABSTAIN_BIAS": "pause.circle.fill"
        case "CAUTION": "exclamationmark.triangle.fill"
        default: "checkmark.circle.fill"
        }
    }

    private func finalActionTint(_ action: String) -> Color {
        switch action.uppercased() {
        case "BUY": FXAITheme.success
        case "SELL": FXAITheme.warning
        default: FXAITheme.textMuted
        }
    }

    private func finalActionSymbol(_ action: String) -> String {
        switch action.uppercased() {
        case "BUY": "arrow.up.circle.fill"
        case "SELL": "arrow.down.circle.fill"
        default: "pause.circle.fill"
        }
    }

    private func barTint(_ label: String) -> Color {
        switch label.uppercased() {
        case "BUY": FXAITheme.success
        case "SELL": FXAITheme.warning
        default: FXAITheme.textMuted
        }
    }

    private var criticalColor: Color { Color(red: 0.97, green: 0.43, blue: 0.47) }
}
