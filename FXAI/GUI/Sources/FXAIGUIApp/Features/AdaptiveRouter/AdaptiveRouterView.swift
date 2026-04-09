import Charts
import FXAIGUICore
import SwiftUI

struct AdaptiveRouterView: View {
    @EnvironmentObject private var model: FXAIGUIModel

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 18) {
                SectionHeader(
                    title: "Adaptive Router",
                    subtitle: "Inspect the live regime classifier, routed plugin trust, abstention posture, and replay evidence that now steers the FXAI plugin zoo."
                )

                if let snapshot = model.adaptiveRouterSnapshot, !snapshot.symbols.isEmpty {
                    symbolPicker(snapshot: snapshot)

                    if let detail = model.selectedAdaptiveRouterDetail {
                        topStatus(detail: detail)
                        regimePanel(detail: detail)
                        pluginPanel(detail: detail)
                        replayPanel(detail: detail, replayHoursBack: snapshot.replayHoursBack)
                        profilePanel(detail: detail)
                    }
                } else {
                    EmptyStateView(
                        title: "Adaptive Router artifacts not found",
                        message: "Generate adaptive router profiles, let the runtime publish live routing state, and refresh the GUI to inspect current regime and plugin routing.",
                        symbolName: "point.3.filled.connected.trianglepath.dotted"
                    )
                }
            }
        }
    }

    private func symbolPicker(snapshot: AdaptiveRouterSnapshot) -> some View {
        ViewThatFits(in: .horizontal) {
            HStack {
                Picker("Symbol", selection: $model.selectedAdaptiveSymbol) {
                    ForEach(snapshot.symbols, id: \.symbol) { detail in
                        Text(detail.symbol).tag(detail.symbol)
                    }
                }
                .pickerStyle(.segmented)

                Spacer()
            }

            HStack {
                Picker("Symbol", selection: $model.selectedAdaptiveSymbol) {
                    ForEach(snapshot.symbols, id: \.symbol) { detail in
                        Text(detail.symbol).tag(detail.symbol)
                    }
                }
                .pickerStyle(.menu)
                Spacer()
            }
        }
    }

    private func topStatus(detail: AdaptiveRouterSymbolSnapshot) -> some View {
        let abstainFootnote = "Abstain bias \(String(format: "%.2f", detail.abstainBias))"
        let executionFootnote = "\(detail.sessionLabel.replacingOccurrences(of: "_", with: " ")) • \(detail.volatilityRegime.capitalized)"

        return LazyVGrid(
            columns: [GridItem(.adaptive(minimum: 180), spacing: 16, alignment: .top)],
            spacing: 16
        ) {
            MetricCard(
                title: "Top Regime",
                value: detail.topRegime.replacingOccurrences(of: "_", with: " "),
                footnote: detail.generatedAt.map { "Live state \($0.formatted(date: .omitted, time: .shortened))" } ?? "Runtime state missing timestamp",
                symbolName: "waveform.path.badge.plus",
                tint: regimeTint(detail.topRegime)
            )
            MetricCard(
                title: "Confidence",
                value: percentString(detail.confidence),
                footnote: "Soft regime confidence from the live classifier.",
                symbolName: "scope",
                tint: FXAITheme.accent
            )
            MetricCard(
                title: "Posture",
                value: detail.tradePosture,
                footnote: abstainFootnote,
                symbolName: postureSymbol(detail.tradePosture),
                tint: postureTint(detail.tradePosture)
            )
            MetricCard(
                title: "Execution",
                value: detail.spreadRegime.capitalized,
                footnote: executionFootnote,
                symbolName: "arrow.left.and.line.vertical.and.arrow.right",
                tint: FXAITheme.accentSoft
            )
            MetricCard(
                title: "News Risk",
                value: percentString(detail.newsRiskScore),
                footnote: detail.eventETAMin.map { "Event ETA \($0)m" } ?? "No imminent scheduled event",
                symbolName: "dot.radiowaves.left.and.right",
                tint: detail.staleNews ? FXAITheme.warning : FXAITheme.success
            )
            MetricCard(
                title: "Liquidity",
                value: percentString(detail.liquidityStress),
                footnote: "Breakout \(percentString(detail.breakoutPressure)) • Trend \(percentString(detail.trendStrength))",
                symbolName: "drop.degreesign.fill",
                tint: detail.liquidityStress >= 0.55 ? FXAITheme.warning : FXAITheme.accentSoft
            )
        }
    }

    private func regimePanel(detail: AdaptiveRouterSymbolSnapshot) -> some View {
        FXAIVisualEffectSurface {
            VStack(alignment: .leading, spacing: 14) {
                Text("Regime State")
                    .font(.headline)
                    .foregroundStyle(FXAITheme.textPrimary)

                if detail.probabilities.isEmpty {
                    Text("Live regime probabilities have not been published yet.")
                        .foregroundStyle(FXAITheme.textSecondary)
                } else {
                    Chart(detail.probabilities) { item in
                        BarMark(
                            x: .value("Regime", item.label),
                            y: .value("Probability", item.probability)
                        )
                        .foregroundStyle(regimeTint(item.label).gradient)
                        .cornerRadius(6)
                    }
                    .chartYAxis {
                        AxisMarks(position: .leading)
                    }
                    .frame(height: 240)
                }

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

    private func pluginPanel(detail: AdaptiveRouterSymbolSnapshot) -> some View {
        FXAIVisualEffectSurface {
            VStack(alignment: .leading, spacing: 14) {
                Text("Plugin Routing")
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

    private func replayPanel(detail: AdaptiveRouterSymbolSnapshot, replayHoursBack: Int) -> some View {
        let recentTransitions: [AdaptiveRouterTransition] = detail.recentTransitions.prefix(10).map { $0 }

        return FXAIVisualEffectSurface {
            VStack(alignment: .leading, spacing: 14) {
                Text("Replay & Drift")
                    .font(.headline)
                    .foregroundStyle(FXAITheme.textPrimary)

                Text("Last \(replayHoursBack)h • \(detail.observationCount) observations")
                    .font(.caption)
                    .foregroundStyle(FXAITheme.textMuted)

                LazyVGrid(
                    columns: [GridItem(.adaptive(minimum: 220), spacing: 14, alignment: .top)],
                    spacing: 14
                ) {
                    metricListCard(title: "Regime Counts", values: detail.replayRegimeCounts)
                    metricListCard(title: "Posture Counts", values: detail.replayPostureCounts)
                    metricListCard(title: "Top Reasons", values: detail.replayTopReasons)
                    metricListCard(title: "Top Plugins", values: detail.replayTopPlugins)
                }

                if !recentTransitions.isEmpty {
                    VStack(alignment: .leading, spacing: 8) {
                        Text("Recent Transitions")
                            .font(.subheadline.weight(.semibold))
                            .foregroundStyle(FXAITheme.textPrimary)
                        transitionRows(recentTransitions)
                    }
                }
            }
        }
    }

    private func profilePanel(detail: AdaptiveRouterSymbolSnapshot) -> some View {
        FXAIVisualEffectSurface {
            VStack(alignment: .leading, spacing: 14) {
                Text("Profile & Thresholds")
                    .font(.headline)
                    .foregroundStyle(FXAITheme.textPrimary)

                LazyVGrid(
                    columns: [GridItem(.adaptive(minimum: 220), spacing: 14, alignment: .top)],
                    spacing: 14
                ) {
                    metricListCard(
                        title: "Thresholds",
                        values: detail.thresholdMetrics,
                        emptyMessage: "No adaptive threshold profile was found."
                    )
                    metricListCard(
                        title: "Regime Bias",
                        values: detail.regimeBiasMetrics,
                        emptyMessage: "No pair-tag regime bias was found."
                    )
                    metricListCard(
                        title: "Pair Tags",
                        values: detail.pairTags.map { KeyValueRecord(key: $0, value: "active") },
                        emptyMessage: "No pair tags declared."
                    )
                    metricListCard(
                        title: "Top Profile Plugins",
                        values: detail.topProfilePlugins.map { KeyValueRecord(key: $0, value: "candidate") },
                        emptyMessage: "No profile plugin summary was found."
                    )
                }
            }
        }
    }

    private func pluginColumn(title: String, plugins: [AdaptiveRouterPluginState], tint: Color) -> some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                Text(title)
                    .font(.subheadline.weight(.semibold))
                    .foregroundStyle(FXAITheme.textPrimary)
                Spacer()
                Text("\(plugins.count)")
                    .font(.caption.weight(.bold))
                    .foregroundStyle(tint)
            }

            if plugins.isEmpty {
                Text("No plugins in this state.")
                    .font(.caption)
                    .foregroundStyle(FXAITheme.textMuted)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(12)
                    .background(
                        RoundedRectangle(cornerRadius: 14, style: .continuous)
                            .fill(FXAITheme.panel.opacity(0.35))
                    )
            } else {
                ForEach(plugins.prefix(8)) { plugin in
                    VStack(alignment: .leading, spacing: 6) {
                        HStack {
                            Text(plugin.name)
                                .font(.caption.weight(.semibold))
                                .foregroundStyle(FXAITheme.textPrimary)
                            Spacer()
                            Text(String(format: "%.2f / %.2f", plugin.weight, plugin.suitability))
                                .font(.caption2.monospacedDigit())
                                .foregroundStyle(FXAITheme.textSecondary)
                        }
                        ForEach(plugin.reasons, id: \.self) { reason in
                            Text(reason)
                                .font(.caption2)
                                .foregroundStyle(FXAITheme.textMuted)
                                .lineLimit(2)
                        }
                    }
                    .padding(12)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .background(
                        RoundedRectangle(cornerRadius: 14, style: .continuous)
                            .fill(tint.opacity(0.10))
                    )
                }
            }
        }
        .frame(maxWidth: .infinity, alignment: .topLeading)
    }

    private func metricListCard(title: String, values: [KeyValueRecord], emptyMessage: String = "No data available.") -> some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(title)
                .font(.subheadline.weight(.semibold))
                .foregroundStyle(FXAITheme.textPrimary)
            if values.isEmpty {
                Text(emptyMessage)
                    .font(.caption)
                    .foregroundStyle(FXAITheme.textMuted)
            } else {
                ForEach(values.prefix(8)) { item in
                    HStack(alignment: .top) {
                        Text(item.key.replacingOccurrences(of: "_", with: " ").capitalized)
                            .font(.caption)
                            .foregroundStyle(FXAITheme.textMuted)
                        Spacer()
                        Text(item.value)
                            .font(.caption.monospacedDigit())
                            .foregroundStyle(FXAITheme.textPrimary)
                    }
                }
            }
        }
        .padding(14)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(
            RoundedRectangle(cornerRadius: 16, style: .continuous)
                .fill(FXAITheme.panel.opacity(0.42))
        )
    }

    @ViewBuilder
    private func transitionRows(_ transitions: [AdaptiveRouterTransition]) -> some View {
        ForEach(transitions, id: \.id) { transition in
            transitionRow(transition)
        }
    }

    private func transitionRow(_ transition: AdaptiveRouterTransition) -> some View {
        let transitionTypeLabel = transition.type.replacingOccurrences(of: "_", with: " ").capitalized
        let transitionObservedLabel = transition.observedAt.map { FXAIFormatting.relativeDateString(for: $0) } ?? "unknown"
        let transitionSubtitle = "\(transitionTypeLabel) • \(transitionObservedLabel)"

        return HStack(alignment: .top, spacing: 8) {
            Image(systemName: transition.type == "posture_change" ? "shield.lefthalf.filled" : "arrow.triangle.branch")
                .foregroundStyle(FXAITheme.accent)
            VStack(alignment: .leading, spacing: 3) {
                Text("\(transition.fromValue) → \(transition.toValue)")
                    .font(.caption.weight(.semibold))
                    .foregroundStyle(FXAITheme.textPrimary)
                Text(transitionSubtitle)
                    .font(.caption)
                    .foregroundStyle(FXAITheme.textMuted)
            }
            Spacer()
        }
        .padding(10)
        .background(
            RoundedRectangle(cornerRadius: 14, style: .continuous)
                .fill(FXAITheme.panel.opacity(0.40))
        )
    }

    private func percentString(_ value: Double) -> String {
        String(format: "%.0f%%", value * 100.0)
    }

    private func regimeTint(_ regime: String) -> Color {
        switch regime {
        case "HIGH_VOL_EVENT", "LIQUIDITY_STRESS":
            return FXAITheme.warning
        case "BREAKOUT_TRANSITION":
            return FXAITheme.accent
        case "RANGE_MEAN_REVERTING":
            return FXAITheme.accentSoft
        case "RISK_ON_OFF_MACRO":
            return FXAITheme.success
        default:
            return FXAITheme.textPrimary
        }
    }

    private func postureTint(_ posture: String) -> Color {
        switch posture {
        case "BLOCK":
            return FXAITheme.warning
        case "ABSTAIN_BIAS":
            return FXAITheme.accent
        case "CAUTION":
            return FXAITheme.warning
        default:
            return FXAITheme.success
        }
    }

    private func postureSymbol(_ posture: String) -> String {
        switch posture {
        case "BLOCK":
            return "hand.raised.fill"
        case "ABSTAIN_BIAS":
            return "pause.circle.fill"
        case "CAUTION":
            return "exclamationmark.shield.fill"
        default:
            return "checkmark.shield.fill"
        }
    }
}
