import Charts
import FXAIGUICore
import SwiftUI

private struct ExecutionQualityRiskBar: Identifiable {
    let id: String
    let label: String
    let value: Double
}

struct ExecutionQualityView: View {
    @EnvironmentObject private var model: FXAIGUIModel

    private var visibleSymbols: [ExecutionQualitySymbolSnapshot] {
        guard let snapshot = model.executionQualitySnapshot else { return [] }
        return Array(snapshot.symbols.prefix(24))
    }

    private var selectedSymbol: ExecutionQualitySymbolSnapshot? {
        guard !visibleSymbols.isEmpty else { return nil }
        return visibleSymbols.first(where: { $0.symbol == model.selectedExecutionQualitySymbol }) ?? visibleSymbols.first
    }

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 18) {
                SectionHeader(
                    title: "Execution Quality",
                    subtitle: "Inspect the live execution forecast that prices spread widening, slippage risk, fill quality, latency sensitivity, and liquidity fragility before the final trade gate commits capital."
                )

                if let snapshot = model.executionQualitySnapshot, !snapshot.symbols.isEmpty {
                    topStatus(snapshot: snapshot)
                    symbolWorkspace(snapshot: snapshot)
                    replayPanel(snapshot: snapshot)
                } else {
                    EmptyStateView(
                        title: "Execution-quality artifacts not found",
                        message: "Run execution-quality-validate so the runtime config and memory exports exist, let MT5 publish per-symbol execution-quality state, and rebuild the replay report for operator visibility.",
                        symbolName: "speedometer"
                    )
                }
            }
        }
    }

    private func topStatus(snapshot: ExecutionQualitySnapshot) -> some View {
        let worstScore = snapshot.symbols.map(\.executionQualityScore).min() ?? 0
        let blockedCount = snapshot.symbols.filter { $0.executionState.uppercased() == "BLOCKED" }.count
        let stressedCount = snapshot.symbols.filter { $0.executionState.uppercased() == "STRESSED" }.count

        return LazyVGrid(columns: [GridItem(.adaptive(minimum: 180), spacing: 16)], spacing: 16) {
            MetricCard(
                title: "Tracked Symbols",
                value: "\(snapshot.symbols.count)",
                footnote: "Execution-quality state files currently visible to the operator GUI.",
                symbolName: "arrow.left.and.right.square.fill",
                tint: FXAITheme.accentSoft
            )
            MetricCard(
                title: "Worst Quality",
                value: percentString(worstScore),
                footnote: "Lowest execution-quality score across the live symbol set.",
                symbolName: "speedometer",
                tint: qualityTint(for: worstScore)
            )
            MetricCard(
                title: "Blocked",
                value: "\(blockedCount)",
                footnote: "Symbols where the execution forecaster currently recommends a hard block.",
                symbolName: "hand.raised.fill",
                tint: criticalColor
            )
            MetricCard(
                title: "Stressed",
                value: "\(stressedCount)",
                footnote: "Symbols that remain tradable only with materially worse execution posture.",
                symbolName: "exclamationmark.triangle.fill",
                tint: FXAITheme.warning
            )
        }
    }

    private func symbolWorkspace(snapshot _: ExecutionQualitySnapshot) -> some View {
        FXAIVisualEffectSurface {
            VStack(alignment: .leading, spacing: 14) {
                Text("Symbol Forecasts")
                    .font(.headline)
                    .foregroundStyle(FXAITheme.textPrimary)

                ViewThatFits(in: .horizontal) {
                    HStack(alignment: .top, spacing: 18) {
                        symbolList
                            .frame(width: 340, alignment: .topLeading)
                        symbolDetail
                    }

                    VStack(alignment: .leading, spacing: 18) {
                        symbolList
                        symbolDetail
                    }
                }
            }
        }
    }

    private var symbolList: some View {
        VStack(alignment: .leading, spacing: 10) {
            ForEach(visibleSymbols) { symbolState in
                Button {
                    model.selectedExecutionQualitySymbol = symbolState.symbol
                } label: {
                    HStack(alignment: .top, spacing: 10) {
                        VStack(alignment: .leading, spacing: 4) {
                            HStack(spacing: 8) {
                                Text(symbolState.symbol)
                                    .font(.subheadline.weight(.semibold))
                                    .foregroundStyle(FXAITheme.textPrimary)
                                Text(symbolState.executionState)
                                    .font(.caption2.weight(.bold))
                                    .foregroundStyle(stateTint(symbolState.executionState))
                                    .padding(.horizontal, 6)
                                    .padding(.vertical, 2)
                                    .background(
                                        Capsule(style: .continuous)
                                            .fill(stateTint(symbolState.executionState).opacity(0.12))
                                    )
                            }
                            Text(symbolState.regimeLabel.replacingOccurrences(of: "_", with: " ").capitalized)
                                .font(.caption)
                                .foregroundStyle(FXAITheme.textSecondary)
                            if let reason = symbolState.reasons.first {
                                Text(reason)
                                    .font(.caption)
                                    .foregroundStyle(FXAITheme.textMuted)
                                    .lineLimit(2)
                            }
                        }
                        Spacer(minLength: 12)
                        VStack(alignment: .trailing, spacing: 4) {
                            Text(percentString(symbolState.executionQualityScore))
                                .font(.subheadline.weight(.semibold))
                                .foregroundStyle(stateTint(symbolState.executionState))
                            Text(pointString(symbolState.spreadExpectedPoints))
                                .font(.caption)
                                .foregroundStyle(FXAITheme.textMuted)
                        }
                    }
                    .padding(12)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .background(
                        RoundedRectangle(cornerRadius: 16, style: .continuous)
                            .fill(model.selectedExecutionQualitySymbol == symbolState.symbol ? FXAITheme.accentSoft.opacity(0.14) : FXAITheme.panel.opacity(0.55))
                    )
                }
                .buttonStyle(.plain)
            }
        }
    }

    private var symbolDetail: some View {
        VStack(alignment: .leading, spacing: 16) {
            if let selectedSymbol {
                let bars = [
                    ExecutionQualityRiskBar(id: "spread", label: "Spread Risk", value: selectedSymbol.spreadWideningRisk),
                    ExecutionQualityRiskBar(id: "slippage", label: "Slippage", value: selectedSymbol.slippageRisk),
                    ExecutionQualityRiskBar(id: "latency", label: "Latency", value: selectedSymbol.latencySensitivityScore),
                    ExecutionQualityRiskBar(id: "fragility", label: "Fragility", value: selectedSymbol.liquidityFragilityScore),
                    ExecutionQualityRiskBar(id: "fill", label: "Fill Quality", value: 1.0 - selectedSymbol.fillQualityScore),
                ]

                VStack(alignment: .leading, spacing: 16) {
                    HStack {
                        Text(selectedSymbol.symbol)
                            .font(.title3.weight(.semibold))
                            .foregroundStyle(FXAITheme.textPrimary)
                        Spacer()
                        StatusBadge(title: "State", value: selectedSymbol.executionState, tint: stateTint(selectedSymbol.executionState))
                    }

                    LazyVGrid(columns: [GridItem(.adaptive(minimum: 150), spacing: 12)], spacing: 12) {
                        detailMetric(title: "Quality", value: percentString(selectedSymbol.executionQualityScore), tint: qualityTint(for: selectedSymbol.executionQualityScore))
                        detailMetric(title: "Spread Now", value: pointString(selectedSymbol.spreadNowPoints), tint: FXAITheme.accentSoft)
                        detailMetric(title: "Spread Expected", value: pointString(selectedSymbol.spreadExpectedPoints), tint: FXAITheme.warning)
                        detailMetric(title: "Expected Slippage", value: pointString(selectedSymbol.expectedSlippagePoints), tint: FXAITheme.warning)
                        detailMetric(title: "Deviation", value: pointString(selectedSymbol.allowedDeviationPoints), tint: FXAITheme.accent)
                        detailMetric(title: "Lot Scale", value: String(format: "%.2fx", selectedSymbol.cautionLotScale), tint: FXAITheme.success)
                    }

                    Chart(bars) { bar in
                        BarMark(
                            x: .value("Metric", bar.label),
                            y: .value("Value", bar.value)
                        )
                        .foregroundStyle(barTint(bar.label).gradient)
                        .cornerRadius(6)
                    }
                    .chartYAxis { AxisMarks(position: .leading) }
                    .frame(height: 220)

                    LazyVGrid(columns: [GridItem(.adaptive(minimum: 210), spacing: 12)], spacing: 12) {
                        metricListCard(
                            title: "Execution Inputs",
                            values: [
                                KeyValueRecord(key: "session_label", value: selectedSymbol.sessionLabel),
                                KeyValueRecord(key: "regime_label", value: selectedSymbol.regimeLabel),
                                KeyValueRecord(key: "tier_kind", value: selectedSymbol.tierKind),
                                KeyValueRecord(key: "tier_support", value: "\(selectedSymbol.support)"),
                                KeyValueRecord(key: "tier_quality", value: String(format: "%.2f", selectedSymbol.quality)),
                                KeyValueRecord(key: "broker_coverage", value: percentString(selectedSymbol.brokerCoverage)),
                            ]
                        )
                        metricListCard(
                            title: "Risk Flags",
                            values: [
                                KeyValueRecord(key: "news_window_active", value: boolLabel(selectedSymbol.newsWindowActive)),
                                KeyValueRecord(key: "rates_repricing_active", value: boolLabel(selectedSymbol.ratesRepricingActive)),
                                KeyValueRecord(key: "support_usable", value: boolLabel(selectedSymbol.supportUsable)),
                                KeyValueRecord(key: "memory_stale", value: boolLabel(selectedSymbol.memoryStale)),
                                KeyValueRecord(key: "data_stale", value: boolLabel(selectedSymbol.dataStale)),
                                KeyValueRecord(key: "fallback_used", value: boolLabel(selectedSymbol.fallbackUsed)),
                            ]
                        )
                        metricListCard(
                            title: "Broker Friction",
                            values: [
                                KeyValueRecord(key: "reject_probability", value: percentString(selectedSymbol.brokerRejectProbability)),
                                KeyValueRecord(key: "partial_fill_probability", value: percentString(selectedSymbol.brokerPartialFillProbability)),
                                KeyValueRecord(key: "fill_quality_score", value: percentString(selectedSymbol.fillQualityScore)),
                                KeyValueRecord(key: "latency_sensitivity", value: percentString(selectedSymbol.latencySensitivityScore)),
                                KeyValueRecord(key: "liquidity_fragility", value: percentString(selectedSymbol.liquidityFragilityScore)),
                                KeyValueRecord(key: "enter_prob_buffer", value: percentString(selectedSymbol.cautionEnterProbBuffer)),
                            ]
                        )
                    }

                    VStack(alignment: .leading, spacing: 8) {
                        Text("Reason Codes")
                            .font(.subheadline.weight(.semibold))
                            .foregroundStyle(FXAITheme.textPrimary)
                        if selectedSymbol.reasons.isEmpty {
                            Text("No explicit execution-quality reason codes were recorded for the latest state.")
                                .font(.caption)
                                .foregroundStyle(FXAITheme.textMuted)
                        } else {
                            ForEach(selectedSymbol.reasons, id: \.self) { reason in
                                HStack(alignment: .top, spacing: 8) {
                                    Image(systemName: "circle.fill")
                                        .font(.system(size: 5))
                                        .foregroundStyle(stateTint(selectedSymbol.executionState))
                                        .padding(.top, 6)
                                    Text(reason)
                                        .font(.caption)
                                        .foregroundStyle(FXAITheme.textMuted)
                                }
                            }
                        }
                    }
                }
            } else {
                Text("No execution-quality symbol is selected.")
                    .foregroundStyle(FXAITheme.textSecondary)
            }
        }
    }

    private func replayPanel(snapshot: ExecutionQualitySnapshot) -> some View {
        FXAIVisualEffectSurface {
            VStack(alignment: .leading, spacing: 14) {
                Text("Replay Summary")
                    .font(.headline)
                    .foregroundStyle(FXAITheme.textPrimary)

                if let selectedSymbol {
                    Text("Last \(snapshot.replayHoursBack)h • \(selectedSymbol.observationCount) observations • min quality \(percentString(selectedSymbol.minExecutionQualityScore))")
                        .font(.caption)
                        .foregroundStyle(FXAITheme.textMuted)

                    LazyVGrid(columns: [GridItem(.adaptive(minimum: 220), spacing: 14)], spacing: 14) {
                        metricListCard(title: "State Counts", values: selectedSymbol.replayStateCounts)
                        metricListCard(title: "Tier Counts", values: selectedSymbol.replayTierCounts)
                        metricListCard(title: "Top Reasons", values: selectedSymbol.replayTopReasons)
                        metricListCard(
                            title: "Replay Extremes",
                            values: [
                                KeyValueRecord(key: "max_spread_widening_risk", value: percentString(selectedSymbol.maxSpreadWideningRisk)),
                                KeyValueRecord(key: "max_slippage_risk", value: percentString(selectedSymbol.maxSlippageRisk)),
                                KeyValueRecord(key: "min_execution_quality", value: percentString(selectedSymbol.minExecutionQualityScore)),
                            ]
                        )
                    }

                    if !selectedSymbol.recentTransitions.isEmpty {
                        VStack(alignment: .leading, spacing: 8) {
                            Text("Recent Transitions")
                                .font(.subheadline.weight(.semibold))
                                .foregroundStyle(FXAITheme.textPrimary)
                            ForEach(selectedSymbol.recentTransitions) { transition in
                                HStack(alignment: .top, spacing: 8) {
                                    Image(systemName: "arrow.triangle.swap")
                                        .foregroundStyle(FXAITheme.accent)
                                        .padding(.top, 2)
                                    VStack(alignment: .leading, spacing: 3) {
                                        Text("\(transition.type): \(transition.fromValue) → \(transition.toValue)")
                                            .font(.caption.weight(.semibold))
                                            .foregroundStyle(FXAITheme.textPrimary)
                                        if let observedAt = transition.observedAt {
                                            Text(observedAt.formatted(date: .abbreviated, time: .shortened))
                                                .font(.caption2)
                                                .foregroundStyle(FXAITheme.textMuted)
                                        }
                                    }
                                }
                            }
                        }
                    }
                } else {
                    Text("Replay output becomes visible after the runtime emits at least one execution-quality state file.")
                        .font(.caption)
                        .foregroundStyle(FXAITheme.textMuted)
                }
            }
        }
    }

    private func metricListCard(title: String, values: [KeyValueRecord]) -> some View {
        VStack(alignment: .leading, spacing: 10) {
            Text(title)
                .font(.subheadline.weight(.semibold))
                .foregroundStyle(FXAITheme.textPrimary)

            if values.isEmpty {
                Text("No data available.")
                    .font(.caption)
                    .foregroundStyle(FXAITheme.textMuted)
            } else {
                ForEach(values) { item in
                    HStack {
                        Text(item.key.replacingOccurrences(of: "_", with: " "))
                            .font(.caption)
                            .foregroundStyle(FXAITheme.textSecondary)
                        Spacer()
                        Text(item.value)
                            .font(.caption.weight(.semibold))
                            .foregroundStyle(FXAITheme.textPrimary)
                    }
                }
            }
        }
        .padding(14)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(
            RoundedRectangle(cornerRadius: 16, style: .continuous)
                .fill(FXAITheme.panel.opacity(0.55))
        )
    }

    private func detailMetric(title: String, value: String, tint: Color) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            Text(title)
                .font(.caption)
                .foregroundStyle(FXAITheme.textSecondary)
            Text(value)
                .font(.headline.weight(.semibold))
                .foregroundStyle(tint)
        }
        .padding(12)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(
            RoundedRectangle(cornerRadius: 16, style: .continuous)
                .fill(FXAITheme.panel.opacity(0.55))
        )
    }

    private func stateTint(_ state: String) -> Color {
        switch state.uppercased() {
        case "BLOCKED": criticalColor
        case "STRESSED": FXAITheme.warning
        case "CAUTION": FXAITheme.accentSoft
        default: FXAITheme.success
        }
    }

    private func qualityTint(for value: Double) -> Color {
        if value < 0.36 { return criticalColor }
        if value < 0.54 { return FXAITheme.warning }
        if value < 0.72 { return FXAITheme.accentSoft }
        return FXAITheme.success
    }

    private func barTint(_ label: String) -> Color {
        if label.contains("Fill") {
            return criticalColor
        }
        if label.contains("Latency") {
            return FXAITheme.warning
        }
        return FXAITheme.accent
    }

    private func percentString(_ value: Double) -> String {
        String(format: "%.0f%%", value * 100)
    }

    private func pointString(_ value: Double) -> String {
        String(format: "%.2f pt", value)
    }

    private func boolLabel(_ value: Bool) -> String {
        value ? "yes" : "no"
    }

    private var criticalColor: Color {
        Color(red: 0.97, green: 0.43, blue: 0.47)
    }
}
