import Charts
import FXAIGUICore
import SwiftUI

private struct ProbabilityBar: Identifiable {
    let id: String
    let label: String
    let phase: String
    let value: Double
}

private struct MoveQuantileBar: Identifiable {
    let id: String
    let label: String
    let value: Double
}

struct ProbCalibrationView: View {
    @EnvironmentObject private var model: FXAIGUIModel

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 18) {
                SectionHeader(
                    title: "Probabilistic Calibration",
                    subtitle: "Inspect the final cost-aware decision layer that converts raw ensemble output into calibrated probabilities, expected move estimates, edge after costs, and explicit abstention reasons."
                )

                if let snapshot = model.probCalibrationSnapshot, !snapshot.symbols.isEmpty {
                    symbolPicker(snapshot: snapshot)

                    if let detail = model.selectedProbCalibrationDetail {
                        statusPanel(detail: detail)
                        probabilityPanel(detail: detail)
                        penaltiesPanel(detail: detail)
                        replayPanel(detail: detail, replayHoursBack: snapshot.replayHoursBack)
                    }
                } else {
                    EmptyStateView(
                        title: "Probabilistic calibration artifacts not found",
                        message: "Run prob-calibration-validate so the runtime config and memory export exist, let the live runtime publish symbol state, and rebuild the replay report to inspect calibrated trade decisions.",
                        symbolName: "checkmark.shield.fill"
                    )
                }
            }
        }
    }

    private func symbolPicker(snapshot: ProbCalibrationSnapshot) -> some View {
        ViewThatFits(in: .horizontal) {
            HStack {
                Picker("Symbol", selection: $model.selectedProbCalibrationSymbol) {
                    ForEach(snapshot.symbols, id: \.symbol) { detail in
                        Text(detail.symbol).tag(detail.symbol)
                    }
                }
                .pickerStyle(.segmented)
                Spacer()
            }

            HStack {
                Picker("Symbol", selection: $model.selectedProbCalibrationSymbol) {
                    ForEach(snapshot.symbols, id: \.symbol) { detail in
                        Text(detail.symbol).tag(detail.symbol)
                    }
                }
                .pickerStyle(.menu)
                Spacer()
            }
        }
    }

    private func statusPanel(detail: ProbCalibrationSymbolSnapshot) -> some View {
        LazyVGrid(columns: [GridItem(.adaptive(minimum: 180), spacing: 16)], spacing: 16) {
            MetricCard(
                title: "Final Action",
                value: detail.finalAction,
                footnote: detail.generatedAt.map { "Live state \($0.formatted(date: .omitted, time: .shortened))" } ?? "Runtime timestamp unavailable",
                symbolName: actionSymbol(detail.finalAction),
                tint: actionTint(detail.finalAction, abstain: detail.abstain)
            )
            MetricCard(
                title: "Calibrated Confidence",
                value: percentString(detail.calibratedConfidence),
                footnote: "\(detail.method) • raw \(percentString(max(detail.rawBuyProb, detail.rawSellProb)))",
                symbolName: "scope",
                tint: detail.calibratedConfidence >= 0.58 ? FXAITheme.success : FXAITheme.warning
            )
            MetricCard(
                title: "Edge After Costs",
                value: pointString(detail.edgeAfterCostsPoints),
                footnote: "Gross \(pointString(detail.expectedGrossEdgePoints)) • spread+slip \(pointString(detail.spreadCostPoints + detail.slippageCostPoints))",
                symbolName: detail.edgeAfterCostsPoints >= 0 ? "plus.circle.fill" : "minus.circle.fill",
                tint: detail.edgeAfterCostsPoints >= 0 ? FXAITheme.success : criticalColor
            )
            MetricCard(
                title: "Uncertainty",
                value: percentString(detail.uncertaintyScore),
                footnote: "Penalty \(pointString(detail.uncertaintyPenaltyPoints)) • risk \(pointString(detail.riskPenaltyPoints))",
                symbolName: "exclamationmark.shield.fill",
                tint: detail.uncertaintyScore >= 0.60 ? FXAITheme.warning : FXAITheme.accentSoft
            )
            MetricCard(
                title: "Calibration Tier",
                value: detail.tierKind.replacingOccurrences(of: "_", with: " "),
                footnote: "Support \(detail.support) • quality \(String(format: "%.2f", detail.quality))",
                symbolName: "square.stack.3d.down.right.fill",
                tint: detail.supportUsable ? FXAITheme.accent : FXAITheme.warning
            )
            MetricCard(
                title: "Move Distribution",
                value: pointString(detail.expectedMoveQ50Points),
                footnote: "Q25 \(pointString(detail.expectedMoveQ25Points)) • Q75 \(pointString(detail.expectedMoveQ75Points))",
                symbolName: "chart.bar.xaxis",
                tint: FXAITheme.accent
            )
        }
    }

    private func probabilityPanel(detail: ProbCalibrationSymbolSnapshot) -> some View {
        let probabilityBars = [
            ProbabilityBar(id: "raw-buy", label: "BUY", phase: "Raw", value: detail.rawBuyProb),
            ProbabilityBar(id: "raw-sell", label: "SELL", phase: "Raw", value: detail.rawSellProb),
            ProbabilityBar(id: "raw-skip", label: "SKIP", phase: "Raw", value: detail.rawSkipProb),
            ProbabilityBar(id: "cal-buy", label: "BUY", phase: "Calibrated", value: detail.calibratedBuyProb),
            ProbabilityBar(id: "cal-sell", label: "SELL", phase: "Calibrated", value: detail.calibratedSellProb),
            ProbabilityBar(id: "cal-skip", label: "SKIP", phase: "Calibrated", value: detail.calibratedSkipProb),
        ]
        let quantiles = [
            MoveQuantileBar(id: "q25", label: "Q25", value: detail.expectedMoveQ25Points),
            MoveQuantileBar(id: "q50", label: "Q50", value: detail.expectedMoveQ50Points),
            MoveQuantileBar(id: "mean", label: "Mean", value: detail.expectedMoveMeanPoints),
            MoveQuantileBar(id: "q75", label: "Q75", value: detail.expectedMoveQ75Points),
        ]

        return ViewThatFits(in: .horizontal) {
            HStack(alignment: .top, spacing: 16) {
                probabilityCard(detail: detail, records: probabilityBars)
                moveCard(detail: detail, records: quantiles)
            }

            VStack(alignment: .leading, spacing: 16) {
                probabilityCard(detail: detail, records: probabilityBars)
                moveCard(detail: detail, records: quantiles)
            }
        }
    }

    private func probabilityCard(detail: ProbCalibrationSymbolSnapshot, records: [ProbabilityBar]) -> some View {
        FXAIVisualEffectSurface {
            VStack(alignment: .leading, spacing: 14) {
                Text("Probability Mapping")
                    .font(.headline)
                    .foregroundStyle(FXAITheme.textPrimary)

                Chart(records) { record in
                    BarMark(
                        x: .value("Side", record.label),
                        y: .value("Probability", record.value)
                    )
                    .position(by: .value("Phase", record.phase))
                    .foregroundStyle(probabilityTint(record.label, phase: record.phase).gradient)
                    .cornerRadius(6)
                }
                .chartYAxis { AxisMarks(position: .leading) }
                .frame(height: 220)

                Text("\(detail.sessionLabel.replacingOccurrences(of: "_", with: " ")) • \(detail.regimeLabel.replacingOccurrences(of: "_", with: " "))")
                    .font(.caption)
                    .foregroundStyle(FXAITheme.textMuted)
            }
        }
    }

    private func moveCard(detail: ProbCalibrationSymbolSnapshot, records: [MoveQuantileBar]) -> some View {
        FXAIVisualEffectSurface {
            VStack(alignment: .leading, spacing: 14) {
                Text("Expected Move")
                    .font(.headline)
                    .foregroundStyle(FXAITheme.textPrimary)

                Chart(records) { record in
                    BarMark(
                        x: .value("Metric", record.label),
                        y: .value("Points", record.value)
                    )
                    .foregroundStyle(FXAITheme.accent.gradient)
                    .cornerRadius(6)
                }
                .chartYAxis { AxisMarks(position: .leading) }
                .frame(height: 220)

                Text("Tier \(detail.tierKey) • \(detail.fallbackUsed ? "fallback" : "primary")")
                    .font(.caption)
                    .foregroundStyle(FXAITheme.textMuted)
                    .lineLimit(2)
            }
        }
    }

    private func penaltiesPanel(detail: ProbCalibrationSymbolSnapshot) -> some View {
        FXAIVisualEffectSurface {
            VStack(alignment: .leading, spacing: 14) {
                Text("Penalties & Reasons")
                    .font(.headline)
                    .foregroundStyle(FXAITheme.textPrimary)

                LazyVGrid(columns: [GridItem(.adaptive(minimum: 220), spacing: 14)], spacing: 14) {
                    metricListCard(
                        title: "Costs",
                        values: [
                            KeyValueRecord(key: "spread_cost_points", value: pointString(detail.spreadCostPoints)),
                            KeyValueRecord(key: "slippage_cost_points", value: pointString(detail.slippageCostPoints)),
                            KeyValueRecord(key: "uncertainty_penalty_points", value: pointString(detail.uncertaintyPenaltyPoints)),
                            KeyValueRecord(key: "risk_penalty_points", value: pointString(detail.riskPenaltyPoints)),
                        ]
                    )
                    metricListCard(
                        title: "Quality Flags",
                        values: [
                            KeyValueRecord(key: "calibration_stale", value: boolLabel(detail.calibrationStale)),
                            KeyValueRecord(key: "input_stale", value: boolLabel(detail.inputStale)),
                            KeyValueRecord(key: "support_usable", value: boolLabel(detail.supportUsable)),
                            KeyValueRecord(key: "fallback_used", value: boolLabel(detail.fallbackUsed)),
                            KeyValueRecord(key: "abstain", value: boolLabel(detail.abstain)),
                        ]
                    )
                    metricListCard(
                        title: "Decision Summary",
                        values: [
                            KeyValueRecord(key: "raw_action", value: detail.rawAction),
                            KeyValueRecord(key: "final_action", value: detail.finalAction),
                            KeyValueRecord(key: "raw_score", value: String(format: "%.3f", detail.rawScore)),
                            KeyValueRecord(key: "selected_quality", value: String(format: "%.2f", detail.quality)),
                        ]
                    )
                }

                VStack(alignment: .leading, spacing: 8) {
                    Text("Reason Codes")
                        .font(.subheadline.weight(.semibold))
                        .foregroundStyle(FXAITheme.textPrimary)

                    if detail.reasons.isEmpty {
                        Text("No explicit abstention or downgrade reasons were recorded for the latest state.")
                            .font(.caption)
                            .foregroundStyle(FXAITheme.textMuted)
                    } else {
                        ForEach(detail.reasons, id: \.self) { reason in
                            HStack(alignment: .top, spacing: 8) {
                                Image(systemName: "circle.fill")
                                    .font(.system(size: 5))
                                    .foregroundStyle(actionTint(detail.finalAction, abstain: detail.abstain))
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
    }

    private func replayPanel(detail: ProbCalibrationSymbolSnapshot, replayHoursBack: Int) -> some View {
        FXAIVisualEffectSurface {
            VStack(alignment: .leading, spacing: 14) {
                Text("Replay & Stability")
                    .font(.headline)
                    .foregroundStyle(FXAITheme.textPrimary)

                Text("Last \(replayHoursBack)h • \(detail.observationCount) observations • abstain \(detail.abstainCount) • fallback \(detail.fallbackCount)")
                    .font(.caption)
                    .foregroundStyle(FXAITheme.textMuted)

                LazyVGrid(columns: [GridItem(.adaptive(minimum: 220), spacing: 14)], spacing: 14) {
                    metricListCard(title: "Action Counts", values: detail.replayActionCounts)
                    metricListCard(title: "Tier Counts", values: detail.replayTierCounts)
                    metricListCard(title: "Top Reasons", values: detail.replayTopReasons)
                    metricListCard(
                        title: "Replay Summary",
                        values: [
                            KeyValueRecord(key: "avg_confidence", value: percentString(detail.averageConfidence)),
                            KeyValueRecord(key: "avg_edge_after_costs", value: pointString(detail.averageEdgeAfterCostsPoints)),
                            KeyValueRecord(key: "avg_uncertainty", value: percentString(detail.averageUncertaintyScore)),
                            KeyValueRecord(key: "min_edge_after_costs", value: pointString(detail.minEdgeAfterCostsPoints)),
                            KeyValueRecord(key: "max_edge_after_costs", value: pointString(detail.maxEdgeAfterCostsPoints)),
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
        }
    }

    private func actionSymbol(_ action: String) -> String {
        switch action.uppercased() {
        case "BUY": "arrow.up.circle.fill"
        case "SELL": "arrow.down.circle.fill"
        default: "pause.circle.fill"
        }
    }

    private func actionTint(_ action: String, abstain: Bool) -> Color {
        if abstain || action.uppercased() == "SKIP" {
            return FXAITheme.warning
        }
        switch action.uppercased() {
        case "BUY": return FXAITheme.success
        case "SELL": return criticalColor
        default: return FXAITheme.warning
        }
    }

    private func probabilityTint(_ label: String, phase: String) -> Color {
        let base: Color
        switch label.uppercased() {
        case "BUY":
            base = FXAITheme.success
        case "SELL":
            base = criticalColor
        default:
            base = FXAITheme.warning
        }
        return phase == "Raw" ? base.opacity(0.55) : base
    }

    private func percentString(_ value: Double) -> String {
        String(format: "%.0f%%", value * 100)
    }

    private func pointString(_ value: Double) -> String {
        String(format: "%.2f pt", value)
    }

    private func boolLabel(_ value: Bool) -> String {
        value ? "YES" : "NO"
    }

    private var criticalColor: Color {
        Color(red: 0.97, green: 0.43, blue: 0.47)
    }
}
