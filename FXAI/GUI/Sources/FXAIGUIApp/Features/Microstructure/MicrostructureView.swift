import FXAIGUICore
import Foundation
import SwiftUI

struct MicrostructureView: View {
    @EnvironmentObject private var model: FXAIGUIModel

    private var visibleSymbols: [MicrostructureSymbolState] {
        guard let snapshot = model.microstructureSnapshot else { return [] }
        return Array(snapshot.symbols.prefix(30))
    }

    private var selectedSymbol: MicrostructureSymbolState? {
        guard !visibleSymbols.isEmpty else { return nil }
        return visibleSymbols.first(where: { $0.symbol == model.selectedMicrostructureSymbol }) ?? visibleSymbols.first
    }

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 18) {
                SectionHeader(
                    title: "Microstructure",
                    subtitle: "Inspect live MT5 tick-flow proxies, liquidity stress, stop-run rejection risk, session handoff behavior, and the runtime trade gate they currently imply."
                )

                if let snapshot = model.microstructureSnapshot {
                    statusGrid(snapshot: snapshot)
                    symbolWorkspace
                    artifactPanel(snapshot: snapshot)
                } else {
                    EmptyStateView(
                        title: "Microstructure service is not visible yet",
                        message: "Install and start FXAI_MicrostructureProbe in MT5 Services, then refresh the GUI so the operator shell can read the latest runtime snapshot.",
                        symbolName: "waveform.path.ecg.rectangle.fill"
                    )
                }
            }
        }
    }

    private func statusGrid(snapshot: MicrostructureSnapshot) -> some View {
        LazyVGrid(columns: [GridItem(.adaptive(minimum: 190), spacing: 16, alignment: .top)], spacing: 16) {
            MetricCard(
                title: "Service",
                value: snapshot.serviceStatus.stale ? "Stale" : (snapshot.serviceStatus.ok ? "Ready" : "Error"),
                footnote: serviceFootnote(snapshot.serviceStatus),
                symbolName: snapshot.serviceStatus.stale ? "clock.badge.exclamationmark.fill" : (snapshot.serviceStatus.ok ? "checkmark.seal.fill" : "xmark.octagon.fill"),
                tint: snapshot.serviceStatus.stale ? FXAITheme.warning : (snapshot.serviceStatus.ok ? FXAITheme.success : criticalColor)
            )
            MetricCard(
                title: "Symbols",
                value: "\(snapshot.symbols.count)",
                footnote: "FX pairs currently exposed through the microstructure runtime snapshot.",
                symbolName: "arrow.left.and.right.square.fill",
                tint: FXAITheme.accentSoft
            )
            MetricCard(
                title: "Hostile",
                value: percentString(snapshot.symbols.map(\.hostileExecutionScore).max() ?? 0),
                footnote: "Highest hostile execution score across the current symbol set.",
                symbolName: "shield.lefthalf.filled.trianglebadge.exclamationmark",
                tint: warningColor
            )
            MetricCard(
                title: "Liquidity Stress",
                value: percentString(snapshot.symbols.map(\.liquidityStressScore).max() ?? 0),
                footnote: "Maximum liquidity stress score seen in the current snapshot.",
                symbolName: "drop.triangle.fill",
                tint: FXAITheme.accentSoft
            )
        }
    }

    private var symbolWorkspace: some View {
        FXAIVisualEffectSurface {
            VStack(alignment: .leading, spacing: 14) {
                Text("Symbol State")
                    .font(.headline)
                    .foregroundStyle(FXAITheme.textPrimary)

                if visibleSymbols.isEmpty {
                    Text("No symbol-level microstructure state is available yet.")
                        .foregroundStyle(FXAITheme.textSecondary)
                } else {
                    ViewThatFits(in: .horizontal) {
                        HStack(alignment: .top, spacing: 18) {
                            symbolList
                                .frame(width: 350, alignment: .topLeading)
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
    }

    private var symbolList: some View {
        VStack(alignment: .leading, spacing: 10) {
            ForEach(visibleSymbols) { symbolState in
                Button {
                    model.selectedMicrostructureSymbol = symbolState.symbol
                } label: {
                    HStack(alignment: .top, spacing: 10) {
                        VStack(alignment: .leading, spacing: 4) {
                            HStack(spacing: 8) {
                                Text(symbolState.symbol)
                                    .font(.subheadline.weight(.semibold))
                                    .foregroundStyle(FXAITheme.textPrimary)
                                Text(symbolState.tradeGate)
                                    .font(.caption2.weight(.bold))
                                    .foregroundStyle(gateColor(symbolState.tradeGate))
                                    .padding(.horizontal, 6)
                                    .padding(.vertical, 2)
                                    .background(
                                        Capsule(style: .continuous)
                                            .fill(gateColor(symbolState.tradeGate).opacity(0.12))
                                    )
                            }
                            Text(symbolState.microstructureRegime.replacingOccurrences(of: "_", with: " ").capitalized)
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
                            Text(percentString(symbolState.hostileExecutionScore))
                                .font(.subheadline.weight(.semibold))
                                .foregroundStyle(gateColor(symbolState.tradeGate))
                            Text(percentString(symbolState.liquidityStressScore))
                                .font(.caption)
                                .foregroundStyle(FXAITheme.textMuted)
                        }
                    }
                    .padding(12)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .background(
                        RoundedRectangle(cornerRadius: 16, style: .continuous)
                            .fill(model.selectedMicrostructureSymbol == symbolState.symbol ? FXAITheme.accentSoft.opacity(0.14) : FXAITheme.panel.opacity(0.55))
                    )
                }
                .buttonStyle(.plain)
            }
        }
    }

    private var symbolDetail: some View {
        VStack(alignment: .leading, spacing: 16) {
            if let selectedSymbol {
                HStack {
                    VStack(alignment: .leading, spacing: 4) {
                        Text(selectedSymbol.symbol)
                            .font(.title3.weight(.semibold))
                            .foregroundStyle(FXAITheme.textPrimary)
                        if !selectedSymbol.brokerSymbol.isEmpty && selectedSymbol.brokerSymbol != selectedSymbol.symbol {
                            Text(selectedSymbol.brokerSymbol)
                                .font(.caption)
                                .foregroundStyle(FXAITheme.textMuted)
                        }
                    }
                    Spacer()
                    StatusBadge(title: "Gate", value: selectedSymbol.tradeGate, tint: gateColor(selectedSymbol.tradeGate))
                }

                LazyVGrid(columns: [GridItem(.adaptive(minimum: 200), spacing: 12, alignment: .top)], spacing: 12) {
                    detailMetric(title: "Hostile Execution", value: percentString(selectedSymbol.hostileExecutionScore), tint: warningColor)
                    detailMetric(title: "Liquidity Stress", value: percentString(selectedSymbol.liquidityStressScore), tint: FXAITheme.accentSoft)
                    detailMetric(title: "Tick Imbalance 30s", value: signedString(selectedSymbol.tickImbalance30s), tint: FXAITheme.accent)
                    detailMetric(title: "Directional Efficiency 60s", value: percentString(selectedSymbol.directionalEfficiency60s), tint: FXAITheme.success)
                    detailMetric(title: "Spread Z-Score 60s", value: optionalDoubleString(selectedSymbol.spreadZScore60s), tint: FXAITheme.warning)
                    detailMetric(title: "Tick Rate Z-Score 60s", value: optionalDoubleString(selectedSymbol.tickRateZScore60s), tint: FXAITheme.accentSoft)
                    detailMetric(title: "Vol Burst 5m", value: optionalDoubleString(selectedSymbol.volBurstScore5m), tint: FXAITheme.warning)
                    detailMetric(title: "Sweep Rejection 60s", value: percentString(selectedSymbol.breakoutReversalScore60s), tint: criticalColor)
                }

                VStack(alignment: .leading, spacing: 8) {
                    metricRow(label: "Session", value: selectedSymbol.sessionTag)
                    metricRow(label: "Handoff", value: selectedSymbol.handoffFlag ? "Active" : "Quiet")
                    metricRow(label: "Minutes Since Open", value: optionalIntString(selectedSymbol.minutesSinceSessionOpen))
                    metricRow(label: "Minutes To Close", value: optionalIntString(selectedSymbol.minutesToSessionClose))
                    metricRow(label: "Open Burst", value: percentString(selectedSymbol.sessionOpenBurstScore))
                    metricRow(label: "Session Spread", value: percentString(selectedSymbol.sessionSpreadBehaviorScore))
                    metricRow(label: "Spread Current", value: optionalDoubleString(selectedSymbol.spreadCurrent))
                    metricRow(label: "Realized Vol 5m", value: optionalDoubleString(selectedSymbol.realizedVol5m))
                    metricRow(label: "Extrema Breach 60s", value: percentString(selectedSymbol.localExtremaBreachScore60s))
                    metricRow(label: "Sweep & Reject", value: selectedSymbol.sweepAndRejectFlag60s ? "Yes" : "No")
                    metricRow(label: "Exhaustion Proxy", value: percentString(selectedSymbol.exhaustionProxy60s))
                    metricRow(label: "Silent Gap", value: "\(Int(selectedSymbol.silentGapSecondsCurrent.rounded()))s")
                }

                if !selectedSymbol.reasons.isEmpty {
                    Divider().overlay(FXAITheme.stroke.opacity(0.75))
                    VStack(alignment: .leading, spacing: 6) {
                        Text("Reasons")
                            .font(.subheadline.weight(.semibold))
                            .foregroundStyle(FXAITheme.textPrimary)
                        ForEach(selectedSymbol.reasons, id: \.self) { reason in
                            Text(reason)
                                .font(.caption)
                                .foregroundStyle(FXAITheme.textMuted)
                        }
                    }
                }
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    private func artifactPanel(snapshot: MicrostructureSnapshot) -> some View {
        FXAIVisualEffectSurface {
            VStack(alignment: .leading, spacing: 14) {
                Text("Health & Artifacts")
                    .font(.headline)
                    .foregroundStyle(FXAITheme.textPrimary)

                VStack(alignment: .leading, spacing: 6) {
                    ForEach(snapshot.healthSummary) { item in
                        metricRow(label: item.key, value: item.value)
                    }
                }

                if !snapshot.artifactPaths.isEmpty {
                    Divider().overlay(FXAITheme.stroke.opacity(0.75))
                    VStack(alignment: .leading, spacing: 6) {
                        ForEach(snapshot.artifactPaths) { item in
                            metricRow(label: item.key, value: item.value)
                        }
                    }
                }
            }
        }
    }

    private func detailMetric(title: String, value: String, tint: Color) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            Text(title)
                .font(.caption.weight(.semibold))
                .foregroundStyle(FXAITheme.textSecondary)
            Text(value)
                .font(.title3.weight(.semibold))
                .foregroundStyle(tint)
        }
        .padding(12)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(
            RoundedRectangle(cornerRadius: 16, style: .continuous)
                .fill(FXAITheme.panel.opacity(0.58))
        )
    }

    private func metricRow(label: String, value: String) -> some View {
        HStack {
            Text(label)
                .foregroundStyle(FXAITheme.textSecondary)
            Spacer(minLength: 12)
            Text(value)
                .foregroundStyle(FXAITheme.textPrimary)
                .multilineTextAlignment(.trailing)
        }
        .font(.subheadline)
    }

    private func percentString(_ value: Double) -> String {
        "\(Int((value * 100).rounded()))%"
    }

    private func signedString(_ value: Double) -> String {
        let sign = value > 0 ? "+" : ""
        return "\(sign)\(String(format: "%.2f", value))"
    }

    private func optionalDoubleString(_ value: Double?) -> String {
        guard let value else { return "n/a" }
        return String(format: "%.2f", value)
    }

    private func optionalIntString(_ value: Int?) -> String {
        guard let value else { return "n/a" }
        return "\(value)"
    }

    private func serviceFootnote(_ status: MicrostructureServiceStatus) -> String {
        var parts: [String] = []
        if let lastSuccessAt = status.lastSuccessAt {
            parts.append("Last success \(FXAIFormatting.relativeDateString(for: lastSuccessAt))")
        }
        if let lastError = status.lastError, !lastError.isEmpty {
            parts.append(lastError)
        }
        if parts.isEmpty {
            parts.append("Awaiting runtime microstructure snapshot.")
        }
        return parts.joined(separator: " • ")
    }

    private func gateColor(_ gate: String) -> Color {
        switch gate.uppercased() {
        case "BLOCK":
            criticalColor
        case "CAUTION":
            warningColor
        case "ALLOW":
            FXAITheme.success
        default:
            FXAITheme.textMuted
        }
    }

    private var warningColor: Color { FXAITheme.warning }
    private var criticalColor: Color { Color(red: 0.97, green: 0.43, blue: 0.47) }
}
