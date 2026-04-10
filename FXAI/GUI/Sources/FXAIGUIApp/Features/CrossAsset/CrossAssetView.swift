import FXAIGUICore
import SwiftUI

struct CrossAssetView: View {
    @EnvironmentObject private var model: FXAIGUIModel

    private var visiblePairs: [CrossAssetPairState] {
        guard let snapshot = model.crossAssetSnapshot else { return [] }
        return Array(snapshot.pairs.prefix(24))
    }

    private var selectedPair: CrossAssetPairState? {
        guard !visiblePairs.isEmpty else { return nil }
        return visiblePairs.first(where: { $0.pair == model.selectedCrossAssetSymbol }) ?? visiblePairs.first
    }

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 18) {
                SectionHeader(
                    title: "Cross-Asset State",
                    subtitle: "Track the shared macro and liquidity regime built from rates, equity-risk, commodities, volatility, and dollar-liquidity proxies before that context reaches routing, calibration, and runtime risk controls."
                )

                if let snapshot = model.crossAssetSnapshot {
                    topStatus(snapshot: snapshot)
                    sourcePanel(snapshot: snapshot)
                    pairWorkspace(snapshot: snapshot)
                    proxyAndTransitionPanel(snapshot: snapshot)
                } else {
                    EmptyStateView(
                        title: "Cross-asset artifacts not found",
                        message: "Run cross-asset-validate, start the MT5 cross-asset probe service, build a shared snapshot with cross-asset-once or cross-asset-daemon, and refresh the GUI.",
                        symbolName: "globe.americas.fill"
                    )
                }
            }
        }
    }

    private func topStatus(snapshot: CrossAssetSnapshot) -> some View {
        let macroState = value(for: "macro_state", in: snapshot.stateLabels) ?? "UNKNOWN"
        let liquidityState = value(for: "liquidity_state", in: snapshot.stateLabels) ?? "UNKNOWN"
        let blockedCount = snapshot.pairs.filter { $0.tradeGate.uppercased() == "BLOCK" }.count
        let staleSourceCount = snapshot.sourceStatuses.filter(\.stale).count

        return LazyVGrid(columns: [GridItem(.adaptive(minimum: 180), spacing: 16)], spacing: 16) {
            MetricCard(
                title: "Macro State",
                value: prettifyState(macroState),
                footnote: "Current top shared cross-asset regime exposed to runtime and GUI surfaces.",
                symbolName: "globe.americas.fill",
                tint: stateTint(macroState)
            )
            MetricCard(
                title: "Liquidity",
                value: prettifyState(liquidityState),
                footnote: "Composite liquidity posture derived from volatility, USD-liquidity, and cross-asset dislocation pressure.",
                symbolName: "drop.fill",
                tint: stateTint(liquidityState)
            )
            MetricCard(
                title: "Blocked Pairs",
                value: "\(blockedCount)",
                footnote: "Pairs currently pushed into hard block by the cross-asset state layer.",
                symbolName: "hand.raised.fill",
                tint: criticalColor
            )
            MetricCard(
                title: "Stale Sources",
                value: "\(staleSourceCount)",
                footnote: "Critical or optional source groups that are currently stale.",
                symbolName: "clock.badge.exclamationmark.fill",
                tint: staleSourceCount > 0 ? FXAITheme.warning : FXAITheme.success
            )
        }
    }

    private func sourcePanel(snapshot: CrossAssetSnapshot) -> some View {
        FXAIVisualEffectSurface {
            VStack(alignment: .leading, spacing: 14) {
                Text("Source Health and State Scores")
                    .font(.headline)
                    .foregroundStyle(FXAITheme.textPrimary)

                LazyVGrid(columns: [GridItem(.adaptive(minimum: 220), spacing: 14)], spacing: 14) {
                    ForEach(snapshot.sourceStatuses) { status in
                        VStack(alignment: .leading, spacing: 8) {
                            HStack {
                                Text(status.id.replacingOccurrences(of: "_", with: " ").capitalized)
                                    .font(.subheadline.weight(.semibold))
                                    .foregroundStyle(FXAITheme.textPrimary)
                                Spacer()
                                StatusBadge(
                                    title: "State",
                                    value: status.stale ? "Stale" : (status.ok ? "Ready" : "Missing"),
                                    tint: status.stale ? FXAITheme.warning : (status.ok ? FXAITheme.success : criticalColor)
                                )
                            }
                            metricRow(label: "Updated", value: shortDate(status.lastUpdateAt))
                            if let proxySymbol = status.proxySymbol, !proxySymbol.isEmpty {
                                metricRow(label: "Proxy", value: proxySymbol)
                            }
                            if let availableSymbols = status.availableSymbols {
                                metricRow(label: "Available", value: "\(availableSymbols)")
                            }
                            if let configuredSymbols = status.configuredSymbols {
                                metricRow(label: "Configured", value: "\(configuredSymbols)")
                            }
                        }
                        .padding(14)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .background(
                            RoundedRectangle(cornerRadius: 16, style: .continuous)
                                .fill((status.stale ? FXAITheme.warning : FXAITheme.accentSoft).opacity(0.10))
                        )
                    }

                    metricListCard(title: "State Scores", values: snapshot.stateScores)
                    metricListCard(title: "Normalized Features", values: Array(snapshot.features.prefix(8)))
                    metricListCard(title: "Quality Flags", values: snapshot.qualityFlags)
                }
            }
        }
    }

    private func pairWorkspace(snapshot: CrossAssetSnapshot) -> some View {
        FXAIVisualEffectSurface {
            VStack(alignment: .leading, spacing: 14) {
                Text("Pair Impact")
                    .font(.headline)
                    .foregroundStyle(FXAITheme.textPrimary)

                ViewThatFits(in: .horizontal) {
                    HStack(alignment: .top, spacing: 18) {
                        pairList
                            .frame(width: 340, alignment: .topLeading)
                        pairDetail(snapshot: snapshot)
                    }

                    VStack(alignment: .leading, spacing: 18) {
                        pairList
                        pairDetail(snapshot: snapshot)
                    }
                }
            }
        }
    }

    private var pairList: some View {
        VStack(alignment: .leading, spacing: 10) {
            ForEach(visiblePairs) { pair in
                Button {
                    model.selectedCrossAssetSymbol = pair.pair
                } label: {
                    HStack(alignment: .top, spacing: 10) {
                        VStack(alignment: .leading, spacing: 4) {
                            HStack(spacing: 8) {
                                Text(pair.pair)
                                    .font(.subheadline.weight(.semibold))
                                    .foregroundStyle(FXAITheme.textPrimary)
                                Text(pair.tradeGate)
                                    .font(.caption2.weight(.bold))
                                    .foregroundStyle(gateColor(pair.tradeGate))
                                    .padding(.horizontal, 6)
                                    .padding(.vertical, 2)
                                    .background(
                                        Capsule(style: .continuous)
                                            .fill(gateColor(pair.tradeGate).opacity(0.12))
                                    )
                            }
                            Text(prettifyState(pair.macroState))
                                .font(.caption)
                                .foregroundStyle(FXAITheme.textSecondary)
                            if let reason = pair.reasons.first {
                                Text(reason)
                                    .font(.caption)
                                    .foregroundStyle(FXAITheme.textMuted)
                                    .lineLimit(2)
                            }
                        }
                        Spacer(minLength: 12)
                        VStack(alignment: .trailing, spacing: 4) {
                            Text(percentString(pair.pairCrossAssetRiskScore))
                                .font(.subheadline.weight(.semibold))
                                .foregroundStyle(gateColor(pair.tradeGate))
                            Text(percentString(pair.pairSensitivity))
                                .font(.caption)
                                .foregroundStyle(FXAITheme.textMuted)
                        }
                    }
                    .padding(12)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .background(
                        RoundedRectangle(cornerRadius: 16, style: .continuous)
                            .fill(model.selectedCrossAssetSymbol == pair.pair ? FXAITheme.accentSoft.opacity(0.14) : FXAITheme.panel.opacity(0.55))
                    )
                }
                .buttonStyle(.plain)
            }
        }
    }

    private func pairDetail(snapshot: CrossAssetSnapshot) -> some View {
        VStack(alignment: .leading, spacing: 16) {
            if let selectedPair {
                HStack {
                    Text(selectedPair.pair)
                        .font(.title3.weight(.semibold))
                        .foregroundStyle(FXAITheme.textPrimary)
                    Spacer()
                    StatusBadge(title: "Gate", value: selectedPair.tradeGate, tint: gateColor(selectedPair.tradeGate))
                }

                LazyVGrid(columns: [GridItem(.adaptive(minimum: 170), spacing: 12)], spacing: 12) {
                    detailMetric(title: "Risk Score", value: percentString(selectedPair.pairCrossAssetRiskScore), tint: gateColor(selectedPair.tradeGate))
                    detailMetric(title: "Sensitivity", value: percentString(selectedPair.pairSensitivity), tint: FXAITheme.accentSoft)
                    detailMetric(title: "Macro", value: prettifyState(selectedPair.macroState), tint: stateTint(selectedPair.macroState))
                    detailMetric(title: "Risk", value: prettifyState(selectedPair.riskState), tint: stateTint(selectedPair.riskState))
                    detailMetric(title: "Liquidity", value: prettifyState(selectedPair.liquidityState), tint: stateTint(selectedPair.liquidityState))
                    detailMetric(title: "Stale", value: selectedPair.stale ? "Yes" : "No", tint: selectedPair.stale ? FXAITheme.warning : FXAITheme.success)
                }

                metricListCard(
                    title: "Current Global Labels",
                    values: snapshot.stateLabels
                )

                if !selectedPair.reasons.isEmpty {
                    VStack(alignment: .leading, spacing: 8) {
                        Text("Reasons")
                            .font(.subheadline.weight(.semibold))
                            .foregroundStyle(FXAITheme.textPrimary)
                        ForEach(selectedPair.reasons, id: \.self) { reason in
                            HStack(alignment: .top, spacing: 8) {
                                Image(systemName: "circle.fill")
                                    .font(.system(size: 5))
                                    .foregroundStyle(gateColor(selectedPair.tradeGate))
                                    .padding(.top, 6)
                                Text(reason)
                                    .font(.caption)
                                    .foregroundStyle(FXAITheme.textMuted)
                            }
                        }
                    }
                }
            } else {
                Text("No pair-level cross-asset state is available.")
                    .foregroundStyle(FXAITheme.textSecondary)
            }
        }
    }

    private func proxyAndTransitionPanel(snapshot: CrossAssetSnapshot) -> some View {
        FXAIVisualEffectSurface {
            VStack(alignment: .leading, spacing: 14) {
                Text("Proxy Rail and Transitions")
                    .font(.headline)
                    .foregroundStyle(FXAITheme.textPrimary)

                ViewThatFits(in: .horizontal) {
                    HStack(alignment: .top, spacing: 18) {
                        metricListCard(title: "Selected Proxies", values: proxyRecords(snapshot.selectedProxies))
                            .frame(maxWidth: .infinity, alignment: .topLeading)
                        transitionCard(snapshot: snapshot)
                    }

                    VStack(alignment: .leading, spacing: 18) {
                        metricListCard(title: "Selected Proxies", values: proxyRecords(snapshot.selectedProxies))
                        transitionCard(snapshot: snapshot)
                    }
                }
            }
        }
    }

    private func transitionCard(snapshot: CrossAssetSnapshot) -> some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("Recent Transitions")
                .font(.subheadline.weight(.semibold))
                .foregroundStyle(FXAITheme.textPrimary)

            if snapshot.recentTransitions.isEmpty {
                Text("No recent macro or pair-gate transitions were recorded.")
                    .font(.caption)
                    .foregroundStyle(FXAITheme.textMuted)
            } else {
                ForEach(snapshot.recentTransitions.prefix(8)) { transition in
                    VStack(alignment: .leading, spacing: 4) {
                        Text("\(transition.target.uppercased()) • \(transition.type.replacingOccurrences(of: "_", with: " "))")
                            .font(.caption.weight(.semibold))
                            .foregroundStyle(FXAITheme.textPrimary)
                        Text("\(prettifyState(transition.fromValue)) -> \(prettifyState(transition.toValue))")
                            .font(.caption)
                            .foregroundStyle(FXAITheme.textSecondary)
                        Text(shortDate(transition.observedAt))
                            .font(.caption2)
                            .foregroundStyle(FXAITheme.textMuted)
                    }
                    .padding(10)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .background(
                        RoundedRectangle(cornerRadius: 14, style: .continuous)
                            .fill(FXAITheme.panel.opacity(0.55))
                    )
                }
            }

            if !snapshot.reasons.isEmpty {
                Divider().overlay(FXAITheme.stroke.opacity(0.65))
                Text("Global Reasons")
                    .font(.subheadline.weight(.semibold))
                    .foregroundStyle(FXAITheme.textPrimary)
                ForEach(snapshot.reasons, id: \.self) { reason in
                    Text(reason)
                        .font(.caption)
                        .foregroundStyle(FXAITheme.textMuted)
                }
            }
        }
        .frame(maxWidth: .infinity, alignment: .topLeading)
    }

    private func proxyRecords(_ proxies: [CrossAssetProxySelection]) -> [KeyValueRecord] {
        proxies.map { proxy in
            let fallbackLabel = proxy.fallbackUsed ? " fallback" : ""
            return KeyValueRecord(
                key: proxy.id,
                value: "\(proxy.symbol)\(fallbackLabel) • \(signedPercent(proxy.changePct1d))"
            )
        }
    }

    private func metricListCard(title: String, values: [KeyValueRecord]) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(title)
                .font(.subheadline.weight(.semibold))
                .foregroundStyle(FXAITheme.textPrimary)
            if values.isEmpty {
                Text("No values available.")
                    .font(.caption)
                    .foregroundStyle(FXAITheme.textMuted)
            } else {
                ForEach(values) { item in
                    metricRow(label: item.key, value: item.value)
                }
            }
        }
        .padding(14)
        .frame(maxWidth: .infinity, alignment: .topLeading)
        .background(
            RoundedRectangle(cornerRadius: 16, style: .continuous)
                .fill(FXAITheme.panel.opacity(0.55))
        )
    }

    private func detailMetric(title: String, value: String, tint: Color) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            Text(title)
                .font(.caption.weight(.semibold))
                .foregroundStyle(FXAITheme.textSecondary)
            Text(value)
                .font(.headline.weight(.semibold))
                .foregroundStyle(tint)
        }
        .padding(12)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(
            RoundedRectangle(cornerRadius: 14, style: .continuous)
                .fill(FXAITheme.panel.opacity(0.55))
        )
    }

    private func metricRow(label: String, value: String) -> some View {
        HStack(alignment: .top, spacing: 12) {
            Text(label.replacingOccurrences(of: "_", with: " ").capitalized)
                .font(.caption)
                .foregroundStyle(FXAITheme.textSecondary)
            Spacer(minLength: 10)
            Text(value)
                .font(.caption.weight(.semibold))
                .foregroundStyle(FXAITheme.textPrimary)
                .multilineTextAlignment(.trailing)
        }
    }

    private func percentString(_ value: Double) -> String {
        String(format: "%.0f%%", value * 100.0)
    }

    private func signedPercent(_ value: Double) -> String {
        String(format: "%+.2f%%", value)
    }

    private func shortDate(_ date: Date?) -> String {
        guard let date else { return "n/a" }
        return date.formatted(date: .abbreviated, time: .shortened)
    }

    private func prettifyState(_ value: String) -> String {
        guard !value.isEmpty else { return "Unknown" }
        return value.replacingOccurrences(of: "_", with: " ").capitalized
    }

    private func value(for key: String, in records: [KeyValueRecord]) -> String? {
        records.first(where: { $0.key == key })?.value
    }

    private func gateColor(_ gate: String) -> Color {
        switch gate.uppercased() {
        case "BLOCK":
            criticalColor
        case "CAUTION":
            FXAITheme.warning
        case "ALLOW":
            FXAITheme.success
        default:
            FXAITheme.textSecondary
        }
    }

    private func stateTint(_ state: String) -> Color {
        switch state.uppercased() {
        case "RATES_REPRICING", "STRESSED":
            criticalColor
        case "RISK_OFF", "CAUTION", "MIXED":
            FXAITheme.warning
        case "NORMAL", "ALLOW":
            FXAITheme.success
        default:
            FXAITheme.accentSoft
        }
    }

    private var criticalColor: Color {
        Color(red: 0.97, green: 0.43, blue: 0.47)
    }
}
