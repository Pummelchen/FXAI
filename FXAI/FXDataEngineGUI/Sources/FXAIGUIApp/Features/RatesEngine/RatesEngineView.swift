import FXAIGUICore
import SwiftUI

struct RatesEngineView: View {
    @EnvironmentObject private var model: FXAIGUIModel

    private var visiblePairs: [RatesEnginePairState] {
        guard let snapshot = model.ratesEngineSnapshot else { return [] }
        return Array(snapshot.pairs.prefix(24))
    }

    private var selectedPair: RatesEnginePairState? {
        guard !visiblePairs.isEmpty else { return nil }
        return visiblePairs.first(where: { $0.pair == model.selectedRatesSymbol }) ?? visiblePairs.first
    }

    private var selectedCurrencyStates: [RatesEngineCurrencyState] {
        guard let snapshot = model.ratesEngineSnapshot, let selectedPair else { return [] }
        let codes = Set([selectedPair.baseCurrency, selectedPair.quoteCurrency])
        return snapshot.currencies.filter { codes.contains($0.currency) }
    }

    private var filteredEvents: [RatesEnginePolicyEvent] {
        guard let snapshot = model.ratesEngineSnapshot else { return [] }
        guard let selectedPair else { return Array(snapshot.recentPolicyEvents.prefix(16)) }
        let codes = Set([selectedPair.baseCurrency, selectedPair.quoteCurrency])
        let filtered = snapshot.recentPolicyEvents.filter { codes.contains($0.currency) }
        return Array((filtered.isEmpty ? snapshot.recentPolicyEvents : filtered).prefix(16))
    }

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 18) {
                SectionHeader(
                    title: "Rates Engine",
                    subtitle: "Inspect provider health, policy repricing, expected-path divergence, pair-level rates gates, and recent central-bank or macro-policy transmission events."
                )

                if let snapshot = model.ratesEngineSnapshot {
                    topStatus(snapshot: snapshot)
                    currencyHeatmap(snapshot: snapshot)
                    pairWorkspace
                    policyTape
                    artifactPanel(snapshot: snapshot)
                } else {
                    EmptyStateView(
                        title: "Rates Engine is not running yet",
                        message: "Validate the rates-engine config, start the daemon, and refresh the GUI so the operator shell can display rates-aware policy state.",
                        symbolName: "chart.line.text.clipboard.fill"
                    )
                }
            }
        }
    }

    private func topStatus(snapshot: RatesEngineSnapshot) -> some View {
        LazyVGrid(columns: [GridItem(.adaptive(minimum: 180), spacing: 16, alignment: .top)], spacing: 16) {
            ForEach(snapshot.sourceStatuses) { status in
                MetricCard(
                    title: status.id.replacingOccurrences(of: "_", with: " ").capitalized,
                    value: status.stale ? "Stale" : (status.ok ? "Ready" : "Error"),
                    footnote: sourceFootnote(for: status),
                    symbolName: status.stale ? "clock.badge.exclamationmark.fill" : (status.ok ? "checkmark.seal.fill" : "xmark.octagon.fill"),
                    tint: status.stale ? FXAITheme.warning : (status.ok ? FXAITheme.success : criticalColor)
                )
            }

            MetricCard(
                title: "Pairs",
                value: "\(snapshot.pairs.count)",
                footnote: "Pair-level rates states currently exposed to runtime and the GUI.",
                symbolName: "arrow.left.and.right.square.fill",
                tint: FXAITheme.accentSoft
            )

            MetricCard(
                title: "Policy Tape",
                value: "\(snapshot.recentPolicyEvents.count)",
                footnote: "Recent events that are contributing to policy-path and repricing context.",
                symbolName: "megaphone.fill",
                tint: FXAITheme.accentSoft
            )
        }
    }

    private func currencyHeatmap(snapshot: RatesEngineSnapshot) -> some View {
        FXAIVisualEffectSurface {
            VStack(alignment: .leading, spacing: 14) {
                Text("Currency Policy State")
                    .font(.headline)
                    .foregroundStyle(FXAITheme.textPrimary)

                LazyVGrid(columns: [GridItem(.adaptive(minimum: 220), spacing: 14, alignment: .top)], spacing: 14) {
                    ForEach(snapshot.currencies) { currency in
                        VStack(alignment: .leading, spacing: 10) {
                            HStack {
                                Text(currency.currency)
                                    .font(.headline)
                                    .foregroundStyle(FXAITheme.textPrimary)
                                Spacer()
                                StatusBadge(
                                    title: "Mode",
                                    value: currency.frontEndBasis == "manual_market_input" ? "Manual" : "Proxy",
                                    tint: currency.frontEndBasis == "manual_market_input" ? FXAITheme.success : FXAITheme.warning
                                )
                            }

                            metricRow(label: "Repricing", value: percentString(currency.policyRepricingScore))
                            metricRow(label: "Surprise", value: percentString(currency.policySurpriseScore))
                            metricRow(label: "Uncertainty", value: percentString(currency.policyUncertaintyScore))
                            metricRow(label: "Direction", value: signedString(currency.policyDirectionScore))
                            metricRow(label: "Path", value: optionalDoubleString(currency.expectedPathLevel))
                            metricRow(label: "Curve", value: curveLabel(currency))

                            if let reason = currency.reasons.first {
                                Text(reason)
                                    .font(.caption)
                                    .foregroundStyle(FXAITheme.textMuted)
                                    .lineLimit(3)
                            }
                        }
                        .padding(16)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .background(
                            RoundedRectangle(cornerRadius: 18, style: .continuous)
                                .fill(currency.stale ? FXAITheme.warning.opacity(0.10) : FXAITheme.accentSoft.opacity(0.08))
                        )
                    }
                }
            }
        }
    }

    private var pairWorkspace: some View {
        FXAIVisualEffectSurface {
            VStack(alignment: .leading, spacing: 14) {
                Text("Pair Divergence")
                    .font(.headline)
                    .foregroundStyle(FXAITheme.textPrimary)

                if visiblePairs.isEmpty {
                    Text("No pair-level rates state is available yet.")
                        .foregroundStyle(FXAITheme.textSecondary)
                } else {
                    ViewThatFits(in: .horizontal) {
                        HStack(alignment: .top, spacing: 18) {
                            pairList
                                .frame(width: 330, alignment: .topLeading)
                            pairDetail
                        }
                        VStack(alignment: .leading, spacing: 18) {
                            pairList
                            pairDetail
                        }
                    }
                }
            }
        }
    }

    private var pairList: some View {
        VStack(alignment: .leading, spacing: 10) {
            ForEach(visiblePairs) { pair in
                Button {
                    model.selectedRatesSymbol = pair.pair
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
                            Text(pair.ratesRegime.replacingOccurrences(of: "_", with: " ").capitalized)
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
                            Text(percentString(pair.ratesRiskScore))
                                .font(.subheadline.weight(.semibold))
                                .foregroundStyle(gateColor(pair.tradeGate))
                            Text(optionalDoubleString(pair.policyDivergenceScore))
                                .font(.caption)
                                .foregroundStyle(FXAITheme.textMuted)
                        }
                    }
                    .padding(12)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .background(
                        RoundedRectangle(cornerRadius: 16, style: .continuous)
                            .fill(model.selectedRatesSymbol == pair.pair ? FXAITheme.accentSoft.opacity(0.14) : FXAITheme.panel.opacity(0.55))
                    )
                }
                .buttonStyle(.plain)
            }
        }
    }

    private var pairDetail: some View {
        VStack(alignment: .leading, spacing: 16) {
            if let selectedPair {
                HStack {
                    Text(selectedPair.pair)
                        .font(.title3.weight(.semibold))
                        .foregroundStyle(FXAITheme.textPrimary)
                    Spacer()
                    StatusBadge(title: "Gate", value: selectedPair.tradeGate, tint: gateColor(selectedPair.tradeGate))
                }

                VStack(alignment: .leading, spacing: 8) {
                    metricRow(label: "Rates Risk", value: percentString(selectedPair.ratesRiskScore))
                    metricRow(label: "Policy Divergence", value: optionalDoubleString(selectedPair.policyDivergenceScore))
                    metricRow(label: "Curve Divergence", value: optionalDoubleString(selectedPair.curveDivergenceScore))
                    metricRow(label: "Expected Path Diff", value: optionalDoubleString(selectedPair.expectedPathDiff))
                    metricRow(label: "Front-End Diff", value: optionalDoubleString(selectedPair.frontEndDiff))
                    metricRow(label: "Alignment", value: selectedPair.policyAlignment)
                }

                if !selectedCurrencyStates.isEmpty {
                    Divider().overlay(FXAITheme.stroke.opacity(0.75))
                    ForEach(selectedCurrencyStates) { state in
                        VStack(alignment: .leading, spacing: 6) {
                            Text(state.currency)
                                .font(.subheadline.weight(.semibold))
                                .foregroundStyle(FXAITheme.textPrimary)
                            metricRow(label: "Mode", value: state.expectedPathBasis)
                            metricRow(label: "Repricing", value: percentString(state.policyRepricingScore))
                            metricRow(label: "Uncertainty", value: percentString(state.policyUncertaintyScore))
                            if let reason = state.reasons.first {
                                Text(reason)
                                    .font(.caption)
                                    .foregroundStyle(FXAITheme.textMuted)
                            }
                        }
                    }
                }

                if !selectedPair.reasons.isEmpty {
                    Divider().overlay(FXAITheme.stroke.opacity(0.75))
                    VStack(alignment: .leading, spacing: 6) {
                        Text("Reasons")
                            .font(.subheadline.weight(.semibold))
                            .foregroundStyle(FXAITheme.textPrimary)
                        ForEach(selectedPair.reasons, id: \.self) { reason in
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

    private var policyTape: some View {
        FXAIVisualEffectSurface {
            VStack(alignment: .leading, spacing: 14) {
                Text("Policy Tape")
                    .font(.headline)
                    .foregroundStyle(FXAITheme.textPrimary)

                if filteredEvents.isEmpty {
                    Text("No recent policy-linked events are available yet.")
                        .foregroundStyle(FXAITheme.textSecondary)
                } else {
                    VStack(spacing: 10) {
                        ForEach(filteredEvents) { event in
                            HStack(alignment: .top, spacing: 12) {
                                VStack(alignment: .leading, spacing: 4) {
                                    HStack(spacing: 8) {
                                        Text(event.currency)
                                            .font(.caption.weight(.bold))
                                            .foregroundStyle(FXAITheme.accent)
                                        Text(event.source.uppercased())
                                            .font(.caption2)
                                            .foregroundStyle(FXAITheme.textMuted)
                                    }
                                    Text(event.title)
                                        .font(.subheadline.weight(.semibold))
                                        .foregroundStyle(FXAITheme.textPrimary)
                                        .lineLimit(2)
                                    Text(event.domain)
                                        .font(.caption)
                                        .foregroundStyle(FXAITheme.textMuted)
                                }
                                Spacer(minLength: 12)
                                VStack(alignment: .trailing, spacing: 4) {
                                    Text(percentString(event.policyRelevanceScore))
                                        .font(.caption.weight(.semibold))
                                        .foregroundStyle(FXAITheme.textPrimary)
                                    Text(signedString(event.direction))
                                        .font(.caption)
                                        .foregroundStyle(event.direction >= 0 ? FXAITheme.success : FXAITheme.warning)
                                    if let publishedAt = event.publishedAt {
                                        Text(FXAIFormatting.relativeDateString(for: publishedAt))
                                            .font(.caption2)
                                            .foregroundStyle(FXAITheme.textMuted)
                                    }
                                }
                            }
                            .padding(12)
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .background(
                                RoundedRectangle(cornerRadius: 16, style: .continuous)
                                    .fill(FXAITheme.panel.opacity(0.60))
                            )
                        }
                    }
                }
            }
        }
    }

    private func artifactPanel(snapshot: RatesEngineSnapshot) -> some View {
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

    private func sourceFootnote(for status: RatesEngineSourceStatus) -> String {
        let updated = status.lastUpdateAt.map { FXAIFormatting.relativeDateString(for: $0) } ?? "unknown"
        if let mode = status.mode, !mode.isEmpty {
            return "\(mode) • updated \(updated)"
        }
        if let coverage = status.coverageRatio {
            return "coverage \(Int((coverage * 100).rounded()))% • updated \(updated)"
        }
        return "updated \(updated)"
    }

    private func gateColor(_ gate: String) -> Color {
        switch gate.uppercased() {
        case "BLOCK": criticalColor
        case "CAUTION": FXAITheme.warning
        default: FXAITheme.success
        }
    }

    private var criticalColor: Color {
        Color(.sRGB, red: 0.90, green: 0.34, blue: 0.30, opacity: 0.96)
    }

    private func metricRow(label: String, value: String) -> some View {
        HStack(alignment: .firstTextBaseline) {
            Text(label)
                .font(.caption)
                .foregroundStyle(FXAITheme.textMuted)
            Spacer()
            Text(value)
                .font(.caption.weight(.semibold))
                .foregroundStyle(FXAITheme.textPrimary)
        }
    }

    private func percentString(_ value: Double) -> String {
        String(format: "%.0f%%", value * 100.0)
    }

    private func optionalDoubleString(_ value: Double?) -> String {
        guard let value else { return "n/a" }
        return String(format: "%.2f", value)
    }

    private func signedString(_ value: Double) -> String {
        if value > 0 {
            return String(format: "+%.2f", value)
        }
        return String(format: "%.2f", value)
    }

    private func curveLabel(_ state: RatesEngineCurrencyState) -> String {
        if let slope = state.curveSlope2s10s {
            return "\(String(format: "%.2f", slope)) • \(state.curveShapeRegime)"
        }
        return state.curveShapeRegime
    }
}
