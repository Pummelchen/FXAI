import FXAIGUICore
import SwiftUI

struct DriftGovernanceView: View {
    @EnvironmentObject private var model: FXAIGUIModel

    private var visibleSymbols: [DriftGovernanceSymbolSnapshot] {
        model.driftGovernanceSnapshot?.symbols ?? []
    }

    private var selectedSymbol: DriftGovernanceSymbolSnapshot? {
        model.selectedDriftGovernanceDetail
    }

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 18) {
                SectionHeader(
                    title: "Drift Governance",
                    subtitle: "Track plugin-health drift, pair-level demotions, challenger promotion eligibility, and the governance actions currently constraining the live plugin zoo."
                )

                if let snapshot = model.driftGovernanceSnapshot, !snapshot.symbols.isEmpty {
                    topStatus(snapshot: snapshot)
                    symbolWorkspace(snapshot: snapshot)
                    statusPanel(snapshot: snapshot)
                } else {
                    EmptyStateView(
                        title: "Drift-governance artifacts not found",
                        message: "Run drift-governance-validate, execute drift-governance-run for the active profile, rebuild the report, and refresh the GUI to inspect plugin health and governance actions.",
                        symbolName: "waveform.and.magnifyingglass"
                    )
                }
            }
        }
    }

    private func topStatus(snapshot: DriftGovernanceSnapshot) -> some View {
        let appliedActions = snapshot.symbols
            .flatMap(\.recentActions)
            .filter(\.actionApplied)
            .count
        let candidates = snapshot.symbols
            .flatMap(\.plugins)
            .filter { $0.challengerEvaluation?.qualifies == true }
            .count

        return LazyVGrid(columns: [GridItem(.adaptive(minimum: 180), spacing: 16)], spacing: 16) {
            MetricCard(
                title: "Tracked Symbols",
                value: "\(snapshot.symbolCount)",
                footnote: "Pairs with active plugin-health monitoring in the current report.",
                symbolName: "point.3.connected.trianglepath.dotted",
                tint: FXAITheme.accentSoft
            )
            MetricCard(
                title: "Tracked Plugins",
                value: "\(snapshot.pluginCount)",
                footnote: "Governed plugin scopes included in the latest cycle.",
                symbolName: "shippingbox.fill",
                tint: FXAITheme.accent
            )
            MetricCard(
                title: "Applied Actions",
                value: "\(appliedActions)",
                footnote: "Latest action count \(snapshot.latestActionCount) • mode \(snapshot.actionMode).",
                symbolName: "shield.lefthalf.filled",
                tint: appliedActions > 0 ? FXAITheme.warning : FXAITheme.success
            )
            MetricCard(
                title: "Eligible Challengers",
                value: "\(candidates)",
                footnote: "Challengers that currently satisfy promotion-review thresholds.",
                symbolName: "rosette",
                tint: candidates > 0 ? FXAITheme.success : FXAITheme.textMuted
            )
        }
    }

    private func symbolWorkspace(snapshot _: DriftGovernanceSnapshot) -> some View {
        FXAIVisualEffectSurface {
            VStack(alignment: .leading, spacing: 14) {
                Text("Symbol Governance")
                    .font(.headline)
                    .foregroundStyle(FXAITheme.textPrimary)

                ViewThatFits(in: .horizontal) {
                    HStack(alignment: .top, spacing: 18) {
                        symbolList
                            .frame(width: 360, alignment: .topLeading)
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
                    model.selectedRuntimeSymbol = symbolState.symbol
                } label: {
                    HStack(alignment: .top, spacing: 10) {
                        VStack(alignment: .leading, spacing: 4) {
                            HStack(spacing: 8) {
                                Text(symbolState.symbol)
                                    .font(.subheadline.weight(.semibold))
                                    .foregroundStyle(FXAITheme.textPrimary)
                                Text(worstGovernanceState(symbolState))
                                    .font(.caption2.weight(.bold))
                                    .foregroundStyle(governanceTint(worstGovernanceState(symbolState)))
                                    .padding(.horizontal, 6)
                                    .padding(.vertical, 2)
                                    .background(
                                        Capsule(style: .continuous)
                                            .fill(governanceTint(worstGovernanceState(symbolState)).opacity(0.12))
                                    )
                            }
                            Text("\(symbolState.pluginCount) plugins • \(appliedActionCount(symbolState)) applied actions")
                                .font(.caption)
                                .foregroundStyle(FXAITheme.textSecondary)
                            if let firstReason = symbolState.plugins.first?.reasonCodes.first {
                                Text(firstReason)
                                    .font(.caption)
                                    .foregroundStyle(FXAITheme.textMuted)
                                    .lineLimit(2)
                            }
                        }
                        Spacer(minLength: 12)
                        VStack(alignment: .trailing, spacing: 4) {
                            Text(percentString(maxRisk(symbolState)))
                                .font(.subheadline.weight(.semibold))
                                .foregroundStyle(riskTint(maxRisk(symbolState)))
                            Text("\(challengerEligibleCount(symbolState)) eligible")
                                .font(.caption)
                                .foregroundStyle(FXAITheme.textMuted)
                        }
                    }
                    .padding(12)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .background(
                        RoundedRectangle(cornerRadius: 16, style: .continuous)
                            .fill(model.selectedRuntimeSymbol == symbolState.symbol ? FXAITheme.accentSoft.opacity(0.14) : FXAITheme.panel.opacity(0.55))
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
                    Text(selectedSymbol.symbol)
                        .font(.title3.weight(.semibold))
                        .foregroundStyle(FXAITheme.textPrimary)
                    Spacer()
                    StatusBadge(
                        title: "Worst State",
                        value: worstGovernanceState(selectedSymbol),
                        tint: governanceTint(worstGovernanceState(selectedSymbol))
                    )
                }

                LazyVGrid(columns: [GridItem(.adaptive(minimum: 170), spacing: 12)], spacing: 12) {
                    detailMetric(title: "Max Risk", value: percentString(maxRisk(selectedSymbol)), tint: riskTint(maxRisk(selectedSymbol)))
                    detailMetric(title: "Applied Actions", value: "\(appliedActionCount(selectedSymbol))", tint: FXAITheme.warning)
                    detailMetric(title: "Eligible Challengers", value: "\(challengerEligibleCount(selectedSymbol))", tint: FXAITheme.success)
                    detailMetric(title: "Shadow/Disabled", value: "\(shadowOrDisabledCount(selectedSymbol))", tint: criticalColor)
                }

                LazyVGrid(columns: [GridItem(.adaptive(minimum: 220), spacing: 12)], spacing: 12) {
                    metricListCard(title: "Health Counts", values: selectedSymbol.healthCounts)
                    metricListCard(title: "Governance Counts", values: selectedSymbol.governanceCounts)
                    metricListCard(title: "Action Counts", values: selectedSymbol.actionCounts)
                    metricListCard(title: "Latest Context", values: selectedSymbol.latestContext, emptyMessage: "No context summary was available.")
                }

                pluginRoster(selectedSymbol)
                recentActionsPanel(selectedSymbol)
            } else {
                Text("No drift-governance symbol is selected.")
                    .foregroundStyle(FXAITheme.textSecondary)
            }
        }
    }

    private func pluginRoster(_ symbol: DriftGovernanceSymbolSnapshot) -> some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("Plugin Roster")
                .font(.subheadline.weight(.semibold))
                .foregroundStyle(FXAITheme.textPrimary)

            ForEach(symbol.plugins) { plugin in
                VStack(alignment: .leading, spacing: 10) {
                    HStack(alignment: .top, spacing: 10) {
                        VStack(alignment: .leading, spacing: 4) {
                            HStack(spacing: 8) {
                                Text(plugin.pluginName)
                                    .font(.subheadline.weight(.semibold))
                                    .foregroundStyle(FXAITheme.textPrimary)
                                Text(plugin.familyName)
                                    .font(.caption2.weight(.bold))
                                    .foregroundStyle(FXAITheme.textSecondary)
                                    .padding(.horizontal, 6)
                                    .padding(.vertical, 2)
                                    .background(
                                        Capsule(style: .continuous)
                                            .fill(FXAITheme.panel.opacity(0.65))
                                    )
                            }
                            Text("Base \(plugin.baseRegistryStatus.uppercased()) • rec \(plugin.actionRecommendation) • weight \(String(format: "%.2f", plugin.weightMultiplier))")
                                .font(.caption)
                                .foregroundStyle(FXAITheme.textSecondary)
                        }
                        Spacer(minLength: 12)
                        VStack(alignment: .trailing, spacing: 4) {
                            Text(percentString(plugin.aggregateRiskScore))
                                .font(.subheadline.weight(.semibold))
                                .foregroundStyle(riskTint(plugin.aggregateRiskScore))
                            HStack(spacing: 6) {
                                badge(plugin.healthState, tint: healthTint(plugin.healthState))
                                badge(plugin.governanceState, tint: governanceTint(plugin.governanceState))
                            }
                        }
                    }

                    if !plugin.reasonCodes.isEmpty {
                        VStack(alignment: .leading, spacing: 4) {
                            ForEach(plugin.reasonCodes.prefix(4), id: \.self) { reason in
                                Text(reason)
                                    .font(.caption)
                                    .foregroundStyle(FXAITheme.textMuted)
                            }
                        }
                    }

                    LazyVGrid(columns: [GridItem(.adaptive(minimum: 200), spacing: 10)], spacing: 10) {
                        metricListCard(title: "Drift Scores", values: plugin.driftScores, emptyMessage: "No drift metrics recorded.")
                        metricListCard(title: "Support", values: plugin.support, emptyMessage: "No support metadata recorded.")
                        metricListCard(title: "Quality Flags", values: plugin.qualityFlags, emptyMessage: "No quality flags recorded.")
                        if let challenger = plugin.challengerEvaluation {
                            metricListCard(title: "Challenger Eval", values: challengerRecords(challenger), emptyMessage: "No challenger evaluation.")
                        }
                    }

                    if !plugin.contextSummary.isEmpty {
                        metricListCard(title: "Context Summary", values: plugin.contextSummary, emptyMessage: "No context summary recorded.")
                    }
                }
                .padding(12)
                .background(
                    RoundedRectangle(cornerRadius: 16, style: .continuous)
                        .fill(governanceTint(plugin.governanceState).opacity(0.08))
                )
            }
        }
    }

    private func recentActionsPanel(_ symbol: DriftGovernanceSymbolSnapshot) -> some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("Recent Actions")
                .font(.subheadline.weight(.semibold))
                .foregroundStyle(FXAITheme.textPrimary)

            if symbol.recentActions.isEmpty {
                Text("No governance actions have been recorded for this symbol yet.")
                    .font(.caption)
                    .foregroundStyle(FXAITheme.textMuted)
            } else {
                ForEach(symbol.recentActions.prefix(12)) { action in
                    HStack(alignment: .top, spacing: 10) {
                        Image(systemName: action.actionApplied ? "checkmark.shield.fill" : "eye.fill")
                            .foregroundStyle(action.actionApplied ? FXAITheme.warning : FXAITheme.textMuted)
                            .padding(.top, 2)
                        VStack(alignment: .leading, spacing: 4) {
                            Text("\(action.pluginName) • \(action.actionKind)")
                                .font(.caption.weight(.semibold))
                                .foregroundStyle(FXAITheme.textPrimary)
                            Text("\(action.previousState) → \(action.newState)")
                                .font(.caption)
                                .foregroundStyle(FXAITheme.textSecondary)
                            if let createdAt = action.createdAt {
                                Text(createdAt.formatted(date: .abbreviated, time: .shortened))
                                    .font(.caption2)
                                    .foregroundStyle(FXAITheme.textMuted)
                            }
                        }
                        Spacer()
                        badge(action.actionApplied ? "APPLIED" : "REC", tint: action.actionApplied ? FXAITheme.warning : FXAITheme.textMuted)
                    }
                    .padding(10)
                    .background(
                        RoundedRectangle(cornerRadius: 14, style: .continuous)
                            .fill(FXAITheme.panel.opacity(0.50))
                    )
                }
            }
        }
    }

    private func statusPanel(snapshot: DriftGovernanceSnapshot) -> some View {
        FXAIVisualEffectSurface {
            VStack(alignment: .leading, spacing: 14) {
                Text("Status and Artifacts")
                    .font(.headline)
                    .foregroundStyle(FXAITheme.textPrimary)

                LazyVGrid(columns: [GridItem(.adaptive(minimum: 220), spacing: 12)], spacing: 12) {
                    metricListCard(title: "Health Counts", values: snapshot.healthCounts)
                    metricListCard(title: "Governance Counts", values: snapshot.governanceCounts)
                    metricListCard(title: "Action Counts", values: snapshot.actionCounts)
                    metricListCard(title: "Status", values: snapshot.statusRecords)
                }

                if !snapshot.artifactPaths.isEmpty {
                    metricListCard(title: "Artifact Paths", values: snapshot.artifactPaths)
                }
            }
        }
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
                ForEach(values.prefix(10)) { item in
                    HStack(alignment: .top) {
                        Text(item.key.replacingOccurrences(of: "_", with: " ").capitalized)
                            .font(.caption)
                            .foregroundStyle(FXAITheme.textSecondary)
                        Spacer(minLength: 16)
                        Text(item.value)
                            .font(.caption.monospacedDigit())
                            .foregroundStyle(FXAITheme.textPrimary)
                            .multilineTextAlignment(.trailing)
                    }
                }
            }
        }
        .padding(12)
        .background(
            RoundedRectangle(cornerRadius: 14, style: .continuous)
                .fill(FXAITheme.panel.opacity(0.50))
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
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(12)
        .background(
            RoundedRectangle(cornerRadius: 14, style: .continuous)
                .fill(FXAITheme.panel.opacity(0.50))
        )
    }

    private func challengerRecords(_ challenger: DriftGovernanceChallengerEvaluation) -> [KeyValueRecord] {
        [
            KeyValueRecord(key: "eligibility_state", value: challenger.eligibilityState),
            KeyValueRecord(key: "qualifies", value: boolLabel(challenger.qualifies)),
            KeyValueRecord(key: "support_count", value: "\(challenger.supportCount)"),
            KeyValueRecord(key: "shadow_support", value: "\(challenger.shadowSupport)"),
            KeyValueRecord(key: "walkforward_score", value: percentString(challenger.walkforwardScore / 100.0)),
            KeyValueRecord(key: "recent_score", value: percentString(challenger.recentScore / 100.0)),
            KeyValueRecord(key: "live_shadow_score", value: percentString(challenger.liveShadowScore)),
            KeyValueRecord(key: "live_reliability", value: percentString(challenger.liveReliability)),
            KeyValueRecord(key: "portfolio_score", value: percentString(challenger.portfolioScore)),
            KeyValueRecord(key: "promotion_margin", value: String(format: "%.2f", challenger.promotionMargin)),
        ]
    }

    private func badge(_ text: String, tint: Color) -> some View {
        Text(text)
            .font(.caption2.weight(.bold))
            .foregroundStyle(tint)
            .padding(.horizontal, 6)
            .padding(.vertical, 2)
            .background(
                Capsule(style: .continuous)
                    .fill(tint.opacity(0.12))
            )
    }

    private func worstGovernanceState(_ symbol: DriftGovernanceSymbolSnapshot) -> String {
        symbol.plugins
            .map(\.governanceState)
            .max(by: { governanceSeverity($0) < governanceSeverity($1) }) ?? "HEALTHY"
    }

    private func governanceSeverity(_ state: String) -> Int {
        switch state.uppercased() {
        case "DISABLED": 6
        case "DEMOTED": 5
        case "SHADOW_ONLY": 4
        case "DEGRADED": 3
        case "CAUTION": 2
        case "CHAMPION_CANDIDATE", "CHALLENGER": 1
        default: 0
        }
    }

    private func appliedActionCount(_ symbol: DriftGovernanceSymbolSnapshot) -> Int {
        symbol.recentActions.filter(\.actionApplied).count
    }

    private func challengerEligibleCount(_ symbol: DriftGovernanceSymbolSnapshot) -> Int {
        symbol.plugins.filter { $0.challengerEvaluation?.qualifies == true }.count
    }

    private func shadowOrDisabledCount(_ symbol: DriftGovernanceSymbolSnapshot) -> Int {
        symbol.plugins.filter { $0.shadowOnly || $0.disabled }.count
    }

    private func maxRisk(_ symbol: DriftGovernanceSymbolSnapshot) -> Double {
        symbol.plugins.map(\.aggregateRiskScore).max() ?? 0
    }

    private func healthTint(_ state: String) -> Color {
        switch state.uppercased() {
        case "DISABLED": criticalColor
        case "SHADOW_ONLY": FXAITheme.warning
        case "DEGRADED": FXAITheme.warning
        case "CAUTION": FXAITheme.accentSoft
        default: FXAITheme.success
        }
    }

    private func governanceTint(_ state: String) -> Color {
        switch state.uppercased() {
        case "DISABLED", "DEMOTED": criticalColor
        case "SHADOW_ONLY", "DEGRADED": FXAITheme.warning
        case "CAUTION": FXAITheme.accentSoft
        case "CHAMPION_CANDIDATE": FXAITheme.success
        case "CHALLENGER": FXAITheme.accent
        default: FXAITheme.success
        }
    }

    private func riskTint(_ value: Double) -> Color {
        if value >= 0.75 { return criticalColor }
        if value >= 0.50 { return FXAITheme.warning }
        if value >= 0.30 { return FXAITheme.accentSoft }
        return FXAITheme.success
    }

    private func percentString(_ value: Double) -> String {
        String(format: "%.0f%%", value * 100.0)
    }

    private func boolLabel(_ value: Bool) -> String {
        value ? "Yes" : "No"
    }

    private var criticalColor: Color {
        Color(red: 0.97, green: 0.43, blue: 0.47)
    }
}
