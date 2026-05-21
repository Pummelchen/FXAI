import FXAIGUICore
import SwiftUI

struct PairNetworkView: View {
    @EnvironmentObject private var model: FXAIGUIModel

    private var visibleSymbols: [PairNetworkSymbolSnapshot] {
        guard let snapshot = model.pairNetworkSnapshot else { return [] }
        return Array(snapshot.symbols.prefix(24))
    }

    private var selectedSymbol: PairNetworkSymbolSnapshot? {
        guard !visibleSymbols.isEmpty else { return nil }
        return visibleSymbols.first(where: { $0.symbol == model.selectedPairNetworkSymbol }) ?? visibleSymbols.first
    }

    private var selectedPairSummary: PairNetworkPairSummary? {
        guard let snapshot = model.pairNetworkSnapshot else { return nil }
        guard let symbol = selectedSymbol?.symbol else { return snapshot.pairSummaries.first }
        return snapshot.pairSummaries.first(where: { $0.pair == symbol }) ?? snapshot.pairSummaries.first
    }

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 18) {
                SectionHeader(
                    title: "Pair Network",
                    subtitle: "Inspect the portfolio-level dependency graph that decomposes pair trades into currency and factor exposures, scores redundancy or contradiction, and resolves overlapping expressions before final order approval."
                )

                if let snapshot = model.pairNetworkSnapshot {
                    topStatus(snapshot: snapshot)
                    symbolWorkspace(snapshot: snapshot)
                    graphWorkspace(snapshot: snapshot)
                } else {
                    EmptyStateView(
                        title: "Pair-network artifacts not found",
                        message: "Run pair-network-validate to publish the default config, build the graph with pair-network-build, let MT5 emit per-symbol runtime state, and refresh the GUI.",
                        symbolName: "point.3.connected.trianglepath.dotted"
                    )
                }
            }
        }
    }

    private func topStatus(snapshot: PairNetworkSnapshot) -> some View {
        let blockedCount = snapshot.symbols.filter {
            let decision = $0.decision.uppercased()
            return decision == "BLOCK_CONTRADICTORY" || decision == "BLOCK_CONCENTRATION"
        }.count
        let reducedCount = snapshot.symbols.filter { $0.decision.uppercased() == "ALLOW_REDUCED" }.count

        return LazyVGrid(columns: [GridItem(.adaptive(minimum: 180), spacing: 16)], spacing: 16) {
            MetricCard(
                title: "Graph Mode",
                value: prettify(snapshot.graphMode),
                footnote: "Current dependency build mode, structural only or structural plus empirical augmentation.",
                symbolName: "point.3.connected.trianglepath.dotted",
                tint: snapshot.fallbackGraphUsed ? FXAITheme.warning : FXAITheme.accentSoft
            )
            MetricCard(
                title: "Tracked Symbols",
                value: "\(snapshot.symbols.count)",
                footnote: "Per-symbol runtime pair-network decisions currently visible to the operator GUI.",
                symbolName: "arrow.left.and.right.square.fill",
                tint: FXAITheme.accent
            )
            MetricCard(
                title: "Blocked",
                value: "\(blockedCount)",
                footnote: "Candidates currently rejected because they contradict existing exposure or increase concentration too far.",
                symbolName: "hand.raised.fill",
                tint: criticalColor
            )
            MetricCard(
                title: "Reduced",
                value: "\(reducedCount)",
                footnote: "Candidates still allowed but resized down because redundancy or concentration is already elevated.",
                symbolName: "arrow.down.right.and.arrow.up.left",
                tint: FXAITheme.warning
            )
        }
    }

    private func symbolWorkspace(snapshot: PairNetworkSnapshot) -> some View {
        FXAIVisualEffectSurface {
            VStack(alignment: .leading, spacing: 14) {
                Text("Runtime Decisions")
                    .font(.headline)
                    .foregroundStyle(FXAITheme.textPrimary)

                ViewThatFits(in: .horizontal) {
                    HStack(alignment: .top, spacing: 18) {
                        symbolList
                            .frame(width: 340, alignment: .topLeading)
                        symbolDetail(snapshot: snapshot)
                    }

                    VStack(alignment: .leading, spacing: 18) {
                        symbolList
                        symbolDetail(snapshot: snapshot)
                    }
                }
            }
        }
    }

    private var symbolList: some View {
        VStack(alignment: .leading, spacing: 10) {
            ForEach(visibleSymbols) { symbolState in
                Button {
                    model.selectedPairNetworkSymbol = symbolState.symbol
                } label: {
                    HStack(alignment: .top, spacing: 10) {
                        VStack(alignment: .leading, spacing: 4) {
                            HStack(spacing: 8) {
                                Text(symbolState.symbol)
                                    .font(.subheadline.weight(.semibold))
                                    .foregroundStyle(FXAITheme.textPrimary)
                                Text(shortDecision(symbolState.decision))
                                    .font(.caption2.weight(.bold))
                                    .foregroundStyle(decisionTint(symbolState.decision))
                                    .padding(.horizontal, 6)
                                    .padding(.vertical, 2)
                                    .background(
                                        Capsule(style: .continuous)
                                            .fill(decisionTint(symbolState.decision).opacity(0.12))
                                    )
                            }
                            Text(symbolState.preferredExpression.isEmpty ? "No alternative preference" : "Prefer \(symbolState.preferredExpression)")
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
                            Text(percentString(symbolState.conflictScore))
                                .font(.subheadline.weight(.semibold))
                                .foregroundStyle(decisionTint(symbolState.decision))
                            Text(String(format: "%.2fx", symbolState.recommendedSizeMultiplier))
                                .font(.caption)
                                .foregroundStyle(FXAITheme.textMuted)
                        }
                    }
                    .padding(12)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .background(
                        RoundedRectangle(cornerRadius: 16, style: .continuous)
                            .fill(model.selectedPairNetworkSymbol == symbolState.symbol ? FXAITheme.accentSoft.opacity(0.14) : FXAITheme.panel.opacity(0.55))
                    )
                }
                .buttonStyle(.plain)
            }
        }
    }

    private func symbolDetail(snapshot: PairNetworkSnapshot) -> some View {
        VStack(alignment: .leading, spacing: 16) {
            if let selectedSymbol {
                HStack {
                    Text(selectedSymbol.symbol)
                        .font(.title3.weight(.semibold))
                        .foregroundStyle(FXAITheme.textPrimary)
                    Spacer()
                    StatusBadge(title: "Decision", value: shortDecision(selectedSymbol.decision), tint: decisionTint(selectedSymbol.decision))
                }

                LazyVGrid(columns: [GridItem(.adaptive(minimum: 160), spacing: 12)], spacing: 12) {
                    detailMetric(title: "Conflict", value: percentString(selectedSymbol.conflictScore), tint: decisionTint(selectedSymbol.decision))
                    detailMetric(title: "Redundancy", value: percentString(selectedSymbol.redundancyScore), tint: FXAITheme.warning)
                    detailMetric(title: "Contradiction", value: percentString(selectedSymbol.contradictionScore), tint: criticalColor)
                    detailMetric(title: "Concentration", value: percentString(selectedSymbol.concentrationScore), tint: FXAITheme.warning)
                    detailMetric(title: "Currency Conc.", value: percentString(selectedSymbol.currencyConcentration), tint: FXAITheme.accentSoft)
                    detailMetric(title: "Factor Conc.", value: percentString(selectedSymbol.factorConcentration), tint: FXAITheme.accent)
                    detailMetric(title: "Size Multiplier", value: String(format: "%.2fx", selectedSymbol.recommendedSizeMultiplier), tint: FXAITheme.success)
                    detailMetric(title: "Preferred", value: selectedSymbol.preferredExpression.isEmpty ? "None" : selectedSymbol.preferredExpression, tint: FXAITheme.textSecondary)
                }

                LazyVGrid(columns: [GridItem(.adaptive(minimum: 220), spacing: 12)], spacing: 12) {
                    metricListCard(title: "Currency Exposure", values: selectedSymbol.currencyExposure)
                    metricListCard(title: "Factor Exposure", values: selectedSymbol.factorExposure)
                    metricListCard(
                        title: "Graph Flags",
                        values: [
                            KeyValueRecord(key: "graph_mode", value: snapshot.graphMode),
                            KeyValueRecord(key: "action_mode", value: snapshot.actionMode),
                            KeyValueRecord(key: "fallback_graph_used", value: boolLabel(selectedSymbol.fallbackGraphUsed)),
                            KeyValueRecord(key: "partial_dependency_data", value: boolLabel(selectedSymbol.partialDependencyData)),
                            KeyValueRecord(key: "graph_stale", value: boolLabel(selectedSymbol.graphStale)),
                        ]
                    )
                }

                VStack(alignment: .leading, spacing: 8) {
                    Text("Reason Codes")
                        .font(.subheadline.weight(.semibold))
                        .foregroundStyle(FXAITheme.textPrimary)
                    if selectedSymbol.reasons.isEmpty {
                        Text("No pair-network reason codes were recorded for the latest symbol state.")
                            .font(.caption)
                            .foregroundStyle(FXAITheme.textMuted)
                    } else {
                        ForEach(selectedSymbol.reasons, id: \.self) { reason in
                            HStack(alignment: .top, spacing: 8) {
                                Image(systemName: "circle.fill")
                                    .font(.system(size: 5))
                                    .foregroundStyle(decisionTint(selectedSymbol.decision))
                                    .padding(.top, 6)
                                Text(reason)
                                    .font(.caption)
                                    .foregroundStyle(FXAITheme.textMuted)
                            }
                        }
                    }
                }
            } else {
                metricListCard(title: "Graph Flags", values: snapshot.statusRecords)
            }
        }
    }

    private func graphWorkspace(snapshot: PairNetworkSnapshot) -> some View {
        FXAIVisualEffectSurface {
            VStack(alignment: .leading, spacing: 14) {
                Text("Dependency Graph")
                    .font(.headline)
                    .foregroundStyle(FXAITheme.textPrimary)

                LazyVGrid(columns: [GridItem(.adaptive(minimum: 220), spacing: 12)], spacing: 12) {
                    metricListCard(
                        title: "Status",
                        values: [
                            KeyValueRecord(key: "graph_mode", value: snapshot.graphMode),
                            KeyValueRecord(key: "action_mode", value: snapshot.actionMode),
                            KeyValueRecord(key: "pair_count", value: "\(snapshot.pairCount)"),
                            KeyValueRecord(key: "currency_count", value: "\(snapshot.currencyCount)"),
                            KeyValueRecord(key: "edge_count", value: "\(snapshot.edgeCount)"),
                        ]
                    )
                    metricListCard(title: "Quality Flags", values: snapshot.qualityFlags)
                    metricListCard(title: "Artifact Paths", values: snapshot.artifactPaths)
                }

                if let selectedPairSummary {
                    VStack(alignment: .leading, spacing: 10) {
                        HStack {
                            Text("Selected Pair Dependencies")
                                .font(.subheadline.weight(.semibold))
                                .foregroundStyle(FXAITheme.textPrimary)
                            Spacer()
                            Text(selectedPairSummary.pair)
                                .font(.caption.weight(.semibold))
                                .foregroundStyle(FXAITheme.textSecondary)
                        }

                        metricListCard(title: "Factor Signature", values: selectedPairSummary.factorSignature)

                        ForEach(Array(selectedPairSummary.topDependencies.prefix(6))) { edge in
                            dependencyRow(edge: edge)
                        }
                    }
                }

                if !snapshot.topEdges.isEmpty {
                    VStack(alignment: .leading, spacing: 10) {
                        Text("Top Overlap Edges")
                            .font(.subheadline.weight(.semibold))
                            .foregroundStyle(FXAITheme.textPrimary)

                        ForEach(Array(snapshot.topEdges.prefix(8))) { edge in
                            dependencyRow(edge: edge)
                        }
                    }
                }

                if !snapshot.reasons.isEmpty {
                    VStack(alignment: .leading, spacing: 8) {
                        Text("Report Reasons")
                            .font(.subheadline.weight(.semibold))
                            .foregroundStyle(FXAITheme.textPrimary)
                        ForEach(snapshot.reasons, id: \.self) { reason in
                            Text(reason)
                                .font(.caption)
                                .foregroundStyle(FXAITheme.textMuted)
                        }
                    }
                }
            }
        }
    }

    private func dependencyRow(edge: PairNetworkDependencyEdge) -> some View {
        HStack(alignment: .top, spacing: 12) {
            VStack(alignment: .leading, spacing: 4) {
                Text(edge.sourcePair.isEmpty ? edge.targetPair : "\(edge.sourcePair) -> \(edge.targetPair)")
                    .font(.subheadline.weight(.semibold))
                    .foregroundStyle(FXAITheme.textPrimary)
                Text(edge.relation.replacingOccurrences(of: "_", with: " ").capitalized)
                    .font(.caption)
                    .foregroundStyle(FXAITheme.textSecondary)
                if !edge.sharedCurrencies.isEmpty {
                    Text("Shared: \(edge.sharedCurrencies.joined(separator: ", "))")
                        .font(.caption)
                        .foregroundStyle(FXAITheme.textMuted)
                }
            }
            Spacer()
            VStack(alignment: .trailing, spacing: 4) {
                Text(percentString(edge.combinedScore))
                    .font(.subheadline.weight(.semibold))
                    .foregroundStyle(FXAITheme.accentSoft)
                Text("S \(percentString(edge.structuralScore)) • E \(percentString(edge.empiricalScore))")
                    .font(.caption)
                    .foregroundStyle(FXAITheme.textMuted)
                if edge.support > 0 {
                    Text("n=\(edge.support)")
                        .font(.caption2)
                        .foregroundStyle(FXAITheme.textMuted)
                }
            }
        }
        .padding(12)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(
            RoundedRectangle(cornerRadius: 14, style: .continuous)
                .fill(FXAITheme.panel.opacity(0.58))
        )
    }

    private func detailMetric(title: String, value: String, tint: Color) -> some View {
        VStack(alignment: .leading, spacing: 5) {
            Text(title)
                .font(.caption.weight(.semibold))
                .foregroundStyle(FXAITheme.textSecondary)
            Text(value)
                .font(.headline.weight(.semibold))
                .foregroundStyle(tint)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(12)
        .background(
            RoundedRectangle(cornerRadius: 16, style: .continuous)
                .fill(tint.opacity(0.08))
        )
    }

    private func metricListCard(title: String, values: [KeyValueRecord]) -> some View {
        VStack(alignment: .leading, spacing: 10) {
            Text(title)
                .font(.subheadline.weight(.semibold))
                .foregroundStyle(FXAITheme.textPrimary)

            if values.isEmpty {
                Text("No values available.")
                    .font(.caption)
                    .foregroundStyle(FXAITheme.textMuted)
            } else {
                ForEach(values) { record in
                    HStack {
                        Text(record.key.replacingOccurrences(of: "_", with: " ").capitalized)
                            .font(.caption)
                            .foregroundStyle(FXAITheme.textSecondary)
                        Spacer()
                        Text(record.value)
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
                .fill(FXAITheme.panel.opacity(0.58))
        )
    }

    private func shortDecision(_ value: String) -> String {
        switch value.uppercased() {
        case "ALLOW_REDUCED": "REDUCE"
        case "SUPPRESS_REDUNDANT": "SUPPRESS"
        case "BLOCK_CONTRADICTORY": "BLOCK"
        case "BLOCK_CONCENTRATION": "BLOCK"
        case "PREFER_ALTERNATIVE_EXPRESSION": "PREFER"
        default: "ALLOW"
        }
    }

    private func decisionTint(_ value: String) -> Color {
        switch value.uppercased() {
        case "BLOCK_CONTRADICTORY", "BLOCK_CONCENTRATION":
            criticalColor
        case "SUPPRESS_REDUNDANT", "PREFER_ALTERNATIVE_EXPRESSION":
            FXAITheme.warning
        case "ALLOW_REDUCED":
            FXAITheme.accent
        default:
            FXAITheme.success
        }
    }

    private func prettify(_ raw: String) -> String {
        raw.replacingOccurrences(of: "_", with: " ").capitalized
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
