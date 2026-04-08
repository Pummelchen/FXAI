import FXAIGUICore
import SwiftUI

struct NewsPulseView: View {
    @EnvironmentObject private var model: FXAIGUIModel

    private var activePairs: Set<String> {
        Set((model.runtimeSnapshot?.deployments ?? []).compactMap { pairID(for: $0.symbol) })
    }

    private var visiblePairs: [NewsPulsePairState] {
        guard let snapshot = model.newsPulseSnapshot else { return [] }
        let featured = snapshot.pairs.filter { activePairs.contains($0.pair) }
        let featuredIDs = Set(featured.map(\.pair))
        let supplemental = snapshot.pairs.filter { !featuredIDs.contains($0.pair) }.prefix(18)
        return featured + supplemental
    }

    private var upcomingCurrencies: [NewsPulseCurrencyState] {
        guard let snapshot = model.newsPulseSnapshot else { return [] }
        return snapshot.currencies
            .filter { $0.nextHighImpactETAMin != nil || $0.inPostEventWindow }
            .sorted {
                let lhsETA = $0.nextHighImpactETAMin ?? Int.max
                let rhsETA = $1.nextHighImpactETAMin ?? Int.max
                if lhsETA == rhsETA {
                    return $0.currency < $1.currency
                }
                return lhsETA < rhsETA
            }
            .prefix(8)
            .map { $0 }
    }

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 18) {
                SectionHeader(
                    title: "NewsPulse",
                    subtitle: "Inspect scheduled macro risk, breaking-news burst state, current pair gates, and source freshness from the shared NewsPulse subsystem."
                )

                if let snapshot = model.newsPulseSnapshot {
                    statusRow(snapshot: snapshot)
                    currencyHeatmap(snapshot: snapshot)
                    upcomingEvents(snapshot: snapshot)
                    pairRiskPanel(snapshot: snapshot)
                    recentTape(snapshot: snapshot)
                    artifactPanel(snapshot: snapshot)
                } else {
                    EmptyStateView(
                        title: "NewsPulse is not running yet",
                        message: "Install the MT5 calendar service, start the NewsPulse daemon, and refresh the GUI so the operator shell can display merged news state.",
                        symbolName: "dot.radiowaves.left.and.right"
                    )
                }
            }
        }
    }

    private func statusRow(snapshot: NewsPulseSnapshot) -> some View {
        let generatedAtText = snapshot.generatedAt.map { FXAIFormatting.relativeDateString(for: $0) } ?? "unknown"

        return LazyVGrid(
            columns: [GridItem(.adaptive(minimum: 180), spacing: 16, alignment: .top)],
            spacing: 16
        ) {
            ForEach(snapshot.sourceStatuses) { status in
                MetricCard(
                    title: status.id.uppercased(),
                    value: status.stale ? "Stale" : (status.ok ? "Ready" : "Error"),
                    footnote: status.lastError?.isEmpty == false
                        ? status.lastError!
                        : status.lastSuccessAt.map { "Last success \($0.formatted(date: .omitted, time: .shortened))" } ?? "Waiting for data",
                    symbolName: statusIcon(for: status),
                    tint: statusColor(for: status)
                )
            }

            MetricCard(
                title: "Snapshot",
                value: generatedAtText,
                footnote: "Merged NewsPulse state visible to runtime gates and the operator shell.",
                symbolName: "clock.arrow.trianglehead.counterclockwise.rotate.90",
                tint: snapshot.hasBlockingIssue ? FXAITheme.warning : FXAITheme.success
            )

            MetricCard(
                title: "Queries",
                value: "\(snapshot.queryCount)",
                footnote: "GDELT query sets used in the latest fusion cycle.",
                symbolName: "magnifyingglass.circle.fill",
                tint: FXAITheme.accentSoft
            )
        }
    }

    private func currencyHeatmap(snapshot: NewsPulseSnapshot) -> some View {
        FXAIVisualEffectSurface {
            VStack(alignment: .leading, spacing: 14) {
                Text("Currency Heatmap")
                    .font(.headline)
                    .foregroundStyle(FXAITheme.textPrimary)

                LazyVGrid(
                    columns: [GridItem(.adaptive(minimum: 170), spacing: 14, alignment: .top)],
                    spacing: 14
                ) {
                    ForEach(snapshot.currencies) { currency in
                        VStack(alignment: .leading, spacing: 10) {
                            HStack {
                                Text(currency.currency)
                                    .font(.headline)
                                    .foregroundStyle(FXAITheme.textPrimary)
                                Spacer()
                                StatusBadge(
                                    title: "Gate",
                                    value: gateLabel(for: currency),
                                    tint: gateColor(risk: currency.riskScore, stale: currency.stale)
                                )
                            }

                            VStack(alignment: .leading, spacing: 6) {
                                metricRow(label: "Risk", value: percentString(currency.riskScore))
                                metricRow(label: "Burst", value: String(format: "%.2f", currency.burstScore15m))
                                metricRow(label: "Intensity", value: String(format: "%.2f", currency.intensity15m))
                                metricRow(label: "Tone", value: String(format: "%.2f", currency.toneMean15m))
                            }

                            if let eta = currency.nextHighImpactETAMin {
                                Text("High-impact event in \(eta)m")
                                    .font(.caption.weight(.semibold))
                                    .foregroundStyle(FXAITheme.warning)
                            } else if let since = currency.timeSinceLastHighImpactMin, currency.inPostEventWindow {
                                Text("High-impact print \(since)m ago")
                                    .font(.caption.weight(.semibold))
                                    .foregroundStyle(FXAITheme.warning)
                            }

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
                                .fill(gateColor(risk: currency.riskScore, stale: currency.stale).opacity(0.08))
                        )
                    }
                }
            }
        }
    }

    private func upcomingEvents(snapshot: NewsPulseSnapshot) -> some View {
        FXAIVisualEffectSurface {
            VStack(alignment: .leading, spacing: 12) {
                Text("Upcoming High-Impact Windows")
                    .font(.headline)
                    .foregroundStyle(FXAITheme.textPrimary)

                if upcomingCurrencies.isEmpty {
                    Text("No current pre-event or immediate post-event windows are active.")
                        .foregroundStyle(FXAITheme.textSecondary)
                } else {
                    ForEach(upcomingCurrencies) { currency in
                        HStack(alignment: .top, spacing: 12) {
                            Text(currency.currency)
                                .font(.subheadline.weight(.semibold))
                                .foregroundStyle(FXAITheme.textPrimary)
                                .frame(width: 52, alignment: .leading)

                            VStack(alignment: .leading, spacing: 4) {
                                if let eta = currency.nextHighImpactETAMin {
                                    Text("Next high-impact event in \(eta) minutes")
                                        .foregroundStyle(FXAITheme.textPrimary)
                                } else if let since = currency.timeSinceLastHighImpactMin {
                                    Text("High-impact event printed \(since) minutes ago")
                                        .foregroundStyle(FXAITheme.textPrimary)
                                }
                                if let reason = currency.reasons.first {
                                    Text(reason)
                                        .font(.caption)
                                        .foregroundStyle(FXAITheme.textMuted)
                                }
                            }

                            Spacer()
                            Text(percentString(currency.riskScore))
                                .font(.subheadline.weight(.semibold))
                                .foregroundStyle(gateColor(risk: currency.riskScore, stale: currency.stale))
                        }
                        .padding(.vertical, 4)
                    }
                }
            }
        }
    }

    private func pairRiskPanel(snapshot: NewsPulseSnapshot) -> some View {
        FXAIVisualEffectSurface {
            VStack(alignment: .leading, spacing: 12) {
                Text("Pair Risk")
                    .font(.headline)
                    .foregroundStyle(FXAITheme.textPrimary)

                if visiblePairs.isEmpty {
                    Text("No pair-level NewsPulse state is available yet.")
                        .foregroundStyle(FXAITheme.textSecondary)
                } else {
                    ForEach(visiblePairs) { pair in
                        HStack(alignment: .top, spacing: 12) {
                            VStack(alignment: .leading, spacing: 4) {
                                HStack(spacing: 8) {
                                    Text(pair.pair)
                                        .font(.subheadline.weight(.semibold))
                                        .foregroundStyle(FXAITheme.textPrimary)
                                    if activePairs.contains(pair.pair) {
                                        Text("LIVE")
                                            .font(.caption2.weight(.bold))
                                            .foregroundStyle(FXAITheme.accent)
                                            .padding(.horizontal, 6)
                                            .padding(.vertical, 2)
                                            .background(
                                                Capsule(style: .continuous)
                                                    .fill(FXAITheme.accent.opacity(0.14))
                                            )
                                    }
                                }
                                Text(pair.reasons.first ?? "No active news risk reason.")
                                    .font(.caption)
                                    .foregroundStyle(FXAITheme.textMuted)
                                    .lineLimit(2)
                            }

                            Spacer()

                            VStack(alignment: .trailing, spacing: 4) {
                                Text(pair.tradeGate)
                                    .font(.caption.weight(.bold))
                                    .foregroundStyle(gateColor(gate: pair.tradeGate, stale: pair.stale))
                                Text(percentString(pair.newsRiskScore))
                                    .font(.subheadline.weight(.semibold))
                                    .foregroundStyle(FXAITheme.textPrimary)
                                if let eta = pair.eventETAMin {
                                    Text("ETA \(eta)m")
                                        .font(.caption)
                                        .foregroundStyle(FXAITheme.textMuted)
                                }
                            }
                        }
                        .padding(.vertical, 6)
                    }
                }
            }
        }
    }

    private func recentTape(snapshot: NewsPulseSnapshot) -> some View {
        FXAIVisualEffectSurface {
            VStack(alignment: .leading, spacing: 12) {
                Text("Recent Tape")
                    .font(.headline)
                    .foregroundStyle(FXAITheme.textPrimary)

                if snapshot.recentItems.isEmpty {
                    Text("No NewsPulse history items are available yet.")
                        .foregroundStyle(FXAITheme.textSecondary)
                } else {
                    ForEach(snapshot.recentItems.prefix(16)) { item in
                        HStack(alignment: .top, spacing: 12) {
                            VStack(alignment: .leading, spacing: 4) {
                                Text(item.title)
                                    .font(.subheadline.weight(.semibold))
                                    .foregroundStyle(FXAITheme.textPrimary)
                                Text("\(item.source.uppercased()) • \(item.domain)")
                                    .font(.caption)
                                    .foregroundStyle(FXAITheme.textMuted)
                                if !item.currencyTags.isEmpty {
                                    Text(item.currencyTags.joined(separator: ", "))
                                        .font(.caption2)
                                        .foregroundStyle(FXAITheme.accentSoft)
                                }
                            }
                            Spacer()
                            VStack(alignment: .trailing, spacing: 4) {
                                if let publishedAt = item.publishedAt {
                                    Text(FXAIFormatting.relativeDateString(for: publishedAt))
                                        .font(.caption)
                                        .foregroundStyle(FXAITheme.textMuted)
                                }
                                Text(String(format: "Tone %.2f", item.tone))
                                    .font(.caption2.weight(.semibold))
                                    .foregroundStyle(item.tone >= 0 ? FXAITheme.success : FXAITheme.warning)
                            }
                        }
                        .padding(.vertical, 4)
                    }
                }
            }
        }
    }

    private func artifactPanel(snapshot: NewsPulseSnapshot) -> some View {
        FXAIVisualEffectSurface {
            VStack(alignment: .leading, spacing: 10) {
                Text("Artifacts")
                    .font(.headline)
                    .foregroundStyle(FXAITheme.textPrimary)
                ForEach(snapshot.artifactPaths) { artifact in
                    HStack(alignment: .top, spacing: 12) {
                        Text(artifact.key)
                            .font(.caption.weight(.semibold))
                            .foregroundStyle(FXAITheme.textSecondary)
                            .frame(width: 110, alignment: .leading)
                        Text(artifact.value)
                            .font(.caption)
                            .foregroundStyle(FXAITheme.textMuted)
                            .textSelection(.enabled)
                        Spacer()
                    }
                }
            }
        }
    }

    private func metricRow(label: String, value: String) -> some View {
        HStack {
            Text(label)
                .font(.caption)
                .foregroundStyle(FXAITheme.textMuted)
            Spacer()
            Text(value)
                .font(.caption.weight(.semibold))
                .foregroundStyle(FXAITheme.textPrimary)
        }
    }

    private func gateLabel(for currency: NewsPulseCurrencyState) -> String {
        if currency.stale {
            return "BLOCK"
        }
        if currency.riskScore >= 0.78 {
            return "BLOCK"
        }
        if currency.riskScore >= 0.45 {
            return "CAUTION"
        }
        return "ALLOW"
    }

    private func gateColor(risk: Double, stale: Bool) -> Color {
        if stale {
            return FXAITheme.warning
        }
        if risk >= 0.78 {
            return FXAITheme.warning
        }
        if risk >= 0.45 {
            return FXAITheme.accentSoft
        }
        return FXAITheme.success
    }

    private func gateColor(gate: String, stale: Bool) -> Color {
        if stale {
            return FXAITheme.warning
        }
        switch gate.uppercased() {
        case "BLOCK":
            return FXAITheme.warning
        case "CAUTION":
            return FXAITheme.accentSoft
        default:
            return FXAITheme.success
        }
    }

    private func statusColor(for status: NewsPulseSourceStatus) -> Color {
        if status.stale || !status.ok {
            return FXAITheme.warning
        }
        return FXAITheme.success
    }

    private func statusIcon(for status: NewsPulseSourceStatus) -> String {
        if status.stale || !status.ok {
            return "exclamationmark.triangle.fill"
        }
        return "checkmark.seal.fill"
    }

    private func percentString(_ value: Double) -> String {
        String(format: "%.0f%%", max(0, min(value, 1)) * 100.0)
    }

    private func pairID(for symbol: String) -> String? {
        let clean = symbol.uppercased()
        guard clean.count >= 6 else { return nil }
        let start = clean.startIndex
        let end = clean.index(start, offsetBy: 6)
        return String(clean[start..<end])
    }
}
