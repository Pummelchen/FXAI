import FXAIGUICore
import SwiftUI

struct NewsPulseView: View {
    @EnvironmentObject private var model: FXAIGUIModel
    @State private var selectedPairID = ""

    private var runtimeSymbols: Set<String> {
        Set((model.runtimeSnapshot?.deployments ?? []).map { $0.symbol.uppercased() })
    }

    private var activePairs: Set<String> {
        guard let snapshot = model.newsPulseSnapshot else {
            return Set((model.runtimeSnapshot?.deployments ?? []).compactMap {
                FXSymbolPairResolver.pairID(from: $0.symbol)
            })
        }
        var out: Set<String> = []
        let preferredPairs = snapshot.pairs.map(\.pair)
        for pair in snapshot.pairs {
            if runtimeSymbols.contains(pair.pair.uppercased()) {
                out.insert(pair.pair)
                continue
            }
            let brokerSymbols = Set(pair.brokerSymbols.map { $0.uppercased() })
            if !runtimeSymbols.isDisjoint(with: brokerSymbols) {
                out.insert(pair.pair)
                continue
            }
            if (model.runtimeSnapshot?.deployments.contains {
                FXSymbolPairResolver.pairID(from: $0.symbol, preferredPairs: preferredPairs) == pair.pair
            } ?? false) {
                out.insert(pair.pair)
            }
        }
        return out
    }

    private var visiblePairs: [NewsPulsePairState] {
        guard let snapshot = model.newsPulseSnapshot else { return [] }
        let featured = snapshot.pairs.filter { activePairs.contains($0.pair) }
        let featuredIDs = Set(featured.map(\.pair))
        let supplemental = snapshot.pairs.filter { !featuredIDs.contains($0.pair) }.prefix(18)
        return featured + supplemental
    }

    private var selectedPair: NewsPulsePairState? {
        guard !visiblePairs.isEmpty else { return nil }
        return visiblePairs.first(where: { $0.pair == selectedPairID }) ?? visiblePairs.first
    }

    private var selectedStories: [NewsPulseStory] {
        guard let snapshot = model.newsPulseSnapshot, let selectedPair else { return [] }
        let storyIDs = Set(selectedPair.storyIDs)
        return snapshot.stories.filter { storyIDs.contains($0.id) }
    }

    private var filteredRecentItems: [NewsPulseRecentItem] {
        guard let snapshot = model.newsPulseSnapshot else { return [] }
        guard let selectedPair else { return Array(snapshot.recentItems.prefix(16)) }
        let currencies = Set([selectedPair.baseCurrency, selectedPair.quoteCurrency])
        let storyIDs = Set(selectedPair.storyIDs)
        let filtered = snapshot.recentItems.filter { item in
            !currencies.isDisjoint(with: Set(item.currencyTags)) || (item.storyID.map { storyIDs.contains($0) } ?? false)
        }
        return Array((filtered.isEmpty ? snapshot.recentItems : filtered).prefix(16))
    }

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 18) {
                SectionHeader(
                    title: "NewsPulse",
                    subtitle: "Inspect scheduled macro risk, breaking-news bursts, official-feed stories, pair-level gates, and the daemon health that drives the shared FXAI news context."
                )

                if let snapshot = model.newsPulseSnapshot {
                    topStatus(snapshot: snapshot)
                    currencyHeatmap(snapshot: snapshot)
                    pairWorkspace(snapshot: snapshot)
                    storyPanel(snapshot: snapshot)
                    recentTape(snapshot: snapshot)
                    sourceHealthPanel(snapshot: snapshot)
                    policyPanel(snapshot: snapshot)
                    artifactPanel(snapshot: snapshot)
                } else {
                    EmptyStateView(
                        title: "NewsPulse is not running yet",
                        message: "Install the MT5 calendar service, start the NewsPulse daemon, and refresh the GUI so the operator shell can display merged news state.",
                        symbolName: "dot.radiowaves.left.and.right"
                    )
                }
            }
            .onAppear(perform: syncSelection)
            .onChange(of: model.newsPulseSnapshot?.pairs.map(\.pair) ?? []) { _, _ in
                syncSelection()
            }
        }
    }

    private func topStatus(snapshot: NewsPulseSnapshot) -> some View {
        let generatedAtText = snapshot.generatedAt.map { FXAIFormatting.relativeDateString(for: $0) } ?? "unknown"

        return LazyVGrid(
            columns: [GridItem(.adaptive(minimum: 180), spacing: 16, alignment: .top)],
            spacing: 16
        ) {
            ForEach(snapshot.sourceStatuses) { status in
                MetricCard(
                    title: status.id.uppercased(),
                    value: status.stale ? "Stale" : (status.ok ? "Ready" : "Error"),
                    footnote: sourceFootnote(for: status),
                    symbolName: statusIcon(for: status),
                    tint: statusColor(for: status)
                )
            }

            MetricCard(
                title: "Snapshot",
                value: generatedAtText,
                footnote: "Merged state visible to runtime gates, replay artifacts, and the operator shell.",
                symbolName: "clock.arrow.trianglehead.counterclockwise.rotate.90",
                tint: snapshot.hasBlockingIssue ? FXAITheme.warning : FXAITheme.success
            )

            MetricCard(
                title: "Stories",
                value: "\(snapshot.stories.count)",
                footnote: "Evolving official and GDELT-linked stories currently tracked by NewsPulse.",
                symbolName: "sparkles.rectangle.stack.fill",
                tint: FXAITheme.accentSoft
            )

            MetricCard(
                title: "Queries",
                value: "\(snapshot.queryCount) + \(snapshot.officialQueryCount)",
                footnote: "GDELT plus official-feed query sets used in the latest fusion cycle.",
                symbolName: "magnifyingglass.circle.fill",
                tint: FXAITheme.accentSoft
            )

            if let daemon = snapshot.daemon {
                MetricCard(
                    title: "Daemon",
                    value: daemon.degraded ? "Degraded" : "Healthy",
                    footnote: daemon.lastError?.isEmpty == false
                        ? daemon.lastError!
                        : "Cycles \(daemon.cyclesCompleted) • interval \(daemon.intervalSeconds)s",
                    symbolName: daemon.degraded ? "bolt.trianglebadge.exclamationmark.fill" : "bolt.heart.fill",
                    tint: daemon.degraded ? FXAITheme.warning : FXAITheme.success
                )
            }
        }
    }

    private func currencyHeatmap(snapshot: NewsPulseSnapshot) -> some View {
        FXAIVisualEffectSurface {
            VStack(alignment: .leading, spacing: 14) {
                Text("Currency Heatmap")
                    .font(.headline)
                    .foregroundStyle(FXAITheme.textPrimary)

                LazyVGrid(
                    columns: [GridItem(.adaptive(minimum: 180), spacing: 14, alignment: .top)],
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
                                metricRow(label: "Stories", value: "\(currency.storyCount15m) • \(String(format: "%.2f", currency.storySeverity15m))")
                                metricRow(label: "Official", value: "\(currency.officialCount24h) / 24h")
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

    private func pairWorkspace(snapshot: NewsPulseSnapshot) -> some View {
        FXAIVisualEffectSurface {
            VStack(alignment: .leading, spacing: 14) {
                Text("Pair Gates")
                    .font(.headline)
                    .foregroundStyle(FXAITheme.textPrimary)

                if visiblePairs.isEmpty {
                    Text("No pair-level NewsPulse state is available yet.")
                        .foregroundStyle(FXAITheme.textSecondary)
                } else {
                    ViewThatFits(in: .horizontal) {
                        HStack(alignment: .top, spacing: 18) {
                            pairList
                                .frame(width: 320, alignment: .topLeading)
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
    }

    private var pairList: some View {
        VStack(alignment: .leading, spacing: 10) {
            ForEach(visiblePairs) { pair in
                Button {
                    selectedPairID = pair.pair
                } label: {
                    HStack(alignment: .top, spacing: 10) {
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
                        }
                    }
                    .padding(12)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .background(
                        RoundedRectangle(cornerRadius: 16, style: .continuous)
                            .fill(selectedPair?.pair == pair.pair ? FXAITheme.accent.opacity(0.12) : FXAITheme.panel.opacity(0.45))
                    )
                }
                .buttonStyle(.plain)
            }
        }
    }

    private func pairDetail(snapshot: NewsPulseSnapshot) -> some View {
        VStack(alignment: .leading, spacing: 14) {
            if let pair = selectedPair {
                VStack(alignment: .leading, spacing: 12) {
                    HStack {
                        VStack(alignment: .leading, spacing: 4) {
                            Text(pair.pair)
                                .font(.title3.weight(.bold))
                                .foregroundStyle(FXAITheme.textPrimary)
                            Text("\(pair.baseCurrency) / \(pair.quoteCurrency)")
                                .font(.caption)
                                .foregroundStyle(FXAITheme.textMuted)
                        }
                        Spacer()
                        StatusBadge(
                            title: "Gate",
                            value: pair.tradeGate,
                            tint: gateColor(gate: pair.tradeGate, stale: pair.stale)
                        )
                    }

                    LazyVGrid(columns: [GridItem(.adaptive(minimum: 140), spacing: 12)], spacing: 12) {
                        infoChip(title: "Risk", value: percentString(pair.newsRiskScore))
                        infoChip(title: "Pressure", value: String(format: "%.2f", pair.newsPressure))
                        infoChip(title: "ETA", value: pair.eventETAMin.map { "\($0)m" } ?? "n/a")
                        infoChip(title: "Changed", value: pair.gateChangedAt.map { FXAIFormatting.relativeDateString(for: $0) } ?? "n/a")
                        infoChip(title: "Session", value: pair.sessionProfile.capitalized)
                        infoChip(title: "Profile", value: pair.calibrationProfile)
                        infoChip(title: "Lot Scale", value: pair.cautionLotScale.map { String(format: "%.2f", $0) } ?? "default")
                        infoChip(title: "Prob Buffer", value: pair.cautionEnterProbBuffer.map { String(format: "%.2f", $0) } ?? "default")
                    }

                    if !pair.watchlistTags.isEmpty || !pair.brokerSymbols.isEmpty {
                        VStack(alignment: .leading, spacing: 6) {
                            if !pair.watchlistTags.isEmpty {
                                Text("Watchlists: \(pair.watchlistTags.joined(separator: ", "))")
                                    .font(.caption)
                                    .foregroundStyle(FXAITheme.textMuted)
                            }
                            if !pair.brokerSymbols.isEmpty {
                                Text("Broker symbols: \(pair.brokerSymbols.joined(separator: ", "))")
                                    .font(.caption)
                                    .foregroundStyle(FXAITheme.textMuted)
                            }
                        }
                    }

                    VStack(alignment: .leading, spacing: 8) {
                        Text("Why \(pair.tradeGate)")
                            .font(.subheadline.weight(.semibold))
                            .foregroundStyle(FXAITheme.textPrimary)
                        ForEach(pair.reasons, id: \.self) { reason in
                            HStack(alignment: .top, spacing: 8) {
                                Image(systemName: "circle.fill")
                                    .font(.system(size: 5))
                                    .foregroundStyle(gateColor(gate: pair.tradeGate, stale: pair.stale))
                                    .padding(.top, 6)
                                Text(reason)
                                    .font(.caption)
                                    .foregroundStyle(FXAITheme.textMuted)
                            }
                        }
                    }

                    timelinePanel(for: pair, snapshot: snapshot)
                }
            } else {
                Text("Select a pair to inspect why NewsPulse is gating it and how the state evolved.")
                    .foregroundStyle(FXAITheme.textSecondary)
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    private func timelinePanel(for pair: NewsPulsePairState, snapshot: NewsPulseSnapshot) -> some View {
        let timeline = snapshot.pairTimelines[pair.pair] ?? []
        return VStack(alignment: .leading, spacing: 8) {
            Text("Gate Timeline")
                .font(.subheadline.weight(.semibold))
                .foregroundStyle(FXAITheme.textPrimary)

            if timeline.isEmpty {
                Text("No timeline history is available yet for this pair.")
                    .font(.caption)
                    .foregroundStyle(FXAITheme.textSecondary)
            } else {
                ForEach(timeline.reversed()) { point in
                    HStack(alignment: .top, spacing: 10) {
                        VStack(alignment: .leading, spacing: 2) {
                            Text(point.tradeGate)
                                .font(.caption.weight(.bold))
                                .foregroundStyle(gateColor(gate: point.tradeGate, stale: point.stale))
                            Text(point.observedAt.map { FXAIFormatting.relativeDateString(for: $0) } ?? "unknown")
                                .font(.caption2)
                                .foregroundStyle(FXAITheme.textMuted)
                        }
                        Spacer()
                        VStack(alignment: .trailing, spacing: 2) {
                            Text(percentString(point.newsRiskScore))
                                .font(.caption.weight(.semibold))
                                .foregroundStyle(FXAITheme.textPrimary)
                            Text(point.eventETAMin.map { "ETA \($0)m" } ?? point.sessionProfile.capitalized)
                                .font(.caption2)
                                .foregroundStyle(FXAITheme.textMuted)
                        }
                    }
                    .padding(.vertical, 4)
                }
            }
        }
    }

    private func storyPanel(snapshot: NewsPulseSnapshot) -> some View {
        FXAIVisualEffectSurface {
            VStack(alignment: .leading, spacing: 12) {
                Text("Evolving Stories")
                    .font(.headline)
                    .foregroundStyle(FXAITheme.textPrimary)

                let stories = selectedStories.isEmpty ? Array(snapshot.stories.prefix(10)) : selectedStories
                if stories.isEmpty {
                    Text("No evolving stories are active right now.")
                        .foregroundStyle(FXAITheme.textSecondary)
                } else {
                    ForEach(stories) { story in
                        HStack(alignment: .top, spacing: 12) {
                            VStack(alignment: .leading, spacing: 4) {
                                Text(story.latestTitle)
                                    .font(.subheadline.weight(.semibold))
                                    .foregroundStyle(FXAITheme.textPrimary)
                                Text("\(story.currencyTags.joined(separator: ", ")) • \(story.domains.joined(separator: ", "))")
                                    .font(.caption)
                                    .foregroundStyle(FXAITheme.textMuted)
                            }
                            Spacer()
                            VStack(alignment: .trailing, spacing: 4) {
                                Text(percentString(story.severityScore))
                                    .font(.caption.weight(.semibold))
                                    .foregroundStyle(story.active ? FXAITheme.warning : FXAITheme.textMuted)
                                Text("\(story.itemCount) items")
                                    .font(.caption2)
                                    .foregroundStyle(FXAITheme.textMuted)
                            }
                        }
                        .padding(.vertical, 4)
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

                if filteredRecentItems.isEmpty {
                    Text("No NewsPulse history items are available yet.")
                        .foregroundStyle(FXAITheme.textSecondary)
                } else {
                    ForEach(filteredRecentItems) { item in
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
                                if let storyID = item.storyID {
                                    Text(storyID)
                                        .font(.caption2.monospaced())
                                        .foregroundStyle(FXAITheme.textMuted)
                                } else {
                                    Text(String(format: "Tone %.2f", item.tone))
                                        .font(.caption2.weight(.semibold))
                                        .foregroundStyle(item.tone >= 0 ? FXAITheme.success : FXAITheme.warning)
                                }
                            }
                        }
                        .padding(.vertical, 4)
                    }
                }
            }
        }
    }

    private func sourceHealthPanel(snapshot: NewsPulseSnapshot) -> some View {
        FXAIVisualEffectSurface {
            VStack(alignment: .leading, spacing: 12) {
                Text("Source Health")
                    .font(.headline)
                    .foregroundStyle(FXAITheme.textPrimary)

                if let daemon = snapshot.daemon {
                    VStack(alignment: .leading, spacing: 6) {
                        Text("Daemon \(daemon.degraded ? "degraded" : "healthy")")
                            .font(.subheadline.weight(.semibold))
                            .foregroundStyle(daemon.degraded ? FXAITheme.warning : FXAITheme.textPrimary)
                        Text("Cycles \(daemon.cyclesCompleted) • interval \(daemon.intervalSeconds)s • failures \(daemon.consecutiveFailures)")
                            .font(.caption)
                            .foregroundStyle(FXAITheme.textMuted)
                        if !daemon.degradedReasons.isEmpty {
                            Text(daemon.degradedReasons.joined(separator: ", "))
                                .font(.caption)
                                .foregroundStyle(FXAITheme.warning)
                        }
                    }
                }

                ForEach(snapshot.sourceHealthTimeline.reversed().prefix(8)) { point in
                    HStack(spacing: 12) {
                        Text(point.observedAt.map { FXAIFormatting.relativeDateString(for: $0) } ?? "unknown")
                            .font(.caption)
                            .foregroundStyle(FXAITheme.textMuted)
                            .frame(width: 130, alignment: .leading)
                        sourceDot(title: "CAL", ok: point.calendarOK, stale: point.calendarStale)
                        sourceDot(title: "GDELT", ok: point.gdeltOK, stale: point.gdeltStale)
                        sourceDot(title: "OFF", ok: point.officialOK, stale: point.officialStale)
                        Spacer()
                    }
                }
            }
        }
    }

    private func policyPanel(snapshot: NewsPulseSnapshot) -> some View {
        guard snapshot.policySummary != nil || snapshot.healthSummary != nil else {
            return AnyView(EmptyView())
        }
        return AnyView(
            FXAIVisualEffectSurface {
                VStack(alignment: .leading, spacing: 12) {
                    Text("Policy & Replay")
                        .font(.headline)
                        .foregroundStyle(FXAITheme.textPrimary)

                    if let policy = snapshot.policySummary {
                        metricRow(label: "Active pairs", value: "\(policy.activePairs.count)")
                        metricRow(label: "Broker symbol map", value: "\(policy.brokerSymbolMapCount)")
                        if let active = policy.watchlists.first(where: { $0.key == "active" }) {
                            Text(active.values.prefix(12).joined(separator: ", "))
                                .font(.caption)
                                .foregroundStyle(FXAITheme.textMuted)
                                .lineLimit(2)
                        }
                    }

                    if let health = snapshot.healthSummary {
                        metricRow(label: "Required stale", value: health.requiredSourcesStale ? "yes" : "no")
                        metricRow(label: "History rows", value: "\(health.historyRecordsLocal)")
                        metricRow(label: "Story count", value: "\(health.storyCount)")
                        if let backoff = health.gdeltBackoffUntil {
                            Text("GDELT backoff until \(FXAIFormatting.relativeDateString(for: backoff))")
                                .font(.caption)
                                .foregroundStyle(FXAITheme.warning)
                        }
                    }
                }
            }
        )
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
                            .frame(width: 130, alignment: .leading)
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

    private func infoChip(title: String, value: String) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(title)
                .font(.caption)
                .foregroundStyle(FXAITheme.textMuted)
            Text(value)
                .font(.caption.weight(.semibold))
                .foregroundStyle(FXAITheme.textPrimary)
        }
        .padding(10)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(
            RoundedRectangle(cornerRadius: 14, style: .continuous)
                .fill(FXAITheme.panel.opacity(0.48))
        )
    }

    private func sourceDot(title: String, ok: Bool, stale: Bool) -> some View {
        HStack(spacing: 6) {
            Circle()
                .fill((ok && !stale) ? FXAITheme.success : FXAITheme.warning)
                .frame(width: 8, height: 8)
            Text(title)
                .font(.caption2.weight(.semibold))
                .foregroundStyle(FXAITheme.textMuted)
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
        if status.required && (status.stale || !status.ok) {
            return FXAITheme.warning
        }
        if !status.enabled {
            return FXAITheme.textMuted
        }
        return status.stale || !status.ok ? FXAITheme.warning : FXAITheme.success
    }

    private func statusIcon(for status: NewsPulseSourceStatus) -> String {
        if !status.enabled {
            return "circle.dashed"
        }
        if status.stale || !status.ok {
            return "exclamationmark.triangle.fill"
        }
        return "checkmark.seal.fill"
    }

    private func sourceFootnote(for status: NewsPulseSourceStatus) -> String {
        if !status.enabled {
            return "Optional source is disabled."
        }
        if let lastError = status.lastError, !lastError.isEmpty {
            return lastError
        }
        if let backoffUntil = status.backoffUntil {
            return "Backoff until \(FXAIFormatting.relativeDateString(for: backoffUntil))"
        }
        return status.lastSuccessAt.map { "Last success \($0.formatted(date: .omitted, time: .shortened))" } ?? "Waiting for data"
    }

    private func percentString(_ value: Double) -> String {
        String(format: "%.0f%%", max(0, min(value, 1)) * 100.0)
    }

    private func syncSelection() {
        guard let first = visiblePairs.first else {
            selectedPairID = ""
            return
        }
        if !visiblePairs.contains(where: { $0.pair == selectedPairID }) {
            selectedPairID = first.pair
        }
    }
}
