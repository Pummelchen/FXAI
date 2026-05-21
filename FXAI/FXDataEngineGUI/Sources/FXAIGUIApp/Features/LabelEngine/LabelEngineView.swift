import FXAIGUICore
import SwiftUI

struct LabelEngineView: View {
    @EnvironmentObject private var model: FXAIGUIModel

    private var visibleBuilds: [LabelEngineBuildSnapshot] {
        model.labelEngineSnapshot?.builds ?? []
    }

    private var selectedBuild: LabelEngineBuildSnapshot? {
        guard !visibleBuilds.isEmpty else { return nil }
        return visibleBuilds.first(where: { $0.datasetKey == model.selectedLabelEngineDatasetKey }) ?? visibleBuilds.first
    }

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 18) {
                SectionHeader(
                    title: "Label Engine",
                    subtitle: "Inspect the shared multi-horizon target layer that now separates direction, move quality, timing, tradeability, and meta-label acceptance across exported FXAI datasets."
                )

                if let snapshot = model.labelEngineSnapshot, !snapshot.builds.isEmpty {
                    topStatus(snapshot: snapshot)
                    buildWorkspace(snapshot: snapshot)
                    statusPanel(snapshot: snapshot)
                } else {
                    EmptyStateView(
                        title: "Label-engine artifacts not found",
                        message: "Run label-engine-validate, build label artifacts for one or more exported datasets with label-engine-build, rebuild the aggregated report, and refresh the GUI.",
                        symbolName: "target"
                    )
                }
            }
        }
    }

    private func topStatus(snapshot: LabelEngineSnapshot) -> some View {
        let horizonCount = selectedBuild?.horizons.count ?? 0
        let metaAcceptance = doubleValue(for: "meta_acceptance_rate", in: selectedBuild?.summaryMetrics ?? [])
        return LazyVGrid(columns: [GridItem(.adaptive(minimum: 180), spacing: 16)], spacing: 16) {
            MetricCard(
                title: "Artifacts",
                value: "\(snapshot.artifactCount)",
                footnote: "Label-engine bundles currently indexed for operator review.",
                symbolName: "tray.full.fill",
                tint: FXAITheme.accentSoft
            )
            MetricCard(
                title: "Selected Symbol",
                value: selectedBuild?.symbol.isEmpty == false ? selectedBuild?.symbol ?? "Unknown" : "Unknown",
                footnote: "Dataset symbol behind the currently selected label bundle.",
                symbolName: "chart.xyaxis.line",
                tint: FXAITheme.accent
            )
            MetricCard(
                title: "Horizons",
                value: "\(horizonCount)",
                footnote: "Configured forecast horizons stored inside the selected bundle.",
                symbolName: "timeline.selection",
                tint: FXAITheme.success
            )
            MetricCard(
                title: "Meta Accept",
                value: percentString(metaAcceptance),
                footnote: "Acceptance rate of the meta-labeling layer over the selected candidate stream.",
                symbolName: "checkmark.shield.fill",
                tint: metaAcceptance >= 0.5 ? FXAITheme.success : FXAITheme.warning
            )
        }
    }

    private func buildWorkspace(snapshot _: LabelEngineSnapshot) -> some View {
        FXAIVisualEffectSurface {
            VStack(alignment: .leading, spacing: 14) {
                Text("Artifact Bundles")
                    .font(.headline)
                    .foregroundStyle(FXAITheme.textPrimary)

                ViewThatFits(in: .horizontal) {
                    HStack(alignment: .top, spacing: 18) {
                        buildList
                            .frame(width: 360, alignment: .topLeading)
                        buildDetail
                    }

                    VStack(alignment: .leading, spacing: 18) {
                        buildList
                        buildDetail
                    }
                }
            }
        }
    }

    private var buildList: some View {
        VStack(alignment: .leading, spacing: 10) {
            ForEach(visibleBuilds) { build in
                Button {
                    model.selectedLabelEngineDatasetKey = build.datasetKey
                } label: {
                    HStack(alignment: .top, spacing: 10) {
                        VStack(alignment: .leading, spacing: 4) {
                            Text(build.symbol)
                                .font(.subheadline.weight(.semibold))
                                .foregroundStyle(FXAITheme.textPrimary)
                            Text(build.datasetKey)
                                .font(.caption)
                                .foregroundStyle(FXAITheme.textSecondary)
                                .lineLimit(2)
                            Text("\(build.barCount) bars • \(build.executionProfile)")
                                .font(.caption)
                                .foregroundStyle(FXAITheme.textMuted)
                        }
                        Spacer(minLength: 12)
                        VStack(alignment: .trailing, spacing: 4) {
                            Text(shortDate(build.generatedAt))
                                .font(.caption.weight(.semibold))
                                .foregroundStyle(FXAITheme.textPrimary)
                            Text("v\(build.labelVersion)")
                                .font(.caption2)
                                .foregroundStyle(FXAITheme.textMuted)
                        }
                    }
                    .padding(12)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .background(
                        RoundedRectangle(cornerRadius: 16, style: .continuous)
                            .fill(model.selectedLabelEngineDatasetKey == build.datasetKey ? FXAITheme.accentSoft.opacity(0.14) : FXAITheme.panel.opacity(0.55))
                    )
                }
                .buttonStyle(.plain)
            }
        }
    }

    private var buildDetail: some View {
        VStack(alignment: .leading, spacing: 16) {
            if let selectedBuild {
                HStack {
                    VStack(alignment: .leading, spacing: 4) {
                        Text(selectedBuild.symbol)
                            .font(.title3.weight(.semibold))
                            .foregroundStyle(FXAITheme.textPrimary)
                        Text(selectedBuild.datasetKey)
                            .font(.caption)
                            .foregroundStyle(FXAITheme.textSecondary)
                    }
                    Spacer()
                    StatusBadge(title: "Execution", value: selectedBuild.executionProfile, tint: FXAITheme.accentSoft)
                }

                LazyVGrid(columns: [GridItem(.adaptive(minimum: 170), spacing: 12)], spacing: 12) {
                    detailMetric(title: "Bars", value: "\(selectedBuild.barCount)", tint: FXAITheme.accentSoft)
                    detailMetric(title: "Point Size", value: decimalString(selectedBuild.pointSize), tint: FXAITheme.accent)
                    detailMetric(title: "Tradeable Long", value: percentString(doubleValue(for: "long_tradeability_rate", in: selectedBuild.summaryMetrics)), tint: FXAITheme.success)
                    detailMetric(title: "Tradeable Short", value: percentString(doubleValue(for: "short_tradeability_rate", in: selectedBuild.summaryMetrics)), tint: FXAITheme.warning)
                    detailMetric(title: "Candidates", value: "\(intValue(for: "candidate_count", in: selectedBuild.summaryMetrics))", tint: FXAITheme.accentSoft)
                    detailMetric(title: "Meta Accept", value: percentString(doubleValue(for: "meta_acceptance_rate", in: selectedBuild.summaryMetrics)), tint: FXAITheme.success)
                }

                metricListCard(title: "Summary Metrics", values: selectedBuild.summaryMetrics)
                metricListCard(title: "Meta Summary", values: selectedBuild.metaSummary)
                metricListCard(title: "Quality Flags", values: selectedBuild.qualityFlags)

                VStack(alignment: .leading, spacing: 8) {
                    Text("Horizon Breakdown")
                        .font(.subheadline.weight(.semibold))
                        .foregroundStyle(FXAITheme.textPrimary)
                    ForEach(selectedBuild.horizons) { horizon in
                        HStack {
                            VStack(alignment: .leading, spacing: 3) {
                                Text("\(horizon.horizonID) • \(horizon.bars) bars")
                                    .font(.subheadline.weight(.semibold))
                                    .foregroundStyle(FXAITheme.textPrimary)
                                Text("\(horizon.sampleCount) rows • \(horizon.candidateCount) candidates")
                                    .font(.caption)
                                    .foregroundStyle(FXAITheme.textMuted)
                            }
                            Spacer()
                            VStack(alignment: .trailing, spacing: 3) {
                                Text("L \(percentString(horizon.longTradeabilityRate)) / S \(percentString(horizon.shortTradeabilityRate))")
                                    .font(.caption.weight(.semibold))
                                    .foregroundStyle(FXAITheme.textPrimary)
                                Text("Accept \(percentString(horizon.candidateAcceptanceRate))")
                                    .font(.caption)
                                    .foregroundStyle(FXAITheme.textSecondary)
                            }
                        }
                        .padding(10)
                        .background(
                            RoundedRectangle(cornerRadius: 14, style: .continuous)
                                .fill(FXAITheme.panel.opacity(0.50))
                        )
                    }
                }

                if !selectedBuild.topReasons.isEmpty {
                    VStack(alignment: .leading, spacing: 8) {
                        Text("Top Failure Reasons")
                            .font(.subheadline.weight(.semibold))
                            .foregroundStyle(FXAITheme.textPrimary)
                        ForEach(selectedBuild.topReasons) { reason in
                            metricRow(label: reason.reason, value: "\(reason.count)")
                        }
                    }
                }

                metricListCard(title: "Artifact Paths", values: selectedBuild.artifactPaths)
            } else {
                Text("No label-engine bundle is selected.")
                    .foregroundStyle(FXAITheme.textSecondary)
            }
        }
    }

    private func statusPanel(snapshot: LabelEngineSnapshot) -> some View {
        FXAIVisualEffectSurface {
            VStack(alignment: .leading, spacing: 14) {
                Text("Status and Aggregation")
                    .font(.headline)
                    .foregroundStyle(FXAITheme.textPrimary)
                metricListCard(title: "Status", values: snapshot.statusRecords)
                if !snapshot.artifactPaths.isEmpty {
                    metricListCard(title: "Report Paths", values: snapshot.artifactPaths)
                }
            }
        }
    }

    private func doubleValue(for key: String, in values: [KeyValueRecord]) -> Double {
        values.first(where: { $0.key == key }).flatMap { Double($0.value) } ?? 0
    }

    private func intValue(for key: String, in values: [KeyValueRecord]) -> Int {
        values.first(where: { $0.key == key }).flatMap { Int($0.value) } ?? 0
    }

    private func decimalString(_ value: Double) -> String {
        String(format: "%.5f", value)
    }

    private func shortDate(_ value: Date?) -> String {
        guard let value else { return "Unknown" }
        return value.formatted(date: .abbreviated, time: .shortened)
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
}
