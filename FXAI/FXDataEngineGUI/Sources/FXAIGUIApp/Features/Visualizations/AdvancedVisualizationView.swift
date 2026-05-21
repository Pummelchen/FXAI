import Charts
import FXAIGUICore
import SwiftUI

struct AdvancedVisualizationView: View {
    @EnvironmentObject private var model: FXAIGUIModel

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 18) {
                SectionHeader(
                    title: "Advanced Visuals",
                    subtitle: "Phase 5 surfaces for dense family stress maps, artifact diff heatmaps, world-plan structure, and promotion or attribution timelines."
                )

                if let snapshot = model.visualizationSnapshot, !snapshot.symbolDetails.isEmpty {
                    summary(snapshot: snapshot)

                    if let heatmap = snapshot.familyStressHeatmap {
                        MetalHeatmapPanel(heatmap: heatmap)
                    }

                    HStack {
                        Picker("Symbol", selection: $model.selectedVisualizationSymbol) {
                            ForEach(snapshot.symbols, id: \.self) { symbol in
                                Text(symbol).tag(symbol)
                            }
                        }
                        .pickerStyle(.segmented)
                        .frame(maxWidth: 420)

                        Spacer()
                    }

                    if let detail = model.selectedVisualizationDetail {
                        worldSection(detail: detail)
                        attributionSection(detail: detail)
                        if let heatmap = detail.artifactDiffHeatmap {
                            MetalHeatmapPanel(heatmap: heatmap)
                        }
                        timelineSection(snapshot: snapshot, detail: detail)
                    }
                } else {
                    EmptyStateView(
                        title: "No advanced visuals available",
                        message: "Generate current runtime and Research OS artifacts first. Phase 5 relies on live deployment, attribution, world, and lineage outputs already emitted by FXAI.",
                        symbolName: "sparkles.tv"
                    )
                }
            }
        }
    }

    private func summary(snapshot: AdvancedVisualizationSnapshot) -> some View {
        LazyVGrid(
            columns: [
                GridItem(.flexible(minimum: 220), spacing: 16),
                GridItem(.flexible(minimum: 220), spacing: 16),
                GridItem(.flexible(minimum: 220), spacing: 16),
                GridItem(.flexible(minimum: 220), spacing: 16)
            ],
            spacing: 16
        ) {
            MetricCard(
                title: "Profile",
                value: snapshot.profileName ?? "unknown",
                footnote: "Research OS profile behind the current advanced visualization state.",
                symbolName: "folder.badge.gearshape",
                tint: FXAITheme.accent
            )
            MetricCard(
                title: "Symbols",
                value: "\(snapshot.symbols.count)",
                footnote: "Symbols with enough deployment and Research OS data to render Phase 5 surfaces.",
                symbolName: "globe.europe.africa.fill",
                tint: FXAITheme.accentSoft
            )
            MetricCard(
                title: "Timeline Events",
                value: "\(snapshot.globalTimeline.count)",
                footnote: "Promotion, branch, audit, deployment, and attribution events rendered in the current timeline.",
                symbolName: "timeline.selection",
                tint: FXAITheme.warning
            )
            MetricCard(
                title: "Heatmaps",
                value: snapshot.familyStressHeatmap == nil ? "1" : "2+",
                footnote: "One global family stress surface and one per-symbol artifact diff surface when data is present.",
                symbolName: "square.grid.3x3.topmiddle.filled",
                tint: FXAITheme.success
            )
        }
    }

    private func worldSection(detail: SymbolVisualizationDetail) -> some View {
        HStack(alignment: .top, spacing: 16) {
            FXAIVisualEffectSurface {
                VStack(alignment: .leading, spacing: 12) {
                    HStack {
                        Text("World Plan Sessions")
                            .font(.headline)
                            .foregroundStyle(FXAITheme.textPrimary)
                        Spacer()
                        if !detail.weakScenarios.isEmpty {
                            Text(detail.weakScenarios.joined(separator: ", "))
                                .font(.caption)
                                .foregroundStyle(FXAITheme.textMuted)
                        }
                    }

                    Chart(detail.worldSessionScales) { point in
                        BarMark(
                            x: .value("Session", point.label),
                            y: .value("Sigma", point.value)
                        )
                        .foregroundStyle(FXAITheme.accent.gradient)

                        if let secondary = point.secondaryValue {
                            BarMark(
                                x: .value("Session", point.label),
                                y: .value("Spread", secondary)
                            )
                            .foregroundStyle(FXAITheme.warning.gradient)
                            .position(by: .value("Type", "Spread"))
                        }
                    }
                    .frame(height: 220)
                }
            }

            FXAIVisualEffectSurface {
                VStack(alignment: .leading, spacing: 12) {
                    Text("Stress and Recovery")
                        .font(.headline)
                        .foregroundStyle(FXAITheme.textPrimary)

                    Chart(detail.worldStressMetrics) { point in
                        BarMark(
                            x: .value("Metric", point.label),
                            y: .value("Value", point.value)
                        )
                        .foregroundStyle(FXAITheme.accentSoft.gradient)
                        .cornerRadius(4)
                    }
                    .chartYAxis {
                        AxisMarks(position: .leading)
                    }
                    .frame(height: 220)
                }
            }
        }
    }

    private func attributionSection(detail: SymbolVisualizationDetail) -> some View {
        HStack(alignment: .top, spacing: 16) {
            FXAIVisualEffectSurface {
                VStack(alignment: .leading, spacing: 12) {
                    Text("Family Weights")
                        .font(.headline)
                        .foregroundStyle(FXAITheme.textPrimary)

                    if detail.familyWeights.isEmpty {
                        Text("No family weights found in the current attribution profile.")
                            .foregroundStyle(FXAITheme.textSecondary)
                    } else {
                        Chart(detail.familyWeights.prefix(8)) { point in
                            BarMark(
                                x: .value("Weight", point.value),
                                y: .value("Family", point.label)
                            )
                            .foregroundStyle(FXAITheme.accent.gradient)
                            .cornerRadius(4)
                        }
                        .frame(height: 260)
                    }
                }
            }

            FXAIVisualEffectSurface {
                VStack(alignment: .leading, spacing: 12) {
                    Text("Feature Attribution")
                        .font(.headline)
                        .foregroundStyle(FXAITheme.textPrimary)

                    if detail.featureWeights.isEmpty {
                        Text("No feature-group attribution weights found.")
                            .foregroundStyle(FXAITheme.textSecondary)
                    } else {
                        Chart(detail.featureWeights.prefix(8)) { point in
                            BarMark(
                                x: .value("Weight", point.value),
                                y: .value("Feature", point.label)
                            )
                            .foregroundStyle(FXAITheme.warning.gradient)
                            .cornerRadius(4)
                        }
                        .frame(height: 260)
                    }
                }
            }

            FXAIVisualEffectSurface {
                VStack(alignment: .leading, spacing: 12) {
                    Text("Plugin Routing")
                        .font(.headline)
                        .foregroundStyle(FXAITheme.textPrimary)

                    if detail.pluginWeights.isEmpty {
                        Text("No plugin routing weights were emitted for this symbol.")
                            .foregroundStyle(FXAITheme.textSecondary)
                    } else {
                        Chart(detail.pluginWeights.prefix(8)) { point in
                            BarMark(
                                x: .value("Weight", point.value),
                                y: .value("Plugin", point.label)
                            )
                            .foregroundStyle(FXAITheme.success.gradient)
                            .cornerRadius(4)
                        }
                        .frame(height: 260)
                    }
                }
            }
        }
    }

    private func timelineSection(snapshot: AdvancedVisualizationSnapshot, detail: SymbolVisualizationDetail) -> some View {
        FXAIVisualEffectSurface {
            VStack(alignment: .leading, spacing: 14) {
                Text("Promotion and Attribution Timeline")
                    .font(.headline)
                    .foregroundStyle(FXAITheme.textPrimary)

                let events = combinedTimeline(snapshot: snapshot, detail: detail)
                if events.isEmpty {
                    Text("No timeline events are available for this symbol.")
                        .foregroundStyle(FXAITheme.textSecondary)
                } else {
                    Chart(events) { event in
                        PointMark(
                            x: .value("Date", event.date),
                            y: .value("Category", event.category)
                        )
                        .symbolSize(90)
                        .foregroundStyle(color(for: event.category))

                        if let score = event.score {
                            RuleMark(
                                x: .value("Date", event.date),
                                yStart: .value("Category", event.category),
                                yEnd: .value("Category", event.category)
                            )
                            .foregroundStyle(color(for: event.category).opacity(0.25))
                            .annotation(position: .top, alignment: .center) {
                                Text(String(format: "%.2f", score))
                                    .font(.caption2)
                                    .foregroundStyle(FXAITheme.textMuted)
                            }
                        }
                    }
                    .frame(height: 260)

                    VStack(spacing: 10) {
                        ForEach(events.suffix(8).reversed()) { event in
                            HStack(alignment: .top, spacing: 12) {
                                Circle()
                                    .fill(color(for: event.category))
                                    .frame(width: 8, height: 8)
                                    .padding(.top, 6)

                                VStack(alignment: .leading, spacing: 4) {
                                    Text(event.title)
                                        .font(.subheadline.weight(.semibold))
                                        .foregroundStyle(FXAITheme.textPrimary)
                                    Text(event.detail)
                                        .font(.caption)
                                        .foregroundStyle(FXAITheme.textSecondary)
                                    Text(event.date.formatted(date: .abbreviated, time: .shortened))
                                        .font(.caption2)
                                        .foregroundStyle(FXAITheme.textMuted)
                                }

                                Spacer()
                            }
                        }
                    }
                }
            }
        }
    }

    private func combinedTimeline(snapshot: AdvancedVisualizationSnapshot, detail: SymbolVisualizationDetail) -> [VisualizationTimelineEvent] {
        let global = snapshot.globalTimeline.filter { $0.title.contains(detail.symbol) || $0.detail.contains(detail.symbol) }
        return (detail.timeline + global)
            .sorted { lhs, rhs in
                if lhs.date == rhs.date {
                    return lhs.title < rhs.title
                }
                return lhs.date < rhs.date
            }
    }

    private func color(for category: String) -> Color {
        switch category {
        case "promotion":
            return FXAITheme.warning
        case "branch":
            return FXAITheme.accent
        case "audit":
            return FXAITheme.success
        case "attribution":
            return FXAITheme.accentSoft
        case "analog":
            return FXAITheme.textSecondary
        default:
            return FXAITheme.textMuted
        }
    }
}
