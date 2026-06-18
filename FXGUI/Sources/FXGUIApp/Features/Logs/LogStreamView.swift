import FXGUICore
import SwiftUI

struct LogStreamView: View {
    @EnvironmentObject private var model: FXGUIModel

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 18) {
                SectionHeader(
                    title: "Log Streams",
                    subtitle: "Inspect categorized log output from FXAI subsystems including certification, database, plugin tests, agent fleet, execution safety, promotion, and runtime."
                )

                if let snapshot = model.logStreamSnapshot, !snapshot.streams.isEmpty {
                    summary(snapshot: snapshot)

                    HStack(alignment: .top, spacing: 16) {
                        streamList(snapshot: snapshot)
                        streamDetail
                    }
                } else {
                    EmptyStateView(
                        title: "No log streams available",
                        message: "Connect to an FXAI project and refresh to populate log stream data.",
                        symbolName: "text.alignleft"
                    )
                }
            }
        }
    }

    private func summary(snapshot: LogStreamSnapshot) -> some View {
        LazyVGrid(
            columns: [
                GridItem(.flexible(minimum: 200), spacing: 16),
                GridItem(.flexible(minimum: 200), spacing: 16),
                GridItem(.flexible(minimum: 200), spacing: 16)
            ],
            spacing: 16
        ) {
            MetricCard(
                title: "Streams",
                value: "\(snapshot.streams.count)",
                footnote: "Active log stream categories.",
                symbolName: "text.alignleft",
                tint: FXAITheme.accent
            )
            MetricCard(
                title: "Total Events",
                value: "\(snapshot.streams.reduce(0) { $0 + $1.eventCount })",
                footnote: "Combined event count across all streams.",
                symbolName: "number",
                tint: FXAITheme.success
            )
            MetricCard(
                title: "Warnings",
                value: "\(snapshot.streams.filter { $0.level == .warning || $0.level == .error || $0.level == .critical }.count)",
                footnote: "Streams with elevated severity.",
                symbolName: "exclamationmark.triangle.fill",
                tint: FXAITheme.warning
            )
        }
    }

    private func streamList(snapshot: LogStreamSnapshot) -> some View {
        FXAIVisualEffectSurface {
            VStack(alignment: .leading, spacing: 8) {
                Text("Stream Categories")
                    .font(.headline)
                    .foregroundStyle(FXAITheme.textPrimary)

                ForEach(snapshot.streams) { stream in
                    Button {
                        model.selectedLogStreamKey = stream.streamKey
                    } label: {
                        HStack {
                            Circle()
                                .fill(levelColor(stream.level))
                                .frame(width: 8, height: 8)
                            VStack(alignment: .leading, spacing: 2) {
                                Text(stream.displayName)
                                    .font(.subheadline.weight(.semibold))
                                    .foregroundStyle(FXAITheme.textPrimary)
                                Text("\(stream.eventCount) events · \(stream.level.title)")
                                    .font(.caption2)
                                    .foregroundStyle(FXAITheme.textMuted)
                            }
                            Spacer()
                        }
                        .padding(10)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .background(
                            RoundedRectangle(cornerRadius: 12)
                                .fill(model.selectedLogStreamKey == stream.streamKey ? FXAITheme.accent.opacity(0.14) : FXAITheme.backgroundSecondary.opacity(0.4))
                        )
                    }
                    .buttonStyle(.plain)
                }
            }
        }
        .frame(maxWidth: 320, alignment: .topLeading)
    }

    @ViewBuilder
    private var streamDetail: some View {
        if let stream = model.selectedLogStream {
            FXAIVisualEffectSurface {
                VStack(alignment: .leading, spacing: 12) {
                    HStack {
                        Text(stream.displayName)
                            .font(.system(size: 20, weight: .semibold, design: .rounded))
                            .foregroundStyle(FXAITheme.textPrimary)
                        Spacer()
                        StatusBadge(
                            title: "Level",
                            value: stream.level.title,
                            tint: levelColor(stream.level)
                        )
                    }

                    LazyVGrid(
                        columns: [
                            GridItem(.flexible(minimum: 180), spacing: 12),
                            GridItem(.flexible(minimum: 180), spacing: 12)
                        ],
                        spacing: 12
                    ) {
                        MetricCard(
                            title: "Events",
                            value: "\(stream.eventCount)",
                            footnote: "Total events in this stream.",
                            symbolName: "number",
                            tint: FXAITheme.accent
                        )
                        MetricCard(
                            title: "Last Event",
                            value: stream.lastEventAt.map { FXAIFormatting.dateTimeString(for: $0) } ?? "N/A",
                            footnote: "Most recent event timestamp.",
                            symbolName: "clock.fill",
                            tint: FXAITheme.accentSoft
                        )
                    }

                    Text("Recent Output")
                        .font(.subheadline.weight(.semibold))
                        .foregroundStyle(FXAITheme.textPrimary)

                    if stream.recentLines.isEmpty {
                        Text("No recent log lines available for this stream.")
                            .font(.caption)
                            .foregroundStyle(FXAITheme.textMuted)
                    } else {
                        VStack(alignment: .leading, spacing: 4) {
                            ForEach(stream.recentLines, id: \.self) { line in
                                Text(line)
                                    .font(.system(.caption, design: .monospaced))
                                    .foregroundStyle(FXAITheme.textSecondary)
                                    .textSelection(.enabled)
                                    .padding(.vertical, 2)
                            }
                        }
                        .padding(12)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .background(
                            RoundedRectangle(cornerRadius: 14)
                                .fill(FXAITheme.backgroundSecondary.opacity(0.6))
                        )
                    }
                }
            }
        } else {
            EmptyStateView(
                title: "No stream selected",
                message: "Select a log stream to inspect its recent output.",
                symbolName: "text.alignleft"
            )
        }
    }

    private func levelColor(_ level: LogLevel) -> Color {
        switch level {
        case .debug: return FXAITheme.textMuted
        case .info: return FXAITheme.accent
        case .warning: return FXAITheme.warning
        case .error: return .orange
        case .critical: return .red
        }
    }
}
