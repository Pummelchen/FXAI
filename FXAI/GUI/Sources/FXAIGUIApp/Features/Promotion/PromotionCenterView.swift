import Charts
import FXAIGUICore
import SwiftUI

struct PromotionCenterView: View {
    @EnvironmentObject private var model: FXAIGUIModel

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 18) {
                SectionHeader(
                    title: "Promotion Center",
                    subtitle: "See which plugins are currently promoted, how champions compare to challengers, and which set paths or tiers are active."
                )

                if let runtimeSnapshot = model.runtimeSnapshot {
                    summary(snapshot: runtimeSnapshot)
                    championChart(champions: runtimeSnapshot.champions)
                    championList(champions: runtimeSnapshot.champions)
                } else {
                    EmptyStateView(
                        title: "No promotion data loaded",
                        message: "Refresh the project after Offline Lab promotion outputs are present.",
                        symbolName: "rosette"
                    )
                }
            }
        }
    }

    private func summary(snapshot: RuntimeOperationsSnapshot) -> some View {
        LazyVGrid(
            columns: [
                GridItem(.flexible(minimum: 220), spacing: 16),
                GridItem(.flexible(minimum: 220), spacing: 16),
                GridItem(.flexible(minimum: 220), spacing: 16)
            ],
            spacing: 16
        ) {
            MetricCard(
                title: "Champions",
                value: "\(snapshot.champions.count)",
                footnote: "Current champion records discovered from ResearchOS promotion output.",
                symbolName: "crown.fill",
                tint: FXAITheme.warning
            )
            MetricCard(
                title: "Deployments",
                value: "\(snapshot.deployments.count)",
                footnote: "Runtime deployments currently visible in the dashboard surface.",
                symbolName: "waveform.path.ecg",
                tint: FXAITheme.accent
            )
            MetricCard(
                title: "Profile",
                value: snapshot.profileName ?? "unknown",
                footnote: "Latest ResearchOS profile the GUI is reading.",
                symbolName: "person.crop.rectangle.stack.fill",
                tint: FXAITheme.accentSoft
            )
        }
    }

    private func championChart(champions: [PromotionChampionRecord]) -> some View {
        FXAIVisualEffectSurface {
            VStack(alignment: .leading, spacing: 12) {
                Text("Champion vs Challenger")
                    .font(.headline)
                    .foregroundStyle(FXAITheme.textPrimary)

                if champions.isEmpty {
                    Text("No champions were found in the current promotion outputs.")
                        .foregroundStyle(FXAITheme.textSecondary)
                } else {
                    Chart {
                        ForEach(champions) { champion in
                            BarMark(
                                x: .value("Symbol", champion.symbol),
                                y: .value("Champion", champion.championScore)
                            )
                            .foregroundStyle(FXAITheme.success.gradient)

                            BarMark(
                                x: .value("Symbol", champion.symbol),
                                y: .value("Challenger", champion.challengerScore)
                            )
                            .foregroundStyle(FXAITheme.warning.gradient)
                            .position(by: .value("Series", "challenger"))
                        }
                    }
                    .frame(height: 250)
                }
            }
        }
    }

    private func championList(champions: [PromotionChampionRecord]) -> some View {
        FXAIVisualEffectSurface {
            VStack(alignment: .leading, spacing: 12) {
                Text("Champion Records")
                    .font(.headline)
                    .foregroundStyle(FXAITheme.textPrimary)

                if champions.isEmpty {
                    Text("No champion records are available.")
                        .foregroundStyle(FXAITheme.textSecondary)
                } else {
                    ForEach(champions) { champion in
                        VStack(alignment: .leading, spacing: 8) {
                            HStack {
                                VStack(alignment: .leading, spacing: 4) {
                                    Text("\(champion.symbol) • \(champion.pluginName)")
                                        .font(.subheadline.weight(.semibold))
                                        .foregroundStyle(FXAITheme.textPrimary)
                                    Text("\(champion.status) • \(champion.promotionTier)")
                                        .font(.caption)
                                        .foregroundStyle(FXAITheme.textSecondary)
                                }
                                Spacer()
                                if let path = champion.setPath {
                                    Button("Reveal Set") {
                                        model.openInFinder(path)
                                    }
                                    .buttonStyle(.bordered)
                                }
                            }

                            HStack(spacing: 18) {
                                Text("Champion: \(champion.championScore, format: .number.precision(.fractionLength(2)))")
                                Text("Challenger: \(champion.challengerScore, format: .number.precision(.fractionLength(2)))")
                                Text("Portfolio: \(champion.portfolioScore, format: .number.precision(.fractionLength(2)))")
                            }
                            .font(.caption)
                            .foregroundStyle(FXAITheme.textSecondary)

                            if let reviewedAt = champion.reviewedAt {
                                Text("Reviewed \(FXAIFormatting.dateTimeString(for: reviewedAt))")
                                    .font(.caption)
                                    .foregroundStyle(FXAITheme.textMuted)
                            }
                        }
                        .padding(.vertical, 8)
                    }
                }
            }
        }
    }
}
