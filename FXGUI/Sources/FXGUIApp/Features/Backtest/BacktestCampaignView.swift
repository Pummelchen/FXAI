import Charts
import FXGUICore
import SwiftUI

struct BacktestCampaignView: View {
    @EnvironmentObject private var model: FXGUIModel

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 18) {
                SectionHeader(
                    title: "Backtest Campaigns",
                    subtitle: "Track multi-scenario backtest runs, pass rates, and performance metrics for each plugin-symbol combination."
                )

                if let snapshot = model.backtestCampaignSnapshot, !snapshot.campaigns.isEmpty {
                    summary(snapshot: snapshot)
                    campaignList(snapshot: snapshot)
                    selectedCampaignDetail
                } else {
                    EmptyStateView(
                        title: "No backtest campaigns",
                        message: "Run multi-scenario backtests through the Backtest Builder to generate campaign tracking data.",
                        symbolName: "flag.checkered"
                    )
                }
            }
        }
    }

    private func summary(snapshot: BacktestCampaignSnapshot) -> some View {
        LazyVGrid(
            columns: [
                GridItem(.flexible(minimum: 200), spacing: 16),
                GridItem(.flexible(minimum: 200), spacing: 16),
                GridItem(.flexible(minimum: 200), spacing: 16),
                GridItem(.flexible(minimum: 200), spacing: 16)
            ],
            spacing: 16
        ) {
            MetricCard(
                title: "Total Campaigns",
                value: "\(snapshot.campaigns.count)",
                footnote: "All tracked backtest campaigns.",
                symbolName: "flag.checkered",
                tint: FXAITheme.accent
            )
            MetricCard(
                title: "Active",
                value: "\(snapshot.activeCount)",
                footnote: "Currently running campaigns.",
                symbolName: "arrow.triangle.2.circlepath",
                tint: FXAITheme.warning
            )
            MetricCard(
                title: "Completed",
                value: "\(snapshot.completedCount)",
                footnote: "Finished campaigns with results.",
                symbolName: "checkmark.circle.fill",
                tint: FXAITheme.success
            )
            MetricCard(
                title: "Avg Pass Rate",
                value: averagePassRate(snapshot),
                footnote: "Mean scenario pass rate across campaigns.",
                symbolName: "percent",
                tint: FXAITheme.accentSoft
            )
        }
    }

    private func campaignList(snapshot: BacktestCampaignSnapshot) -> some View {
        FXAIVisualEffectSurface {
            VStack(alignment: .leading, spacing: 10) {
                Text("Campaigns")
                    .font(.headline)
                    .foregroundStyle(FXAITheme.textPrimary)

                ForEach(snapshot.campaigns) { campaign in
                    Button {
                        model.selectedCampaignID = campaign.id
                    } label: {
                        HStack {
                            Image(systemName: campaign.status.symbolName)
                                .foregroundStyle(campaignStatusColor(campaign.status))
                            VStack(alignment: .leading, spacing: 2) {
                                Text(campaign.name)
                                    .font(.subheadline.weight(.semibold))
                                    .foregroundStyle(FXAITheme.textPrimary)
                                HStack(spacing: 8) {
                                    Text(campaign.status.title)
                                        .font(.caption2.weight(.semibold))
                                        .foregroundStyle(campaignStatusColor(campaign.status))
                                    Text("\(campaign.passedCount)/\(campaign.scenarioCount) passed")
                                        .font(.caption2)
                                        .foregroundStyle(FXAITheme.textMuted)
                                }
                            }
                            Spacer()
                            Text("\(Int(campaign.passRate * 100))%")
                                .font(.caption.weight(.bold))
                                .foregroundStyle(campaign.passRate >= 0.7 ? FXAITheme.success : FXAITheme.warning)
                        }
                        .padding(12)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .background(
                            RoundedRectangle(cornerRadius: 14)
                                .fill(model.selectedCampaignID == campaign.id ? FXAITheme.accent.opacity(0.14) : FXAITheme.backgroundSecondary.opacity(0.5))
                        )
                    }
                    .buttonStyle(.plain)
                }
            }
        }
    }

    @ViewBuilder
    private var selectedCampaignDetail: some View {
        if let campaign = model.selectedCampaign {
            FXAIVisualEffectSurface {
                VStack(alignment: .leading, spacing: 14) {
                    Text(campaign.name)
                        .font(.system(size: 20, weight: .semibold, design: .rounded))
                        .foregroundStyle(FXAITheme.textPrimary)

                    LazyVGrid(
                        columns: [
                            GridItem(.flexible(minimum: 160), spacing: 12),
                            GridItem(.flexible(minimum: 160), spacing: 12),
                            GridItem(.flexible(minimum: 160), spacing: 12),
                            GridItem(.flexible(minimum: 160), spacing: 12)
                        ],
                        spacing: 12
                    ) {
                        MetricCard(
                            title: "Plugin",
                            value: campaign.pluginName,
                            footnote: "Strategy plugin under test.",
                            symbolName: "shippingbox.fill",
                            tint: FXAITheme.accent
                        )
                        MetricCard(
                            title: "Symbol",
                            value: campaign.symbol,
                            footnote: "Market symbol tested.",
                            symbolName: "chart.bar.xaxis",
                            tint: FXAITheme.accentSoft
                        )
                        MetricCard(
                            title: "Best Sharpe",
                            value: campaign.bestSharpe.map { String(format: "%.4f", $0) } ?? "N/A",
                            footnote: "Best Sharpe ratio across scenarios.",
                            symbolName: "chart.line.uptrend.xyaxis",
                            tint: FXAITheme.success
                        )
                        MetricCard(
                            title: "Worst Drawdown",
                            value: campaign.worstDrawdown.map { String(format: "%.4f", $0) } ?? "N/A",
                            footnote: "Worst drawdown across scenarios.",
                            symbolName: "chart.line.downtrend.xyaxis",
                            tint: FXAITheme.warning
                        )
                    }

                    if campaign.scenarioCount > 0 {
                        passRateBar(campaign: campaign)
                    }

                    if let startedAt = campaign.startedAt {
                        Text("Started: \(FXAIFormatting.dateTimeString(for: startedAt))")
                            .font(.caption2)
                            .foregroundStyle(FXAITheme.textMuted)
                    }
                    if let completedAt = campaign.completedAt {
                        Text("Completed: \(FXAIFormatting.dateTimeString(for: completedAt))")
                            .font(.caption2)
                            .foregroundStyle(FXAITheme.textMuted)
                    }
                }
            }
        }
    }

    private func passRateBar(campaign: BacktestCampaign) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("Scenario Pass Rate")
                .font(.caption.weight(.semibold))
                .foregroundStyle(FXAITheme.textSecondary)

            GeometryReader { geo in
                ZStack(alignment: .leading) {
                    RoundedRectangle(cornerRadius: 6)
                        .fill(FXAITheme.backgroundSecondary.opacity(0.6))
                    RoundedRectangle(cornerRadius: 6)
                        .fill(campaign.passRate >= 0.7 ? FXAITheme.success : FXAITheme.warning)
                        .frame(width: geo.size.width * campaign.passRate)
                }
            }
            .frame(height: 12)

            HStack {
                Text("\(campaign.passedCount) passed")
                    .font(.caption2)
                    .foregroundStyle(FXAITheme.success)
                Spacer()
                Text("\(campaign.failedCount) failed")
                    .font(.caption2)
                    .foregroundStyle(FXAITheme.warning)
            }
        }
    }

    private func averagePassRate(_ snapshot: BacktestCampaignSnapshot) -> String {
        let rates = snapshot.campaigns.map(\.passRate)
        guard !rates.isEmpty else { return "N/A" }
        let avg = rates.reduce(0, +) / Double(rates.count)
        return "\(Int(avg * 100))%"
    }

    private func campaignStatusColor(_ status: CampaignStatus) -> Color {
        switch status {
        case .pending: return FXAITheme.textMuted
        case .running: return FXAITheme.warning
        case .completed: return FXAITheme.success
        case .failed: return .red
        case .cancelled: return FXAITheme.textMuted
        }
    }
}
