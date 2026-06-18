import FXGUICore
import SwiftUI

struct EvidencePackView: View {
    @EnvironmentObject private var model: FXGUIModel

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 18) {
                SectionHeader(
                    title: "Evidence Packs",
                    subtitle: "Exportable evidence bundles for strategy candidates, including scores, approvals, audit trail, and configuration."
                )

                if model.evidencePacks.isEmpty {
                    EmptyStateView(
                        title: "No evidence packs generated",
                        message: "Open the Promotion Center, select a candidate, and generate an evidence pack to export.",
                        symbolName: "doc.badge.arrow.up"
                    )
                } else {
                    summary

                    packList
                }

                if let candidate = model.selectedCandidate {
                    generateSection(candidate: candidate)
                }
            }
        }
    }

    private var summary: some View {
        LazyVGrid(
            columns: [
                GridItem(.flexible(minimum: 220), spacing: 16),
                GridItem(.flexible(minimum: 220), spacing: 16),
                GridItem(.flexible(minimum: 220), spacing: 16)
            ],
            spacing: 16
        ) {
            MetricCard(
                title: "Total Packs",
                value: "\(model.evidencePacks.count)",
                footnote: "Evidence packs generated in this session.",
                symbolName: "doc.badge.arrow.up.fill",
                tint: FXAITheme.accent
            )
            MetricCard(
                title: "Total Artifacts",
                value: "\(model.evidencePacks.reduce(0) { $0 + $1.totalArtifacts })",
                footnote: "Combined evidence artifacts across all packs.",
                symbolName: "tray.full.fill",
                tint: FXAITheme.success
            )
            MetricCard(
                title: "Candidates Available",
                value: "\(model.promotionCandidates.count)",
                footnote: "Promotion candidates eligible for evidence export.",
                symbolName: "rosette",
                tint: FXAITheme.accentSoft
            )
        }
    }

    private var packList: some View {
        ForEach(model.evidencePacks) { pack in
            FXAIVisualEffectSurface {
                VStack(alignment: .leading, spacing: 12) {
                    HStack {
                        VStack(alignment: .leading, spacing: 4) {
                            Text("\(pack.symbol) — \(pack.pluginName)")
                                .font(.headline)
                                .foregroundStyle(FXAITheme.textPrimary)
                            HStack(spacing: 8) {
                                Label(pack.stage.title, systemImage: pack.stage.symbolName)
                                    .font(.caption.weight(.semibold))
                                    .foregroundStyle(FXAITheme.accent)
                                Text("\(pack.totalArtifacts) artifacts")
                                    .font(.caption)
                                    .foregroundStyle(FXAITheme.textMuted)
                            }
                        }
                        Spacer()
                        Button {
                            exportPack(pack)
                        } label: {
                            Label("Export JSON", systemImage: "square.and.arrow.up")
                        }
                        .buttonStyle(.bordered)
                    }

                    ForEach(pack.sections) { section in
                        VStack(alignment: .leading, spacing: 6) {
                            Text(section.title)
                                .font(.subheadline.weight(.semibold))
                                .foregroundStyle(FXAITheme.textPrimary)
                            ForEach(section.artifacts) { artifact in
                                HStack(alignment: .top, spacing: 8) {
                                    Text(artifact.name)
                                        .font(.caption.weight(.semibold))
                                        .foregroundStyle(FXAITheme.textSecondary)
                                    Text(artifact.summary)
                                        .font(.caption)
                                        .foregroundStyle(FXAITheme.textMuted)
                                    Spacer()
                                }
                                .padding(.vertical, 2)
                            }
                        }
                        .padding(.vertical, 4)
                    }
                }
            }
        }
    }

    private func generateSection(candidate: PromotionCandidate) -> some View {
        FXAIVisualEffectSurface {
            VStack(alignment: .leading, spacing: 12) {
                Text("Generate New Pack")
                    .font(.headline)
                    .foregroundStyle(FXAITheme.textPrimary)
                Text("Selected: \(candidate.symbol) / \(candidate.pluginName) at \(candidate.currentStage.title)")
                    .font(.caption)
                    .foregroundStyle(FXAITheme.textSecondary)
                Button {
                    _ = model.generateEvidencePack(for: candidate)
                } label: {
                    Label("Generate Evidence Pack", systemImage: "doc.badge.arrow.up.fill")
                }
                .buttonStyle(.borderedProminent)
                .tint(FXAITheme.accent)
            }
        }
    }

    private func exportPack(_ pack: EvidencePack) {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        encoder.dateEncodingStrategy = .iso8601

        let exportData: [String: Any] = [
            "id": pack.id,
            "candidateID": pack.candidateID,
            "symbol": pack.symbol,
            "pluginName": pack.pluginName,
            "stage": pack.stage.rawValue,
            "generatedAt": pack.generatedAt.description,
            "sections": pack.sections.map { section in
                [
                    "title": section.title,
                    "artifacts": section.artifacts.map { [
                        "name": $0.name,
                        "summary": $0.summary,
                        "path": $0.path?.path ?? ""
                    ]}
                ] as [String: Any]
            }
        ]

        if let jsonData = try? JSONSerialization.data(withJSONObject: exportData, options: [.prettyPrinted, .sortedKeys]),
           let jsonString = String(data: jsonData, encoding: .utf8) {
            model.copyToPasteboard(jsonString)
        }
    }
}
