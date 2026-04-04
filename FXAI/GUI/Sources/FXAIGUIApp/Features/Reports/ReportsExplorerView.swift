import FXAIGUICore
import SwiftUI

struct ReportsExplorerView: View {
    @EnvironmentObject private var model: FXAIGUIModel

    private var categories: [String] {
        guard let snapshot = model.snapshot else { return ["All"] }
        return ["All"] + snapshot.reportCategories.map(\.category)
    }

    private var filteredArtifacts: [ReportArtifact] {
        guard let snapshot = model.snapshot else { return [] }
        return snapshot.recentArtifacts.filter { artifact in
            model.reportCategoryFilter == "All" || artifact.category == model.reportCategoryFilter
        }
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 18) {
            SectionHeader(
                title: "Reports Explorer",
                subtitle: "Browse the current baseline, profile, ResearchOS, and distillation artifacts without leaving the project tree."
            )

            HStack(spacing: 12) {
                Picker("Category", selection: $model.reportCategoryFilter) {
                    ForEach(categories, id: \.self) { category in
                        Text(category).tag(category)
                    }
                }
                .pickerStyle(.segmented)

                Spacer()
            }

            if let snapshot = model.snapshot {
                HStack(spacing: 16) {
                    ForEach(snapshot.reportCategories) { category in
                        MetricCard(
                            title: category.category,
                            value: "\(category.fileCount)",
                            footnote: category.latestModifiedAt.map { "Latest: \(FXAIFormatting.relativeDateString(for: $0))" } ?? "No timestamps available",
                            symbolName: "doc.on.doc.fill",
                            tint: FXAITheme.accentSoft
                        )
                    }
                }
            }

            FXAIVisualEffectSurface {
                VStack(alignment: .leading, spacing: 12) {
                    Text("Recent Files")
                        .font(.headline)
                        .foregroundStyle(FXAITheme.textPrimary)

                    if filteredArtifacts.isEmpty {
                        Text("No report artifacts match the current category.")
                            .foregroundStyle(FXAITheme.textSecondary)
                    } else {
                        ForEach(filteredArtifacts) { artifact in
                            HStack(alignment: .top, spacing: 12) {
                                VStack(alignment: .leading, spacing: 4) {
                                    Text(artifact.name)
                                        .font(.subheadline.weight(.semibold))
                                        .foregroundStyle(FXAITheme.textPrimary)
                                    Text(artifact.path.path)
                                        .font(.caption)
                                        .foregroundStyle(FXAITheme.textMuted)
                                        .lineLimit(1)
                                }
                                Spacer()
                                VStack(alignment: .trailing, spacing: 6) {
                                    Text(artifact.category)
                                        .font(.caption.weight(.semibold))
                                        .foregroundStyle(FXAITheme.accent)
                                    Button("Reveal") {
                                        model.openInFinder(artifact.path)
                                    }
                                    .buttonStyle(.plain)
                                    .foregroundStyle(FXAITheme.accentSoft)
                                }
                            }
                            .padding(.vertical, 6)
                        }
                    }
                }
            }
        }
    }
}
