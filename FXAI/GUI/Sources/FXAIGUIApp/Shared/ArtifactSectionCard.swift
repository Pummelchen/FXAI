import FXAIGUICore
import SwiftUI

struct ArtifactSectionCard: View {
    let section: RuntimeArtifactSection
    let limit: Int
    let onReveal: ((URL) -> Void)?

    init(section: RuntimeArtifactSection, limit: Int = 12, onReveal: ((URL) -> Void)? = nil) {
        self.section = section
        self.limit = limit
        self.onReveal = onReveal
    }

    var body: some View {
        FXAIVisualEffectSurface {
            VStack(alignment: .leading, spacing: 12) {
                HStack {
                    VStack(alignment: .leading, spacing: 4) {
                        Text(section.title)
                            .font(.headline)
                            .foregroundStyle(FXAITheme.textPrimary)
                        Text("\(section.values.count) values")
                            .font(.caption)
                            .foregroundStyle(FXAITheme.textSecondary)
                    }
                    Spacer()
                    if let path = section.sourcePath, let onReveal {
                        Button("Reveal") {
                            onReveal(path)
                        }
                        .buttonStyle(.bordered)
                    }
                }

                if section.values.isEmpty {
                    Text("No values available for this artifact.")
                        .foregroundStyle(FXAITheme.textSecondary)
                } else {
                    ForEach(Array(section.values.prefix(limit))) { record in
                        HStack(alignment: .firstTextBaseline, spacing: 12) {
                            Text(record.key)
                                .font(.caption.weight(.semibold))
                                .foregroundStyle(FXAITheme.textSecondary)
                            Spacer(minLength: 12)
                            Text(record.value)
                                .font(.system(.caption, design: .monospaced))
                                .foregroundStyle(FXAITheme.textPrimary)
                                .multilineTextAlignment(.trailing)
                        }
                    }
                }
            }
        }
    }
}
