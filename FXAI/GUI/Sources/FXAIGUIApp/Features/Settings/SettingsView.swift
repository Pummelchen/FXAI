import FXAIGUICore
import SwiftUI

struct SettingsView: View {
    @EnvironmentObject private var model: FXAIGUIModel

    var body: some View {
        VStack(alignment: .leading, spacing: 18) {
            SectionHeader(
                title: "Settings",
                subtitle: "Project-root selection, environment awareness, and documentation entry points."
            )

            FXAIVisualEffectSurface {
                VStack(alignment: .leading, spacing: 14) {
                    Text("Project Root")
                        .font(.headline)
                        .foregroundStyle(FXAITheme.textPrimary)

                    Text(model.projectPathLabel)
                        .font(.callout)
                        .foregroundStyle(FXAITheme.textSecondary)
                        .textSelection(.enabled)

                    HStack {
                        Button("Choose Project") {
                            model.chooseProjectRoot()
                        }
                        .buttonStyle(.borderedProminent)
                        .tint(FXAITheme.accent)

                        Button("Reveal in Finder") {
                            model.openProjectRootInFinder()
                        }
                        .buttonStyle(.bordered)
                    }
                }
            }

            if let snapshot = model.snapshot {
                FXAIVisualEffectSurface {
                    VStack(alignment: .leading, spacing: 14) {
                        Text("Environment")
                            .font(.headline)
                            .foregroundStyle(FXAITheme.textPrimary)

                        HStack(spacing: 18) {
                            StatusBadge(
                                title: "Database",
                                value: snapshot.tursoSummary.localDatabasePresent ? "Ready" : "Missing",
                                tint: snapshot.tursoSummary.localDatabasePresent ? FXAITheme.success : FXAITheme.warning
                            )
                            StatusBadge(
                                title: "Embedded Replica",
                                value: snapshot.tursoSummary.embeddedReplicaConfigured ? "Configured" : "Local Only",
                                tint: snapshot.tursoSummary.embeddedReplicaConfigured ? FXAITheme.accent : FXAITheme.accentSoft
                            )
                            StatusBadge(
                                title: "Encryption",
                                value: snapshot.tursoSummary.encryptionConfigured ? "On" : "Off",
                                tint: snapshot.tursoSummary.encryptionConfigured ? FXAITheme.success : FXAITheme.warning
                            )
                        }
                    }
                }
            }

            Spacer()
        }
    }
}
