import FXAIGUICore
import SwiftUI

struct OfflineLabBuilderView: View {
    @EnvironmentObject private var model: FXAIGUIModel

    private var command: String {
        guard let root = model.projectRoot else { return "" }
        return RunBuilderCommandFactory.offlineWorkflow(projectRoot: root, draft: model.offlineDraft)
    }

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 18) {
                SectionHeader(
                    title: "Offline Lab Builder",
                    subtitle: "Build practical research OS flows for tuning, promotion, deployment, lineage, and governance without remembering long CLI chains."
                )

                FXAIVisualEffectSurface {
                    VStack(alignment: .leading, spacing: 14) {
                        Text("Workflow Preset")
                            .font(.headline)
                            .foregroundStyle(FXAITheme.textPrimary)

                        HStack(spacing: 10) {
                            workflowButton(.smoke)
                            workflowButton(.continuous)
                            workflowButton(.promotion)
                            workflowButton(.governance)
                        }
                    }
                }

                FXAIVisualEffectSurface {
                    VStack(alignment: .leading, spacing: 16) {
                        Text("Configuration")
                            .font(.headline)
                            .foregroundStyle(FXAITheme.textPrimary)

                        HStack(spacing: 14) {
                            labeledTextField("Profile", text: $model.offlineDraft.profileName)
                            labeledTextField("Symbol", text: $model.offlineDraft.symbol)
                            labeledTextField("Months", text: $model.offlineDraft.monthsList)
                        }

                        HStack(spacing: 14) {
                            Picker("Symbol Pack", selection: $model.offlineDraft.symbolPack) {
                                ForEach(FXAISymbolPack.allCases) { pack in
                                    Text(pack.title).tag(pack)
                                }
                            }
                            .pickerStyle(.menu)

                            Picker("Runtime Mode", selection: $model.offlineDraft.runtimeMode) {
                                Text("Research").tag("research")
                                Text("Production").tag("production")
                            }
                            .pickerStyle(.menu)
                        }

                        HStack(spacing: 14) {
                            labeledStepper("Top Plugins", value: $model.offlineDraft.topPlugins, range: 1...32, step: 1)
                            labeledStepper("Experiments", value: $model.offlineDraft.limitExperiments, range: 1...128, step: 1)
                            labeledStepper("Runs", value: $model.offlineDraft.limitRuns, range: 1...512, step: 1)
                        }

                        VStack(alignment: .leading, spacing: 10) {
                            Text("Outputs")
                                .font(.subheadline.weight(.semibold))
                                .foregroundStyle(FXAITheme.textPrimary)
                            Toggle("Bootstrap first", isOn: $model.offlineDraft.includeBootstrap)
                            Toggle("Auto export exact windows", isOn: $model.offlineDraft.autoExport)
                            Toggle("Promote best parameters", isOn: $model.offlineDraft.includeBestParams)
                            Toggle("Emit deployment profiles", isOn: $model.offlineDraft.includeDeployProfiles)
                            Toggle("Render lineage report", isOn: $model.offlineDraft.includeLineage)
                            Toggle("Build minimal live bundle", isOn: $model.offlineDraft.includeMinimalBundle)
                        }
                    }
                }

                CommandPreviewCard(
                    title: "Generated Offline Lab Workflow",
                    summary: "This is the practical command chain the GUI recommends for the current research OS preset.",
                    command: command,
                    onCopy: { model.copyToPasteboard(command) },
                    onTerminal: { model.handoffCommandToTerminal(command) }
                )
            }
        }
    }

    private func workflowButton(_ preset: OfflineWorkflowPreset) -> some View {
        Button(preset.title) {
            model.offlineDraft.workflowPreset = preset
            switch preset {
            case .smoke:
                model.offlineDraft.profileName = "smoke"
                model.offlineDraft.symbolPack = .none
                model.offlineDraft.includeBootstrap = true
                model.offlineDraft.includeBestParams = false
                model.offlineDraft.includeDeployProfiles = false
                model.offlineDraft.includeLineage = false
                model.offlineDraft.includeMinimalBundle = false
            case .continuous:
                model.offlineDraft.profileName = "continuous"
                model.offlineDraft.symbolPack = .majors
                model.offlineDraft.includeBootstrap = true
                model.offlineDraft.includeBestParams = true
                model.offlineDraft.includeDeployProfiles = true
                model.offlineDraft.includeLineage = true
                model.offlineDraft.includeMinimalBundle = false
            case .promotion:
                model.offlineDraft.profileName = "bestparams"
                model.offlineDraft.symbolPack = .majors
                model.offlineDraft.includeBootstrap = false
                model.offlineDraft.includeBestParams = true
                model.offlineDraft.includeDeployProfiles = true
                model.offlineDraft.includeLineage = true
                model.offlineDraft.includeMinimalBundle = true
            case .governance:
                model.offlineDraft.profileName = "continuous"
                model.offlineDraft.symbolPack = .majors
                model.offlineDraft.includeBootstrap = false
                model.offlineDraft.includeBestParams = true
                model.offlineDraft.includeDeployProfiles = true
                model.offlineDraft.includeLineage = true
                model.offlineDraft.includeMinimalBundle = true
            }
        }
        .buttonStyle(.bordered)
    }

    private func labeledTextField(_ title: String, text: Binding<String>) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            Text(title)
                .font(.caption.weight(.semibold))
                .foregroundStyle(FXAITheme.textSecondary)
            TextField(title, text: text)
                .textFieldStyle(.roundedBorder)
        }
    }

    private func labeledStepper(_ title: String, value: Binding<Int>, range: ClosedRange<Int>, step: Int) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            Text(title)
                .font(.caption.weight(.semibold))
                .foregroundStyle(FXAITheme.textSecondary)
            Stepper("\(value.wrappedValue)", value: value, in: range, step: step)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }
}
