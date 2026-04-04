import FXAIGUICore
import SwiftUI

struct AuditLabBuilderView: View {
    @EnvironmentObject private var model: FXAIGUIModel

    private var availablePlugins: [String] {
        let names = model.pluginNames
        return names.isEmpty ? [model.auditDraft.pluginName] : names
    }

    private var command: String {
        guard let root = model.projectRoot else { return "" }
        return RunBuilderCommandFactory.auditCommand(projectRoot: root, draft: model.auditDraft)
    }

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 18) {
                SectionHeader(
                    title: "Audit Lab Builder",
                    subtitle: "Build exact `run-audit` commands from the real FXAI CLI surface instead of remembering flags."
                )

                FXAIVisualEffectSurface {
                    VStack(alignment: .leading, spacing: 14) {
                        Text("Preset")
                            .font(.headline)
                            .foregroundStyle(FXAITheme.textPrimary)

                        HStack(spacing: 10) {
                            presetButton(title: "Smoke", preset: .smoke)
                            presetButton(title: "Walk-Forward", preset: .walkForward)
                            presetButton(title: "Macro", preset: .macro)
                            presetButton(title: "Portfolio", preset: .portfolio)
                            presetButton(title: "Adversarial", preset: .adversarial)
                        }
                    }
                }

                FXAIVisualEffectSurface {
                    VStack(alignment: .leading, spacing: 16) {
                        Text("Configuration")
                            .font(.headline)
                            .foregroundStyle(FXAITheme.textPrimary)

                        Toggle("All Plugins", isOn: $model.auditDraft.allPlugins)

                        HStack(spacing: 14) {
                            Picker("Plugin", selection: $model.auditDraft.pluginName) {
                                ForEach(availablePlugins, id: \.self) { name in
                                    Text(name).tag(name)
                                }
                            }
                            .pickerStyle(.menu)
                            .disabled(model.auditDraft.allPlugins)

                            Picker("Execution", selection: $model.auditDraft.executionProfile) {
                                ForEach(FXAIExecutionProfile.allCases) { profile in
                                    Text(profile.title).tag(profile)
                                }
                            }
                            .pickerStyle(.menu)

                            Picker("Symbol Pack", selection: $model.auditDraft.symbolPack) {
                                ForEach(FXAISymbolPack.allCases) { pack in
                                    Text(pack.title).tag(pack)
                                }
                            }
                            .pickerStyle(.menu)
                        }

                        HStack(spacing: 14) {
                            labeledTextField("Symbol", text: $model.auditDraft.symbol)
                                .disabled(model.auditDraft.symbolPack != .none)
                            labeledTextField("Normalization", text: $model.auditDraft.normalization)
                            labeledTextField("Schema", text: $model.auditDraft.schemaID)
                        }

                        HStack(spacing: 14) {
                            labeledStepper("Bars", value: $model.auditDraft.bars, range: 500...10000, step: 100)
                            labeledStepper("Horizon", value: $model.auditDraft.horizon, range: 1...64, step: 1)
                            labeledStepper("Sequence Bars", value: $model.auditDraft.sequenceBars, range: 8...256, step: 8)
                            labeledStepper("Seed", value: $model.auditDraft.seed, range: 1...9999, step: 1)
                        }
                    }
                }

                CommandPreviewCard(
                    title: "Generated Audit Command",
                    summary: "This command matches the actual FXAI Audit Lab CLI and is ready to hand off to Terminal.",
                    command: command,
                    onCopy: { model.copyToPasteboard(command) },
                    onTerminal: { model.handoffCommandToTerminal(command) }
                )
            }
        }
    }

    private func presetButton(title: String, preset: AuditScenarioPreset) -> some View {
        Button(title) {
            model.auditDraft.scenarioPreset = preset
            switch preset {
            case .smoke:
                model.auditDraft.bars = 1000
                model.auditDraft.sequenceBars = 32
                model.auditDraft.executionProfile = .default
                model.auditDraft.symbolPack = .none
            case .walkForward:
                model.auditDraft.bars = 1500
                model.auditDraft.sequenceBars = 64
                model.auditDraft.executionProfile = .default
                model.auditDraft.symbolPack = .none
            case .macro:
                model.auditDraft.bars = 1800
                model.auditDraft.sequenceBars = 64
                model.auditDraft.executionProfile = .primeECN
                model.auditDraft.symbolPack = .none
            case .portfolio:
                model.auditDraft.bars = 2400
                model.auditDraft.sequenceBars = 96
                model.auditDraft.executionProfile = .primeECN
                model.auditDraft.symbolPack = .majors
            case .adversarial:
                model.auditDraft.bars = 2200
                model.auditDraft.sequenceBars = 96
                model.auditDraft.executionProfile = .stress
                model.auditDraft.symbolPack = .none
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
