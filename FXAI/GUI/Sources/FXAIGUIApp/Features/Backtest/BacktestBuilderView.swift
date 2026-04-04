import FXAIGUICore
import SwiftUI

struct BacktestBuilderView: View {
    @EnvironmentObject private var model: FXAIGUIModel

    private var availablePlugins: [String] {
        let names = model.pluginNames
        return names.isEmpty ? [model.backtestDraft.pluginName] : names
    }

    private var command: String {
        guard let root = model.projectRoot else { return "" }
        return RunBuilderCommandFactory.backtestWorkflow(projectRoot: root, draft: model.backtestDraft)
    }

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 18) {
                SectionHeader(
                    title: "Backtest Builder",
                    subtitle: "Prepare the realistic FXAI workflow that exists today: compile, audit, baseline, then hand off into MT5 Strategy Tester."
                )

                FXAIVisualEffectSurface {
                    VStack(alignment: .leading, spacing: 16) {
                        Text("Backtest Workflow")
                            .font(.headline)
                            .foregroundStyle(FXAITheme.textPrimary)

                        HStack(spacing: 14) {
                            Picker("Plugin", selection: $model.backtestDraft.pluginName) {
                                ForEach(availablePlugins, id: \.self) { name in
                                    Text(name).tag(name)
                                }
                            }
                            .pickerStyle(.menu)

                            Picker("Scenario Pack", selection: $model.backtestDraft.scenarioPreset) {
                                ForEach(AuditScenarioPreset.allCases) { preset in
                                    Text(preset.title).tag(preset)
                                }
                            }
                            .pickerStyle(.menu)

                            Picker("Execution", selection: $model.backtestDraft.executionProfile) {
                                ForEach(FXAIExecutionProfile.allCases) { profile in
                                    Text(profile.title).tag(profile)
                                }
                            }
                            .pickerStyle(.menu)
                        }

                        HStack(spacing: 14) {
                            labeledTextField("Symbol", text: $model.backtestDraft.symbol)
                            labeledTextField("Baseline Name", text: $model.backtestDraft.baselineName)
                        }

                        HStack(spacing: 14) {
                            labeledStepper("Bars", value: $model.backtestDraft.bars, range: 500...10000, step: 100)
                            labeledStepper("Sequence Bars", value: $model.backtestDraft.sequenceBars, range: 8...256, step: 8)
                        }
                    }
                }

                CommandPreviewCard(
                    title: "Generated Backtest Prep Workflow",
                    summary: "Copy this into Terminal first, then move into MT5 Strategy Tester with the same plugin and symbol configuration.",
                    command: command,
                    onCopy: { model.copyToPasteboard(command) },
                    onTerminal: { model.handoffCommandToTerminal(command) }
                )

                FXAIVisualEffectSurface {
                    VStack(alignment: .leading, spacing: 10) {
                        Text("Manual Strategy Tester Handoff")
                            .font(.headline)
                            .foregroundStyle(FXAITheme.textPrimary)
                        Label("After the command workflow completes, open MT5 Strategy Tester and run the same symbol, plugin, and `.set` surface you just certified.", systemImage: "arrowshape.turn.up.right.fill")
                            .foregroundStyle(FXAITheme.textSecondary)
                        Label("Use the audit result and saved baseline to decide whether the tester run is worth expanding.", systemImage: "checkmark.shield.fill")
                            .foregroundStyle(FXAITheme.textSecondary)
                    }
                }
            }
        }
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
