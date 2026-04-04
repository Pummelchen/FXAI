import Foundation

public enum RunBuilderCommandFactory {
    public static func auditCommand(projectRoot: URL, draft: AuditLabDraft) -> String {
        var parts: [String] = [
            "cd \(shellQuoted(projectRoot.path))",
            "python3 Tools/fxai_testlab.py run-audit"
        ]

        if draft.allPlugins {
            parts[1] += " --all-plugins"
        } else {
            parts[1] += " --plugin-list \(shellQuoted("{\(draft.pluginName)}"))"
        }

        parts[1] += " --scenario-list \(shellQuoted(draft.scenarioPreset.cliList))"
        parts[1] += " --bars \(draft.bars)"
        parts[1] += " --horizon \(draft.horizon)"
        parts[1] += " --sequence-bars \(draft.sequenceBars)"
        parts[1] += " --normalization \(shellQuoted(draft.normalization))"
        parts[1] += " --schema-id \(shellQuoted(draft.schemaID))"
        parts[1] += " --seed \(draft.seed)"
        parts[1] += " --execution-profile \(draft.executionProfile.rawValue)"

        if draft.symbolPack == .none {
            parts[1] += " --symbol \(shellQuoted(draft.symbol))"
        } else {
            parts[1] += " --symbol-pack \(draft.symbolPack.rawValue)"
        }

        return parts.joined(separator: "\n")
    }

    public static func backtestWorkflow(projectRoot: URL, draft: BacktestBuilderDraft) -> String {
        let auditDraft = AuditLabDraft(
            pluginName: draft.pluginName,
            allPlugins: false,
            scenarioPreset: draft.scenarioPreset,
            symbol: draft.symbol,
            symbolPack: .none,
            executionProfile: draft.executionProfile,
            bars: draft.bars,
            horizon: 5,
            sequenceBars: draft.sequenceBars,
            normalization: "auto",
            schemaID: "default",
            seed: 42
        )

        return [
            "cd \(shellQuoted(projectRoot.path))",
            "python3 Tools/fxai_testlab.py compile-main",
            "python3 Tools/fxai_testlab.py compile-audit",
            auditCommand(projectRoot: projectRoot, draft: auditDraft).split(separator: "\n").last.map(String.init) ?? "",
            "python3 Tools/fxai_testlab.py baseline-save --name \(shellQuoted(draft.baselineName))"
        ].joined(separator: "\n")
    }

    public static func offlineWorkflow(projectRoot: URL, draft: OfflineLabDraft) -> String {
        var commands: [String] = ["cd \(shellQuoted(projectRoot.path))"]

        if draft.includeBootstrap {
            commands.append("python3 Tools/fxai_offline_lab.py bootstrap")
        }

        switch draft.workflowPreset {
        case .smoke:
            commands.append("python3 Tools/fxai_offline_lab.py seed-demo --profile \(shellQuoted(draft.profileName)) --symbol \(shellQuoted(draft.symbol)) --runtime-mode \(shellQuoted(draft.runtimeMode))")
            commands.append("python3 Tools/fxai_offline_lab.py dashboard --profile \(shellQuoted(draft.profileName))")
        case .continuous, .promotion, .governance:
            var tune = "python3 Tools/fxai_offline_lab.py tune-zoo --profile \(shellQuoted(draft.profileName))"
            if draft.autoExport {
                tune += " --auto-export"
            }
            if draft.symbolPack == .none {
                tune += " --symbol \(shellQuoted(draft.symbol))"
            } else {
                tune += " --symbol-pack \(draft.symbolPack.rawValue)"
            }
            tune += " --months-list \(shellQuoted(draft.monthsList))"
            tune += " --top-plugins \(draft.topPlugins)"
            tune += " --limit-experiments \(draft.limitExperiments)"
            tune += " --limit-runs \(draft.limitRuns)"
            commands.append(tune)

            if draft.includeBestParams {
                if draft.symbolPack == .none {
                    commands.append("python3 Tools/fxai_offline_lab.py best-params --profile \(shellQuoted(draft.profileName)) --symbol \(shellQuoted(draft.symbol))")
                } else {
                    commands.append("python3 Tools/fxai_offline_lab.py best-params --profile \(shellQuoted(draft.profileName)) --symbol-pack \(draft.symbolPack.rawValue)")
                }
            }

            if draft.includeDeployProfiles {
                commands.append("python3 Tools/fxai_offline_lab.py deploy-profiles --profile \(shellQuoted(draft.profileName)) --runtime-mode \(shellQuoted(draft.runtimeMode))")
            }

            if draft.workflowPreset == .governance {
                commands.append("python3 Tools/fxai_offline_lab.py autonomous-governance --profile \(shellQuoted(draft.profileName))")
            }

            if draft.includeLineage {
                commands.append("python3 Tools/fxai_offline_lab.py lineage-report --profile \(shellQuoted(draft.profileName)) --symbol \(shellQuoted(draft.symbol))")
            }

            if draft.includeMinimalBundle {
                commands.append("python3 Tools/fxai_offline_lab.py minimal-bundle --profile \(shellQuoted(draft.profileName))")
            }
        }

        return commands.joined(separator: "\n")
    }

    private static func shellQuoted(_ value: String) -> String {
        "'" + value.replacingOccurrences(of: "'", with: "'\"'\"'") + "'"
    }
}
