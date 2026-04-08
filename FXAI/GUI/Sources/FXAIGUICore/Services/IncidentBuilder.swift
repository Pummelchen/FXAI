import Foundation

public struct IncidentBuilder {
    public init() {}

    public func build(
        projectRoot: URL,
        snapshot: FXAIProjectSnapshot?,
        runtimeSnapshot: RuntimeOperationsSnapshot?,
        researchSnapshot: ResearchOSControlSnapshot?,
        newsPulseSnapshot: NewsPulseSnapshot?
    ) -> IncidentCenterSnapshot {
        var incidents: [FXAIIncident] = []
        let effectiveProfile = runtimeSnapshot?.profileName ?? researchSnapshot?.profileName ?? "continuous"

        if let snapshot {
            let missingTargets = snapshot.buildTargets.filter { !$0.exists }
            if !missingTargets.isEmpty {
                let targetList = missingTargets.map(\.name).sorted()
                incidents.append(
                    FXAIIncident(
                        id: "build::missing-targets",
                        severity: .critical,
                        category: .build,
                        title: "Build targets are missing",
                        summary: "One or more MT5 outputs are absent. Live inspection and tester workflows should not be trusted until the build surface is clean again.",
                        detailLines: targetList.map { "Missing build target: \($0)" },
                        actions: [
                            IncidentAction(
                                title: "Run full verification",
                                summary: "Compile, pytest, and deterministic checks through the standard FXAI validation path.",
                                command: verifyAllCommand(projectRoot: projectRoot),
                                destinationSelection: "commands"
                            ),
                            IncidentAction(
                                title: "Compile MT5 targets",
                                summary: "Rebuild the main EA, audit runner, and export runner directly.",
                                command: compileTargetsCommand(projectRoot: projectRoot),
                                destinationSelection: "backtestBuilder"
                            )
                        ],
                        playbook: RecoveryPlaybook(
                            title: "Recover the build surface",
                            summary: "Bring the MT5 outputs back to a known-good state before you inspect or promote anything else.",
                            steps: [
                                RecoveryStep(
                                    title: "Run verify-all",
                                    summary: "Confirm whether the failure is limited to MT5 targets or broader Python and artifact validation.",
                                    command: verifyAllCommand(projectRoot: projectRoot),
                                    destinationSelection: "commands"
                                ),
                                RecoveryStep(
                                    title: "Compile all MT5 targets",
                                    summary: "Rebuild the EA, Audit Runner, and Offline Export Runner.",
                                    command: compileTargetsCommand(projectRoot: projectRoot),
                                    destinationSelection: "backtestBuilder"
                                )
                            ]
                        )
                    )
                )
            }

            if snapshot.runtimeProfiles.isEmpty,
               runtimeSnapshot?.deployments.isEmpty != true {
                incidents.append(
                    FXAIIncident(
                        id: "promotion::no-runtime-profiles",
                        severity: .warning,
                        category: .promotion,
                        title: "No runtime profiles are available",
                        summary: "The GUI cannot show a promoted live surface because no deployment profiles were discovered in the current ResearchOS outputs.",
                        actions: [
                            IncidentAction(
                                title: "Deploy profiles",
                                summary: "Emit the promoted deployment payloads for MT5 runtime consumption.",
                                command: deployProfilesCommand(projectRoot: projectRoot, profileName: effectiveProfile, runtimeMode: "research"),
                                destinationSelection: "offlineLab"
                            ),
                            IncidentAction(
                                title: "Run autonomous governance",
                                summary: "Refresh champions, lineage, profiles, and the operator dashboard together.",
                                command: autonomousGovernanceCommand(projectRoot: projectRoot, profileName: effectiveProfile),
                                destinationSelection: "researchControl"
                            )
                        ],
                        playbook: RecoveryPlaybook(
                            title: "Restore live deployment visibility",
                            summary: "Regenerate promoted payloads and the operator dashboard so runtime state becomes inspectable again.",
                            steps: [
                                RecoveryStep(
                                    title: "Run autonomous governance",
                                    summary: "Refresh the research profile and operator dashboard first.",
                                    command: autonomousGovernanceCommand(projectRoot: projectRoot, profileName: effectiveProfile),
                                    destinationSelection: "researchControl"
                                ),
                                RecoveryStep(
                                    title: "Emit deployment profiles",
                                    summary: "Rebuild the runtime deployment payloads and MT5 bundle artifacts.",
                                    command: deployProfilesCommand(projectRoot: projectRoot, profileName: effectiveProfile, runtimeMode: "research"),
                                    destinationSelection: "offlineLab"
                                )
                            ]
                        )
                    )
                )
            }
        }

        if let researchSnapshot {
            if let environment = researchSnapshot.environment, let configError = nonEmpty(environment.configError) {
                incidents.append(
                    FXAIIncident(
                        id: "research-os::config-error",
                        severity: .critical,
                        category: .researchOS,
                        title: "Research OS environment is misconfigured",
                        summary: "The Turso-backed environment reported a configuration error. Governance and recovery flows may generate incomplete or misleading artifacts.",
                        detailLines: [configError],
                        actions: [
                            IncidentAction(
                                title: "Run diagnostics",
                                summary: "Validate the environment and rebuild the operator dashboard.",
                                command: ResearchOSCommandFactory.environmentDiagnostics(projectRoot: projectRoot, profileName: effectiveProfile),
                                destinationSelection: "researchControl"
                            ),
                            IncidentAction(
                                title: "Bootstrap the research OS",
                                summary: "Re-check the environment and seed the expected local surfaces.",
                                command: bootstrapCommand(projectRoot: projectRoot),
                                destinationSelection: "commands"
                            )
                        ],
                        playbook: RecoveryPlaybook(
                            title: "Repair the Research OS environment",
                            summary: "Fix the Turso or local libSQL configuration first so every later workflow has a stable source of truth.",
                            steps: [
                                RecoveryStep(
                                    title: "Validate environment",
                                    summary: "Re-run the environment checks and refresh the dashboard.",
                                    command: ResearchOSCommandFactory.environmentDiagnostics(projectRoot: projectRoot, profileName: effectiveProfile),
                                    destinationSelection: "researchControl"
                                ),
                                RecoveryStep(
                                    title: "Bootstrap local surfaces",
                                    summary: "Seed or repair the expected environment files and example artifacts.",
                                    command: bootstrapCommand(projectRoot: projectRoot),
                                    destinationSelection: "commands"
                                )
                            ]
                        )
                    )
                )
            }
        } else {
            incidents.append(
                FXAIIncident(
                    id: "research-os::missing-dashboard",
                    severity: .warning,
                    category: .researchOS,
                    title: "No Research OS dashboard was found",
                    summary: "Branch state, audit events, and vector surfaces are unavailable until the operator dashboard is rebuilt.",
                    actions: [
                        IncidentAction(
                            title: "Rebuild operator dashboard",
                            summary: "Generate the ResearchOS dashboard from the current profile.",
                            command: dashboardCommand(projectRoot: projectRoot, profileName: effectiveProfile),
                            destinationSelection: "researchControl"
                        )
                    ],
                    playbook: RecoveryPlaybook(
                        title: "Rebuild Research OS visibility",
                        summary: "Generate the operator dashboard again so the GUI can inspect the research platform state.",
                        steps: [
                            RecoveryStep(
                                title: "Run environment diagnostics",
                                summary: "Validate the backend and refresh the dashboard in one pass.",
                                command: ResearchOSCommandFactory.environmentDiagnostics(projectRoot: projectRoot, profileName: effectiveProfile),
                                destinationSelection: "researchControl"
                            ),
                            RecoveryStep(
                                title: "Run dashboard generation",
                                summary: "Rebuild the operator summary and supporting artifacts for the current profile.",
                                command: dashboardCommand(projectRoot: projectRoot, profileName: effectiveProfile),
                                destinationSelection: "researchControl"
                            )
                        ]
                    )
                )
            )
        }

        if let runtimeSnapshot {
            if runtimeSnapshot.deployments.isEmpty {
                incidents.append(
                    FXAIIncident(
                        id: "runtime::missing-deployments",
                        severity: .warning,
                        category: .runtime,
                        title: "No runtime deployments are inspectable",
                        summary: "The GUI found no current live deployments. Runtime and promotion views will stay empty until profiles and artifacts are regenerated.",
                        actions: [
                            IncidentAction(
                                title: "Deploy profiles",
                                summary: "Regenerate the MT5 deployment payloads for the active profile.",
                                command: deployProfilesCommand(projectRoot: projectRoot, profileName: effectiveProfile, runtimeMode: "research"),
                                destinationSelection: "offlineLab"
                            )
                        ],
                        playbook: RecoveryPlaybook(
                            title: "Restore runtime artifacts",
                            summary: "Recreate the promoted deployment payloads so runtime monitoring becomes available again.",
                            steps: [
                                RecoveryStep(
                                    title: "Run autonomous governance",
                                    summary: "Refresh champion state and supporting dashboard artifacts.",
                                    command: autonomousGovernanceCommand(projectRoot: projectRoot, profileName: effectiveProfile),
                                    destinationSelection: "researchControl"
                                ),
                                RecoveryStep(
                                    title: "Deploy profiles",
                                    summary: "Emit the live runtime payloads and MT5 promotions.",
                                    command: deployProfilesCommand(projectRoot: projectRoot, profileName: effectiveProfile, runtimeMode: "research"),
                                    destinationSelection: "offlineLab"
                                )
                            ]
                        )
                    )
                )
            }

            for deployment in runtimeSnapshot.deployments {
                let health = deployment.artifactHealth
                let missingFlags = [
                    ("deployment profile", health.missingDeployment),
                    ("student router", health.missingRouter),
                    ("supervisor service", health.missingSupervisorService),
                    ("supervisor command", health.missingSupervisorCommand),
                    ("world plan", health.missingWorldPlan)
                ]
                let missingNames = missingFlags.compactMap { $0.1 ? $0.0 : nil }

                if !missingNames.isEmpty || health.staleArtifact {
                    let severity: IncidentSeverity = missingNames.isEmpty ? .warning : .critical
                    var details = missingNames.map { "Missing \($0)" }
                    if health.staleArtifact {
                        details.append("Artifact age is \(health.artifactAgeSeconds)s and is marked stale")
                    }

                    incidents.append(
                        FXAIIncident(
                            id: "runtime::\(deployment.symbol)::artifacts",
                            severity: severity,
                            category: .runtime,
                            title: "Runtime artifacts need recovery for \(deployment.symbol)",
                            summary: "The promoted runtime surface for \(deployment.symbol) is incomplete or stale. Live policy, routing, or world-state decisions should not be trusted until the artifacts are refreshed.",
                            affectedSymbol: deployment.symbol,
                            detailLines: details,
                            actions: [
                                IncidentAction(
                                    title: "Run recovery bundle",
                                    summary: "Regenerate runtime artifacts, lineage, and minimal bundle outputs for this profile.",
                                    command: ResearchOSCommandFactory.recoveryCommand(
                                        projectRoot: projectRoot,
                                        draft: ResearchOSRecoveryDraft(profileName: deployment.profileName, runtimeMode: deployment.runtimeMode)
                                    ),
                                    destinationSelection: "incidents"
                                ),
                                IncidentAction(
                                    title: "Deploy profiles",
                                    summary: "Re-emit promoted deployment payloads for the current profile.",
                                    command: deployProfilesCommand(projectRoot: projectRoot, profileName: deployment.profileName, runtimeMode: deployment.runtimeMode),
                                    destinationSelection: "offlineLab"
                                )
                            ],
                            playbook: RecoveryPlaybook(
                                title: "Repair \(deployment.symbol) runtime state",
                                summary: "Rebuild the runtime payloads in a controlled order so MT5 and the GUI agree on the active deployment state again.",
                                steps: [
                                    RecoveryStep(
                                        title: "Run recovery bundle",
                                        summary: "Repair generated runtime artifacts, lineage, and the minimal live bundle.",
                                        command: ResearchOSCommandFactory.recoveryCommand(
                                            projectRoot: projectRoot,
                                            draft: ResearchOSRecoveryDraft(profileName: deployment.profileName, runtimeMode: deployment.runtimeMode)
                                        ),
                                        destinationSelection: "incidents"
                                    ),
                                    RecoveryStep(
                                        title: "Deploy runtime profiles again",
                                        summary: "Re-emit deployment payloads for the affected profile and runtime mode.",
                                        command: deployProfilesCommand(projectRoot: projectRoot, profileName: deployment.profileName, runtimeMode: deployment.runtimeMode),
                                        destinationSelection: "offlineLab"
                                    ),
                                    RecoveryStep(
                                        title: "Verify the full platform",
                                        summary: "Run compile, deterministic, pytest, and MT5 verification before trusting the repaired state.",
                                        command: verifyAllCommand(projectRoot: projectRoot),
                                        destinationSelection: "commands"
                                    )
                                ]
                            )
                        )
                    )
                }

                if !health.performanceFailures.isEmpty || !health.artifactSizeFailures.isEmpty {
                    let detailLines = health.performanceFailures.map { "Performance gate: \($0)" }
                        + health.artifactSizeFailures.map { "Artifact size gate: \($0)" }

                    incidents.append(
                        FXAIIncident(
                            id: "performance::\(deployment.symbol)",
                            severity: .warning,
                            category: .performance,
                            title: "Runtime performance gates failed for \(deployment.symbol)",
                            summary: "The promoted profile for \(deployment.symbol) violates runtime or artifact-size expectations. Keep it out of production until the budgets are back inside limits.",
                            affectedSymbol: deployment.symbol,
                            detailLines: detailLines,
                            actions: [
                                IncidentAction(
                                    title: "Inspect runtime monitor",
                                    summary: "Review deployment metrics, router state, and world-plan highlights for the symbol.",
                                    command: verifyAllCommand(projectRoot: projectRoot),
                                    destinationSelection: "runtimeMonitor"
                                ),
                                IncidentAction(
                                    title: "Regenerate promotion state",
                                    summary: "Refresh governance, profiles, and sizing outputs under the current profile.",
                                    command: autonomousGovernanceCommand(projectRoot: projectRoot, profileName: deployment.profileName),
                                    destinationSelection: "researchControl"
                                )
                            ],
                            playbook: RecoveryPlaybook(
                                title: "Bring \(deployment.symbol) back inside runtime budget",
                                summary: "Use the operator surfaces to inspect the failing symbol, then regenerate profiles under the current governance state.",
                                steps: [
                                    RecoveryStep(
                                        title: "Review symbol runtime state",
                                        summary: "Inspect which metrics and artifacts are pushing the profile outside runtime budget.",
                                        command: verifyAllCommand(projectRoot: projectRoot),
                                        destinationSelection: "runtimeMonitor"
                                    ),
                                    RecoveryStep(
                                        title: "Refresh governance outputs",
                                        summary: "Regenerate champions, profiles, and budgets from the current research state.",
                                        command: autonomousGovernanceCommand(projectRoot: projectRoot, profileName: deployment.profileName),
                                        destinationSelection: "researchControl"
                                    )
                                ]
                            )
                        )
                    )
                }
            }
        }

        if let newsPulseSnapshot {
            let staleSources = newsPulseSnapshot.sourceStatuses.filter { !$0.ok || $0.stale }
            if !staleSources.isEmpty {
                incidents.append(
                    FXAIIncident(
                        id: "newspulse::stale-sources",
                        severity: .warning,
                        category: .runtime,
                        title: "NewsPulse source freshness is degraded",
                        summary: "The shared NewsPulse surface is stale or failing. Runtime gates should treat missing news context as unknown until the collector path is healthy again.",
                        detailLines: staleSources.map { "\($0.id.uppercased()): \($0.lastError ?? "stale or missing updates")" },
                        actions: [
                            IncidentAction(
                                title: "Open NewsPulse",
                                summary: "Inspect current source state, pair risk, and recent tape in the operator shell.",
                                command: newspulseOnceCommand(projectRoot: projectRoot),
                                destinationSelection: "newsPulse"
                            ),
                            IncidentAction(
                                title: "Run NewsPulse refresh",
                                summary: "Force one local fusion cycle from the current calendar export and GDELT state.",
                                command: newspulseOnceCommand(projectRoot: projectRoot),
                                destinationSelection: "commands"
                            )
                        ],
                        playbook: RecoveryPlaybook(
                            title: "Recover NewsPulse freshness",
                            summary: "Restore the calendar and GDELT collectors before you trust any news-aware runtime gate again.",
                            steps: [
                                RecoveryStep(
                                    title: "Validate NewsPulse config",
                                    summary: "Check whitelist, query scaffolding, and local artifact paths first.",
                                    command: newspulseValidateCommand(projectRoot: projectRoot),
                                    destinationSelection: "newsPulse"
                                ),
                                RecoveryStep(
                                    title: "Refresh one fusion cycle",
                                    summary: "Generate a fresh merged snapshot from the latest collector state.",
                                    command: newspulseOnceCommand(projectRoot: projectRoot),
                                    destinationSelection: "commands"
                                )
                            ]
                        )
                    )
                )
            }

            let runtimePairs = Set((runtimeSnapshot?.deployments ?? []).compactMap { activePairID(for: $0.symbol) })
            let activeBlockedPairs = newsPulseSnapshot.pairs.filter {
                runtimePairs.contains($0.pair) && ($0.tradeGate.uppercased() == "BLOCK" || $0.tradeGate.uppercased() == "CAUTION")
            }
            for pair in activeBlockedPairs.prefix(4) {
                let severity: IncidentSeverity = pair.tradeGate.uppercased() == "BLOCK" ? .critical : .warning
                incidents.append(
                    FXAIIncident(
                        id: "newspulse::\(pair.pair)",
                        severity: severity,
                        category: .runtime,
                        title: "NewsPulse gate is active for \(pair.pair)",
                        summary: "The shared NewsPulse subsystem is currently flagging \(pair.pair) as \(pair.tradeGate). Treat new entries as risk-sensitive until the scheduled or breaking-news window clears.",
                        affectedSymbol: pair.pair,
                        detailLines: pair.reasons + [pair.eventETAMin.map { "Next event ETA: \($0) minutes" } ?? "No scheduled ETA available"],
                        actions: [
                            IncidentAction(
                                title: "Inspect NewsPulse",
                                summary: "Open pair-level reasons, currency heatmap, and recent tape for the active gate.",
                                command: newspulseOnceCommand(projectRoot: projectRoot),
                                destinationSelection: "newsPulse"
                            )
                        ],
                        playbook: RecoveryPlaybook(
                            title: "Work through the active NewsPulse gate",
                            summary: "Use the shared news surface to decide whether to wait, reduce risk, or keep the pair blocked through the event window.",
                            steps: [
                                RecoveryStep(
                                    title: "Inspect the pair gate",
                                    summary: "Review pair reasons, source health, and current event timing.",
                                    command: newspulseOnceCommand(projectRoot: projectRoot),
                                    destinationSelection: "newsPulse"
                                ),
                                RecoveryStep(
                                    title: "Refresh the snapshot",
                                    summary: "Re-run one fusion cycle to confirm whether the gate is still active.",
                                    command: newspulseOnceCommand(projectRoot: projectRoot),
                                    destinationSelection: "commands"
                                )
                            ]
                        )
                    )
                )
            }
        }

        let sorted = incidents.sorted { lhs, rhs in
            if lhs.severity == rhs.severity {
                return lhs.title < rhs.title
            }
            return lhs.severity > rhs.severity
        }

        return IncidentCenterSnapshot(incidents: sorted)
    }

    private func verifyAllCommand(projectRoot: URL) -> String {
        [
            "cd \(shellQuoted(projectRoot.path))",
            "python3 Tools/fxai_testlab.py verify-all"
        ].joined(separator: "\n")
    }

    private func compileTargetsCommand(projectRoot: URL) -> String {
        [
            "cd \(shellQuoted(projectRoot.path))",
            "python3 Tools/fxai_testlab.py compile-main",
            "python3 Tools/fxai_testlab.py compile-audit",
            "python3 Tools/fxai_testlab.py compile-export"
        ].joined(separator: "\n")
    }

    private func deployProfilesCommand(projectRoot: URL, profileName: String, runtimeMode: String) -> String {
        [
            "cd \(shellQuoted(projectRoot.path))",
            "python3 Tools/fxai_offline_lab.py deploy-profiles --profile \(shellQuoted(profileName)) --runtime-mode \(shellQuoted(runtimeMode))"
        ].joined(separator: "\n")
    }

    private func autonomousGovernanceCommand(projectRoot: URL, profileName: String) -> String {
        [
            "cd \(shellQuoted(projectRoot.path))",
            "python3 Tools/fxai_offline_lab.py autonomous-governance --profile \(shellQuoted(profileName))",
            "python3 Tools/fxai_offline_lab.py dashboard --profile \(shellQuoted(profileName))"
        ].joined(separator: "\n")
    }

    private func dashboardCommand(projectRoot: URL, profileName: String) -> String {
        [
            "cd \(shellQuoted(projectRoot.path))",
            "python3 Tools/fxai_offline_lab.py dashboard --profile \(shellQuoted(profileName))"
        ].joined(separator: "\n")
    }

    private func bootstrapCommand(projectRoot: URL) -> String {
        [
            "cd \(shellQuoted(projectRoot.path))",
            "python3 Tools/fxai_offline_lab.py bootstrap --seed-demo"
        ].joined(separator: "\n")
    }

    private func newspulseValidateCommand(projectRoot: URL) -> String {
        [
            "cd \(shellQuoted(projectRoot.path))",
            "python3 Tools/fxai_offline_lab.py newspulse-validate"
        ].joined(separator: "\n")
    }

    private func newspulseOnceCommand(projectRoot: URL) -> String {
        [
            "cd \(shellQuoted(projectRoot.path))",
            "python3 Tools/fxai_offline_lab.py newspulse-once"
        ].joined(separator: "\n")
    }

    private func activePairID(for symbol: String) -> String? {
        let upper = symbol.uppercased()
        guard upper.count >= 6 else { return nil }
        let start = upper.startIndex
        let end = upper.index(start, offsetBy: 6)
        return String(upper[start..<end])
    }

    private func nonEmpty(_ value: String?) -> String? {
        guard let value else { return nil }
        let trimmed = value.trimmingCharacters(in: .whitespacesAndNewlines)
        return trimmed.isEmpty ? nil : trimmed
    }

    private func shellQuoted(_ value: String) -> String {
        "'" + value.replacingOccurrences(of: "'", with: "'\"'\"'") + "'"
    }
}
