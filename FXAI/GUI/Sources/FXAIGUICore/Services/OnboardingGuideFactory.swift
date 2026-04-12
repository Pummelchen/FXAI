import Foundation

public enum OnboardingGuideFactory {
    public static func guide(role: WorkspaceRole, projectRoot: URL) -> RoleOnboardingGuide {
        let commands = CommandFactory.recipes(projectRoot: projectRoot).filter { $0.role == role }

        switch role {
        case .liveTrader:
            return RoleOnboardingGuide(
                role: role,
                headline: "Start with the promoted truth, not raw model internals.",
                summary: "Live traders should first confirm the deployed symbol state, artifact freshness, and policy constraints before caring about plugin details.",
                steps: [
                    OnboardingStep(
                        title: "Check Live Overview first",
                        summary: "Confirm build health, runtime profiles, and current research database state before trusting any live interpretation.",
                        destination: OnboardingDestinationHint(title: "Live Overview", selection: "liveOverview")
                    ),
                    OnboardingStep(
                        title: "Inspect Runtime Monitor per symbol",
                        summary: "Verify the active deployment profile, router weights, supervisor state, and world plan loaded for the traded symbol.",
                        destination: OnboardingDestinationHint(title: "Runtime Monitor", selection: "runtimeMonitor")
                    ),
                    OnboardingStep(
                        title: "Use Incident Center for anything stale or missing",
                        summary: "If runtime artifacts are stale or incomplete, follow the guided recovery playbook before opening MT5 or forcing a deployment.",
                        destination: OnboardingDestinationHint(title: "Incident Center", selection: "incidents")
                    )
                ],
                recommendedDestinations: [
                    OnboardingDestinationHint(title: "Live Overview", selection: "liveOverview"),
                    OnboardingDestinationHint(title: "Runtime Monitor", selection: "runtimeMonitor"),
                    OnboardingDestinationHint(title: "Promotion Center", selection: "promotionCenter")
                ],
                recommendedCommands: commands
            )
        case .demoTrader:
            return RoleOnboardingGuide(
                role: role,
                headline: "Use demo mode to compare expected behavior against current behavior.",
                summary: "Demo traders benefit most from the runtime view, audit builder, and reports explorer, with less focus on Turso administration.",
                steps: [
                    OnboardingStep(
                        title: "Start in Demo Overview",
                        summary: "Use Runtime Monitor to see what the promoted deployment actually loaded before you evaluate demo behavior.",
                        destination: OnboardingDestinationHint(title: "Demo Overview", selection: "demoOverview")
                    ),
                    OnboardingStep(
                        title: "Launch a focused audit pack",
                        summary: "Run the demo candidate through a recent, walk-forward, and macro-aware audit before trusting the chart behavior.",
                        destination: OnboardingDestinationHint(title: "Audit Lab Builder", selection: "auditLab")
                    ),
                    OnboardingStep(
                        title: "Compare current artifacts and reports",
                        summary: "Use the report explorer to review the latest summaries instead of diffing raw files manually.",
                        destination: OnboardingDestinationHint(title: "Reports Explorer", selection: "reports")
                    )
                ],
                recommendedDestinations: [
                    OnboardingDestinationHint(title: "Demo Overview", selection: "demoOverview"),
                    OnboardingDestinationHint(title: "Runtime Monitor", selection: "runtimeMonitor"),
                    OnboardingDestinationHint(title: "Audit Lab Builder", selection: "auditLab"),
                    OnboardingDestinationHint(title: "Reports Explorer", selection: "reports")
                ],
                recommendedCommands: commands
            )
        case .backtester:
            return RoleOnboardingGuide(
                role: role,
                headline: "Build a realistic test flow before you open Strategy Tester.",
                summary: "Backtesters should use the GUI to generate compile, audit, and baseline workflows, then hand off to the MT5 tester with clean assumptions.",
                steps: [
                    OnboardingStep(
                        title: "Prepare builds first",
                        summary: "Compile the EA and Audit Runner before any serious Strategy Tester session.",
                        destination: OnboardingDestinationHint(title: "Backtest Builder", selection: "backtestBuilder")
                    ),
                    OnboardingStep(
                        title: "Pick scenario realism, not just speed",
                        summary: "Use audit and backtest presets that reflect execution pressure, walk-forward structure, and macro risk.",
                        destination: OnboardingDestinationHint(title: "Audit Lab Builder", selection: "auditLab")
                    ),
                    OnboardingStep(
                        title: "Save views for recurring test setups",
                        summary: "Persist your preferred scenario, symbol, and baseline combinations so repetitive test cycles stay fast and consistent."
                    )
                ],
                recommendedDestinations: [
                    OnboardingDestinationHint(title: "Backtest Builder", selection: "backtestBuilder"),
                    OnboardingDestinationHint(title: "Audit Lab Builder", selection: "auditLab"),
                    OnboardingDestinationHint(title: "Command Center", selection: "commands")
                ],
                recommendedCommands: commands
            )
        case .researcher:
            return RoleOnboardingGuide(
                role: role,
                headline: "Drive the model zoo through artifacts, not guesswork.",
                summary: "Researchers should pivot between the plugin zoo, Offline Lab workflows, promotion state, and advanced visuals instead of tuning one plugin in isolation.",
                steps: [
                    OnboardingStep(
                        title: "Start in Research Workspace",
                        summary: "Filter the model inventory, review family coverage, and pick realistic candidates before launching tuning runs.",
                        destination: OnboardingDestinationHint(title: "Research Workspace", selection: "researchWorkspace")
                    ),
                    OnboardingStep(
                        title: "Use Offline Lab Builder for campaign chains",
                        summary: "Generate tune, best-params, deploy-profile, lineage, and minimal-bundle flows without stitching commands manually.",
                        destination: OnboardingDestinationHint(title: "Offline Lab Builder", selection: "offlineLab")
                    ),
                    OnboardingStep(
                        title: "Check Promotion Center and Advanced Visuals",
                        summary: "Use promotion status, attribution, world plans, and family stress maps to understand why a candidate won or was pruned.",
                        destination: OnboardingDestinationHint(title: "Advanced Visuals", selection: "advancedVisuals")
                    )
                ],
                recommendedDestinations: [
                    OnboardingDestinationHint(title: "Research Workspace", selection: "researchWorkspace"),
                    OnboardingDestinationHint(title: "Plugin Zoo", selection: "plugins"),
                    OnboardingDestinationHint(title: "Offline Lab Builder", selection: "offlineLab"),
                    OnboardingDestinationHint(title: "Promotion Center", selection: "promotionCenter"),
                    OnboardingDestinationHint(title: "Advanced Visuals", selection: "advancedVisuals")
                ],
                recommendedCommands: commands
            )
        case .architect:
            return RoleOnboardingGuide(
                role: role,
                headline: "Operate the research OS as a platform, not a folder collection.",
                summary: "System architects should anchor on environment health, branches, incident state, recovery, and packaging, while keeping the terminal available for exact commands.",
                steps: [
                    OnboardingStep(
                        title: "Start in Platform Control",
                        summary: "Open the Research OS Control surface first and inspect backend, sync mode, encryption, branches, and audit-event freshness.",
                        destination: OnboardingDestinationHint(title: "Platform Control", selection: "platformControl")
                    ),
                    OnboardingStep(
                        title: "Use Incident Center for drift or breakage",
                        summary: "If the live tree, research state, or build targets drift, follow the generated recovery playbook instead of improvising repairs.",
                        destination: OnboardingDestinationHint(title: "Incident Center", selection: "incidents")
                    ),
                    OnboardingStep(
                        title: "Package and document the GUI as an operator product",
                        summary: "Use the release packaging flow and docs so the GUI remains a reliable operator surface, not just a dev-only prototype."
                    )
                ],
                recommendedDestinations: [
                    OnboardingDestinationHint(title: "Platform Control", selection: "platformControl"),
                    OnboardingDestinationHint(title: "Research OS Control", selection: "researchControl"),
                    OnboardingDestinationHint(title: "Incident Center", selection: "incidents"),
                    OnboardingDestinationHint(title: "Settings", selection: "settings"),
                    OnboardingDestinationHint(title: "Command Center", selection: "commands")
                ],
                recommendedCommands: commands
            )
        }
    }
}
