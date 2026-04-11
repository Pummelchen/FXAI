import Foundation
import Testing
@testable import FXAIGUICore

struct Phase6WorkflowTests {
    @Test
    func savedWorkspaceStoreRoundTripsViewsAndOnboardingState() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        let storageURL = root.appendingPathComponent("workspace_state.json")
        let store = SavedWorkspaceStore(storageURL: storageURL)

        let savedView = SavedWorkspaceView(
            name: "Live EURUSD",
            projectRootPath: "/tmp/fxai",
            selection: "runtimeMonitor",
            selectedRole: .liveTrader,
            selectedRuntimeSymbol: "EURUSD",
            selectedRatesSymbol: "EURUSD",
            selectedCrossAssetSymbol: "EURUSD",
            selectedMicrostructureSymbol: "EURUSD",
            selectedAdaptiveSymbol: "EURUSD",
            selectedDynamicEnsembleSymbol: "EURUSD",
            selectedProbCalibrationSymbol: "EURUSD",
            selectedExecutionQualitySymbol: "EURUSD",
            selectedLabelEngineDatasetKey: "continuous:EURUSD:m1:labels",
            selectedResearchSymbol: "EURUSD",
            selectedVisualizationSymbol: "EURUSD",
            pluginSearchText: "mlp",
            selectedPluginFamily: "Sequence",
            reportCategoryFilter: "ResearchOS",
            auditDraft: AuditLabDraft(pluginName: "ai_mlp"),
            backtestDraft: BacktestBuilderDraft(pluginName: "ai_mlp"),
            offlineDraft: OfflineLabDraft(profileName: "continuous"),
            researchBranchDraft: ResearchOSBranchDraft(profileName: "continuous"),
            researchAuditDraft: ResearchOSAuditDraft(limit: 50, pages: 2),
            researchVectorDraft: ResearchOSVectorDraft(profileName: "continuous", symbol: "EURUSD", limit: 6),
            researchRecoveryDraft: ResearchOSRecoveryDraft(profileName: "continuous", runtimeMode: "production"),
            overviewLayout: .default()
        )

        let state = FXAIGUIPersistenceState(
            savedViews: [savedView],
            lastWorkspace: savedView,
            completedOnboardingRoles: [.liveTrader, .architect],
            preferredProjectRootPath: "/tmp/fxai",
            autoReconnectEnabled: false
        )

        try store.save(state)
        let loaded = store.load()

        #expect(loaded.savedViews.count == 1)
        #expect(loaded.savedViews.first?.name == "Live EURUSD")
        #expect(loaded.lastWorkspace?.selection == "runtimeMonitor")
        #expect(loaded.savedViews.first?.selectedProbCalibrationSymbol == "EURUSD")
        #expect(loaded.savedViews.first?.overviewLayout.sections.count == OverviewDashboardSectionKind.allCases.count)
        #expect(Set(loaded.completedOnboardingRoles) == Set([.liveTrader, .architect]))
        #expect(loaded.preferredProjectRootPath == "/tmp/fxai")
        #expect(loaded.autoReconnectEnabled == false)
    }

    @Test
    func incidentBuilderFlagsBuildRuntimeAndResearchProblems() {
        let projectRoot = URL(fileURLWithPath: "/tmp/fxai", isDirectory: true)
        let snapshot = FXAIProjectSnapshot(
            projectRoot: projectRoot,
            generatedAt: Date(),
            buildTargets: [
                BuildTargetStatus(name: "FXAI.ex5", relativePath: "FXAI.ex5", exists: false, modifiedAt: nil)
            ],
            pluginFamilies: [],
            plugins: [],
            reportCategories: [],
            recentArtifacts: [],
            runtimeProfiles: [],
            operatorSummary: OperatorSummary(profileName: "continuous", championCount: 0, deploymentCount: 0, latestReviewedAt: nil),
            tursoSummary: TursoSummary(
                localDatabasePresent: true,
                localDatabasePath: nil,
                embeddedReplicaConfigured: true,
                encryptionConfigured: true
            )
        )

        let runtimeSnapshot = RuntimeOperationsSnapshot(
            generatedAt: Date(),
            profileName: "continuous",
            deployments: [
                RuntimeDeploymentDetail(
                    id: "EURUSD",
                    symbol: "EURUSD",
                    profileName: "continuous",
                    pluginName: "ai_mlp",
                    promotionTier: "audit-approved",
                    runtimeMode: "research",
                    createdAt: nil,
                    reviewedAt: nil,
                    artifactHealth: RuntimeArtifactHealth(
                        artifactExists: true,
                        staleArtifact: true,
                        missingDeployment: false,
                        missingRouter: true,
                        missingSupervisorService: false,
                        missingSupervisorCommand: false,
                        missingWorldPlan: true,
                        artifactAgeSeconds: 900,
                        performanceFailures: ["router latency > budget"],
                        artifactSizeFailures: []
                    ),
                    deploymentPath: nil,
                    studentRouterPath: nil,
                    supervisorServicePath: nil,
                    supervisorCommandPath: nil,
                    worldPlanPath: nil,
                    deploymentSections: [],
                    routerSections: [],
                    supervisorSections: [],
                    commandSections: [],
                    worldSections: [],
                    attributionSections: [],
                    featureHighlights: [],
                    studentRouterWeights: [],
                    familyWeights: [],
                    prunedPlugins: []
                )
            ],
            champions: []
        )

        let researchSnapshot = ResearchOSControlSnapshot(
            generatedAt: Date(),
            profileName: "continuous",
            environment: ResearchOSEnvironmentStatus(
                backend: "turso",
                syncMode: "embedded-replica",
                databasePath: nil,
                databaseName: "fxai-prod",
                organizationSlug: "openai",
                groupName: "trading",
                locationName: "fra",
                cliConfigPath: nil,
                syncIntervalSeconds: 30,
                encryptionEnabled: true,
                platformAPIEnabled: true,
                syncEnabled: true,
                authTokenConfigured: true,
                apiTokenConfigured: true,
                configError: "missing TURSO_DATABASE_URL"
            ),
            branches: [],
            auditEvents: [],
            symbols: [],
            sourceOfTruth: []
        )

        let incidents = IncidentBuilder().build(
            projectRoot: projectRoot,
            snapshot: snapshot,
            runtimeSnapshot: runtimeSnapshot,
            researchSnapshot: researchSnapshot,
            newsPulseSnapshot: nil
        )

        let hasCriticalBuild = incidents.incidents.contains { incident in
            incident.category == IncidentCategory.build && incident.severity == .critical
        }
        let hasRuntimeSymbol = incidents.incidents.contains { incident in
            incident.category == IncidentCategory.runtime && incident.affectedSymbol == "EURUSD"
        }
        let hasCriticalResearch = incidents.incidents.contains { incident in
            incident.category == IncidentCategory.researchOS && incident.severity == .critical
        }
        let hasPerformance = incidents.incidents.contains { incident in
            incident.category == IncidentCategory.performance
        }

        #expect(hasCriticalBuild)
        #expect(hasRuntimeSymbol)
        #expect(hasCriticalResearch)
        #expect(hasPerformance)
    }

    @Test
    func onboardingGuideProvidesRoleSpecificDestinationsAndCommands() {
        let projectRoot = URL(fileURLWithPath: "/tmp/fxai", isDirectory: true)
        let guide = OnboardingGuideFactory.guide(role: .architect, projectRoot: projectRoot)

        #expect(guide.steps.count >= 3)
        #expect(guide.recommendedDestinations.contains(where: { $0.selection == "researchControl" }))
        #expect(guide.recommendedCommands.contains(where: { $0.command.contains("autonomous-governance") }))
    }

    @Test
    func projectConnectionCoordinatorSupportsDetachedAndReconnectStates() throws {
        let projectRoot = try makeProjectRootFixture()
        let coordinator = ProjectConnectionCoordinator(defaultProjectRootProvider: { nil })

        let waiting = coordinator.resolve(
            currentProjectRoot: nil,
            preferredProjectRoot: nil,
            autoReconnectEnabled: true
        )
        #expect(waiting.state == .waitingForProject)
        #expect(waiting.activeProjectRoot == nil)

        let connected = coordinator.resolve(
            currentProjectRoot: nil,
            preferredProjectRoot: projectRoot,
            autoReconnectEnabled: true
        )
        #expect(connected.state == .connected)
        #expect(connected.activeProjectRoot == projectRoot)

        let disconnected = coordinator.resolve(
            currentProjectRoot: nil,
            preferredProjectRoot: projectRoot,
            autoReconnectEnabled: false
        )
        #expect(disconnected.state == .disconnectedByUser)
        #expect(disconnected.activeProjectRoot == nil)
    }

    private func makeProjectRootFixture() throws -> URL {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: root.appendingPathComponent("Plugins", isDirectory: true), withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: root.appendingPathComponent("Tools", isDirectory: true), withIntermediateDirectories: true)
        try Data().write(to: root.appendingPathComponent("FXAI.mq5"))
        return root
    }
}
