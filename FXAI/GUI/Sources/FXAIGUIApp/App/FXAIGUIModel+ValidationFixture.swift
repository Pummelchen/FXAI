import FXAIGUICore
import Foundation

extension FXAIGUIModel {
    static func validationFixture(
        projectRoot: URL = URL(fileURLWithPath: "/tmp/FXAI"),
        selection: SidebarDestination = .overview,
        resourceMonitor: GUIResourceMonitor = GUIResourceMonitor(initialProfile: .default)
    ) -> FXAIGUIModel {
        let model = FXAIGUIModel(resourceMonitor: resourceMonitor, performInitialConnectionCheck: false)
        let snapshot = GUIValidationFixtures.projectSnapshot(projectRoot: projectRoot)
        let runtimeSnapshot = GUIValidationFixtures.runtimeSnapshot(projectRoot: projectRoot)
        let ratesEngineSnapshot = GUIValidationFixtures.ratesEngineSnapshot(projectRoot: projectRoot)
        let microstructureSnapshot = GUIValidationFixtures.microstructureSnapshot(projectRoot: projectRoot)
        let adaptiveRouterSnapshot = GUIValidationFixtures.adaptiveRouterSnapshot(projectRoot: projectRoot)
        let dynamicEnsembleSnapshot = GUIValidationFixtures.dynamicEnsembleSnapshot(projectRoot: projectRoot)
        let probCalibrationSnapshot = GUIValidationFixtures.probCalibrationSnapshot(projectRoot: projectRoot)
        let executionQualitySnapshot = GUIValidationFixtures.executionQualitySnapshot(projectRoot: projectRoot)
        let researchSnapshot = GUIValidationFixtures.researchSnapshot(projectRoot: projectRoot)
        let visualizationSnapshot = GUIValidationFixtures.visualizationSnapshot(projectRoot: projectRoot)
        let incidentSnapshot = GUIValidationFixtures.incidentSnapshot(projectRoot: projectRoot)

        model.projectRoot = projectRoot
        model.connectionState = .connected
        model.snapshot = snapshot
        model.runtimeSnapshot = runtimeSnapshot
        model.ratesEngineSnapshot = ratesEngineSnapshot
        model.microstructureSnapshot = microstructureSnapshot
        model.adaptiveRouterSnapshot = adaptiveRouterSnapshot
        model.dynamicEnsembleSnapshot = dynamicEnsembleSnapshot
        model.probCalibrationSnapshot = probCalibrationSnapshot
        model.executionQualitySnapshot = executionQualitySnapshot
        model.researchSnapshot = researchSnapshot
        model.visualizationSnapshot = visualizationSnapshot
        model.incidentSnapshot = incidentSnapshot
        model.selection = selection
        model.lastConnectionCheckAt = Date()
        model.selectedRole = .liveTrader
        model.selectedRuntimeSymbol = runtimeSnapshot.symbols.first ?? ""
        model.selectedRatesSymbol = ratesEngineSnapshot.pairs.first?.pair ?? ""
        model.selectedMicrostructureSymbol = microstructureSnapshot.symbols.first?.symbol ?? ""
        model.selectedAdaptiveSymbol = adaptiveRouterSnapshot.symbols.first?.symbol ?? ""
        model.selectedDynamicEnsembleSymbol = dynamicEnsembleSnapshot.symbols.first?.symbol ?? ""
        model.selectedProbCalibrationSymbol = probCalibrationSnapshot.symbols.first?.symbol ?? ""
        model.selectedExecutionQualitySymbol = executionQualitySnapshot.symbols.first?.symbol ?? ""
        model.selectedResearchSymbol = researchSnapshot.symbols.first?.symbol ?? ""
        model.selectedVisualizationSymbol = visualizationSnapshot.symbols.first ?? ""
        model.selectedIncidentID = incidentSnapshot.incidents.first?.id
        model.selectedPluginFamily = "All"
        model.reportCategoryFilter = "All"
        model.savedViews = [
            SavedWorkspaceView(
                id: UUID(),
                name: "Live Runtime Review",
                projectRootPath: projectRoot.path,
                createdAt: Date(),
                updatedAt: Date(),
                selection: SidebarDestination.runtimeMonitor.rawValue,
                selectedRole: .liveTrader,
                selectedRuntimeSymbol: runtimeSnapshot.symbols.first ?? "EURUSD",
                selectedRatesSymbol: ratesEngineSnapshot.pairs.first?.pair ?? "EURUSD",
                selectedMicrostructureSymbol: microstructureSnapshot.symbols.first?.symbol ?? "EURUSD",
                selectedAdaptiveSymbol: adaptiveRouterSnapshot.symbols.first?.symbol ?? "EURUSD",
                selectedDynamicEnsembleSymbol: dynamicEnsembleSnapshot.symbols.first?.symbol ?? "EURUSD",
                selectedProbCalibrationSymbol: probCalibrationSnapshot.symbols.first?.symbol ?? "EURUSD",
                selectedExecutionQualitySymbol: executionQualitySnapshot.symbols.first?.symbol ?? "EURUSD",
                selectedResearchSymbol: researchSnapshot.symbols.first?.symbol ?? "EURUSD",
                selectedVisualizationSymbol: visualizationSnapshot.symbols.first ?? "EURUSD",
                pluginSearchText: "",
                selectedPluginFamily: "All",
                reportCategoryFilter: "All",
                auditDraft: model.auditDraft,
                backtestDraft: model.backtestDraft,
                offlineDraft: model.offlineDraft,
                researchBranchDraft: model.researchBranchDraft,
                researchAuditDraft: model.researchAuditDraft,
                researchVectorDraft: model.researchVectorDraft,
                researchRecoveryDraft: model.researchRecoveryDraft,
                overviewLayout: model.overviewLayout
            )
        ]
        model.completedOnboardingRoles = [.liveTrader]
        return model
    }
}
