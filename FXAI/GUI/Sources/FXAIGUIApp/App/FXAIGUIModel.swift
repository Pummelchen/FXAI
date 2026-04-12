import AppKit
import Combine
import FXAIGUICore
import Foundation

@MainActor
final class FXAIGUIModel: ObservableObject {
    @Published var projectRoot: URL?
    @Published var connectionState: ProjectConnectionState = .waitingForProject
    @Published var autoReconnectEnabled = true
    @Published var lastConnectionCheckAt: Date?
    @Published var snapshot: FXAIProjectSnapshot?
    @Published var runtimeSnapshot: RuntimeOperationsSnapshot?
    @Published var newsPulseSnapshot: NewsPulseSnapshot?
    @Published var ratesEngineSnapshot: RatesEngineSnapshot?
    @Published var crossAssetSnapshot: CrossAssetSnapshot?
    @Published var pairNetworkSnapshot: PairNetworkSnapshot?
    @Published var microstructureSnapshot: MicrostructureSnapshot?
    @Published var adaptiveRouterSnapshot: AdaptiveRouterSnapshot?
    @Published var driftGovernanceSnapshot: DriftGovernanceSnapshot?
    @Published var dynamicEnsembleSnapshot: DynamicEnsembleSnapshot?
    @Published var probCalibrationSnapshot: ProbCalibrationSnapshot?
    @Published var executionQualitySnapshot: ExecutionQualitySnapshot?
    @Published var labelEngineSnapshot: LabelEngineSnapshot?
    @Published var researchSnapshot: ResearchOSControlSnapshot?
    @Published var visualizationSnapshot: AdvancedVisualizationSnapshot?
    @Published var incidentSnapshot: IncidentCenterSnapshot?
    @Published var selection: SidebarDestination? = .overview
    @Published var selectedRole: WorkspaceRole = .liveTrader
    @Published var selectedRuntimeSymbol = ""
    @Published var selectedRatesSymbol = ""
    @Published var selectedCrossAssetSymbol = ""
    @Published var selectedPairNetworkSymbol = ""
    @Published var selectedMicrostructureSymbol = ""
    @Published var selectedAdaptiveSymbol = ""
    @Published var selectedDynamicEnsembleSymbol = ""
    @Published var selectedProbCalibrationSymbol = ""
    @Published var selectedExecutionQualitySymbol = ""
    @Published var selectedLabelEngineDatasetKey = ""
    @Published var selectedResearchSymbol = ""
    @Published var selectedVisualizationSymbol = ""
    @Published var selectedIncidentID: String?
    @Published var overviewLayout = OverviewDashboardLayoutState.default()
    @Published var roleWorkspaceLayouts = RoleWorkspaceDashboardLayouts.default()
    @Published var pluginSearchText = ""
    @Published var selectedPluginFamily = "All"
    @Published var reportCategoryFilter = "All"
    @Published var saveViewNameDraft = ""
    @Published var auditDraft = AuditLabDraft()
    @Published var backtestDraft = BacktestBuilderDraft()
    @Published var offlineDraft = OfflineLabDraft()
    @Published var researchBranchDraft = ResearchOSBranchDraft()
    @Published var researchAuditDraft = ResearchOSAuditDraft()
    @Published var researchVectorDraft = ResearchOSVectorDraft()
    @Published var researchRecoveryDraft = ResearchOSRecoveryDraft()
    @Published var savedViews: [SavedWorkspaceView] = []
    @Published var completedOnboardingRoles: Set<WorkspaceRole> = []
    @Published var lastErrorMessage: String?
    @Published var isRefreshing = false
    @Published private(set) var renderingProfile: GUIRenderingProfile

    private let scanner = ProjectScanner()
    private let connectionCoordinator = ProjectConnectionCoordinator()
    private let runtimeReader = RuntimeArtifactReader()
    private let newsPulseReader = NewsPulseArtifactReader()
    private let ratesEngineReader = RatesEngineArtifactReader()
    private let crossAssetReader = CrossAssetArtifactReader()
    private let pairNetworkReader = PairNetworkArtifactReader()
    private let microstructureReader = MicrostructureArtifactReader()
    private let adaptiveRouterReader = AdaptiveRouterArtifactReader()
    private let driftGovernanceReader = DriftGovernanceArtifactReader()
    private let dynamicEnsembleReader = DynamicEnsembleArtifactReader()
    private let probCalibrationReader = ProbCalibrationArtifactReader()
    private let executionQualityReader = ExecutionQualityArtifactReader()
    private let labelEngineReader = LabelEngineArtifactReader()
    private let researchReader = ResearchOSArtifactReader()
    private let visualizationBuilder = AdvancedVisualizationBuilder()
    private let incidentBuilder = IncidentBuilder()
    private let workspaceStore = SavedWorkspaceStore()
    private let resourceMonitor: GUIResourceMonitor
    private var cancellables: Set<AnyCancellable> = []
    private var connectionTimerCancellable: AnyCancellable?
    private var isRestoringPersistedState = false
    private var preferredProjectRoot: URL?

    init(
        resourceMonitor: GUIResourceMonitor = .shared,
        performInitialConnectionCheck: Bool = true
    ) {
        self.resourceMonitor = resourceMonitor
        self.renderingProfile = resourceMonitor.profile
        loadPersistedState()
        bindPersistence()
        bindResourceManagement()

        if performInitialConnectionCheck {
            Task {
                await performSoftConnectionCheck(forceRefresh: true)
            }
        }
    }

    var commandRecipes: [CommandRecipe] {
        guard let projectRoot else { return [] }
        return CommandFactory.recipes(projectRoot: projectRoot)
    }

    var pluginNames: [String] {
        guard let snapshot else { return [] }
        return snapshot.plugins.map(\.name)
    }

    var projectPathLabel: String {
        projectRoot?.path ?? preferredProjectRoot?.path ?? "No FXAI project connected"
    }

    var connectionStatusLabel: String {
        connectionState.title
    }

    var selectedRuntimeDetail: RuntimeDeploymentDetail? {
        guard let runtimeSnapshot else { return nil }
        if let selected = runtimeSnapshot.deployments.first(where: { $0.symbol == selectedRuntimeSymbol }) {
            return selected
        }
        return runtimeSnapshot.deployments.first
    }

    var selectedResearchSymbolDetail: ResearchOSSymbolControl? {
        guard let researchSnapshot else { return nil }
        if let selected = researchSnapshot.symbols.first(where: { $0.symbol == selectedResearchSymbol }) {
            return selected
        }
        return researchSnapshot.symbols.first
    }

    var selectedAdaptiveRouterDetail: AdaptiveRouterSymbolSnapshot? {
        guard let adaptiveRouterSnapshot else { return nil }
        if let selected = adaptiveRouterSnapshot.symbols.first(where: { $0.symbol == selectedAdaptiveSymbol }) {
            return selected
        }
        return adaptiveRouterSnapshot.symbols.first
    }

    var selectedDriftGovernanceDetail: DriftGovernanceSymbolSnapshot? {
        guard let driftGovernanceSnapshot else { return nil }
        if let selected = driftGovernanceSnapshot.symbols.first(where: { $0.symbol == selectedRuntimeSymbol }) {
            return selected
        }
        return driftGovernanceSnapshot.symbols.first
    }

    var selectedDynamicEnsembleDetail: DynamicEnsembleSymbolSnapshot? {
        guard let dynamicEnsembleSnapshot else { return nil }
        if let selected = dynamicEnsembleSnapshot.symbols.first(where: { $0.symbol == selectedDynamicEnsembleSymbol }) {
            return selected
        }
        return dynamicEnsembleSnapshot.symbols.first
    }

    var selectedProbCalibrationDetail: ProbCalibrationSymbolSnapshot? {
        guard let probCalibrationSnapshot else { return nil }
        if let selected = probCalibrationSnapshot.symbols.first(where: { $0.symbol == selectedProbCalibrationSymbol }) {
            return selected
        }
        return probCalibrationSnapshot.symbols.first
    }

    var selectedExecutionQualityDetail: ExecutionQualitySymbolSnapshot? {
        guard let executionQualitySnapshot else { return nil }
        if let selected = executionQualitySnapshot.symbols.first(where: { $0.symbol == selectedExecutionQualitySymbol }) {
            return selected
        }
        return executionQualitySnapshot.symbols.first
    }

    var selectedLabelEngineDetail: LabelEngineBuildSnapshot? {
        guard let labelEngineSnapshot else { return nil }
        if let selected = labelEngineSnapshot.builds.first(where: { $0.datasetKey == selectedLabelEngineDatasetKey }) {
            return selected
        }
        return labelEngineSnapshot.builds.first
    }

    var selectedMicrostructureDetail: MicrostructureSymbolState? {
        guard let microstructureSnapshot else { return nil }
        if let selected = microstructureSnapshot.symbols.first(where: { $0.symbol == selectedMicrostructureSymbol }) {
            return selected
        }
        return microstructureSnapshot.symbols.first
    }

    var selectedRatesEngineDetail: RatesEnginePairState? {
        guard let ratesEngineSnapshot else { return nil }
        if let selected = ratesEngineSnapshot.pairs.first(where: { $0.pair == selectedRatesSymbol }) {
            return selected
        }
        return ratesEngineSnapshot.pairs.first
    }

    var selectedCrossAssetDetail: CrossAssetPairState? {
        guard let crossAssetSnapshot else { return nil }
        if let selected = crossAssetSnapshot.pairs.first(where: { $0.pair == selectedCrossAssetSymbol }) {
            return selected
        }
        return crossAssetSnapshot.pairs.first
    }

    var selectedPairNetworkDetail: PairNetworkSymbolSnapshot? {
        guard let pairNetworkSnapshot else { return nil }
        if let selected = pairNetworkSnapshot.symbols.first(where: { $0.symbol == selectedPairNetworkSymbol }) {
            return selected
        }
        return pairNetworkSnapshot.symbols.first
    }

    var selectedVisualizationDetail: SymbolVisualizationDetail? {
        guard let visualizationSnapshot else { return nil }
        if let selected = visualizationSnapshot.symbolDetails.first(where: { $0.symbol == selectedVisualizationSymbol }) {
            return selected
        }
        return visualizationSnapshot.symbolDetails.first
    }

    var selectedIncident: FXAIIncident? {
        guard let incidentSnapshot else { return nil }
        if let selectedIncidentID,
           let selected = incidentSnapshot.incidents.first(where: { $0.id == selectedIncidentID }) {
            return selected
        }
        return incidentSnapshot.incidents.first
    }

    var currentOnboardingGuide: RoleOnboardingGuide? {
        guard let projectRoot else { return nil }
        return OnboardingGuideFactory.guide(role: selectedRole, projectRoot: projectRoot)
    }

    var currentSelectionTitle: String {
        selection?.title ?? SidebarDestination.overview.title
    }

    var highestIncidentSeverity: IncidentSeverity? {
        incidentSnapshot?.highestSeverity
    }

    func refresh() async {
        guard let projectRoot, ProjectPathResolver.isProjectRoot(projectRoot) else {
            clearSnapshots()
            connectionState = autoReconnectEnabled ? .waitingForProject : .disconnectedByUser
            if autoReconnectEnabled {
                lastErrorMessage = nil
            }
            return
        }

        isRefreshing = true
        defer { isRefreshing = false }

        do {
            snapshot = try scanner.scan(projectRoot: projectRoot)
            runtimeSnapshot = runtimeReader.read(projectRoot: projectRoot)
            newsPulseSnapshot = newsPulseReader.read(projectRoot: projectRoot)
            ratesEngineSnapshot = ratesEngineReader.read(projectRoot: projectRoot)
            crossAssetSnapshot = crossAssetReader.read(projectRoot: projectRoot)
            pairNetworkSnapshot = pairNetworkReader.read(projectRoot: projectRoot)
            microstructureSnapshot = microstructureReader.read(projectRoot: projectRoot)
            adaptiveRouterSnapshot = adaptiveRouterReader.read(projectRoot: projectRoot)
            driftGovernanceSnapshot = driftGovernanceReader.read(projectRoot: projectRoot)
            dynamicEnsembleSnapshot = dynamicEnsembleReader.read(projectRoot: projectRoot)
            probCalibrationSnapshot = probCalibrationReader.read(projectRoot: projectRoot)
            executionQualitySnapshot = executionQualityReader.read(projectRoot: projectRoot)
            labelEngineSnapshot = labelEngineReader.read(projectRoot: projectRoot)
            researchSnapshot = researchReader.read(projectRoot: projectRoot)
            visualizationSnapshot = visualizationBuilder.build(
                projectRoot: projectRoot,
                runtimeSnapshot: runtimeSnapshot,
                researchSnapshot: researchSnapshot
            )
            incidentSnapshot = incidentBuilder.build(
                projectRoot: projectRoot,
                snapshot: snapshot,
                runtimeSnapshot: runtimeSnapshot,
                researchSnapshot: researchSnapshot,
                newsPulseSnapshot: newsPulseSnapshot
            )
            syncBuilderDefaults()
            syncRuntimeSelection()
            syncRatesSelection()
            syncCrossAssetSelection()
            syncPairNetworkSelection()
            syncMicrostructureSelection()
            syncAdaptiveSelection()
            syncDriftGovernanceSelection()
            syncDynamicEnsembleSelection()
            syncProbCalibrationSelection()
            syncExecutionQualitySelection()
            syncLabelEngineSelection()
            syncResearchSelection()
            syncVisualizationSelection()
            syncIncidentSelection()
            connectionState = .connected
            preferredProjectRoot = projectRoot
            lastErrorMessage = nil
        } catch {
            clearSnapshots()
            connectionState = autoReconnectEnabled ? .waitingForProject : .disconnectedByUser
            lastErrorMessage = error.localizedDescription
        }
    }

    func chooseProjectRoot() {
        let panel = NSOpenPanel()
        panel.canChooseDirectories = true
        panel.canChooseFiles = false
        panel.allowsMultipleSelection = false
        panel.prompt = "Use FXAI Project"
        panel.message = "Choose the FXAI project root that contains FXAI.mq5, Plugins, and Tools."
        panel.directoryURL = projectRoot

        guard panel.runModal() == .OK, let url = panel.url else {
            return
        }

        preferredProjectRoot = url
        autoReconnectEnabled = true
        Task {
            await performSoftConnectionCheck(forceRefresh: true)
        }
    }

    func disconnectProject() {
        preferredProjectRoot = projectRoot ?? preferredProjectRoot
        projectRoot = nil
        autoReconnectEnabled = false
        connectionState = .disconnectedByUser
        clearSnapshots()
        lastErrorMessage = nil
    }

    func reconnectProject() {
        autoReconnectEnabled = true
        Task {
            await performSoftConnectionCheck(forceRefresh: true)
        }
    }

    func navigate(to destination: SidebarDestination) {
        selection = destination
    }

    func activateRoleWorkspace(_ role: WorkspaceRole) {
        selectedRole = role
        selection = role.defaultDestination
    }

    func openProjectRootInFinder() {
        guard let projectRoot else { return }
        NSWorkspace.shared.open(projectRoot)
    }

    func openInFinder(_ url: URL) {
        NSWorkspace.shared.activateFileViewerSelecting([url])
    }

    func openURL(_ url: URL) {
        NSWorkspace.shared.open(url)
    }

    func copyToPasteboard(_ value: String) {
        let pasteboard = NSPasteboard.general
        pasteboard.clearContents()
        pasteboard.setString(value, forType: .string)
    }

    func handoffCommandToTerminal(_ command: String) {
        copyToPasteboard(command)

        guard let projectRoot else { return }

        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/open")
        process.arguments = ["-a", "Terminal", projectRoot.path]
        try? process.run()
    }

    func saveCurrentView(named proposedName: String? = nil) {
        let fallbackName = defaultSavedViewName()
        let trimmedName = (proposedName ?? saveViewNameDraft)
            .trimmingCharacters(in: .whitespacesAndNewlines)
        let targetName = trimmedName.isEmpty ? fallbackName : trimmedName
        let now = Date()

        if let existingIndex = savedViews.firstIndex(where: { $0.name.caseInsensitiveCompare(targetName) == .orderedSame }) {
            let existingID = savedViews[existingIndex].id
            var updated = currentWorkspaceSnapshot(name: targetName, id: existingID, createdAt: savedViews[existingIndex].createdAt, updatedAt: now)
            updated.updatedAt = now
            savedViews[existingIndex] = updated
        } else {
            let savedView = currentWorkspaceSnapshot(name: targetName, updatedAt: now)
            savedViews.insert(savedView, at: 0)
        }

        savedViews.sort { lhs, rhs in
            if lhs.updatedAt == rhs.updatedAt {
                return lhs.name.localizedCaseInsensitiveCompare(rhs.name) == .orderedAscending
            }
            return lhs.updatedAt > rhs.updatedAt
        }
        saveViewNameDraft = ""
    }

    func applySavedView(_ savedView: SavedWorkspaceView) {
        let applyState = {
            self.selection = SidebarDestination(rawValue: savedView.selection) ?? .overview
            self.selectedRole = savedView.selectedRole
            self.selectedRuntimeSymbol = savedView.selectedRuntimeSymbol
            self.selectedRatesSymbol = savedView.selectedRatesSymbol
            self.selectedCrossAssetSymbol = savedView.selectedCrossAssetSymbol
            self.selectedPairNetworkSymbol = savedView.selectedPairNetworkSymbol
            self.selectedMicrostructureSymbol = savedView.selectedMicrostructureSymbol
            self.selectedAdaptiveSymbol = savedView.selectedAdaptiveSymbol
            self.selectedDynamicEnsembleSymbol = savedView.selectedDynamicEnsembleSymbol
            self.selectedProbCalibrationSymbol = savedView.selectedProbCalibrationSymbol
            self.selectedExecutionQualitySymbol = savedView.selectedExecutionQualitySymbol
            self.selectedLabelEngineDatasetKey = savedView.selectedLabelEngineDatasetKey
            self.selectedResearchSymbol = savedView.selectedResearchSymbol
            self.selectedVisualizationSymbol = savedView.selectedVisualizationSymbol
            self.overviewLayout = savedView.overviewLayout.normalized()
            self.roleWorkspaceLayouts = savedView.roleWorkspaceLayouts.normalized()
            self.pluginSearchText = savedView.pluginSearchText
            self.selectedPluginFamily = savedView.selectedPluginFamily
            self.reportCategoryFilter = savedView.reportCategoryFilter
            self.auditDraft = savedView.auditDraft
            self.backtestDraft = savedView.backtestDraft
            self.offlineDraft = savedView.offlineDraft
            self.researchBranchDraft = savedView.researchBranchDraft
            self.researchAuditDraft = savedView.researchAuditDraft
            self.researchVectorDraft = savedView.researchVectorDraft
            self.researchRecoveryDraft = savedView.researchRecoveryDraft
            self.syncBuilderDefaults()
            self.syncRuntimeSelection()
            self.syncRatesSelection()
            self.syncCrossAssetSelection()
            self.syncMicrostructureSelection()
            self.syncAdaptiveSelection()
            self.syncDynamicEnsembleSelection()
            self.syncProbCalibrationSelection()
            self.syncExecutionQualitySelection()
            self.syncLabelEngineSelection()
            self.syncResearchSelection()
            self.syncVisualizationSelection()
        }

        if let targetRootPath = savedView.projectRootPath,
           targetRootPath != projectRoot?.path {
            projectRoot = URL(fileURLWithPath: targetRootPath, isDirectory: true)
            applyState()
            Task { await refresh() }
            return
        }

        applyState()
    }

    func deleteSavedView(_ savedView: SavedWorkspaceView) {
        savedViews.removeAll { $0.id == savedView.id }
    }

    func resetOverviewLayout() {
        overviewLayout = .default()
    }

    func roleWorkspaceLayout(for role: WorkspaceRole) -> RoleWorkspaceDashboardLayoutState {
        guard let kind = RoleWorkspaceDashboardKind(role: role) else {
            return .default(for: .liveOverview)
        }
        return roleWorkspaceLayouts.layout(for: kind)
    }

    func resetRoleWorkspaceLayout(for role: WorkspaceRole) {
        guard let kind = RoleWorkspaceDashboardKind(role: role) else { return }
        updateRoleWorkspaceLayout(for: kind) { layout in
            layout = .default(for: kind)
        }
    }

    func moveOverviewSection(_ sectionID: UUID, by delta: Int) {
        updateOverviewLayout { layout in
            guard let index = layout.sections.firstIndex(where: { $0.id == sectionID }) else { return }
            let targetIndex = min(max(0, index + delta), layout.sections.count - 1)
            guard targetIndex != index else { return }
            let section = layout.sections.remove(at: index)
            layout.sections.insert(section, at: targetIndex)
        }
    }

    func reorderOverviewSection(draggedSectionID: UUID, before targetSectionID: UUID) {
        updateOverviewLayout { layout in
            guard
                let sourceIndex = layout.sections.firstIndex(where: { $0.id == draggedSectionID }),
                let targetIndex = layout.sections.firstIndex(where: { $0.id == targetSectionID }),
                sourceIndex != targetIndex
            else { return }

            let section = layout.sections.remove(at: sourceIndex)
            let insertionIndex = sourceIndex < targetIndex ? targetIndex - 1 : targetIndex
            layout.sections.insert(section, at: insertionIndex)
        }
    }

    func moveOverviewWidget(sectionID: UUID, widgetID: UUID, by delta: Int) {
        updateOverviewLayout { layout in
            guard let sectionIndex = layout.sections.firstIndex(where: { $0.id == sectionID }) else { return }
            guard let widgetIndex = layout.sections[sectionIndex].widgets.firstIndex(where: { $0.id == widgetID }) else { return }
            let targetIndex = min(max(0, widgetIndex + delta), layout.sections[sectionIndex].widgets.count - 1)
            guard targetIndex != widgetIndex else { return }
            let widget = layout.sections[sectionIndex].widgets.remove(at: widgetIndex)
            layout.sections[sectionIndex].widgets.insert(widget, at: targetIndex)
        }
    }

    func moveOverviewWidgetOnGrid(sectionID: UUID, widgetID: UUID, columnDelta: Int = 0, rowDelta: Int = 0) {
        updateOverviewLayout { layout in
            guard let sectionIndex = layout.sections.firstIndex(where: { $0.id == sectionID }) else { return }
            guard let widgetIndex = layout.sections[sectionIndex].widgets.firstIndex(where: { $0.id == widgetID }) else { return }

            var widget = layout.sections[sectionIndex].widgets.remove(at: widgetIndex)
            widget.columnUnits = max(0, widget.columnUnits + columnDelta)
            widget.rowUnits = max(0, widget.rowUnits + rowDelta)
            layout.sections[sectionIndex].widgets.insert(widget, at: 0)
        }
    }

    func reorderOverviewWidget(
        draggedWidgetID: UUID,
        from sourceSectionID: UUID,
        to targetSectionID: UUID,
        before targetWidgetID: UUID? = nil
    ) {
        updateOverviewLayout { layout in
            guard let sourceSectionIndex = layout.sections.firstIndex(where: { $0.id == sourceSectionID }) else { return }
            guard let draggedWidgetIndex = layout.sections[sourceSectionIndex].widgets.firstIndex(where: { $0.id == draggedWidgetID }) else { return }

            let widget = layout.sections[sourceSectionIndex].widgets.remove(at: draggedWidgetIndex)
            guard let targetSectionIndex = layout.sections.firstIndex(where: { $0.id == targetSectionID }) else { return }

            if let targetWidgetID,
               let targetIndex = layout.sections[targetSectionIndex].widgets.firstIndex(where: { $0.id == targetWidgetID }) {
                let adjustedIndex: Int
                if sourceSectionIndex == targetSectionIndex && draggedWidgetIndex < targetIndex {
                    adjustedIndex = max(0, targetIndex - 1)
                } else {
                    adjustedIndex = targetIndex
                }
                layout.sections[targetSectionIndex].widgets.insert(widget, at: adjustedIndex)
            } else {
                layout.sections[targetSectionIndex].widgets.append(widget)
            }
        }
    }

    func resizeOverviewWidget(sectionID: UUID, widgetID: UUID, widthDelta: Int = 0, heightDelta: Int = 0) {
        updateOverviewLayout { layout in
            guard let sectionIndex = layout.sections.firstIndex(where: { $0.id == sectionID }) else { return }
            guard let widgetIndex = layout.sections[sectionIndex].widgets.firstIndex(where: { $0.id == widgetID }) else { return }
            var widget = layout.sections[sectionIndex].widgets[widgetIndex]
            let spec = OverviewDashboardLayoutState.spec(for: widget.kind)
            widget.widthUnits = min(max(widget.widthUnits + widthDelta, spec.minimumWidthUnits), spec.maximumWidthUnits)
            widget.heightUnits = min(max(widget.heightUnits + heightDelta, spec.minimumHeightUnits), spec.maximumHeightUnits)
            layout.sections[sectionIndex].widgets[widgetIndex] = widget
        }
    }

    func moveRoleWorkspacePanelOnGrid(role: WorkspaceRole, panelID: UUID, columnDelta: Int = 0, rowDelta: Int = 0) {
        guard let kind = RoleWorkspaceDashboardKind(role: role) else { return }
        updateRoleWorkspaceLayout(for: kind) { layout in
            guard let panelIndex = layout.panels.firstIndex(where: { $0.id == panelID }) else { return }

            var panel = layout.panels.remove(at: panelIndex)
            panel.columnUnits = max(0, panel.columnUnits + columnDelta)
            panel.rowUnits = max(0, panel.rowUnits + rowDelta)
            layout.panels.insert(panel, at: 0)
        }
    }

    func resizeRoleWorkspacePanel(role: WorkspaceRole, panelID: UUID, widthDelta: Int = 0, heightDelta: Int = 0) {
        guard let kind = RoleWorkspaceDashboardKind(role: role) else { return }
        updateRoleWorkspaceLayout(for: kind) { layout in
            guard let panelIndex = layout.panels.firstIndex(where: { $0.id == panelID }) else { return }
            var panel = layout.panels[panelIndex]
            let spec = RoleWorkspaceDashboardLayoutState.spec(for: panel.kind)
            panel.widthUnits = min(max(panel.widthUnits + widthDelta, spec.minimumWidthUnits), spec.maximumWidthUnits)
            panel.heightUnits = min(max(panel.heightUnits + heightDelta, spec.minimumHeightUnits), spec.maximumHeightUnits)
            layout.panels[panelIndex] = panel
        }
    }

    func markOnboardingCompleted(for role: WorkspaceRole) {
        completedOnboardingRoles.insert(role)
    }

    func resetOnboarding(for role: WorkspaceRole) {
        completedOnboardingRoles.remove(role)
    }

    func hasCompletedOnboarding(for role: WorkspaceRole) -> Bool {
        completedOnboardingRoles.contains(role)
    }

    private func syncBuilderDefaults() {
        let names = pluginNames
        guard let firstPlugin = names.first else { return }

        if auditDraft.pluginName.isEmpty || !names.contains(auditDraft.pluginName) {
            auditDraft.pluginName = firstPlugin
        }
        if backtestDraft.pluginName.isEmpty || !names.contains(backtestDraft.pluginName) {
            backtestDraft.pluginName = firstPlugin
        }
    }

    private func syncRuntimeSelection() {
        guard let runtimeSnapshot else {
            selectedRuntimeSymbol = ""
            return
        }

        let symbols = runtimeSnapshot.symbols
        guard let first = symbols.first else {
            selectedRuntimeSymbol = ""
            return
        }

        if !symbols.contains(selectedRuntimeSymbol) {
            selectedRuntimeSymbol = first
        }
    }

    private func syncRatesSelection() {
        guard let ratesEngineSnapshot else {
            selectedRatesSymbol = ""
            return
        }

        let symbols = ratesEngineSnapshot.pairs.map(\.pair)
        guard let first = symbols.first else {
            selectedRatesSymbol = ""
            return
        }

        if !symbols.contains(selectedRatesSymbol) {
            selectedRatesSymbol = selectedRuntimeSymbol.isEmpty ? first : selectedRuntimeSymbol
            if !symbols.contains(selectedRatesSymbol) {
                selectedRatesSymbol = first
            }
        }
    }

    private func syncCrossAssetSelection() {
        guard let crossAssetSnapshot else {
            selectedCrossAssetSymbol = ""
            return
        }

        let symbols = crossAssetSnapshot.pairs.map(\.pair)
        guard let first = symbols.first else {
            selectedCrossAssetSymbol = ""
            return
        }

        if !symbols.contains(selectedCrossAssetSymbol) {
            selectedCrossAssetSymbol = selectedRuntimeSymbol.isEmpty ? first : selectedRuntimeSymbol
            if !symbols.contains(selectedCrossAssetSymbol) {
                selectedCrossAssetSymbol = first
            }
        }
    }

    private func syncPairNetworkSelection() {
        guard let pairNetworkSnapshot else {
            selectedPairNetworkSymbol = ""
            return
        }

        let symbols = pairNetworkSnapshot.symbols.map(\.symbol)
        guard let first = symbols.first else {
            selectedPairNetworkSymbol = ""
            return
        }

        if !symbols.contains(selectedPairNetworkSymbol) {
            selectedPairNetworkSymbol = selectedRuntimeSymbol.isEmpty ? first : selectedRuntimeSymbol
            if !symbols.contains(selectedPairNetworkSymbol) {
                selectedPairNetworkSymbol = first
            }
        }
    }

    private func syncMicrostructureSelection() {
        guard let microstructureSnapshot else {
            selectedMicrostructureSymbol = ""
            return
        }

        let symbols = microstructureSnapshot.symbols.map(\.symbol)
        guard let first = symbols.first else {
            selectedMicrostructureSymbol = ""
            return
        }

        if !symbols.contains(selectedMicrostructureSymbol) {
            selectedMicrostructureSymbol = selectedRuntimeSymbol.isEmpty ? first : selectedRuntimeSymbol
            if !symbols.contains(selectedMicrostructureSymbol) {
                selectedMicrostructureSymbol = first
            }
        }
    }

    private func syncAdaptiveSelection() {
        guard let adaptiveRouterSnapshot else {
            selectedAdaptiveSymbol = ""
            return
        }

        let symbols = adaptiveRouterSnapshot.symbols.map(\.symbol)
        guard let first = symbols.first else {
            selectedAdaptiveSymbol = ""
            return
        }

        if !symbols.contains(selectedAdaptiveSymbol) {
            selectedAdaptiveSymbol = selectedRuntimeSymbol.isEmpty ? first : selectedRuntimeSymbol
            if !symbols.contains(selectedAdaptiveSymbol) {
                selectedAdaptiveSymbol = first
            }
        }
    }

    private func syncDriftGovernanceSelection() {
        guard let driftGovernanceSnapshot else { return }

        let symbols = driftGovernanceSnapshot.symbols.map(\.symbol)
        guard let first = symbols.first else { return }

        if !symbols.contains(selectedRuntimeSymbol) {
            selectedRuntimeSymbol = first
        }
    }

    private func syncDynamicEnsembleSelection() {
        guard let dynamicEnsembleSnapshot else {
            selectedDynamicEnsembleSymbol = ""
            return
        }

        let symbols = dynamicEnsembleSnapshot.symbols.map(\.symbol)
        guard let first = symbols.first else {
            selectedDynamicEnsembleSymbol = ""
            return
        }

        if !symbols.contains(selectedDynamicEnsembleSymbol) {
            selectedDynamicEnsembleSymbol = selectedRuntimeSymbol.isEmpty ? first : selectedRuntimeSymbol
            if !symbols.contains(selectedDynamicEnsembleSymbol) {
                selectedDynamicEnsembleSymbol = first
            }
        }
    }

    private func syncProbCalibrationSelection() {
        guard let probCalibrationSnapshot else {
            selectedProbCalibrationSymbol = ""
            return
        }

        let symbols = probCalibrationSnapshot.symbols.map(\.symbol)
        guard let first = symbols.first else {
            selectedProbCalibrationSymbol = ""
            return
        }

        if !symbols.contains(selectedProbCalibrationSymbol) {
            selectedProbCalibrationSymbol = selectedRuntimeSymbol.isEmpty ? first : selectedRuntimeSymbol
            if !symbols.contains(selectedProbCalibrationSymbol) {
                selectedProbCalibrationSymbol = first
            }
        }
    }

    private func syncExecutionQualitySelection() {
        guard let executionQualitySnapshot else {
            selectedExecutionQualitySymbol = ""
            return
        }

        let symbols = executionQualitySnapshot.symbols.map(\.symbol)
        guard let first = symbols.first else {
            selectedExecutionQualitySymbol = ""
            return
        }

        if !symbols.contains(selectedExecutionQualitySymbol) {
            selectedExecutionQualitySymbol = selectedRuntimeSymbol.isEmpty ? first : selectedRuntimeSymbol
            if !symbols.contains(selectedExecutionQualitySymbol) {
                selectedExecutionQualitySymbol = first
            }
        }
    }

    private func syncLabelEngineSelection() {
        guard let labelEngineSnapshot else {
            selectedLabelEngineDatasetKey = ""
            return
        }

        let datasetKeys = labelEngineSnapshot.builds.map(\.datasetKey)
        guard let first = datasetKeys.first else {
            selectedLabelEngineDatasetKey = ""
            return
        }

        if !datasetKeys.contains(selectedLabelEngineDatasetKey) {
            selectedLabelEngineDatasetKey = first
        }
    }

    private func syncResearchSelection() {
        guard let researchSnapshot else {
            selectedResearchSymbol = ""
            return
        }

        let symbols = researchSnapshot.symbols.map(\.symbol)
        if let first = symbols.first {
            if !symbols.contains(selectedResearchSymbol) {
                selectedResearchSymbol = first
            }
            if researchVectorDraft.symbol.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty || !symbols.contains(researchVectorDraft.symbol) {
                researchVectorDraft.symbol = first
            }
        } else {
            selectedResearchSymbol = ""
        }

        if let profileName = researchSnapshot.profileName, !profileName.isEmpty {
            researchBranchDraft.profileName = profileName
            researchVectorDraft.profileName = profileName
            researchRecoveryDraft.profileName = profileName
        }

        if let environment = researchSnapshot.environment {
            if researchBranchDraft.sourceDatabase.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                researchBranchDraft.sourceDatabase = environment.databaseName
            }
            if researchBranchDraft.groupName.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                researchBranchDraft.groupName = environment.groupName
            }
            if researchBranchDraft.locationName.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                researchBranchDraft.locationName = environment.locationName
            }
        }
    }

    private func syncVisualizationSelection() {
        guard let visualizationSnapshot else {
            selectedVisualizationSymbol = ""
            return
        }

        let symbols = visualizationSnapshot.symbols
        guard let first = symbols.first else {
            selectedVisualizationSymbol = ""
            return
        }

        if !symbols.contains(selectedVisualizationSymbol) {
            selectedVisualizationSymbol = selectedRuntimeSymbol.isEmpty ? first : selectedRuntimeSymbol
            if !symbols.contains(selectedVisualizationSymbol) {
                selectedVisualizationSymbol = first
            }
        }
    }

    private func syncIncidentSelection() {
        guard let incidentSnapshot, !incidentSnapshot.incidents.isEmpty else {
            selectedIncidentID = nil
            return
        }

        if let selectedIncidentID,
           incidentSnapshot.incidents.contains(where: { $0.id == selectedIncidentID }) {
            return
        }

        selectedIncidentID = incidentSnapshot.incidents.first?.id
    }

    private func defaultSavedViewName() -> String {
        let formatter = DateFormatter()
        formatter.dateFormat = "MMM d HH:mm"
        return "\(currentSelectionTitle) • \(formatter.string(from: Date()))"
    }

    private func bindPersistence() {
        let publishers: [AnyPublisher<Void, Never>] = [
            $projectRoot.dropFirst().map { _ in () }.eraseToAnyPublisher(),
            $connectionState.dropFirst().map { _ in () }.eraseToAnyPublisher(),
            $autoReconnectEnabled.dropFirst().map { _ in () }.eraseToAnyPublisher(),
            $selection.dropFirst().map { _ in () }.eraseToAnyPublisher(),
            $selectedRole.dropFirst().map { _ in () }.eraseToAnyPublisher(),
            $selectedRuntimeSymbol.dropFirst().map { _ in () }.eraseToAnyPublisher(),
            $selectedRatesSymbol.dropFirst().map { _ in () }.eraseToAnyPublisher(),
            $selectedCrossAssetSymbol.dropFirst().map { _ in () }.eraseToAnyPublisher(),
            $selectedPairNetworkSymbol.dropFirst().map { _ in () }.eraseToAnyPublisher(),
            $selectedMicrostructureSymbol.dropFirst().map { _ in () }.eraseToAnyPublisher(),
            $selectedAdaptiveSymbol.dropFirst().map { _ in () }.eraseToAnyPublisher(),
            $selectedDynamicEnsembleSymbol.dropFirst().map { _ in () }.eraseToAnyPublisher(),
            $selectedProbCalibrationSymbol.dropFirst().map { _ in () }.eraseToAnyPublisher(),
            $selectedExecutionQualitySymbol.dropFirst().map { _ in () }.eraseToAnyPublisher(),
            $selectedLabelEngineDatasetKey.dropFirst().map { _ in () }.eraseToAnyPublisher(),
            $selectedResearchSymbol.dropFirst().map { _ in () }.eraseToAnyPublisher(),
            $selectedVisualizationSymbol.dropFirst().map { _ in () }.eraseToAnyPublisher(),
            $overviewLayout.dropFirst().map { _ in () }.eraseToAnyPublisher(),
            $roleWorkspaceLayouts.dropFirst().map { _ in () }.eraseToAnyPublisher(),
            $pluginSearchText.dropFirst().map { _ in () }.eraseToAnyPublisher(),
            $selectedPluginFamily.dropFirst().map { _ in () }.eraseToAnyPublisher(),
            $reportCategoryFilter.dropFirst().map { _ in () }.eraseToAnyPublisher(),
            $auditDraft.dropFirst().map { _ in () }.eraseToAnyPublisher(),
            $backtestDraft.dropFirst().map { _ in () }.eraseToAnyPublisher(),
            $offlineDraft.dropFirst().map { _ in () }.eraseToAnyPublisher(),
            $researchBranchDraft.dropFirst().map { _ in () }.eraseToAnyPublisher(),
            $researchAuditDraft.dropFirst().map { _ in () }.eraseToAnyPublisher(),
            $researchVectorDraft.dropFirst().map { _ in () }.eraseToAnyPublisher(),
            $researchRecoveryDraft.dropFirst().map { _ in () }.eraseToAnyPublisher(),
            $savedViews.dropFirst().map { _ in () }.eraseToAnyPublisher(),
            $completedOnboardingRoles.dropFirst().map { _ in () }.eraseToAnyPublisher()
        ]

        Publishers.MergeMany(publishers)
            .debounce(for: .milliseconds(180), scheduler: DispatchQueue.main)
            .sink { [weak self] in
                self?.persistState()
            }
            .store(in: &cancellables)
    }

    private func bindResourceManagement() {
        resourceMonitor.$profile
            .receive(on: DispatchQueue.main)
            .sink { [weak self] profile in
                guard let self else { return }
                self.renderingProfile = profile
                self.configureConnectionTimer()
            }
            .store(in: &cancellables)

        $autoReconnectEnabled
            .dropFirst()
            .sink { [weak self] _ in
                self?.configureConnectionTimer()
            }
            .store(in: &cancellables)

        configureConnectionTimer()
    }

    private func configureConnectionTimer() {
        connectionTimerCancellable?.cancel()
        connectionTimerCancellable = nil

        guard autoReconnectEnabled else { return }

        connectionTimerCancellable = Timer.publish(
            every: renderingProfile.softReconnectInterval,
            on: .main,
            in: .common
        )
        .autoconnect()
        .sink { [weak self] _ in
            guard let self else { return }
            Task {
                await self.performSoftConnectionCheck(forceRefresh: false)
            }
        }
    }

    private func performSoftConnectionCheck(forceRefresh: Bool) async {
        guard forceRefresh || renderingProfile.allowsPeriodicReconnectChecks else {
            return
        }

        lastConnectionCheckAt = Date()

        let resolution = connectionCoordinator.resolve(
            currentProjectRoot: projectRoot,
            preferredProjectRoot: preferredProjectRoot,
            autoReconnectEnabled: autoReconnectEnabled
        )

        preferredProjectRoot = resolution.preferredProjectRoot
        connectionState = resolution.state

        guard let resolvedRoot = resolution.activeProjectRoot else {
            if projectRoot != nil {
                projectRoot = nil
            }
            clearSnapshots()
            lastErrorMessage = nil
            return
        }

        let rootChanged = projectRoot?.standardizedFileURL != resolvedRoot.standardizedFileURL
        if rootChanged {
            projectRoot = resolvedRoot
        }

        if forceRefresh || rootChanged || snapshot == nil {
            await refresh()
        }
    }

    private func persistState() {
        guard !isRestoringPersistedState else { return }

        let state = FXAIGUIPersistenceState(
            savedViews: savedViews,
            lastWorkspace: currentWorkspaceSnapshot(name: "Last Session"),
            completedOnboardingRoles: completedOnboardingRoles.sorted { $0.rawValue < $1.rawValue },
            preferredProjectRootPath: preferredProjectRoot?.path,
            autoReconnectEnabled: autoReconnectEnabled
        )

        do {
            try workspaceStore.save(state)
        } catch {
            lastErrorMessage = error.localizedDescription
        }
    }

    private func loadPersistedState() {
        isRestoringPersistedState = true
        defer { isRestoringPersistedState = false }

        let state = workspaceStore.load()
        savedViews = state.savedViews.sorted { lhs, rhs in
            if lhs.updatedAt == rhs.updatedAt {
                return lhs.name.localizedCaseInsensitiveCompare(rhs.name) == .orderedAscending
            }
            return lhs.updatedAt > rhs.updatedAt
        }
        completedOnboardingRoles = Set(state.completedOnboardingRoles)
        autoReconnectEnabled = state.autoReconnectEnabled
        if let preferredProjectRootPath = state.preferredProjectRootPath, !preferredProjectRootPath.isEmpty {
            preferredProjectRoot = URL(fileURLWithPath: preferredProjectRootPath, isDirectory: true)
        }

        if let workspace = state.lastWorkspace {
            if autoReconnectEnabled {
                if let path = workspace.projectRootPath, !path.isEmpty {
                    projectRoot = URL(fileURLWithPath: path, isDirectory: true)
                    preferredProjectRoot = projectRoot
                }
            } else {
                projectRoot = nil
            }
            selection = SidebarDestination(rawValue: workspace.selection) ?? .overview
            selectedRole = workspace.selectedRole
            selectedRuntimeSymbol = workspace.selectedRuntimeSymbol
            selectedRatesSymbol = workspace.selectedRatesSymbol
            selectedCrossAssetSymbol = workspace.selectedCrossAssetSymbol
            selectedPairNetworkSymbol = workspace.selectedPairNetworkSymbol
            selectedMicrostructureSymbol = workspace.selectedMicrostructureSymbol
            selectedAdaptiveSymbol = workspace.selectedAdaptiveSymbol
            selectedDynamicEnsembleSymbol = workspace.selectedDynamicEnsembleSymbol
            selectedProbCalibrationSymbol = workspace.selectedProbCalibrationSymbol
            selectedExecutionQualitySymbol = workspace.selectedExecutionQualitySymbol
            selectedLabelEngineDatasetKey = workspace.selectedLabelEngineDatasetKey
            selectedResearchSymbol = workspace.selectedResearchSymbol
            selectedVisualizationSymbol = workspace.selectedVisualizationSymbol
            overviewLayout = workspace.overviewLayout.normalized()
            roleWorkspaceLayouts = workspace.roleWorkspaceLayouts.normalized()
            pluginSearchText = workspace.pluginSearchText
            selectedPluginFamily = workspace.selectedPluginFamily
            reportCategoryFilter = workspace.reportCategoryFilter
            auditDraft = workspace.auditDraft
            backtestDraft = workspace.backtestDraft
            offlineDraft = workspace.offlineDraft
            researchBranchDraft = workspace.researchBranchDraft
            researchAuditDraft = workspace.researchAuditDraft
            researchVectorDraft = workspace.researchVectorDraft
            researchRecoveryDraft = workspace.researchRecoveryDraft
        }
    }

    private func clearSnapshots() {
        snapshot = nil
        runtimeSnapshot = nil
        newsPulseSnapshot = nil
        ratesEngineSnapshot = nil
        crossAssetSnapshot = nil
        pairNetworkSnapshot = nil
        microstructureSnapshot = nil
        adaptiveRouterSnapshot = nil
        driftGovernanceSnapshot = nil
        dynamicEnsembleSnapshot = nil
        probCalibrationSnapshot = nil
        executionQualitySnapshot = nil
        labelEngineSnapshot = nil
        researchSnapshot = nil
        visualizationSnapshot = nil
        incidentSnapshot = nil
        selectedIncidentID = nil
    }

    private func currentWorkspaceSnapshot(
        name: String,
        id: UUID = UUID(),
        createdAt: Date = Date(),
        updatedAt: Date = Date()
    ) -> SavedWorkspaceView {
        SavedWorkspaceView(
            id: id,
            name: name,
            projectRootPath: projectRoot?.path,
            createdAt: createdAt,
            updatedAt: updatedAt,
            selection: selection?.rawValue ?? SidebarDestination.overview.rawValue,
            selectedRole: selectedRole,
            selectedRuntimeSymbol: selectedRuntimeSymbol,
            selectedRatesSymbol: selectedRatesSymbol,
            selectedCrossAssetSymbol: selectedCrossAssetSymbol,
            selectedPairNetworkSymbol: selectedPairNetworkSymbol,
            selectedMicrostructureSymbol: selectedMicrostructureSymbol,
            selectedAdaptiveSymbol: selectedAdaptiveSymbol,
            selectedDynamicEnsembleSymbol: selectedDynamicEnsembleSymbol,
            selectedProbCalibrationSymbol: selectedProbCalibrationSymbol,
            selectedExecutionQualitySymbol: selectedExecutionQualitySymbol,
            selectedLabelEngineDatasetKey: selectedLabelEngineDatasetKey,
            selectedResearchSymbol: selectedResearchSymbol,
            selectedVisualizationSymbol: selectedVisualizationSymbol,
            pluginSearchText: pluginSearchText,
            selectedPluginFamily: selectedPluginFamily,
            reportCategoryFilter: reportCategoryFilter,
            auditDraft: auditDraft,
            backtestDraft: backtestDraft,
            offlineDraft: offlineDraft,
            researchBranchDraft: researchBranchDraft,
            researchAuditDraft: researchAuditDraft,
            researchVectorDraft: researchVectorDraft,
            researchRecoveryDraft: researchRecoveryDraft,
            overviewLayout: overviewLayout.normalized(),
            roleWorkspaceLayouts: roleWorkspaceLayouts.normalized()
        )
    }

    private func updateOverviewLayout(_ mutate: (inout OverviewDashboardLayoutState) -> Void) {
        var updated = overviewLayout
        mutate(&updated)
        overviewLayout = updated.normalized()
    }

    private func updateRoleWorkspaceLayout(
        for kind: RoleWorkspaceDashboardKind,
        _ mutate: (inout RoleWorkspaceDashboardLayoutState) -> Void
    ) {
        var layouts = roleWorkspaceLayouts
        var updated = layouts.layout(for: kind)
        mutate(&updated)
        layouts.setLayout(updated.normalized(), for: kind)
        roleWorkspaceLayouts = layouts.normalized()
    }
}
