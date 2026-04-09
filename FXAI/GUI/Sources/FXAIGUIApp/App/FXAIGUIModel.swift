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
    @Published var microstructureSnapshot: MicrostructureSnapshot?
    @Published var adaptiveRouterSnapshot: AdaptiveRouterSnapshot?
    @Published var researchSnapshot: ResearchOSControlSnapshot?
    @Published var visualizationSnapshot: AdvancedVisualizationSnapshot?
    @Published var incidentSnapshot: IncidentCenterSnapshot?
    @Published var selection: SidebarDestination? = .overview
    @Published var selectedRole: WorkspaceRole = .liveTrader
    @Published var selectedRuntimeSymbol = ""
    @Published var selectedRatesSymbol = ""
    @Published var selectedMicrostructureSymbol = ""
    @Published var selectedAdaptiveSymbol = ""
    @Published var selectedResearchSymbol = ""
    @Published var selectedVisualizationSymbol = ""
    @Published var selectedIncidentID: String?
    @Published var overviewLayout = OverviewDashboardLayoutState.default()
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
    private let microstructureReader = MicrostructureArtifactReader()
    private let adaptiveRouterReader = AdaptiveRouterArtifactReader()
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
            microstructureSnapshot = microstructureReader.read(projectRoot: projectRoot)
            adaptiveRouterSnapshot = adaptiveRouterReader.read(projectRoot: projectRoot)
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
            syncMicrostructureSelection()
            syncAdaptiveSelection()
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
            self.selectedMicrostructureSymbol = savedView.selectedMicrostructureSymbol
            self.selectedAdaptiveSymbol = savedView.selectedAdaptiveSymbol
            self.selectedResearchSymbol = savedView.selectedResearchSymbol
            self.selectedVisualizationSymbol = savedView.selectedVisualizationSymbol
            self.overviewLayout = savedView.overviewLayout.normalized()
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
            self.syncAdaptiveSelection()
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
            $selectedMicrostructureSymbol.dropFirst().map { _ in () }.eraseToAnyPublisher(),
            $selectedAdaptiveSymbol.dropFirst().map { _ in () }.eraseToAnyPublisher(),
            $selectedResearchSymbol.dropFirst().map { _ in () }.eraseToAnyPublisher(),
            $selectedVisualizationSymbol.dropFirst().map { _ in () }.eraseToAnyPublisher(),
            $overviewLayout.dropFirst().map { _ in () }.eraseToAnyPublisher(),
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
            selectedMicrostructureSymbol = workspace.selectedMicrostructureSymbol
            selectedAdaptiveSymbol = workspace.selectedAdaptiveSymbol
            selectedResearchSymbol = workspace.selectedResearchSymbol
            selectedVisualizationSymbol = workspace.selectedVisualizationSymbol
            overviewLayout = workspace.overviewLayout.normalized()
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
        microstructureSnapshot = nil
        adaptiveRouterSnapshot = nil
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
            selectedMicrostructureSymbol: selectedMicrostructureSymbol,
            selectedAdaptiveSymbol: selectedAdaptiveSymbol,
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
            overviewLayout: overviewLayout.normalized()
        )
    }

    private func updateOverviewLayout(_ mutate: (inout OverviewDashboardLayoutState) -> Void) {
        var updated = overviewLayout
        mutate(&updated)
        overviewLayout = updated.normalized()
    }
}
