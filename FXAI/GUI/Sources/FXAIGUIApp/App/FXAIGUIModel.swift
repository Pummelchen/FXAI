import AppKit
import Combine
import FXAIGUICore
import Foundation

@MainActor
final class FXAIGUIModel: ObservableObject {
    @Published var projectRoot: URL?
    @Published var snapshot: FXAIProjectSnapshot?
    @Published var runtimeSnapshot: RuntimeOperationsSnapshot?
    @Published var researchSnapshot: ResearchOSControlSnapshot?
    @Published var visualizationSnapshot: AdvancedVisualizationSnapshot?
    @Published var incidentSnapshot: IncidentCenterSnapshot?
    @Published var selection: SidebarDestination? = .overview
    @Published var selectedRole: WorkspaceRole = .liveTrader
    @Published var selectedRuntimeSymbol = ""
    @Published var selectedResearchSymbol = ""
    @Published var selectedVisualizationSymbol = ""
    @Published var selectedIncidentID: String?
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

    private let scanner = ProjectScanner()
    private let runtimeReader = RuntimeArtifactReader()
    private let researchReader = ResearchOSArtifactReader()
    private let visualizationBuilder = AdvancedVisualizationBuilder()
    private let incidentBuilder = IncidentBuilder()
    private let workspaceStore = SavedWorkspaceStore()
    private var cancellables: Set<AnyCancellable> = []
    private var isRestoringPersistedState = false

    init() {
        loadPersistedState()
        if projectRoot == nil {
            projectRoot = ProjectPathResolver.defaultProjectRoot()
        }
        bindPersistence()

        Task {
            await refresh()
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
        projectRoot?.path ?? "No FXAI project selected"
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
        guard let projectRoot else {
            snapshot = nil
            researchSnapshot = nil
            incidentSnapshot = nil
            lastErrorMessage = "Select an FXAI project root to load the dashboard."
            return
        }

        isRefreshing = true
        defer { isRefreshing = false }

        do {
            snapshot = try scanner.scan(projectRoot: projectRoot)
            runtimeSnapshot = runtimeReader.read(projectRoot: projectRoot)
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
                researchSnapshot: researchSnapshot
            )
            syncBuilderDefaults()
            syncRuntimeSelection()
            syncResearchSelection()
            syncVisualizationSelection()
            syncIncidentSelection()
            lastErrorMessage = nil
        } catch {
            snapshot = nil
            runtimeSnapshot = nil
            researchSnapshot = nil
            visualizationSnapshot = nil
            incidentSnapshot = nil
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

        projectRoot = url
        Task {
            await refresh()
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
            self.selectedResearchSymbol = savedView.selectedResearchSymbol
            self.selectedVisualizationSymbol = savedView.selectedVisualizationSymbol
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
            $selection.dropFirst().map { _ in () }.eraseToAnyPublisher(),
            $selectedRole.dropFirst().map { _ in () }.eraseToAnyPublisher(),
            $selectedRuntimeSymbol.dropFirst().map { _ in () }.eraseToAnyPublisher(),
            $selectedResearchSymbol.dropFirst().map { _ in () }.eraseToAnyPublisher(),
            $selectedVisualizationSymbol.dropFirst().map { _ in () }.eraseToAnyPublisher(),
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

    private func persistState() {
        guard !isRestoringPersistedState else { return }

        let state = FXAIGUIPersistenceState(
            savedViews: savedViews,
            lastWorkspace: currentWorkspaceSnapshot(name: "Last Session"),
            completedOnboardingRoles: completedOnboardingRoles.sorted { $0.rawValue < $1.rawValue }
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

        if let workspace = state.lastWorkspace {
            if let path = workspace.projectRootPath, !path.isEmpty {
                projectRoot = URL(fileURLWithPath: path, isDirectory: true)
            }
            selection = SidebarDestination(rawValue: workspace.selection) ?? .overview
            selectedRole = workspace.selectedRole
            selectedRuntimeSymbol = workspace.selectedRuntimeSymbol
            selectedResearchSymbol = workspace.selectedResearchSymbol
            selectedVisualizationSymbol = workspace.selectedVisualizationSymbol
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
            researchRecoveryDraft: researchRecoveryDraft
        )
    }
}
