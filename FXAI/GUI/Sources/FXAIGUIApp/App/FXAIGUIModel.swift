import AppKit
import FXAIGUICore
import Foundation

@MainActor
final class FXAIGUIModel: ObservableObject {
    @Published var projectRoot: URL?
    @Published var snapshot: FXAIProjectSnapshot?
    @Published var runtimeSnapshot: RuntimeOperationsSnapshot?
    @Published var selection: SidebarDestination? = .overview
    @Published var selectedRole: WorkspaceRole = .liveTrader
    @Published var selectedRuntimeSymbol = ""
    @Published var pluginSearchText = ""
    @Published var selectedPluginFamily = "All"
    @Published var reportCategoryFilter = "All"
    @Published var auditDraft = AuditLabDraft()
    @Published var backtestDraft = BacktestBuilderDraft()
    @Published var offlineDraft = OfflineLabDraft()
    @Published var lastErrorMessage: String?
    @Published var isRefreshing = false

    private let scanner = ProjectScanner()
    private let runtimeReader = RuntimeArtifactReader()

    init() {
        self.projectRoot = ProjectPathResolver.defaultProjectRoot()

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

    func refresh() async {
        guard let projectRoot else {
            snapshot = nil
            lastErrorMessage = "Select an FXAI project root to load the dashboard."
            return
        }

        isRefreshing = true
        defer { isRefreshing = false }

        do {
            snapshot = try scanner.scan(projectRoot: projectRoot)
            runtimeSnapshot = runtimeReader.read(projectRoot: projectRoot)
            syncBuilderDefaults()
            syncRuntimeSelection()
            lastErrorMessage = nil
        } catch {
            snapshot = nil
            runtimeSnapshot = nil
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
}
