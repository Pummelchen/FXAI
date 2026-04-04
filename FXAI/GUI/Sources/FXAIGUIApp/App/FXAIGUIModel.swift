import AppKit
import FXAIGUICore
import Foundation

@MainActor
final class FXAIGUIModel: ObservableObject {
    @Published var projectRoot: URL?
    @Published var snapshot: FXAIProjectSnapshot?
    @Published var selection: SidebarDestination? = .overview
    @Published var selectedRole: WorkspaceRole = .liveTrader
    @Published var pluginSearchText = ""
    @Published var selectedPluginFamily = "All"
    @Published var reportCategoryFilter = "All"
    @Published var lastErrorMessage: String?
    @Published var isRefreshing = false

    private let scanner = ProjectScanner()

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

    var projectPathLabel: String {
        projectRoot?.path ?? "No FXAI project selected"
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
            lastErrorMessage = nil
        } catch {
            snapshot = nil
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
}
