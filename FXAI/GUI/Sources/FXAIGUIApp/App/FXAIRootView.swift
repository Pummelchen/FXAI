import FXAIGUICore
import SwiftUI

struct FXAIRootView: View {
    @EnvironmentObject private var model: FXAIGUIModel

    var body: some View {
        NavigationSplitView {
            sidebar
        } detail: {
            detailView
                .padding(20)
        }
        .navigationSplitViewStyle(.balanced)
        .toolbar {
            ToolbarItemGroup(placement: .primaryAction) {
                Button {
                    Task { await model.refresh() }
                } label: {
                    Label("Refresh", systemImage: "arrow.clockwise")
                }
                .help("Refresh FXAI project state")

                Button {
                    model.chooseProjectRoot()
                } label: {
                    Label("Choose Project", systemImage: "folder")
                }
                .help("Choose a different FXAI project root")
            }
        }
    }

    private var sidebar: some View {
        List(selection: $model.selection) {
            Section("Workspace") {
                ForEach(SidebarDestination.allCases) { destination in
                    Label(destination.title, systemImage: destination.symbolName)
                        .tag(Optional(destination))
                }
            }

            Section("Project") {
                VStack(alignment: .leading, spacing: 8) {
                    Text("Active Root")
                        .font(.caption.weight(.semibold))
                        .foregroundStyle(FXAITheme.textSecondary)
                    Text(model.projectPathLabel)
                        .font(.caption)
                        .foregroundStyle(FXAITheme.textMuted)
                        .lineLimit(4)
                }
                .padding(.vertical, 6)
            }
        }
        .scrollContentBackground(.hidden)
        .background(FXAITheme.backgroundSecondary.opacity(0.6))
        .navigationSplitViewColumnWidth(min: 220, ideal: 250, max: 280)
    }

    @ViewBuilder
    private var detailView: some View {
        switch model.selection ?? .overview {
        case .overview:
            OverviewDashboardView()
        case .roles:
            RoleWorkspacesView()
        case .plugins:
            PluginZooView()
        case .reports:
            ReportsExplorerView()
        case .commands:
            CommandCenterView()
        case .settings:
            SettingsView()
        }
    }
}
