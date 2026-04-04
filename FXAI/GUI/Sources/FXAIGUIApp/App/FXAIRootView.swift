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

                Button {
                    model.saveCurrentView()
                } label: {
                    Label("Save View", systemImage: "bookmark.fill")
                }
                .help("Save the current GUI state as a reusable workspace view")

                Button {
                    model.navigate(to: .incidents)
                } label: {
                    Label("Incidents", systemImage: "exclamationmark.triangle.fill")
                }
                .help("Open the incident and recovery surface")
            }
        }
    }

    private var sidebar: some View {
        List(selection: $model.selection) {
            Section("Workspace") {
                ForEach(SidebarDestination.allCases) { destination in
                    sidebarLabel(for: destination)
                        .tag(Optional(destination))
                }
            }

            if !model.savedViews.isEmpty {
                Section("Saved Views") {
                    ForEach(model.savedViews.prefix(8)) { savedView in
                        HStack(spacing: 8) {
                            Button {
                                model.applySavedView(savedView)
                            } label: {
                                VStack(alignment: .leading, spacing: 3) {
                                    Text(savedView.name)
                                        .font(.subheadline.weight(.semibold))
                                        .foregroundStyle(FXAITheme.textPrimary)
                                    Text(savedView.titleSummary)
                                        .font(.caption)
                                        .foregroundStyle(FXAITheme.textMuted)
                                        .lineLimit(1)
                                }
                                .frame(maxWidth: .infinity, alignment: .leading)
                            }
                            .buttonStyle(.plain)

                            Button {
                                model.deleteSavedView(savedView)
                            } label: {
                                Image(systemName: "trash")
                                    .foregroundStyle(FXAITheme.textMuted)
                            }
                            .buttonStyle(.borderless)
                            .help("Delete saved view")
                        }
                        .padding(.vertical, 3)
                    }
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
        case .onboarding:
            OnboardingGuideView()
        case .roles:
            RoleWorkspacesView()
        case .incidents:
            IncidentCenterView()
        case .auditLab:
            AuditLabBuilderView()
        case .backtestBuilder:
            BacktestBuilderView()
        case .offlineLab:
            OfflineLabBuilderView()
        case .runtimeMonitor:
            RuntimeMonitorView()
        case .promotionCenter:
            PromotionCenterView()
        case .researchControl:
            ResearchOSControlView()
        case .advancedVisuals:
            AdvancedVisualizationView()
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

    @ViewBuilder
    private func sidebarLabel(for destination: SidebarDestination) -> some View {
        HStack {
            Label(destination.title, systemImage: destination.symbolName)
            if destination == .incidents, let incidentSnapshot = model.incidentSnapshot, !incidentSnapshot.incidents.isEmpty {
                Spacer()
                Text("\(incidentSnapshot.incidents.count)")
                    .font(.caption2.weight(.semibold))
                    .foregroundStyle(FXAITheme.warning)
                    .padding(.horizontal, 6)
                    .padding(.vertical, 2)
                    .background(
                        Capsule(style: .continuous)
                            .fill(FXAITheme.warning.opacity(0.14))
                    )
            }
        }
    }
}
