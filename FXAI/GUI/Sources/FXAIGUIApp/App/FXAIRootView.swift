import FXAIGUICore
import SwiftUI

struct FXAIRootView: View {
    @EnvironmentObject private var model: FXAIGUIModel
    @EnvironmentObject private var themeEnvironment: ThemeEnvironment
    private let sidebarSections: [(title: String, destinations: [SidebarDestination])] = [
        ("Start", [.overview, .roles, .onboarding, .incidents]),
        ("Build", [.auditLab, .backtestBuilder, .offlineLab]),
        ("Operate", [.newsPulse, .ratesEngine, .microstructure, .adaptiveRouter, .dynamicEnsemble, .probCalibration, .runtimeMonitor, .promotionCenter, .researchControl]),
        ("Inspect", [.plugins, .reports, .commands, .advancedVisuals]),
        ("System", [.settings])
    ]

    var body: some View {
        NavigationSplitView {
            sidebar
        } detail: {
            detailView
                .padding(20)
                .background(FXAIBackgroundView().opacity(0.92))
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
                    if model.projectRoot == nil {
                        model.reconnectProject()
                    } else {
                        model.chooseProjectRoot()
                    }
                } label: {
                    Label(model.projectRoot == nil ? "Connect" : "Choose Project", systemImage: model.projectRoot == nil ? "link.circle.fill" : "folder")
                }
                .help(model.projectRoot == nil ? "Reconnect or connect to an FXAI project" : "Choose a different FXAI project root")

                if model.projectRoot != nil || model.connectionState == .disconnectedByUser {
                    Button {
                        if model.projectRoot != nil {
                            model.disconnectProject()
                        } else {
                            model.reconnectProject()
                        }
                    } label: {
                        Label(model.projectRoot != nil ? "Disconnect" : "Reconnect", systemImage: model.projectRoot != nil ? "eject.circle.fill" : "arrow.triangle.2.circlepath.circle.fill")
                    }
                    .help(model.projectRoot != nil ? "Disconnect the GUI from the current FXAI project" : "Reconnect the GUI to the preferred FXAI project")
                }

                Button {
                    model.saveCurrentView()
                } label: {
                    Label("Save View", systemImage: "bookmark.fill")
                }
                .help("Save the current GUI state as a reusable workspace view")

                if model.selection == .overview {
                    Button {
                        model.resetOverviewLayout()
                    } label: {
                        Label("Reset Dashboard", systemImage: "rectangle.3.group.bubble.left.fill")
                    }
                    .help("Reset the overview dashboard layout to the shipped default")
                }

                Menu {
                    ForEach(themeEnvironment.allThemes, id: \.themeID) { theme in
                        Button(theme.displayName) {
                            themeEnvironment.activateTheme(theme.themeID)
                        }
                    }
                } label: {
                    Label("Theme", systemImage: "paintpalette.fill")
                }
                .help("Switch the active GUI theme")

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
            ForEach(sidebarSections, id: \.title) { section in
                Section(section.title) {
                    ForEach(section.destinations) { destination in
                        sidebarLabel(for: destination)
                            .tag(Optional(destination))
                    }
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
                    Text("Connection")
                        .font(.caption.weight(.semibold))
                        .foregroundStyle(FXAITheme.textSecondary)
                    HStack {
                        Circle()
                            .fill(connectionColor)
                            .frame(width: 8, height: 8)
                        Text(model.connectionStatusLabel)
                            .font(.caption.weight(.semibold))
                            .foregroundStyle(FXAITheme.textPrimary)
                    }
                    Text(model.projectPathLabel)
                        .font(.caption)
                        .foregroundStyle(FXAITheme.textMuted)
                        .lineLimit(4)
                }
                .padding(.vertical, 6)
            }
        }
        .scrollContentBackground(.hidden)
        .background(
            FXAIGlassRoundedBackground(cornerRadius: 24, style: .sidebar, tint: FXAITheme.accentSoft.opacity(0.10))
                .padding(.vertical, 8)
                .padding(.leading, 8)
                .padding(.trailing, 6)
        )
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
        case .newsPulse:
            NewsPulseView()
        case .ratesEngine:
            RatesEngineView()
        case .microstructure:
            MicrostructureView()
        case .adaptiveRouter:
            AdaptiveRouterView()
        case .dynamicEnsemble:
            DynamicEnsembleView()
        case .probCalibration:
            ProbCalibrationView()
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

    private var connectionColor: Color {
        switch model.connectionState {
        case .connected:
            FXAITheme.success
        case .waitingForProject:
            FXAITheme.warning
        case .disconnectedByUser:
            FXAITheme.textMuted
        }
    }
}
