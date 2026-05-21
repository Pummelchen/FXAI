import Charts
import FXAIGUICore
import SwiftUI

struct OverviewDashboardView: View {
    @EnvironmentObject private var model: FXAIGUIModel
    @EnvironmentObject private var themeEnvironment: ThemeEnvironment
    @State private var customizationEnabled = false

    var body: some View {
        GeometryReader { geometry in
            ScrollView {
                VStack(alignment: .leading, spacing: 18) {
                    SectionHeader(
                        title: "FXAI Overview",
                        subtitle: "See project health, promoted runtime state, research outputs, and the next operator action at a glance."
                    )

                    overviewControls

                    if let snapshot = model.snapshot {
                        let sectionWidth = max(360, geometry.size.width - 8)
                        ForEach(model.overviewLayout.sections) { section in
                            let widgets = visibleWidgets(in: section, snapshot: snapshot)
                            if customizationEnabled || !widgets.isEmpty {
                                OverviewDashboardSectionView(
                                    section: section,
                                    availableWidth: sectionWidth,
                                    customizationEnabled: customizationEnabled,
                                    widgets: widgets
                                ) { widget in
                                    AnyView(widgetContent(for: widget.kind, snapshot: snapshot))
                                }
                            }
                        }
                    } else {
                        EmptyStateView(
                            title: emptyStateTitle,
                            message: emptyStateMessage,
                            symbolName: emptyStateSymbol
                        )
                    }
                }
                .padding(.bottom, 22)
            }
        }
        .scrollContentBackground(.hidden)
    }

    private var overviewControls: some View {
        FXAIVisualEffectSurface(style: .card, cornerRadius: 20, contentPadding: 14, tint: FXAITheme.accent.opacity(0.08)) {
            ViewThatFits(in: .horizontal) {
                HStack(alignment: .center, spacing: 16) {
                    controlSummary
                    Spacer(minLength: 12)
                    controlButtons
                }
                VStack(alignment: .leading, spacing: 12) {
                    controlSummary
                    controlButtons
                }
            }
        }
    }

    private var controlSummary: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("Dynamic Dashboard")
                .font(.headline)
                .foregroundStyle(FXAITheme.textPrimary)
            Text("Move and resize widgets on a 1 cm snap grid, drag the handle chip to reposition them, and the GUI saves every change automatically.")
                .font(.subheadline)
                .foregroundStyle(FXAITheme.textSecondary)
                .fixedSize(horizontal: false, vertical: true)
            Text("Grid unit: 1 cm snap (\(Int(model.overviewLayout.gridUnitPoints.rounded())) pt base) • Sidebar stays pinned • Reset restores the shipped layout")
                .font(.caption)
                .foregroundStyle(FXAITheme.textMuted)
        }
    }

    private var controlButtons: some View {
        Group {
            Button(customizationEnabled ? "Done Customizing" : "Customize Dashboard") {
                customizationEnabled.toggle()
            }
            .buttonStyle(.borderedProminent)
            .tint(FXAITheme.accent)

            Button("Reset Layout") {
                model.resetOverviewLayout()
            }
            .buttonStyle(.bordered)
        }
    }

    private func hero(snapshot: FXAIProjectSnapshot) -> some View {
        FXAIVisualEffectSurface(style: .hero, cornerRadius: 26, contentPadding: 18, tint: FXAITheme.accent.opacity(0.10)) {
            VStack(alignment: .leading, spacing: 18) {
                HStack(alignment: .top, spacing: 18) {
                    VStack(alignment: .leading, spacing: 8) {
                        Text("FXAI Operator GUI")
                            .font(.caption.weight(.semibold))
                            .foregroundStyle(FXAITheme.textSecondary)

                        Text("Terminal-first control and visibility for the full FXAI stack.")
                            .font(.system(size: 30, weight: .semibold, design: .rounded))
                            .foregroundStyle(FXAITheme.textPrimary)

                        Text("Project root: \(snapshot.projectRoot.path)")
                            .font(.subheadline)
                            .foregroundStyle(FXAITheme.textSecondary)
                            .lineLimit(2)
                    }

                    Spacer(minLength: 12)

                    VStack(alignment: .trailing, spacing: 12) {
                        StatusBadge(
                            title: "Research DB",
                            value: snapshot.tursoSummary.embeddedReplicaConfigured ? "Embedded Replica" : "Local Turso",
                            tint: snapshot.tursoSummary.embeddedReplicaConfigured ? FXAITheme.accent : FXAITheme.accentSoft
                        )
                    StatusBadge(
                        title: "Updated",
                        value: FXAIFormatting.dateTimeString(for: snapshot.generatedAt),
                        tint: FXAITheme.warning
                    )
                    StatusBadge(
                        title: "Theme",
                        value: themeEnvironment.currentTheme.displayName,
                        tint: FXAITheme.accentSoft
                    )
                }
                }

                Rectangle()
                    .fill(FXAITheme.stroke)
                    .frame(height: 1)

                ViewThatFits(in: .horizontal) {
                    HStack(spacing: 18) {
                        heroStatusBadges(snapshot: snapshot)
                    }
                    VStack(alignment: .leading, spacing: 12) {
                        heroStatusBadges(snapshot: snapshot)
                    }
                }
            }
            .padding(8)
            .background(
                RoundedRectangle(cornerRadius: 26, style: .continuous)
                    .fill(FXAITheme.heroGradient.opacity(0.55))
            )
        }
    }

    @ViewBuilder
    private func metricCard(for kind: OverviewDashboardWidgetKind, snapshot: FXAIProjectSnapshot) -> some View {
        switch kind {
        case .buildTargetsMetric:
            MetricCard(
                title: "Build Targets",
                value: "\(snapshot.cleanBuildTargetCount)/\(snapshot.buildTargets.count)",
                footnote: "MT5 outputs currently present in the project tree.",
                symbolName: "hammer.fill",
                tint: FXAITheme.warning
            )
        case .pluginsMetric:
            MetricCard(
                title: "Plugins",
                value: "\(snapshot.totalPluginCount)",
                footnote: "Discovered across \(snapshot.pluginFamilies.count) families.",
                symbolName: "shippingbox.fill",
                tint: FXAITheme.accent
            )
        case .artifactsMetric:
            MetricCard(
                title: "Artifacts",
                value: "\(snapshot.totalReportCount)",
                footnote: "Baselines, ResearchOS, profiles, and distillation outputs.",
                symbolName: "tray.full.fill",
                tint: FXAITheme.accentSoft
            )
        case .runtimeProfilesMetric:
            MetricCard(
                title: "Runtime Profiles",
                value: "\(snapshot.runtimeProfiles.count)",
                footnote: "Current deployment payloads discovered under ResearchOS.",
                symbolName: "waveform.path.ecg",
                tint: FXAITheme.success
            )
        case .incidentsMetric:
            MetricCard(
                title: "Incidents",
                value: "\(model.incidentSnapshot?.incidents.count ?? 0)",
                footnote: "Generated operator issues with guided recovery steps.",
                symbolName: "exclamationmark.triangle.fill",
                tint: (model.incidentSnapshot?.incidents.isEmpty == false) ? FXAITheme.warning : FXAITheme.success
            )
        case .savedViewsMetric:
            MetricCard(
                title: "Saved Views",
                value: "\(model.savedViews.count)",
                footnote: "Reusable GUI workspace states for repeatable operator flows.",
                symbolName: "bookmark.fill",
                tint: FXAITheme.accentSoft
            )
        default:
            EmptyView()
        }
    }

    private func pluginChart(snapshot: FXAIProjectSnapshot) -> some View {
        FXAIVisualEffectSurface {
            VStack(alignment: .leading, spacing: 14) {
                Text("Plugin Family Footprint")
                    .font(.headline)
                    .foregroundStyle(FXAITheme.textPrimary)

                Chart(snapshot.pluginFamilies) { family in
                    BarMark(
                        x: .value("Family", family.family),
                        y: .value("Plugins", family.pluginCount)
                    )
                    .foregroundStyle(FXAITheme.accent.gradient)
                    .cornerRadius(6)
                }
                .chartYAxis {
                    AxisMarks(position: .leading)
                }
                .frame(minHeight: 240)
            }
        }
        .frame(maxWidth: .infinity)
    }

    private func reportChart(snapshot: FXAIProjectSnapshot) -> some View {
        FXAIVisualEffectSurface {
            VStack(alignment: .leading, spacing: 14) {
                Text("Artifact Surface")
                    .font(.headline)
                    .foregroundStyle(FXAITheme.textPrimary)

                Chart(snapshot.reportCategories) { category in
                    BarMark(
                        x: .value("Category", category.category),
                        y: .value("Files", category.fileCount)
                    )
                    .foregroundStyle(FXAITheme.accentSoft.gradient)
                    .cornerRadius(6)
                }
                .chartYAxis {
                    AxisMarks(position: .leading)
                }
                .frame(minHeight: 240)
            }
        }
        .frame(maxWidth: .infinity)
    }

    private func recentArtifacts(snapshot: FXAIProjectSnapshot) -> some View {
        FXAIVisualEffectSurface {
            VStack(alignment: .leading, spacing: 12) {
                Text("Recent Artifacts")
                    .font(.headline)
                    .foregroundStyle(FXAITheme.textPrimary)

                ScrollView(.vertical, showsIndicators: true) {
                    LazyVStack(alignment: .leading, spacing: 0) {
                        ForEach(snapshot.recentArtifacts.prefix(8)) { artifact in
                            HStack(alignment: .top) {
                                VStack(alignment: .leading, spacing: 4) {
                                    Text(artifact.name)
                                        .font(.subheadline.weight(.semibold))
                                        .foregroundStyle(FXAITheme.textPrimary)

                                    Text(artifact.category)
                                        .font(.caption)
                                        .foregroundStyle(FXAITheme.textMuted)
                                }
                                Spacer()
                                Text(FXAIFormatting.byteCountString(for: artifact.sizeBytes))
                                    .font(.caption)
                                    .foregroundStyle(FXAITheme.textSecondary)
                            }
                            .padding(.vertical, 4)
                        }
                    }
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
    }

    private func runtimeProfiles(snapshot: FXAIProjectSnapshot) -> some View {
        FXAIVisualEffectSurface {
            VStack(alignment: .leading, spacing: 12) {
                Text("Deployment Profiles")
                    .font(.headline)
                    .foregroundStyle(FXAITheme.textPrimary)

                if snapshot.runtimeProfiles.isEmpty {
                    Text("No live deployment profiles were discovered in the current ResearchOS output.")
                        .font(.subheadline)
                        .foregroundStyle(FXAITheme.textSecondary)
                } else {
                    ScrollView(.vertical, showsIndicators: true) {
                        LazyVStack(alignment: .leading, spacing: 0) {
                            ForEach(snapshot.runtimeProfiles.prefix(8)) { profile in
                                VStack(alignment: .leading, spacing: 4) {
                                    HStack {
                                        Text(profile.symbol)
                                            .font(.subheadline.weight(.semibold))
                                            .foregroundStyle(FXAITheme.textPrimary)
                                        Spacer()
                                        Text(profile.runtimeMode)
                                            .font(.caption.weight(.semibold))
                                            .foregroundStyle(FXAITheme.accent)
                                    }

                                    Text("\(profile.pluginName) • \(profile.promotionTier)")
                                        .font(.caption)
                                        .foregroundStyle(FXAITheme.textSecondary)
                                }
                                .padding(.vertical, 4)
                            }
                        }
                    }
                    .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
                }
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
    }

    private func onboardingPrompt(guide: RoleOnboardingGuide) -> some View {
        FXAIVisualEffectSurface {
            VStack(alignment: .leading, spacing: 12) {
                ViewThatFits(in: .horizontal) {
                    HStack {
                        VStack(alignment: .leading, spacing: 6) {
                            Text("Next Best Step For \(guide.role.title)")
                                .font(.headline)
                                .foregroundStyle(FXAITheme.textPrimary)
                            Text(guide.headline)
                                .foregroundStyle(FXAITheme.textSecondary)
                        }
                        Spacer()
                        Button("Open Guide") {
                            model.navigate(to: .onboarding)
                        }
                        .buttonStyle(.borderedProminent)
                        .tint(FXAITheme.accent)
                    }
                    VStack(alignment: .leading, spacing: 12) {
                        VStack(alignment: .leading, spacing: 6) {
                            Text("Next Best Step For \(guide.role.title)")
                                .font(.headline)
                                .foregroundStyle(FXAITheme.textPrimary)
                            Text(guide.headline)
                                .foregroundStyle(FXAITheme.textSecondary)
                        }
                        Button("Open Guide") {
                            model.navigate(to: .onboarding)
                        }
                        .buttonStyle(.borderedProminent)
                        .tint(FXAITheme.accent)
                    }
                }
            }
        }
    }

    @ViewBuilder
    private func widgetContent(for kind: OverviewDashboardWidgetKind, snapshot: FXAIProjectSnapshot) -> some View {
        switch kind {
        case .heroSummary:
            hero(snapshot: snapshot)
        case .buildTargetsMetric, .pluginsMetric, .artifactsMetric, .runtimeProfilesMetric, .incidentsMetric, .savedViewsMetric:
            metricCard(for: kind, snapshot: snapshot)
        case .pluginChart:
            pluginChart(snapshot: snapshot)
        case .reportChart:
            reportChart(snapshot: snapshot)
        case .recentArtifacts:
            recentArtifacts(snapshot: snapshot)
        case .deploymentProfiles:
            runtimeProfiles(snapshot: snapshot)
        case .onboardingPrompt:
            if let guide = model.currentOnboardingGuide, !model.hasCompletedOnboarding(for: guide.role) {
                onboardingPrompt(guide: guide)
            } else {
                EmptyView()
            }
        }
    }

    private func visibleWidgets(in section: OverviewDashboardSectionLayout, snapshot: FXAIProjectSnapshot) -> [OverviewDashboardWidgetLayout] {
        section.widgets.filter { widget in
            switch widget.kind {
            case .onboardingPrompt:
                if let guide = model.currentOnboardingGuide {
                    return !model.hasCompletedOnboarding(for: guide.role)
                }
                return false
            case .heroSummary, .buildTargetsMetric, .pluginsMetric, .artifactsMetric, .runtimeProfilesMetric, .incidentsMetric, .savedViewsMetric, .pluginChart, .reportChart, .recentArtifacts, .deploymentProfiles:
                return true
            }
        }
    }

    private func heroStatusBadges(snapshot: FXAIProjectSnapshot) -> some View {
        Group {
            StatusBadge(
                title: "Connection",
                value: model.connectionStatusLabel,
                tint: connectionTint
            )
            StatusBadge(
                title: "Champions",
                value: "\(snapshot.operatorSummary.championCount)",
                tint: FXAITheme.success
            )
            StatusBadge(
                title: "Deployments",
                value: "\(snapshot.operatorSummary.deploymentCount)",
                tint: FXAITheme.accent
            )
            StatusBadge(
                title: "Turso",
                value: snapshot.tursoSummary.localDatabasePresent ? "Present" : "Missing",
                tint: snapshot.tursoSummary.localDatabasePresent ? FXAITheme.success : FXAITheme.warning
            )
            StatusBadge(
                title: "Encryption",
                value: snapshot.tursoSummary.encryptionConfigured ? "Configured" : "Off",
                tint: snapshot.tursoSummary.encryptionConfigured ? FXAITheme.success : FXAITheme.warning
            )
        }
    }

    private var emptyStateTitle: String {
        switch model.connectionState {
        case .connected:
            "No FXAI project loaded"
        case .waitingForProject:
            "Waiting for FXAI"
        case .disconnectedByUser:
            "FXAI is disconnected"
        }
    }

    private var emptyStateMessage: String {
        switch model.connectionState {
        case .connected:
            model.lastErrorMessage ?? "Choose a project root to load plugin, report, and runtime inventory."
        case .waitingForProject:
            "The GUI is running in detached mode and will softly try to reconnect every 10 seconds. You can also pick a project root manually at any time."
        case .disconnectedByUser:
            "The GUI was intentionally disconnected from the framework. Use Connect or Choose Project when you want to attach it again."
        }
    }

    private var emptyStateSymbol: String {
        switch model.connectionState {
        case .connected:
            "folder.badge.questionmark"
        case .waitingForProject:
            "link.badge.plus"
        case .disconnectedByUser:
            "eject.circle"
        }
    }

    private var connectionTint: Color {
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
