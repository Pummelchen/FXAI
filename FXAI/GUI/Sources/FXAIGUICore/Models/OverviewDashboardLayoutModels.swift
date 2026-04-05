import Foundation

public enum OverviewDashboardSectionKind: String, CaseIterable, Codable, Identifiable, Hashable, Sendable {
    case hero
    case metrics
    case analytics
    case operations
    case guidance

    public var id: String { rawValue }

    public var title: String {
        switch self {
        case .hero:
            "Overview"
        case .metrics:
            "Metrics"
        case .analytics:
            "Analytics"
        case .operations:
            "Operations"
        case .guidance:
            "Guidance"
        }
    }

    public var subtitle: String {
        switch self {
        case .hero:
            "Primary status and immediate operator posture."
        case .metrics:
            "Fast numerical health checks across the framework."
        case .analytics:
            "Charts and surfaces that explain where the framework is leaning."
        case .operations:
            "Artifacts and deployment surfaces operators revisit most often."
        case .guidance:
            "Context-sensitive next actions and operator help."
        }
    }
}

public enum OverviewDashboardWidgetKind: String, CaseIterable, Codable, Identifiable, Hashable, Sendable {
    case heroSummary
    case buildTargetsMetric
    case pluginsMetric
    case artifactsMetric
    case runtimeProfilesMetric
    case incidentsMetric
    case savedViewsMetric
    case pluginChart
    case reportChart
    case recentArtifacts
    case deploymentProfiles
    case onboardingPrompt

    public var id: String { rawValue }

    public var title: String {
        switch self {
        case .heroSummary:
            "Overview Hero"
        case .buildTargetsMetric:
            "Build Targets"
        case .pluginsMetric:
            "Plugins"
        case .artifactsMetric:
            "Artifacts"
        case .runtimeProfilesMetric:
            "Runtime Profiles"
        case .incidentsMetric:
            "Incidents"
        case .savedViewsMetric:
            "Saved Views"
        case .pluginChart:
            "Plugin Family Footprint"
        case .reportChart:
            "Artifact Surface"
        case .recentArtifacts:
            "Recent Artifacts"
        case .deploymentProfiles:
            "Deployment Profiles"
        case .onboardingPrompt:
            "Onboarding Prompt"
        }
    }
}

public struct OverviewDashboardWidgetSpec: Hashable, Sendable {
    public let kind: OverviewDashboardWidgetKind
    public let defaultSection: OverviewDashboardSectionKind
    public let minimumWidthUnits: Int
    public let maximumWidthUnits: Int
    public let minimumHeightUnits: Int
    public let maximumHeightUnits: Int
    public let defaultWidthUnits: Int
    public let defaultHeightUnits: Int
    public let priority: Int

    public init(
        kind: OverviewDashboardWidgetKind,
        defaultSection: OverviewDashboardSectionKind,
        minimumWidthUnits: Int,
        maximumWidthUnits: Int,
        minimumHeightUnits: Int,
        maximumHeightUnits: Int,
        defaultWidthUnits: Int,
        defaultHeightUnits: Int,
        priority: Int
    ) {
        self.kind = kind
        self.defaultSection = defaultSection
        self.minimumWidthUnits = minimumWidthUnits
        self.maximumWidthUnits = maximumWidthUnits
        self.minimumHeightUnits = minimumHeightUnits
        self.maximumHeightUnits = maximumHeightUnits
        self.defaultWidthUnits = defaultWidthUnits
        self.defaultHeightUnits = defaultHeightUnits
        self.priority = priority
    }
}

public struct OverviewDashboardWidgetLayout: Identifiable, Codable, Hashable, Sendable {
    public let id: UUID
    public var kind: OverviewDashboardWidgetKind
    public var widthUnits: Int
    public var heightUnits: Int

    public init(
        id: UUID = UUID(),
        kind: OverviewDashboardWidgetKind,
        widthUnits: Int,
        heightUnits: Int
    ) {
        self.id = id
        self.kind = kind
        self.widthUnits = widthUnits
        self.heightUnits = heightUnits
    }

    public init(kind: OverviewDashboardWidgetKind) {
        let spec = OverviewDashboardLayoutState.spec(for: kind)
        self.init(
            kind: kind,
            widthUnits: spec.defaultWidthUnits,
            heightUnits: spec.defaultHeightUnits
        )
    }

    public func clamped(for availableColumns: Int? = nil) -> OverviewDashboardWidgetLayout {
        let spec = OverviewDashboardLayoutState.spec(for: kind)
        let maximumWidth = min(spec.maximumWidthUnits, max(spec.minimumWidthUnits, availableColumns ?? spec.maximumWidthUnits))
        return OverviewDashboardWidgetLayout(
            id: id,
            kind: kind,
            widthUnits: min(max(widthUnits, spec.minimumWidthUnits), maximumWidth),
            heightUnits: min(max(heightUnits, spec.minimumHeightUnits), spec.maximumHeightUnits)
        )
    }
}

public struct OverviewDashboardSectionLayout: Identifiable, Codable, Hashable, Sendable {
    public let id: UUID
    public var kind: OverviewDashboardSectionKind
    public var widgets: [OverviewDashboardWidgetLayout]

    public init(
        id: UUID = UUID(),
        kind: OverviewDashboardSectionKind,
        widgets: [OverviewDashboardWidgetLayout]
    ) {
        self.id = id
        self.kind = kind
        self.widgets = widgets
    }
}

public struct OverviewDashboardLayoutState: Codable, Hashable, Sendable {
    public var gridUnitPoints: Double
    public var sections: [OverviewDashboardSectionLayout]

    public init(
        gridUnitPoints: Double = 40,
        sections: [OverviewDashboardSectionLayout] = []
    ) {
        self.gridUnitPoints = max(40, gridUnitPoints)
        self.sections = Self.normalizedSections(from: sections.isEmpty ? Self.defaultSections() : sections)
    }

    public static func `default`() -> OverviewDashboardLayoutState {
        OverviewDashboardLayoutState()
    }

    public static func spec(for kind: OverviewDashboardWidgetKind) -> OverviewDashboardWidgetSpec {
        specs[kind]!
    }

    public func normalized() -> OverviewDashboardLayoutState {
        OverviewDashboardLayoutState(gridUnitPoints: max(40, gridUnitPoints), sections: Self.normalizedSections(from: sections))
    }

    private static func defaultSections() -> [OverviewDashboardSectionLayout] {
        [
            OverviewDashboardSectionLayout(
                kind: .hero,
                widgets: [
                    OverviewDashboardWidgetLayout(kind: .heroSummary)
                ]
            ),
            OverviewDashboardSectionLayout(
                kind: .metrics,
                widgets: [
                    OverviewDashboardWidgetLayout(kind: .buildTargetsMetric),
                    OverviewDashboardWidgetLayout(kind: .pluginsMetric),
                    OverviewDashboardWidgetLayout(kind: .artifactsMetric),
                    OverviewDashboardWidgetLayout(kind: .runtimeProfilesMetric),
                    OverviewDashboardWidgetLayout(kind: .incidentsMetric),
                    OverviewDashboardWidgetLayout(kind: .savedViewsMetric)
                ]
            ),
            OverviewDashboardSectionLayout(
                kind: .analytics,
                widgets: [
                    OverviewDashboardWidgetLayout(kind: .pluginChart),
                    OverviewDashboardWidgetLayout(kind: .reportChart)
                ]
            ),
            OverviewDashboardSectionLayout(
                kind: .operations,
                widgets: [
                    OverviewDashboardWidgetLayout(kind: .recentArtifacts),
                    OverviewDashboardWidgetLayout(kind: .deploymentProfiles)
                ]
            ),
            OverviewDashboardSectionLayout(
                kind: .guidance,
                widgets: [
                    OverviewDashboardWidgetLayout(kind: .onboardingPrompt)
                ]
            )
        ]
    }

    private static func normalizedSections(from sections: [OverviewDashboardSectionLayout]) -> [OverviewDashboardSectionLayout] {
        let defaultSections = defaultSections()
        let validSectionKinds = Set(OverviewDashboardSectionKind.allCases)
        let validWidgetKinds = Set(OverviewDashboardWidgetKind.allCases)

        var deduplicatedSections: [OverviewDashboardSectionLayout] = []
        var seenSections = Set<OverviewDashboardSectionKind>()
        for section in sections where validSectionKinds.contains(section.kind) && !seenSections.contains(section.kind) {
            seenSections.insert(section.kind)
            var seenWidgets = Set<OverviewDashboardWidgetKind>()
            let normalizedWidgets = section.widgets.compactMap { widget -> OverviewDashboardWidgetLayout? in
                guard validWidgetKinds.contains(widget.kind), !seenWidgets.contains(widget.kind) else { return nil }
                seenWidgets.insert(widget.kind)
                return widget.clamped()
            }
            deduplicatedSections.append(
                OverviewDashboardSectionLayout(id: section.id, kind: section.kind, widgets: normalizedWidgets)
            )
        }

        for defaultSection in defaultSections where !seenSections.contains(defaultSection.kind) {
            deduplicatedSections.append(defaultSection)
        }

        return deduplicatedSections.map { section in
            let defaultWidgets = defaultSections.first(where: { $0.kind == section.kind })?.widgets ?? []
            var seenWidgets = Set(section.widgets.map(\.kind))
            var widgets = section.widgets
            for widget in defaultWidgets where !seenWidgets.contains(widget.kind) {
                widgets.append(widget)
                seenWidgets.insert(widget.kind)
            }
            return OverviewDashboardSectionLayout(id: section.id, kind: section.kind, widgets: widgets.map { $0.clamped() })
        }
    }

    private static let specs: [OverviewDashboardWidgetKind: OverviewDashboardWidgetSpec] = [
        .heroSummary: OverviewDashboardWidgetSpec(kind: .heroSummary, defaultSection: .hero, minimumWidthUnits: 6, maximumWidthUnits: 12, minimumHeightUnits: 4, maximumHeightUnits: 7, defaultWidthUnits: 12, defaultHeightUnits: 5, priority: 100),
        .buildTargetsMetric: OverviewDashboardWidgetSpec(kind: .buildTargetsMetric, defaultSection: .metrics, minimumWidthUnits: 2, maximumWidthUnits: 6, minimumHeightUnits: 2, maximumHeightUnits: 4, defaultWidthUnits: 3, defaultHeightUnits: 3, priority: 92),
        .pluginsMetric: OverviewDashboardWidgetSpec(kind: .pluginsMetric, defaultSection: .metrics, minimumWidthUnits: 2, maximumWidthUnits: 6, minimumHeightUnits: 2, maximumHeightUnits: 4, defaultWidthUnits: 3, defaultHeightUnits: 3, priority: 90),
        .artifactsMetric: OverviewDashboardWidgetSpec(kind: .artifactsMetric, defaultSection: .metrics, minimumWidthUnits: 2, maximumWidthUnits: 6, minimumHeightUnits: 2, maximumHeightUnits: 4, defaultWidthUnits: 3, defaultHeightUnits: 3, priority: 89),
        .runtimeProfilesMetric: OverviewDashboardWidgetSpec(kind: .runtimeProfilesMetric, defaultSection: .metrics, minimumWidthUnits: 2, maximumWidthUnits: 6, minimumHeightUnits: 2, maximumHeightUnits: 4, defaultWidthUnits: 3, defaultHeightUnits: 3, priority: 88),
        .incidentsMetric: OverviewDashboardWidgetSpec(kind: .incidentsMetric, defaultSection: .metrics, minimumWidthUnits: 2, maximumWidthUnits: 6, minimumHeightUnits: 2, maximumHeightUnits: 4, defaultWidthUnits: 3, defaultHeightUnits: 3, priority: 87),
        .savedViewsMetric: OverviewDashboardWidgetSpec(kind: .savedViewsMetric, defaultSection: .metrics, minimumWidthUnits: 2, maximumWidthUnits: 6, minimumHeightUnits: 2, maximumHeightUnits: 4, defaultWidthUnits: 3, defaultHeightUnits: 3, priority: 86),
        .pluginChart: OverviewDashboardWidgetSpec(kind: .pluginChart, defaultSection: .analytics, minimumWidthUnits: 4, maximumWidthUnits: 12, minimumHeightUnits: 4, maximumHeightUnits: 8, defaultWidthUnits: 6, defaultHeightUnits: 5, priority: 82),
        .reportChart: OverviewDashboardWidgetSpec(kind: .reportChart, defaultSection: .analytics, minimumWidthUnits: 4, maximumWidthUnits: 12, minimumHeightUnits: 4, maximumHeightUnits: 8, defaultWidthUnits: 6, defaultHeightUnits: 5, priority: 81),
        .recentArtifacts: OverviewDashboardWidgetSpec(kind: .recentArtifacts, defaultSection: .operations, minimumWidthUnits: 4, maximumWidthUnits: 12, minimumHeightUnits: 4, maximumHeightUnits: 9, defaultWidthUnits: 6, defaultHeightUnits: 6, priority: 76),
        .deploymentProfiles: OverviewDashboardWidgetSpec(kind: .deploymentProfiles, defaultSection: .operations, minimumWidthUnits: 4, maximumWidthUnits: 12, minimumHeightUnits: 4, maximumHeightUnits: 9, defaultWidthUnits: 6, defaultHeightUnits: 6, priority: 75),
        .onboardingPrompt: OverviewDashboardWidgetSpec(kind: .onboardingPrompt, defaultSection: .guidance, minimumWidthUnits: 4, maximumWidthUnits: 12, minimumHeightUnits: 3, maximumHeightUnits: 6, defaultWidthUnits: 12, defaultHeightUnits: 3, priority: 70)
    ]
}
