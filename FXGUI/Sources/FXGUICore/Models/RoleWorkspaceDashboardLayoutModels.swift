import Foundation

public enum RoleWorkspaceDashboardKind: String, CaseIterable, Codable, Identifiable, Hashable, Sendable {
    case liveOverview
    case demoOverview
    case researchWorkspace
    case platformControl

    public var id: String { rawValue }

    public init?(role: WorkspaceRole) {
        switch role {
        case .liveTrader:
            self = .liveOverview
        case .demoTrader:
            self = .demoOverview
        case .researcher:
            self = .researchWorkspace
        case .architect:
            self = .platformControl
        case .backtester:
            return nil
        }
    }
}

public enum RoleWorkspaceDashboardPanelKind: String, CaseIterable, Codable, Identifiable, Hashable, Sendable {
    case hero
    case statusSummary
    case benefits
    case scenarios
    case quickScreens
    case commands

    public var id: String { rawValue }

    public var title: String {
        switch self {
        case .hero:
            "Role Overview"
        case .statusSummary:
            "Status Summary"
        case .benefits:
            "Benefits"
        case .scenarios:
            "Example Scenarios"
        case .quickScreens:
            "Quick Screens"
        case .commands:
            "Suggested Commands"
        }
    }
}

public struct RoleWorkspaceDashboardPanelSpec: Hashable, Sendable {
    public let kind: RoleWorkspaceDashboardPanelKind
    public let minimumWidthUnits: Int
    public let maximumWidthUnits: Int
    public let minimumHeightUnits: Int
    public let maximumHeightUnits: Int
    public let defaultWidthUnits: Int
    public let defaultHeightUnits: Int
    public let defaultColumnUnits: Int
    public let defaultRowUnits: Int
    public let priority: Int

    public init(
        kind: RoleWorkspaceDashboardPanelKind,
        minimumWidthUnits: Int,
        maximumWidthUnits: Int,
        minimumHeightUnits: Int,
        maximumHeightUnits: Int,
        defaultWidthUnits: Int,
        defaultHeightUnits: Int,
        defaultColumnUnits: Int,
        defaultRowUnits: Int,
        priority: Int
    ) {
        self.kind = kind
        self.minimumWidthUnits = minimumWidthUnits
        self.maximumWidthUnits = maximumWidthUnits
        self.minimumHeightUnits = minimumHeightUnits
        self.maximumHeightUnits = maximumHeightUnits
        self.defaultWidthUnits = defaultWidthUnits
        self.defaultHeightUnits = defaultHeightUnits
        self.defaultColumnUnits = defaultColumnUnits
        self.defaultRowUnits = defaultRowUnits
        self.priority = priority
    }
}

public struct RoleWorkspaceDashboardPanelLayout: Identifiable, Codable, Hashable, Sendable {
    public let id: UUID
    public var kind: RoleWorkspaceDashboardPanelKind
    public var widthUnits: Int
    public var heightUnits: Int
    public var columnUnits: Int
    public var rowUnits: Int

    enum CodingKeys: String, CodingKey {
        case id
        case kind
        case widthUnits
        case heightUnits
        case columnUnits
        case rowUnits
    }

    public init(
        id: UUID = UUID(),
        kind: RoleWorkspaceDashboardPanelKind,
        widthUnits: Int,
        heightUnits: Int,
        columnUnits: Int = 0,
        rowUnits: Int = 0
    ) {
        self.id = id
        self.kind = kind
        self.widthUnits = widthUnits
        self.heightUnits = heightUnits
        self.columnUnits = max(0, columnUnits)
        self.rowUnits = max(0, rowUnits)
    }

    public init(kind: RoleWorkspaceDashboardPanelKind) {
        let spec = RoleWorkspaceDashboardLayoutState.spec(for: kind)
        self.init(
            kind: kind,
            widthUnits: spec.defaultWidthUnits,
            heightUnits: spec.defaultHeightUnits,
            columnUnits: spec.defaultColumnUnits,
            rowUnits: spec.defaultRowUnits
        )
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        id = try container.decodeIfPresent(UUID.self, forKey: .id) ?? UUID()
        kind = try container.decode(RoleWorkspaceDashboardPanelKind.self, forKey: .kind)
        let spec = RoleWorkspaceDashboardLayoutState.spec(for: kind)
        widthUnits = try container.decodeIfPresent(Int.self, forKey: .widthUnits) ?? spec.defaultWidthUnits
        heightUnits = try container.decodeIfPresent(Int.self, forKey: .heightUnits) ?? spec.defaultHeightUnits
        columnUnits = max(0, try container.decodeIfPresent(Int.self, forKey: .columnUnits) ?? spec.defaultColumnUnits)
        rowUnits = max(0, try container.decodeIfPresent(Int.self, forKey: .rowUnits) ?? spec.defaultRowUnits)
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(id, forKey: .id)
        try container.encode(kind, forKey: .kind)
        try container.encode(widthUnits, forKey: .widthUnits)
        try container.encode(heightUnits, forKey: .heightUnits)
        try container.encode(columnUnits, forKey: .columnUnits)
        try container.encode(rowUnits, forKey: .rowUnits)
    }

    public func clamped(for availableColumns: Int? = nil) -> RoleWorkspaceDashboardPanelLayout {
        let spec = RoleWorkspaceDashboardLayoutState.spec(for: kind)
        let maximumWidth = min(spec.maximumWidthUnits, max(spec.minimumWidthUnits, availableColumns ?? spec.maximumWidthUnits))
        let clampedWidth = min(max(widthUnits, spec.minimumWidthUnits), maximumWidth)
        let clampedHeight = min(max(heightUnits, spec.minimumHeightUnits), spec.maximumHeightUnits)
        let clampedColumn: Int
        if let availableColumns {
            clampedColumn = min(max(0, columnUnits), max(0, availableColumns - clampedWidth))
        } else {
            clampedColumn = max(0, columnUnits)
        }
        return RoleWorkspaceDashboardPanelLayout(
            id: id,
            kind: kind,
            widthUnits: clampedWidth,
            heightUnits: clampedHeight,
            columnUnits: clampedColumn,
            rowUnits: max(0, rowUnits)
        )
    }
}

public struct RoleWorkspaceDashboardLayoutState: Codable, Hashable, Sendable {
    public var gridUnitPoints: Double
    public var panels: [RoleWorkspaceDashboardPanelLayout]

    public init(
        gridUnitPoints: Double = 72.0 / 2.54,
        panels: [RoleWorkspaceDashboardPanelLayout] = []
    ) {
        self.gridUnitPoints = max(72.0 / 2.54, gridUnitPoints)
        self.panels = Self.normalizedPanels(from: panels.isEmpty ? Self.defaultPanels() : panels)
    }

    public static func `default`(for _: RoleWorkspaceDashboardKind) -> RoleWorkspaceDashboardLayoutState {
        RoleWorkspaceDashboardLayoutState()
    }

    public static func spec(for kind: RoleWorkspaceDashboardPanelKind) -> RoleWorkspaceDashboardPanelSpec {
        specs[kind]!
    }

    public func normalized() -> RoleWorkspaceDashboardLayoutState {
        RoleWorkspaceDashboardLayoutState(
            gridUnitPoints: max(72.0 / 2.54, gridUnitPoints),
            panels: Self.normalizedPanels(from: panels)
        )
    }

    private static func defaultPanels() -> [RoleWorkspaceDashboardPanelLayout] {
        [
            RoleWorkspaceDashboardPanelLayout(kind: .hero),
            RoleWorkspaceDashboardPanelLayout(kind: .statusSummary),
            RoleWorkspaceDashboardPanelLayout(kind: .benefits),
            RoleWorkspaceDashboardPanelLayout(kind: .scenarios),
            RoleWorkspaceDashboardPanelLayout(kind: .quickScreens),
            RoleWorkspaceDashboardPanelLayout(kind: .commands)
        ]
    }

    private static func normalizedPanels(from panels: [RoleWorkspaceDashboardPanelLayout]) -> [RoleWorkspaceDashboardPanelLayout] {
        let defaults = defaultPanels()
        let validKinds = Set(RoleWorkspaceDashboardPanelKind.allCases)
        var deduplicated: [RoleWorkspaceDashboardPanelLayout] = []
        var seen = Set<RoleWorkspaceDashboardPanelKind>()

        for panel in panels where validKinds.contains(panel.kind) && !seen.contains(panel.kind) {
            deduplicated.append(panel.clamped())
            seen.insert(panel.kind)
        }

        for panel in defaults where !seen.contains(panel.kind) {
            deduplicated.append(panel)
            seen.insert(panel.kind)
        }

        return deduplicated.map { $0.clamped() }
    }

    private static let specs: [RoleWorkspaceDashboardPanelKind: RoleWorkspaceDashboardPanelSpec] = [
        .hero: RoleWorkspaceDashboardPanelSpec(kind: .hero, minimumWidthUnits: 6, maximumWidthUnits: 12, minimumHeightUnits: 4, maximumHeightUnits: 7, defaultWidthUnits: 12, defaultHeightUnits: 5, defaultColumnUnits: 0, defaultRowUnits: 0, priority: 100),
        .statusSummary: RoleWorkspaceDashboardPanelSpec(kind: .statusSummary, minimumWidthUnits: 6, maximumWidthUnits: 12, minimumHeightUnits: 4, maximumHeightUnits: 7, defaultWidthUnits: 12, defaultHeightUnits: 5, defaultColumnUnits: 0, defaultRowUnits: 5, priority: 98),
        .benefits: RoleWorkspaceDashboardPanelSpec(kind: .benefits, minimumWidthUnits: 4, maximumWidthUnits: 8, minimumHeightUnits: 4, maximumHeightUnits: 8, defaultWidthUnits: 6, defaultHeightUnits: 5, defaultColumnUnits: 0, defaultRowUnits: 10, priority: 95),
        .scenarios: RoleWorkspaceDashboardPanelSpec(kind: .scenarios, minimumWidthUnits: 4, maximumWidthUnits: 8, minimumHeightUnits: 4, maximumHeightUnits: 8, defaultWidthUnits: 6, defaultHeightUnits: 5, defaultColumnUnits: 6, defaultRowUnits: 10, priority: 94),
        .quickScreens: RoleWorkspaceDashboardPanelSpec(kind: .quickScreens, minimumWidthUnits: 5, maximumWidthUnits: 8, minimumHeightUnits: 5, maximumHeightUnits: 9, defaultWidthUnits: 7, defaultHeightUnits: 6, defaultColumnUnits: 0, defaultRowUnits: 15, priority: 92),
        .commands: RoleWorkspaceDashboardPanelSpec(kind: .commands, minimumWidthUnits: 4, maximumWidthUnits: 7, minimumHeightUnits: 4, maximumHeightUnits: 9, defaultWidthUnits: 5, defaultHeightUnits: 6, defaultColumnUnits: 7, defaultRowUnits: 15, priority: 90)
    ]
}

public struct RoleWorkspaceDashboardLayouts: Codable, Hashable, Sendable {
    public var liveOverview: RoleWorkspaceDashboardLayoutState
    public var demoOverview: RoleWorkspaceDashboardLayoutState
    public var researchWorkspace: RoleWorkspaceDashboardLayoutState
    public var platformControl: RoleWorkspaceDashboardLayoutState

    public init(
        liveOverview: RoleWorkspaceDashboardLayoutState = .default(for: .liveOverview),
        demoOverview: RoleWorkspaceDashboardLayoutState = .default(for: .demoOverview),
        researchWorkspace: RoleWorkspaceDashboardLayoutState = .default(for: .researchWorkspace),
        platformControl: RoleWorkspaceDashboardLayoutState = .default(for: .platformControl)
    ) {
        self.liveOverview = liveOverview.normalized()
        self.demoOverview = demoOverview.normalized()
        self.researchWorkspace = researchWorkspace.normalized()
        self.platformControl = platformControl.normalized()
    }

    public static func `default`() -> RoleWorkspaceDashboardLayouts {
        RoleWorkspaceDashboardLayouts()
    }

    public func normalized() -> RoleWorkspaceDashboardLayouts {
        RoleWorkspaceDashboardLayouts(
            liveOverview: liveOverview.normalized(),
            demoOverview: demoOverview.normalized(),
            researchWorkspace: researchWorkspace.normalized(),
            platformControl: platformControl.normalized()
        )
    }

    public func layout(for kind: RoleWorkspaceDashboardKind) -> RoleWorkspaceDashboardLayoutState {
        switch kind {
        case .liveOverview:
            liveOverview
        case .demoOverview:
            demoOverview
        case .researchWorkspace:
            researchWorkspace
        case .platformControl:
            platformControl
        }
    }

    public mutating func setLayout(_ layout: RoleWorkspaceDashboardLayoutState, for kind: RoleWorkspaceDashboardKind) {
        switch kind {
        case .liveOverview:
            liveOverview = layout.normalized()
        case .demoOverview:
            demoOverview = layout.normalized()
        case .researchWorkspace:
            researchWorkspace = layout.normalized()
        case .platformControl:
            platformControl = layout.normalized()
        }
    }
}
