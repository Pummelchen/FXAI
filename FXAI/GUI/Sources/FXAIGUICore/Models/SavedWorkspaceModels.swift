import Foundation

public struct SavedWorkspaceView: Identifiable, Codable, Hashable, Sendable {
    public let id: UUID
    public var name: String
    public var projectRootPath: String?
    public var createdAt: Date
    public var updatedAt: Date
    public var selection: String
    public var selectedRole: WorkspaceRole
    public var selectedRuntimeSymbol: String
    public var selectedRatesSymbol: String
    public var selectedMicrostructureSymbol: String
    public var selectedAdaptiveSymbol: String
    public var selectedDynamicEnsembleSymbol: String
    public var selectedResearchSymbol: String
    public var selectedVisualizationSymbol: String
    public var pluginSearchText: String
    public var selectedPluginFamily: String
    public var reportCategoryFilter: String
    public var auditDraft: AuditLabDraft
    public var backtestDraft: BacktestBuilderDraft
    public var offlineDraft: OfflineLabDraft
    public var researchBranchDraft: ResearchOSBranchDraft
    public var researchAuditDraft: ResearchOSAuditDraft
    public var researchVectorDraft: ResearchOSVectorDraft
    public var researchRecoveryDraft: ResearchOSRecoveryDraft
    public var overviewLayout: OverviewDashboardLayoutState

    public init(
        id: UUID = UUID(),
        name: String,
        projectRootPath: String?,
        createdAt: Date = Date(),
        updatedAt: Date = Date(),
        selection: String,
        selectedRole: WorkspaceRole,
        selectedRuntimeSymbol: String,
        selectedRatesSymbol: String,
        selectedMicrostructureSymbol: String,
        selectedAdaptiveSymbol: String,
        selectedDynamicEnsembleSymbol: String,
        selectedResearchSymbol: String,
        selectedVisualizationSymbol: String,
        pluginSearchText: String,
        selectedPluginFamily: String,
        reportCategoryFilter: String,
        auditDraft: AuditLabDraft,
        backtestDraft: BacktestBuilderDraft,
        offlineDraft: OfflineLabDraft,
        researchBranchDraft: ResearchOSBranchDraft,
        researchAuditDraft: ResearchOSAuditDraft,
        researchVectorDraft: ResearchOSVectorDraft,
        researchRecoveryDraft: ResearchOSRecoveryDraft,
        overviewLayout: OverviewDashboardLayoutState = .default()
    ) {
        self.id = id
        self.name = name
        self.projectRootPath = projectRootPath
        self.createdAt = createdAt
        self.updatedAt = updatedAt
        self.selection = selection
        self.selectedRole = selectedRole
        self.selectedRuntimeSymbol = selectedRuntimeSymbol
        self.selectedRatesSymbol = selectedRatesSymbol
        self.selectedMicrostructureSymbol = selectedMicrostructureSymbol
        self.selectedAdaptiveSymbol = selectedAdaptiveSymbol
        self.selectedDynamicEnsembleSymbol = selectedDynamicEnsembleSymbol
        self.selectedResearchSymbol = selectedResearchSymbol
        self.selectedVisualizationSymbol = selectedVisualizationSymbol
        self.pluginSearchText = pluginSearchText
        self.selectedPluginFamily = selectedPluginFamily
        self.reportCategoryFilter = reportCategoryFilter
        self.auditDraft = auditDraft
        self.backtestDraft = backtestDraft
        self.offlineDraft = offlineDraft
        self.researchBranchDraft = researchBranchDraft
        self.researchAuditDraft = researchAuditDraft
        self.researchVectorDraft = researchVectorDraft
        self.researchRecoveryDraft = researchRecoveryDraft
        self.overviewLayout = overviewLayout
    }

    private enum CodingKeys: String, CodingKey {
        case id
        case name
        case projectRootPath
        case createdAt
        case updatedAt
        case selection
        case selectedRole
        case selectedRuntimeSymbol
        case selectedRatesSymbol
        case selectedMicrostructureSymbol
        case selectedAdaptiveSymbol
        case selectedDynamicEnsembleSymbol
        case selectedResearchSymbol
        case selectedVisualizationSymbol
        case pluginSearchText
        case selectedPluginFamily
        case reportCategoryFilter
        case auditDraft
        case backtestDraft
        case offlineDraft
        case researchBranchDraft
        case researchAuditDraft
        case researchVectorDraft
        case researchRecoveryDraft
        case overviewLayout
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        id = try container.decodeIfPresent(UUID.self, forKey: .id) ?? UUID()
        name = try container.decode(String.self, forKey: .name)
        projectRootPath = try container.decodeIfPresent(String.self, forKey: .projectRootPath)
        createdAt = try container.decodeIfPresent(Date.self, forKey: .createdAt) ?? Date()
        updatedAt = try container.decodeIfPresent(Date.self, forKey: .updatedAt) ?? createdAt
        selection = try container.decode(String.self, forKey: .selection)
        selectedRole = try container.decode(WorkspaceRole.self, forKey: .selectedRole)
        selectedRuntimeSymbol = try container.decodeIfPresent(String.self, forKey: .selectedRuntimeSymbol) ?? ""
        selectedRatesSymbol = try container.decodeIfPresent(String.self, forKey: .selectedRatesSymbol) ?? selectedRuntimeSymbol
        selectedMicrostructureSymbol = try container.decodeIfPresent(String.self, forKey: .selectedMicrostructureSymbol) ?? selectedRuntimeSymbol
        selectedAdaptiveSymbol = try container.decodeIfPresent(String.self, forKey: .selectedAdaptiveSymbol) ?? ""
        selectedDynamicEnsembleSymbol = try container.decodeIfPresent(String.self, forKey: .selectedDynamicEnsembleSymbol) ?? selectedAdaptiveSymbol
        selectedResearchSymbol = try container.decodeIfPresent(String.self, forKey: .selectedResearchSymbol) ?? ""
        selectedVisualizationSymbol = try container.decodeIfPresent(String.self, forKey: .selectedVisualizationSymbol) ?? ""
        pluginSearchText = try container.decodeIfPresent(String.self, forKey: .pluginSearchText) ?? ""
        selectedPluginFamily = try container.decodeIfPresent(String.self, forKey: .selectedPluginFamily) ?? "All"
        reportCategoryFilter = try container.decodeIfPresent(String.self, forKey: .reportCategoryFilter) ?? "All"
        auditDraft = try container.decodeIfPresent(AuditLabDraft.self, forKey: .auditDraft) ?? AuditLabDraft()
        backtestDraft = try container.decodeIfPresent(BacktestBuilderDraft.self, forKey: .backtestDraft) ?? BacktestBuilderDraft()
        offlineDraft = try container.decodeIfPresent(OfflineLabDraft.self, forKey: .offlineDraft) ?? OfflineLabDraft()
        researchBranchDraft = try container.decodeIfPresent(ResearchOSBranchDraft.self, forKey: .researchBranchDraft) ?? ResearchOSBranchDraft()
        researchAuditDraft = try container.decodeIfPresent(ResearchOSAuditDraft.self, forKey: .researchAuditDraft) ?? ResearchOSAuditDraft()
        researchVectorDraft = try container.decodeIfPresent(ResearchOSVectorDraft.self, forKey: .researchVectorDraft) ?? ResearchOSVectorDraft()
        researchRecoveryDraft = try container.decodeIfPresent(ResearchOSRecoveryDraft.self, forKey: .researchRecoveryDraft) ?? ResearchOSRecoveryDraft()
        overviewLayout = try container.decodeIfPresent(OverviewDashboardLayoutState.self, forKey: .overviewLayout)?.normalized() ?? .default()
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(id, forKey: .id)
        try container.encode(name, forKey: .name)
        try container.encodeIfPresent(projectRootPath, forKey: .projectRootPath)
        try container.encode(createdAt, forKey: .createdAt)
        try container.encode(updatedAt, forKey: .updatedAt)
        try container.encode(selection, forKey: .selection)
        try container.encode(selectedRole, forKey: .selectedRole)
        try container.encode(selectedRuntimeSymbol, forKey: .selectedRuntimeSymbol)
        try container.encode(selectedRatesSymbol, forKey: .selectedRatesSymbol)
        try container.encode(selectedMicrostructureSymbol, forKey: .selectedMicrostructureSymbol)
        try container.encode(selectedAdaptiveSymbol, forKey: .selectedAdaptiveSymbol)
        try container.encode(selectedDynamicEnsembleSymbol, forKey: .selectedDynamicEnsembleSymbol)
        try container.encode(selectedResearchSymbol, forKey: .selectedResearchSymbol)
        try container.encode(selectedVisualizationSymbol, forKey: .selectedVisualizationSymbol)
        try container.encode(pluginSearchText, forKey: .pluginSearchText)
        try container.encode(selectedPluginFamily, forKey: .selectedPluginFamily)
        try container.encode(reportCategoryFilter, forKey: .reportCategoryFilter)
        try container.encode(auditDraft, forKey: .auditDraft)
        try container.encode(backtestDraft, forKey: .backtestDraft)
        try container.encode(offlineDraft, forKey: .offlineDraft)
        try container.encode(researchBranchDraft, forKey: .researchBranchDraft)
        try container.encode(researchAuditDraft, forKey: .researchAuditDraft)
        try container.encode(researchVectorDraft, forKey: .researchVectorDraft)
        try container.encode(researchRecoveryDraft, forKey: .researchRecoveryDraft)
        try container.encode(overviewLayout.normalized(), forKey: .overviewLayout)
    }

    public var titleSummary: String {
        let selectionLabel = selection
            .unicodeScalars
            .enumerated()
            .reduce(into: "") { result, item in
                let scalar = item.element
                if item.offset > 0,
                   CharacterSet.uppercaseLetters.contains(scalar),
                   result.last != " " {
                    result.append(" ")
                }
                result.append(String(scalar))
            }
            .replacingOccurrences(of: "_", with: " ")
            .capitalized
        return "\(selectedRole.title) • \(selectionLabel)"
    }
}

public struct FXAIGUIPersistenceState: Codable, Hashable, Sendable {
    public var savedViews: [SavedWorkspaceView]
    public var lastWorkspace: SavedWorkspaceView?
    public var completedOnboardingRoles: [WorkspaceRole]
    public var preferredProjectRootPath: String?
    public var autoReconnectEnabled: Bool

    public init(
        savedViews: [SavedWorkspaceView] = [],
        lastWorkspace: SavedWorkspaceView? = nil,
        completedOnboardingRoles: [WorkspaceRole] = [],
        preferredProjectRootPath: String? = nil,
        autoReconnectEnabled: Bool = true
    ) {
        self.savedViews = savedViews
        self.lastWorkspace = lastWorkspace
        self.completedOnboardingRoles = completedOnboardingRoles
        self.preferredProjectRootPath = preferredProjectRootPath
        self.autoReconnectEnabled = autoReconnectEnabled
    }

    private enum CodingKeys: String, CodingKey {
        case savedViews
        case lastWorkspace
        case completedOnboardingRoles
        case preferredProjectRootPath
        case autoReconnectEnabled
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        savedViews = try container.decodeIfPresent([SavedWorkspaceView].self, forKey: .savedViews) ?? []
        lastWorkspace = try container.decodeIfPresent(SavedWorkspaceView.self, forKey: .lastWorkspace)
        completedOnboardingRoles = try container.decodeIfPresent([WorkspaceRole].self, forKey: .completedOnboardingRoles) ?? []
        preferredProjectRootPath = try container.decodeIfPresent(String.self, forKey: .preferredProjectRootPath)
        autoReconnectEnabled = try container.decodeIfPresent(Bool.self, forKey: .autoReconnectEnabled) ?? true
    }
}
