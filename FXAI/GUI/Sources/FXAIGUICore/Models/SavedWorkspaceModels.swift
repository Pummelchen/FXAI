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

    public init(
        id: UUID = UUID(),
        name: String,
        projectRootPath: String?,
        createdAt: Date = Date(),
        updatedAt: Date = Date(),
        selection: String,
        selectedRole: WorkspaceRole,
        selectedRuntimeSymbol: String,
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
        researchRecoveryDraft: ResearchOSRecoveryDraft
    ) {
        self.id = id
        self.name = name
        self.projectRootPath = projectRootPath
        self.createdAt = createdAt
        self.updatedAt = updatedAt
        self.selection = selection
        self.selectedRole = selectedRole
        self.selectedRuntimeSymbol = selectedRuntimeSymbol
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
