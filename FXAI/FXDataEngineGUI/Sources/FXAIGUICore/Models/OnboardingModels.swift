import Foundation

public struct OnboardingDestinationHint: Identifiable, Hashable, Sendable {
    public let id: String
    public let title: String
    public let selection: String

    public init(title: String, selection: String) {
        self.id = selection
        self.title = title
        self.selection = selection
    }
}

public struct OnboardingStep: Identifiable, Hashable, Sendable {
    public let id: String
    public let title: String
    public let summary: String
    public let destination: OnboardingDestinationHint?

    public init(title: String, summary: String, destination: OnboardingDestinationHint? = nil) {
        self.id = title
        self.title = title
        self.summary = summary
        self.destination = destination
    }
}

public struct RoleOnboardingGuide: Hashable, Sendable {
    public let role: WorkspaceRole
    public let headline: String
    public let summary: String
    public let steps: [OnboardingStep]
    public let recommendedDestinations: [OnboardingDestinationHint]
    public let recommendedCommands: [CommandRecipe]
}

