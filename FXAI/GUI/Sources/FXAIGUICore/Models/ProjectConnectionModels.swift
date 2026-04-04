import Foundation

public enum ProjectConnectionState: String, Codable, Hashable, Sendable {
    case connected
    case waitingForProject = "waiting_for_project"
    case disconnectedByUser = "disconnected_by_user"

    public var title: String {
        switch self {
        case .connected: "Connected"
        case .waitingForProject: "Waiting"
        case .disconnectedByUser: "Disconnected"
        }
    }
}

public struct ProjectConnectionResolution: Hashable, Sendable {
    public let activeProjectRoot: URL?
    public let preferredProjectRoot: URL?
    public let state: ProjectConnectionState

    public init(activeProjectRoot: URL?, preferredProjectRoot: URL?, state: ProjectConnectionState) {
        self.activeProjectRoot = activeProjectRoot
        self.preferredProjectRoot = preferredProjectRoot
        self.state = state
    }
}
