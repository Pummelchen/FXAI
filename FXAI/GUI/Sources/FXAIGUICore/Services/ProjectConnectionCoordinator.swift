import Foundation

public struct ProjectConnectionCoordinator {
    private let defaultProjectRootProvider: @Sendable () -> URL?

    public init(defaultProjectRootProvider: @escaping @Sendable () -> URL? = {
        ProjectPathResolver.defaultProjectRoot()
    }) {
        self.defaultProjectRootProvider = defaultProjectRootProvider
    }

    public func resolve(
        currentProjectRoot: URL?,
        preferredProjectRoot: URL?,
        autoReconnectEnabled: Bool
    ) -> ProjectConnectionResolution {
        if let currentProjectRoot, ProjectPathResolver.isProjectRoot(currentProjectRoot) {
            return ProjectConnectionResolution(
                activeProjectRoot: currentProjectRoot,
                preferredProjectRoot: currentProjectRoot,
                state: .connected
            )
        }

        if autoReconnectEnabled {
            if let preferredProjectRoot, ProjectPathResolver.isProjectRoot(preferredProjectRoot) {
                return ProjectConnectionResolution(
                    activeProjectRoot: preferredProjectRoot,
                    preferredProjectRoot: preferredProjectRoot,
                    state: .connected
                )
            }

            if let defaultProjectRoot = defaultProjectRootProvider() {
                return ProjectConnectionResolution(
                    activeProjectRoot: defaultProjectRoot,
                    preferredProjectRoot: preferredProjectRoot ?? defaultProjectRoot,
                    state: .connected
                )
            }

            return ProjectConnectionResolution(
                activeProjectRoot: nil,
                preferredProjectRoot: preferredProjectRoot,
                state: .waitingForProject
            )
        }

        return ProjectConnectionResolution(
            activeProjectRoot: nil,
            preferredProjectRoot: preferredProjectRoot,
            state: .disconnectedByUser
        )
    }
}
