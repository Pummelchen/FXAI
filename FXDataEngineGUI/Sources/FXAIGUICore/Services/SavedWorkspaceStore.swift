import Foundation

public struct SavedWorkspaceStore {
    public let storageURL: URL

    public init(storageURL: URL = SavedWorkspaceStore.defaultStorageURL()) {
        self.storageURL = storageURL
    }

    public func load() -> FXAIGUIPersistenceState {
        guard let data = try? Data(contentsOf: storageURL) else {
            return FXAIGUIPersistenceState()
        }

        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601

        guard let state = try? decoder.decode(FXAIGUIPersistenceState.self, from: data) else {
            return FXAIGUIPersistenceState()
        }

        return state
    }

    public func save(_ state: FXAIGUIPersistenceState) throws {
        let fm = FileManager.default
        try fm.createDirectory(
            at: storageURL.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )

        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        let data = try encoder.encode(state)
        try data.write(to: storageURL, options: .atomic)
    }

    public static func defaultStorageURL() -> URL {
        let fm = FileManager.default
        let base = fm.urls(for: .applicationSupportDirectory, in: .userDomainMask).first
            ?? URL(fileURLWithPath: NSHomeDirectory(), isDirectory: true)
                .appendingPathComponent("Library/Application Support", isDirectory: true)

        return base
            .appendingPathComponent("FXAIGUI", isDirectory: true)
            .appendingPathComponent("workspace_state.json")
    }
}

