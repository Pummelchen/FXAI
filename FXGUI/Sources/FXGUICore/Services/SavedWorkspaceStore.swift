import Foundation

public struct SavedWorkspaceStore {
    public let storageURL: URL

    public init(storageURL: URL = SavedWorkspaceStore.defaultStorageURL()) {
        self.storageURL = storageURL
    }

    public func load() -> FXGUIPersistenceState {
        guard let data = try? Data(contentsOf: storageURL) else {
            return FXGUIPersistenceState()
        }

        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601

        guard let state = try? decoder.decode(FXGUIPersistenceState.self, from: data) else {
            return FXGUIPersistenceState()
        }

        return state
    }

    public func save(_ state: FXGUIPersistenceState) throws {
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
            .appendingPathComponent("FXGUI", isDirectory: true)
            .appendingPathComponent("workspace_state.json")
    }
}

