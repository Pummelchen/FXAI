import Foundation

public enum AuditRuntimeArtifactConstants {
    public static let directory = "FXAI/Audit/Runtime"
    public static let magic = 1_179_874_889
    public static let version = 1
}

public enum AuditRuntimeArtifactPaths {
    public static func safeKey(_ raw: String, defaultValue: String = "audit") -> String {
        var key = raw.isEmpty ? defaultValue : raw
        for character in ["\\", "/", ":", "*", "?", "\"", "<", ">", "|", " "] {
            key = key.replacingOccurrences(of: character, with: "_")
        }
        return key.isEmpty ? defaultValue : key
    }

    public static func runtimeArtifactFile(symbol: String) -> String {
        "\(AuditRuntimeArtifactConstants.directory)/fxai_audit_runtime_\(safeKey(symbol)).bin"
    }

    public static func pluginStateFile(symbol: String, manifest: PluginManifestV4) -> String {
        "\(AuditRuntimeArtifactConstants.directory)/fxai_audit_plugin_\(safeKey(symbol))_\(safeKey(manifest.aiName, defaultValue: "plugin")).bin"
    }
}

public struct AuditRuntimePluginState: Codable, Hashable, Sendable {
    public var manifest: PluginManifestV4
    public var data: Data

    public init(manifest: PluginManifestV4, data: Data) {
        self.manifest = manifest
        self.data = data
    }
}

public struct AuditRuntimeArtifactSaveResult: Codable, Hashable, Sendable {
    public var savedConformalState: Bool
    public var savedPluginState: Bool
    public var lastSaveTimeUTC: Int64
    public var conformalStateFile: String
    public var pluginStateFile: String?

    public var savedAny: Bool {
        savedConformalState || savedPluginState
    }
}

public struct AuditRuntimeArtifactLoadResult: Codable, Hashable, Sendable {
    public var loadedConformalState: Bool
    public var loadedPluginState: Bool
    public var lastSaveTimeUTC: Int64
    public var conformalState: ConformalCalibrationState?
    public var pluginState: AuditRuntimePluginState?

    public var loadedAny: Bool {
        loadedConformalState || loadedPluginState
    }
}

public enum AuditRuntimeArtifactCodec {
    public static func encodeConformalState(_ state: ConformalCalibrationState) throws -> Data {
        var writer = RuntimeArtifactBinaryWriter()
        try writer.appendInt32(AuditRuntimeArtifactConstants.magic)
        try writer.appendInt32(AuditRuntimeArtifactConstants.version)
        var data = writer.data
        data.append(try RuntimeConformalCalibrationCodec.encode(state))
        return data
    }

    public static func decodeConformalState(from data: Data) throws -> ConformalCalibrationState {
        var reader = RuntimeArtifactBinaryReader(data: data)
        let magic = try reader.readInt32()
        let version = try reader.readInt32()
        guard magic == AuditRuntimeArtifactConstants.magic,
              version == AuditRuntimeArtifactConstants.version else {
            throw FXDataEngineError.validation("audit runtime artifact header")
        }
        let payload = data.suffix(reader.remainingByteCount)
        return try RuntimeConformalCalibrationCodec.decode(from: Data(payload))
    }
}

public struct AuditRuntimeArtifactRepository: Sendable {
    public var rootURL: URL

    public init(rootURL: URL) {
        self.rootURL = rootURL
    }

    public func runtimeArtifactFile(symbol: String) -> String {
        AuditRuntimeArtifactPaths.runtimeArtifactFile(symbol: symbol)
    }

    public func pluginStateFile(symbol: String, manifest: PluginManifestV4) -> String {
        AuditRuntimeArtifactPaths.pluginStateFile(symbol: symbol, manifest: manifest)
    }

    public func saveRuntimeArtifacts(
        symbol: String,
        conformalState: ConformalCalibrationState,
        pluginState: AuditRuntimePluginState? = nil,
        nowUTC: Int64
    ) throws -> AuditRuntimeArtifactSaveResult {
        let conformalPath = runtimeArtifactFile(symbol: symbol)
        let conformalURL = url(forRelativePath: conformalPath)
        try createParentDirectory(for: conformalURL)
        try AuditRuntimeArtifactCodec.encodeConformalState(conformalState).write(to: conformalURL, options: .atomic)

        var pluginPath: String?
        if let pluginState {
            let path = pluginStateFile(symbol: symbol, manifest: pluginState.manifest)
            let url = url(forRelativePath: path)
            try createParentDirectory(for: url)
            try pluginState.data.write(to: url, options: .atomic)
            pluginPath = path
        }

        return AuditRuntimeArtifactSaveResult(
            savedConformalState: true,
            savedPluginState: pluginState != nil,
            lastSaveTimeUTC: max(0, nowUTC),
            conformalStateFile: conformalPath,
            pluginStateFile: pluginPath
        )
    }

    public func loadRuntimeArtifacts(
        symbol: String,
        pluginManifest: PluginManifestV4? = nil,
        nowUTC: Int64
    ) throws -> AuditRuntimeArtifactLoadResult {
        var conformalState: ConformalCalibrationState?
        let conformalPath = runtimeArtifactFile(symbol: symbol)
        let conformalURL = url(forRelativePath: conformalPath)
        if FileManager.default.fileExists(atPath: conformalURL.path) {
            conformalState = try? AuditRuntimeArtifactCodec.decodeConformalState(from: Data(contentsOf: conformalURL))
        }

        var pluginState: AuditRuntimePluginState?
        if let pluginManifest {
            let pluginURL = url(forRelativePath: pluginStateFile(symbol: symbol, manifest: pluginManifest))
            if FileManager.default.fileExists(atPath: pluginURL.path) {
                pluginState = try? AuditRuntimePluginState(
                    manifest: pluginManifest,
                    data: Data(contentsOf: pluginURL)
                )
            }
        }

        return AuditRuntimeArtifactLoadResult(
            loadedConformalState: conformalState != nil,
            loadedPluginState: pluginState != nil,
            lastSaveTimeUTC: (conformalState != nil || pluginState != nil) ? max(0, nowUTC) : 0,
            conformalState: conformalState,
            pluginState: pluginState
        )
    }

    public func maybeSaveRuntimeArtifacts(
        symbol: String,
        conformalState: ConformalCalibrationState,
        pluginState: AuditRuntimePluginState? = nil,
        dirty: Bool,
        lastSaveTimeUTC: Int64,
        barTimeUTC: Int64,
        nowUTC: Int64
    ) throws -> AuditRuntimeArtifactSaveResult? {
        guard RuntimeArtifactSavePolicy.shouldSave(
            dirty: dirty,
            lastSaveTimeUTC: lastSaveTimeUTC,
            barTimeUTC: barTimeUTC,
            nowUTC: nowUTC
        ) else {
            return nil
        }
        let effectiveNow = barTimeUTC > 0 ? barTimeUTC : nowUTC
        return try saveRuntimeArtifacts(
            symbol: symbol,
            conformalState: conformalState,
            pluginState: pluginState,
            nowUTC: effectiveNow
        )
    }

    private func url(forRelativePath path: String) -> URL {
        rootURL.appendingPathComponent(path, isDirectory: false)
    }

    private func createParentDirectory(for url: URL) throws {
        try FileManager.default.createDirectory(
            at: url.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
    }
}
