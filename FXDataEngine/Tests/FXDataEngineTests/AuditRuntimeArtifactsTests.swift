import Foundation
import XCTest
@testable import FXDataEngine

final class AuditRuntimeArtifactsTests: XCTestCase {
    func testAuditRuntimeSafeKeyAndPathsMatchLegacyWrapper() {
        XCTAssertEqual(AuditRuntimeArtifactPaths.safeKey(""), "audit")
        XCTAssertEqual(AuditRuntimeArtifactPaths.safeKey("EUR/USD live:*?"), "EUR_USD_live___")
        XCTAssertEqual(
            AuditRuntimeArtifactPaths.runtimeArtifactFile(symbol: "EUR/USD live:*?"),
            "FXAI/Audit/Runtime/fxai_audit_runtime_EUR_USD_live___.bin"
        )
    }

    func testAuditRuntimeConformalEnvelopeRoundTripsMagicVersionAndState() throws {
        var state = ConformalCalibrationState()
        let slot = TrainingSampleTools.horizonSlot(horizonMinutes: 13)
        _ = state.pushScore(aiIndex: 2, regimeID: 3, horizonSlot: slot, classScore: 0.42, moveScore: 1.10, pathScore: 0.25)

        let encoded = try AuditRuntimeArtifactCodec.encodeConformalState(state)
        var reader = RuntimeArtifactBinaryReader(data: encoded)
        XCTAssertEqual(try reader.readInt32(), AuditRuntimeArtifactConstants.magic)
        XCTAssertEqual(try reader.readInt32(), AuditRuntimeArtifactConstants.version)

        let decoded = try AuditRuntimeArtifactCodec.decodeConformalState(from: encoded)
        XCTAssertEqual(
            decoded.quantile(aiIndex: 2, regimeID: 3, horizonSlot: slot, scoreKind: .classScore, fallback: 0.0),
            0.42,
            accuracy: 1e-12
        )
        XCTAssertThrowsError(try AuditRuntimeArtifactCodec.decodeConformalState(from: Data([0, 0, 0, 0, 1, 0, 0, 0])))
    }

    func testAuditRuntimeRepositorySavesLoadsAndThrottlesGlobalAndPluginState() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("AuditRuntimeArtifactsTests-\(UUID().uuidString)", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let repository = AuditRuntimeArtifactRepository(rootURL: root)

        var state = ConformalCalibrationState()
        let slot = TrainingSampleTools.horizonSlot(horizonMinutes: 21)
        _ = state.pushScore(aiIndex: 1, regimeID: 2, horizonSlot: slot, classScore: 0.31, moveScore: 0.90, pathScore: 0.15)
        let manifest = PluginManifestV4(
            aiID: 4,
            aiName: "Audit Plugin",
            family: .transformer,
            capabilityMask: [.selfTest, .onlineLearning]
        )
        let pluginState = AuditRuntimePluginState(manifest: manifest, data: Data([7, 8, 9]))

        XCTAssertNil(try repository.maybeSaveRuntimeArtifacts(
            symbol: "EUR/USD live",
            conformalState: state,
            pluginState: pluginState,
            dirty: false,
            lastSaveTimeUTC: 1_000,
            barTimeUTC: 2_000,
            nowUTC: 2_000
        ))
        XCTAssertNil(try repository.maybeSaveRuntimeArtifacts(
            symbol: "EUR/USD live",
            conformalState: state,
            pluginState: pluginState,
            dirty: true,
            lastSaveTimeUTC: 1_000,
            barTimeUTC: 1_899,
            nowUTC: 1_899
        ))

        let saved = try repository.maybeSaveRuntimeArtifacts(
            symbol: "EUR/USD live",
            conformalState: state,
            pluginState: pluginState,
            dirty: true,
            lastSaveTimeUTC: 1_000,
            barTimeUTC: 1_900,
            nowUTC: 1_900
        )
        XCTAssertTrue(saved?.savedAny ?? false)
        XCTAssertEqual(saved?.lastSaveTimeUTC, 1_900)
        XCTAssertNotNil(saved?.pluginStateFile)

        let loaded = try repository.loadRuntimeArtifacts(
            symbol: "EUR/USD live",
            pluginManifest: manifest,
            nowUTC: 2_100
        )
        XCTAssertTrue(loaded.loadedAny)
        XCTAssertEqual(loaded.lastSaveTimeUTC, 2_100)
        XCTAssertEqual(loaded.pluginState?.data, Data([7, 8, 9]))
        XCTAssertEqual(
            loaded.conformalState?.quantile(aiIndex: 1, regimeID: 2, horizonSlot: slot, scoreKind: .classScore, fallback: 0.0) ?? 0.0,
            0.31,
            accuracy: 1e-12
        )

        let corruptURL = root.appendingPathComponent(repository.runtimeArtifactFile(symbol: "GBP/USD"), isDirectory: false)
        try FileManager.default.createDirectory(
            at: corruptURL.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        try Data([0, 0, 0, 0, 1, 0, 0, 0]).write(to: corruptURL)
        let pluginURL = root.appendingPathComponent(repository.pluginStateFile(symbol: "GBP/USD", manifest: manifest), isDirectory: false)
        try Data([1, 2, 3]).write(to: pluginURL)

        let corruptLoaded = try repository.loadRuntimeArtifacts(symbol: "GBP/USD", pluginManifest: manifest, nowUTC: 2_200)
        XCTAssertFalse(corruptLoaded.loadedConformalState)
        XCTAssertTrue(corruptLoaded.loadedPluginState)
        XCTAssertEqual(corruptLoaded.pluginState?.data, Data([1, 2, 3]))
    }
}
