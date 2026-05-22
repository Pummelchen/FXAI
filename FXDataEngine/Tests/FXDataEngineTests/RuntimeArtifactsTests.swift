import XCTest
@testable import FXDataEngine

final class RuntimeArtifactsTests: XCTestCase {
    func testRuntimeArtifactPathsMatchLegacyShape() {
        XCTAssertEqual(RuntimeArtifactPaths.safeSymbol("EUR/USD:live"), "EUR_USD_live")
        XCTAssertEqual(RuntimeArtifactPaths.safeSymbol("EUR USD"), "EUR USD")
        XCTAssertEqual(
            RuntimeArtifactPaths.runtimeArtifactFile(symbol: "EUR/USD"),
            "FXAI/Runtime/fxai_runtime_EUR_USD.bin"
        )
        XCTAssertEqual(
            RuntimeArtifactPaths.persistenceManifestFile(symbol: "EURUSD"),
            "FXAI/Runtime/fxai_persistence_EURUSD.tsv"
        )
        XCTAssertEqual(
            RuntimeArtifactPaths.featureManifestFile(symbol: "EURUSD"),
            "FXAI/Runtime/fxai_features_EURUSD.tsv"
        )
        XCTAssertEqual(
            RuntimeArtifactPaths.macroManifestFile(symbol: "EURUSD"),
            "FXAI/Runtime/fxai_macro_EURUSD.tsv"
        )
        XCTAssertEqual(
            RuntimeArtifactPaths.performanceManifestFile(symbol: "EURUSD"),
            "FXAI/Runtime/fxai_perf_EURUSD.tsv"
        )
        XCTAssertEqual(
            RuntimeArtifactPaths.shadowLedgerFile(symbol: "EURUSD"),
            "FXAI/Runtime/fxai_shadow_EURUSD.tsv"
        )
    }

    func testRuntimeArtifactHeaderUsesLegacyVersionAndDimensions() {
        let header = RuntimeArtifactHeader()
        XCTAssertTrue(header.isCompatibleWithCurrentContract)
        XCTAssertEqual(header.version, 15)
        XCTAssertEqual(header.featureCount, 180)
        XCTAssertEqual(header.normalizationMethodCount, 17)
        XCTAssertEqual(header.normalizationRollWindowMax, 512)
        XCTAssertEqual(header.replayCapacity, 384)
        XCTAssertEqual(header.aiCount, 63)
        XCTAssertEqual(header.regimeCount, 12)
        XCTAssertEqual(header.maxHorizons, 8)
        XCTAssertEqual(header.conformalDepth, 96)
        XCTAssertEqual(header.reliabilityPendingCapacity, 2_048)

        var incompatible = header
        incompatible.version = 14
        XCTAssertFalse(incompatible.isCompatibleWithCurrentContract)
    }

    func testFeatureManifestRowsUseSwiftVolumeContractAndLegacyClipBounds() {
        let volume = FeatureRegistryManifestRow(featureIndex: 6)
        XCTAssertEqual(volume.featureName, "volume_norm")
        XCTAssertEqual(volume.featureGroup, "volume")
        XCTAssertEqual(volume.provenance, "price_bar")
        XCTAssertTrue(volume.leakageGuarded)
        XCTAssertEqual(volume.clipLower, 0.0)
        XCTAssertEqual(volume.clipUpper, 12.0)

        let mtfRange = FeatureRegistryManifestRow(featureIndex: FXDataEngineConstants.mainMTFFeatureOffset + 2)
        XCTAssertEqual(mtfRange.clipLower, -6.0)
        XCTAssertEqual(mtfRange.clipUpper, 6.0)

        let macroSurprise = FeatureRegistryManifestRow(featureIndex: FXDataEngineConstants.macroEventFeatureOffset + 3)
        XCTAssertEqual(macroSurprise.provenance, "event_macro")
        XCTAssertEqual(macroSurprise.clipLower, -6.0)
        XCTAssertEqual(macroSurprise.clipUpper, 6.0)
    }

    func testPersistenceCoverageManifestDerivesPromotionReadiness() {
        let blocked = PersistenceCoverageManifestRow(
            aiID: 1,
            aiName: "Stateful",
            referenceTier: .fullNative,
            coverageTag: "none",
            checkpointDepth: "none",
            persistent: true,
            stateVersion: 1,
            capabilityMask: [.stateful, .selfTest],
            nativeSnapshot: false,
            deterministicReplay: false,
            stateFileSize: 0,
            stateFile: "state.bin"
        )

        XCTAssertTrue(blocked.statefulCheckpoint)
        XCTAssertTrue(blocked.nativeRequired)
        XCTAssertFalse(blocked.promotionReady)
        XCTAssertTrue(blocked.coverageNote.contains("blocked from live promotion"))

        let ready = PersistenceCoverageManifestRow(
            aiID: 2,
            aiName: "Native",
            referenceTier: .fullNative,
            coverageTag: "native_model",
            checkpointDepth: "full",
            persistent: true,
            stateVersion: 2,
            capabilityMask: [.onlineLearning, .selfTest],
            nativeSnapshot: true,
            deterministicReplay: true,
            stateFileSize: 1024,
            stateFile: "state.bin"
        )

        XCTAssertTrue(ready.promotionReady)
        XCTAssertEqual(ready.coverageNote, "native checkpoint verified")
        XCTAssertEqual(PersistenceCoverageManifestRow.header.count, 16)
        XCTAssertEqual(FeatureRegistryManifestRow.header.count, 7)
        XCTAssertEqual(RuntimeStage.controlPlane.name, "control_plane")
    }
}
