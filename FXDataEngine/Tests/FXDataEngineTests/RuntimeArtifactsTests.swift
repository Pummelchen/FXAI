import XCTest
@testable import FXDataEngine

final class RuntimeArtifactsTests: XCTestCase {
    private func temporaryRepository() throws -> (URL, RuntimeArtifactFileRepository) {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("RuntimeArtifactsTests-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        return (root, RuntimeArtifactFileRepository(rootURL: root))
    }

    private func read(_ relativePath: String, root: URL) throws -> String {
        try String(contentsOf: root.appendingPathComponent(relativePath), encoding: .utf8)
    }

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

    func testManifestRowsSerializeWithLegacyHeadersAndFormatting() {
        let persistence = PersistenceCoverageManifestRow(
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
        XCTAssertEqual(persistence.tsvColumns[2], "full_native")
        XCTAssertEqual(persistence.tsvColumns[7], "65")
        XCTAssertEqual(persistence.tsvColumns[12], "1")

        let macro = MacroDatasetManifestRow(
            symbol: "EUR/USD",
            schemaVersion: 3,
            recordCount: 12,
            avgImportance: 0.42,
            leakageGuardScore: 0.91,
            leakageSafe: true
        )
        XCTAssertEqual(MacroDatasetManifestRow.header.count, 28)
        XCTAssertEqual(macro.tsvColumns[0], "EUR_USD")
        XCTAssertEqual(macro.tsvColumns[17], "0.420000")
        XCTAssertEqual(macro.tsvColumns[27], "1")

        let stage = RuntimePerformanceManifestRow.stage(.router, meanMS: 1.25, maxMS: 3.5, observations: 9, activeModels: 4)
        let plugin = RuntimePerformanceManifestRow.plugin(
            aiID: 4,
            aiName: "Plugin\tName",
            predictMeanMS: 0.5,
            predictMaxMS: 1.5,
            predictObservations: 7,
            updateMeanMS: 0.25,
            updateMaxMS: 0.75,
            updateObservations: 3,
            workingSetKB: 42.1254,
            activeModels: 4
        )
        XCTAssertEqual(RuntimePerformanceManifestRow.header.count, 15)
        XCTAssertEqual(stage.tsvColumns[0], "stage")
        XCTAssertEqual(stage.tsvColumns[1], "router")
        XCTAssertEqual(stage.tsvColumns[14], "4")
        XCTAssertEqual(plugin.tsvColumns[0], "plugin")
        XCTAssertEqual(plugin.tsvColumns[7], "0.500000")
        XCTAssertEqual(plugin.tsvColumns[13], "42.125")

        let shadow = ShadowFleetLedgerRow(
            symbol: "EUR/USD",
            aiID: 1,
            aiName: "Shadow",
            family: .transformer,
            metaWeight: 3.5,
            reliability: 0.5,
            globalEdge: 0.25,
            contextEdge: 0.1,
            contextRegret: 0.2,
            portfolioObjective: 0.3,
            portfolioStability: 0.4,
            portfolioCorrelationPenalty: 0.1,
            portfolioDiversification: 0.2,
            routeValue: 0.6,
            routeRegret: 0.2,
            routeCounterfactual: 0.5,
            regimeID: 2,
            horizonMinutes: 13,
            observations: 5,
            policyEnterProb: 0.7,
            policyNoTradeProb: 1.4,
            policyExitProb: 0.2,
            policyAddProb: 0.1,
            policyReduceProb: 0.05,
            policyTimeoutProb: 0.0,
            policyTightenProb: 0.3,
            policyPortfolioFit: 0.6,
            policyCapitalEfficiency: 0.8,
            policyLifecycleAction: .enter,
            portfolioPressure: 1.1,
            controlPlaneScore: 1.2,
            portfolioSupervisorScore: 1.3
        )
        XCTAssertEqual(ShadowFleetLedgerRow.header.count, 33)
        XCTAssertEqual(shadow.tsvColumns[0], "EUR_USD")
        XCTAssertEqual(shadow.tsvColumns[3], String(AIFamily.transformer.rawValue))
        XCTAssertEqual(shadow.tsvColumns[4], "3.000000")
        XCTAssertEqual(shadow.tsvColumns[21], "1.000000")
        XCTAssertEqual(shadow.tsvColumns[29], String(PolicyLifecycleAction.enter.rawValue))
        XCTAssertEqual(shadow.shadowScore, 0.366, accuracy: 1e-12)
    }

    func testRuntimeArtifactFileRepositoryWritesTSVManifests() throws {
        let (root, repository) = try temporaryRepository()
        defer { try? FileManager.default.removeItem(at: root) }

        let persistence = PersistenceCoverageManifestRow(
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
        try repository.writePersistenceManifest(symbol: "EUR/USD", rows: [persistence])
        try repository.writeFeatureRegistryManifest(symbol: "EUR/USD")
        try repository.writeMacroDatasetManifest(
            symbol: "EUR/USD",
            row: MacroDatasetManifestRow(symbol: "EUR/USD", schemaVersion: 3, recordCount: 12, leakageSafe: true)
        )
        try repository.writeRuntimePerformanceManifest(
            symbol: "EUR/USD",
            rows: [.stage(.router, meanMS: 1.25, maxMS: 3.5, observations: 9, activeModels: 4)]
        )
        try repository.writeShadowFleetLedger(
            symbol: "EUR/USD",
            rows: [
                ShadowFleetLedgerRow(
                    symbol: "EUR/USD",
                    aiID: 1,
                    aiName: "Shadow",
                    family: .transformer,
                    metaWeight: 1.0,
                    reliability: 0.5,
                    globalEdge: 0.25,
                    contextEdge: 0.1,
                    contextRegret: 0.2,
                    portfolioObjective: 0.3,
                    portfolioStability: 0.4,
                    portfolioCorrelationPenalty: 0.1,
                    portfolioDiversification: 0.2,
                    routeValue: 0.6,
                    routeRegret: 0.2,
                    routeCounterfactual: 0.5,
                    regimeID: 2,
                    horizonMinutes: 13,
                    observations: 5,
                    policyEnterProb: 0.7,
                    policyNoTradeProb: 0.4,
                    policyExitProb: 0.2,
                    policyAddProb: 0.1,
                    policyReduceProb: 0.05,
                    policyTimeoutProb: 0.0,
                    policyTightenProb: 0.3,
                    policyPortfolioFit: 0.6,
                    policyCapitalEfficiency: 0.8,
                    policyLifecycleAction: .enter,
                    portfolioPressure: 1.1,
                    controlPlaneScore: 1.2,
                    portfolioSupervisorScore: 1.3
                )
            ]
        )

        XCTAssertTrue(try read(RuntimeArtifactPaths.persistenceManifestFile(symbol: "EUR/USD"), root: root).contains("native checkpoint verified"))
        XCTAssertEqual(
            try read(RuntimeArtifactPaths.featureManifestFile(symbol: "EUR/USD"), root: root)
                .components(separatedBy: "\r\n")
                .filter { !$0.isEmpty }
                .count,
            FXDataEngineConstants.aiFeatures + 1
        )
        XCTAssertTrue(try read(RuntimeArtifactPaths.macroManifestFile(symbol: "EUR/USD"), root: root).contains("EUR_USD\t3\t12"))
        XCTAssertTrue(try read(RuntimeArtifactPaths.performanceManifestFile(symbol: "EUR/USD"), root: root).contains("stage\trouter"))
        XCTAssertTrue(try read(RuntimeArtifactPaths.shadowLedgerFile(symbol: "EUR/USD"), root: root).contains("EUR_USD\t1\tShadow"))
    }
}
