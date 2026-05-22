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
        XCTAssertEqual(header.aiCount, 65)
        XCTAssertEqual(header.regimeCount, 12)
        XCTAssertEqual(header.maxHorizons, 8)
        XCTAssertEqual(header.conformalDepth, 96)
        XCTAssertEqual(header.reliabilityPendingCapacity, 2_048)

        var incompatible = header
        incompatible.version = 14
        XCTAssertFalse(incompatible.isCompatibleWithCurrentContract)
    }

    func testRuntimeArtifactBinaryCodecPreservesMQLHeaderAndPayload() throws {
        let header = RuntimeArtifactHeader()
        let encodedHeader = try RuntimeArtifactBinaryCodec.encodeHeader(header)

        XCTAssertEqual(encodedHeader.count, RuntimeArtifactHeader.binaryByteCount)
        XCTAssertEqual(Array(encodedHeader.prefix(8)), [15, 0, 0, 0, 180, 0, 0, 0])
        XCTAssertEqual(try RuntimeArtifactBinaryCodec.decodeHeader(from: encodedHeader), header)

        let payload = Data([0xAA, 0xBB, 0xCC, 0xDD])
        let envelope = RuntimeArtifactEnvelope(header: header, payload: payload)
        let decoded = try RuntimeArtifactBinaryCodec.decodeEnvelope(
            from: try RuntimeArtifactBinaryCodec.encodeEnvelope(envelope)
        )
        XCTAssertEqual(decoded.header, header)
        XCTAssertEqual(decoded.payload, payload)
        XCTAssertTrue(decoded.isCompatibleWithCurrentContract)
        XCTAssertEqual(decoded.payloadByteCount, 4)

        var incompatibleHeader = header
        incompatibleHeader.featureCount = 999
        let incompatibleEnvelope = RuntimeArtifactEnvelope(header: incompatibleHeader, payload: Data())
        XCTAssertFalse(incompatibleEnvelope.isCompatibleWithCurrentContract)
        XCTAssertThrowsError(try RuntimeArtifactBinaryCodec.decodeHeader(from: Data(count: 12)))
    }

    func testRuntimeArtifactPayloadLayoutMaterializesLegacySections() throws {
        let layout = RuntimeArtifactPayloadMaterializer.layout()
        XCTAssertEqual(layout.sections.count, RuntimeArtifactPayloadSectionKind.allCases.count)
        XCTAssertEqual(layout.sections.first?.offset, 0)

        let windows = try XCTUnwrap(layout.section(.normalizationWindows))
        let replay = try XCTUnwrap(layout.section(.replayReservoir))
        let featureDrift = try XCTUnwrap(layout.section(.featureDrift))
        let sharedTransfer = try XCTUnwrap(layout.section(.sharedTransferTensor))
        let analog = try XCTUnwrap(layout.section(.analogMemory))
        let broker = try XCTUnwrap(layout.section(.brokerExecution))

        XCTAssertEqual(windows.byteCount, 728)
        XCTAssertEqual(replay.byteCount, 645_576)
        XCTAssertEqual(featureDrift.byteCount, 396)
        XCTAssertEqual(sharedTransfer.owner, .fxPlugins)
        XCTAssertEqual(analog.owner, .fxDataEngine)
        XCTAssertEqual(broker.owner, .fxBacktest)
        XCTAssertEqual(analog.offset + analog.byteCount, broker.offset)
        XCTAssertEqual(layout.payloadByteCount, broker.offset + broker.byteCount)

        let payload = Data(count: layout.payloadByteCount)
        XCTAssertEqual(try layout.slice(payload, kind: .analogMemory).count, RuntimeArtifactPayloadMaterializer.analogMemoryByteCount)
    }

    func testPreparedSampleCodecRoundTripsLegacyBinaryOrderWithNoSpreadDefaults() throws {
        let modelInput = [1.0, 0.25, -0.50] + Array(repeating: 0.0, count: FXDataEngineConstants.aiWeights - 3)
        let sample = PreparedTrainingSample(
            valid: true,
            labelClass: .buy,
            regimeID: 2,
            horizonMinutes: 15,
            horizonSlot: 3,
            movePoints: 12.5,
            minMovePoints: 8.0,
            costPoints: 0.0,
            sampleWeight: 1.4,
            qualityScore: 1.7,
            mfePoints: 20.0,
            maePoints: 4.0,
            timeToHitFraction: 0.25,
            pathFlags: [.dualHit, .slowHit],
            pathRisk: 0.30,
            fillRisk: 0.0,
            maskedStepTarget: 1.2,
            nextVolumeTarget: 3.4,
            regimeShiftTarget: 0.2,
            contextLeadTarget: 0.7,
            pointValue: 0.00001,
            domainHash: 0.42,
            sampleTimeUTC: 1_704_067_200,
            x: modelInput
        )
        let runtimeSample = RuntimeArtifactPreparedSample(sample: sample)
        let encoded = try RuntimeArtifactPreparedSampleCodec.encode(runtimeSample)
        let decoded = try RuntimeArtifactPreparedSampleCodec.decode(from: encoded)

        XCTAssertEqual(encoded.count, RuntimeArtifactPreparedSampleCodec.byteCount)
        XCTAssertEqual(encoded.prefix(20), Data([1, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 15, 0, 0, 0, 3, 0, 0, 0]))
        XCTAssertEqual(decoded.valid, true)
        XCTAssertEqual(decoded.labelClass, .buy)
        XCTAssertEqual(decoded.liquidityStress, 0.0)
        XCTAssertEqual(decoded.traceLiquidityMeanRatio, 0.0)
        XCTAssertEqual(decoded.traceGapRatio, 0.30, accuracy: 1e-12)
        XCTAssertEqual(decoded.traceReversalRatio, 0.30, accuracy: 1e-12)
        XCTAssertEqual(decoded.preparedTrainingSample.fillRisk, 0.0, accuracy: 1e-12)
        XCTAssertEqual(decoded.x[0], 1.0, accuracy: 1e-12)
        XCTAssertEqual(decoded.x[2], -0.50, accuracy: 1e-12)

        let legacyZeroX = RuntimeArtifactPreparedSample(x: Array(repeating: 0.0, count: FXDataEngineConstants.aiWeights))
        let decodedLegacyZeroX = try RuntimeArtifactPreparedSampleCodec.decode(
            from: try RuntimeArtifactPreparedSampleCodec.encode(legacyZeroX)
        )
        XCTAssertEqual(decodedLegacyZeroX.x[0], 0.0, accuracy: 1e-12)

        var fillRiskSample = sample
        fillRiskSample.fillRisk = 0.25
        let fillRiskRuntimeSample = RuntimeArtifactPreparedSample(sample: fillRiskSample)
        XCTAssertEqual(fillRiskRuntimeSample.liquidityStress, 0.0, accuracy: 1e-12)
        XCTAssertEqual(fillRiskRuntimeSample.preparedTrainingSample.fillRisk, 0.25, accuracy: 1e-12)
    }

    func testPreparedSampleJSONDecodesLegacyStressKeysAndEncodesLiquidityNames() throws {
        let legacyJSON = Data("""
        {
          "valid": true,
          "labelClass": 1,
          "spreadStress": 0.7,
          "traceSpreadMeanRatio": 0.8,
          "traceSpreadPeakRatio": 1.2,
          "x": [1.0, 0.25, -0.5]
        }
        """.utf8)

        let decoded = try JSONDecoder().decode(RuntimeArtifactPreparedSample.self, from: legacyJSON)
        XCTAssertTrue(decoded.valid)
        XCTAssertEqual(decoded.labelClass, .buy)
        XCTAssertEqual(decoded.liquidityStress, 0.7, accuracy: 1e-12)
        XCTAssertEqual(decoded.traceLiquidityMeanRatio, 0.8, accuracy: 1e-12)
        XCTAssertEqual(decoded.traceLiquidityPeakRatio, 1.2, accuracy: 1e-12)

        let encoded = String(data: try JSONEncoder().encode(decoded), encoding: .utf8) ?? ""
        XCTAssertTrue(encoded.contains("\"liquidityStress\""))
        XCTAssertTrue(encoded.contains("\"traceLiquidityMeanRatio\""))
        XCTAssertFalse(encoded.contains("\"spreadStress\""))
        XCTAssertFalse(encoded.contains("\"traceSpreadMeanRatio\""))
    }

    func testFeatureDriftCodecRoundTripsLegacyGroupShape() throws {
        let state = RuntimeFeatureDriftState(
            ready: true,
            lastTimeUTC: 1_704_067_200,
            groups: [
                RuntimeFeatureDriftGroupState(
                    baselineObservations: 10,
                    liveObservations: 4,
                    baselineMean: 0.20,
                    baselineAbs: 0.30,
                    liveMean: -0.10,
                    liveAbs: 0.40,
                    driftEMA: 0.50
                )
            ]
        )

        let encoded = try RuntimeFeatureDriftCodec.encode(state)
        let decoded = try RuntimeFeatureDriftCodec.decode(from: encoded)

        XCTAssertEqual(encoded.count, RuntimeFeatureDriftCodec.byteCount)
        XCTAssertTrue(decoded.ready)
        XCTAssertEqual(decoded.lastTimeUTC, 1_704_067_200)
        XCTAssertEqual(decoded.groups.count, FeatureGroup.allCases.count)
        XCTAssertEqual(decoded.groups[0].baselineObservations, 10)
        XCTAssertEqual(decoded.groups[0].liveMean, -0.10, accuracy: 1e-12)
        XCTAssertEqual(decoded.groups[1].baselineObservations, 0)
    }

    func testAnalogMemoryCodecRoundTripsLegacySectionAndSanitizesDecodedShape() throws {
        var store = AnalogMemoryStore()
        let modelInput = [1.0, 0.10, -0.20, 0.30, -0.40] + Array(repeating: 0.0, count: FXDataEngineConstants.aiWeights - 5)
        for index in 0..<3 {
            store.update(
                modelInput: modelInput,
                regimeID: index,
                sessionBucket: index,
                horizonMinutes: 5 + index,
                domainHash: 0.25,
                movePoints: Double(index + 1) * 10.0,
                minMovePoints: 5.0,
                qualityScore: 1.2,
                pathRisk: 0.20,
                fillRisk: 0.0,
                sampleTimeUTC: 1_704_067_200 + Int64(index),
                sampleWeight: 1.0
            )
        }

        let encoded = try RuntimeArtifactAnalogMemoryCodec.encode(store)
        let decoded = try RuntimeArtifactAnalogMemoryCodec.decode(from: encoded)

        XCTAssertEqual(encoded.count, RuntimeArtifactAnalogMemoryCodec.byteCount)
        XCTAssertTrue(decoded.ready)
        XCTAssertEqual(decoded.size, 3)
        XCTAssertEqual(decoded.head, 3)
        XCTAssertEqual(decoded.entries[0].sampleTimeUTC, 1_704_067_200)
        XCTAssertEqual(decoded.entries[2].regimeID, 2)
        XCTAssertEqual(decoded.entries[2].fillRisk, 0.0, accuracy: 1e-12)

        let malformed = AnalogMemoryStore(
            capacity: FXDataEngineConstants.analogMemoryCapacity,
            head: 999,
            size: 999,
            entries: [AnalogMemoryEntry()]
        )
        XCTAssertEqual(malformed.size, FXDataEngineConstants.analogMemoryCapacity)
        XCTAssertEqual(malformed.head, 0)
    }

    func testRuntimeArtifactSavePolicyMatchesLegacyDirtyThrottle() {
        XCTAssertFalse(RuntimeArtifactSavePolicy.shouldSave(
            dirty: false,
            lastSaveTimeUTC: 1_000,
            barTimeUTC: 2_000,
            nowUTC: 2_000
        ))
        XCTAssertFalse(RuntimeArtifactSavePolicy.shouldSave(
            dirty: true,
            lastSaveTimeUTC: 1_000,
            barTimeUTC: 1_899,
            nowUTC: 1_899
        ))
        XCTAssertTrue(RuntimeArtifactSavePolicy.shouldSave(
            dirty: true,
            lastSaveTimeUTC: 1_000,
            barTimeUTC: 1_900,
            nowUTC: 1_900
        ))
        XCTAssertTrue(RuntimeArtifactSavePolicy.shouldSave(
            dirty: true,
            lastSaveTimeUTC: 0,
            barTimeUTC: 0,
            nowUTC: 0
        ))
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

        let envelope = RuntimeArtifactEnvelope(payload: Data([1, 2, 3, 4, 5]))
        try repository.writeRuntimeArtifact(symbol: "EUR/USD", envelope: envelope)
        let loadedEnvelope = try XCTUnwrap(repository.readRuntimeArtifact(symbol: "EUR/USD"))
        XCTAssertEqual(loadedEnvelope.header, RuntimeArtifactHeader())
        XCTAssertEqual(loadedEnvelope.payload, envelope.payload)
        XCTAssertEqual(
            try repository.runtimeArtifactFileSize(symbol: "EUR/USD"),
            Int64(RuntimeArtifactHeader.binaryByteCount + envelope.payload.count)
        )

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
