import XCTest
@testable import FXDataEngine

final class PluginContractTests: XCTestCase {
    func testManifestValidationRequiresCoherentCapabilities() throws {
        let valid = PluginManifestV4(
            aiID: 2,
            aiName: "SequenceModel",
            family: .transformer,
            capabilityMask: [.selfTest, .onlineLearning, .replay, .windowContext],
            minSequenceBars: 4,
            maxSequenceBars: 32
        )
        XCTAssertNoThrow(try valid.validate())
        XCTAssertEqual(valid.resolvedSequenceBars(horizonMinutes: 3), 24)

        let invalid = PluginManifestV4(
            aiID: 3,
            aiName: "BadReplay",
            family: .linear,
            capabilityMask: [.selfTest, .replay]
        )
        XCTAssertThrowsError(try invalid.validate())
    }

    func testPredictRequestValidationChecksWindowContract() throws {
        let x = Array(repeating: 0.0, count: FXDataEngineConstants.aiWeights)
        let context = PluginContextV4(sequenceBars: 2, dataHasVolume: true)
        let valid = PredictRequestV4(valid: true, context: context, windowSize: 1, x: x, xWindow: [x])

        XCTAssertNoThrow(try valid.validate())

        let invalid = PredictRequestV4(valid: true, context: context, windowSize: 0, x: x)
        XCTAssertThrowsError(try invalid.validate())
    }

    func testManifestContextCompatibilityMatchesLegacyWindowRules() throws {
        let x = Array(repeating: 0.0, count: FXDataEngineConstants.aiWeights)
        let stateless = PluginManifestV4(
            aiID: 1,
            aiName: "Stateless",
            family: .linear,
            minHorizonMinutes: 2,
            maxHorizonMinutes: 10,
            minSequenceBars: 1,
            maxSequenceBars: 1
        )
        let windowed = PluginManifestV4(
            aiID: 2,
            aiName: "Windowed",
            family: .transformer,
            capabilityMask: [.selfTest, .windowContext],
            minHorizonMinutes: 2,
            maxHorizonMinutes: 10,
            minSequenceBars: 2,
            maxSequenceBars: 8
        )

        XCTAssertNoThrow(try PluginContractTools.validateCompatibility(
            manifest: stateless,
            context: PluginContextV4(horizonMinutes: 5, sequenceBars: 1)
        ))
        XCTAssertThrowsError(try PluginContractTools.validateCompatibility(
            manifest: stateless,
            context: PluginContextV4(horizonMinutes: 5, sequenceBars: 2)
        )) { error in
            XCTAssertEqual(String(describing: error), "validation failed: ctx.sequence_manifest")
        }
        XCTAssertThrowsError(try PluginContractTools.validateCompatibility(
            manifest: windowed,
            context: PluginContextV4(horizonMinutes: 1, sequenceBars: 2)
        )) { error in
            XCTAssertEqual(String(describing: error), "validation failed: ctx.horizon_manifest")
        }

        let request = PredictRequestV4(
            valid: true,
            context: PluginContextV4(horizonMinutes: 5, sequenceBars: 2),
            windowSize: 1,
            x: x,
            xWindow: [x]
        )
        XCTAssertNoThrow(try request.validate())
        XCTAssertNoThrow(try PluginContractTools.validateCompatibility(
            manifest: windowed,
            context: request.context
        ))
    }

    func testPluginInvocationWrappersRecordPerformanceAndWorkingSet() throws {
        let x = Array(repeating: 0.0, count: FXDataEngineConstants.aiWeights)
        let manifest = PluginManifestV4(
            aiID: 3,
            aiName: "Invocation",
            family: .transformer,
            capabilityMask: [.selfTest, .onlineLearning, .windowContext],
            minHorizonMinutes: 1,
            maxHorizonMinutes: 20,
            minSequenceBars: 2,
            maxSequenceBars: 8
        )
        let context = PluginContextV4(
            horizonMinutes: 5,
            sequenceBars: 4,
            sampleTimeUTC: 1_800_018_000,
            dataHasVolume: true
        )
        let predictRequest = PredictRequestV4(
            valid: true,
            context: context,
            windowSize: 3,
            x: x,
            xWindow: [x, x, x]
        )
        let trainRequest = TrainRequestV4(
            valid: true,
            context: context,
            labelClass: .buy,
            movePoints: 2.0,
            sampleWeight: 1.0,
            fillRisk: 0.1,
            windowSize: 3,
            x: x,
            xWindow: [x, x, x]
        )
        var plugin = InvocationRecordingPlugin(manifest: manifest)
        var performance = RuntimePerformanceState()

        let prediction = try PluginInvocationTools.predictViaV4(
            plugin: plugin,
            request: predictRequest,
            hyperParameters: HyperParameters(),
            performance: &performance,
            measuredElapsedMS: 2.5
        )
        try prediction.validate()
        try PluginInvocationTools.trainViaV4(
            plugin: &plugin,
            request: trainRequest,
            hyperParameters: HyperParameters(),
            performance: &performance,
            measuredElapsedMS: 4.0
        )

        let aiID = manifest.aiID
        XCTAssertEqual(plugin.trainCalls, 1)
        XCTAssertEqual(performance.pluginPredictObservations[aiID], 1)
        XCTAssertEqual(performance.pluginPredictMeanMS[aiID], 2.5, accuracy: 0.0)
        XCTAssertEqual(performance.pluginUpdateObservations[aiID], 1)
        XCTAssertEqual(performance.pluginUpdateMeanMS[aiID], 4.0, accuracy: 0.0)
        XCTAssertEqual(performance.lastTimeUTC, 1_800_018_000)
        XCTAssertEqual(
            performance.pluginWorkingSetKB[aiID],
            RuntimePerformanceState.estimatePluginWorkingSetKB(manifest: manifest, sequenceBars: 4),
            accuracy: 1e-12
        )
    }

    func testPluginInvocationRejectsIncompatibleContextBeforeCallingPlugin() throws {
        let x = Array(repeating: 0.0, count: FXDataEngineConstants.aiWeights)
        let manifest = PluginManifestV4(
            aiID: 4,
            aiName: "Stateless",
            family: .linear,
            minSequenceBars: 1,
            maxSequenceBars: 8
        )
        let request = PredictRequestV4(
            valid: true,
            context: PluginContextV4(sequenceBars: 2),
            windowSize: 1,
            x: x,
            xWindow: [x]
        )
        var performance = RuntimePerformanceState()

        XCTAssertThrowsError(try PluginInvocationTools.predictViaV4(
            plugin: InvocationRecordingPlugin(manifest: manifest),
            request: request,
            hyperParameters: HyperParameters(),
            performance: &performance,
            measuredElapsedMS: 1.0
        )) { error in
            XCTAssertEqual(String(describing: error), "validation failed: ctx.sequence_unexpected")
        }
        XCTAssertEqual(performance.pluginPredictObservations[manifest.aiID], 0)
        XCTAssertEqual(performance.pluginWorkingSetKB[manifest.aiID], 0.0, accuracy: 0.0)
    }

    func testPluginInvocationRecordsTimingWhenPluginThrowsAfterValidation() throws {
        let x = Array(repeating: 0.0, count: FXDataEngineConstants.aiWeights)
        let manifest = PluginManifestV4(aiID: 5, aiName: "Throwing", family: .linear)
        let context = PluginContextV4(sampleTimeUTC: 1_800_019_000)
        let predictRequest = PredictRequestV4(valid: true, context: context, x: x)
        let trainRequest = TrainRequestV4(
            valid: true,
            context: context,
            labelClass: .skip,
            movePoints: 0.0,
            sampleWeight: 1.0,
            x: x
        )
        var plugin = InvocationThrowingPlugin(manifest: manifest)
        var performance = RuntimePerformanceState()

        XCTAssertThrowsError(try PluginInvocationTools.predictViaV4(
            plugin: plugin,
            request: predictRequest,
            hyperParameters: HyperParameters(),
            performance: &performance,
            measuredElapsedMS: 1.25
        ))
        XCTAssertThrowsError(try PluginInvocationTools.trainViaV4(
            plugin: &plugin,
            request: trainRequest,
            hyperParameters: HyperParameters(),
            performance: &performance,
            measuredElapsedMS: 3.5
        ))

        XCTAssertEqual(performance.pluginPredictObservations[manifest.aiID], 1)
        XCTAssertEqual(performance.pluginPredictMeanMS[manifest.aiID], 1.25, accuracy: 0.0)
        XCTAssertEqual(performance.pluginUpdateObservations[manifest.aiID], 1)
        XCTAssertEqual(performance.pluginUpdateMeanMS[manifest.aiID], 3.5, accuracy: 0.0)
        XCTAssertEqual(performance.lastTimeUTC, 1_800_019_000)
    }

    func testPluginPersistenceMetadataMatchesLegacyStateTagsAndPaths() throws {
        let stateless = PluginManifestV4(
            aiID: 6,
            aiName: "Rule/Base",
            family: .ruleBased,
            referenceTier: .ruleBaseline
        )
        let replay = PluginManifestV4(
            aiID: 7,
            aiName: "Replay:Plugin",
            family: .transformer,
            capabilityMask: [.selfTest, .onlineLearning, .replay, .windowContext],
            minSequenceBars: 2,
            maxSequenceBars: 16
        )
        let nativeDescriptor = PluginPersistenceDescriptor(
            supportsPersistentState: true,
            stateVersion: 12,
            supportsDeterministicReplayCheckpoint: true,
            supportsNativeParameterSnapshot: true
        )

        XCTAssertEqual(PluginPersistenceConstants.directory, "FXAI/Runtime/Plugins")
        XCTAssertEqual(PluginPersistenceConstants.artifactVersion, 12)
        XCTAssertEqual(
            PluginPersistenceTools.stateFile(symbol: "EUR/USD:live", aiName: "Replay:Plugin"),
            "FXAI/Runtime/Plugins/fxai_plugin_EUR_USD_live_Replay_Plugin.bin"
        )
        XCTAssertEqual(PluginPersistenceTools.depthTag(manifest: stateless), "stateless")
        XCTAssertEqual(PluginPersistenceTools.defaultReferenceTier(aiID: AIModelID.chronos.rawValue), .surrogate)
        XCTAssertEqual(PluginPersistenceTools.defaultReferenceTier(aiID: AIModelID.lightgbm.rawValue), .compressedNative)
        XCTAssertEqual(PluginPersistenceTools.defaultReferenceTier(aiID: AIModelID.buyOnly.rawValue), .ruleBaseline)
        XCTAssertEqual(PluginPersistenceTools.coverageTag(manifest: stateless), "compressed_native")
        XCTAssertEqual(PluginPersistenceTools.depthTag(manifest: replay), "deterministic_replay")
        XCTAssertEqual(PluginPersistenceTools.coverageTag(manifest: replay), "native_replay")
        XCTAssertEqual(PluginPersistenceTools.depthTag(manifest: replay, descriptor: nativeDescriptor), "native_parameters")
        XCTAssertEqual(PluginPersistenceTools.coverageTag(manifest: replay, descriptor: nativeDescriptor), "native_model")

        let replayRow = try PluginPersistenceTools.coverageManifestRow(
            manifest: replay,
            symbol: "EUR/USD:live",
            stateFileSize: 2048
        )
        XCTAssertEqual(replayRow.coverageTag, "native_replay")
        XCTAssertEqual(replayRow.checkpointDepth, "deterministic_replay")
        XCTAssertTrue(replayRow.nativeRequired)
        XCTAssertFalse(replayRow.promotionReady)
        XCTAssertEqual(replayRow.stateFileSize, 2048)
        XCTAssertEqual(replayRow.stateFile, "FXAI/Runtime/Plugins/fxai_plugin_EUR_USD_live_Replay_Plugin.bin")

        let nativeRow = try PluginPersistenceTools.coverageManifestRow(
            manifest: replay,
            symbol: "EURUSD",
            descriptor: nativeDescriptor
        )
        XCTAssertEqual(nativeRow.coverageTag, "native_model")
        XCTAssertEqual(nativeRow.checkpointDepth, "native_parameters")
        XCTAssertTrue(nativeRow.promotionReady)
    }

    func testMLPayloadCarriesVolumeAvailability() {
        let x = Array(repeating: 0.0, count: FXDataEngineConstants.aiWeights)
        let context = PluginContextV4(dataHasVolume: true)
        let request = PredictRequestV4(valid: true, context: context, x: x)
        let descriptor = MLBackendDescriptor(mode: .inProcess(.metal), modelIdentifier: "model")
        let payload = MLBackendFactory.inferencePayload(descriptor: descriptor, request: request)

        XCTAssertEqual(payload.framework, .metal)
        XCTAssertTrue(payload.dataHasVolume)
        XCTAssertTrue(descriptor.usesVolumeFeatures)
    }
}

private struct InvocationRecordingPlugin: FXAIPluginV4 {
    let manifest: PluginManifestV4
    var trainCalls = 0

    mutating func reset() {}

    mutating func train(_ request: TrainRequestV4, hyperParameters: HyperParameters) throws {
        try request.validate()
        trainCalls += 1
    }

    func predict(_ request: PredictRequestV4, hyperParameters: HyperParameters) throws -> PredictionV4 {
        try request.validate()
        return PredictionV4(
            classProbabilities: [0.3, 0.2, 0.5],
            moveMeanPoints: 1.0,
            moveQ25Points: 0.5,
            moveQ50Points: 1.0,
            moveQ75Points: 1.5,
            mfeMeanPoints: 1.8,
            maeMeanPoints: 0.4,
            hitTimeFraction: 0.5,
            pathRisk: 0.2,
            fillRisk: request.context.dataHasVolume ? 0.1 : 0.0,
            confidence: 0.6,
            reliability: 0.7
        )
    }
}

private struct InvocationThrowingPlugin: FXAIPluginV4 {
    let manifest: PluginManifestV4

    mutating func reset() {}

    mutating func train(_ request: TrainRequestV4, hyperParameters: HyperParameters) throws {
        try request.validate()
        throw FXDataEngineError.externalBackend("train failed")
    }

    func predict(_ request: PredictRequestV4, hyperParameters: HyperParameters) throws -> PredictionV4 {
        try request.validate()
        throw FXDataEngineError.externalBackend("predict failed")
    }
}
