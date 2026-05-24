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

    func testPluginContractsRequireLatestAPIVersion() throws {
        let staleManifest = PluginManifestV4(
            apiVersion: FXDataEngineConstants.latestPluginAPIVersion - 1,
            aiID: 2,
            aiName: "Stale",
            family: .linear
        )
        XCTAssertThrowsError(try staleManifest.validate()) { error in
            XCTAssertEqual(String(describing: error), "validation failed: manifest.apiVersion")
        }

        let staleContext = PluginContextV4(apiVersion: FXDataEngineConstants.latestPluginAPIVersion - 1)
        XCTAssertThrowsError(try staleContext.validate()) { error in
            XCTAssertEqual(String(describing: error), "validation failed: ctx.apiVersion")
        }

        let staleTokenizer = PluginTokenizerContractV4(version: "fxai-tokenizer-v0")
        XCTAssertThrowsError(try staleTokenizer.validate()) { error in
            XCTAssertEqual(String(describing: error), "validation failed: ctx.tokenizer.version")
        }

        let stalePrediction = PredictionV4(apiVersion: FXDataEngineConstants.latestPluginAPIVersion - 1)
        XCTAssertThrowsError(try stalePrediction.validate()) { error in
            XCTAssertEqual(String(describing: error), "validation failed: pred.apiVersion")
        }
    }

    func testPluginContextDecodingRequiresExplicitAPIVersion() {
        let json = """
        {
          "regimeID": 0,
          "sessionBucket": 0,
          "horizonMinutes": 1,
          "featureSchema": 1,
          "normalizationMethod": 0,
          "sequenceBars": 1,
          "pointValue": 1.0,
          "domainHash": 0.25,
          "sampleTimeUTC": 1800020000,
          "dataHasVolume": true
        }
        """.data(using: .utf8)!

        XCTAssertThrowsError(try JSONDecoder().decode(PluginContextV4.self, from: json))
    }

    func testPredictRequestValidationChecksWindowContract() throws {
        let x = Array(repeating: 0.0, count: FXDataEngineConstants.aiWeights)
        let context = PluginContextV4(sequenceBars: 2, dataHasVolume: true)
        let valid = PredictRequestV4(valid: true, context: context, windowSize: 1, x: x, xWindow: [x])

        XCTAssertNoThrow(try valid.validate())

        let invalid = PredictRequestV4(valid: true, context: context, windowSize: 0, x: x)
        XCTAssertThrowsError(try invalid.validate())
    }

    func testPluginWindowValidationRequiresExactPayloadRowCount() throws {
        let x = Array(repeating: 0.0, count: FXDataEngineConstants.aiWeights)
        let context = PluginContextV4(sequenceBars: 3, dataHasVolume: true)

        let missingRow = PredictRequestV4(valid: true, context: context, windowSize: 2, x: x, xWindow: [x])
        XCTAssertThrowsError(try missingRow.validate()) { error in
            XCTAssertEqual(String(describing: error), "validation failed: req.windowSizePayload")
        }

        let extraRow = PredictRequestV4(valid: true, context: context, windowSize: 1, x: x, xWindow: [x, x])
        XCTAssertThrowsError(try extraRow.validate()) { error in
            XCTAssertEqual(String(describing: error), "validation failed: req.windowSizePayload")
        }
    }

    func testPayloadBuilderRejectsWindowOutsideSequenceContract() throws {
        let x = Array(repeating: 0.0, count: FXDataEngineConstants.aiWeights)
        let normalization = NormalizationCore()

        XCTAssertThrowsError(try normalization.buildPayloadFrame(NormalizationPayloadRequest(
            valid: true,
            sequenceBars: 2,
            windowSize: 2,
            x: x,
            xWindow: [x, x]
        ))) { error in
            XCTAssertEqual(String(describing: error), "validation failed: payload.windowSizeContext")
        }

        XCTAssertThrowsError(try normalization.buildPayloadFrame(NormalizationPayloadRequest(
            valid: true,
            sequenceBars: 3,
            windowSize: 2,
            x: x,
            xWindow: [x]
        ))) { error in
            XCTAssertEqual(String(describing: error), "validation failed: payload.windowSizePayload")
        }
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

    func testPluginContextRuntimeWindowHelpersMatchLegacyIndexing() {
        func row(_ feature0: Double, _ feature1: Double, _ cost: Double = 0.0) -> [Double] {
            var values = Array(repeating: 0.0, count: FXDataEngineConstants.aiWeights)
            values[1] = feature0
            values[2] = feature1
            values[7] = cost
            return values
        }
        let window = [
            row(10.0, 1.0, 0.35),
            row(8.0, 3.0),
            row(4.0, 5.0),
            row(2.0, 7.0)
        ]

        XCTAssertEqual(PluginContextRuntimeTools.effectiveWindowSize(window, declaredSize: 99), 4)
        XCTAssertEqual(PluginContextRuntimeTools.windowValue(window, barIndex: 0, inputIndex: 1), 10.0, accuracy: 0.0)
        XCTAssertEqual(
            PluginContextRuntimeTools.windowSliceMean(window, inputIndex: 1, startBar: 1, count: 2),
            6.0,
            accuracy: 0.0
        )
        XCTAssertEqual(PluginContextRuntimeTools.currentWindowFeatureMean(window, featureIndex: 0), 3.95, accuracy: 1e-12)
        XCTAssertEqual(PluginContextRuntimeTools.currentWindowFeatureRecentMean(window, featureIndex: 0, recentBars: 2), 9.0, accuracy: 0.0)
        XCTAssertEqual(PluginContextRuntimeTools.currentWindowFeatureStd(window, featureIndex: 0), sqrt(10.0), accuracy: 1e-12)
        XCTAssertEqual(PluginContextRuntimeTools.currentWindowFeatureRange(window, featureIndex: 0, recentBars: 2), 2.0, accuracy: 0.0)
        XCTAssertEqual(PluginContextRuntimeTools.currentWindowFeatureSlope(window, featureIndex: 0), 8.0 / 3.0, accuracy: 1e-12)
        XCTAssertEqual(PluginContextRuntimeTools.currentWindowFeatureRecentDelta(window, featureIndex: 0, recentBars: 3), 6.0, accuracy: 0.0)
        XCTAssertEqual(PluginContextRuntimeTools.currentWindowFeatureEMAMean(window, featureIndex: 0, decay: 0.5), 8.133333333333333, accuracy: 1e-12)
    }

    func testPluginContextRuntimeClassAndPredictionHelpersMatchLegacyRules() throws {
        var x = Array(repeating: 0.0, count: FXDataEngineConstants.aiWeights)
        x[7] = 0.35

        XCTAssertEqual(PluginContextRuntimeTools.contextHorizonBucket(horizonMinutes: 1), 0)
        XCTAssertEqual(PluginContextRuntimeTools.contextHorizonBucket(horizonMinutes: 34), 6)
        XCTAssertEqual(PluginContextRuntimeTools.contextHorizonBucket(horizonMinutes: 35), 7)
        XCTAssertEqual(
            PluginContextRuntimeTools.normalizeClassLabel(rawLabel: 99, x: x, movePoints: 0.20),
            .skip
        )
        XCTAssertEqual(
            PluginContextRuntimeTools.normalizeClassLabel(rawLabel: 3, x: x, movePoints: 2.0, priceCostPoints: 0.0),
            .buy
        )
        XCTAssertEqual(
            PluginContextRuntimeTools.normalizeClassLabel(rawLabel: -1, x: x, movePoints: -2.0, priceCostPoints: 0.0),
            .sell
        )
        XCTAssertEqual(
            PluginContextRuntimeTools.normalizeClassDistribution([Double.nan, -1.0, 0.0]),
            [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]
        )

        let context = PluginContextV4(
            horizonMinutes: 5,
            priceCostPoints: 0.60,
            minMovePoints: 1.40,
            dataHasVolume: true
        )
        try context.validate()
        let legacyContextJSON = """
        {
          "apiVersion": 4,
          "regimeID": 2,
          "sessionBucket": 3,
          "horizonMinutes": 5,
          "featureSchema": 1,
          "normalizationMethod": 0,
          "sequenceBars": 1,
          "pointValue": 1.0,
          "domainHash": 0.25,
          "sampleTimeUTC": 1800020000,
          "dataHasVolume": true
        }
        """.data(using: .utf8)!
        let decodedLegacyContext = try JSONDecoder().decode(PluginContextV4.self, from: legacyContextJSON)
        XCTAssertEqual(decodedLegacyContext.priceCostPoints, 0.0, accuracy: 0.0)
        XCTAssertEqual(decodedLegacyContext.minMovePoints, 0.0, accuracy: 0.0)
        XCTAssertTrue(decodedLegacyContext.dataHasVolume)
        XCTAssertTrue(String(data: try JSONEncoder().encode(context), encoding: .utf8)!.contains("priceCostPoints"))

        let nativeShape = PluginModelOutputV4(
            classProbabilities: [0.20, 0.50, 0.30],
            moveMeanPoints: 4.0,
            moveQ25Points: 1.0,
            moveQ50Points: 2.0,
            moveQ75Points: 6.0,
            mfeMeanPoints: 8.0,
            maeMeanPoints: 2.0,
            hitTimeFraction: 0.40,
            pathRisk: 0.30,
            fillRisk: 0.20,
            confidence: 0.70,
            reliability: 0.60,
            hasQuantiles: true,
            hasConfidence: true,
            hasPathQuality: true
        )
        let prediction = PluginContextRuntimeTools.fillPrediction(
            modelOutput: nativeShape,
            calibratedMoveMeanPoints: 6.0,
            context: context
        )
        try prediction.validate()
        XCTAssertEqual(prediction.moveQ25Points, 1.5, accuracy: 1e-12)
        XCTAssertEqual(prediction.moveQ50Points, 3.0, accuracy: 1e-12)
        XCTAssertEqual(prediction.moveQ75Points, 9.0, accuracy: 1e-12)
        XCTAssertEqual(prediction.mfeMeanPoints, 12.0, accuracy: 1e-12)
        XCTAssertEqual(prediction.maeMeanPoints, 3.0, accuracy: 1e-12)
        XCTAssertEqual(prediction.pathRisk, 0.30, accuracy: 1e-12)
        XCTAssertEqual(prediction.fillRisk, 0.20, accuracy: 1e-12)

        let fallback = PluginContextRuntimeTools.fillPrediction(
            modelOutput: PluginModelOutputV4(classProbabilities: [0.80, 0.10, 0.10]),
            calibratedMoveMeanPoints: 2.0,
            context: context
        )
        try fallback.validate()
        XCTAssertEqual(fallback.moveQ25Points, 1.775, accuracy: 1e-12)
        XCTAssertEqual(fallback.moveQ50Points, 2.0, accuracy: 1e-12)
        XCTAssertEqual(fallback.moveQ75Points, 2.225, accuracy: 1e-12)
        XCTAssertEqual(fallback.mfeMeanPoints, 2.225, accuracy: 1e-12)
        XCTAssertEqual(fallback.maeMeanPoints, 0.70, accuracy: 1e-12)
        XCTAssertEqual(fallback.hitTimeFraction, 0.46, accuracy: 1e-12)
        XCTAssertEqual(fallback.pathRisk, 0.201, accuracy: 1e-12)
        XCTAssertEqual(fallback.fillRisk, 0.27941176470588236, accuracy: 1e-12)
        XCTAssertEqual(fallback.confidence, 0.80, accuracy: 1e-12)
        XCTAssertEqual(fallback.reliability, 0.95, accuracy: 1e-12)
    }

    func testPluginTernaryCalibratorMatchesLegacyUpdateAndCalibrationRules() {
        var calibrator = PluginTernaryCalibrator()
        let initial = calibrator.calibrated([0.20, 0.30, 0.50])
        XCTAssertEqual(initial[0], 0.20, accuracy: 1e-12)
        XCTAssertEqual(initial[1], 0.30, accuracy: 1e-12)
        XCTAssertEqual(initial[2], 0.50, accuracy: 1e-12)

        calibrator.update(
            rawProbabilities: [0.20, 0.30, 0.50],
            labelClass: .buy,
            sampleWeight: 2.0,
            learningRate: 0.10
        )

        XCTAssertEqual(calibrator.steps, 1)
        XCTAssertEqual(calibrator.biases[0], -0.004, accuracy: 1e-12)
        XCTAssertEqual(calibrator.biases[1], 0.014, accuracy: 1e-12)
        XCTAssertEqual(calibrator.biases[2], -0.010, accuracy: 1e-12)
        XCTAssertEqual(calibrator.weights[0][0], 1.0 + 0.02 * (-0.20 * log(0.20)), accuracy: 1e-12)
        XCTAssertEqual(calibrator.weights[1][1], 1.0 + 0.02 * (0.70 * log(0.30)), accuracy: 1e-12)
        XCTAssertEqual(calibrator.weights[2][2], 1.0 + 0.02 * (-0.50 * log(0.50)), accuracy: 1e-12)
        XCTAssertEqual(calibrator.isotonicCount[0][2], 2.0, accuracy: 0.0)
        XCTAssertEqual(calibrator.isotonicPositive[1][3], 2.0, accuracy: 0.0)
        XCTAssertEqual(calibrator.isotonicCount[2][6], 2.0, accuracy: 0.0)

        for _ in 0..<29 {
            calibrator.update(
                rawProbabilities: [0.20, 0.30, 0.50],
                labelClass: .buy,
                sampleWeight: 2.0,
                learningRate: 0.10
            )
        }

        let learned = calibrator.calibrated([0.20, 0.30, 0.50])
        XCTAssertEqual(calibrator.steps, 30)
        XCTAssertEqual(learned.reduce(0.0, +), 1.0, accuracy: 1e-12)
        XCTAssertGreaterThan(learned[LabelClass.buy.rawValue], 0.30)
        XCTAssertTrue(learned.allSatisfy { $0 >= 0.0005 && $0 <= 0.9990 })
    }

    func testPluginBinaryCalibratorMatchesLegacyPlattAndIsotonicRules() {
        var calibrator = PluginBinaryCalibrator()
        XCTAssertEqual(calibrator.calibrated(0.25), 0.25, accuracy: 1e-12)

        calibrator.update(rawProbability: 0.25, target: true, sampleWeight: 2.0)

        let z = log(0.25 / 0.75)
        XCTAssertEqual(calibrator.steps, 1)
        XCTAssertFalse(calibrator.ready)
        XCTAssertEqual(calibrator.scale, 1.0 + 0.03 * (0.75 * z), accuracy: 1e-12)
        XCTAssertEqual(calibrator.bias, 0.0225, accuracy: 1e-12)
        XCTAssertEqual(calibrator.isotonicCount[3], 2.0, accuracy: 0.0)
        XCTAssertEqual(calibrator.isotonicPositive[3], 2.0, accuracy: 0.0)

        for _ in 0..<19 {
            calibrator.update(rawProbability: 0.25, target: true, sampleWeight: 2.0)
        }

        let calibrated = calibrator.calibrated(0.25)
        XCTAssertTrue(calibrator.ready)
        XCTAssertEqual(calibrator.steps, 20)
        XCTAssertGreaterThan(calibrated, 0.25)
        XCTAssertLessThanOrEqual(calibrated, 0.999)
    }

    func testPluginQualitySupportMatchesLegacyMoveWeightAndReplayRules() {
        var x = Array(repeating: 0.0, count: FXDataEngineConstants.aiWeights)
        x[0] = 1.0
        x[1] = 0.5
        x[7] = 1.0

        var head = PluginMoveHead()
        let hyperParameters = HyperParameters(learningRate: 0.10, l2: 0.02)
        head.update(x: x, movePoints: 5.0, hyperParameters: hyperParameters, sampleWeight: 2.0)

        XCTAssertEqual(head.steps, 1)
        XCTAssertFalse(head.ready)
        XCTAssertEqual(head.weights[0], 0.08, accuracy: 1e-12)
        XCTAssertEqual(head.weights[1], 0.04, accuracy: 1e-12)
        XCTAssertEqual(head.predictRaw(x), 0.18, accuracy: 1e-12)

        let scaled = PluginSupportTools.scaleHyperParametersForMove(hyperParameters, movePoints: 50.0)
        XCTAssertEqual(scaled.learningRate, 0.30, accuracy: 1e-12)
        XCTAssertEqual(scaled.ftrlAlpha, 0.15, accuracy: 1e-12)
        XCTAssertEqual(PluginSupportTools.moveEdgeWeight(movePoints: 4.0, priceCostPoints: 1.0), 3.50, accuracy: 1e-12)

        let targets = PluginQualityTargets(
            mfePoints: 6.0,
            maePoints: 1.5,
            hitTimeFraction: 0.25,
            pathRisk: 0.20,
            fillRisk: 0.30
        )
        let weighted = PluginSupportTools.moveSampleWeight(
            x: x,
            movePoints: 4.0,
            priceCostPoints: 1.0,
            minMovePoints: 1.0,
            qualityTargets: targets
        )
        let quality = 1.0 + 0.18 * 1.5 + 0.14 * 0.75 - 0.16 * 0.25 - 0.18 * 0.24
        XCTAssertEqual(weighted, 3.50 * quality, accuracy: 1e-12)

        XCTAssertEqual(
            PluginSupportTools.computeReplayPriority(
                rawLabelClass: -1,
                probabilities: [0.70, 0.20, 0.10],
                movePoints: -4.0,
                priceCostPoints: 1.0,
                minMovePoints: 2.0
            ),
            1.325,
            accuracy: 1e-12
        )
        XCTAssertEqual(
            PluginSupportTools.computeReplayPriority(
                rawLabelClass: LabelClass.skip.rawValue,
                probabilities: [0.70, 0.20, 0.10],
                movePoints: 0.2,
                priceCostPoints: 1.0,
                minMovePoints: 2.0
            ),
            1.55,
            accuracy: 1e-12
        )
    }

    func testPluginQualityAndContextCalibrationBanksMatchLegacyUpdates() {
        let context = PluginContextV4(
            regimeID: 2,
            sessionBucket: 3,
            horizonMinutes: 5,
            minMovePoints: 1.0
        )
        var qualityBank = PluginQualityBank()
        qualityBank.update(
            targets: PluginQualityTargets(
                mfePoints: 10.0,
                maePoints: 3.0,
                hitTimeFraction: 0.40,
                pathRisk: 0.20,
                fillRisk: 0.10
            ),
            context: context,
            sampleWeight: 2.0
        )
        XCTAssertTrue(qualityBank.ready)
        XCTAssertTrue(qualityBank.cells.isEmpty)

        qualityBank.update(
            targets: PluginQualityTargets(
                mfePoints: 20.0,
                maePoints: 7.0,
                hitTimeFraction: 0.80,
                pathRisk: 0.60,
                fillRisk: 0.50
            ),
            context: context,
            sampleWeight: 2.0
        )
        let priors = qualityBank.priors(context: context)
        XCTAssertEqual(qualityBank.mfeEMA, 11.2, accuracy: 1e-12)
        XCTAssertEqual(priors.mfePoints, 12.08, accuracy: 1e-12)
        XCTAssertEqual(priors.maePoints, 3.8320000000000003, accuracy: 1e-12)
        XCTAssertEqual(priors.hitTimeFraction, 0.4832000000000001, accuracy: 1e-12)
        XCTAssertEqual(priors.pathRisk, 0.2832, accuracy: 1e-12)
        XCTAssertEqual(priors.fillRisk, 0.18320000000000003, accuracy: 1e-12)
        XCTAssertEqual(priors.trust, 0.415, accuracy: 1e-12)

        var calibrationBank = PluginContextCalibrationBank()
        XCTAssertEqual(
            calibrationBank.classCalibrated(probabilities: [0.20, 0.30, 0.50], context: context),
            [0.20, 0.30, 0.50]
        )

        calibrationBank.update(
            labelClass: .buy,
            expectedMovePoints: 4.0,
            movePoints: 6.0,
            sampleWeight: 2.0,
            context: context
        )
        let calibrated = calibrationBank.classCalibrated(probabilities: [0.20, 0.30, 0.50], context: context)
        XCTAssertEqual(calibrated[LabelClass.sell.rawValue], 0.1996153846153846, accuracy: 1e-12)
        XCTAssertEqual(calibrated[LabelClass.buy.rawValue], 0.31384615384615383, accuracy: 1e-12)
        XCTAssertEqual(calibrated[LabelClass.skip.rawValue], 0.48653846153846153, accuracy: 1e-12)
        XCTAssertEqual(calibrationBank.expectedMoveCalibrated(4.0, context: context), 4.15, accuracy: 1e-6)
    }

    func testPluginDeterministicRNGMatchesLegacyLCG() {
        var rng = PluginDeterministicRNG(aiID: 4)
        let seeded = UInt32(5) &* 747_796_405 &+ 2_891_336_453
        XCTAssertEqual(rng.state, seeded)

        let nextState = 1_664_525 &* seeded &+ 1_013_904_223
        let expected = (Double(nextState) + 0.5) / 4_294_967_296.0
        XCTAssertEqual(rng.next01(), expected, accuracy: 1e-15)
        XCTAssertEqual(rng.state, nextState)
        XCTAssertEqual(rng.nextIndex(0), -1)
    }

    func testMLPayloadCarriesVolumeAvailability() {
        let x = Array(repeating: 0.0, count: FXDataEngineConstants.aiWeights)
        let context = PluginContextV4(priceCostPoints: 1.25, minMovePoints: 2.5, dataHasVolume: true)
        let request = PredictRequestV4(valid: true, context: context, x: x)
        let descriptor = MLBackendDescriptor(mode: .inProcess(.metal), modelIdentifier: "model")
        let payload = MLBackendFactory.inferencePayload(descriptor: descriptor, request: request)

        XCTAssertEqual(payload.framework, .metal)
        XCTAssertTrue(payload.dataHasVolume)
        XCTAssertEqual(payload.priceCostPoints, 1.25, accuracy: 0.0)
        XCTAssertEqual(payload.minMovePoints, 2.5, accuracy: 0.0)
        XCTAssertEqual(payload.horizonMinutes, 1)
        XCTAssertEqual(payload.sequenceBars, 1)
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
