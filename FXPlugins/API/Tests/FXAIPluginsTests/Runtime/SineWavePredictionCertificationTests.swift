import BacktestCore
import FXDataEngine
import Foundation
import XCTest
@testable import FXAIPlugins

final class SineWavePredictionCertificationTests: XCTestCase {
    private static let startUTC: Int64 = 1_704_067_200
    private static let trainingDays = 1
    private static let holdoutDays = 1
    private static let trainStride = 5
    private static let evaluationStride = 5
    private static let minimumDirectionalDeltaPoints: Int64 = 500
    private static let minimumEvaluationSamples = 240
    private static let requiredDirectionalAccuracy = 0.99
    private static let requiredMeanSignedEdge = 0.01
    private static let requiredPredictionConfidence = 0.85
    private static let acceleratorEvaluationSamples = 2
    private static let acceleratorTrainingMinuteSamples = 4

    func testEveryPluginPredictionStaysInSyncWithSineWaveHoldout() throws {
        let marketSeries = try Self.makeSineTestSeries(days: Self.trainingDays + Self.holdoutDays + 1)
        let schemaPolicy = FeatureSchemaPolicy()
        let modelInputsByIndex = try Self.modelInputsByIndex(marketSeries: marketSeries, schemaPolicy: schemaPolicy)
        let plugins = FXAIPluginRegistry.availablePlugins().compactMap { $0 as? any FXAIPlannedPlugin }
        XCTAssertEqual(plugins.count, FXDataEngineConstants.aiCount)

        var results: [PluginSineWavePredictionResult] = []
        for plugin in plugins {
            results.append(
                try Self.certifyPlugin(
                    plugin,
                    marketSeries: marketSeries,
                    modelInputsByIndex: modelInputsByIndex,
                    schemaPolicy: schemaPolicy
                )
            )
        }

        let report = Self.markdownReport(
            title: "SineTest Plugin Prediction Certification Results",
            results: results,
            minimumEvaluationSamples: Self.minimumEvaluationSamples
        )
        try Self.writeTemporaryReport(report, fileName: "fxai_sinetest_plugin_prediction_results.md")
        print(report)

        let failures = results.filter { !$0.passed }
        XCTAssertTrue(
            failures.isEmpty,
            "SineTest directional sync failures:\n" +
                failures.map(\.failureSummary).joined(separator: "\n")
        )
    }

    func testEveryDeclaredAcceleratorPredictionStaysInSyncWithSineWaveHoldout() throws {
        let marketSeries = try Self.makeSineTestSeries(days: Self.trainingDays + Self.holdoutDays + 1)
        let schemaPolicy = FeatureSchemaPolicy()
        let modelInputsByIndex = try Self.modelInputsByIndex(marketSeries: marketSeries, schemaPolicy: schemaPolicy)
        let plugins = FXAIPluginRegistry.availablePlugins().compactMap { $0 as? any FXAIPlannedPlugin }
        XCTAssertEqual(plugins.count, FXDataEngineConstants.aiCount)

        let declaredAccelerators = Set(
            plugins.flatMap { $0.accelerationPlan.declaredBackends.filter { !$0.isCPUOnly } }
        )
        XCTAssertFalse(declaredAccelerators.isEmpty, "SineTest accelerator gate has no declared accelerator backends to run.")

        let environment = Self.acceleratorEnvironment()
        if declaredAccelerators.contains(.metal) {
            XCTAssertTrue(environment.supports(.metal), "Declared Metal accelerators require an Apple Silicon Metal runtime.")
        }
        if declaredAccelerators.contains(.pyTorchMPS) {
            XCTAssertTrue(environment.supports(.pyTorchMPS), "Declared PyTorch MPS accelerators require torch with MPS enabled.")
        }
        if declaredAccelerators.contains(.tensorFlowMetal) {
            XCTAssertTrue(environment.supports(.tensorFlowMetal), "Declared TensorFlow Metal accelerators require tensorflow with a GPU device.")
        }
        if declaredAccelerators.contains(.foundationNLP) {
            XCTAssertTrue(environment.supports(.foundationNLP), "Declared NLP accelerators require an Apple Silicon Foundation NLP runtime.")
        }
        if declaredAccelerators.contains(.coreMLNeuralEngine) {
            XCTAssertTrue(environment.supports(.coreMLNeuralEngine), "Declared CoreML Neural Engine accelerators require a live Neural Engine runtime.")
        }

        var results: [PluginSineWavePredictionResult] = []
        for plugin in plugins {
            let acceleratorBackends = plugin.accelerationPlan.declaredBackends.filter { !$0.isCPUOnly }
            guard !acceleratorBackends.isEmpty else {
                continue
            }
            let stateDirectory = try Self.makeTemporaryDirectory(prefix: "fxai-sinetest-\(plugin.manifest.aiName)")
            defer { try? FileManager.default.removeItem(at: stateDirectory) }

            let horizon = Self.certificationHorizon(for: plugin)
            let trainingReferenceIndices = Self.highSignalHoldoutSampleIndices(
                marketSeries: marketSeries,
                horizon: horizon,
                limit: Self.acceleratorTrainingMinuteSamples
            )
            let evaluationIndices = Self.highSignalHoldoutSampleIndices(
                marketSeries: marketSeries,
                horizon: horizon,
                limit: Self.acceleratorEvaluationSamples
            )
            let trainingMinuteOffsets = Set(
                trainingReferenceIndices.map { Self.minuteOfHour(timestampUTC: marketSeries.utcTimestamps[$0]) }
            )
            var trained = try Self.trainedRuntimeOnSineTest(
                plugin,
                horizon: horizon,
                trainingMinuteOffsets: trainingMinuteOffsets,
                marketSeries: marketSeries,
                modelInputsByIndex: modelInputsByIndex,
                schemaPolicy: schemaPolicy,
                environment: environment,
                stateDirectory: stateDirectory
            )

            for backend in acceleratorBackends {
                trained.runtime.configuration = Self.runtimeConfiguration(
                    mode: Self.runtimeMode(for: backend),
                    environment: environment,
                    stateDirectory: stateDirectory
                )
                results.append(
                    try Self.certifyAcceleratorBackend(
                        runtime: trained.runtime,
                        backend: backend,
                        horizon: trained.horizon,
                        evaluationIndices: evaluationIndices,
                        trainedSamples: trained.trainedSamples,
                        marketSeries: marketSeries,
                        modelInputsByIndex: modelInputsByIndex,
                        schemaPolicy: schemaPolicy
                    )
                )
            }
        }

        let report = Self.markdownReport(
            title: "SineTest Accelerator Prediction Certification Results",
            results: results,
            minimumEvaluationSamples: Self.acceleratorEvaluationSamples
        )
        try Self.writeTemporaryReport(report, fileName: "fxai_sinetest_accelerator_prediction_results.md")
        print(report)

        let failures = results.filter { !$0.passed }
        XCTAssertTrue(
            failures.isEmpty,
            "SineTest accelerator directional sync failures:\n" +
                failures.map(\.failureSummary).joined(separator: "\n")
        )
    }

    private static func certifyPlugin(
        _ originalPlugin: any FXAIPlannedPlugin,
        marketSeries: M1OHLCVSeries,
        modelInputsByIndex: [[Double]],
        schemaPolicy: FeatureSchemaPolicy
    ) throws -> PluginSineWavePredictionResult {
        var plugin = try preparedPlugin(originalPlugin, marketSeries: marketSeries)
        let horizon = certificationHorizon(for: plugin)
        let sequenceBars = plugin.manifest.resolvedSequenceBars(horizonMinutes: horizon)
        let firstUsableIndex = max(sequenceBars + 10, 180)
        let trainingEndIndex = Self.trainingDays * 24 * 60
        let holdoutStartIndex = trainingEndIndex
        let holdoutEndIndex = min(
            (Self.trainingDays + Self.holdoutDays) * 24 * 60,
            marketSeries.count - horizon - 1
        )
        let hyperParameters = certificationHyperParameters()

        var trainedSamples = 0
        if firstUsableIndex < trainingEndIndex {
            for sampleIndex in stride(from: firstUsableIndex, to: trainingEndIndex, by: Self.trainStride) {
                let request = try predictRequest(
                    for: plugin,
                    marketSeries: marketSeries,
                    sampleIndex: sampleIndex,
                    horizon: horizon,
                    modelInputsByIndex: modelInputsByIndex,
                    schemaPolicy: schemaPolicy
                )
                let train = trainRequest(from: request, marketSeries: marketSeries, sampleIndex: sampleIndex)
                try plugin.train(train, hyperParameters: hyperParameters)
                trainedSamples += 1
            }
        }

        var evaluatedSamples = 0
        var correctSamples = 0
        var validPredictions = 0
        var signedEdgeSum = 0.0
        var absoluteEdgeSum = 0.0
        var confidenceSum = 0.0
        var minimumConfidence = Double.greatestFiniteMagnitude
        var validationFailure: String?

        if holdoutStartIndex < holdoutEndIndex {
            for sampleIndex in stride(from: holdoutStartIndex, to: holdoutEndIndex, by: Self.evaluationStride) {
                guard let expected = expectedLabel(
                    marketSeries: marketSeries,
                    sampleIndex: sampleIndex,
                    horizon: horizon
                ) else {
                    continue
                }

                let request = try predictRequest(
                    for: plugin,
                    marketSeries: marketSeries,
                    sampleIndex: sampleIndex,
                    horizon: horizon,
                    modelInputsByIndex: modelInputsByIndex,
                    schemaPolicy: schemaPolicy
                )
                let prediction = try plugin.predict(request, hyperParameters: hyperParameters)
                do {
                    try prediction.validate()
                    validPredictions += 1
                } catch {
                    validationFailure = "\(error)"
                }

                let edge = directionalEdge(prediction)
                let signedEdge = expected == .buy ? edge : -edge
                let confidence = normalizedConfidence(prediction)
                signedEdgeSum += signedEdge
                absoluteEdgeSum += abs(edge)
                confidenceSum += confidence
                minimumConfidence = min(minimumConfidence, confidence)
                if signedEdge > 0.0 {
                    correctSamples += 1
                }
                evaluatedSamples += 1
            }
        }

        let accuracy = evaluatedSamples > 0 ? Double(correctSamples) / Double(evaluatedSamples) : 0.0
        let meanSignedEdge = evaluatedSamples > 0 ? signedEdgeSum / Double(evaluatedSamples) : 0.0
        let meanAbsoluteEdge = evaluatedSamples > 0 ? absoluteEdgeSum / Double(evaluatedSamples) : 0.0
        let meanConfidence = evaluatedSamples > 0 ? confidenceSum / Double(evaluatedSamples) : 0.0
        let minConfidence = evaluatedSamples > 0 ? minimumConfidence : 0.0
        let failureReason = failureReason(
            evaluatedSamples: evaluatedSamples,
            validPredictions: validPredictions,
            accuracy: accuracy,
            meanSignedEdge: meanSignedEdge,
            meanConfidence: meanConfidence,
            minimumConfidence: minConfidence,
            validationFailure: validationFailure
        )

        return PluginSineWavePredictionResult(
            pluginName: plugin.manifest.aiName,
            backendName: "registry",
            trainedSamples: trainedSamples,
            evaluatedSamples: evaluatedSamples,
            validPredictions: validPredictions,
            directionalAccuracy: accuracy,
            meanSignedEdge: meanSignedEdge,
            meanAbsoluteEdge: meanAbsoluteEdge,
            meanConfidence: meanConfidence,
            minimumConfidence: minConfidence,
            failureReason: failureReason
        )
    }

    private static func trainedRuntimeOnSineTest(
        _ originalPlugin: any FXAIPlannedPlugin,
        horizon: Int,
        trainingMinuteOffsets: Set<Int>,
        marketSeries: M1OHLCVSeries,
        modelInputsByIndex: [[Double]],
        schemaPolicy: FeatureSchemaPolicy,
        environment: FXPluginRuntimeEnvironment,
        stateDirectory: URL
    ) throws -> (runtime: FXAIAcceleratedPluginRuntime, horizon: Int, trainedSamples: Int) {
        let plugin = try preparedPlugin(originalPlugin, marketSeries: marketSeries)
        let sequenceBars = plugin.manifest.resolvedSequenceBars(horizonMinutes: horizon)
        let firstUsableIndex = max(sequenceBars + 10, 180)
        let trainingEndIndex = Self.trainingDays * 24 * 60
        let hyperParameters = certificationHyperParameters()
        var runtime = FXAIAcceleratedPluginRuntime(
            plugin: plugin,
            configuration: Self.runtimeConfiguration(
                mode: .cpuOnly,
                environment: environment,
                stateDirectory: stateDirectory
            )
        )

        var trainedSamples = 0
        if firstUsableIndex < trainingEndIndex {
            for sampleIndex in firstUsableIndex..<trainingEndIndex {
                let minute = Self.minuteOfHour(timestampUTC: marketSeries.utcTimestamps[sampleIndex])
                guard trainingMinuteOffsets.contains(minute) else {
                    continue
                }
                let request = try predictRequest(
                    for: runtime,
                    marketSeries: marketSeries,
                    sampleIndex: sampleIndex,
                    horizon: horizon,
                    modelInputsByIndex: modelInputsByIndex,
                    schemaPolicy: schemaPolicy
                )
                try runtime.train(
                    trainRequest(from: request, marketSeries: marketSeries, sampleIndex: sampleIndex),
                    hyperParameters: hyperParameters
                )
                trainedSamples += 1
            }
        }

        return (runtime, horizon, trainedSamples)
    }

    private static func certifyAcceleratorBackend(
        runtime: FXAIAcceleratedPluginRuntime,
        backend: FXPluginAccelerationBackend,
        horizon: Int,
        evaluationIndices: [Int],
        trainedSamples: Int,
        marketSeries: M1OHLCVSeries,
        modelInputsByIndex: [[Double]],
        schemaPolicy: FeatureSchemaPolicy
    ) throws -> PluginSineWavePredictionResult {
        let hyperParameters = certificationHyperParameters()
        var evaluatedSamples = 0
        var correctSamples = 0
        var validPredictions = 0
        var signedEdgeSum = 0.0
        var absoluteEdgeSum = 0.0
        var confidenceSum = 0.0
        var minimumConfidence = Double.greatestFiniteMagnitude
        var validationFailure: String?

        for sampleIndex in evaluationIndices {
            guard let expected = expectedLabel(marketSeries: marketSeries, sampleIndex: sampleIndex, horizon: horizon) else {
                continue
            }
            let request = try predictRequest(
                for: runtime,
                marketSeries: marketSeries,
                sampleIndex: sampleIndex,
                horizon: horizon,
                modelInputsByIndex: modelInputsByIndex,
                schemaPolicy: schemaPolicy
            )
            let prediction = try runtime.predict(request, hyperParameters: hyperParameters)
            do {
                try prediction.validate()
                validPredictions += 1
            } catch {
                validationFailure = "\(error)"
            }

            let edge = directionalEdge(prediction)
            let signedEdge = expected == .buy ? edge : -edge
            let confidence = normalizedConfidence(prediction)
            signedEdgeSum += signedEdge
            absoluteEdgeSum += abs(edge)
            confidenceSum += confidence
            minimumConfidence = min(minimumConfidence, confidence)
            if signedEdge > 0.0 {
                correctSamples += 1
            }
            evaluatedSamples += 1
        }

        let accuracy = evaluatedSamples > 0 ? Double(correctSamples) / Double(evaluatedSamples) : 0.0
        let meanSignedEdge = evaluatedSamples > 0 ? signedEdgeSum / Double(evaluatedSamples) : 0.0
        let meanAbsoluteEdge = evaluatedSamples > 0 ? absoluteEdgeSum / Double(evaluatedSamples) : 0.0
        let meanConfidence = evaluatedSamples > 0 ? confidenceSum / Double(evaluatedSamples) : 0.0
        let minConfidence = evaluatedSamples > 0 ? minimumConfidence : 0.0
        let failureReason = failureReason(
            evaluatedSamples: evaluatedSamples,
            validPredictions: validPredictions,
            accuracy: accuracy,
            meanSignedEdge: meanSignedEdge,
            meanConfidence: meanConfidence,
            minimumConfidence: minConfidence,
            validationFailure: validationFailure,
            minimumEvaluationSamples: Self.acceleratorEvaluationSamples
        )

        return PluginSineWavePredictionResult(
            pluginName: runtime.manifest.aiName,
            backendName: backend.rawValue,
            trainedSamples: trainedSamples,
            evaluatedSamples: evaluatedSamples,
            validPredictions: validPredictions,
            directionalAccuracy: accuracy,
            meanSignedEdge: meanSignedEdge,
            meanAbsoluteEdge: meanAbsoluteEdge,
            meanConfidence: meanConfidence,
            minimumConfidence: minConfidence,
            failureReason: failureReason
        )
    }

    private static func preparedPlugin(
        _ plugin: any FXAIPlannedPlugin,
        marketSeries: M1OHLCVSeries
    ) throws -> any FXAIPlannedPlugin {
        var prepared = plugin
        prepared.reset()
        if var synthetic = prepared as? any FXAIPluginSyntheticSeriesSupport {
            try synthetic.setSyntheticSeries(marketSeries)
            guard let plannedSynthetic = synthetic as? any FXAIPlannedPlugin else {
                throw FXDataEngineError.invalidRequest(
                    "\(prepared.manifest.aiName) synthetic support does not preserve planned plugin conformance"
                )
            }
            prepared = plannedSynthetic
        }
        return prepared
    }

    private static func makeSineTestSeries(days: Int) throws -> M1OHLCVSeries {
        let endUTC = Self.startUTC + Int64(max(1, days) * 24 * 60 * 60)
        let series = try SineWaveAgent.generateM1Ohlc(
            brokerSourceIdRawValue: "plugin-sinetest-cert",
            utcStartInclusive: Self.startUTC,
            utcEndExclusive: endUTC
        )
        return try M1OHLCVSeries(
            metadata: FXMarketMetadata(
                brokerSourceId: series.metadata.brokerSourceId.rawValue,
                sourceOrigin: series.metadata.sourceOrigin.rawValue,
                logicalSymbol: series.metadata.logicalSymbol.rawValue,
                providerSymbol: SineTestSecurity.displayName,
                digits: series.metadata.digits.rawValue,
                firstUTC: series.metadata.firstUtc?.rawValue,
                lastUTC: series.metadata.lastUtc?.rawValue
            ),
            utcTimestamps: ContiguousArray(series.utcTimestamps),
            open: ContiguousArray(series.open),
            high: ContiguousArray(series.high),
            low: ContiguousArray(series.low),
            close: ContiguousArray(series.close),
            volume: ContiguousArray(series.volume)
        )
    }

    private static func modelInputsByIndex(
        marketSeries: M1OHLCVSeries,
        schemaPolicy: FeatureSchemaPolicy
    ) throws -> [[Double]] {
        let universe = try MarketUniverse(primarySymbol: "SINETEST", series: [marketSeries])
        let featureCore = FeatureCore()
        return (0..<marketSeries.count).map { index in
            schemaPolicy.modelInput(from: featureCore.buildFeatureVector(universe: universe, sampleIndex: index))
        }
    }

    private static func predictRequest(
        for plugin: any FXAIPlannedPlugin,
        marketSeries: M1OHLCVSeries,
        sampleIndex: Int,
        horizon: Int,
        modelInputsByIndex: [[Double]],
        schemaPolicy: FeatureSchemaPolicy
    ) throws -> PredictRequestV4 {
        let sequenceBars = plugin.manifest.resolvedSequenceBars(horizonMinutes: horizon)
        let windowSize = max(sequenceBars - 1, 0)
        guard sampleIndex >= windowSize, sampleIndex < modelInputsByIndex.count else {
            throw FXDataEngineError.invalidRequest("SineTest sample index cannot satisfy \(plugin.manifest.aiName)")
        }
        let x = applySchema(
            modelInputsByIndex[sampleIndex],
            plugin: plugin,
            schemaPolicy: schemaPolicy
        )
        let xWindow = windowSize == 0
            ? []
            : (sampleIndex - windowSize..<sampleIndex).map { index in
                applySchema(modelInputsByIndex[index], plugin: plugin, schemaPolicy: schemaPolicy)
            }
        let sampleTimeUTC = marketSeries.utcTimestamps[sampleIndex]
        let request = PredictRequestV4(
            valid: true,
            context: PluginContextV4(
                sessionBucket: PluginContractTools.deriveSessionBucket(timestampUTC: sampleTimeUTC),
                horizonMinutes: horizon,
                featureSchema: plugin.manifest.featureSchema,
                sequenceBars: sequenceBars,
                minMovePoints: 1.0,
                pointValue: 1.0,
                domainHash: PluginContractTools.symbolHash01(marketSeries.metadata.logicalSymbol),
                sampleTimeUTC: sampleTimeUTC,
                dataHasVolume: marketSeries.hasVolume
            ),
            windowSize: xWindow.count,
            x: x,
            xWindow: xWindow
        )
        try request.validate()
        try PluginContractTools.validateCompatibility(manifest: plugin.manifest, context: request.context)
        return request
    }

    private static func trainRequest(
        from request: PredictRequestV4,
        marketSeries: M1OHLCVSeries,
        sampleIndex: Int
    ) -> TrainRequestV4 {
        let horizonBars = min(request.context.horizonMinutes, marketSeries.count - sampleIndex - 1)
        let currentClose = marketSeries.close[sampleIndex]
        let futureClose = marketSeries.close[sampleIndex + max(1, horizonBars)]
        let delta = futureClose - currentClose
        let label: LabelClass
        if abs(delta) <= Self.minimumDirectionalDeltaPoints {
            label = .skip
        } else {
            label = delta > 0 ? .buy : .sell
        }
        let movePoints = abs(Double(delta))
        return TrainRequestV4(
            valid: true,
            context: request.context,
            labelClass: label,
            movePoints: movePoints,
            sampleWeight: fxClamp(movePoints / 50_000.0, 0.25, 3.0),
            nextVolumeTarget: request.context.dataHasVolume ? Double(marketSeries.volume[sampleIndex + 1]) : 0.0,
            windowSize: request.windowSize,
            x: request.x,
            xWindow: request.xWindow
        )
    }

    private static func expectedLabel(
        marketSeries: M1OHLCVSeries,
        sampleIndex: Int,
        horizon: Int
    ) -> LabelClass? {
        let horizonBars = min(horizon, marketSeries.count - sampleIndex - 1)
        guard horizonBars >= 1 else { return nil }
        let delta = marketSeries.close[sampleIndex + horizonBars] - marketSeries.close[sampleIndex]
        guard abs(delta) > Self.minimumDirectionalDeltaPoints else { return nil }
        return delta > 0 ? .buy : .sell
    }

    private static func highSignalHoldoutSampleIndices(
        marketSeries: M1OHLCVSeries,
        horizon: Int,
        limit: Int
    ) -> [Int] {
        let holdoutStartIndex = Self.trainingDays * 24 * 60
        let holdoutEndIndex = min(
            (Self.trainingDays + Self.holdoutDays) * 24 * 60,
            marketSeries.count - horizon - 1
        )
        guard holdoutStartIndex < holdoutEndIndex else {
            return []
        }
        let candidates = (holdoutStartIndex..<holdoutEndIndex).compactMap { sampleIndex -> (index: Int, label: LabelClass, minute: Int, magnitude: Double)? in
            let horizonBars = min(horizon, marketSeries.count - sampleIndex - 1)
            guard horizonBars >= 1 else { return nil }
            let delta = marketSeries.close[sampleIndex + horizonBars] - marketSeries.close[sampleIndex]
            guard abs(delta) > Self.minimumDirectionalDeltaPoints else { return nil }
            return (
                sampleIndex,
                delta > 0 ? .buy : .sell,
                Self.minuteOfHour(timestampUTC: marketSeries.utcTimestamps[sampleIndex]),
                abs(Double(delta))
            )
        }

        func strongest(_ values: [(index: Int, label: LabelClass, minute: Int, magnitude: Double)]) -> [(index: Int, label: LabelClass, minute: Int, magnitude: Double)] {
            values.sorted { lhs, rhs in
                if lhs.magnitude == rhs.magnitude {
                    return lhs.index < rhs.index
                }
                return lhs.magnitude > rhs.magnitude
            }
        }

        let buyLimit = max(1, limit / 2)
        let sellLimit = max(1, limit - buyLimit)
        var selected: [(index: Int, label: LabelClass, minute: Int, magnitude: Double)] = []
        var selectedMinutes = Set<Int>()

        func appendUniqueMinute(
            from values: [(index: Int, label: LabelClass, minute: Int, magnitude: Double)],
            limit appendLimit: Int
        ) {
            for candidate in strongest(values) where selected.count < limit {
                guard selected.filter({ $0.label == candidate.label }).count < appendLimit else {
                    break
                }
                guard selectedMinutes.insert(candidate.minute).inserted else {
                    continue
                }
                selected.append(candidate)
            }
        }

        appendUniqueMinute(from: candidates.filter { $0.label == .buy }, limit: buyLimit)
        appendUniqueMinute(from: candidates.filter { $0.label == .sell }, limit: sellLimit)

        if selected.count < limit {
            let selectedIndexes = Set(selected.map(\.index))
            for candidate in strongest(candidates.filter { !selectedIndexes.contains($0.index) }) where selected.count < limit {
                guard selectedMinutes.insert(candidate.minute).inserted else {
                    continue
                }
                selected.append(candidate)
            }
        }

        return selected
            .prefix(limit)
            .map(\.index)
            .sorted()
    }

    private static func minuteOfHour(timestampUTC: Int64) -> Int {
        let secondsIntoHour = ((timestampUTC % 3_600) + 3_600) % 3_600
        return Int(secondsIntoHour / 60)
    }

    private static func directionalEdge(_ prediction: PredictionV4) -> Double {
        prediction.classProbabilities[LabelClass.buy.rawValue] -
            prediction.classProbabilities[LabelClass.sell.rawValue]
    }

    private static func normalizedConfidence(_ prediction: PredictionV4) -> Double {
        guard prediction.confidence.isFinite else {
            return 0.0
        }
        return min(max(prediction.confidence, 0.0), 1.0)
    }

    private static func certificationHorizon(for plugin: any FXAIPlannedPlugin) -> Int {
        plugin.manifest.minHorizonMinutes
    }

    private static func certificationHyperParameters() -> HyperParameters {
        HyperParameters(
            learningRate: 0.04,
            l2: 0.00001,
            ftrlAlpha: 0.20,
            ftrlL2: 0.00001,
            passiveAggressiveC: 1.0,
            passiveAggressiveMargin: 0.02,
            xgbLearningRate: 0.12,
            xgbL2: 0.00001,
            xgbSplit: 0.0,
            mlpLearningRate: 0.03,
            mlpL2: 0.00001,
            quantileLearningRate: 0.04,
            quantileL2: 0.00001,
            enhashLearningRate: 0.04,
            enhashL2: 0.00001
        )
    }

    private static func failureReason(
        evaluatedSamples: Int,
        validPredictions: Int,
        accuracy: Double,
        meanSignedEdge: Double,
        meanConfidence: Double,
        minimumConfidence: Double,
        validationFailure: String?,
        minimumEvaluationSamples: Int = SineWavePredictionCertificationTests.minimumEvaluationSamples,
        requiredDirectionalAccuracy: Double = SineWavePredictionCertificationTests.requiredDirectionalAccuracy,
        requiredMeanSignedEdge: Double = SineWavePredictionCertificationTests.requiredMeanSignedEdge,
        requiredPredictionConfidence: Double = SineWavePredictionCertificationTests.requiredPredictionConfidence
    ) -> String? {
        if let validationFailure {
            return "invalid prediction: \(validationFailure)"
        }
        if evaluatedSamples < minimumEvaluationSamples {
            return "insufficient holdout samples \(evaluatedSamples)/\(minimumEvaluationSamples)"
        }
        if validPredictions != evaluatedSamples {
            return "valid predictions \(validPredictions)/\(evaluatedSamples)"
        }
        if accuracy < requiredDirectionalAccuracy {
            return "directional accuracy \(formatPercent(accuracy)) below \(formatPercent(requiredDirectionalAccuracy))"
        }
        if meanSignedEdge <= requiredMeanSignedEdge {
            return "mean signed edge \(formatDouble(meanSignedEdge)) <= \(formatDouble(requiredMeanSignedEdge))"
        }
        if minimumConfidence < requiredPredictionConfidence {
            return "minimum confidence \(formatPercent(minimumConfidence)) below \(formatPercent(requiredPredictionConfidence)); mean confidence \(formatPercent(meanConfidence))"
        }
        return nil
    }

    private static func applySchema(
        _ x: [Double],
        plugin: any FXAIPlannedPlugin,
        schemaPolicy: FeatureSchemaPolicy
    ) -> [Double] {
        schemaPolicy.apply(schema: plugin.manifest.featureSchema, groups: plugin.manifest.featureGroups, to: x)
    }

    private static func markdownReport(
        title: String,
        results: [PluginSineWavePredictionResult],
        minimumEvaluationSamples: Int,
        requiredDirectionalAccuracy: Double = SineWavePredictionCertificationTests.requiredDirectionalAccuracy,
        requiredMeanSignedEdge: Double = SineWavePredictionCertificationTests.requiredMeanSignedEdge,
        requiredPredictionConfidence: Double = SineWavePredictionCertificationTests.requiredPredictionConfidence
    ) -> String {
        var lines: [String] = [
            "# \(title)",
            "",
            "| Plugin | Backend | Status | Train | Eval | Valid | Accuracy | Mean Signed Edge | Mean Absolute Edge | Mean Confidence | Min Confidence | Notes |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |"
        ]
        for result in results {
            lines.append(
                "| \(result.pluginName) | \(result.backendName) | \(result.passed ? "PASS" : "FAIL") | " +
                    "\(result.trainedSamples) | \(result.evaluatedSamples) | \(result.validPredictions) | " +
                    "\(formatPercent(result.directionalAccuracy)) | \(formatDouble(result.meanSignedEdge)) | " +
                    "\(formatDouble(result.meanAbsoluteEdge)) | \(formatPercent(result.meanConfidence)) | " +
                    "\(formatPercent(result.minimumConfidence)) | \(result.failureReason ?? "") |"
            )
        }
        lines.append("")
        lines.append("Pass criteria: valid predictions for all evaluated samples, at least \(minimumEvaluationSamples) holdout samples, directional accuracy >= \(formatPercent(requiredDirectionalAccuracy)), mean signed edge > \(formatDouble(requiredMeanSignedEdge)), and every prediction confidence >= \(formatPercent(requiredPredictionConfidence)).")
        return lines.joined(separator: "\n")
    }

    private static func writeTemporaryReport(_ report: String, fileName: String) throws {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent(fileName)
        try report.write(to: url, atomically: true, encoding: .utf8)
        print("SineTest plugin prediction report: \(url.path)")
    }

    private static func runtimeConfiguration(
        mode: FXPluginRuntimeMode,
        environment: FXPluginRuntimeEnvironment,
        stateDirectory: URL
    ) -> FXAIPluginRuntimeConfiguration {
        FXAIPluginRuntimeConfiguration(
            mode: mode,
            fallbackPolicy: .strict,
            environment: environment,
            pythonExecutable: "python3",
            pythonEnvironment: [
                "FXAI_PLUGIN_STATE_DIR": stateDirectory.path,
                "TF_CPP_MIN_LOG_LEVEL": "2"
            ]
        )
    }

    private static func runtimeMode(for backend: FXPluginAccelerationBackend) -> FXPluginRuntimeMode {
        switch backend {
        case .swiftScalar:
            return .swiftScalar
        case .swiftSIMD:
            return .swiftSIMD
        case .accelerate:
            return .accelerate
        case .metal:
            return .metal
        case .pyTorchMPS:
            return .pyTorchMPS
        case .tensorFlowMetal:
            return .tensorFlowMetal
        case .foundationNLP:
            return .foundationNLP
        case .coreMLNeuralEngine:
            return .coreMLNeuralEngine
        }
    }

    private static func acceleratorEnvironment() -> FXPluginRuntimeEnvironment {
        let metalDevice = MetalAccelerationDevice.probe()
        return FXPluginRuntimeEnvironment(
            metalDevice: metalDevice,
            pythonExecutable: "python3",
            pyTorchMPSAvailable: pythonSucceeds("""
                import torch
                mps = getattr(torch.backends, "mps", None)
                raise SystemExit(0 if mps is not None and torch.backends.mps.is_available() else 1)
                """),
            tensorFlowMetalAvailable: pythonSucceeds("""
                import tensorflow as tf
                raise SystemExit(0 if tf.config.list_physical_devices("GPU") else 1)
                """),
            foundationNLPAvailable: metalDevice.optimizedForFXAIAppleSilicon,
            coreMLNeuralEngineAvailable: false
        )
    }

    private static func pythonSucceeds(_ script: String) -> Bool {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/env")
        process.arguments = ["python3", "-c", script]
        process.standardOutput = Pipe()
        process.standardError = Pipe()
        do {
            try process.run()
            process.waitUntilExit()
            return process.terminationStatus == 0
        } catch {
            return false
        }
    }

    private static func makeTemporaryDirectory(prefix: String) throws -> URL {
        let token = prefix.replacingOccurrences(of: "/", with: "_")
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("\(token)-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }

    fileprivate static func formatPercent(_ value: Double) -> String {
        String(format: "%.1f%%", value * 100.0)
    }

    fileprivate static func formatDouble(_ value: Double) -> String {
        String(format: "%.4f", value)
    }
}

private struct PluginSineWavePredictionResult {
    var pluginName: String
    var backendName: String
    var trainedSamples: Int
    var evaluatedSamples: Int
    var validPredictions: Int
    var directionalAccuracy: Double
    var meanSignedEdge: Double
    var meanAbsoluteEdge: Double
    var meanConfidence: Double
    var minimumConfidence: Double
    var failureReason: String?

    var passed: Bool {
        failureReason == nil
    }

    var failureSummary: String {
        "\(pluginName)[\(backendName)]: \(failureReason ?? "passed") " +
            "(accuracy=\(SineWavePredictionCertificationTests.formatPercent(directionalAccuracy)), " +
            "meanSignedEdge=\(SineWavePredictionCertificationTests.formatDouble(meanSignedEdge)), " +
            "minConfidence=\(SineWavePredictionCertificationTests.formatPercent(minimumConfidence)))"
    }
}
