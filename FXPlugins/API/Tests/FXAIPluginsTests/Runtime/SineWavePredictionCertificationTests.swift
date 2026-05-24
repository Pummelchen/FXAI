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
    private static let requiredDirectionalAccuracy = 0.68
    private static let requiredMeanSignedEdge = 0.01

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

        let report = Self.markdownReport(results: results)
        try Self.writeTemporaryReport(report)
        print(report)

        let failures = results.filter { !$0.passed }
        XCTAssertTrue(
            failures.isEmpty,
            "SineTest directional sync failures:\n" +
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
                signedEdgeSum += signedEdge
                absoluteEdgeSum += abs(edge)
                if signedEdge > 0.0 {
                    correctSamples += 1
                }
                evaluatedSamples += 1
            }
        }

        let accuracy = evaluatedSamples > 0 ? Double(correctSamples) / Double(evaluatedSamples) : 0.0
        let meanSignedEdge = evaluatedSamples > 0 ? signedEdgeSum / Double(evaluatedSamples) : 0.0
        let meanAbsoluteEdge = evaluatedSamples > 0 ? absoluteEdgeSum / Double(evaluatedSamples) : 0.0
        let failureReason = failureReason(
            evaluatedSamples: evaluatedSamples,
            validPredictions: validPredictions,
            accuracy: accuracy,
            meanSignedEdge: meanSignedEdge,
            validationFailure: validationFailure
        )

        return PluginSineWavePredictionResult(
            pluginName: plugin.manifest.aiName,
            trainedSamples: trainedSamples,
            evaluatedSamples: evaluatedSamples,
            validPredictions: validPredictions,
            directionalAccuracy: accuracy,
            meanSignedEdge: meanSignedEdge,
            meanAbsoluteEdge: meanAbsoluteEdge,
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

    private static func directionalEdge(_ prediction: PredictionV4) -> Double {
        prediction.classProbabilities[LabelClass.buy.rawValue] -
            prediction.classProbabilities[LabelClass.sell.rawValue]
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
        validationFailure: String?
    ) -> String? {
        if let validationFailure {
            return "invalid prediction: \(validationFailure)"
        }
        if evaluatedSamples < Self.minimumEvaluationSamples {
            return "insufficient holdout samples \(evaluatedSamples)/\(Self.minimumEvaluationSamples)"
        }
        if validPredictions != evaluatedSamples {
            return "valid predictions \(validPredictions)/\(evaluatedSamples)"
        }
        if accuracy < Self.requiredDirectionalAccuracy {
            return "directional accuracy \(formatPercent(accuracy)) below \(formatPercent(Self.requiredDirectionalAccuracy))"
        }
        if meanSignedEdge <= Self.requiredMeanSignedEdge {
            return "mean signed edge \(formatDouble(meanSignedEdge)) <= \(formatDouble(Self.requiredMeanSignedEdge))"
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

    private static func markdownReport(results: [PluginSineWavePredictionResult]) -> String {
        var lines: [String] = [
            "# SineTest Plugin Prediction Certification Results",
            "",
            "| Plugin | Status | Train | Eval | Valid | Accuracy | Mean Signed Edge | Mean Absolute Edge | Notes |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |"
        ]
        for result in results {
            lines.append(
                "| \(result.pluginName) | \(result.passed ? "PASS" : "FAIL") | " +
                    "\(result.trainedSamples) | \(result.evaluatedSamples) | \(result.validPredictions) | " +
                    "\(formatPercent(result.directionalAccuracy)) | \(formatDouble(result.meanSignedEdge)) | " +
                    "\(formatDouble(result.meanAbsoluteEdge)) | \(result.failureReason ?? "") |"
            )
        }
        lines.append("")
        lines.append("Pass criteria: valid predictions for all evaluated samples, at least \(Self.minimumEvaluationSamples) holdout samples, directional accuracy >= \(formatPercent(Self.requiredDirectionalAccuracy)), mean signed edge > \(formatDouble(Self.requiredMeanSignedEdge)).")
        return lines.joined(separator: "\n")
    }

    private static func writeTemporaryReport(_ report: String) throws {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("fxai_sinetest_plugin_prediction_results.md")
        try report.write(to: url, atomically: true, encoding: .utf8)
        print("SineTest plugin prediction report: \(url.path)")
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
    var trainedSamples: Int
    var evaluatedSamples: Int
    var validPredictions: Int
    var directionalAccuracy: Double
    var meanSignedEdge: Double
    var meanAbsoluteEdge: Double
    var failureReason: String?

    var passed: Bool {
        failureReason == nil
    }

    var failureSummary: String {
        "\(pluginName): \(failureReason ?? "passed") " +
            "(accuracy=\(SineWavePredictionCertificationTests.formatPercent(directionalAccuracy)), " +
            "meanSignedEdge=\(SineWavePredictionCertificationTests.formatDouble(meanSignedEdge)))"
    }
}
