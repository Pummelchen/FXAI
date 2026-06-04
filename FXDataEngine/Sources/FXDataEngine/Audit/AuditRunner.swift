import Foundation

public struct AuditRunnerConfiguration: Codable, Hashable, Sendable {
    public var horizonMinutes: Int
    public var pointValue: Double
    public var priceCostPoints: Double
    public var evThresholdPoints: Double
    public var normalizationMethod: FeatureNormalizationMethod
    public var maxSamples: Int
    public var walkForwardTrainBars: Int
    public var walkForwardTestBars: Int
    public var walkForwardPurgeBars: Int
    public var walkForwardEmbargoBars: Int
    public var walkForwardFolds: Int
    public var walkForwardAnchored: Bool

    public init(
        horizonMinutes: Int,
        pointValue: Double = 0.0001,
        priceCostPoints: Double = 0.0,
        evThresholdPoints: Double = 0.25,
        normalizationMethod: FeatureNormalizationMethod = .existing,
        maxSamples: Int = Int.max,
        walkForwardTrainBars: Int = 256,
        walkForwardTestBars: Int = 64,
        walkForwardPurgeBars: Int = 32,
        walkForwardEmbargoBars: Int = 24,
        walkForwardFolds: Int = 6,
        walkForwardAnchored: Bool = false
    ) {
        self.horizonMinutes = TrainingSampleTools.clampHorizon(horizonMinutes)
        let sanitizedPointValue = fxSafeFinite(pointValue)
        self.pointValue = sanitizedPointValue > 0.0 ? sanitizedPointValue : 0.0001
        self.priceCostPoints = max(0.0, fxSafeFinite(priceCostPoints))
        self.evThresholdPoints = max(0.0, fxSafeFinite(evThresholdPoints))
        self.normalizationMethod = normalizationMethod
        self.maxSamples = max(0, maxSamples)
        self.walkForwardTrainBars = max(12, walkForwardTrainBars)
        self.walkForwardTestBars = max(12, walkForwardTestBars)
        self.walkForwardPurgeBars = max(0, walkForwardPurgeBars)
        self.walkForwardEmbargoBars = max(0, walkForwardEmbargoBars)
        self.walkForwardFolds = min(max(1, walkForwardFolds), 64)
        self.walkForwardAnchored = walkForwardAnchored
    }
}

public enum AuditRunnerTools {
    public static func runScenario<Plugin: FXAIPluginV4>(
        plugin: inout Plugin,
        generated: AuditGeneratedScenarioSeries,
        spec: AuditScenarioSpec,
        configuration: AuditRunnerConfiguration,
        hyperParameters: HyperParameters = HyperParameters()
    ) throws -> AuditScenarioMetrics {
        let manifest = plugin.manifest
        try manifest.validate()
        plugin.reset()

        var metrics = AuditScenarioMetrics(
            aiID: manifest.aiID,
            aiName: manifest.aiName,
            family: manifest.family.rawValue,
            scenario: spec.name,
            barsTotal: generated.primary.count,
            resetDelta: -1.0,
            sequenceDelta: -1.0
        )

        guard generated.primary.isConsistent,
              generated.primary.count > 0,
              configuration.maxSamples > 0 else {
            return AuditScoringTools.finalizeScenarioMetrics(metrics)
        }

        let horizon = configuration.horizonMinutes
        let startIndex = horizon + 1
        var endIndex = generated.primary.count - 220
        if endIndex <= startIndex { endIndex = generated.primary.count - 32 }
        if endIndex <= startIndex { endIndex = generated.primary.count - 2 }
        guard endIndex > startIndex else {
            return AuditScoringTools.finalizeScenarioMetrics(metrics)
        }

        if spec.name == "market_walkforward" {
            return try runWalkForwardScenario(
                plugin: &plugin,
                generated: generated,
                spec: spec,
                configuration: configuration,
                manifest: manifest,
                startIndex: startIndex,
                endIndex: endIndex,
                hyperParameters: hyperParameters,
                baseMetrics: metrics
            )
        }

        var heldRequest: PredictRequestV4?
        var processed = 0
        var index = startIndex
        while index < endIndex, processed < configuration.maxSamples {
            let sample: AuditPreparedSample
            do {
                sample = try AuditSampleTools.buildSample(
                    generated: generated,
                    sampleIndexAsSeries: index,
                    horizonMinutes: horizon,
                    manifest: manifest,
                    pointValue: configuration.pointValue,
                    priceCostPoints: configuration.priceCostPoints,
                    evThresholdPoints: configuration.evThresholdPoints,
                    normalizationMethod: configuration.normalizationMethod
                )
            } catch {
                index += 1
                continue
            }

            processed += 1
            recordTruth(sample.payload.sample.labelClass, into: &metrics)
            let request = sample.predictRequest
            heldRequest = request

            do {
                try request.validate()
                let prediction = try plugin.predict(request, hyperParameters: hyperParameters)
                try prediction.validate()
                recordValidPrediction(
                    prediction,
                    sample: sample,
                    spec: spec,
                    into: &metrics
                )
            } catch {
                metrics.invalidPredictions += 1
            }

            try sample.trainRequest.validate()
            try plugin.train(sample.trainRequest, hyperParameters: hyperParameters)
            index += 1
        }

        if let heldRequest {
            applyResetAndSequenceChecks(
                plugin: &plugin,
                heldRequest: heldRequest,
                hyperParameters: hyperParameters,
                metrics: &metrics
            )
        }

        return AuditScoringTools.finalizeScenarioMetrics(metrics)
    }

    private static func runWalkForwardScenario<Plugin: FXAIPluginV4>(
        plugin: inout Plugin,
        generated: AuditGeneratedScenarioSeries,
        spec: AuditScenarioSpec,
        configuration: AuditRunnerConfiguration,
        manifest: PluginManifestV4,
        startIndex: Int,
        endIndex: Int,
        hyperParameters: HyperParameters,
        baseMetrics: AuditScenarioMetrics
    ) throws -> AuditScenarioMetrics {
        let trainBars = max(12, configuration.walkForwardTrainBars)
        let testBars = max(12, configuration.walkForwardTestBars)
        let purgeBars = max(0, configuration.walkForwardPurgeBars)
        let embargoBars = max(0, configuration.walkForwardEmbargoBars)
        let step = max(1, testBars + embargoBars)
        var combinedTestMetrics = baseMetrics
        var trainFolds: [AuditFoldMetrics] = []
        var testFolds: [AuditFoldMetrics] = []
        var heldRequest: PredictRequestV4?
        var processed = 0

        for fold in 0..<configuration.walkForwardFolds {
            if processed >= configuration.maxSamples { break }
            plugin.reset()

            let trainStart = configuration.walkForwardAnchored ? startIndex : startIndex + fold * step
            let trainEnd = (configuration.walkForwardAnchored ? startIndex + trainBars + fold * step : trainStart + trainBars)
            let testStart = trainEnd + purgeBars
            let testEnd = testStart + testBars
            if trainStart < startIndex || trainEnd <= trainStart || testStart < trainEnd || testEnd > endIndex {
                break
            }

            var trainMetrics = baseMetrics
            var testMetrics = baseMetrics

            for index in trainStart..<trainEnd {
                if processed >= configuration.maxSamples { break }
                if try processAuditSample(
                    plugin: &plugin,
                    generated: generated,
                    spec: spec,
                    configuration: configuration,
                    manifest: manifest,
                    sampleIndex: index,
                    hyperParameters: hyperParameters,
                    metrics: &trainMetrics,
                    heldRequest: nil,
                    trainAfterPrediction: true
                ) {
                    processed += 1
                }
            }

            for index in testStart..<testEnd {
                if processed >= configuration.maxSamples { break }
                if try processAuditSample(
                    plugin: &plugin,
                    generated: generated,
                    spec: spec,
                    configuration: configuration,
                    manifest: manifest,
                    sampleIndex: index,
                    hyperParameters: hyperParameters,
                    metrics: &testMetrics,
                    heldRequest: &heldRequest,
                    trainAfterPrediction: false
                ) {
                    processed += 1
                }
            }

            mergeScenarioMetrics(testMetrics, into: &combinedTestMetrics)
            trainFolds.append(foldMetrics(from: trainMetrics))
            testFolds.append(foldMetrics(from: testMetrics))
        }

        var output = AuditScoringTools.finalizeWalkForward(
            scenario: combinedTestMetrics,
            trainFolds: trainFolds,
            testFolds: testFolds
        )
        if let heldRequest {
            applyResetAndSequenceChecks(
                plugin: &plugin,
                heldRequest: heldRequest,
                hyperParameters: hyperParameters,
                metrics: &output
            )
        }
        return AuditScoringTools.finalizeScenarioMetrics(output)
    }

    private static func processAuditSample<Plugin: FXAIPluginV4>(
        plugin: inout Plugin,
        generated: AuditGeneratedScenarioSeries,
        spec: AuditScenarioSpec,
        configuration: AuditRunnerConfiguration,
        manifest: PluginManifestV4,
        sampleIndex: Int,
        hyperParameters: HyperParameters,
        metrics: inout AuditScenarioMetrics,
        heldRequest: inout PredictRequestV4?,
        trainAfterPrediction: Bool
    ) throws -> Bool {
        let sample: AuditPreparedSample
        do {
            sample = try AuditSampleTools.buildSample(
                generated: generated,
                sampleIndexAsSeries: sampleIndex,
                horizonMinutes: configuration.horizonMinutes,
                manifest: manifest,
                pointValue: configuration.pointValue,
                priceCostPoints: configuration.priceCostPoints,
                evThresholdPoints: configuration.evThresholdPoints,
                normalizationMethod: configuration.normalizationMethod
            )
        } catch {
            return false
        }

        recordTruth(sample.payload.sample.labelClass, into: &metrics)
        let request = sample.predictRequest
        heldRequest = request

        do {
            try request.validate()
            let prediction = try plugin.predict(request, hyperParameters: hyperParameters)
            try prediction.validate()
            recordValidPrediction(
                prediction,
                sample: sample,
                spec: spec,
                into: &metrics
            )
        } catch {
            metrics.invalidPredictions += 1
        }

        if trainAfterPrediction {
            try sample.trainRequest.validate()
            try plugin.train(sample.trainRequest, hyperParameters: hyperParameters)
        }
        return true
    }

    private static func processAuditSample<Plugin: FXAIPluginV4>(
        plugin: inout Plugin,
        generated: AuditGeneratedScenarioSeries,
        spec: AuditScenarioSpec,
        configuration: AuditRunnerConfiguration,
        manifest: PluginManifestV4,
        sampleIndex: Int,
        hyperParameters: HyperParameters,
        metrics: inout AuditScenarioMetrics,
        heldRequest: PredictRequestV4?,
        trainAfterPrediction: Bool
    ) throws -> Bool {
        var mutableHeldRequest = heldRequest
        return try processAuditSample(
            plugin: &plugin,
            generated: generated,
            spec: spec,
            configuration: configuration,
            manifest: manifest,
            sampleIndex: sampleIndex,
            hyperParameters: hyperParameters,
            metrics: &metrics,
            heldRequest: &mutableHeldRequest,
            trainAfterPrediction: trainAfterPrediction
        )
    }

    private static func mergeScenarioMetrics(_ source: AuditScenarioMetrics, into target: inout AuditScenarioMetrics) {
        target.samplesTotal += source.samplesTotal
        target.validPredictions += source.validPredictions
        target.invalidPredictions += source.invalidPredictions
        target.buyCount += source.buyCount
        target.sellCount += source.sellCount
        target.skipCount += source.skipCount
        target.trueBuyCount += source.trueBuyCount
        target.trueSellCount += source.trueSellCount
        target.trueSkipCount += source.trueSkipCount
        target.exactMatchCount += source.exactMatchCount
        target.directionalEvaluationCount += source.directionalEvaluationCount
        target.directionalCorrectCount += source.directionalCorrectCount
        target.trendAlignmentSum += source.trendAlignmentSum
        target.trendAlignmentCount += source.trendAlignmentCount
        target.confidenceSum += source.confidenceSum
        target.reliabilitySum += source.reliabilitySum
        target.moveSum += source.moveSum
        target.directionalConfidenceSum += source.directionalConfidenceSum
        target.directionalHitSum += source.directionalHitSum
        target.brierSum += source.brierSum
        target.calibrationAbsSum += source.calibrationAbsSum
        target.pathQualityAbsSum += source.pathQualityAbsSum
        target.pathQualityCount += source.pathQualityCount
        target.netSum += source.netSum
        target.macroEventRate += source.macroEventRate
        target.macroPreRate += source.macroPreRate
        target.macroPostRate += source.macroPostRate
        target.macroImportanceMean += source.macroImportanceMean
        target.macroSurpriseAbsMean += source.macroSurpriseAbsMean
        target.macroDataCoverage += source.macroDataCoverage
        target.macroSurpriseZAbsMean += source.macroSurpriseZAbsMean
        target.macroRevisionAbsMean += source.macroRevisionAbsMean
        target.macroCurrencyRelevanceMean += source.macroCurrencyRelevanceMean
        target.macroProvenanceTrustMean += source.macroProvenanceTrustMean
        target.macroRatesRate += source.macroRatesRate
        target.macroInflationRate += source.macroInflationRate
        target.macroLaborRate += source.macroLaborRate
        target.macroGrowthRate += source.macroGrowthRate
    }

    private static func foldMetrics(from metrics: AuditScenarioMetrics) -> AuditFoldMetrics {
        AuditFoldMetrics(
            samplesTotal: metrics.samplesTotal,
            validPredictions: metrics.validPredictions,
            invalidPredictions: metrics.invalidPredictions,
            buyCount: metrics.buyCount,
            sellCount: metrics.sellCount,
            skipCount: metrics.skipCount,
            directionalEvaluationCount: metrics.directionalEvaluationCount,
            directionalCorrectCount: metrics.directionalCorrectCount,
            confidenceSum: metrics.confidenceSum,
            reliabilitySum: metrics.reliabilitySum,
            moveSum: metrics.moveSum,
            brierSum: metrics.brierSum,
            calibrationAbsSum: metrics.calibrationAbsSum,
            pathQualityAbsSum: metrics.pathQualityAbsSum,
            pathQualityCount: metrics.pathQualityCount,
            netSum: metrics.netSum
        )
    }

    private static func recordTruth(_ label: LabelClass, into metrics: inout AuditScenarioMetrics) {
        metrics.samplesTotal += 1
        switch label {
        case .buy:
            metrics.trueBuyCount += 1
        case .sell:
            metrics.trueSellCount += 1
        case .skip:
            metrics.trueSkipCount += 1
        }
    }

    private static func recordValidPrediction(
        _ prediction: PredictionV4,
        sample: AuditPreparedSample,
        spec: AuditScenarioSpec,
        into metrics: inout AuditScenarioMetrics
    ) {
        let label = sample.payload.sample.labelClass
        let decision = AuditScoringTools.decision(from: prediction)
        let movePoints = sample.payload.sample.movePoints
        let minMovePoints = sample.payload.sample.minMovePoints
        let pathFlags = sample.payload.sample.pathFlags
        let target = targetProbabilities(label)
        var brier = 0.0
        for index in 0..<3 {
            let delta = probability(prediction, index: index) - target[index]
            brier += delta * delta
        }

        let directionalEvaluation = decision != .skip
        let directionalCorrect = decision == label
        let directionalConfidence = max(
            probability(prediction, index: LabelClass.buy.rawValue),
            probability(prediction, index: LabelClass.sell.rawValue)
        )
        let calibrationAbs = directionalEvaluation
            ? abs(directionalConfidence - (directionalCorrect ? 1.0 : 0.0))
            : 0.0
        let moveScale = max(abs(movePoints), max(abs(prediction.moveMeanPoints), 0.50))
        let pathQuality =
            0.25 * fxClamp(abs(prediction.mfeMeanPoints - sample.payload.sample.mfePoints) / moveScale, 0.0, 3.0) +
            0.20 * fxClamp(abs(prediction.maeMeanPoints - sample.payload.sample.maePoints) / moveScale, 0.0, 3.0) +
            0.20 * abs(prediction.hitTimeFraction - sample.payload.sample.timeToHitFraction) +
            0.20 * abs(prediction.pathRisk - sample.payload.sample.pathRisk) +
            0.15 * abs(prediction.fillRisk - fxClamp(sample.payload.sample.fillRisk + (pathFlags.contains(.dualHit) ? 0.25 : 0.0), 0.0, 1.0))
        let netPoints = realizedNetPoints(decision: decision, movePoints: movePoints, minMovePoints: minMovePoints)

        metrics.validPredictions += 1
        switch decision {
        case .buy:
            metrics.buyCount += 1
        case .sell:
            metrics.sellCount += 1
        case .skip:
            metrics.skipCount += 1
        }
        if decision == label {
            metrics.exactMatchCount += 1
        }
        recordTrendAlignment(decision: decision, label: label, scenarioName: spec.name, into: &metrics)

        metrics.confidenceSum += prediction.confidence
        metrics.reliabilitySum += prediction.reliability
        metrics.moveSum += prediction.moveMeanPoints
        metrics.brierSum += brier
        metrics.netSum += netPoints

        if directionalEvaluation {
            metrics.directionalEvaluationCount += 1
            metrics.directionalConfidenceSum += directionalConfidence
            metrics.directionalHitSum += directionalCorrect ? 1.0 : 0.0
            metrics.calibrationAbsSum += calibrationAbs
            if directionalCorrect {
                metrics.directionalCorrectCount += 1
            }
        }
        metrics.pathQualityAbsSum += pathQuality
        metrics.pathQualityCount += 1
    }

    private static func recordTrendAlignment(
        decision: LabelClass,
        label: LabelClass,
        scenarioName: String,
        into metrics: inout AuditScenarioMetrics
    ) {
        if scenarioName == "drift_up" || scenarioName == "monotonic_up" {
            if decision == .buy { metrics.trendAlignmentSum += 1.0 }
            if decision == .sell { metrics.trendAlignmentSum -= 1.0 }
            metrics.trendAlignmentCount += 1
        } else if scenarioName == "drift_down" || scenarioName == "monotonic_down" {
            if decision == .sell { metrics.trendAlignmentSum += 1.0 }
            if decision == .buy { metrics.trendAlignmentSum -= 1.0 }
            metrics.trendAlignmentCount += 1
        } else if scenarioName == "market_trend" || scenarioName == "market_walkforward" {
            if label == .buy || label == .sell {
                if decision == label { metrics.trendAlignmentSum += 1.0 }
                if decision != .skip, decision != label { metrics.trendAlignmentSum -= 1.0 }
                metrics.trendAlignmentCount += 1
            }
        }
    }

    private static func applyResetAndSequenceChecks<Plugin: FXAIPluginV4>(
        plugin: inout Plugin,
        heldRequest: PredictRequestV4,
        hyperParameters: HyperParameters,
        metrics: inout AuditScenarioMetrics
    ) {
        let heldPrediction = validatedPrediction {
            try plugin.predict(heldRequest, hyperParameters: hyperParameters)
        }
        plugin.reset()
        let resetPrediction = validatedPrediction {
            try plugin.predict(heldRequest, hyperParameters: hyperParameters)
        }
        if let heldPrediction, let resetPrediction {
            metrics.resetDelta = AuditScoringTools.comparePredictions(heldPrediction, resetPrediction)
        }

        guard heldRequest.context.sequenceBars > 1 else { return }
        var shortContext = heldRequest.context
        shortContext.sequenceBars = 1
        let shortRequest = PredictRequestV4(
            valid: heldRequest.valid,
            context: shortContext,
            windowSize: 0,
            x: heldRequest.x,
            xWindow: []
        )
        plugin.reset()
        let shortPrediction = validatedPrediction {
            try plugin.predict(shortRequest, hyperParameters: hyperParameters)
        }
        plugin.reset()
        let longPrediction = validatedPrediction {
            try plugin.predict(heldRequest, hyperParameters: hyperParameters)
        }
        if let shortPrediction, let longPrediction {
            metrics.sequenceDelta = AuditScoringTools.comparePredictions(shortPrediction, longPrediction)
        }
    }

    private static func validatedPrediction(_ body: () throws -> PredictionV4) -> PredictionV4? {
        do {
            let prediction = try body()
            try prediction.validate()
            return prediction
        } catch {
            return nil
        }
    }

    private static func targetProbabilities(_ label: LabelClass) -> [Double] {
        var target = [0.0, 0.0, 0.0]
        target[label.rawValue] = 1.0
        return target
    }

    private static func probability(_ prediction: PredictionV4, index: Int) -> Double {
        guard index >= 0, index < prediction.classProbabilities.count else { return 0.0 }
        return fxSafeFinite(prediction.classProbabilities[index])
    }

    private static func realizedNetPoints(
        decision: LabelClass,
        movePoints: Double,
        minMovePoints: Double
    ) -> Double {
        switch decision {
        case .buy:
            return movePoints - minMovePoints
        case .sell:
            return -movePoints - minMovePoints
        case .skip:
            return 0.0
        }
    }
}
