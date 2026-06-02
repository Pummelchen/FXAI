import Foundation

public struct PreparedPluginPayload: Sendable {
    public let dataBundle: DataCoreBundle
    public let featureFrame: FeatureCoreFrame
    public let normalizationFrame: NormalizationCoreFrame
    public let payloadFrame: NormalizationPayloadFrame
    public let context: PluginContextV4

    public var predictRequest: PredictRequestV4 {
        PredictRequestV4(
            valid: true,
            context: context,
            windowSize: payloadFrame.windowSize,
            x: payloadFrame.x,
            xWindow: payloadFrame.xWindow
        )
    }
}

public struct FXDataEnginePipeline: Sendable {
    public let dataCore: DataCore
    public let featureCore: FeatureCore
    public let normalizationCore: NormalizationCore
    public let schemaPolicy: FeatureSchemaPolicy

    public init(
        dataCore: DataCore = DataCore(),
        featureCore: FeatureCore = FeatureCore(),
        normalizationCore: NormalizationCore = NormalizationCore(),
        schemaPolicy: FeatureSchemaPolicy = FeatureSchemaPolicy()
    ) {
        self.dataCore = dataCore
        self.featureCore = featureCore
        self.normalizationCore = normalizationCore
        self.schemaPolicy = schemaPolicy
    }

    public func preparePredictPayload(
        universe: MarketUniverse,
        request dataRequest: DataCoreRequest,
        manifest: PluginManifestV4,
        horizonMinutes: Int,
        normalizationMethod: FeatureNormalizationMethod = .existing,
        normalizationFitState: NormalizationFitState? = nil,
        configuredHorizons: [Int] = HorizonTools.defaultConfiguredHorizons
    ) throws -> PreparedPluginPayload {
        try manifest.validate()
        let dataBundle = try dataCore.buildBundle(request: dataRequest, universe: universe)
        let featureFrame = try featureCore.buildFrame(
            bundle: dataBundle,
            request: FeatureCoreRequest(
                sampleIndex: dataBundle.sampleIndex,
                horizonMinutes: horizonMinutes,
                normalizationMethod: normalizationMethod
            )
        )
        let normalizationFrame = try normalizationCore.buildInputFrame(
            from: featureFrame,
            fitState: normalizationFitState,
            configuredHorizons: configuredHorizons
        )
        let sequenceBars = manifest.resolvedSequenceBars(horizonMinutes: horizonMinutes)
        let xWindow = buildInputWindow(
            universe: universe,
            centerIndex: dataBundle.sampleIndex,
            sequenceBars: sequenceBars,
            horizonMinutes: horizonMinutes,
            normalizationMethod: normalizationMethod,
            normalizationFitState: normalizationFitState,
            configuredHorizons: configuredHorizons
        )
        let payloadFrame = try normalizationCore.buildPayloadFrame(NormalizationPayloadRequest(
            valid: true,
            featureSchema: manifest.featureSchema,
            featureGroups: manifest.featureGroups,
            normalizationMethod: normalizationMethod,
            horizonMinutes: horizonMinutes,
            sequenceBars: sequenceBars,
            sampleTimeUTC: featureFrame.sampleTimeUTC,
            windowSize: xWindow.count,
            x: normalizationFrame.modelInput,
            xWindow: xWindow
        ))
        let context = PluginContextV4(
            regimeID: 0,
            sessionBucket: PluginContractTools.deriveSessionBucket(timestampUTC: featureFrame.sampleTimeUTC),
            horizonMinutes: horizonMinutes,
            featureSchema: manifest.featureSchema,
            normalizationMethod: normalizationMethod,
            sequenceBars: sequenceBars,
            pointValue: 1.0 / pow(10.0, Double(universe.primary.metadata.digits)),
            domainHash: PluginContractTools.symbolHash01(universe.primarySymbol),
            sampleTimeUTC: featureFrame.sampleTimeUTC,
            dataHasVolume: featureFrame.hasVolume
        )
        return PreparedPluginPayload(
            dataBundle: dataBundle,
            featureFrame: featureFrame,
            normalizationFrame: normalizationFrame,
            payloadFrame: payloadFrame,
            context: context
        )
    }

    public func prepareTrainPayload(
        universe: MarketUniverse,
        request dataRequest: DataCoreRequest,
        manifest: PluginManifestV4,
        horizonMinutes: Int,
        roundTripCostPoints: Double = 0.0,
        evThresholdPoints: Double = 0.0,
        normalizationMethod: FeatureNormalizationMethod = .existing,
        tradeKillerMinutes: Int? = nil,
        normalizationFitState: NormalizationFitState? = nil,
        configuredHorizons: [Int] = HorizonTools.defaultConfiguredHorizons
    ) throws -> PreparedTrainingPayload {
        try manifest.validate()
        let dataBundle = try dataCore.buildBundle(request: dataRequest, universe: universe)
        let horizon = TrainingSampleTools.clampHorizon(horizonMinutes)
        guard dataBundle.sampleIndex + horizon < dataBundle.primary.count else {
            throw FXDataEngineError.insufficientData("sample index \(dataBundle.sampleIndex) does not leave \(horizon) future M1 bars for training label")
        }

        let featureFrame = try featureCore.buildFrame(
            bundle: dataBundle,
            request: FeatureCoreRequest(
                sampleIndex: dataBundle.sampleIndex,
                horizonMinutes: horizon,
                normalizationMethod: normalizationMethod
            )
        )
        let normalizationFrame = try normalizationCore.buildInputFrame(
            from: featureFrame,
            fitState: normalizationFitState,
            configuredHorizons: configuredHorizons
        )
        let label = TrainingSampleTools.buildTripleBarrierLabel(
            series: dataBundle.primary,
            index: dataBundle.sampleIndex,
            horizonMinutes: horizon,
            roundTripCostPoints: roundTripCostPoints,
            evThresholdPoints: evThresholdPoints,
            tradeKillerMinutes: tradeKillerMinutes
        )
        let sequenceBars = manifest.resolvedSequenceBars(horizonMinutes: horizon)
        let xWindow = buildInputWindow(
            universe: universe,
            centerIndex: dataBundle.sampleIndex,
            sequenceBars: sequenceBars,
            horizonMinutes: horizon,
            normalizationMethod: normalizationMethod,
            normalizationFitState: normalizationFitState,
            configuredHorizons: configuredHorizons
        )
        let payloadFrame = try normalizationCore.buildPayloadFrame(NormalizationPayloadRequest(
            valid: true,
            featureSchema: manifest.featureSchema,
            featureGroups: manifest.featureGroups,
            normalizationMethod: normalizationMethod,
            horizonMinutes: horizon,
            sequenceBars: sequenceBars,
            sampleTimeUTC: featureFrame.sampleTimeUTC,
            windowSize: xWindow.count,
            x: normalizationFrame.modelInput,
            xWindow: xWindow
        ))

        let primary = dataBundle.primary
        let pointValue = 1.0 / pow(10.0, Double(primary.metadata.digits))
        let minMovePoints = max(0.0, roundTripCostPoints)
        let pathRisk = TrainingSampleTools.pathRisk(
            mfePoints: label.mfePoints,
            maePoints: label.maePoints,
            minMovePoints: minMovePoints,
            timeToHitFraction: label.timeToHitFraction,
            pathFlags: label.pathFlags
        )
        let edge = max(abs(label.realizedMovePoints) - minMovePoints, 0.0)
        let quality = trainingQuality(
            label: label,
            minMovePoints: minMovePoints,
            liquidityStress: 0.0
        )
        let sampleWeight = fxClamp(
            (label.labelClass == .skip ? 0.85 : 1.20) *
                quality *
                (0.75 + edge / max(minMovePoints, 0.50)),
            0.25,
            7.50
        )
        let sample = PreparedTrainingSample(
            valid: true,
            labelClass: label.labelClass,
            regimeID: TrainingSampleTools.staticRegimeID(series: primary, index: dataBundle.sampleIndex),
            horizonMinutes: horizon,
            horizonSlot: TrainingSampleTools.horizonSlot(horizonMinutes: horizon),
            movePoints: label.realizedMovePoints,
            minMovePoints: minMovePoints,
            costPoints: minMovePoints,
            sampleWeight: sampleWeight,
            qualityScore: quality,
            mfePoints: label.mfePoints,
            maePoints: label.maePoints,
            timeToHitFraction: label.timeToHitFraction,
            pathFlags: label.pathFlags,
            pathRisk: pathRisk,
            fillRisk: TrainingSampleTools.fillRisk(liquidityStressPoints: 0.0, minMovePoints: minMovePoints, costPoints: minMovePoints),
            maskedStepTarget: maskedStepTarget(series: primary, index: dataBundle.sampleIndex),
            nextVolumeTarget: nextVolumeTarget(series: primary, index: dataBundle.sampleIndex, horizonMinutes: horizon),
            regimeShiftTarget: regimeShiftTarget(series: primary, index: dataBundle.sampleIndex, horizonMinutes: horizon),
            contextLeadTarget: contextLeadTarget(bundle: dataBundle, movePoints: label.realizedMovePoints),
            pointValue: pointValue,
            domainHash: PluginContractTools.symbolHash01(universe.primarySymbol),
            sampleTimeUTC: featureFrame.sampleTimeUTC,
            x: normalizationFrame.modelInput
        )
        let context = PluginContextV4(
            regimeID: sample.regimeID,
            sessionBucket: PluginContractTools.deriveSessionBucket(timestampUTC: featureFrame.sampleTimeUTC),
            horizonMinutes: horizon,
            featureSchema: manifest.featureSchema,
            normalizationMethod: normalizationMethod,
            sequenceBars: sequenceBars,
            pointValue: pointValue,
            domainHash: sample.domainHash,
            sampleTimeUTC: featureFrame.sampleTimeUTC,
            dataHasVolume: featureFrame.hasVolume
        )
        return PreparedTrainingPayload(
            dataBundle: dataBundle,
            featureFrame: featureFrame,
            normalizationFrame: normalizationFrame,
            payloadFrame: payloadFrame,
            sample: sample,
            context: context
        )
    }

    public func prepareTrainingDataset(
        universe: MarketUniverse,
        baseRequest: DataCoreRequest,
        manifest: PluginManifestV4,
        datasetRequest: TrainingDatasetRequest
    ) throws -> PreparedTrainingDataset {
        try manifest.validate()
        let horizon = TrainingSampleTools.clampHorizon(datasetRequest.horizonMinutes)
        let lowerBound = max(baseRequest.neededBars - 1, 0)
        let upperBound = universe.primary.count - horizon - 1
        let start = min(max(datasetRequest.startIndex ?? lowerBound, lowerBound), max(lowerBound, upperBound))
        let end = min(max(datasetRequest.endIndex ?? upperBound, start), upperBound)
        guard upperBound >= lowerBound, datasetRequest.maxSamples > 0 else {
            return PreparedTrainingDataset(
                symbol: baseRequest.symbol,
                horizonMinutes: horizon,
                startIndex: start,
                endIndex: end,
                stride: datasetRequest.stride,
                normalizationFitStartIndex: nil,
                normalizationFitEndIndex: nil,
                normalizationFitSampleCount: 0,
                payloads: []
            )
        }

        let sampleIndices = trainingSampleIndices(
            start: start,
            end: end,
            stride: datasetRequest.stride,
            maxSamples: datasetRequest.maxSamples
        )
        let fitSampleIndices = normalizationFitSampleIndices(
            sampleIndices: sampleIndices,
            sampleStart: start,
            sampleEnd: end,
            stride: datasetRequest.stride,
            horizonMinutes: horizon,
            normalizationMethod: datasetRequest.normalizationMethod
        )
        let fitState = try normalizationFitState(
            universe: universe,
            baseRequest: baseRequest,
            sampleIndices: fitSampleIndices,
            horizonMinutes: horizon,
            normalizationMethod: datasetRequest.normalizationMethod
        )
        let hasFit = datasetRequest.normalizationMethod.usesFittedStats && fitState != nil

        var payloads: [PreparedTrainingPayload] = []
        payloads.reserveCapacity(sampleIndices.count)
        for sampleIndex in sampleIndices {
            let request = DataCoreRequest(
                liveMode: baseRequest.liveMode,
                symbol: baseRequest.symbol,
                neededBars: baseRequest.neededBars,
                alignUpToIndex: sampleIndex,
                contextSymbols: baseRequest.contextSymbols
            )
            let payload = try prepareTrainPayload(
                universe: universe,
                request: request,
                manifest: manifest,
                horizonMinutes: horizon,
                roundTripCostPoints: datasetRequest.roundTripCostPoints,
                evThresholdPoints: datasetRequest.evThresholdPoints,
                normalizationMethod: datasetRequest.normalizationMethod,
                tradeKillerMinutes: datasetRequest.tradeKillerMinutes,
                normalizationFitState: fitState
            )
            payloads.append(payload)
        }

        return PreparedTrainingDataset(
            symbol: baseRequest.symbol,
            horizonMinutes: horizon,
            startIndex: start,
            endIndex: end,
            stride: datasetRequest.stride,
            normalizationFitStartIndex: hasFit ? fitSampleIndices.first : nil,
            normalizationFitEndIndex: hasFit ? fitSampleIndices.last : nil,
            normalizationFitSampleCount: hasFit ? fitSampleIndices.count : 0,
            payloads: payloads
        )
    }

    public func buildInputWindow(
        universe: MarketUniverse,
        centerIndex: Int,
        sequenceBars: Int,
        horizonMinutes: Int = 1,
        normalizationMethod: FeatureNormalizationMethod,
        normalizationFitState: NormalizationFitState? = nil,
        configuredHorizons: [Int] = HorizonTools.defaultConfiguredHorizons
    ) -> [[Double]] {
        let capped = min(max(1, sequenceBars), FXDataEngineConstants.maxSequenceBars)
        guard capped > 1, centerIndex > 0 else { return [] }
        let rows = min(capped - 1, centerIndex)
        var window: [[Double]] = []
        window.reserveCapacity(rows)
        for offset in stride(from: rows, through: 1, by: -1) {
            let featureFrame = FeatureCoreFrame(
                valid: true,
                sampleIndex: centerIndex - offset,
                horizonMinutes: TrainingSampleTools.clampHorizon(horizonMinutes),
                normalizationMethod: normalizationMethod,
                sampleTimeUTC: universe.primary.utcTimestamps[centerIndex - offset],
                hasVolume: FeatureCore.hasUsableVolume(universe),
                hasPrevious: centerIndex - offset > 0,
                raw: featureCore.buildFeatureVector(universe: universe, sampleIndex: centerIndex - offset),
                previous: centerIndex - offset > 0
                    ? featureCore.buildFeatureVector(universe: universe, sampleIndex: centerIndex - offset - 1)
                    : Array(repeating: 0.0, count: FXDataEngineConstants.aiFeatures)
            )
            if let normalized = try? normalizationCore.buildInputFrame(
                from: featureFrame,
                fitState: normalizationFitState,
                configuredHorizons: configuredHorizons
            ) {
                window.append(normalized.modelInput)
            }
        }
        return window
    }

    private func trainingSampleIndices(start: Int, end: Int, stride: Int, maxSamples: Int) -> [Int] {
        var indices: [Int] = []
        indices.reserveCapacity(min(maxSamples, max(0, ((end - start) / max(stride, 1)) + 1)))
        var sampleIndex = start
        while sampleIndex <= end, indices.count < maxSamples {
            indices.append(sampleIndex)
            sampleIndex += max(stride, 1)
        }
        return indices
    }

    private func normalizationFitSampleIndices(
        sampleIndices: [Int],
        sampleStart: Int,
        sampleEnd: Int,
        stride: Int,
        horizonMinutes: Int,
        normalizationMethod: FeatureNormalizationMethod
    ) -> [Int] {
        guard normalizationMethod.usesFittedStats else { return sampleIndices }
        guard let split = WarmupTools.normalizationCandidateSplit(
            horizonMinutes: horizonMinutes,
            startIndex: sampleStart,
            endIndex: sampleEnd
        ) else {
            return sampleIndices
        }

        let trainRange = split.trainingStart...split.trainingEnd
        let candidateIndices = sampleIndices.filter { trainRange.contains($0) }
        if candidateIndices.count >= 8 {
            return candidateIndices
        }

        return trainingSampleIndices(
            start: split.trainingStart,
            end: split.trainingEnd,
            stride: 1,
            maxSamples: max(8, min(512, max(sampleIndices.count, 8)))
        )
    }

    private func normalizationFitState(
        universe: MarketUniverse,
        baseRequest: DataCoreRequest,
        sampleIndices: [Int],
        horizonMinutes: Int,
        normalizationMethod: FeatureNormalizationMethod
    ) throws -> NormalizationFitState? {
        guard normalizationMethod.usesFittedStats else { return nil }
        var rawRows: [[Double]] = []
        rawRows.reserveCapacity(sampleIndices.count)
        for sampleIndex in sampleIndices {
            let request = DataCoreRequest(
                liveMode: baseRequest.liveMode,
                symbol: baseRequest.symbol,
                neededBars: baseRequest.neededBars,
                alignUpToIndex: sampleIndex,
                contextSymbols: baseRequest.contextSymbols
            )
            let bundle = try dataCore.buildBundle(request: request, universe: universe)
            let frame = try featureCore.buildFrame(
                bundle: bundle,
                request: FeatureCoreRequest(
                    sampleIndex: bundle.sampleIndex,
                    horizonMinutes: horizonMinutes,
                    normalizationMethod: normalizationMethod
                )
            )
            rawRows.append(frame.raw)
        }
        var fit = NormalizationFitState()
        guard fit.fit(method: normalizationMethod, horizonMinutes: horizonMinutes, rawRows: rawRows) else {
            throw FXDataEngineError.insufficientData(
                "normalization method \(normalizationMethod) needs at least 8 training rows to fit without leakage-prone fallbacks"
            )
        }
        return fit
    }

    private func trainingQuality(
        label: TripleBarrierLabelResult,
        minMovePoints: Double,
        liquidityStress: Double
    ) -> Double {
        let quality: Double
        if label.labelClass == .skip {
            quality = 0.75 - (0.10 * liquidityStress)
        } else {
            let mfeRatio = label.mfePoints / max(minMovePoints, 0.50)
            let adverseRatio = label.maePoints / max(label.mfePoints, minMovePoints)
            let speedBonus = 1.0 - fxClamp(label.timeToHitFraction, 0.0, 1.0)
            var directionalQuality = 0.85 +
                0.20 * fxClamp(mfeRatio, 0.0, 4.0) +
                0.20 * speedBonus -
                0.15 * fxClamp(adverseRatio, 0.0, 3.0) -
                0.10 * liquidityStress
            if label.pathFlags.contains(.dualHit) { directionalQuality -= 0.12 }
            if label.pathFlags.contains(.killedEarly) { directionalQuality -= 0.10 }
            quality = directionalQuality
        }
        return fxClamp(quality, 0.35, 2.20)
    }

    private func maskedStepTarget(series: M1OHLCVSeries, index: Int) -> Double {
        guard index >= 0, index + 1 < series.count else { return 0.0 }
        return TrainingSampleTools.movePoints(from: series.close[index], to: series.close[index + 1])
    }

    private func nextVolumeTarget(
        series: M1OHLCVSeries,
        index: Int,
        horizonMinutes: Int
    ) -> Double {
        guard series.hasVolume, index >= 0, index + 1 < series.count else { return 0.0 }
        let targetIndex = min(index + TrainingSampleTools.clampHorizon(horizonMinutes), series.count - 1)
        guard targetIndex > index else { return 0.0 }
        return Double(series.volume[targetIndex])
    }

    private func regimeShiftTarget(series: M1OHLCVSeries, index: Int, horizonMinutes: Int) -> Double {
        let auxHorizon = min(max(1, horizonMinutes), 8)
        guard index >= 0, index + auxHorizon < series.count else { return 0.0 }
        let currentRegime = TrainingSampleTools.staticRegimeID(series: series, index: index)
        let futureRegime = TrainingSampleTools.staticRegimeID(series: series, index: index + auxHorizon)
        return currentRegime == futureRegime ? 0.0 : 1.0
    }

    private func contextLeadTarget(bundle: DataCoreBundle, movePoints: Double) -> Double {
        let index = bundle.sampleIndex
        guard index >= 0,
              index < bundle.contextAggregates.mean.count,
              index < bundle.contextAggregates.standardDeviation.count else {
            return 0.5
        }
        let mean = bundle.contextAggregates.mean[index]
        let standardDeviation = bundle.contextAggregates.standardDeviation[index]
        let signal = standardDeviation > 1e-6 ? mean / standardDeviation : mean
        return fxClamp(0.5 + 0.5 * sign(signal) * sign(movePoints), 0.0, 1.0)
    }

    private func sign(_ value: Double) -> Double {
        if value > 0 { return 1.0 }
        if value < 0 { return -1.0 }
        return 0.0
    }
}
