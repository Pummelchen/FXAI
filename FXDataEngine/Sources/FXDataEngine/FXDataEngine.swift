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
        normalizationMethod: FeatureNormalizationMethod = .existing
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
        let normalizationFrame = try normalizationCore.buildInputFrame(from: featureFrame)
        let sequenceBars = manifest.resolvedSequenceBars(horizonMinutes: horizonMinutes)
        let xWindow = buildInputWindow(
            universe: universe,
            centerIndex: dataBundle.sampleIndex,
            sequenceBars: sequenceBars,
            normalizationMethod: normalizationMethod
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
        tradeKillerMinutes: Int? = nil
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
        let normalizationFrame = try normalizationCore.buildInputFrame(from: featureFrame)
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
            normalizationMethod: normalizationMethod
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
            spreadStress: 0.0
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
            fillRisk: TrainingSampleTools.fillRisk(spreadStressPoints: 0.0, minMovePoints: minMovePoints, costPoints: minMovePoints),
            maskedStepTarget: maskedStepTarget(series: primary, index: dataBundle.sampleIndex),
            nextVolumeTarget: nextVolatilityTarget(series: primary, index: dataBundle.sampleIndex, horizonMinutes: horizon, fallback: abs(label.realizedMovePoints)),
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

    public func buildInputWindow(
        universe: MarketUniverse,
        centerIndex: Int,
        sequenceBars: Int,
        normalizationMethod: FeatureNormalizationMethod
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
                horizonMinutes: 1,
                normalizationMethod: normalizationMethod,
                sampleTimeUTC: universe.primary.utcTimestamps[centerIndex - offset],
                hasVolume: FeatureCore.hasUsableVolume(universe),
                hasPrevious: centerIndex - offset > 0,
                raw: featureCore.buildFeatureVector(universe: universe, sampleIndex: centerIndex - offset),
                previous: centerIndex - offset > 0
                    ? featureCore.buildFeatureVector(universe: universe, sampleIndex: centerIndex - offset - 1)
                    : Array(repeating: 0.0, count: FXDataEngineConstants.aiFeatures)
            )
            if let normalized = try? normalizationCore.buildInputFrame(from: featureFrame) {
                window.append(normalized.modelInput)
            }
        }
        return window
    }

    private func trainingQuality(
        label: TripleBarrierLabelResult,
        minMovePoints: Double,
        spreadStress: Double
    ) -> Double {
        let quality: Double
        if label.labelClass == .skip {
            quality = 0.75 - (0.10 * spreadStress)
        } else {
            let mfeRatio = label.mfePoints / max(minMovePoints, 0.50)
            let adverseRatio = label.maePoints / max(label.mfePoints, minMovePoints)
            let speedBonus = 1.0 - fxClamp(label.timeToHitFraction, 0.0, 1.0)
            var directionalQuality = 0.85 +
                0.20 * fxClamp(mfeRatio, 0.0, 4.0) +
                0.20 * speedBonus -
                0.15 * fxClamp(adverseRatio, 0.0, 3.0) -
                0.10 * spreadStress
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

    private func nextVolatilityTarget(
        series: M1OHLCVSeries,
        index: Int,
        horizonMinutes: Int,
        fallback: Double
    ) -> Double {
        let auxHorizon = min(max(1, horizonMinutes), 8)
        guard index >= 0, index + 1 < series.count else { return abs(fallback) }
        var sum = 0.0
        var count = 0
        for step in 1...auxHorizon {
            let futureIndex = index + step
            guard futureIndex < series.count else { break }
            sum += abs(TrainingSampleTools.movePoints(from: series.close[index], to: series.close[futureIndex]))
            count += 1
        }
        return count > 0 ? sum / Double(count) : abs(fallback)
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
