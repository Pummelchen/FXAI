import Foundation

public struct SamplePathFlags: OptionSet, Codable, Sendable, Hashable {
    public let rawValue: Int

    public init(rawValue: Int) {
        self.rawValue = rawValue
    }

    public static let dualHit = SamplePathFlags(rawValue: 1)
    public static let killedEarly = SamplePathFlags(rawValue: 2)
    public static let spreadStress = SamplePathFlags(rawValue: 4)
    public static let slowHit = SamplePathFlags(rawValue: 8)
}

public struct TripleBarrierLabelResult: Codable, Hashable, Sendable {
    public var labelClass: LabelClass
    public var realizedMovePoints: Double
    public var mfePoints: Double
    public var maePoints: Double
    public var timeToHitFraction: Double
    public var pathFlags: SamplePathFlags

    public init(
        labelClass: LabelClass = .skip,
        realizedMovePoints: Double = 0.0,
        mfePoints: Double = 0.0,
        maePoints: Double = 0.0,
        timeToHitFraction: Double = 1.0,
        pathFlags: SamplePathFlags = []
    ) {
        self.labelClass = labelClass
        self.realizedMovePoints = fxSafeFinite(realizedMovePoints)
        self.mfePoints = max(0.0, fxSafeFinite(mfePoints))
        self.maePoints = max(0.0, fxSafeFinite(maePoints))
        self.timeToHitFraction = fxClamp(timeToHitFraction, 0.0, 1.0)
        self.pathFlags = pathFlags
    }
}

public struct PreparedTrainingSample: Sendable {
    public var valid: Bool
    public var labelClass: LabelClass
    public var regimeID: Int
    public var horizonMinutes: Int
    public var horizonSlot: Int
    public var movePoints: Double
    public var minMovePoints: Double
    public var costPoints: Double
    public var sampleWeight: Double
    public var qualityScore: Double
    public var mfePoints: Double
    public var maePoints: Double
    public var timeToHitFraction: Double
    public var pathFlags: SamplePathFlags
    public var pathRisk: Double
    public var fillRisk: Double
    public var maskedStepTarget: Double
    public var nextVolumeTarget: Double
    public var regimeShiftTarget: Double
    public var contextLeadTarget: Double
    public var pointValue: Double
    public var domainHash: Double
    public var sampleTimeUTC: Int64
    public var x: [Double]

    public init(
        valid: Bool = false,
        labelClass: LabelClass = .skip,
        regimeID: Int = 0,
        horizonMinutes: Int = 1,
        horizonSlot: Int = 0,
        movePoints: Double = 0.0,
        minMovePoints: Double = 0.0,
        costPoints: Double = 0.0,
        sampleWeight: Double = 1.0,
        qualityScore: Double = 1.0,
        mfePoints: Double = 0.0,
        maePoints: Double = 0.0,
        timeToHitFraction: Double = 1.0,
        pathFlags: SamplePathFlags = [],
        pathRisk: Double = 0.0,
        fillRisk: Double = 0.0,
        maskedStepTarget: Double = 0.0,
        nextVolumeTarget: Double = 0.0,
        regimeShiftTarget: Double = 0.0,
        contextLeadTarget: Double = 0.5,
        pointValue: Double = 1.0,
        domainHash: Double = 0.0,
        sampleTimeUTC: Int64 = 0,
        x: [Double] = Array(repeating: 0.0, count: FXDataEngineConstants.aiWeights)
    ) {
        self.valid = valid
        self.labelClass = labelClass
        self.regimeID = Int(fxClamp(Double(regimeID), 0.0, Double(FXDataEngineConstants.pluginRegimeBuckets - 1)))
        self.horizonMinutes = TrainingSampleTools.clampHorizon(horizonMinutes)
        self.horizonSlot = Int(fxClamp(Double(horizonSlot), 0.0, Double(RuntimeArtifactConstants.maxHorizons - 1)))
        self.movePoints = fxSafeFinite(movePoints)
        self.minMovePoints = max(0.0, fxSafeFinite(minMovePoints))
        self.costPoints = max(0.0, fxSafeFinite(costPoints))
        self.sampleWeight = fxClamp(sampleWeight, 0.0, 10.0)
        self.qualityScore = fxClamp(qualityScore, 0.0, 4.0)
        self.mfePoints = max(0.0, fxSafeFinite(mfePoints))
        self.maePoints = max(0.0, fxSafeFinite(maePoints))
        self.timeToHitFraction = fxClamp(timeToHitFraction, 0.0, 1.0)
        self.pathFlags = pathFlags
        self.pathRisk = fxClamp(pathRisk, 0.0, 1.0)
        self.fillRisk = fxClamp(fillRisk, 0.0, 1.0)
        self.maskedStepTarget = fxSafeFinite(maskedStepTarget)
        self.nextVolumeTarget = max(0.0, fxSafeFinite(nextVolumeTarget))
        self.regimeShiftTarget = fxClamp(regimeShiftTarget, 0.0, 1.0)
        self.contextLeadTarget = fxClamp(contextLeadTarget, 0.0, 1.0)
        self.pointValue = pointValue > 0 ? pointValue : 1.0
        self.domainHash = fxClamp(domainHash, 0.0, 1.0)
        self.sampleTimeUTC = sampleTimeUTC
        self.x = TrainingSampleTools.sanitizeModelInput(x)
    }
}

public struct PreparedTrainingPayload: Sendable {
    public let dataBundle: DataCoreBundle
    public let featureFrame: FeatureCoreFrame
    public let normalizationFrame: NormalizationCoreFrame
    public let payloadFrame: NormalizationPayloadFrame
    public let sample: PreparedTrainingSample
    public let context: PluginContextV4

    public var trainRequest: TrainRequestV4 {
        TrainRequestV4(
            valid: sample.valid,
            context: context,
            labelClass: sample.labelClass,
            movePoints: sample.movePoints,
            sampleWeight: sample.sampleWeight,
            mfePoints: sample.mfePoints,
            maePoints: sample.maePoints,
            timeToHitFraction: sample.timeToHitFraction,
            pathFlags: sample.pathFlags.rawValue,
            pathRisk: sample.pathRisk,
            fillRisk: sample.fillRisk,
            maskedStepTarget: sample.maskedStepTarget,
            nextVolumeTarget: sample.nextVolumeTarget,
            regimeShiftTarget: sample.regimeShiftTarget,
            contextLeadTarget: sample.contextLeadTarget,
            windowSize: payloadFrame.windowSize,
            x: payloadFrame.x,
            xWindow: payloadFrame.xWindow
        )
    }
}

public struct TrainingDatasetRequest: Sendable {
    public var startIndex: Int?
    public var endIndex: Int?
    public var stride: Int
    public var maxSamples: Int
    public var horizonMinutes: Int
    public var roundTripCostPoints: Double
    public var evThresholdPoints: Double
    public var normalizationMethod: FeatureNormalizationMethod
    public var tradeKillerMinutes: Int?

    public init(
        startIndex: Int? = nil,
        endIndex: Int? = nil,
        stride: Int = 1,
        maxSamples: Int = Int.max,
        horizonMinutes: Int,
        roundTripCostPoints: Double = 0.0,
        evThresholdPoints: Double = 0.0,
        normalizationMethod: FeatureNormalizationMethod = .existing,
        tradeKillerMinutes: Int? = nil
    ) {
        self.startIndex = startIndex
        self.endIndex = endIndex
        self.stride = max(1, stride)
        self.maxSamples = max(0, maxSamples)
        self.horizonMinutes = TrainingSampleTools.clampHorizon(horizonMinutes)
        self.roundTripCostPoints = max(0.0, fxSafeFinite(roundTripCostPoints))
        self.evThresholdPoints = max(0.0, fxSafeFinite(evThresholdPoints))
        self.normalizationMethod = normalizationMethod
        self.tradeKillerMinutes = tradeKillerMinutes
    }
}

public struct PreparedTrainingDataset: Sendable {
    public let symbol: String
    public let horizonMinutes: Int
    public let startIndex: Int
    public let endIndex: Int
    public let stride: Int
    public let payloads: [PreparedTrainingPayload]

    public var trainRequests: [TrainRequestV4] {
        payloads.map(\.trainRequest)
    }
}

public enum TrainingSampleTools {
    public static let defaultConfiguredHorizons = HorizonTools.defaultConfiguredHorizons

    public static func clampHorizon(_ horizonMinutes: Int) -> Int {
        HorizonTools.clampHorizon(horizonMinutes)
    }

    public static func horizonSlot(
        horizonMinutes: Int,
        configuredHorizons: [Int] = defaultConfiguredHorizons
    ) -> Int {
        HorizonTools.horizonSlot(horizonMinutes: horizonMinutes, configuredHorizons: configuredHorizons)
    }

    public static func movePoints(from entry: Int64, to exit: Int64) -> Double {
        Double(exit - entry)
    }

    public static func buildEVClassLabel(
        movePoints: Double,
        roundTripCostPoints: Double,
        evThresholdPoints: Double
    ) -> LabelClass {
        let evMin = max(0.0, evThresholdPoints)
        let cost = max(0.0, roundTripCostPoints)
        let buyEV = movePoints - cost
        let sellEV = -movePoints - cost
        if buyEV >= evMin, buyEV > sellEV { return .buy }
        if sellEV >= evMin, sellEV > buyEV { return .sell }
        return .skip
    }

    public static func buildTripleBarrierLabel(
        series: M1OHLCVSeries,
        index: Int,
        horizonMinutes: Int,
        roundTripCostPoints: Double,
        evThresholdPoints: Double,
        tradeKillerMinutes: Int? = nil
    ) -> TripleBarrierLabelResult {
        buildTripleBarrierLabel(
            series: series,
            index: index,
            horizonMinutes: horizonMinutes,
            roundTripCostPoints: roundTripCostPoints,
            evThresholdPoints: evThresholdPoints,
            maxFutureIndex: series.count - 1,
            tradeKillerMinutes: tradeKillerMinutes
        )
    }

    public static func buildTripleBarrierLabel(
        series: M1OHLCVSeries,
        index: Int,
        horizonMinutes: Int,
        roundTripCostPoints: Double,
        evThresholdPoints: Double,
        maxFutureIndex: Int,
        tradeKillerMinutes: Int? = nil
    ) -> TripleBarrierLabelResult {
        let horizon = clampHorizon(horizonMinutes)
        guard index >= 0, index < series.count, horizon >= 1 else {
            return TripleBarrierLabelResult()
        }
        let boundedFutureIndex = min(maxFutureIndex, series.count - 1)
        let maxStep = min(horizon, boundedFutureIndex - index)
        guard maxStep >= 1 else { return TripleBarrierLabelResult() }

        let entry = series.close[index]
        guard entry > 0 else { return TripleBarrierLabelResult() }

        let evMin = max(0.0, evThresholdPoints)
        let barrierBase = max(max(0.0, roundTripCostPoints) + evMin, 0.10)
        let drift: Double
        if index >= 5 {
            let momentum = movePoints(from: series.close[index - 5], to: series.close[index])
            drift = fxClamp(momentum / max(barrierBase, 0.10), -1.0, 1.0)
        } else {
            drift = 0.0
        }

        let rangeStart = max(0, index - 9)
        var rangeSum = 0.0
        var rangeCount = 0
        for row in rangeStart...index {
            rangeSum += max(0.0, Double(series.high[row] - series.low[row]))
            rangeCount += 1
        }
        let rangeAverage = rangeCount > 0 ? rangeSum / Double(rangeCount) : barrierBase
        let volatilityScale = fxClamp(rangeAverage / max(barrierBase, 0.10), 0.7, 1.8)
        let buyBarrier = max(barrierBase * volatilityScale * (1.0 - 0.10 * drift), 0.10)
        let sellBarrier = max(barrierBase * volatilityScale * (1.0 + 0.10 * drift), 0.10)

        var bestUp = 0.0
        var bestDown = 0.0
        for step in 1...maxStep {
            let futureIndex = index + step
            let upMove = movePoints(from: entry, to: series.high[futureIndex])
            let downMove = movePoints(from: entry, to: series.low[futureIndex])
            bestUp = max(bestUp, upMove)
            bestDown = max(bestDown, abs(downMove))

            let hitUp = upMove >= buyBarrier
            let hitDown = downMove <= -sellBarrier
            let hitFraction = fxClamp(Double(step) / Double(max(maxStep, 1)), 0.0, 1.0)

            if hitUp, !hitDown {
                var flags: SamplePathFlags = []
                if hitFraction > 0.75 { flags.insert(.slowHit) }
                return TripleBarrierLabelResult(
                    labelClass: .buy,
                    realizedMovePoints: max(upMove, buyBarrier),
                    mfePoints: bestUp,
                    maePoints: bestDown,
                    timeToHitFraction: hitFraction,
                    pathFlags: flags
                )
            }
            if hitDown, !hitUp {
                var flags: SamplePathFlags = []
                if hitFraction > 0.75 { flags.insert(.slowHit) }
                return TripleBarrierLabelResult(
                    labelClass: .sell,
                    realizedMovePoints: min(downMove, -sellBarrier),
                    mfePoints: bestDown,
                    maePoints: bestUp,
                    timeToHitFraction: hitFraction,
                    pathFlags: flags
                )
            }
            if hitUp, hitDown {
                let closeMove = movePoints(from: entry, to: series.close[futureIndex])
                let upExcess = upMove - buyBarrier
                let downExcess = -downMove - sellBarrier
                if closeMove > 0.0, upExcess >= downExcess {
                    return TripleBarrierLabelResult(
                        labelClass: .buy,
                        realizedMovePoints: max(closeMove, buyBarrier),
                        mfePoints: bestUp,
                        maePoints: bestDown,
                        timeToHitFraction: hitFraction,
                        pathFlags: [.dualHit]
                    )
                }
                if closeMove < 0.0, downExcess >= upExcess {
                    return TripleBarrierLabelResult(
                        labelClass: .sell,
                        realizedMovePoints: min(closeMove, -sellBarrier),
                        mfePoints: bestDown,
                        maePoints: bestUp,
                        timeToHitFraction: hitFraction,
                        pathFlags: [.dualHit]
                    )
                }
                let excursions = assignExcursions(realizedMovePoints: closeMove, bestUpPoints: bestUp, bestDownPoints: bestDown)
                return TripleBarrierLabelResult(
                    labelClass: buildEVClassLabel(
                        movePoints: closeMove,
                        roundTripCostPoints: roundTripCostPoints,
                        evThresholdPoints: evThresholdPoints
                    ),
                    realizedMovePoints: closeMove,
                    mfePoints: excursions.mfe,
                    maePoints: excursions.mae,
                    timeToHitFraction: hitFraction,
                    pathFlags: [.dualHit]
                )
            }
        }

        let terminalIndex = index + maxStep
        let realizedMove = movePoints(from: entry, to: series.close[terminalIndex])
        let excursions = assignExcursions(realizedMovePoints: realizedMove, bestUpPoints: bestUp, bestDownPoints: bestDown)
        var flags: SamplePathFlags = []
        if let tradeKillerMinutes, tradeKillerMinutes > 0, horizon > tradeKillerMinutes {
            flags.insert(.killedEarly)
        }
        return TripleBarrierLabelResult(
            labelClass: buildEVClassLabel(
                movePoints: realizedMove,
                roundTripCostPoints: roundTripCostPoints,
                evThresholdPoints: evThresholdPoints
            ),
            realizedMovePoints: realizedMove,
            mfePoints: excursions.mfe,
            maePoints: excursions.mae,
            timeToHitFraction: 1.0,
            pathFlags: flags
        )
    }

    public static func pathRisk(
        mfePoints: Double,
        maePoints: Double,
        minMovePoints: Double,
        timeToHitFraction: Double,
        pathFlags: SamplePathFlags
    ) -> Double {
        let mfe = max(abs(mfePoints), max(minMovePoints, 0.10))
        let adverseRatio = fxClamp(abs(maePoints) / mfe, 0.0, 3.0)
        var risk = 0.45 * adverseRatio + 0.30 * fxClamp(timeToHitFraction, 0.0, 1.0)
        if pathFlags.contains(.dualHit) { risk += 0.15 }
        if pathFlags.contains(.killedEarly) { risk += 0.10 }
        if pathFlags.contains(.slowHit) { risk += 0.08 }
        return fxClamp(risk, 0.0, 1.0)
    }

    public static func fillRisk(spreadStressPoints: Double, minMovePoints: Double, costPoints: Double) -> Double {
        let denominator = max(minMovePoints + max(costPoints, 0.0), 0.25)
        return fxClamp(abs(spreadStressPoints) / denominator, 0.0, 1.0)
    }

    public static func staticRegimeID(series: M1OHLCVSeries, index: Int) -> Int {
        let timestamp = index >= 0 && index < series.count ? series.utcTimestamps[index] : 0
        let session = sessionGroup(timestampUTC: timestamp)
        let volatilityRef = max(rollingAbsMovePoints(series: series, index: index, window: 64), 1e-6)
        var volatilityProxy = rollingMoveStdPoints(series: series, index: index, window: 10)
        if volatilityProxy < 1e-6 {
            volatilityProxy = rollingAbsMovePoints(series: series, index: index, window: 10)
        }
        let volatilityHigh = abs(volatilityProxy) > (1.15 * volatilityRef + 0.02)
        let regime = session * 4 + (volatilityHigh ? 2 : 0)
        return Int(fxClamp(Double(regime), 0.0, Double(FXDataEngineConstants.pluginRegimeBuckets - 1)))
    }

    public static func sanitizeModelInput(_ x: [Double]) -> [Double] {
        var output = Array(repeating: 0.0, count: FXDataEngineConstants.aiWeights)
        output[0] = 1.0
        let count = min(x.count, FXDataEngineConstants.aiWeights)
        for index in 1..<count {
            output[index] = fxSafeFinite(x[index])
        }
        return output
    }

    static func assignExcursions(realizedMovePoints: Double, bestUpPoints: Double, bestDownPoints: Double) -> (mfe: Double, mae: Double) {
        if realizedMovePoints > 0.0 {
            return (bestUpPoints, bestDownPoints)
        }
        if realizedMovePoints < 0.0 {
            return (bestDownPoints, bestUpPoints)
        }
        let flat = max(bestUpPoints, bestDownPoints)
        return (flat, flat)
    }

    static func rollingAbsMovePoints(series: M1OHLCVSeries, index: Int, window: Int) -> Double {
        guard index > 0, index < series.count, window > 0 else { return 0.0 }
        let start = max(1, index - window + 1)
        var sum = 0.0
        var count = 0
        for row in start...index {
            sum += abs(movePoints(from: series.close[row - 1], to: series.close[row]))
            count += 1
        }
        return count > 0 ? sum / Double(count) : 0.0
    }

    static func rollingMoveStdPoints(series: M1OHLCVSeries, index: Int, window: Int) -> Double {
        guard index > 0, index < series.count, window > 0 else { return 0.0 }
        let start = max(1, index - window + 1)
        var moves: [Double] = []
        moves.reserveCapacity(index - start + 1)
        for row in start...index {
            moves.append(movePoints(from: series.close[row - 1], to: series.close[row]))
        }
        return moves.standardDeviation
    }

    static func sessionGroup(timestampUTC: Int64) -> Int {
        HorizonTools.sessionGroup(timestampUTC: timestampUTC)
    }
}

private extension Array where Element == Double {
    var standardDeviation: Double {
        guard count > 1 else { return 0.0 }
        let mean = reduce(0.0, +) / Double(count)
        let variance = reduce(0.0) { $0 + ($1 - mean) * ($1 - mean) } / Double(count - 1)
        return sqrt(Swift.max(0.0, variance))
    }
}
