import FXDataEngine
import Foundation

public struct RuleM1SyncPlugin: FXAIPlannedPlugin, FXAIPluginSyntheticSeriesSupport {
    public let manifest = PluginManifestV4(
        aiID: AIModelID.m1Sync.rawValue,
        aiName: "rule_m1sync",
        family: .ruleBased,
        referenceTier: .ruleBaseline,
        capabilityMask: [.selfTest, .onlineLearning, .multiHorizon],
        featureGroups: .all,
        minHorizonMinutes: 1,
        maxHorizonMinutes: 240,
        minSequenceBars: 1,
        maxSequenceBars: 1,
        requiresVolumeWhenAvailable: true
    )

    public let accelerationPlan = FXPluginAccelerationPlan(
        pluginName: "rule_m1sync",
        primaryBackends: [.swiftScalar],
        candidateBackends: [.swiftSIMD],
        usesVolumeWhenAvailable: true,
        notes: "M1 chain rule over a tiny rolling OHLCV window. Uses volume confirmation when available; keep scalar for parity, SIMD only helps future batched audit sweeps."
    )

    private var hitEMA = 0.55
    private var edgeEMA = 0.0
    private var steps = 0
    private var syntheticSeries: M1OHLCVSeries?
    private let m1SyncBars: Int

    public init(m1SyncBars: Int = 3) {
        self.m1SyncBars = min(max(m1SyncBars, 2), 12)
    }

    public mutating func reset() {
        hitEMA = 0.55
        edgeEMA = 0.0
        steps = 0
        syntheticSeries = nil
    }

    public mutating func setSyntheticSeries(_ series: M1OHLCVSeries) throws {
        guard series.count > 0 else {
            throw FXDataEngineError.invalidRequest("rule_m1sync synthetic series is empty")
        }
        syntheticSeries = series
    }

    public mutating func clearSyntheticSeries() {
        syntheticSeries = nil
    }

    public mutating func train(_ request: TrainRequestV4, hyperParameters: HyperParameters) throws {
        try request.validate()
        try PluginContractTools.validateCompatibility(manifest: manifest, context: request.context)
        let evaluation = evaluate(request.context)
        let observed = request.labelClass
        let hit: Double
        if evaluation.label == .skip {
            hit = observed == .skip ? 1.0 : 0.0
        } else {
            hit = evaluation.label == observed ? 1.0 : 0.0
        }

        hitEMA = 0.985 * hitEMA + 0.015 * hit
        edgeEMA = 0.980 * edgeEMA + 0.020 * evaluation.expectedMovePoints
        steps += 1
    }

    public func predict(_ request: PredictRequestV4, hyperParameters: HyperParameters) throws -> PredictionV4 {
        try request.validate()
        try PluginContractTools.validateCompatibility(manifest: manifest, context: request.context)

        var evaluation = evaluate(request.context)
        var reliability = fxClamp(hitEMA, 0.25, 0.95)
        let costPoints = max(request.context.priceCostPoints, 0.0)
        let minimumMove = request.context.minMovePoints > 0.0 ? request.context.minMovePoints : 0.10
        let moveScale = max(evaluation.expectedMovePoints, max(minimumMove, 0.10))
        let executionDrag = fxClamp(costPoints / max(moveScale, 0.25), 0.0, 1.5)
        let sessionPenalty = (request.context.sessionBucket == 0 ||
            request.context.sessionBucket == FXDataEngineConstants.pluginSessionBuckets - 1) ? 0.10 : 0.0
        let volumeBoost = 0.06 * evaluation.volumeConfirmation

        reliability = fxClamp(reliability * (1.0 - 0.22 * executionDrag - sessionPenalty) + volumeBoost, 0.10, 0.95)
        evaluation.strength = fxClamp(evaluation.strength * (1.0 - 0.28 * executionDrag - 0.50 * sessionPenalty) + 0.08 * evaluation.volumeConfirmation, 0.0, 1.0)
        evaluation.expectedMovePoints = max(0.0, evaluation.expectedMovePoints * (1.0 - 0.25 * executionDrag - 0.35 * sessionPenalty))
        if executionDrag >= 0.95 {
            evaluation.label = .skip
            evaluation.expectedMovePoints = 0.0
            evaluation.strength = 0.0
        }

        let classProbabilities: [Double]
        switch evaluation.label {
        case .buy:
            let buy = fxClamp(0.90 + 0.08 * evaluation.strength + 0.04 * (reliability - 0.50), 0.85, 0.995)
            classProbabilities = normalized([0.01, buy, max(0.02, 1.0 - buy - 0.01)])
        case .sell:
            let sell = fxClamp(0.90 + 0.08 * evaluation.strength + 0.04 * (reliability - 0.50), 0.85, 0.995)
            classProbabilities = normalized([sell, 0.01, max(0.02, 1.0 - sell - 0.01)])
        case .skip:
            classProbabilities = [0.02, 0.02, 0.96]
        }

        let sigma = max(0.10, 0.35 * evaluation.expectedMovePoints)
        let q25 = max(0.0, evaluation.expectedMovePoints - 0.55 * sigma)
        let q50 = max(q25, evaluation.expectedMovePoints)
        let q75 = max(q50, evaluation.expectedMovePoints + 0.55 * sigma)
        let directionalConfidence = max(classProbabilities[LabelClass.buy.rawValue], classProbabilities[LabelClass.sell.rawValue])

        return PredictionV4(
            classProbabilities: classProbabilities,
            moveMeanPoints: evaluation.expectedMovePoints,
            moveQ25Points: q25,
            moveQ50Points: q50,
            moveQ75Points: q75,
            mfeMeanPoints: evaluation.expectedMovePoints,
            maeMeanPoints: max(0.0, 0.35 * evaluation.expectedMovePoints),
            hitTimeFraction: 1.0,
            pathRisk: 1.0 - fxClamp(1.0 - classProbabilities[LabelClass.skip.rawValue], 0.0, 1.0),
            fillRisk: 0.0,
            confidence: directionalConfidence,
            reliability: reliability
        )
    }

    private func evaluate(_ context: PluginContextV4) -> M1SyncEvaluation {
        guard let series = syntheticSeries,
              series.count >= m1SyncBars + 1 else {
            return M1SyncEvaluation(label: .skip, expectedMovePoints: 0.0, strength: 0.0, volumeConfirmation: 0.0)
        }

        let closedIndex = findClosedIndex(series: series, sampleTimeUTC: context.sampleTimeUTC)
        guard closedIndex >= m1SyncBars - 1 else {
            return M1SyncEvaluation(label: .skip, expectedMovePoints: 0.0, strength: 0.0, volumeConfirmation: 0.0)
        }

        let start = closedIndex - m1SyncBars + 1
        let closes = (start...closedIndex).map { Double(series.close[$0]) }
        guard closes.allSatisfy({ $0 > 0.0 }) else {
            return M1SyncEvaluation(label: .skip, expectedMovePoints: 0.0, strength: 0.0, volumeConfirmation: 0.0)
        }

        let nowPrice: Double
        if closedIndex + 1 < series.count, series.open[closedIndex + 1] > 0 {
            nowPrice = Double(series.open[closedIndex + 1])
        } else {
            nowPrice = Double(series.close[closedIndex])
        }
        guard nowPrice > 0.0 else {
            return M1SyncEvaluation(label: .skip, expectedMovePoints: 0.0, strength: 0.0, volumeConfirmation: 0.0)
        }

        let point = 1.0
        let costPoints = max(context.priceCostPoints, 0.0)
        let minimumMove = context.minMovePoints > 0.0 ? context.minMovePoints : max(0.10, costPoints)
        let epsilon = max(0.10 * point, 0.02 * costPoints * point)
        var upChain = true
        var downChain = true
        var minStepPoints = Double.greatestFiniteMagnitude
        var previous = closes[0]

        for close in closes.dropFirst() {
            let step = close - previous
            if step <= epsilon { upChain = false }
            if step >= -epsilon { downChain = false }
            minStepPoints = min(minStepPoints, abs(step) / point)
            previous = close
        }

        let finalStep = nowPrice - closes[closes.count - 1]
        if finalStep <= epsilon { upChain = false }
        if finalStep >= -epsilon { downChain = false }
        let finalStepPoints = abs(finalStep) / point
        minStepPoints = min(minStepPoints, finalStepPoints)

        guard upChain || downChain else {
            return M1SyncEvaluation(label: .skip, expectedMovePoints: 0.0, strength: 0.0, volumeConfirmation: 0.0)
        }

        let totalPoints = abs(nowPrice - closes[0]) / point
        let edgePoints = totalPoints - costPoints
        let totalScore = sigmoid(edgePoints / max(minimumMove, 0.10))
        let stepScore = sigmoid((minStepPoints / max(minimumMove, 0.10)) - 0.15)
        let volumeConfirmation = series.hasVolume
            ? max(0.35, computeVolumeConfirmation(series: series, start: start, closedIndex: closedIndex))
            : 0.0
        let strength = fxClamp(0.60 * totalScore + 0.40 * stepScore + 0.04 * volumeConfirmation, 0.0, 1.0)
        return M1SyncEvaluation(
            label: upChain ? .buy : .sell,
            expectedMovePoints: max(totalPoints, 0.0),
            strength: strength,
            volumeConfirmation: volumeConfirmation
        )
    }

    private func computeVolumeConfirmation(series: M1OHLCVSeries, start: Int, closedIndex: Int) -> Double {
        guard start >= 0, closedIndex >= start, closedIndex < series.volume.count else {
            return 0.0
        }
        let window = series.volume[start...closedIndex].map(Double.init)
        let positive = window.filter { $0 > 0.0 }
        guard !positive.isEmpty else { return 0.0 }
        let average = positive.reduce(0.0, +) / Double(positive.count)
        guard average > 0.0 else { return 0.0 }
        let recent = Double(series.volume[closedIndex])
        let relative = recent / average
        return fxClamp((relative - 0.70) / 0.80, 0.0, 1.0)
    }

    private func findClosedIndex(series: M1OHLCVSeries, sampleTimeUTC: Int64) -> Int {
        if sampleTimeUTC <= 0 {
            return max(series.count - 2, 0)
        }
        var candidate = -1
        for index in 0..<series.count where series.utcTimestamps[index] <= sampleTimeUTC {
            candidate = index
        }
        if candidate < 0 {
            return max(series.count - 2, 0)
        }
        return min(candidate, max(series.count - 1, 0))
    }

    private func normalized(_ probabilities: [Double]) -> [Double] {
        let sum = probabilities.reduce(0.0, +)
        guard sum > 0.0 else { return [0.02, 0.02, 0.96] }
        return probabilities.map { $0 / sum }
    }

    private func sigmoid(_ value: Double) -> Double {
        1.0 / (1.0 + exp(-value))
    }
}

private struct M1SyncEvaluation {
    var label: LabelClass
    var expectedMovePoints: Double
    var strength: Double
    var volumeConfirmation: Double
}
