import FXDataEngine
import Foundation

public struct MixMoeConformalCPUModel: Sendable {
    private static let expertCount = 4
    private static let regimeCount = 11
    private static let featureCount = 32
    private static let weightCount = featureCount + 1
    private static let scoreCount = 128
    private static let bucketCount = 12
    private static let bucketDepth = 64
    private static let calibrationFeatureCount = 5
    private static let classCount = 3

    private var steps: Int
    private var calibrationHead: Int
    private var calibrationFilled: Int
    private var router: [[Double]]
    private var gateWeights: [[Double]]
    private var directionWeights: [[Double]]
    private var moveWeights: [[Double]]
    private var scores: [Double]
    private var bucketScores: [[Double]]
    private var bucketHeads: [Int]
    private var bucketFilled: [Int]
    private var usageEMA: [Double]
    private var calibrationWeights: [[Double]]
    private var qualityBank: PluginQualityBank

    public init() {
        self.steps = 0
        self.calibrationHead = 0
        self.calibrationFilled = 0
        self.router = Array(
            repeating: Array(repeating: 0.0, count: Self.regimeCount),
            count: Self.expertCount
        )
        self.gateWeights = Array(
            repeating: Array(repeating: 0.0, count: Self.weightCount),
            count: Self.expertCount
        )
        self.directionWeights = Array(
            repeating: Array(repeating: 0.0, count: Self.weightCount),
            count: Self.expertCount
        )
        self.moveWeights = Array(
            repeating: Array(repeating: 0.0, count: Self.weightCount),
            count: Self.expertCount
        )
        self.scores = Array(repeating: 0.40, count: Self.scoreCount)
        self.bucketScores = Array(
            repeating: Array(repeating: 0.40, count: Self.bucketDepth),
            count: Self.bucketCount
        )
        self.bucketHeads = Array(repeating: 0, count: Self.bucketCount)
        self.bucketFilled = Array(repeating: 0, count: Self.bucketCount)
        self.usageEMA = Array(repeating: 1.0 / Double(Self.expertCount), count: Self.expertCount)
        self.calibrationWeights = Array(
            repeating: Array(repeating: 0.0, count: Self.calibrationFeatureCount),
            count: Self.classCount
        )
        self.qualityBank = PluginQualityBank()
        seedRouter()
    }

    public mutating func reset() {
        self = MixMoeConformalCPUModel()
    }

    public mutating func train(_ request: TrainRequestV4, hyperParameters: HyperParameters) {
        let x = Self.preparedFeatures(request.x, dataHasVolume: request.context.dataHasVolume)
        let label = PluginContextRuntimeTools.normalizeClassLabel(
            rawLabel: request.labelClass.rawValue,
            x: x,
            movePoints: request.movePoints,
            priceCostPoints: request.context.priceCostPoints
        )
        let regime = buildRegime(x)
        let features = buildFeatures(x)
        let gates = routerSoftmax(regime)
        let baseLearningRate = fxClamp(hyperParameters.learningRate, 0.0001, 0.0300)
        let sampleWeight = fxClamp(
            request.sampleWeight *
                PluginSupportTools.moveSampleWeight(
                    x: x,
                    movePoints: request.movePoints,
                    priceCostPoints: request.context.priceCostPoints,
                    minMovePoints: request.context.minMovePoints,
                    qualityTargets: PluginQualityTargets(request: request)
                ),
            0.20,
            4.00
        )
        let learningRate = fxClamp(baseLearningRate * min(sampleWeight, 2.25), 0.0001, 0.0450)
        let l2 = fxClamp(hyperParameters.l2, 0.0, 0.10)
        let targetTrade = label == .skip ? 0.0 : 1.0
        let targetUp = label == .buy ? 1.0 : 0.0
        let cost = max(0.0, request.context.priceCostPoints)
        let targetMove = max(0.0, abs(request.movePoints) - cost)

        var routerLoss = Array(repeating: 0.0, count: Self.expertCount)
        var routerTotal = 0.0
        for expert in 0..<Self.expertCount {
            let gate = gates[expert]
            let tradeProbability = Self.clampProbability(PluginSupportTools.sigmoid(dotHead(gateWeights[expert], features)))
            let upProbability = Self.clampProbability(PluginSupportTools.sigmoid(dotHead(directionWeights[expert], features)))
            let moveEstimate = dotHead(moveWeights[expert], features)
            let tradeError = (targetTrade - tradeProbability) * gate
            let upError = (targetUp - upProbability) * gate
            let moveError = (targetMove - moveEstimate) * gate
            routerLoss[expert] = abs(targetTrade - tradeProbability) +
                (label == .skip ? 0.0 : abs(targetUp - upProbability)) +
                0.10 * abs(moveError)
            routerTotal += 1.0 / max(routerLoss[expert], 0.05)

            gateWeights[expert][0] += learningRate * (tradeError - l2 * gateWeights[expert][0])
            directionWeights[expert][0] += learningRate * (upError - l2 * directionWeights[expert][0])
            moveWeights[expert][0] += learningRate * 0.20 * (moveError - l2 * moveWeights[expert][0])
            for index in 0..<Self.featureCount {
                let value = features[index]
                gateWeights[expert][index + 1] = Self.clipWeight(
                    gateWeights[expert][index + 1] + learningRate * (tradeError * value - l2 * gateWeights[expert][index + 1]),
                    limit: 6.0
                )
                if label != .skip {
                    directionWeights[expert][index + 1] = Self.clipWeight(
                        directionWeights[expert][index + 1] + learningRate * (upError * value - l2 * directionWeights[expert][index + 1]),
                        limit: 6.0
                    )
                }
                moveWeights[expert][index + 1] = Self.clipWeight(
                    moveWeights[expert][index + 1] + learningRate * 0.20 * (moveError * value - l2 * moveWeights[expert][index + 1]),
                    limit: 12.0
                )
            }
        }

        for expert in 0..<Self.expertCount {
            let reward = (1.0 / max(routerLoss[expert], 0.05)) / max(routerTotal, 1.0e-6)
            usageEMA[expert] = 0.985 * usageEMA[expert] + 0.015 * gates[expert]
            for index in 0..<Self.regimeCount {
                router[expert][index] = Self.clipWeight(
                    router[expert][index] + learningRate * 0.10 * ((reward - gates[expert]) * regime[index] - 0.002 * router[expert][index]),
                    limit: 3.0
                )
            }
        }

        let core = predictCore(x: x, context: request.context)
        let trueIndex = label.rawValue
        let trueProbability = trueIndex < core.calibratedProbabilities.count ?
            core.calibratedProbabilities[trueIndex] :
            core.calibratedProbabilities[LabelClass.skip.rawValue]
        let score = 1.0 - fxClamp(trueProbability, 0.0005, 0.9990)
        storeCalibrationScore(score)
        storeBucketScore(score, bucket: bucketIndex(x: x, context: request.context))
        updateCalibrator(
            label: label,
            baseProbabilities: core.rawProbabilities,
            expectedMovePoints: core.expectedMove,
            context: request.context,
            learningRate: baseLearningRate * sampleWeight
        )
        qualityBank.update(request: request, sampleWeight: sampleWeight)
        steps += 1
    }

    public func predict(_ request: PredictRequestV4, hyperParameters _: HyperParameters) -> PredictionV4 {
        let x = Self.preparedFeatures(request.x, dataHasVolume: request.context.dataHasVolume)
        let window = Self.preparedWindow(request.xWindow, dataHasVolume: request.context.dataHasVolume)
        let core = predictCore(x: x, context: request.context)
        let probabilities = core.calibratedProbabilities
        let expectedMove = max(0.0, core.expectedMove)
        let sigma = max(0.10, 0.30 * expectedMove + 0.25 * (steps > 0 ? 1.0 : 0.0))
        let confidence = fxClamp(
            max(probabilities[LabelClass.buy.rawValue], probabilities[LabelClass.sell.rawValue]),
            0.0,
            1.0
        )
        let reliability = fxClamp(
            0.45 +
                0.25 * min(Double(steps) / 64.0, 1.0) +
                0.20 * (1.0 - probabilities[LabelClass.skip.rawValue]),
            0.0,
            1.0
        )
        let q25 = max(0.0, expectedMove - 0.55 * sigma)
        let q50 = max(q25, expectedMove)
        let q75 = max(q50, expectedMove + 0.55 * sigma)
        let baseOutput = PluginModelOutputV4(
            classProbabilities: probabilities,
            moveMeanPoints: expectedMove,
            moveQ25Points: q25,
            moveQ50Points: q50,
            moveQ75Points: q75,
            mfeMeanPoints: max(q75, expectedMove),
            maeMeanPoints: max(0.0, 0.35 * expectedMove),
            hitTimeFraction: fxClamp(0.64 - 0.24 * confidence + 0.15 * probabilities[LabelClass.skip.rawValue], 0.0, 1.0),
            pathRisk: fxClamp(0.34 * probabilities[LabelClass.skip.rawValue] + 0.22 * (1.0 - reliability), 0.0, 1.0),
            fillRisk: fxClamp(request.context.priceCostPoints / max(expectedMove + request.context.minMovePoints, 0.25), 0.0, 1.0),
            confidence: confidence,
            reliability: reliability,
            hasQuantiles: true,
            hasConfidence: true,
            hasPathQuality: true
        )
        let output = PluginPathQualityTools.populatedOutput(
            baseOutput,
            x: x,
            window: window,
            context: request.context,
            family: .mixture,
            activityGate: 1.0 - probabilities[LabelClass.skip.rawValue],
            structuralQuality: reliability,
            qualityPriors: qualityBank.priors(context: request.context),
            declaredWindowSize: request.windowSize
        )
        return PluginContextRuntimeTools.fillPrediction(
            modelOutput: output,
            calibratedMoveMeanPoints: expectedMove,
            context: request.context
        )
    }

    private mutating func seedRouter() {
        for expert in 0..<Self.expertCount {
            let index = min(expert + 1, Self.regimeCount - 1)
            router[expert][index] = 0.10
        }
    }

    private func predictCore(
        x: [Double],
        context: PluginContextV4
    ) -> (rawProbabilities: [Double], calibratedProbabilities: [Double], expectedMove: Double) {
        let regime = buildRegime(x)
        let features = buildFeatures(x)
        let gates = routerSoftmax(regime)
        var tradeProbability = 0.0
        var upProbability = 0.0
        var expectedMove = 0.0
        for expert in 0..<Self.expertCount {
            let gate = gates[expert]
            tradeProbability += gate * Self.clampProbability(PluginSupportTools.sigmoid(dotHead(gateWeights[expert], features)))
            upProbability += gate * Self.clampProbability(PluginSupportTools.sigmoid(dotHead(directionWeights[expert], features)))
            expectedMove += gate * abs(dotHead(moveWeights[expert], features))
        }
        expectedMove = max(0.0, expectedMove)

        let conformalCutoff = quantile90(bucket: bucketIndex(x: x, context: context))
        var buy = tradeProbability * upProbability
        var sell = tradeProbability * (1.0 - upProbability)
        var skip = 1.0 - tradeProbability
        let allowBuy = (1.0 - buy) <= conformalCutoff
        let allowSell = (1.0 - sell) <= conformalCutoff
        if allowBuy == allowSell {
            skip = max(skip, 0.55)
            buy *= 0.50
            sell *= 0.50
        }

        let raw = PluginContextRuntimeTools.normalizeClassDistribution([
            Self.clampProbability(sell),
            Self.clampProbability(buy),
            Self.clampProbability(skip)
        ])
        let calibrated = applyCalibrator(
            baseProbabilities: raw,
            expectedMovePoints: expectedMove,
            context: context
        )
        return (raw, calibrated, expectedMove)
    }

    private func buildRegime(_ x: [Double]) -> [Double] {
        let r1 = safeX(x, 0)
        let r5 = safeX(x, 1)
        let r15 = safeX(x, 2)
        let r60 = safeX(x, 3)
        let volatility = abs(safeX(x, 4))
        return [
            1.0,
            fxClamp(r1, -10.0, 10.0),
            fxClamp(r5, -10.0, 10.0),
            fxClamp(r15, -10.0, 10.0),
            fxClamp(r60, -10.0, 10.0),
            fxClamp(volatility, 0.0, 10.0),
            fxClamp(r1 - r5, -10.0, 10.0),
            fxClamp(r5 - r15, -10.0, 10.0),
            fxClamp((r1 + r5 + r15) / max(volatility, 1.0e-6), -10.0, 10.0),
            fxClamp(safeX(x, 5), -10.0, 10.0),
            fxClamp(safeX(x, 6), -10.0, 10.0)
        ]
    }

    private func buildFeatures(_ x: [Double]) -> [Double] {
        var features = Array(repeating: 0.0, count: Self.featureCount)
        let count = min(x.count, Self.featureCount)
        for index in 0..<count {
            features[index] = fxClamp(fxSafeFinite(x[index]), -10.0, 10.0)
        }
        return features
    }

    private func routerSoftmax(_ regime: [Double]) -> [Double] {
        var logits = Array(repeating: 0.0, count: Self.expertCount)
        var maximum = -Double.greatestFiniteMagnitude
        for expert in 0..<Self.expertCount {
            var value = Self.dot(router[expert], regime)
            value -= 0.35 * (usageEMA[expert] - 1.0 / Double(Self.expertCount))
            value = PluginSupportTools.clipSymmetric(value, limit: 30.0)
            logits[expert] = value
            maximum = max(maximum, value)
        }
        var sum = 0.0
        var values = Array(repeating: 0.0, count: Self.expertCount)
        for expert in 0..<Self.expertCount {
            let value = exp(fxClamp(logits[expert] - maximum, -30.0, 30.0))
            values[expert] = value
            sum += value
        }
        guard sum > 0.0 else {
            return Array(repeating: 1.0 / Double(Self.expertCount), count: Self.expertCount)
        }
        return values.map { $0 / sum }
    }

    private func dotHead(_ weights: [Double], _ features: [Double]) -> Double {
        var value = weights[0]
        let count = min(features.count, weights.count - 1)
        for index in 0..<count {
            value += weights[index + 1] * features[index]
        }
        return PluginSupportTools.clipSymmetric(value, limit: 40.0)
    }

    private func quantile90(bucket: Int) -> Double {
        let safeBucket = min(max(bucket, 0), Self.bucketCount - 1)
        let count = min(max(bucketFilled[safeBucket], 0), Self.bucketDepth)
        guard count > 8 else { return 0.40 }
        var values = Array(bucketScores[safeBucket].prefix(count))
        values.sort()
        let index = min(max(Int(floor(0.90 * Double(count - 1))), 0), count - 1)
        return values[index]
    }

    private func applyCalibrator(
        baseProbabilities: [Double],
        expectedMovePoints: Double,
        context: PluginContextV4
    ) -> [Double] {
        let features = calibrationFeatures(
            baseProbabilities: baseProbabilities,
            expectedMovePoints: expectedMovePoints,
            context: context
        )
        var logits = Array(repeating: 0.0, count: Self.classCount)
        for classIndex in 0..<Self.classCount {
            var value = log(max(baseProbabilities[classIndex], 1.0e-6))
            for featureIndex in 0..<Self.calibrationFeatureCount {
                value += calibrationWeights[classIndex][featureIndex] * features[featureIndex]
            }
            logits[classIndex] = value
        }
        return Self.softmax3(logits)
    }

    private mutating func updateCalibrator(
        label: LabelClass,
        baseProbabilities: [Double],
        expectedMovePoints: Double,
        context: PluginContextV4,
        learningRate: Double
    ) {
        let features = calibrationFeatures(
            baseProbabilities: baseProbabilities,
            expectedMovePoints: expectedMovePoints,
            context: context
        )
        let probabilities = applyCalibrator(
            baseProbabilities: baseProbabilities,
            expectedMovePoints: expectedMovePoints,
            context: context
        )
        let lr = fxClamp(0.25 * learningRate, 0.0002, 0.0200)
        for classIndex in 0..<Self.classCount {
            let target = classIndex == label.rawValue ? 1.0 : 0.0
            let error = target - probabilities[classIndex]
            for featureIndex in 0..<Self.calibrationFeatureCount {
                calibrationWeights[classIndex][featureIndex] = Self.clipWeight(
                    calibrationWeights[classIndex][featureIndex] +
                        lr * (error * features[featureIndex] - 0.002 * calibrationWeights[classIndex][featureIndex]),
                    limit: 6.0
                )
            }
        }
    }

    private func calibrationFeatures(
        baseProbabilities: [Double],
        expectedMovePoints: Double,
        context: PluginContextV4
    ) -> [Double] {
        let minimumMove = max(context.minMovePoints, 0.10)
        let priceCost = max(context.priceCostPoints, 0.0)
        return [
            1.0,
            fxClamp(baseProbabilities[LabelClass.buy.rawValue] - baseProbabilities[LabelClass.sell.rawValue], -1.0, 1.0),
            fxClamp(baseProbabilities[LabelClass.skip.rawValue], 0.0, 1.0),
            fxClamp(expectedMovePoints / minimumMove, 0.0, 12.0),
            fxClamp(priceCost / minimumMove, 0.0, 4.0)
        ]
    }

    private mutating func storeCalibrationScore(_ score: Double) {
        scores[calibrationHead] = fxClamp(fxSafeFinite(score, fallback: 0.40), 0.0, 1.0)
        calibrationHead = (calibrationHead + 1) % Self.scoreCount
        calibrationFilled = min(calibrationFilled + 1, Self.scoreCount)
    }

    private mutating func storeBucketScore(_ score: Double, bucket: Int) {
        let safeBucket = min(max(bucket, 0), Self.bucketCount - 1)
        let slot = bucketHeads[safeBucket]
        bucketScores[safeBucket][slot] = fxClamp(fxSafeFinite(score, fallback: 0.40), 0.0, 1.0)
        bucketHeads[safeBucket] = (slot + 1) % Self.bucketDepth
        bucketFilled[safeBucket] = min(bucketFilled[safeBucket] + 1, Self.bucketDepth)
    }

    private func bucketIndex(x: [Double], context: PluginContextV4) -> Int {
        let session = Self.sessionBucket(context)
        let volatility = abs(safeX(x, 4))
        let regime: Int
        if volatility < 0.75 {
            regime = 0
        } else if volatility < 1.75 {
            regime = 1
        } else {
            regime = 2
        }
        return min(max(3 * session + regime, 0), Self.bucketCount - 1)
    }

    private static func sessionBucket(_ context: PluginContextV4) -> Int {
        min(max(context.sessionBucket, 0), 3)
    }

    private func safeX(_ x: [Double], _ index: Int) -> Double {
        guard index >= 0, index < x.count else { return 0.0 }
        return fxSafeFinite(x[index])
    }

    private static func preparedFeatures(_ x: [Double], dataHasVolume: Bool) -> [Double] {
        var output = Array(repeating: 0.0, count: FXDataEngineConstants.aiWeights)
        for index in 0..<FXDataEngineConstants.aiWeights {
            output[index] = fxClamp(index < x.count ? fxSafeFinite(x[index]) : 0.0, -50.0, 50.0)
        }
        if !dataHasVolume {
            zeroVolumeFeatures(&output)
        }
        return output
    }

    private static func preparedWindow(_ window: [[Double]], dataHasVolume: Bool) -> [[Double]] {
        window.map { preparedFeatures($0, dataHasVolume: dataHasVolume) }
    }

    private static func zeroVolumeFeatures(_ features: inout [Double]) {
        for index in [6, 68, 69, 70, 71, 74, 75, 76, 77, 78, 80, 81, 82, 83] where index < features.count {
            features[index] = 0.0
        }
    }

    private static func clampProbability(_ probability: Double) -> Double {
        fxClamp(fxSafeFinite(probability, fallback: 0.5), 0.001, 0.999)
    }

    private static func clipWeight(_ weight: Double, limit: Double) -> Double {
        PluginSupportTools.clipSymmetric(fxSafeFinite(weight), limit: limit)
    }

    private static func dot(_ lhs: [Double], _ rhs: [Double]) -> Double {
        var value = 0.0
        let count = min(lhs.count, rhs.count)
        for index in 0..<count {
            value += lhs[index] * rhs[index]
        }
        return value
    }

    private static func softmax3(_ logits: [Double]) -> [Double] {
        let safe = [
            fxSafeFinite(logits[safe: 0] ?? 0.0),
            fxSafeFinite(logits[safe: 1] ?? 0.0),
            fxSafeFinite(logits[safe: 2] ?? 0.0)
        ]
        let maximum = safe.max() ?? 0.0
        let expValues = safe.map { exp(fxClamp($0 - maximum, -30.0, 30.0)) }
        let sum = expValues.reduce(0.0, +)
        guard sum > 0.0 else {
            return [0.10, 0.10, 0.80]
        }
        return expValues.map { $0 / sum }
    }
}

private extension Array {
    subscript(safe index: Int) -> Element? {
        indices.contains(index) ? self[index] : nil
    }
}
