import Foundation

public struct PluginModelOutputV4: Codable, Hashable, Sendable {
    public var classProbabilities: [Double]
    public var moveMeanPoints: Double
    public var moveQ25Points: Double
    public var moveQ50Points: Double
    public var moveQ75Points: Double
    public var mfeMeanPoints: Double
    public var maeMeanPoints: Double
    public var hitTimeFraction: Double
    public var pathRisk: Double
    public var fillRisk: Double
    public var confidence: Double
    public var reliability: Double
    public var hasQuantiles: Bool
    public var hasConfidence: Bool
    public var hasPathQuality: Bool

    public init(
        classProbabilities: [Double] = [0.10, 0.10, 0.80],
        moveMeanPoints: Double = 0.0,
        moveQ25Points: Double = 0.0,
        moveQ50Points: Double = 0.0,
        moveQ75Points: Double = 0.0,
        mfeMeanPoints: Double = 0.0,
        maeMeanPoints: Double = 0.0,
        hitTimeFraction: Double = 1.0,
        pathRisk: Double = 0.0,
        fillRisk: Double = 0.0,
        confidence: Double = 0.0,
        reliability: Double = 0.0,
        hasQuantiles: Bool = false,
        hasConfidence: Bool = false,
        hasPathQuality: Bool = false
    ) {
        self.classProbabilities = classProbabilities
        self.moveMeanPoints = moveMeanPoints
        self.moveQ25Points = moveQ25Points
        self.moveQ50Points = moveQ50Points
        self.moveQ75Points = moveQ75Points
        self.mfeMeanPoints = mfeMeanPoints
        self.maeMeanPoints = maeMeanPoints
        self.hitTimeFraction = hitTimeFraction
        self.pathRisk = pathRisk
        self.fillRisk = fillRisk
        self.confidence = confidence
        self.reliability = reliability
        self.hasQuantiles = hasQuantiles
        self.hasConfidence = hasConfidence
        self.hasPathQuality = hasPathQuality
    }
}

public enum PluginContextRuntimeTools {
    public static func inputPriceCostPoints(_ x: [Double], explicitCostPoints: Double? = nil) -> Double {
        if let explicitCostPoints, explicitCostPoints.isFinite, explicitCostPoints >= 0.0 {
            return explicitCostPoints
        }
        guard x.count > 7 else { return 0.0 }
        return max(0.0, abs(fxSafeFinite(x[7])))
    }

    public static func normalizeClassLabel(rawLabel: Int, x: [Double], movePoints: Double, priceCostPoints: Double? = nil) -> LabelClass {
        if let label = LabelClass(rawValue: rawLabel) {
            return label
        }
        let cost = inputPriceCostPoints(x, explicitCostPoints: priceCostPoints)
        let edge = abs(fxSafeFinite(movePoints)) - cost
        let skipBand = 0.10 + 0.25 * max(cost, 0.0)
        if edge <= skipBand {
            return .skip
        }
        if rawLabel > 0 {
            return .buy
        }
        if rawLabel == 0 {
            return .sell
        }
        return movePoints >= 0.0 ? .buy : .sell
    }

    public static func normalizeClassDistribution(_ probabilities: [Double]) -> [Double] {
        guard probabilities.count >= 3 else { return [0.10, 0.10, 0.80] }
        var output = Array(probabilities.prefix(3)).map { fxClamp(fxSafeFinite($0), 0.0005, 0.9990) }
        let sum = output.reduce(0.0, +)
        guard sum.isFinite, sum > 0.0 else { return [0.10, 0.10, 0.80] }
        output = output.map { $0 / sum }
        return output
    }

    public static func contextHorizonBucket(horizonMinutes: Int) -> Int {
        let horizon = max(1, horizonMinutes)
        if horizon <= 1 { return 0 }
        if horizon <= 3 { return 1 }
        if horizon <= 5 { return 2 }
        if horizon <= 8 { return 3 }
        if horizon <= 13 { return 4 }
        if horizon <= 21 { return 5 }
        if horizon <= 34 { return 6 }
        return FXDataEngineConstants.pluginHorizonBuckets - 1
    }

    public static func effectiveWindowSize(_ window: [[Double]], declaredSize: Int? = nil) -> Int {
        let declared = declaredSize.map { min(max(0, $0), FXDataEngineConstants.maxSequenceBars) } ?? window.count
        return min(max(0, declared), window.count, FXDataEngineConstants.maxSequenceBars)
    }

    public static func windowValue(
        _ window: [[Double]],
        barIndex: Int,
        inputIndex: Int,
        declaredSize: Int? = nil
    ) -> Double {
        let size = effectiveWindowSize(window, declaredSize: declaredSize)
        guard barIndex >= 0, barIndex < size,
              inputIndex >= 0, inputIndex < FXDataEngineConstants.aiWeights,
              window[barIndex].count > inputIndex else {
            return 0.0
        }
        return fxSafeFinite(window[barIndex][inputIndex])
    }

    public static func windowSliceMean(
        _ window: [[Double]],
        inputIndex: Int,
        startBar: Int,
        count: Int,
        declaredSize: Int? = nil
    ) -> Double {
        let size = effectiveWindowSize(window, declaredSize: declaredSize)
        guard inputIndex >= 0,
              inputIndex < FXDataEngineConstants.aiWeights,
              size > 0,
              count > 0 else {
            return 0.0
        }
        let first = max(0, startBar)
        guard first < size else { return 0.0 }
        let last = min(size, first + count)
        guard last > first else { return 0.0 }

        var sum = 0.0
        var used = 0
        for index in first..<last {
            sum += windowValue(window, barIndex: index, inputIndex: inputIndex, declaredSize: size)
            used += 1
        }
        return used > 0 ? sum / Double(used) : 0.0
    }

    public static func currentWindowFeatureMean(_ window: [[Double]], featureIndex: Int, declaredSize: Int? = nil) -> Double {
        let inputIndex = featureIndex + 1
        let size = effectiveWindowSize(window, declaredSize: declaredSize)
        guard inputIndex >= 1, inputIndex < FXDataEngineConstants.aiWeights, size > 0 else {
            return 0.0
        }
        let full = windowSliceMean(window, inputIndex: inputIndex, startBar: 0, count: size, declaredSize: size)
        let halfCount = max(size / 2, 1)
        let quarterCount = max(size / 4, 1)
        let half = windowSliceMean(window, inputIndex: inputIndex, startBar: size - halfCount, count: halfCount, declaredSize: size)
        let quarter = windowSliceMean(window, inputIndex: inputIndex, startBar: size - quarterCount, count: quarterCount, declaredSize: size)
        return 0.40 * full + 0.35 * half + 0.25 * quarter
    }

    public static func currentWindowFeatureRecentMean(
        _ window: [[Double]],
        featureIndex: Int,
        recentBars: Int,
        declaredSize: Int? = nil
    ) -> Double {
        let inputIndex = featureIndex + 1
        let size = effectiveWindowSize(window, declaredSize: declaredSize)
        guard inputIndex >= 1, inputIndex < FXDataEngineConstants.aiWeights, size > 0 else {
            return 0.0
        }
        let count = min(max(1, recentBars), size)
        return windowSliceMean(window, inputIndex: inputIndex, startBar: 0, count: count, declaredSize: size)
    }

    public static func currentWindowFeatureStd(_ window: [[Double]], featureIndex: Int, declaredSize: Int? = nil) -> Double {
        let inputIndex = featureIndex + 1
        let size = effectiveWindowSize(window, declaredSize: declaredSize)
        guard inputIndex >= 1, inputIndex < FXDataEngineConstants.aiWeights, size > 1 else {
            return 0.0
        }
        let mean = windowSliceMean(window, inputIndex: inputIndex, startBar: 0, count: size, declaredSize: size)
        var accumulator = 0.0
        for index in 0..<size {
            let delta = windowValue(window, barIndex: index, inputIndex: inputIndex, declaredSize: size) - mean
            accumulator += delta * delta
        }
        return sqrt(accumulator / Double(max(size, 1)))
    }

    public static func currentWindowFeatureRange(
        _ window: [[Double]],
        featureIndex: Int,
        recentBars: Int = 0,
        declaredSize: Int? = nil
    ) -> Double {
        let inputIndex = featureIndex + 1
        let size = effectiveWindowSize(window, declaredSize: declaredSize)
        guard inputIndex >= 1, inputIndex < FXDataEngineConstants.aiWeights, size > 0 else {
            return 0.0
        }
        let count = recentBars <= 0 ? size : min(recentBars, size)
        var low = windowValue(window, barIndex: 0, inputIndex: inputIndex, declaredSize: size)
        var high = low
        for index in 0..<count {
            let value = windowValue(window, barIndex: index, inputIndex: inputIndex, declaredSize: size)
            low = min(low, value)
            high = max(high, value)
        }
        return high - low
    }

    public static func currentWindowFeatureSlope(_ window: [[Double]], featureIndex: Int, declaredSize: Int? = nil) -> Double {
        let inputIndex = featureIndex + 1
        let size = effectiveWindowSize(window, declaredSize: declaredSize)
        guard inputIndex >= 1, inputIndex < FXDataEngineConstants.aiWeights, size > 1 else {
            return 0.0
        }
        let first = windowValue(window, barIndex: 0, inputIndex: inputIndex, declaredSize: size)
        let last = windowValue(window, barIndex: size - 1, inputIndex: inputIndex, declaredSize: size)
        return (first - last) / Double(max(size - 1, 1))
    }

    public static func currentWindowFeatureRecentDelta(
        _ window: [[Double]],
        featureIndex: Int,
        recentBars: Int,
        declaredSize: Int? = nil
    ) -> Double {
        let inputIndex = featureIndex + 1
        let size = effectiveWindowSize(window, declaredSize: declaredSize)
        guard inputIndex >= 1, inputIndex < FXDataEngineConstants.aiWeights, size > 0 else {
            return 0.0
        }
        var count = recentBars
        if count <= 1 {
            count = max(size / 4, 2)
        }
        count = min(count, size)
        let lastIndex = max(count - 1, 0)
        return windowValue(window, barIndex: 0, inputIndex: inputIndex, declaredSize: size) -
            windowValue(window, barIndex: lastIndex, inputIndex: inputIndex, declaredSize: size)
    }

    public static func currentWindowFeatureEMAMean(
        _ window: [[Double]],
        featureIndex: Int,
        decay: Double = 0.72,
        declaredSize: Int? = nil
    ) -> Double {
        let inputIndex = featureIndex + 1
        let size = effectiveWindowSize(window, declaredSize: declaredSize)
        guard inputIndex >= 1, inputIndex < FXDataEngineConstants.aiWeights, size > 0 else {
            return 0.0
        }
        let alpha = fxClamp(decay, 0.05, 0.98)
        var weight = 1.0
        var weightSum = 0.0
        var sum = 0.0
        for index in 0..<size {
            sum += weight * windowValue(window, barIndex: index, inputIndex: inputIndex, declaredSize: size)
            weightSum += weight
            weight *= alpha
        }
        return weightSum > 0.0 ? sum / weightSum : 0.0
    }

    public static func fillPrediction(
        modelOutput: PluginModelOutputV4,
        calibratedMoveMeanPoints: Double,
        context: PluginContextV4
    ) -> PredictionV4 {
        let probabilities = normalizeClassDistribution(modelOutput.classProbabilities)
        let buyProbability = probabilities[LabelClass.buy.rawValue]
        let sellProbability = probabilities[LabelClass.sell.rawValue]
        let skipProbability = probabilities[LabelClass.skip.rawValue]
        let directionalConfidence = max(buyProbability, sellProbability)
        let uncertainty = fxClamp(1.0 - directionalConfidence + 0.50 * skipProbability, 0.10, 1.50)
        let meanMove = calibratedMoveMeanPoints.isFinite && calibratedMoveMeanPoints > 0.0 ? calibratedMoveMeanPoints : 0.0
        let rawMean = modelOutput.moveMeanPoints.isFinite && modelOutput.moveMeanPoints > 0.0 ? modelOutput.moveMeanPoints : 0.0
        let scale = rawMean > 1e-9 ? meanMove / rawMean : 1.0

        let moveQ25: Double
        let moveQ50: Double
        let moveQ75: Double
        if modelOutput.hasQuantiles, meanMove > 0.0 {
            moveQ25 = max(0.0, modelOutput.moveQ25Points * scale)
            moveQ50 = max(moveQ25, modelOutput.moveQ50Points * scale)
            moveQ75 = max(moveQ50, modelOutput.moveQ75Points * scale)
        } else if meanMove > 0.0 {
            moveQ25 = max(0.0, meanMove * max(0.25, 1.0 - 0.45 * uncertainty))
            moveQ50 = meanMove
            moveQ75 = max(moveQ50, meanMove * (1.0 + 0.45 * uncertainty))
        } else {
            moveQ25 = 0.0
            moveQ50 = 0.0
            moveQ75 = 0.0
        }

        let mfe: Double
        let mae: Double
        let hitTime: Double
        let pathRisk: Double
        let fillRisk: Double
        if modelOutput.hasPathQuality {
            mfe = max(0.0, modelOutput.mfeMeanPoints * scale)
            mae = max(0.0, modelOutput.maeMeanPoints * scale)
            hitTime = fxClamp(modelOutput.hitTimeFraction, 0.0, 1.0)
            pathRisk = fxClamp(modelOutput.pathRisk, 0.0, 1.0)
            fillRisk = fxClamp(modelOutput.fillRisk, 0.0, 1.0)
        } else {
            mfe = max(moveQ75, meanMove)
            mae = max(0.0, 0.35 * meanMove)
            hitTime = fxClamp(0.60 - 0.20 * directionalConfidence + 0.20 * skipProbability, 0.0, 1.0)
            pathRisk = fxClamp(0.40 * skipProbability + 0.35 * hitTime, 0.0, 1.0)
            fillRisk = fxClamp(
                (context.priceCostPoints + 0.25 * context.minMovePoints) /
                    max(meanMove + context.minMovePoints, 0.25),
                0.0,
                1.0
            )
        }

        return PredictionV4(
            classProbabilities: probabilities,
            moveMeanPoints: meanMove,
            moveQ25Points: moveQ25,
            moveQ50Points: moveQ50,
            moveQ75Points: moveQ75,
            mfeMeanPoints: mfe,
            maeMeanPoints: mae,
            hitTimeFraction: hitTime,
            pathRisk: pathRisk,
            fillRisk: fillRisk,
            confidence: fxClamp(modelOutput.hasConfidence ? modelOutput.confidence : directionalConfidence, 0.0, 1.0),
            reliability: fxClamp(modelOutput.hasConfidence ? modelOutput.reliability : 1.0 - 0.50 * skipProbability, 0.0, 1.0)
        )
    }
}
