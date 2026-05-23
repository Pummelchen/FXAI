import FXDataEngine
import Foundation

public struct LinElasticLogitCPUModel: Sendable {
    private static let classCount = 3
    private static let stateDimension = 16
    private static let aiID = AIModelID.linElasticLogit.rawValue

    private var steps: Int
    private var weights: [[Double]]
    private var velocity: [[Double]]
    private var moveEMA: Double
    private var moveReady: Bool
    private var classMass: [Double]
    private var policyOld: [Double]

    public init() {
        self.steps = 0
        self.weights = Array(repeating: Array(repeating: 0.0, count: Self.stateDimension), count: Self.classCount)
        self.velocity = Array(repeating: Array(repeating: 0.0, count: Self.stateDimension), count: Self.classCount)
        self.moveEMA = 0.0
        self.moveReady = false
        self.classMass = Array(repeating: 1.0, count: Self.classCount)
        self.policyOld = Array(repeating: 1.0 / 3.0, count: Self.classCount)
        for classIndex in 0..<Self.classCount {
            for stateIndex in 0..<Self.stateDimension {
                self.weights[classIndex][stateIndex] = Self.seedWeight(Self.aiID, classIndex + 1, stateIndex + 1)
            }
        }
    }

    public mutating func reset() {
        self = LinElasticLogitCPUModel()
    }

    public mutating func train(_ request: TrainRequestV4, hyperParameters: HyperParameters) {
        let x = Self.preparedFeatures(request.x, dataHasVolume: request.context.dataHasVolume)
        let window = Self.preparedWindow(request.xWindow, dataHasVolume: request.context.dataHasVolume)
        var label = PluginContextRuntimeTools.normalizeClassLabel(
            rawLabel: request.labelClass.rawValue,
            x: x,
            movePoints: request.movePoints,
            priceCostPoints: request.context.priceCostPoints
        )
        if !(0..<Self.classCount).contains(label.rawValue) {
            label = request.movePoints > 0.0 ? .buy : (request.movePoints < 0.0 ? .sell : .skip)
        }
        let state = buildState(
            x: x,
            window: window,
            windowSize: request.windowSize,
            horizonMinutes: request.context.horizonMinutes,
            dataHasVolume: request.context.dataHasVolume
        )
        let signedMove: Double
        switch label {
        case .buy:
            signedMove = abs(request.movePoints)
        case .sell:
            signedMove = -abs(request.movePoints)
        case .skip:
            signedMove = 0.0
        }
        let cost = PluginContextRuntimeTools.inputPriceCostPoints(x, explicitCostPoints: request.context.priceCostPoints)
        let sampleWeight = fxClamp(
            request.sampleWeight * PluginSupportTools.moveEdgeWeight(movePoints: request.movePoints, priceCostPoints: cost),
            0.15,
            6.0
        )
        updateLinearPolicy(label: label.rawValue, state: state, signedMove: signedMove, hyperParameters: hyperParameters, costPoints: cost, sampleWeight: sampleWeight)
        classMass[label.rawValue] += sampleWeight
        updateMoveEMA(movePoints: request.movePoints)
        steps += 1
    }

    public func predict(_ request: PredictRequestV4, hyperParameters: HyperParameters) -> PredictionV4 {
        let x = Self.preparedFeatures(request.x, dataHasVolume: request.context.dataHasVolume)
        let window = Self.preparedWindow(request.xWindow, dataHasVolume: request.context.dataHasVolume)
        let state = buildState(
            x: x,
            window: window,
            windowSize: request.windowSize,
            horizonMinutes: request.context.horizonMinutes,
            dataHasVolume: request.context.dataHasVolume
        )
        let cost = PluginContextRuntimeTools.inputPriceCostPoints(x, explicitCostPoints: request.context.priceCostPoints)
        let minMove = max(request.context.minMovePoints, 0.10)
        let forecastVolatility = windowVolatility(window: window, windowSize: request.windowSize)
        let modelConfidence = 0.55
        let margin = PluginSupportTools.clipSymmetric(linearMargin(state), limit: 8.0)
        let scale = max(max(forecastVolatility, minMove * 0.20), 0.05)
        let directionalProbability = PluginSupportTools.sigmoid(margin / scale)
        let edge = abs(margin) * max(moveReady ? moveEMA : minMove, minMove)
        let active = PluginSupportTools.sigmoid((edge - cost) / max(minMove, 0.10))
        let weak = fxClamp(1.0 - modelConfidence, 0.0, 1.0)
        let skip = fxClamp(0.12 + 0.58 * (1.0 - active) + 0.25 * weak, 0.05, 0.92)
        let probabilities = PluginContextRuntimeTools.normalizeClassDistribution([
            (1.0 - skip) * (1.0 - directionalProbability),
            (1.0 - skip) * directionalProbability,
            skip
        ])
        let mean = max(0.0, max(edge, moveReady ? moveEMA : 0.0))
        let sigma = max(0.10, forecastVolatility + 0.25 * mean + 0.25 * minMove)
        let q25 = max(0.0, mean - 0.55 * sigma)
        let q50 = max(q25, mean)
        let q75 = max(q50, mean + 0.55 * sigma)
        let confidence = fxClamp(0.50 * max(probabilities[0], probabilities[1]) + 0.35 * modelConfidence + 0.15 * active, 0.0, 1.0)
        let output = PluginModelOutputV4(
            classProbabilities: probabilities,
            moveMeanPoints: mean,
            moveQ25Points: q25,
            moveQ50Points: q50,
            moveQ75Points: q75,
            mfeMeanPoints: max(q75, mean * (1.05 + 0.35 * confidence)),
            maeMeanPoints: max(0.0, mean * (0.30 + 0.35 * skip + 0.15 * weak)),
            hitTimeFraction: fxClamp(0.70 - 0.35 * active + 0.25 * skip, 0.0, 1.0),
            pathRisk: fxClamp(0.35 * skip + 0.30 * weak + 0.35 * (mean > 0.0 ? (0.30 + 0.35 * skip + 0.15 * weak) : 1.0), 0.0, 1.0),
            fillRisk: fxClamp(cost / max(mean + minMove, 0.10), 0.0, 1.0),
            confidence: confidence,
            reliability: fxClamp(0.30 + 0.30 * min(Double(steps) / 128.0, 1.0) + 0.25 * modelConfidence + 0.15 * (moveReady ? 1.0 : 0.0), 0.0, 1.0),
            hasQuantiles: true,
            hasConfidence: true,
            hasPathQuality: true
        )
        return PluginContextRuntimeTools.fillPrediction(
            modelOutput: output,
            calibratedMoveMeanPoints: mean,
            context: request.context
        )
    }

    private mutating func updateLinearPolicy(
        label: Int,
        state: [Double],
        signedMove: Double,
        hyperParameters: HyperParameters,
        costPoints: Double,
        sampleWeight: Double
    ) {
        let logits = (0..<Self.classCount).map { classIndex in
            Self.dot(weights[classIndex], state)
        }
        let probabilities = Self.softmax3(logits)
        let learningRate = fxClamp(hyperParameters.learningRate, 0.0002, 0.08) * fxClamp(sampleWeight, 0.2, 5.0)
        let l2 = fxClamp(hyperParameters.l2, 0.0, 0.20)
        let l1 = 0.0008
        for classIndex in 0..<Self.classCount {
            let target = classIndex == label ? 1.0 : 0.0
            let error = target - probabilities[classIndex]
            for stateIndex in 0..<Self.stateDimension {
                let sign = weights[classIndex][stateIndex] > 0.0 ? 1.0 : (weights[classIndex][stateIndex] < 0.0 ? -1.0 : 0.0)
                let shrink = l2 * weights[classIndex][stateIndex] + l1 * sign
                let gradient = error * state[stateIndex] - shrink
                velocity[classIndex][stateIndex] = 0.85 * velocity[classIndex][stateIndex] + 0.15 * gradient
                weights[classIndex][stateIndex] = fxClamp(
                    weights[classIndex][stateIndex] + learningRate * velocity[classIndex][stateIndex],
                    -8.0,
                    8.0
                )
            }
        }
        policyOld = probabilities
    }

    private func buildState(
        x: [Double],
        window: [[Double]],
        windowSize: Int,
        horizonMinutes: Int,
        dataHasVolume: Bool
    ) -> [Double] {
        var state = Array(repeating: 0.0, count: Self.stateDimension)
        state[0] = 1.0
        state[1] = x[safe: 1]
        state[2] = x[safe: 2]
        state[3] = x[safe: 3]
        state[4] = x[safe: 4]
        state[5] = x[safe: 7]
        state[6] = x[safe: 12]
        let volumeTerm = dataHasVolume ? fxClamp(0.35 * x[safe: 6] + 0.15 * x[safe: 80] + 0.10 * x[safe: 81], -2.0, 2.0) : 0.0
        state[7] = x[safe: 40] + volumeTerm
        state[8] = PluginContextRuntimeTools.currentWindowFeatureSlope(window, featureIndex: 0, declaredSize: windowSize)
        state[9] = PluginContextRuntimeTools.currentWindowFeatureStd(window, featureIndex: 0, declaredSize: windowSize)
        state[10] = PluginContextRuntimeTools.currentWindowFeatureRange(window, featureIndex: 0, recentBars: 16, declaredSize: windowSize)
        state[11] = PluginContextRuntimeTools.currentWindowFeatureRecentDelta(window, featureIndex: 0, recentBars: 8, declaredSize: windowSize)
        state[12] = PluginContextRuntimeTools.currentWindowFeatureEMAMean(window, featureIndex: 1, decay: 0.70, declaredSize: windowSize)
        state[13] = x[safe: FXDataEngineConstants.macroEventFeatureOffset + 14]
        state[14] = x[safe: FXDataEngineConstants.macroEventFeatureOffset + 19] + 0.20 * volumeTerm
        state[15] = fxClamp(Double(horizonMinutes) / 60.0, 0.0, 2.0)
        for index in 1..<Self.stateDimension {
            state[index] = fxClamp(state[index], -8.0, 8.0)
        }
        return state
    }

    private func linearMargin(_ state: [Double]) -> Double {
        PluginSupportTools.clipSymmetric(
            Self.dot(weights[LabelClass.buy.rawValue], state) - Self.dot(weights[LabelClass.sell.rawValue], state),
            limit: 12.0
        )
    }

    private func windowVolatility(window: [[Double]], windowSize: Int) -> Double {
        var value = max(PluginContextRuntimeTools.currentWindowFeatureStd(window, featureIndex: 0, declaredSize: windowSize), 0.0)
        if value <= 1.0e-6 {
            value = max(abs(PluginContextRuntimeTools.currentWindowFeatureRecentDelta(window, featureIndex: 0, recentBars: 8, declaredSize: windowSize)), 0.0)
        }
        if value <= 1.0e-6 {
            value = max(moveReady ? 0.01 * moveEMA : 0.05, 0.01)
        }
        return value
    }

    private mutating func updateMoveEMA(movePoints: Double) {
        let absoluteMove = abs(fxSafeFinite(movePoints))
        if !moveReady {
            moveEMA = absoluteMove
            moveReady = true
        } else {
            moveEMA = 0.97 * moveEMA + 0.03 * absoluteMove
        }
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

    private static func seedWeight(_ aiID: Int, _ a: Int, _ b: Int) -> Double {
        0.035 * sin(Double((aiID + 17) * (a + 3) * 37 + (b + 11) * 101))
    }

    private static func dot(_ weights: [Double], _ values: [Double]) -> Double {
        var value = 0.0
        let count = min(weights.count, values.count)
        for index in 0..<count {
            value += weights[index] * values[index]
        }
        return value
    }

    private static func softmax3(_ logits: [Double]) -> [Double] {
        let safe = (0..<Self.classCount).map { index -> Double in
            index < logits.count ? fxSafeFinite(logits[index]) : 0.0
        }
        let maximum = safe.max() ?? 0.0
        var exponentials = Array(repeating: 0.0, count: Self.classCount)
        var sum = 0.0
        for index in 0..<Self.classCount {
            let value = exp(fxClamp(safe[index] - maximum, -35.0, 35.0))
            exponentials[index] = value
            sum += value
        }
        guard sum > 0.0 else { return [0.10, 0.10, 0.80] }
        return exponentials.map { $0 / sum }
    }
}

private extension Array where Element == Double {
    subscript(safe index: Int) -> Double {
        indices.contains(index) ? self[index] : 0.0
    }
}
