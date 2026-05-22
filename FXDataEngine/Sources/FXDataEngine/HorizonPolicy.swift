import Foundation

public struct OOFHorizonPriorCell: Codable, Hashable, Sendable {
    public var scoreEMA: Double
    public var edgeEMA: Double
    public var qualityEMA: Double
    public var tradeRateEMA: Double
    public var ready: Bool
    public var observations: Int

    public init(
        scoreEMA: Double = 0.0,
        edgeEMA: Double = 0.0,
        qualityEMA: Double = 0.0,
        tradeRateEMA: Double = 0.0,
        ready: Bool = false,
        observations: Int = 0
    ) {
        self.scoreEMA = fxSafeFinite(scoreEMA)
        self.edgeEMA = fxSafeFinite(edgeEMA)
        self.qualityEMA = fxClamp(qualityEMA, 0.0, 1.0)
        self.tradeRateEMA = fxClamp(tradeRateEMA, 0.0, 1.0)
        self.ready = ready
        self.observations = min(max(observations, 0), HorizonPolicyTools.observationCap)
    }
}

public struct HorizonPolicyNetworkState: Codable, Hashable, Sendable {
    public var inputWeights: [[Double]]
    public var hiddenBias: [Double]
    public var outputWeights: [Double]
    public var outputBias: Double
    public var ready: Bool
    public var observations: Int

    public init(
        inputWeights: [[Double]] = [],
        hiddenBias: [Double] = [],
        outputWeights: [Double] = [],
        outputBias: Double = 0.0,
        ready: Bool = false,
        observations: Int = 0
    ) {
        self.inputWeights = Self.normalizedMatrix(
            inputWeights,
            rows: FXDataEngineConstants.horizonPolicyHidden,
            columns: FXDataEngineConstants.horizonPolicyFeatures
        )
        self.hiddenBias = Self.normalizedVector(hiddenBias, count: FXDataEngineConstants.horizonPolicyHidden)
        self.outputWeights = Self.normalizedVector(outputWeights, count: FXDataEngineConstants.horizonPolicyHidden)
        self.outputBias = fxSafeFinite(outputBias)
        self.ready = ready
        self.observations = min(max(observations, 0), HorizonPolicyTools.observationCap)
    }

    private static func normalizedVector(_ values: [Double], count: Int) -> [Double] {
        var output = Array(repeating: 0.0, count: count)
        for index in 0..<min(values.count, count) {
            output[index] = fxSafeFinite(values[index])
        }
        return output
    }

    private static func normalizedMatrix(_ values: [[Double]], rows: Int, columns: Int) -> [[Double]] {
        var output = Array(repeating: Array(repeating: 0.0, count: columns), count: rows)
        for row in 0..<min(values.count, rows) {
            for column in 0..<min(values[row].count, columns) {
                output[row][column] = fxSafeFinite(values[row][column])
            }
        }
        return output
    }
}

public struct HorizonPolicyPrediction: Codable, Hashable, Sendable {
    public var value: Double
    public var hidden: [Double]

    public init(value: Double, hidden: [Double]) {
        self.value = fxSafeFinite(value)
        var resolved = Array(repeating: 0.0, count: FXDataEngineConstants.horizonPolicyHidden)
        for index in 0..<min(hidden.count, resolved.count) {
            resolved[index] = fxSafeFinite(hidden[index])
        }
        self.hidden = resolved
    }
}

public enum HorizonPolicyTools {
    public static let observationCap = 200_000

    public static func updatedOOFPriorCell(
        _ cell: OOFHorizonPriorCell,
        scoreProxy: Double,
        edgeRatio: Double,
        quality: Double,
        tradeTarget: Bool
    ) -> OOFHorizonPriorCell {
        let score = fxClamp(scoreProxy, -4.0, 8.0)
        let edge = fxClamp(edgeRatio, -2.0, 4.0)
        let quality = fxClamp(quality, 0.0, 1.0)
        let tradeRate = tradeTarget ? 1.0 : 0.0

        guard cell.ready else {
            return OOFHorizonPriorCell(
                scoreEMA: score,
                edgeEMA: edge,
                qualityEMA: quality,
                tradeRateEMA: tradeRate,
                ready: true,
                observations: 1
            )
        }

        let observations = max(cell.observations, 0)
        let alpha = fxClamp(0.18 / sqrt(1.0 + 0.03 * Double(observations)), 0.025, 0.12)
        return OOFHorizonPriorCell(
            scoreEMA: (1.0 - alpha) * cell.scoreEMA + alpha * score,
            edgeEMA: (1.0 - alpha) * cell.edgeEMA + alpha * edge,
            qualityEMA: (1.0 - alpha) * cell.qualityEMA + alpha * quality,
            tradeRateEMA: (1.0 - alpha) * cell.tradeRateEMA + alpha * tradeRate,
            ready: true,
            observations: min(observations + 1, observationCap)
        )
    }

    public static func oofHorizonPriorScore(_ cell: OOFHorizonPriorCell) -> Double {
        guard cell.ready else { return 0.0 }
        let trust = fxClamp(Double(cell.observations) / 48.0, 0.05, 0.35)
        let prior = 0.28 * cell.scoreEMA +
            0.22 * cell.edgeEMA +
            0.28 * (2.0 * cell.qualityEMA - 1.0) +
            0.22 * (2.0 * cell.tradeRateEMA - 1.0)
        return trust * fxClamp(prior, -3.0, 3.0)
    }

    public static func oofTradeGatePrior(_ cell: OOFHorizonPriorCell) -> Double {
        guard cell.ready else { return -1.0 }
        let trust = fxClamp(Double(cell.observations) / 64.0, 0.10, 0.45)
        let prior = 0.18 +
            0.42 * cell.tradeRateEMA +
            0.20 * cell.qualityEMA +
            0.12 * fxClamp(cell.edgeEMA, 0.0, 2.0) / 2.0 +
            0.08 * fxClamp(cell.scoreEMA, 0.0, 4.0) / 4.0
        return fxClamp((1.0 - trust) * 0.50 + trust * prior, 0.01, 0.99)
    }

    public static func predictValue(
        _ state: HorizonPolicyNetworkState,
        features: [Double]
    ) -> HorizonPolicyPrediction {
        let features = normalizedFeatures(features)
        var hidden = Array(repeating: 0.0, count: FXDataEngineConstants.horizonPolicyHidden)
        for hiddenIndex in 0..<FXDataEngineConstants.horizonPolicyHidden {
            var z = state.hiddenBias[hiddenIndex]
            for featureIndex in 0..<FXDataEngineConstants.horizonPolicyFeatures {
                z += state.inputWeights[hiddenIndex][featureIndex] * features[featureIndex]
            }
            hidden[hiddenIndex] = legacyTanh(z)
        }

        var prediction = state.outputBias
        for hiddenIndex in 0..<FXDataEngineConstants.horizonPolicyHidden {
            prediction += state.outputWeights[hiddenIndex] * hidden[hiddenIndex]
        }
        return HorizonPolicyPrediction(value: prediction, hidden: hidden)
    }

    public static func updatedNetwork(
        _ state: HorizonPolicyNetworkState,
        features: [Double],
        rewardScaled: Double
    ) -> HorizonPolicyNetworkState {
        let features = normalizedFeatures(features)
        let prediction = predictValue(state, features: features)
        let error = fxClamp(fxSafeFinite(rewardScaled) - prediction.value, -4.0, 4.0)
        let learningRate = fxClamp(
            0.020 / sqrt(1.0 + 0.02 * Double(max(state.observations, 0))),
            0.0015,
            0.020
        )

        let oldOutputWeights = state.outputWeights
        var next = state
        next.outputBias += learningRate * error
        for hiddenIndex in 0..<FXDataEngineConstants.horizonPolicyHidden {
            let outputRegularization = 0.0008 * state.outputWeights[hiddenIndex]
            next.outputWeights[hiddenIndex] += learningRate * (
                error * prediction.hidden[hiddenIndex] - outputRegularization
            )
        }

        for hiddenIndex in 0..<FXDataEngineConstants.horizonPolicyHidden {
            let hiddenValue = prediction.hidden[hiddenIndex]
            let delta = (1.0 - hiddenValue * hiddenValue) * oldOutputWeights[hiddenIndex] * error
            next.hiddenBias[hiddenIndex] += learningRate * delta
            for featureIndex in 0..<FXDataEngineConstants.horizonPolicyFeatures {
                let inputRegularization = 0.0006 * state.inputWeights[hiddenIndex][featureIndex]
                next.inputWeights[hiddenIndex][featureIndex] += learningRate * (
                    delta * features[featureIndex] - inputRegularization
                )
            }
        }

        next.observations = min(max(state.observations, 0) + 1, observationCap)
        next.ready = true
        return next
    }

    public static func legacyTanh(_ value: Double) -> Double {
        let value = fxSafeFinite(value)
        if value > 18.0 {
            return 1.0
        }
        if value < -18.0 {
            return -1.0
        }
        let exp2 = exp(2.0 * value)
        return (exp2 - 1.0) / (exp2 + 1.0)
    }

    private static func normalizedFeatures(_ features: [Double]) -> [Double] {
        var output = Array(repeating: 0.0, count: FXDataEngineConstants.horizonPolicyFeatures)
        for index in 0..<min(features.count, output.count) {
            output[index] = fxSafeFinite(features[index])
        }
        return output
    }
}
