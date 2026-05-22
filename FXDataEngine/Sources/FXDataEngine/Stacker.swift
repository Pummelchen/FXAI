import Foundation

public struct StackRouterActionCell: Codable, Hashable, Sendable {
    public var value: Double
    public var regret: Double
    public var counterfactual: Double
    public var ready: Bool
    public var observations: Int

    public init(
        value: Double = 0.0,
        regret: Double = 0.0,
        counterfactual: Double = 0.0,
        ready: Bool = false,
        observations: Int = 0
    ) {
        self.value = fxSafeFinite(value)
        self.regret = fxClamp(regret, 0.0, 1.0)
        self.counterfactual = fxClamp(counterfactual, -1.0, 1.0)
        self.ready = ready
        self.observations = min(max(observations, 0), StackerTools.observationCap)
    }
}

public struct StackFeatureInputs: Codable, Hashable, Sendable {
    public var buyPct: Double
    public var sellPct: Double
    public var skipPct: Double
    public var avgBuyEV: Double
    public var avgSellEV: Double
    public var minMovePoints: Double
    public var expectedMovePoints: Double
    public var volProxy: Double
    public var horizonMinutes: Int
    public var maxConfiguredHorizonMinutes: Int?
    public var avgConfidence: Double
    public var avgReliability: Double
    public var moveDispersion: Double
    public var directionalMargin: Double
    public var activeFamilyRatio: Double
    public var dominantFamilyRatio: Double
    public var contextStrength: Double
    public var contextQuality: Double
    public var avgHitTime: Double
    public var avgPathRisk: Double
    public var avgFillRisk: Double
    public var avgMFERatio: Double
    public var avgMAERatio: Double
    public var avgContextEdgeNorm: Double
    public var avgContextRegret: Double
    public var avgGlobalEdgeNorm: Double
    public var bestCounterfactualEdgeNorm: Double
    public var ensembleVsBestGapNorm: Double
    public var avgPortfolioEdgeNorm: Double
    public var avgPortfolioStability: Double
    public var avgPortfolioCorrelationPenalty: Double
    public var avgPortfolioDiversification: Double
    public var bestModelShare: Double
    public var bestBuyShare: Double
    public var bestSellShare: Double
    public var avgContextTrust: Double
    public var foundationTrust: Double
    public var foundationDirectionBias: Double
    public var foundationMoveRatio: Double
    public var studentTrust: Double
    public var studentTradability: Double
    public var analogSimilarity: Double
    public var analogEdgeNorm: Double
    public var analogQuality: Double
    public var hierarchyConsistency: Double
    public var hierarchyTradability: Double
    public var hierarchyExecutionViability: Double
    public var hierarchyHorizonFit: Double

    public init(
        buyPct: Double = 0.0,
        sellPct: Double = 0.0,
        skipPct: Double = 0.0,
        avgBuyEV: Double = 0.0,
        avgSellEV: Double = 0.0,
        minMovePoints: Double = 0.0,
        expectedMovePoints: Double = 0.0,
        volProxy: Double = 0.0,
        horizonMinutes: Int = 0,
        maxConfiguredHorizonMinutes: Int? = nil,
        avgConfidence: Double = 0.0,
        avgReliability: Double = 0.0,
        moveDispersion: Double = 0.0,
        directionalMargin: Double = 0.0,
        activeFamilyRatio: Double = 0.0,
        dominantFamilyRatio: Double = 0.0,
        contextStrength: Double = 0.0,
        contextQuality: Double = 0.0,
        avgHitTime: Double = 0.0,
        avgPathRisk: Double = 0.0,
        avgFillRisk: Double = 0.0,
        avgMFERatio: Double = 0.0,
        avgMAERatio: Double = 0.0,
        avgContextEdgeNorm: Double = 0.0,
        avgContextRegret: Double = 0.0,
        avgGlobalEdgeNorm: Double = 0.0,
        bestCounterfactualEdgeNorm: Double = 0.0,
        ensembleVsBestGapNorm: Double = 0.0,
        avgPortfolioEdgeNorm: Double = 0.0,
        avgPortfolioStability: Double = 0.0,
        avgPortfolioCorrelationPenalty: Double = 0.0,
        avgPortfolioDiversification: Double = 0.0,
        bestModelShare: Double = 0.0,
        bestBuyShare: Double = 0.0,
        bestSellShare: Double = 0.0,
        avgContextTrust: Double = 0.0,
        foundationTrust: Double = 0.0,
        foundationDirectionBias: Double = 0.0,
        foundationMoveRatio: Double = 0.0,
        studentTrust: Double = 0.0,
        studentTradability: Double = 0.0,
        analogSimilarity: Double = 0.0,
        analogEdgeNorm: Double = 0.0,
        analogQuality: Double = 0.0,
        hierarchyConsistency: Double = 0.0,
        hierarchyTradability: Double = 0.0,
        hierarchyExecutionViability: Double = 0.0,
        hierarchyHorizonFit: Double = 0.0
    ) {
        self.buyPct = buyPct
        self.sellPct = sellPct
        self.skipPct = skipPct
        self.avgBuyEV = avgBuyEV
        self.avgSellEV = avgSellEV
        self.minMovePoints = minMovePoints
        self.expectedMovePoints = expectedMovePoints
        self.volProxy = volProxy
        self.horizonMinutes = horizonMinutes
        self.maxConfiguredHorizonMinutes = maxConfiguredHorizonMinutes
        self.avgConfidence = avgConfidence
        self.avgReliability = avgReliability
        self.moveDispersion = moveDispersion
        self.directionalMargin = directionalMargin
        self.activeFamilyRatio = activeFamilyRatio
        self.dominantFamilyRatio = dominantFamilyRatio
        self.contextStrength = contextStrength
        self.contextQuality = contextQuality
        self.avgHitTime = avgHitTime
        self.avgPathRisk = avgPathRisk
        self.avgFillRisk = avgFillRisk
        self.avgMFERatio = avgMFERatio
        self.avgMAERatio = avgMAERatio
        self.avgContextEdgeNorm = avgContextEdgeNorm
        self.avgContextRegret = avgContextRegret
        self.avgGlobalEdgeNorm = avgGlobalEdgeNorm
        self.bestCounterfactualEdgeNorm = bestCounterfactualEdgeNorm
        self.ensembleVsBestGapNorm = ensembleVsBestGapNorm
        self.avgPortfolioEdgeNorm = avgPortfolioEdgeNorm
        self.avgPortfolioStability = avgPortfolioStability
        self.avgPortfolioCorrelationPenalty = avgPortfolioCorrelationPenalty
        self.avgPortfolioDiversification = avgPortfolioDiversification
        self.bestModelShare = bestModelShare
        self.bestBuyShare = bestBuyShare
        self.bestSellShare = bestSellShare
        self.avgContextTrust = avgContextTrust
        self.foundationTrust = foundationTrust
        self.foundationDirectionBias = foundationDirectionBias
        self.foundationMoveRatio = foundationMoveRatio
        self.studentTrust = studentTrust
        self.studentTradability = studentTradability
        self.analogSimilarity = analogSimilarity
        self.analogEdgeNorm = analogEdgeNorm
        self.analogQuality = analogQuality
        self.hierarchyConsistency = hierarchyConsistency
        self.hierarchyTradability = hierarchyTradability
        self.hierarchyExecutionViability = hierarchyExecutionViability
        self.hierarchyHorizonFit = hierarchyHorizonFit
    }
}

public struct StackNetworkState: Codable, Hashable, Sendable {
    public var inputWeights: [[Double]]
    public var hiddenBias: [Double]
    public var outputWeights: [[Double]]
    public var outputBias: [Double]
    public var ready: Bool
    public var observations: Int

    public init(
        inputWeights: [[Double]] = [],
        hiddenBias: [Double] = [],
        outputWeights: [[Double]] = [],
        outputBias: [Double] = [],
        ready: Bool = false,
        observations: Int = 0
    ) {
        self.inputWeights = Self.normalizedMatrix(
            inputWeights,
            rows: FXDataEngineConstants.stackHidden,
            columns: FXDataEngineConstants.stackFeatures
        )
        self.hiddenBias = Self.normalizedVector(hiddenBias, count: FXDataEngineConstants.stackHidden)
        self.outputWeights = Self.normalizedMatrix(
            outputWeights,
            rows: LabelClass.allCases.count,
            columns: FXDataEngineConstants.stackHidden
        )
        self.outputBias = Self.normalizedVector(outputBias, count: LabelClass.allCases.count)
        self.ready = ready
        self.observations = min(max(observations, 0), StackerTools.observationCap)
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

public struct StackPrediction: Codable, Hashable, Sendable {
    public var probabilities: [Double]
    public var rawProbabilities: [Double]
    public var hidden: [Double]

    public init(probabilities: [Double], rawProbabilities: [Double], hidden: [Double]) {
        self.probabilities = Self.normalizedVector(probabilities, count: LabelClass.allCases.count)
        self.rawProbabilities = Self.normalizedVector(rawProbabilities, count: LabelClass.allCases.count)
        self.hidden = Self.normalizedVector(hidden, count: FXDataEngineConstants.stackHidden)
    }

    private static func normalizedVector(_ values: [Double], count: Int) -> [Double] {
        var output = Array(repeating: 0.0, count: count)
        for index in 0..<min(values.count, count) {
            output[index] = fxSafeFinite(values[index])
        }
        return output
    }
}

public enum StackerTools {
    public static let observationCap = 200_000

    public static func isModelInList(aiID: Int, modelIDs: [Int]) -> Bool {
        modelIDs.contains(aiID)
    }

    public static func stackFeature(_ features: [Double], _ index: Int, default defaultValue: Double = 0.0) -> Double {
        guard index >= 0, index < features.count else { return defaultValue }
        return fxSafeFinite(features[index], fallback: defaultValue)
    }

    public static func buildStackFeatures(_ input: StackFeatureInputs) -> [Double] {
        var features = Array(repeating: 0.0, count: FXDataEngineConstants.stackFeatures)

        let minMove = fxSafeFinite(input.minMovePoints)
        let expectedMove = fxSafeFinite(input.expectedMovePoints)
        let mm = max(minMove, 0.10)
        let em = max(expectedMove, mm)
        var buyProbability = fxClamp(input.buyPct / 100.0, 0.0, 1.0)
        var sellProbability = fxClamp(input.sellPct / 100.0, 0.0, 1.0)
        var skipProbability = fxClamp(input.skipPct / 100.0, 0.0, 1.0)
        let probabilitySum = buyProbability + sellProbability + skipProbability
        if probabilitySum > 0.0 {
            buyProbability /= probabilitySum
            sellProbability /= probabilitySum
            skipProbability /= probabilitySum
        }

        var entropy = 0.0
        if buyProbability > 1e-9 { entropy -= buyProbability * log(buyProbability) }
        if sellProbability > 1e-9 { entropy -= sellProbability * log(sellProbability) }
        if skipProbability > 1e-9 { entropy -= skipProbability * log(skipProbability) }
        let entropyNorm = 1.0 - fxClamp(entropy / log(3.0), 0.0, 1.0)

        features[0] = 1.0
        features[1] = fxClamp((buyProbability - 0.5) * 2.0, -1.0, 1.0)
        features[2] = fxClamp((sellProbability - 0.5) * 2.0, -1.0, 1.0)
        features[3] = fxClamp((skipProbability - 0.5) * 2.0, -1.0, 1.0)
        features[4] = fxClamp(input.avgBuyEV / mm, -3.0, 3.0) / 3.0
        features[5] = fxClamp(input.avgSellEV / mm, -3.0, 3.0) / 3.0
        features[6] = fxClamp(buyProbability - sellProbability, -1.0, 1.0)
        features[7] = fxClamp(max(buyProbability, sellProbability), 0.0, 1.0)
        features[8] = fxClamp(entropyNorm, 0.0, 1.0)
        features[9] = fxClamp((input.avgBuyEV - input.avgSellEV) / mm, -4.0, 4.0) / 4.0
        features[10] = fxClamp(input.minMovePoints / em, 0.0, 2.0) - 0.5
        features[11] = fxClamp(input.volProxy / 4.0, 0.0, 1.0)
        let maxConfiguredHorizon = max(input.maxConfiguredHorizonMinutes ?? input.horizonMinutes, 1)
        features[12] = fxClamp(Double(input.horizonMinutes) / Double(maxConfiguredHorizon), 0.0, 1.0)
        features[13] = fxClamp(skipProbability - max(buyProbability, sellProbability), -1.0, 1.0)
        let topProbability = max(buyProbability, sellProbability, skipProbability)
        let secondProbability: Double
        if topProbability == buyProbability {
            secondProbability = max(sellProbability, skipProbability)
        } else if topProbability == sellProbability {
            secondProbability = max(buyProbability, skipProbability)
        } else {
            secondProbability = max(buyProbability, sellProbability)
        }
        features[14] = fxClamp(topProbability - secondProbability, 0.0, 1.0)
        features[15] = fxClamp(abs(buyProbability - sellProbability), 0.0, 1.0)
        features[16] = fxClamp(min(buyProbability, sellProbability), 0.0, 0.5) * 2.0
        features[17] = fxClamp(max(input.avgBuyEV, input.avgSellEV) / mm, -2.0, 6.0) / 4.0
        features[18] = fxClamp(input.expectedMovePoints / mm, 0.0, 8.0) / 4.0
        features[19] = fxClamp((buyProbability + sellProbability) - skipProbability, -1.0, 1.0)
        features[20] = fxClamp(input.avgConfidence, 0.0, 1.0)
        features[21] = fxClamp(input.avgReliability, 0.0, 1.0)
        features[22] = fxClamp(input.moveDispersion / mm, 0.0, 4.0) / 2.0
        features[23] = fxClamp(input.directionalMargin, 0.0, 1.0)
        features[24] = fxClamp(input.activeFamilyRatio, 0.0, 1.0)
        features[25] = fxClamp(input.dominantFamilyRatio, 0.0, 1.0)
        features[26] = fxClamp(input.contextStrength, 0.0, 4.0) / 2.0
        features[27] = fxClamp(input.contextQuality, -1.0, 2.0) / 2.0
        features[28] = fxClamp(buyProbability + sellProbability, 0.0, 1.0)
        features[29] = fxClamp(input.avgConfidence * input.avgReliability, 0.0, 1.0)
        features[30] = fxClamp((buyProbability - sellProbability) * input.directionalMargin, -1.0, 1.0)
        features[31] = fxClamp(input.moveDispersion / em, 0.0, 4.0) / 2.0
        features[32] = fxClamp((input.contextStrength * max(input.contextQuality, 0.0)) / 4.0, 0.0, 2.0) - 0.5
        features[33] = fxClamp(input.dominantFamilyRatio * input.avgReliability, 0.0, 1.0)
        features[34] = fxClamp((input.avgBuyEV + input.avgSellEV) / (2.0 * mm), -3.0, 3.0) / 3.0
        features[35] = fxClamp(entropyNorm * 0.5 * (input.avgConfidence + input.avgReliability), 0.0, 1.0)
        features[36] = fxClamp(max(input.avgBuyEV, input.avgSellEV) / max(input.expectedMovePoints, mm), -2.0, 6.0) / 4.0
        features[37] = fxClamp(input.avgConfidence - input.avgReliability, -1.0, 1.0)
        features[38] = fxClamp(input.avgReliability * max(input.contextQuality, 0.0), 0.0, 1.5) - 0.25
        features[39] = fxClamp(input.directionalMargin * input.contextStrength, 0.0, 4.0) / 2.0 - 0.5
        features[40] = fxClamp(input.activeFamilyRatio * input.dominantFamilyRatio, 0.0, 1.0)
        features[41] = fxClamp(input.moveDispersion / max(input.expectedMovePoints, mm), 0.0, 4.0) / 2.0
        features[42] = fxClamp((buyProbability + sellProbability - skipProbability) * input.avgConfidence, -1.0, 1.0)
        features[43] = fxClamp(
            ((input.avgBuyEV - input.avgSellEV) / max(input.expectedMovePoints, mm)) * input.directionalMargin,
            -2.0,
            2.0
        ) / 2.0
        features[44] = fxClamp(entropyNorm * input.dominantFamilyRatio, 0.0, 1.0)
        features[45] = fxClamp(input.avgConfidence * input.directionalMargin * (1.0 - skipProbability), 0.0, 1.0)
        features[46] = fxClamp(
            input.contextStrength * input.dominantFamilyRatio * input.avgReliability,
            0.0,
            4.0
        ) / 2.0 - 0.5
        features[47] = fxClamp(
            (input.avgBuyEV + input.avgSellEV) / max(input.expectedMovePoints + mm, mm),
            -2.0,
            2.0
        ) / 2.0
        features[48] = fxClamp(input.avgHitTime, 0.0, 1.0)
        features[49] = fxClamp(input.avgPathRisk, 0.0, 1.0)
        features[50] = fxClamp(input.avgFillRisk, 0.0, 1.0)
        features[51] = fxClamp(input.avgMFERatio, 0.0, 4.0) / 2.0 - 0.5
        features[52] = fxClamp(input.avgMAERatio, 0.0, 2.0) - 0.5
        features[53] = fxClamp((1.0 - input.avgPathRisk) * input.avgConfidence, 0.0, 1.0)
        features[54] = fxClamp((1.0 - input.avgFillRisk) * input.dominantFamilyRatio, 0.0, 1.0)
        features[55] = fxClamp(
            input.avgMFERatio * (1.0 - input.avgMAERatio) * max(input.contextQuality + 1.0, 0.0),
            0.0,
            4.0
        ) / 2.0 - 0.5
        features[56] = fxClamp(input.avgContextEdgeNorm, -1.0, 1.0)
        features[57] = fxClamp(input.avgContextRegret, 0.0, 1.0)
        features[58] = fxClamp(input.avgGlobalEdgeNorm, -1.0, 1.0)
        features[59] = fxClamp(input.bestCounterfactualEdgeNorm, -1.0, 1.0)
        features[60] = fxClamp(input.ensembleVsBestGapNorm, 0.0, 1.0)
        features[61] = fxClamp(input.avgPortfolioEdgeNorm, -1.0, 1.0)
        features[62] = fxClamp(input.avgPortfolioStability, 0.0, 1.0)
        features[63] = fxClamp(input.avgPortfolioCorrelationPenalty, 0.0, 1.0)
        features[64] = fxClamp(input.avgPortfolioDiversification, 0.0, 1.0)
        features[65] = fxClamp(input.bestModelShare, 0.0, 1.0)
        features[66] = fxClamp(input.bestBuyShare, 0.0, 1.0)
        features[67] = fxClamp(input.bestSellShare, 0.0, 1.0)
        features[68] = fxClamp(input.avgContextTrust, 0.0, 1.0)
        features[69] = fxClamp(input.bestCounterfactualEdgeNorm - input.avgContextRegret, -1.0, 1.0)
        features[70] = fxClamp(input.avgPortfolioStability * (1.0 - input.avgContextRegret), 0.0, 1.0)
        features[71] = fxClamp(
            (input.avgContextEdgeNorm + input.avgGlobalEdgeNorm + input.avgPortfolioEdgeNorm) / 3.0,
            -1.0,
            1.0
        )
        features[72] = fxClamp(input.foundationTrust, 0.0, 1.0)
        features[73] = fxClamp(input.foundationDirectionBias, -1.0, 1.0)
        features[74] = fxClamp((input.foundationMoveRatio - 1.0) / 1.5, -1.0, 1.0)
        features[75] = fxClamp(input.studentTrust, 0.0, 1.0)
        features[76] = fxClamp(input.studentTradability, 0.0, 1.0)
        features[77] = fxClamp(input.analogSimilarity, 0.0, 1.0)
        features[78] = fxClamp(input.analogEdgeNorm, -1.0, 1.0)
        features[79] = fxClamp(input.analogQuality, 0.0, 1.0)
        features[80] = fxClamp(input.hierarchyConsistency, 0.0, 1.0)
        features[81] = fxClamp(input.hierarchyTradability, 0.0, 1.0)
        features[82] = fxClamp(input.hierarchyExecutionViability, 0.0, 1.0)
        features[83] = fxClamp(input.hierarchyHorizonFit, 0.0, 1.0)
        return features
    }

    public static func predictStackProbabilities(
        _ state: StackNetworkState,
        features: [Double],
        actionCells: [StackRouterActionCell] = []
    ) -> StackPrediction {
        let features = normalizedStackFeatures(features)
        guard state.ready else {
            let probabilities = stackRouterBlend(
                probabilities: heuristicStackProbabilities(features: features),
                features: features,
                actionCells: actionCells
            )
            return StackPrediction(
                probabilities: probabilities,
                rawProbabilities: probabilities,
                hidden: []
            )
        }

        let forward = stackNetworkForward(state, features: features)
        var probabilities = forward.rawProbabilities
        let portfolioObjective = stackPortfolioObjective(features: features)
        let routingObjective = stackRoutingObjective(features: features)
        let directionBias = fxClamp(
            0.08 * routingObjective * fxClamp(stackFeature(features, 6), -1.0, 1.0) +
            0.05 * portfolioObjective * fxClamp(stackFeature(features, 71), -1.0, 1.0),
            -0.12,
            0.12
        )
        probabilities[LabelClass.sell.rawValue] = fxClamp(
            probabilities[LabelClass.sell.rawValue] + max(-directionBias, 0.0),
            0.0005,
            0.9990
        )
        probabilities[LabelClass.buy.rawValue] = fxClamp(
            probabilities[LabelClass.buy.rawValue] + max(directionBias, 0.0),
            0.0005,
            0.9990
        )
        probabilities[LabelClass.skip.rawValue] = fxClamp(
            probabilities[LabelClass.skip.rawValue] - 0.04 * portfolioObjective,
            0.0005,
            0.9990
        )
        probabilities = normalizedProbabilities(probabilities)
        probabilities = stackRouterBlend(
            probabilities: probabilities,
            features: features,
            actionCells: actionCells
        )
        return StackPrediction(
            probabilities: probabilities,
            rawProbabilities: forward.rawProbabilities,
            hidden: forward.hidden
        )
    }

    public static func updatedStackNetwork(
        _ state: StackNetworkState,
        features: [Double],
        labelClass: LabelClass,
        sampleWeight: Double
    ) -> StackNetworkState {
        let features = normalizedStackFeatures(features)
        let forward = stackNetworkForward(state, features: features)
        let learningRate = fxClamp(
            0.025 / sqrt(1.0 + 0.02 * Double(max(state.observations, 0))),
            0.002,
            0.025
        )
        let portfolioObjective = stackPortfolioObjective(features: features)
        let routingObjective = stackRoutingObjective(features: features)
        let sampleWeight = fxClamp(
            sampleWeight * fxClamp(0.90 + 0.30 * portfolioObjective + 0.20 * routingObjective, 0.45, 1.60),
            0.20,
            7.50
        )

        var deltaOutput = Array(repeating: 0.0, count: LabelClass.allCases.count)
        for label in LabelClass.allCases {
            let index = label.rawValue
            let target = label == labelClass ? 1.0 : 0.0
            deltaOutput[index] = fxClamp(
                (target - forward.rawProbabilities[index]) * sampleWeight,
                -3.0,
                3.0
            )
        }

        var deltaHidden = Array(repeating: 0.0, count: FXDataEngineConstants.stackHidden)
        for hiddenIndex in 0..<FXDataEngineConstants.stackHidden {
            var back = 0.0
            for label in LabelClass.allCases {
                back += deltaOutput[label.rawValue] * state.outputWeights[label.rawValue][hiddenIndex]
            }
            let hiddenValue = forward.hidden[hiddenIndex]
            deltaHidden[hiddenIndex] = fxClamp(back * (1.0 - hiddenValue * hiddenValue), -3.0, 3.0)
        }

        var outputWeights = state.outputWeights
        var outputBias = state.outputBias
        for label in LabelClass.allCases {
            let labelIndex = label.rawValue
            outputBias[labelIndex] += learningRate * deltaOutput[labelIndex]
            for hiddenIndex in 0..<FXDataEngineConstants.stackHidden {
                let regularization = 0.0005 * outputWeights[labelIndex][hiddenIndex]
                outputWeights[labelIndex][hiddenIndex] += learningRate *
                    (deltaOutput[labelIndex] * forward.hidden[hiddenIndex] - regularization)
            }
        }

        var inputWeights = state.inputWeights
        var hiddenBias = state.hiddenBias
        for hiddenIndex in 0..<FXDataEngineConstants.stackHidden {
            hiddenBias[hiddenIndex] += learningRate * deltaHidden[hiddenIndex]
            for featureIndex in 0..<FXDataEngineConstants.stackFeatures {
                let regularization = featureIndex == 0 ? 0.0 : 0.0004 * inputWeights[hiddenIndex][featureIndex]
                inputWeights[hiddenIndex][featureIndex] += learningRate *
                    (deltaHidden[hiddenIndex] * features[featureIndex] - regularization)
            }
        }

        return StackNetworkState(
            inputWeights: inputWeights,
            hiddenBias: hiddenBias,
            outputWeights: outputWeights,
            outputBias: outputBias,
            ready: true,
            observations: min(max(state.observations, 0) + 1, observationCap)
        )
    }

    public static func stackPortfolioObjective(features: [Double]) -> Double {
        fxClamp(
            0.30 * stackFeature(features, 61) +
            0.28 * stackFeature(features, 62) -
            0.22 * stackFeature(features, 63) +
            0.24 * stackFeature(features, 64) +
            0.10 * stackFeature(features, 70) +
            0.10 * stackFeature(features, 71),
            -1.0,
            1.0
        )
    }

    public static func stackRoutingObjective(features: [Double]) -> Double {
        fxClamp(
            0.22 * stackFeature(features, 56) -
            0.30 * stackFeature(features, 57) +
            0.18 * stackFeature(features, 58) +
            0.20 * stackFeature(features, 59) -
            0.18 * stackFeature(features, 60) +
            0.10 * stackFeature(features, 68) +
            0.16 * stackFeature(features, 69),
            -1.0,
            1.0
        )
    }

    public static func stackRouterContextTrust(features: [Double]) -> Double {
        fxClamp(
            0.18 +
            0.22 * stackFeature(features, 68) +
            0.16 * stackFeature(features, 62) +
            0.12 * stackFeature(features, 70) +
            0.10 * stackFeature(features, 71) -
            0.14 * stackFeature(features, 57) -
            0.10 * stackFeature(features, 63),
            0.0,
            1.0
        )
    }

    public static func stackRouterActionUtility(
        action: LabelClass,
        labelClass: LabelClass,
        realizedEdge: Double,
        qualityScore: Double
    ) -> Double {
        let realizedEdge = fxSafeFinite(realizedEdge)
        let edgeNorm = fxClamp(realizedEdge / max(abs(realizedEdge), 1.0), -1.0, 1.0)
        let quality = fxClamp(qualityScore, 0.0, 2.0)

        if action == .skip {
            if labelClass == .skip {
                return fxClamp(0.28 + 0.22 * quality - 0.12 * edgeNorm, -1.0, 1.0)
            }
            return fxClamp(-0.28 - 0.46 * max(edgeNorm, 0.0) - 0.10 * quality, -1.0, 1.0)
        }

        if action == labelClass {
            return fxClamp(0.32 + 0.52 * edgeNorm + 0.18 * quality, -1.0, 1.0)
        }
        if labelClass == .skip {
            return fxClamp(-0.22 - 0.18 * quality - 0.18 * abs(edgeNorm), -1.0, 1.0)
        }
        return fxClamp(-0.34 - 0.52 * max(edgeNorm, 0.0) - 0.12 * quality, -1.0, 1.0)
    }

    public static func observedRouterCells(
        _ cells: [StackRouterActionCell],
        labelClass: LabelClass,
        realizedEdge: Double,
        qualityScore: Double,
        features: [Double],
        predictedProbabilities: [Double],
        sampleWeight: Double
    ) -> [StackRouterActionCell] {
        let currentCells = normalizedCells(cells)
        let utilities = LabelClass.allCases.map {
            stackRouterActionUtility(
                action: $0,
                labelClass: labelClass,
                realizedEdge: realizedEdge,
                qualityScore: qualityScore
            )
        }
        var baseline = 0.0
        var bestUtility = -Double.greatestFiniteMagnitude
        for label in LabelClass.allCases {
            let index = label.rawValue
            baseline += vectorValue(predictedProbabilities, index, default: 0.3333333) * utilities[index]
            bestUtility = max(bestUtility, utilities[index])
        }

        let trust = fxClamp(
            sampleWeight * (0.40 + 0.60 * stackRouterContextTrust(features: features)),
            0.10,
            6.0
        )
        return LabelClass.allCases.map { label in
            let index = label.rawValue
            let cell = currentCells[index]
            let observations = max(cell.observations, 0)
            let alpha = fxClamp(0.18 / sqrt(1.0 + 0.05 * Double(observations)), 0.02, 0.18)
            let utility = utilities[index]
            let counterfactual = clipSym(utility - baseline, limit: 1.0)
            let regret = fxClamp(bestUtility - utility, 0.0, 1.0)
            if observations <= 0 {
                return StackRouterActionCell(
                    value: utility,
                    regret: regret,
                    counterfactual: counterfactual,
                    ready: true,
                    observations: 1
                )
            }

            let blend = fxClamp(alpha * trust, 0.01, 0.25)
            return StackRouterActionCell(
                value: (1.0 - blend) * cell.value + blend * utility,
                regret: (1.0 - blend) * cell.regret + blend * regret,
                counterfactual: (1.0 - blend) * cell.counterfactual + blend * counterfactual,
                ready: true,
                observations: min(observations + 1, observationCap)
            )
        }
    }

    public static func stackRouterBlend(
        probabilities: [Double],
        features: [Double],
        actionCells: [StackRouterActionCell]
    ) -> [Double] {
        var output = [
            vectorValue(probabilities, LabelClass.sell.rawValue, default: 0.3333333),
            vectorValue(probabilities, LabelClass.buy.rawValue, default: 0.3333333),
            vectorValue(probabilities, LabelClass.skip.rawValue, default: 0.3333334)
        ]
        let contextTrust = stackRouterContextTrust(features: features)
        guard contextTrust > 1e-6 else { return output }

        let cells = normalizedCells(actionCells)
        let routingObjective = stackRoutingObjective(features: features)
        let portfolioObjective = stackPortfolioObjective(features: features)
        let directionBias = fxClamp(stackFeature(features, 6), -1.0, 1.0)
        let correlationPenalty = fxClamp(stackFeature(features, 63), 0.0, 1.0)
        for label in LabelClass.allCases {
            let index = label.rawValue
            let cell = cells[index]
            guard cell.ready, cell.observations > 0 else { continue }

            let observationTrust = fxClamp(Double(cell.observations) / 48.0, 0.0, 1.0)
            var routerScore = 0.70 * cell.value +
                0.35 * cell.counterfactual -
                0.55 * cell.regret
            if label == .buy {
                routerScore += 0.10 * routingObjective * directionBias
            } else if label == .sell {
                routerScore -= 0.10 * routingObjective * directionBias
            } else {
                routerScore += 0.06 * portfolioObjective * correlationPenalty
            }

            let multiplier = exp(clipSym(contextTrust * observationTrust * routerScore, limit: 1.2))
            output[index] = fxClamp(output[index] * multiplier, 0.0005, 0.9990)
        }

        let denominator = max(output.reduce(0.0, +), 1e-12)
        return output.map { $0 / denominator }
    }

    public static func heuristicStackProbabilities(features: [Double]) -> [Double] {
        let features = normalizedStackFeatures(features)
        var sellProbability = fxClamp(
            0.26 +
            0.31 * stackFeature(features, 2) -
            0.12 * stackFeature(features, 3) +
            0.16 * stackFeature(features, 5) -
            0.07 * stackFeature(features, 10) +
            0.08 * stackFeature(features, 20) +
            0.08 * stackFeature(features, 21) +
            0.05 * stackFeature(features, 23) +
            0.05 * stackFeature(features, 30) +
            0.04 * stackFeature(features, 34) +
            0.05 * stackFeature(features, 58) +
            0.04 * stackFeature(features, 59) +
            0.04 * stackFeature(features, 61) +
            0.04 * stackFeature(features, 62) -
            0.05 * stackFeature(features, 57) -
            0.04 * stackFeature(features, 63) -
            0.03 * stackFeature(features, 73) +
            0.03 * stackFeature(features, 78) +
            0.03 * stackFeature(features, 80) +
            0.02 * stackFeature(features, 82),
            0.01,
            0.98
        )
        var buyProbability = fxClamp(
            0.26 +
            0.31 * stackFeature(features, 1) -
            0.12 * stackFeature(features, 3) +
            0.16 * stackFeature(features, 4) -
            0.07 * stackFeature(features, 10) +
            0.08 * stackFeature(features, 20) +
            0.08 * stackFeature(features, 21) +
            0.05 * stackFeature(features, 23) -
            0.05 * stackFeature(features, 30) +
            0.04 * stackFeature(features, 34) +
            0.05 * stackFeature(features, 58) +
            0.04 * stackFeature(features, 59) +
            0.04 * stackFeature(features, 61) +
            0.04 * stackFeature(features, 62) -
            0.05 * stackFeature(features, 57) -
            0.04 * stackFeature(features, 63) +
            0.03 * stackFeature(features, 73) +
            0.03 * stackFeature(features, 78) +
            0.03 * stackFeature(features, 80) +
            0.02 * stackFeature(features, 82),
            0.01,
            0.98
        )
        var skipProbability = fxClamp(
            0.18 +
            0.32 * stackFeature(features, 3) +
            0.18 * stackFeature(features, 10) -
            0.08 * stackFeature(features, 8) -
            0.07 * stackFeature(features, 20) -
            0.07 * stackFeature(features, 21) +
            0.08 * stackFeature(features, 31) -
            0.04 * stackFeature(features, 28) -
            0.03 * stackFeature(features, 32) +
            0.08 * stackFeature(features, 57) +
            0.07 * stackFeature(features, 60) +
            0.05 * stackFeature(features, 63) -
            0.05 * stackFeature(features, 62) -
            0.05 * stackFeature(features, 80) -
            0.05 * stackFeature(features, 81) -
            0.04 * stackFeature(features, 82) -
            0.03 * stackFeature(features, 77),
            0.01,
            0.98
        )
        let portfolioObjective = stackPortfolioObjective(features: features)
        let routingObjective = stackRoutingObjective(features: features)
        sellProbability = fxClamp(
            sellProbability + 0.05 * portfolioObjective - 0.06 * routingObjective * fxClamp(stackFeature(features, 6), -1.0, 1.0),
            0.01,
            0.98
        )
        buyProbability = fxClamp(
            buyProbability + 0.05 * portfolioObjective + 0.06 * routingObjective * fxClamp(stackFeature(features, 6), -1.0, 1.0),
            0.01,
            0.98
        )
        skipProbability = fxClamp(
            skipProbability - 0.05 * portfolioObjective - 0.04 * routingObjective * fxClamp(stackFeature(features, 19), -1.0, 1.0),
            0.01,
            0.98
        )
        return normalizedProbabilities([sellProbability, buyProbability, skipProbability])
    }

    public static func legacyTanh(_ value: Double) -> Double {
        if value > 18.0 { return 1.0 }
        if value < -18.0 { return -1.0 }
        let exp2 = exp(2.0 * value)
        return (exp2 - 1.0) / (exp2 + 1.0)
    }

    private static func vectorValue(_ values: [Double], _ index: Int, default defaultValue: Double) -> Double {
        guard index >= 0, index < values.count else { return defaultValue }
        return fxSafeFinite(values[index], fallback: defaultValue)
    }

    private static func stackNetworkForward(
        _ state: StackNetworkState,
        features: [Double]
    ) -> (hidden: [Double], rawProbabilities: [Double]) {
        let features = normalizedStackFeatures(features)
        var hidden = Array(repeating: 0.0, count: FXDataEngineConstants.stackHidden)
        for hiddenIndex in 0..<FXDataEngineConstants.stackHidden {
            var z = state.hiddenBias[hiddenIndex]
            for featureIndex in 0..<FXDataEngineConstants.stackFeatures {
                z += state.inputWeights[hiddenIndex][featureIndex] * features[featureIndex]
            }
            hidden[hiddenIndex] = legacyTanh(z)
        }

        var logits = Array(repeating: 0.0, count: LabelClass.allCases.count)
        for label in LabelClass.allCases {
            let labelIndex = label.rawValue
            var z = state.outputBias[labelIndex]
            for hiddenIndex in 0..<FXDataEngineConstants.stackHidden {
                z += state.outputWeights[labelIndex][hiddenIndex] * hidden[hiddenIndex]
            }
            logits[labelIndex] = z
        }
        return (hidden, softmax3(logits))
    }

    private static func softmax3(_ logits: [Double]) -> [Double] {
        let sell = vectorValue(logits, LabelClass.sell.rawValue, default: 0.0)
        let buy = vectorValue(logits, LabelClass.buy.rawValue, default: 0.0)
        let skip = vectorValue(logits, LabelClass.skip.rawValue, default: 0.0)
        let maximum = max(sell, buy, skip)
        let eSell = exp(sell - maximum)
        let eBuy = exp(buy - maximum)
        let eSkip = exp(skip - maximum)
        let denominator = max(eSell + eBuy + eSkip, 1.0)
        return [eSell / denominator, eBuy / denominator, eSkip / denominator]
    }

    private static func normalizedProbabilities(_ probabilities: [Double]) -> [Double] {
        var output = [
            vectorValue(probabilities, LabelClass.sell.rawValue, default: 0.3333),
            vectorValue(probabilities, LabelClass.buy.rawValue, default: 0.3333),
            vectorValue(probabilities, LabelClass.skip.rawValue, default: 0.3334)
        ]
        let denominator = output.reduce(0.0, +)
        guard denominator > 0.0 else { return [0.3333, 0.3333, 0.3334] }
        output[0] /= denominator
        output[1] /= denominator
        output[2] /= denominator
        return output
    }

    private static func normalizedStackFeatures(_ features: [Double]) -> [Double] {
        var output = Array(repeating: 0.0, count: FXDataEngineConstants.stackFeatures)
        for index in 0..<min(features.count, output.count) {
            output[index] = fxSafeFinite(features[index])
        }
        return output
    }

    private static func normalizedCells(_ cells: [StackRouterActionCell]) -> [StackRouterActionCell] {
        var output = Array(repeating: StackRouterActionCell(), count: LabelClass.allCases.count)
        for index in 0..<min(cells.count, output.count) {
            output[index] = cells[index]
        }
        return output
    }

    private static func clipSym(_ value: Double, limit: Double) -> Double {
        let limit = max(fxSafeFinite(limit), 0.0)
        return fxClamp(value, -limit, limit)
    }
}
