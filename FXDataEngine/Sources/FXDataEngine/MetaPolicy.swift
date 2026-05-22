import Foundation

public struct MetaPolicyDecision: Codable, Hashable, Sendable {
    public var tradeProbability: Double
    public var noTradeProbability: Double
    public var enterProbability: Double
    public var exitProbability: Double
    public var directionBias: Double
    public var sizeMultiplier: Double
    public var holdQuality: Double
    public var expectedUtility: Double
    public var confidence: Double
    public var portfolioFit: Double
    public var capitalEfficiency: Double
    public var addProbability: Double
    public var reduceProbability: Double
    public var tightenProbability: Double
    public var timeoutProbability: Double
    public var action: PolicyLifecycleAction

    public init(
        tradeProbability: Double = 0.0,
        noTradeProbability: Double = 1.0,
        enterProbability: Double = 0.0,
        exitProbability: Double = 0.0,
        directionBias: Double = 0.0,
        sizeMultiplier: Double = 1.0,
        holdQuality: Double = 0.0,
        expectedUtility: Double = 0.0,
        confidence: Double = 0.0,
        portfolioFit: Double = 0.0,
        capitalEfficiency: Double = 0.0,
        addProbability: Double = 0.0,
        reduceProbability: Double = 0.0,
        tightenProbability: Double = 0.0,
        timeoutProbability: Double = 0.0,
        action: PolicyLifecycleAction = .noTrade
    ) {
        self.tradeProbability = fxClamp(tradeProbability, 0.0, 1.0)
        self.noTradeProbability = fxClamp(noTradeProbability, 0.0, 1.0)
        self.enterProbability = fxClamp(enterProbability, 0.0, 1.0)
        self.exitProbability = fxClamp(exitProbability, 0.0, 1.0)
        self.directionBias = fxClamp(directionBias, -1.0, 1.0)
        self.sizeMultiplier = fxClamp(sizeMultiplier, 0.0, 3.0)
        self.holdQuality = fxClamp(holdQuality, 0.0, 1.0)
        self.expectedUtility = fxSafeFinite(expectedUtility)
        self.confidence = fxClamp(confidence, 0.0, 1.0)
        self.portfolioFit = fxClamp(portfolioFit, 0.0, 1.0)
        self.capitalEfficiency = fxClamp(capitalEfficiency, 0.0, 1.0)
        self.addProbability = fxClamp(addProbability, 0.0, 1.0)
        self.reduceProbability = fxClamp(reduceProbability, 0.0, 1.0)
        self.tightenProbability = fxClamp(tightenProbability, 0.0, 1.0)
        self.timeoutProbability = fxClamp(timeoutProbability, 0.0, 1.0)
        self.action = action
    }
}

public struct MetaPolicyNetworkState: Codable, Hashable, Sendable {
    public var inputWeights: [[Double]]
    public var hiddenBias: [Double]
    public var tradeWeights: [Double]
    public var directionWeights: [Double]
    public var sizeWeights: [Double]
    public var holdWeights: [Double]
    public var tradeBias: Double
    public var directionBias: Double
    public var sizeBias: Double
    public var holdBias: Double
    public var ready: Bool
    public var observations: Int

    public init(
        inputWeights: [[Double]] = [],
        hiddenBias: [Double] = [],
        tradeWeights: [Double] = [],
        directionWeights: [Double] = [],
        sizeWeights: [Double] = [],
        holdWeights: [Double] = [],
        tradeBias: Double = 0.0,
        directionBias: Double = 0.0,
        sizeBias: Double = 0.0,
        holdBias: Double = 0.0,
        ready: Bool = false,
        observations: Int = 0
    ) {
        self.inputWeights = Self.normalizedMatrix(
            inputWeights,
            rows: FXDataEngineConstants.policyHidden,
            columns: FXDataEngineConstants.policyFeatures
        )
        self.hiddenBias = Self.normalizedVector(hiddenBias, count: FXDataEngineConstants.policyHidden)
        self.tradeWeights = Self.normalizedVector(tradeWeights, count: FXDataEngineConstants.policyHidden)
        self.directionWeights = Self.normalizedVector(directionWeights, count: FXDataEngineConstants.policyHidden)
        self.sizeWeights = Self.normalizedVector(sizeWeights, count: FXDataEngineConstants.policyHidden)
        self.holdWeights = Self.normalizedVector(holdWeights, count: FXDataEngineConstants.policyHidden)
        self.tradeBias = fxSafeFinite(tradeBias)
        self.directionBias = fxSafeFinite(directionBias)
        self.sizeBias = fxSafeFinite(sizeBias)
        self.holdBias = fxSafeFinite(holdBias)
        self.ready = ready
        self.observations = min(max(observations, 0), MetaPolicyTools.observationCap)
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

public struct MetaPolicyPrediction: Codable, Hashable, Sendable {
    public var decision: MetaPolicyDecision
    public var heuristicDecision: MetaPolicyDecision
    public var learnedTradeProbability: Double?
    public var learnedDirectionBias: Double?
    public var learnedSizeMultiplier: Double?
    public var learnedHoldQuality: Double?
    public var hidden: [Double]

    public init(
        decision: MetaPolicyDecision,
        heuristicDecision: MetaPolicyDecision,
        learnedTradeProbability: Double? = nil,
        learnedDirectionBias: Double? = nil,
        learnedSizeMultiplier: Double? = nil,
        learnedHoldQuality: Double? = nil,
        hidden: [Double] = []
    ) {
        self.decision = decision
        self.heuristicDecision = heuristicDecision
        self.learnedTradeProbability = learnedTradeProbability.map { fxClamp($0, 0.0, 1.0) }
        self.learnedDirectionBias = learnedDirectionBias.map { fxClamp($0, -1.0, 1.0) }
        self.learnedSizeMultiplier = learnedSizeMultiplier.map { fxClamp($0, 0.25, 1.60) }
        self.learnedHoldQuality = learnedHoldQuality.map { fxClamp($0, 0.0, 1.0) }
        var resolvedHidden = Array(repeating: 0.0, count: FXDataEngineConstants.policyHidden)
        for index in 0..<min(hidden.count, resolvedHidden.count) {
            resolvedHidden[index] = fxSafeFinite(hidden[index])
        }
        self.hidden = resolvedHidden
    }
}

public struct MetaPolicyFeatureInputs: Codable, Hashable, Sendable {
    public var stackFeatures: [Double]
    public var tradeGate: Double
    public var tradeEdgePoints: Double
    public var expectedMovePoints: Double
    public var minMovePoints: Double
    public var macroQuality: Double
    public var contextQuality: Double
    public var contextStrength: Double
    public var foundationTrust: Double
    public var foundationDirectionBias: Double
    public var studentTrust: Double
    public var analogSimilarity: Double
    public var analogQuality: Double
    public var regime: RegimeGraphQuery
    public var deployment: LiveDeploymentProfile
    public var portfolioPressureHint: Double

    public init(
        stackFeatures: [Double] = [],
        tradeGate: Double = 0.0,
        tradeEdgePoints: Double = 0.0,
        expectedMovePoints: Double = 0.0,
        minMovePoints: Double = 0.0,
        macroQuality: Double = 0.0,
        contextQuality: Double = 0.0,
        contextStrength: Double = 0.0,
        foundationTrust: Double = 0.0,
        foundationDirectionBias: Double = 0.0,
        studentTrust: Double = 0.0,
        analogSimilarity: Double = 0.0,
        analogQuality: Double = 0.0,
        regime: RegimeGraphQuery = RegimeGraphQuery(),
        deployment: LiveDeploymentProfile = LiveDeploymentProfile(),
        portfolioPressureHint: Double = 0.0
    ) {
        self.stackFeatures = stackFeatures
        self.tradeGate = tradeGate
        self.tradeEdgePoints = tradeEdgePoints
        self.expectedMovePoints = expectedMovePoints
        self.minMovePoints = minMovePoints
        self.macroQuality = macroQuality
        self.contextQuality = contextQuality
        self.contextStrength = contextStrength
        self.foundationTrust = foundationTrust
        self.foundationDirectionBias = foundationDirectionBias
        self.studentTrust = studentTrust
        self.analogSimilarity = analogSimilarity
        self.analogQuality = analogQuality
        self.regime = regime
        self.deployment = deployment
        self.portfolioPressureHint = portfolioPressureHint
    }
}

public enum MetaPolicyTools {
    public static let observationCap = 200_000

    public static func stackFeature(_ features: [Double], _ index: Int, default defaultValue: Double = 0.0) -> Double {
        guard index >= 0, index < features.count else { return defaultValue }
        return fxSafeFinite(features[index], fallback: defaultValue)
    }

    public static func buildPolicyFeatures(_ input: MetaPolicyFeatureInputs) -> [Double] {
        var features = Array(repeating: 0.0, count: FXDataEngineConstants.policyFeatures)
        let deployment = input.deployment
        let regime = input.regime

        let mm = max(fxSafeFinite(input.minMovePoints), 0.10)
        let analogWeight = fxClamp(deployment.analogWeight, 0.0, 0.80)
        let transitionWeight = fxClamp(deployment.regimeTransitionWeight, 0.0, 1.0)
        let macroFloor = fxClamp(deployment.macroQualityFloor, 0.0, 1.0)
        let teacherGain = fxClamp(deployment.teacherSignalGain, 0.40, 1.80)
        let studentGain = fxClamp(deployment.studentSignalGain, 0.40, 1.80)
        let foundationGain = fxClamp(deployment.foundationQualityGain, 0.40, 1.80)
        let macroGain = fxClamp(deployment.macroStateGain, 0.40, 1.80)
        let lifecycleGain = fxClamp(deployment.policyLifecycleGain, 0.40, 1.80)
        let stack = input.stackFeatures

        features[0] = 1.0
        features[1] = fxClamp(0.5 + 0.5 * stackFeature(stack, 1), 0.0, 1.0)
        features[2] = fxClamp(0.5 + 0.5 * stackFeature(stack, 2), 0.0, 1.0)
        features[3] = fxClamp(0.5 + 0.5 * stackFeature(stack, 3), 0.0, 1.0)
        features[4] = fxClamp(input.expectedMovePoints / mm, 0.0, 8.0) / 4.0 - 0.5
        features[5] = fxClamp(input.tradeEdgePoints / mm, -4.0, 4.0) / 4.0
        features[6] = fxClamp(stackFeature(stack, 20), 0.0, 1.0)
        features[7] = fxClamp(stackFeature(stack, 21), 0.0, 1.0)
        features[8] = fxClamp(input.tradeGate, 0.0, 1.0)
        features[9] = fxClamp(stackFeature(stack, 80), 0.0, 1.0)
        features[10] = fxClamp(stackFeature(stack, 81), 0.0, 1.0)
        features[11] = fxClamp(stackFeature(stack, 82), 0.0, 1.0)
        features[12] = fxClamp(stackFeature(stack, 83), 0.0, 1.0)
        features[13] = 1.0 - fxClamp(stackFeature(stack, 49, default: 1.0), 0.0, 1.0)
        features[14] = 1.0 - fxClamp(stackFeature(stack, 50, default: 1.0), 0.0, 1.0)
        features[15] = fxClamp(input.macroQuality * (0.80 + 0.20 * macroGain), 0.0, 1.0)
        features[16] = fxClamp(0.5 + 0.5 * input.contextQuality, 0.0, 1.0)
        features[17] = fxClamp(input.contextStrength / 3.0, 0.0, 1.0)
        features[18] = fxClamp(
            input.foundationTrust *
                (0.82 + 0.50 * fxClamp(deployment.foundationWeight, 0.0, 0.90)) *
                (0.75 + 0.25 * foundationGain),
            0.0,
            1.0
        )
        features[19] = fxClamp(input.foundationDirectionBias, -1.0, 1.0)
        features[20] = fxClamp(input.studentTrust * (0.78 + 0.22 * studentGain), 0.0, 1.0)
        features[21] = fxClamp(input.analogSimilarity * (0.80 + 0.60 * analogWeight), 0.0, 1.0)
        features[22] = fxClamp(input.analogQuality * (0.80 + 0.60 * analogWeight), 0.0, 1.0)
        features[23] = fxClamp(regime.persistence, 0.0, 1.0)
        features[24] = fxClamp(regime.transitionConfidence * (0.65 + 0.35 * transitionWeight), 0.0, 1.0)
        features[25] = fxClamp(regime.instability * (0.70 + 0.30 * transitionWeight), 0.0, 1.0)
        features[26] = fxClamp(regime.edgeBias, -1.0, 1.0)
        features[27] = fxClamp(regime.qualityBias, 0.0, 1.0)
        features[28] = fxClamp(
            0.60 * regime.macroAlignment +
                0.25 * fxClamp(input.macroQuality - macroFloor + 0.50, 0.0, 1.0) +
                0.15 * (1.0 - fxClamp(max(macroFloor - input.macroQuality, 0.0), 0.0, 1.0)),
            0.0,
            1.0
        )
        features[29] = fxClamp((deployment.teacherWeight - 0.50) * teacherGain, -0.75, 0.75)
        features[30] = fxClamp(
            (0.65 * deployment.policyTradeFloor + 0.35 * macroFloor) * (0.80 + 0.20 * lifecycleGain),
            0.0,
            1.0
        )
        features[31] = fxClamp(input.portfolioPressureHint, 0.0, 1.5) / 1.5
        return features
    }

    public static func predictPolicy(
        _ state: MetaPolicyNetworkState,
        features: [Double],
        deployment: LiveDeploymentProfile = LiveDeploymentProfile()
    ) -> MetaPolicyPrediction {
        let features = normalizedPolicyFeatures(features)
        let heuristic = heuristicPolicyDecision(features: features, deployment: deployment)
        guard state.ready else {
            return MetaPolicyPrediction(decision: heuristic.decision, heuristicDecision: heuristic.decision)
        }

        let forward = policyForward(state, features: features)
        let mix = fxClamp(Double(max(state.observations, 0)) / 160.0, 0.18, 0.82)
        var tradeProbability = fxClamp(
            (1.0 - mix) * heuristic.tradeProbability + mix * forward.tradeProbability,
            0.0,
            1.0
        )
        let directionBias = fxClamp(
            (1.0 - mix) * heuristic.directionBias + mix * forward.directionBias,
            -1.0,
            1.0
        )
        var holdQuality = fxClamp(
            (1.0 - mix) * heuristic.holdQuality + mix * forward.holdQuality,
            0.0,
            1.0
        )
        var sizeMultiplier = fxClamp(
            (1.0 - mix) * heuristic.sizeMultiplier + mix * forward.sizeMultiplier,
            0.25,
            1.60
        )
        let portfolioFit = fxClamp(
            0.54 * (1.0 - policyFeature(features, 31)) +
                0.18 * policyFeature(features, 16) +
                0.12 * policyFeature(features, 17) +
                0.16 * policyFeature(features, 27),
            0.0,
            1.0
        )
        let capitalEfficiency = fxClamp(
            0.28 +
                0.26 * max(policyFeature(features, 5), 0.0) +
                0.18 * policyFeature(features, 4) +
                0.14 * holdQuality +
                0.14 * tradeProbability,
            0.0,
            1.0
        )
        var expectedUtility = fxClamp(
            0.60 * policyFeature(features, 5) +
                0.20 * tradeProbability +
                0.20 * holdQuality,
            -1.0,
            1.0
        )
        var confidence = fxClamp(
            0.35 * tradeProbability +
                0.25 * holdQuality +
                0.20 * (1.0 - policyFeature(features, 31)) +
                0.20 * policyFeature(features, 24),
            0.0,
            1.0
        )

        let macroGuard = fxClamp(1.0 - 0.55 * heuristic.macroShortfall, 0.15, 1.0)
        let transitionGuard = fxClamp(1.0 - 0.35 * heuristic.transitionPressure, 0.35, 1.0)
        tradeProbability = fxClamp(tradeProbability * macroGuard * transitionGuard, 0.0, 1.0)
        holdQuality = fxClamp(holdQuality * macroGuard * transitionGuard, 0.0, 1.0)
        sizeMultiplier = fxClamp(
            sizeMultiplier *
                fxClamp(1.0 - 0.20 * heuristic.macroShortfall - 0.15 * heuristic.transitionPressure, 0.40, 1.0),
            0.25,
            1.60
        )
        expectedUtility = fxClamp(
            expectedUtility - 0.25 * heuristic.macroShortfall - 0.18 * heuristic.transitionPressure,
            -1.0,
            1.0
        )
        confidence = fxClamp(
            confidence *
                fxClamp(1.0 - 0.30 * heuristic.macroShortfall - 0.20 * heuristic.transitionPressure, 0.20, 1.0),
            0.0,
            1.0
        )

        let noTradeProbability = fxClamp(
            0.58 * (1.0 - tradeProbability) +
                0.16 * heuristic.macroShortfall +
                0.14 * heuristic.transitionPressure +
                0.12 * policyFeature(features, 31),
            0.0,
            1.0
        )
        let enterProbability = fxClamp(
            tradeProbability *
                (0.46 + 0.28 * portfolioFit + 0.26 * capitalEfficiency) *
                (1.0 - 0.40 * noTradeProbability),
            0.0,
            1.0
        )
        let exitProbability = fxClamp(
            0.20 +
                0.18 * (1.0 - holdQuality) +
                0.16 * heuristic.macroShortfall +
                0.14 * heuristic.transitionPressure +
                0.14 * policyFeature(features, 31) +
                0.18 * max(-policyFeature(features, 5), 0.0),
            0.0,
            1.0
        )
        let addProbability = fxClamp(
            0.40 * enterProbability +
                0.18 * holdQuality +
                0.18 * capitalEfficiency +
                0.14 * portfolioFit +
                0.10 * max(expectedUtility, 0.0) -
                0.22 * noTradeProbability,
            0.0,
            1.0
        )
        let reduceProbability = fxClamp(
            0.32 * noTradeProbability +
                0.28 * exitProbability +
                0.16 * (1.0 - holdQuality) +
                0.14 * heuristic.transitionPressure +
                0.10 * policyFeature(features, 31),
            0.0,
            1.0
        )
        let tightenProbability = fxClamp(
            0.46 * reduceProbability +
                0.20 * heuristic.transitionPressure +
                0.18 * heuristic.macroShortfall +
                0.16 * (1.0 - portfolioFit),
            0.0,
            1.0
        )
        let timeoutProbability = fxClamp(
            0.28 * reduceProbability +
                0.24 * exitProbability +
                0.24 * heuristic.transitionPressure +
                0.24 * heuristic.macroShortfall,
            0.0,
            1.0
        )
        let holdScore = fxClamp(
            0.52 * holdQuality + 0.28 * portfolioFit + 0.20 * confidence,
            0.0,
            1.0
        )
        let action: PolicyLifecycleAction
        if enterProbability > max(noTradeProbability, max(exitProbability, holdScore)),
           enterProbability >= 0.40 {
            action = .enter
        } else if exitProbability > max(noTradeProbability, holdScore), exitProbability >= 0.52 {
            action = .exit
        } else if holdScore > noTradeProbability, holdScore >= 0.50 {
            action = .hold
        } else {
            action = .noTrade
        }

        let decision = MetaPolicyDecision(
            tradeProbability: tradeProbability,
            noTradeProbability: noTradeProbability,
            enterProbability: enterProbability,
            exitProbability: exitProbability,
            directionBias: directionBias,
            sizeMultiplier: sizeMultiplier,
            holdQuality: holdQuality,
            expectedUtility: expectedUtility,
            confidence: confidence,
            portfolioFit: portfolioFit,
            capitalEfficiency: capitalEfficiency,
            addProbability: addProbability,
            reduceProbability: reduceProbability,
            tightenProbability: tightenProbability,
            timeoutProbability: timeoutProbability,
            action: action
        )
        return MetaPolicyPrediction(
            decision: decision,
            heuristicDecision: heuristic.decision,
            learnedTradeProbability: forward.tradeProbability,
            learnedDirectionBias: forward.directionBias,
            learnedSizeMultiplier: forward.sizeMultiplier,
            learnedHoldQuality: forward.holdQuality,
            hidden: forward.hidden
        )
    }

    public static func updatedPolicyNetwork(
        _ state: MetaPolicyNetworkState,
        features: [Double],
        tradeTarget: Double,
        directionTarget: Double,
        sizeTarget: Double,
        holdTarget: Double,
        sampleWeight: Double
    ) -> MetaPolicyNetworkState {
        let features = normalizedPolicyFeatures(features)
        let forward = policyForward(state, features: features)
        let sampleWeight = fxClamp(sampleWeight, 0.20, 8.00)
        let learningRate = fxClamp(
            0.018 / sqrt(1.0 + 0.02 * Double(max(state.observations, 0))),
            0.0015,
            0.018
        )
        let tradeError = fxClamp((tradeTarget - forward.tradeProbability) * sampleWeight, -3.0, 3.0)
        let directionError = fxClamp((directionTarget - forward.directionBias) * sampleWeight, -3.0, 3.0)
        let sizeError = fxClamp(((sizeTarget - forward.sizeMultiplier) / 1.35) * sampleWeight, -3.0, 3.0)
        let holdError = fxClamp((holdTarget - forward.holdQuality) * sampleWeight, -3.0, 3.0)

        let oldTradeWeights = state.tradeWeights
        let oldDirectionWeights = state.directionWeights
        let oldSizeWeights = state.sizeWeights
        let oldHoldWeights = state.holdWeights

        var tradeBias = state.tradeBias + learningRate * tradeError
        var directionBias = state.directionBias + learningRate * directionError
        var sizeBias = state.sizeBias + learningRate * sizeError
        var holdBias = state.holdBias + learningRate * holdError
        tradeBias = fxSafeFinite(tradeBias)
        directionBias = fxSafeFinite(directionBias)
        sizeBias = fxSafeFinite(sizeBias)
        holdBias = fxSafeFinite(holdBias)

        var tradeWeights = state.tradeWeights
        var directionWeights = state.directionWeights
        var sizeWeights = state.sizeWeights
        var holdWeights = state.holdWeights
        for hiddenIndex in 0..<FXDataEngineConstants.policyHidden {
            tradeWeights[hiddenIndex] += learningRate *
                (tradeError * forward.hidden[hiddenIndex] - 0.0005 * tradeWeights[hiddenIndex])
            directionWeights[hiddenIndex] += learningRate *
                (directionError * forward.hidden[hiddenIndex] - 0.0005 * directionWeights[hiddenIndex])
            sizeWeights[hiddenIndex] += learningRate *
                (sizeError * forward.hidden[hiddenIndex] - 0.0005 * sizeWeights[hiddenIndex])
            holdWeights[hiddenIndex] += learningRate *
                (holdError * forward.hidden[hiddenIndex] - 0.0005 * holdWeights[hiddenIndex])
        }

        var hiddenBias = state.hiddenBias
        var inputWeights = state.inputWeights
        for hiddenIndex in 0..<FXDataEngineConstants.policyHidden {
            let back = oldTradeWeights[hiddenIndex] * tradeError +
                oldDirectionWeights[hiddenIndex] * directionError +
                oldSizeWeights[hiddenIndex] * sizeError +
                oldHoldWeights[hiddenIndex] * holdError
            let deltaHidden = fxClamp(back * (1.0 - forward.hidden[hiddenIndex] * forward.hidden[hiddenIndex]), -3.0, 3.0)
            hiddenBias[hiddenIndex] += learningRate * deltaHidden
            for featureIndex in 0..<FXDataEngineConstants.policyFeatures {
                let regularization = featureIndex == 0 ? 0.0 : 0.0004 * inputWeights[hiddenIndex][featureIndex]
                inputWeights[hiddenIndex][featureIndex] += learningRate *
                    (deltaHidden * features[featureIndex] - regularization)
            }
        }

        return MetaPolicyNetworkState(
            inputWeights: inputWeights,
            hiddenBias: hiddenBias,
            tradeWeights: tradeWeights,
            directionWeights: directionWeights,
            sizeWeights: sizeWeights,
            holdWeights: holdWeights,
            tradeBias: tradeBias,
            directionBias: directionBias,
            sizeBias: sizeBias,
            holdBias: holdBias,
            ready: true,
            observations: min(max(state.observations, 0) + 1, observationCap)
        )
    }

    private static func heuristicPolicyDecision(
        features: [Double],
        deployment: LiveDeploymentProfile
    ) -> (
        decision: MetaPolicyDecision,
        macroShortfall: Double,
        transitionPressure: Double,
        tradeProbability: Double,
        directionBias: Double,
        holdQuality: Double,
        sizeMultiplier: Double
    ) {
        let macroFloor = fxClamp(deployment.macroQualityFloor, 0.0, 1.0)
        let teacherGain = fxClamp(deployment.teacherSignalGain, 0.40, 1.80)
        let studentGain = fxClamp(deployment.studentSignalGain, 0.40, 1.80)
        let foundationGain = fxClamp(deployment.foundationQualityGain, 0.40, 1.80)
        let macroGain = fxClamp(deployment.macroStateGain, 0.40, 1.80)
        let lifecycleGain = fxClamp(deployment.policyLifecycleGain, 0.40, 1.80)
        let macroShortfall = fxClamp(macroFloor - policyFeature(features, 15), 0.0, 1.0)
        let transitionPressure = fxClamp(deployment.regimeTransitionWeight, 0.0, 1.0) * policyFeature(features, 25)
        let analogBonus = 0.5 * (policyFeature(features, 21) + policyFeature(features, 22))

        let tradeProbability = fxClamp(
            0.16 +
                0.15 * policyFeature(features, 1) +
                0.15 * policyFeature(features, 2) -
                0.12 * policyFeature(features, 3) +
                0.10 * policyFeature(features, 5) +
                0.09 * policyFeature(features, 6) +
                0.09 * policyFeature(features, 7) +
                0.10 * policyFeature(features, 8) +
                0.08 * policyFeature(features, 9) +
                0.06 * policyFeature(features, 10) +
                0.06 * policyFeature(features, 11) +
                0.04 * policyFeature(features, 12) +
                0.06 * policyFeature(features, 15) +
                0.05 * policyFeature(features, 18) +
                0.04 * policyFeature(features, 20) +
                0.05 * analogBonus +
                0.03 * policyFeature(features, 23) -
                0.08 * transitionPressure -
                0.14 * macroShortfall * macroGain -
                0.10 * policyFeature(features, 31) +
                0.08 * policyFeature(features, 29) * teacherGain +
                0.08 * policyFeature(features, 30) * lifecycleGain +
                0.04 * policyFeature(features, 18) * foundationGain +
                0.04 * policyFeature(features, 20) * studentGain,
            0.01,
            0.99
        )
        let directionBias = fxClamp(
            0.70 * (policyFeature(features, 1, default: 0.5) - policyFeature(features, 2, default: 0.5)) +
                0.30 * policyFeature(features, 5) +
                0.12 * policyFeature(features, 19) +
                0.10 * policyFeature(features, 26),
            -1.0,
            1.0
        )
        let holdQuality = fxClamp(
            0.18 +
                0.18 * policyFeature(features, 13) +
                0.18 * policyFeature(features, 14) +
                0.14 * policyFeature(features, 9) +
                0.12 * policyFeature(features, 11) +
                0.10 * policyFeature(features, 12) +
                0.08 * policyFeature(features, 27) -
                0.08 * transitionPressure -
                0.12 * macroShortfall -
                0.08 * policyFeature(features, 31),
            0.0,
            1.0
        )
        var sizeMultiplier = fxClamp(
            0.42 +
                0.26 * tradeProbability +
                0.18 * max(policyFeature(features, 5), 0.0) +
                0.12 * policyFeature(features, 7) +
                0.10 * policyFeature(features, 27) -
                0.06 * transitionPressure -
                0.10 * macroShortfall -
                0.18 * policyFeature(features, 31),
            0.25,
            1.50
        )
        sizeMultiplier *= fxClamp(deployment.policySizeBias, 0.40, 1.60) *
            fxClamp(0.80 + 0.20 * lifecycleGain, 0.40, 1.80)
        sizeMultiplier = fxClamp(sizeMultiplier, 0.25, 1.60)

        let noTradeProbability = fxClamp(
            0.60 - 0.45 * tradeProbability +
                0.20 * macroShortfall +
                0.18 * transitionPressure +
                0.12 * policyFeature(features, 31),
            0.0,
            1.0
        )
        let portfolioFit = fxClamp(
            0.62 * (1.0 - policyFeature(features, 31)) +
                0.20 * policyFeature(features, 16) +
                0.18 * policyFeature(features, 27),
            0.0,
            1.0
        )
        let capitalEfficiency = fxClamp(
            0.34 +
                0.30 * max(policyFeature(features, 5), 0.0) +
                0.22 * policyFeature(features, 4) +
                0.14 * holdQuality,
            0.0,
            1.0
        )
        let enterProbability = fxClamp(
            tradeProbability *
                (0.52 + 0.24 * portfolioFit + 0.24 * capitalEfficiency) *
                (1.0 - 0.45 * noTradeProbability),
            0.0,
            1.0
        )
        let exitProbability = fxClamp(
            0.24 +
                0.22 * (1.0 - holdQuality) +
                0.18 * macroShortfall +
                0.16 * transitionPressure +
                0.20 * policyFeature(features, 31),
            0.0,
            1.0
        )
        let expectedUtility = fxClamp(
            0.55 * policyFeature(features, 5) +
                0.25 * tradeProbability +
                0.20 * holdQuality,
            -1.0,
            1.0
        )
        let confidence = fxClamp(
            0.45 * tradeProbability +
                0.30 * holdQuality +
                0.25 * (1.0 - policyFeature(features, 31)),
            0.0,
            1.0
        )
        let addProbability = fxClamp(
            0.42 * enterProbability +
                0.20 * holdQuality +
                0.20 * capitalEfficiency +
                0.18 * portfolioFit -
                0.26 * noTradeProbability,
            0.0,
            1.0
        )
        let reduceProbability = fxClamp(
            0.34 * noTradeProbability +
                0.28 * exitProbability +
                0.20 * (1.0 - holdQuality) +
                0.18 * policyFeature(features, 31),
            0.0,
            1.0
        )
        let tightenProbability = fxClamp(
            0.44 * reduceProbability +
                0.22 * transitionPressure +
                0.18 * macroShortfall +
                0.16 * (1.0 - portfolioFit),
            0.0,
            1.0
        )
        let timeoutProbability = fxClamp(
            0.32 * reduceProbability +
                0.26 * transitionPressure +
                0.20 * macroShortfall +
                0.22 * (1.0 - holdQuality),
            0.0,
            1.0
        )
        let action: PolicyLifecycleAction
        if enterProbability > max(noTradeProbability, exitProbability), enterProbability >= 0.42 {
            action = .enter
        } else if exitProbability > noTradeProbability, exitProbability >= 0.55 {
            action = .exit
        } else if holdQuality >= 0.55, tradeProbability >= 0.40 {
            action = .hold
        } else {
            action = .noTrade
        }

        return (
            MetaPolicyDecision(
                tradeProbability: tradeProbability,
                noTradeProbability: noTradeProbability,
                enterProbability: enterProbability,
                exitProbability: exitProbability,
                directionBias: directionBias,
                sizeMultiplier: sizeMultiplier,
                holdQuality: holdQuality,
                expectedUtility: expectedUtility,
                confidence: confidence,
                portfolioFit: portfolioFit,
                capitalEfficiency: capitalEfficiency,
                addProbability: addProbability,
                reduceProbability: reduceProbability,
                tightenProbability: tightenProbability,
                timeoutProbability: timeoutProbability,
                action: action
            ),
            macroShortfall,
            transitionPressure,
            tradeProbability,
            directionBias,
            holdQuality,
            sizeMultiplier
        )
    }

    private static func policyForward(
        _ state: MetaPolicyNetworkState,
        features: [Double]
    ) -> (
        hidden: [Double],
        tradeProbability: Double,
        directionBias: Double,
        sizeMultiplier: Double,
        holdQuality: Double
    ) {
        let features = normalizedPolicyFeatures(features)
        var hidden = Array(repeating: 0.0, count: FXDataEngineConstants.policyHidden)
        for hiddenIndex in 0..<FXDataEngineConstants.policyHidden {
            var z = state.hiddenBias[hiddenIndex]
            for featureIndex in 0..<FXDataEngineConstants.policyFeatures {
                z += state.inputWeights[hiddenIndex][featureIndex] * features[featureIndex]
            }
            hidden[hiddenIndex] = StackerTools.legacyTanh(z)
        }

        var tradeZ = state.tradeBias
        var directionZ = state.directionBias
        var sizeZ = state.sizeBias
        var holdZ = state.holdBias
        for hiddenIndex in 0..<FXDataEngineConstants.policyHidden {
            tradeZ += state.tradeWeights[hiddenIndex] * hidden[hiddenIndex]
            directionZ += state.directionWeights[hiddenIndex] * hidden[hiddenIndex]
            sizeZ += state.sizeWeights[hiddenIndex] * hidden[hiddenIndex]
            holdZ += state.holdWeights[hiddenIndex] * hidden[hiddenIndex]
        }
        return (
            hidden,
            StackerTools.legacySigmoid(tradeZ),
            StackerTools.legacyTanh(directionZ),
            fxClamp(0.25 + 1.35 * StackerTools.legacySigmoid(sizeZ), 0.25, 1.60),
            StackerTools.legacySigmoid(holdZ)
        )
    }

    private static func policyFeature(_ features: [Double], _ index: Int, default defaultValue: Double = 0.0) -> Double {
        guard index >= 0, index < features.count else { return defaultValue }
        return fxSafeFinite(features[index], fallback: defaultValue)
    }

    private static func normalizedPolicyFeatures(_ features: [Double]) -> [Double] {
        var output = Array(repeating: 0.0, count: FXDataEngineConstants.policyFeatures)
        for index in 0..<min(features.count, output.count) {
            output[index] = fxSafeFinite(features[index])
        }
        return output
    }
}
