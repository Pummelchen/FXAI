import Foundation

public struct MetaPendingReplayOutcomeConfig: Codable, Hashable, Sendable {
    public var currentBarIndex: Int?
    public var priceCostPoints: Double
    public var commissionPoints: Double
    public var baseCostBufferPoints: Double
    public var evThresholdPoints: Double
    public var executionProfile: ExecutionProfile
    public var tradeKillerMinutes: Int?
    public var currentRegimeID: Int
    public var currentMacroQuality: Double

    public init(
        currentBarIndex: Int? = nil,
        priceCostPoints: Double = 0.0,
        commissionPoints: Double = 0.0,
        baseCostBufferPoints: Double = 0.0,
        evThresholdPoints: Double = 0.0,
        executionProfile: ExecutionProfile = ExecutionProfile(),
        tradeKillerMinutes: Int? = nil,
        currentRegimeID: Int = 0,
        currentMacroQuality: Double = 0.0
    ) {
        self.currentBarIndex = currentBarIndex
        self.priceCostPoints = max(0.0, fxSafeFinite(priceCostPoints))
        self.commissionPoints = max(0.0, fxSafeFinite(commissionPoints))
        self.baseCostBufferPoints = max(0.0, fxSafeFinite(baseCostBufferPoints))
        self.evThresholdPoints = max(0.0, fxSafeFinite(evThresholdPoints))
        self.executionProfile = executionProfile
        self.tradeKillerMinutes = tradeKillerMinutes
        self.currentRegimeID = Int(fxClamp(
            Double(currentRegimeID),
            0.0,
            Double(FXDataEngineConstants.pluginRegimeBuckets - 1)
        ))
        self.currentMacroQuality = fxClamp(currentMacroQuality, 0.0, 1.0)
    }
}

public struct MetaStackPendingReplayTargets: Codable, Hashable, Sendable {
    public var sampleWeight: Double
    public var qualityScore: Double
    public var tradeTarget: Bool
    public var predictedProbabilities: [Double]

    public init(
        sampleWeight: Double,
        qualityScore: Double,
        tradeTarget: Bool,
        predictedProbabilities: [Double]
    ) {
        self.sampleWeight = max(0.0, fxSafeFinite(sampleWeight))
        self.qualityScore = fxClamp(qualityScore, 0.25, 1.75)
        self.tradeTarget = tradeTarget
        self.predictedProbabilities = MetaPendingReplayOutcomeTargets.normalizedProbabilities(predictedProbabilities)
    }
}

public struct MetaPolicyPendingReplayTargets: Codable, Hashable, Sendable {
    public var tradeTarget: Double
    public var directionTarget: Double
    public var sizeTarget: Double
    public var holdTarget: Double
    public var sampleWeight: Double
    public var qualityScore: Double

    public init(
        tradeTarget: Double,
        directionTarget: Double,
        sizeTarget: Double,
        holdTarget: Double,
        sampleWeight: Double,
        qualityScore: Double
    ) {
        self.tradeTarget = fxClamp(tradeTarget, 0.0, 1.0)
        self.directionTarget = fxClamp(directionTarget, -1.0, 1.0)
        self.sizeTarget = fxClamp(sizeTarget, 0.25, 1.60)
        self.holdTarget = fxClamp(holdTarget, 0.0, 1.0)
        self.sampleWeight = fxClamp(sampleWeight, 0.20, 6.0)
        self.qualityScore = fxClamp(qualityScore, 0.0, 2.0)
    }
}

public struct MetaPendingReplayOutcomeTargets: Codable, Hashable, Sendable {
    public var action: MetaPendingReplayOutcomeAction
    public var currentBarIndex: Int
    public var predictionSeriesIndex: Int
    public var horizonMinutes: Int
    public var minMovePoints: Double
    public var label: TripleBarrierLabelResult
    public var realizedEdgePoints: Double
    public var qualityScore: Double
    public var stack: MetaStackPendingReplayTargets
    public var policy: MetaPolicyPendingReplayTargets
    public var horizonPolicyReward: Double

    public var labelClass: LabelClass {
        label.labelClass
    }

    public init(
        action: MetaPendingReplayOutcomeAction,
        currentBarIndex: Int,
        predictionSeriesIndex: Int,
        horizonMinutes: Int,
        minMovePoints: Double,
        label: TripleBarrierLabelResult,
        realizedEdgePoints: Double,
        qualityScore: Double,
        stack: MetaStackPendingReplayTargets,
        policy: MetaPolicyPendingReplayTargets,
        horizonPolicyReward: Double
    ) {
        self.action = action
        self.currentBarIndex = max(0, currentBarIndex)
        self.predictionSeriesIndex = max(0, predictionSeriesIndex)
        self.horizonMinutes = HorizonTools.clampHorizon(horizonMinutes)
        self.minMovePoints = max(0.0, fxSafeFinite(minMovePoints))
        self.label = label
        self.realizedEdgePoints = fxSafeFinite(realizedEdgePoints)
        self.qualityScore = fxClamp(qualityScore, 0.0, 2.0)
        self.stack = stack
        self.policy = policy
        self.horizonPolicyReward = fxClamp(horizonPolicyReward, -2.0, 6.0)
    }

    public static func normalizedProbabilities(_ probabilities: [Double]) -> [Double] {
        var output = [
            HorizonTools.value(in: probabilities, index: LabelClass.sell.rawValue, default: 0.0),
            HorizonTools.value(in: probabilities, index: LabelClass.buy.rawValue, default: 0.0),
            HorizonTools.value(in: probabilities, index: LabelClass.skip.rawValue, default: 0.0)
        ].map { fxClamp($0, 0.0, 1.0) }
        let sum = output.reduce(0.0, +)
        guard sum > 1e-12 else { return [0.0, 0.0, 0.0] }
        output[0] /= sum
        output[1] /= sum
        output[2] /= sum
        return output
    }
}

public extension MetaPendingReplayTools {
    static func outcomeTargets(
        for action: MetaPendingReplayOutcomeAction,
        series: M1OHLCVSeries,
        config: MetaPendingReplayOutcomeConfig = MetaPendingReplayOutcomeConfig()
    ) -> MetaPendingReplayOutcomeTargets? {
        guard action.canEvaluate, series.count > 0 else { return nil }
        let currentBarIndex = min(max(config.currentBarIndex ?? (series.count - 1), 0), series.count - 1)
        let predictionSeriesIndex = currentBarIndex - action.predictionIndex
        let horizon = HorizonTools.clampHorizon(action.entry.horizonMinutes)
        guard predictionSeriesIndex >= 0,
              predictionSeriesIndex < series.count,
              predictionSeriesIndex + horizon <= currentBarIndex else {
            return nil
        }

        let minMovePoints = ExecutionReplayTools.entryCostPoints(
            priceCostPoints: config.priceCostPoints,
            commissionPoints: config.commissionPoints,
            baseCostBufferPoints: config.baseCostBufferPoints,
            profile: config.executionProfile
        )
        let label = TrainingSampleTools.buildTripleBarrierLabel(
            series: series,
            index: predictionSeriesIndex,
            horizonMinutes: horizon,
            roundTripCostPoints: minMovePoints,
            evThresholdPoints: config.evThresholdPoints,
            maxFutureIndex: currentBarIndex,
            tradeKillerMinutes: config.tradeKillerMinutes
        )
        let realizedEdge = realizedEdgePoints(label: label, minMovePoints: minMovePoints)
        let stack = stackTargets(
            entry: action.entry,
            label: label,
            minMovePoints: minMovePoints,
            realizedEdgePoints: realizedEdge
        )
        let policy = policyTargets(
            label: label,
            minMovePoints: minMovePoints,
            realizedEdgePoints: realizedEdge
        )
        let effectiveQuality = action.entry.kind == .stack ? stack.qualityScore : policy.qualityScore
        return MetaPendingReplayOutcomeTargets(
            action: action,
            currentBarIndex: currentBarIndex,
            predictionSeriesIndex: predictionSeriesIndex,
            horizonMinutes: horizon,
            minMovePoints: minMovePoints,
            label: label,
            realizedEdgePoints: realizedEdge,
            qualityScore: effectiveQuality,
            stack: stack,
            policy: policy,
            horizonPolicyReward: horizonPolicyReward(
                label: label,
                minMovePoints: minMovePoints
            )
        )
    }

    static func updatedStackNetwork(
        _ state: StackNetworkState,
        with outcome: MetaPendingReplayOutcomeTargets
    ) -> StackNetworkState {
        guard outcome.action.entry.kind == .stack else { return state }
        return StackerTools.updatedStackNetwork(
            state,
            features: outcome.action.entry.features,
            labelClass: outcome.labelClass,
            sampleWeight: outcome.stack.sampleWeight
        )
    }

    static func observedStackRouterCells(
        _ cells: [StackRouterActionCell],
        with outcome: MetaPendingReplayOutcomeTargets
    ) -> [StackRouterActionCell] {
        guard outcome.action.entry.kind == .stack else { return cells }
        return StackerTools.observedRouterCells(
            cells,
            labelClass: outcome.labelClass,
            realizedEdge: outcome.realizedEdgePoints,
            qualityScore: outcome.stack.qualityScore,
            features: outcome.action.entry.features,
            predictedProbabilities: outcome.stack.predictedProbabilities,
            sampleWeight: outcome.stack.sampleWeight
        )
    }

    static func updatedTradeGateNetwork(
        _ state: TradeGateNetworkState,
        with outcome: MetaPendingReplayOutcomeTargets
    ) -> TradeGateNetworkState {
        guard outcome.action.entry.kind == .stack else { return state }
        return StackerTools.updatedTradeGateNetwork(
            state,
            features: outcome.action.entry.features,
            tradeTarget: outcome.stack.tradeTarget,
            sampleWeight: outcome.stack.sampleWeight
        )
    }

    static func updatedPolicyNetwork(
        _ state: MetaPolicyNetworkState,
        with outcome: MetaPendingReplayOutcomeTargets
    ) -> MetaPolicyNetworkState {
        guard outcome.action.entry.kind == .policy else { return state }
        return MetaPolicyTools.updatedPolicyNetwork(
            state,
            features: outcome.action.entry.features,
            tradeTarget: outcome.policy.tradeTarget,
            directionTarget: outcome.policy.directionTarget,
            sizeTarget: outcome.policy.sizeTarget,
            holdTarget: outcome.policy.holdTarget,
            sampleWeight: outcome.policy.sampleWeight
        )
    }

    static func updatedHorizonPolicyNetwork(
        _ state: HorizonPolicyNetworkState,
        with outcome: MetaPendingReplayOutcomeTargets
    ) -> HorizonPolicyNetworkState {
        guard outcome.action.entry.kind == .horizonPolicy else { return state }
        return HorizonPolicyTools.updatedNetwork(
            state,
            features: outcome.action.entry.features,
            rewardScaled: outcome.horizonPolicyReward
        )
    }

    static func updatedRegimeGraph(
        _ state: RegimeGraphState,
        with outcome: MetaPendingReplayOutcomeTargets,
        config: MetaPendingReplayOutcomeConfig
    ) -> RegimeGraphState {
        guard outcome.action.entry.kind == .policy else { return state }
        var next = state
        next.updateFeedback(
            fromRegime: outcome.action.entry.regimeID,
            toRegime: config.currentRegimeID,
            realizedEdgePoints: outcome.realizedEdgePoints,
            qualityScore: outcome.policy.qualityScore,
            macroQuality: config.currentMacroQuality
        )
        return next
    }

    private static func stackTargets(
        entry: MetaPendingReplayEntry,
        label: TripleBarrierLabelResult,
        minMovePoints: Double,
        realizedEdgePoints: Double
    ) -> MetaStackPendingReplayTargets {
        var sampleWeight = moveEdgeWeight(
            movePoints: label.realizedMovePoints,
            costPoints: minMovePoints
        )
        let quality = stackQuality(label: label, minMovePoints: minMovePoints)
        sampleWeight *= quality
        if entry.signal == -1, label.labelClass != .skip {
            sampleWeight *= 0.80
        }
        let tradeTarget = label.labelClass != .skip &&
            realizedEdgePoints > 0.0 &&
            quality > 0.70 &&
            label.timeToHitFraction < 0.95
        return MetaStackPendingReplayTargets(
            sampleWeight: sampleWeight,
            qualityScore: quality,
            tradeTarget: tradeTarget,
            predictedProbabilities: entry.probabilities
        )
    }

    private static func policyTargets(
        label: TripleBarrierLabelResult,
        minMovePoints: Double,
        realizedEdgePoints: Double
    ) -> MetaPolicyPendingReplayTargets {
        let qualityScore = policyQuality(label: label, minMovePoints: minMovePoints)
        let tradeTarget = label.labelClass != .skip &&
            realizedEdgePoints > 0.0 &&
            qualityScore > 0.70 ? 1.0 : 0.0
        let directionTarget: Double
        switch label.labelClass {
        case .buy:
            directionTarget = 1.0
        case .sell:
            directionTarget = -1.0
        case .skip:
            directionTarget = 0.0
        }
        var sizeTarget = fxClamp(
            0.25 +
                0.45 * max(realizedEdgePoints / max(minMovePoints, 0.10), 0.0) +
                0.20 * qualityScore,
            0.25,
            1.60
        )
        if tradeTarget < 0.5 {
            sizeTarget = 0.25
        }
        let maeRatio = adverseRatio(label: label, minMovePoints: minMovePoints)
        let holdTarget = fxClamp(
            0.28 +
                0.24 * speedBonus(label) +
                0.22 * qualityScore / 2.0 +
                0.16 * (1.0 - maeRatio / 3.0) -
                0.12 * (label.pathFlags.contains(.dualHit) ? 1.0 : 0.0),
            0.0,
            1.0
        )
        let sampleWeight = fxClamp(
            0.40 +
                0.45 * qualityScore +
                0.25 * max(abs(realizedEdgePoints) / max(minMovePoints, 0.10), 0.0),
            0.20,
            6.0
        )
        return MetaPolicyPendingReplayTargets(
            tradeTarget: tradeTarget,
            directionTarget: directionTarget,
            sizeTarget: sizeTarget,
            holdTarget: holdTarget,
            sampleWeight: sampleWeight,
            qualityScore: qualityScore
        )
    }

    private static func horizonPolicyReward(
        label: TripleBarrierLabelResult,
        minMovePoints: Double
    ) -> Double {
        let edge = max(abs(label.realizedMovePoints) - minMovePoints, 0.0)
        var reward = -0.25
        if label.labelClass != .skip {
            var quality = 1.0 +
                0.20 * speedBonus(label) -
                0.12 * adverseRatio(label: label, minMovePoints: minMovePoints)
            if label.pathFlags.contains(.dualHit) {
                quality -= 0.10
            }
            reward = quality * edge / max(minMovePoints, 0.50)
        }
        return fxClamp(reward, -2.0, 6.0)
    }

    private static func realizedEdgePoints(
        label: TripleBarrierLabelResult,
        minMovePoints: Double
    ) -> Double {
        switch label.labelClass {
        case .buy:
            label.realizedMovePoints - minMovePoints
        case .sell:
            -label.realizedMovePoints - minMovePoints
        case .skip:
            -max(abs(label.realizedMovePoints) - minMovePoints, 0.0)
        }
    }

    private static func stackQuality(
        label: TripleBarrierLabelResult,
        minMovePoints: Double
    ) -> Double {
        var quality = 1.0 +
            0.25 * speedBonus(label) -
            0.15 * adverseRatio(label: label, minMovePoints: minMovePoints)
        if label.pathFlags.contains(.dualHit) {
            quality -= 0.10
        }
        return fxClamp(quality, 0.25, 1.75)
    }

    private static func policyQuality(
        label: TripleBarrierLabelResult,
        minMovePoints: Double
    ) -> Double {
        var quality = 1.0 +
            0.28 * speedBonus(label) -
            0.16 * adverseRatio(label: label, minMovePoints: minMovePoints)
        if label.pathFlags.contains(.dualHit) {
            quality -= 0.10
        }
        return fxClamp(quality, 0.0, 2.0)
    }

    private static func moveEdgeWeight(movePoints: Double, costPoints: Double) -> Double {
        let move = abs(fxSafeFinite(movePoints))
        let cost = max(0.0, fxSafeFinite(costPoints))
        let edge = move - cost
        return fxClamp(0.50 + edge / max(cost, 1.0), 0.25, 4.00)
    }

    private static func speedBonus(_ label: TripleBarrierLabelResult) -> Double {
        1.0 - fxClamp(label.timeToHitFraction, 0.0, 1.0)
    }

    private static func adverseRatio(
        label: TripleBarrierLabelResult,
        minMovePoints: Double
    ) -> Double {
        fxClamp(label.maePoints / max(label.mfePoints, minMovePoints, 0.10), 0.0, 3.0)
    }
}
