import Foundation

public struct RuntimeEnsemblePolicyInputs: Codable, Hashable, Sendable {
    public var buyPercent: Double
    public var sellPercent: Double
    public var skipPercent: Double
    public var ensembleBuyProbability: Double
    public var ensembleSkipProbability: Double
    public var averageBuyEVPoints: Double
    public var averageSellEVPoints: Double
    public var stackBuyEVPoints: Double
    public var stackSellEVPoints: Double
    public var buyPolicyScore: Double
    public var sellPolicyScore: Double
    public var policyGate: Double
    public var tradeGateThreshold: Double
    public var agreePercent: Double
    public var evThresholdPoints: Double
    public var hierarchyConsistency: Double
    public var hierarchyExecutionViability: Double
    public var macroLeakageSafe: Bool
    public var macroQuality: Double
    public var policyDecision: MetaPolicyDecision
    public var deployment: LiveDeploymentProfile

    public init(
        buyPercent: Double = 0.0,
        sellPercent: Double = 0.0,
        skipPercent: Double = 100.0,
        ensembleBuyProbability: Double = 0.0,
        ensembleSkipProbability: Double = 1.0,
        averageBuyEVPoints: Double = 0.0,
        averageSellEVPoints: Double = 0.0,
        stackBuyEVPoints: Double = 0.0,
        stackSellEVPoints: Double = 0.0,
        buyPolicyScore: Double = 0.0,
        sellPolicyScore: Double = 0.0,
        policyGate: Double = 0.0,
        tradeGateThreshold: Double = 0.0,
        agreePercent: Double = 0.0,
        evThresholdPoints: Double = 0.0,
        hierarchyConsistency: Double = 0.0,
        hierarchyExecutionViability: Double = 0.0,
        macroLeakageSafe: Bool = false,
        macroQuality: Double = 0.0,
        policyDecision: MetaPolicyDecision = MetaPolicyDecision(),
        deployment: LiveDeploymentProfile = LiveDeploymentProfile()
    ) {
        self.buyPercent = fxClamp(buyPercent, 0.0, 100.0)
        self.sellPercent = fxClamp(sellPercent, 0.0, 100.0)
        self.skipPercent = fxClamp(skipPercent, 0.0, 100.0)
        self.ensembleBuyProbability = fxClamp(ensembleBuyProbability, 0.0, 1.0)
        self.ensembleSkipProbability = fxClamp(ensembleSkipProbability, 0.0, 1.0)
        self.averageBuyEVPoints = fxSafeFinite(averageBuyEVPoints)
        self.averageSellEVPoints = fxSafeFinite(averageSellEVPoints)
        self.stackBuyEVPoints = fxSafeFinite(stackBuyEVPoints)
        self.stackSellEVPoints = fxSafeFinite(stackSellEVPoints)
        self.buyPolicyScore = fxClamp(buyPolicyScore, 0.0, 1.25)
        self.sellPolicyScore = fxClamp(sellPolicyScore, 0.0, 1.25)
        self.policyGate = fxClamp(policyGate, 0.0, 1.0)
        self.tradeGateThreshold = fxClamp(tradeGateThreshold, 0.0, 1.0)
        self.agreePercent = fxClamp(agreePercent, 0.0, 100.0)
        self.evThresholdPoints = fxSafeFinite(evThresholdPoints)
        self.hierarchyConsistency = fxClamp(hierarchyConsistency, 0.0, 1.0)
        self.hierarchyExecutionViability = fxClamp(hierarchyExecutionViability, 0.0, 1.0)
        self.macroLeakageSafe = macroLeakageSafe
        self.macroQuality = fxClamp(macroQuality, 0.0, 1.0)
        self.policyDecision = policyDecision
        self.deployment = deployment
    }
}

public struct RuntimePolicyDecisionResult: Codable, Hashable, Sendable {
    public var decision: Int
    public var chosenEdgePoints: Double
    public var policyGateFloor: Double
    public var reason: String

    public init(decision: Int, chosenEdgePoints: Double, policyGateFloor: Double = 0.0, reason: String) {
        self.decision = RuntimePolicyStageTools.normalizedDecision(decision)
        self.chosenEdgePoints = fxSafeFinite(chosenEdgePoints)
        self.policyGateFloor = fxClamp(policyGateFloor, 0.0, 1.0)
        self.reason = reason
    }
}

public struct RuntimePolicyMutableState: Codable, Hashable, Sendable {
    public var decision: Int
    public var tradeGate: Double
    public var policyDecision: MetaPolicyDecision

    public init(decision: Int = -1, tradeGate: Double = 0.0, policyDecision: MetaPolicyDecision = MetaPolicyDecision()) {
        self.decision = RuntimePolicyStageTools.normalizedDecision(decision)
        self.tradeGate = fxClamp(tradeGate, 0.0, 1.0)
        self.policyDecision = policyDecision
    }
}

public enum RuntimePolicyStageTools {
    public static func resolveEnsembleDecision(_ input: RuntimeEnsemblePolicyInputs) -> RuntimePolicyDecisionResult {
        var policyGateFloor = 0.0
        var decision = -1
        var reason = "policy_no_trade"
        var chosenEdge = max(input.stackBuyEVPoints, input.stackSellEVPoints)

        if input.hierarchyConsistency < 0.38 || input.hierarchyExecutionViability < 0.32 {
            reason = "hierarchy_block"
        } else if input.policyDecision.action == .noTrade ||
                    input.policyDecision.noTradeProbability > fxClamp(input.deployment.policyNoTradeCap, 0.25, 0.95) {
            reason = "policy_no_trade"
        } else if input.macroLeakageSafe &&
                    input.macroQuality < fxClamp(input.deployment.macroQualityFloor, 0.0, 1.0) {
            reason = "macro_quality_floor"
        } else if input.ensembleSkipProbability >= 0.58 || input.skipPercent >= 75.0 {
            reason = "ensemble_skip_block"
        } else {
            policyGateFloor = max(input.tradeGateThreshold, fxClamp(input.deployment.policyTradeFloor, 0.20, 0.90))
            policyGateFloor = max(policyGateFloor, input.policyDecision.enterProbability)
            if input.policyGate < policyGateFloor {
                reason = "policy_gate_floor"
            } else if input.buyPolicyScore >= input.sellPolicyScore &&
                        input.buyPercent >= input.agreePercent &&
                        input.stackBuyEVPoints >= input.evThresholdPoints &&
                        input.averageBuyEVPoints > input.averageSellEVPoints {
                decision = 1
                reason = "policy_buy"
            } else if input.sellPolicyScore > input.buyPolicyScore &&
                        input.sellPercent >= input.agreePercent &&
                        input.stackSellEVPoints >= input.evThresholdPoints &&
                        input.averageSellEVPoints > input.averageBuyEVPoints {
                decision = 0
                reason = "policy_sell"
            } else if input.buyPercent >= input.agreePercent &&
                        input.averageBuyEVPoints >= input.evThresholdPoints &&
                        input.averageBuyEVPoints > input.averageSellEVPoints {
                decision = 1
                reason = "fallback_buy"
            } else if input.sellPercent >= input.agreePercent &&
                        input.averageSellEVPoints >= input.evThresholdPoints &&
                        input.averageSellEVPoints > input.averageBuyEVPoints {
                decision = 0
                reason = "fallback_sell"
            } else {
                reason = "policy_uncertain"
            }
        }

        if decision == 1 {
            chosenEdge = input.stackBuyEVPoints
        } else if decision == 0 {
            chosenEdge = input.stackSellEVPoints
        }
        return RuntimePolicyDecisionResult(
            decision: decision,
            chosenEdgePoints: chosenEdge,
            policyGateFloor: policyGateFloor,
            reason: reason
        )
    }

    public static func applyDynamicEnsemblePosture(
        _ state: RuntimePolicyMutableState,
        dynamicState: DynamicEnsembleRuntimeState,
        dynamicEnsembleEnabled: Bool = true,
        dynamicEnsembleApplied: Bool = true,
        cautionEnterProbabilityBuffer: Double = 0.0,
        abstainEnterProbabilityFloor: Double = 0.05
    ) -> RuntimePolicyMutableState {
        guard dynamicEnsembleEnabled, dynamicEnsembleApplied, dynamicState.ready else { return state }
        var output = state
        var policy = output.policyDecision
        let buffer = fxClamp(cautionEnterProbabilityBuffer, 0.0, 0.25)

        switch dynamicState.tradePosture {
        case "CAUTION":
            policy.sizeMultiplier = fxClamp(policy.sizeMultiplier * 0.86, 0.10, 1.60)
            policy.enterProbability = fxClamp(policy.enterProbability - buffer, 0.0, 1.0)
            output.tradeGate = fxClamp(output.tradeGate * 0.93, 0.0, 1.0)
            policy.noTradeProbability = fxClamp(policy.noTradeProbability + dynamicState.abstainBias, 0.0, 1.0)
        case "ABSTAIN_BIAS":
            policy.sizeMultiplier = fxClamp(policy.sizeMultiplier * 0.72, 0.05, 1.60)
            policy.enterProbability = fxClamp(policy.enterProbability - max(0.08, buffer), 0.0, 1.0)
            output.tradeGate = fxClamp(output.tradeGate * 0.84, 0.0, 1.0)
            policy.noTradeProbability = fxClamp(policy.noTradeProbability + max(dynamicState.abstainBias, 0.18), 0.0, 1.0)
            if policy.enterProbability < fxClamp(abstainEnterProbabilityFloor, 0.05, 0.95) {
                output.decision = -1
            }
        case "BLOCK":
            policy.sizeMultiplier = fxClamp(policy.sizeMultiplier * 0.25, 0.01, 1.60)
            policy.enterProbability = 0.0
            output.tradeGate = fxClamp(output.tradeGate * 0.42, 0.0, 1.0)
            policy.noTradeProbability = fxClamp(max(policy.noTradeProbability, 0.96), 0.0, 1.0)
            output.decision = -1
        default:
            break
        }

        if output.decision != -1 {
            if output.decision == 1, dynamicState.finalScore < -0.08 {
                output.decision = -1
            } else if output.decision == 0, dynamicState.finalScore > 0.08 {
                output.decision = -1
            }
        }
        output.policyDecision = policy
        return output
    }

    public static func applyAdaptiveRouterPosture(
        _ state: RuntimePolicyMutableState,
        posture: String,
        abstainBias: Double
    ) -> RuntimePolicyMutableState {
        var output = state
        var policy = output.policyDecision
        let bias = fxClamp(abstainBias, 0.0, 0.98)

        switch posture {
        case "CAUTION":
            policy.sizeMultiplier = fxClamp(policy.sizeMultiplier * 0.84, 0.10, 1.60)
            policy.enterProbability = fxClamp(policy.enterProbability - 0.05, 0.0, 1.0)
            output.tradeGate = fxClamp(output.tradeGate * 0.92, 0.0, 1.0)
            policy.noTradeProbability = fxClamp(policy.noTradeProbability + bias, 0.0, 1.0)
        case "ABSTAIN_BIAS":
            policy.sizeMultiplier = fxClamp(policy.sizeMultiplier * 0.68, 0.05, 1.60)
            policy.enterProbability = fxClamp(policy.enterProbability - 0.14, 0.0, 1.0)
            output.tradeGate = fxClamp(output.tradeGate * 0.82, 0.0, 1.0)
            policy.noTradeProbability = fxClamp(policy.noTradeProbability + max(bias, 0.18), 0.0, 1.0)
            if policy.enterProbability < 0.32 {
                output.decision = -1
            }
        case "BLOCK":
            policy.sizeMultiplier = fxClamp(policy.sizeMultiplier * 0.25, 0.01, 1.60)
            policy.enterProbability = 0.0
            output.tradeGate = fxClamp(output.tradeGate * 0.40, 0.0, 1.0)
            policy.noTradeProbability = fxClamp(max(policy.noTradeProbability, 0.96), 0.0, 1.0)
            output.decision = -1
        default:
            break
        }

        output.policyDecision = policy
        return output
    }

    public static func normalizedDecision(_ decision: Int) -> Int {
        decision == 1 ? 1 : (decision == 0 ? 0 : -1)
    }
}
