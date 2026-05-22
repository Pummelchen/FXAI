import Foundation

public enum RiskPositionSizing: Int, Codable, Sendable, CaseIterable {
    case fixedLot = 0
    case conviction = 1
    case volatilityTarget = 2
}

public struct RiskPolicyConfig: Codable, Hashable, Sendable {
    public var baseLot: Double
    public var positionSizing: RiskPositionSizing
    public var riskPerTradePct: Double
    public var evThresholdPoints: Double
    public var minConfidence: Double
    public var minReliability: Double
    public var maxPathRisk: Double
    public var maxFillRisk: Double
    public var minTradeGate: Double
    public var minHierarchyScore: Double
    public var minHierarchyConsistency: Double
    public var minHierarchyTradability: Double
    public var minHierarchyExecution: Double
    public var minMacroStateQuality: Double
    public var killTradeGate: Double
    public var killPathRisk: Double
    public var killFillRisk: Double

    public init(
        baseLot: Double = 0.01,
        positionSizing: RiskPositionSizing = .conviction,
        riskPerTradePct: Double = 0.35,
        evThresholdPoints: Double = 0.0,
        minConfidence: Double = 0.52,
        minReliability: Double = 0.48,
        maxPathRisk: Double = 0.72,
        maxFillRisk: Double = 0.68,
        minTradeGate: Double = 0.52,
        minHierarchyScore: Double = 0.46,
        minHierarchyConsistency: Double = 0.40,
        minHierarchyTradability: Double = 0.38,
        minHierarchyExecution: Double = 0.34,
        minMacroStateQuality: Double = 0.24,
        killTradeGate: Double = 0.24,
        killPathRisk: Double = 0.92,
        killFillRisk: Double = 0.90
    ) {
        self.baseLot = max(0.0, fxSafeFinite(baseLot))
        self.positionSizing = positionSizing
        self.riskPerTradePct = max(0.0, fxSafeFinite(riskPerTradePct))
        self.evThresholdPoints = max(0.0, fxSafeFinite(evThresholdPoints))
        self.minConfidence = minConfidence
        self.minReliability = minReliability
        self.maxPathRisk = maxPathRisk
        self.maxFillRisk = maxFillRisk
        self.minTradeGate = minTradeGate
        self.minHierarchyScore = minHierarchyScore
        self.minHierarchyConsistency = minHierarchyConsistency
        self.minHierarchyTradability = minHierarchyTradability
        self.minHierarchyExecution = minHierarchyExecution
        self.minMacroStateQuality = minMacroStateQuality
        self.killTradeGate = killTradeGate
        self.killPathRisk = killPathRisk
        self.killFillRisk = killFillRisk
    }
}

public struct RiskPolicySignalState: Codable, Hashable, Sendable {
    public var confidence: Double
    public var reliability: Double
    public var tradeGate: Double
    public var pathRisk: Double
    public var fillRisk: Double
    public var minMovePoints: Double
    public var tradeEdgePoints: Double
    public var expectedMovePoints: Double
    public var hierarchyScore: Double
    public var hierarchyConsistency: Double
    public var hierarchyTradability: Double
    public var hierarchyExecution: Double
    public var contextStrength: Double
    public var macroStateQuality: Double
    public var policyAction: PolicyLifecycleAction
    public var policyEnterProb: Double
    public var policyNoTradeProb: Double
    public var policyHoldQuality: Double
    public var policySizeMultiplier: Double
    public var policyPortfolioFit: Double
    public var policyCapitalEfficiency: Double

    public init(
        confidence: Double = 0.0,
        reliability: Double = 0.0,
        tradeGate: Double = 0.0,
        pathRisk: Double = 0.0,
        fillRisk: Double = 0.0,
        minMovePoints: Double = 0.25,
        tradeEdgePoints: Double = 0.0,
        expectedMovePoints: Double = 0.0,
        hierarchyScore: Double = 0.0,
        hierarchyConsistency: Double = 0.0,
        hierarchyTradability: Double = 0.0,
        hierarchyExecution: Double = 0.0,
        contextStrength: Double = 0.0,
        macroStateQuality: Double = 0.0,
        policyAction: PolicyLifecycleAction = .noTrade,
        policyEnterProb: Double = 0.0,
        policyNoTradeProb: Double = 1.0,
        policyHoldQuality: Double = 0.0,
        policySizeMultiplier: Double = 1.0,
        policyPortfolioFit: Double = 0.0,
        policyCapitalEfficiency: Double = 0.0
    ) {
        self.confidence = confidence
        self.reliability = reliability
        self.tradeGate = tradeGate
        self.pathRisk = pathRisk
        self.fillRisk = fillRisk
        self.minMovePoints = minMovePoints
        self.tradeEdgePoints = tradeEdgePoints
        self.expectedMovePoints = expectedMovePoints
        self.hierarchyScore = hierarchyScore
        self.hierarchyConsistency = hierarchyConsistency
        self.hierarchyTradability = hierarchyTradability
        self.hierarchyExecution = hierarchyExecution
        self.contextStrength = contextStrength
        self.macroStateQuality = macroStateQuality
        self.policyAction = policyAction
        self.policyEnterProb = policyEnterProb
        self.policyNoTradeProb = policyNoTradeProb
        self.policyHoldQuality = policyHoldQuality
        self.policySizeMultiplier = policySizeMultiplier
        self.policyPortfolioFit = policyPortfolioFit
        self.policyCapitalEfficiency = policyCapitalEfficiency
    }
}

public struct RiskBudgetInput: Codable, Hashable, Sendable {
    public var equity: Double
    public var moneyPerPointPerLot: Double
    public var riskPoints: Double

    public init(equity: Double = 0.0, moneyPerPointPerLot: Double = 0.0, riskPoints: Double = 0.0) {
        self.equity = equity
        self.moneyPerPointPerLot = moneyPerPointPerLot
        self.riskPoints = riskPoints
    }
}

public struct RiskPolicyDecision: Codable, Hashable, Sendable {
    public var allowed: Bool
    public var reason: String

    public init(allowed: Bool = true, reason: String = "ok") {
        self.allowed = allowed
        self.reason = reason
    }
}

public struct RiskSizingResult: Codable, Hashable, Sendable {
    public var requestedLot: Double
    public var hardCapLot: Double
    public var riskBudgetLot: Double
    public var reason: String

    public init(
        requestedLot: Double = 0.0,
        hardCapLot: Double = 1_000_000.0,
        riskBudgetLot: Double = 0.0,
        reason: String = "ok"
    ) {
        self.requestedLot = requestedLot
        self.hardCapLot = hardCapLot
        self.riskBudgetLot = riskBudgetLot
        self.reason = reason
    }
}

public enum RiskPolicyTools {
    public static func regimeKillSwitch(
        config: RiskPolicyConfig,
        signal: RiskPolicySignalState
    ) -> RiskPolicyDecision {
        if signal.tradeGate <= fxClamp(config.killTradeGate, 0.0, 1.0) {
            return RiskPolicyDecision(allowed: false, reason: "kill_trade_gate")
        }
        if signal.policyNoTradeProb >= 0.92 {
            return RiskPolicyDecision(allowed: false, reason: "kill_policy_no_trade")
        }
        if signal.policyAction == .exit, signal.policyEnterProb < 0.25 {
            return RiskPolicyDecision(allowed: false, reason: "kill_policy_exit")
        }
        if signal.pathRisk >= fxClamp(config.killPathRisk, 0.0, 1.0) {
            return RiskPolicyDecision(allowed: false, reason: "kill_path_risk")
        }
        if signal.fillRisk >= fxClamp(config.killFillRisk, 0.0, 1.0) {
            return RiskPolicyDecision(allowed: false, reason: "kill_fill_risk")
        }
        if signal.policyHoldQuality < 0.18,
           signal.tradeGate < 0.36 || signal.pathRisk > 0.82 {
            return RiskPolicyDecision(allowed: false, reason: "kill_policy_hold")
        }
        return RiskPolicyDecision()
    }

    public static func admissionDecision(
        config: RiskPolicyConfig,
        signal: RiskPolicySignalState,
        direction: Int,
        deployment: LiveDeploymentProfile = LiveDeploymentProfile(),
        systemHealth: SystemHealthState? = nil,
        factorContext: PairFactorContext? = nil,
        macroEventLeakageSafe: Bool = false
    ) -> RiskPolicyDecision {
        guard direction == 0 || direction == 1 else {
            return RiskPolicyDecision(allowed: false, reason: "invalid_direction")
        }
        if let systemHealth,
           systemHealth.ready,
           systemHealth.posture == .degraded,
           systemHealth.healthScore < 0.35 {
            return RiskPolicyDecision(allowed: false, reason: "risk_system_health_degraded")
        }
        if let factorContext,
           factorContext.ready,
           factorContext.biasDirection >= 0,
           factorContext.biasDirection != direction,
           abs(factorContext.blendedScore) > 0.42 {
            return RiskPolicyDecision(allowed: false, reason: "risk_factor_context_opposed")
        }
        if signal.confidence < fxClamp(config.minConfidence, 0.0, 1.0) {
            return RiskPolicyDecision(allowed: false, reason: "risk_confidence_floor")
        }
        if signal.reliability < fxClamp(config.minReliability, 0.0, 1.0) {
            return RiskPolicyDecision(allowed: false, reason: "risk_reliability_floor")
        }
        if signal.pathRisk > fxClamp(config.maxPathRisk, 0.0, 1.0) {
            return RiskPolicyDecision(allowed: false, reason: "risk_path_cap")
        }
        if signal.fillRisk > fxClamp(config.maxFillRisk, 0.0, 1.0) {
            return RiskPolicyDecision(allowed: false, reason: "risk_fill_cap")
        }
        if signal.tradeGate < fxClamp(config.minTradeGate, 0.0, 1.0) {
            return RiskPolicyDecision(allowed: false, reason: "risk_trade_gate_floor")
        }
        if signal.policyEnterProb < 0.05 {
            return RiskPolicyDecision(allowed: false, reason: "risk_policy_enter_floor")
        }
        if signal.policyNoTradeProb > fxClamp(deployment.policyNoTradeCap, 0.25, 0.95) {
            return RiskPolicyDecision(allowed: false, reason: "risk_policy_no_trade")
        }
        if signal.hierarchyScore < fxClamp(config.minHierarchyScore, 0.0, 1.0) {
            return RiskPolicyDecision(allowed: false, reason: "risk_hierarchy_score_floor")
        }
        if signal.hierarchyConsistency < fxClamp(config.minHierarchyConsistency, 0.0, 1.0) {
            return RiskPolicyDecision(allowed: false, reason: "risk_hierarchy_consistency_floor")
        }
        if signal.hierarchyTradability < fxClamp(config.minHierarchyTradability, 0.0, 1.0) {
            return RiskPolicyDecision(allowed: false, reason: "risk_hierarchy_tradability_floor")
        }
        if signal.hierarchyExecution < fxClamp(config.minHierarchyExecution, 0.0, 1.0) {
            return RiskPolicyDecision(allowed: false, reason: "risk_hierarchy_execution_floor")
        }
        let macroFloor = max(
            fxClamp(config.minMacroStateQuality, 0.0, 1.0),
            fxClamp(deployment.macroQualityFloor, 0.0, 1.0)
        )
        if macroEventLeakageSafe, signal.macroStateQuality < macroFloor {
            return RiskPolicyDecision(allowed: false, reason: "risk_macro_state_floor")
        }
        return RiskPolicyDecision()
    }

    public static func conviction(
        signal: RiskPolicySignalState,
        direction: Int,
        deployment: LiveDeploymentProfile = LiveDeploymentProfile(),
        systemHealth: SystemHealthState? = nil,
        factorContext: PairFactorContext? = nil
    ) -> Double {
        let edgeScale = fxClamp(signal.tradeEdgePoints / max(signal.minMovePoints, 0.25), -1.0, 4.0)
        var conviction = fxClamp(
            0.20 +
                0.22 * fxClamp(signal.confidence, 0.0, 1.0) +
                0.18 * fxClamp(signal.reliability, 0.0, 1.0) +
                0.16 * fxClamp(signal.tradeGate, 0.0, 1.0) +
                0.12 * fxClamp(signal.hierarchyScore, 0.0, 1.0) +
                0.08 * fxClamp(signal.hierarchyConsistency, 0.0, 1.0) +
                0.10 * fxClamp(signal.contextStrength / 2.0, 0.0, 1.0) +
                0.10 * (1.0 - fxClamp(signal.pathRisk, 0.0, 1.0)) +
                0.08 * (1.0 - fxClamp(signal.fillRisk, 0.0, 1.0)) +
                0.06 * fxClamp(signal.macroStateQuality, 0.0, 1.0) +
                0.10 * fxClamp(edgeScale / 2.0, 0.0, 1.0),
            0.20,
            1.60
        )
        if let factorContext, factorContext.ready {
            let factorConviction = fxClamp(0.85 + 0.35 * abs(factorContext.blendedScore), 0.60, 1.25)
            if factorContext.biasDirection == direction {
                conviction *= factorConviction
            } else if factorContext.biasDirection >= 0 {
                conviction *= fxClamp(1.10 - 0.55 * abs(factorContext.blendedScore), 0.40, 1.10)
            }
        }
        if let systemHealth, systemHealth.ready {
            conviction *= fxClamp(0.65 + 0.45 * systemHealth.healthScore, 0.35, 1.10)
        }
        conviction *= fxClamp(signal.policySizeMultiplier, 0.25, 1.60)
        conviction *= fxClamp(deployment.portfolioBudgetBias, 0.40, 1.60)
        conviction *= fxClamp(0.75 + 0.35 * signal.policyPortfolioFit, 0.25, 1.25)
        conviction *= fxClamp(deployment.capitalEfficiencyBias, 0.40, 1.80)
        conviction *= fxClamp(0.70 + 0.45 * signal.policyCapitalEfficiency, 0.25, 1.40)
        return fxClamp(conviction, 0.20, 2.20)
    }

    public static func riskBudgetLot(config: RiskPolicyConfig, budget: RiskBudgetInput?) -> Double {
        guard let budget else { return 0.0 }
        let riskBudgetPct = max(config.riskPerTradePct, 0.0)
        guard riskBudgetPct > 0.0,
              budget.equity > 0.0,
              budget.moneyPerPointPerLot > 0.0,
              budget.riskPoints > 0.0 else {
            return 0.0
        }
        return (budget.equity * (riskBudgetPct / 100.0)) / (budget.moneyPerPointPerLot * budget.riskPoints)
    }

    public static func sizingResult(
        config: RiskPolicyConfig,
        signal: RiskPolicySignalState,
        conviction: Double,
        budget: RiskBudgetInput? = nil
    ) -> RiskSizingResult {
        var requestedLot = config.baseLot
        var hardCapLot = 1_000_000.0
        let riskBudgetLot = riskBudgetLot(config: config, budget: budget)
        if riskBudgetLot > 0.0, riskBudgetLot < hardCapLot {
            hardCapLot = riskBudgetLot
        }

        if config.positionSizing == .conviction {
            requestedLot *= conviction
        }

        if config.positionSizing == .volatilityTarget {
            if riskBudgetLot > 0.0 {
                let riskPoints = max(budget?.riskPoints ?? 0.0, 0.25)
                let volScale = fxClamp(
                    0.55 +
                        0.25 * conviction +
                        0.10 * fxClamp(signal.expectedMovePoints / riskPoints, 0.0, 1.5),
                    0.35,
                    1.00
                )
                requestedLot = riskBudgetLot * volScale
            } else {
                requestedLot *= fxClamp(0.50 + 0.35 * conviction, 0.35, 1.10)
            }
        } else if riskBudgetLot > 0.0, requestedLot > riskBudgetLot {
            requestedLot = riskBudgetLot
        }

        if signal.tradeEdgePoints < max(config.evThresholdPoints * 0.50, 0.0) {
            return RiskSizingResult(requestedLot: 0.0, hardCapLot: hardCapLot, riskBudgetLot: riskBudgetLot, reason: "risk_edge_floor")
        }
        if requestedLot > hardCapLot + 1e-9 {
            return RiskSizingResult(requestedLot: 0.0, hardCapLot: hardCapLot, riskBudgetLot: riskBudgetLot, reason: "risk_min_volume_cap")
        }
        if !requestedLot.isFinite || requestedLot <= 0.0 {
            return RiskSizingResult(requestedLot: 0.0, hardCapLot: hardCapLot, riskBudgetLot: riskBudgetLot, reason: "risk_lot_invalid")
        }
        return RiskSizingResult(requestedLot: requestedLot, hardCapLot: hardCapLot, riskBudgetLot: riskBudgetLot)
    }
}
