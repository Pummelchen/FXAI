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
    public var riskTargetMovePoints: Double
    public var evThresholdPoints: Double
    public var maxPortfolioPressure: Double
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
        riskTargetMovePoints: Double = 12.0,
        evThresholdPoints: Double = 0.0,
        maxPortfolioPressure: Double = 0.78,
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
        self.riskTargetMovePoints = riskTargetMovePoints
        self.evThresholdPoints = max(0.0, fxSafeFinite(evThresholdPoints))
        self.maxPortfolioPressure = maxPortfolioPressure
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

public struct RiskPortfolioExposureState: Codable, Hashable, Sendable {
    public var grossExposureLots: Double
    public var correlatedExposureLots: Double
    public var directionalClusterLots: Double
    public var maxPortfolioExposureLots: Double
    public var maxCorrelatedExposureLots: Double
    public var maxDirectionalClusterLots: Double

    public init(
        grossExposureLots: Double = 0.0,
        correlatedExposureLots: Double = 0.0,
        directionalClusterLots: Double = 0.0,
        maxPortfolioExposureLots: Double = 0.0,
        maxCorrelatedExposureLots: Double = 0.0,
        maxDirectionalClusterLots: Double = 0.0
    ) {
        self.grossExposureLots = grossExposureLots
        self.correlatedExposureLots = correlatedExposureLots
        self.directionalClusterLots = directionalClusterLots
        self.maxPortfolioExposureLots = maxPortfolioExposureLots
        self.maxCorrelatedExposureLots = maxCorrelatedExposureLots
        self.maxDirectionalClusterLots = maxDirectionalClusterLots
    }
}

public struct RiskPortfolioPressureResult: Codable, Hashable, Sendable {
    public var pressure: Double
    public var controlPlaneScore: Double

    public init(pressure: Double = 0.0, controlPlaneScore: Double = 0.0) {
        self.pressure = pressure
        self.controlPlaneScore = controlPlaneScore
    }
}

public struct RiskControlPlaneOverlayState: Codable, Hashable, Sendable {
    public var aggregate: ControlPlaneAggregate
    public var supervisor: PortfolioSupervisorProfile
    public var serviceState: SupervisorServiceState
    public var commandState: SupervisorCommandState
    public var supervisorScore: Double?
    public var serviceScore: Double?
    public var controlPlaneScore: Double?
    public var controlPlaneBuyScore: Double
    public var controlPlaneSellScore: Double

    public init(
        aggregate: ControlPlaneAggregate = ControlPlaneAggregate(),
        supervisor: PortfolioSupervisorProfile = PortfolioSupervisorProfile(),
        serviceState: SupervisorServiceState = SupervisorServiceState(),
        commandState: SupervisorCommandState = SupervisorCommandState(),
        supervisorScore: Double? = nil,
        serviceScore: Double? = nil,
        controlPlaneScore: Double? = nil,
        controlPlaneBuyScore: Double = 0.0,
        controlPlaneSellScore: Double = 0.0
    ) {
        self.aggregate = aggregate
        self.supervisor = supervisor
        self.serviceState = serviceState
        self.commandState = commandState
        self.supervisorScore = supervisorScore
        self.serviceScore = serviceScore
        self.controlPlaneScore = controlPlaneScore
        self.controlPlaneBuyScore = controlPlaneBuyScore
        self.controlPlaneSellScore = controlPlaneSellScore
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
    private static func resolvedSupervisorScore(direction: Int, overlay: RiskControlPlaneOverlayState) -> Double {
        fxSafeFinite(overlay.supervisorScore ??
            ControlPlaneScoring.portfolioSupervisorScore(direction: direction, aggregate: overlay.aggregate, profile: overlay.supervisor)
        )
    }

    private static func resolvedServiceScore(direction: Int, overlay: RiskControlPlaneOverlayState) -> Double {
        fxSafeFinite(overlay.serviceScore ??
            ControlPlaneScoring.supervisorServiceScore(direction: direction, state: overlay.serviceState)
        )
    }

    private static func computedControlPlaneScore(
        deployment: LiveDeploymentProfile,
        aggregate: ControlPlaneAggregate,
        supervisor: PortfolioSupervisorProfile,
        supervisorScore: Double,
        serviceScore: Double
    ) -> Double {
        let supervisorBlend = fxClamp(
            0.55 * fxClamp(supervisor.supervisorWeight, 0.0, 1.0) +
                0.45 * fxClamp(deployment.supervisorBlend, 0.0, 1.0),
            0.0,
            1.0
        )
        return fxClamp(
            (1.0 - supervisorBlend) * aggregate.score +
                0.55 * supervisorBlend * supervisorScore +
                0.45 * supervisorBlend * serviceScore,
            0.0,
            3.0
        )
    }

    public static func estimatedRiskPoints(config: RiskPolicyConfig, signal: RiskPolicySignalState) -> Double {
        let baseCost = max(signal.minMovePoints, 0.25)
        let expectedMove = max(signal.expectedMovePoints, baseCost)
        let configuredTarget = max(config.riskTargetMovePoints, 0.0)
        var riskPoints = baseCost +
            expectedMove * (0.45 + 0.65 * fxClamp(signal.pathRisk, 0.0, 1.0)) +
            baseCost * (0.25 + 0.50 * fxClamp(signal.fillRisk, 0.0, 1.0))
        if configuredTarget > 0.0, configuredTarget > riskPoints {
            riskPoints = configuredTarget
        }
        if !riskPoints.isFinite || riskPoints <= 0.0 {
            riskPoints = max(configuredTarget, baseCost)
        }
        return max(riskPoints, 0.25)
    }

    public static func portfolioPressure(
        exposure: RiskPortfolioExposureState,
        signal: RiskPolicySignalState,
        deployment: LiveDeploymentProfile = LiveDeploymentProfile(),
        aggregate: ControlPlaneAggregate = ControlPlaneAggregate(),
        supervisor: PortfolioSupervisorProfile = PortfolioSupervisorProfile(),
        serviceState: SupervisorServiceState = SupervisorServiceState(),
        direction: Int,
        macroEventLeakageSafe: Bool = false,
        supervisorScore explicitSupervisorScore: Double? = nil,
        serviceScore explicitServiceScore: Double? = nil
    ) -> RiskPortfolioPressureResult {
        let portfolioCap = max(exposure.maxPortfolioExposureLots, 0.0)
        let corrCap = max(exposure.maxCorrelatedExposureLots, 0.0)
        let dirCap = max(exposure.maxDirectionalClusterLots, 0.0)
        let grossRatio = portfolioCap > 1e-9 ? exposure.grossExposureLots / portfolioCap : 0.0
        let corrRatio = corrCap > 1e-9 ? exposure.correlatedExposureLots / corrCap : 0.0
        let dirRatio = dirCap > 1e-9 ? exposure.directionalClusterLots / dirCap : 0.0
        let hierarchyPenalty = 1.0 - fxClamp(signal.hierarchyScore, 0.0, 1.0)
        let macroPenalty = macroEventLeakageSafe ? (1.0 - fxClamp(signal.macroStateQuality, 0.0, 1.0)) : 0.0
        let supervisorScore = explicitSupervisorScore ??
            ControlPlaneScoring.portfolioSupervisorScore(direction: direction, aggregate: aggregate, profile: supervisor)
        let serviceScore = explicitServiceScore ??
            ControlPlaneScoring.supervisorServiceScore(direction: direction, state: serviceState)
        let controlPlaneScore = computedControlPlaneScore(
            deployment: deployment,
            aggregate: aggregate,
            supervisor: supervisor,
            supervisorScore: supervisorScore,
            serviceScore: serviceScore
        )
        let pressure = fxClamp(
            0.30 * fxClamp(grossRatio / max(supervisor.grossBudgetBias, 0.40), 0.0, 2.0) +
                0.28 * fxClamp(corrRatio, 0.0, 2.0) +
                0.22 * fxClamp(dirRatio, 0.0, 2.0) +
                0.12 * fxClamp(aggregate.score, 0.0, 1.5) +
                0.08 * fxClamp(supervisorScore, 0.0, 1.5) +
                0.08 * fxClamp(serviceScore, 0.0, 1.5) +
                0.06 * fxClamp(aggregate.macroOverlap, 0.0, 1.0) +
                0.06 * fxClamp(serviceState.macroPressure, 0.0, 1.0) +
                0.10 * hierarchyPenalty +
                0.06 * macroPenalty,
            0.0,
            1.5
        )
        return RiskPortfolioPressureResult(pressure: pressure, controlPlaneScore: controlPlaneScore)
    }

    public static func supervisorOverlayDecision(
        signal: RiskPolicySignalState,
        direction: Int,
        overlay: RiskControlPlaneOverlayState
    ) -> RiskPolicyDecision {
        let supervisorScore = resolvedSupervisorScore(direction: direction, overlay: overlay)
        let serviceScore = resolvedServiceScore(direction: direction, overlay: overlay)
        if overlay.aggregate.maxCapitalRiskPct > fxClamp(overlay.supervisor.capitalRiskCapPct, 0.10, 10.0) {
            return RiskPolicyDecision(allowed: false, reason: "risk_supervisor_capital")
        }
        if supervisorScore > fxClamp(overlay.supervisor.hardBlockScore, 0.20, 3.0) {
            return RiskPolicyDecision(allowed: false, reason: "risk_supervisor_block")
        }
        if overlay.serviceState.ready {
            if serviceScore > fxClamp(overlay.serviceState.blockScore, 0.20, 3.0) {
                return RiskPolicyDecision(allowed: false, reason: "risk_supervisor_service_block")
            }
            if signal.policyEnterProb < max(fxClamp(overlay.serviceState.entryFloor, 0.10, 0.95), 0.05) {
                return RiskPolicyDecision(allowed: false, reason: "risk_supervisor_service_entry_floor")
            }
        }
        if overlay.commandState.ready, overlay.commandState.blocksDirection(direction) {
            return RiskPolicyDecision(allowed: false, reason: "risk_supervisor_command_block")
        }
        return RiskPolicyDecision()
    }

    public static func supervisorOverlayLotMultiplier(
        signal: RiskPolicySignalState,
        direction: Int,
        portfolioPressure: Double,
        deployment: LiveDeploymentProfile = LiveDeploymentProfile(),
        overlay: RiskControlPlaneOverlayState
    ) -> Double {
        let supervisorScore = resolvedSupervisorScore(direction: direction, overlay: overlay)
        let serviceScore = resolvedServiceScore(direction: direction, overlay: overlay)
        var controlPlanePressure = overlay.controlPlaneScore ?? computedControlPlaneScore(
            deployment: deployment,
            aggregate: overlay.aggregate,
            supervisor: overlay.supervisor,
            supervisorScore: supervisorScore,
            serviceScore: serviceScore
        )
        if direction == 1 {
            controlPlanePressure = max(controlPlanePressure, overlay.controlPlaneBuyScore)
        } else if direction == 0 {
            controlPlanePressure = max(controlPlanePressure, overlay.controlPlaneSellScore)
        }
        let serviceEntryBudget = fxClamp(
            overlay.serviceState.ready
                ? (direction == 1
                    ? overlay.serviceState.longEntryBudgetMultiplier
                    : (direction == 0
                        ? overlay.serviceState.shortEntryBudgetMultiplier
                        : overlay.serviceState.budgetMultiplier))
                : 1.0,
            0.10,
            1.20
        )
        return fxClamp(
            (1.08 - 0.60 * portfolioPressure) *
                (1.02 - 0.25 * fxClamp(controlPlanePressure, 0.0, 1.5)) *
                fxClamp(1.04 - 0.22 * supervisorScore, 0.25, 1.10) *
                serviceEntryBudget *
                overlay.commandState.entryBudgetMultiplier(for: direction) *
                fxClamp(signal.policySizeMultiplier, 0.25, 1.60) *
                fxClamp(0.70 + 0.30 * signal.policyEnterProb, 0.20, 1.10) *
                fxClamp(0.72 + 0.28 * signal.policyCapitalEfficiency, 0.25, 1.15),
            0.20,
            1.10
        )
    }

    public static func applyExposureCaps(
        requestedLot: Double,
        hardCapLot: Double,
        riskBudgetLot: Double = 0.0,
        exposure: RiskPortfolioExposureState,
        supervisor: PortfolioSupervisorProfile
    ) -> RiskSizingResult {
        var requestedLot = requestedLot
        var hardCapLot = fxSafeFinite(hardCapLot)

        func capResult(available: Double, reason: String) -> RiskSizingResult? {
            if available <= 0.0 {
                return RiskSizingResult(requestedLot: 0.0, hardCapLot: hardCapLot, riskBudgetLot: riskBudgetLot, reason: reason)
            }
            if requestedLot > available {
                requestedLot = available
            }
            if available < hardCapLot {
                hardCapLot = available
            }
            return nil
        }

        let portfolioCap = max(exposure.maxPortfolioExposureLots, 0.0)
        if portfolioCap > 0.0,
           let blocked = capResult(
            available: portfolioCap * max(supervisor.grossBudgetBias, 0.40) - exposure.grossExposureLots,
            reason: "risk_portfolio_cap"
           ) {
            return blocked
        }

        let correlatedCap = max(exposure.maxCorrelatedExposureLots, 0.0)
        if correlatedCap > 0.0,
           let blocked = capResult(
            available: correlatedCap * max(supervisor.correlatedBudgetBias, 0.40) - exposure.correlatedExposureLots,
            reason: "risk_correlated_cap"
           ) {
            return blocked
        }

        let directionalCap = max(exposure.maxDirectionalClusterLots, 0.0)
        if directionalCap > 0.0,
           let blocked = capResult(
            available: directionalCap * max(supervisor.directionalBudgetBias, 0.40) - exposure.directionalClusterLots,
            reason: "risk_directional_cluster_cap"
           ) {
            return blocked
        }

        if !requestedLot.isFinite || requestedLot <= 0.0 {
            return RiskSizingResult(requestedLot: 0.0, hardCapLot: hardCapLot, riskBudgetLot: riskBudgetLot, reason: "risk_lot_invalid")
        }

        return RiskSizingResult(
            requestedLot: requestedLot,
            hardCapLot: hardCapLot,
            riskBudgetLot: riskBudgetLot
        )
    }

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
        budget: RiskBudgetInput? = nil,
        portfolioPressure: Double? = nil
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
        if let portfolioPressure,
           portfolioPressure > fxClamp(config.maxPortfolioPressure, 0.0, 1.5) {
            return RiskSizingResult(requestedLot: 0.0, hardCapLot: hardCapLot, riskBudgetLot: riskBudgetLot, reason: "risk_portfolio_pressure")
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
