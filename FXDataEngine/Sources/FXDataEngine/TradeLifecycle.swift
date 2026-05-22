import Foundation

public struct TradeLifecyclePositionState: Codable, Hashable, Sendable {
    public var hasPosition: Bool
    public var direction: Int
    public var totalVolume: Double
    public var openProfit: Double
    public var profitPoints: Double
    public var cycleRealizedProfit: Double
    public var cyclePeakProfit: Double
    public var givebackFraction: Double
    public var pendingCount: Int
    public var heldBars: Int

    public init(
        hasPosition: Bool = false,
        direction: Int = -1,
        totalVolume: Double = 0.0,
        openProfit: Double = 0.0,
        profitPoints: Double = 0.0,
        cycleRealizedProfit: Double = 0.0,
        cyclePeakProfit: Double = 0.0,
        givebackFraction: Double = 0.0,
        pendingCount: Int = 0,
        heldBars: Int = 0
    ) {
        self.hasPosition = hasPosition
        self.direction = [-1, 0, 1].contains(direction) ? direction : -1
        self.totalVolume = max(0.0, fxSafeFinite(totalVolume))
        self.openProfit = fxSafeFinite(openProfit)
        self.profitPoints = fxSafeFinite(profitPoints)
        self.cycleRealizedProfit = fxSafeFinite(cycleRealizedProfit)
        self.cyclePeakProfit = fxSafeFinite(cyclePeakProfit)
        self.givebackFraction = givebackFraction
        self.pendingCount = max(0, pendingCount)
        self.heldBars = max(0, heldBars)
    }
}

public struct TradeLifecyclePolicyState: Codable, Hashable, Sendable {
    public var addProb: Double
    public var reduceProb: Double
    public var timeoutProb: Double
    public var tightenProb: Double
    public var exitProb: Double
    public var holdQuality: Double
    public var portfolioFit: Double
    public var capitalEfficiency: Double
    public var portfolioPressure: Double

    public init(
        addProb: Double = 0.0,
        reduceProb: Double = 0.0,
        timeoutProb: Double = 0.0,
        tightenProb: Double = 0.0,
        exitProb: Double = 0.0,
        holdQuality: Double = 0.0,
        portfolioFit: Double = 0.0,
        capitalEfficiency: Double = 0.0,
        portfolioPressure: Double = 0.0
    ) {
        self.addProb = addProb
        self.reduceProb = reduceProb
        self.timeoutProb = timeoutProb
        self.tightenProb = tightenProb
        self.exitProb = exitProb
        self.holdQuality = holdQuality
        self.portfolioFit = portfolioFit
        self.capitalEfficiency = capitalEfficiency
        self.portfolioPressure = portfolioPressure
    }
}

public struct TradeLifecycleProbabilities: Codable, Hashable, Sendable {
    public var add: Double
    public var reduce: Double
    public var timeout: Double
    public var tighten: Double
    public var exit: Double

    public init(add: Double = 0.0, reduce: Double = 0.0, timeout: Double = 0.0, tighten: Double = 0.0, exit: Double = 0.0) {
        self.add = add
        self.reduce = reduce
        self.timeout = timeout
        self.tighten = tighten
        self.exit = exit
    }
}

public struct TradeLifecycleRecommendation: Codable, Hashable, Sendable {
    public var action: PolicyLifecycleAction
    public var reason: String
    public var probabilities: TradeLifecycleProbabilities
    public var reduceFraction: Double
    public var addSizeMultiplier: Double

    public init(
        action: PolicyLifecycleAction = .noTrade,
        reason: String = "lifecycle_hold",
        probabilities: TradeLifecycleProbabilities = TradeLifecycleProbabilities(),
        reduceFraction: Double = 0.0,
        addSizeMultiplier: Double = 0.0
    ) {
        self.action = action
        self.reason = reason
        self.probabilities = probabilities
        self.reduceFraction = reduceFraction
        self.addSizeMultiplier = addSizeMultiplier
    }
}

public enum TradeLifecycleTools {
    public static func probabilities(
        position: TradeLifecyclePositionState,
        signalDirection: Int,
        policy: TradeLifecyclePolicyState,
        minMovePoints: Double,
        deployment: LiveDeploymentProfile = LiveDeploymentProfile(),
        serviceState: SupervisorServiceState = SupervisorServiceState(),
        commandState: SupervisorCommandState = SupervisorCommandState(),
        controlPlaneScore: Double = 0.0,
        serviceScore explicitServiceScore: Double? = nil
    ) -> TradeLifecycleProbabilities {
        let serviceScore = explicitServiceScore ?? ControlPlaneScoring.supervisorServiceScore(direction: position.direction, state: serviceState)
        let lifecycleGain = fxClamp(deployment.policyLifecycleGain, 0.40, 1.80)
        let directionMatch = signalDirection == position.direction
        let oppositeSignal = signalDirection >= 0 && signalDirection != position.direction
        let profitNorm = fxClamp(position.profitPoints / max(minMovePoints, 0.25), -4.0, 4.0) / 4.0
        let cycleProfitNorm = fxClamp(
            (position.openProfit + position.cycleRealizedProfit) / max(abs(position.cyclePeakProfit) + 1e-6, 1.0),
            -1.0,
            1.0
        )
        let giveback = fxClamp(position.givebackFraction, 0.0, 1.0)
        let timeSoft = deployment.softTimeoutBars > 0
            ? fxClamp(Double(position.heldBars) / Double(deployment.softTimeoutBars), 0.0, 3.0)
            : 0.0
        let timeHard = deployment.hardTimeoutBars > 0
            ? fxClamp(Double(position.heldBars) / Double(deployment.hardTimeoutBars), 0.0, 3.0)
            : 0.0
        let controlPressure = fxClamp(controlPlaneScore / 2.0, 0.0, 1.5)
        let servicePressure = fxClamp(serviceScore / max(serviceState.blockScore, 0.20), 0.0, 1.5)
        let holdBudget = fxClamp(commandState.ready ? commandState.holdBudgetMultiplier : 1.0, 0.10, 1.20)
        let commandPressure = commandState.ready
            ? fxClamp(
                0.34 * commandState.reduceBias +
                    0.30 * commandState.exitBias +
                    0.20 * commandState.tightenBias +
                    0.16 * commandState.timeoutBias,
                0.0,
                1.0
            )
            : 0.0

        let addProb = fxClamp(
            policy.addProb +
                (directionMatch ? 0.16 : -0.18) +
                0.10 * max(profitNorm, 0.0) +
                0.08 * max(cycleProfitNorm, 0.0) -
                0.14 * giveback +
                0.10 * (lifecycleGain - 1.0) +
                0.08 * (holdBudget - 1.0) +
                0.10 * policy.capitalEfficiency +
                0.08 * policy.portfolioFit -
                0.14 * fxClamp(policy.portfolioPressure / 1.5, 0.0, 1.0) -
                0.12 * controlPressure -
                0.10 * servicePressure -
                0.12 * commandPressure +
                0.10 * (commandState.ready ? commandState.addCapMultiplier - 1.0 : 0.0),
            0.0,
            1.0
        )
        let reduceProb = fxClamp(
            policy.reduceProb +
                0.16 * fxClamp(policy.portfolioPressure / 1.5, 0.0, 1.0) +
                0.10 * controlPressure +
                0.12 * servicePressure +
                0.14 * (1.0 - holdBudget) +
                0.12 * giveback +
                0.08 * max(1.0 - lifecycleGain, 0.0) +
                0.10 * commandPressure +
                0.12 * max(-profitNorm, 0.0) +
                (oppositeSignal ? 0.22 : 0.0),
            0.0,
            1.0
        )
        let timeoutProb = fxClamp(
            policy.timeoutProb +
                0.20 * fxClamp(timeSoft / 1.5, 0.0, 1.0) +
                0.18 * fxClamp(timeHard - 1.0, 0.0, 1.0) +
                0.10 * (1.0 - holdBudget) +
                0.10 * servicePressure +
                0.16 * giveback +
                0.08 * max(1.0 - lifecycleGain, 0.0) +
                0.14 * (commandState.ready ? commandState.timeoutBias : 0.0),
            0.0,
            1.0
        )
        let tightenProb = fxClamp(
            policy.tightenProb +
                0.18 * fxClamp(timeSoft / 1.5, 0.0, 1.0) +
                0.12 * max(profitNorm, 0.0) +
                0.10 * (1.0 - holdBudget) +
                0.18 * giveback +
                0.06 * (lifecycleGain - 1.0) +
                0.10 * serviceState.reduceBias +
                0.14 * (commandState.ready ? commandState.tightenBias : 0.0),
            0.0,
            1.0
        )
        let exitProb = fxClamp(
            policy.exitProb +
                0.18 * serviceState.exitBias +
                0.14 * fxClamp(timeHard - 1.0, 0.0, 1.0) +
                0.10 * (1.0 - holdBudget) +
                0.18 * giveback +
                0.08 * max(1.0 - lifecycleGain, 0.0) +
                0.12 * (commandState.ready ? commandState.exitBias : 0.0) +
                0.12 * max(-profitNorm, 0.0) +
                (oppositeSignal ? 0.24 : 0.0),
            0.0,
            1.0
        )

        return TradeLifecycleProbabilities(add: addProb, reduce: reduceProb, timeout: timeoutProb, tighten: tightenProb, exit: exitProb)
    }

    public static func recommendation(
        position: TradeLifecyclePositionState,
        signalDirection: Int,
        policy: TradeLifecyclePolicyState,
        minMovePoints: Double,
        deployment: LiveDeploymentProfile = LiveDeploymentProfile(),
        serviceState: SupervisorServiceState = SupervisorServiceState(),
        commandState: SupervisorCommandState = SupervisorCommandState(),
        controlPlaneScore: Double = 0.0,
        serviceScore: Double? = nil
    ) -> TradeLifecycleRecommendation {
        guard position.hasPosition else {
            return TradeLifecycleRecommendation(reason: "lifecycle_idle")
        }
        let probabilities = probabilities(
            position: position,
            signalDirection: signalDirection,
            policy: policy,
            minMovePoints: minMovePoints,
            deployment: deployment,
            serviceState: serviceState,
            commandState: commandState,
            controlPlaneScore: controlPlaneScore,
            serviceScore: serviceScore
        )
        if position.heldBars >= deployment.hardTimeoutBars,
           deployment.hardTimeoutBars > 0,
           probabilities.timeout >= fxClamp(deployment.policyTimeoutFloor, 0.30, 0.99) ||
            probabilities.exit >= fxClamp(deployment.policyExitFloor, 0.20, 0.99) {
            return TradeLifecycleRecommendation(action: .timeout, reason: "lifecycle_timeout_exit", probabilities: probabilities)
        }
        if probabilities.exit >= fxClamp(deployment.policyExitFloor, 0.20, 0.99) {
            return TradeLifecycleRecommendation(action: .exit, reason: "lifecycle_exit", probabilities: probabilities)
        }
        if probabilities.reduce >= fxClamp(deployment.policyReduceFloor, 0.25, 0.99) {
            let reduceFraction = fxClamp(
                deployment.reduceFraction *
                    (0.70 + 0.30 * probabilities.reduce) *
                    (serviceState.ready ? (0.85 + 0.35 * serviceState.reduceBias) : 1.0),
                0.05,
                0.95
            )
            return TradeLifecycleRecommendation(
                action: .reduce,
                reason: "lifecycle_reduce",
                probabilities: probabilities,
                reduceFraction: reduceFraction
            )
        }
        if probabilities.tighten >= 0.52 ||
            (position.heldBars >= deployment.softTimeoutBars && deployment.softTimeoutBars > 0 && position.profitPoints > 0.0) {
            return TradeLifecycleRecommendation(action: .tighten, reason: "lifecycle_tighten", probabilities: probabilities)
        }
        if signalDirection == position.direction,
           position.pendingCount <= 0,
           probabilities.add >= fxClamp(deployment.policyAddFloor, 0.30, 0.99) {
            let addSizeMultiplier = fxClamp(deployment.maxAddFraction, 0.05, 1.00) *
                (serviceState.ready ? serviceState.addMultiplier : 1.0) *
                (commandState.ready ? commandState.addCapMultiplier : 1.0)
            return TradeLifecycleRecommendation(
                action: .add,
                reason: "lifecycle_add",
                probabilities: probabilities,
                addSizeMultiplier: addSizeMultiplier
            )
        }
        let action: PolicyLifecycleAction = policy.holdQuality >= fxClamp(deployment.policyHoldFloor, 0.20, 0.95)
            ? .hold
            : .noTrade
        return TradeLifecycleRecommendation(action: action, reason: "lifecycle_hold", probabilities: probabilities)
    }
}
