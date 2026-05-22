import Foundation

public enum TradeExecutionOrderType: String, Codable, Sendable, CaseIterable {
    case buy = "BUY"
    case sell = "SELL"
    case buyLimit = "BUY_LIMIT"
    case sellLimit = "SELL_LIMIT"
    case buyStop = "BUY_STOP"
    case sellStop = "SELL_STOP"
    case buyStopLimit = "BUY_STOP_LIMIT"
    case sellStopLimit = "SELL_STOP_LIMIT"
}

public struct TradeExecutionMarketContext: Codable, Hashable, Sendable {
    public var referencePrice: Double
    public var pointValue: Double
    public var digits: Int
    public var minDistancePoints: Double
    public var generatedAtUTC: Int64

    public init(
        referencePrice: Double = 0.0,
        pointValue: Double = 0.0001,
        digits: Int = 5,
        minDistancePoints: Double = 2.0,
        generatedAtUTC: Int64 = 0
    ) {
        let resolvedPoint = fxSafeFinite(pointValue, fallback: 0.0001)
        self.referencePrice = max(0.0, fxSafeFinite(referencePrice))
        self.pointValue = resolvedPoint > 0.0 ? resolvedPoint : 0.0001
        self.digits = digits >= 0 ? max(0, min(digits, 10)) : 5
        self.minDistancePoints = max(2.0, fxSafeFinite(minDistancePoints))
        self.generatedAtUTC = max(0, generatedAtUTC)
    }
}

public struct TradeExecutionPlanningInput: Codable, Hashable, Sendable {
    public var symbol: String
    public var direction: Int
    public var market: TradeExecutionMarketContext
    public var signal: RiskPolicySignalState
    public var riskConfig: RiskPolicyConfig
    public var estimatedRiskPoints: Double
    public var systemHealth: SystemHealthState?
    public var executionQualityState: String
    public var newsPulseTradeGate: String
    public var previousOrderRequestPending: Bool

    public init(
        symbol: String = "",
        direction: Int = -1,
        market: TradeExecutionMarketContext = TradeExecutionMarketContext(),
        signal: RiskPolicySignalState = RiskPolicySignalState(),
        riskConfig: RiskPolicyConfig = RiskPolicyConfig(),
        estimatedRiskPoints: Double = 0.0,
        systemHealth: SystemHealthState? = nil,
        executionQualityState: String = "",
        newsPulseTradeGate: String = "",
        previousOrderRequestPending: Bool = false
    ) {
        self.symbol = symbol
        self.direction = RuntimePolicyStageTools.normalizedDecision(direction)
        self.market = market
        self.signal = signal
        self.riskConfig = riskConfig
        self.estimatedRiskPoints = max(0.0, fxSafeFinite(estimatedRiskPoints))
        self.systemHealth = systemHealth
        self.executionQualityState = executionQualityState.uppercased()
        self.newsPulseTradeGate = newsPulseTradeGate.uppercased()
        self.previousOrderRequestPending = previousOrderRequestPending
    }
}

public struct TradeExecutionPlan: Codable, Hashable, Sendable {
    public var ready: Bool
    public var usePending: Bool
    public var orderType: TradeExecutionOrderType
    public var entryPrice: Double
    public var stopLoss: Double
    public var takeProfit: Double
    public var expiryTimeUTC: Int64
    public var mode: String

    public init(
        ready: Bool = false,
        usePending: Bool = false,
        orderType: TradeExecutionOrderType = .buy,
        entryPrice: Double = 0.0,
        stopLoss: Double = 0.0,
        takeProfit: Double = 0.0,
        expiryTimeUTC: Int64 = 0,
        mode: String = "MARKET"
    ) {
        self.ready = ready
        self.usePending = usePending
        self.orderType = orderType
        self.entryPrice = max(0.0, fxSafeFinite(entryPrice))
        self.stopLoss = max(0.0, fxSafeFinite(stopLoss))
        self.takeProfit = max(0.0, fxSafeFinite(takeProfit))
        self.expiryTimeUTC = max(0, expiryTimeUTC)
        self.mode = mode.isEmpty ? "MARKET" : mode
    }
}

public enum TradeExecutionPlanTools {
    public static func normalizeOrderPrice(_ price: Double, digits: Int) -> Double {
        guard price.isFinite, price > 0.0 else { return 0.0 }
        let safeDigits = max(0, min(digits, 10))
        let scale = pow(10.0, Double(safeDigits))
        return (price * scale).rounded() / scale
    }

    public static func tradeDistanceMinPoints(_ market: TradeExecutionMarketContext) -> Double {
        max(2.0, fxSafeFinite(market.minDistancePoints))
    }

    public static func orderTypeBucket(_ orderType: TradeExecutionOrderType) -> Int {
        switch orderType {
        case .buy, .sell:
            return 1
        case .buyLimit, .sellLimit:
            return 2
        case .buyStop, .sellStop:
            return 3
        case .buyStopLimit, .sellStopLimit:
            return 4
        }
    }

    public static func protectiveLevels(
        direction: Int,
        entryPrice: Double,
        market: TradeExecutionMarketContext,
        signal: RiskPolicySignalState,
        riskConfig: RiskPolicyConfig = RiskPolicyConfig(),
        estimatedRiskPoints: Double = 0.0
    ) -> (stopLoss: Double, takeProfit: Double) {
        let normalizedDirection = RuntimePolicyStageTools.normalizedDecision(direction)
        guard normalizedDirection == 0 || normalizedDirection == 1,
              entryPrice > 0.0,
              market.pointValue > 0.0 else {
            return (0.0, 0.0)
        }
        let minPoints = tradeDistanceMinPoints(market)
        let riskEstimate = estimatedRiskPoints > 0.0
            ? estimatedRiskPoints
            : RiskPolicyTools.estimatedRiskPoints(config: riskConfig, signal: signal)
        let riskPoints = max(riskEstimate, minPoints + 2.0)
        let rewardPoints = max(
            1.35 * max(signal.expectedMovePoints, riskPoints),
            riskPoints * 1.15
        )

        let rawStop: Double
        let rawTakeProfit: Double
        if normalizedDirection == 1 {
            rawStop = entryPrice - riskPoints * market.pointValue
            rawTakeProfit = entryPrice + rewardPoints * market.pointValue
        } else {
            rawStop = entryPrice + riskPoints * market.pointValue
            rawTakeProfit = entryPrice - rewardPoints * market.pointValue
        }
        return (
            normalizeOrderPrice(rawStop, digits: market.digits),
            normalizeOrderPrice(rawTakeProfit, digits: market.digits)
        )
    }

    public static func prefersPendingOrder(
        systemHealth: SystemHealthState?,
        executionQualityState: String,
        newsPulseTradeGate: String,
        previousOrderRequestPending: Bool
    ) -> Bool {
        if let systemHealth, systemHealth.ready, systemHealth.posture != .healthy {
            return true
        }
        let executionState = executionQualityState.uppercased()
        if executionState == "CAUTION" || executionState == "STRESSED" {
            return true
        }
        if newsPulseTradeGate.uppercased() == "CAUTION" {
            return true
        }
        return previousOrderRequestPending
    }

    public static func buildPlan(_ input: TradeExecutionPlanningInput) -> TradeExecutionPlan {
        guard input.direction == 0 || input.direction == 1,
              input.market.referencePrice > 0.0,
              input.market.pointValue > 0.0 else {
            return TradeExecutionPlan()
        }

        let minPoints = tradeDistanceMinPoints(input.market)
        let setupPoints = max(max(input.signal.minMovePoints, 4.0), minPoints + 1.0)
        let riskEstimate = input.estimatedRiskPoints > 0.0
            ? input.estimatedRiskPoints
            : RiskPolicyTools.estimatedRiskPoints(config: input.riskConfig, signal: input.signal)
        let riskPoints = max(riskEstimate, minPoints + 2.0)
        let entryOffsetPoints = max(
            0.30 * riskPoints,
            max(minPoints + 1.0, 0.35 * setupPoints)
        )
        let preferPending = prefersPendingOrder(
            systemHealth: input.systemHealth,
            executionQualityState: input.executionQualityState,
            newsPulseTradeGate: input.newsPulseTradeGate,
            previousOrderRequestPending: input.previousOrderRequestPending
        )

        var plan = TradeExecutionPlan()
        let referencePrice = input.market.referencePrice
        if preferPending {
            let useBreakout = input.signal.tradeGate >= 0.60 &&
                input.signal.hierarchyExecution >= 0.55 &&
                input.signal.pathRisk <= 0.55

            if input.direction == 1 {
                if useBreakout {
                    plan.orderType = .buyStop
                    plan.entryPrice = referencePrice + entryOffsetPoints * input.market.pointValue
                    plan.mode = "BUY_STOP"
                } else {
                    plan.orderType = .buyLimit
                    plan.entryPrice = referencePrice - entryOffsetPoints * input.market.pointValue
                    plan.mode = "BUY_LIMIT"
                }
            } else {
                if useBreakout {
                    plan.orderType = .sellStop
                    plan.entryPrice = referencePrice - entryOffsetPoints * input.market.pointValue
                    plan.mode = "SELL_STOP"
                } else {
                    plan.orderType = .sellLimit
                    plan.entryPrice = referencePrice + entryOffsetPoints * input.market.pointValue
                    plan.mode = "SELL_LIMIT"
                }
            }

            plan.usePending = true
            plan.entryPrice = normalizeOrderPrice(plan.entryPrice, digits: input.market.digits)
            plan.expiryTimeUTC = input.market.generatedAtUTC > 0 ? input.market.generatedAtUTC + 20 * 60 : 0
        } else {
            plan.usePending = false
            plan.orderType = input.direction == 1 ? .buy : .sell
            plan.entryPrice = normalizeOrderPrice(referencePrice, digits: input.market.digits)
            plan.mode = "MARKET"
        }

        guard plan.entryPrice > 0.0 else { return TradeExecutionPlan() }
        let initialProtection = protectiveLevels(
            direction: input.direction,
            entryPrice: plan.entryPrice,
            market: input.market,
            signal: input.signal,
            riskConfig: input.riskConfig,
            estimatedRiskPoints: riskPoints
        )
        plan.stopLoss = initialProtection.stopLoss
        plan.takeProfit = initialProtection.takeProfit

        if plan.usePending {
            let minDistance = minPoints * input.market.pointValue
            switch plan.orderType {
            case .buyLimit where plan.entryPrice >= referencePrice - minDistance:
                plan.entryPrice = normalizeOrderPrice(
                    referencePrice - (minPoints + 1.0) * input.market.pointValue,
                    digits: input.market.digits
                )
            case .sellLimit where plan.entryPrice <= referencePrice + minDistance:
                plan.entryPrice = normalizeOrderPrice(
                    referencePrice + (minPoints + 1.0) * input.market.pointValue,
                    digits: input.market.digits
                )
            case .buyStop where plan.entryPrice <= referencePrice + minDistance:
                plan.entryPrice = normalizeOrderPrice(
                    referencePrice + (minPoints + 1.0) * input.market.pointValue,
                    digits: input.market.digits
                )
            case .sellStop where plan.entryPrice >= referencePrice - minDistance:
                plan.entryPrice = normalizeOrderPrice(
                    referencePrice - (minPoints + 1.0) * input.market.pointValue,
                    digits: input.market.digits
                )
            default:
                break
            }

            let adjustedProtection = protectiveLevels(
                direction: input.direction,
                entryPrice: plan.entryPrice,
                market: input.market,
                signal: input.signal,
                riskConfig: input.riskConfig,
                estimatedRiskPoints: riskPoints
            )
            plan.stopLoss = adjustedProtection.stopLoss
            plan.takeProfit = adjustedProtection.takeProfit
        }

        plan.ready = true
        return plan
    }
}
