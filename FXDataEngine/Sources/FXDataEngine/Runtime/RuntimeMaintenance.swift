import Foundation

public enum RuntimeMaintenanceAction: String, Codable, Sendable, CaseIterable {
    case resetCycleState = "RESET_CYCLE_STATE"
    case recoverCycleState = "RECOVER_CYCLE_STATE"
    case clearLastOrderRequest = "CLEAR_LAST_ORDER_REQUEST"
}

public struct RuntimeInvariantInput: Codable, Hashable, Sendable {
    public var symbolExposureCount: Int
    public var cycleActive: Bool
    public var lastOrderRequestPending: Bool
    public var lastOrderRequestUsesPendingOrder: Bool
    public var lastOrderRequestSymbol: String
    public var lastOrderRequestSymbolOrderCount: Int
    public var lastOrderRequestTimeUTC: Int64
    public var nowUTC: Int64

    public init(
        symbolExposureCount: Int = 0,
        cycleActive: Bool = false,
        lastOrderRequestPending: Bool = false,
        lastOrderRequestUsesPendingOrder: Bool = false,
        lastOrderRequestSymbol: String = "",
        lastOrderRequestSymbolOrderCount: Int = 0,
        lastOrderRequestTimeUTC: Int64 = 0,
        nowUTC: Int64 = 0
    ) {
        self.symbolExposureCount = max(0, symbolExposureCount)
        self.cycleActive = cycleActive
        self.lastOrderRequestPending = lastOrderRequestPending
        self.lastOrderRequestUsesPendingOrder = lastOrderRequestUsesPendingOrder
        self.lastOrderRequestSymbol = lastOrderRequestSymbol
        self.lastOrderRequestSymbolOrderCount = max(0, lastOrderRequestSymbolOrderCount)
        self.lastOrderRequestTimeUTC = max(0, lastOrderRequestTimeUTC)
        self.nowUTC = max(0, nowUTC)
    }
}

public struct RuntimeInvariantPlan: Codable, Hashable, Sendable {
    public var reason: String
    public var actions: [RuntimeMaintenanceAction]

    public init(reason: String = "ok", actions: [RuntimeMaintenanceAction] = []) {
        self.reason = reason.isEmpty ? "ok" : reason
        self.actions = actions
    }

    public var hasAction: Bool {
        !actions.isEmpty
    }
}

public struct RuntimeMaintenanceRunPlan: Codable, Hashable, Sendable {
    public var symbol: String
    public var timeContext: RuntimeTimeContext
    public var invariantPlan: RuntimeInvariantPlan
    public var systemHealth: SystemHealthState
    public var emitDebug: Bool
    public var debugMessage: String

    public init(
        symbol: String,
        timeContext: RuntimeTimeContext,
        invariantPlan: RuntimeInvariantPlan,
        systemHealth: SystemHealthState,
        emitDebug: Bool = false
    ) {
        self.symbol = symbol
        self.timeContext = timeContext
        self.invariantPlan = invariantPlan
        self.systemHealth = systemHealth
        self.emitDebug = emitDebug
        if emitDebug, invariantPlan.reason != "ok" {
            self.debugMessage = "runtime maintenance action: reason=\(invariantPlan.reason)"
        } else {
            self.debugMessage = ""
        }
    }

    public var ready: Bool {
        systemHealth.ready
    }
}

public enum RuntimeMaintenanceTools {
    public static let orderRequestExpirySeconds: Int64 = 7_200

    public static func checkRuntimeInvariants(_ input: RuntimeInvariantInput) -> RuntimeInvariantPlan {
        var actions: [RuntimeMaintenanceAction] = []
        var reason = "ok"
        var requestPending = input.lastOrderRequestPending

        if input.symbolExposureCount <= 0 {
            if input.cycleActive {
                appendUnique(.resetCycleState, to: &actions)
                reason = "cycle_reset_no_exposure"
            }
            if requestPending {
                appendUnique(.clearLastOrderRequest, to: &actions)
                requestPending = false
                if reason == "ok" {
                    reason = "cleared_stale_order_request"
                }
            }
            return RuntimeInvariantPlan(reason: reason, actions: actions)
        }

        if !input.cycleActive {
            appendUnique(.recoverCycleState, to: &actions)
            reason = "cycle_recovered"
        }

        if requestPending,
           input.lastOrderRequestUsesPendingOrder,
           !input.lastOrderRequestSymbol.isEmpty,
           input.lastOrderRequestSymbolOrderCount <= 0 {
            appendUnique(.clearLastOrderRequest, to: &actions)
            requestPending = false
            if reason == "ok" {
                reason = "cleared_completed_pending_request"
            }
        }

        if requestPending,
           input.lastOrderRequestTimeUTC > 0,
           input.nowUTC > input.lastOrderRequestTimeUTC,
           input.nowUTC - input.lastOrderRequestTimeUTC > orderRequestExpirySeconds {
            appendUnique(.clearLastOrderRequest, to: &actions)
            reason = "expired_order_request_state"
        }

        return RuntimeInvariantPlan(reason: reason, actions: actions)
    }

    public static func buildRunPlan(
        symbol: String,
        timeContext: RuntimeTimeContext,
        systemHealth: SystemHealthState,
        invariantInput: RuntimeInvariantInput,
        emitDebug: Bool = false
    ) -> RuntimeMaintenanceRunPlan {
        RuntimeMaintenanceRunPlan(
            symbol: symbol,
            timeContext: timeContext,
            invariantPlan: checkRuntimeInvariants(invariantInput),
            systemHealth: systemHealth,
            emitDebug: emitDebug
        )
    }

    private static func appendUnique(
        _ action: RuntimeMaintenanceAction,
        to actions: inout [RuntimeMaintenanceAction]
    ) {
        if !actions.contains(action) {
            actions.append(action)
        }
    }
}
