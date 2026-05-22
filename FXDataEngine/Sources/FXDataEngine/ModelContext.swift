import Foundation

public struct MoveEMAState: Codable, Hashable, Sendable {
    public var ready: Bool
    public var emaAbsMove: Double

    public init(ready: Bool = false, emaAbsMove: Double = 0.0) {
        self.ready = ready
        self.emaAbsMove = emaAbsMove
    }

    public mutating func update(movePoints: Double, alpha: Double) {
        guard movePoints.isFinite else { return }

        let clampedAlpha = fxClamp(alpha, 0.001, 0.500)
        let value = abs(movePoints)

        if !ready {
            emaAbsMove = value
            ready = true
            return
        }

        emaAbsMove = (1.0 - clampedAlpha) * emaAbsMove + clampedAlpha * value
    }
}

public enum ModelContextTools {
    public static func coreClampHorizon(_ horizonMinutes: Int) -> Int {
        min(max(horizonMinutes, 1), 1_440)
    }

    public static func symbolModelScale(_ symbol: String) -> Double {
        let uppercased = symbol.uppercased()
        if uppercased.contains("XAU") || uppercased.contains("GOLD")
            || uppercased.contains("XAG") || uppercased.contains("SILVER") {
            return 1.18
        }
        if uppercased.contains("US30") || uppercased.contains("NAS")
            || uppercased.contains("SPX") || uppercased.contains("DAX")
            || uppercased.contains("GER40") || uppercased.contains("JP225") {
            return 1.15
        }
        if uppercased.contains("OIL") || uppercased.contains("WTI")
            || uppercased.contains("BRENT") || uppercased.contains("NGAS") {
            return 1.12
        }
        if uppercased.contains("JPY") || uppercased.contains("GBP") {
            return 1.06
        }
        return 1.00
    }

    public static func horizonModelScale(_ horizonMinutes: Int) -> Double {
        let horizon = coreClampHorizon(horizonMinutes)
        if horizon <= 5 {
            return 0.92
        }
        if horizon <= 15 {
            return 0.98
        }
        if horizon <= 60 {
            return 1.00
        }
        if horizon <= 240 {
            return 1.10
        }
        return 1.18
    }

    public static func modelCapacityScale(symbol: String, horizonMinutes: Int) -> Double {
        fxClamp(symbolModelScale(symbol) * horizonModelScale(horizonMinutes), 0.85, 1.35)
    }

    public static func contextSequenceSpan(
        maxCap: Int,
        horizonMinutes: Int,
        symbol: String,
        baseMin: Int = 8
    ) -> Int {
        let cap = max(baseMin, maxCap)
        let scale = modelCapacityScale(symbol: symbol, horizonMinutes: horizonMinutes)
        var span = Int((Double(cap) * fxClamp(0.55 + 0.35 * scale, 0.45, 1.10)).rounded())
        if span < baseMin {
            span = baseMin
        }
        if span > maxCap {
            span = maxCap
        }
        return span
    }

    public static func contextBatchSpan(
        maxCap: Int,
        horizonMinutes: Int,
        symbol: String,
        baseMin: Int = 4
    ) -> Int {
        let cap = max(baseMin, maxCap)
        let scale = modelCapacityScale(symbol: symbol, horizonMinutes: horizonMinutes)
        var span = Int((Double(cap) * fxClamp(0.60 + 0.30 * scale, 0.45, 1.00)).rounded())
        if span < baseMin {
            span = baseMin
        }
        if span > maxCap {
            span = maxCap
        }
        return span
    }

    public static func contextTreeBudget(
        maxCap: Int,
        horizonMinutes: Int,
        symbol: String,
        baseMin: Int
    ) -> Int {
        let cap = max(baseMin, maxCap)
        let scale = modelCapacityScale(symbol: symbol, horizonMinutes: horizonMinutes)
        var budget = Int((Double(cap) * fxClamp(0.55 + 0.40 * scale, 0.50, 1.15)).rounded())
        if budget < baseMin {
            budget = baseMin
        }
        if budget > maxCap {
            budget = maxCap
        }
        return budget
    }

    public static func updatedMoveEMA(
        emaAbsMove: Double,
        ready: Bool,
        movePoints: Double,
        alpha: Double
    ) -> MoveEMAState {
        var state = MoveEMAState(ready: ready, emaAbsMove: emaAbsMove)
        state.update(movePoints: movePoints, alpha: alpha)
        return state
    }

    public static func threeWayBranch(_ x: Double, split: Double) -> Int {
        if x < split - 0.50 {
            return 0
        }
        if x > split + 0.50 {
            return 2
        }
        return 1
    }
}
