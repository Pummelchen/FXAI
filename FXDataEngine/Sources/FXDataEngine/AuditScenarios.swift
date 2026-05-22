import Foundation

public struct AuditScenarioSpec: Codable, Hashable, Sendable {
    public var id: Int
    public var name: String
    public var driftPerBar: Double
    public var sigmaPerBar: Double
    public var meanRevertStrength: Double
    public var volatilityCluster: Double
    public var spikeProbability: Double
    public var spikeScale: Double
    public var fillRiskPoints: Double
    public var macroFocus: Double
    public var worldSigmaScale: Double
    public var worldDriftBias: Double
    public var worldFillRiskScale: Double
    public var worldGapProbability: Double
    public var worldGapScale: Double
    public var worldFlipProbability: Double
    public var worldContextCorrelationBias: Double
    public var worldLiquidityStress: Double
    public var worldSessionEdgeFocus: Double
    public var worldTrendPersistence: Double
    public var worldShockMemory: Double
    public var worldRecoveryBias: Double
    public var worldLiquidityShockProbability: Double
    public var worldLiquidityShockScale: Double
    public var worldRegimeTransitionBurst: Double
    public var worldTransitionEntropy: Double
    public var worldMeanRevertBias: Double
    public var worldVolatilityClusterBias: Double
    public var worldShockDecay: Double
    public var worldAsiaSigmaScale: Double
    public var worldLondonSigmaScale: Double
    public var worldNewYorkSigmaScale: Double
    public var worldAsiaFillRiskScale: Double
    public var worldLondonFillRiskScale: Double
    public var worldNewYorkFillRiskScale: Double

    public init(
        id: Int = 0,
        name: String = "random_walk",
        driftPerBar: Double = 0.0,
        sigmaPerBar: Double = 0.00018,
        meanRevertStrength: Double = 0.0,
        volatilityCluster: Double = 0.0,
        spikeProbability: Double = 0.0,
        spikeScale: Double = 0.0,
        fillRiskPoints: Double = 1.2,
        macroFocus: Double = 0.0,
        worldSigmaScale: Double = 1.0,
        worldDriftBias: Double = 0.0,
        worldFillRiskScale: Double = 1.0,
        worldGapProbability: Double = 0.0,
        worldGapScale: Double = 0.0,
        worldFlipProbability: Double = 0.0,
        worldContextCorrelationBias: Double = 0.0,
        worldLiquidityStress: Double = 0.0,
        worldSessionEdgeFocus: Double = 0.0,
        worldTrendPersistence: Double = 0.5,
        worldShockMemory: Double = 0.0,
        worldRecoveryBias: Double = 0.0,
        worldLiquidityShockProbability: Double = 0.0,
        worldLiquidityShockScale: Double = 1.0,
        worldRegimeTransitionBurst: Double = 0.0,
        worldTransitionEntropy: Double = 0.0,
        worldMeanRevertBias: Double = 0.0,
        worldVolatilityClusterBias: Double = 0.0,
        worldShockDecay: Double = 0.6,
        worldAsiaSigmaScale: Double = 1.0,
        worldLondonSigmaScale: Double = 1.0,
        worldNewYorkSigmaScale: Double = 1.0,
        worldAsiaFillRiskScale: Double = 1.0,
        worldLondonFillRiskScale: Double = 1.0,
        worldNewYorkFillRiskScale: Double = 1.0
    ) {
        self.id = id
        self.name = name
        self.driftPerBar = fxSafeFinite(driftPerBar)
        self.sigmaPerBar = fxSafeFinite(sigmaPerBar)
        self.meanRevertStrength = fxSafeFinite(meanRevertStrength)
        self.volatilityCluster = fxSafeFinite(volatilityCluster)
        self.spikeProbability = fxSafeFinite(spikeProbability)
        self.spikeScale = fxSafeFinite(spikeScale)
        self.fillRiskPoints = fxSafeFinite(fillRiskPoints)
        self.macroFocus = fxSafeFinite(macroFocus)
        self.worldSigmaScale = fxSafeFinite(worldSigmaScale)
        self.worldDriftBias = fxSafeFinite(worldDriftBias)
        self.worldFillRiskScale = fxSafeFinite(worldFillRiskScale)
        self.worldGapProbability = fxSafeFinite(worldGapProbability)
        self.worldGapScale = fxSafeFinite(worldGapScale)
        self.worldFlipProbability = fxSafeFinite(worldFlipProbability)
        self.worldContextCorrelationBias = fxSafeFinite(worldContextCorrelationBias)
        self.worldLiquidityStress = fxSafeFinite(worldLiquidityStress)
        self.worldSessionEdgeFocus = fxSafeFinite(worldSessionEdgeFocus)
        self.worldTrendPersistence = fxSafeFinite(worldTrendPersistence)
        self.worldShockMemory = fxSafeFinite(worldShockMemory)
        self.worldRecoveryBias = fxSafeFinite(worldRecoveryBias)
        self.worldLiquidityShockProbability = fxSafeFinite(worldLiquidityShockProbability)
        self.worldLiquidityShockScale = fxSafeFinite(worldLiquidityShockScale)
        self.worldRegimeTransitionBurst = fxSafeFinite(worldRegimeTransitionBurst)
        self.worldTransitionEntropy = fxSafeFinite(worldTransitionEntropy)
        self.worldMeanRevertBias = fxSafeFinite(worldMeanRevertBias)
        self.worldVolatilityClusterBias = fxSafeFinite(worldVolatilityClusterBias)
        self.worldShockDecay = fxSafeFinite(worldShockDecay)
        self.worldAsiaSigmaScale = fxSafeFinite(worldAsiaSigmaScale)
        self.worldLondonSigmaScale = fxSafeFinite(worldLondonSigmaScale)
        self.worldNewYorkSigmaScale = fxSafeFinite(worldNewYorkSigmaScale)
        self.worldAsiaFillRiskScale = fxSafeFinite(worldAsiaFillRiskScale)
        self.worldLondonFillRiskScale = fxSafeFinite(worldLondonFillRiskScale)
        self.worldNewYorkFillRiskScale = fxSafeFinite(worldNewYorkFillRiskScale)
    }
}

public struct AuditScenarioDoubleBar: Codable, Hashable, Sendable {
    public var timestampUTC: Int64
    public var open: Double
    public var high: Double
    public var low: Double
    public var close: Double
    public var volume: Double
    public var fillRiskPoints: Double

    public init(
        timestampUTC: Int64,
        open: Double,
        high: Double,
        low: Double,
        close: Double,
        volume: Double = 0.0,
        fillRiskPoints: Double = 0.0
    ) {
        self.timestampUTC = timestampUTC
        self.open = fxSafeFinite(open)
        self.high = fxSafeFinite(high)
        self.low = fxSafeFinite(low)
        self.close = fxSafeFinite(close)
        self.volume = max(0.0, fxSafeFinite(volume))
        self.fillRiskPoints = max(0.0, fxSafeFinite(fillRiskPoints))
    }
}

public struct AuditCloseTimeframeSeries: Codable, Hashable, Sendable {
    public var timeUTC: [Int64]
    public var close: [Double]
    public var alignedIndexMap: [Int]

    public init(timeUTC: [Int64] = [], close: [Double] = [], alignedIndexMap: [Int] = []) {
        self.timeUTC = timeUTC
        self.close = close.map { fxSafeFinite($0) }
        self.alignedIndexMap = alignedIndexMap
    }
}

public struct AuditGeneratedScenarioSeries: Codable, Hashable, Sendable {
    public var primary: AuditAsSeriesOHLCV
    public var m5: AuditCloseTimeframeSeries
    public var m15: AuditCloseTimeframeSeries
    public var m30: AuditCloseTimeframeSeries
    public var h1: AuditCloseTimeframeSeries
    public var contexts: [AuditAsSeriesOHLCV]
    public var contextFeatures: AuditContextFeatureSet

    public init(
        primary: AuditAsSeriesOHLCV,
        m5: AuditCloseTimeframeSeries,
        m15: AuditCloseTimeframeSeries,
        m30: AuditCloseTimeframeSeries,
        h1: AuditCloseTimeframeSeries,
        contexts: [AuditAsSeriesOHLCV],
        contextFeatures: AuditContextFeatureSet
    ) {
        self.primary = primary
        self.m5 = m5
        self.m15 = m15
        self.m30 = m30
        self.h1 = h1
        self.contexts = contexts
        self.contextFeatures = contextFeatures
    }
}

struct AuditScenarioRandomGenerator: Sendable {
    private var state: UInt64 = 881_726_454

    init(seed: UInt64) {
        self.state = seed == 0 ? 881_726_454 : seed
    }

    mutating func nextUInt64() -> UInt64 {
        state ^= state &<< 13
        state ^= state >> 7
        state ^= state &<< 17
        return state
    }

    mutating func nextUnit() -> Double {
        let value = nextUInt64() % 2_147_483_647
        return fxClamp(Double(value) / 2_147_483_646.0, 0.0, 1.0)
    }

    mutating func nextNormal() -> Double {
        let u1 = max(nextUnit(), 1e-9)
        let u2 = nextUnit()
        return sqrt(-2.0 * log(u1)) * cos(2.0 * Double.pi * u2)
    }
}

public enum AuditScenarioTools {
    public static func worldPlanFile(symbol: String) -> String {
        "\(ControlPlaneConstants.promotionsDirectory)/fxai_world_plan_" +
            "\(ControlPlanePaths.safeToken(symbol, defaultValue: "default")).tsv"
    }

    public static func scenarioSpec(
        scenarioID: Int,
        worldPlanTSV: String? = nil
    ) -> AuditScenarioSpec {
        var spec = AuditScenarioSpec(id: scenarioID)

        switch scenarioID {
        case 1:
            spec.name = "drift_up"
            spec.driftPerBar = 0.00010
            spec.sigmaPerBar = 0.00015
            spec.fillRiskPoints = 1.0
        case 2:
            spec.name = "drift_down"
            spec.driftPerBar = -0.00010
            spec.sigmaPerBar = 0.00015
            spec.fillRiskPoints = 1.0
        case 3:
            spec.name = "mean_revert"
            spec.meanRevertStrength = 0.22
            spec.fillRiskPoints = 1.3
        case 4:
            spec.name = "vol_cluster"
            spec.volatilityCluster = 0.85
            spec.spikeProbability = 0.01
            spec.spikeScale = 4.0
            spec.fillRiskPoints = 1.8
        case 5:
            spec.name = "monotonic_up"
            spec.driftPerBar = 0.00022
            spec.sigmaPerBar = 0.00003
            spec.fillRiskPoints = 0.8
        case 6:
            spec.name = "monotonic_down"
            spec.driftPerBar = -0.00022
            spec.sigmaPerBar = 0.00003
            spec.fillRiskPoints = 0.8
        case 7:
            spec.name = "regime_shift"
            spec.driftPerBar = 0.00008
            spec.sigmaPerBar = 0.00015
            spec.volatilityCluster = 0.55
            spec.spikeProbability = 0.005
            spec.spikeScale = 3.0
            spec.fillRiskPoints = 1.5
        case 8:
            spec.name = "market_recent"
            spec.fillRiskPoints = 1.2
        case 9:
            spec.name = "market_trend"
            spec.fillRiskPoints = 1.2
        case 10:
            spec.name = "market_chop"
            spec.fillRiskPoints = 1.4
        case 11:
            spec.name = "market_session_edges"
            spec.fillRiskPoints = 1.6
        case 12:
            spec.name = "market_liquidity_shock"
            spec.fillRiskPoints = 2.2
        case 13:
            spec.name = "market_walkforward"
            spec.fillRiskPoints = 1.5
        case 14:
            spec.name = "market_macro_event"
            spec.fillRiskPoints = 1.7
            spec.macroFocus = 1.0
        case 15:
            spec.name = "market_adversarial"
            spec.fillRiskPoints = 1.9
            spec.macroFocus = 0.5
        default:
            break
        }

        if scenarioID >= 11, let worldPlanTSV {
            applyWorldPlan(tsv: worldPlanTSV, to: &spec)
        }

        return spec
    }

    public static func applyWorldPlan(tsv: String, to spec: inout AuditScenarioSpec) {
        for line in tsv.components(separatedBy: .newlines) where !line.isEmpty {
            let parts = line.split(separator: "\t", maxSplits: 1, omittingEmptySubsequences: false)
            guard parts.count >= 2 else { continue }
            let key = String(parts[0]).trimmingCharacters(in: .whitespacesAndNewlines)
            let value = mqlStringToDouble(String(parts[1]).trimmingCharacters(in: .whitespacesAndNewlines))

            switch key {
            case "sigma_scale":
                spec.worldSigmaScale = value
            case "drift_bias":
                spec.worldDriftBias = value
            case "spread_scale", "fill_risk_scale", "price_cost_scale":
                spec.worldFillRiskScale = value
            case "gap_prob":
                spec.worldGapProbability = value
            case "gap_scale":
                spec.worldGapScale = value
            case "flip_prob":
                spec.worldFlipProbability = value
            case "context_corr_bias":
                spec.worldContextCorrelationBias = value
            case "liquidity_stress":
                spec.worldLiquidityStress = value
            case "session_edge_focus":
                spec.worldSessionEdgeFocus = value
            case "trend_persistence":
                spec.worldTrendPersistence = value
            case "shock_memory":
                spec.worldShockMemory = value
            case "recovery_bias":
                spec.worldRecoveryBias = value
            case "spread_shock_prob", "liquidity_shock_prob":
                spec.worldLiquidityShockProbability = value
            case "spread_shock_scale", "liquidity_shock_scale":
                spec.worldLiquidityShockScale = value
            case "regime_transition_burst":
                spec.worldRegimeTransitionBurst = value
            case "transition_entropy":
                spec.worldTransitionEntropy = value
            case "mean_revert_bias":
                spec.worldMeanRevertBias = value
            case "vol_cluster_bias":
                spec.worldVolatilityClusterBias = value
            case "shock_decay":
                spec.worldShockDecay = value
            case "asia_sigma_scale":
                spec.worldAsiaSigmaScale = value
            case "london_sigma_scale":
                spec.worldLondonSigmaScale = value
            case "newyork_sigma_scale":
                spec.worldNewYorkSigmaScale = value
            case "asia_spread_scale", "asia_fill_risk_scale":
                spec.worldAsiaFillRiskScale = value
            case "london_spread_scale", "london_fill_risk_scale":
                spec.worldLondonFillRiskScale = value
            case "newyork_spread_scale", "newyork_fill_risk_scale":
                spec.worldNewYorkFillRiskScale = value
            case "macro_focus":
                spec.macroFocus = value
            default:
                continue
            }
        }

        clampWorldPlan(&spec)
    }

    public static func worldHashUnit(timestampUTC: Int64, salt: Int) -> Double {
        var x = timestampUTC &+ (Int64(salt) &+ 1) &* 1_315_423_911
        x = x ^ (x &<< 13)
        x = x ^ (x >> 17)
        x = x ^ (x &<< 5)
        if x < 0 {
            x = x == Int64.min ? Int64.max : -x
        }
        return Double(x % 1_000_000) / 1_000_000.0
    }

    public static func worldSign(timestampUTC: Int64, salt: Int) -> Double {
        worldHashUnit(timestampUTC: timestampUTC, salt: salt) < 0.5 ? -1.0 : 1.0
    }

    public static func hourOf(timestampUTC: Int64) -> Int {
        let date = Date(timeIntervalSince1970: TimeInterval(max(0, timestampUTC)))
        var calendar = Calendar(identifier: .gregorian)
        calendar.timeZone = TimeZone(secondsFromGMT: 0)!
        return calendar.component(.hour, from: date)
    }

    public static func sessionEdgeStrength(timestampUTC: Int64) -> Double {
        let hour = Double(hourOf(timestampUTC: timestampUTC))
        let londonDistance = abs(hour - 8.0)
        let newYorkDistance = abs(hour - 16.0)
        let rolloverDistance = min(abs(hour - 23.0), abs(hour - 0.0))
        let edge = max(
            1.0 - londonDistance / 4.0,
            1.0 - newYorkDistance / 4.0,
            1.0 - rolloverDistance / 3.0
        )
        return fxClamp(edge, 0.0, 1.0)
    }

    public static func sessionSigmaScale(spec: AuditScenarioSpec, hour: Int) -> Double {
        if hour >= 7, hour <= 12 {
            return fxClamp(spec.worldLondonSigmaScale, 0.50, 3.00)
        }
        if hour >= 13, hour <= 20 {
            return fxClamp(spec.worldNewYorkSigmaScale, 0.50, 3.00)
        }
        return fxClamp(spec.worldAsiaSigmaScale, 0.50, 3.00)
    }

    public static func sessionFillRiskScale(spec: AuditScenarioSpec, hour: Int) -> Double {
        if hour >= 7, hour <= 12 {
            return fxClamp(spec.worldLondonFillRiskScale, 0.50, 4.00)
        }
        if hour >= 13, hour <= 20 {
            return fxClamp(spec.worldNewYorkFillRiskScale, 0.50, 4.00)
        }
        return fxClamp(spec.worldAsiaFillRiskScale, 0.50, 4.00)
    }

    public static func normalizeSyntheticBar(_ bar: AuditScenarioDoubleBar, point: Double) -> AuditScenarioDoubleBar {
        let pt = point > 0.0 ? point : 1e-5
        var output = bar
        if output.open <= pt {
            output.open = max(output.close, 10.0 * pt)
        }
        if output.close <= pt {
            output.close = max(output.open, 10.0 * pt)
        }

        let bodyHigh = max(output.open, output.close)
        let bodyLow = min(output.open, output.close)
        if output.high < bodyHigh {
            output.high = bodyHigh
        }
        if output.low <= 0.0 || output.low > bodyLow {
            output.low = max(pt, bodyLow - pt)
        }
        if output.high - output.low < 2.0 * pt {
            output.high = bodyHigh + pt
            output.low = max(pt, bodyLow - pt)
        }
        output.volume = max(0.0, fxSafeFinite(output.volume))
        output.fillRiskPoints = max(0.0, fxSafeFinite(output.fillRiskPoints))
        return output
    }

    public static func applyWorldPlanToAsSeriesBars(
        _ bars: [AuditScenarioDoubleBar],
        spec: AuditScenarioSpec,
        point: Double
    ) -> [AuditScenarioDoubleBar] {
        guard !bars.isEmpty, point > 0.0 else { return bars }

        var output = bars
        var previousTransformedClose = 0.0
        var previousTransformedReturn = 0.0
        var previousShockStrength = 0.0
        let pt = max(point, 1e-5)

        for index in stride(from: output.count - 1, through: 0, by: -1) {
            var bar = output[index]
            let originalOpen = max(bar.open, pt)
            let originalClose = max(bar.close, pt)
            let originalHigh = max(bar.high, max(originalOpen, originalClose))
            let originalLow = max(pt, min(bar.low, min(originalOpen, originalClose)))
            var originalPreviousClose = originalOpen
            if index < output.count - 1 {
                originalPreviousClose = max(output[index + 1].close, pt)
            }
            if previousTransformedClose <= pt {
                previousTransformedClose = originalPreviousClose
            }

            var baseReturn = originalPreviousClose > pt ? (originalClose - originalPreviousClose) / originalPreviousClose : 0.0
            let baseGap = originalPreviousClose > pt ? (originalOpen - originalPreviousClose) / originalPreviousClose : 0.0
            let baseRange = max(originalHigh - originalLow, 2.0 * pt)
            let bodyHigh = max(originalOpen, originalClose)
            let bodyLow = min(originalOpen, originalClose)
            let upperRatio = max(originalHigh - bodyHigh, 0.0) / baseRange
            let lowerRatio = max(bodyLow - originalLow, 0.0) / baseRange
            let sessionEdge = sessionEdgeStrength(timestampUTC: bar.timestampUTC)
            let hour = hourOf(timestampUTC: bar.timestampUTC)

            var sigmaScale = fxClamp(spec.worldSigmaScale, 0.50, 3.00)
            let edgeFocus = fxClamp(spec.worldSessionEdgeFocus, 0.0, 1.5)
            let persistence = fxClamp(spec.worldTrendPersistence, 0.0, 1.0)
            let shockMemory = fxClamp(spec.worldShockMemory, 0.0, 1.0)
            let shockDecay = fxClamp(spec.worldShockDecay, 0.0, 1.5)
            let recoveryBias = fxClamp(spec.worldRecoveryBias, -1.0, 1.0)
            let transitionBurst = fxClamp(spec.worldRegimeTransitionBurst, 0.0, 1.0)
            let transitionEntropy = fxClamp(spec.worldTransitionEntropy, 0.0, 1.0)
            let meanRevertBias = fxClamp(spec.worldMeanRevertBias, 0.0, 1.0)
            let volatilityClusterBias = fxClamp(spec.worldVolatilityClusterBias, 0.0, 1.0)
            let gapProbability = fxClamp(spec.worldGapProbability, 0.0, 0.30)
            let gapScale = fxClamp(spec.worldGapScale, 0.0, 8.0)
            var fillRiskScale = fxClamp(spec.worldFillRiskScale, 0.50, 4.00)
            sigmaScale *= sessionSigmaScale(spec: spec, hour: hour)
            fillRiskScale *= sessionFillRiskScale(spec: spec, hour: hour)
            let liquidity = fxClamp(spec.worldLiquidityStress, 0.0, 3.0)
            let liquidityShockProbability = fxClamp(spec.worldLiquidityShockProbability, 0.0, 0.50)
            let liquidityShockScale = fxClamp(spec.worldLiquidityShockScale, 1.0, 8.0)
            let flipProbability = fxClamp(spec.worldFlipProbability, 0.0, 0.50)

            let edgeVolatilityMultiplier = 1.0 +
                0.32 * edgeFocus * sessionEdge +
                0.18 * volatilityClusterBias * previousShockStrength
            let trendBias = (persistence - 0.50) * (0.30 - 0.18 * meanRevertBias) * abs(previousTransformedReturn)
            if previousTransformedReturn >= 0.0 {
                baseReturn += trendBias
            } else {
                baseReturn -= trendBias
            }
            if previousShockStrength > 0.0 {
                let shockTerm = previousShockStrength * (0.18 * shockMemory - 0.14 * recoveryBias)
                if previousTransformedReturn >= 0.0 {
                    baseReturn += shockTerm
                } else {
                    baseReturn -= shockTerm
                }
            }

            var transformedReturn = baseReturn * sigmaScale * edgeVolatilityMultiplier +
                spec.worldDriftBias * (0.70 + 0.30 * sessionEdge)
            let liveFlipProbability = fxClamp(
                flipProbability +
                    0.14 * meanRevertBias +
                    0.10 * transitionBurst * (1.0 - sessionEdge) +
                    0.08 * transitionEntropy,
                0.0,
                0.65
            )
            if worldHashUnit(timestampUTC: bar.timestampUTC, salt: 5) < liveFlipProbability {
                transformedReturn *= -1.0
            }

            let microGap = 0.40 * pt / max(previousTransformedClose, pt)
            var gapTerm = baseGap * (0.65 + 0.35 * sigmaScale) +
                0.06 * sessionEdge * edgeFocus * worldSign(timestampUTC: bar.timestampUTC, salt: 7) * microGap
            let liveGapProbability = fxClamp(gapProbability + 0.10 * transitionBurst, 0.0, 0.40)
            if worldHashUnit(timestampUTC: bar.timestampUTC, salt: 11) < liveGapProbability {
                gapTerm += gapScale *
                    max(abs(baseReturn), 0.20 * pt / max(previousTransformedClose, pt)) *
                    worldSign(timestampUTC: bar.timestampUTC, salt: 13)
            }

            var newOpen = previousTransformedClose * (1.0 + gapTerm)
            if newOpen <= pt {
                newOpen = previousTransformedClose
            }
            if newOpen <= pt {
                newOpen = originalOpen
            }

            var newClose = previousTransformedClose * (1.0 + transformedReturn)
            if newClose <= pt {
                newClose = max(newOpen, originalClose)
            }

            var rangeScale = (0.78 + 0.22 * sigmaScale) *
                (1.0 + 0.22 * edgeFocus * sessionEdge + 0.10 * liquidity + 0.12 * transitionBurst)
            if previousShockStrength > 0.0 {
                rangeScale *= 1.0 +
                    0.20 * previousShockStrength *
                    (0.60 + shockMemory + 0.35 * volatilityClusterBias) *
                    (1.0 + 0.20 * transitionEntropy)
            }
            let newRange = max(2.0 * pt, baseRange * rangeScale)
            var wickUp = max(0.5 * pt, newRange * max(upperRatio, 0.15))
            var wickDown = max(0.5 * pt, newRange * max(lowerRatio, 0.15))
            if recoveryBias > 0.0, previousShockStrength > 0.0 {
                if transformedReturn >= 0.0 {
                    wickDown *= 1.0 + 0.20 * recoveryBias * previousShockStrength
                } else {
                    wickUp *= 1.0 + 0.20 * recoveryBias * previousShockStrength
                }
            }

            let newBodyHigh = max(newOpen, newClose)
            let newBodyLow = min(newOpen, newClose)
            bar.open = newOpen
            bar.close = newClose
            bar.high = newBodyHigh + wickUp
            bar.low = max(pt, newBodyLow - wickDown)

            var liveFillRiskScale = fillRiskScale * (1.0 + 0.20 * edgeFocus * sessionEdge + 0.16 * liquidity)
            let liquidityShock = worldHashUnit(timestampUTC: bar.timestampUTC, salt: 17) < liquidityShockProbability
            if liquidityShock {
                liveFillRiskScale *= liquidityShockScale
            }
            let baseFillRisk = max(bar.fillRiskPoints, spec.fillRiskPoints, 0.25)
            bar.fillRiskPoints = baseFillRisk * liveFillRiskScale
            if bar.volume > 0.0 {
                let shockVolumeDrag = liquidityShock ? 0.08 * liquidityShockScale : 0.0
                let volumeScale = fxClamp(
                    1.0 + 0.10 * edgeFocus * sessionEdge - 0.18 * liquidity - shockVolumeDrag,
                    0.10,
                    3.0
                )
                bar.volume = max(0.0, bar.volume * volumeScale)
            }

            bar = normalizeSyntheticBar(bar, point: pt)
            output[index] = bar

            previousTransformedReturn = previousTransformedClose > pt
                ? (bar.close - previousTransformedClose) / previousTransformedClose
                : 0.0
            previousShockStrength = fxClamp(
                (abs(previousTransformedReturn) / max(abs(baseReturn) + 1e-6, 1e-6)) *
                    (1.0 - 0.25 * shockDecay) +
                    0.10 * transitionEntropy,
                0.0,
                3.0
            )
            previousTransformedClose = bar.close
        }

        return output
    }

    public static func generateMarketScenarioSeries(
        spec: AuditScenarioSpec,
        marketSeries: M1OHLCVSeries,
        bars: Int,
        point: Double,
        applyWorldPlan: Bool = true
    ) -> AuditGeneratedScenarioSeries? {
        guard spec.id >= 8,
              bars >= 512,
              point > 0.0,
              marketSeries.count >= bars + 64,
              let startIndex = selectMarketScenarioStart(
                spec: spec,
                marketSeries: marketSeries,
                bars: bars,
                point: point
              ) else {
            return nil
        }

        let selectedChronological = marketScenarioBars(
            from: marketSeries,
            range: startIndex..<(startIndex + bars),
            point: point
        )
        guard selectedChronological.count == bars else { return nil }

        var asSeriesBars = Array(selectedChronological.reversed())
        if applyWorldPlan {
            asSeriesBars = applyWorldPlanToAsSeriesBars(asSeriesBars, spec: spec, point: point)
        }
        let transformedChronological = Array(asSeriesBars.reversed())
        let primary = asSeriesOHLCV(fromAsSeriesBars: asSeriesBars)
        guard primary.isConsistent, primary.count == bars else { return nil }

        let contexts = [
            AuditContextSeriesTools.deriveContextSeriesFromBase(point: point, base: primary, transformID: 0),
            AuditContextSeriesTools.deriveContextSeriesFromBase(point: point, base: primary, transformID: 1),
            AuditContextSeriesTools.deriveContextSeriesFromBase(point: point, base: primary, transformID: 2)
        ]

        return buildGeneratedScenarioSeries(
            primary: primary,
            chronologicalBars: transformedChronological,
            point: point,
            contexts: contexts
        )
    }

    public static func generatedScenarioSeries(
        chronologicalBars: [AuditScenarioDoubleBar],
        point: Double
    ) -> AuditGeneratedScenarioSeries? {
        guard !chronologicalBars.isEmpty else { return nil }
        let primary = AuditContextSeriesTools.reverseChronologicalBarsToSeries(chronologicalBars)
        guard primary.isConsistent, primary.count > 0 else { return nil }
        let contexts = [
            AuditContextSeriesTools.deriveContextSeriesFromBase(point: point, base: primary, transformID: 0),
            AuditContextSeriesTools.deriveContextSeriesFromBase(point: point, base: primary, transformID: 1),
            AuditContextSeriesTools.deriveContextSeriesFromBase(point: point, base: primary, transformID: 2)
        ]
        return buildGeneratedScenarioSeries(
            primary: primary,
            chronologicalBars: chronologicalBars,
            point: point,
            contexts: contexts
        )
    }

    public static func generateSyntheticScenarioSeries(
        spec: AuditScenarioSpec,
        bars: Int,
        seed: UInt64,
        point: Double
    ) -> AuditGeneratedScenarioSeries? {
        guard bars >= 512, point > 0.0, spec.id < 8 else { return nil }

        var rng = AuditScenarioRandomGenerator(seed: seed ^ (UInt64(max(spec.id + 1, 0)) &* 2_654_435_761))
        let startUTC: Int64 = 1_704_067_200
        let anchor = 1.10000
        let pt = max(point, 1e-5)
        var chronologicalBars: [AuditScenarioDoubleBar] = []
        var context1Bars: [AuditScenarioDoubleBar] = []
        var context2Bars: [AuditScenarioDoubleBar] = []
        var context3Bars: [AuditScenarioDoubleBar] = []
        chronologicalBars.reserveCapacity(bars)
        context1Bars.reserveCapacity(bars)
        context2Bars.reserveCapacity(bars)
        context3Bars.reserveCapacity(bars)

        var previous = anchor
        var previousSigma = spec.sigmaPerBar
        var previousReturn = 0.0
        var previousShockStrength = 0.0
        var contextPrevious1 = anchor * 0.97
        var contextPrevious2 = anchor * 1.03
        var contextPrevious3 = anchor * 0.91

        for offset in 0..<bars {
            let timestampUTC = startUTC + Int64(60 * offset)
            let sessionEdge = sessionEdgeStrength(timestampUTC: timestampUTC)
            let edgeFocus = fxClamp(spec.worldSessionEdgeFocus, 0.0, 1.5)
            let persistence = fxClamp(spec.worldTrendPersistence, 0.0, 1.0)
            let shockMemory = fxClamp(spec.worldShockMemory, 0.0, 1.0)
            let shockDecay = fxClamp(spec.worldShockDecay, 0.0, 1.5)
            let recoveryBias = fxClamp(spec.worldRecoveryBias, -1.0, 1.0)
            let transitionBurst = fxClamp(spec.worldRegimeTransitionBurst, 0.0, 1.0)
            let transitionEntropy = fxClamp(spec.worldTransitionEntropy, 0.0, 1.0)
            let meanRevertBias = fxClamp(spec.worldMeanRevertBias, 0.0, 1.0)
            let volatilityClusterBias = fxClamp(spec.worldVolatilityClusterBias, 0.0, 1.0)
            let hour = hourOf(timestampUTC: timestampUTC)
            let sessionSigmaScaleValue = sessionSigmaScale(spec: spec, hour: hour)
            let sessionFillRiskScaleValue = sessionFillRiskScale(spec: spec, hour: hour)

            var sigma = spec.sigmaPerBar
            sigma *= sessionSigmaScaleValue
            if spec.volatilityCluster > 0.0 {
                sigma = max(
                    1e-6,
                    spec.volatilityCluster * previousSigma +
                        (1.0 - spec.volatilityCluster) * spec.sigmaPerBar * (0.5 + 1.5 * rng.nextUnit())
                )
                previousSigma = sigma
            }
            sigma *= fxClamp(spec.worldSigmaScale, 0.50, 3.00) *
                (1.0 + 0.28 * edgeFocus * sessionEdge +
                 0.14 * volatilityClusterBias * previousShockStrength +
                 0.08 * transitionEntropy)

            var drift = spec.driftPerBar + spec.worldDriftBias
            if spec.id == 7, offset > bars / 2 {
                drift = -spec.driftPerBar * 1.25
                sigma *= 1.8
            }

            var returnValue = drift + sigma * rng.nextNormal()
            let persistenceBias = (persistence - 0.50) * (0.35 - 0.18 * meanRevertBias) * abs(previousReturn)
            if previousReturn >= 0.0 {
                returnValue += persistenceBias
            } else {
                returnValue -= persistenceBias
            }
            if previousShockStrength > 0.0 {
                let shockTerm = previousShockStrength * (0.22 * shockMemory - 0.16 * recoveryBias)
                if previousReturn >= 0.0 {
                    returnValue += shockTerm
                } else {
                    returnValue -= shockTerm
                }
            }
            if spec.meanRevertStrength > 0.0 {
                returnValue += spec.meanRevertStrength * ((anchor - previous) / max(previous, point))
            }
            if spec.spikeProbability > 0.0, rng.nextUnit() < spec.spikeProbability {
                returnValue += spec.spikeScale * sigma * rng.nextNormal()
            }
            let liveGapProbability = fxClamp(spec.worldGapProbability + 0.10 * transitionBurst, 0.0, 0.40)
            if spec.worldGapProbability > 0.0, rng.nextUnit() < liveGapProbability {
                returnValue += spec.worldGapScale * sigma * (rng.nextUnit() < 0.5 ? -1.0 : 1.0)
            }
            let liveFlipProbability = fxClamp(
                spec.worldFlipProbability +
                    0.14 * meanRevertBias +
                    0.10 * transitionBurst +
                    0.08 * transitionEntropy,
                0.0,
                0.65
            )
            if spec.worldFlipProbability > 0.0, rng.nextUnit() < liveFlipProbability {
                returnValue *= -1.0
            }

            if spec.id == 5 {
                returnValue = abs(drift) + 0.10 * sigma * abs(rng.nextNormal())
            } else if spec.id == 6 {
                returnValue = -(abs(drift) + 0.10 * sigma * abs(rng.nextNormal()))
            }

            let open = previous
            var close = previous * (1.0 + returnValue)
            if close <= point {
                close = previous + 10.0 * point
            }

            var wick = max(
                abs(close - open) * (0.30 + 0.40 * rng.nextUnit()),
                point * (2.0 + 8.0 * rng.nextUnit())
            )
            wick *= 1.0 + 0.18 * edgeFocus * sessionEdge + 0.10 * previousShockStrength
            let high = max(open, close) + wick
            let low = max(point, min(open, close) - wick)
            let fillRisk = syntheticFillRisk(
                spec: spec,
                sessionScale: sessionFillRiskScaleValue,
                sessionEdge: sessionEdge,
                edgeFocus: edgeFocus,
                volatilityRatio: abs(returnValue) / max(sigma, 1e-6),
                rng: &rng,
                multiplier: 1.0,
                shockEligible: true
            )
            let volume = syntheticVolume(
                returnValue: returnValue,
                sigma: sigma,
                sessionEdge: sessionEdge,
                liquidityStress: spec.worldLiquidityStress,
                edgeFocus: edgeFocus,
                shockStrength: previousShockStrength
            )

            chronologicalBars.append(normalizeSyntheticBar(
                AuditScenarioDoubleBar(
                    timestampUTC: timestampUTC,
                    open: open,
                    high: high,
                    low: low,
                    close: close,
                    volume: volume,
                    fillRiskPoints: fillRisk
                ),
                point: pt
            ))

            let contextNoise1 = 0.35 * sigma * rng.nextNormal()
            let contextNoise2 = 0.45 * sigma * rng.nextNormal()
            let contextNoise3 = 0.55 * sigma * rng.nextNormal()
            let correlationBias = fxClamp(spec.worldContextCorrelationBias, -1.0, 1.0)
            let contextReturn1 = (0.80 + 0.15 * correlationBias) * returnValue + contextNoise1
            let contextReturn2 = (-0.45 + 0.20 * correlationBias) * returnValue + contextNoise2
            let contextReturn3 = (0.35 + 0.25 * correlationBias) * returnValue + 0.50 * contextNoise3

            let contextBar1 = syntheticContextBar(
                timestampUTC: timestampUTC,
                open: contextPrevious1,
                returnValue: contextReturn1,
                fallbackPrice: open,
                point: point,
                sigma: sigma,
                wickBase: 1.5,
                wickRange: 5.0,
                fillRiskMultiplier: 0.95 + 0.18 * fxClamp(abs(contextReturn1) / max(sigma, 1e-6), 0.0, 2.0),
                volumeBase: volume,
                spec: spec,
                sessionFillRiskScale: sessionFillRiskScaleValue,
                sessionEdge: sessionEdge,
                edgeFocus: edgeFocus,
                rng: &rng
            )
            let contextBar2 = syntheticContextBar(
                timestampUTC: timestampUTC,
                open: contextPrevious2,
                returnValue: contextReturn2,
                fallbackPrice: open,
                point: point,
                sigma: sigma,
                wickBase: 1.5,
                wickRange: 5.5,
                fillRiskMultiplier: 0.88 + 0.12 * fxClamp(abs(contextReturn2) / max(sigma, 1e-6), 0.0, 2.0),
                volumeBase: volume,
                spec: spec,
                sessionFillRiskScale: sessionFillRiskScaleValue,
                sessionEdge: sessionEdge,
                edgeFocus: edgeFocus,
                rng: &rng
            )
            let contextBar3 = syntheticContextBar(
                timestampUTC: timestampUTC,
                open: contextPrevious3,
                returnValue: contextReturn3,
                fallbackPrice: open,
                point: point,
                sigma: sigma,
                wickBase: 1.5,
                wickRange: 6.0,
                fillRiskMultiplier: 1.02 + 0.16 * fxClamp(abs(contextReturn3) / max(sigma, 1e-6), 0.0, 2.0),
                volumeBase: volume,
                spec: spec,
                sessionFillRiskScale: sessionFillRiskScaleValue,
                sessionEdge: sessionEdge,
                edgeFocus: edgeFocus,
                rng: &rng
            )
            context1Bars.append(contextBar1)
            context2Bars.append(contextBar2)
            context3Bars.append(contextBar3)
            contextPrevious1 = contextBar1.close
            contextPrevious2 = contextBar2.close
            contextPrevious3 = contextBar3.close

            previousReturn = previous > point ? (close - previous) / previous : 0.0
            previousShockStrength = fxClamp(
                (abs(previousReturn) / max(sigma, 1e-6)) *
                    (1.0 - 0.25 * shockDecay) +
                    0.10 * transitionEntropy,
                0.0,
                3.0
            )
            previous = close
        }

        let primary = AuditContextSeriesTools.reverseChronologicalBarsToSeries(chronologicalBars)
        let contextSeries = [
            AuditContextSeriesTools.reverseChronologicalBarsToSeries(context1Bars),
            AuditContextSeriesTools.reverseChronologicalBarsToSeries(context2Bars),
            AuditContextSeriesTools.reverseChronologicalBarsToSeries(context3Bars)
        ]

        return buildGeneratedScenarioSeries(
            primary: primary,
            chronologicalBars: chronologicalBars,
            point: point,
            contexts: contextSeries
        )
    }

    private static func selectMarketScenarioStart(
        spec: AuditScenarioSpec,
        marketSeries: M1OHLCVSeries,
        bars: Int,
        point: Double
    ) -> Int? {
        let maxStart = marketSeries.count - bars
        guard maxStart >= 0 else { return nil }

        var bestStart = maxStart
        var bestScore = -Double.greatestFiniteMagnitude
        var start = maxStart
        while start >= 0 {
            let recencyOffset = maxStart - start
            let score = marketScenarioScore(
                spec: spec,
                marketSeries: marketSeries,
                start: start,
                bars: bars,
                point: point,
                recencyOffset: recencyOffset
            )
            if score > bestScore {
                bestScore = score
                bestStart = start
            }
            start -= 1
        }
        return bestStart
    }

    private static func marketScenarioScore(
        spec: AuditScenarioSpec,
        marketSeries: M1OHLCVSeries,
        start: Int,
        bars: Int,
        point: Double,
        recencyOffset: Int
    ) -> Double {
        let pt = max(point, 1e-6)
        let lastIndex = start + bars - 1
        let net = abs(marketSeries.price(marketSeries.close[lastIndex]) - marketSeries.price(marketSeries.close[start]))
        var absoluteMoveSum = 0.0
        var rangeSum = 0.0
        for index in start..<lastIndex {
            let closeA = marketSeries.price(marketSeries.close[index])
            let closeB = marketSeries.price(marketSeries.close[index + 1])
            absoluteMoveSum += abs(closeB - closeA)
            rangeSum += abs(marketSeries.price(marketSeries.high[index]) - marketSeries.price(marketSeries.low[index]))
        }
        absoluteMoveSum = max(absoluteMoveSum, pt)
        let trendiness = net / absoluteMoveSum
        let averageRangePoints = (rangeSum / max(Double(bars - 1), 1.0)) / pt
        let volumeShock = marketVolumeShockScore(marketSeries: marketSeries, start: start, bars: bars)

        switch spec.id {
        case 8:
            return -0.001 * Double(recencyOffset)
        case 9:
            return trendiness + 0.05 * averageRangePoints
        case 10:
            return (1.0 - trendiness) + 0.03 * averageRangePoints
        case 11:
            let midpointTime = marketSeries.utcTimestamps[start + bars / 2]
            let hour = Double(hourOf(timestampUTC: midpointTime))
            let sessionEdge = 1.0 - min(abs(hour - 8.0), abs(hour - 16.0)) / 8.0
            return 0.60 * sessionEdge + 0.25 * averageRangePoints + 0.15 * (1.0 - trendiness)
        case 12:
            return volumeShock + 0.04 * averageRangePoints
        case 14:
            let midpointTime = marketSeries.utcTimestamps[start + bars / 2]
            let macroProxy = fxClamp(
                0.50 * sessionEdgeStrength(timestampUTC: midpointTime) +
                    0.30 * volumeShock +
                    0.20 * fxClamp(averageRangePoints / 8.0, 0.0, 1.0),
                0.0,
                1.0
            )
            return 0.68 * macroProxy +
                0.22 * volumeShock +
                0.06 * trendiness +
                0.04 * fxClamp(averageRangePoints, 0.0, 8.0)
        default:
            let recentA = marketSeries.price(marketSeries.close[lastIndex])
            let recentB = marketSeries.price(marketSeries.close[max(start, lastIndex - bars / 3)])
            let olderA = marketSeries.price(marketSeries.close[max(start, lastIndex - (2 * bars) / 3)])
            let olderB = marketSeries.price(marketSeries.close[start])
            let recentReturn = recentB > 0.0 ? (recentA - recentB) / recentB : 0.0
            let olderReturn = olderB > 0.0 ? (olderA - olderB) / olderB : 0.0
            return (1.0 - abs(recentReturn - olderReturn)) +
                0.20 * (1.0 - trendiness) +
                0.03 * averageRangePoints
        }
    }

    private static func marketVolumeShockScore(
        marketSeries: M1OHLCVSeries,
        start: Int,
        bars: Int
    ) -> Double {
        var positiveCount = 0
        var sum = 0.0
        var maxVolume = 0.0
        var minVolume = Double.greatestFiniteMagnitude
        for index in start..<(start + bars) {
            let volume = Double(marketSeries.volume[index])
            guard volume > 0.0 else { continue }
            positiveCount += 1
            sum += volume
            maxVolume = max(maxVolume, volume)
            minVolume = min(minVolume, volume)
        }
        guard positiveCount > 0 else { return 0.0 }
        let average = max(sum / Double(positiveCount), 1.0)
        let highShock = (maxVolume - average) / average
        let lowShock = (average - minVolume) / average
        return fxClamp(max(highShock, lowShock), 0.0, 12.0)
    }

    public static func marketScenarioBars(
        from marketSeries: M1OHLCVSeries,
        range: Range<Int>,
        point: Double
    ) -> [AuditScenarioDoubleBar] {
        guard range.lowerBound >= 0, range.upperBound <= marketSeries.count else { return [] }
        var averagePositiveVolume = 0.0
        var positiveVolumeCount = 0
        for index in range {
            let volume = Double(marketSeries.volume[index])
            if volume > 0.0 {
                averagePositiveVolume += volume
                positiveVolumeCount += 1
            }
        }
        if positiveVolumeCount > 0 {
            averagePositiveVolume /= Double(positiveVolumeCount)
        }

        var output: [AuditScenarioDoubleBar] = []
        output.reserveCapacity(range.count)
        for index in range {
            let open = marketSeries.price(marketSeries.open[index])
            let high = marketSeries.price(marketSeries.high[index])
            let low = marketSeries.price(marketSeries.low[index])
            let close = marketSeries.price(marketSeries.close[index])
            let volume = Double(marketSeries.volume[index])
            output.append(normalizeSyntheticBar(
                AuditScenarioDoubleBar(
                    timestampUTC: marketSeries.utcTimestamps[index],
                    open: open,
                    high: high,
                    low: low,
                    close: close,
                    volume: volume,
                    fillRiskPoints: marketFillRiskPoints(
                        high: high,
                        low: low,
                        volume: volume,
                        averagePositiveVolume: averagePositiveVolume,
                        point: point
                    )
                ),
                point: point
            ))
        }
        return output
    }

    private static func marketFillRiskPoints(
        high: Double,
        low: Double,
        volume: Double,
        averagePositiveVolume: Double,
        point: Double
    ) -> Double {
        let pt = max(point, 1e-6)
        let rangePoints = max(0.0, (high - low) / pt)
        var liquidityStress = 0.0
        if averagePositiveVolume > 0.0, volume > 0.0 {
            let ratio = volume / averagePositiveVolume
            liquidityStress = ratio < 1.0
                ? 2.0 * (1.0 - ratio)
                : 0.25 * fxClamp(ratio - 1.0, 0.0, 4.0)
        } else {
            liquidityStress = 0.50
        }
        return max(0.25, fxClamp(0.35 + 0.04 * rangePoints + liquidityStress, 0.25, 12.0))
    }

    private static func asSeriesOHLCV(fromAsSeriesBars bars: [AuditScenarioDoubleBar]) -> AuditAsSeriesOHLCV {
        AuditAsSeriesOHLCV(
            timeUTC: bars.map(\.timestampUTC),
            open: bars.map(\.open),
            high: bars.map(\.high),
            low: bars.map(\.low),
            close: bars.map(\.close),
            volume: bars.map(\.volume),
            fillRiskPoints: bars.map(\.fillRiskPoints)
        )
    }

    private static func buildGeneratedScenarioSeries(
        primary: AuditAsSeriesOHLCV,
        chronologicalBars: [AuditScenarioDoubleBar],
        point: Double,
        contexts: [AuditAsSeriesOHLCV]
    ) -> AuditGeneratedScenarioSeries {
        let m5Aggregate = AuditContextSeriesTools.aggregateCloseTimeframe(chronologicalBars: chronologicalBars, step: 5)
        let m15Aggregate = AuditContextSeriesTools.aggregateCloseTimeframe(chronologicalBars: chronologicalBars, step: 15)
        let m30Aggregate = AuditContextSeriesTools.aggregateCloseTimeframe(chronologicalBars: chronologicalBars, step: 30)
        let h1Aggregate = AuditContextSeriesTools.aggregateCloseTimeframe(chronologicalBars: chronologicalBars, step: 60)
        let contextFeatures = AuditContextSeriesTools.buildContextFeatures(
            mainClose: primary.close,
            point: point,
            contexts: contexts
        )

        return AuditGeneratedScenarioSeries(
            primary: primary,
            m5: AuditCloseTimeframeSeries(
                timeUTC: m5Aggregate.timeUTC,
                close: m5Aggregate.close,
                alignedIndexMap: buildAlignedIndexMapAsSeries(
                    referenceTimeUTC: primary.timeUTC,
                    targetTimeUTC: m5Aggregate.timeUTC,
                    maxLagSeconds: 2 * 5 * 60
                )
            ),
            m15: AuditCloseTimeframeSeries(
                timeUTC: m15Aggregate.timeUTC,
                close: m15Aggregate.close,
                alignedIndexMap: buildAlignedIndexMapAsSeries(
                    referenceTimeUTC: primary.timeUTC,
                    targetTimeUTC: m15Aggregate.timeUTC,
                    maxLagSeconds: 2 * 15 * 60
                )
            ),
            m30: AuditCloseTimeframeSeries(
                timeUTC: m30Aggregate.timeUTC,
                close: m30Aggregate.close,
                alignedIndexMap: buildAlignedIndexMapAsSeries(
                    referenceTimeUTC: primary.timeUTC,
                    targetTimeUTC: m30Aggregate.timeUTC,
                    maxLagSeconds: 2 * 30 * 60
                )
            ),
            h1: AuditCloseTimeframeSeries(
                timeUTC: h1Aggregate.timeUTC,
                close: h1Aggregate.close,
                alignedIndexMap: buildAlignedIndexMapAsSeries(
                    referenceTimeUTC: primary.timeUTC,
                    targetTimeUTC: h1Aggregate.timeUTC,
                    maxLagSeconds: 2 * 60 * 60
                )
            ),
            contexts: contexts,
            contextFeatures: contextFeatures
        )
    }

    private static func syntheticFillRisk(
        spec: AuditScenarioSpec,
        sessionScale: Double,
        sessionEdge: Double,
        edgeFocus: Double,
        volatilityRatio: Double,
        rng: inout AuditScenarioRandomGenerator,
        multiplier: Double,
        shockEligible: Bool
    ) -> Double {
        var scale = fxClamp(spec.worldFillRiskScale, 0.50, 4.00) *
            sessionScale *
            multiplier *
            (0.85 + 0.30 * rng.nextUnit() +
             0.18 * fxClamp(spec.worldLiquidityStress, 0.0, 3.0) +
             0.16 * edgeFocus * sessionEdge +
             0.06 * fxClamp(volatilityRatio, 0.0, 2.0))
        if shockEligible,
           rng.nextUnit() < fxClamp(spec.worldLiquidityShockProbability, 0.0, 0.50) {
            scale *= fxClamp(spec.worldLiquidityShockScale, 1.0, 8.0)
        }
        return max(0.25, spec.fillRiskPoints * scale)
    }

    private static func syntheticVolume(
        returnValue: Double,
        sigma: Double,
        sessionEdge: Double,
        liquidityStress: Double,
        edgeFocus: Double,
        shockStrength: Double
    ) -> Double {
        let movement = fxClamp(abs(returnValue) / max(sigma, 1e-6), 0.0, 6.0)
        let stress = fxClamp(liquidityStress, 0.0, 3.0)
        let scale = fxClamp(
            1.0 + 0.18 * movement + 0.10 * edgeFocus * sessionEdge + 0.08 * shockStrength - 0.14 * stress,
            0.15,
            4.0
        )
        return max(1.0, 100.0 * scale)
    }

    private static func syntheticContextBar(
        timestampUTC: Int64,
        open: Double,
        returnValue: Double,
        fallbackPrice: Double,
        point: Double,
        sigma: Double,
        wickBase: Double,
        wickRange: Double,
        fillRiskMultiplier: Double,
        volumeBase: Double,
        spec: AuditScenarioSpec,
        sessionFillRiskScale: Double,
        sessionEdge: Double,
        edgeFocus: Double,
        rng: inout AuditScenarioRandomGenerator
    ) -> AuditScenarioDoubleBar {
        var close = open * (1.0 + returnValue)
        if close <= point {
            close = max(open, fallbackPrice)
        }
        let wick = max(
            abs(close - open) * (0.25 + 0.35 * rng.nextUnit()),
            point * (wickBase + wickRange * rng.nextUnit())
        )
        let fillRisk = syntheticFillRisk(
            spec: spec,
            sessionScale: sessionFillRiskScale,
            sessionEdge: sessionEdge,
            edgeFocus: edgeFocus,
            volatilityRatio: abs(returnValue) / max(sigma, 1e-6),
            rng: &rng,
            multiplier: fillRiskMultiplier,
            shockEligible: false
        )
        let volumeScale = fxClamp(
            0.80 + 0.20 * rng.nextUnit() - 0.08 * fxClamp(spec.worldLiquidityStress, 0.0, 3.0),
            0.15,
            2.0
        )
        return normalizeSyntheticBar(
            AuditScenarioDoubleBar(
                timestampUTC: timestampUTC,
                open: open,
                high: max(open, close) + wick,
                low: max(point, min(open, close) - wick),
                close: close,
                volume: max(1.0, volumeBase * volumeScale),
                fillRiskPoints: fillRisk
            ),
            point: point
        )
    }

    private static func buildAlignedIndexMapAsSeries(
        referenceTimeUTC: [Int64],
        targetTimeUTC: [Int64],
        maxLagSeconds: Int
    ) -> [Int] {
        var output = Array(repeating: -1, count: referenceTimeUTC.count)
        guard !referenceTimeUTC.isEmpty, !targetTimeUTC.isEmpty else { return output }
        var targetIndex = 0

        for referenceIndex in referenceTimeUTC.indices {
            let referenceTime = referenceTimeUTC[referenceIndex]
            guard referenceTime > 0 else { continue }
            while targetIndex < targetTimeUTC.count, targetTimeUTC[targetIndex] > referenceTime {
                targetIndex += 1
            }
            if targetIndex >= targetTimeUTC.count {
                break
            }

            let lag = referenceTime - targetTimeUTC[targetIndex]
            guard lag >= 0 else { continue }
            if maxLagSeconds > 0, lag > Int64(maxLagSeconds) {
                continue
            }
            output[referenceIndex] = targetIndex
        }

        return output
    }

    private static func clampWorldPlan(_ spec: inout AuditScenarioSpec) {
        spec.worldSigmaScale = fxClamp(spec.worldSigmaScale, 0.50, 3.00)
        spec.worldDriftBias = fxClamp(spec.worldDriftBias, -3.0 * spec.sigmaPerBar, 3.0 * spec.sigmaPerBar)
        spec.worldFillRiskScale = fxClamp(spec.worldFillRiskScale, 0.50, 4.00)
        spec.worldGapProbability = fxClamp(spec.worldGapProbability, 0.0, 0.30)
        spec.worldGapScale = fxClamp(spec.worldGapScale, 0.0, 8.0)
        spec.worldFlipProbability = fxClamp(spec.worldFlipProbability, 0.0, 0.50)
        spec.worldContextCorrelationBias = fxClamp(spec.worldContextCorrelationBias, -1.0, 1.0)
        spec.worldLiquidityStress = fxClamp(spec.worldLiquidityStress, 0.0, 3.0)
        spec.worldSessionEdgeFocus = fxClamp(spec.worldSessionEdgeFocus, 0.0, 1.5)
        spec.worldTrendPersistence = fxClamp(spec.worldTrendPersistence, 0.0, 1.0)
        spec.worldShockMemory = fxClamp(spec.worldShockMemory, 0.0, 1.0)
        spec.worldRecoveryBias = fxClamp(spec.worldRecoveryBias, -1.0, 1.0)
        spec.worldLiquidityShockProbability = fxClamp(spec.worldLiquidityShockProbability, 0.0, 0.50)
        spec.worldLiquidityShockScale = fxClamp(spec.worldLiquidityShockScale, 1.0, 8.0)
        spec.worldRegimeTransitionBurst = fxClamp(spec.worldRegimeTransitionBurst, 0.0, 1.0)
        spec.worldTransitionEntropy = fxClamp(spec.worldTransitionEntropy, 0.0, 1.0)
        spec.worldMeanRevertBias = fxClamp(spec.worldMeanRevertBias, 0.0, 1.0)
        spec.worldVolatilityClusterBias = fxClamp(spec.worldVolatilityClusterBias, 0.0, 1.0)
        spec.worldShockDecay = fxClamp(spec.worldShockDecay, 0.0, 1.5)
        spec.worldAsiaSigmaScale = fxClamp(spec.worldAsiaSigmaScale, 0.50, 3.00)
        spec.worldLondonSigmaScale = fxClamp(spec.worldLondonSigmaScale, 0.50, 3.00)
        spec.worldNewYorkSigmaScale = fxClamp(spec.worldNewYorkSigmaScale, 0.50, 3.00)
        spec.worldAsiaFillRiskScale = fxClamp(spec.worldAsiaFillRiskScale, 0.50, 4.00)
        spec.worldLondonFillRiskScale = fxClamp(spec.worldLondonFillRiskScale, 0.50, 4.00)
        spec.worldNewYorkFillRiskScale = fxClamp(spec.worldNewYorkFillRiskScale, 0.50, 4.00)
        spec.macroFocus = fxClamp(spec.macroFocus, 0.0, 1.5)
    }

    private static func mqlStringToDouble(_ raw: String) -> Double {
        let scanner = Scanner(string: raw)
        scanner.charactersToBeSkipped = nil
        return fxSafeFinite(scanner.scanDouble() ?? 0.0)
    }
}
