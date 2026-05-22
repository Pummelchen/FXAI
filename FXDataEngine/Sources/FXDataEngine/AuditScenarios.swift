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
            spec.name = "market_spread_shock"
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
