import Foundation

public enum AdaptiveRouterRuntimeConstants {
    public static let maxReasons = 6
    public static let runtimeDirectory = "FXAI/Runtime"
}

public enum AdaptiveRouterRuntimeStatus: Int, Codable, Sendable, CaseIterable {
    case suppressed = 0
    case downweighted
    case active
    case upweighted

    public var label: String {
        switch self {
        case .suppressed: "SUPPRESSED"
        case .downweighted: "DOWNWEIGHTED"
        case .active: "ACTIVE"
        case .upweighted: "UPWEIGHTED"
        }
    }
}

public struct AdaptiveRegimeState: Codable, Hashable, Sendable {
    public var valid: Bool
    public var symbol: String
    public var generatedAt: Int64
    public var topLabel: String
    public var confidence: Double
    public var probabilities: [Double]
    public var sessionLabel: String
    public var priceCostRegime: String
    public var volatilityRegime: String
    public var newsRiskScore: Double
    public var newsPressure: Double
    public var eventETAMinutes: Int
    public var staleNews: Bool
    public var liquidityStress: Double
    public var breakoutPressure: Double
    public var trendStrength: Double
    public var rangePressure: Double
    public var macroPressure: Double
    public var reasons: [String]

    public init(
        valid: Bool = false,
        symbol: String = "",
        generatedAt: Int64 = 0,
        topLabel: String = "TREND_PERSISTENT",
        confidence: Double = 0.0,
        probabilities: [Double] = Array(repeating: 0.0, count: ControlPlaneConstants.adaptiveRouterRegimeLabels.count),
        sessionLabel: String = "ASIA",
        priceCostRegime: String = "NORMAL",
        volatilityRegime: String = "NORMAL",
        newsRiskScore: Double = 0.0,
        newsPressure: Double = 0.0,
        eventETAMinutes: Int = -1,
        staleNews: Bool = true,
        liquidityStress: Double = 0.0,
        breakoutPressure: Double = 0.0,
        trendStrength: Double = 0.0,
        rangePressure: Double = 0.0,
        macroPressure: Double = 0.0,
        reasons: [String] = []
    ) {
        self.valid = valid
        self.symbol = symbol.uppercased()
        self.generatedAt = max(0, generatedAt)
        self.topLabel = topLabel.isEmpty ? "TREND_PERSISTENT" : topLabel
        self.confidence = fxClamp(confidence, 0.0, 1.0)
        self.probabilities = Self.normalizedProbabilities(probabilities)
        self.sessionLabel = sessionLabel.isEmpty ? "ASIA" : sessionLabel
        self.priceCostRegime = priceCostRegime.isEmpty ? "NORMAL" : priceCostRegime
        self.volatilityRegime = volatilityRegime.isEmpty ? "NORMAL" : volatilityRegime
        self.newsRiskScore = fxClamp(newsRiskScore, 0.0, 1.0)
        self.newsPressure = fxClamp(newsPressure, -1.0, 1.0)
        self.eventETAMinutes = eventETAMinutes
        self.staleNews = staleNews
        self.liquidityStress = fxClamp(liquidityStress, 0.0, 1.0)
        self.breakoutPressure = fxClamp(breakoutPressure, 0.0, 1.0)
        self.trendStrength = fxClamp(trendStrength, 0.0, 1.0)
        self.rangePressure = fxClamp(rangePressure, 0.0, 1.0)
        self.macroPressure = fxClamp(macroPressure, 0.0, 1.0)
        self.reasons = Self.uniqueReasons(reasons)
    }

    public static var reset: AdaptiveRegimeState {
        AdaptiveRegimeState()
    }

    public var reasonsCSV: String {
        reasons.filter { !$0.isEmpty }.joined(separator: "; ")
    }

    public mutating func appendReason(_ reason: String) {
        let value = reason.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !value.isEmpty,
              !reasons.contains(value),
              reasons.count < AdaptiveRouterRuntimeConstants.maxReasons else {
            return
        }
        reasons.append(value)
    }

    private static func normalizedProbabilities(_ raw: [Double]) -> [Double] {
        let count = ControlPlaneConstants.adaptiveRouterRegimeLabels.count
        var output = Array(raw.prefix(count)).map { max(0.0, fxSafeFinite($0)) }
        if output.count < count {
            output.append(contentsOf: Array(repeating: 0.0, count: count - output.count))
        }
        return output
    }

    private static func uniqueReasons(_ input: [String]) -> [String] {
        var output: [String] = []
        output.reserveCapacity(min(input.count, AdaptiveRouterRuntimeConstants.maxReasons))
        for raw in input {
            let value = raw.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !value.isEmpty,
                  !output.contains(value),
                  output.count < AdaptiveRouterRuntimeConstants.maxReasons else {
                continue
            }
            output.append(value)
        }
        return output
    }
}

public struct AdaptiveRouterRegimeInputs: Sendable {
    public var symbol: String
    public var series: M1OHLCVSeries
    public var index: Int
    public var pointValue: Double
    public var priceCostPoints: Double
    public var priceCostReferencePoints: Double?
    public var volatilityProxyAbs: Double
    public var volatilityReference: Double?
    public var minMovePoints: Double
    public var contextStrength: Double
    public var contextQuality: Double
    public var regimeGraph: RegimeGraphQuery
    public var newsState: NewsPulsePairState
    public var crossAssetState: CrossAssetPairState
    public var microstructureState: MicrostructurePairState

    public init(
        symbol: String,
        series: M1OHLCVSeries,
        index: Int,
        pointValue: Double = 1.0,
        priceCostPoints: Double = 0.0,
        priceCostReferencePoints: Double? = nil,
        volatilityProxyAbs: Double = 0.0,
        volatilityReference: Double? = nil,
        minMovePoints: Double = 0.0,
        contextStrength: Double = 0.0,
        contextQuality: Double = 0.0,
        regimeGraph: RegimeGraphQuery = RegimeGraphQuery(),
        newsState: NewsPulsePairState = .reset,
        crossAssetState: CrossAssetPairState = .reset,
        microstructureState: MicrostructurePairState = .reset
    ) {
        self.symbol = symbol.uppercased()
        self.series = series
        self.index = index
        self.pointValue = pointValue > 0.0 ? pointValue : 1.0
        self.priceCostPoints = max(0.0, fxSafeFinite(priceCostPoints))
        self.priceCostReferencePoints = priceCostReferencePoints.map { max(0.0, fxSafeFinite($0)) }
        self.volatilityProxyAbs = max(0.0, fxSafeFinite(volatilityProxyAbs))
        self.volatilityReference = volatilityReference.map { max(0.0, fxSafeFinite($0)) }
        self.minMovePoints = max(0.0, fxSafeFinite(minMovePoints))
        self.contextStrength = fxSafeFinite(contextStrength)
        self.contextQuality = fxClamp(contextQuality, 0.0, 1.0)
        self.regimeGraph = regimeGraph
        self.newsState = newsState
        self.crossAssetState = crossAssetState
        self.microstructureState = microstructureState
    }
}

public struct AdaptiveRouterPluginRoute: Codable, Hashable, Sendable {
    public var name: String
    public var eligible: Bool
    public var routedWeight: Double
    public var suitability: Double
    public var status: AdaptiveRouterRuntimeStatus
    public var reasons: [String]

    public init(
        name: String,
        eligible: Bool = false,
        routedWeight: Double = 0.0,
        suitability: Double = 0.0,
        status: AdaptiveRouterRuntimeStatus = .active,
        reasons: [String] = []
    ) {
        self.name = name
        self.eligible = eligible
        self.routedWeight = max(0.0, fxSafeFinite(routedWeight))
        self.suitability = max(0.0, fxSafeFinite(suitability))
        self.status = status
        self.reasons = reasons
    }
}

public enum AdaptiveRouterRuntimeTools {
    public static func runtimeStatePath(symbol: String) -> String {
        "\(AdaptiveRouterRuntimeConstants.runtimeDirectory)/fxai_regime_router_\(ControlPlanePaths.safeToken(symbol)).tsv"
    }

    public static func runtimeHistoryPath(symbol: String) -> String {
        "\(AdaptiveRouterRuntimeConstants.runtimeDirectory)/fxai_regime_router_history_\(ControlPlanePaths.safeToken(symbol)).ndjson"
    }

    public static func regimeLabel(_ index: Int) -> String {
        guard index >= 0, index < ControlPlaneConstants.adaptiveRouterRegimeLabels.count else {
            return "TREND_PERSISTENT"
        }
        return ControlPlaneConstants.adaptiveRouterRegimeLabels[index]
    }

    public static func sessionLabel(sampleTimeUTC: Int64) -> String {
        let timestamp = max(sampleTimeUTC, 0)
        var calendar = Calendar(identifier: .gregorian)
        calendar.timeZone = TimeZone(secondsFromGMT: 0) ?? .gmt
        let hour = calendar.component(.hour, from: Date(timeIntervalSince1970: TimeInterval(timestamp)))
        if hour >= 21 || hour < 1 {
            return "ROLLOVER"
        }
        if hour < 7 {
            return "ASIA"
        }
        if hour < 12 {
            return "LONDON"
        }
        if hour < 16 {
            return "LONDON_NY_OVERLAP"
        }
        return "NEWYORK"
    }

    public static func macroPairSensitivity(symbol: String) -> Double {
        let upper = symbol.uppercased()
        var score = 0.10
        if upper.contains("JPY") || upper.contains("CHF") {
            score += 0.32
        }
        if upper.contains("USD") {
            score += 0.22
        }
        if upper.contains("AUD") || upper.contains("NZD") || upper.contains("CAD") {
            score += 0.18
        }
        if upper.contains("EUR") || upper.contains("GBP") {
            score += 0.16
        }
        return fxClamp(score, 0.10, 1.0)
    }

    public static func buildRegimeState(inputs: AdaptiveRouterRegimeInputs) -> AdaptiveRegimeState {
        guard inputs.index >= 0, inputs.index < inputs.series.count else {
            return .reset
        }

        var state = AdaptiveRegimeState.reset
        state.valid = true
        state.symbol = inputs.symbol
        state.generatedAt = inputs.series.utcTimestamps[inputs.index]
        state.sessionLabel = sessionLabel(sampleTimeUTC: state.generatedAt)
        state.newsRiskScore = fxClamp(inputs.newsState.ready ? inputs.newsState.newsRiskScore : 0.0, 0.0, 1.0)
        state.newsPressure = fxClamp(inputs.newsState.ready ? inputs.newsState.newsPressure : 0.0, -1.0, 1.0)
        state.eventETAMinutes = inputs.newsState.ready ? inputs.newsState.eventETAMinutes : -1
        state.staleNews = !inputs.newsState.ready || inputs.newsState.stale

        let crossReady = inputs.crossAssetState.ready && !inputs.crossAssetState.stale
        let crossPairRisk = crossReady ? fxClamp(inputs.crossAssetState.pairCrossAssetRiskScore, 0.0, 1.0) : 0.0
        let crossRiskOff = crossReady ? fxClamp(inputs.crossAssetState.riskOffScore, 0.0, 1.0) : 0.0
        let crossLiquidity = crossReady
            ? fxClamp(max(inputs.crossAssetState.usdLiquidityStressScore, inputs.crossAssetState.crossAssetDislocationScore), 0.0, 1.0)
            : 0.0

        let microReady = inputs.microstructureState.ready && !inputs.microstructureState.stale
        let microLiquidity = microReady ? fxClamp(inputs.microstructureState.liquidityStressScore, 0.0, 1.0) : 0.0
        let microHostile = microReady ? fxClamp(inputs.microstructureState.hostileExecutionScore, 0.0, 1.0) : 0.0
        let microBreakout = microReady ? fxClamp(inputs.microstructureState.localExtremaBreachScore60s, 0.0, 1.0) : 0.0
        let microTickPressure = microReady ? fxClamp(abs(inputs.microstructureState.tickImbalance30s), 0.0, 1.0) : 0.0
        let microDirectionalEfficiency = microReady ? fxClamp(inputs.microstructureState.directionalEfficiency60s, 0.0, 1.0) : 0.0
        let microSweepRisk = microReady
            ? fxClamp(max(inputs.microstructureState.breakoutReversalScore60s, inputs.microstructureState.exhaustionProxy60s), 0.0, 1.0)
            : 0.0
        let microHandoff = microReady && inputs.microstructureState.handoffFlag
        let microSessionBurst = microReady
            ? fxClamp(max(inputs.microstructureState.sessionOpenBurstScore, inputs.microstructureState.sessionSpreadBehaviorScore), 0.0, 1.0)
            : 0.0

        let priceCostReference = max(inputs.priceCostReferencePoints ?? max(inputs.priceCostPoints, 0.10), 0.10)
        let volatilityReference = max(inputs.volatilityReference ?? max(abs(inputs.volatilityProxyAbs), 1e-6), 1e-6)
        var priceCostRatio = fxClamp(inputs.priceCostPoints / priceCostReference, 0.25, 4.0)
        var volatilityRatio = fxClamp(abs(inputs.volatilityProxyAbs) / volatilityReference, 0.25, 4.0)
        if microReady {
            let microCostRatio = fxClamp(1.0 + max(inputs.microstructureState.spreadZscore60s, 0.0) / 2.0, 0.60, 4.0)
            let microVolatilityRatio = fxClamp(1.0 + max(inputs.microstructureState.volBurstScore5m - 1.0, 0.0), 0.60, 4.0)
            priceCostRatio = fxClamp(0.72 * priceCostRatio + 0.28 * microCostRatio, 0.25, 4.0)
            volatilityRatio = fxClamp(0.70 * volatilityRatio + 0.30 * microVolatilityRatio, 0.25, 4.0)
        }
        state.priceCostRegime = priceCostRatio >= 1.45 ? "ELEVATED" : (priceCostRatio <= 0.85 ? "CALM" : "NORMAL")
        state.volatilityRegime = volatilityRatio >= 1.35 ? "HIGH" : (volatilityRatio <= 0.82 ? "LOW" : "NORMAL")

        let slopePoints = abs(priceDeltaPoints(series: inputs.series, index: inputs.index, lookback: 6, pointValue: inputs.pointValue))
        let longSlopePoints = abs(priceDeltaPoints(series: inputs.series, index: inputs.index, lookback: 12, pointValue: inputs.pointValue))
        let moveFloor = max(inputs.minMovePoints, 0.50)
        let slopeNorm = fxClamp((0.55 * slopePoints + 0.45 * longSlopePoints) / max(moveFloor, 0.50), 0.0, 3.0) / 3.0
        let flipRatio = recentFlipRatio(series: inputs.series, index: inputs.index, count: 8)
        let rangePointsValue = rangePoints(series: inputs.series, index: inputs.index, count: 12, pointValue: inputs.pointValue)
        let rangeTightness = 1.0 - fxClamp(rangePointsValue / max(4.0 * moveFloor, 1.0), 0.0, 1.0)
        let reversalPressure = fxClamp(0.55 * flipRatio + 0.45 * rangeTightness, 0.0, 1.0)
        let breakoutPressure = fxClamp(
            0.34 * fxClamp(volatilityRatio - 1.0, 0.0, 1.5) +
                0.20 * fxClamp(abs(inputs.regimeGraph.edgeBias), 0.0, 1.0) +
                0.18 * fxClamp(inputs.regimeGraph.transitionConfidence, 0.0, 1.0) +
                0.14 * (1.0 - rangeTightness) +
                0.14 * microBreakout,
            0.0,
            1.0
        )
        let trendStrength = fxClamp(
            0.30 * fxClamp(inputs.regimeGraph.persistence, 0.0, 1.0) +
                0.26 * slopeNorm +
                0.14 * (1.0 - reversalPressure) +
                0.10 * fxClamp(inputs.contextStrength / 2.0, 0.0, 1.0) +
                0.10 * microTickPressure +
                0.10 * microDirectionalEfficiency,
            0.0,
            1.0
        )
        let rangePressure = fxClamp(
            0.31 * reversalPressure +
                0.24 * rangeTightness +
                0.18 * (1.0 - slopeNorm) +
                0.13 * (1.0 - breakoutPressure) +
                0.08 * (microReady && inputs.microstructureState.microstructureRegime == "CHOPPY_HIGH_ACTIVITY" ? 1.0 : 0.0) +
                0.06 * (1.0 - microDirectionalEfficiency),
            0.0,
            1.0
        )
        let pairMacroSensitivity = macroPairSensitivity(symbol: inputs.symbol)
        let macroPressure = fxClamp(
            0.36 * fxClamp(inputs.regimeGraph.macroAlignment, 0.0, 1.0) +
                0.24 * abs(state.newsPressure) +
                0.20 * pairMacroSensitivity +
                0.12 * fxClamp(inputs.contextStrength / 2.0, 0.0, 1.0) +
                0.08 * crossPairRisk,
            0.0,
            1.0
        )
        let liquidityStress = fxClamp(
            0.28 * fxClamp(priceCostRatio - 1.0, 0.0, 2.0) +
                0.14 * fxClamp(inputs.regimeGraph.instability, 0.0, 1.0) +
                0.12 * (state.sessionLabel == "ROLLOVER" ? 1.0 : 0.0) +
                0.10 * fxClamp(volatilityRatio - 1.0, 0.0, 2.0) +
                0.10 * (state.staleNews ? 1.0 : 0.0) +
                0.16 * microLiquidity +
                0.06 * microHostile +
                0.04 * crossLiquidity,
            0.0,
            1.0
        )
        state.breakoutPressure = breakoutPressure
        state.trendStrength = trendStrength
        state.rangePressure = rangePressure
        state.macroPressure = macroPressure
        state.liquidityStress = liquidityStress

        var sessionFlowScore = 0.12
        switch state.sessionLabel {
        case "ASIA":
            sessionFlowScore += 0.20 * rangePressure + 0.12 * (1.0 - state.newsRiskScore) + 0.08 * microSessionBurst
        case "LONDON":
            sessionFlowScore += 0.22 * breakoutPressure + 0.10 * trendStrength + 0.08 * microSessionBurst
        case "LONDON_NY_OVERLAP":
            sessionFlowScore += 0.18 * macroPressure + 0.16 * breakoutPressure + 0.08 * microSessionBurst
        case "NEWYORK":
            sessionFlowScore += 0.18 * trendStrength + 0.12 * macroPressure + 0.08 * microSessionBurst
        default:
            sessionFlowScore += 0.20 * liquidityStress + 0.08 * microSessionBurst
        }
        sessionFlowScore = fxClamp(sessionFlowScore, 0.0, 1.0)

        let eventWindow = inputs.newsState.ready &&
            !inputs.newsState.stale &&
            ((inputs.newsState.eventETAMinutes >= 0 && inputs.newsState.eventETAMinutes <= 45) ||
                inputs.newsState.tradeGate == "BLOCK" ||
                inputs.newsState.tradeGate == "CAUTION")

        var raw = Array(repeating: 0.0, count: ControlPlaneConstants.adaptiveRouterRegimeLabels.count)
        raw[0] = 0.14 + 0.46 * trendStrength + 0.16 * fxClamp(inputs.regimeGraph.persistence, 0.0, 1.0) + 0.10 * (1.0 - liquidityStress) + 0.14 * (1.0 - state.newsRiskScore)
        raw[1] = 0.12 + 0.44 * rangePressure + 0.12 * (1.0 - breakoutPressure) + 0.16 * (1.0 - state.newsRiskScore) + 0.16 * (1.0 - liquidityStress)
        raw[2] = 0.12 + 0.42 * breakoutPressure + 0.14 * trendStrength + 0.10 * fxClamp(inputs.regimeGraph.transitionConfidence, 0.0, 1.0) + 0.10 * fxClamp(volatilityRatio - 1.0, 0.0, 2.0) + 0.12 * microSweepRisk
        raw[3] = 0.08 + 0.52 * state.newsRiskScore + 0.18 * (eventWindow ? 1.0 : 0.0) + 0.10 * fxClamp(volatilityRatio - 1.0, 0.0, 2.0) + 0.12 * abs(state.newsPressure)
        raw[4] = 0.08 + 0.38 * macroPressure + 0.16 * abs(state.newsPressure) + 0.18 * pairMacroSensitivity + 0.10 * fxClamp(inputs.contextQuality, 0.0, 1.0) + 0.10 * crossRiskOff
        raw[5] = 0.08 + 0.42 * liquidityStress + 0.10 * fxClamp(inputs.regimeGraph.instability, 0.0, 1.0) + 0.08 * (state.sessionLabel == "ROLLOVER" ? 1.0 : 0.0) + 0.10 * (state.staleNews ? 1.0 : 0.0) + 0.12 * microHostile + 0.10 * crossLiquidity
        raw[6] = 0.12 + 0.48 * sessionFlowScore + 0.14 * (1.0 - state.newsRiskScore) + 0.08 * fxClamp(inputs.contextStrength / 2.0, 0.0, 1.0) + 0.08 * (1.0 - liquidityStress) + 0.10 * (microHandoff ? 1.0 : 0.0)

        var total = 0.0
        var topIndex = 0
        var secondIndex = 1
        for index in raw.indices {
            raw[index] = max(raw[index], 0.0001)
            total += raw[index]
            if raw[index] > raw[topIndex] {
                secondIndex = topIndex
                topIndex = index
            } else if index != topIndex && raw[index] > raw[secondIndex] {
                secondIndex = index
            }
        }
        if total <= 0.0 {
            total = 1.0
        }
        state.probabilities = raw.map { $0 / total }
        state.topLabel = regimeLabel(topIndex)
        state.confidence = fxClamp(
            0.60 * state.probabilities[topIndex] +
                0.40 * (state.probabilities[topIndex] - state.probabilities[secondIndex]),
            0.0,
            1.0
        )

        if eventWindow {
            state.appendReason("NewsPulse event window active")
        }
        if state.staleNews {
            state.appendReason("NewsPulse stale or unavailable")
        }
        if priceCostRatio >= 1.35 {
            state.appendReason("Spread regime elevated")
        }
        if volatilityRatio >= 1.25 {
            state.appendReason("Volatility expansion detected")
        }
        if trendStrength >= 0.62 {
            state.appendReason("Directional persistence elevated")
        }
        if rangePressure >= 0.62 {
            state.appendReason("Range reversion pressure elevated")
        }
        if macroPressure >= 0.58 {
            state.appendReason("Macro repricing pressure elevated")
        }
        if liquidityStress >= 0.58 {
            state.appendReason("Liquidity stress elevated")
        }
        if crossReady && inputs.crossAssetState.macroState != "NORMAL" {
            state.appendReason("Cross-asset macro regime active")
        }
        if crossReady && inputs.crossAssetState.tradeGate == "BLOCK" {
            state.appendReason("Cross-asset stress blocking")
        }
        if microReady && microHostile >= 0.58 {
            state.appendReason("Microstructure hostile execution elevated")
        }
        if microReady && microSweepRisk >= 0.58 {
            state.appendReason("Microstructure sweep rejection risk elevated")
        }
        if microHandoff && microSessionBurst >= 0.50 {
            state.appendReason("Session handoff burst active")
        }
        if sessionFlowScore >= 0.58 {
            state.appendReason("\(state.sessionLabel) session flow dominant")
        }
        return state
    }

    public static func pluginSuitability(
        profile: AdaptiveRouterProfile,
        state: AdaptiveRegimeState,
        pluginName: String,
        adaptiveRouterEnabled: Bool = true
    ) -> Double {
        guard adaptiveRouterEnabled, profile.ready, profile.enabled else { return 1.0 }
        var regimeBlend = 0.0
        for index in 0..<ControlPlaneConstants.adaptiveRouterRegimeLabels.count {
            regimeBlend += state.probabilities[index] *
                profile.regimeWeight(pluginName: pluginName, regimeLabel: regimeLabel(index))
        }
        if regimeBlend <= 0.0 {
            regimeBlend = 1.0
        }
        let sessionWeight = profile.sessionWeight(pluginName: pluginName, sessionLabel: state.sessionLabel)
        let globalWeight = profile.globalWeight(pluginName: pluginName)
        let newsCompatibility = profile.newsCompatibility(pluginName: pluginName)
        let liquidityRobustness = profile.liquidityRobustness(pluginName: pluginName)

        let newsFactor: Double
        if state.staleNews {
            newsFactor = profile.staleNewsForceCaution ? 0.86 : 0.95
        } else if state.newsRiskScore > 0.45 || state.topLabel == "HIGH_VOL_EVENT" {
            newsFactor = fxClamp(0.70 + 0.30 * newsCompatibility, 0.20, 1.60)
        } else {
            newsFactor = 1.0
        }

        let liquidityFactor = state.liquidityStress > 0.40
            ? fxClamp(0.68 + 0.32 * liquidityRobustness, 0.20, 1.60)
            : 1.0
        let suitability = globalWeight * regimeBlend * sessionWeight * newsFactor * liquidityFactor
        return fxClamp(suitability, profile.minPluginWeight, profile.maxPluginWeight)
    }

    public static func suitabilityStatus(
        profile: AdaptiveRouterProfile,
        suitability: Double,
        adaptiveRouterEnabled: Bool = true
    ) -> AdaptiveRouterRuntimeStatus {
        guard adaptiveRouterEnabled, profile.ready, profile.enabled else { return .active }
        if suitability < profile.suppressionThreshold {
            return .suppressed
        }
        if suitability < profile.downweightThreshold {
            return .downweighted
        }
        if suitability > 1.05 {
            return .upweighted
        }
        return .active
    }

    public static func computePosture(
        profile: AdaptiveRouterProfile,
        state: AdaptiveRegimeState,
        bestSuitability: Double,
        eligibleCount: Int,
        adaptiveRouterEnabled: Bool = true
    ) -> String {
        guard adaptiveRouterEnabled, profile.ready, profile.enabled else { return "NORMAL" }
        if eligibleCount <= 0 || bestSuitability <= profile.blockThreshold {
            return "BLOCK"
        }
        if state.staleNews && profile.staleNewsForceCaution {
            if bestSuitability <= profile.abstainThreshold {
                return "ABSTAIN_BIAS"
            }
            return "CAUTION"
        }
        if state.topLabel == "LIQUIDITY_STRESS" &&
            (state.liquidityStress >= 0.74 || bestSuitability <= profile.abstainThreshold) {
            return "BLOCK"
        }
        if state.topLabel == "HIGH_VOL_EVENT" {
            if bestSuitability <= profile.abstainThreshold || state.newsRiskScore >= 0.82 {
                return "ABSTAIN_BIAS"
            }
            return "CAUTION"
        }
        if bestSuitability <= profile.abstainThreshold || state.confidence < profile.confidenceFloor {
            return "ABSTAIN_BIAS"
        }
        if bestSuitability <= profile.cautionThreshold ||
            state.liquidityStress >= 0.56 ||
            state.breakoutPressure >= 0.72 {
            return "CAUTION"
        }
        return "NORMAL"
    }

    public static func postureAbstainBias(
        profile: AdaptiveRouterProfile,
        state: AdaptiveRegimeState,
        posture: String
    ) -> Double {
        var bias = 0.0
        switch posture {
        case "CAUTION":
            bias = 0.12
        case "ABSTAIN_BIAS":
            bias = 0.30
        case "BLOCK":
            bias = 0.92
        default:
            bias = 0.0
        }
        if state.staleNews {
            bias += profile.staleNewsAbstainBias
        }
        return fxClamp(bias, 0.0, 0.98)
    }

    public static func runtimeStateTSV(
        symbol: String,
        state: AdaptiveRegimeState,
        posture: String,
        abstainBias: Double,
        routes: [AdaptiveRouterPluginRoute] = []
    ) -> String? {
        guard state.valid, !symbol.isEmpty else { return nil }
        let csv = routeCSVs(routes: routes)
        return runtimeStateRows(
            symbol: symbol,
            state: state,
            posture: posture,
            abstainBias: abstainBias,
            activeCSV: csv.active,
            downweightedCSV: csv.downweighted,
            suppressedCSV: csv.suppressed
        )
        .map { key, value in
            "\(RuntimeArtifactTSV.field(key))\t\(RuntimeArtifactTSV.field(value))"
        }
        .joined(separator: "\r\n") + "\r\n"
    }

    public static func runtimeHistoryNDJSONLine(
        symbol: String,
        profile: AdaptiveRouterProfile,
        state: AdaptiveRegimeState,
        posture: String,
        abstainBias: Double,
        routes: [AdaptiveRouterPluginRoute] = []
    ) -> String? {
        guard state.valid, !symbol.isEmpty else { return nil }
        let probabilities = ControlPlaneConstants.adaptiveRouterRegimeLabels.enumerated()
            .map { index, label in
                "\(jsonQuoted(label)):\(RuntimeArtifactTSV.double(state.probabilities[index]))"
            }
            .joined(separator: ",")
        let reasons = state.reasons.map(jsonQuoted).joined(separator: ",")
        let plugins = pluginJSON(routes: routes, topLabel: state.topLabel)
        return "{" +
            "\"schema_version\":1," +
            "\"generated_at\":\(jsonQuoted(iso8601UTC(state.generatedAt)))," +
            "\"symbol\":\(jsonQuoted(symbol))," +
            "\"regime\":{" +
            "\"top_label\":\(jsonQuoted(state.topLabel))," +
            "\"confidence\":\(RuntimeArtifactTSV.double(state.confidence))," +
            "\"probabilities\":{\(probabilities)}," +
            "\"reasons\":[\(reasons)]," +
            "\"session\":\(jsonQuoted(state.sessionLabel))," +
            "\"spread_regime\":\(jsonQuoted(state.priceCostRegime))," +
            "\"volatility_regime\":\(jsonQuoted(state.volatilityRegime))," +
            "\"news_risk_score\":\(RuntimeArtifactTSV.double(state.newsRiskScore))," +
            "\"news_pressure\":\(RuntimeArtifactTSV.double(state.newsPressure))," +
            "\"event_eta_min\":\(state.eventETAMinutes)," +
            "\"stale_news\":\(state.staleNews ? "true" : "false")" +
            "}," +
            "\"router\":{" +
            "\"mode\":\(jsonQuoted(profile.routerMode))," +
            "\"top_regime\":\(jsonQuoted(state.topLabel))," +
            "\"trade_posture\":\(jsonQuoted(posture))," +
            "\"abstain_bias\":\(RuntimeArtifactTSV.double(abstainBias))," +
            "\"reasons\":[\(reasons)]" +
            "}," +
            "\"plugins\":[\(plugins)]" +
            "}"
    }

    public static func readRegimeState(
        symbol _: String,
        stateTSV: String?
    ) -> AdaptiveRegimeState? {
        guard let stateTSV else { return nil }
        let state = parseState(tsv: stateTSV)
        return state.valid ? state : nil
    }

    public static func parseState(tsv: String) -> AdaptiveRegimeState {
        var state = AdaptiveRegimeState.reset
        for line in tsv.components(separatedBy: .newlines) where !line.isEmpty {
            let parts = line.split(separator: "\t", omittingEmptySubsequences: false)
            guard parts.count >= 2 else { continue }
            let key = String(parts[0])
            let value = String(parts[1])
            state.valid = true
            switch key {
            case "symbol":
                state.symbol = value
            case "generated_at":
                state.generatedAt = Int64(value) ?? 0
            case "top_regime_label", "top_regime":
                state.topLabel = value
            case "regime_confidence":
                state.confidence = Double(value) ?? 0.0
            case "session_label":
                state.sessionLabel = value
            case "spread_regime", "price_cost_regime":
                state.priceCostRegime = value
            case "volatility_regime":
                state.volatilityRegime = value
            case "news_risk_score":
                state.newsRiskScore = Double(value) ?? 0.0
            case "news_pressure":
                state.newsPressure = Double(value) ?? 0.0
            case "event_eta_min":
                state.eventETAMinutes = Int(value) ?? -1
            case "stale_news":
                state.staleNews = (Int(value) ?? 0) != 0
            case "liquidity_stress":
                state.liquidityStress = Double(value) ?? 0.0
            case "breakout_pressure":
                state.breakoutPressure = Double(value) ?? 0.0
            case "trend_strength":
                state.trendStrength = Double(value) ?? 0.0
            case "range_pressure":
                state.rangePressure = Double(value) ?? 0.0
            case "macro_pressure":
                state.macroPressure = Double(value) ?? 0.0
            case "reasons_csv":
                for reason in value.split(separator: ";", omittingEmptySubsequences: false) {
                    state.appendReason(String(reason))
                }
            case "probabilities_csv":
                state.probabilities = parseProbabilitiesCSV(value)
            default:
                break
            }
        }
        return state
    }

    private static func priceDeltaPoints(
        series: M1OHLCVSeries,
        index: Int,
        lookback: Int,
        pointValue: Double
    ) -> Double {
        guard index >= lookback,
              index < series.count,
              pointValue > 0.0 else {
            return 0.0
        }
        return Double(series.close[index] - series.close[index - lookback]) / pointValue
    }

    private static func recentFlipRatio(
        series: M1OHLCVSeries,
        index: Int,
        count: Int
    ) -> Double {
        guard index >= count, count > 1 else { return 0.0 }
        var flips = 0
        var previousSign = 0
        for offset in 0..<count {
            let newer = index - offset
            let older = newer - 1
            let delta = series.close[newer] - series.close[older]
            let sign = delta > 0 ? 1 : (delta < 0 ? -1 : 0)
            if sign == 0 {
                continue
            }
            if previousSign != 0, sign != previousSign {
                flips += 1
            }
            previousSign = sign
        }
        return fxClamp(Double(flips) / Double(max(count - 1, 1)), 0.0, 1.0)
    }

    private static func rangePoints(
        series: M1OHLCVSeries,
        index: Int,
        count: Int,
        pointValue: Double
    ) -> Double {
        guard index >= 0,
              index < series.count,
              count > 0,
              pointValue > 0.0 else {
            return 0.0
        }
        let lowerBound = max(0, index - count + 1)
        var highest = series.high[index]
        var lowest = series.low[index]
        for cursor in lowerBound...index {
            highest = max(highest, series.high[cursor])
            lowest = min(lowest, series.low[cursor])
        }
        return max(0.0, Double(highest - lowest) / pointValue)
    }

    private static func routeCSVs(routes: [AdaptiveRouterPluginRoute]) -> (active: String, downweighted: String, suppressed: String) {
        let selectedTotal = routes
            .filter { $0.eligible && $0.routedWeight > 0.0 }
            .reduce(0.0) { $0 + $1.routedWeight }
        var active: [String] = []
        var downweighted: [String] = []
        var suppressed: [String] = []

        for route in routes where route.suitability > 0.0 || route.routedWeight > 0.0 || route.eligible {
            let normalizedWeight = selectedTotal > 0.0 && route.eligible ? route.routedWeight / selectedTotal : 0.0
            let token = "\(route.name):\(RuntimeArtifactTSV.double(normalizedWeight, decimals: 4)):\(RuntimeArtifactTSV.double(route.suitability, decimals: 4))"
            switch route.status {
            case .suppressed:
                suppressed.append(token)
            case .downweighted:
                downweighted.append(token)
            default:
                active.append(token)
            }
        }
        return (
            active.joined(separator: "|"),
            downweighted.joined(separator: "|"),
            suppressed.joined(separator: "|")
        )
    }

    private static func pluginJSON(routes: [AdaptiveRouterPluginRoute], topLabel: String) -> String {
        let selectedTotal = routes
            .filter { $0.eligible && $0.routedWeight > 0.0 }
            .reduce(0.0) { $0 + $1.routedWeight }
        return routes
            .filter { $0.suitability > 0.0 || $0.routedWeight > 0.0 || $0.eligible }
            .map { route in
                let normalizedWeight = selectedTotal > 0.0 && route.eligible ? route.routedWeight / selectedTotal : 0.0
                let reasons = route.reasons.isEmpty
                    ? [defaultPluginReason(status: route.status, topLabel: topLabel)]
                    : route.reasons
                return "{" +
                    "\"name\":\(jsonQuoted(route.name))," +
                    "\"eligible\":\(route.eligible ? "true" : "false")," +
                    "\"weight\":\(RuntimeArtifactTSV.double(normalizedWeight))," +
                    "\"suitability\":\(RuntimeArtifactTSV.double(route.suitability))," +
                    "\"status\":\(jsonQuoted(route.status.label))," +
                    "\"reasons\":[\(reasons.map(jsonQuoted).joined(separator: ","))]" +
                    "}"
            }
            .joined(separator: ",")
    }

    private static func defaultPluginReason(status: AdaptiveRouterRuntimeStatus, topLabel: String) -> String {
        switch status {
        case .suppressed:
            if topLabel == "HIGH_VOL_EVENT" {
                return "Suppressed in event regime"
            }
            if topLabel == "LIQUIDITY_STRESS" {
                return "Suppressed in liquidity stress"
            }
            return "Suppressed by low regime fit"
        case .downweighted:
            return "Downweighted by moderate regime fit"
        case .upweighted:
            return "Upweighted by strong regime fit"
        case .active:
            return "Balanced regime fit"
        }
    }

    private static func runtimeStateRows(
        symbol: String,
        state: AdaptiveRegimeState,
        posture: String,
        abstainBias: Double,
        activeCSV: String,
        downweightedCSV: String,
        suppressedCSV: String
    ) -> [(String, String)] {
        [
            ("schema_version", "1"),
            ("symbol", symbol),
            ("generated_at", "\(state.generatedAt)"),
            ("top_regime_label", state.topLabel),
            ("regime_confidence", RuntimeArtifactTSV.double(state.confidence)),
            ("trade_posture", posture),
            ("abstain_bias", RuntimeArtifactTSV.double(abstainBias)),
            ("session_label", state.sessionLabel),
            ("spread_regime", state.priceCostRegime),
            ("volatility_regime", state.volatilityRegime),
            ("news_risk_score", RuntimeArtifactTSV.double(state.newsRiskScore)),
            ("news_pressure", RuntimeArtifactTSV.double(state.newsPressure)),
            ("event_eta_min", "\(state.eventETAMinutes)"),
            ("stale_news", RuntimeArtifactTSV.bool(state.staleNews)),
            ("liquidity_stress", RuntimeArtifactTSV.double(state.liquidityStress)),
            ("breakout_pressure", RuntimeArtifactTSV.double(state.breakoutPressure)),
            ("trend_strength", RuntimeArtifactTSV.double(state.trendStrength)),
            ("range_pressure", RuntimeArtifactTSV.double(state.rangePressure)),
            ("macro_pressure", RuntimeArtifactTSV.double(state.macroPressure)),
            ("reasons_csv", state.reasonsCSV),
            ("probabilities_csv", probabilitiesCSV(state)),
            ("active_plugins_csv", activeCSV),
            ("downweighted_plugins_csv", downweightedCSV),
            ("suppressed_plugins_csv", suppressedCSV)
        ]
    }

    private static func probabilitiesCSV(_ state: AdaptiveRegimeState) -> String {
        ControlPlaneConstants.adaptiveRouterRegimeLabels.enumerated()
            .map { index, label in
                "\(label)=\(RuntimeArtifactTSV.double(state.probabilities[index]))"
            }
            .joined(separator: ",")
    }

    private static func parseProbabilitiesCSV(_ csv: String) -> [Double] {
        var output = Array(repeating: 0.0, count: ControlPlaneConstants.adaptiveRouterRegimeLabels.count)
        for token in csv.split(separator: ",", omittingEmptySubsequences: false) {
            let parts = token.split(separator: "=", maxSplits: 1, omittingEmptySubsequences: false)
            guard parts.count == 2,
                  let index = ControlPlaneConstants.adaptiveRouterRegimeLabels.firstIndex(of: String(parts[0])) else {
                continue
            }
            output[index] = Double(parts[1]) ?? 0.0
        }
        return output
    }

    private static func iso8601UTC(_ timestamp: Int64) -> String {
        guard timestamp > 0 else { return "" }
        var calendar = Calendar(identifier: .gregorian)
        calendar.timeZone = TimeZone(secondsFromGMT: 0) ?? .gmt
        let components = calendar.dateComponents(
            [.year, .month, .day, .hour, .minute, .second],
            from: Date(timeIntervalSince1970: TimeInterval(timestamp))
        )
        return String(
            format: "%04d-%02d-%02dT%02d:%02d:%02dZ",
            locale: Locale(identifier: "en_US_POSIX"),
            components.year ?? 0,
            components.month ?? 0,
            components.day ?? 0,
            components.hour ?? 0,
            components.minute ?? 0,
            components.second ?? 0
        )
    }

    private static func jsonQuoted(_ value: String) -> String {
        let escaped = value
            .replacingOccurrences(of: "\\", with: "\\\\")
            .replacingOccurrences(of: "\"", with: "\\\"")
            .replacingOccurrences(of: "\r", with: " ")
            .replacingOccurrences(of: "\n", with: " ")
        return "\"\(escaped)\""
    }
}

public extension RuntimeArtifactFileRepository {
    func writeAdaptiveRouterRuntimeArtifacts(
        symbol: String,
        profile: AdaptiveRouterProfile,
        state: AdaptiveRegimeState,
        posture: String,
        abstainBias: Double,
        routes: [AdaptiveRouterPluginRoute] = []
    ) throws {
        guard let stateTSV = AdaptiveRouterRuntimeTools.runtimeStateTSV(
            symbol: symbol,
            state: state,
            posture: posture,
            abstainBias: abstainBias,
            routes: routes
        ),
            let historyLine = AdaptiveRouterRuntimeTools.runtimeHistoryNDJSONLine(
                symbol: symbol,
                profile: profile,
                state: state,
                posture: posture,
                abstainBias: abstainBias,
                routes: routes
            ) else {
            return
        }

        let stateURL = url(for: AdaptiveRouterRuntimeTools.runtimeStatePath(symbol: symbol))
        try fileManager.createDirectory(
            at: stateURL.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        try stateTSV.write(to: stateURL, atomically: true, encoding: .utf8)

        let historyURL = url(for: AdaptiveRouterRuntimeTools.runtimeHistoryPath(symbol: symbol))
        try fileManager.createDirectory(
            at: historyURL.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        let historyData = Data((historyLine + "\r\n").utf8)
        if fileManager.fileExists(atPath: historyURL.path) {
            let handle = try FileHandle(forWritingTo: historyURL)
            defer { try? handle.close() }
            try handle.seekToEnd()
            try handle.write(contentsOf: historyData)
        } else {
            try historyData.write(to: historyURL, options: .atomic)
        }
    }
}
