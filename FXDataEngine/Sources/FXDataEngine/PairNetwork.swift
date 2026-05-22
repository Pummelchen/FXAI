import Foundation

public enum PairNetworkConstants {
    public static let maxReasons = 12
    public static let maxCurrencies = 24
    public static let maxCurrencyProfiles = 24
    public static let factorCount = 7
    public static let configFile = "FXAI/Runtime/pair_network_config.tsv"
    public static let statusFile = "FXAI/Runtime/pair_network_status.tsv"
}

public enum PairNetworkDecisionCode: Int, Codable, Sendable, CaseIterable {
    case allow = 0
    case allowReduced = 1
    case suppressRedundant = 2
    case blockContradictory = 3
    case blockConcentration = 4
    case preferAlternative = 5
}

public enum PairNetworkFactor: Int, Codable, Sendable, CaseIterable {
    case usdBloc = 0
    case eurRates = 1
    case safeHaven = 2
    case commodityFX = 3
    case riskOn = 4
    case liquidityStress = 5
    case macroShock = 6

    public var legacyName: String {
        switch self {
        case .usdBloc: "usd_bloc"
        case .eurRates: "eur_rates"
        case .safeHaven: "safe_haven"
        case .commodityFX: "commodity_fx"
        case .riskOn: "risk_on"
        case .liquidityStress: "liquidity_stress"
        case .macroShock: "macro_shock"
        }
    }
}

public struct PairNetworkCurrencyProfile: Codable, Hashable, Sendable {
    public var currency: String
    public var usdBloc: Double
    public var eurRates: Double
    public var safeHaven: Double
    public var commodityFX: Double
    public var riskOn: Double
    public var liquidityStress: Double
    public var macroShock: Double

    public init(
        currency: String = "",
        usdBloc: Double = 0.0,
        eurRates: Double = 0.0,
        safeHaven: Double = 0.0,
        commodityFX: Double = 0.0,
        riskOn: Double = 0.0,
        liquidityStress: Double = 0.0,
        macroShock: Double = 0.20
    ) {
        self.currency = currency
        self.usdBloc = usdBloc
        self.eurRates = eurRates
        self.safeHaven = safeHaven
        self.commodityFX = commodityFX
        self.riskOn = riskOn
        self.liquidityStress = liquidityStress
        self.macroShock = macroShock
    }
}

public struct PairNetworkConfig: Codable, Hashable, Sendable {
    public var ready: Bool
    public var enabled: Bool
    public var fallbackStructuralOnly: Bool
    public var autoApply: Bool
    public var fallbackGraphUsed: Bool
    public var partialDependencyData: Bool
    public var graphStale: Bool
    public var generatedAt: Int64
    public var graphMode: String
    public var graphStaleAfterSeconds: Int
    public var historyPoints: Int
    public var maxEdgesPerPair: Int
    public var minEmpiricalOverlap: Int
    public var empiricalLookbackBars: Int
    public var structuralWeight: Double
    public var empiricalWeight: Double
    public var redundancyThreshold: Double
    public var contradictionThreshold: Double
    public var concentrationReduceThreshold: Double
    public var concentrationBlockThreshold: Double
    public var executionOverlapThreshold: Double
    public var reducedSizeMultiplierFloor: Double
    public var preferredExpressionMargin: Double
    public var minIncrementalEdgeScore: Double
    public var weightEdgeAfterCosts: Double
    public var weightExecutionQuality: Double
    public var weightCalibrationQuality: Double
    public var weightPortfolioFit: Double
    public var weightDiversification: Double
    public var weightMacroFit: Double
    public var currencyProfiles: [PairNetworkCurrencyProfile]

    public init(
        ready: Bool = true,
        enabled: Bool = true,
        fallbackStructuralOnly: Bool = true,
        autoApply: Bool = true,
        fallbackGraphUsed: Bool = false,
        partialDependencyData: Bool = false,
        graphStale: Bool = true,
        generatedAt: Int64 = 0,
        graphMode: String = "STRUCTURAL_ONLY",
        graphStaleAfterSeconds: Int = 43_200,
        historyPoints: Int = 192,
        maxEdgesPerPair: Int = 10,
        minEmpiricalOverlap: Int = 128,
        empiricalLookbackBars: Int = 512,
        structuralWeight: Double = 0.72,
        empiricalWeight: Double = 0.28,
        redundancyThreshold: Double = 0.68,
        contradictionThreshold: Double = 0.74,
        concentrationReduceThreshold: Double = 0.58,
        concentrationBlockThreshold: Double = 0.80,
        executionOverlapThreshold: Double = 0.62,
        reducedSizeMultiplierFloor: Double = 0.45,
        preferredExpressionMargin: Double = 0.04,
        minIncrementalEdgeScore: Double = 0.12,
        weightEdgeAfterCosts: Double = 0.34,
        weightExecutionQuality: Double = 0.20,
        weightCalibrationQuality: Double = 0.16,
        weightPortfolioFit: Double = 0.14,
        weightDiversification: Double = 0.10,
        weightMacroFit: Double = 0.06,
        currencyProfiles: [PairNetworkCurrencyProfile] = []
    ) {
        self.ready = ready
        self.enabled = enabled
        self.fallbackStructuralOnly = fallbackStructuralOnly
        self.autoApply = autoApply
        self.fallbackGraphUsed = fallbackGraphUsed
        self.partialDependencyData = partialDependencyData
        self.graphStale = graphStale
        self.generatedAt = max(0, generatedAt)
        self.graphMode = graphMode
        self.graphStaleAfterSeconds = graphStaleAfterSeconds
        self.historyPoints = historyPoints
        self.maxEdgesPerPair = maxEdgesPerPair
        self.minEmpiricalOverlap = minEmpiricalOverlap
        self.empiricalLookbackBars = empiricalLookbackBars
        self.structuralWeight = structuralWeight
        self.empiricalWeight = empiricalWeight
        self.redundancyThreshold = redundancyThreshold
        self.contradictionThreshold = contradictionThreshold
        self.concentrationReduceThreshold = concentrationReduceThreshold
        self.concentrationBlockThreshold = concentrationBlockThreshold
        self.executionOverlapThreshold = executionOverlapThreshold
        self.reducedSizeMultiplierFloor = reducedSizeMultiplierFloor
        self.preferredExpressionMargin = preferredExpressionMargin
        self.minIncrementalEdgeScore = minIncrementalEdgeScore
        self.weightEdgeAfterCosts = weightEdgeAfterCosts
        self.weightExecutionQuality = weightExecutionQuality
        self.weightCalibrationQuality = weightCalibrationQuality
        self.weightPortfolioFit = weightPortfolioFit
        self.weightDiversification = weightDiversification
        self.weightMacroFit = weightMacroFit
        self.currencyProfiles = Array(currencyProfiles.prefix(PairNetworkConstants.maxCurrencyProfiles))
    }
}

public struct PairNetworkExposure: Codable, Hashable, Sendable {
    public var currencyKeys: [String]
    public var currencyValues: [Double]
    public var factorValues: [Double]

    public init(
        currencyKeys: [String] = [],
        currencyValues: [Double] = [],
        factorValues: [Double] = Array(repeating: 0.0, count: PairNetworkConstants.factorCount)
    ) {
        let count = min(currencyKeys.count, currencyValues.count, PairNetworkConstants.maxCurrencies)
        self.currencyKeys = Array(currencyKeys.prefix(count))
        self.currencyValues = Array(currencyValues.prefix(count))
        if factorValues.count >= PairNetworkConstants.factorCount {
            self.factorValues = Array(factorValues.prefix(PairNetworkConstants.factorCount))
        } else {
            self.factorValues = factorValues + Array(repeating: 0.0, count: PairNetworkConstants.factorCount - factorValues.count)
        }
    }

    public var currencyCount: Int {
        currencyKeys.count
    }
}

public enum PairNetworkTools {
    public static func runtimeStatePath(symbol: String) -> String {
        "FXAI/Runtime/fxai_pair_network_\(ControlPlanePaths.safeToken(symbol)).tsv"
    }

    public static func runtimeHistoryPath(symbol: String) -> String {
        "FXAI/Runtime/fxai_pair_network_history_\(ControlPlanePaths.safeToken(symbol)).ndjson"
    }

    public static func decisionLabel(_ code: PairNetworkDecisionCode) -> String {
        switch code {
        case .allow: "ALLOW"
        case .allowReduced: "ALLOW_REDUCED"
        case .suppressRedundant: "SUPPRESS_REDUNDANT"
        case .blockContradictory: "BLOCK_CONTRADICTORY"
        case .blockConcentration: "BLOCK_CONCENTRATION"
        case .preferAlternative: "PREFER_ALTERNATIVE_EXPRESSION"
        }
    }

    public static func factorName(_ index: Int) -> String {
        PairNetworkFactor(rawValue: index)?.legacyName ?? "unknown"
    }

    public static func parseConfig(configTSV: String? = nil, statusTSV: String? = nil) -> PairNetworkConfig {
        var config = PairNetworkConfig()

        if let configTSV {
            for line in configTSV.components(separatedBy: .newlines) where !line.isEmpty {
                let parts = line.split(separator: "\t", omittingEmptySubsequences: false)
                guard parts.count >= 2 else { continue }
                let key = String(parts[0])
                let value = String(parts[1])
                switch key {
                case "enabled": config.enabled = (Int(value) ?? 0) != 0
                case "graph_stale_after_sec": config.graphStaleAfterSeconds = Int(value) ?? 0
                case "history_points": config.historyPoints = Int(value) ?? 0
                case "max_edges_per_pair": config.maxEdgesPerPair = Int(value) ?? 0
                case "fallback_structural_only": config.fallbackStructuralOnly = (Int(value) ?? 0) != 0
                case "min_empirical_overlap": config.minEmpiricalOverlap = Int(value) ?? 0
                case "empirical_lookback_bars": config.empiricalLookbackBars = Int(value) ?? 0
                case "structural_weight": config.structuralWeight = Double(value) ?? 0.0
                case "empirical_weight": config.empiricalWeight = Double(value) ?? 0.0
                case "redundancy_threshold": config.redundancyThreshold = Double(value) ?? 0.0
                case "contradiction_threshold": config.contradictionThreshold = Double(value) ?? 0.0
                case "concentration_reduce_threshold": config.concentrationReduceThreshold = Double(value) ?? 0.0
                case "concentration_block_threshold": config.concentrationBlockThreshold = Double(value) ?? 0.0
                case "execution_overlap_threshold": config.executionOverlapThreshold = Double(value) ?? 0.0
                case "reduced_size_multiplier_floor": config.reducedSizeMultiplierFloor = Double(value) ?? 0.0
                case "preferred_expression_margin": config.preferredExpressionMargin = Double(value) ?? 0.0
                case "min_incremental_edge_score": config.minIncrementalEdgeScore = Double(value) ?? 0.0
                case "action_mode": config.autoApply = value != "RECOMMEND_ONLY"
                case "selection_weight" where parts.count >= 3:
                    setSelectionWeight(&config, key: String(parts[1]), value: Double(String(parts[2])) ?? 0.0)
                case "currency_profile" where parts.count >= 4:
                    setCurrencyProfileFactor(
                        &config,
                        currency: String(parts[1]),
                        factorName: String(parts[2]),
                        value: Double(String(parts[3])) ?? 0.0
                    )
                default:
                    break
                }
            }
        }

        if let statusTSV {
            for line in statusTSV.components(separatedBy: .newlines) where !line.isEmpty {
                let parts = line.split(separator: "\t", omittingEmptySubsequences: false)
                guard parts.count >= 2 else { continue }
                let key = String(parts[0])
                let value = String(parts[1])
                switch key {
                case "graph_mode": config.graphMode = value
                case "fallback_graph_used": config.fallbackGraphUsed = (Int(value) ?? 0) != 0
                case "partial_dependency_data": config.partialDependencyData = (Int(value) ?? 0) != 0
                case "graph_stale": config.graphStale = (Int(value) ?? 0) != 0
                default: break
                }
            }
        }

        config.ready = true
        return config
    }

    public static func profileFactorValue(_ profile: PairNetworkCurrencyProfile, factor: PairNetworkFactor) -> Double {
        switch factor {
        case .usdBloc: profile.usdBloc
        case .eurRates: profile.eurRates
        case .safeHaven: profile.safeHaven
        case .commodityFX: profile.commodityFX
        case .riskOn: profile.riskOn
        case .liquidityStress: profile.liquidityStress
        case .macroShock: profile.macroShock
        }
    }

    public static func currencyFactorWeight(_ config: PairNetworkConfig, currency: String, factor: PairNetworkFactor) -> Double {
        guard let profile = config.currencyProfiles.first(where: { $0.currency == currency }) else {
            return 0.0
        }
        return profileFactorValue(profile, factor: factor)
    }

    public static func buildSymbolExposure(
        config: PairNetworkConfig,
        symbol: String,
        direction: Int,
        sizeUnits: Double
    ) -> PairNetworkExposure {
        var exposure = PairNetworkExposure()
        let legs = ExposureTools.parseSymbolLegs(symbol)
        guard legs.isValidPair, direction == 0 || direction == 1 else { return exposure }

        let signedUnits = abs(sizeUnits) * (direction == 1 ? 1.0 : -1.0)
        addKeyValue(&exposure, key: legs.base, delta: signedUnits)
        addKeyValue(&exposure, key: legs.quote, delta: -signedUnits)

        for factor in PairNetworkFactor.allCases {
            exposure.factorValues[factor.rawValue] += signedUnits * currencyFactorWeight(config, currency: legs.base, factor: factor)
            exposure.factorValues[factor.rawValue] -= signedUnits * currencyFactorWeight(config, currency: legs.quote, factor: factor)
        }
        return exposure
    }

    public static func structuralOverlap(config: PairNetworkConfig, lhsSymbol: String, rhsSymbol: String) -> Double {
        let lhs = ExposureTools.parseSymbolLegs(lhsSymbol)
        let rhs = ExposureTools.parseSymbolLegs(rhsSymbol)
        guard lhs.isValidPair, rhs.isValidPair else { return 0.0 }

        var currencyScore = 0.0
        if lhs.base == rhs.base, lhs.quote == rhs.quote {
            currencyScore = 1.0
        } else if lhs.base == rhs.quote, lhs.quote == rhs.base {
            currencyScore = 0.96
        } else if lhs.base == rhs.base || lhs.quote == rhs.quote {
            currencyScore = 0.84
        } else if lhs.base == rhs.quote || lhs.quote == rhs.base {
            currencyScore = 0.72
        } else if ExposureTools.symbolsShareCurrency(lhsSymbol, rhsSymbol) {
            currencyScore = 0.56
        }

        var lhsFactors = buildSymbolExposure(config: config, symbol: lhsSymbol, direction: 1, sizeUnits: 1.0).factorValues.map(abs)
        var rhsFactors = buildSymbolExposure(config: config, symbol: rhsSymbol, direction: 1, sizeUnits: 1.0).factorValues.map(abs)
        normalizeFactorVectorLength(&lhsFactors)
        normalizeFactorVectorLength(&rhsFactors)
        let factorScore = fxClamp((factorCosine(lhsFactors, rhsFactors) + 1.0) * 0.5, 0.0, 1.0)

        var clusterBonus = 0.0
        if lhsSymbol.contains("USD"), rhsSymbol.contains("USD") {
            clusterBonus += 0.08
        }
        if containsCommodityBloc(lhsSymbol), containsCommodityBloc(rhsSymbol) {
            clusterBonus += 0.12
        }
        if containsSafeHavenBloc(lhsSymbol), containsSafeHavenBloc(rhsSymbol) {
            clusterBonus += 0.10
        }
        return fxClamp(0.64 * currencyScore + 0.28 * factorScore + clusterBonus, 0.0, 1.0)
    }

    public static func qualityScore(
        config: PairNetworkConfig,
        edgeScore: Double,
        executionQualityScore: Double,
        calibrationQuality: Double,
        portfolioFit: Double,
        macroFit: Double,
        overlapScore: Double
    ) -> Double {
        let diversification = fxClamp(1.0 - overlapScore, 0.0, 1.0)
        return fxClamp(
            config.weightEdgeAfterCosts * edgeScore +
                config.weightExecutionQuality * fxClamp(executionQualityScore, 0.0, 1.0) +
                config.weightCalibrationQuality * fxClamp(calibrationQuality, 0.0, 1.0) +
                config.weightPortfolioFit * fxClamp(portfolioFit, 0.0, 1.0) +
                config.weightDiversification * diversification +
                config.weightMacroFit * fxClamp(macroFit, 0.0, 1.0),
            0.0,
            1.0
        )
    }

    static func factorCosine(_ lhs: [Double], _ rhs: [Double]) -> Double {
        let lhsNorm = factorNorm(lhs)
        let rhsNorm = factorNorm(rhs)
        guard lhsNorm > 0.0, rhsNorm > 0.0 else { return 0.0 }
        var dot = 0.0
        for index in 0..<PairNetworkConstants.factorCount {
            dot += lhs[index] * rhs[index]
        }
        return fxClamp(dot / (lhsNorm * rhsNorm), -1.0, 1.0)
    }

    private static func factorNorm(_ values: [Double]) -> Double {
        var total = 0.0
        for index in 0..<PairNetworkConstants.factorCount {
            total += values[index] * values[index]
        }
        return total > 0.0 ? sqrt(total) : 0.0
    }

    private static func addKeyValue(_ exposure: inout PairNetworkExposure, key: String, delta: Double) {
        guard !key.isEmpty else { return }
        if let index = exposure.currencyKeys.firstIndex(of: key) {
            exposure.currencyValues[index] += delta
            return
        }
        guard exposure.currencyKeys.count < PairNetworkConstants.maxCurrencies else { return }
        exposure.currencyKeys.append(key)
        exposure.currencyValues.append(delta)
    }

    private static func setSelectionWeight(_ config: inout PairNetworkConfig, key: String, value: Double) {
        switch key {
        case "edge_after_costs": config.weightEdgeAfterCosts = value
        case "execution_quality": config.weightExecutionQuality = value
        case "calibration_quality": config.weightCalibrationQuality = value
        case "portfolio_fit": config.weightPortfolioFit = value
        case "diversification": config.weightDiversification = value
        case "macro_fit": config.weightMacroFit = value
        default: break
        }
    }

    private static func setCurrencyProfileFactor(
        _ config: inout PairNetworkConfig,
        currency: String,
        factorName: String,
        value: Double
    ) {
        guard !currency.isEmpty else { return }
        var index = config.currencyProfiles.firstIndex(where: { $0.currency == currency })
        if index == nil, config.currencyProfiles.count < PairNetworkConstants.maxCurrencyProfiles {
            config.currencyProfiles.append(PairNetworkCurrencyProfile(currency: currency))
            index = config.currencyProfiles.count - 1
        }
        guard let index else { return }
        switch factorName {
        case "usd_bloc": config.currencyProfiles[index].usdBloc = value
        case "eur_rates": config.currencyProfiles[index].eurRates = value
        case "safe_haven": config.currencyProfiles[index].safeHaven = value
        case "commodity_fx": config.currencyProfiles[index].commodityFX = value
        case "risk_on": config.currencyProfiles[index].riskOn = value
        case "liquidity_stress": config.currencyProfiles[index].liquidityStress = value
        case "macro_shock": config.currencyProfiles[index].macroShock = value
        default: break
        }
    }

    private static func normalizeFactorVectorLength(_ values: inout [Double]) {
        if values.count < PairNetworkConstants.factorCount {
            values += Array(repeating: 0.0, count: PairNetworkConstants.factorCount - values.count)
        } else if values.count > PairNetworkConstants.factorCount {
            values = Array(values.prefix(PairNetworkConstants.factorCount))
        }
    }

    private static func containsCommodityBloc(_ symbol: String) -> Bool {
        symbol.contains("AUD") || symbol.contains("CAD") || symbol.contains("NZD") || symbol.contains("NOK")
    }

    private static func containsSafeHavenBloc(_ symbol: String) -> Bool {
        symbol.contains("JPY") || symbol.contains("CHF")
    }
}
