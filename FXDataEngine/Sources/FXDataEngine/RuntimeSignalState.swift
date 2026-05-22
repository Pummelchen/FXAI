import Foundation

public struct RuntimeSignalCache: Codable, Hashable, Sendable {
    public var expectedMovePoints = 0.0
    public var tradeEdgePoints = 0.0
    public var confidence = 0.0
    public var reliability = 0.0
    public var pathRisk = 1.0
    public var fillRisk = 1.0
    public var tradeGate = 0.0
    public var hierarchyScore = 0.0
    public var hierarchyConsistency = 0.0
    public var hierarchyTradability = 0.0
    public var hierarchyExecution = 0.0
    public var hierarchyHorizonFit = 0.0
    public var macroStateQuality = 0.0
    public var portfolioPressure = 0.0
    public var contextQuality = 0.0
    public var contextStrength = 0.0
    public var minMovePoints = 0.0
    public var horizonMinutes = 0
    public var regimeID = 0

    public var policyTradeProbability = 0.0
    public var policyNoTradeProbability = 1.0
    public var policyEnterProbability = 0.0
    public var policyExitProbability = 0.0
    public var policyDirectionBias = 0.0
    public var policySizeMultiplier = 1.0
    public var policyHoldQuality = 0.0
    public var policyExpectedUtility = 0.0
    public var policyConfidence = 0.0
    public var policyPortfolioFit = 0.0
    public var policyCapitalEfficiency = 0.0
    public var policyAddProbability = 0.0
    public var policyReduceProbability = 0.0
    public var policyTightenProbability = 0.0
    public var policyTimeoutProbability = 0.0
    public var policyAction: PolicyLifecycleAction = .noTrade

    public var controlPlaneScore = 0.0
    public var controlPlaneBuyScore = 0.0
    public var controlPlaneSellScore = 0.0
    public var controlPlaneSymbol = ""
    public var controlPlaneBarTimeUTC: Int64 = 0

    public var adaptiveRouterReady = false
    public var adaptiveRouterTopLabel = "UNKNOWN"
    public var adaptiveRouterConfidence = 0.0
    public var adaptiveRouterPosture = "NORMAL"
    public var adaptiveRouterAbstainBias = 0.0
    public var adaptiveRouterSession = ""
    public var adaptiveRouterPriceCostRegime = ""
    public var adaptiveRouterVolatilityRegime = ""
    public var adaptiveRouterNewsRisk = 0.0
    public var adaptiveRouterLiquidityStress = 0.0
    public var adaptiveRouterGeneratedAt: Int64 = 0
    public var adaptiveRouterReasonsCSV = ""
    public var adaptiveRouterActivePluginsCSV = ""
    public var adaptiveRouterDownweightedPluginsCSV = ""
    public var adaptiveRouterSuppressedPluginsCSV = ""

    public var dynamicEnsembleReady = false
    public var dynamicEnsembleQuality = 0.0
    public var dynamicEnsembleAbstainBias = 0.0
    public var dynamicEnsembleTradePosture = "NORMAL"
    public var dynamicEnsembleTopRegime = "UNKNOWN"
    public var dynamicEnsembleSession = ""
    public var dynamicEnsembleGeneratedAt: Int64 = 0
    public var dynamicEnsembleBuyProbability = 0.0
    public var dynamicEnsembleSellProbability = 0.0
    public var dynamicEnsembleSkipProbability = 1.0
    public var dynamicEnsembleReasonsCSV = ""
    public var dynamicEnsembleActivePluginsCSV = ""
    public var dynamicEnsembleDownweightedPluginsCSV = ""
    public var dynamicEnsembleSuppressedPluginsCSV = ""

    public var executionQualityReady = false
    public var executionQualityFallbackUsed = false
    public var executionQualityMemoryStale = true
    public var executionQualityDataStale = true
    public var executionQualitySupportUsable = false
    public var executionQualityGeneratedAt: Int64 = 0
    public var executionQualityMethod = "SCORECARD_V1"
    public var executionQualityTierKind = "GLOBAL"
    public var executionQualityTierKey = "GLOBAL|*|*|*"
    public var executionQualitySupport = 0
    public var executionQualityQuality = 0.0
    public var executionQualityPriceCostNowPoints = 0.0
    public var executionQualityPriceCostExpectedPoints = 0.0
    public var executionQualityPriceCostWideningRisk = 0.0
    public var executionQualityExpectedSlippagePoints = 0.0
    public var executionQualitySlippageRisk = 0.0
    public var executionQualityFillQuality = 0.0
    public var executionQualityLatencySensitivity = 0.0
    public var executionQualityLiquidityFragility = 0.0
    public var executionQualityQualityScore = 0.0
    public var executionQualityAllowedDeviationPoints = 0.0
    public var executionQualityCautionLotScale = 1.0
    public var executionQualityCautionEnterProbabilityBuffer = 0.0
    public var executionQualityState = "UNKNOWN"
    public var executionQualityReasonsCSV = ""

    public var probabilityCalibrationReady = false
    public var probabilityCalibrationFallbackUsed = false
    public var probabilityCalibrationAbstain = false
    public var probabilityCalibrationCalibrationStale = true
    public var probabilityCalibrationInputStale = true
    public var probabilityCalibrationGeneratedAt: Int64 = 0
    public var probabilityCalibrationMethod = "LOGISTIC_AFFINE"
    public var probabilityCalibrationTierKind = "GLOBAL"
    public var probabilityCalibrationTierKey = "GLOBAL|*|*|*"
    public var probabilityCalibrationSupport = 0
    public var probabilityCalibrationQuality = 0.0
    public var probabilityCalibrationRawScore = 0.0
    public var probabilityCalibrationRawAction = "SKIP"
    public var probabilityCalibrationRawBuyProbability = 0.0
    public var probabilityCalibrationRawSellProbability = 0.0
    public var probabilityCalibrationRawSkipProbability = 1.0
    public var probabilityCalibrationBuyProbability = 0.0
    public var probabilityCalibrationSellProbability = 0.0
    public var probabilityCalibrationSkipProbability = 1.0
    public var probabilityCalibrationConfidence = 0.0
    public var probabilityCalibrationMoveMeanPoints = 0.0
    public var probabilityCalibrationMoveQ25Points = 0.0
    public var probabilityCalibrationMoveQ50Points = 0.0
    public var probabilityCalibrationMoveQ75Points = 0.0
    public var probabilityCalibrationPriceCostPoints = 0.0
    public var probabilityCalibrationSlippageCostPoints = 0.0
    public var probabilityCalibrationUncertaintyScore = 0.0
    public var probabilityCalibrationUncertaintyPenaltyPoints = 0.0
    public var probabilityCalibrationRiskPenaltyPoints = 0.0
    public var probabilityCalibrationGrossEdgePoints = 0.0
    public var probabilityCalibrationEdgeAfterCostsPoints = 0.0
    public var probabilityCalibrationFinalAction = "SKIP"
    public var probabilityCalibrationSession = "UNKNOWN"
    public var probabilityCalibrationRegime = "UNKNOWN"
    public var probabilityCalibrationReasonsCSV = ""
    public var probabilityCalibrationPrimaryReason = ""

    public init() {}

    public static var reset: RuntimeSignalCache {
        RuntimeSignalCache()
    }

    public var riskPolicySignalState: RiskPolicySignalState {
        RiskPolicySignalState(
            confidence: confidence,
            reliability: reliability,
            tradeGate: tradeGate,
            pathRisk: pathRisk,
            fillRisk: fillRisk,
            minMovePoints: minMovePoints,
            tradeEdgePoints: tradeEdgePoints,
            expectedMovePoints: expectedMovePoints,
            hierarchyScore: hierarchyScore,
            hierarchyConsistency: hierarchyConsistency,
            hierarchyTradability: hierarchyTradability,
            hierarchyExecution: hierarchyExecution,
            contextStrength: contextStrength,
            macroStateQuality: macroStateQuality,
            policyAction: policyAction,
            policyEnterProb: policyEnterProbability,
            policyNoTradeProb: policyNoTradeProbability,
            policyHoldQuality: policyHoldQuality,
            policySizeMultiplier: policySizeMultiplier,
            policyPortfolioFit: policyPortfolioFit,
            policyCapitalEfficiency: policyCapitalEfficiency
        )
    }

    public mutating func apply(
        signal: RiskPolicySignalState,
        portfolioPressure: Double = 0.0,
        contextQuality: Double = 0.0,
        hierarchyHorizonFit: Double = 0.0,
        horizonMinutes: Int = 0,
        regimeID: Int = 0
    ) {
        expectedMovePoints = signal.expectedMovePoints
        tradeEdgePoints = signal.tradeEdgePoints
        confidence = signal.confidence
        reliability = signal.reliability
        pathRisk = signal.pathRisk
        fillRisk = signal.fillRisk
        tradeGate = signal.tradeGate
        hierarchyScore = signal.hierarchyScore
        hierarchyConsistency = signal.hierarchyConsistency
        hierarchyTradability = signal.hierarchyTradability
        hierarchyExecution = signal.hierarchyExecution
        self.hierarchyHorizonFit = hierarchyHorizonFit
        macroStateQuality = signal.macroStateQuality
        self.portfolioPressure = portfolioPressure
        self.contextQuality = contextQuality
        contextStrength = signal.contextStrength
        minMovePoints = signal.minMovePoints
        self.horizonMinutes = horizonMinutes
        self.regimeID = regimeID

        policyTradeProbability = signal.policyEnterProb
        policyNoTradeProbability = signal.policyNoTradeProb
        policyEnterProbability = signal.policyEnterProb
        policyExitProbability = 0.0
        policyDirectionBias = 0.0
        policySizeMultiplier = signal.policySizeMultiplier
        policyHoldQuality = signal.policyHoldQuality
        policyExpectedUtility = 0.0
        policyConfidence = signal.confidence
        policyPortfolioFit = signal.policyPortfolioFit
        policyCapitalEfficiency = signal.policyCapitalEfficiency
        policyAddProbability = 0.0
        policyReduceProbability = 0.0
        policyTightenProbability = 0.0
        policyTimeoutProbability = 0.0
        policyAction = signal.policyAction
    }

    public mutating func apply(policy: MetaPolicyDecision) {
        policyTradeProbability = policy.tradeProbability
        policyNoTradeProbability = policy.noTradeProbability
        policyEnterProbability = policy.enterProbability
        policyExitProbability = policy.exitProbability
        policyDirectionBias = policy.directionBias
        policySizeMultiplier = policy.sizeMultiplier
        policyHoldQuality = policy.holdQuality
        policyExpectedUtility = policy.expectedUtility
        policyConfidence = policy.confidence
        policyPortfolioFit = policy.portfolioFit
        policyCapitalEfficiency = policy.capitalEfficiency
        policyAddProbability = policy.addProbability
        policyReduceProbability = policy.reduceProbability
        policyTightenProbability = policy.tightenProbability
        policyTimeoutProbability = policy.timeoutProbability
        policyAction = policy.action
    }

    public mutating func apply(
        controlPlaneOverlay: RiskControlPlaneOverlayState,
        symbol: String = "",
        barTimeUTC: Int64 = 0
    ) {
        controlPlaneScore = controlPlaneOverlay.controlPlaneScore ?? 0.0
        controlPlaneBuyScore = controlPlaneOverlay.controlPlaneBuyScore
        controlPlaneSellScore = controlPlaneOverlay.controlPlaneSellScore
        controlPlaneSymbol = symbol
        controlPlaneBarTimeUTC = max(0, barTimeUTC)
    }

    public mutating func apply(
        adaptiveRouterState state: AdaptiveRegimeState,
        posture: String,
        abstainBias: Double,
        routes: [AdaptiveRouterPluginRoute] = []
    ) {
        guard state.valid else {
            adaptiveRouterReady = false
            adaptiveRouterTopLabel = "UNKNOWN"
            adaptiveRouterConfidence = 0.0
            adaptiveRouterPosture = "NORMAL"
            adaptiveRouterAbstainBias = 0.0
            adaptiveRouterSession = ""
            adaptiveRouterPriceCostRegime = ""
            adaptiveRouterVolatilityRegime = ""
            adaptiveRouterNewsRisk = 0.0
            adaptiveRouterLiquidityStress = 0.0
            adaptiveRouterGeneratedAt = 0
            adaptiveRouterReasonsCSV = ""
            adaptiveRouterActivePluginsCSV = ""
            adaptiveRouterDownweightedPluginsCSV = ""
            adaptiveRouterSuppressedPluginsCSV = ""
            return
        }
        adaptiveRouterReady = state.valid
        adaptiveRouterTopLabel = state.topLabel
        adaptiveRouterConfidence = state.confidence
        adaptiveRouterPosture = posture.isEmpty ? "NORMAL" : posture
        adaptiveRouterAbstainBias = fxClamp(abstainBias, 0.0, 0.98)
        adaptiveRouterSession = state.sessionLabel
        adaptiveRouterPriceCostRegime = state.priceCostRegime
        adaptiveRouterVolatilityRegime = state.volatilityRegime
        adaptiveRouterNewsRisk = state.newsRiskScore
        adaptiveRouterLiquidityStress = state.liquidityStress
        adaptiveRouterGeneratedAt = state.generatedAt
        adaptiveRouterReasonsCSV = state.reasonsCSV

        let csv = RuntimeSignalStateTools.adaptiveRouterPluginCSVs(routes: routes)
        adaptiveRouterActivePluginsCSV = csv.active
        adaptiveRouterDownweightedPluginsCSV = csv.downweighted
        adaptiveRouterSuppressedPluginsCSV = csv.suppressed
    }

    public mutating func apply(
        dynamicEnsembleState state: DynamicEnsembleRuntimeState,
        records: [DynamicEnsemblePluginRecord] = []
    ) {
        dynamicEnsembleReady = state.ready
        dynamicEnsembleQuality = state.ensembleQuality
        dynamicEnsembleAbstainBias = state.abstainBias
        dynamicEnsembleTradePosture = state.tradePosture
        dynamicEnsembleTopRegime = state.topRegime
        dynamicEnsembleSession = state.sessionLabel
        dynamicEnsembleGeneratedAt = state.generatedAt
        dynamicEnsembleBuyProbability = state.buyProbability
        dynamicEnsembleSellProbability = state.sellProbability
        dynamicEnsembleSkipProbability = state.skipProbability
        dynamicEnsembleReasonsCSV = state.reasonsCSV

        let csv = RuntimeSignalStateTools.dynamicEnsemblePluginCSVs(records: records)
        dynamicEnsembleActivePluginsCSV = csv.active
        dynamicEnsembleDownweightedPluginsCSV = csv.downweighted
        dynamicEnsembleSuppressedPluginsCSV = csv.suppressed
    }

    public mutating func apply(executionQualityState state: ExecutionQualityPairState) {
        executionQualityReady = state.ready
        executionQualityFallbackUsed = state.fallbackUsed
        executionQualityMemoryStale = state.memoryStale
        executionQualityDataStale = state.dataStale
        executionQualitySupportUsable = state.supportUsable
        executionQualityGeneratedAt = state.generatedAt
        executionQualityMethod = state.method
        executionQualityTierKind = state.selectedTierKind
        executionQualityTierKey = state.selectedTierKey
        executionQualitySupport = state.selectedSupport
        executionQualityQuality = state.selectedQuality
        executionQualityPriceCostNowPoints = state.spreadNowPoints
        executionQualityPriceCostExpectedPoints = state.spreadExpectedPoints
        executionQualityPriceCostWideningRisk = state.spreadWideningRisk
        executionQualityExpectedSlippagePoints = state.expectedSlippagePoints
        executionQualitySlippageRisk = state.slippageRisk
        executionQualityFillQuality = state.fillQualityScore
        executionQualityLatencySensitivity = state.latencySensitivityScore
        executionQualityLiquidityFragility = state.liquidityFragilityScore
        executionQualityQualityScore = state.executionQualityScore
        executionQualityAllowedDeviationPoints = state.allowedDeviationPoints
        executionQualityCautionLotScale = state.cautionLotScale
        executionQualityCautionEnterProbabilityBuffer = state.cautionEnterProbabilityBuffer
        executionQualityState = state.executionState
        executionQualityReasonsCSV = state.reasonsCSV
    }

    public mutating func apply(probabilityCalibrationState state: ProbabilityCalibrationRuntimeState) {
        probabilityCalibrationReady = state.ready
        probabilityCalibrationFallbackUsed = state.fallbackUsed
        probabilityCalibrationAbstain = state.abstain
        probabilityCalibrationCalibrationStale = state.calibrationStale
        probabilityCalibrationInputStale = state.inputStale
        probabilityCalibrationGeneratedAt = state.generatedAt
        probabilityCalibrationMethod = state.method
        probabilityCalibrationTierKind = state.selectedTierKind
        probabilityCalibrationTierKey = state.selectedTierKey
        probabilityCalibrationSupport = state.selectedSupport
        probabilityCalibrationQuality = state.selectedQuality
        probabilityCalibrationRawScore = state.rawScore
        probabilityCalibrationRawAction = state.rawAction
        probabilityCalibrationRawBuyProbability = state.rawBuyProbability
        probabilityCalibrationRawSellProbability = state.rawSellProbability
        probabilityCalibrationRawSkipProbability = state.rawSkipProbability
        probabilityCalibrationBuyProbability = state.calibratedBuyProbability
        probabilityCalibrationSellProbability = state.calibratedSellProbability
        probabilityCalibrationSkipProbability = state.calibratedSkipProbability
        probabilityCalibrationConfidence = state.calibratedConfidence
        probabilityCalibrationMoveMeanPoints = state.expectedMoveMeanPoints
        probabilityCalibrationMoveQ25Points = state.expectedMoveQ25Points
        probabilityCalibrationMoveQ50Points = state.expectedMoveQ50Points
        probabilityCalibrationMoveQ75Points = state.expectedMoveQ75Points
        probabilityCalibrationPriceCostPoints = state.priceCostPoints
        probabilityCalibrationSlippageCostPoints = state.slippageCostPoints
        probabilityCalibrationUncertaintyScore = state.uncertaintyScore
        probabilityCalibrationUncertaintyPenaltyPoints = state.uncertaintyPenaltyPoints
        probabilityCalibrationRiskPenaltyPoints = state.riskPenaltyPoints
        probabilityCalibrationGrossEdgePoints = state.expectedGrossEdgePoints
        probabilityCalibrationEdgeAfterCostsPoints = state.edgeAfterCostsPoints
        probabilityCalibrationFinalAction = state.finalAction
        probabilityCalibrationSession = state.sessionLabel
        probabilityCalibrationRegime = state.regimeLabel
        probabilityCalibrationReasonsCSV = state.reasonsCSV
        probabilityCalibrationPrimaryReason = state.reasons.first ?? ""
    }
}

public enum RuntimeSignalStateTools {
    public static func controlPlaneSnapshot(
        symbol: String,
        direction: Int,
        signalIntensity: Double,
        login: Int64,
        magic: UInt64,
        chartID: Int64,
        barTimeUTC: Int64,
        cache: RuntimeSignalCache,
        exposure: RiskPortfolioExposureState = RiskPortfolioExposureState(),
        capitalRiskPct: Double = 0.0
    ) -> ControlPlaneSnapshot {
        var snapshot = ControlPlaneSnapshot()
        snapshot.login = login
        snapshot.magic = magic
        snapshot.chartID = chartID
        snapshot.symbol = symbol
        snapshot.barTimeUTC = barTimeUTC > 0 ? barTimeUTC : cache.controlPlaneBarTimeUTC
        snapshot.direction = direction

        var intensity = fxClamp(signalIntensity, 0.0, 4.0)
        if direction < 0 {
            intensity *= 0.35
        }
        snapshot.signalIntensity = intensity

        let minimumMove = max(cache.minMovePoints, 0.10)
        snapshot.confidence = cache.confidence
        snapshot.reliability = cache.reliability
        snapshot.tradeGate = cache.tradeGate
        snapshot.hierarchyScore = cache.hierarchyScore
        snapshot.macroQuality = cache.macroStateQuality
        snapshot.tradeEdgeNorm = fxClamp(cache.tradeEdgePoints / minimumMove, -4.0, 4.0) / 4.0
        snapshot.expectedMoveNorm = fxClamp(cache.expectedMovePoints / minimumMove, 0.0, 8.0) / 2.0
        snapshot.policyTradeProb = cache.policyTradeProbability
        snapshot.policyNoTradeProb = cache.policyNoTradeProbability
        snapshot.policyEnterProb = cache.policyEnterProbability
        snapshot.policyExitProb = cache.policyExitProbability
        snapshot.policyAddProb = cache.policyAddProbability
        snapshot.policyReduceProb = cache.policyReduceProbability
        snapshot.policyTightenProb = cache.policyTightenProbability
        snapshot.policyTimeoutProb = cache.policyTimeoutProbability
        snapshot.policySizeMultiplier = cache.policySizeMultiplier
        snapshot.policyPortfolioFit = cache.policyPortfolioFit
        snapshot.policyCapitalEfficiency = cache.policyCapitalEfficiency
        snapshot.policyLifecycleAction = cache.policyAction
        snapshot.grossExposureLots = exposure.grossExposureLots
        snapshot.correlatedExposureLots = exposure.correlatedExposureLots
        snapshot.directionalClusterLots = exposure.directionalClusterLots
        snapshot.capitalRiskPct = capitalRiskPct
        snapshot.portfolioPressure = cache.portfolioPressure
        return snapshot.normalized()
    }

    public static func idleControlPlaneSnapshot(
        symbol: String,
        login: Int64,
        magic: UInt64,
        chartID: Int64,
        barTimeUTC: Int64,
        cache: RuntimeSignalCache = .reset
    ) -> ControlPlaneSnapshot {
        controlPlaneSnapshot(
            symbol: symbol,
            direction: -1,
            signalIntensity: 0.0,
            login: login,
            magic: magic,
            chartID: chartID,
            barTimeUTC: barTimeUTC,
            cache: cache
        )
    }

    public static func controlPlaneSnapshotTSV(_ snapshot: ControlPlaneSnapshot) -> String {
        let rows: [(String, String)] = [
            ("login", String(snapshot.login)),
            ("magic", String(snapshot.magic)),
            ("chart_id", String(snapshot.chartID % 2_147_483_647)),
            ("symbol", snapshot.symbol),
            ("bar_time", String(snapshot.barTimeUTC)),
            ("direction", String(snapshot.direction)),
            ("signal_intensity", RuntimeArtifactTSV.double(snapshot.signalIntensity)),
            ("confidence", RuntimeArtifactTSV.double(snapshot.confidence)),
            ("reliability", RuntimeArtifactTSV.double(snapshot.reliability)),
            ("trade_gate", RuntimeArtifactTSV.double(snapshot.tradeGate)),
            ("hierarchy_score", RuntimeArtifactTSV.double(snapshot.hierarchyScore)),
            ("macro_quality", RuntimeArtifactTSV.double(snapshot.macroQuality)),
            ("trade_edge_norm", RuntimeArtifactTSV.double(snapshot.tradeEdgeNorm)),
            ("expected_move_norm", RuntimeArtifactTSV.double(snapshot.expectedMoveNorm)),
            ("policy_trade_prob", RuntimeArtifactTSV.double(snapshot.policyTradeProb)),
            ("policy_no_trade_prob", RuntimeArtifactTSV.double(snapshot.policyNoTradeProb)),
            ("policy_enter_prob", RuntimeArtifactTSV.double(snapshot.policyEnterProb)),
            ("policy_exit_prob", RuntimeArtifactTSV.double(snapshot.policyExitProb)),
            ("policy_add_prob", RuntimeArtifactTSV.double(snapshot.policyAddProb)),
            ("policy_reduce_prob", RuntimeArtifactTSV.double(snapshot.policyReduceProb)),
            ("policy_tighten_prob", RuntimeArtifactTSV.double(snapshot.policyTightenProb)),
            ("policy_timeout_prob", RuntimeArtifactTSV.double(snapshot.policyTimeoutProb)),
            ("policy_size_mult", RuntimeArtifactTSV.double(snapshot.policySizeMultiplier)),
            ("policy_portfolio_fit", RuntimeArtifactTSV.double(snapshot.policyPortfolioFit)),
            ("policy_capital_efficiency", RuntimeArtifactTSV.double(snapshot.policyCapitalEfficiency)),
            ("policy_lifecycle_action", String(snapshot.policyLifecycleAction.rawValue)),
            ("gross_exposure_lots", RuntimeArtifactTSV.double(snapshot.grossExposureLots)),
            ("correlated_exposure_lots", RuntimeArtifactTSV.double(snapshot.correlatedExposureLots)),
            ("directional_cluster_lots", RuntimeArtifactTSV.double(snapshot.directionalClusterLots)),
            ("capital_risk_pct", RuntimeArtifactTSV.double(snapshot.capitalRiskPct)),
            ("portfolio_pressure", RuntimeArtifactTSV.double(snapshot.portfolioPressure))
        ]
        return rows
            .map { "\(RuntimeArtifactTSV.field($0.0))\t\(RuntimeArtifactTSV.field($0.1))" }
            .joined(separator: "\r\n") + "\r\n"
    }

    public static func adaptiveRouterPluginCSVs(
        routes: [AdaptiveRouterPluginRoute]
    ) -> (active: String, downweighted: String, suppressed: String) {
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

    public static func dynamicEnsemblePluginCSVs(
        records: [DynamicEnsemblePluginRecord]
    ) -> (active: String, downweighted: String, suppressed: String) {
        var active: [String] = []
        var downweighted: [String] = []
        var suppressed: [String] = []
        for record in records where record.ready {
            let token = "\(record.aiName):\(RuntimeArtifactTSV.double(record.normalizedWeight, decimals: 4)):\(RuntimeArtifactTSV.double(record.trustScore, decimals: 4))"
            switch record.status {
            case .suppressed:
                suppressed.append(token)
            case .downweighted:
                downweighted.append(token)
            case .active:
                active.append(token)
            case .excluded:
                continue
            }
        }
        return (
            active.joined(separator: "|"),
            downweighted.joined(separator: "|"),
            suppressed.joined(separator: "|")
        )
    }
}

public extension RuntimeArtifactFileRepository {
    @discardableResult
    func writeControlPlaneLocalSnapshot(_ snapshot: ControlPlaneSnapshot) throws -> URL {
        let fileURL = url(
            for: ControlPlanePaths.snapshotFile(
                symbol: snapshot.symbol,
                login: snapshot.login,
                magic: snapshot.magic,
                chartID: snapshot.chartID
            )
        )
        try fileManager.createDirectory(
            at: fileURL.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        try RuntimeSignalStateTools.controlPlaneSnapshotTSV(snapshot)
            .write(to: fileURL, atomically: true, encoding: .utf8)
        return fileURL
    }
}
