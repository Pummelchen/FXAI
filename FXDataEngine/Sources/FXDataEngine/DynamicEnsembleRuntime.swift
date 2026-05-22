import Foundation

public enum DynamicEnsembleRuntimeConstants {
    public static let maxPluginReasons = 4
    public static let maxReasons = 8
    public static let runtimeDirectory = "FXAI/Runtime"
    public static let configPath = "FXAI/Runtime/dynamic_ensemble_config.tsv"
}

public enum DynamicEnsembleStatus: Int, Codable, Sendable, CaseIterable {
    case excluded = 0
    case suppressed
    case downweighted
    case active

    public var label: String {
        switch self {
        case .excluded: "EXCLUDED"
        case .suppressed: "SUPPRESSED"
        case .downweighted: "DOWNWEIGHTED"
        case .active: "ACTIVE"
        }
    }
}

public enum DynamicEnsembleRuntimeTools {
    public static func runtimeStatePath(symbol: String) -> String {
        "\(DynamicEnsembleRuntimeConstants.runtimeDirectory)/fxai_dynamic_ensemble_\(ControlPlanePaths.safeToken(symbol)).tsv"
    }

    public static func runtimeHistoryPath(symbol: String) -> String {
        "\(DynamicEnsembleRuntimeConstants.runtimeDirectory)/fxai_dynamic_ensemble_history_\(ControlPlanePaths.safeToken(symbol)).ndjson"
    }

    public static func actionLabel(_ action: Int) -> String {
        if action == 1 {
            return "BUY"
        }
        if action == 0 {
            return "SELL"
        }
        return "SKIP"
    }

    public static func actionCode(_ label: String) -> Int {
        switch label.uppercased() {
        case "BUY": 1
        case "SELL": 0
        default: -1
        }
    }

    public static func familySlot(_ familyName: String) -> AIFamily {
        let clean = familyName.trimmingCharacters(in: .whitespacesAndNewlines)
        return AIFamily.allCases.first { $0.controlPlaneName == clean } ?? .other
    }

    public static func centerDirection(records: [DynamicEnsemblePluginRecord]) -> Double {
        var numerator = 0.0
        var denominator = 0.0
        for record in records where record.ready && record.baseMetaWeight > 0.0 {
            let direction = fxClamp(record.buyProbability - record.sellProbability, -1.0, 1.0)
            numerator += record.baseMetaWeight * direction
            denominator += record.baseMetaWeight
        }
        guard denominator > 0.0 else { return 0.0 }
        return fxClamp(numerator / denominator, -1.0, 1.0)
    }

    public static func riskStress(
        newsState: NewsPulsePairState,
        ratesState: RatesEnginePairState,
        crossAssetState: CrossAssetPairState,
        microstructureState: MicrostructurePairState,
        driftNorm: Double
    ) -> Double {
        let newsStress: Double
        if !newsState.ready || newsState.stale {
            newsStress = 0.50
        } else {
            let gateStress = newsState.tradeGate == "BLOCK" ? 0.95 : (newsState.tradeGate == "CAUTION" ? 0.64 : 0.0)
            newsStress = fxClamp(max(newsState.newsRiskScore, gateStress), 0.0, 1.0)
        }

        let ratesStress: Double
        if !ratesState.ready || ratesState.stale {
            ratesStress = 0.40
        } else {
            let gateStress = ratesState.tradeGate == "BLOCK" ? 0.90 : (ratesState.tradeGate == "CAUTION" ? 0.60 : 0.0)
            ratesStress = fxClamp(max(ratesState.ratesRiskScore, gateStress), 0.0, 1.0)
        }

        let microStress: Double
        if !microstructureState.ready || microstructureState.stale {
            microStress = 0.50
        } else {
            let gateStress = microstructureState.tradeGate == "BLOCK" ? 0.96 : (microstructureState.tradeGate == "CAUTION" ? 0.64 : 0.0)
            microStress = fxClamp(
                max(microstructureState.hostileExecutionScore, max(microstructureState.liquidityStressScore, gateStress)),
                0.0,
                1.0
            )
        }

        let crossStress: Double
        if !crossAssetState.ready || crossAssetState.stale {
            crossStress = 0.36
        } else {
            let gateStress = crossAssetState.tradeGate == "BLOCK" ? 0.92 : (crossAssetState.tradeGate == "CAUTION" ? 0.62 : 0.0)
            crossStress = fxClamp(
                max(crossAssetState.pairCrossAssetRiskScore, max(crossAssetState.usdLiquidityStressScore, gateStress)),
                0.0,
                1.0
            )
        }

        return fxClamp(
            0.30 * newsStress +
                0.18 * ratesStress +
                0.22 * crossStress +
                0.18 * microStress +
                0.12 * fxClamp(driftNorm, 0.0, 1.0),
            0.0,
            1.0
        )
    }

    public static func evaluate(inputs: DynamicEnsembleInputs) -> DynamicEnsembleEvaluation {
        var state = DynamicEnsembleRuntimeState.reset
        var records = inputs.records
        let config = inputs.config
        guard inputs.dynamicEnsembleEnabled,
              config.ready,
              config.enabled,
              !records.isEmpty else {
            state.fallbackUsed = true
            return DynamicEnsembleEvaluation(applied: false, state: state, records: records)
        }

        state.ready = true
        state.generatedAt = max(0, inputs.generatedAt)
        state.symbol = inputs.symbol
        state.topRegime = inputs.regimeState.topLabel
        state.sessionLabel = inputs.regimeState.sessionLabel
        state.skipSupport = 0.0

        let center = centerDirection(records: records)
        var qualityContextSum = 0.0
        var qualityTrustSum = 0.0
        var qualityContextDenominator = 0.0
        var dominantShare = 0.0

        for index in records.indices {
            let record = records[index]
            if !record.ready || record.baseMetaWeight <= 0.0 {
                records[index].status = .excluded
                records[index].normalizedWeight = 0.0
                continue
            }

            if record.adaptiveStatus == .suppressed {
                records[index].status = .suppressed
                records[index].trustScore = 0.0
                records[index].normalizedWeight = 0.0
                records[index].appendReason("suppressed_by_adaptive_router")
                continue
            }

            let family = record.family
            var priorMultiplier = 1.0
            if record.adaptiveStatus == .upweighted {
                priorMultiplier += config.weightAdaptiveUpweightGain
                records[index].appendReason("adaptive_router_upweighted")
            } else if record.adaptiveStatus == .downweighted {
                priorMultiplier *= max(0.35, 1.0 - config.weightAdaptiveDownweightPenalty)
                records[index].appendReason("adaptive_router_downweighted")
            }

            let portfolioBlend = fxClamp(
                0.45 * record.portfolioStability +
                    0.35 * record.portfolioDiversity +
                    0.20 * max(record.portfolioEdgeNorm, 0.0),
                0.0,
                1.0
            )
            let empiricalMultiplier = fxClamp(
                0.48 +
                    config.weightReliabilityGain * fxClamp(record.reliability, 0.0, 1.0) +
                    config.weightContextEdgeGain * max(record.contextEdgeNorm, 0.0) +
                    config.weightGlobalEdgeGain * max(record.globalEdgeNorm, 0.0) +
                    config.weightPortfolioGain * portfolioBlend +
                    config.weightContextTrustGain * fxClamp(record.contextTrust, 0.0, 1.0) -
                    0.24 * fxClamp(record.contextRegret, 0.0, 1.0) -
                    0.12 * fxClamp(record.portfolioCorrelation, 0.0, 1.0),
                0.20,
                1.80
            )

            let confidenceGap = max(record.confidence - record.reliability, 0.0)
            let confidenceCap = fxClamp(config.familyConfidenceCap[family], 0.55, 1.00)
            let calibrationMultiplier = fxClamp(
                confidenceCap -
                    config.penaltyConfidenceGap * confidenceGap -
                    config.penaltyContextRegret * fxClamp(record.contextRegret, 0.0, 1.0),
                0.30,
                1.00
            )
            records[index].calibrationShrink = calibrationMultiplier
            if calibrationMultiplier < 0.70 {
                records[index].appendReason("confidence_shrunk")
            }

            let directionScore = fxClamp(record.buyProbability - record.sellProbability, -1.0, 1.0)
            let disagreement = abs(directionScore - center)
            let stabilityMultiplier = fxClamp(
                1.0 -
                    config.penaltyDisagreement * disagreement / max(config.familyDisagreementTolerance[family], 0.20) -
                    config.penaltyDrift * fxClamp(inputs.driftNorm, 0.0, 1.0),
                0.35,
                1.10
            )
            if disagreement >= 0.45 {
                records[index].appendReason("directional_disagreement")
            }

            let costRatio = fxClamp(inputs.priceCostPoints / max(inputs.minMovePoints, 0.50), 0.0, 2.5) / 2.5
            let familyNews = fxClamp(config.familyNewsCompatibility[family], 0.40, 1.30)
            let familyRates = fxClamp(config.familyRatesCompatibility[family], 0.40, 1.30)
            let familyMicro = fxClamp(config.familyMicroCompatibility[family], 0.40, 1.30)
            let familyCost = fxClamp(config.familyCostRobustness[family], 0.40, 1.30)
            var riskMultiplier = 1.0

            applyRiskContext(
                config: config,
                newsState: inputs.newsState,
                ratesState: inputs.ratesState,
                crossAssetState: inputs.crossAssetState,
                microstructureState: inputs.microstructureState,
                familyNews: familyNews,
                familyRates: familyRates,
                familyMicro: familyMicro,
                record: &records[index],
                riskMultiplier: &riskMultiplier
            )
            riskMultiplier -= config.penaltyCost * costRatio * fxClamp(1.15 - familyCost, 0.25, 1.20)
            riskMultiplier = fxClamp(riskMultiplier, 0.10, 1.20)

            let trust = fxClamp(
                record.baseMetaWeight * priorMultiplier * empiricalMultiplier * calibrationMultiplier * stabilityMultiplier * riskMultiplier,
                0.0,
                4.0
            )
            records[index].trustScore = trust

            if trust < config.suppressTrustThreshold {
                records[index].status = .suppressed
                records[index].normalizedWeight = 0.0
                records[index].appendReason("trust_below_suppress_threshold")
            } else if trust < config.downweightTrustThreshold {
                records[index].status = .downweighted
                records[index].appendReason("trust_below_active_threshold")
            } else {
                records[index].status = .active
            }
        }

        normalizeWeights(records: &records, maxShare: config.maxWeightShare)
        var weightedCandidates = records.filter { $0.status.rawValue >= DynamicEnsembleStatus.downweighted.rawValue && $0.normalizedWeight > 0.0 }.count
        var activeSum = 0.0
        for index in records.indices where records[index].normalizedWeight > 0.0 {
            if records[index].normalizedWeight < config.minEffectiveWeight,
               weightedCandidates > config.minActivePlugins {
                records[index].status = .suppressed
                records[index].normalizedWeight = 0.0
                weightedCandidates -= 1
                records[index].appendReason("weight_below_min_effective_weight")
                continue
            }
            activeSum += records[index].normalizedWeight
        }

        guard activeSum > 0.0 else {
            state.ready = false
            state.fallbackUsed = true
            return DynamicEnsembleEvaluation(applied: false, state: state, records: records)
        }

        var buyProbability = 0.0
        var sellProbability = 0.0
        var skipProbability = 0.0
        var agreementNumerator = 0.0
        var agreementDenominator = 0.0

        for index in records.indices where records[index].normalizedWeight > 0.0 {
            records[index].normalizedWeight /= activeSum
            dominantShare = max(dominantShare, records[index].normalizedWeight)
            if records[index].status == .active {
                state.participatingCount += 1
            } else if records[index].status == .downweighted {
                state.downweightedCount += 1
            }

            let direction = fxClamp(records[index].buyProbability - records[index].sellProbability, -1.0, 1.0)
            agreementNumerator += records[index].normalizedWeight * direction
            agreementDenominator += records[index].normalizedWeight * abs(direction)
            buyProbability += records[index].normalizedWeight * fxClamp(records[index].buyProbability * records[index].calibrationShrink, 0.0, 1.0)
            sellProbability += records[index].normalizedWeight * fxClamp(records[index].sellProbability * records[index].calibrationShrink, 0.0, 1.0)
            skipProbability += records[index].normalizedWeight * fxClamp(records[index].skipProbability + (1.0 - records[index].calibrationShrink) * 0.35, 0.0, 1.0)

            qualityContextSum += records[index].normalizedWeight * fxClamp(
                0.40 * max(records[index].contextEdgeNorm, 0.0) +
                    0.18 * max(records[index].globalEdgeNorm, 0.0) +
                    0.22 * fxClamp(records[index].contextTrust, 0.0, 1.0) +
                    0.10 * fxClamp(records[index].portfolioStability, 0.0, 1.0) +
                    0.10 * fxClamp(records[index].portfolioDiversity, 0.0, 1.0),
                0.0,
                1.0
            )
            qualityTrustSum += records[index].normalizedWeight * fxClamp(records[index].trustScore, 0.0, 1.20)
            qualityContextDenominator += records[index].normalizedWeight

            if records[index].signal == 1 {
                state.buySupport += records[index].normalizedWeight
            } else if records[index].signal == 0 {
                state.sellSupport += records[index].normalizedWeight
            } else {
                state.skipSupport += records[index].normalizedWeight
            }
        }

        state.suppressedCount = records.filter { $0.status == .suppressed }.count
        var probabilitySum = buyProbability + sellProbability + skipProbability
        if probabilitySum <= 0.0 {
            probabilitySum = 1.0
        }
        state.buyProbability = fxClamp(buyProbability / probabilitySum, 0.0, 1.0)
        state.sellProbability = fxClamp(sellProbability / probabilitySum, 0.0, 1.0)
        state.skipProbability = fxClamp(skipProbability / probabilitySum, 0.0, 1.0)
        state.finalScore = fxClamp(state.buyProbability - state.sellProbability, -1.0, 1.0)
        state.agreementScore = fxClamp(abs(agreementNumerator) / max(agreementDenominator, 1e-6), 0.0, 1.0)
        state.contextFitScore = fxClamp(qualityContextSum / max(qualityContextDenominator, 1e-6), 0.0, 1.0)
        state.dominantPluginShare = dominantShare

        let effectiveParticipants = state.participatingCount + state.downweightedCount
        let trustStrength = fxClamp(qualityTrustSum / max(qualityContextDenominator, 1e-6), 0.0, 1.20) / 1.20
        let stress = riskStress(
            newsState: inputs.newsState,
            ratesState: inputs.ratesState,
            crossAssetState: inputs.crossAssetState,
            microstructureState: inputs.microstructureState,
            driftNorm: inputs.driftNorm
        )
        let executionSafety = fxClamp(1.0 - stress, 0.0, 1.0)
        let concentrationPenalty = config.penaltyConcentrationQuality *
            fxClamp((dominantShare - config.maxWeightShare) / max(1.0 - config.maxWeightShare, 0.10), 0.0, 1.0)
        let singlePenalty = effectiveParticipants <= 1 ? config.penaltySinglePluginQuality : 0.0
        state.ensembleQuality = fxClamp(
            0.34 * state.agreementScore +
                0.26 * trustStrength +
                0.18 * state.contextFitScore +
                0.22 * executionSafety -
                concentrationPenalty -
                singlePenalty,
            0.0,
            1.0
        )

        if effectiveParticipants <= 0 || state.ensembleQuality <= config.blockQualityThreshold {
            state.tradePosture = "BLOCK"
        } else if state.ensembleQuality <= config.abstainQualityThreshold {
            state.tradePosture = "ABSTAIN_BIAS"
        } else if state.ensembleQuality <= config.cautionQualityThreshold || stress >= 0.56 || state.agreementScore <= 0.20 {
            state.tradePosture = "CAUTION"
        } else {
            state.tradePosture = "NORMAL"
        }

        switch state.tradePosture {
        case "BLOCK":
            state.abstainBias = 0.92
        case "ABSTAIN_BIAS":
            state.abstainBias = 0.34
        case "CAUTION":
            state.abstainBias = 0.14
        default:
            state.abstainBias = 0.04
        }
        if !inputs.newsState.ready || inputs.newsState.stale ||
            !inputs.ratesState.ready || inputs.ratesState.stale ||
            !inputs.microstructureState.ready || inputs.microstructureState.stale {
            state.abstainBias = fxClamp(state.abstainBias + 0.10, 0.0, 0.98)
        }
        if state.agreementScore <= 0.20 {
            state.abstainBias = fxClamp(state.abstainBias + 0.08, 0.0, 0.98)
        }

        appendRuntimeReasons(
            state: &state,
            newsState: inputs.newsState,
            ratesState: inputs.ratesState,
            microstructureState: inputs.microstructureState
        )

        if state.buyProbability >= state.sellProbability && state.buyProbability > state.skipProbability {
            state.finalAction = 1
        } else if state.sellProbability > state.buyProbability && state.sellProbability > state.skipProbability {
            state.finalAction = 0
        } else {
            state.finalAction = -1
        }

        return DynamicEnsembleEvaluation(applied: true, state: state, records: records)
    }

    public static func runtimeStateTSV(
        symbol: String,
        state: DynamicEnsembleRuntimeState,
        records: [DynamicEnsemblePluginRecord] = [],
        finalDecision: Int? = nil
    ) -> String? {
        guard state.ready, !symbol.isEmpty else { return nil }
        let finalState = state.withFinalAction(finalDecision ?? state.finalAction)
        let csv = pluginCSVs(records: records)
        return runtimeRows(
            symbol: symbol,
            state: finalState,
            activeCSV: csv.active,
            downweightedCSV: csv.downweighted,
            suppressedCSV: csv.suppressed
        )
        .map { key, value in "\(RuntimeArtifactTSV.field(key))\t\(RuntimeArtifactTSV.field(value))" }
        .joined(separator: "\r\n") + "\r\n"
    }

    public static func runtimeHistoryNDJSONLine(
        symbol: String,
        state: DynamicEnsembleRuntimeState,
        records: [DynamicEnsemblePluginRecord] = [],
        finalDecision: Int? = nil
    ) -> String? {
        guard state.ready, !symbol.isEmpty else { return nil }
        let finalState = state.withFinalAction(finalDecision ?? state.finalAction)
        let reasons = finalState.reasons.map(jsonQuoted).joined(separator: ",")
        return "{" +
            "\"schema_version\":1," +
            "\"generated_at\":\(jsonQuoted(iso8601UTC(finalState.generatedAt)))," +
            "\"symbol\":\(jsonQuoted(symbol))," +
            "\"ensemble\":{" +
            "\"top_regime\":\(jsonQuoted(finalState.topRegime))," +
            "\"session_label\":\(jsonQuoted(finalState.sessionLabel))," +
            "\"trade_posture\":\(jsonQuoted(finalState.tradePosture))," +
            "\"ensemble_quality\":\(RuntimeArtifactTSV.double(finalState.ensembleQuality))," +
            "\"abstain_bias\":\(RuntimeArtifactTSV.double(finalState.abstainBias))," +
            "\"agreement_score\":\(RuntimeArtifactTSV.double(finalState.agreementScore))," +
            "\"context_fit_score\":\(RuntimeArtifactTSV.double(finalState.contextFitScore))," +
            "\"dominant_plugin_share\":\(RuntimeArtifactTSV.double(finalState.dominantPluginShare))," +
            "\"buy_prob\":\(RuntimeArtifactTSV.double(finalState.buyProbability))," +
            "\"sell_prob\":\(RuntimeArtifactTSV.double(finalState.sellProbability))," +
            "\"skip_prob\":\(RuntimeArtifactTSV.double(finalState.skipProbability))," +
            "\"buy_support\":\(RuntimeArtifactTSV.double(finalState.buySupport))," +
            "\"sell_support\":\(RuntimeArtifactTSV.double(finalState.sellSupport))," +
            "\"skip_support\":\(RuntimeArtifactTSV.double(finalState.skipSupport))," +
            "\"final_score\":\(RuntimeArtifactTSV.double(finalState.finalScore))," +
            "\"final_action\":\(jsonQuoted(actionLabel(finalState.finalAction)))," +
            "\"reasons\":[\(reasons)]" +
            "}," +
            "\"plugins\":[\(pluginJSON(records: records))]" +
            "}"
    }

    public static func readRuntimeState(symbol _: String, stateTSV: String?) -> DynamicEnsembleRuntimeState? {
        guard let stateTSV else { return nil }
        let state = parseRuntimeState(tsv: stateTSV)
        return state.ready ? state : nil
    }

    public static func parseRuntimeState(tsv: String) -> DynamicEnsembleRuntimeState {
        var state = DynamicEnsembleRuntimeState.reset
        for line in tsv.components(separatedBy: .newlines) where !line.isEmpty {
            let parts = line.split(separator: "\t", omittingEmptySubsequences: false)
            guard parts.count >= 2 else { continue }
            state.ready = true
            let key = String(parts[0])
            let value = String(parts[1])
            switch key {
            case "symbol": state.symbol = value
            case "generated_at": state.generatedAt = Int64(value) ?? 0
            case "top_regime": state.topRegime = value
            case "session_label": state.sessionLabel = value
            case "trade_posture": state.tradePosture = value
            case "ensemble_quality": state.ensembleQuality = Double(value) ?? 0.0
            case "abstain_bias": state.abstainBias = Double(value) ?? 0.0
            case "agreement_score": state.agreementScore = Double(value) ?? 0.0
            case "context_fit_score": state.contextFitScore = Double(value) ?? 0.0
            case "dominant_plugin_share": state.dominantPluginShare = Double(value) ?? 0.0
            case "participating_count": state.participatingCount = Int(value) ?? 0
            case "downweighted_count": state.downweightedCount = Int(value) ?? 0
            case "suppressed_count": state.suppressedCount = Int(value) ?? 0
            case "buy_support": state.buySupport = Double(value) ?? 0.0
            case "sell_support": state.sellSupport = Double(value) ?? 0.0
            case "skip_support": state.skipSupport = Double(value) ?? 0.0
            case "buy_prob": state.buyProbability = Double(value) ?? 0.0
            case "sell_prob": state.sellProbability = Double(value) ?? 0.0
            case "skip_prob": state.skipProbability = Double(value) ?? 1.0
            case "final_score": state.finalScore = Double(value) ?? 0.0
            case "final_action": state.finalAction = actionCode(value)
            case "fallback_used": state.fallbackUsed = (Int(value) ?? 0) != 0
            case "reasons_csv":
                for reason in value.split(separator: ";", omittingEmptySubsequences: false) {
                    state.appendReason(String(reason))
                }
            default:
                break
            }
        }
        return state
    }

    private static func applyRiskContext(
        config: DynamicEnsembleConfig,
        newsState: NewsPulsePairState,
        ratesState: RatesEnginePairState,
        crossAssetState: CrossAssetPairState,
        microstructureState: MicrostructurePairState,
        familyNews: Double,
        familyRates: Double,
        familyMicro: Double,
        record: inout DynamicEnsemblePluginRecord,
        riskMultiplier: inout Double
    ) {
        if !newsState.ready || newsState.stale {
            riskMultiplier -= config.penaltyStaleContext * 0.45
            record.appendReason("newspulse_stale")
        } else if newsState.tradeGate == "BLOCK" || newsState.tradeGate == "CAUTION" {
            riskMultiplier -= config.penaltyNews *
                fxClamp(max(newsState.newsRiskScore, 0.40), 0.0, 1.0) *
                fxClamp(1.20 - familyNews, 0.25, 1.20)
            record.appendReason(newsState.tradeGate == "BLOCK" ? "newspulse_block_context" : "newspulse_caution_context")
        }

        if !ratesState.ready || ratesState.stale {
            riskMultiplier -= config.penaltyStaleContext * 0.30
            record.appendReason("rates_state_stale")
        } else if ratesState.tradeGate == "BLOCK" || ratesState.tradeGate == "CAUTION" {
            riskMultiplier -= config.penaltyRates *
                fxClamp(max(ratesState.ratesRiskScore, 0.35), 0.0, 1.0) *
                fxClamp(1.20 - familyRates, 0.25, 1.20)
            record.appendReason(ratesState.tradeGate == "BLOCK" ? "rates_block_context" : "rates_caution_context")
        }

        if !crossAssetState.ready || crossAssetState.stale {
            riskMultiplier -= config.penaltyStaleContext * 0.28
            record.appendReason("cross_asset_stale")
        } else if crossAssetState.tradeGate == "BLOCK" || crossAssetState.tradeGate == "CAUTION" {
            riskMultiplier -= (config.penaltyRates * 0.85) *
                fxClamp(max(crossAssetState.pairCrossAssetRiskScore, 0.35), 0.0, 1.0)
            record.appendReason(crossAssetState.tradeGate == "BLOCK" ? "cross_asset_block_context" : "cross_asset_caution_context")
        }

        if !microstructureState.ready || microstructureState.stale {
            riskMultiplier -= config.penaltyStaleContext * 0.42
            record.appendReason("microstructure_stale")
        } else if microstructureState.tradeGate == "BLOCK" || microstructureState.tradeGate == "CAUTION" {
            let microStress = max(microstructureState.hostileExecutionScore, microstructureState.liquidityStressScore)
            riskMultiplier -= config.penaltyMicro *
                fxClamp(max(microStress, 0.35), 0.0, 1.0) *
                fxClamp(1.20 - familyMicro, 0.25, 1.20)
            record.appendReason(microstructureState.tradeGate == "BLOCK" ? "microstructure_block_context" : "microstructure_caution_context")
        }
    }

    private static func normalizeWeights(records: inout [DynamicEnsemblePluginRecord], maxShare: Double) {
        let cappedMaxShare = fxClamp(maxShare, 0.05, 1.0)
        let total = records
            .filter { $0.status.rawValue >= DynamicEnsembleStatus.downweighted.rawValue && $0.trustScore > 0.0 }
            .reduce(0.0) { $0 + $1.trustScore }
        guard total > 0.0 else { return }

        for index in records.indices {
            if records[index].status.rawValue >= DynamicEnsembleStatus.downweighted.rawValue && records[index].trustScore > 0.0 {
                records[index].normalizedWeight = records[index].trustScore / total
            } else {
                records[index].normalizedWeight = 0.0
            }
        }

        var cappedSum = 0.0
        var uncappedTotal = 0.0
        var needsRedistribution = false
        for index in records.indices where records[index].normalizedWeight > 0.0 {
            if records[index].normalizedWeight > cappedMaxShare {
                records[index].normalizedWeight = cappedMaxShare
                needsRedistribution = true
            } else {
                uncappedTotal += records[index].trustScore
            }
            cappedSum += records[index].normalizedWeight
        }
        guard needsRedistribution else { return }

        let residual = 1.0 - cappedSum
        guard residual > 0.0, uncappedTotal > 0.0 else { return }
        for index in records.indices {
            if records[index].normalizedWeight <= 0.0 || records[index].normalizedWeight >= cappedMaxShare - 1e-9 {
                continue
            }
            records[index].normalizedWeight = residual * records[index].trustScore / uncappedTotal
        }
    }

    private static func appendRuntimeReasons(
        state: inout DynamicEnsembleRuntimeState,
        newsState: NewsPulsePairState,
        ratesState: RatesEnginePairState,
        microstructureState: MicrostructurePairState
    ) {
        if state.tradePosture == "BLOCK" {
            state.appendReason("ensemble_quality_below_block_floor")
        } else if state.tradePosture == "ABSTAIN_BIAS" {
            state.appendReason("ensemble_quality_below_abstain_floor")
        } else if state.tradePosture == "CAUTION" {
            state.appendReason("ensemble_quality_caution")
        }
        if state.agreementScore >= 0.66 {
            state.appendReason("strong_plugin_agreement")
        } else if state.agreementScore <= 0.24 {
            state.appendReason("plugin_disagreement_elevated")
        }
        if state.dominantPluginShare >= 0.58 {
            state.appendReason("plugin_concentration_elevated")
        }
        if newsState.ready && !newsState.stale && newsState.tradeGate == "CAUTION" {
            state.appendReason("newspulse_caution_active")
        }
        if ratesState.ready && !ratesState.stale && ratesState.tradeGate == "CAUTION" {
            state.appendReason("rates_caution_active")
        }
        if microstructureState.ready && !microstructureState.stale && microstructureState.tradeGate == "CAUTION" {
            state.appendReason("microstructure_caution_active")
        }
        if !newsState.ready || newsState.stale || !ratesState.ready || ratesState.stale || !microstructureState.ready || microstructureState.stale {
            state.appendReason("context_state_stale")
        }
    }

    private static func pluginCSVs(records: [DynamicEnsemblePluginRecord]) -> (active: String, downweighted: String, suppressed: String) {
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

    private static func pluginJSON(records: [DynamicEnsemblePluginRecord]) -> String {
        records.filter(\.ready).map { record in
            "{" +
                "\"name\":\(jsonQuoted(record.aiName))," +
                "\"family\":\(jsonQuoted(record.family.controlPlaneName))," +
                "\"status\":\(jsonQuoted(record.status.label))," +
                "\"signal\":\(jsonQuoted(actionLabel(record.signal)))," +
                "\"weight\":\(RuntimeArtifactTSV.double(record.normalizedWeight))," +
                "\"trust\":\(RuntimeArtifactTSV.double(record.trustScore))," +
                "\"calibration_shrink\":\(RuntimeArtifactTSV.double(record.calibrationShrink))," +
                "\"reasons\":[\(record.reasons.map(jsonQuoted).joined(separator: ","))]" +
                "}"
        }
        .joined(separator: ",")
    }

    private static func runtimeRows(
        symbol: String,
        state: DynamicEnsembleRuntimeState,
        activeCSV: String,
        downweightedCSV: String,
        suppressedCSV: String
    ) -> [(String, String)] {
        [
            ("schema_version", "1"),
            ("symbol", symbol),
            ("generated_at", "\(state.generatedAt)"),
            ("top_regime", state.topRegime),
            ("session_label", state.sessionLabel),
            ("trade_posture", state.tradePosture),
            ("ensemble_quality", RuntimeArtifactTSV.double(state.ensembleQuality)),
            ("abstain_bias", RuntimeArtifactTSV.double(state.abstainBias)),
            ("agreement_score", RuntimeArtifactTSV.double(state.agreementScore)),
            ("context_fit_score", RuntimeArtifactTSV.double(state.contextFitScore)),
            ("dominant_plugin_share", RuntimeArtifactTSV.double(state.dominantPluginShare)),
            ("participating_count", "\(state.participatingCount)"),
            ("downweighted_count", "\(state.downweightedCount)"),
            ("suppressed_count", "\(state.suppressedCount)"),
            ("buy_support", RuntimeArtifactTSV.double(state.buySupport)),
            ("sell_support", RuntimeArtifactTSV.double(state.sellSupport)),
            ("skip_support", RuntimeArtifactTSV.double(state.skipSupport)),
            ("buy_prob", RuntimeArtifactTSV.double(state.buyProbability)),
            ("sell_prob", RuntimeArtifactTSV.double(state.sellProbability)),
            ("skip_prob", RuntimeArtifactTSV.double(state.skipProbability)),
            ("final_score", RuntimeArtifactTSV.double(state.finalScore)),
            ("final_action", actionLabel(state.finalAction)),
            ("fallback_used", RuntimeArtifactTSV.bool(state.fallbackUsed)),
            ("reasons_csv", state.reasonsCSV),
            ("active_plugins_csv", activeCSV),
            ("downweighted_plugins_csv", downweightedCSV),
            ("suppressed_plugins_csv", suppressedCSV)
        ]
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

public struct DynamicEnsembleConfig: Codable, Hashable, Sendable {
    public var ready: Bool
    public var enabled: Bool
    public var fallbackToRoutedEnsemble: Bool
    public var suppressTrustThreshold: Double
    public var downweightTrustThreshold: Double
    public var cautionQualityThreshold: Double
    public var abstainQualityThreshold: Double
    public var blockQualityThreshold: Double
    public var minEffectiveWeight: Double
    public var maxWeightShare: Double
    public var minActivePlugins: Int
    public var penaltyConfidenceGap: Double
    public var penaltyContextRegret: Double
    public var penaltyDisagreement: Double
    public var penaltyDrift: Double
    public var penaltyCost: Double
    public var penaltyNews: Double
    public var penaltyRates: Double
    public var penaltyMicro: Double
    public var penaltyStaleContext: Double
    public var penaltySinglePluginQuality: Double
    public var penaltyConcentrationQuality: Double
    public var weightReliabilityGain: Double
    public var weightContextEdgeGain: Double
    public var weightGlobalEdgeGain: Double
    public var weightPortfolioGain: Double
    public var weightContextTrustGain: Double
    public var weightAdaptiveUpweightGain: Double
    public var weightAdaptiveDownweightPenalty: Double
    public var familyNewsCompatibility: [Double]
    public var familyRatesCompatibility: [Double]
    public var familyMicroCompatibility: [Double]
    public var familyCostRobustness: [Double]
    public var familyConfidenceCap: [Double]
    public var familyDisagreementTolerance: [Double]

    public init(
        ready: Bool = true,
        enabled: Bool = true,
        fallbackToRoutedEnsemble: Bool = true,
        suppressTrustThreshold: Double = 0.30,
        downweightTrustThreshold: Double = 0.72,
        cautionQualityThreshold: Double = 0.56,
        abstainQualityThreshold: Double = 0.36,
        blockQualityThreshold: Double = 0.18,
        minEffectiveWeight: Double = 0.04,
        maxWeightShare: Double = 0.66,
        minActivePlugins: Int = 1,
        penaltyConfidenceGap: Double = 0.52,
        penaltyContextRegret: Double = 0.38,
        penaltyDisagreement: Double = 0.28,
        penaltyDrift: Double = 0.18,
        penaltyCost: Double = 0.28,
        penaltyNews: Double = 0.24,
        penaltyRates: Double = 0.20,
        penaltyMicro: Double = 0.30,
        penaltyStaleContext: Double = 0.24,
        penaltySinglePluginQuality: Double = 0.16,
        penaltyConcentrationQuality: Double = 0.22,
        weightReliabilityGain: Double = 0.34,
        weightContextEdgeGain: Double = 0.18,
        weightGlobalEdgeGain: Double = 0.10,
        weightPortfolioGain: Double = 0.16,
        weightContextTrustGain: Double = 0.18,
        weightAdaptiveUpweightGain: Double = 0.05,
        weightAdaptiveDownweightPenalty: Double = 0.14,
        familyNewsCompatibility: [Double]? = nil,
        familyRatesCompatibility: [Double]? = nil,
        familyMicroCompatibility: [Double]? = nil,
        familyCostRobustness: [Double]? = nil,
        familyConfidenceCap: [Double]? = nil,
        familyDisagreementTolerance: [Double]? = nil
    ) {
        self.ready = ready
        self.enabled = enabled
        self.fallbackToRoutedEnsemble = fallbackToRoutedEnsemble
        self.suppressTrustThreshold = suppressTrustThreshold
        self.downweightTrustThreshold = downweightTrustThreshold
        self.cautionQualityThreshold = cautionQualityThreshold
        self.abstainQualityThreshold = abstainQualityThreshold
        self.blockQualityThreshold = blockQualityThreshold
        self.minEffectiveWeight = minEffectiveWeight
        self.maxWeightShare = maxWeightShare
        self.minActivePlugins = max(0, minActivePlugins)
        self.penaltyConfidenceGap = penaltyConfidenceGap
        self.penaltyContextRegret = penaltyContextRegret
        self.penaltyDisagreement = penaltyDisagreement
        self.penaltyDrift = penaltyDrift
        self.penaltyCost = penaltyCost
        self.penaltyNews = penaltyNews
        self.penaltyRates = penaltyRates
        self.penaltyMicro = penaltyMicro
        self.penaltyStaleContext = penaltyStaleContext
        self.penaltySinglePluginQuality = penaltySinglePluginQuality
        self.penaltyConcentrationQuality = penaltyConcentrationQuality
        self.weightReliabilityGain = weightReliabilityGain
        self.weightContextEdgeGain = weightContextEdgeGain
        self.weightGlobalEdgeGain = weightGlobalEdgeGain
        self.weightPortfolioGain = weightPortfolioGain
        self.weightContextTrustGain = weightContextTrustGain
        self.weightAdaptiveUpweightGain = weightAdaptiveUpweightGain
        self.weightAdaptiveDownweightPenalty = weightAdaptiveDownweightPenalty
        self.familyNewsCompatibility = Self.normalizedFamilyValues(familyNewsCompatibility ?? Self.defaultFamilyNewsCompatibility())
        self.familyRatesCompatibility = Self.normalizedFamilyValues(familyRatesCompatibility ?? Self.defaultFamilyRatesCompatibility())
        self.familyMicroCompatibility = Self.normalizedFamilyValues(familyMicroCompatibility ?? Self.defaultFamilyMicroCompatibility())
        self.familyCostRobustness = Self.normalizedFamilyValues(familyCostRobustness ?? Self.defaultFamilyCostRobustness())
        self.familyConfidenceCap = Self.normalizedFamilyValues(familyConfidenceCap ?? Self.defaultFamilyConfidenceCap())
        self.familyDisagreementTolerance = Self.normalizedFamilyValues(familyDisagreementTolerance ?? Self.defaultFamilyDisagreementTolerance())
    }

    public static var defaults: DynamicEnsembleConfig {
        DynamicEnsembleConfig()
    }

    public static func parse(tsv: String) -> DynamicEnsembleConfig {
        let doc = ControlPlaneKeyValueDocument(tsv: tsv)
        var config = DynamicEnsembleConfig.defaults
        config.enabled = doc.bool("enabled", default: config.enabled)
        config.fallbackToRoutedEnsemble = doc.bool("fallback_to_routed_ensemble", default: config.fallbackToRoutedEnsemble)
        config.suppressTrustThreshold = doc.double("threshold_suppress_trust_threshold", default: config.suppressTrustThreshold)
        config.downweightTrustThreshold = doc.double("threshold_downweight_trust_threshold", default: config.downweightTrustThreshold)
        config.cautionQualityThreshold = doc.double("threshold_caution_quality_threshold", default: config.cautionQualityThreshold)
        config.abstainQualityThreshold = doc.double("threshold_abstain_quality_threshold", default: config.abstainQualityThreshold)
        config.blockQualityThreshold = doc.double("threshold_block_quality_threshold", default: config.blockQualityThreshold)
        config.minEffectiveWeight = doc.double("threshold_min_effective_weight", default: config.minEffectiveWeight)
        config.maxWeightShare = doc.double("threshold_max_weight_share", default: config.maxWeightShare)
        config.minActivePlugins = doc.int("threshold_min_active_plugins", default: config.minActivePlugins)
        config.penaltyConfidenceGap = doc.double("penalty_confidence_gap_penalty", default: config.penaltyConfidenceGap)
        config.penaltyContextRegret = doc.double("penalty_context_regret_penalty", default: config.penaltyContextRegret)
        config.penaltyDisagreement = doc.double("penalty_disagreement_penalty", default: config.penaltyDisagreement)
        config.penaltyDrift = doc.double("penalty_drift_penalty", default: config.penaltyDrift)
        config.penaltyCost = doc.double(
            "penalty_price_cost_penalty",
            default: doc.double(
                "penalty_cost_penalty",
                default: doc.double("penalty_spread_cost_penalty", default: config.penaltyCost)
            )
        )
        config.penaltyNews = doc.double("penalty_news_penalty", default: config.penaltyNews)
        config.penaltyRates = doc.double("penalty_rates_penalty", default: config.penaltyRates)
        config.penaltyMicro = doc.double("penalty_micro_penalty", default: config.penaltyMicro)
        config.penaltyStaleContext = doc.double("penalty_stale_context_penalty", default: config.penaltyStaleContext)
        config.penaltySinglePluginQuality = doc.double("penalty_single_plugin_quality_penalty", default: config.penaltySinglePluginQuality)
        config.penaltyConcentrationQuality = doc.double("penalty_concentration_quality_penalty", default: config.penaltyConcentrationQuality)
        config.weightReliabilityGain = doc.double("weight_reliability_gain", default: config.weightReliabilityGain)
        config.weightContextEdgeGain = doc.double("weight_context_edge_gain", default: config.weightContextEdgeGain)
        config.weightGlobalEdgeGain = doc.double("weight_global_edge_gain", default: config.weightGlobalEdgeGain)
        config.weightPortfolioGain = doc.double("weight_portfolio_gain", default: config.weightPortfolioGain)
        config.weightContextTrustGain = doc.double("weight_context_trust_gain", default: config.weightContextTrustGain)
        config.weightAdaptiveUpweightGain = doc.double("weight_adaptive_upweight_gain", default: config.weightAdaptiveUpweightGain)
        config.weightAdaptiveDownweightPenalty = doc.double("weight_adaptive_downweight_penalty", default: config.weightAdaptiveDownweightPenalty)

        for family in AIFamily.allCases {
            let name = family.controlPlaneName
            config.familyNewsCompatibility[family] = doc.double("family_news_compat_\(name)", default: config.familyNewsCompatibility[family])
            config.familyRatesCompatibility[family] = doc.double("family_rates_compat_\(name)", default: config.familyRatesCompatibility[family])
            config.familyMicroCompatibility[family] = doc.double("family_micro_compat_\(name)", default: config.familyMicroCompatibility[family])
            config.familyCostRobustness[family] = doc.double("family_cost_robustness_\(name)", default: config.familyCostRobustness[family])
            config.familyConfidenceCap[family] = doc.double("family_confidence_cap_\(name)", default: config.familyConfidenceCap[family])
            config.familyDisagreementTolerance[family] = doc.double("family_disagreement_tolerance_\(name)", default: config.familyDisagreementTolerance[family])
        }
        return config
    }

    private static func normalizedFamilyValues(_ raw: [Double]) -> [Double] {
        var values = Array(raw.prefix(AIFamily.allCases.count)).map { fxSafeFinite($0) }
        if values.count < AIFamily.allCases.count {
            values.append(contentsOf: Array(repeating: 1.0, count: AIFamily.allCases.count - values.count))
        }
        return values
    }

    private static func defaultFamilyNewsCompatibility() -> [Double] {
        var values = Array(repeating: 1.0, count: AIFamily.allCases.count)
        values[.linear] = 0.82
        values[.transformer] = 1.08
        values[.ruleBased] = 0.76
        return values
    }

    private static func defaultFamilyRatesCompatibility() -> [Double] {
        var values = Array(repeating: 1.0, count: AIFamily.allCases.count)
        values[.linear] = 0.92
        values[.transformer] = 1.00
        values[.ruleBased] = 0.88
        return values
    }

    private static func defaultFamilyMicroCompatibility() -> [Double] {
        var values = Array(repeating: 1.0, count: AIFamily.allCases.count)
        values[.linear] = 0.96
        values[.transformer] = 0.84
        values[.ruleBased] = 1.02
        return values
    }

    private static func defaultFamilyCostRobustness() -> [Double] {
        var values = Array(repeating: 1.0, count: AIFamily.allCases.count)
        values[.linear] = 0.98
        values[.transformer] = 0.82
        values[.ruleBased] = 1.02
        return values
    }

    private static func defaultFamilyConfidenceCap() -> [Double] {
        var values = Array(repeating: 0.96, count: AIFamily.allCases.count)
        values[.linear] = 0.84
        values[.transformer] = 0.86
        values[.ruleBased] = 0.78
        return values
    }

    private static func defaultFamilyDisagreementTolerance() -> [Double] {
        var values = Array(repeating: 1.0, count: AIFamily.allCases.count)
        values[.linear] = 0.92
        values[.transformer] = 0.92
        values[.ruleBased] = 0.90
        return values
    }
}

public struct DynamicEnsemblePluginRecord: Codable, Hashable, Sendable {
    public var ready: Bool
    public var aiIndex: Int
    public var aiName: String
    public var family: AIFamily
    public var signal: Int
    public var buyProbability: Double
    public var sellProbability: Double
    public var skipProbability: Double
    public var expectedMove: Double
    public var moveQ25: Double
    public var moveQ50: Double
    public var moveQ75: Double
    public var confidence: Double
    public var reliability: Double
    public var margin: Double
    public var hitTimeFraction: Double
    public var pathRisk: Double
    public var fillRisk: Double
    public var mfeRatio: Double
    public var maeRatio: Double
    public var buyEV: Double
    public var sellEV: Double
    public var baseMetaWeight: Double
    public var adaptiveSuitability: Double
    public var adaptiveStatus: AdaptiveRouterRuntimeStatus
    public var contextEdgeNorm: Double
    public var contextRegret: Double
    public var globalEdgeNorm: Double
    public var portfolioEdgeNorm: Double
    public var portfolioStability: Double
    public var portfolioCorrelation: Double
    public var portfolioDiversity: Double
    public var contextTrust: Double
    public var calibrationShrink: Double
    public var trustScore: Double
    public var normalizedWeight: Double
    public var status: DynamicEnsembleStatus
    public var reasons: [String]

    public init(
        ready: Bool = false,
        aiIndex: Int = -1,
        aiName: String = "",
        family: AIFamily = .other,
        signal: Int = -1,
        buyProbability: Double = 0.0,
        sellProbability: Double = 0.0,
        skipProbability: Double = 1.0,
        expectedMove: Double = 0.0,
        moveQ25: Double = 0.0,
        moveQ50: Double = 0.0,
        moveQ75: Double = 0.0,
        confidence: Double = 0.0,
        reliability: Double = 0.0,
        margin: Double = 0.0,
        hitTimeFraction: Double = 0.0,
        pathRisk: Double = 1.0,
        fillRisk: Double = 1.0,
        mfeRatio: Double = 0.0,
        maeRatio: Double = 0.0,
        buyEV: Double = 0.0,
        sellEV: Double = 0.0,
        baseMetaWeight: Double = 0.0,
        adaptiveSuitability: Double = 1.0,
        adaptiveStatus: AdaptiveRouterRuntimeStatus = .active,
        contextEdgeNorm: Double = 0.0,
        contextRegret: Double = 0.0,
        globalEdgeNorm: Double = 0.0,
        portfolioEdgeNorm: Double = 0.0,
        portfolioStability: Double = 0.0,
        portfolioCorrelation: Double = 0.0,
        portfolioDiversity: Double = 0.0,
        contextTrust: Double = 0.0,
        calibrationShrink: Double = 1.0,
        trustScore: Double = 0.0,
        normalizedWeight: Double = 0.0,
        status: DynamicEnsembleStatus = .excluded,
        reasons: [String] = []
    ) {
        self.ready = ready
        self.aiIndex = aiIndex
        self.aiName = aiName
        self.family = family
        self.signal = signal
        self.buyProbability = fxClamp(buyProbability, 0.0, 1.0)
        self.sellProbability = fxClamp(sellProbability, 0.0, 1.0)
        self.skipProbability = fxClamp(skipProbability, 0.0, 1.0)
        self.expectedMove = max(0.0, fxSafeFinite(expectedMove))
        self.moveQ25 = max(0.0, fxSafeFinite(moveQ25))
        self.moveQ50 = max(0.0, fxSafeFinite(moveQ50))
        self.moveQ75 = max(0.0, fxSafeFinite(moveQ75))
        self.confidence = fxClamp(confidence, 0.0, 1.0)
        self.reliability = fxClamp(reliability, 0.0, 1.0)
        self.margin = fxClamp(margin, 0.0, 1.0)
        self.hitTimeFraction = fxClamp(hitTimeFraction, 0.0, 1.0)
        self.pathRisk = fxClamp(pathRisk, 0.0, 1.0)
        self.fillRisk = fxClamp(fillRisk, 0.0, 1.0)
        self.mfeRatio = fxClamp(mfeRatio, 0.0, 4.0)
        self.maeRatio = fxClamp(maeRatio, 0.0, 2.0)
        self.buyEV = fxSafeFinite(buyEV)
        self.sellEV = fxSafeFinite(sellEV)
        self.baseMetaWeight = max(0.0, fxSafeFinite(baseMetaWeight))
        self.adaptiveSuitability = max(0.0, fxSafeFinite(adaptiveSuitability))
        self.adaptiveStatus = adaptiveStatus
        self.contextEdgeNorm = fxSafeFinite(contextEdgeNorm)
        self.contextRegret = fxClamp(contextRegret, 0.0, 1.0)
        self.globalEdgeNorm = fxSafeFinite(globalEdgeNorm)
        self.portfolioEdgeNorm = fxSafeFinite(portfolioEdgeNorm)
        self.portfolioStability = fxClamp(portfolioStability, 0.0, 1.0)
        self.portfolioCorrelation = fxClamp(portfolioCorrelation, 0.0, 1.0)
        self.portfolioDiversity = fxClamp(portfolioDiversity, 0.0, 1.0)
        self.contextTrust = fxClamp(contextTrust, 0.0, 1.0)
        self.calibrationShrink = fxClamp(calibrationShrink, 0.0, 1.0)
        self.trustScore = max(0.0, fxSafeFinite(trustScore))
        self.normalizedWeight = max(0.0, fxSafeFinite(normalizedWeight))
        self.status = status
        self.reasons = Self.uniqueReasons(reasons)
    }

    public static var reset: DynamicEnsemblePluginRecord {
        DynamicEnsemblePluginRecord()
    }

    public var reasonsCSV: String {
        reasons.filter { !$0.isEmpty }.joined(separator: "; ")
    }

    public mutating func appendReason(_ reason: String) {
        let value = reason.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !value.isEmpty,
              !reasons.contains(value),
              reasons.count < DynamicEnsembleRuntimeConstants.maxPluginReasons else {
            return
        }
        reasons.append(value)
    }

    private static func uniqueReasons(_ input: [String]) -> [String] {
        var output: [String] = []
        output.reserveCapacity(min(input.count, DynamicEnsembleRuntimeConstants.maxPluginReasons))
        for raw in input {
            let value = raw.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !value.isEmpty,
                  !output.contains(value),
                  output.count < DynamicEnsembleRuntimeConstants.maxPluginReasons else {
                continue
            }
            output.append(value)
        }
        return output
    }
}

public struct DynamicEnsembleRuntimeState: Codable, Hashable, Sendable {
    public var ready: Bool
    public var fallbackUsed: Bool
    public var generatedAt: Int64
    public var symbol: String
    public var topRegime: String
    public var sessionLabel: String
    public var tradePosture: String
    public var ensembleQuality: Double
    public var abstainBias: Double
    public var agreementScore: Double
    public var contextFitScore: Double
    public var dominantPluginShare: Double
    public var participatingCount: Int
    public var downweightedCount: Int
    public var suppressedCount: Int
    public var buySupport: Double
    public var sellSupport: Double
    public var skipSupport: Double
    public var buyProbability: Double
    public var sellProbability: Double
    public var skipProbability: Double
    public var finalScore: Double
    public var finalAction: Int
    public var reasons: [String]

    public init(
        ready: Bool = false,
        fallbackUsed: Bool = false,
        generatedAt: Int64 = 0,
        symbol: String = "",
        topRegime: String = "UNKNOWN",
        sessionLabel: String = "UNKNOWN",
        tradePosture: String = "NORMAL",
        ensembleQuality: Double = 0.0,
        abstainBias: Double = 0.0,
        agreementScore: Double = 0.0,
        contextFitScore: Double = 0.0,
        dominantPluginShare: Double = 0.0,
        participatingCount: Int = 0,
        downweightedCount: Int = 0,
        suppressedCount: Int = 0,
        buySupport: Double = 0.0,
        sellSupport: Double = 0.0,
        skipSupport: Double = 1.0,
        buyProbability: Double = 0.0,
        sellProbability: Double = 0.0,
        skipProbability: Double = 1.0,
        finalScore: Double = 0.0,
        finalAction: Int = -1,
        reasons: [String] = []
    ) {
        self.ready = ready
        self.fallbackUsed = fallbackUsed
        self.generatedAt = max(0, generatedAt)
        self.symbol = symbol.uppercased()
        self.topRegime = topRegime.isEmpty ? "UNKNOWN" : topRegime
        self.sessionLabel = sessionLabel.isEmpty ? "UNKNOWN" : sessionLabel
        self.tradePosture = tradePosture.isEmpty ? "NORMAL" : tradePosture
        self.ensembleQuality = fxClamp(ensembleQuality, 0.0, 1.0)
        self.abstainBias = fxClamp(abstainBias, 0.0, 0.98)
        self.agreementScore = fxClamp(agreementScore, 0.0, 1.0)
        self.contextFitScore = fxClamp(contextFitScore, 0.0, 1.0)
        self.dominantPluginShare = fxClamp(dominantPluginShare, 0.0, 1.0)
        self.participatingCount = max(0, participatingCount)
        self.downweightedCount = max(0, downweightedCount)
        self.suppressedCount = max(0, suppressedCount)
        self.buySupport = max(0.0, fxSafeFinite(buySupport))
        self.sellSupport = max(0.0, fxSafeFinite(sellSupport))
        self.skipSupport = max(0.0, fxSafeFinite(skipSupport))
        self.buyProbability = fxClamp(buyProbability, 0.0, 1.0)
        self.sellProbability = fxClamp(sellProbability, 0.0, 1.0)
        self.skipProbability = fxClamp(skipProbability, 0.0, 1.0)
        self.finalScore = fxClamp(finalScore, -1.0, 1.0)
        self.finalAction = finalAction
        self.reasons = Self.uniqueReasons(reasons)
    }

    public static var reset: DynamicEnsembleRuntimeState {
        DynamicEnsembleRuntimeState()
    }

    public var reasonsCSV: String {
        reasons.filter { !$0.isEmpty }.joined(separator: "; ")
    }

    public mutating func appendReason(_ reason: String) {
        let value = reason.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !value.isEmpty,
              !reasons.contains(value),
              reasons.count < DynamicEnsembleRuntimeConstants.maxReasons else {
            return
        }
        reasons.append(value)
    }

    public func withFinalAction(_ finalAction: Int) -> DynamicEnsembleRuntimeState {
        var copy = self
        copy.finalAction = finalAction
        return copy
    }

    private static func uniqueReasons(_ input: [String]) -> [String] {
        var output: [String] = []
        output.reserveCapacity(min(input.count, DynamicEnsembleRuntimeConstants.maxReasons))
        for raw in input {
            let value = raw.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !value.isEmpty,
                  !output.contains(value),
                  output.count < DynamicEnsembleRuntimeConstants.maxReasons else {
                continue
            }
            output.append(value)
        }
        return output
    }
}

public struct DynamicEnsembleInputs: Sendable {
    public var symbol: String
    public var generatedAt: Int64
    public var priceCostPoints: Double
    public var minMovePoints: Double
    public var driftNorm: Double
    public var regimeState: AdaptiveRegimeState
    public var newsState: NewsPulsePairState
    public var ratesState: RatesEnginePairState
    public var crossAssetState: CrossAssetPairState
    public var microstructureState: MicrostructurePairState
    public var records: [DynamicEnsemblePluginRecord]
    public var config: DynamicEnsembleConfig
    public var dynamicEnsembleEnabled: Bool

    public init(
        symbol: String,
        generatedAt: Int64,
        priceCostPoints: Double = 0.0,
        minMovePoints: Double = 0.0,
        driftNorm: Double = 0.0,
        regimeState: AdaptiveRegimeState = .reset,
        newsState: NewsPulsePairState = .reset,
        ratesState: RatesEnginePairState = .reset,
        crossAssetState: CrossAssetPairState = .reset,
        microstructureState: MicrostructurePairState = .reset,
        records: [DynamicEnsemblePluginRecord] = [],
        config: DynamicEnsembleConfig = .defaults,
        dynamicEnsembleEnabled: Bool = true
    ) {
        self.symbol = symbol.uppercased()
        self.generatedAt = max(0, generatedAt)
        self.priceCostPoints = max(0.0, fxSafeFinite(priceCostPoints))
        self.minMovePoints = max(0.0, fxSafeFinite(minMovePoints))
        self.driftNorm = fxClamp(driftNorm, 0.0, 1.0)
        self.regimeState = regimeState
        self.newsState = newsState
        self.ratesState = ratesState
        self.crossAssetState = crossAssetState
        self.microstructureState = microstructureState
        self.records = records
        self.config = config
        self.dynamicEnsembleEnabled = dynamicEnsembleEnabled
    }
}

public struct DynamicEnsembleEvaluation: Codable, Hashable, Sendable {
    public var applied: Bool
    public var state: DynamicEnsembleRuntimeState
    public var records: [DynamicEnsemblePluginRecord]

    public init(applied: Bool, state: DynamicEnsembleRuntimeState, records: [DynamicEnsemblePluginRecord]) {
        self.applied = applied
        self.state = state
        self.records = records
    }
}

extension Array where Element == Double {
    subscript(_ family: AIFamily) -> Double {
        get {
            guard family.rawValue >= 0, family.rawValue < count else { return 1.0 }
            return self[family.rawValue]
        }
        set {
            guard family.rawValue >= 0, family.rawValue < count else { return }
            self[family.rawValue] = newValue
        }
    }
}

public extension RuntimeArtifactFileRepository {
    func writeDynamicEnsembleRuntimeArtifacts(
        symbol: String,
        state: DynamicEnsembleRuntimeState,
        records: [DynamicEnsemblePluginRecord],
        finalDecision: Int? = nil
    ) throws {
        guard let stateTSV = DynamicEnsembleRuntimeTools.runtimeStateTSV(
            symbol: symbol,
            state: state,
            records: records,
            finalDecision: finalDecision
        ),
            let historyLine = DynamicEnsembleRuntimeTools.runtimeHistoryNDJSONLine(
                symbol: symbol,
                state: state,
                records: records,
                finalDecision: finalDecision
            ) else {
            return
        }

        let stateURL = url(for: DynamicEnsembleRuntimeTools.runtimeStatePath(symbol: symbol))
        try fileManager.createDirectory(
            at: stateURL.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        try stateTSV.write(to: stateURL, atomically: true, encoding: .utf8)

        let historyURL = url(for: DynamicEnsembleRuntimeTools.runtimeHistoryPath(symbol: symbol))
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
