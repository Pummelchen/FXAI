import Foundation

extension GUIValidationFixtures {
    public static func driftGovernanceSnapshot(projectRoot: URL) -> DriftGovernanceSnapshot {
        let now = Date()
        let eurusd = DriftGovernanceSymbolSnapshot(
            symbol: "EURUSD",
            pluginCount: 3,
            healthCounts: [
                KeyValueRecord(key: "CAUTION", value: "1"),
                KeyValueRecord(key: "DEGRADED", value: "1"),
                KeyValueRecord(key: "HEALTHY", value: "1"),
            ],
            governanceCounts: [
                KeyValueRecord(key: "CAUTION", value: "1"),
                KeyValueRecord(key: "DEGRADED", value: "1"),
                KeyValueRecord(key: "CHAMPION", value: "1"),
            ],
            actionCounts: [
                KeyValueRecord(key: "DOWNWEIGHT", value: "2"),
                KeyValueRecord(key: "NONE", value: "1"),
            ],
            latestContext: [
                KeyValueRecord(key: "execution.state", value: "CAUTION"),
                KeyValueRecord(key: "router.regime", value: "HIGH_VOL_EVENT"),
                KeyValueRecord(key: "newspulse.trade_gate", value: "CAUTION"),
                KeyValueRecord(key: "ensemble.max_abstain", value: "0.380"),
            ],
            plugins: [
                DriftGovernancePluginSnapshot(
                    pluginName: "ai_tft",
                    familyID: 31,
                    familyName: "Transformer",
                    baseRegistryStatus: "champion",
                    healthState: "DEGRADED",
                    governanceState: "DEGRADED",
                    recommendedGovernanceState: "DEGRADED",
                    actionRecommendation: "DOWNWEIGHT",
                    actionApplied: true,
                    weightMultiplier: 0.55,
                    restrictLive: false,
                    shadowOnly: false,
                    disabled: false,
                    aggregateRiskScore: 0.71,
                    driftScores: [
                        KeyValueRecord(key: "calibration_drift_score", value: "0.810"),
                        KeyValueRecord(key: "pair_decay_score", value: "0.760"),
                        KeyValueRecord(key: "performance_drift_score", value: "0.790"),
                    ],
                    support: [
                        KeyValueRecord(key: "sample_count_recent", value: "842"),
                        KeyValueRecord(key: "sample_count_reference", value: "10420"),
                        KeyValueRecord(key: "reference_scope", value: "SYMBOL_PLUGIN"),
                    ],
                    reasonCodes: [
                        "CALIBRATION_DRIFT_ELEVATED",
                        "PAIR_DECAY_EURUSD",
                        "RECENT_COST_ADJUSTED_UTILITY_WEAK",
                    ],
                    qualityFlags: [
                        KeyValueRecord(key: "low_support", value: "false"),
                        KeyValueRecord(key: "fallback_thresholds_used", value: "false"),
                    ],
                    contextSummary: [
                        KeyValueRecord(key: "execution.min_quality", value: "0.440"),
                        KeyValueRecord(key: "router.posture", value: "CAUTION"),
                    ],
                    challengerEvaluation: nil
                ),
                DriftGovernancePluginSnapshot(
                    pluginName: "ai_gha",
                    familyID: 24,
                    familyName: "Macro",
                    baseRegistryStatus: "challenger",
                    healthState: "CAUTION",
                    governanceState: "CHAMPION_CANDIDATE",
                    recommendedGovernanceState: "CHAMPION_CANDIDATE",
                    actionRecommendation: "PROMOTION_REVIEW",
                    actionApplied: false,
                    weightMultiplier: 1.0,
                    restrictLive: false,
                    shadowOnly: false,
                    disabled: false,
                    aggregateRiskScore: 0.34,
                    driftScores: [
                        KeyValueRecord(key: "feature_drift_score", value: "0.210"),
                        KeyValueRecord(key: "execution_drift_score", value: "0.410"),
                    ],
                    support: [
                        KeyValueRecord(key: "sample_count_recent", value: "516"),
                        KeyValueRecord(key: "sample_count_reference", value: "8021"),
                        KeyValueRecord(key: "reference_scope", value: "SYMBOL_PLUGIN"),
                    ],
                    reasonCodes: [
                        "CHALLENGER_PROMOTION_ELIGIBLE",
                    ],
                    qualityFlags: [
                        KeyValueRecord(key: "low_support", value: "false"),
                    ],
                    contextSummary: [
                        KeyValueRecord(key: "prob.uncertainty", value: "0.270"),
                        KeyValueRecord(key: "cross_asset.macro", value: "RATES_REPRICING"),
                    ],
                    challengerEvaluation: DriftGovernanceChallengerEvaluation(
                        eligibilityState: "QUALIFIED",
                        qualifies: true,
                        supportCount: 14,
                        shadowSupport: 516,
                        walkforwardScore: 74.0,
                        recentScore: 72.0,
                        adversarialScore: 68.0,
                        macroEventScore: 71.0,
                        calibrationError: 0.08,
                        issueCount: 0.0,
                        liveShadowScore: 0.66,
                        liveReliability: 0.71,
                        portfolioScore: 0.74,
                        promotionMargin: 0.05
                    )
                ),
                DriftGovernancePluginSnapshot(
                    pluginName: "tree_catboost",
                    familyID: 12,
                    familyName: "Tree",
                    baseRegistryStatus: "champion",
                    healthState: "HEALTHY",
                    governanceState: "CHAMPION",
                    recommendedGovernanceState: "CHAMPION",
                    actionRecommendation: "NONE",
                    actionApplied: false,
                    weightMultiplier: 1.0,
                    restrictLive: false,
                    shadowOnly: false,
                    disabled: false,
                    aggregateRiskScore: 0.18,
                    driftScores: [
                        KeyValueRecord(key: "feature_drift_score", value: "0.120"),
                        KeyValueRecord(key: "regime_drift_score", value: "0.090"),
                    ],
                    support: [
                        KeyValueRecord(key: "sample_count_recent", value: "910"),
                        KeyValueRecord(key: "sample_count_reference", value: "13210"),
                    ],
                    reasonCodes: [],
                    qualityFlags: [
                        KeyValueRecord(key: "low_support", value: "false"),
                    ],
                    contextSummary: [
                        KeyValueRecord(key: "ensemble.quality", value: "0.670"),
                    ],
                    challengerEvaluation: nil
                ),
            ],
            recentActions: [
                DriftGovernanceActionRecord(
                    pluginName: "ai_tft",
                    previousState: "CHAMPION",
                    newState: "DEGRADED",
                    actionKind: "DOWNWEIGHT",
                    actionApplied: true,
                    createdAt: now.addingTimeInterval(-900)
                ),
                DriftGovernanceActionRecord(
                    pluginName: "ai_gha",
                    previousState: "CHALLENGER",
                    newState: "CHAMPION_CANDIDATE",
                    actionKind: "PROMOTION_REVIEW",
                    actionApplied: false,
                    createdAt: now.addingTimeInterval(-1200)
                ),
            ]
        )

        let usdjpy = DriftGovernanceSymbolSnapshot(
            symbol: "USDJPY",
            pluginCount: 2,
            healthCounts: [
                KeyValueRecord(key: "HEALTHY", value: "2"),
            ],
            governanceCounts: [
                KeyValueRecord(key: "CHAMPION", value: "1"),
                KeyValueRecord(key: "CHALLENGER", value: "1"),
            ],
            actionCounts: [
                KeyValueRecord(key: "NONE", value: "2"),
            ],
            latestContext: [
                KeyValueRecord(key: "execution.state", value: "NORMAL"),
                KeyValueRecord(key: "router.regime", value: "TREND_PERSISTENT"),
            ],
            plugins: [
                DriftGovernancePluginSnapshot(
                    pluginName: "tree_catboost",
                    familyID: 12,
                    familyName: "Tree",
                    baseRegistryStatus: "champion",
                    healthState: "HEALTHY",
                    governanceState: "CHAMPION",
                    recommendedGovernanceState: "CHAMPION",
                    actionRecommendation: "NONE",
                    actionApplied: false,
                    weightMultiplier: 1.0,
                    restrictLive: false,
                    shadowOnly: false,
                    disabled: false,
                    aggregateRiskScore: 0.11,
                    driftScores: [
                        KeyValueRecord(key: "feature_drift_score", value: "0.050"),
                        KeyValueRecord(key: "performance_drift_score", value: "0.090"),
                    ],
                    support: [
                        KeyValueRecord(key: "sample_count_recent", value: "688"),
                        KeyValueRecord(key: "sample_count_reference", value: "9401"),
                    ],
                    reasonCodes: [],
                    qualityFlags: [
                        KeyValueRecord(key: "low_support", value: "false"),
                    ],
                    contextSummary: [
                        KeyValueRecord(key: "execution.min_quality", value: "0.720"),
                    ],
                    challengerEvaluation: nil
                ),
                DriftGovernancePluginSnapshot(
                    pluginName: "ai_tft",
                    familyID: 31,
                    familyName: "Transformer",
                    baseRegistryStatus: "challenger",
                    healthState: "HEALTHY",
                    governanceState: "CHALLENGER",
                    recommendedGovernanceState: "CHALLENGER",
                    actionRecommendation: "NONE",
                    actionApplied: false,
                    weightMultiplier: 1.0,
                    restrictLive: false,
                    shadowOnly: false,
                    disabled: false,
                    aggregateRiskScore: 0.22,
                    driftScores: [
                        KeyValueRecord(key: "pair_decay_score", value: "0.140"),
                    ],
                    support: [
                        KeyValueRecord(key: "sample_count_recent", value: "402"),
                        KeyValueRecord(key: "sample_count_reference", value: "5770"),
                    ],
                    reasonCodes: [],
                    qualityFlags: [
                        KeyValueRecord(key: "low_support", value: "false"),
                    ],
                    contextSummary: [
                        KeyValueRecord(key: "newspulse.trade_gate", value: "ALLOW"),
                    ],
                    challengerEvaluation: DriftGovernanceChallengerEvaluation(
                        eligibilityState: "INSUFFICIENT",
                        qualifies: false,
                        supportCount: 8,
                        shadowSupport: 402,
                        walkforwardScore: 66.0,
                        recentScore: 61.0,
                        adversarialScore: 58.0,
                        macroEventScore: 60.0,
                        calibrationError: 0.12,
                        issueCount: 1.0,
                        liveShadowScore: 0.58,
                        liveReliability: 0.62,
                        portfolioScore: 0.60,
                        promotionMargin: 0.01
                    )
                ),
            ],
            recentActions: []
        )

        return DriftGovernanceSnapshot(
            generatedAt: now,
            profileName: "continuous",
            policyVersion: 1,
            actionMode: "AUTO_APPLY_PROTECTIVE",
            symbolCount: 2,
            pluginCount: 5,
            latestActionCount: 2,
            healthCounts: [
                KeyValueRecord(key: "CAUTION", value: "1"),
                KeyValueRecord(key: "DEGRADED", value: "1"),
                KeyValueRecord(key: "HEALTHY", value: "3"),
            ],
            governanceCounts: [
                KeyValueRecord(key: "CAUTION", value: "1"),
                KeyValueRecord(key: "CHALLENGER", value: "1"),
                KeyValueRecord(key: "CHAMPION", value: "2"),
                KeyValueRecord(key: "CHAMPION_CANDIDATE", value: "1"),
            ],
            actionCounts: [
                KeyValueRecord(key: "DOWNWEIGHT", value: "1"),
                KeyValueRecord(key: "NONE", value: "3"),
                KeyValueRecord(key: "PROMOTION_REVIEW", value: "1"),
            ],
            statusRecords: [
                KeyValueRecord(key: "applied_action_count", value: "1"),
                KeyValueRecord(key: "plugin_count", value: "5"),
                KeyValueRecord(key: "policy_version", value: "1"),
                KeyValueRecord(key: "symbol_count", value: "2"),
            ],
            artifactPaths: [
                KeyValueRecord(key: "history_path", value: projectRoot.appendingPathComponent("FXDataEngine/Tools/OfflineLab/DriftGovernance/drift_governance_history.ndjson").path),
                KeyValueRecord(key: "report_path", value: projectRoot.appendingPathComponent("FXDataEngine/Tools/OfflineLab/DriftGovernance/Reports/drift_governance_report.json").path),
                KeyValueRecord(key: "status_path", value: projectRoot.appendingPathComponent("FXDataEngine/Tools/OfflineLab/DriftGovernance/drift_governance_status.json").path),
            ],
            symbols: [eurusd, usdjpy]
        )
    }

    public static func probCalibrationSnapshot(projectRoot: URL) -> ProbCalibrationSnapshot {
        let now = Date()
        let eurusd = ProbCalibrationSymbolSnapshot(
            symbol: "EURUSD",
            generatedAt: now.addingTimeInterval(-80),
            method: "LOGISTIC_AFFINE",
            sessionLabel: "LONDON_NY_OVERLAP",
            regimeLabel: "HIGH_VOL_EVENT",
            tierKind: "PAIR_REGIME",
            tierKey: "PAIR_REGIME|EURUSD|*|HIGH_VOL_EVENT",
            support: 118,
            quality: 0.61,
            rawAction: "BUY",
            rawScore: 0.19,
            rawBuyProb: 0.57,
            rawSellProb: 0.21,
            rawSkipProb: 0.22,
            calibratedBuyProb: 0.46,
            calibratedSellProb: 0.24,
            calibratedSkipProb: 0.30,
            calibratedConfidence: 0.46,
            expectedMoveMeanPoints: 8.4,
            expectedMoveQ25Points: 2.7,
            expectedMoveQ50Points: 6.1,
            expectedMoveQ75Points: 11.3,
            priceCostPoints: 1.6,
            slippageCostPoints: 0.9,
            uncertaintyScore: 0.63,
            uncertaintyPenaltyPoints: 2.1,
            riskPenaltyPoints: 1.2,
            expectedGrossEdgePoints: 1.85,
            edgeAfterCostsPoints: -3.95,
            finalAction: "SKIP",
            abstain: true,
            fallbackUsed: false,
            calibrationStale: false,
            inputStale: false,
            supportUsable: true,
            reasons: [
                "EDGE_TOO_SMALL",
                "UNCERTAINTY_TOO_HIGH",
                "NEWS_RISK_BLOCK",
            ],
            replayActionCounts: [
                KeyValueRecord(key: "SKIP", value: "12"),
                KeyValueRecord(key: "BUY", value: "5"),
            ],
            replayTierCounts: [
                KeyValueRecord(key: "PAIR_REGIME", value: "13"),
                KeyValueRecord(key: "REGIME", value: "4"),
            ],
            replayTopReasons: [
                KeyValueRecord(key: "EDGE_TOO_SMALL", value: "10"),
                KeyValueRecord(key: "UNCERTAINTY_TOO_HIGH", value: "6"),
            ],
            recentTransitions: [
                ProbCalibrationTransition(type: "action_change", fromValue: "BUY", toValue: "SKIP", observedAt: now.addingTimeInterval(-1600)),
            ],
            observationCount: 18,
            abstainCount: 12,
            fallbackCount: 1,
            averageConfidence: 0.49,
            averageEdgeAfterCostsPoints: -0.84,
            averageUncertaintyScore: 0.47,
            minEdgeAfterCostsPoints: -4.10,
            maxEdgeAfterCostsPoints: 2.40
        )

        let usdjpy = ProbCalibrationSymbolSnapshot(
            symbol: "USDJPY",
            generatedAt: now.addingTimeInterval(-100),
            method: "LOGISTIC_AFFINE",
            sessionLabel: "LONDON",
            regimeLabel: "TREND_PERSISTENT",
            tierKind: "GLOBAL",
            tierKey: "GLOBAL|*|*|*",
            support: 504,
            quality: 0.72,
            rawAction: "BUY",
            rawScore: 0.41,
            rawBuyProb: 0.64,
            rawSellProb: 0.14,
            rawSkipProb: 0.22,
            calibratedBuyProb: 0.61,
            calibratedSellProb: 0.13,
            calibratedSkipProb: 0.26,
            calibratedConfidence: 0.61,
            expectedMoveMeanPoints: 10.6,
            expectedMoveQ25Points: 4.0,
            expectedMoveQ50Points: 8.2,
            expectedMoveQ75Points: 14.8,
            priceCostPoints: 1.4,
            slippageCostPoints: 0.6,
            uncertaintyScore: 0.24,
            uncertaintyPenaltyPoints: 0.8,
            riskPenaltyPoints: 0.5,
            expectedGrossEdgePoints: 5.09,
            edgeAfterCostsPoints: 1.79,
            finalAction: "BUY",
            abstain: false,
            fallbackUsed: true,
            calibrationStale: false,
            inputStale: false,
            supportUsable: true,
            reasons: [],
            replayActionCounts: [
                KeyValueRecord(key: "BUY", value: "14"),
                KeyValueRecord(key: "SKIP", value: "3"),
            ],
            replayTierCounts: [
                KeyValueRecord(key: "GLOBAL", value: "17"),
            ],
            replayTopReasons: [
                KeyValueRecord(key: "EDGE_TOO_SMALL", value: "2"),
            ],
            recentTransitions: [],
            observationCount: 17,
            abstainCount: 3,
            fallbackCount: 5,
            averageConfidence: 0.58,
            averageEdgeAfterCostsPoints: 1.22,
            averageUncertaintyScore: 0.29,
            minEdgeAfterCostsPoints: -0.40,
            maxEdgeAfterCostsPoints: 3.10
        )

        return ProbCalibrationSnapshot(
            generatedAt: now,
            replayHoursBack: 48,
            symbols: [eurusd, usdjpy]
        )
    }

    public static func executionQualitySnapshot(projectRoot: URL) -> ExecutionQualitySnapshot {
        let now = Date()
        let eurusd = ExecutionQualitySymbolSnapshot(
            symbol: "EURUSD",
            generatedAt: now.addingTimeInterval(-70),
            method: "SCORECARD_V1",
            sessionLabel: "LONDON_NY_OVERLAP",
            regimeLabel: "HIGH_VOL_EVENT",
            tierKind: "PAIR_REGIME",
            tierKey: "PAIR_REGIME|EURUSD|LONDON_NY_OVERLAP|HIGH_VOL_EVENT",
            support: 92,
            quality: 0.58,
            fallbackUsed: false,
            memoryStale: false,
            dataStale: false,
            supportUsable: true,
            newsWindowActive: true,
            ratesRepricingActive: false,
            brokerCoverage: 0.71,
            brokerRejectProbability: 0.19,
            brokerPartialFillProbability: 0.14,
            priceCostNowPoints: 1.3,
            priceCostExpectedPoints: 2.8,
            priceCostWideningRisk: 0.68,
            expectedSlippagePoints: 1.1,
            slippageRisk: 0.57,
            fillQualityScore: 0.49,
            latencySensitivityScore: 0.63,
            liquidityFragilityScore: 0.61,
            executionQualityScore: 0.44,
            allowedDeviationPoints: 6.0,
            cautionLotScale: 0.82,
            cautionEnterProbBuffer: 0.04,
            executionState: "CAUTION",
            reasons: [
                "NEWS_WINDOW_ACTIVE",
                "PRICE_COST_ALREADY_ELEVATED",
                "LATENCY_SENSITIVITY_HIGH",
            ],
            replayStateCounts: [
                KeyValueRecord(key: "CAUTION", value: "11"),
                KeyValueRecord(key: "NORMAL", value: "7"),
            ],
            replayTierCounts: [
                KeyValueRecord(key: "PAIR_REGIME", value: "12"),
                KeyValueRecord(key: "REGIME", value: "6"),
            ],
            replayTopReasons: [
                KeyValueRecord(key: "NEWS_WINDOW_ACTIVE", value: "9"),
                KeyValueRecord(key: "PRICE_COST_ALREADY_ELEVATED", value: "7"),
            ],
            recentTransitions: [
                ExecutionQualityTransition(type: "execution_state", fromValue: "NORMAL", toValue: "CAUTION", observedAt: now.addingTimeInterval(-900)),
            ],
            observationCount: 18,
            maxPriceCostWideningRisk: 0.79,
            maxSlippageRisk: 0.66,
            minExecutionQualityScore: 0.39
        )

        let usdjpy = ExecutionQualitySymbolSnapshot(
            symbol: "USDJPY",
            generatedAt: now.addingTimeInterval(-95),
            method: "SCORECARD_V1",
            sessionLabel: "LONDON",
            regimeLabel: "TREND_PERSISTENT",
            tierKind: "GLOBAL",
            tierKey: "GLOBAL|*|*|*",
            support: 244,
            quality: 0.74,
            fallbackUsed: false,
            memoryStale: false,
            dataStale: false,
            supportUsable: true,
            newsWindowActive: false,
            ratesRepricingActive: false,
            brokerCoverage: 0.84,
            brokerRejectProbability: 0.07,
            brokerPartialFillProbability: 0.05,
            priceCostNowPoints: 1.1,
            priceCostExpectedPoints: 1.4,
            priceCostWideningRisk: 0.29,
            expectedSlippagePoints: 0.4,
            slippageRisk: 0.22,
            fillQualityScore: 0.74,
            latencySensitivityScore: 0.31,
            liquidityFragilityScore: 0.28,
            executionQualityScore: 0.73,
            allowedDeviationPoints: 3.0,
            cautionLotScale: 1.0,
            cautionEnterProbBuffer: 0.0,
            executionState: "NORMAL",
            reasons: [],
            replayStateCounts: [
                KeyValueRecord(key: "NORMAL", value: "17"),
            ],
            replayTierCounts: [
                KeyValueRecord(key: "GLOBAL", value: "17"),
            ],
            replayTopReasons: [
                KeyValueRecord(key: "LOW_LIQUIDITY_SESSION", value: "2"),
            ],
            recentTransitions: [],
            observationCount: 17,
            maxPriceCostWideningRisk: 0.42,
            maxSlippageRisk: 0.33,
            minExecutionQualityScore: 0.66
        )

        return ExecutionQualitySnapshot(
            generatedAt: now,
            replayHoursBack: 48,
            symbols: [eurusd, usdjpy]
        )
    }

}
