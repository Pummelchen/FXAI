import Foundation

extension GUIValidationFixtures {
    public static func adaptiveRouterSnapshot(projectRoot: URL) -> AdaptiveRouterSnapshot {
        let now = Date()
        let eurusd = AdaptiveRouterSymbolSnapshot(
            symbol: "EURUSD",
            profileName: "continuous",
            routerMode: "WEIGHTED_ENSEMBLE",
            topRegime: "HIGH_VOL_EVENT",
            confidence: 0.81,
            tradePosture: "CAUTION",
            abstainBias: 0.18,
            sessionLabel: "LONDON_NY_OVERLAP",
            priceCostRegime: "ELEVATED",
            volatilityRegime: "HIGH",
            newsRiskScore: 0.76,
            newsPressure: -0.14,
            eventETAMin: 14,
            staleNews: false,
            liquidityStress: 0.58,
            breakoutPressure: 0.64,
            trendStrength: 0.49,
            rangePressure: 0.24,
            macroPressure: 0.72,
            generatedAt: now.addingTimeInterval(-90),
            profileGeneratedAt: now.addingTimeInterval(-600),
            reasons: [
                "NewsPulse event window active",
                "Price-cost regime elevated",
                "Volatility expansion detected",
            ],
            probabilities: [
                AdaptiveRouterProbabilityRecord(label: "HIGH_VOL_EVENT", probability: 0.81),
                AdaptiveRouterProbabilityRecord(label: "LIQUIDITY_STRESS", probability: 0.46),
                AdaptiveRouterProbabilityRecord(label: "RISK_ON_OFF_MACRO", probability: 0.32),
                AdaptiveRouterProbabilityRecord(label: "BREAKOUT_TRANSITION", probability: 0.24),
            ],
            activePlugins: [
                AdaptiveRouterPluginState(name: "ai_gha", weight: 0.46, suitability: 1.28, status: "UPWEIGHTED", reasons: ["Strong macro/event regime fit"]),
                AdaptiveRouterPluginState(name: "ai_tesseract", weight: 0.30, suitability: 1.14, status: "ACTIVE", reasons: ["Breakout-transition fit remains acceptable"]),
            ],
            downweightedPlugins: [
                AdaptiveRouterPluginState(name: "ai_tft", weight: 0.18, suitability: 0.79, status: "DOWNWEIGHTED", reasons: ["Recent calibration softer in event windows"]),
            ],
            suppressedPlugins: [
                AdaptiveRouterPluginState(name: "lin_pa", weight: 0.0, suitability: 0.22, status: "SUPPRESSED", reasons: ["Mean reversion suppressed in event regime"]),
            ],
            pairTags: ["active", "dollar_core", "macro_sensitive"],
            topProfilePlugins: ["ai_gha", "ai_tesseract", "ai_tft"],
            thresholdMetrics: [
                KeyValueRecord(key: "caution_threshold", value: "0.53"),
                KeyValueRecord(key: "abstain_threshold", value: "0.34"),
                KeyValueRecord(key: "block_threshold", value: "0.17"),
                KeyValueRecord(key: "suppression_threshold", value: "0.34"),
            ],
            regimeBiasMetrics: [
                KeyValueRecord(key: "regime_bias_HIGH_VOL_EVENT", value: "1.10"),
                KeyValueRecord(key: "regime_bias_RISK_ON_OFF_MACRO", value: "1.06"),
            ],
            replayRegimeCounts: [
                KeyValueRecord(key: "HIGH_VOL_EVENT", value: "18"),
                KeyValueRecord(key: "BREAKOUT_TRANSITION", value: "7"),
            ],
            replayPostureCounts: [
                KeyValueRecord(key: "CAUTION", value: "16"),
                KeyValueRecord(key: "ABSTAIN_BIAS", value: "4"),
            ],
            replayTopReasons: [
                KeyValueRecord(key: "NewsPulse event window active", value: "18"),
                KeyValueRecord(key: "Price-cost regime elevated", value: "12"),
            ],
            replayTopPlugins: [
                KeyValueRecord(key: "ai_gha", value: "14"),
                KeyValueRecord(key: "ai_tesseract", value: "8"),
            ],
            recentTransitions: [
                AdaptiveRouterTransition(type: "regime_change", fromValue: "BREAKOUT_TRANSITION", toValue: "HIGH_VOL_EVENT", observedAt: now.addingTimeInterval(-1800)),
                AdaptiveRouterTransition(type: "posture_change", fromValue: "NORMAL", toValue: "CAUTION", observedAt: now.addingTimeInterval(-1700)),
            ],
            observationCount: 24
        )

        let usdjpy = AdaptiveRouterSymbolSnapshot(
            symbol: "USDJPY",
            profileName: "continuous",
            routerMode: "WEIGHTED_ENSEMBLE",
            topRegime: "TREND_PERSISTENT",
            confidence: 0.68,
            tradePosture: "NORMAL",
            abstainBias: 0.06,
            sessionLabel: "LONDON",
            priceCostRegime: "NORMAL",
            volatilityRegime: "NORMAL",
            newsRiskScore: 0.22,
            newsPressure: 0.18,
            eventETAMin: nil,
            staleNews: false,
            liquidityStress: 0.24,
            breakoutPressure: 0.32,
            trendStrength: 0.71,
            rangePressure: 0.21,
            macroPressure: 0.44,
            generatedAt: now.addingTimeInterval(-120),
            profileGeneratedAt: now.addingTimeInterval(-600),
            reasons: [
                "Directional persistence elevated",
                "London session flow dominant",
            ],
            probabilities: [
                AdaptiveRouterProbabilityRecord(label: "TREND_PERSISTENT", probability: 0.68),
                AdaptiveRouterProbabilityRecord(label: "SESSION_FLOW", probability: 0.42),
                AdaptiveRouterProbabilityRecord(label: "BREAKOUT_TRANSITION", probability: 0.19),
            ],
            activePlugins: [
                AdaptiveRouterPluginState(name: "ai_tft", weight: 0.44, suitability: 1.19, status: "UPWEIGHTED", reasons: ["Strong trend persistence fit"]),
                AdaptiveRouterPluginState(name: "tree_catboost", weight: 0.33, suitability: 1.05, status: "ACTIVE", reasons: ["Stable baseline fit"]),
            ],
            downweightedPlugins: [],
            suppressedPlugins: [],
            pairTags: ["yen_cross"],
            topProfilePlugins: ["ai_tft", "tree_catboost"],
            thresholdMetrics: [
                KeyValueRecord(key: "caution_threshold", value: "0.50"),
                KeyValueRecord(key: "abstain_threshold", value: "0.32"),
            ],
            regimeBiasMetrics: [
                KeyValueRecord(key: "regime_bias_TREND_PERSISTENT", value: "1.00"),
                KeyValueRecord(key: "regime_bias_LIQUIDITY_STRESS", value: "1.08"),
            ],
            replayRegimeCounts: [
                KeyValueRecord(key: "TREND_PERSISTENT", value: "21"),
            ],
            replayPostureCounts: [
                KeyValueRecord(key: "NORMAL", value: "19"),
            ],
            replayTopReasons: [
                KeyValueRecord(key: "Directional persistence elevated", value: "21"),
            ],
            replayTopPlugins: [
                KeyValueRecord(key: "ai_tft", value: "16"),
            ],
            recentTransitions: [],
            observationCount: 21
        )

        return AdaptiveRouterSnapshot(
            generatedAt: now,
            profileName: "continuous",
            replayHoursBack: 72,
            symbols: [eurusd, usdjpy]
        )
    }

    public static func dynamicEnsembleSnapshot(projectRoot: URL) -> DynamicEnsembleSnapshot {
        let now = Date()
        let eurusd = DynamicEnsembleSymbolSnapshot(
            symbol: "EURUSD",
            generatedAt: now.addingTimeInterval(-90),
            topRegime: "HIGH_VOL_EVENT",
            sessionLabel: "LONDON_NY_OVERLAP",
            tradePosture: "CAUTION",
            ensembleQuality: 0.44,
            abstainBias: 0.22,
            agreementScore: 0.31,
            contextFitScore: 0.58,
            dominantPluginShare: 0.41,
            buyProb: 0.37,
            sellProb: 0.29,
            skipProb: 0.34,
            finalScore: 0.08,
            finalAction: "SKIP",
            fallbackUsed: false,
            reasons: [
                "plugin_disagreement_elevated",
                "newspulse_caution_active",
                "ensemble_quality_caution",
            ],
            activePlugins: [
                DynamicEnsemblePluginState(name: "ai_gha", family: "memory", status: "ACTIVE", signal: "BUY", weight: 0.39, trust: 0.74, calibrationShrink: 0.88, reasons: ["adaptive_router_upweighted"]),
                DynamicEnsemblePluginState(name: "ai_tesseract", family: "transformer", status: "ACTIVE", signal: "BUY", weight: 0.27, trust: 0.58, calibrationShrink: 0.74, reasons: ["confidence_shrunk"]),
            ],
            downweightedPlugins: [
                DynamicEnsemblePluginState(name: "ai_tft", family: "transformer", status: "DOWNWEIGHTED", signal: "SELL", weight: 0.18, trust: 0.46, calibrationShrink: 0.69, reasons: ["directional_disagreement"]),
            ],
            suppressedPlugins: [
                DynamicEnsemblePluginState(name: "lin_pa", family: "rule", status: "SUPPRESSED", signal: "SELL", weight: 0.0, trust: 0.18, calibrationShrink: 0.62, reasons: ["trust_below_suppress_threshold"]),
            ],
            replayPostureCounts: [
                KeyValueRecord(key: "CAUTION", value: "11"),
                KeyValueRecord(key: "ABSTAIN_BIAS", value: "4"),
            ],
            replayActionCounts: [
                KeyValueRecord(key: "SKIP", value: "9"),
                KeyValueRecord(key: "BUY", value: "6"),
            ],
            replayStatusCounts: [
                KeyValueRecord(key: "ACTIVE", value: "24"),
                KeyValueRecord(key: "DOWNWEIGHTED", value: "9"),
                KeyValueRecord(key: "SUPPRESSED", value: "7"),
            ],
            replayTopReasons: [
                KeyValueRecord(key: "plugin_disagreement_elevated", value: "8"),
                KeyValueRecord(key: "newspulse_caution_active", value: "6"),
            ],
            replayTopDominantPlugins: [
                KeyValueRecord(key: "ai_gha", value: "10"),
                KeyValueRecord(key: "ai_tesseract", value: "4"),
            ],
            recentTransitions: [
                DynamicEnsembleTransition(type: "posture_change", fromValue: "NORMAL", toValue: "CAUTION", observedAt: now.addingTimeInterval(-1700)),
                DynamicEnsembleTransition(type: "action_change", fromValue: "BUY", toValue: "SKIP", observedAt: now.addingTimeInterval(-1600)),
            ],
            observationCount: 19,
            averageQuality: 0.52,
            maxAbstainBias: 0.38
        )

        let usdjpy = DynamicEnsembleSymbolSnapshot(
            symbol: "USDJPY",
            generatedAt: now.addingTimeInterval(-120),
            topRegime: "TREND_PERSISTENT",
            sessionLabel: "LONDON",
            tradePosture: "NORMAL",
            ensembleQuality: 0.71,
            abstainBias: 0.07,
            agreementScore: 0.74,
            contextFitScore: 0.66,
            dominantPluginShare: 0.48,
            buyProb: 0.56,
            sellProb: 0.18,
            skipProb: 0.26,
            finalScore: 0.38,
            finalAction: "BUY",
            fallbackUsed: false,
            reasons: [
                "strong_plugin_agreement",
            ],
            activePlugins: [
                DynamicEnsemblePluginState(name: "ai_tft", family: "transformer", status: "ACTIVE", signal: "BUY", weight: 0.48, trust: 0.92, calibrationShrink: 0.86, reasons: ["adaptive_router_upweighted"]),
                DynamicEnsemblePluginState(name: "tree_catboost", family: "tree", status: "ACTIVE", signal: "BUY", weight: 0.32, trust: 0.71, calibrationShrink: 0.91, reasons: ["Active in the current dynamic ensemble"]),
            ],
            downweightedPlugins: [],
            suppressedPlugins: [],
            replayPostureCounts: [
                KeyValueRecord(key: "NORMAL", value: "17"),
            ],
            replayActionCounts: [
                KeyValueRecord(key: "BUY", value: "12"),
                KeyValueRecord(key: "SKIP", value: "3"),
            ],
            replayStatusCounts: [
                KeyValueRecord(key: "ACTIVE", value: "20"),
            ],
            replayTopReasons: [
                KeyValueRecord(key: "strong_plugin_agreement", value: "11"),
            ],
            replayTopDominantPlugins: [
                KeyValueRecord(key: "ai_tft", value: "13"),
            ],
            recentTransitions: [],
            observationCount: 18,
            averageQuality: 0.67,
            maxAbstainBias: 0.21
        )

        return DynamicEnsembleSnapshot(
            generatedAt: now,
            replayHoursBack: 48,
            symbols: [eurusd, usdjpy]
        )
    }

}
