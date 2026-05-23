import Foundation

extension GUIValidationFixtures {
    public static func labelEngineSnapshot(projectRoot _: URL) -> LabelEngineSnapshot {
        let now = Date()
        let build = LabelEngineBuildSnapshot(
            datasetKey: "continuous:EURUSD:m1:labels",
            profileName: "continuous",
            symbol: "EURUSD",
            timeframe: "M1",
            barCount: 96,
            pointSize: 0.00001,
            executionProfile: "default",
            labelVersion: 1,
            generatedAt: now.addingTimeInterval(-320),
            summaryMetrics: [
                KeyValueRecord(key: "label_row_count", value: "224"),
                KeyValueRecord(key: "meta_label_count", value: "28"),
                KeyValueRecord(key: "candidate_count", value: "28"),
                KeyValueRecord(key: "long_tradeability_rate", value: "0.61"),
                KeyValueRecord(key: "short_tradeability_rate", value: "0.47"),
                KeyValueRecord(key: "meta_acceptance_rate", value: "0.43"),
            ],
            metaSummary: [
                KeyValueRecord(key: "candidate_mode", value: "BASELINE_MOMENTUM"),
                KeyValueRecord(key: "accepted_count", value: "12"),
                KeyValueRecord(key: "rejected_count", value: "16"),
                KeyValueRecord(key: "min_raw_signal_strength", value: "0.15"),
            ],
            qualityFlags: [
                KeyValueRecord(key: "path_approximation_used", value: "true"),
                KeyValueRecord(key: "partial_cost_model", value: "true"),
                KeyValueRecord(key: "external_candidates_used", value: "false"),
            ],
            artifactPaths: [
                KeyValueRecord(key: "bundle_json", value: "/tmp/label_bundle.json"),
                KeyValueRecord(key: "labels_ndjson", value: "/tmp/labels.ndjson"),
                KeyValueRecord(key: "meta_labels_ndjson", value: "/tmp/meta_labels.ndjson"),
            ],
            topReasons: [
                LabelEngineReasonCount(reason: "MOVE_TOO_SMALL_AFTER_COSTS", count: 44),
                LabelEngineReasonCount(reason: "ADVERSE_HIT_FIRST", count: 19),
            ],
            horizons: [
                LabelEngineHorizonSnapshot(
                    horizonID: "M5",
                    bars: 5,
                    sampleCount: 56,
                    longTradeabilityRate: 0.66,
                    shortTradeabilityRate: 0.52,
                    candidateCount: 8,
                    candidateAcceptanceRate: 0.50,
                    meanCostAdjustedReturnPoints: 3.2,
                    medianTimeToFavorableHitSec: 180
                ),
                LabelEngineHorizonSnapshot(
                    horizonID: "M15",
                    bars: 15,
                    sampleCount: 56,
                    longTradeabilityRate: 0.61,
                    shortTradeabilityRate: 0.47,
                    candidateCount: 8,
                    candidateAcceptanceRate: 0.43,
                    meanCostAdjustedReturnPoints: 4.8,
                    medianTimeToFavorableHitSec: 420
                ),
                LabelEngineHorizonSnapshot(
                    horizonID: "H1",
                    bars: 60,
                    sampleCount: 56,
                    longTradeabilityRate: 0.54,
                    shortTradeabilityRate: 0.38,
                    candidateCount: 12,
                    candidateAcceptanceRate: 0.33,
                    meanCostAdjustedReturnPoints: 6.4,
                    medianTimeToFavorableHitSec: 1320
                ),
            ]
        )
        return LabelEngineSnapshot(
            generatedAt: now,
            artifactCount: 1,
            latestDatasetKey: build.datasetKey,
            builds: [build],
            statusRecords: [
                KeyValueRecord(key: "artifact_count", value: "1"),
                KeyValueRecord(key: "latest_dataset_key", value: build.datasetKey),
                KeyValueRecord(key: "profile_name", value: "continuous"),
            ],
            artifactPaths: [
                KeyValueRecord(key: "report_json", value: "/tmp/label_engine_report.json"),
                KeyValueRecord(key: "runtime_summary_json", value: "/tmp/label_engine_summary.json"),
            ]
        )
    }

    public static func crossAssetSnapshot(projectRoot _: URL) -> CrossAssetSnapshot {
        let now = Date()
        return CrossAssetSnapshot(
            generatedAt: now,
            sourceStatuses: [
                CrossAssetSourceStatus(id: "rates", ok: true, stale: false, lastUpdateAt: now.addingTimeInterval(-90), proxySymbol: nil, availableSymbols: nil, configuredSymbols: nil),
                CrossAssetSourceStatus(id: "context_service", ok: true, stale: false, lastUpdateAt: now.addingTimeInterval(-45), proxySymbol: nil, availableSymbols: 18, configuredSymbols: 21),
                CrossAssetSourceStatus(id: "equities", ok: true, stale: false, lastUpdateAt: nil, proxySymbol: "US500", availableSymbols: nil, configuredSymbols: nil),
                CrossAssetSourceStatus(id: "commodities", ok: true, stale: false, lastUpdateAt: nil, proxySymbol: "XBRUSD", availableSymbols: nil, configuredSymbols: nil),
                CrossAssetSourceStatus(id: "volatility", ok: true, stale: false, lastUpdateAt: nil, proxySymbol: "BTCUSD", availableSymbols: nil, configuredSymbols: nil),
                CrossAssetSourceStatus(id: "liquidity", ok: true, stale: false, lastUpdateAt: nil, proxySymbol: "US10Y", availableSymbols: nil, configuredSymbols: nil),
            ],
            features: [
                KeyValueRecord(key: "front_end_rate_divergence_z", value: "1.42"),
                KeyValueRecord(key: "equity_risk_state_z", value: "1.11"),
                KeyValueRecord(key: "commodity_shock_oil_z", value: "0.84"),
                KeyValueRecord(key: "volatility_stress_z", value: "1.63"),
                KeyValueRecord(key: "usd_liquidity_stress_z", value: "0.98"),
                KeyValueRecord(key: "cross_asset_dislocation_z", value: "1.06"),
                KeyValueRecord(key: "global_macro_stress_z", value: "1.22"),
            ],
            stateScores: [
                KeyValueRecord(key: "rates_repricing_score", value: "0.74"),
                KeyValueRecord(key: "risk_off_score", value: "0.69"),
                KeyValueRecord(key: "commodity_shock_score", value: "0.51"),
                KeyValueRecord(key: "volatility_shock_score", value: "0.77"),
                KeyValueRecord(key: "usd_liquidity_stress_score", value: "0.66"),
                KeyValueRecord(key: "cross_asset_dislocation_score", value: "0.62"),
            ],
            stateLabels: [
                KeyValueRecord(key: "macro_state", value: "RATES_REPRICING"),
                KeyValueRecord(key: "risk_state", value: "RISK_OFF"),
                KeyValueRecord(key: "liquidity_state", value: "STRESSED"),
            ],
            selectedProxies: [
                CrossAssetProxySelection(id: "equities", symbol: "US500", fallbackUsed: false, available: true, changePct1d: -1.26, rangeRatio1d: 1.44),
                CrossAssetProxySelection(id: "oil", symbol: "XBRUSD", fallbackUsed: false, available: true, changePct1d: 1.08, rangeRatio1d: 1.22),
                CrossAssetProxySelection(id: "gold", symbol: "XAUUSD", fallbackUsed: false, available: true, changePct1d: 0.66, rangeRatio1d: 1.14),
                CrossAssetProxySelection(id: "volatility", symbol: "BTCUSD", fallbackUsed: true, available: true, changePct1d: 2.42, rangeRatio1d: 1.58),
                CrossAssetProxySelection(id: "dollar_liquidity", symbol: "US10Y", fallbackUsed: true, available: true, changePct1d: 0.73, rangeRatio1d: 1.10),
            ],
            pairs: [
                CrossAssetPairState(
                    pair: "EURUSD",
                    baseCurrency: "EUR",
                    quoteCurrency: "USD",
                    macroState: "RATES_REPRICING",
                    riskState: "RISK_OFF",
                    liquidityState: "STRESSED",
                    pairCrossAssetRiskScore: 0.81,
                    pairSensitivity: 0.79,
                    tradeGate: "BLOCK",
                    stale: false,
                    reasons: ["FRONT_END_RATES_DIVERGING", "USD_LIQUIDITY_PRESSURE_RISING", "CROSS_ASSET_DISLOCATION_ELEVATED"]
                ),
                CrossAssetPairState(
                    pair: "USDJPY",
                    baseCurrency: "USD",
                    quoteCurrency: "JPY",
                    macroState: "RATES_REPRICING",
                    riskState: "RISK_OFF",
                    liquidityState: "STRESSED",
                    pairCrossAssetRiskScore: 0.63,
                    pairSensitivity: 0.91,
                    tradeGate: "CAUTION",
                    stale: false,
                    reasons: ["RISK_SENTIMENT_SENSITIVE_PAIR", "USD_LIQUIDITY_SENSITIVE_PAIR"]
                ),
            ],
            recentTransitions: [
                CrossAssetTransition(type: "macro_state", target: "global", fromValue: "NORMAL", toValue: "RATES_REPRICING", observedAt: now.addingTimeInterval(-600)),
                CrossAssetTransition(type: "pair_gate", target: "EURUSD", fromValue: "CAUTION", toValue: "BLOCK", observedAt: now.addingTimeInterval(-420)),
            ],
            reasons: [
                "FRONT_END_RATES_DIVERGING",
                "EQUITY_RISK_PROXY_WEAK",
                "VOLATILITY_STRESS_ELEVATED",
                "USD_LIQUIDITY_PRESSURE_RISING",
            ],
            qualityFlags: [
                KeyValueRecord(key: "fallback_proxy_used", value: "true"),
                KeyValueRecord(key: "partial_data", value: "false"),
                KeyValueRecord(key: "data_stale", value: "false"),
            ],
            healthSummary: [
                KeyValueRecord(key: "pair_count", value: "54"),
                KeyValueRecord(key: "feature_count", value: "10"),
                KeyValueRecord(key: "snapshot_stale_after_sec", value: "900"),
            ],
            artifactPaths: [
                KeyValueRecord(key: "snapshot_json", value: "/tmp/cross_asset_snapshot.json"),
                KeyValueRecord(key: "snapshot_flat", value: "/tmp/cross_asset_snapshot_flat.tsv"),
                KeyValueRecord(key: "symbol_map_tsv", value: "/tmp/cross_asset_symbol_map.tsv"),
            ]
        )
    }

    public static func pairNetworkSnapshot(projectRoot _: URL) -> PairNetworkSnapshot {
        let now = Date()
        let eurusdDependencies = [
            PairNetworkDependencyEdge(
                sourcePair: "EURUSD",
                targetPair: "GBPUSD",
                combinedScore: 0.82,
                structuralScore: 0.78,
                empiricalScore: 0.74,
                correlation: 0.74,
                support: 288,
                relation: "SHARED_CURRENCY",
                sharedCurrencies: ["USD"]
            ),
            PairNetworkDependencyEdge(
                sourcePair: "EURUSD",
                targetPair: "AUDUSD",
                combinedScore: 0.76,
                structuralScore: 0.72,
                empiricalScore: 0.68,
                correlation: 0.68,
                support: 288,
                relation: "SHARED_CURRENCY",
                sharedCurrencies: ["USD"]
            ),
            PairNetworkDependencyEdge(
                sourcePair: "EURUSD",
                targetPair: "USDCHF",
                combinedScore: 0.73,
                structuralScore: 0.76,
                empiricalScore: 0.60,
                correlation: -0.60,
                support: 288,
                relation: "INVERSE",
                sharedCurrencies: ["USD"]
            ),
        ]

        return PairNetworkSnapshot(
            generatedAt: now,
            graphMode: "STRUCTURAL_PLUS_EMPIRICAL",
            actionMode: "AUTO_APPLY",
            pairCount: 54,
            currencyCount: 20,
            edgeCount: 420,
            fallbackGraphUsed: false,
            partialDependencyData: false,
            graphStale: false,
            symbols: [
                PairNetworkSymbolSnapshot(
                    symbol: "EURUSD",
                    generatedAt: now.addingTimeInterval(-60),
                    decision: "ALLOW_REDUCED",
                    fallbackGraphUsed: false,
                    partialDependencyData: false,
                    graphStale: false,
                    conflictScore: 0.68,
                    redundancyScore: 0.66,
                    contradictionScore: 0.18,
                    concentrationScore: 0.61,
                    currencyConcentration: 0.58,
                    factorConcentration: 0.55,
                    recommendedSizeMultiplier: 0.72,
                    preferredExpression: "",
                    currencyExposure: [
                        KeyValueRecord(key: "EUR", value: "1.4000"),
                        KeyValueRecord(key: "USD", value: "-2.1000"),
                        KeyValueRecord(key: "JPY", value: "0.7000"),
                    ],
                    factorExposure: [
                        KeyValueRecord(key: "usd_bloc", value: "-1.8800"),
                        KeyValueRecord(key: "risk_on", value: "0.8200"),
                        KeyValueRecord(key: "macro_shock", value: "0.6100"),
                    ],
                    reasons: [
                        "DUPLICATES_EXISTING_USD_SHORT_EXPOSURE",
                        "FACTOR_CONCENTRATION_ELEVATED",
                    ]
                ),
                PairNetworkSymbolSnapshot(
                    symbol: "NZDUSD",
                    generatedAt: now.addingTimeInterval(-45),
                    decision: "PREFER_ALTERNATIVE_EXPRESSION",
                    fallbackGraphUsed: false,
                    partialDependencyData: false,
                    graphStale: false,
                    conflictScore: 0.79,
                    redundancyScore: 0.83,
                    contradictionScore: 0.05,
                    concentrationScore: 0.57,
                    currencyConcentration: 0.60,
                    factorConcentration: 0.62,
                    recommendedSizeMultiplier: 0.0,
                    preferredExpression: "AUDUSD",
                    currencyExposure: [
                        KeyValueRecord(key: "NZD", value: "1.0000"),
                        KeyValueRecord(key: "USD", value: "-1.0000"),
                    ],
                    factorExposure: [
                        KeyValueRecord(key: "commodity_fx", value: "1.0600"),
                        KeyValueRecord(key: "risk_on", value: "1.0800"),
                    ],
                    reasons: [
                        "BETTER_ALTERNATIVE_EXPRESSION",
                        "HIGH_COMMODITY_BLOC_OVERLAP",
                    ]
                ),
                PairNetworkSymbolSnapshot(
                    symbol: "USDCHF",
                    generatedAt: now.addingTimeInterval(-30),
                    decision: "BLOCK_CONTRADICTORY",
                    fallbackGraphUsed: false,
                    partialDependencyData: false,
                    graphStale: false,
                    conflictScore: 0.88,
                    redundancyScore: 0.34,
                    contradictionScore: 0.88,
                    concentrationScore: 0.49,
                    currencyConcentration: 0.62,
                    factorConcentration: 0.57,
                    recommendedSizeMultiplier: 0.0,
                    preferredExpression: "",
                    currencyExposure: [
                        KeyValueRecord(key: "USD", value: "1.0000"),
                        KeyValueRecord(key: "CHF", value: "-1.0000"),
                    ],
                    factorExposure: [
                        KeyValueRecord(key: "usd_bloc", value: "1.0400"),
                        KeyValueRecord(key: "safe_haven", value: "-0.2000"),
                    ],
                    reasons: [
                        "DIRECT_SYMBOL_CONTRADICTION",
                        "CURRENCY_EXPOSURE_CONFLICT",
                    ]
                ),
            ],
            topEdges: eurusdDependencies + [
                PairNetworkDependencyEdge(
                    sourcePair: "AUDUSD",
                    targetPair: "NZDUSD",
                    combinedScore: 0.85,
                    structuralScore: 0.80,
                    empiricalScore: 0.77,
                    correlation: 0.77,
                    support: 288,
                    relation: "SHARED_CURRENCY",
                    sharedCurrencies: ["USD"]
                )
            ],
            pairSummaries: [
                PairNetworkPairSummary(
                    pair: "EURUSD",
                    baseCurrency: "EUR",
                    quoteCurrency: "USD",
                    factorSignature: [
                        KeyValueRecord(key: "usd_bloc", value: "1.2000"),
                        KeyValueRecord(key: "eur_rates", value: "0.8400"),
                        KeyValueRecord(key: "macro_shock", value: "0.2600"),
                    ],
                    topDependencies: eurusdDependencies
                ),
                PairNetworkPairSummary(
                    pair: "NZDUSD",
                    baseCurrency: "NZD",
                    quoteCurrency: "USD",
                    factorSignature: [
                        KeyValueRecord(key: "commodity_fx", value: "1.0600"),
                        KeyValueRecord(key: "risk_on", value: "1.0800"),
                        KeyValueRecord(key: "usd_bloc", value: "1.1800"),
                    ],
                    topDependencies: [
                        PairNetworkDependencyEdge(
                            sourcePair: "NZDUSD",
                            targetPair: "AUDUSD",
                            combinedScore: 0.85,
                            structuralScore: 0.80,
                            empiricalScore: 0.77,
                            correlation: 0.77,
                            support: 288,
                            relation: "SHARED_CURRENCY",
                            sharedCurrencies: ["USD"]
                        )
                    ]
                ),
            ],
            reasons: [
                "STRUCTURAL_PLUS_EMPIRICAL_GRAPH_READY",
            ],
            qualityFlags: [
                KeyValueRecord(key: "fallback_graph_used", value: "false"),
                KeyValueRecord(key: "partial_dependency_data", value: "false"),
                KeyValueRecord(key: "graph_stale", value: "false"),
            ],
            statusRecords: [
                KeyValueRecord(key: "action_mode", value: "AUTO_APPLY"),
                KeyValueRecord(key: "graph_mode", value: "STRUCTURAL_PLUS_EMPIRICAL"),
                KeyValueRecord(key: "pair_count", value: "54"),
                KeyValueRecord(key: "edge_count", value: "420"),
            ],
            artifactPaths: [
                KeyValueRecord(key: "config_path", value: "/tmp/pair_network_config.json"),
                KeyValueRecord(key: "runtime_config_path", value: "/tmp/pair_network_config.tsv"),
                KeyValueRecord(key: "report_path", value: "/tmp/pair_network_report.json"),
            ]
        )
    }

}
