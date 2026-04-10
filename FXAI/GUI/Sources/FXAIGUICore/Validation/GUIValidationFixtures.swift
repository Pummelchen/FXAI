import Foundation

public enum GUIValidationFixtures {
    public static func projectSnapshot(projectRoot: URL) -> FXAIProjectSnapshot {
        let now = Date()
        let plugins = [
            PluginDescriptor(name: "ai_tft", family: "Sequence", sourcePath: projectRoot.appendingPathComponent("Plugins/Sequence/ai_tft"), sourceKind: .folder),
            PluginDescriptor(name: "ai_patchtst", family: "Sequence", sourcePath: projectRoot.appendingPathComponent("Plugins/Sequence/ai_patchtst"), sourceKind: .folder),
            PluginDescriptor(name: "ai_gha", family: "Sequence", sourcePath: projectRoot.appendingPathComponent("Plugins/Sequence/ai_gha.mqh"), sourceKind: .file),
            PluginDescriptor(name: "ai_qcew", family: "Sequence", sourcePath: projectRoot.appendingPathComponent("Plugins/Sequence/ai_qcew.mqh"), sourceKind: .file),
            PluginDescriptor(name: "tree_catboost", family: "Tree", sourcePath: projectRoot.appendingPathComponent("Plugins/Tree/tree_catboost"), sourceKind: .folder),
            PluginDescriptor(name: "tree_lgbm", family: "Tree", sourcePath: projectRoot.appendingPathComponent("Plugins/Tree/tree_lgbm"), sourceKind: .folder),
            PluginDescriptor(name: "lin_pa", family: "Linear", sourcePath: projectRoot.appendingPathComponent("Plugins/Linear/lin_pa"), sourceKind: .folder),
            PluginDescriptor(name: "native_transformer", family: "TensorCore", sourcePath: projectRoot.appendingPathComponent("TensorCore"), sourceKind: .folder)
        ]
        let pluginFamilies = Dictionary(grouping: plugins, by: \.family)
            .map { family, familyPlugins in
                PluginFamilySummary(id: family, family: family, pluginCount: familyPlugins.count)
            }
            .sorted { lhs, rhs in
                if lhs.pluginCount == rhs.pluginCount {
                    return lhs.family < rhs.family
                }
                return lhs.pluginCount > rhs.pluginCount
            }

        return FXAIProjectSnapshot(
            projectRoot: projectRoot,
            generatedAt: now,
            buildTargets: [
                BuildTargetStatus(name: "FXAI.ex5", relativePath: "FXAI.ex5", exists: true, modifiedAt: now),
                BuildTargetStatus(name: "FXAI_AuditRunner.ex5", relativePath: "Tests/FXAI_AuditRunner.ex5", exists: true, modifiedAt: now),
                BuildTargetStatus(name: "FXAI_OfflineExportRunner.ex5", relativePath: "Tests/FXAI_OfflineExportRunner.ex5", exists: true, modifiedAt: now)
            ],
            pluginFamilies: pluginFamilies,
            plugins: plugins,
            reportCategories: [
                ReportCategorySummary(id: "audit", category: "Audit", fileCount: 16, latestModifiedAt: now),
                ReportCategorySummary(id: "profiles", category: "Profiles", fileCount: 9, latestModifiedAt: now.addingTimeInterval(-1800)),
                ReportCategorySummary(id: "research", category: "ResearchOS", fileCount: 27, latestModifiedAt: now.addingTimeInterval(-900)),
                ReportCategorySummary(id: "distillation", category: "Distillation", fileCount: 8, latestModifiedAt: now.addingTimeInterval(-7200))
            ],
            recentArtifacts: [
                ReportArtifact(category: "Audit", name: "audit_latest.summary.json", path: projectRoot.appendingPathComponent("Tools/Reports/audit_latest.summary.json"), modifiedAt: now, sizeBytes: 14_220),
                ReportArtifact(category: "Profiles", name: "EURUSD_profile.tsv", path: projectRoot.appendingPathComponent("Tools/OfflineLab/Profiles/EURUSD_profile.tsv"), modifiedAt: now.addingTimeInterval(-600), sizeBytes: 3_620),
                ReportArtifact(category: "ResearchOS", name: "research_os_dashboard.json", path: projectRoot.appendingPathComponent("Tools/OfflineLab/research_os_dashboard.json"), modifiedAt: now.addingTimeInterval(-480), sizeBytes: 24_918),
                ReportArtifact(category: "Distillation", name: "teacher_bundle.json", path: projectRoot.appendingPathComponent("Tools/OfflineLab/foundation_teacher_bundle.json"), modifiedAt: now.addingTimeInterval(-2500), sizeBytes: 42_112)
            ],
            runtimeProfiles: [
                RuntimeProfileSummary(
                    id: "EURUSD",
                    symbol: "EURUSD",
                    pluginName: "ai_tft",
                    profileName: "continuous",
                    promotionTier: "production",
                    runtimeMode: "live",
                    sourcePath: projectRoot.appendingPathComponent("Tools/OfflineLab/Profiles/EURUSD_profile.tsv")
                ),
                RuntimeProfileSummary(
                    id: "USDJPY",
                    symbol: "USDJPY",
                    pluginName: "tree_catboost",
                    profileName: "continuous",
                    promotionTier: "audit-approved",
                    runtimeMode: "demo",
                    sourcePath: projectRoot.appendingPathComponent("Tools/OfflineLab/Profiles/USDJPY_profile.tsv")
                )
            ],
            operatorSummary: OperatorSummary(
                profileName: "continuous",
                championCount: 6,
                deploymentCount: 2,
                latestReviewedAt: now.addingTimeInterval(-900)
            ),
            tursoSummary: TursoSummary(
                localDatabasePresent: true,
                localDatabasePath: projectRoot.appendingPathComponent("Tools/OfflineLab/fxai_offline_lab.turso.db"),
                embeddedReplicaConfigured: true,
                encryptionConfigured: true
            )
        )
    }

    public static func runtimeSnapshot(projectRoot: URL) -> RuntimeOperationsSnapshot {
        let now = Date()
        let deploymentSections = [
            RuntimeArtifactSection(
                title: "Deployment Profile",
                sourcePath: projectRoot.appendingPathComponent("Tools/OfflineLab/Profiles/EURUSD_profile.tsv"),
                values: [
                    KeyValueRecord(key: "runtime_mode", value: "live"),
                    KeyValueRecord(key: "plugin_name", value: "ai_tft"),
                    KeyValueRecord(key: "policy_trade_floor", value: "0.74"),
                    KeyValueRecord(key: "policy_no_trade_cap", value: "0.22"),
                    KeyValueRecord(key: "budget_multiplier", value: "1.10"),
                    KeyValueRecord(key: "entry_floor", value: "0.61")
                ]
            )
        ]

        let supervisorSections = [
            RuntimeArtifactSection(
                title: "Supervisor",
                sourcePath: projectRoot.appendingPathComponent("Tools/OfflineLab/supervisor_service.json"),
                values: [
                    KeyValueRecord(key: "supervisor_blend", value: "0.81"),
                    KeyValueRecord(key: "reduce_bias", value: "0.27"),
                    KeyValueRecord(key: "exit_bias", value: "0.38")
                ]
            )
        ]

        let worldSections = [
            RuntimeArtifactSection(
                title: "World Plan",
                sourcePath: projectRoot.appendingPathComponent("Tools/OfflineLab/world_plan_EURUSD.json"),
                values: [
                    KeyValueRecord(key: "sigma_scale", value: "1.14"),
                    KeyValueRecord(key: "spread_scale", value: "1.07"),
                    KeyValueRecord(key: "shock_decay", value: "0.42")
                ]
            )
        ]

        let eurusd = RuntimeDeploymentDetail(
            id: "EURUSD",
            symbol: "EURUSD",
            profileName: "continuous",
            pluginName: "ai_tft",
            promotionTier: "production",
            runtimeMode: "live",
            createdAt: now.addingTimeInterval(-7200),
            reviewedAt: now.addingTimeInterval(-900),
            artifactHealth: RuntimeArtifactHealth(
                artifactExists: true,
                staleArtifact: false,
                missingDeployment: false,
                missingRouter: false,
                missingSupervisorService: false,
                missingSupervisorCommand: false,
                missingWorldPlan: false,
                artifactAgeSeconds: 240,
                performanceFailures: [],
                artifactSizeFailures: []
            ),
            deploymentPath: projectRoot.appendingPathComponent("Tools/OfflineLab/Profiles/EURUSD_profile.tsv"),
            studentRouterPath: projectRoot.appendingPathComponent("Tools/OfflineLab/student_router_EURUSD.tsv"),
            supervisorServicePath: projectRoot.appendingPathComponent("Tools/OfflineLab/supervisor_service.json"),
            supervisorCommandPath: projectRoot.appendingPathComponent("Tools/OfflineLab/supervisor_command.json"),
            worldPlanPath: projectRoot.appendingPathComponent("Tools/OfflineLab/world_plan_EURUSD.json"),
            deploymentSections: deploymentSections,
            routerSections: [
                RuntimeArtifactSection(
                    title: "Student Router",
                    sourcePath: projectRoot.appendingPathComponent("Tools/OfflineLab/student_router_EURUSD.tsv"),
                    values: [
                        KeyValueRecord(key: "teacher_signal_gain", value: "1.18"),
                        KeyValueRecord(key: "student_signal_gain", value: "0.94")
                    ]
                )
            ],
            supervisorSections: supervisorSections,
            commandSections: [
                RuntimeArtifactSection(
                    title: "Supervisor Commands",
                    sourcePath: projectRoot.appendingPathComponent("Tools/OfflineLab/supervisor_command.json"),
                    values: [
                        KeyValueRecord(key: "mode", value: "coordinated"),
                        KeyValueRecord(key: "net_bias_limit", value: "0.46")
                    ]
                )
            ],
            worldSections: worldSections,
            attributionSections: [
                RuntimeArtifactSection(
                    title: "Attribution",
                    sourcePath: projectRoot.appendingPathComponent("Tools/OfflineLab/attribution_EURUSD.tsv"),
                    values: [
                        KeyValueRecord(key: "macro_family", value: "0.33"),
                        KeyValueRecord(key: "trend_family", value: "0.41")
                    ]
                )
            ],
            featureHighlights: [
                KeyValueRecord(key: "macro_surprise", value: "0.29"),
                KeyValueRecord(key: "session_pressure", value: "0.41")
            ],
            studentRouterWeights: [
                KeyValueRecord(key: "ai_tft", value: "0.58"),
                KeyValueRecord(key: "tree_catboost", value: "0.24")
            ],
            familyWeights: [
                KeyValueRecord(key: "Sequence", value: "0.67"),
                KeyValueRecord(key: "Tree", value: "0.21")
            ],
            prunedPlugins: ["lin_pa", "tree_lgbm"]
        )

        let usdjpy = RuntimeDeploymentDetail(
            id: "USDJPY",
            symbol: "USDJPY",
            profileName: "continuous",
            pluginName: "tree_catboost",
            promotionTier: "audit-approved",
            runtimeMode: "demo",
            createdAt: now.addingTimeInterval(-10400),
            reviewedAt: now.addingTimeInterval(-3600),
            artifactHealth: RuntimeArtifactHealth(
                artifactExists: true,
                staleArtifact: false,
                missingDeployment: false,
                missingRouter: false,
                missingSupervisorService: false,
                missingSupervisorCommand: false,
                missingWorldPlan: false,
                artifactAgeSeconds: 340,
                performanceFailures: [],
                artifactSizeFailures: []
            ),
            deploymentPath: projectRoot.appendingPathComponent("Tools/OfflineLab/Profiles/USDJPY_profile.tsv"),
            studentRouterPath: projectRoot.appendingPathComponent("Tools/OfflineLab/student_router_USDJPY.tsv"),
            supervisorServicePath: projectRoot.appendingPathComponent("Tools/OfflineLab/supervisor_service.json"),
            supervisorCommandPath: projectRoot.appendingPathComponent("Tools/OfflineLab/supervisor_command.json"),
            worldPlanPath: projectRoot.appendingPathComponent("Tools/OfflineLab/world_plan_USDJPY.json"),
            deploymentSections: [
                RuntimeArtifactSection(
                    title: "Deployment Profile",
                    sourcePath: projectRoot.appendingPathComponent("Tools/OfflineLab/Profiles/USDJPY_profile.tsv"),
                    values: [
                        KeyValueRecord(key: "runtime_mode", value: "demo"),
                        KeyValueRecord(key: "plugin_name", value: "tree_catboost"),
                        KeyValueRecord(key: "policy_trade_floor", value: "0.69")
                    ]
                )
            ],
            routerSections: [],
            supervisorSections: supervisorSections,
            commandSections: [],
            worldSections: worldSections,
            attributionSections: [],
            featureHighlights: [KeyValueRecord(key: "carry_pressure", value: "0.38")],
            studentRouterWeights: [KeyValueRecord(key: "tree_catboost", value: "0.62")],
            familyWeights: [KeyValueRecord(key: "Tree", value: "0.59")],
            prunedPlugins: ["ai_patchtst"]
        )

        return RuntimeOperationsSnapshot(
            generatedAt: now,
            profileName: "continuous",
            deployments: [eurusd, usdjpy],
            champions: [
                PromotionChampionRecord(
                    symbol: "EURUSD",
                    pluginName: "ai_tft",
                    status: "champion",
                    promotionTier: "production",
                    championScore: 0.82,
                    challengerScore: 0.71,
                    portfolioScore: 0.78,
                    reviewedAt: now.addingTimeInterval(-900),
                    setPath: projectRoot.appendingPathComponent("Profiles/Tester/EURUSD_ai_tft.set"),
                    profileName: "continuous"
                ),
                PromotionChampionRecord(
                    symbol: "USDJPY",
                    pluginName: "tree_catboost",
                    status: "champion",
                    promotionTier: "audit-approved",
                    championScore: 0.74,
                    challengerScore: 0.69,
                    portfolioScore: 0.71,
                    reviewedAt: now.addingTimeInterval(-3600),
                    setPath: projectRoot.appendingPathComponent("Profiles/Tester/USDJPY_tree_catboost.set"),
                    profileName: "continuous"
                )
            ]
        )
    }

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
            spreadRegime: "ELEVATED",
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
                "Spread regime elevated",
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
                KeyValueRecord(key: "Spread regime elevated", value: "12"),
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
            spreadRegime: "NORMAL",
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
            spreadCostPoints: 1.6,
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
            spreadCostPoints: 1.4,
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

    public static func researchSnapshot(projectRoot: URL) -> ResearchOSControlSnapshot {
        let now = Date()
        return ResearchOSControlSnapshot(
            generatedAt: now,
            profileName: "continuous",
            environment: ResearchOSEnvironmentStatus(
                backend: "turso",
                syncMode: "embedded-replica",
                databasePath: projectRoot.appendingPathComponent("Tools/OfflineLab/fxai_offline_lab.turso.db"),
                databaseName: "fxai-offline-lab",
                organizationSlug: "fxai",
                groupName: "default",
                locationName: "fra",
                cliConfigPath: projectRoot.appendingPathComponent("Tools/OfflineLab/turso.toml"),
                syncIntervalSeconds: 30,
                encryptionEnabled: true,
                platformAPIEnabled: true,
                syncEnabled: true,
                authTokenConfigured: true,
                apiTokenConfigured: true,
                configError: nil
            ),
            branches: [
                ResearchOSBranchRecord(
                    name: "continuous",
                    sourceDatabase: "fxai-offline-lab",
                    parentName: "main",
                    branchKind: "champion",
                    status: "active",
                    groupName: "default",
                    locationName: "fra",
                    hostname: "fxai.turso.io",
                    syncURL: "libsql://fxai.turso.io",
                    envArtifactPath: projectRoot.appendingPathComponent("Tools/OfflineLab/branches/continuous.env"),
                    isBranch: true,
                    createdAt: now.addingTimeInterval(-86_400),
                    sourceTimestamp: "2026-04-05T07:00:00Z"
                )
            ],
            auditEvents: [
                ResearchOSAuditEvent(
                    organizationSlug: "fxai",
                    eventID: "evt-001",
                    eventType: "branch.created",
                    targetName: "continuous",
                    occurredAt: now.addingTimeInterval(-12_000),
                    observedAt: now.addingTimeInterval(-11_900)
                ),
                ResearchOSAuditEvent(
                    organizationSlug: "fxai",
                    eventID: "evt-002",
                    eventType: "governance.promoted",
                    targetName: "EURUSD",
                    occurredAt: now.addingTimeInterval(-2_400),
                    observedAt: now.addingTimeInterval(-2_350)
                )
            ],
            symbols: [
                ResearchOSSymbolControl(
                    symbol: "EURUSD",
                    analogNeighbors: [
                        ResearchOSAnalogNeighbor(
                            sourceKey: "EURUSD::ai_tft",
                            pluginName: "ai_tft",
                            distance: 0.12,
                            similarity: 0.88,
                            score: 0.79,
                            sourceType: "shadow",
                            scope: "live",
                            payload: [
                                KeyValueRecord(key: "session", value: "London"),
                                KeyValueRecord(key: "regime", value: "trend")
                            ]
                        )
                    ],
                    deploymentArtifactPath: projectRoot.appendingPathComponent("Tools/OfflineLab/Profiles/EURUSD_profile.tsv"),
                    deploymentCreatedAt: now.addingTimeInterval(-900)
                ),
                ResearchOSSymbolControl(
                    symbol: "USDJPY",
                    analogNeighbors: [
                        ResearchOSAnalogNeighbor(
                            sourceKey: "USDJPY::tree_catboost",
                            pluginName: "tree_catboost",
                            distance: 0.19,
                            similarity: 0.81,
                            score: 0.72,
                            sourceType: "audit",
                            scope: "research",
                            payload: [
                                KeyValueRecord(key: "session", value: "Tokyo"),
                                KeyValueRecord(key: "regime", value: "carry")
                            ]
                        )
                    ],
                    deploymentArtifactPath: projectRoot.appendingPathComponent("Tools/OfflineLab/Profiles/USDJPY_profile.tsv"),
                    deploymentCreatedAt: now.addingTimeInterval(-3600)
                )
            ],
            sourceOfTruth: [
                KeyValueRecord(key: "profiles", value: "FILE_COMMON/FXAI/Offline/Promotions"),
                KeyValueRecord(key: "database", value: "Turso"),
                KeyValueRecord(key: "runtime", value: "MT5 FILE_COMMON")
            ]
        )
    }

    public static func visualizationSnapshot(projectRoot: URL) -> AdvancedVisualizationSnapshot {
        let now = Date()
        return AdvancedVisualizationSnapshot(
            generatedAt: now,
            profileName: "continuous",
            familyStressHeatmap: VisualizationHeatmap(
                title: "Family Stress",
                subtitle: "Cross-family stress and fit",
                rowLabels: ["Sequence", "Tree", "Linear"],
                columnLabels: ["Macro", "Trend", "Carry"],
                values: [
                    [0.71, 0.84, 0.43],
                    [0.52, 0.66, 0.81],
                    [0.33, 0.48, 0.54]
                ]
            ),
            symbolDetails: [
                SymbolVisualizationDetail(
                    symbol: "EURUSD",
                    worldSessionScales: [
                        VisualizationSeriesPoint(label: "Asia", value: 0.72),
                        VisualizationSeriesPoint(label: "London", value: 1.08),
                        VisualizationSeriesPoint(label: "New York", value: 0.91)
                    ],
                    worldStressMetrics: [
                        VisualizationSeriesPoint(label: "Sigma", value: 1.14),
                        VisualizationSeriesPoint(label: "Spread", value: 1.07),
                        VisualizationSeriesPoint(label: "Shock", value: 0.42)
                    ],
                    familyWeights: [
                        VisualizationSeriesPoint(label: "Sequence", value: 0.67),
                        VisualizationSeriesPoint(label: "Tree", value: 0.21)
                    ],
                    featureWeights: [
                        VisualizationSeriesPoint(label: "Macro", value: 0.33),
                        VisualizationSeriesPoint(label: "Session", value: 0.41)
                    ],
                    pluginWeights: [
                        VisualizationSeriesPoint(label: "ai_tft", value: 0.58),
                        VisualizationSeriesPoint(label: "tree_catboost", value: 0.24)
                    ],
                    artifactDiffHeatmap: VisualizationHeatmap(
                        title: "Artifact Diff",
                        subtitle: "Router / supervisor / world alignment",
                        rowLabels: ["Router", "Supervisor", "World"],
                        columnLabels: ["Freshness", "Fit", "Risk"],
                        values: [
                            [0.12, 0.84, 0.26],
                            [0.08, 0.78, 0.31],
                            [0.14, 0.69, 0.35]
                        ]
                    ),
                    timeline: [
                        VisualizationTimelineEvent(category: "promotion", title: "Champion Promoted", detail: "ai_tft promoted for EURUSD", date: now.addingTimeInterval(-3600), score: 0.82)
                    ],
                    weakScenarios: ["Macro shock cluster", "Late NY spread widening"]
                )
            ],
            globalTimeline: [
                VisualizationTimelineEvent(category: "governance", title: "Continuous cycle", detail: "Promotion review completed", date: now.addingTimeInterval(-5400), score: 0.78)
            ]
        )
    }

    public static func incidentSnapshot(projectRoot: URL) -> IncidentCenterSnapshot {
        IncidentCenterSnapshot(
            generatedAt: Date(),
            incidents: [
                FXAIIncident(
                    id: "incident-runtime-freshness",
                    severity: .warning,
                    category: .runtime,
                    title: "Runtime artifact freshness drift",
                    summary: "One deployment profile is close to the freshness threshold.",
                    affectedSymbol: "USDJPY",
                    detailLines: ["Supervisor snapshot age is increasing."],
                    actions: [
                        IncidentAction(title: "Open Runtime", summary: "Inspect runtime deployment sections", command: "python3 Tools/fxai_testlab.py verify-all", destinationSelection: "runtimeMonitor")
                    ],
                    playbook: RecoveryPlaybook(
                        title: "Refresh runtime artifacts",
                        summary: "Regenerate and verify the deployed runtime state.",
                        steps: [
                            RecoveryStep(title: "Verify all", summary: "Confirm clean compiles and fixtures.", command: "python3 Tools/fxai_testlab.py verify-all", destinationSelection: "runtimeMonitor")
                        ]
                    )
                ),
                FXAIIncident(
                    id: "incident-research-branch",
                    severity: .info,
                    category: .researchOS,
                    title: "Research branch ready",
                    summary: "A continuous branch is available for operator review.",
                    actions: [
                        IncidentAction(title: "Open Research OS", summary: "Inspect branch status and audit events", command: "python3 Tools/fxai_offline_lab.py dashboard", destinationSelection: "researchControl")
                    ],
                    playbook: RecoveryPlaybook(
                        title: "Review branch health",
                        summary: "Inspect recent Research OS branch and audit state.",
                        steps: [
                            RecoveryStep(title: "Open Research OS", summary: "Review branch inventory.", command: "python3 Tools/fxai_offline_lab.py branch-inventory", destinationSelection: "researchControl")
                        ]
                    )
                )
            ]
        )
    }

    public static func ratesEngineSnapshot(projectRoot: URL) -> RatesEngineSnapshot {
        let now = Date()
        return RatesEngineSnapshot(
            generatedAt: now,
            sourceStatuses: [
                RatesEngineSourceStatus(id: "manual_inputs", ok: true, stale: false, enabled: true, required: false, lastUpdateAt: now.addingTimeInterval(-1800), mode: "manual_market_input", coverageRatio: 0.3, updatedCurrencies: 3),
                RatesEngineSourceStatus(id: "proxy_engine", ok: true, stale: false, enabled: true, required: false, lastUpdateAt: now.addingTimeInterval(-120), mode: "newspulse_policy_proxy", coverageRatio: nil, updatedCurrencies: nil),
                RatesEngineSourceStatus(id: "newspulse", ok: true, stale: false, enabled: true, required: false, lastUpdateAt: now.addingTimeInterval(-120), mode: "shared_context", coverageRatio: nil, updatedCurrencies: nil),
            ],
            currencies: [
                RatesEngineCurrencyState(
                    currency: "USD",
                    frontEndLevel: 0.84,
                    frontEndBasis: "policy_proxy_index",
                    frontEndChange1d: 0.16,
                    frontEndChange5d: 0.42,
                    expectedPathLevel: 1.08,
                    expectedPathBasis: "policy_proxy_index",
                    expectedPathChange1d: 0.24,
                    expectedPathChange5d: 0.56,
                    curveSlope2s10s: nil,
                    curveBasis: "unavailable",
                    curveShapeRegime: "UNAVAILABLE",
                    policyRepricingScore: 0.81,
                    policySurpriseScore: 0.62,
                    policyUncertaintyScore: 0.44,
                    policyDirectionScore: 0.55,
                    policyRelevanceScore: 0.78,
                    preCBEventWindow: false,
                    postCBEventWindow: true,
                    preMacroPolicyWindow: false,
                    postMacroPolicyWindow: true,
                    meetingPathRepriceNow: true,
                    macroToRatesTransmissionScore: 0.69,
                    stale: false,
                    reasons: ["USD central-bank repricing window active"]
                ),
                RatesEngineCurrencyState(
                    currency: "EUR",
                    frontEndLevel: 0.22,
                    frontEndBasis: "manual_market_input",
                    frontEndChange1d: -0.08,
                    frontEndChange5d: -0.14,
                    expectedPathLevel: 0.18,
                    expectedPathBasis: "manual_market_input",
                    expectedPathChange1d: -0.05,
                    expectedPathChange5d: -0.12,
                    curveSlope2s10s: -0.34,
                    curveBasis: "manual_market_input",
                    curveShapeRegime: "INVERSION_LIKE",
                    policyRepricingScore: 0.36,
                    policySurpriseScore: 0.18,
                    policyUncertaintyScore: 0.24,
                    policyDirectionScore: -0.14,
                    policyRelevanceScore: 0.33,
                    preCBEventWindow: false,
                    postCBEventWindow: false,
                    preMacroPolicyWindow: false,
                    postMacroPolicyWindow: false,
                    meetingPathRepriceNow: false,
                    macroToRatesTransmissionScore: 0.22,
                    stale: false,
                    reasons: ["EUR manual front-end and curve inputs available"]
                ),
            ],
            pairs: [
                RatesEnginePairState(
                    pair: "EURUSD",
                    baseCurrency: "EUR",
                    quoteCurrency: "USD",
                    frontEndDiff: -0.62,
                    expectedPathDiff: -0.90,
                    curveDivergenceScore: 0.21,
                    policyDivergenceScore: 0.74,
                    ratesRegime: "UNSTABLE",
                    ratesRiskScore: 0.79,
                    tradeGate: "BLOCK",
                    policyAlignment: "quote_hawkish",
                    meetingPathRepriceNow: true,
                    macroToRatesTransmissionScore: 0.69,
                    stale: false,
                    brokerSymbols: ["EURUSD"],
                    reasons: ["meeting path repricing active", "policy divergence meaningful"]
                ),
                RatesEnginePairState(
                    pair: "USDJPY",
                    baseCurrency: "USD",
                    quoteCurrency: "JPY",
                    frontEndDiff: 0.48,
                    expectedPathDiff: 0.54,
                    curveDivergenceScore: 0.11,
                    policyDivergenceScore: 0.58,
                    ratesRegime: "SUPPORTIVE",
                    ratesRiskScore: 0.38,
                    tradeGate: "ALLOW",
                    policyAlignment: "base_hawkish",
                    meetingPathRepriceNow: false,
                    macroToRatesTransmissionScore: 0.47,
                    stale: false,
                    brokerSymbols: ["USDJPY"],
                    reasons: ["policy divergence meaningful"]
                ),
            ],
            recentPolicyEvents: [
                RatesEnginePolicyEvent(
                    id: "evt-usd-fed",
                    currency: "USD",
                    source: "official",
                    domain: "federalreserve.gov",
                    publishedAt: now.addingTimeInterval(-600),
                    title: "Federal Reserve policy statement",
                    url: URL(string: "https://example.test/fed"),
                    policyRelevanceScore: 0.91,
                    direction: 0.56,
                    centralBankEvent: true,
                    macroPolicyEvent: false
                ),
                RatesEnginePolicyEvent(
                    id: "evt-eur-cpi",
                    currency: "EUR",
                    source: "calendar",
                    domain: "mt5-calendar",
                    publishedAt: now.addingTimeInterval(-1800),
                    title: "Euro Area CPI",
                    url: nil,
                    policyRelevanceScore: 0.62,
                    direction: -0.18,
                    centralBankEvent: false,
                    macroPolicyEvent: true
                ),
            ],
            healthSummary: [
                KeyValueRecord(key: "pair_count", value: "2"),
                KeyValueRecord(key: "currency_count", value: "2"),
            ],
            artifactPaths: [
                KeyValueRecord(key: "snapshot_json", value: projectRoot.appendingPathComponent("Tools/OfflineLab/RatesEngine/rates_engine_status.json").path),
            ]
        )
    }

    public static func microstructureSnapshot(projectRoot: URL) -> MicrostructureSnapshot {
        let now = Date()
        return MicrostructureSnapshot(
            generatedAt: now,
            serviceStatus: MicrostructureServiceStatus(
                ok: true,
                stale: false,
                enabled: true,
                pollIntervalMS: 5000,
                symbolRefreshSec: 300,
                snapshotStaleAfterSec: 45,
                lastPollAt: now.addingTimeInterval(-5),
                lastSuccessAt: now.addingTimeInterval(-5),
                lastSymbolRefreshAt: now.addingTimeInterval(-120),
                lastError: nil
            ),
            symbols: [
                MicrostructureSymbolState(
                    symbol: "EURUSD",
                    brokerSymbol: "EURUSD",
                    available: true,
                    stale: false,
                    generatedAt: now,
                    spreadCurrent: 0.8,
                    silentGapSecondsCurrent: 0.3,
                    sessionTag: "LONDON_NEWYORK_OVERLAP",
                    handoffFlag: false,
                    minutesSinceSessionOpen: 74,
                    minutesToSessionClose: 166,
                    sessionOpenBurstScore: 0.24,
                    sessionSpreadBehaviorScore: 0.18,
                    liquidityStressScore: 0.32,
                    hostileExecutionScore: 0.28,
                    microstructureRegime: "TRENDING_CLEAN",
                    tradeGate: "ALLOW",
                    tickImbalance30s: 0.34,
                    directionalEfficiency60s: 0.72,
                    spreadZScore60s: 0.44,
                    tickRate60s: 126,
                    tickRateZScore60s: 1.18,
                    realizedVol5m: 0.61,
                    volBurstScore5m: 1.12,
                    localExtremaBreachScore60s: 0.21,
                    sweepAndRejectFlag60s: false,
                    breakoutReversalScore60s: 0.18,
                    exhaustionProxy60s: 0.22,
                    reasons: ["Tick imbalance supports a clean short-horizon trend"]
                ),
                MicrostructureSymbolState(
                    symbol: "GBPJPY",
                    brokerSymbol: "GBPJPY",
                    available: true,
                    stale: false,
                    generatedAt: now,
                    spreadCurrent: 2.6,
                    silentGapSecondsCurrent: 1.6,
                    sessionTag: "LONDON_NEWYORK_OVERLAP",
                    handoffFlag: true,
                    minutesSinceSessionOpen: 12,
                    minutesToSessionClose: 18,
                    sessionOpenBurstScore: 0.74,
                    sessionSpreadBehaviorScore: 0.66,
                    liquidityStressScore: 0.79,
                    hostileExecutionScore: 0.72,
                    microstructureRegime: "STOP_RUN_RISK",
                    tradeGate: "CAUTION",
                    tickImbalance30s: -0.11,
                    directionalEfficiency60s: 0.37,
                    spreadZScore60s: 2.24,
                    tickRate60s: 188,
                    tickRateZScore60s: 2.11,
                    realizedVol5m: 1.42,
                    volBurstScore5m: 1.91,
                    localExtremaBreachScore60s: 0.76,
                    sweepAndRejectFlag60s: true,
                    breakoutReversalScore60s: 0.81,
                    exhaustionProxy60s: 0.73,
                    reasons: ["Recent breakout rejection detected", "Spread instability elevated"]
                ),
            ],
            healthSummary: [
                KeyValueRecord(key: "active_symbol_count", value: "2"),
                KeyValueRecord(key: "snapshot_stale_after_sec", value: "45"),
            ],
            artifactPaths: [
                KeyValueRecord(key: "snapshot_json", value: projectRoot.appendingPathComponent("FILE_COMMON/FXAI/Runtime/microstructure_snapshot.json").path),
                KeyValueRecord(key: "history_ndjson", value: projectRoot.appendingPathComponent("FILE_COMMON/FXAI/Runtime/microstructure_history.ndjson").path),
            ]
        )
    }
}
