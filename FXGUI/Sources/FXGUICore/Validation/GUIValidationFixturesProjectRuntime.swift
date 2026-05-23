import Foundation

extension GUIValidationFixtures {
    public static func projectSnapshot(projectRoot: URL) -> FXAIProjectSnapshot {
        let now = Date()
        let plugins = [
            PluginDescriptor(name: "ai_tft", family: "Sequence", sourcePath: projectRoot.appendingPathComponent("FXPlugins/ai_tft/AITFTPlugin.swift"), sourceKind: .file),
            PluginDescriptor(name: "ai_patchtst", family: "Sequence", sourcePath: projectRoot.appendingPathComponent("FXPlugins/ai_patchtst/AIPatchtstPlugin.swift"), sourceKind: .file),
            PluginDescriptor(name: "ai_gha", family: "Sequence", sourcePath: projectRoot.appendingPathComponent("FXPlugins/ai_gha/AIGHAPlugin.swift"), sourceKind: .file),
            PluginDescriptor(name: "ai_qcew", family: "Sequence", sourcePath: projectRoot.appendingPathComponent("FXPlugins/ai_qcew/AIQCEWPlugin.swift"), sourceKind: .file),
            PluginDescriptor(name: "tree_catboost", family: "Tree", sourcePath: projectRoot.appendingPathComponent("FXPlugins/tree_catboost/TreeCatboostPlugin.swift"), sourceKind: .file),
            PluginDescriptor(name: "tree_lgbm", family: "Tree", sourcePath: projectRoot.appendingPathComponent("FXPlugins/tree_lgbm/TreeLgbmPlugin.swift"), sourceKind: .file),
            PluginDescriptor(name: "lin_pa", family: "Linear", sourcePath: projectRoot.appendingPathComponent("FXPlugins/lin_pa/LinPAPlugin.swift"), sourceKind: .file),
            PluginDescriptor(name: "fxbacktest_moving_average_cross", family: "Demo", sourcePath: projectRoot.appendingPathComponent("FXPlugins/fxbacktest_moving_average_cross/MovingAverageCrossFXDataEnginePlugin.swift"), sourceKind: .file)
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
                BuildTargetStatus(name: "FXDataEngine Swift Package", relativePath: "FXDataEngine/Package.swift", exists: true, modifiedAt: now),
                BuildTargetStatus(name: "FXPlugins Swift Package", relativePath: "FXPlugins/Package.swift", exists: true, modifiedAt: now),
                BuildTargetStatus(name: "FXBacktest Swift Package", relativePath: "FXBacktest/Package.swift", exists: true, modifiedAt: now),
                BuildTargetStatus(name: "FXDatabase Swift Package", relativePath: "FXDatabase/Package.swift", exists: true, modifiedAt: now)
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
                ReportArtifact(category: "Audit", name: "audit_latest.summary.json", path: projectRoot.appendingPathComponent("FXDataEngine/Tools/Reports/audit_latest.summary.json"), modifiedAt: now, sizeBytes: 14_220),
                ReportArtifact(category: "Profiles", name: "EURUSD_profile.tsv", path: projectRoot.appendingPathComponent("FXDataEngine/Tools/OfflineLab/Profiles/EURUSD_profile.tsv"), modifiedAt: now.addingTimeInterval(-600), sizeBytes: 3_620),
                ReportArtifact(category: "ResearchOS", name: "research_os_dashboard.json", path: projectRoot.appendingPathComponent("FXDataEngine/Tools/OfflineLab/research_os_dashboard.json"), modifiedAt: now.addingTimeInterval(-480), sizeBytes: 24_918),
                ReportArtifact(category: "Distillation", name: "teacher_bundle.json", path: projectRoot.appendingPathComponent("FXDataEngine/Tools/OfflineLab/foundation_teacher_bundle.json"), modifiedAt: now.addingTimeInterval(-2500), sizeBytes: 42_112)
            ],
            runtimeProfiles: [
                RuntimeProfileSummary(
                    id: "EURUSD",
                    symbol: "EURUSD",
                    pluginName: "ai_tft",
                    profileName: "continuous",
                    promotionTier: "production",
                    runtimeMode: "live",
                    sourcePath: projectRoot.appendingPathComponent("FXDataEngine/Tools/OfflineLab/Profiles/EURUSD_profile.tsv")
                ),
                RuntimeProfileSummary(
                    id: "USDJPY",
                    symbol: "USDJPY",
                    pluginName: "tree_catboost",
                    profileName: "continuous",
                    promotionTier: "audit-approved",
                    runtimeMode: "demo",
                    sourcePath: projectRoot.appendingPathComponent("FXDataEngine/Tools/OfflineLab/Profiles/USDJPY_profile.tsv")
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
                localDatabasePath: projectRoot.appendingPathComponent("FXDataEngine/Tools/OfflineLab/fxai_offline_lab.turso.db"),
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
                sourcePath: projectRoot.appendingPathComponent("FXDataEngine/Tools/OfflineLab/Profiles/EURUSD_profile.tsv"),
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
                sourcePath: projectRoot.appendingPathComponent("FXDataEngine/Tools/OfflineLab/supervisor_service.json"),
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
                sourcePath: projectRoot.appendingPathComponent("FXDataEngine/Tools/OfflineLab/world_plan_EURUSD.json"),
                values: [
                    KeyValueRecord(key: "sigma_scale", value: "1.14"),
                    KeyValueRecord(key: "price_cost_scale", value: "1.07"),
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
            deploymentPath: projectRoot.appendingPathComponent("FXDataEngine/Tools/OfflineLab/Profiles/EURUSD_profile.tsv"),
            studentRouterPath: projectRoot.appendingPathComponent("FXDataEngine/Tools/OfflineLab/student_router_EURUSD.tsv"),
            supervisorServicePath: projectRoot.appendingPathComponent("FXDataEngine/Tools/OfflineLab/supervisor_service.json"),
            supervisorCommandPath: projectRoot.appendingPathComponent("FXDataEngine/Tools/OfflineLab/supervisor_command.json"),
            worldPlanPath: projectRoot.appendingPathComponent("FXDataEngine/Tools/OfflineLab/world_plan_EURUSD.json"),
            deploymentSections: deploymentSections,
            routerSections: [
                RuntimeArtifactSection(
                    title: "Student Router",
                    sourcePath: projectRoot.appendingPathComponent("FXDataEngine/Tools/OfflineLab/student_router_EURUSD.tsv"),
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
                    sourcePath: projectRoot.appendingPathComponent("FXDataEngine/Tools/OfflineLab/supervisor_command.json"),
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
                    sourcePath: projectRoot.appendingPathComponent("FXDataEngine/Tools/OfflineLab/attribution_EURUSD.tsv"),
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
            deploymentPath: projectRoot.appendingPathComponent("FXDataEngine/Tools/OfflineLab/Profiles/USDJPY_profile.tsv"),
            studentRouterPath: projectRoot.appendingPathComponent("FXDataEngine/Tools/OfflineLab/student_router_USDJPY.tsv"),
            supervisorServicePath: projectRoot.appendingPathComponent("FXDataEngine/Tools/OfflineLab/supervisor_service.json"),
            supervisorCommandPath: projectRoot.appendingPathComponent("FXDataEngine/Tools/OfflineLab/supervisor_command.json"),
            worldPlanPath: projectRoot.appendingPathComponent("FXDataEngine/Tools/OfflineLab/world_plan_USDJPY.json"),
            deploymentSections: [
                RuntimeArtifactSection(
                    title: "Deployment Profile",
                    sourcePath: projectRoot.appendingPathComponent("FXDataEngine/Tools/OfflineLab/Profiles/USDJPY_profile.tsv"),
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

}
