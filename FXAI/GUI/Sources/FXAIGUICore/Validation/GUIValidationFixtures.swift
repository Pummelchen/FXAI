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
}
