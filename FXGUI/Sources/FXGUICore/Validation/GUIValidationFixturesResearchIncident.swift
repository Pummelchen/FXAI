import Foundation

extension GUIValidationFixtures {
    public static func researchSnapshot(projectRoot: URL) -> ResearchOSControlSnapshot {
        let now = Date()
        return ResearchOSControlSnapshot(
            generatedAt: now,
            profileName: "continuous",
            environment: ResearchOSEnvironmentStatus(
                backend: "turso",
                syncMode: "embedded-replica",
                databasePath: projectRoot.appendingPathComponent("FXDataEngine/Tools/OfflineLab/fxai_offline_lab.turso.db"),
                databaseName: "fxai-offline-lab",
                organizationSlug: "fxai",
                groupName: "default",
                locationName: "fra",
                cliConfigPath: projectRoot.appendingPathComponent("FXDataEngine/Tools/OfflineLab/turso.toml"),
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
                    envArtifactPath: projectRoot.appendingPathComponent("FXDataEngine/Tools/OfflineLab/branches/continuous.env"),
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
                    deploymentArtifactPath: projectRoot.appendingPathComponent("FXDataEngine/Tools/OfflineLab/Profiles/EURUSD_profile.tsv"),
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
                    deploymentArtifactPath: projectRoot.appendingPathComponent("FXDataEngine/Tools/OfflineLab/Profiles/USDJPY_profile.tsv"),
                    deploymentCreatedAt: now.addingTimeInterval(-3600)
                )
            ],
            sourceOfTruth: [
                KeyValueRecord(key: "profiles", value: "FXAI/Offline/Promotions"),
                KeyValueRecord(key: "database", value: "Turso"),
                KeyValueRecord(key: "runtime", value: "Swift runtime artifacts")
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
                        VisualizationSeriesPoint(label: "Price Cost", value: 1.07),
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
                    weakScenarios: ["Macro shock cluster", "Late NY price-cost widening"]
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
                        IncidentAction(title: "Open Runtime", summary: "Inspect runtime deployment sections", command: "python3 FXDataEngine/Tools/fxai_testlab.py verify-all", destinationSelection: "runtimeMonitor")
                    ],
                    playbook: RecoveryPlaybook(
                        title: "Refresh runtime artifacts",
                        summary: "Regenerate and verify the deployed runtime state.",
                        steps: [
                            RecoveryStep(title: "Verify all", summary: "Confirm clean compiles and fixtures.", command: "python3 FXDataEngine/Tools/fxai_testlab.py verify-all", destinationSelection: "runtimeMonitor")
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
                        IncidentAction(title: "Open Research OS", summary: "Inspect branch status and audit events", command: "python3 FXDataEngine/Tools/fxai_offline_lab.py dashboard", destinationSelection: "researchControl")
                    ],
                    playbook: RecoveryPlaybook(
                        title: "Review branch health",
                        summary: "Inspect recent Research OS branch and audit state.",
                        steps: [
                            RecoveryStep(title: "Open Research OS", summary: "Review branch inventory.", command: "python3 FXDataEngine/Tools/fxai_offline_lab.py branch-inventory", destinationSelection: "researchControl")
                        ]
                    )
                )
            ]
        )
    }

}
