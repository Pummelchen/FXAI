import Foundation
import Testing
@testable import FXAIGUICore

struct AdvancedVisualizationBuilderTests {
    @Test
    func buildsHeatmapsAndTimelinesFromResearchArtifacts() throws {
        let root = try makeVisualizationFixture()
        let builder = AdvancedVisualizationBuilder()

        let runtimeSnapshot = RuntimeOperationsSnapshot(
            generatedAt: Date(),
            profileName: "viz-profile",
            deployments: [
                RuntimeDeploymentDetail(
                    id: "EURUSD",
                    symbol: "EURUSD",
                    profileName: "viz-profile",
                    pluginName: "ai_mlp",
                    promotionTier: "audit-approved",
                    runtimeMode: "research",
                    createdAt: Date(timeIntervalSince1970: 1_700_000_000),
                    reviewedAt: Date(timeIntervalSince1970: 1_700_000_100),
                    artifactHealth: RuntimeArtifactHealth(
                        artifactExists: true,
                        staleArtifact: false,
                        missingDeployment: false,
                        missingRouter: false,
                        missingSupervisorService: false,
                        missingSupervisorCommand: false,
                        missingWorldPlan: false,
                        artifactAgeSeconds: 0,
                        performanceFailures: [],
                        artifactSizeFailures: []
                    ),
                    deploymentPath: nil,
                    studentRouterPath: nil,
                    supervisorServicePath: nil,
                    supervisorCommandPath: nil,
                    worldPlanPath: nil,
                    deploymentSections: [
                        RuntimeArtifactSection(
                            title: "Deployment",
                            sourcePath: nil,
                            values: [
                                KeyValueRecord(key: "policy_trade_floor", value: "0.62"),
                                KeyValueRecord(key: "policy_no_trade_cap", value: "0.71"),
                                KeyValueRecord(key: "student_signal_gain", value: "1.05"),
                                KeyValueRecord(key: "teacher_signal_gain", value: "1.08")
                            ]
                        )
                    ],
                    routerSections: [
                        RuntimeArtifactSection(
                            title: "Router",
                            sourcePath: nil,
                            values: [
                                KeyValueRecord(key: "family_weight_recurrent", value: "1.24"),
                                KeyValueRecord(key: "plugin_weights_csv", value: "ai_mlp=1.20")
                            ]
                        )
                    ],
                    supervisorSections: [
                        RuntimeArtifactSection(
                            title: "Supervisor",
                            sourcePath: nil,
                            values: [
                                KeyValueRecord(key: "budget_multiplier", value: "1.11"),
                                KeyValueRecord(key: "entry_floor", value: "0.41"),
                                KeyValueRecord(key: "reduce_bias", value: "0.14")
                            ]
                        )
                    ],
                    commandSections: [
                        RuntimeArtifactSection(
                            title: "Command",
                            sourcePath: nil,
                            values: [
                                KeyValueRecord(key: "exit_bias", value: "0.13")
                            ]
                        )
                    ],
                    worldSections: [
                        RuntimeArtifactSection(
                            title: "World",
                            sourcePath: nil,
                            values: [
                                KeyValueRecord(key: "sigma_scale", value: "0.77"),
                                KeyValueRecord(key: "spread_scale", value: "0.87")
                            ]
                        )
                    ],
                    attributionSections: [
                        RuntimeArtifactSection(
                            title: "Attribution",
                            sourcePath: nil,
                            values: [
                                KeyValueRecord(key: "feature_weight_price", value: "1.08"),
                                KeyValueRecord(key: "feature_weight_volatility", value: "1.08")
                            ]
                        )
                    ],
                    featureHighlights: [],
                    studentRouterWeights: [],
                    familyWeights: [],
                    prunedPlugins: []
                )
            ],
            champions: [
                PromotionChampionRecord(
                    symbol: "EURUSD",
                    pluginName: "ai_mlp",
                    status: "champion",
                    promotionTier: "audit-approved",
                    championScore: 83.5,
                    challengerScore: 81.2,
                    portfolioScore: 0.72,
                    reviewedAt: Date(timeIntervalSince1970: 1_700_000_100),
                    setPath: nil,
                    profileName: "viz-profile"
                )
            ]
        )

        let researchSnapshot = ResearchOSControlSnapshot(
            generatedAt: Date(),
            profileName: "viz-profile",
            environment: nil,
            branches: [
                ResearchOSBranchRecord(
                    name: "viz-branch",
                    sourceDatabase: "fxai-prod",
                    parentName: "fxai-prod",
                    branchKind: "campaign",
                    status: "active",
                    groupName: "trading",
                    locationName: "fra",
                    hostname: "",
                    syncURL: "",
                    envArtifactPath: nil,
                    isBranch: true,
                    createdAt: Date(timeIntervalSince1970: 1_700_000_050),
                    sourceTimestamp: ""
                )
            ],
            auditEvents: [
                ResearchOSAuditEvent(
                    organizationSlug: "openai",
                    eventID: "evt_1",
                    eventType: "branch.created",
                    targetName: "viz-branch",
                    occurredAt: Date(timeIntervalSince1970: 1_700_000_055),
                    observedAt: Date(timeIntervalSince1970: 1_700_000_060)
                )
            ],
            symbols: [
                ResearchOSSymbolControl(
                    symbol: "EURUSD",
                    analogNeighbors: [
                        ResearchOSAnalogNeighbor(
                            sourceKey: "plugin:ai_mlp",
                            pluginName: "ai_mlp",
                            distance: 0.09,
                            similarity: 0.91,
                            score: 0.84,
                            sourceType: "shadow",
                            scope: "analog_shadow",
                            payload: []
                        )
                    ],
                    deploymentArtifactPath: nil,
                    deploymentCreatedAt: Date(timeIntervalSince1970: 1_700_000_000)
                )
            ],
            sourceOfTruth: []
        )

        let snapshot = builder.build(
            projectRoot: root,
            runtimeSnapshot: runtimeSnapshot,
            researchSnapshot: researchSnapshot
        )

        #expect(snapshot.profileName == "viz-profile")
        #expect(snapshot.familyStressHeatmap != nil)
        #expect(snapshot.familyStressHeatmap?.rowLabels == ["EURUSD"])
        #expect(snapshot.familyStressHeatmap?.columnLabels.contains("recurrent") == true)
        #expect(snapshot.globalTimeline.count >= 3)

        let detail = try #require(snapshot.symbolDetails.first)
        #expect(detail.worldSessionScales.count == 3)
        #expect(detail.worldStressMetrics.contains(where: { $0.label == "Shock Decay" }))
        #expect(detail.familyWeights.contains(where: { $0.label == "recurrent" }))
        #expect(detail.featureWeights.contains(where: { $0.label == "price" }))
        #expect(detail.pluginWeights.contains(where: { $0.label == "ai_mlp" }))
        #expect(detail.artifactDiffHeatmap != nil)
        #expect(detail.weakScenarios == ["market_adversarial", "market_macro_event"])
        #expect(detail.timeline.contains(where: { $0.category == "promotion" }))
        #expect(detail.timeline.contains(where: { $0.category == "attribution" }))
    }

    private func makeVisualizationFixture() throws -> URL {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)

        let profileRoot = root.appendingPathComponent("Tools/OfflineLab/ResearchOS/viz-profile", isDirectory: true)
        try FileManager.default.createDirectory(at: profileRoot, withIntermediateDirectories: true)

        try Data(
            """
            {
              "family_weights": {
                "recurrent": 1.24,
                "tree": 0.90
              },
              "feature_group_weights": {
                "price": 1.08,
                "volatility": 1.08,
                "context": 1.03
              }
            }
            """.utf8
        ).write(to: profileRoot.appendingPathComponent("attribution_EURUSD.json"))

        try Data(
            """
            {
              "plugin_weights": {
                "ai_mlp": 1.21
              }
            }
            """.utf8
        ).write(to: profileRoot.appendingPathComponent("student_router_EURUSD.json"))

        try Data(
            """
            {
              "asia_sigma_scale": 0.5,
              "asia_spread_scale": 1.0,
              "london_sigma_scale": 0.6,
              "london_spread_scale": 1.1,
              "newyork_sigma_scale": 0.55,
              "newyork_spread_scale": 1.05,
              "sigma_scale": 0.77,
              "spread_scale": 0.87,
              "shock_decay": 0.73,
              "liquidity_stress": 0.17,
              "transition_entropy": 0.11,
              "recovery_bias": -0.01,
              "weak_scenarios": ["market_adversarial", "market_macro_event"]
            }
            """.utf8
        ).write(to: profileRoot.appendingPathComponent("world_simulator_EURUSD.json"))

        try Data(
            """
            {
              "supervisor_score": 0.44,
              "pressure_velocity": 0.03
            }
            """.utf8
        ).write(to: profileRoot.appendingPathComponent("supervisor_service_EURUSD.json"))

        try Data(
            """
            {
              "deployments": [
                {
                  "champions": [
                    {
                      "plugin_name": "ai_mlp",
                      "promoted_at": 1700000080,
                      "reviewed_at": 1700000100,
                      "champion_score": 83.5,
                      "portfolio_score": 0.72,
                      "status": "champion"
                    }
                  ]
                }
              ]
            }
            """.utf8
        ).write(to: profileRoot.appendingPathComponent("lineage_EURUSD.json"))

        return root
    }
}
