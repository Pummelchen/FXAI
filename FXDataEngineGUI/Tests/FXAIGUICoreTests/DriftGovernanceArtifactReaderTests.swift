import FXAIGUICore
import Foundation
import Testing

struct DriftGovernanceArtifactReaderTests {
    @Test
    func readerParsesReportAndStatusArtifacts() throws {
        let projectRoot = FileManager.default.temporaryDirectory
            .appendingPathComponent("fxai-drift-governance-reader-\(UUID().uuidString)", isDirectory: true)
        let reportDirectory = projectRoot
            .appendingPathComponent("Tools/OfflineLab/DriftGovernance/Reports", isDirectory: true)
        let statusDirectory = projectRoot
            .appendingPathComponent("Tools/OfflineLab/DriftGovernance", isDirectory: true)
        try FileManager.default.createDirectory(at: reportDirectory, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: statusDirectory, withIntermediateDirectories: true)
        defer {
            try? FileManager.default.removeItem(at: projectRoot)
        }

        let reportJSON: [String: Any] = [
            "generated_at": "2026-04-11T12:00:00Z",
            "profile_name": "continuous",
            "policy_version": 1,
            "action_mode": "AUTO_APPLY_PROTECTIVE",
            "symbol_count": 1,
            "plugin_count": 1,
            "latest_action_count": 2,
            "health_counts": [
                "DEGRADED": 1,
            ],
            "governance_counts": [
                "DEMOTED": 1,
            ],
            "action_counts": [
                "DEMOTE": 1,
            ],
            "symbols": [
                [
                    "symbol": "EURUSD",
                    "plugin_count": 1,
                    "health_counts": ["DEGRADED": 1],
                    "governance_counts": ["DEMOTED": 1],
                    "action_counts": ["DEMOTE": 1],
                    "latest_context": [
                        "execution_quality": [
                            "min_execution_quality_score": 0.44,
                            "latest": ["execution_state": "CAUTION"],
                        ],
                        "newspulse": [
                            "trade_gate": "CAUTION",
                            "news_risk_score": 0.71,
                        ],
                    ],
                    "plugins": [
                        [
                            "plugin_name": "ai_tft",
                            "family_id": 31,
                            "family_name": "Transformer",
                            "base_registry_status": "champion",
                            "health_state": "DEGRADED",
                            "governance_state": "DEMOTED",
                            "recommended_governance_state": "DEMOTED",
                            "action_recommendation": "DEMOTE",
                            "action_applied": true,
                            "weight_multiplier": 0.02,
                            "restrict_live": true,
                            "shadow_only": true,
                            "disabled": false,
                            "aggregate_risk_score": 0.81,
                            "drift_scores": [
                                "calibration_drift_score": 0.88,
                                "pair_decay_score": 0.76,
                            ],
                            "support": [
                                "sample_count_recent": 842,
                                "sample_count_reference": 10420,
                            ],
                            "reason_codes": [
                                "CALIBRATION_DRIFT_ELEVATED",
                                "PAIR_DECAY_EURUSD",
                            ],
                            "quality_flags": [
                                "low_support": false,
                                "reference_fallback_scope": "SYMBOL_PLUGIN",
                            ],
                            "challenger_evaluation": [
                                "eligibility_state": "INSUFFICIENT",
                                "qualifies": false,
                                "support_count": 3,
                                "shadow_support": 12,
                                "walkforward_score": 71.0,
                                "recent_score": 69.0,
                                "adversarial_score": 68.0,
                                "macro_event_score": 65.0,
                                "calibration_error": 0.08,
                                "issue_count": 0.0,
                                "live_shadow_score": 0.54,
                                "live_reliability": 0.51,
                                "portfolio_score": 0.57,
                                "promotion_margin": 0.02,
                            ],
                            "context": [
                                "dynamic_ensemble": [
                                    "average_quality": 0.52,
                                    "max_abstain_bias": 0.38,
                                ],
                            ],
                        ],
                    ],
                    "recent_actions": [
                        [
                            "plugin_name": "ai_tft",
                            "previous_state": "CHAMPION",
                            "new_state": "DEMOTED",
                            "action_kind": "DEMOTE",
                            "action_applied": true,
                            "created_at": 1_776_000_000,
                        ],
                    ],
                ],
            ],
            "artifacts": [
                "report_path": "/tmp/drift_governance_report.json",
                "status_path": "/tmp/drift_governance_status.json",
            ],
        ]
        let statusJSON: [String: Any] = [
            "profile_name": "continuous",
            "plugin_count": 1,
            "symbol_count": 1,
            "applied_action_count": 1,
            "artifacts": [
                "report_path": "/tmp/drift_governance_report.json",
                "status_path": "/tmp/drift_governance_status.json",
                "history_path": "/tmp/drift_governance_history.ndjson",
            ],
        ]

        let reportData = try JSONSerialization.data(withJSONObject: reportJSON, options: [.prettyPrinted, .sortedKeys])
        let statusData = try JSONSerialization.data(withJSONObject: statusJSON, options: [.prettyPrinted, .sortedKeys])
        try reportData.write(to: reportDirectory.appendingPathComponent("drift_governance_report.json"))
        try statusData.write(to: statusDirectory.appendingPathComponent("drift_governance_status.json"))

        let snapshot = try #require(DriftGovernanceArtifactReader().read(projectRoot: projectRoot))
        #expect(snapshot.profileName == "continuous")
        #expect(snapshot.symbolCount == 1)
        #expect(snapshot.pluginCount == 1)
        #expect(snapshot.symbols.first?.symbol == "EURUSD")
        #expect(snapshot.symbols.first?.plugins.first?.governanceState == "DEMOTED")
        #expect(snapshot.symbols.first?.plugins.first?.reasonCodes.contains("PAIR_DECAY_EURUSD") == true)
        #expect(snapshot.symbols.first?.recentActions.first?.actionKind == "DEMOTE")
        #expect(snapshot.symbols.first?.latestContext.contains(where: { $0.key == "execution.state" && $0.value == "CAUTION" }) == true)
    }

    @Test
    func readerReturnsNilWhenReportIsMissing() {
        let projectRoot = FileManager.default.temporaryDirectory
            .appendingPathComponent("fxai-drift-governance-reader-missing-\(UUID().uuidString)", isDirectory: true)
        defer {
            try? FileManager.default.removeItem(at: projectRoot)
        }

        #expect(DriftGovernanceArtifactReader().read(projectRoot: projectRoot) == nil)
    }
}
