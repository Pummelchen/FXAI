import FXAIGUICore
import Foundation
import Testing

struct AdaptiveRouterArtifactReaderTests {
    @Test
    func readerParsesAdaptiveRouterArtifactsFromDashboardAndReplay() throws {
        let projectRoot = FileManager.default.temporaryDirectory
            .appendingPathComponent("fxai-adaptive-router-reader-\(UUID().uuidString)", isDirectory: true)
        let profileDirectory = projectRoot
            .appendingPathComponent("Tools/OfflineLab/ResearchOS/continuous", isDirectory: true)
        let replayDirectory = projectRoot
            .appendingPathComponent("Tools/OfflineLab/AdaptiveRouter/Reports", isDirectory: true)
        try FileManager.default.createDirectory(at: profileDirectory, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: replayDirectory, withIntermediateDirectories: true)

        defer {
            try? FileManager.default.removeItem(at: projectRoot)
        }

        let dashboard: [String: Any] = [
            "generated_at": "2026-04-09T09:30:00Z",
            "deployments": [
                [
                    "symbol": "EURUSD",
                    "live_state": [
                        "adaptive_router_tsv": [
                            "router_mode": "WEIGHTED_ENSEMBLE",
                            "pair_tags_csv": "dollar_core,macro_sensitive",
                            "caution_threshold": "0.53",
                            "abstain_threshold": "0.34",
                            "regime_bias_HIGH_VOL_EVENT": "1.10",
                        ],
                        "adaptive_runtime_tsv": [
                            "generated_at": "1775727000",
                            "top_regime_label": "HIGH_VOL_EVENT",
                            "regime_confidence": "0.810000",
                            "trade_posture": "CAUTION",
                            "abstain_bias": "0.180000",
                            "session_label": "LONDON_NY_OVERLAP",
                            "spread_regime": "ELEVATED",
                            "volatility_regime": "HIGH",
                            "news_risk_score": "0.760000",
                            "news_pressure": "-0.140000",
                            "event_eta_min": "14",
                            "stale_news": "0",
                            "liquidity_stress": "0.580000",
                            "breakout_pressure": "0.640000",
                            "trend_strength": "0.490000",
                            "range_pressure": "0.240000",
                            "macro_pressure": "0.720000",
                            "reasons_csv": "NewsPulse event window active; Spread regime elevated",
                            "probabilities_csv": "HIGH_VOL_EVENT=0.810000,LIQUIDITY_STRESS=0.460000,BREAKOUT_TRANSITION=0.240000",
                            "active_plugins_csv": "ai_gha:0.4600:1.2800|ai_tesseract:0.3000:1.1400",
                            "downweighted_plugins_csv": "ai_tft:0.1800:0.7900",
                            "suppressed_plugins_csv": "lin_pa:0.0000:0.2200",
                        ],
                    ],
                ],
            ],
        ]

        let profileJSON: [String: Any] = [
            "router_mode": "WEIGHTED_ENSEMBLE",
            "summary": [
                "generated_at": "2026-04-09T09:20:00Z",
                "top_plugins": ["ai_gha", "ai_tesseract", "ai_tft"],
            ],
        ]

        let replayJSON: [String: Any] = [
            "hours_back": 48,
            "symbols": [
                [
                    "symbol": "EURUSD",
                    "observations": 12,
                    "latest": [
                        "generated_at": "2026-04-09T09:30:00Z",
                        "regime": [
                            "top_label": "HIGH_VOL_EVENT",
                            "confidence": 0.81,
                            "probabilities": [
                                "HIGH_VOL_EVENT": 0.81,
                                "LIQUIDITY_STRESS": 0.46,
                            ],
                            "reasons": ["NewsPulse event window active"],
                        ],
                        "router": [
                            "mode": "WEIGHTED_ENSEMBLE",
                            "trade_posture": "CAUTION",
                            "abstain_bias": 0.18,
                            "reasons": ["NewsPulse event window active"],
                        ],
                        "plugins": [
                            ["name": "ai_gha", "weight": 0.46, "suitability": 1.28, "status": "UPWEIGHTED", "reasons": ["Strong macro/event regime fit"]],
                        ],
                    ],
                    "regime_counts": ["HIGH_VOL_EVENT": 9, "BREAKOUT_TRANSITION": 3],
                    "posture_counts": ["CAUTION": 8, "ABSTAIN_BIAS": 2],
                    "top_reasons": [["reason": "NewsPulse event window active", "count": 9]],
                    "top_plugins": [["plugin": "ai_gha", "count": 7]],
                    "recent_transitions": [
                        ["type": "regime_change", "from": "BREAKOUT_TRANSITION", "to": "HIGH_VOL_EVENT", "at": "2026-04-09T09:16:00Z"],
                    ],
                ],
            ],
        ]

        try writeJSON(dashboard, to: profileDirectory.appendingPathComponent("operator_dashboard.json"))
        try writeJSON(profileJSON, to: profileDirectory.appendingPathComponent("adaptive_router_EURUSD.json"))
        try writeJSON(replayJSON, to: replayDirectory.appendingPathComponent("adaptive_router_replay_report.json"))

        let snapshot = try #require(AdaptiveRouterArtifactReader().read(projectRoot: projectRoot))
        #expect(snapshot.profileName == "continuous")
        #expect(snapshot.replayHoursBack == 48)
        #expect(snapshot.symbols.count == 1)
        #expect(snapshot.symbols.first?.topRegime == "HIGH_VOL_EVENT")
        #expect(snapshot.symbols.first?.activePlugins.first?.name == "ai_gha")
        #expect(snapshot.symbols.first?.suppressedPlugins.first?.name == "lin_pa")
        #expect(snapshot.symbols.first?.pairTags.contains("dollar_core") == true)
        #expect(snapshot.symbols.first?.recentTransitions.first?.toValue == "HIGH_VOL_EVENT")
    }

    @Test
    func readerReturnsNilWhenAdaptiveRouterArtifactsAreMissing() {
        let projectRoot = FileManager.default.temporaryDirectory
            .appendingPathComponent("fxai-adaptive-router-reader-missing-\(UUID().uuidString)", isDirectory: true)
        defer {
            try? FileManager.default.removeItem(at: projectRoot)
        }

        #expect(AdaptiveRouterArtifactReader().read(projectRoot: projectRoot) == nil)
    }

    private func writeJSON(_ payload: [String: Any], to url: URL) throws {
        try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        let data = try JSONSerialization.data(withJSONObject: payload, options: [.prettyPrinted, .sortedKeys])
        try data.write(to: url)
    }
}
