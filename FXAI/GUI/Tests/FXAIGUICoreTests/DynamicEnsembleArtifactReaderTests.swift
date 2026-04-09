import FXAIGUICore
import Foundation
import Testing

struct DynamicEnsembleArtifactReaderTests {
    @Test
    func readerParsesRuntimeAndReplayArtifacts() throws {
        let projectRoot = FileManager.default.temporaryDirectory
            .appendingPathComponent("fxai-dynamic-ensemble-reader-\(UUID().uuidString)", isDirectory: true)
        let runtimeDirectory = projectRoot
            .appendingPathComponent("FILE_COMMON/FXAI/Runtime", isDirectory: true)
        let replayDirectory = projectRoot
            .appendingPathComponent("Tools/OfflineLab/DynamicEnsemble/Reports", isDirectory: true)
        try FileManager.default.createDirectory(at: runtimeDirectory, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: replayDirectory, withIntermediateDirectories: true)
        defer {
            try? FileManager.default.removeItem(at: projectRoot)
        }

        let runtimeState = """
        schema_version\t1
        symbol\tEURUSD
        generated_at\t1775727000
        top_regime\tHIGH_VOL_EVENT
        session_label\tLONDON_NY_OVERLAP
        trade_posture\tCAUTION
        ensemble_quality\t0.410000
        abstain_bias\t0.240000
        agreement_score\t0.330000
        context_fit_score\t0.570000
        dominant_plugin_share\t0.410000
        buy_prob\t0.370000
        sell_prob\t0.290000
        skip_prob\t0.340000
        final_score\t0.080000
        final_action\tSKIP
        fallback_used\t0
        reasons_csv\tplugin_disagreement_elevated; newspulse_caution_active
        active_plugins_csv\tai_gha:0.3900:0.7400|ai_tesseract:0.2700:0.5800
        downweighted_plugins_csv\tai_tft:0.1800:0.4600
        suppressed_plugins_csv\tlin_pa:0.0000:0.1800
        """
        try runtimeState.write(
            to: runtimeDirectory.appendingPathComponent("fxai_dynamic_ensemble_EURUSD.tsv"),
            atomically: true,
            encoding: .utf8
        )

        let replayJSON: [String: Any] = [
            "hours_back": 48,
            "symbols": [
                [
                    "symbol": "EURUSD",
                    "observations": 14,
                    "average_quality": 0.52,
                    "max_abstain_bias": 0.38,
                    "latest": [
                        "generated_at": "2026-04-09T09:30:00Z",
                        "ensemble": [
                            "top_regime": "HIGH_VOL_EVENT",
                            "session_label": "LONDON_NY_OVERLAP",
                            "trade_posture": "CAUTION",
                            "ensemble_quality": 0.41,
                            "abstain_bias": 0.24,
                            "buy_prob": 0.37,
                            "sell_prob": 0.29,
                            "skip_prob": 0.34,
                            "final_score": 0.08,
                            "final_action": "SKIP",
                            "reasons": ["plugin_disagreement_elevated"],
                        ],
                        "plugins": [
                            ["name": "ai_gha", "family": "memory", "status": "ACTIVE", "signal": "BUY", "weight": 0.39, "trust": 0.74, "calibration_shrink": 0.88, "reasons": ["adaptive_router_upweighted"]],
                            ["name": "lin_pa", "family": "rule", "status": "SUPPRESSED", "signal": "SELL", "weight": 0.0, "trust": 0.18, "calibration_shrink": 0.62, "reasons": ["trust_below_suppress_threshold"]],
                        ],
                    ],
                    "posture_counts": ["CAUTION": 9, "ABSTAIN_BIAS": 2],
                    "action_counts": ["SKIP": 8, "BUY": 4],
                    "plugin_status_counts": ["ACTIVE": 18, "SUPPRESSED": 4],
                    "top_reasons": [["reason": "plugin_disagreement_elevated", "count": 6]],
                    "top_dominant_plugins": [["plugin": "ai_gha", "count": 7]],
                    "recent_transitions": [
                        ["type": "action_change", "from": "BUY", "to": "SKIP", "at": "2026-04-09T09:16:00Z"],
                    ],
                ],
            ],
        ]
        let replayData = try JSONSerialization.data(withJSONObject: replayJSON, options: [.prettyPrinted, .sortedKeys])
        try replayData.write(to: replayDirectory.appendingPathComponent("dynamic_ensemble_replay_report.json"))

        let snapshot = try #require(DynamicEnsembleArtifactReader().read(projectRoot: projectRoot))
        #expect(snapshot.replayHoursBack == 48)
        #expect(snapshot.symbols.count == 1)
        #expect(snapshot.symbols.first?.tradePosture == "CAUTION")
        #expect(snapshot.symbols.first?.activePlugins.first?.name == "ai_gha")
        #expect(snapshot.symbols.first?.suppressedPlugins.first?.name == "lin_pa")
        #expect(snapshot.symbols.first?.recentTransitions.first?.toValue == "SKIP")
    }

    @Test
    func readerReturnsNilWhenArtifactsAreMissing() {
        let projectRoot = FileManager.default.temporaryDirectory
            .appendingPathComponent("fxai-dynamic-ensemble-reader-missing-\(UUID().uuidString)", isDirectory: true)
        defer {
            try? FileManager.default.removeItem(at: projectRoot)
        }

        #expect(DynamicEnsembleArtifactReader().read(projectRoot: projectRoot) == nil)
    }
}
