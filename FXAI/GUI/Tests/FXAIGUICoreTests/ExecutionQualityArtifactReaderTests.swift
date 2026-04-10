import FXAIGUICore
import Foundation
import Testing

struct ExecutionQualityArtifactReaderTests {
    @Test
    func readerParsesRuntimeAndReplayArtifacts() throws {
        let projectRoot = FileManager.default.temporaryDirectory
            .appendingPathComponent("fxai-execution-quality-reader-\(UUID().uuidString)", isDirectory: true)
        let runtimeDirectory = projectRoot
            .appendingPathComponent("FILE_COMMON/FXAI/Runtime", isDirectory: true)
        let replayDirectory = projectRoot
            .appendingPathComponent("Tools/OfflineLab/ExecutionQuality/Reports", isDirectory: true)
        try FileManager.default.createDirectory(at: runtimeDirectory, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: replayDirectory, withIntermediateDirectories: true)
        defer {
            try? FileManager.default.removeItem(at: projectRoot)
        }

        let runtimeState = """
        symbol\tEURUSD
        generated_at\t1775728200
        method\tSCORECARD_V1
        session_label\tLONDON_NY_OVERLAP
        regime_label\tHIGH_VOL_EVENT
        selected_tier_kind\tPAIR_REGIME
        selected_tier_key\tPAIR_REGIME|EURUSD|LONDON_NY_OVERLAP|HIGH_VOL_EVENT
        selected_support\t92
        selected_quality\t0.580000
        fallback_used\t0
        memory_stale\t0
        data_stale\t0
        support_usable\t1
        news_window_active\t1
        rates_repricing_active\t0
        broker_coverage\t0.710000
        broker_reject_prob\t0.190000
        broker_partial_fill_prob\t0.140000
        spread_now_points\t1.300000
        spread_expected_points\t2.800000
        spread_widening_risk\t0.680000
        expected_slippage_points\t1.100000
        slippage_risk\t0.570000
        fill_quality_score\t0.490000
        latency_sensitivity_score\t0.630000
        liquidity_fragility_score\t0.610000
        execution_quality_score\t0.440000
        allowed_deviation_points\t6.000000
        caution_lot_scale\t0.820000
        caution_enter_prob_buffer\t0.040000
        execution_state\tCAUTION
        reasons_csv\tNEWS_WINDOW_ACTIVE; SPREAD_ALREADY_ELEVATED; LATENCY_SENSITIVITY_HIGH
        """
        try runtimeState.write(
            to: runtimeDirectory.appendingPathComponent("fxai_execution_quality_EURUSD.tsv"),
            atomically: true,
            encoding: .utf8
        )

        let replayJSON: [String: Any] = [
            "hours_back": 48,
            "symbols": [
                [
                    "symbol": "EURUSD",
                    "observations": 18,
                    "state_counts": ["CAUTION": 11, "NORMAL": 7],
                    "tier_counts": ["PAIR_REGIME": 12, "REGIME": 6],
                    "max_spread_widening_risk": 0.79,
                    "max_slippage_risk": 0.66,
                    "min_execution_quality_score": 0.39,
                    "top_reasons": [
                        ["reason": "NEWS_WINDOW_ACTIVE", "count": 9],
                        ["reason": "SPREAD_ALREADY_ELEVATED", "count": 7],
                    ],
                    "recent_transitions": [
                        ["type": "execution_state", "from": "NORMAL", "to": "CAUTION", "at": "2026-04-10T09:16:00Z"],
                    ],
                    "latest": [
                        "generated_at": "2026-04-10T09:50:00Z",
                        "state": [
                            "method": "SCORECARD_V1",
                            "session_label": "LONDON_NY_OVERLAP",
                            "regime_label": "HIGH_VOL_EVENT",
                            "selected_tier_kind": "PAIR_REGIME",
                            "selected_tier_key": "PAIR_REGIME|EURUSD|LONDON_NY_OVERLAP|HIGH_VOL_EVENT",
                            "selected_support": 92,
                            "selected_quality": 0.58,
                            "fallback_used": false,
                            "memory_stale": false,
                            "data_stale": false,
                            "support_usable": true,
                            "news_window_active": true,
                            "rates_repricing_active": false,
                            "broker_coverage": 0.71,
                            "broker_reject_prob": 0.19,
                            "broker_partial_fill_prob": 0.14,
                            "spread_now_points": 1.3,
                            "spread_expected_points": 2.8,
                            "spread_widening_risk": 0.68,
                            "expected_slippage_points": 1.1,
                            "slippage_risk": 0.57,
                            "fill_quality_score": 0.49,
                            "latency_sensitivity_score": 0.63,
                            "liquidity_fragility_score": 0.61,
                            "execution_quality_score": 0.44,
                            "allowed_deviation_points": 6.0,
                            "caution_lot_scale": 0.82,
                            "caution_enter_prob_buffer": 0.04,
                            "execution_state": "CAUTION",
                            "reason_codes": [
                                "NEWS_WINDOW_ACTIVE",
                                "SPREAD_ALREADY_ELEVATED",
                                "LATENCY_SENSITIVITY_HIGH",
                            ],
                        ],
                    ],
                ],
            ],
        ]
        let replayData = try JSONSerialization.data(withJSONObject: replayJSON, options: [.prettyPrinted, .sortedKeys])
        try replayData.write(to: replayDirectory.appendingPathComponent("execution_quality_replay_report.json"))

        let snapshot = try #require(ExecutionQualityArtifactReader().read(projectRoot: projectRoot))
        #expect(snapshot.replayHoursBack == 48)
        #expect(snapshot.symbols.count == 1)
        #expect(snapshot.symbols.first?.executionState == "CAUTION")
        #expect(snapshot.symbols.first?.spreadExpectedPoints == 2.8)
        #expect(snapshot.symbols.first?.replayTopReasons.first?.key == "NEWS_WINDOW_ACTIVE")
        #expect(snapshot.symbols.first?.recentTransitions.first?.toValue == "CAUTION")
    }

    @Test
    func readerReturnsNilWhenArtifactsAreMissing() {
        let projectRoot = FileManager.default.temporaryDirectory
            .appendingPathComponent("fxai-execution-quality-reader-missing-\(UUID().uuidString)", isDirectory: true)
        defer {
            try? FileManager.default.removeItem(at: projectRoot)
        }

        #expect(ExecutionQualityArtifactReader().read(projectRoot: projectRoot) == nil)
    }
}
