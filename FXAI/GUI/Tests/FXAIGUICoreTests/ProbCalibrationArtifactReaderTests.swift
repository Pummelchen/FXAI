import FXAIGUICore
import Foundation
import Testing

struct ProbCalibrationArtifactReaderTests {
    @Test
    func readerParsesRuntimeAndReplayArtifacts() throws {
        let projectRoot = FileManager.default.temporaryDirectory
            .appendingPathComponent("fxai-prob-calibration-reader-\(UUID().uuidString)", isDirectory: true)
        let runtimeDirectory = projectRoot
            .appendingPathComponent("FILE_COMMON/FXAI/Runtime", isDirectory: true)
        let replayDirectory = projectRoot
            .appendingPathComponent("Tools/OfflineLab/ProbabilisticCalibration/Reports", isDirectory: true)
        try FileManager.default.createDirectory(at: runtimeDirectory, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: replayDirectory, withIntermediateDirectories: true)
        defer {
            try? FileManager.default.removeItem(at: projectRoot)
        }

        let runtimeState = """
        schema_version\t1
        symbol\tEURUSD
        generated_at\t1775727000
        method\tLOGISTIC_AFFINE
        session_label\tLONDON_NY_OVERLAP
        regime_label\tHIGH_VOL_EVENT
        selected_tier_kind\tPAIR_REGIME
        selected_tier_key\tPAIR_REGIME|EURUSD|*|HIGH_VOL_EVENT
        selected_support\t118
        selected_quality\t0.610000
        raw_action\tBUY
        raw_score\t0.190000
        raw_buy_prob\t0.570000
        raw_sell_prob\t0.210000
        raw_skip_prob\t0.220000
        calibrated_buy_prob\t0.460000
        calibrated_sell_prob\t0.240000
        calibrated_skip_prob\t0.300000
        calibrated_confidence\t0.460000
        expected_move_mean_points\t8.400000
        expected_move_q25_points\t2.700000
        expected_move_q50_points\t6.100000
        expected_move_q75_points\t11.300000
        spread_cost_points\t1.600000
        slippage_cost_points\t0.900000
        uncertainty_score\t0.630000
        uncertainty_penalty_points\t2.100000
        risk_penalty_points\t1.200000
        expected_gross_edge_points\t1.850000
        edge_after_costs_points\t-3.950000
        final_action\tSKIP
        abstain\t1
        fallback_used\t0
        calibration_stale\t0
        input_stale\t0
        support_usable\t1
        reasons_csv\tEDGE_TOO_SMALL; UNCERTAINTY_TOO_HIGH
        """
        try runtimeState.write(
            to: runtimeDirectory.appendingPathComponent("fxai_prob_calibration_EURUSD.tsv"),
            atomically: true,
            encoding: .utf8
        )

        let replayJSON: [String: Any] = [
            "hours_back": 48,
            "symbols": [
                [
                    "symbol": "EURUSD",
                    "observations": 18,
                    "abstain_count": 12,
                    "fallback_count": 1,
                    "average_confidence": 0.49,
                    "average_edge_after_costs_points": -0.84,
                    "average_uncertainty_score": 0.47,
                    "min_edge_after_costs_points": -4.10,
                    "max_edge_after_costs_points": 2.40,
                    "latest": [
                        "generated_at": "2026-04-10T09:30:00Z",
                        "state": [
                            "method": "LOGISTIC_AFFINE",
                            "session_label": "LONDON_NY_OVERLAP",
                            "regime_label": "HIGH_VOL_EVENT",
                            "selected_tier_kind": "PAIR_REGIME",
                            "selected_tier_key": "PAIR_REGIME|EURUSD|*|HIGH_VOL_EVENT",
                            "selected_support": 118,
                            "selected_quality": 0.61,
                            "raw_action": "BUY",
                            "raw_score": 0.19,
                            "raw_buy_prob": 0.57,
                            "raw_sell_prob": 0.21,
                            "raw_skip_prob": 0.22,
                            "calibrated_buy_prob": 0.46,
                            "calibrated_sell_prob": 0.24,
                            "calibrated_skip_prob": 0.30,
                            "calibrated_confidence": 0.46,
                            "expected_move_mean_points": 8.4,
                            "expected_move_q25_points": 2.7,
                            "expected_move_q50_points": 6.1,
                            "expected_move_q75_points": 11.3,
                            "spread_cost_points": 1.6,
                            "slippage_cost_points": 0.9,
                            "uncertainty_score": 0.63,
                            "uncertainty_penalty_points": 2.1,
                            "risk_penalty_points": 1.2,
                            "expected_gross_edge_points": 1.85,
                            "edge_after_costs_points": -3.95,
                            "final_action": "SKIP",
                            "abstain": true,
                            "fallback_used": false,
                            "calibration_stale": false,
                            "input_stale": false,
                            "support_usable": true,
                            "reason_codes": ["EDGE_TOO_SMALL", "UNCERTAINTY_TOO_HIGH"],
                        ],
                    ],
                    "action_counts": ["BUY": 5, "SKIP": 12],
                    "tier_counts": ["PAIR_REGIME": 13, "REGIME": 4],
                    "top_reasons": [["reason": "EDGE_TOO_SMALL", "count": 10]],
                    "recent_transitions": [
                        ["type": "action_change", "from": "BUY", "to": "SKIP", "at": "2026-04-10T09:16:00Z"],
                    ],
                ],
            ],
        ]
        let replayData = try JSONSerialization.data(withJSONObject: replayJSON, options: [.prettyPrinted, .sortedKeys])
        try replayData.write(to: replayDirectory.appendingPathComponent("prob_calibration_replay_report.json"))

        let snapshot = try #require(ProbCalibrationArtifactReader().read(projectRoot: projectRoot))
        #expect(snapshot.replayHoursBack == 48)
        #expect(snapshot.symbols.count == 1)
        #expect(snapshot.symbols.first?.tierKind == "PAIR_REGIME")
        #expect(snapshot.symbols.first?.finalAction == "SKIP")
        #expect(snapshot.symbols.first?.abstain == true)
        #expect(snapshot.symbols.first?.replayTopReasons.first?.key == "EDGE_TOO_SMALL")
        #expect(snapshot.symbols.first?.recentTransitions.first?.toValue == "SKIP")
    }

    @Test
    func readerReturnsNilWhenArtifactsAreMissing() {
        let projectRoot = FileManager.default.temporaryDirectory
            .appendingPathComponent("fxai-prob-calibration-reader-missing-\(UUID().uuidString)", isDirectory: true)
        defer {
            try? FileManager.default.removeItem(at: projectRoot)
        }

        #expect(ProbCalibrationArtifactReader().read(projectRoot: projectRoot) == nil)
    }
}
