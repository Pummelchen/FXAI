import FXAIGUICore
import Foundation
import Testing

struct MicrostructureArtifactReaderTests {
    @Test
    func readerParsesProjectLocalStatusPayload() throws {
        let projectRoot = FileManager.default.temporaryDirectory
            .appendingPathComponent("fxai-microstructure-reader-\(UUID().uuidString)", isDirectory: true)
        let statusURL = projectRoot
            .appendingPathComponent("Tools/OfflineLab/Microstructure", isDirectory: true)
            .appendingPathComponent("microstructure_status.json", isDirectory: false)
        try FileManager.default.createDirectory(at: statusURL.deletingLastPathComponent(), withIntermediateDirectories: true)
        defer {
            try? FileManager.default.removeItem(at: projectRoot)
        }

        let payload: [String: Any] = [
            "generated_at": "2026-04-09T09:30:00Z",
            "service": [
                "ok": 1,
                "stale": 0,
                "enabled": 1,
                "poll_interval_ms": 5000,
                "symbol_refresh_sec": 300,
                "snapshot_stale_after_sec": 45,
                "last_poll_at": "2026-04-09T09:29:55Z",
                "last_success_at": "2026-04-09T09:29:55Z",
                "last_symbol_refresh_at": "2026-04-09T09:28:00Z",
            ],
            "symbols": [
                "EURUSD": [
                    "broker_symbol": "EURUSD",
                    "available": 1,
                    "stale": 0,
                    "generated_at": "2026-04-09T09:30:00Z",
                    "spread_current": 0.8,
                    "silent_gap_seconds_current": 0.4,
                    "session_tag": "LONDON_NEWYORK_OVERLAP",
                    "handoff_flag": 0,
                    "minutes_since_session_open": 75,
                    "minutes_to_session_close": 165,
                    "session_open_burst_score": 0.22,
                    "session_spread_behavior_score": 0.18,
                    "liquidity_stress_score": 0.34,
                    "hostile_execution_score": 0.29,
                    "microstructure_regime": "TRENDING_CLEAN",
                    "trade_gate": "ALLOW",
                    "tick_imbalance_30s": 0.31,
                    "directional_efficiency_60s": 0.71,
                    "spread_zscore_60s": 0.48,
                    "tick_rate_60s": 129,
                    "tick_rate_zscore_60s": 1.21,
                    "realized_vol_5m": 0.62,
                    "vol_burst_score_5m": 1.16,
                    "local_extrema_breach_score_60s": 0.19,
                    "sweep_and_reject_flag_60s": 0,
                    "breakout_reversal_score_60s": 0.15,
                    "exhaustion_proxy_60s": 0.21,
                    "reasons": ["Tick imbalance supports a clean short-horizon trend"],
                ],
                "GBPJPY": [
                    "broker_symbol": "GBPJPY",
                    "available": 1,
                    "stale": 0,
                    "generated_at": "2026-04-09T09:30:00Z",
                    "spread_current": 2.6,
                    "silent_gap_seconds_current": 1.2,
                    "session_tag": "LONDON_NEWYORK_OVERLAP",
                    "handoff_flag": 1,
                    "minutes_since_session_open": 12,
                    "minutes_to_session_close": 18,
                    "session_open_burst_score": 0.74,
                    "session_spread_behavior_score": 0.66,
                    "liquidity_stress_score": 0.81,
                    "hostile_execution_score": 0.73,
                    "microstructure_regime": "STOP_RUN_RISK",
                    "trade_gate": "CAUTION",
                    "tick_imbalance_30s": -0.12,
                    "directional_efficiency_60s": 0.38,
                    "spread_zscore_60s": 2.24,
                    "tick_rate_60s": 188,
                    "tick_rate_zscore_60s": 2.11,
                    "realized_vol_5m": 1.42,
                    "vol_burst_score_5m": 1.91,
                    "local_extrema_breach_score_60s": 0.76,
                    "sweep_and_reject_flag_60s": 1,
                    "breakout_reversal_score_60s": 0.81,
                    "exhaustion_proxy_60s": 0.73,
                    "reasons": ["Recent breakout rejection detected", "Spread instability elevated"],
                ],
            ],
            "health": [
                "active_symbol_count": 2,
                "snapshot_stale_after_sec": 45,
            ],
            "artifacts": [
                "snapshot_json": "/tmp/microstructure_snapshot.json",
                "history_ndjson": "/tmp/microstructure_history.ndjson",
            ],
        ]
        let data = try JSONSerialization.data(withJSONObject: payload, options: [.prettyPrinted, .sortedKeys])
        try data.write(to: statusURL)

        let snapshot = try #require(MicrostructureArtifactReader().read(projectRoot: projectRoot))
        #expect(snapshot.serviceStatus.ok == true)
        #expect(snapshot.symbols.count == 2)
        #expect(snapshot.symbols.first?.symbol == "GBPJPY")
        #expect(snapshot.symbols.first?.tradeGate == "CAUTION")
        #expect(snapshot.symbols.first?.sweepAndRejectFlag60s == true)
        #expect(snapshot.healthSummary.contains(where: { $0.key == "active_symbol_count" }))
        #expect(snapshot.artifactPaths.contains(where: { $0.key == "snapshot_json" }))
    }

    @Test
    func readerFallsBackToRuntimeFixtureDirectory() throws {
        let projectRoot = FileManager.default.temporaryDirectory
            .appendingPathComponent("fxai-microstructure-runtime-\(UUID().uuidString)", isDirectory: true)
        let runtimeStatus = projectRoot
            .appendingPathComponent("FILE_COMMON/FXAI/Runtime", isDirectory: true)
            .appendingPathComponent("microstructure_status.json", isDirectory: false)
        try FileManager.default.createDirectory(at: runtimeStatus.deletingLastPathComponent(), withIntermediateDirectories: true)
        defer {
            try? FileManager.default.removeItem(at: projectRoot)
        }

        let payload: [String: Any] = [
            "generated_at": "2026-04-09T09:30:00Z",
            "service": ["ok": true, "stale": false, "enabled": true],
            "symbols": [
                "EURUSD": [
                    "broker_symbol": "EURUSD.r",
                    "available": true,
                    "stale": false,
                    "generated_at": "2026-04-09T09:30:00Z",
                    "liquidity_stress_score": 0.22,
                    "hostile_execution_score": 0.18,
                    "microstructure_regime": "NORMAL",
                    "trade_gate": "ALLOW",
                    "tick_imbalance_30s": 0.05,
                    "directional_efficiency_60s": 0.41,
                    "spread_zscore_60s": 0.12,
                    "tick_rate_60s": 118,
                    "tick_rate_zscore_60s": 0.85,
                    "realized_vol_5m": 0.51,
                    "vol_burst_score_5m": 1.02,
                    "local_extrema_breach_score_60s": 0.08,
                    "sweep_and_reject_flag_60s": false,
                    "breakout_reversal_score_60s": 0.06,
                    "exhaustion_proxy_60s": 0.12,
                    "reasons": [],
                ],
            ],
            "health": [:],
            "artifacts": [:],
        ]
        let data = try JSONSerialization.data(withJSONObject: payload, options: [.prettyPrinted, .sortedKeys])
        try data.write(to: runtimeStatus)

        let snapshot = try #require(MicrostructureArtifactReader().read(projectRoot: projectRoot))
        #expect(snapshot.symbols.first?.brokerSymbol == "EURUSD.r")
        #expect(snapshot.symbols.first?.tradeGate == "ALLOW")
    }
}
