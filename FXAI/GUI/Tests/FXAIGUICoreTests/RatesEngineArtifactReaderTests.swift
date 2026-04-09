import FXAIGUICore
import Foundation
import Testing

struct RatesEngineArtifactReaderTests {
    @Test
    func readerParsesRatesEngineStatusPayload() throws {
        let projectRoot = FileManager.default.temporaryDirectory
            .appendingPathComponent("fxai-rates-reader-\(UUID().uuidString)", isDirectory: true)
        let statusURL = projectRoot
            .appendingPathComponent("Tools/OfflineLab/RatesEngine", isDirectory: true)
            .appendingPathComponent("rates_engine_status.json", isDirectory: false)
        try FileManager.default.createDirectory(at: statusURL.deletingLastPathComponent(), withIntermediateDirectories: true)

        defer {
            try? FileManager.default.removeItem(at: projectRoot)
        }

        let payload: [String: Any] = [
            "generated_at": "2026-04-09T09:30:00Z",
            "source_status": [
                "manual_inputs": [
                    "ok": true,
                    "stale": false,
                    "enabled": true,
                    "required": false,
                    "last_update_at": "2026-04-09T08:00:00Z",
                    "mode": "manual_market_input",
                    "coverage_ratio": 0.3,
                    "updated_currencies": 3,
                ],
                "proxy_engine": [
                    "ok": true,
                    "stale": false,
                    "enabled": true,
                    "required": false,
                    "last_update_at": "2026-04-09T09:30:00Z",
                    "mode": "newspulse_policy_proxy",
                ],
            ],
            "currencies": [
                "USD": [
                    "front_end_level": 0.88,
                    "front_end_basis": "policy_proxy_index",
                    "front_end_change_1d": 0.14,
                    "front_end_change_5d": 0.38,
                    "expected_path_level": 1.04,
                    "expected_path_basis": "policy_proxy_index",
                    "expected_path_change_1d": 0.22,
                    "expected_path_change_5d": 0.51,
                    "curve_shape_regime": "UNAVAILABLE",
                    "curve_basis": "unavailable",
                    "policy_repricing_score": 0.82,
                    "policy_surprise_score": 0.61,
                    "policy_uncertainty_score": 0.35,
                    "policy_direction_score": 0.47,
                    "policy_relevance_score": 0.78,
                    "pre_cb_event_window": false,
                    "post_cb_event_window": true,
                    "pre_macro_policy_window": false,
                    "post_macro_policy_window": true,
                    "meeting_path_reprice_now": true,
                    "macro_to_rates_transmission_score": 0.69,
                    "stale": false,
                    "reasons": ["USD central-bank repricing window active"],
                ],
            ],
            "pairs": [
                "EURUSD": [
                    "base_currency": "EUR",
                    "quote_currency": "USD",
                    "front_end_diff": -0.62,
                    "expected_path_diff": -0.91,
                    "curve_divergence_score": 0.19,
                    "policy_divergence_score": 0.73,
                    "rates_regime": "UNSTABLE",
                    "rates_risk_score": 0.81,
                    "trade_gate": "BLOCK",
                    "policy_alignment": "quote_hawkish",
                    "meeting_path_reprice_now": true,
                    "macro_to_rates_transmission_score": 0.69,
                    "stale": false,
                    "broker_symbols": ["EURUSD", "EURUSD.r"],
                    "reasons": ["meeting path repricing active", "policy divergence meaningful"],
                ],
            ],
            "recent_policy_events": [
                [
                    "id": "evt-fed-1",
                    "currency": "USD",
                    "source": "official",
                    "domain": "federalreserve.gov",
                    "published_at": "2026-04-09T09:24:00Z",
                    "title": "Federal Reserve policy statement",
                    "url": "https://example.test/fed",
                    "policy_relevance_score": 0.92,
                    "direction": 0.58,
                    "central_bank_event": true,
                    "macro_policy_event": false,
                ],
            ],
            "health": [
                "required_sources_stale": false,
                "history_records_local": 12,
                "pair_count": 54,
                "currency_count": 10,
            ],
            "artifacts": [
                "snapshot_json": "/tmp/rates_snapshot.json",
                "symbol_map_tsv": "/tmp/rates_symbol_map.tsv",
            ],
        ]

        let data = try JSONSerialization.data(withJSONObject: payload, options: [.prettyPrinted, .sortedKeys])
        try data.write(to: statusURL)

        let snapshot = try #require(RatesEngineArtifactReader().read(projectRoot: projectRoot))
        #expect(snapshot.sourceStatuses.count == 2)
        #expect(snapshot.currencies.first?.currency == "USD")
        #expect(snapshot.currencies.first?.postCBEventWindow == true)
        #expect(snapshot.pairs.first?.pair == "EURUSD")
        #expect(snapshot.pairs.first?.tradeGate == "BLOCK")
        #expect(snapshot.pairs.first?.brokerSymbols.contains("EURUSD.r") == true)
        #expect(snapshot.recentPolicyEvents.first?.currency == "USD")
        #expect(snapshot.healthSummary.contains(where: { $0.key == "pair_count" }))
        #expect(snapshot.artifactPaths.contains(where: { $0.key == "snapshot_json" }))
    }

    @Test
    func readerReturnsNilWhenStatusArtifactIsMissing() {
        let projectRoot = FileManager.default.temporaryDirectory
            .appendingPathComponent("fxai-rates-reader-missing-\(UUID().uuidString)", isDirectory: true)
        defer {
            try? FileManager.default.removeItem(at: projectRoot)
        }

        #expect(RatesEngineArtifactReader().read(projectRoot: projectRoot) == nil)
    }
}
