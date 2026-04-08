import FXAIGUICore
import Foundation
import Testing

struct NewsPulseArtifactReaderTests {
    @Test
    func readerParsesRichNewsPulseStatusPayload() throws {
        let projectRoot = FileManager.default.temporaryDirectory
            .appendingPathComponent("fxai-newspulse-reader-\(UUID().uuidString)", isDirectory: true)
        let statusURL = projectRoot
            .appendingPathComponent("Tools/OfflineLab/NewsPulse", isDirectory: true)
            .appendingPathComponent("newspulse_status.json", isDirectory: false)
        try FileManager.default.createDirectory(at: statusURL.deletingLastPathComponent(), withIntermediateDirectories: true)

        defer {
            try? FileManager.default.removeItem(at: projectRoot)
        }

        let payload: [String: Any] = [
            "generated_at": "2026-04-08T09:30:00Z",
            "query_count": 8,
            "official_query_count": 3,
            "source_status": [
                "calendar": [
                    "ok": true,
                    "stale": false,
                    "enabled": true,
                    "required": true,
                    "last_update_at": "2026-04-08T09:29:00Z",
                    "cursor": "42",
                ],
                "gdelt": [
                    "ok": true,
                    "stale": false,
                    "enabled": true,
                    "required": true,
                    "last_poll_at": "2026-04-08T09:30:00Z",
                    "last_success_at": "2026-04-08T09:30:00Z",
                ],
                "official": [
                    "ok": false,
                    "stale": true,
                    "enabled": true,
                    "required": false,
                    "last_poll_at": "2026-04-08T09:30:00Z",
                    "backoff_until": "2026-04-08T09:35:00Z",
                    "last_error": "timeout",
                ],
            ],
            "currencies": [
                "USD": [
                    "breaking_count_15m": 4,
                    "intensity_15m": 1.42,
                    "tone_mean_15m": -0.8,
                    "tone_abs_mean_15m": 1.1,
                    "burst_score_15m": 1.7,
                    "story_count_15m": 2,
                    "story_severity_15m": 0.66,
                    "official_count_24h": 1,
                    "dominant_story_ids": ["story-usd-1"],
                    "next_high_impact_eta_min": 11,
                    "in_pre_event_window": true,
                    "in_post_event_window": false,
                    "stale": false,
                    "risk_score": 0.81,
                    "reasons": ["USD high-impact event in 11 minutes"],
                ],
            ],
            "pairs": [
                "EURUSD": [
                    "base_currency": "EUR",
                    "quote_currency": "USD",
                    "event_eta_min": 11,
                    "news_risk_score": 0.83,
                    "trade_gate": "BLOCK",
                    "news_pressure": -0.14,
                    "stale": false,
                    "reasons": ["pair event block window active (11m <= 12m)"],
                    "story_ids": ["story-usd-1"],
                    "watchlist_tags": ["active", "macro_sensitive", "dollar_core"],
                    "broker_symbols": ["EURUSD", "EURUSD.r"],
                    "session_profile": "overlap",
                    "calibration_profile": "overlap:dollar_core:pair",
                    "caution_lot_scale": 0.56,
                    "caution_enter_prob_buffer": 0.07,
                    "gate_changed_at": "2026-04-08T09:24:00Z",
                ],
            ],
            "stories": [
                [
                    "id": "story-usd-1",
                    "latest_title": "Fed officials signal patience after inflation surprise",
                    "first_published_at": "2026-04-08T09:05:00Z",
                    "last_published_at": "2026-04-08T09:28:00Z",
                    "currency_tags": ["USD"],
                    "topic_tags": ["monetary_policy"],
                    "domains": ["reuters.com", "federalreserve.gov"],
                    "representative_url": "https://example.test/story",
                    "item_count": 3,
                    "source_count": 2,
                    "official_hits": 1,
                    "severity_score": 0.72,
                    "active": true,
                    "item_ids": ["item-1", "item-2"],
                ],
            ],
            "recent_items": [
                [
                    "id": "item-1",
                    "source": "official",
                    "published_at": "2026-04-08T09:28:00Z",
                    "seen_at": "2026-04-08T09:30:00Z",
                    "currency_tags": ["USD"],
                    "topic_tags": ["monetary_policy"],
                    "domain": "federalreserve.gov",
                    "title": "Federal Reserve policy statement",
                    "url": "https://example.test/official",
                    "importance": "official",
                    "tone": 0.0,
                    "story_id": "story-usd-1",
                ],
            ],
            "pair_timelines": [
                "EURUSD": [
                    [
                        "observed_at": "2026-04-08T09:20:00Z",
                        "trade_gate": "CAUTION",
                        "news_risk_score": 0.63,
                        "news_pressure": -0.09,
                        "stale": false,
                        "event_eta_min": 21,
                        "session_profile": "london",
                        "calibration_profile": "london:dollar_core:pair",
                        "watchlist_tags": ["active"],
                        "story_ids": ["story-usd-1"],
                        "reasons": ["pair event caution window active (21m <= 28m)"],
                    ],
                    [
                        "observed_at": "2026-04-08T09:28:00Z",
                        "trade_gate": "BLOCK",
                        "news_risk_score": 0.83,
                        "news_pressure": -0.14,
                        "stale": false,
                        "event_eta_min": 11,
                        "session_profile": "overlap",
                        "calibration_profile": "overlap:dollar_core:pair",
                        "watchlist_tags": ["active", "macro_sensitive"],
                        "story_ids": ["story-usd-1"],
                        "reasons": ["pair event block window active (11m <= 12m)"],
                    ],
                ],
            ],
            "source_health_timeline": [
                [
                    "observed_at": "2026-04-08T09:20:00Z",
                    "calendar_ok": true,
                    "calendar_stale": false,
                    "gdelt_ok": true,
                    "gdelt_stale": false,
                    "official_ok": true,
                    "official_stale": false,
                ],
                [
                    "observed_at": "2026-04-08T09:30:00Z",
                    "calendar_ok": true,
                    "calendar_stale": false,
                    "gdelt_ok": true,
                    "gdelt_stale": false,
                    "official_ok": false,
                    "official_stale": true,
                ],
            ],
            "daemon": [
                "mode": "daemon",
                "heartbeat_at": "2026-04-08T09:30:00Z",
                "last_cycle_started_at": "2026-04-08T09:29:50Z",
                "last_cycle_finished_at": "2026-04-08T09:30:00Z",
                "last_cycle_duration_sec": 4.2,
                "interval_seconds": 60,
                "cycles_completed": 18,
                "consecutive_failures": 1,
                "degraded": true,
                "degraded_reasons": ["official_feed_not_ready"],
                "last_error": "timeout",
            ],
            "policy_summary": [
                "active_pairs": ["EURUSD", "GBPUSD"],
                "watchlists": [
                    "active": ["EURUSD", "GBPUSD"],
                    "macro_sensitive": ["EURUSD"],
                ],
                "broker_symbol_map_count": 2,
            ],
            "health": [
                "required_sources_stale": false,
                "gdelt_backoff_until": "2026-04-08T09:34:00Z",
                "history_records_local": 24,
                "story_count": 3,
            ],
            "artifacts": [
                "snapshot_json": "/tmp/news_snapshot.json",
                "policy_json": "/tmp/newspulse_policy.json",
                "replay_report_json": "/tmp/newspulse_replay_report.json",
            ],
        ]

        let data = try JSONSerialization.data(withJSONObject: payload, options: [.prettyPrinted, .sortedKeys])
        try data.write(to: statusURL)

        let snapshot = try #require(NewsPulseArtifactReader().read(projectRoot: projectRoot))
        #expect(snapshot.queryCount == 8)
        #expect(snapshot.officialQueryCount == 3)
        #expect(snapshot.sourceStatuses.count == 3)
        #expect(snapshot.stories.count == 1)
        #expect(snapshot.pairs.first?.pair == "EURUSD")
        #expect(snapshot.pairs.first?.sessionProfile == "overlap")
        #expect(snapshot.pairs.first?.brokerSymbols.contains("EURUSD.r") == true)
        #expect(snapshot.pairTimelines["EURUSD"]?.count == 2)
        #expect(snapshot.sourceHealthTimeline.count == 2)
        #expect(snapshot.daemon?.degraded == true)
        #expect(snapshot.policySummary?.brokerSymbolMapCount == 2)
        #expect(snapshot.healthSummary?.historyRecordsLocal == 24)
        #expect(snapshot.artifactPaths.contains(where: { $0.key == "policy_json" }))
    }

    @Test
    func readerReturnsNilWhenStatusArtifactIsMissing() {
        let projectRoot = FileManager.default.temporaryDirectory
            .appendingPathComponent("fxai-newspulse-reader-missing-\(UUID().uuidString)", isDirectory: true)
        defer {
            try? FileManager.default.removeItem(at: projectRoot)
        }

        #expect(NewsPulseArtifactReader().read(projectRoot: projectRoot) == nil)
    }
}
