from __future__ import annotations

import json
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import offline_lab.newspulse_contracts as contracts
from offline_lab.newspulse_config import build_query_specs, load_config, validate_config_payload
from offline_lab.newspulse_daemon import newspulse_health_snapshot, run_newspulse_daemon
from offline_lab.newspulse_fusion import run_newspulse_cycle
from offline_lab.newspulse_gdelt import merge_normalized_articles, normalize_articles, query_gdelt
from offline_lab.newspulse_official import query_official_feeds
from offline_lab.newspulse_policy import canonical_pair_for_symbol, load_policy, pair_calibration
from offline_lab.newspulse_replay import rebuild_replay_report_from_history
from offline_lab.newspulse_story import build_story_clusters
from offline_lab.fixtures import patched_paths


def _write_calendar_fixture(now: datetime) -> None:
    contracts.COMMON_NEWSPULSE_CALENDAR_STATE.write_text(
        "\n".join(
            [
                "ok\t1",
                "stale\t0",
                f"last_update_at\t{now.replace(microsecond=0).isoformat().replace('+00:00', 'Z')}",
                "change_id\t42",
                "record_count\t2",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    feed = "\t".join(
        [
            "event_id",
            "event_key",
            "title",
            "country_code",
            "country_name",
            "currency",
            "event_time_unix",
            "importance",
            "actual",
            "forecast",
            "previous",
            "revised_previous",
            "surprise_proxy",
            "collector_seen_unix",
            "change_id",
        ]
    )
    rows = [
        "\t".join(
            [
                "1",
                "1|future|0",
                "Fed rate decision",
                "US",
                "United States",
                "USD",
                str(int((now + timedelta(minutes=12)).timestamp())),
                "3",
                "",
                "",
                "",
                "",
                "",
                str(int(now.timestamp())),
                "42",
            ]
        ),
        "\t".join(
            [
                "2",
                "2|past|0",
                "ECB inflation release",
                "EU",
                "Euro Area",
                "EUR",
                str(int((now - timedelta(minutes=4)).timestamp())),
                "3",
                "2.8",
                "2.1",
                "2.0",
                "2.0",
                "0.31",
                str(int(now.timestamp())),
                "42",
            ]
        ),
    ]
    contracts.COMMON_NEWSPULSE_CALENDAR_FEED.write_text(feed + "\n" + "\n".join(rows) + "\n", encoding="utf-8")


def test_newspulse_query_specs_and_config_validation():
    with tempfile.TemporaryDirectory(prefix="fxai_newspulse_cfg_") as tmp_dir:
        with patched_paths(Path(tmp_dir)):
            config, sources = load_config()
            validate_config_payload(config, sources)
            specs = build_query_specs(config)
            assert specs
            assert any(spec["currency"] == "USD" for spec in specs)
            assert all("query_terms" in spec for spec in specs)
            assert int(config["gdelt"]["rate_limit_backoff_sec"]) > 0

            import offline_lab.newspulse_fusion as fusion

            state = {"query_rotation_index": 0}
            first_slice = fusion._select_query_specs(config, state, specs)
            second_slice = fusion._select_query_specs(config, state, specs)
            assert 1 <= len(first_slice) <= int(config["gdelt"]["max_query_sets_per_cycle"])
            assert 1 <= len(second_slice) <= int(config["gdelt"]["max_query_sets_per_cycle"])
            if len(specs) > len(first_slice):
                assert first_slice != second_slice


def test_newspulse_merge_normalized_articles_deduplicates_title_signature():
    items = [
        {
            "id": "a",
            "source": "gdelt",
            "published_at": "2026-04-08T00:00:00Z",
            "seen_at": "2026-04-08T00:01:00Z",
            "currency_tags": ["USD"],
            "topic_tags": ["monetary_policy"],
            "domain": "reuters.com",
            "title": "Fed says rates stay higher for longer",
            "title_signature": "fed says rates stay higher for longer",
            "url": "https://www.reuters.com/a",
            "importance": "",
            "tone": -1.5,
            "tier_weight": 1.0,
            "topic_weight": 1.0,
        },
        {
            "id": "b",
            "source": "gdelt",
            "published_at": "2026-04-08T00:00:00Z",
            "seen_at": "2026-04-08T00:02:00Z",
            "currency_tags": ["USD"],
            "topic_tags": ["inflation"],
            "domain": "reuters.com",
            "title": "Fed says rates stay higher for longer",
            "title_signature": "fed says rates stay higher for longer",
            "url": "",
            "importance": "",
            "tone": -1.2,
            "tier_weight": 1.0,
            "topic_weight": 0.9,
        },
    ]
    merged = merge_normalized_articles(items)
    assert len(merged) == 1
    assert merged[0]["topic_tags"] == ["monetary_policy", "inflation"]


def test_newspulse_normalize_articles_enforces_whitelist():
    with tempfile.TemporaryDirectory(prefix="fxai_newspulse_norm_") as tmp_dir:
        with patched_paths(Path(tmp_dir)):
            _config, sources = load_config()
            accepted = normalize_articles(
                [
                    {
                        "url": "https://www.reuters.com/world/example",
                        "domain": "reuters.com",
                        "title": "Fed officials warn inflation remains sticky",
                        "seendate": "20260408093000",
                        "tone": "-2.5",
                    },
                    {
                        "url": "https://example.com/world/example",
                        "domain": "example.com",
                        "title": "Fed officials warn inflation remains sticky",
                        "seendate": "20260408093000",
                        "tone": "-2.5",
                    },
                ],
                {
                    "currency": "USD",
                    "topic": "inflation",
                    "aliases": ["USD", "Fed"],
                    "query_terms": ["inflation"],
                    "source_countries": ["us"],
                    "weight": 1.0,
                },
                sources,
                "2026-04-08T09:31:00Z",
            )
            assert len(accepted) == 1
            assert accepted[0]["domain"] == "reuters.com"


def test_newspulse_cycle_builds_snapshot_and_flat_export():
    with tempfile.TemporaryDirectory(prefix="fxai_newspulse_cycle_") as tmp_dir:
        with patched_paths(Path(tmp_dir)):
            config, sources = load_config()
            gdelt_now = datetime(2026, 4, 8, 9, 30, tzinfo=timezone.utc)
            _write_calendar_fixture(gdelt_now)

            def fake_query_gdelt(_config, _sources, _query_specs, seen_at_iso):
                return (
                    [
                        {
                            "id": "gdelt-usd-1",
                            "source": "gdelt",
                            "published_at": "2026-04-08T09:25:00Z",
                            "seen_at": seen_at_iso,
                            "currency_tags": ["USD"],
                            "topic_tags": ["monetary_policy"],
                            "domain": "reuters.com",
                            "title": "Dollar jitters ahead of Fed decision",
                            "url": "https://www.reuters.com/markets/usd-1",
                            "importance": "",
                            "tone": -1.8,
                            "tier_weight": 1.0,
                            "topic_weight": 1.0,
                            "title_signature": "dollar jitters ahead of fed decision",
                        }
                    ],
                    {"errors": [], "query_count": len(build_query_specs(config))},
                )

            import offline_lab.newspulse_fusion as fusion

            saved = {
                "query_gdelt": fusion.query_gdelt,
                "query_official_feeds": fusion.query_official_feeds,
                "utc_now": fusion.utc_now,
            }
            fusion.query_gdelt = fake_query_gdelt
            fusion.query_official_feeds = lambda _config, _sources, _seen_at_iso: ([], {"errors": [], "query_count": 0, "success_count": 0})
            fusion.utc_now = lambda: gdelt_now
            try:
                payload = run_newspulse_cycle()
            finally:
                fusion.query_gdelt = saved["query_gdelt"]
                fusion.query_official_feeds = saved["query_official_feeds"]
                fusion.utc_now = saved["utc_now"]

            assert payload["currency_count"] >= 2
            assert contracts.COMMON_NEWSPULSE_JSON.exists()
            assert contracts.COMMON_NEWSPULSE_FLAT.exists()
            assert contracts.COMMON_NEWSPULSE_HISTORY.exists()
            assert contracts.COMMON_NEWSPULSE_REPLAY_FLAT.exists()
            assert contracts.COMMON_NEWSPULSE_SYMBOL_MAP.exists()
            assert contracts.NEWSPULSE_REPLAY_REPORT_PATH.exists()
            snapshot = json.loads(contracts.COMMON_NEWSPULSE_JSON.read_text(encoding="utf-8"))
            assert snapshot["schema_version"] == 2
            assert snapshot["pairs"]["EURUSD"]["trade_gate"] in {"CAUTION", "BLOCK"}
            assert snapshot["pairs"]["EURUSD"]["event_eta_min"] == 12
            assert "USD high-impact event in 12 minutes" in snapshot["pairs"]["EURUSD"]["reasons"]
            assert snapshot["pairs"]["EURUSD"]["session_profile"]
            assert snapshot["pairs"]["EURUSD"]["calibration_profile"]
            flat = contracts.COMMON_NEWSPULSE_FLAT.read_text(encoding="utf-8")
            assert "pair\tEURUSD\ttrade_gate\t" in flat
            assert "currency\tUSD\tburst_score_15m\t" in flat
            assert "pair\tEURUSD\tcaution_lot_scale\t" in flat
            symbol_map = contracts.COMMON_NEWSPULSE_SYMBOL_MAP.read_text(encoding="utf-8")
            assert "symbol\tEURUSD\tEURUSD" in symbol_map
            replay = json.loads(contracts.NEWSPULSE_REPLAY_REPORT_PATH.read_text(encoding="utf-8"))
            assert "EURUSD" in replay["pair_timelines"]


def test_newspulse_stale_source_blocks_pairs():
    with tempfile.TemporaryDirectory(prefix="fxai_newspulse_stale_") as tmp_dir:
        with patched_paths(Path(tmp_dir)):
            _write_calendar_fixture(datetime(2026, 4, 8, 9, 30, tzinfo=timezone.utc))
            contracts.COMMON_NEWSPULSE_CALENDAR_STATE.write_text(
                "ok\t0\nstale\t1\nlast_update_at\t2026-04-08T09:30:00Z\nchange_id\t42\nrecord_count\t0\n",
                encoding="utf-8",
            )

            def fake_query_gdelt(_config, _sources, _query_specs, _seen_at_iso):
                return [], {"errors": ["network"], "query_count": 0}

            import offline_lab.newspulse_fusion as fusion

            saved = {
                "query_gdelt": fusion.query_gdelt,
                "query_official_feeds": fusion.query_official_feeds,
                "utc_now": fusion.utc_now,
            }
            fusion.query_gdelt = fake_query_gdelt
            fusion.query_official_feeds = lambda _config, _sources, _seen_at_iso: ([], {"errors": [], "query_count": 0, "success_count": 0})
            fusion.utc_now = lambda: datetime(2026, 4, 8, 9, 31, tzinfo=timezone.utc)
            try:
                run_newspulse_cycle()
            finally:
                fusion.query_gdelt = saved["query_gdelt"]
                fusion.query_official_feeds = saved["query_official_feeds"]
                fusion.utc_now = saved["utc_now"]

            snapshot = json.loads(contracts.COMMON_NEWSPULSE_JSON.read_text(encoding="utf-8"))
            assert snapshot["pairs"]["EURUSD"]["trade_gate"] == "BLOCK"
            assert snapshot["currencies"]["USD"]["stale"] is True


def test_newspulse_future_state_timestamps_are_sanitized():
    with tempfile.TemporaryDirectory(prefix="fxai_newspulse_future_") as tmp_dir:
        with patched_paths(Path(tmp_dir)):
            _write_calendar_fixture(datetime(2026, 4, 8, 9, 30, tzinfo=timezone.utc))

            import offline_lab.newspulse_fusion as fusion

            future_now = datetime(2026, 4, 8, 9, 31, tzinfo=timezone.utc)
            fusion.NEWSPULSE_STATE_PATH.write_text(
                json.dumps(
                    {
                        "last_gdelt_poll_at": "2026-04-08T12:00:00Z",
                        "last_gdelt_success_at": "2026-04-08T12:00:00Z",
                        "gdelt_backoff_until": "2026-04-08T12:10:00Z",
                        "currency_baselines": {},
                        "seen_items": {},
                        "query_rotation_index": 0,
                    }
                ),
                encoding="utf-8",
            )

            def fake_query_gdelt(_config, _sources, _query_specs, _seen_at_iso):
                return [], {"errors": [], "query_count": 0, "success_count": 0, "throttled": False}

            saved = {
                "query_gdelt": fusion.query_gdelt,
                "query_official_feeds": fusion.query_official_feeds,
                "utc_now": fusion.utc_now,
            }
            fusion.query_gdelt = fake_query_gdelt
            fusion.query_official_feeds = lambda _config, _sources, _seen_at_iso: ([], {"errors": [], "query_count": 0, "success_count": 0})
            fusion.utc_now = lambda: future_now
            try:
                run_newspulse_cycle()
            finally:
                fusion.query_gdelt = saved["query_gdelt"]
                fusion.query_official_feeds = saved["query_official_feeds"]
                fusion.utc_now = saved["utc_now"]

            state = json.loads(fusion.NEWSPULSE_STATE_PATH.read_text(encoding="utf-8"))
            assert state["last_gdelt_poll_at"] == "2026-04-08T09:31:00Z"
            assert state["last_gdelt_success_at"] == ""
            assert state["gdelt_backoff_until"] == ""


def test_newspulse_gdelt_query_stops_on_rate_limit():
    import offline_lab.newspulse_gdelt as gdelt

    calls: list[str] = []

    def fake_fetch(query_spec, _sources, _timeout, _allow_insecure_tls_fallback):
        calls.append(f"{query_spec['currency']}:{query_spec['topic']}")
        if len(calls) == 2:
            from urllib.error import HTTPError

            raise HTTPError("https://example.test", 429, "Too Many Requests", hdrs=None, fp=None)
        return []

    config = {
        "gdelt": {
            "request_timeout_sec": 1,
            "allow_insecure_tls_fallback": False,
        }
    }
    specs = [
        {"currency": "USD", "topic": "monetary_policy", "aliases": ["USD"], "query_terms": ["rates"]},
        {"currency": "EUR", "topic": "inflation", "aliases": ["EUR"], "query_terms": ["inflation"]},
        {"currency": "JPY", "topic": "banking_stress", "aliases": ["JPY"], "query_terms": ["bank"]},
    ]
    saved = gdelt.fetch_query_results
    gdelt.fetch_query_results = fake_fetch
    try:
        _items, meta = query_gdelt(config, {"domains": {}, "tier_weights": {}}, specs, "2026-04-08T09:31:00Z")
    finally:
        gdelt.fetch_query_results = saved

    assert calls == ["USD:monetary_policy", "EUR:inflation"]
    assert meta["throttled"] is True
    assert meta["success_count"] == 1
    assert meta["query_count"] == 2


def test_newspulse_policy_mapping_and_calibration():
    with tempfile.TemporaryDirectory(prefix="fxai_newspulse_policy_") as tmp_dir:
        with patched_paths(Path(tmp_dir)):
            policy = load_policy()
            policy["broker_symbol_map"]["EURUSD.r"] = "EURUSD"
            mapped = canonical_pair_for_symbol("EURUSD.r", policy)
            assert mapped == "EURUSD"
            heuristic = canonical_pair_for_symbol("mGBPJPY.pro", policy)
            assert heuristic == "GBPJPY"

            calibration = pair_calibration("GBPJPY", policy, datetime(2026, 4, 8, 13, 0, tzinfo=timezone.utc))
            assert calibration["session_profile"] in {"newyork", "overlap"}
            assert calibration["calibration_profile"]
            assert float(calibration["caution_lot_scale"]) < 0.65
            assert float(calibration["enter_prob_buffer"]) >= 0.06


def test_newspulse_official_feed_queries_respect_whitelist_and_keywords():
    with tempfile.TemporaryDirectory(prefix="fxai_newspulse_official_") as tmp_dir:
        with patched_paths(Path(tmp_dir)):
            config, sources = load_config()

            def fake_fetch(feed_spec, _timeout):
                return [
                    {
                        "title": "Federal Reserve issues monetary policy statement",
                        "url": "https://www.federalreserve.gov/newsevents/pressreleases/monetary20260408a.htm",
                        "published_at": "2026-04-08T09:20:00Z",
                        "identifier": "fed-a",
                    },
                    {
                        "title": "Community event update",
                        "url": "https://www.federalreserve.gov/newsevents/community.htm",
                        "published_at": "2026-04-08T09:18:00Z",
                        "identifier": "fed-b",
                    },
                ]

            import offline_lab.newspulse_official as official

            saved = official.fetch_official_feed
            official.fetch_official_feed = fake_fetch
            try:
                items, meta = query_official_feeds(config, sources, "2026-04-08T09:30:00Z")
            finally:
                official.fetch_official_feed = saved

            assert meta["query_count"] >= 1
            assert meta["success_count"] >= 1
            assert any(item["source"] == "official" for item in items)
            assert all(item["domain"] in {"federalreserve.gov", "ecb.europa.eu", "bankofengland.co.uk", "bankofcanada.ca"} for item in items)
            assert any("policy statement" in item["title"].lower() for item in items)


def test_newspulse_story_clusters_track_evolving_story():
    now = datetime(2026, 4, 8, 10, 0, tzinfo=timezone.utc)
    items = [
        {
            "id": "a",
            "currency_tags": ["USD"],
            "topic_tags": ["monetary_policy"],
            "domain": "reuters.com",
            "title": "Fed officials signal patience after inflation surprise",
            "url": "https://example.com/a",
            "published_at": "2026-04-08T09:30:00Z",
            "source": "gdelt",
            "tier_weight": 1.0,
        },
        {
            "id": "b",
            "currency_tags": ["USD"],
            "topic_tags": ["monetary_policy"],
            "domain": "federalreserve.gov",
            "title": "Fed officials signal patience after inflation surprise",
            "url": "https://example.com/b",
            "published_at": "2026-04-08T09:42:00Z",
            "source": "official",
            "tier_weight": 1.2,
        },
    ]
    stories, item_map = build_story_clusters(items, now)
    assert len(stories) == 1
    assert item_map["a"] == item_map["b"] == stories[0]["id"]
    assert stories[0]["official_hits"] == 1
    assert stories[0]["active"] is True
    assert stories[0]["severity_score"] > 0.5


def test_newspulse_replay_report_rebuilds_from_history():
    with tempfile.TemporaryDirectory(prefix="fxai_newspulse_replay_") as tmp_dir:
        with patched_paths(Path(tmp_dir)):
            contracts.COMMON_NEWSPULSE_HISTORY.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "record_type": "snapshot",
                                "snapshot": {
                                    "generated_at": "2026-04-08T09:30:00Z",
                                    "pairs": {
                                        "EURUSD": {
                                            "trade_gate": "CAUTION",
                                            "news_risk_score": 0.62,
                                            "news_pressure": -0.08,
                                            "stale": False,
                                            "event_eta_min": 12,
                                            "session_profile": "london",
                                            "calibration_profile": "london:dollar_core:pair",
                                            "watchlist_tags": ["active", "macro_sensitive"],
                                            "story_ids": ["story-1"],
                                            "reasons": ["USD high-impact event in 12 minutes"],
                                        }
                                    },
                                    "currencies": {
                                        "USD": {"risk_score": 0.62},
                                        "EUR": {"risk_score": 0.44},
                                    },
                                    "source_status": {
                                        "calendar": {"ok": True, "stale": False},
                                        "gdelt": {"ok": True, "stale": False},
                                        "official": {"ok": True, "stale": False},
                                    },
                                    "stories": [{"id": "story-1", "latest_title": "Fed preview"}],
                                },
                            },
                            sort_keys=True,
                        ),
                        "",
                    ]
                ),
                encoding="utf-8",
            )

            report = rebuild_replay_report_from_history(history_path=contracts.COMMON_NEWSPULSE_HISTORY, hours_back=72)
            assert "EURUSD" in report["pair_timelines"]
            assert report["top_pairs"][0]["pair"] == "EURUSD"
            assert contracts.NEWSPULSE_REPLAY_REPORT_PATH.exists()


def test_newspulse_daemon_health_captures_failures_and_exits():
    with tempfile.TemporaryDirectory(prefix="fxai_newspulse_daemon_") as tmp_dir:
        with patched_paths(Path(tmp_dir)):
            import offline_lab.newspulse_daemon as daemon

            saved = {
                "run_newspulse_cycle": daemon.run_newspulse_cycle,
                "load_config": daemon.load_config,
                "time_sleep": daemon.time.sleep,
            }
            daemon.load_config = lambda: ({"poll_interval_sec": 15}, {})
            daemon.run_newspulse_cycle = lambda daemon_context=None: (_ for _ in ()).throw(RuntimeError("boom"))
            daemon.time.sleep = lambda _seconds: None
            try:
                payload = run_newspulse_daemon(iterations=1, interval_seconds=15)
            finally:
                daemon.run_newspulse_cycle = saved["run_newspulse_cycle"]
                daemon.load_config = saved["load_config"]
                daemon.time.sleep = saved["time_sleep"]

            health = newspulse_health_snapshot()
            assert payload["iterations"] == 1
            assert payload["successful_iterations"] == 0
            assert payload["consecutive_failures"] == 1
            assert health["daemon"]["degraded"] is True
            assert health["daemon"]["last_error"] == "boom"
