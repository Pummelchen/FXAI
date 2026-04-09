from __future__ import annotations

import json
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import offline_lab.newspulse_contracts as newspulse_contracts
import offline_lab.rates_engine as engine
import offline_lab.rates_engine_contracts as contracts
import offline_lab.rates_engine_replay as replay
from offline_lab.fixtures import patched_paths
from offline_lab.rates_engine_daemon import validate_rates_engine_config
from offline_lab.rates_engine_inputs import load_inputs
from offline_lab.rates_engine_newspulse import apply_rates_enrichment


def _iso(dt: datetime) -> str:
    return dt.replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _write_newspulse_snapshot(
    now: datetime,
    *,
    generated_at: datetime | None = None,
    extra_items: list[dict] | None = None,
) -> None:
    snapshot_generated_at = generated_at or now
    payload = {
        "schema_version": 2,
        "generated_at": _iso(snapshot_generated_at),
        "source_status": {
            "calendar": {
                "ok": True,
                "stale": False,
                "last_update_at": _iso(snapshot_generated_at),
            },
            "gdelt": {
                "ok": True,
                "stale": False,
                "last_success_at": _iso(snapshot_generated_at),
            },
        },
        "currencies": {
            "USD": {
                "breaking_count_15m": 3,
                "burst_score_15m": 1.1,
                "story_severity_15m": 0.58,
                "official_count_24h": 1,
                "last_surprise_proxy": 0.46,
                "risk_score": 0.62,
                "stale": False,
            },
            "EUR": {
                "breaking_count_15m": 1,
                "burst_score_15m": 0.18,
                "story_severity_15m": 0.14,
                "official_count_24h": 0,
                "last_surprise_proxy": -0.08,
                "risk_score": 0.22,
                "stale": False,
            },
        },
        "recent_items": [
            {
                "id": "usd-fed",
                "source": "official",
                "published_at": _iso(now - timedelta(minutes=6)),
                "seen_at": _iso(now),
                "currency_tags": ["USD"],
                "topic_tags": ["monetary_policy"],
                "domain": "federalreserve.gov",
                "title": "Federal Reserve policy statement signals higher for longer stance",
                "url": "https://example.test/fed",
                "tone": -0.4,
            },
            {
                "id": "eur-cpi",
                "source": "calendar",
                "published_at": _iso(now - timedelta(minutes=12)),
                "seen_at": _iso(now),
                "currency_tags": ["EUR"],
                "topic_tags": ["scheduled_macro", "inflation"],
                "domain": "",
                "title": "Euro area CPI surprise holds above forecast",
                "url": "",
                "tone": 0.0,
            },
            *list(extra_items or []),
        ],
    }
    newspulse_contracts.COMMON_NEWSPULSE_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    newspulse_contracts.NEWSPULSE_STATUS_PATH.write_text(
        json.dumps(
            {
                "generated_at": _iso(snapshot_generated_at),
                "health": {"snapshot_stale_after_sec": 900},
                "source_status": payload["source_status"],
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def test_rates_engine_validate_creates_default_config_and_inputs():
    with tempfile.TemporaryDirectory(prefix="fxai_rates_validate_") as tmp_dir:
        with patched_paths(Path(tmp_dir)):
            payload = validate_rates_engine_config()
            assert payload["ok"] is True
            assert Path(payload["config_path"]).exists()
            assert Path(payload["inputs_path"]).exists()
            assert "USD" in payload["currencies"]
            assert payload["manual_input_currencies"] == []


def test_rates_engine_cycle_builds_snapshot_exports_and_manual_proxy_mix():
    with tempfile.TemporaryDirectory(prefix="fxai_rates_cycle_") as tmp_dir:
        with patched_paths(Path(tmp_dir)):
            now = datetime(2026, 4, 9, 9, 30, tzinfo=timezone.utc)
            _write_newspulse_snapshot(now)

            inputs = load_inputs()
            inputs["currencies"]["EUR"].update(
                {
                    "front_end_level": 0.25,
                    "expected_path_level": 0.18,
                    "curve_2y_level": 2.05,
                    "curve_10y_level": 1.62,
                    "last_update_at": _iso(now - timedelta(hours=3)),
                    "basis": "manual_market_input",
                }
            )
            contracts.RATES_ENGINE_INPUTS_PATH.write_text(
                json.dumps(inputs, indent=2, sort_keys=True),
                encoding="utf-8",
            )

            saved_engine_now = engine.utc_now
            saved_replay_now = replay.utc_now
            engine.utc_now = lambda: now
            replay.utc_now = lambda: now
            try:
                payload = engine.run_rates_engine_cycle()
            finally:
                engine.utc_now = saved_engine_now
                replay.utc_now = saved_replay_now

            assert payload["currency_count"] >= 8
            assert payload["pair_count"] > 0
            assert Path(payload["snapshot_path"]).exists()
            assert Path(payload["flat_path"]).exists()
            assert Path(payload["history_path"]).exists()
            assert contracts.COMMON_RATES_SYMBOL_MAP.exists()
            assert contracts.RATES_ENGINE_STATUS_PATH.exists()
            assert contracts.RATES_ENGINE_REPLAY_REPORT_PATH.exists()

            snapshot = json.loads(contracts.COMMON_RATES_JSON.read_text(encoding="utf-8"))
            assert snapshot["schema_version"] == contracts.RATES_ENGINE_SCHEMA_VERSION
            assert snapshot["currencies"]["EUR"]["front_end_basis"] == "manual_market_input"
            assert snapshot["currencies"]["USD"]["front_end_basis"] == "policy_proxy_index"
            assert snapshot["pairs"]["EURUSD"]["trade_gate"] in {"ALLOW", "CAUTION", "BLOCK"}
            assert snapshot["pairs"]["EURUSD"]["broker_symbols"] == []
            assert snapshot["recent_policy_events"]

            flat = contracts.COMMON_RATES_FLAT.read_text(encoding="utf-8")
            assert "pair\tEURUSD\ttrade_gate\t" in flat
            assert "currency\tUSD\tpolicy_repricing_score\t" in flat

            symbol_map = contracts.COMMON_RATES_SYMBOL_MAP.read_text(encoding="utf-8")
            assert "symbol\tEURUSD\tEURUSD" in symbol_map

            status = json.loads(contracts.RATES_ENGINE_STATUS_PATH.read_text(encoding="utf-8"))
            assert status["health"]["currency_count"] >= 8
            assert status["health"]["snapshot_stale_after_sec"] == 900


def test_rates_engine_blocks_pairs_when_proxy_state_is_stale_and_no_manual_inputs():
    with tempfile.TemporaryDirectory(prefix="fxai_rates_stale_") as tmp_dir:
        with patched_paths(Path(tmp_dir)):
            now = datetime(2026, 4, 9, 12, 0, tzinfo=timezone.utc)
            _write_newspulse_snapshot(now, generated_at=now - timedelta(hours=2))

            saved_engine_now = engine.utc_now
            saved_replay_now = replay.utc_now
            engine.utc_now = lambda: now
            replay.utc_now = lambda: now
            try:
                engine.run_rates_engine_cycle()
            finally:
                engine.utc_now = saved_engine_now
                replay.utc_now = saved_replay_now

            snapshot = json.loads(contracts.COMMON_RATES_JSON.read_text(encoding="utf-8"))
            assert snapshot["source_status"]["proxy_engine"]["stale"] is True
            assert snapshot["currencies"]["USD"]["stale"] is True
            assert snapshot["pairs"]["EURUSD"]["trade_gate"] == "BLOCK"
            assert "rates state stale or incomplete" in snapshot["pairs"]["EURUSD"]["reasons"]


def test_rates_engine_enriches_newspulse_snapshot_with_rates_context():
    with tempfile.TemporaryDirectory(prefix="fxai_rates_enrichment_") as tmp_dir:
        with patched_paths(Path(tmp_dir)):
            rates_snapshot = {
                "generated_at": "2026-04-09T09:30:00Z",
                "source_status": {"manual_inputs": {"ok": True, "stale": False}},
                "currencies": {
                    "USD": {
                        "policy_relevance_score": 0.77,
                        "policy_repricing_score": 0.66,
                        "policy_surprise_score": 0.52,
                        "policy_uncertainty_score": 0.29,
                        "curve_shape_regime": "NEUTRAL",
                        "front_end_basis": "policy_proxy_index",
                        "expected_path_basis": "policy_proxy_index",
                        "stale": False,
                        "reasons": ["USD central-bank repricing window active"],
                        "macro_to_rates_transmission_score": 0.61,
                        "meeting_path_reprice_now": True,
                    }
                },
                "pairs": {
                    "EURUSD": {
                        "rates_risk_score": 0.71,
                        "trade_gate": "CAUTION",
                        "rates_regime": "UNSTABLE",
                        "policy_divergence_score": 0.68,
                        "curve_divergence_score": 0.18,
                        "meeting_path_reprice_now": True,
                        "macro_to_rates_transmission_score": 0.61,
                        "policy_alignment": "quote_hawkish",
                        "stale": False,
                        "reasons": ["meeting path repricing active"],
                    }
                },
            }
            contracts.COMMON_RATES_JSON.write_text(json.dumps(rates_snapshot, indent=2, sort_keys=True), encoding="utf-8")

            enriched = apply_rates_enrichment(
                {
                    "currencies": {"USD": {"stale": False}},
                    "pairs": {"EURUSD": {"trade_gate": "BLOCK"}},
                    "recent_items": [
                        {
                            "id": "item-1",
                            "currency_tags": ["USD"],
                            "topic_tags": ["monetary_policy"],
                            "title": "Fed statement",
                        }
                    ],
                }
            )

            assert enriched["rates_enrichment"]["available"] is True
            assert enriched["currencies"]["USD"]["rates_context"]["policy_relevance_score"] == 0.77
            assert enriched["pairs"]["EURUSD"]["rates_context"]["trade_gate"] == "CAUTION"
            assert enriched["recent_items"][0]["path_repricing_after_event"] is True
            assert enriched["recent_items"][0]["macro_release_rates_impact"] == 0.61


def test_rates_engine_replay_report_tracks_transitions():
    with tempfile.TemporaryDirectory(prefix="fxai_rates_replay_") as tmp_dir:
        with patched_paths(Path(tmp_dir)):
            now = datetime(2026, 4, 9, 15, 0, tzinfo=timezone.utc)
            records = [
                {
                    "record_type": "snapshot",
                    "generated_at": _iso(now - timedelta(hours=2)),
                    "snapshot": {
                        "generated_at": _iso(now - timedelta(hours=2)),
                        "pairs": {
                            "EURUSD": {
                                "trade_gate": "ALLOW",
                                "rates_regime": "NEUTRAL",
                                "rates_risk_score": 0.21,
                                "policy_divergence_score": 0.31,
                                "meeting_path_reprice_now": False,
                                "reasons": ["policy divergence meaningful"],
                            }
                        },
                        "currencies": {},
                    },
                },
                {
                    "record_type": "snapshot",
                    "generated_at": _iso(now - timedelta(minutes=20)),
                    "snapshot": {
                        "generated_at": _iso(now - timedelta(minutes=20)),
                        "pairs": {
                            "EURUSD": {
                                "trade_gate": "BLOCK",
                                "rates_regime": "UNSTABLE",
                                "rates_risk_score": 0.82,
                                "policy_divergence_score": 0.74,
                                "meeting_path_reprice_now": True,
                                "reasons": ["meeting path repricing active"],
                            }
                        },
                        "currencies": {},
                    },
                },
            ]
            contracts.COMMON_RATES_HISTORY.write_text(
                "\n".join(json.dumps(record, sort_keys=True) for record in records) + "\n",
                encoding="utf-8",
            )

            saved_now = replay.utc_now
            replay.utc_now = lambda: now
            try:
                report = replay.build_rates_replay_report(symbol="EURUSD", hours_back=24)
            finally:
                replay.utc_now = saved_now

            assert report["symbols"][0]["symbol"] == "EURUSD"
            assert report["symbols"][0]["gate_counts"]["ALLOW"] == 1
            assert report["symbols"][0]["gate_counts"]["BLOCK"] == 1
            assert any(
                transition["type"] == "trade_gate" and transition["to"] == "BLOCK"
                for transition in report["symbols"][0]["recent_transitions"]
            )
            assert contracts.RATES_ENGINE_REPLAY_REPORT_PATH.exists()
