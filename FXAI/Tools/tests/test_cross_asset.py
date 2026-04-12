from __future__ import annotations

import json
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import offline_lab.cross_asset_contracts as contracts
import offline_lab.cross_asset_engine as engine
import offline_lab.cross_asset_replay as replay
from offline_lab.cross_asset_config import export_runtime_probe_config, load_config, resolve_probe_symbols, validate_config_payload
from offline_lab.fixtures import patched_paths


def _iso(dt: datetime) -> str:
    return dt.replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _write_probe_snapshot(now: datetime, *, generated_at: datetime | None = None) -> None:
    stamped = generated_at or now
    payload = {
        "schema_version": 1,
        "generated_at": _iso(stamped),
        "symbols": {
            "US500": {
                "available": True,
                "updated_at": _iso(stamped),
                "last_price": 5234.0,
                "change_pct_1h": -0.42,
                "change_pct_4h": -0.88,
                "change_pct_1d": -1.26,
                "change_pct_5d": -1.90,
                "range_ratio_1d": 1.44,
            },
            "XBRUSD": {
                "available": True,
                "updated_at": _iso(stamped),
                "last_price": 88.1,
                "change_pct_1h": 0.32,
                "change_pct_4h": 0.66,
                "change_pct_1d": 1.08,
                "change_pct_5d": 2.48,
                "range_ratio_1d": 1.22,
            },
            "XAUUSD": {
                "available": True,
                "updated_at": _iso(stamped),
                "last_price": 2321.4,
                "change_pct_1h": 0.18,
                "change_pct_4h": 0.34,
                "change_pct_1d": 0.66,
                "change_pct_5d": 1.74,
                "range_ratio_1d": 1.14,
            },
            "BTCUSD": {
                "available": True,
                "updated_at": _iso(stamped),
                "last_price": 81_400.0,
                "change_pct_1h": 0.84,
                "change_pct_4h": 1.26,
                "change_pct_1d": 2.42,
                "change_pct_5d": 5.11,
                "range_ratio_1d": 1.58,
            },
            "US10Y": {
                "available": True,
                "updated_at": _iso(stamped),
                "last_price": 4.38,
                "change_pct_1h": 0.06,
                "change_pct_4h": 0.18,
                "change_pct_1d": 0.73,
                "change_pct_5d": 1.11,
                "range_ratio_1d": 1.10,
            },
        },
    }
    status = {
        "generated_at": _iso(stamped),
        "service": {
            "ok": True,
            "stale": False,
        },
    }
    contracts.COMMON_CROSS_ASSET_PROBE_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    contracts.COMMON_CROSS_ASSET_PROBE_STATUS.write_text(json.dumps(status, indent=2, sort_keys=True), encoding="utf-8")


def _write_rates_snapshot(now: datetime, *, generated_at: datetime | None = None) -> None:
    stamped = generated_at or now
    payload = {
        "generated_at": _iso(stamped),
        "currencies": {
            "USD": {
                "curve_slope_2s10s": -0.22,
                "policy_repricing_score": 0.78,
                "policy_uncertainty_score": 0.46,
                "stale": False,
            },
            "EUR": {
                "curve_slope_2s10s": -0.36,
                "policy_repricing_score": 0.34,
                "policy_uncertainty_score": 0.22,
                "stale": False,
            },
            "JPY": {
                "curve_slope_2s10s": 0.08,
                "policy_repricing_score": 0.28,
                "policy_uncertainty_score": 0.24,
                "stale": False,
            },
        },
        "pairs": {
            "EURUSD": {
                "front_end_diff": -0.62,
                "expected_path_diff": -0.91,
                "stale": False,
            },
            "USDJPY": {
                "front_end_diff": 0.54,
                "expected_path_diff": 0.66,
                "stale": False,
            },
        },
    }
    from offline_lab.rates_engine_contracts import COMMON_RATES_JSON

    COMMON_RATES_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def test_cross_asset_validate_creates_default_files():
    with tempfile.TemporaryDirectory(prefix="fxai_cross_asset_validate_") as tmp_dir:
        with patched_paths(Path(tmp_dir)):
            payload = engine.validate_cross_asset_config()
            config = load_config()
            assert payload["ok"] is True
            assert Path(payload["config_path"]).exists()
            assert Path(payload["probe_config_path"]).exists()
            assert "EURUSD" in payload["market_pairs"]
            assert config["enabled"] is True


def test_cross_asset_validate_rejects_invalid_threshold_order():
    payload = load_config()
    payload["label_thresholds"]["pair_caution_min"] = 0.90
    payload["label_thresholds"]["pair_block_min"] = 0.80
    try:
        validate_config_payload(payload)
    except Exception as exc:  # noqa: BLE001
        assert "pair thresholds" in str(exc)
    else:
        raise AssertionError("invalid cross-asset pair thresholds should fail validation")


def test_cross_asset_probe_symbols_preserve_broker_symbol_case():
    payload = load_config()
    payload["market_universe"]["indicator_symbols"] = ["MidDE50", "TecDE30", "XAUUSD"]
    payload["proxy_candidates"]["equities"] = ["MidDE50", "US500"]
    payload["proxy_candidates"]["gold"] = ["XAUUSD"]

    resolved = resolve_probe_symbols(payload)

    assert "MidDE50" in resolved
    assert "TecDE30" in resolved
    assert "MIDDE50" not in resolved
    assert "TECDE30" not in resolved

    runtime_path = export_runtime_probe_config(payload)
    runtime_lines = runtime_path.read_text(encoding="utf-8").splitlines()

    assert "symbol\tMidDE50" in runtime_lines
    assert "symbol\tTecDE30" in runtime_lines
    assert "symbol\tMIDDE50" not in runtime_lines
    assert "symbol\tTECDE30" not in runtime_lines


def test_cross_asset_cycle_builds_snapshot_and_runtime_exports():
    with tempfile.TemporaryDirectory(prefix="fxai_cross_asset_cycle_") as tmp_dir:
        with patched_paths(Path(tmp_dir)):
            now = datetime(2026, 4, 11, 10, 0, tzinfo=timezone.utc)
            _write_probe_snapshot(now)
            _write_rates_snapshot(now)

            saved_engine_now = engine.utc_now
            saved_replay_now = replay.utc_now
            engine.utc_now = lambda: now
            replay.utc_now = lambda: now
            try:
                payload = engine.run_cross_asset_cycle()
            finally:
                engine.utc_now = saved_engine_now
                replay.utc_now = saved_replay_now

            assert payload["pair_count"] > 0
            assert payload["data_stale"] is False
            assert Path(payload["snapshot_path"]).exists()
            assert Path(payload["flat_path"]).exists()
            assert Path(payload["history_path"]).exists()
            assert contracts.COMMON_CROSS_ASSET_SYMBOL_MAP.exists()
            assert contracts.CROSS_ASSET_REPLAY_REPORT_PATH.exists()

            snapshot = json.loads(contracts.COMMON_CROSS_ASSET_JSON.read_text(encoding="utf-8"))
            assert snapshot["schema_version"] == contracts.CROSS_ASSET_SCHEMA_VERSION
            assert snapshot["state_labels"]["macro_state"] in {
                "RATES_REPRICING",
                "VOLATILITY_SHOCK",
                "USD_LIQUIDITY_STRESS",
                "CROSS_ASSET_DISLOCATION",
                "COMMODITY_SHOCK",
                "MIXED",
            }
            assert snapshot["source_status"]["rates"]["stale"] is False
            assert snapshot["pair_states"]["EURUSD"]["trade_gate"] in {"ALLOW", "CAUTION", "BLOCK"}
            assert "pair\tEURUSD\ttrade_gate\t" in contracts.COMMON_CROSS_ASSET_FLAT.read_text(encoding="utf-8")
            assert "symbol\tEURUSD\tEURUSD" in contracts.COMMON_CROSS_ASSET_SYMBOL_MAP.read_text(encoding="utf-8")


def test_cross_asset_rates_only_fallback_keeps_runtime_live_when_probe_is_missing():
    with tempfile.TemporaryDirectory(prefix="fxai_cross_asset_rates_only_") as tmp_dir:
        with patched_paths(Path(tmp_dir)):
            import offline_lab.cross_asset_config as config_module

            now = datetime(2026, 4, 11, 10, 0, tzinfo=timezone.utc)
            _write_rates_snapshot(now)

            config = load_config()
            config["probe_required_for_live_gates"] = False
            config_module.CROSS_ASSET_CONFIG_PATH.write_text(json.dumps(config, indent=2, sort_keys=True), encoding="utf-8")

            saved_engine_now = engine.utc_now
            saved_replay_now = replay.utc_now
            engine.utc_now = lambda: now
            replay.utc_now = lambda: now
            try:
                engine.run_cross_asset_cycle()
            finally:
                engine.utc_now = saved_engine_now
                replay.utc_now = saved_replay_now

            snapshot = json.loads(contracts.COMMON_CROSS_ASSET_JSON.read_text(encoding="utf-8"))
            assert snapshot["quality_flags"]["partial_data"] is True
            assert snapshot["quality_flags"]["data_stale"] is False
            assert snapshot["quality_flags"]["rates_only_fallback"] is True
            assert snapshot["source_status"]["context_service"]["stale"] is True
            assert snapshot["pair_states"]["EURUSD"]["stale"] is False
            assert "cross-asset state stale or incomplete" not in snapshot["pair_states"]["EURUSD"]["reasons"]


def test_cross_asset_can_require_probe_for_live_gates():
    with tempfile.TemporaryDirectory(prefix="fxai_cross_asset_probe_required_") as tmp_dir:
        with patched_paths(Path(tmp_dir)):
            import offline_lab.cross_asset_config as config_module

            now = datetime(2026, 4, 11, 10, 0, tzinfo=timezone.utc)
            _write_rates_snapshot(now)

            config = load_config()
            config["probe_required_for_live_gates"] = True
            config_module.CROSS_ASSET_CONFIG_PATH.write_text(json.dumps(config, indent=2, sort_keys=True), encoding="utf-8")

            saved_engine_now = engine.utc_now
            saved_replay_now = replay.utc_now
            engine.utc_now = lambda: now
            replay.utc_now = lambda: now
            try:
                engine.run_cross_asset_cycle()
            finally:
                engine.utc_now = saved_engine_now
                replay.utc_now = saved_replay_now

            snapshot = json.loads(contracts.COMMON_CROSS_ASSET_JSON.read_text(encoding="utf-8"))
            assert snapshot["quality_flags"]["data_stale"] is True
            assert snapshot["pair_states"]["EURUSD"]["stale"] is True
            assert snapshot["pair_states"]["EURUSD"]["trade_gate"] == "BLOCK"


def test_cross_asset_blocks_pairs_when_critical_sources_are_stale():
    with tempfile.TemporaryDirectory(prefix="fxai_cross_asset_stale_") as tmp_dir:
        with patched_paths(Path(tmp_dir)):
            now = datetime(2026, 4, 11, 11, 0, tzinfo=timezone.utc)
            stale_time = now - timedelta(hours=2)
            _write_probe_snapshot(now, generated_at=stale_time)
            _write_rates_snapshot(now, generated_at=stale_time)

            saved_engine_now = engine.utc_now
            saved_replay_now = replay.utc_now
            engine.utc_now = lambda: now
            replay.utc_now = lambda: now
            try:
                engine.run_cross_asset_cycle()
            finally:
                engine.utc_now = saved_engine_now
                replay.utc_now = saved_replay_now

            snapshot = json.loads(contracts.COMMON_CROSS_ASSET_JSON.read_text(encoding="utf-8"))
            assert snapshot["quality_flags"]["data_stale"] is True
            assert snapshot["source_status"]["rates"]["stale"] is True
            assert snapshot["source_status"]["context_service"]["stale"] is True
            assert snapshot["pair_states"]["EURUSD"]["trade_gate"] == "BLOCK"
            assert "cross-asset state stale or incomplete" in snapshot["pair_states"]["EURUSD"]["reasons"]


def test_cross_asset_replay_report_summarizes_pair_history():
    with tempfile.TemporaryDirectory(prefix="fxai_cross_asset_replay_") as tmp_dir:
        with patched_paths(Path(tmp_dir)):
            contracts.COMMON_CROSS_ASSET_HISTORY.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "record_type": "snapshot",
                                "generated_at": "2026-04-11T09:00:00Z",
                                "snapshot": {
                                    "generated_at": "2026-04-11T09:00:00Z",
                                    "state_labels": {
                                        "macro_state": "NORMAL",
                                        "risk_state": "NORMAL",
                                        "liquidity_state": "NORMAL",
                                    },
                                    "reason_codes": ["BASELINE"],
                                    "pair_states": {
                                        "EURUSD": {
                                            "trade_gate": "ALLOW",
                                            "macro_state": "NORMAL",
                                            "risk_state": "NORMAL",
                                            "liquidity_state": "NORMAL",
                                            "pair_cross_asset_risk_score": 0.24,
                                            "reasons": ["BASELINE"],
                                        }
                                    },
                                },
                            }
                        ),
                        json.dumps(
                            {
                                "record_type": "snapshot",
                                "generated_at": "2026-04-11T10:00:00Z",
                                "snapshot": {
                                    "generated_at": "2026-04-11T10:00:00Z",
                                    "state_labels": {
                                        "macro_state": "RATES_REPRICING",
                                        "risk_state": "RISK_OFF",
                                        "liquidity_state": "STRESSED",
                                    },
                                    "reason_codes": ["FRONT_END_RATES_DIVERGING"],
                                    "pair_states": {
                                        "EURUSD": {
                                            "trade_gate": "BLOCK",
                                            "macro_state": "RATES_REPRICING",
                                            "risk_state": "RISK_OFF",
                                            "liquidity_state": "STRESSED",
                                            "pair_cross_asset_risk_score": 0.82,
                                            "reasons": ["FRONT_END_RATES_DIVERGING"],
                                        }
                                    },
                                },
                            }
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            payload = replay.build_cross_asset_replay_report(symbol="EURUSD", hours_back=72)
            assert payload["symbol_count"] == 1
            assert payload["symbols"][0]["gate_counts"]["ALLOW"] == 1
            assert payload["symbols"][0]["gate_counts"]["BLOCK"] == 1
            assert payload["symbols"][0]["recent_transitions"]
