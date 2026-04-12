from __future__ import annotations

import json
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

from offline_lab.fixtures import patched_paths
from offline_lab.microstructure_config import default_config, validate_config_payload
from offline_lab.microstructure_reference import (
    burst_scores,
    classify_microstructure_state,
    detect_sweep_and_reject,
    directional_efficiency_from_changes,
    resolve_session,
    spread_metrics,
    tick_imbalance_from_changes,
)
from offline_lab.microstructure_replay import build_microstructure_replay_report
from offline_lab.microstructure_service import (
    microstructure_health_snapshot,
    sync_local_status_from_runtime,
    validate_microstructure_config,
)
import offline_lab.microstructure_contracts as contracts


def _iso_hours_ago(hours_ago: float) -> str:
    return (datetime.now(timezone.utc) - timedelta(hours=hours_ago)).strftime("%Y-%m-%dT%H:%M:%SZ")


def test_microstructure_validate_creates_default_files_and_required_windows():
    with tempfile.TemporaryDirectory(prefix="fxai_micro_validate_") as tmp_dir:
        with patched_paths(Path(tmp_dir)):
            payload = validate_microstructure_config()
            assert payload["ok"] is True
            assert Path(payload["config_path"]).exists()
            assert Path(payload["service_config_path"]).exists()
            assert payload["windows_sec"] == [10, 30, 60, 300, 900]


def test_microstructure_validate_rejects_missing_required_windows():
    payload = default_config()
    payload["windows_sec"] = [10, 60, 300]
    try:
        validate_config_payload(payload)
    except Exception as exc:  # noqa: BLE001
        assert "required phase-1 windows" in str(exc)
    else:
        raise AssertionError("missing required windows should fail validation")


def test_microstructure_reference_identifies_clean_trend_state():
    changes = [1.2, 0.9, 1.1, 1.0, -0.2, 0.8, 1.1]
    imbalance = tick_imbalance_from_changes(changes)
    efficiency = directional_efficiency_from_changes(changes)
    state = classify_microstructure_state(
        spread_instability=0.18,
        spread_zscore_60s=0.44,
        wide_spread_fraction_60s=0.02,
        session_spread_behavior_score=0.12,
        vol_burst_score_5m=1.08,
        intensity_burst_score_30s=1.14,
        silent_gap_seconds_current=0.4,
        handoff_flag=False,
        local_extrema_breach_score_60s=0.16,
        breakout_reversal_score_60s=0.12,
        exhaustion_proxy_30s=0.18,
        directional_efficiency_60s=efficiency,
        tick_imbalance_30s=imbalance,
        session_open_burst_score=0.18,
        thresholds=default_config()["thresholds"],
    )
    assert imbalance > 0.5
    assert efficiency > 0.7
    assert state["microstructure_regime"] == "TRENDING_CLEAN"
    assert state["trade_gate"] == "ALLOW"


def test_microstructure_reference_detects_thin_and_wide_regime():
    spreads = [1.1, 1.2, 1.1, 1.4, 3.6, 4.1, 4.5]
    spread_state = spread_metrics(
        spreads,
        wide_spread_zscore=default_config()["thresholds"]["wide_spread_zscore"],
        wide_spread_absolute_points_floor=default_config()["thresholds"]["wide_spread_absolute_points_floor"],
    )
    state = classify_microstructure_state(
        spread_instability=spread_state["spread_instability"],
        spread_zscore_60s=spread_state["spread_zscore"],
        wide_spread_fraction_60s=max(spread_state["wide_spread_fraction"], 0.14),
        session_spread_behavior_score=0.62,
        vol_burst_score_5m=1.55,
        intensity_burst_score_30s=1.72,
        silent_gap_seconds_current=8.0,
        handoff_flag=True,
        local_extrema_breach_score_60s=0.21,
        breakout_reversal_score_60s=0.18,
        exhaustion_proxy_30s=0.24,
        directional_efficiency_60s=0.28,
        tick_imbalance_30s=0.04,
        session_open_burst_score=0.66,
        thresholds=default_config()["thresholds"],
    )
    assert spread_state["spread_zscore"] > 1.0
    assert state["microstructure_regime"] == "THIN_AND_WIDE"
    assert state["trade_gate"] in {"CAUTION", "BLOCK"}
    assert "current spread regime is abnormally wide" in state["reasons"]


def test_microstructure_reference_detects_sweep_and_reject_pattern():
    sweep = detect_sweep_and_reject(
        [100.0, 100.2, 100.4, 100.5, 101.4, 101.0, 100.7],
        point_value=0.1,
        spread_current=1.2,
        shock_move_points_factor=1.8,
        stop_run_reversal_fraction=0.45,
        directional_efficiency=0.34,
        shock_move_count=2,
    )
    assert sweep["local_extrema_breach_score"] > 0.5
    assert sweep["sweep_and_reject_flag"] is True
    assert sweep["breakout_reversal_score"] > 0.5


def test_microstructure_reference_resolves_session_overlap_and_handoff():
    session = resolve_session(datetime(2026, 4, 9, 12, 5, tzinfo=timezone.utc), default_config()["session_model"])
    assert session["session_tag"] == "LONDON_NEWYORK_OVERLAP"
    assert session["handoff_flag"] is True
    assert session["minutes_since_session_open"] == 5


def test_microstructure_burst_scores_detect_activity_jump():
    burst = burst_scores(190.0, [90.0, 95.0, 98.0, 100.0])
    assert burst["ratio"] > 1.8
    assert burst["zscore"] > 2.0


def test_microstructure_health_snapshot_handles_missing_runtime_artifacts():
    with tempfile.TemporaryDirectory(prefix="fxai_micro_health_") as tmp_dir:
        with patched_paths(Path(tmp_dir)):
            payload = microstructure_health_snapshot()
            assert payload["service"]["ok"] is False
            assert payload["service"]["stale"] is True
            assert payload["symbols"] == {}


def test_microstructure_sync_local_status_persists_runtime_status_mtime():
    with tempfile.TemporaryDirectory(prefix="fxai_micro_status_sync_") as tmp_dir:
        with patched_paths(Path(tmp_dir)):
            contracts.COMMON_MICROSTRUCTURE_STATUS.write_text(
                json.dumps(
                    {
                        "generated_at": _iso_hours_ago(1.0),
                        "service": {"ok": True, "stale": False},
                        "health": {"snapshot_stale_after_sec": 45},
                        "symbols": {},
                    },
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
            payload = sync_local_status_from_runtime()
            saved = json.loads(contracts.MICROSTRUCTURE_STATUS_PATH.read_text(encoding="utf-8"))
            assert "runtime_status_mtime" in payload["health"]
            assert saved["health"]["runtime_status_mtime"] == payload["health"]["runtime_status_mtime"]


def test_microstructure_replay_report_summarizes_history():
    with tempfile.TemporaryDirectory(prefix="fxai_micro_replay_") as tmp_dir:
        with patched_paths(Path(tmp_dir)):
            history_records = [
                {
                    "generated_at": _iso_hours_ago(2.0),
                    "symbol": "EURUSD",
                    "state": {
                        "microstructure_regime": "NORMAL",
                        "trade_gate": "ALLOW",
                        "hostile_execution_score": 0.18,
                        "liquidity_stress_score": 0.22,
                        "reasons": ["baseline stable"],
                    },
                },
                {
                    "generated_at": _iso_hours_ago(1.95),
                    "symbol": "EURUSD",
                    "state": {
                        "microstructure_regime": "STOP_RUN_RISK",
                        "trade_gate": "CAUTION",
                        "hostile_execution_score": 0.74,
                        "liquidity_stress_score": 0.69,
                        "reasons": ["recent breakout rejection detected"],
                    },
                },
            ]
            contracts.COMMON_MICROSTRUCTURE_HISTORY.write_text(
                "\n".join(json.dumps(item, sort_keys=True) for item in history_records) + "\n",
                encoding="utf-8",
            )
            payload = build_microstructure_replay_report(symbol="EURUSD", hours_back=48)
            assert payload["symbol_count"] == 1
            assert payload["symbols"][0]["regime_counts"]["STOP_RUN_RISK"] == 1
            assert payload["symbols"][0]["gate_counts"]["CAUTION"] == 1
            assert payload["symbols"][0]["recent_transitions"]
