from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
import tempfile
from pathlib import Path

from offline_lab.execution_quality_config import (
    default_config,
    default_memory,
    load_config,
    load_memory,
    validate_config_payload,
    validate_memory_payload,
)
from offline_lab.execution_quality_math import (
    classify_session_thinness,
    compute_execution_quality_forecast,
    select_execution_quality_tier,
)
from offline_lab.execution_quality_replay import build_execution_quality_replay_report
from offline_lab.fixtures import patched_paths
import offline_lab.execution_quality_contracts as contracts


def test_execution_quality_validate_creates_default_files():
    with tempfile.TemporaryDirectory(prefix="fxai_execquality_validate_") as tmp_dir:
        with patched_paths(Path(tmp_dir)):
            config = load_config()
            memory = load_memory()
            assert config["enabled"] is True
            assert Path(contracts.EXECUTION_QUALITY_CONFIG_PATH).exists()
            assert Path(contracts.EXECUTION_QUALITY_RUNTIME_CONFIG_PATH).exists()
            assert Path(contracts.EXECUTION_QUALITY_MEMORY_PATH).exists()
            assert Path(contracts.EXECUTION_QUALITY_RUNTIME_MEMORY_PATH).exists()
            assert memory["tiers"]


def test_execution_quality_validate_rejects_bad_threshold_order():
    payload = default_config()
    payload["state_thresholds"]["normal_min"] = 0.40
    payload["state_thresholds"]["caution_min"] = 0.50
    try:
        validate_config_payload(payload)
    except Exception as exc:  # noqa: BLE001
        assert "state_thresholds" in str(exc)
    else:
        raise AssertionError("invalid execution-quality thresholds should fail validation")


def test_execution_quality_validate_rejects_duplicate_memory_tiers():
    payload = default_memory()
    payload["tiers"].append(dict(payload["tiers"][0]))
    try:
        validate_memory_payload(payload)
    except Exception as exc:  # noqa: BLE001
        assert "Duplicate execution-quality tier" in str(exc)
    else:
        raise AssertionError("duplicate execution-quality tiers should fail validation")


def test_execution_quality_selects_specific_supported_tier():
    records = [
        {"kind": "GLOBAL", "symbol": "*", "session": "*", "regime": "*", "support": 400, "quality": 0.60},
        {"kind": "REGIME", "symbol": "*", "session": "*", "regime": "HIGH_VOL_EVENT", "support": 96, "quality": 0.48},
        {"kind": "PAIR_REGIME", "symbol": "EURUSD", "session": "*", "regime": "HIGH_VOL_EVENT", "support": 120, "quality": 0.54},
    ]
    selected = select_execution_quality_tier(
        records,
        symbol="EURUSD",
        session="LONDON_NY_OVERLAP",
        regime="HIGH_VOL_EVENT",
        soft_support_floor=64,
        hard_support_floor=16,
    )
    assert selected["kind"] == "PAIR_REGIME"
    assert selected["fallback_used"] is False
    assert selected["support_usable"] is True


def test_execution_quality_session_thinness_detects_asia_and_handoff():
    assert classify_session_thinness("ASIA", False) > classify_session_thinness("LONDON_NY_OVERLAP", False)
    assert classify_session_thinness("LONDON", True) >= 0.55


def test_execution_quality_forecast_enters_blocked_state_under_stress():
    config = default_config()
    tier = default_memory()["tiers"][0]
    result = compute_execution_quality_forecast(
        config=config,
        tier={**tier, "fallback_used": False},
        symbol="EURUSD",
        session_label="ASIA",
        regime_label="HIGH_VOL_EVENT",
        current_spread_points=4.2,
        broker_slippage_points=3.1,
        broker_latency_points=2.8,
        broker_reject_prob=0.54,
        broker_partial_fill_prob=0.48,
        broker_fill_ratio_mean=0.52,
        broker_event_burst_penalty=0.76,
        micro_spread_zscore=3.2,
        micro_hostile_execution=0.88,
        micro_liquidity_stress=0.84,
        micro_vol_burst=2.4,
        micro_tick_rate_zscore=2.2,
        micro_tick_imbalance=0.61,
        news_risk_score=0.82,
        rates_risk_score=0.72,
        stale_context_count=1,
        news_window_active=True,
        rates_repricing_active=True,
        handoff_flag=True,
        support_usable=True,
        data_stale=False,
        base_allowed_deviation_points=6.0,
    )
    assert result["execution_state"] == "BLOCKED"
    assert result["spread_expected_points"] >= result["spread_now_points"]
    assert result["slippage_risk"] > 0.6
    assert "NEWS_WINDOW_ACTIVE" in result["reason_codes"]


def test_execution_quality_forecast_stays_normal_in_supported_calm_state():
    config = default_config()
    tier = {**default_memory()["tiers"][0], "fallback_used": False}
    result = compute_execution_quality_forecast(
        config=config,
        tier=tier,
        symbol="EURUSD",
        session_label="LONDON_NY_OVERLAP",
        regime_label="TREND_PERSISTENT",
        current_spread_points=0.8,
        broker_slippage_points=0.2,
        broker_latency_points=0.4,
        broker_reject_prob=0.02,
        broker_partial_fill_prob=0.03,
        broker_fill_ratio_mean=0.98,
        broker_event_burst_penalty=0.02,
        micro_spread_zscore=0.30,
        micro_hostile_execution=0.16,
        micro_liquidity_stress=0.18,
        micro_vol_burst=0.50,
        micro_tick_rate_zscore=0.40,
        micro_tick_imbalance=0.22,
        news_risk_score=0.08,
        rates_risk_score=0.10,
        stale_context_count=0,
        news_window_active=False,
        rates_repricing_active=False,
        handoff_flag=False,
        support_usable=True,
        data_stale=False,
        base_allowed_deviation_points=4.0,
    )
    assert result["execution_state"] == "NORMAL"
    assert result["execution_quality_score"] > 0.70
    assert result["fill_quality_score"] > 0.55


def test_execution_quality_replay_report_summarizes_history():
    with tempfile.TemporaryDirectory(prefix="fxai_execquality_replay_") as tmp_dir:
        with patched_paths(Path(tmp_dir)):
            now = datetime.now(timezone.utc).replace(microsecond=0)
            earlier = now - timedelta(hours=2)
            later = now - timedelta(hours=1)
            history_path = contracts.execution_quality_runtime_history_path("EURUSD")
            history_path.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "generated_at": earlier.isoformat().replace("+00:00", "Z"),
                                "symbol": "EURUSD",
                                "state": {
                                    "symbol": "EURUSD",
                                    "execution_state": "CAUTION",
                                    "selected_tier_kind": "GLOBAL",
                                    "spread_widening_risk": 0.42,
                                    "slippage_risk": 0.38,
                                    "execution_quality_score": 0.58,
                                    "reason_codes": ["SPREAD_ALREADY_ELEVATED"],
                                },
                            }
                        ),
                        json.dumps(
                            {
                                "generated_at": later.isoformat().replace("+00:00", "Z"),
                                "symbol": "EURUSD",
                                "state": {
                                    "symbol": "EURUSD",
                                    "execution_state": "STRESSED",
                                    "selected_tier_kind": "REGIME",
                                    "spread_widening_risk": 0.76,
                                    "slippage_risk": 0.68,
                                    "execution_quality_score": 0.31,
                                    "reason_codes": ["VOLATILITY_BURST", "SLIPPAGE_RISK_ELEVATED"],
                                },
                            }
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            payload = build_execution_quality_replay_report(symbol="EURUSD", hours_back=72)
            assert payload["symbol_count"] == 1
            latest = payload["symbols"][0]["latest"]["state"]
            assert latest["execution_state"] == "STRESSED"
            assert payload["symbols"][0]["state_counts"]["CAUTION"] == 1
            assert payload["symbols"][0]["state_counts"]["STRESSED"] == 1
