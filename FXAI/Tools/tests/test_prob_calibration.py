from __future__ import annotations

import json
import tempfile
from pathlib import Path

from offline_lab.fixtures import patched_paths
from offline_lab.prob_calibration_config import (
    default_config,
    default_memory,
    load_config,
    load_memory,
    validate_config_payload,
    validate_memory_payload,
)
from offline_lab.prob_calibration_contracts import (
    PROB_CALIBRATION_MEMORY_PATH,
)
from offline_lab.prob_calibration_math import (
    calibrate_probabilities,
    compute_uncertainty_score,
    decide_action,
    scale_move_distribution,
    select_calibration_tier,
)
from offline_lab.prob_calibration_replay import build_prob_calibration_replay_report
import offline_lab.prob_calibration_contracts as contracts


def test_prob_calibration_validate_creates_default_files():
    with tempfile.TemporaryDirectory(prefix="fxai_probcal_validate_") as tmp_dir:
        with patched_paths(Path(tmp_dir)):
            config = load_config()
            memory = load_memory()
            assert config["enabled"] is True
            assert Path(contracts.PROB_CALIBRATION_CONFIG_PATH).exists()
            assert Path(contracts.PROB_CALIBRATION_RUNTIME_CONFIG_PATH).exists()
            assert Path(contracts.PROB_CALIBRATION_MEMORY_PATH).exists()
            assert Path(contracts.PROB_CALIBRATION_RUNTIME_MEMORY_PATH).exists()
            assert memory["tiers"]


def test_prob_calibration_validate_rejects_invalid_support_floors():
    payload = default_config()
    payload["support_soft_floor"] = 4
    try:
        validate_config_payload(payload)
    except Exception as exc:  # noqa: BLE001
        assert "support_soft_floor" in str(exc)
    else:
        raise AssertionError("invalid support_soft_floor should fail validation")


def test_prob_calibration_validate_rejects_duplicate_memory_tiers():
    payload = default_memory()
    payload["tiers"].append(dict(payload["tiers"][0]))
    try:
        validate_memory_payload(payload)
    except Exception as exc:  # noqa: BLE001
        assert "Duplicate probabilistic calibration tier" in str(exc)
    else:
        raise AssertionError("duplicate tier definitions should fail validation")


def test_prob_calibration_selects_most_specific_supported_tier():
    records = [
        {
            "kind": "GLOBAL",
            "symbol": "*",
            "session": "*",
            "regime": "*",
            "support": 500,
            "calibration_quality": 0.58,
        },
        {
            "kind": "REGIME",
            "symbol": "*",
            "session": "*",
            "regime": "HIGH_VOL_EVENT",
            "support": 84,
            "calibration_quality": 0.52,
        },
        {
            "kind": "PAIR_REGIME",
            "symbol": "EURUSD",
            "session": "*",
            "regime": "HIGH_VOL_EVENT",
            "support": 92,
            "calibration_quality": 0.57,
        },
    ]
    selected = select_calibration_tier(
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


def test_prob_calibration_falls_back_when_support_is_soft_low():
    records = [
        {
            "kind": "PAIR_SESSION_REGIME",
            "symbol": "EURUSD",
            "session": "LONDON_NY_OVERLAP",
            "regime": "HIGH_VOL_EVENT",
            "support": 24,
            "calibration_quality": 0.49,
        },
        {
            "kind": "GLOBAL",
            "symbol": "*",
            "session": "*",
            "regime": "*",
            "support": 400,
            "calibration_quality": 0.58,
        },
    ]
    selected = select_calibration_tier(
        records,
        symbol="EURUSD",
        session="LONDON_NY_OVERLAP",
        regime="HIGH_VOL_EVENT",
        soft_support_floor=64,
        hard_support_floor=16,
    )
    assert selected["kind"] == "PAIR_SESSION_REGIME"
    assert selected["fallback_used"] is True
    assert selected["support_usable"] is True


def test_prob_calibration_probability_mapping_shrinks_and_preserves_sum():
    config = default_config()
    tier = default_memory()["tiers"][0]
    result = calibrate_probabilities(
        raw_buy_prob=0.58,
        raw_sell_prob=0.21,
        raw_skip_prob=0.21,
        tier=tier,
        config=config,
        uncertainty_score=0.40,
    )
    total = (
        result["calibrated_buy_prob"]
        + result["calibrated_sell_prob"]
        + result["calibrated_skip_prob"]
    )
    assert abs(total - 1.0) < 1e-6
    assert 0.0 <= result["calibrated_confidence"] <= 1.0
    assert result["calibrated_skip_prob"] >= 0.0


def test_prob_calibration_move_distribution_scales_conservatively_under_uncertainty():
    tier = default_memory()["tiers"][0]
    scaled = scale_move_distribution(
        move_mean_points=12.0,
        move_q25_points=5.0,
        move_q50_points=9.0,
        move_q75_points=16.0,
        tier=tier,
        uncertainty_score=0.70,
    )
    assert scaled["expected_move_q25_points"] <= scaled["expected_move_q50_points"] <= scaled["expected_move_q75_points"]
    assert scaled["expected_move_mean_points"] >= scaled["expected_move_q50_points"]
    assert scaled["expected_move_mean_points"] < 12.0


def test_prob_calibration_uncertainty_penalty_increases_for_weak_context():
    config = default_config()
    tier = default_memory()["tiers"][0]
    base = compute_uncertainty_score(
        config=config,
        tier=tier,
        min_move_points=3.0,
        expected_move_mean_points=8.0,
        expected_move_q25_points=3.5,
        expected_move_q75_points=12.0,
        agreement_score=0.82,
        news_risk_score=0.12,
        rates_risk_score=0.10,
        micro_risk_score=0.14,
        dynamic_abstain_bias=0.04,
        adaptive_abstain_bias=0.06,
        stale_context_count=0,
    )
    stressed = compute_uncertainty_score(
        config=config,
        tier={**tier, "support": 12, "calibration_quality": 0.30, "uncertainty_mult": 1.4},
        min_move_points=3.0,
        expected_move_mean_points=6.0,
        expected_move_q25_points=1.4,
        expected_move_q75_points=13.5,
        agreement_score=0.34,
        news_risk_score=0.82,
        rates_risk_score=0.74,
        micro_risk_score=0.88,
        dynamic_abstain_bias=0.46,
        adaptive_abstain_bias=0.41,
        stale_context_count=2,
    )
    assert stressed["uncertainty_score"] > base["uncertainty_score"]
    assert stressed["uncertainty_penalty_points"] > base["uncertainty_penalty_points"]


def test_prob_calibration_decide_action_prefers_skip_when_edge_is_not_enough():
    outcome = decide_action(
        upstream_action="BUY",
        calibrated_buy_prob=0.56,
        calibrated_sell_prob=0.19,
        expected_gross_edge_points=2.6,
        edge_after_costs_points=-0.2,
        edge_floor_points=0.25,
        uncertainty_score=0.96,
        uncertainty_limit=0.92,
        calibration_quality=0.41,
        min_calibration_quality=0.44,
        support=12,
        support_hard_floor=16,
        raw_score=0.03,
        signal_zero_band=0.035,
        expected_move_q25_points=0.8,
        cost_floor_points=1.4,
        context_flags={
            "calibration_stale": True,
            "input_stale": False,
            "news_risk_block": False,
            "microstructure_stress": True,
            "rates_risk_block": False,
        },
    )
    assert outcome["final_action"] == "SKIP"
    assert outcome["abstain"] is True
    assert "EDGE_TOO_SMALL" in outcome["reason_codes"]
    assert "MICROSTRUCTURE_STRESS" in outcome["reason_codes"]


def test_prob_calibration_replay_report_summarizes_history():
    with tempfile.TemporaryDirectory(prefix="fxai_probcal_replay_") as tmp_dir:
        with patched_paths(Path(tmp_dir)):
            history_rows = [
                {
                    "generated_at": "2026-04-10T10:00:00Z",
                    "symbol": "EURUSD",
                    "state": {
                        "selected_tier_kind": "REGIME",
                        "calibrated_confidence": 0.57,
                        "edge_after_costs_points": 0.42,
                        "uncertainty_score": 0.36,
                        "final_action": "BUY",
                        "abstain": False,
                        "fallback_used": False,
                        "reason_codes": [],
                    },
                },
                {
                    "generated_at": "2026-04-10T10:05:00Z",
                    "symbol": "EURUSD",
                    "state": {
                        "selected_tier_kind": "REGIME",
                        "calibrated_confidence": 0.51,
                        "edge_after_costs_points": -0.18,
                        "uncertainty_score": 0.94,
                        "final_action": "SKIP",
                        "abstain": True,
                        "fallback_used": True,
                        "reason_codes": ["EDGE_TOO_SMALL", "UNCERTAINTY_TOO_HIGH"],
                    },
                },
            ]
            contracts.prob_calibration_runtime_history_path("EURUSD").write_text(
                "\n".join(json.dumps(item, sort_keys=True) for item in history_rows) + "\n",
                encoding="utf-8",
            )
            payload = build_prob_calibration_replay_report(symbol="EURUSD", hours_back=48)
            assert payload["symbol_count"] == 1
            symbol_payload = payload["symbols"][0]
            assert symbol_payload["action_counts"]["BUY"] == 1
            assert symbol_payload["action_counts"]["SKIP"] == 1
            assert symbol_payload["fallback_count"] == 1
            assert symbol_payload["abstain_count"] == 1
            assert symbol_payload["recent_transitions"]


def test_prob_calibration_load_memory_rewrites_generated_unix_line():
    with tempfile.TemporaryDirectory(prefix="fxai_probcal_memory_") as tmp_dir:
        with patched_paths(Path(tmp_dir)):
            payload = default_memory()
            payload["generated_at"] = "2026-04-10T10:00:00Z"
            PROB_CALIBRATION_MEMORY_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            load_memory()
            runtime_text = contracts.PROB_CALIBRATION_RUNTIME_MEMORY_PATH.read_text(encoding="utf-8")
            assert "generated_at_unix" in runtime_text
