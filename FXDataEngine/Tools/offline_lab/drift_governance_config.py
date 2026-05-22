from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from .common import OfflineLabError
from .drift_governance_contracts import (
    DRIFT_GOVERNANCE_ACTIONS,
    DRIFT_GOVERNANCE_CONFIG_PATH,
    DRIFT_GOVERNANCE_CONFIG_VERSION,
    DRIFT_GOVERNANCE_MODES,
    ensure_drift_governance_dirs,
    json_dump,
)


def default_config() -> dict[str, Any]:
    return {
        "schema_version": DRIFT_GOVERNANCE_CONFIG_VERSION,
        "policy_version": 1,
        "enabled": True,
        "action_mode": "AUTO_APPLY_PROTECTIVE",
        "recent_window": {
            "max_observations": 24,
            "max_age_hours": 96,
        },
        "reference_window": {
            "max_observations": 120,
            "max_age_hours": 1440,
        },
        "support": {
            "min_recent_observations": 4,
            "min_reference_observations": 8,
            "min_challenger_runs": 3,
            "min_challenger_shadow_observations": 2,
        },
        "hysteresis": {
            "cooldown_hours": 24,
            "recovery_hold_hours": 48,
        },
        "aggregation_weights": {
            "feature_drift_score": 0.18,
            "regime_drift_score": 0.12,
            "calibration_drift_score": 0.18,
            "pair_decay_score": 0.16,
            "post_event_drift_score": 0.10,
            "execution_drift_score": 0.10,
            "performance_drift_score": 0.16,
        },
        "metric_keys": {
            "feature_keys": [
                "meta_weight",
                "reliability",
                "global_edge",
                "context_edge",
                "context_regret",
                "route_value",
                "route_regret",
                "policy_enter_prob",
                "policy_no_trade_prob",
                "portfolio_pressure",
                "control_plane_score",
                "portfolio_supervisor_score",
                "shadow_score",
            ],
            "performance_keys": [
                "shadow_score",
                "route_value",
                "portfolio_objective",
                "portfolio_stability",
                "reliability",
            ],
            "execution_keys": [
                "portfolio_pressure",
                "control_plane_score",
                "portfolio_supervisor_score",
                "policy_no_trade_prob",
            ],
        },
        "thresholds": {
            "feature_psi_warn": 0.15,
            "feature_psi_degrade": 0.30,
            "regime_mix_warn": 0.20,
            "regime_mix_degrade": 0.40,
            "calibration_warn_delta": 0.08,
            "calibration_degrade_delta": 0.18,
            "performance_warn_delta": 0.08,
            "performance_degrade_delta": 0.18,
            "execution_warn_delta": 0.08,
            "execution_degrade_delta": 0.18,
            "caution_score": 0.34,
            "degraded_score": 0.56,
            "shadow_only_score": 0.74,
            "disable_score": 0.88,
            "recover_score": 0.22,
            "recover_to_healthy_score": 0.14,
            "max_low_support_score": 0.58,
        },
        "context_modifiers": {
            "stressed_execution_quality_floor": 0.45,
            "elevated_uncertainty_floor": 0.60,
            "elevated_cross_asset_risk": 0.65,
            "event_or_posture_risk_gates": ["CAUTION", "BLOCK", "STRESSED", "HIGH_VOL_EVENT"],
            "event_score_multiplier": 1.08,
            "execution_score_multiplier": 1.10,
            "cross_asset_score_multiplier": 1.05,
            "calibration_score_multiplier": 1.05,
        },
        "actions": {
            "caution_weight_multiplier": 0.82,
            "degraded_weight_multiplier": 0.58,
            "shadow_only_weight_multiplier": 0.02,
            "disabled_weight_multiplier": 0.0,
            "restrict_live_states": ["SHADOW_ONLY", "DEMOTED", "DISABLED"],
            "suppress_router_states": ["SHADOW_ONLY", "DISABLED"],
            "auto_disable": True,
            "auto_demote": True,
            "auto_promote": False,
        },
        "challenger_policy": {
            "discover_from_best_configs": True,
            "max_candidates_per_symbol": 3,
            "min_walkforward_score": 72.0,
            "min_recent_score": 74.0,
            "min_adversarial_score": 68.0,
            "min_macro_event_score": 64.0,
            "max_calibration_error": 0.30,
            "max_issue_count": 1.50,
            "min_portfolio_score": 0.52,
            "min_shadow_score": 0.02,
            "min_reliability": 0.52,
            "promotion_margin": 0.50,
            "rollback_hours": 96,
        },
    }


def _merge_defaults(default_value: Any, payload_value: Any) -> Any:
    if isinstance(default_value, dict):
        payload_dict = payload_value if isinstance(payload_value, dict) else {}
        merged = {key: _merge_defaults(default_value[key], payload_dict.get(key)) for key in default_value}
        for key, value in payload_dict.items():
            if key not in merged:
                merged[key] = deepcopy(value)
        return merged
    if isinstance(default_value, list):
        return deepcopy(payload_value) if isinstance(payload_value, list) else deepcopy(default_value)
    if payload_value is None:
        return deepcopy(default_value)
    return deepcopy(payload_value)


def _validate_positive_int(payload: dict[str, Any], key: str) -> int:
    value = int(payload.get(key, 0) or 0)
    if value <= 0:
        raise OfflineLabError(f"Drift-governance {key} must be > 0")
    payload[key] = value
    return value


def _validate_probability_like(payload: dict[str, Any], key: str, *, lo: float = 0.0, hi: float = 1.0) -> float:
    value = float(payload.get(key, 0.0) or 0.0)
    if value < lo or value > hi:
        raise OfflineLabError(f"Drift-governance {key} must be in [{lo}, {hi}]")
    payload[key] = value
    return value


def validate_config_payload(payload: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise OfflineLabError("Drift-governance config must be a JSON object")
    schema_version = int(payload.get("schema_version", 0) or 0)
    if schema_version != DRIFT_GOVERNANCE_CONFIG_VERSION:
        raise OfflineLabError(
            f"Drift-governance schema_version must be {DRIFT_GOVERNANCE_CONFIG_VERSION}, received {schema_version}"
        )
    if not isinstance(payload.get("enabled", True), bool):
        raise OfflineLabError("Drift-governance enabled must be a boolean")
    payload["policy_version"] = max(int(payload.get("policy_version", 1) or 1), 1)

    action_mode = str(payload.get("action_mode", "AUTO_APPLY_PROTECTIVE") or "AUTO_APPLY_PROTECTIVE").strip().upper()
    if action_mode not in DRIFT_GOVERNANCE_MODES:
        raise OfflineLabError(
            f"Unsupported drift-governance action_mode {action_mode}; expected one of {', '.join(DRIFT_GOVERNANCE_MODES)}"
        )
    payload["action_mode"] = action_mode

    for section_name in ("recent_window", "reference_window", "support", "hysteresis", "aggregation_weights", "metric_keys", "thresholds", "context_modifiers", "actions", "challenger_policy"):
        if not isinstance(payload.get(section_name, {}), dict):
            raise OfflineLabError(f"Drift-governance {section_name} must be an object")

    _validate_positive_int(payload["recent_window"], "max_observations")
    _validate_positive_int(payload["recent_window"], "max_age_hours")
    _validate_positive_int(payload["reference_window"], "max_observations")
    _validate_positive_int(payload["reference_window"], "max_age_hours")
    _validate_positive_int(payload["support"], "min_recent_observations")
    _validate_positive_int(payload["support"], "min_reference_observations")
    _validate_positive_int(payload["support"], "min_challenger_runs")
    _validate_positive_int(payload["support"], "min_challenger_shadow_observations")
    _validate_positive_int(payload["hysteresis"], "cooldown_hours")
    _validate_positive_int(payload["hysteresis"], "recovery_hold_hours")

    weights = payload["aggregation_weights"]
    expected_weight_keys = {
        "feature_drift_score",
        "regime_drift_score",
        "calibration_drift_score",
        "pair_decay_score",
        "post_event_drift_score",
        "execution_drift_score",
        "performance_drift_score",
    }
    if set(weights.keys()) != expected_weight_keys:
        raise OfflineLabError("Drift-governance aggregation_weights must define the full drift-score key set")
    for key, value in weights.items():
        value_f = float(value or 0.0)
        if value_f < 0.0:
            raise OfflineLabError(f"Drift-governance aggregation weight {key} must be >= 0")
        weights[key] = value_f
    if sum(float(value) for value in weights.values()) <= 0.0:
        raise OfflineLabError("Drift-governance aggregation weights must sum to > 0")

    metric_keys = payload["metric_keys"]
    for key in ("feature_keys", "performance_keys", "execution_keys"):
        items = metric_keys.get(key, [])
        if not isinstance(items, list) or not items:
            raise OfflineLabError(f"Drift-governance {key} must be a non-empty array")
        normalized = []
        seen: set[str] = set()
        for item in items:
            token = str(item or "").strip()
            if not token or token in seen:
                continue
            seen.add(token)
            normalized.append(token)
        if not normalized:
            raise OfflineLabError(f"Drift-governance {key} must contain at least one non-empty metric key")
        metric_keys[key] = normalized

    thresholds = payload["thresholds"]
    for key in (
        "feature_psi_warn",
        "feature_psi_degrade",
        "regime_mix_warn",
        "regime_mix_degrade",
        "calibration_warn_delta",
        "calibration_degrade_delta",
        "performance_warn_delta",
        "performance_degrade_delta",
        "execution_warn_delta",
        "execution_degrade_delta",
        "caution_score",
        "degraded_score",
        "shadow_only_score",
        "disable_score",
        "recover_score",
        "recover_to_healthy_score",
        "max_low_support_score",
    ):
        thresholds[key] = float(thresholds.get(key, 0.0) or 0.0)
        if thresholds[key] < 0.0:
            raise OfflineLabError(f"Drift-governance threshold {key} must be >= 0")
    if not (thresholds["feature_psi_warn"] < thresholds["feature_psi_degrade"]):
        raise OfflineLabError("Drift-governance feature PSI warn threshold must be < degrade threshold")
    if not (thresholds["regime_mix_warn"] < thresholds["regime_mix_degrade"]):
        raise OfflineLabError("Drift-governance regime_mix_warn must be < regime_mix_degrade")
    if not (thresholds["caution_score"] < thresholds["degraded_score"] < thresholds["shadow_only_score"] < thresholds["disable_score"]):
        raise OfflineLabError("Drift-governance action thresholds must be strictly increasing")
    if thresholds["recover_to_healthy_score"] > thresholds["recover_score"]:
        raise OfflineLabError("Drift-governance recover_to_healthy_score must be <= recover_score")

    modifiers = payload["context_modifiers"]
    _validate_probability_like(modifiers, "stressed_execution_quality_floor")
    _validate_probability_like(modifiers, "elevated_uncertainty_floor")
    _validate_probability_like(modifiers, "elevated_cross_asset_risk")
    for key in (
        "event_score_multiplier",
        "execution_score_multiplier",
        "cross_asset_score_multiplier",
        "calibration_score_multiplier",
    ):
        value = float(modifiers.get(key, 0.0) or 0.0)
        if value <= 0.0:
            raise OfflineLabError(f"Drift-governance context modifier {key} must be > 0")
        modifiers[key] = value
    gates = modifiers.get("event_or_posture_risk_gates", [])
    if not isinstance(gates, list):
        raise OfflineLabError("Drift-governance event_or_posture_risk_gates must be an array")
    modifiers["event_or_posture_risk_gates"] = [str(item).strip().upper() for item in gates if str(item).strip()]

    actions = payload["actions"]
    for key in (
        "caution_weight_multiplier",
        "degraded_weight_multiplier",
        "shadow_only_weight_multiplier",
        "disabled_weight_multiplier",
    ):
        value = float(actions.get(key, 0.0) or 0.0)
        if value < 0.0:
            raise OfflineLabError(f"Drift-governance action weight {key} must be >= 0")
        actions[key] = value
    restrict_live_states = [str(item).strip().upper() for item in list(actions.get("restrict_live_states", [])) if str(item).strip()]
    suppress_router_states = [str(item).strip().upper() for item in list(actions.get("suppress_router_states", [])) if str(item).strip()]
    actions["restrict_live_states"] = restrict_live_states
    actions["suppress_router_states"] = suppress_router_states
    for key in ("auto_disable", "auto_demote", "auto_promote"):
        if not isinstance(actions.get(key, False), bool):
            raise OfflineLabError(f"Drift-governance actions.{key} must be a boolean")

    challenger = payload["challenger_policy"]
    if not isinstance(challenger.get("discover_from_best_configs", True), bool):
        raise OfflineLabError("Drift-governance challenger_policy.discover_from_best_configs must be a boolean")
    _validate_positive_int(challenger, "max_candidates_per_symbol")
    _validate_positive_int(challenger, "rollback_hours")
    for key in (
        "min_walkforward_score",
        "min_recent_score",
        "min_adversarial_score",
        "min_macro_event_score",
        "max_calibration_error",
        "max_issue_count",
        "min_portfolio_score",
        "min_shadow_score",
        "min_reliability",
        "promotion_margin",
    ):
        challenger[key] = float(challenger.get(key, 0.0) or 0.0)
    if challenger["max_calibration_error"] <= 0.0:
        raise OfflineLabError("Drift-governance max_calibration_error must be > 0")
    if challenger["promotion_margin"] < 0.0:
        raise OfflineLabError("Drift-governance promotion_margin must be >= 0")
    return payload


def load_config(path: Path | None = None) -> dict[str, Any]:
    ensure_drift_governance_dirs()
    path = path or DRIFT_GOVERNANCE_CONFIG_PATH
    defaults = default_config()
    if not path.exists():
        json_dump(path, defaults)
        return deepcopy(defaults)
    payload = json.loads(path.read_text(encoding="utf-8"))
    merged = _merge_defaults(defaults, payload)
    validated = validate_config_payload(merged)
    if json.dumps(validated, sort_keys=True) != json.dumps(payload, sort_keys=True):
        json_dump(path, validated)
    return validated
