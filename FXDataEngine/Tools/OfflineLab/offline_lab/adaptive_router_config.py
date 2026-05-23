from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from .common import OfflineLabError
from .adaptive_router_contracts import (
    ADAPTIVE_ROUTER_CONFIG_PATH,
    ADAPTIVE_ROUTER_CONFIG_VERSION,
    ADAPTIVE_ROUTER_REGIMES,
    ADAPTIVE_ROUTER_SESSIONS,
    ensure_adaptive_router_dirs,
)

_FAMILIES = [
    "linear",
    "tree",
    "recurrent",
    "convolutional",
    "transformer",
    "state_space",
    "distribution",
    "mixture",
    "memory",
    "world",
    "rule",
    "other",
]


def _regime_dict(value: float = 1.0) -> dict[str, float]:
    return {label: float(value) for label in ADAPTIVE_ROUTER_REGIMES}


def _session_dict(value: float = 1.0) -> dict[str, float]:
    return {label: float(value) for label in ADAPTIVE_ROUTER_SESSIONS}


def default_config() -> dict[str, Any]:
    family_defaults: dict[str, dict[str, Any]] = {}
    for family in _FAMILIES:
        family_defaults[family] = {
            "regime_weights": _regime_dict(1.0),
            "session_weights": _session_dict(1.0),
            "news_compatibility": 1.0,
            "liquidity_robustness": 1.0,
        }

    family_defaults["linear"]["regime_weights"].update({
        "RANGE_MEAN_REVERTING": 1.12,
        "HIGH_VOL_EVENT": 0.82,
        "LIQUIDITY_STRESS": 0.88,
        "SESSION_FLOW": 1.06,
    })
    family_defaults["tree"]["regime_weights"].update({
        "TREND_PERSISTENT": 1.04,
        "RANGE_MEAN_REVERTING": 1.06,
        "HIGH_VOL_EVENT": 0.88,
        "LIQUIDITY_STRESS": 0.90,
    })
    family_defaults["recurrent"]["regime_weights"].update({
        "TREND_PERSISTENT": 1.14,
        "BREAKOUT_TRANSITION": 1.10,
        "HIGH_VOL_EVENT": 1.02,
        "LIQUIDITY_STRESS": 0.90,
    })
    family_defaults["convolutional"]["regime_weights"].update({
        "BREAKOUT_TRANSITION": 1.08,
        "SESSION_FLOW": 1.06,
        "HIGH_VOL_EVENT": 0.92,
    })
    family_defaults["transformer"]["regime_weights"].update({
        "TREND_PERSISTENT": 1.12,
        "BREAKOUT_TRANSITION": 1.12,
        "HIGH_VOL_EVENT": 1.08,
        "LIQUIDITY_STRESS": 0.86,
    })
    family_defaults["state_space"]["regime_weights"].update({
        "TREND_PERSISTENT": 1.16,
        "BREAKOUT_TRANSITION": 1.08,
        "HIGH_VOL_EVENT": 1.00,
        "LIQUIDITY_STRESS": 0.88,
    })
    family_defaults["distribution"]["regime_weights"].update({
        "HIGH_VOL_EVENT": 1.10,
        "RISK_ON_OFF_MACRO": 1.04,
        "LIQUIDITY_STRESS": 0.96,
    })
    family_defaults["mixture"]["regime_weights"].update({
        "TREND_PERSISTENT": 1.06,
        "RANGE_MEAN_REVERTING": 1.06,
        "BREAKOUT_TRANSITION": 1.04,
        "HIGH_VOL_EVENT": 1.04,
    })
    family_defaults["memory"]["regime_weights"].update({
        "RISK_ON_OFF_MACRO": 1.12,
        "HIGH_VOL_EVENT": 1.08,
        "TREND_PERSISTENT": 1.04,
        "LIQUIDITY_STRESS": 0.92,
    })
    family_defaults["world"]["regime_weights"].update({
        "RISK_ON_OFF_MACRO": 1.14,
        "HIGH_VOL_EVENT": 1.08,
        "BREAKOUT_TRANSITION": 1.06,
    })
    family_defaults["rule"]["regime_weights"].update({
        "RANGE_MEAN_REVERTING": 1.10,
        "SESSION_FLOW": 1.12,
        "HIGH_VOL_EVENT": 0.76,
        "LIQUIDITY_STRESS": 0.82,
    })

    family_defaults["linear"]["news_compatibility"] = 0.78
    family_defaults["tree"]["news_compatibility"] = 0.86
    family_defaults["recurrent"]["news_compatibility"] = 1.04
    family_defaults["convolutional"]["news_compatibility"] = 0.92
    family_defaults["transformer"]["news_compatibility"] = 1.10
    family_defaults["state_space"]["news_compatibility"] = 1.06
    family_defaults["distribution"]["news_compatibility"] = 1.10
    family_defaults["mixture"]["news_compatibility"] = 1.02
    family_defaults["memory"]["news_compatibility"] = 1.08
    family_defaults["world"]["news_compatibility"] = 1.08
    family_defaults["rule"]["news_compatibility"] = 0.70

    family_defaults["linear"]["liquidity_robustness"] = 0.90
    family_defaults["tree"]["liquidity_robustness"] = 0.94
    family_defaults["recurrent"]["liquidity_robustness"] = 0.88
    family_defaults["convolutional"]["liquidity_robustness"] = 0.92
    family_defaults["transformer"]["liquidity_robustness"] = 0.82
    family_defaults["state_space"]["liquidity_robustness"] = 0.86
    family_defaults["distribution"]["liquidity_robustness"] = 0.96
    family_defaults["mixture"]["liquidity_robustness"] = 0.90
    family_defaults["memory"]["liquidity_robustness"] = 0.88
    family_defaults["world"]["liquidity_robustness"] = 0.90
    family_defaults["rule"]["liquidity_robustness"] = 0.84

    pair_tag_overrides = {
        "dollar_core": {
            "regime_bias": {
                "HIGH_VOL_EVENT": 1.10,
                "RISK_ON_OFF_MACRO": 1.06,
                "SESSION_FLOW": 1.02,
            },
            "caution_threshold": 0.54,
            "abstain_threshold": 0.36,
            "block_threshold": 0.18,
        },
        "yen_cross": {
            "regime_bias": {
                "HIGH_VOL_EVENT": 1.14,
                "RISK_ON_OFF_MACRO": 1.12,
                "LIQUIDITY_STRESS": 1.08,
            },
            "caution_threshold": 0.50,
            "abstain_threshold": 0.32,
            "block_threshold": 0.15,
        },
        "commodity_fx": {
            "regime_bias": {
                "RISK_ON_OFF_MACRO": 1.06,
                "SESSION_FLOW": 1.04,
                "RANGE_MEAN_REVERTING": 1.02,
            },
            "caution_threshold": 0.56,
            "abstain_threshold": 0.37,
            "block_threshold": 0.20,
        },
        "europe_rates": {
            "regime_bias": {
                "HIGH_VOL_EVENT": 1.08,
                "RISK_ON_OFF_MACRO": 1.04,
                "TREND_PERSISTENT": 1.03,
            },
            "caution_threshold": 0.53,
            "abstain_threshold": 0.34,
            "block_threshold": 0.17,
        },
    }

    return {
        "schema_version": ADAPTIVE_ROUTER_CONFIG_VERSION,
        "enabled": True,
        "router_mode": "WEIGHTED_ENSEMBLE",
        "fallback_to_student_router_only": True,
        "thresholds": {
            "caution_threshold": 0.55,
            "abstain_threshold": 0.35,
            "block_threshold": 0.16,
            "confidence_floor": 0.12,
            "suppression_threshold": 0.34,
            "downweight_threshold": 0.78,
            "stale_news_abstain_bias": 0.24,
            "stale_news_force_caution": True,
            "min_plugin_weight": 0.05,
            "max_plugin_weight": 1.80,
            "max_active_weight_share": 0.72,
        },
        "family_defaults": family_defaults,
        "pair_tag_overrides": pair_tag_overrides,
        "plugin_patterns": [
            {
                "id": "macro_regime_models",
                "match_any": ["gha", "stmn", "macro", "world", "memory", "tesseract"],
                "global_weight_mult": 1.04,
                "regime_weights": {
                    "HIGH_VOL_EVENT": 1.12,
                    "RISK_ON_OFF_MACRO": 1.10,
                    "LIQUIDITY_STRESS": 0.92,
                },
                "news_compatibility": 1.14,
                "liquidity_robustness": 0.90,
            },
            {
                "id": "trend_breakout_models",
                "match_any": ["tft", "patch", "trend", "momentum", "state", "qcew", "tesseract"],
                "global_weight_mult": 1.02,
                "regime_weights": {
                    "TREND_PERSISTENT": 1.10,
                    "BREAKOUT_TRANSITION": 1.12,
                    "RANGE_MEAN_REVERTING": 0.88,
                },
                "news_compatibility": 1.04,
                "liquidity_robustness": 0.88,
            },
            {
                "id": "mean_reversion_models",
                "match_any": ["mean", "revert", "ou", "pa", "kalman", "fewc"],
                "global_weight_mult": 1.00,
                "regime_weights": {
                    "RANGE_MEAN_REVERTING": 1.12,
                    "HIGH_VOL_EVENT": 0.76,
                    "BREAKOUT_TRANSITION": 0.82,
                },
                "news_compatibility": 0.76,
                "liquidity_robustness": 0.92,
            },
        ],
        "plugin_overrides": {
            "ai_fewc": {
                "regime_weights": {
                    "RANGE_MEAN_REVERTING": 1.12,
                    "SESSION_FLOW": 1.06,
                    "HIGH_VOL_EVENT": 0.84,
                }
            },
            "ai_gha": {
                "regime_weights": {
                    "HIGH_VOL_EVENT": 1.10,
                    "RISK_ON_OFF_MACRO": 1.10,
                    "TREND_PERSISTENT": 1.04,
                },
                "news_compatibility": 1.16,
            },
            "ai_tesseract": {
                "regime_weights": {
                    "TREND_PERSISTENT": 1.10,
                    "BREAKOUT_TRANSITION": 1.10,
                    "HIGH_VOL_EVENT": 1.06,
                }
            },
            "ai_qcew": {
                "regime_weights": {
                    "BREAKOUT_TRANSITION": 1.08,
                    "RISK_ON_OFF_MACRO": 1.04,
                    "LIQUIDITY_STRESS": 0.84,
                },
                "liquidity_robustness": 0.82,
            },
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
        if isinstance(payload_value, list):
            return deepcopy(payload_value)
        return deepcopy(default_value)
    if payload_value is None:
        return deepcopy(default_value)
    return deepcopy(payload_value)


def ensure_default_config_file(path: Path | None = None) -> Path:
    config_path = path or ADAPTIVE_ROUTER_CONFIG_PATH
    ensure_adaptive_router_dirs()
    config_path.parent.mkdir(parents=True, exist_ok=True)
    if not config_path.exists():
        config_path.write_text(
            json.dumps(default_config(), indent=2, sort_keys=True),
            encoding="utf-8",
        )
    return config_path


def _validate_regime_weights(payload: dict[str, Any], prefix: str) -> None:
    for regime in ADAPTIVE_ROUTER_REGIMES:
        try:
            value = float(payload.get(regime, 1.0))
        except Exception as exc:
            raise OfflineLabError(f"{prefix}.{regime} must be numeric") from exc
        if value < 0.05 or value > 2.50:
            raise OfflineLabError(f"{prefix}.{regime} must stay within 0.05..2.50")


def _validate_session_weights(payload: dict[str, Any], prefix: str) -> None:
    for session in ADAPTIVE_ROUTER_SESSIONS:
        try:
            value = float(payload.get(session, 1.0))
        except Exception as exc:
            raise OfflineLabError(f"{prefix}.{session} must be numeric") from exc
        if value < 0.05 or value > 2.50:
            raise OfflineLabError(f"{prefix}.{session} must stay within 0.05..2.50")


def validate_config_payload(payload: dict[str, Any]) -> dict[str, Any]:
    version = int(payload.get("schema_version", 0) or 0)
    if version != ADAPTIVE_ROUTER_CONFIG_VERSION:
        raise OfflineLabError(
            f"Adaptive Router config schema mismatch: expected {ADAPTIVE_ROUTER_CONFIG_VERSION}, got {version}"
        )

    thresholds = payload.get("thresholds")
    if not isinstance(thresholds, dict):
        raise OfflineLabError("Adaptive Router thresholds are missing")
    numeric_thresholds = [
        "caution_threshold",
        "abstain_threshold",
        "block_threshold",
        "confidence_floor",
        "suppression_threshold",
        "downweight_threshold",
        "stale_news_abstain_bias",
        "min_plugin_weight",
        "max_plugin_weight",
        "max_active_weight_share",
    ]
    for key in numeric_thresholds:
        try:
            float(thresholds.get(key, 0.0))
        except Exception as exc:
            raise OfflineLabError(f"Adaptive Router threshold must be numeric: {key}") from exc

    family_defaults = payload.get("family_defaults")
    if not isinstance(family_defaults, dict) or not family_defaults:
        raise OfflineLabError("Adaptive Router family_defaults are missing")
    for family_name, spec in family_defaults.items():
        if not isinstance(spec, dict):
            raise OfflineLabError(f"Adaptive Router family spec must be an object: {family_name}")
        _validate_regime_weights(spec.get("regime_weights", {}), f"family_defaults.{family_name}.regime_weights")
        _validate_session_weights(spec.get("session_weights", {}), f"family_defaults.{family_name}.session_weights")
        for key in ("news_compatibility", "liquidity_robustness"):
            try:
                value = float(spec.get(key, 1.0))
            except Exception as exc:
                raise OfflineLabError(f"Adaptive Router family {key} must be numeric: {family_name}") from exc
            if value < 0.05 or value > 2.50:
                raise OfflineLabError(f"Adaptive Router family {key} must stay within 0.05..2.50: {family_name}")

    pair_tags = payload.get("pair_tag_overrides")
    if not isinstance(pair_tags, dict):
        raise OfflineLabError("Adaptive Router pair_tag_overrides must be an object")
    for tag_name, spec in pair_tags.items():
        if not isinstance(spec, dict):
            raise OfflineLabError(f"Adaptive Router pair_tag_overrides entry must be an object: {tag_name}")
        _validate_regime_weights(spec.get("regime_bias", {}), f"pair_tag_overrides.{tag_name}.regime_bias")

    plugin_patterns = payload.get("plugin_patterns")
    if not isinstance(plugin_patterns, list):
        raise OfflineLabError("Adaptive Router plugin_patterns must be an array")
    for idx, spec in enumerate(plugin_patterns):
        if not isinstance(spec, dict):
            raise OfflineLabError(f"Adaptive Router plugin pattern must be an object: {idx}")
        match_any = spec.get("match_any", [])
        if not isinstance(match_any, list) or not match_any:
            raise OfflineLabError(f"Adaptive Router plugin pattern must declare match_any: {idx}")
        _validate_regime_weights(spec.get("regime_weights", {}), f"plugin_patterns[{idx}].regime_weights")

    overrides = payload.get("plugin_overrides")
    if not isinstance(overrides, dict):
        raise OfflineLabError("Adaptive Router plugin_overrides must be an object")
    for plugin_name, spec in overrides.items():
        if not isinstance(spec, dict):
            raise OfflineLabError(f"Adaptive Router plugin override must be an object: {plugin_name}")
        if "regime_weights" in spec:
            _validate_regime_weights(spec["regime_weights"], f"plugin_overrides.{plugin_name}.regime_weights")

    return payload


def load_config(path: Path | None = None) -> dict[str, Any]:
    config_path = ensure_default_config_file(path)
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise OfflineLabError(f"Adaptive Router config missing: {config_path}") from exc
    except json.JSONDecodeError as exc:
        raise OfflineLabError(f"Adaptive Router config is not valid JSON: {config_path}") from exc
    if not isinstance(payload, dict):
        raise OfflineLabError(f"Adaptive Router config must be a JSON object: {config_path}")
    merged = _merge_defaults(default_config(), payload)
    validate_config_payload(merged)
    return merged
