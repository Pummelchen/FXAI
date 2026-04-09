from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from .common import OfflineLabError
from .dynamic_ensemble_contracts import (
    DYNAMIC_ENSEMBLE_CONFIG_PATH,
    DYNAMIC_ENSEMBLE_CONFIG_VERSION,
    DYNAMIC_ENSEMBLE_FAMILIES,
    DYNAMIC_ENSEMBLE_RUNTIME_CONFIG_PATH,
    ensure_dynamic_ensemble_dirs,
)


def default_config() -> dict[str, Any]:
    family_defaults: dict[str, dict[str, float]] = {
        family: {
            "news_compat": 1.0,
            "rates_compat": 1.0,
            "micro_compat": 1.0,
            "cost_robustness": 1.0,
            "confidence_cap": 0.96,
            "disagreement_tolerance": 1.0,
        }
        for family in DYNAMIC_ENSEMBLE_FAMILIES
    }

    family_defaults["linear"].update({
        "news_compat": 0.82,
        "rates_compat": 0.92,
        "micro_compat": 0.96,
        "cost_robustness": 0.98,
        "confidence_cap": 0.84,
        "disagreement_tolerance": 0.92,
    })
    family_defaults["tree"].update({
        "news_compat": 0.90,
        "rates_compat": 0.94,
        "micro_compat": 0.96,
        "cost_robustness": 0.98,
        "confidence_cap": 0.88,
        "disagreement_tolerance": 0.96,
    })
    family_defaults["recurrent"].update({
        "news_compat": 1.00,
        "rates_compat": 0.98,
        "micro_compat": 0.90,
        "cost_robustness": 0.86,
        "confidence_cap": 0.90,
        "disagreement_tolerance": 1.00,
    })
    family_defaults["convolutional"].update({
        "news_compat": 0.92,
        "rates_compat": 0.92,
        "micro_compat": 0.98,
        "cost_robustness": 0.92,
        "confidence_cap": 0.90,
        "disagreement_tolerance": 0.96,
    })
    family_defaults["transformer"].update({
        "news_compat": 1.08,
        "rates_compat": 1.00,
        "micro_compat": 0.84,
        "cost_robustness": 0.82,
        "confidence_cap": 0.86,
        "disagreement_tolerance": 0.92,
    })
    family_defaults["state_space"].update({
        "news_compat": 1.02,
        "rates_compat": 1.02,
        "micro_compat": 0.88,
        "cost_robustness": 0.86,
        "confidence_cap": 0.88,
        "disagreement_tolerance": 0.94,
    })
    family_defaults["distribution"].update({
        "news_compat": 1.08,
        "rates_compat": 1.04,
        "micro_compat": 0.90,
        "cost_robustness": 0.96,
        "confidence_cap": 0.92,
        "disagreement_tolerance": 0.96,
    })
    family_defaults["mixture"].update({
        "news_compat": 1.02,
        "rates_compat": 0.98,
        "micro_compat": 0.92,
        "cost_robustness": 0.92,
        "confidence_cap": 0.90,
        "disagreement_tolerance": 1.02,
    })
    family_defaults["memory"].update({
        "news_compat": 1.06,
        "rates_compat": 1.06,
        "micro_compat": 0.86,
        "cost_robustness": 0.88,
        "confidence_cap": 0.88,
        "disagreement_tolerance": 0.96,
    })
    family_defaults["world"].update({
        "news_compat": 1.06,
        "rates_compat": 1.08,
        "micro_compat": 0.88,
        "cost_robustness": 0.90,
        "confidence_cap": 0.90,
        "disagreement_tolerance": 0.96,
    })
    family_defaults["rule"].update({
        "news_compat": 0.76,
        "rates_compat": 0.88,
        "micro_compat": 1.02,
        "cost_robustness": 1.02,
        "confidence_cap": 0.78,
        "disagreement_tolerance": 0.90,
    })

    return {
        "schema_version": DYNAMIC_ENSEMBLE_CONFIG_VERSION,
        "enabled": True,
        "fallback_to_routed_ensemble": True,
        "thresholds": {
            "suppress_trust_threshold": 0.30,
            "downweight_trust_threshold": 0.72,
            "caution_quality_threshold": 0.56,
            "abstain_quality_threshold": 0.36,
            "block_quality_threshold": 0.18,
            "min_effective_weight": 0.04,
            "max_weight_share": 0.66,
            "min_active_plugins": 1,
        },
        "penalties": {
            "confidence_gap_penalty": 0.52,
            "context_regret_penalty": 0.38,
            "disagreement_penalty": 0.28,
            "drift_penalty": 0.18,
            "spread_cost_penalty": 0.28,
            "news_penalty": 0.24,
            "rates_penalty": 0.20,
            "micro_penalty": 0.30,
            "stale_context_penalty": 0.24,
            "single_plugin_quality_penalty": 0.16,
            "concentration_quality_penalty": 0.22,
        },
        "weights": {
            "reliability_gain": 0.34,
            "context_edge_gain": 0.18,
            "global_edge_gain": 0.10,
            "portfolio_gain": 0.16,
            "context_trust_gain": 0.18,
            "adaptive_upweight_gain": 0.05,
            "adaptive_downweight_penalty": 0.14,
        },
        "families": family_defaults,
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


def ensure_default_files(path: Path | None = None) -> Path:
    path = path or DYNAMIC_ENSEMBLE_CONFIG_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text(json.dumps(default_config(), indent=2, sort_keys=True), encoding="utf-8")
    return path


def validate_config_payload(payload: dict[str, Any]) -> dict[str, Any]:
    schema_version = int(payload.get("schema_version", 0) or 0)
    if schema_version != DYNAMIC_ENSEMBLE_CONFIG_VERSION:
        raise OfflineLabError(
            f"Dynamic ensemble config schema mismatch: expected {DYNAMIC_ENSEMBLE_CONFIG_VERSION}, got {schema_version}"
        )
    thresholds = payload.get("thresholds")
    penalties = payload.get("penalties")
    weights = payload.get("weights")
    families = payload.get("families")
    if not isinstance(thresholds, dict) or not thresholds:
        raise OfflineLabError("Dynamic ensemble thresholds must be a JSON object")
    if not isinstance(penalties, dict) or not penalties:
        raise OfflineLabError("Dynamic ensemble penalties must be a JSON object")
    if not isinstance(weights, dict) or not weights:
        raise OfflineLabError("Dynamic ensemble weights must be a JSON object")
    if not isinstance(families, dict) or not families:
        raise OfflineLabError("Dynamic ensemble families must be a JSON object")
    if float(thresholds.get("max_weight_share", 0.0) or 0.0) <= 0.0:
        raise OfflineLabError("Dynamic ensemble max_weight_share must be positive")
    if int(thresholds.get("min_active_plugins", 0) or 0) < 1:
        raise OfflineLabError("Dynamic ensemble min_active_plugins must be at least 1")
    for family in DYNAMIC_ENSEMBLE_FAMILIES:
        spec = families.get(family)
        if not isinstance(spec, dict):
            raise OfflineLabError(f"Dynamic ensemble family spec missing for {family}")
    return payload


def export_runtime_config(payload: dict[str, Any]) -> Path:
    ensure_dynamic_ensemble_dirs()
    normalized = validate_config_payload(payload)
    thresholds = dict(normalized["thresholds"])
    penalties = dict(normalized["penalties"])
    weights = dict(normalized["weights"])
    families = dict(normalized["families"])

    lines = [
        f"schema_version\t{DYNAMIC_ENSEMBLE_CONFIG_VERSION}",
        f"enabled\t{1 if normalized.get('enabled', True) else 0}",
        f"fallback_to_routed_ensemble\t{1 if normalized.get('fallback_to_routed_ensemble', True) else 0}",
    ]
    for key, value in sorted(thresholds.items()):
        lines.append(f"threshold_{key}\t{value}")
    for key, value in sorted(penalties.items()):
        lines.append(f"penalty_{key}\t{value}")
    for key, value in sorted(weights.items()):
        lines.append(f"weight_{key}\t{value}")
    for family in DYNAMIC_ENSEMBLE_FAMILIES:
        spec = dict(families.get(family, {}))
        for key, value in sorted(spec.items()):
            lines.append(f"family_{key}_{family}\t{value}")

    DYNAMIC_ENSEMBLE_RUNTIME_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    DYNAMIC_ENSEMBLE_RUNTIME_CONFIG_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return DYNAMIC_ENSEMBLE_RUNTIME_CONFIG_PATH


def load_config(path: Path | None = None) -> dict[str, Any]:
    path = path or DYNAMIC_ENSEMBLE_CONFIG_PATH
    ensure_default_files(path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise OfflineLabError(f"Dynamic ensemble config missing: {path}") from exc
    except json.JSONDecodeError as exc:
        raise OfflineLabError(f"Dynamic ensemble config is not valid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise OfflineLabError(f"Dynamic ensemble config must be a JSON object: {path}")
    merged = _merge_defaults(default_config(), payload)
    validate_config_payload(merged)
    export_runtime_config(merged)
    return merged
