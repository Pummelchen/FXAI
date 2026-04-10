from __future__ import annotations

import json
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .common import OfflineLabError
from .prob_calibration_contracts import (
    PROB_CALIBRATION_CONFIG_PATH,
    PROB_CALIBRATION_CONFIG_VERSION,
    PROB_CALIBRATION_MEMORY_PATH,
    PROB_CALIBRATION_MEMORY_VERSION,
    PROB_CALIBRATION_METHODS,
    PROB_CALIBRATION_RUNTIME_CONFIG_PATH,
    PROB_CALIBRATION_RUNTIME_MEMORY_PATH,
    PROB_CALIBRATION_TIER_KINDS,
    ensure_prob_calibration_dirs,
    isoformat_utc,
    json_dump,
)

_REGIME_TIER_DEFAULTS: list[dict[str, Any]] = [
    {
        "kind": "REGIME",
        "symbol": "*",
        "session": "*",
        "regime": "TREND_PERSISTENT",
        "support": 196,
        "prob_scale": 2.40,
        "prob_bias": 0.02,
        "skip_bias": 0.04,
        "move_mean_scale": 0.90,
        "move_q25_scale": 0.78,
        "move_q50_scale": 0.86,
        "move_q75_scale": 0.98,
        "calibration_quality": 0.66,
        "uncertainty_mult": 0.86,
        "confidence_cap": 0.66,
    },
    {
        "kind": "REGIME",
        "symbol": "*",
        "session": "*",
        "regime": "RANGE_MEAN_REVERTING",
        "support": 172,
        "prob_scale": 2.05,
        "prob_bias": 0.00,
        "skip_bias": 0.05,
        "move_mean_scale": 0.82,
        "move_q25_scale": 0.72,
        "move_q50_scale": 0.80,
        "move_q75_scale": 0.92,
        "calibration_quality": 0.61,
        "uncertainty_mult": 0.96,
        "confidence_cap": 0.63,
    },
    {
        "kind": "REGIME",
        "symbol": "*",
        "session": "*",
        "regime": "BREAKOUT_TRANSITION",
        "support": 128,
        "prob_scale": 1.92,
        "prob_bias": 0.00,
        "skip_bias": 0.07,
        "move_mean_scale": 0.80,
        "move_q25_scale": 0.64,
        "move_q50_scale": 0.76,
        "move_q75_scale": 0.94,
        "calibration_quality": 0.55,
        "uncertainty_mult": 1.08,
        "confidence_cap": 0.61,
    },
    {
        "kind": "REGIME",
        "symbol": "*",
        "session": "*",
        "regime": "HIGH_VOL_EVENT",
        "support": 96,
        "prob_scale": 1.74,
        "prob_bias": 0.00,
        "skip_bias": 0.13,
        "move_mean_scale": 0.72,
        "move_q25_scale": 0.50,
        "move_q50_scale": 0.64,
        "move_q75_scale": 0.86,
        "calibration_quality": 0.48,
        "uncertainty_mult": 1.34,
        "confidence_cap": 0.58,
    },
    {
        "kind": "REGIME",
        "symbol": "*",
        "session": "*",
        "regime": "RISK_ON_OFF_MACRO",
        "support": 88,
        "prob_scale": 1.88,
        "prob_bias": 0.00,
        "skip_bias": 0.08,
        "move_mean_scale": 0.78,
        "move_q25_scale": 0.62,
        "move_q50_scale": 0.72,
        "move_q75_scale": 0.90,
        "calibration_quality": 0.54,
        "uncertainty_mult": 1.10,
        "confidence_cap": 0.60,
    },
    {
        "kind": "REGIME",
        "symbol": "*",
        "session": "*",
        "regime": "LIQUIDITY_STRESS",
        "support": 84,
        "prob_scale": 1.62,
        "prob_bias": 0.00,
        "skip_bias": 0.14,
        "move_mean_scale": 0.70,
        "move_q25_scale": 0.48,
        "move_q50_scale": 0.60,
        "move_q75_scale": 0.84,
        "calibration_quality": 0.44,
        "uncertainty_mult": 1.40,
        "confidence_cap": 0.57,
    },
    {
        "kind": "REGIME",
        "symbol": "*",
        "session": "*",
        "regime": "SESSION_FLOW",
        "support": 110,
        "prob_scale": 1.82,
        "prob_bias": 0.00,
        "skip_bias": 0.07,
        "move_mean_scale": 0.76,
        "move_q25_scale": 0.60,
        "move_q50_scale": 0.70,
        "move_q75_scale": 0.88,
        "calibration_quality": 0.52,
        "uncertainty_mult": 1.06,
        "confidence_cap": 0.60,
    },
]


def default_config() -> dict[str, Any]:
    return {
        "schema_version": PROB_CALIBRATION_CONFIG_VERSION,
        "enabled": True,
        "allow_abstain_flag": True,
        "neutral_blend_gain": 0.65,
        "skip_uncertainty_gain": 0.12,
        "skip_calibration_credit": 0.05,
        "skip_floor": 0.02,
        "skip_cap": 0.96,
        "base_uncertainty_score": 0.18,
        "support_soft_floor": 64,
        "support_hard_floor": 16,
        "memory_stale_after_hours": 96,
        "min_calibration_quality": 0.44,
        "max_uncertainty_score": 0.92,
        "signal_zero_band": 0.035,
        "edge_floor_mult": 0.08,
        "trade_edge_floor_points": 0.05,
        "soft_fallback": {
            "prob_scale": 1.60,
            "skip_bias": 0.08,
            "move_mean_scale": 0.78,
            "move_q25_scale": 0.60,
            "move_q50_scale": 0.72,
            "move_q75_scale": 0.88,
            "confidence_cap": 0.58,
        },
        "uncertainty_penalties": {
            "support": 0.34,
            "quality": 0.28,
            "disagreement": 0.26,
            "distribution_width": 0.22,
            "news": 0.18,
            "rates": 0.14,
            "micro": 0.24,
            "dynamic_abstain": 0.20,
            "adaptive_abstain": 0.22,
            "stale_context": 0.16,
        },
        "risk_penalties": {
            "news_block_mult": 0.32,
            "rates_block_mult": 0.24,
            "micro_block_mult": 0.36,
            "caution_posture_mult": 0.14,
            "abstain_posture_mult": 0.24,
            "block_posture_mult": 0.42,
            "fill_risk_mult": 0.20,
            "path_risk_mult": 0.16,
        },
        "bucket_hierarchy": [
            "PAIR_SESSION_REGIME",
            "PAIR_REGIME",
            "REGIME",
            "GLOBAL",
        ],
    }


def default_memory() -> dict[str, Any]:
    return {
        "schema_version": PROB_CALIBRATION_MEMORY_VERSION,
        "generated_at": isoformat_utc(),
        "default_method": "LOGISTIC_AFFINE",
        "tiers": [
            {
                "kind": "GLOBAL",
                "symbol": "*",
                "session": "*",
                "regime": "*",
                "support": 512,
                "prob_scale": 2.12,
                "prob_bias": 0.0,
                "skip_bias": 0.05,
                "move_mean_scale": 0.84,
                "move_q25_scale": 0.68,
                "move_q50_scale": 0.78,
                "move_q75_scale": 0.92,
                "calibration_quality": 0.58,
                "uncertainty_mult": 1.0,
                "confidence_cap": 0.62,
            },
            *_REGIME_TIER_DEFAULTS,
        ],
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


def _require_bool(payload: dict[str, Any], key: str) -> None:
    if key in payload and not isinstance(payload[key], bool):
        raise OfflineLabError(f"Probabilistic calibration {key} must be a boolean")


def validate_config_payload(payload: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise OfflineLabError("Probabilistic calibration config must be a JSON object")
    schema_version = int(payload.get("schema_version", 0) or 0)
    if schema_version != PROB_CALIBRATION_CONFIG_VERSION:
        raise OfflineLabError(
            f"Probabilistic calibration config schema mismatch: expected {PROB_CALIBRATION_CONFIG_VERSION}, got {schema_version}"
        )
    _require_bool(payload, "enabled")
    _require_bool(payload, "allow_abstain_flag")
    support_soft_floor = int(payload.get("support_soft_floor", 0) or 0)
    support_hard_floor = int(payload.get("support_hard_floor", 0) or 0)
    if support_soft_floor < 8:
        raise OfflineLabError("Probabilistic calibration support_soft_floor must be at least 8")
    if support_hard_floor < 1:
        raise OfflineLabError("Probabilistic calibration support_hard_floor must be at least 1")
    if support_hard_floor > support_soft_floor:
        raise OfflineLabError("Probabilistic calibration support_hard_floor must be <= support_soft_floor")
    bucket_hierarchy = payload.get("bucket_hierarchy")
    if not isinstance(bucket_hierarchy, list) or not bucket_hierarchy:
        raise OfflineLabError("Probabilistic calibration bucket_hierarchy must be a non-empty array")
    normalized_hierarchy: list[str] = []
    seen: set[str] = set()
    for value in bucket_hierarchy:
        label = str(value or "").strip().upper()
        if label not in PROB_CALIBRATION_TIER_KINDS:
            raise OfflineLabError(f"Probabilistic calibration bucket_hierarchy contains unknown tier kind: {label}")
        if label in seen:
            raise OfflineLabError(f"Probabilistic calibration bucket_hierarchy duplicates tier kind: {label}")
        seen.add(label)
        normalized_hierarchy.append(label)
    payload = dict(payload)
    payload["bucket_hierarchy"] = normalized_hierarchy
    return payload


def validate_memory_payload(payload: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise OfflineLabError("Probabilistic calibration memory must be a JSON object")
    schema_version = int(payload.get("schema_version", 0) or 0)
    if schema_version != PROB_CALIBRATION_MEMORY_VERSION:
        raise OfflineLabError(
            f"Probabilistic calibration memory schema mismatch: expected {PROB_CALIBRATION_MEMORY_VERSION}, got {schema_version}"
        )
    method = str(payload.get("default_method", "") or "").upper()
    if method not in PROB_CALIBRATION_METHODS:
        raise OfflineLabError(f"Probabilistic calibration default_method must be one of {PROB_CALIBRATION_METHODS}")
    tiers = payload.get("tiers")
    if not isinstance(tiers, list) or not tiers:
        raise OfflineLabError("Probabilistic calibration tiers must be a non-empty array")
    normalized_tiers: list[dict[str, Any]] = []
    seen_keys: set[tuple[str, str, str, str]] = set()
    for index, raw in enumerate(tiers):
        if not isinstance(raw, dict):
            raise OfflineLabError(f"Probabilistic calibration tier at index {index} must be a JSON object")
        kind = str(raw.get("kind", "") or "").strip().upper()
        if kind not in PROB_CALIBRATION_TIER_KINDS:
            raise OfflineLabError(f"Probabilistic calibration tier[{index}] has invalid kind: {kind}")
        tier = {
            "kind": kind,
            "symbol": str(raw.get("symbol", "*") or "*").strip().upper() or "*",
            "session": str(raw.get("session", "*") or "*").strip().upper() or "*",
            "regime": str(raw.get("regime", "*") or "*").strip().upper() or "*",
            "support": int(raw.get("support", 0) or 0),
            "prob_scale": float(raw.get("prob_scale", 0.0) or 0.0),
            "prob_bias": float(raw.get("prob_bias", 0.0) or 0.0),
            "skip_bias": float(raw.get("skip_bias", 0.0) or 0.0),
            "move_mean_scale": float(raw.get("move_mean_scale", 0.0) or 0.0),
            "move_q25_scale": float(raw.get("move_q25_scale", 0.0) or 0.0),
            "move_q50_scale": float(raw.get("move_q50_scale", 0.0) or 0.0),
            "move_q75_scale": float(raw.get("move_q75_scale", 0.0) or 0.0),
            "calibration_quality": float(raw.get("calibration_quality", 0.0) or 0.0),
            "uncertainty_mult": float(raw.get("uncertainty_mult", 0.0) or 0.0),
            "confidence_cap": float(raw.get("confidence_cap", 0.0) or 0.0),
        }
        if tier["support"] < 0:
            raise OfflineLabError(f"Probabilistic calibration tier[{index}] support must be >= 0")
        if tier["prob_scale"] <= 0.0:
            raise OfflineLabError(f"Probabilistic calibration tier[{index}] prob_scale must be positive")
        if not (0.0 <= tier["skip_bias"] <= 0.5):
            raise OfflineLabError(f"Probabilistic calibration tier[{index}] skip_bias must be within [0, 0.5]")
        for key in ("move_mean_scale", "move_q25_scale", "move_q50_scale", "move_q75_scale", "uncertainty_mult"):
            if tier[key] <= 0.0:
                raise OfflineLabError(f"Probabilistic calibration tier[{index}] {key} must be positive")
        if not (0.0 <= tier["calibration_quality"] <= 1.0):
            raise OfflineLabError(f"Probabilistic calibration tier[{index}] calibration_quality must be within [0, 1]")
        if not (0.50 <= tier["confidence_cap"] <= 0.95):
            raise OfflineLabError(f"Probabilistic calibration tier[{index}] confidence_cap must be within [0.50, 0.95]")
        if kind == "PAIR_SESSION_REGIME":
            if tier["symbol"] == "*" or tier["session"] == "*" or tier["regime"] == "*":
                raise OfflineLabError("PAIR_SESSION_REGIME tiers must provide exact symbol, session, and regime")
        elif kind == "PAIR_REGIME":
            if tier["symbol"] == "*" or tier["regime"] == "*":
                raise OfflineLabError("PAIR_REGIME tiers must provide exact symbol and regime")
        elif kind == "REGIME":
            if tier["regime"] == "*":
                raise OfflineLabError("REGIME tiers must provide an exact regime")
        key_tuple = (tier["kind"], tier["symbol"], tier["session"], tier["regime"])
        if key_tuple in seen_keys:
            raise OfflineLabError(f"Duplicate probabilistic calibration tier: {key_tuple}")
        seen_keys.add(key_tuple)
        normalized_tiers.append(tier)
    payload = dict(payload)
    payload["default_method"] = method
    payload["tiers"] = normalized_tiers
    return payload


def export_runtime_config(payload: dict[str, Any]) -> Path:
    ensure_prob_calibration_dirs()
    normalized = validate_config_payload(payload)
    lines = [
        f"schema_version\t{PROB_CALIBRATION_CONFIG_VERSION}",
        f"enabled\t{1 if normalized.get('enabled', True) else 0}",
        f"allow_abstain_flag\t{1 if normalized.get('allow_abstain_flag', True) else 0}",
    ]
    for key, value in sorted(normalized.items()):
        if key in {"schema_version", "enabled", "allow_abstain_flag", "bucket_hierarchy", "soft_fallback", "uncertainty_penalties", "risk_penalties"}:
            continue
        lines.append(f"{key}\t{value}")
    for value in normalized["bucket_hierarchy"]:
        lines.append(f"bucket_hierarchy\t{value}")
    for key, value in sorted(dict(normalized.get("soft_fallback", {})).items()):
        lines.append(f"soft_{key}\t{value}")
    for key, value in sorted(dict(normalized.get("uncertainty_penalties", {})).items()):
        lines.append(f"uncertainty_{key}\t{value}")
    for key, value in sorted(dict(normalized.get("risk_penalties", {})).items()):
        lines.append(f"risk_{key}\t{value}")
    PROB_CALIBRATION_RUNTIME_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    PROB_CALIBRATION_RUNTIME_CONFIG_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return PROB_CALIBRATION_RUNTIME_CONFIG_PATH


def export_runtime_memory(payload: dict[str, Any]) -> Path:
    ensure_prob_calibration_dirs()
    normalized = validate_memory_payload(payload)
    generated_at_raw = str(normalized.get("generated_at", "") or "")
    generated_at_unix = 0
    if generated_at_raw:
        try:
            generated_at_unix = int(
                datetime.fromisoformat(generated_at_raw.replace("Z", "+00:00"))
                .astimezone(timezone.utc)
                .timestamp()
            )
        except Exception:
            generated_at_unix = 0
    lines = [
        f"schema_version\t{PROB_CALIBRATION_MEMORY_VERSION}",
        f"generated_at\t{generated_at_raw}",
        f"generated_at_unix\t{generated_at_unix}",
        f"default_method\t{normalized['default_method']}",
    ]
    for tier in normalized["tiers"]:
        lines.append(
            "\t".join(
                [
                    "tier",
                    tier["kind"],
                    tier["symbol"],
                    tier["session"],
                    tier["regime"],
                    str(tier["support"]),
                    str(tier["prob_scale"]),
                    str(tier["prob_bias"]),
                    str(tier["skip_bias"]),
                    str(tier["move_mean_scale"]),
                    str(tier["move_q25_scale"]),
                    str(tier["move_q50_scale"]),
                    str(tier["move_q75_scale"]),
                    str(tier["calibration_quality"]),
                    str(tier["uncertainty_mult"]),
                    str(tier["confidence_cap"]),
                ]
            )
        )
    PROB_CALIBRATION_RUNTIME_MEMORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    PROB_CALIBRATION_RUNTIME_MEMORY_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return PROB_CALIBRATION_RUNTIME_MEMORY_PATH


def ensure_default_files() -> tuple[dict[str, Any], dict[str, Any]]:
    ensure_prob_calibration_dirs()
    if not PROB_CALIBRATION_CONFIG_PATH.exists():
        json_dump(PROB_CALIBRATION_CONFIG_PATH, default_config())
    if not PROB_CALIBRATION_MEMORY_PATH.exists():
        json_dump(PROB_CALIBRATION_MEMORY_PATH, default_memory())
    config_payload = load_config()
    memory_payload = load_memory()
    return config_payload, memory_payload


def load_config(path: Path | None = None) -> dict[str, Any]:
    path = path or PROB_CALIBRATION_CONFIG_PATH
    ensure_prob_calibration_dirs()
    if not path.exists():
        json_dump(path, default_config())
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise OfflineLabError(f"Probabilistic calibration config missing: {path}") from exc
    except json.JSONDecodeError as exc:
        raise OfflineLabError(f"Probabilistic calibration config is not valid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise OfflineLabError(f"Probabilistic calibration config must be a JSON object: {path}")
    merged = _merge_defaults(default_config(), payload)
    validate_config_payload(merged)
    export_runtime_config(merged)
    return merged


def load_memory(path: Path | None = None) -> dict[str, Any]:
    path = path or PROB_CALIBRATION_MEMORY_PATH
    ensure_prob_calibration_dirs()
    if not path.exists():
        json_dump(path, default_memory())
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise OfflineLabError(f"Probabilistic calibration memory missing: {path}") from exc
    except json.JSONDecodeError as exc:
        raise OfflineLabError(f"Probabilistic calibration memory is not valid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise OfflineLabError(f"Probabilistic calibration memory must be a JSON object: {path}")
    merged = dict(default_memory())
    merged["generated_at"] = payload.get("generated_at", merged["generated_at"])
    merged["default_method"] = payload.get("default_method", merged["default_method"])
    merged["tiers"] = payload.get("tiers", merged["tiers"])
    validate_memory_payload(merged)
    export_runtime_memory(merged)
    return merged
