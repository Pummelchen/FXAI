from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from testlab.shared import EXECUTION_PROFILES

from .common import OfflineLabError
from .label_engine_contracts import (
    LABEL_ENGINE_ALLOWED_CANDIDATE_MODES,
    LABEL_ENGINE_CONFIG_PATH,
    LABEL_ENGINE_CONFIG_VERSION,
    ensure_label_engine_dirs,
    json_dump,
)


def default_config() -> dict[str, Any]:
    return {
        "schema_version": LABEL_ENGINE_CONFIG_VERSION,
        "label_version": 1,
        "enabled": True,
        "timeframe": "M1",
        "execution_profile": "default",
        "sample_start_bars": 16,
        "overlap_stride_bars": 1,
        "direction_zero_band_points": 0.50,
        "spread_multiplier": 1.0,
        "commission_points": 0.0,
        "safety_margin_points": 0.25,
        "default_point_sizes": {
            "fx": 0.00001,
            "fx_jpy": 0.001,
            "metal": 0.01,
            "crypto": 0.01,
            "index": 0.10,
            "share": 0.01,
            "other": 0.0001,
        },
        "symbol_point_overrides": {},
        "horizons": [
            {"id": "M5", "bars": 5},
            {"id": "M15", "bars": 15},
            {"id": "M30", "bars": 30},
            {"id": "H1", "bars": 60},
        ],
        "tradeability": {
            "favorable_cost_mult": 1.25,
            "tradeability_cost_mult": 1.10,
            "adverse_cost_mult": 1.00,
            "max_mae_cost_mult": 1.35,
            "min_favorable_points": 2.0,
            "min_adverse_points": 2.0,
            "max_time_to_favorable_ratio": 0.65,
            "require_favorable_before_adverse": True,
        },
        "meta_labeling": {
            "enabled": True,
            "candidate_mode": "BASELINE_MOMENTUM",
            "baseline_lookback_bars": 5,
            "baseline_signal_threshold_points": 6.0,
            "raw_score_scale_points": 20.0,
            "min_raw_signal_strength": 0.15,
            "execution_penalty_points": 0.0,
            "news_penalty_points": 0.0,
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


def validate_config_payload(payload: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise OfflineLabError("Label engine config must be a JSON object")
    schema_version = int(payload.get("schema_version", 0) or 0)
    if schema_version != LABEL_ENGINE_CONFIG_VERSION:
        raise OfflineLabError(
            f"Label engine schema_version must be {LABEL_ENGINE_CONFIG_VERSION}, received {schema_version}"
        )
    if not isinstance(payload.get("enabled", True), bool):
        raise OfflineLabError("Label engine enabled must be a boolean")
    execution_profile = str(payload.get("execution_profile", "default") or "default").strip()
    if execution_profile not in EXECUTION_PROFILES:
        raise OfflineLabError(f"Unknown label-engine execution profile: {execution_profile}")
    payload["execution_profile"] = execution_profile

    sample_start_bars = int(payload.get("sample_start_bars", 0) or 0)
    if sample_start_bars < 0:
        raise OfflineLabError("Label engine sample_start_bars must be >= 0")
    payload["sample_start_bars"] = sample_start_bars

    overlap_stride_bars = int(payload.get("overlap_stride_bars", 1) or 1)
    if overlap_stride_bars <= 0:
        raise OfflineLabError("Label engine overlap_stride_bars must be >= 1")
    payload["overlap_stride_bars"] = overlap_stride_bars

    if float(payload.get("direction_zero_band_points", 0.0) or 0.0) < 0.0:
        raise OfflineLabError("Label engine direction_zero_band_points must be >= 0")
    if float(payload.get("spread_multiplier", 1.0) or 1.0) <= 0.0:
        raise OfflineLabError("Label engine spread_multiplier must be > 0")

    default_point_sizes = payload.get("default_point_sizes", {})
    if not isinstance(default_point_sizes, dict) or not default_point_sizes:
        raise OfflineLabError("Label engine default_point_sizes must be a non-empty object")
    for key, value in default_point_sizes.items():
        if float(value or 0.0) <= 0.0:
            raise OfflineLabError(f"Label engine point size for {key} must be > 0")

    symbol_overrides = payload.get("symbol_point_overrides", {})
    if not isinstance(symbol_overrides, dict):
        raise OfflineLabError("Label engine symbol_point_overrides must be an object")
    for symbol, value in symbol_overrides.items():
        if not str(symbol or "").strip():
            raise OfflineLabError("Label engine symbol_point_overrides cannot use empty keys")
        if float(value or 0.0) <= 0.0:
            raise OfflineLabError(f"Label engine point override for {symbol} must be > 0")

    horizons = payload.get("horizons", [])
    if not isinstance(horizons, list) or not horizons:
        raise OfflineLabError("Label engine horizons must be a non-empty array")
    seen_ids: set[str] = set()
    normalized_horizons: list[dict[str, Any]] = []
    for item in horizons:
        if not isinstance(item, dict):
            raise OfflineLabError("Each label-engine horizon must be an object")
        horizon_id = str(item.get("id", "") or "").strip().upper()
        bars = int(item.get("bars", 0) or 0)
        if not horizon_id:
            raise OfflineLabError("Each label-engine horizon requires a non-empty id")
        if horizon_id in seen_ids:
            raise OfflineLabError(f"Duplicate label-engine horizon id: {horizon_id}")
        if bars <= 0:
            raise OfflineLabError(f"Label-engine horizon {horizon_id} must use bars > 0")
        seen_ids.add(horizon_id)
        normalized_horizons.append({"id": horizon_id, "bars": bars})
    payload["horizons"] = sorted(normalized_horizons, key=lambda item: (int(item["bars"]), str(item["id"])))

    tradeability = payload.get("tradeability", {})
    if not isinstance(tradeability, dict):
        raise OfflineLabError("Label engine tradeability config must be an object")
    favorable_mult = float(tradeability.get("favorable_cost_mult", 0.0) or 0.0)
    tradeability_mult = float(tradeability.get("tradeability_cost_mult", 0.0) or 0.0)
    adverse_mult = float(tradeability.get("adverse_cost_mult", 0.0) or 0.0)
    max_mae_mult = float(tradeability.get("max_mae_cost_mult", 0.0) or 0.0)
    time_ratio = float(tradeability.get("max_time_to_favorable_ratio", 0.0) or 0.0)
    if favorable_mult <= 0.0 or tradeability_mult <= 0.0 or adverse_mult <= 0.0 or max_mae_mult <= 0.0:
        raise OfflineLabError("Label engine tradeability cost multiples must all be > 0")
    if not 0.0 < time_ratio <= 1.0:
        raise OfflineLabError("Label engine max_time_to_favorable_ratio must be in (0, 1]")
    if favorable_mult < tradeability_mult:
        raise OfflineLabError("Label engine favorable_cost_mult must be >= tradeability_cost_mult")

    meta = payload.get("meta_labeling", {})
    if not isinstance(meta, dict):
        raise OfflineLabError("Label engine meta_labeling config must be an object")
    if not isinstance(meta.get("enabled", True), bool):
        raise OfflineLabError("Label engine meta_labeling.enabled must be a boolean")
    candidate_mode = str(meta.get("candidate_mode", "BASELINE_MOMENTUM") or "BASELINE_MOMENTUM").strip().upper()
    if candidate_mode not in LABEL_ENGINE_ALLOWED_CANDIDATE_MODES:
        raise OfflineLabError(
            f"Unsupported label-engine candidate_mode {candidate_mode}; expected one of {', '.join(LABEL_ENGINE_ALLOWED_CANDIDATE_MODES)}"
        )
    if int(meta.get("baseline_lookback_bars", 0) or 0) <= 0:
        raise OfflineLabError("Label engine baseline_lookback_bars must be > 0")
    if float(meta.get("baseline_signal_threshold_points", 0.0) or 0.0) <= 0.0:
        raise OfflineLabError("Label engine baseline_signal_threshold_points must be > 0")
    if float(meta.get("raw_score_scale_points", 0.0) or 0.0) <= 0.0:
        raise OfflineLabError("Label engine raw_score_scale_points must be > 0")
    if not 0.0 <= float(meta.get("min_raw_signal_strength", 0.0) or 0.0) <= 1.0:
        raise OfflineLabError("Label engine min_raw_signal_strength must be in [0, 1]")
    meta["candidate_mode"] = candidate_mode
    payload["meta_labeling"] = meta
    return payload


def load_config(path: Path | None = None) -> dict[str, Any]:
    ensure_label_engine_dirs()
    path = path or LABEL_ENGINE_CONFIG_PATH
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
