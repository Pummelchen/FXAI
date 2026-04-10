from __future__ import annotations

from copy import deepcopy
from typing import Any

from .common import OfflineLabError
from .execution_quality_contracts import (
    EXECUTION_QUALITY_CONFIG_PATH,
    EXECUTION_QUALITY_CONFIG_VERSION,
    EXECUTION_QUALITY_MEMORY_PATH,
    EXECUTION_QUALITY_MEMORY_VERSION,
    EXECUTION_QUALITY_METHODS,
    EXECUTION_QUALITY_RUNTIME_CONFIG_PATH,
    EXECUTION_QUALITY_RUNTIME_MEMORY_PATH,
    EXECUTION_QUALITY_TIER_KINDS,
    ensure_execution_quality_dirs,
    isoformat_utc,
    json_dump,
    json_load,
)

_REGIME_TIER_DEFAULTS: list[dict[str, Any]] = [
    {
        "kind": "REGIME",
        "symbol": "*",
        "session": "*",
        "regime": "TREND_PERSISTENT",
        "support": 240,
        "quality": 0.68,
        "spread_mult": 1.02,
        "slippage_mult": 0.96,
        "fill_quality_bias": 0.05,
        "latency_mult": 1.02,
        "fragility_mult": 0.92,
        "deviation_mult": 1.00,
    },
    {
        "kind": "REGIME",
        "symbol": "*",
        "session": "*",
        "regime": "RANGE_MEAN_REVERTING",
        "support": 210,
        "quality": 0.64,
        "spread_mult": 1.04,
        "slippage_mult": 1.00,
        "fill_quality_bias": 0.02,
        "latency_mult": 1.04,
        "fragility_mult": 0.98,
        "deviation_mult": 1.02,
    },
    {
        "kind": "REGIME",
        "symbol": "*",
        "session": "*",
        "regime": "BREAKOUT_TRANSITION",
        "support": 156,
        "quality": 0.58,
        "spread_mult": 1.12,
        "slippage_mult": 1.16,
        "fill_quality_bias": -0.04,
        "latency_mult": 1.12,
        "fragility_mult": 1.08,
        "deviation_mult": 1.08,
    },
    {
        "kind": "REGIME",
        "symbol": "*",
        "session": "*",
        "regime": "HIGH_VOL_EVENT",
        "support": 112,
        "quality": 0.46,
        "spread_mult": 1.28,
        "slippage_mult": 1.36,
        "fill_quality_bias": -0.12,
        "latency_mult": 1.24,
        "fragility_mult": 1.22,
        "deviation_mult": 1.14,
    },
    {
        "kind": "REGIME",
        "symbol": "*",
        "session": "*",
        "regime": "RISK_ON_OFF_MACRO",
        "support": 128,
        "quality": 0.54,
        "spread_mult": 1.14,
        "slippage_mult": 1.18,
        "fill_quality_bias": -0.06,
        "latency_mult": 1.10,
        "fragility_mult": 1.12,
        "deviation_mult": 1.08,
    },
    {
        "kind": "REGIME",
        "symbol": "*",
        "session": "*",
        "regime": "LIQUIDITY_STRESS",
        "support": 98,
        "quality": 0.42,
        "spread_mult": 1.34,
        "slippage_mult": 1.42,
        "fill_quality_bias": -0.15,
        "latency_mult": 1.18,
        "fragility_mult": 1.26,
        "deviation_mult": 1.10,
    },
    {
        "kind": "REGIME",
        "symbol": "*",
        "session": "*",
        "regime": "SESSION_FLOW",
        "support": 140,
        "quality": 0.56,
        "spread_mult": 1.10,
        "slippage_mult": 1.08,
        "fill_quality_bias": -0.02,
        "latency_mult": 1.10,
        "fragility_mult": 1.04,
        "deviation_mult": 1.05,
    },
]


def default_config() -> dict[str, Any]:
    return {
        "schema_version": EXECUTION_QUALITY_CONFIG_VERSION,
        "enabled": True,
        "block_on_unknown": True,
        "allow_block_state": True,
        "support_soft_floor": 64,
        "support_hard_floor": 16,
        "memory_stale_after_hours": 168,
        "bucket_hierarchy": [
            "PAIR_SESSION_REGIME",
            "PAIR_REGIME",
            "SESSION_REGIME",
            "REGIME",
            "GLOBAL",
        ],
        "state_thresholds": {
            "normal_min": 0.72,
            "caution_min": 0.54,
            "stressed_min": 0.36,
        },
        "lot_scales": {
            "normal": 1.00,
            "caution": 0.82,
            "stressed": 0.58,
            "blocked": 0.00,
        },
        "enter_prob_buffers": {
            "normal": 0.00,
            "caution": 0.04,
            "stressed": 0.08,
            "blocked": 1.00,
        },
        "forecast_caps": {
            "spread_expected_mult": 4.50,
            "expected_slippage_points": 18.0,
            "allowed_deviation_points_min": 2.0,
            "allowed_deviation_points_max": 25.0,
        },
        "weights": {
            "spread_zscore": 0.22,
            "news_risk": 0.18,
            "rates_risk": 0.10,
            "micro_liquidity": 0.18,
            "micro_hostile": 0.18,
            "volatility_burst": 0.14,
            "tick_rate_burst": 0.12,
            "session_thinness": 0.10,
            "broker_reject": 0.16,
            "broker_partial": 0.14,
            "broker_latency": 0.14,
            "broker_event_burst": 0.12,
            "stale_context": 0.10,
            "support_shortfall": 0.08,
        },
    }


def default_memory() -> dict[str, Any]:
    return {
        "schema_version": EXECUTION_QUALITY_MEMORY_VERSION,
        "generated_at": isoformat_utc(),
        "default_method": "SCORECARD_V1",
        "tiers": [
            {
                "kind": "GLOBAL",
                "symbol": "*",
                "session": "*",
                "regime": "*",
                "support": 512,
                "quality": 0.60,
                "spread_mult": 1.06,
                "slippage_mult": 1.08,
                "fill_quality_bias": 0.0,
                "latency_mult": 1.04,
                "fragility_mult": 1.00,
                "deviation_mult": 1.04,
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
        raise OfflineLabError(f"Execution-quality {key} must be a boolean")


def validate_config_payload(payload: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise OfflineLabError("Execution-quality config must be a JSON object")
    schema_version = int(payload.get("schema_version", 0) or 0)
    if schema_version != EXECUTION_QUALITY_CONFIG_VERSION:
        raise OfflineLabError(
            f"Execution-quality config schema_version must be {EXECUTION_QUALITY_CONFIG_VERSION}, got {schema_version}"
        )
    _require_bool(payload, "enabled")
    _require_bool(payload, "block_on_unknown")
    _require_bool(payload, "allow_block_state")

    support_soft_floor = int(payload.get("support_soft_floor", 0) or 0)
    support_hard_floor = int(payload.get("support_hard_floor", 0) or 0)
    if support_soft_floor < 16:
        raise OfflineLabError("Execution-quality support_soft_floor must be at least 16")
    if support_hard_floor < 4:
        raise OfflineLabError("Execution-quality support_hard_floor must be at least 4")
    if support_hard_floor > support_soft_floor:
        raise OfflineLabError("Execution-quality support_hard_floor cannot exceed support_soft_floor")

    thresholds = payload.get("state_thresholds", {})
    if not isinstance(thresholds, dict):
        raise OfflineLabError("Execution-quality state_thresholds must be an object")
    normal_min = float(thresholds.get("normal_min", 0.0) or 0.0)
    caution_min = float(thresholds.get("caution_min", 0.0) or 0.0)
    stressed_min = float(thresholds.get("stressed_min", 0.0) or 0.0)
    if not (0.0 <= stressed_min <= caution_min <= normal_min <= 1.0):
        raise OfflineLabError("Execution-quality state_thresholds must satisfy 0 <= stressed <= caution <= normal <= 1")

    hierarchy = payload.get("bucket_hierarchy", [])
    if not isinstance(hierarchy, list) or not hierarchy:
        raise OfflineLabError("Execution-quality bucket_hierarchy must be a non-empty list")
    seen: set[str] = set()
    for raw_kind in hierarchy:
        kind = str(raw_kind or "").upper()
        if kind not in EXECUTION_QUALITY_TIER_KINDS:
            raise OfflineLabError(f"Unsupported execution-quality tier kind: {raw_kind}")
        if kind in seen:
            raise OfflineLabError(f"Duplicate execution-quality tier kind in bucket_hierarchy: {kind}")
        seen.add(kind)

    for section_name in ("lot_scales", "enter_prob_buffers", "forecast_caps", "weights"):
        section = payload.get(section_name, {})
        if not isinstance(section, dict):
            raise OfflineLabError(f"Execution-quality {section_name} must be an object")

    return payload


def validate_memory_payload(payload: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise OfflineLabError("Execution-quality memory must be a JSON object")
    schema_version = int(payload.get("schema_version", 0) or 0)
    if schema_version != EXECUTION_QUALITY_MEMORY_VERSION:
        raise OfflineLabError(
            f"Execution-quality memory schema_version must be {EXECUTION_QUALITY_MEMORY_VERSION}, got {schema_version}"
        )
    method = str(payload.get("default_method", "") or "").upper()
    if method not in EXECUTION_QUALITY_METHODS:
        raise OfflineLabError(f"Unsupported execution-quality default_method: {method}")

    tiers = payload.get("tiers", [])
    if not isinstance(tiers, list) or not tiers:
        raise OfflineLabError("Execution-quality memory must define at least one tier")

    seen_keys: set[tuple[str, str, str, str]] = set()
    for index, tier in enumerate(tiers):
        if not isinstance(tier, dict):
            raise OfflineLabError(f"Execution-quality tier #{index} must be an object")
        kind = str(tier.get("kind", "") or "").upper()
        symbol = str(tier.get("symbol", "*") or "*").upper()
        session = str(tier.get("session", "*") or "*").upper()
        regime = str(tier.get("regime", "*") or "*").upper()
        if kind not in EXECUTION_QUALITY_TIER_KINDS:
            raise OfflineLabError(f"Unsupported execution-quality tier kind: {kind}")
        key = (kind, symbol, session, regime)
        if key in seen_keys:
            raise OfflineLabError(f"Duplicate execution-quality tier: {kind}|{symbol}|{session}|{regime}")
        seen_keys.add(key)
        if int(tier.get("support", 0) or 0) < 0:
            raise OfflineLabError("Execution-quality tier support must be non-negative")
        if not (0.0 <= float(tier.get("quality", 0.0) or 0.0) <= 1.0):
            raise OfflineLabError("Execution-quality tier quality must be within [0, 1]")
    return payload


def _write_runtime_config(config: dict[str, Any]) -> None:
    lines = [
        f"enabled\t{1 if bool(config.get('enabled')) else 0}",
        f"block_on_unknown\t{1 if bool(config.get('block_on_unknown')) else 0}",
        f"allow_block_state\t{1 if bool(config.get('allow_block_state')) else 0}",
        f"support_soft_floor\t{int(config.get('support_soft_floor', 64) or 64)}",
        f"support_hard_floor\t{int(config.get('support_hard_floor', 16) or 16)}",
        f"memory_stale_after_hours\t{int(config.get('memory_stale_after_hours', 168) or 168)}",
    ]
    thresholds = dict(config.get("state_thresholds", {}))
    lines.extend([
        f"threshold_normal_min\t{float(thresholds.get('normal_min', 0.72) or 0.72):.6f}",
        f"threshold_caution_min\t{float(thresholds.get('caution_min', 0.54) or 0.54):.6f}",
        f"threshold_stressed_min\t{float(thresholds.get('stressed_min', 0.36) or 0.36):.6f}",
    ])
    lot_scales = dict(config.get("lot_scales", {}))
    lines.extend([
        f"lot_scale_normal\t{float(lot_scales.get('normal', 1.0) or 1.0):.6f}",
        f"lot_scale_caution\t{float(lot_scales.get('caution', 0.82) or 0.82):.6f}",
        f"lot_scale_stressed\t{float(lot_scales.get('stressed', 0.58) or 0.58):.6f}",
        f"lot_scale_blocked\t{float(lot_scales.get('blocked', 0.0) or 0.0):.6f}",
    ])
    enter_prob = dict(config.get("enter_prob_buffers", {}))
    lines.extend([
        f"enter_prob_buffer_normal\t{float(enter_prob.get('normal', 0.0) or 0.0):.6f}",
        f"enter_prob_buffer_caution\t{float(enter_prob.get('caution', 0.04) or 0.04):.6f}",
        f"enter_prob_buffer_stressed\t{float(enter_prob.get('stressed', 0.08) or 0.08):.6f}",
        f"enter_prob_buffer_blocked\t{float(enter_prob.get('blocked', 1.0) or 1.0):.6f}",
    ])
    caps = dict(config.get("forecast_caps", {}))
    lines.extend([
        f"cap_spread_expected_mult\t{float(caps.get('spread_expected_mult', 4.5) or 4.5):.6f}",
        f"cap_expected_slippage_points\t{float(caps.get('expected_slippage_points', 18.0) or 18.0):.6f}",
        f"cap_allowed_deviation_points_min\t{float(caps.get('allowed_deviation_points_min', 2.0) or 2.0):.6f}",
        f"cap_allowed_deviation_points_max\t{float(caps.get('allowed_deviation_points_max', 25.0) or 25.0):.6f}",
    ])
    weights = dict(config.get("weights", {}))
    for key in sorted(weights):
        lines.append(f"weight_{key}\t{float(weights[key] or 0.0):.6f}")
    hierarchy = [str(item or "").upper() for item in config.get("bucket_hierarchy", [])]
    lines.append(f"bucket_count\t{len(hierarchy)}")
    for index, kind in enumerate(hierarchy):
        lines.append(f"bucket_{index}\t{kind}")
    EXECUTION_QUALITY_RUNTIME_CONFIG_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_runtime_memory(memory: dict[str, Any]) -> None:
    lines = [
        f"meta\tgenerated_at\t{memory.get('generated_at', isoformat_utc())}",
        f"meta\tdefault_method\t{str(memory.get('default_method', 'SCORECARD_V1')).upper()}",
    ]
    for tier in memory.get("tiers", []):
        lines.append(
            "\t".join(
                [
                    "tier",
                    str(tier.get("kind", "GLOBAL") or "GLOBAL").upper(),
                    str(tier.get("symbol", "*") or "*").upper(),
                    str(tier.get("session", "*") or "*").upper(),
                    str(tier.get("regime", "*") or "*").upper(),
                    str(int(tier.get("support", 0) or 0)),
                    f"{float(tier.get('quality', 0.0) or 0.0):.6f}",
                    f"{float(tier.get('spread_mult', 1.0) or 1.0):.6f}",
                    f"{float(tier.get('slippage_mult', 1.0) or 1.0):.6f}",
                    f"{float(tier.get('fill_quality_bias', 0.0) or 0.0):.6f}",
                    f"{float(tier.get('latency_mult', 1.0) or 1.0):.6f}",
                    f"{float(tier.get('fragility_mult', 1.0) or 1.0):.6f}",
                    f"{float(tier.get('deviation_mult', 1.0) or 1.0):.6f}",
                ]
            )
        )
    EXECUTION_QUALITY_RUNTIME_MEMORY_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_config() -> dict[str, Any]:
    ensure_execution_quality_dirs()
    merged = _merge_defaults(default_config(), json_load(EXECUTION_QUALITY_CONFIG_PATH))
    validate_config_payload(merged)
    json_dump(EXECUTION_QUALITY_CONFIG_PATH, merged)
    _write_runtime_config(merged)
    return merged


def load_memory() -> dict[str, Any]:
    ensure_execution_quality_dirs()
    merged = _merge_defaults(default_memory(), json_load(EXECUTION_QUALITY_MEMORY_PATH))
    validate_memory_payload(merged)
    json_dump(EXECUTION_QUALITY_MEMORY_PATH, merged)
    _write_runtime_memory(merged)
    return merged
