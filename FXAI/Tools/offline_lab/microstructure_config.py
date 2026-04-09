from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .common import OfflineLabError
from .market_universe import default_market_universe_config
from .microstructure_contracts import (
    COMMON_MICROSTRUCTURE_CONFIG,
    MICROSTRUCTURE_CONFIG_PATH,
    MICROSTRUCTURE_CONFIG_VERSION,
    ensure_microstructure_dirs,
    json_dump,
    json_load,
)


def _default_pairs() -> list[str]:
    payload = default_market_universe_config()
    records = payload.get("symbol_records", [])
    return [
        str(record.get("symbol", "")).strip()
        for record in records
        if isinstance(record, dict) and str(record.get("role", "")).strip().lower() == "tradable"
    ]


def default_config() -> dict[str, Any]:
    return {
        "schema_version": MICROSTRUCTURE_CONFIG_VERSION,
        "enabled": True,
        "collector_mode": "mt5_service",
        "poll_interval_ms": 5000,
        "symbol_refresh_sec": 300,
        "snapshot_stale_after_sec": 45,
        "max_history_window_sec": 960,
        "windows_sec": [10, 30, 60, 300, 900],
        "session_model": {
            "timezone": "UTC",
            "handoff_minutes": 20,
            "sessions": {
                "ASIA": {"start_hour": 0, "end_hour": 7},
                "LONDON": {"start_hour": 7, "end_hour": 12},
                "LONDON_NEWYORK_OVERLAP": {"start_hour": 12, "end_hour": 16},
                "NEWYORK": {"start_hour": 16, "end_hour": 21},
                "ROLLOVER": {"start_hour": 21, "end_hour": 24},
            },
        },
        "thresholds": {
            "wide_spread_zscore": 1.45,
            "wide_spread_absolute_points_floor": 2.0,
            "spread_instability_caution": 0.56,
            "spread_instability_block": 0.82,
            "tick_burst_ratio_caution": 1.35,
            "tick_burst_ratio_block": 1.85,
            "vol_burst_ratio_caution": 1.40,
            "vol_burst_ratio_block": 1.90,
            "shock_move_points_factor": 1.80,
            "stop_run_reversal_fraction": 0.45,
            "stop_run_rejection_score_flag": 0.58,
            "liquidity_stress_caution": 0.58,
            "liquidity_stress_block": 0.86,
            "hostile_execution_caution": 0.56,
            "hostile_execution_block": 0.84,
            "clean_trend_efficiency_floor": 0.62,
            "clean_trend_imbalance_floor": 0.22,
        },
        "runtime_policy": {
            "block_on_unknown": False,
            "caution_lot_scale": 0.72,
            "caution_enter_prob_buffer": 0.04,
        },
        "symbol_universe": {
            "trading_scope": "FX_ONLY",
            "canonical_pairs": _default_pairs(),
        },
    }


def validate_config_payload(payload: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise OfflineLabError("microstructure config must be a JSON object")
    schema_version = int(payload.get("schema_version", 0) or 0)
    if schema_version != MICROSTRUCTURE_CONFIG_VERSION:
        raise OfflineLabError(f"microstructure schema_version must be {MICROSTRUCTURE_CONFIG_VERSION}")

    poll_interval_ms = int(payload.get("poll_interval_ms", 0) or 0)
    symbol_refresh_sec = int(payload.get("symbol_refresh_sec", 0) or 0)
    stale_after_sec = int(payload.get("snapshot_stale_after_sec", 0) or 0)
    max_history_window_sec = int(payload.get("max_history_window_sec", 0) or 0)
    if poll_interval_ms < 1000:
        raise OfflineLabError("microstructure poll_interval_ms must be at least 1000")
    if symbol_refresh_sec < 30:
        raise OfflineLabError("microstructure symbol_refresh_sec must be at least 30")
    if stale_after_sec < 10:
        raise OfflineLabError("microstructure snapshot_stale_after_sec must be at least 10")
    if max_history_window_sec < 900:
        raise OfflineLabError("microstructure max_history_window_sec must be at least 900")

    windows = payload.get("windows_sec")
    if not isinstance(windows, list) or not windows:
        raise OfflineLabError("microstructure windows_sec must be a non-empty array")
    normalized_windows: list[int] = []
    for index, value in enumerate(windows):
        window = int(value or 0)
        if window <= 0:
            raise OfflineLabError(f"microstructure windows_sec[{index}] must be positive")
        normalized_windows.append(window)
    normalized_windows = sorted(set(normalized_windows))
    if normalized_windows[-1] > max_history_window_sec:
        raise OfflineLabError("microstructure largest window must be <= max_history_window_sec")
    required_windows = {10, 30, 60, 300, 900}
    missing_windows = sorted(required_windows.difference(normalized_windows))
    if missing_windows:
        raise OfflineLabError(
            "microstructure windows_sec must include the required phase-1 windows "
            f"{sorted(required_windows)}; missing {missing_windows}"
        )

    thresholds = payload.get("thresholds")
    if not isinstance(thresholds, dict) or not thresholds:
        raise OfflineLabError("microstructure thresholds must be a JSON object")
    runtime_policy = payload.get("runtime_policy")
    if not isinstance(runtime_policy, dict) or not runtime_policy:
        raise OfflineLabError("microstructure runtime_policy must be a JSON object")

    symbol_universe = payload.get("symbol_universe")
    if not isinstance(symbol_universe, dict):
        raise OfflineLabError("microstructure symbol_universe must be a JSON object")
    canonical_pairs = symbol_universe.get("canonical_pairs")
    if not isinstance(canonical_pairs, list) or not canonical_pairs:
        raise OfflineLabError("microstructure symbol_universe.canonical_pairs must be a non-empty array")
    normalized_pairs: list[str] = []
    seen_pairs: set[str] = set()
    for index, value in enumerate(canonical_pairs):
        pair = str(value or "").strip().upper()
        if len(pair) != 6:
            raise OfflineLabError(f"microstructure canonical pair at index {index} must be 6 uppercase letters")
        if pair in seen_pairs:
            raise OfflineLabError(f"duplicate microstructure canonical pair: {pair}")
        seen_pairs.add(pair)
        normalized_pairs.append(pair)

    payload = dict(payload)
    payload["windows_sec"] = normalized_windows
    payload["symbol_universe"] = {
        "trading_scope": "FX_ONLY",
        "canonical_pairs": normalized_pairs,
    }
    return payload


def export_runtime_service_config(payload: dict[str, Any]) -> Path:
    normalized = validate_config_payload(payload)
    ensure_microstructure_dirs()
    lines = [
        f"schema_version\t{MICROSTRUCTURE_CONFIG_VERSION}",
        f"enabled\t{1 if normalized.get('enabled', True) else 0}",
        f"poll_interval_ms\t{int(normalized['poll_interval_ms'])}",
        f"symbol_refresh_sec\t{int(normalized['symbol_refresh_sec'])}",
        f"snapshot_stale_after_sec\t{int(normalized['snapshot_stale_after_sec'])}",
        f"max_history_window_sec\t{int(normalized['max_history_window_sec'])}",
        f"windows_csv\t{','.join(str(value) for value in normalized['windows_sec'])}",
    ]
    for key, value in sorted(dict(normalized.get("thresholds", {})).items()):
        lines.append(f"threshold_{key}\t{value}")
    for key, value in sorted(dict(normalized.get("runtime_policy", {})).items()):
        if isinstance(value, bool):
            value = 1 if value else 0
        lines.append(f"runtime_{key}\t{value}")
    for key, value in sorted(dict(normalized.get("session_model", {})).items()):
        if key == "sessions":
            continue
        lines.append(f"session_{key}\t{value}")
    sessions = dict(normalized.get("session_model", {}).get("sessions", {}))
    for label, spec in sorted(sessions.items()):
        if not isinstance(spec, dict):
            continue
        lines.append(f"session_window\t{label},{int(spec.get('start_hour', 0) or 0)},{int(spec.get('end_hour', 0) or 0)}")
    for pair in normalized["symbol_universe"]["canonical_pairs"]:
        lines.append(f"pair\t{pair}")
    COMMON_MICROSTRUCTURE_CONFIG.parent.mkdir(parents=True, exist_ok=True)
    COMMON_MICROSTRUCTURE_CONFIG.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return COMMON_MICROSTRUCTURE_CONFIG


def ensure_default_files() -> dict[str, Any]:
    ensure_microstructure_dirs()
    if not MICROSTRUCTURE_CONFIG_PATH.exists():
        json_dump(MICROSTRUCTURE_CONFIG_PATH, default_config())
    payload = validate_config_payload(json_load(MICROSTRUCTURE_CONFIG_PATH))
    json_dump(MICROSTRUCTURE_CONFIG_PATH, payload)
    export_runtime_service_config(payload)
    return payload


def load_config() -> dict[str, Any]:
    return ensure_default_files()
