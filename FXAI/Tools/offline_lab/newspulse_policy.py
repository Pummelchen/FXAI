from __future__ import annotations

import json
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .common import OfflineLabError
from .newspulse_contracts import NEWSPULSE_POLICY_PATH, NEWSPULSE_POLICY_VERSION

_SUPPORTED_CURRENCIES = {"USD", "EUR", "GBP", "JPY", "CHF", "CAD", "AUD", "NZD", "SEK", "NOK"}


def default_policy() -> dict[str, Any]:
    return {
        "schema_version": NEWSPULSE_POLICY_VERSION,
        "watchlists": {
            "active": [
                "EURUSD",
                "GBPUSD",
                "USDJPY",
                "USDCHF",
                "USDCAD",
                "AUDUSD",
                "NZDUSD",
                "EURGBP",
                "EURJPY",
                "GBPJPY",
                "AUDJPY",
                "AUDCAD",
                "NZDCAD",
            ],
            "macro_sensitive": ["EURUSD", "GBPUSD", "USDJPY", "USDCHF", "USDCAD", "EURJPY", "GBPJPY"],
            "yen_crosses": ["USDJPY", "EURJPY", "GBPJPY", "AUDJPY", "NZDJPY", "CADJPY", "CHFJPY"],
            "commodity_cycle": ["AUDUSD", "NZDUSD", "USDCAD", "AUDCAD", "NZDCAD", "AUDNZD"],
        },
        "broker_symbol_map": {},
        "heuristic_symbol_scan": True,
        "default_pair_policy": {
            "caution_threshold_mult": 1.0,
            "block_threshold_mult": 1.0,
            "burst_risk_mult": 1.0,
            "pre_window_mult": 1.0,
            "post_window_mult": 1.0,
            "caution_lot_scale": 0.65,
            "enter_prob_buffer": 0.05,
            "event_block_eta_min": 8,
            "event_caution_eta_min": 28,
            "post_block_min": 6,
            "post_caution_min": 24,
        },
        "pair_groups": {
            "dollar_core": {
                "pairs": ["EURUSD", "GBPUSD", "USDJPY", "USDCHF", "USDCAD", "AUDUSD", "NZDUSD"],
                "caution_threshold_mult": 0.96,
                "block_threshold_mult": 0.94,
                "burst_risk_mult": 1.08,
                "pre_window_mult": 1.12,
                "post_window_mult": 1.10,
            },
            "yen_cross": {
                "pairs": ["USDJPY", "EURJPY", "GBPJPY", "AUDJPY", "NZDJPY", "CADJPY", "CHFJPY"],
                "caution_threshold_mult": 0.93,
                "block_threshold_mult": 0.90,
                "burst_risk_mult": 1.10,
                "pre_window_mult": 1.15,
                "post_window_mult": 1.15,
                "caution_lot_scale": 0.58,
                "enter_prob_buffer": 0.07,
            },
            "commodity_cycle": {
                "pairs": ["AUDUSD", "NZDUSD", "USDCAD", "AUDCAD", "NZDCAD", "AUDNZD", "CADJPY"],
                "caution_threshold_mult": 1.05,
                "block_threshold_mult": 1.03,
                "burst_risk_mult": 0.94,
                "pre_window_mult": 0.94,
                "post_window_mult": 0.98,
                "caution_lot_scale": 0.72,
                "enter_prob_buffer": 0.04,
            },
            "europe_rates": {
                "pairs": ["EURUSD", "EURGBP", "EURJPY", "GBPUSD", "GBPJPY", "EURCHF"],
                "caution_threshold_mult": 0.97,
                "block_threshold_mult": 0.95,
                "burst_risk_mult": 1.03,
                "pre_window_mult": 1.08,
                "post_window_mult": 1.05,
            },
        },
        "session_profiles": {
            "asia": {
                "hours_utc": [0, 6],
                "caution_threshold_mult": 1.03,
                "block_threshold_mult": 1.04,
                "burst_risk_mult": 0.96,
                "pre_window_mult": 1.06,
                "post_window_mult": 1.08,
                "caution_lot_scale": 0.70,
                "enter_prob_buffer": 0.04,
            },
            "london": {
                "hours_utc": [7, 11],
                "caution_threshold_mult": 0.97,
                "block_threshold_mult": 0.95,
                "burst_risk_mult": 1.06,
                "pre_window_mult": 1.05,
                "post_window_mult": 1.04,
                "caution_lot_scale": 0.64,
                "enter_prob_buffer": 0.05,
            },
            "newyork": {
                "hours_utc": [12, 16],
                "caution_threshold_mult": 0.95,
                "block_threshold_mult": 0.93,
                "burst_risk_mult": 1.08,
                "pre_window_mult": 1.10,
                "post_window_mult": 1.08,
                "caution_lot_scale": 0.62,
                "enter_prob_buffer": 0.06,
            },
            "overlap": {
                "hours_utc": [12, 14],
                "caution_threshold_mult": 0.92,
                "block_threshold_mult": 0.90,
                "burst_risk_mult": 1.12,
                "pre_window_mult": 1.15,
                "post_window_mult": 1.12,
                "caution_lot_scale": 0.56,
                "enter_prob_buffer": 0.07,
            },
            "rollover": {
                "hours_utc": [21, 23],
                "caution_threshold_mult": 0.98,
                "block_threshold_mult": 0.96,
                "burst_risk_mult": 1.02,
                "pre_window_mult": 1.04,
                "post_window_mult": 1.10,
                "caution_lot_scale": 0.60,
                "enter_prob_buffer": 0.05,
            },
        },
        "pair_overrides": {
            "EURUSD": {
                "pre_window_mult": 1.15,
                "post_window_mult": 1.12,
                "event_block_eta_min": 12,
                "event_caution_eta_min": 35,
            },
            "GBPJPY": {
                "caution_threshold_mult": 0.88,
                "block_threshold_mult": 0.86,
                "burst_risk_mult": 1.16,
                "caution_lot_scale": 0.52,
                "enter_prob_buffer": 0.08,
                "post_block_min": 9,
                "post_caution_min": 28,
            },
            "AUDCAD": {
                "caution_threshold_mult": 1.06,
                "block_threshold_mult": 1.04,
                "pre_window_mult": 0.92,
                "post_window_mult": 0.96,
                "caution_lot_scale": 0.74,
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


def ensure_default_policy_file() -> Path:
    NEWSPULSE_POLICY_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not NEWSPULSE_POLICY_PATH.exists():
        NEWSPULSE_POLICY_PATH.write_text(json.dumps(default_policy(), indent=2, sort_keys=True), encoding="utf-8")
    return NEWSPULSE_POLICY_PATH


def load_policy(path: Path = NEWSPULSE_POLICY_PATH) -> dict[str, Any]:
    ensure_default_policy_file()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise OfflineLabError(f"NewsPulse policy missing: {path}") from exc
    except json.JSONDecodeError as exc:
        raise OfflineLabError(f"NewsPulse policy is not valid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise OfflineLabError(f"NewsPulse policy must be a JSON object: {path}")
    merged = _merge_defaults(default_policy(), payload)
    validate_policy_payload(merged)
    return merged


def validate_policy_payload(payload: dict[str, Any]) -> dict[str, Any]:
    schema_version = int(payload.get("schema_version", 0) or 0)
    if schema_version != NEWSPULSE_POLICY_VERSION:
        raise OfflineLabError(
            f"NewsPulse policy schema mismatch: expected {NEWSPULSE_POLICY_VERSION}, got {schema_version}"
        )

    watchlists = payload.get("watchlists")
    if not isinstance(watchlists, dict) or not watchlists:
        raise OfflineLabError("NewsPulse policy watchlists are missing")
    for watchlist_name, entries in watchlists.items():
        if not isinstance(entries, list):
            raise OfflineLabError(f"NewsPulse watchlist must be an array: {watchlist_name}")
        for pair_id in entries:
            _validate_pair_id(str(pair_id), f"watchlists.{watchlist_name}")

    symbol_map = payload.get("broker_symbol_map")
    if not isinstance(symbol_map, dict):
        raise OfflineLabError("NewsPulse broker_symbol_map must be an object")
    for raw_symbol, pair_id in symbol_map.items():
        if not str(raw_symbol).strip():
            raise OfflineLabError("NewsPulse broker_symbol_map contains an empty symbol key")
        _validate_pair_id(str(pair_id), f"broker_symbol_map.{raw_symbol}")

    _validate_policy_table(payload.get("default_pair_policy"), "default_pair_policy")

    pair_groups = payload.get("pair_groups")
    if not isinstance(pair_groups, dict) or not pair_groups:
        raise OfflineLabError("NewsPulse pair_groups are missing")
    for group_name, spec in pair_groups.items():
        if not isinstance(spec, dict):
            raise OfflineLabError(f"NewsPulse pair group must be an object: {group_name}")
        pairs = spec.get("pairs")
        if not isinstance(pairs, list) or not pairs:
            raise OfflineLabError(f"NewsPulse pair group must list pairs: {group_name}")
        for pair_id in pairs:
            _validate_pair_id(str(pair_id), f"pair_groups.{group_name}")
        _validate_policy_table(spec, f"pair_groups.{group_name}", require_pairs=False)

    session_profiles = payload.get("session_profiles")
    if not isinstance(session_profiles, dict) or not session_profiles:
        raise OfflineLabError("NewsPulse session_profiles are missing")
    for session_name, spec in session_profiles.items():
        if not isinstance(spec, dict):
            raise OfflineLabError(f"NewsPulse session profile must be an object: {session_name}")
        hours_utc = spec.get("hours_utc")
        if not isinstance(hours_utc, list) or len(hours_utc) != 2:
            raise OfflineLabError(f"NewsPulse session profile needs [start,end] UTC hours: {session_name}")
        start_hour = int(hours_utc[0])
        end_hour = int(hours_utc[1])
        if start_hour < 0 or start_hour > 23 or end_hour < 0 or end_hour > 23:
            raise OfflineLabError(f"NewsPulse session hours must be 0-23: {session_name}")
        _validate_policy_table(spec, f"session_profiles.{session_name}", require_hours=False)

    pair_overrides = payload.get("pair_overrides")
    if not isinstance(pair_overrides, dict):
        raise OfflineLabError("NewsPulse pair_overrides must be an object")
    for pair_id, spec in pair_overrides.items():
        _validate_pair_id(str(pair_id), "pair_overrides")
        if not isinstance(spec, dict):
            raise OfflineLabError(f"NewsPulse pair override must be an object: {pair_id}")
        _validate_policy_table(spec, f"pair_overrides.{pair_id}")

    return payload


def _validate_policy_table(spec: Any,
                           label: str,
                           *,
                           require_pairs: bool = True,
                           require_hours: bool = True) -> None:
    if not isinstance(spec, dict):
        raise OfflineLabError(f"NewsPulse policy table must be an object: {label}")
    numeric_keys = (
        "caution_threshold_mult",
        "block_threshold_mult",
        "burst_risk_mult",
        "pre_window_mult",
        "post_window_mult",
        "caution_lot_scale",
        "enter_prob_buffer",
        "event_block_eta_min",
        "event_caution_eta_min",
        "post_block_min",
        "post_caution_min",
    )
    for key in numeric_keys:
        if key not in spec:
            continue
        try:
            float(spec[key])
        except (TypeError, ValueError) as exc:
            raise OfflineLabError(f"NewsPulse policy value must be numeric: {label}.{key}") from exc


def _validate_pair_id(value: str, label: str) -> None:
    pair_id = str(value or "").strip().upper()
    if len(pair_id) != 6:
        raise OfflineLabError(f"NewsPulse pair id must be six characters in {label}: {value}")
    base = pair_id[:3]
    quote = pair_id[3:]
    if base not in _SUPPORTED_CURRENCIES or quote not in _SUPPORTED_CURRENCIES or base == quote:
        raise OfflineLabError(f"NewsPulse pair id is not supported in {label}: {value}")


def active_pairs(policy: dict[str, Any]) -> list[str]:
    entries = policy.get("watchlists", {}).get("active", [])
    if not isinstance(entries, list):
        return []
    out = [str(pair_id).strip().upper() for pair_id in entries if str(pair_id).strip()]
    return sorted(dict.fromkeys(out))


def canonical_pair_for_symbol(symbol: str,
                              policy: dict[str, Any],
                              fallback_pairs: list[str] | None = None) -> str:
    raw = str(symbol or "").strip()
    if not raw:
        return ""
    upper = raw.upper()
    explicit = policy.get("broker_symbol_map", {})
    if isinstance(explicit, dict):
        mapped = str(explicit.get(raw, "") or explicit.get(upper, "")).strip().upper()
        if mapped:
            return mapped

    candidates = set(pair_id.upper() for pair_id in (fallback_pairs or active_pairs(policy)) if str(pair_id).strip())
    alpha_only = "".join(ch for ch in upper if "A" <= ch <= "Z")
    for start in range(0, max(len(alpha_only) - 5, 0)):
        candidate = alpha_only[start:start + 6]
        if len(candidate) != 6:
            continue
        base = candidate[:3]
        quote = candidate[3:]
        if base == quote or base not in _SUPPORTED_CURRENCIES or quote not in _SUPPORTED_CURRENCIES:
            continue
        if not candidates or candidate in candidates:
            return candidate
    return ""


def broker_symbols_for_pair(pair_id: str, policy: dict[str, Any]) -> list[str]:
    normalized = str(pair_id or "").strip().upper()
    out: list[str] = []
    explicit = policy.get("broker_symbol_map", {})
    if not isinstance(explicit, dict):
        return out
    for raw_symbol, mapped in explicit.items():
        if str(mapped or "").strip().upper() == normalized:
            out.append(str(raw_symbol))
    return sorted(dict.fromkeys(out))


def watchlist_tags_for_pair(pair_id: str, policy: dict[str, Any]) -> list[str]:
    normalized = str(pair_id or "").strip().upper()
    tags: list[str] = []
    watchlists = policy.get("watchlists", {})
    if isinstance(watchlists, dict):
        for tag, entries in watchlists.items():
            if isinstance(entries, list) and normalized in {str(item).strip().upper() for item in entries}:
                tags.append(str(tag))
    for group_name, spec in (policy.get("pair_groups", {}) or {}).items():
        if not isinstance(spec, dict):
            continue
        pairs = spec.get("pairs", [])
        if isinstance(pairs, list) and normalized in {str(item).strip().upper() for item in pairs}:
            tags.append(str(group_name))
    return sorted(dict.fromkeys(tags))


def session_profile_name(now_dt: datetime, policy: dict[str, Any]) -> str:
    hour = now_dt.astimezone(timezone.utc).hour
    profiles = policy.get("session_profiles", {})
    if not isinstance(profiles, dict):
        return "default"
    matching: list[tuple[int, str]] = []
    for name, spec in profiles.items():
        if not isinstance(spec, dict):
            continue
        hours = spec.get("hours_utc")
        if not isinstance(hours, list) or len(hours) != 2:
            continue
        start_hour = int(hours[0])
        end_hour = int(hours[1])
        if start_hour <= end_hour:
            in_range = start_hour <= hour <= end_hour
            span = end_hour - start_hour
        else:
            in_range = hour >= start_hour or hour <= end_hour
            span = (24 - start_hour) + end_hour
        if in_range:
            matching.append((span, str(name)))
    if not matching:
        return "default"
    matching.sort()
    return matching[0][1]


def pair_calibration(pair_id: str, policy: dict[str, Any], now_dt: datetime) -> dict[str, Any]:
    result = deepcopy(policy.get("default_pair_policy", {}))
    session_name = session_profile_name(now_dt, policy)
    calibration_parts: list[str] = [session_name]

    session_spec = policy.get("session_profiles", {}).get(session_name, {})
    if isinstance(session_spec, dict):
        result.update({key: deepcopy(value) for key, value in session_spec.items() if key != "hours_utc"})

    tags = watchlist_tags_for_pair(pair_id, policy)
    for group_name, spec in (policy.get("pair_groups", {}) or {}).items():
        if group_name not in tags or not isinstance(spec, dict):
            continue
        calibration_parts.append(group_name)
        for key, value in spec.items():
            if key == "pairs":
                continue
            result[key] = deepcopy(value)

    override = policy.get("pair_overrides", {}).get(str(pair_id).upper(), {})
    if isinstance(override, dict) and override:
        calibration_parts.append("pair")
        for key, value in override.items():
            result[key] = deepcopy(value)

    result["session_profile"] = session_name
    result["watchlist_tags"] = tags
    result["calibration_profile"] = ":".join(dict.fromkeys(calibration_parts))
    return result
