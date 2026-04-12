from __future__ import annotations

from copy import deepcopy
from typing import Any

from .common import DEFAULT_DB, OfflineLabError, close_db, connect_db
from .cross_asset_contracts import (
    COMMON_CROSS_ASSET_CONFIG,
    CROSS_ASSET_CONFIG_PATH,
    CROSS_ASSET_CONFIG_VERSION,
    ensure_cross_asset_dirs,
    json_dump,
    json_load,
)
from .market_universe import (
    default_market_universe_config,
    load_market_universe_config,
    validate_market_universe_config,
)


def _safe_market_universe() -> dict[str, Any]:
    try:
        conn = connect_db(DEFAULT_DB)
        try:
            return validate_market_universe_config(load_market_universe_config(conn))
        finally:
            close_db(conn)
    except Exception:
        return default_market_universe_config()


def _indicator_symbols(payload: dict[str, Any]) -> list[str]:
    out: list[str] = []
    for record in list(payload.get("symbol_records", [])):
        if not isinstance(record, dict):
            continue
        if str(record.get("role", "") or "").strip().lower() != "indicator_only":
            continue
        symbol = str(record.get("symbol", "") or "").strip()
        if symbol:
            out.append(symbol)
    return out


def _tradable_pairs(payload: dict[str, Any]) -> list[str]:
    out: list[str] = []
    for record in list(payload.get("symbol_records", [])):
        if not isinstance(record, dict):
            continue
        if str(record.get("role", "") or "").strip().lower() != "tradable":
            continue
        symbol = str(record.get("symbol", "") or "").strip().upper()
        if len(symbol) == 6:
            out.append(symbol)
    return out


def default_config() -> dict[str, Any]:
    universe = _safe_market_universe()
    return {
        "schema_version": CROSS_ASSET_CONFIG_VERSION,
        "enabled": True,
        "poll_interval_sec": 120,
        "snapshot_stale_after_sec": 900,
        "probe_stale_after_sec": 300,
        "rates_stale_after_sec": 1200,
        "history_points": 192,
        "max_recent_transitions": 24,
        "probe_required_for_live_gates": False,
        "critical_sources": ["rates", "context_service"],
        "use_market_universe_indicator_symbols": True,
        "proxy_candidates": {
            "equities": ["US500", "USTEC", "US30", "DE40", "STOXX50", "JP225", "HK50", "CHINA50", "AUS200", "CA60"],
            "oil": ["XBRUSD", "XTIUSD", "USO.NYSE"],
            "gold": ["XAUUSD", "IAU.NYSE", "XAGUSD"],
            "metals": ["XPDUSD", "XPTUSD", "XAGUSD"],
            "volatility": ["VIX", "VIXUSD", "VOLX", "BTCUSD"],
            "dollar_liquidity": ["DXY", "USDX", "US02Y", "US10Y", "TNX", "USB10", "BUND", "GILT", "JGB"],
        },
        "feature_weights": {
            "rates_repricing": {
                "front_end_divergence": 0.40,
                "policy_repricing": 0.28,
                "policy_uncertainty": 0.20,
                "curve_divergence": 0.12,
            },
            "risk_off": {
                "equity_risk": 0.46,
                "volatility": 0.22,
                "gold": 0.14,
                "usd_liquidity": 0.18,
            },
            "commodity_shock": {
                "oil": 0.58,
                "gold": 0.26,
                "metals": 0.16,
            },
            "volatility_shock": {
                "volatility": 0.68,
                "equity_risk": 0.18,
                "cross_asset_dislocation": 0.14,
            },
            "usd_liquidity_stress": {
                "usd_liquidity": 0.46,
                "risk_off": 0.18,
                "rates_uncertainty": 0.18,
                "volatility": 0.18,
            },
            "cross_asset_dislocation": {
                "dispersion": 0.40,
                "rates_repricing": 0.20,
                "commodity_shock": 0.16,
                "volatility_shock": 0.24,
            },
        },
        "label_thresholds": {
            "normal_max": 0.42,
            "mixed_delta_max": 0.08,
            "risk_off_min": 0.58,
            "liquidity_caution_min": 0.48,
            "liquidity_stressed_min": 0.66,
            "pair_caution_min": 0.56,
            "pair_block_min": 0.82,
        },
        "pair_component_weights": {
            "rates": 0.28,
            "risk_off": 0.22,
            "commodity": 0.16,
            "volatility": 0.18,
            "liquidity": 0.16,
        },
        "currency_profiles": {
            "USD": {"safe_haven": 0.54, "rates": 1.00, "commodity": 0.06, "liquidity": 1.00, "risk": 0.42},
            "EUR": {"safe_haven": 0.18, "rates": 0.88, "commodity": 0.10, "liquidity": 0.52, "risk": 0.36},
            "GBP": {"safe_haven": 0.16, "rates": 0.84, "commodity": 0.08, "liquidity": 0.48, "risk": 0.38},
            "JPY": {"safe_haven": 1.00, "rates": 0.54, "commodity": 0.02, "liquidity": 0.72, "risk": 0.86},
            "CHF": {"safe_haven": 0.92, "rates": 0.52, "commodity": 0.02, "liquidity": 0.68, "risk": 0.80},
            "CAD": {"safe_haven": 0.12, "rates": 0.70, "commodity": 0.98, "liquidity": 0.50, "risk": 0.58},
            "AUD": {"safe_haven": 0.10, "rates": 0.66, "commodity": 0.84, "liquidity": 0.42, "risk": 0.72},
            "NZD": {"safe_haven": 0.08, "rates": 0.62, "commodity": 0.72, "liquidity": 0.40, "risk": 0.68},
            "NOK": {"safe_haven": 0.06, "rates": 0.58, "commodity": 0.92, "liquidity": 0.36, "risk": 0.62},
            "SEK": {"safe_haven": 0.08, "rates": 0.56, "commodity": 0.24, "liquidity": 0.38, "risk": 0.56},
        },
        "market_universe": {
            "tradable_pairs": _tradable_pairs(universe),
            "indicator_symbols": _indicator_symbols(universe),
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


def resolve_probe_symbols(payload: dict[str, Any]) -> list[str]:
    normalized = validate_config_payload(payload)
    symbols: list[str] = []
    seen: set[str] = set()
    for raw_symbol in normalized["market_universe"]["indicator_symbols"]:
        symbol = str(raw_symbol or "").strip()
        symbol_key = symbol.upper()
        if symbol and symbol_key not in seen:
            seen.add(symbol_key)
            symbols.append(symbol)
    for symbol_list in dict(normalized.get("proxy_candidates", {})).values():
        for raw_symbol in list(symbol_list or []):
            symbol = str(raw_symbol or "").strip()
            symbol_key = symbol.upper()
            if symbol and symbol_key not in seen:
                seen.add(symbol_key)
                symbols.append(symbol)
    return symbols


def validate_config_payload(payload: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise OfflineLabError("Cross-asset config must be a JSON object")
    merged = _merge_defaults(default_config(), payload)
    schema_version = int(merged.get("schema_version", 0) or 0)
    if schema_version != CROSS_ASSET_CONFIG_VERSION:
        raise OfflineLabError(
            f"Cross-asset config schema_version must be {CROSS_ASSET_CONFIG_VERSION}, got {schema_version}"
        )
    if not isinstance(merged.get("enabled"), bool):
        raise OfflineLabError("Cross-asset enabled must be a boolean")
    if int(merged.get("poll_interval_sec", 0) or 0) < 30:
        raise OfflineLabError("Cross-asset poll_interval_sec must be at least 30")
    if int(merged.get("snapshot_stale_after_sec", 0) or 0) < 60:
        raise OfflineLabError("Cross-asset snapshot_stale_after_sec must be at least 60")
    if int(merged.get("probe_stale_after_sec", 0) or 0) < 30:
        raise OfflineLabError("Cross-asset probe_stale_after_sec must be at least 30")
    if int(merged.get("rates_stale_after_sec", 0) or 0) < 60:
        raise OfflineLabError("Cross-asset rates_stale_after_sec must be at least 60")
    if int(merged.get("history_points", 0) or 0) < 32:
        raise OfflineLabError("Cross-asset history_points must be at least 32")
    if not isinstance(merged.get("probe_required_for_live_gates"), bool):
        raise OfflineLabError("Cross-asset probe_required_for_live_gates must be a boolean")

    critical_sources = [str(item or "").strip() for item in list(merged.get("critical_sources", []))]
    if not critical_sources:
        raise OfflineLabError("Cross-asset critical_sources must not be empty")
    unsupported = sorted(
        source
        for source in critical_sources
        if source not in {"rates", "context_service", "equities", "commodities", "volatility", "liquidity"}
    )
    if unsupported:
        raise OfflineLabError(f"Unsupported cross-asset critical source(s): {', '.join(unsupported)}")

    proxy_candidates = merged.get("proxy_candidates", {})
    if not isinstance(proxy_candidates, dict) or not proxy_candidates:
        raise OfflineLabError("Cross-asset proxy_candidates must be a non-empty object")
    for group, values in proxy_candidates.items():
        if not isinstance(values, list) or not values:
            raise OfflineLabError(f"Cross-asset proxy_candidates.{group} must be a non-empty array")

    thresholds = merged.get("label_thresholds", {})
    if not isinstance(thresholds, dict):
        raise OfflineLabError("Cross-asset label_thresholds must be an object")
    normal_max = float(thresholds.get("normal_max", 0.0) or 0.0)
    liquidity_caution_min = float(thresholds.get("liquidity_caution_min", 0.0) or 0.0)
    liquidity_stressed_min = float(thresholds.get("liquidity_stressed_min", 0.0) or 0.0)
    pair_caution_min = float(thresholds.get("pair_caution_min", 0.0) or 0.0)
    pair_block_min = float(thresholds.get("pair_block_min", 0.0) or 0.0)
    if not (0.0 <= normal_max < liquidity_caution_min < liquidity_stressed_min <= 1.0):
        raise OfflineLabError("Cross-asset label_thresholds must satisfy normal_max < liquidity_caution_min < liquidity_stressed_min within [0,1]")
    if not (0.0 < pair_caution_min < pair_block_min <= 1.0):
        raise OfflineLabError("Cross-asset pair thresholds must satisfy 0 < pair_caution_min < pair_block_min <= 1")

    universe = merged.get("market_universe", {})
    if not isinstance(universe, dict):
        raise OfflineLabError("Cross-asset market_universe must be an object")
    tradable_pairs = [str(item or "").strip().upper() for item in list(universe.get("tradable_pairs", []))]
    if not tradable_pairs:
        raise OfflineLabError("Cross-asset market_universe.tradable_pairs must not be empty")
    if any(len(symbol) != 6 for symbol in tradable_pairs):
        raise OfflineLabError("Cross-asset tradable_pairs must contain 6-character FX symbols")
    indicator_symbols: list[str] = []
    seen_indicator_symbols: set[str] = set()
    for raw_symbol in list(universe.get("indicator_symbols", [])):
        symbol = str(raw_symbol or "").strip()
        if not symbol:
            continue
        symbol_key = symbol.upper()
        if symbol_key in seen_indicator_symbols:
            continue
        seen_indicator_symbols.add(symbol_key)
        indicator_symbols.append(symbol)
    merged["market_universe"] = {
        "tradable_pairs": sorted(dict.fromkeys(tradable_pairs)),
        "indicator_symbols": indicator_symbols,
    }

    profiles = merged.get("currency_profiles", {})
    if not isinstance(profiles, dict) or not profiles:
        raise OfflineLabError("Cross-asset currency_profiles must be a non-empty object")
    required_profile_keys = {"safe_haven", "rates", "commodity", "liquidity", "risk"}
    for currency, spec in profiles.items():
        if not isinstance(spec, dict):
            raise OfflineLabError(f"Cross-asset currency_profiles.{currency} must be an object")
        missing = required_profile_keys.difference(spec)
        if missing:
            raise OfflineLabError(f"Cross-asset currency_profiles.{currency} is missing: {', '.join(sorted(missing))}")

    return merged


def export_runtime_probe_config(payload: dict[str, Any]) -> Any:
    normalized = validate_config_payload(payload)
    ensure_cross_asset_dirs()
    symbols = resolve_probe_symbols(normalized)
    lines = [
        f"schema_version\t{CROSS_ASSET_CONFIG_VERSION}",
        f"enabled\t{1 if normalized.get('enabled', True) else 0}",
        f"poll_interval_ms\t{int(normalized['poll_interval_sec']) * 1000}",
        f"snapshot_stale_after_sec\t{int(normalized['probe_stale_after_sec'])}",
        f"symbol_count\t{len(symbols)}",
    ]
    for symbol in symbols:
        lines.append(f"symbol\t{symbol}")
    COMMON_CROSS_ASSET_CONFIG.parent.mkdir(parents=True, exist_ok=True)
    COMMON_CROSS_ASSET_CONFIG.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return COMMON_CROSS_ASSET_CONFIG


def ensure_default_files() -> dict[str, Any]:
    ensure_cross_asset_dirs()
    if not CROSS_ASSET_CONFIG_PATH.exists():
        json_dump(CROSS_ASSET_CONFIG_PATH, default_config())
    payload = validate_config_payload(json_load(CROSS_ASSET_CONFIG_PATH))
    json_dump(CROSS_ASSET_CONFIG_PATH, payload)
    export_runtime_probe_config(payload)
    return payload


def load_config() -> dict[str, Any]:
    return ensure_default_files()
