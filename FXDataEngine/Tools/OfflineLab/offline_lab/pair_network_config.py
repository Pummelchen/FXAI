from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from .common import DEFAULT_DB, OfflineLabError, close_db, connect_db
from .market_universe import (
    default_market_universe_config,
    load_market_universe_config,
    validate_market_universe_config,
)
from .pair_network_contracts import (
    PAIR_NETWORK_CONFIG_PATH,
    PAIR_NETWORK_CONFIG_VERSION,
    PAIR_NETWORK_RUNTIME_CONFIG_PATH,
    PAIR_NETWORK_RUNTIME_STATUS_PATH,
    ensure_pair_network_dirs,
    json_dump,
    json_load,
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


def _tradable_pairs(payload: dict[str, Any]) -> list[str]:
    pairs: list[str] = []
    for record in list(payload.get("symbol_records", [])):
        if not isinstance(record, dict):
            continue
        if str(record.get("role", "") or "").strip().lower() != "tradable":
            continue
        symbol = str(record.get("symbol", "") or "").strip().upper()
        if len(symbol) == 6:
            pairs.append(symbol)
    return sorted(set(pairs))


def _supported_currencies(pairs: list[str]) -> list[str]:
    currencies = {pair[:3] for pair in pairs if len(pair) == 6} | {pair[3:] for pair in pairs if len(pair) == 6}
    return sorted(code for code in currencies if len(code) == 3)


def default_currency_profiles(currencies: list[str]) -> dict[str, dict[str, float]]:
    defaults: dict[str, dict[str, float]] = {
        "USD": {
            "usd_bloc": 1.00,
            "eur_rates": 0.16,
            "safe_haven": 0.72,
            "commodity_fx": -0.18,
            "risk_on": -0.42,
            "liquidity_stress": 1.00,
            "macro_shock": 0.78,
        },
        "EUR": {
            "usd_bloc": -0.20,
            "eur_rates": 1.00,
            "safe_haven": -0.06,
            "commodity_fx": 0.06,
            "risk_on": 0.22,
            "liquidity_stress": -0.18,
            "macro_shock": 0.52,
        },
        "GBP": {
            "usd_bloc": -0.12,
            "eur_rates": 0.82,
            "safe_haven": -0.10,
            "commodity_fx": 0.08,
            "risk_on": 0.26,
            "liquidity_stress": -0.10,
            "macro_shock": 0.48,
        },
        "JPY": {
            "usd_bloc": -0.08,
            "eur_rates": 0.16,
            "safe_haven": 1.00,
            "commodity_fx": -0.22,
            "risk_on": -0.88,
            "liquidity_stress": 0.56,
            "macro_shock": 0.70,
        },
        "CHF": {
            "usd_bloc": -0.04,
            "eur_rates": 0.28,
            "safe_haven": 0.92,
            "commodity_fx": -0.18,
            "risk_on": -0.76,
            "liquidity_stress": 0.44,
            "macro_shock": 0.64,
        },
        "AUD": {
            "usd_bloc": -0.18,
            "eur_rates": 0.18,
            "safe_haven": -0.36,
            "commodity_fx": 1.00,
            "risk_on": 0.96,
            "liquidity_stress": -0.28,
            "macro_shock": 0.58,
        },
        "CAD": {
            "usd_bloc": -0.08,
            "eur_rates": 0.18,
            "safe_haven": -0.30,
            "commodity_fx": 0.92,
            "risk_on": 0.68,
            "liquidity_stress": -0.12,
            "macro_shock": 0.54,
        },
        "NZD": {
            "usd_bloc": -0.18,
            "eur_rates": 0.12,
            "safe_haven": -0.32,
            "commodity_fx": 0.88,
            "risk_on": 0.90,
            "liquidity_stress": -0.24,
            "macro_shock": 0.52,
        },
        "NOK": {
            "usd_bloc": -0.10,
            "eur_rates": 0.34,
            "safe_haven": -0.26,
            "commodity_fx": 0.94,
            "risk_on": 0.62,
            "liquidity_stress": -0.18,
            "macro_shock": 0.56,
        },
        "SEK": {
            "usd_bloc": -0.10,
            "eur_rates": 0.54,
            "safe_haven": -0.22,
            "commodity_fx": 0.28,
            "risk_on": 0.54,
            "liquidity_stress": -0.16,
            "macro_shock": 0.46,
        },
        "SGD": {
            "usd_bloc": -0.08,
            "eur_rates": 0.16,
            "safe_haven": 0.10,
            "commodity_fx": 0.22,
            "risk_on": 0.38,
            "liquidity_stress": 0.12,
            "macro_shock": 0.36,
        },
        "CNH": {
            "usd_bloc": -0.12,
            "eur_rates": 0.06,
            "safe_haven": -0.12,
            "commodity_fx": 0.20,
            "risk_on": 0.42,
            "liquidity_stress": 0.08,
            "macro_shock": 0.40,
        },
        "MXN": {
            "usd_bloc": -0.10,
            "eur_rates": 0.08,
            "safe_haven": -0.30,
            "commodity_fx": 0.34,
            "risk_on": 0.72,
            "liquidity_stress": -0.18,
            "macro_shock": 0.52,
        },
        "ZAR": {
            "usd_bloc": -0.12,
            "eur_rates": 0.10,
            "safe_haven": -0.36,
            "commodity_fx": 0.42,
            "risk_on": 0.84,
            "liquidity_stress": -0.28,
            "macro_shock": 0.56,
        },
        "PLN": {
            "usd_bloc": -0.10,
            "eur_rates": 0.62,
            "safe_haven": -0.24,
            "commodity_fx": 0.20,
            "risk_on": 0.44,
            "liquidity_stress": -0.16,
            "macro_shock": 0.44,
        },
        "CZK": {
            "usd_bloc": -0.08,
            "eur_rates": 0.60,
            "safe_haven": -0.22,
            "commodity_fx": 0.16,
            "risk_on": 0.40,
            "liquidity_stress": -0.14,
            "macro_shock": 0.42,
        },
        "HUF": {
            "usd_bloc": -0.12,
            "eur_rates": 0.56,
            "safe_haven": -0.28,
            "commodity_fx": 0.18,
            "risk_on": 0.46,
            "liquidity_stress": -0.20,
            "macro_shock": 0.46,
        },
        "THB": {
            "usd_bloc": -0.10,
            "eur_rates": 0.08,
            "safe_haven": -0.18,
            "commodity_fx": 0.16,
            "risk_on": 0.42,
            "liquidity_stress": -0.10,
            "macro_shock": 0.36,
        },
        "DKK": {
            "usd_bloc": -0.08,
            "eur_rates": 0.88,
            "safe_haven": -0.10,
            "commodity_fx": 0.04,
            "risk_on": 0.18,
            "liquidity_stress": -0.08,
            "macro_shock": 0.34,
        },
        "HKD": {
            "usd_bloc": 0.78,
            "eur_rates": 0.06,
            "safe_haven": 0.02,
            "commodity_fx": 0.02,
            "risk_on": 0.14,
            "liquidity_stress": 0.84,
            "macro_shock": 0.26,
        },
    }
    out: dict[str, dict[str, float]] = {}
    for currency in currencies:
        out[currency] = deepcopy(defaults.get(currency, {
            "usd_bloc": 0.0,
            "eur_rates": 0.0,
            "safe_haven": 0.0,
            "commodity_fx": 0.0,
            "risk_on": 0.0,
            "liquidity_stress": 0.0,
            "macro_shock": 0.20,
        }))
    return out


def default_config() -> dict[str, Any]:
    universe = _safe_market_universe()
    tradable_pairs = _tradable_pairs(universe)
    currencies = _supported_currencies(tradable_pairs)
    return {
        "schema_version": PAIR_NETWORK_CONFIG_VERSION,
        "enabled": True,
        "graph_stale_after_sec": 43200,
        "history_points": 192,
        "max_edges_per_pair": 10,
        "action_mode": "AUTO_APPLY",
        "fallback_structural_only": True,
        "min_empirical_overlap": 128,
        "empirical_lookback_bars": 512,
        "structural_weight": 0.72,
        "empirical_weight": 0.28,
        "redundancy_threshold": 0.68,
        "contradiction_threshold": 0.74,
        "concentration_reduce_threshold": 0.58,
        "concentration_block_threshold": 0.80,
        "execution_overlap_threshold": 0.62,
        "reduced_size_multiplier_floor": 0.45,
        "preferred_expression_margin": 0.04,
        "min_incremental_edge_score": 0.12,
        "selection_weights": {
            "edge_after_costs": 0.34,
            "execution_quality": 0.20,
            "calibration_quality": 0.16,
            "portfolio_fit": 0.14,
            "diversification": 0.10,
            "macro_fit": 0.06,
        },
        "market_universe": {
            "tradable_pairs": tradable_pairs,
            "currencies": currencies,
        },
        "currency_profiles": default_currency_profiles(currencies),
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
        raise OfflineLabError("Pair-network config must be a JSON object")
    merged = _merge_defaults(default_config(), payload)
    if int(merged.get("schema_version", 0) or 0) != PAIR_NETWORK_CONFIG_VERSION:
        raise OfflineLabError(f"Pair-network schema_version must be {PAIR_NETWORK_CONFIG_VERSION}")
    if not isinstance(merged.get("enabled"), bool):
        raise OfflineLabError("Pair-network enabled must be a boolean")
    if int(merged.get("graph_stale_after_sec", 0) or 0) < 300:
        raise OfflineLabError("Pair-network graph_stale_after_sec must be at least 300")
    if int(merged.get("history_points", 0) or 0) < 24:
        raise OfflineLabError("Pair-network history_points must be at least 24")
    if int(merged.get("max_edges_per_pair", 0) or 0) < 3:
        raise OfflineLabError("Pair-network max_edges_per_pair must be at least 3")
    action_mode = str(merged.get("action_mode", "") or "").strip().upper()
    if action_mode not in {"AUTO_APPLY", "RECOMMEND_ONLY"}:
        raise OfflineLabError("Pair-network action_mode must be AUTO_APPLY or RECOMMEND_ONLY")
    merged["action_mode"] = action_mode

    structural_weight = float(merged.get("structural_weight", 0.0) or 0.0)
    empirical_weight = float(merged.get("empirical_weight", 0.0) or 0.0)
    if structural_weight < 0.0 or empirical_weight < 0.0:
        raise OfflineLabError("Pair-network structural_weight and empirical_weight must be non-negative")
    if structural_weight + empirical_weight <= 0.0:
        raise OfflineLabError("Pair-network structural_weight + empirical_weight must be positive")

    redundancy_threshold = float(merged.get("redundancy_threshold", 0.0) or 0.0)
    contradiction_threshold = float(merged.get("contradiction_threshold", 0.0) or 0.0)
    concentration_reduce = float(merged.get("concentration_reduce_threshold", 0.0) or 0.0)
    concentration_block = float(merged.get("concentration_block_threshold", 0.0) or 0.0)
    if not (0.0 < redundancy_threshold < 1.0):
        raise OfflineLabError("Pair-network redundancy_threshold must be within (0,1)")
    if not (0.0 < contradiction_threshold <= 1.0):
        raise OfflineLabError("Pair-network contradiction_threshold must be within (0,1]")
    if not (0.0 < concentration_reduce < concentration_block <= 1.0):
        raise OfflineLabError(
            "Pair-network concentration thresholds must satisfy 0 < concentration_reduce_threshold < concentration_block_threshold <= 1"
        )

    selection_weights = dict(merged.get("selection_weights", {}))
    if not selection_weights:
        raise OfflineLabError("Pair-network selection_weights must not be empty")
    if sum(max(float(value or 0.0), 0.0) for value in selection_weights.values()) <= 0.0:
        raise OfflineLabError("Pair-network selection_weights must have positive total weight")

    market_universe = dict(merged.get("market_universe", {}))
    tradable_pairs = [str(item or "").strip().upper() for item in list(market_universe.get("tradable_pairs", []))]
    if not tradable_pairs:
        raise OfflineLabError("Pair-network market_universe.tradable_pairs must not be empty")
    if any(len(pair) != 6 for pair in tradable_pairs):
        raise OfflineLabError("Pair-network tradable pairs must contain 6-character FX symbols")
    currencies = [str(item or "").strip().upper() for item in list(market_universe.get("currencies", []))]
    if not currencies:
        currencies = _supported_currencies(tradable_pairs)

    profiles = dict(merged.get("currency_profiles", {}))
    if not profiles:
        raise OfflineLabError("Pair-network currency_profiles must not be empty")
    required_factors = set(next(iter(default_currency_profiles(["USD"]).values())).keys())
    normalized_profiles: dict[str, dict[str, float]] = {}
    for currency in currencies:
        raw_profile = dict(profiles.get(currency, {}))
        if not raw_profile:
            raise OfflineLabError(f"Pair-network missing currency profile for {currency}")
        normalized: dict[str, float] = {}
        for factor in required_factors:
            try:
                normalized[factor] = float(raw_profile.get(factor, 0.0) or 0.0)
            except Exception as exc:  # noqa: BLE001
                raise OfflineLabError(f"Pair-network currency profile {currency}.{factor} must be numeric") from exc
        normalized_profiles[currency] = normalized

    merged["market_universe"] = {
        "tradable_pairs": sorted(set(tradable_pairs)),
        "currencies": sorted(set(currencies)),
    }
    merged["currency_profiles"] = normalized_profiles
    merged["structural_weight"] = structural_weight
    merged["empirical_weight"] = empirical_weight
    return merged


def save_config(payload: dict[str, Any]) -> dict[str, Any]:
    ensure_pair_network_dirs()
    normalized = validate_config_payload(payload)
    json_dump(PAIR_NETWORK_CONFIG_PATH, normalized)
    return normalized


def load_config(path: str | None = None) -> dict[str, Any]:
    ensure_pair_network_dirs()
    target = PAIR_NETWORK_CONFIG_PATH if path is None else Path(path)
    payload = json_load(target)
    if not payload:
        payload = default_config()
        json_dump(target, payload)
    normalized = validate_config_payload(payload)
    if target == PAIR_NETWORK_CONFIG_PATH and payload != normalized:
        json_dump(target, normalized)
    return normalized


def export_runtime_config(payload: dict[str, Any]) -> Any:
    normalized = validate_config_payload(payload)
    ensure_pair_network_dirs()
    lines = [
        f"schema_version\t{PAIR_NETWORK_CONFIG_VERSION}",
        f"enabled\t{1 if normalized['enabled'] else 0}",
        f"graph_stale_after_sec\t{int(normalized['graph_stale_after_sec'])}",
        f"history_points\t{int(normalized['history_points'])}",
        f"max_edges_per_pair\t{int(normalized['max_edges_per_pair'])}",
        f"action_mode\t{normalized['action_mode']}",
        f"fallback_structural_only\t{1 if normalized['fallback_structural_only'] else 0}",
        f"min_empirical_overlap\t{int(normalized['min_empirical_overlap'])}",
        f"empirical_lookback_bars\t{int(normalized['empirical_lookback_bars'])}",
        f"structural_weight\t{float(normalized['structural_weight']):.6f}",
        f"empirical_weight\t{float(normalized['empirical_weight']):.6f}",
        f"redundancy_threshold\t{float(normalized['redundancy_threshold']):.6f}",
        f"contradiction_threshold\t{float(normalized['contradiction_threshold']):.6f}",
        f"concentration_reduce_threshold\t{float(normalized['concentration_reduce_threshold']):.6f}",
        f"concentration_block_threshold\t{float(normalized['concentration_block_threshold']):.6f}",
        f"execution_overlap_threshold\t{float(normalized['execution_overlap_threshold']):.6f}",
        f"reduced_size_multiplier_floor\t{float(normalized['reduced_size_multiplier_floor']):.6f}",
        f"preferred_expression_margin\t{float(normalized['preferred_expression_margin']):.6f}",
        f"min_incremental_edge_score\t{float(normalized['min_incremental_edge_score']):.6f}",
    ]
    for key, value in sorted(dict(normalized.get("selection_weights", {})).items()):
        lines.append(f"selection_weight\t{key}\t{float(value):.6f}")
    for currency, profile in sorted(dict(normalized.get("currency_profiles", {})).items()):
        for factor, value in sorted(profile.items()):
            lines.append(f"currency_profile\t{currency}\t{factor}\t{float(value):.6f}")
    for pair in list(dict(normalized.get("market_universe", {})).get("tradable_pairs", [])):
        lines.append(f"tradable_pair\t{pair}\t1")
    PAIR_NETWORK_RUNTIME_CONFIG_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return PAIR_NETWORK_RUNTIME_CONFIG_PATH


def export_runtime_status(payload: dict[str, Any]) -> Any:
    ensure_pair_network_dirs()
    lines = []
    for key, value in payload.items():
        if isinstance(value, bool):
            encoded = "1" if value else "0"
        else:
            encoded = str(value)
        lines.append(f"{key}\t{encoded}")
    PAIR_NETWORK_RUNTIME_STATUS_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return PAIR_NETWORK_RUNTIME_STATUS_PATH
