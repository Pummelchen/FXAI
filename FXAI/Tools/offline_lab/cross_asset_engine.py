from __future__ import annotations

import json
import time
from typing import Any

from .cross_asset_config import load_config, resolve_probe_symbols
from .cross_asset_contracts import (
    COMMON_CROSS_ASSET_CONFIG,
    COMMON_CROSS_ASSET_FLAT,
    COMMON_CROSS_ASSET_HISTORY,
    COMMON_CROSS_ASSET_JSON,
    COMMON_CROSS_ASSET_PROBE_JSON,
    COMMON_CROSS_ASSET_PROBE_STATUS,
    COMMON_CROSS_ASSET_STATUS,
    COMMON_CROSS_ASSET_SYMBOL_MAP,
    CROSS_ASSET_CONFIG_PATH,
    CROSS_ASSET_LOCAL_HISTORY_PATH,
    CROSS_ASSET_SCHEMA_VERSION,
    CROSS_ASSET_STATE_PATH,
    CROSS_ASSET_STATUS_PATH,
    ensure_cross_asset_dirs,
    isoformat_utc,
    json_dump,
    json_load,
    ndjson_append,
    parse_iso8601,
    sanitize_utc_timestamp,
    utc_now,
)
from .cross_asset_math import (
    assign_liquidity_state,
    bounded_zscore,
    build_pair_cross_asset_risk,
    clamp01,
    magnitude_score_from_z,
    positive_score_from_z,
    safe_mean,
    safe_pstdev,
    select_proxy,
    top_label_from_scores,
)
from .rates_engine_contracts import COMMON_RATES_JSON
from .cross_asset_replay import build_cross_asset_replay_report


def _safe_float(value: Any) -> float:
    try:
        return float(value or 0.0)
    except Exception:
        return 0.0


def _symbol_records(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    raw = payload.get("symbols", {})
    if not isinstance(raw, dict):
        return {}
    records: dict[str, dict[str, Any]] = {}
    for key, value in raw.items():
        if not isinstance(value, dict):
            continue
        symbol = str(key or "").strip().upper()
        if symbol:
            records[symbol] = value
    return records


def _proxy_change(record: dict[str, Any], field: str) -> float:
    return _safe_float(record.get(field))


def _proxy_abs_change(record: dict[str, Any], *fields: str) -> float:
    values = [_safe_float(record.get(field)) for field in fields]
    values = [abs(value) for value in values if value == value]
    return max(values) if values else 0.0


def _select_group(symbols: dict[str, dict[str, Any]], candidates: list[str]) -> dict[str, Any]:
    return select_proxy(symbols, candidates)


def _rates_currency_state(payload: dict[str, Any], currency: str) -> dict[str, Any]:
    raw = payload.get("currencies", {})
    if not isinstance(raw, dict):
        return {}
    state = raw.get(currency)
    return state if isinstance(state, dict) else {}


def _rates_pair_states(payload: dict[str, Any]) -> list[dict[str, Any]]:
    raw = payload.get("pairs", {})
    if not isinstance(raw, dict):
        return []
    return [value for value in raw.values() if isinstance(value, dict)]


def _load_state() -> dict[str, Any]:
    payload = json_load(CROSS_ASSET_STATE_PATH)
    return payload if isinstance(payload, dict) else {}


def _history_values(state: dict[str, Any], key: str) -> list[float]:
    feature_history = state.get("feature_history", {})
    if not isinstance(feature_history, dict):
        return []
    records = feature_history.get(key, [])
    if not isinstance(records, list):
        return []
    values: list[float] = []
    for record in records:
        if isinstance(record, dict):
            values.append(_safe_float(record.get("value")))
        else:
            values.append(_safe_float(record))
    return values


def _append_history(state: dict[str, Any], key: str, value: float, generated_at: str, limit: int) -> None:
    feature_history = state.setdefault("feature_history", {})
    if not isinstance(feature_history, dict):
        feature_history = {}
        state["feature_history"] = feature_history
    records = feature_history.setdefault(key, [])
    if not isinstance(records, list):
        records = []
        feature_history[key] = records
    records.append({"at": generated_at, "value": round(float(value), 8)})
    if len(records) > limit:
        del records[:-limit]


def _source_status(ok: bool, stale: bool, **extra: Any) -> dict[str, Any]:
    payload = {"ok": bool(ok), "stale": bool(stale)}
    payload.update(extra)
    return payload


def _compute_raw_metrics(
    *,
    config: dict[str, Any],
    probe_payload: dict[str, Any],
    rates_payload: dict[str, Any],
) -> tuple[dict[str, float], dict[str, Any], dict[str, dict[str, Any]], list[str]]:
    symbols = _symbol_records(probe_payload)
    candidate_map = dict(config.get("proxy_candidates", {}))

    selected = {
        "equities": _select_group(symbols, list(candidate_map.get("equities", []))),
        "oil": _select_group(symbols, list(candidate_map.get("oil", []))),
        "gold": _select_group(symbols, list(candidate_map.get("gold", []))),
        "metals": _select_group(symbols, list(candidate_map.get("metals", []))),
        "volatility": _select_group(symbols, list(candidate_map.get("volatility", []))),
        "dollar_liquidity": _select_group(symbols, list(candidate_map.get("dollar_liquidity", []))),
    }

    proxy_records = {group: dict(selection.get("record", {})) for group, selection in selected.items()}
    fallback_reasons: list[str] = []
    for group, selection in selected.items():
        if selection.get("symbol"):
            if selection.get("fallback_used"):
                fallback_reasons.append(f"{group}_fallback_proxy_used")
        else:
            fallback_reasons.append(f"{group}_proxy_missing")

    rates_pairs = _rates_pair_states(rates_payload)
    front_end_divergences = [
        abs(_safe_float(pair.get("front_end_diff")))
        for pair in rates_pairs
        if not bool(pair.get("stale", False))
    ]
    expected_path_divergences = [
        abs(_safe_float(pair.get("expected_path_diff")))
        for pair in rates_pairs
        if not bool(pair.get("stale", False))
    ]
    policy_repricing_values = [
        _safe_float(state.get("policy_repricing_score"))
        for state in dict(rates_payload.get("currencies", {})).values()
        if isinstance(state, dict) and not bool(state.get("stale", False))
    ]
    policy_uncertainty_values = [
        _safe_float(state.get("policy_uncertainty_score"))
        for state in dict(rates_payload.get("currencies", {})).values()
        if isinstance(state, dict) and not bool(state.get("stale", False))
    ]

    usd_state = _rates_currency_state(rates_payload, "USD")
    eur_state = _rates_currency_state(rates_payload, "EUR")

    equity_record = proxy_records["equities"]
    oil_record = proxy_records["oil"]
    gold_record = proxy_records["gold"]
    metals_record = proxy_records["metals"]
    vol_record = proxy_records["volatility"]
    dollar_record = proxy_records["dollar_liquidity"]

    equity_risk_raw = max(
        -_proxy_change(equity_record, "change_pct_1d"),
        0.0,
    ) + 0.40 * max(-_proxy_change(equity_record, "change_pct_4h"), 0.0)
    oil_shock_raw = _proxy_abs_change(oil_record, "change_pct_4h", "change_pct_1d")
    gold_shock_raw = max(_proxy_change(gold_record, "change_pct_1d"), 0.0) + 0.35 * _proxy_abs_change(gold_record, "change_pct_4h")
    metals_shock_raw = _proxy_abs_change(metals_record, "change_pct_4h", "change_pct_1d")
    volatility_proxy_raw = max(_proxy_change(vol_record, "change_pct_1d"), 0.0)
    if volatility_proxy_raw <= 0.0:
        volatility_proxy_raw = 0.55 * equity_risk_raw + 0.45 * safe_mean(
            [
                _safe_float(equity_record.get("range_ratio_1d")),
                _safe_float(oil_record.get("range_ratio_1d")),
                _safe_float(gold_record.get("range_ratio_1d")),
            ]
        )
        fallback_reasons.append("volatility_proxy_derived_from_market_ranges")
    usd_liquidity_raw = (
        max(_proxy_change(dollar_record, "change_pct_1d"), 0.0)
        + 0.35 * _proxy_abs_change(dollar_record, "change_pct_4h")
        + 0.25 * safe_mean(policy_uncertainty_values)
        + 0.18 * equity_risk_raw
    )
    front_end_rate_divergence_raw = 0.60 * safe_mean(front_end_divergences) + 0.40 * safe_mean(expected_path_divergences)
    yield_curve_slope_us_raw = _safe_float(usd_state.get("curve_slope_2s10s"))
    yield_curve_slope_eu_raw = _safe_float(eur_state.get("curve_slope_2s10s"))

    cross_asset_dispersion_raw = safe_pstdev(
        [
            _proxy_change(equity_record, "change_pct_1d"),
            _proxy_change(oil_record, "change_pct_1d"),
            _proxy_change(gold_record, "change_pct_1d"),
            _proxy_change(dollar_record, "change_pct_1d"),
        ]
    ) + 0.45 * front_end_rate_divergence_raw
    global_macro_stress_raw = (
        0.24 * abs(front_end_rate_divergence_raw)
        + 0.18 * equity_risk_raw
        + 0.16 * oil_shock_raw
        + 0.12 * gold_shock_raw
        + 0.16 * volatility_proxy_raw
        + 0.14 * usd_liquidity_raw
    )

    raw_metrics = {
        "front_end_rate_divergence": round(front_end_rate_divergence_raw, 8),
        "yield_curve_slope_us": round(yield_curve_slope_us_raw, 8),
        "yield_curve_slope_eu": round(yield_curve_slope_eu_raw, 8),
        "equity_risk_state": round(equity_risk_raw, 8),
        "commodity_shock_oil": round(oil_shock_raw, 8),
        "commodity_shock_gold": round(gold_shock_raw, 8),
        "commodity_shock_metals": round(metals_shock_raw, 8),
        "volatility_stress": round(volatility_proxy_raw, 8),
        "usd_liquidity_stress": round(usd_liquidity_raw, 8),
        "cross_asset_dislocation": round(cross_asset_dispersion_raw, 8),
        "global_macro_stress": round(global_macro_stress_raw, 8),
    }
    rates_context = {
        "policy_repricing_mean": safe_mean(policy_repricing_values),
        "policy_uncertainty_mean": safe_mean(policy_uncertainty_values),
        "policy_repricing_values": policy_repricing_values,
        "policy_uncertainty_values": policy_uncertainty_values,
    }
    return raw_metrics, rates_context, selected, fallback_reasons


def _compute_features(
    *,
    generated_at: str,
    raw_metrics: dict[str, float],
    state: dict[str, Any],
    history_points: int,
) -> dict[str, float]:
    features: dict[str, float] = {}
    mapping = {
        "front_end_rate_divergence": "front_end_rate_divergence_z",
        "yield_curve_slope_us": "yield_curve_slope_us_z",
        "yield_curve_slope_eu": "yield_curve_slope_eu_z",
        "equity_risk_state": "equity_risk_state_z",
        "commodity_shock_oil": "commodity_shock_oil_z",
        "commodity_shock_gold": "commodity_shock_gold_z",
        "volatility_stress": "volatility_stress_z",
        "usd_liquidity_stress": "usd_liquidity_stress_z",
        "global_macro_stress": "global_macro_stress_z",
        "cross_asset_dislocation": "cross_asset_dislocation_z",
    }
    for raw_key, feature_key in mapping.items():
        value = raw_metrics.get(raw_key, 0.0)
        history = _history_values(state, raw_key)
        features[feature_key] = round(bounded_zscore(value, history), 6)
        _append_history(state, raw_key, value, generated_at, history_points)
    return features


def _compute_state_scores(
    *,
    config: dict[str, Any],
    features: dict[str, float],
    rates_context: dict[str, Any],
) -> dict[str, float]:
    weights = dict(config.get("feature_weights", {}))
    rates_weights = dict(weights.get("rates_repricing", {}))
    risk_weights = dict(weights.get("risk_off", {}))
    commodity_weights = dict(weights.get("commodity_shock", {}))
    vol_weights = dict(weights.get("volatility_shock", {}))
    liquidity_weights = dict(weights.get("usd_liquidity_stress", {}))
    dislocation_weights = dict(weights.get("cross_asset_dislocation", {}))

    policy_repricing_score = clamp01(safe_mean(list(rates_context.get("policy_repricing_values", []))))
    policy_uncertainty_score = clamp01(safe_mean(list(rates_context.get("policy_uncertainty_values", []))))
    curve_divergence_score = magnitude_score_from_z(
        features.get("yield_curve_slope_us_z", 0.0) - features.get("yield_curve_slope_eu_z", 0.0)
    )

    rates_repricing_score = clamp01(
        float(rates_weights.get("front_end_divergence", 0.0) or 0.0) * magnitude_score_from_z(features.get("front_end_rate_divergence_z", 0.0))
        + float(rates_weights.get("policy_repricing", 0.0) or 0.0) * policy_repricing_score
        + float(rates_weights.get("policy_uncertainty", 0.0) or 0.0) * policy_uncertainty_score
        + float(rates_weights.get("curve_divergence", 0.0) or 0.0) * curve_divergence_score
    )
    risk_off_score = clamp01(
        float(risk_weights.get("equity_risk", 0.0) or 0.0) * positive_score_from_z(features.get("equity_risk_state_z", 0.0))
        + float(risk_weights.get("volatility", 0.0) or 0.0) * positive_score_from_z(features.get("volatility_stress_z", 0.0))
        + float(risk_weights.get("gold", 0.0) or 0.0) * positive_score_from_z(features.get("commodity_shock_gold_z", 0.0))
        + float(risk_weights.get("usd_liquidity", 0.0) or 0.0) * positive_score_from_z(features.get("usd_liquidity_stress_z", 0.0))
    )
    commodity_shock_score = clamp01(
        float(commodity_weights.get("oil", 0.0) or 0.0) * magnitude_score_from_z(features.get("commodity_shock_oil_z", 0.0))
        + float(commodity_weights.get("gold", 0.0) or 0.0) * magnitude_score_from_z(features.get("commodity_shock_gold_z", 0.0))
        + float(commodity_weights.get("metals", 0.0) or 0.0) * magnitude_score_from_z(features.get("cross_asset_dislocation_z", 0.0))
    )
    volatility_shock_score = clamp01(
        float(vol_weights.get("volatility", 0.0) or 0.0) * positive_score_from_z(features.get("volatility_stress_z", 0.0))
        + float(vol_weights.get("equity_risk", 0.0) or 0.0) * positive_score_from_z(features.get("equity_risk_state_z", 0.0))
        + float(vol_weights.get("cross_asset_dislocation", 0.0) or 0.0) * positive_score_from_z(features.get("cross_asset_dislocation_z", 0.0))
    )
    usd_liquidity_stress_score = clamp01(
        float(liquidity_weights.get("usd_liquidity", 0.0) or 0.0) * positive_score_from_z(features.get("usd_liquidity_stress_z", 0.0))
        + float(liquidity_weights.get("risk_off", 0.0) or 0.0) * risk_off_score
        + float(liquidity_weights.get("rates_uncertainty", 0.0) or 0.0) * policy_uncertainty_score
        + float(liquidity_weights.get("volatility", 0.0) or 0.0) * volatility_shock_score
    )
    cross_asset_dislocation_score = clamp01(
        float(dislocation_weights.get("dispersion", 0.0) or 0.0) * positive_score_from_z(features.get("cross_asset_dislocation_z", 0.0))
        + float(dislocation_weights.get("rates_repricing", 0.0) or 0.0) * rates_repricing_score
        + float(dislocation_weights.get("commodity_shock", 0.0) or 0.0) * commodity_shock_score
        + float(dislocation_weights.get("volatility_shock", 0.0) or 0.0) * volatility_shock_score
    )
    return {
        "rates_repricing_score": round(rates_repricing_score, 6),
        "risk_off_score": round(risk_off_score, 6),
        "commodity_shock_score": round(commodity_shock_score, 6),
        "volatility_shock_score": round(volatility_shock_score, 6),
        "usd_liquidity_stress_score": round(usd_liquidity_stress_score, 6),
        "cross_asset_dislocation_score": round(cross_asset_dislocation_score, 6),
    }


def _assign_labels(
    *,
    config: dict[str, Any],
    scores: dict[str, float],
) -> tuple[dict[str, str], list[str]]:
    thresholds = dict(config.get("label_thresholds", {}))
    macro_label, label_reasons = top_label_from_scores(
        {
            "RATES_REPRICING": scores.get("rates_repricing_score", 0.0),
            "COMMODITY_SHOCK": scores.get("commodity_shock_score", 0.0),
            "VOLATILITY_SHOCK": scores.get("volatility_shock_score", 0.0),
            "USD_LIQUIDITY_STRESS": scores.get("usd_liquidity_stress_score", 0.0),
            "CROSS_ASSET_DISLOCATION": scores.get("cross_asset_dislocation_score", 0.0),
        },
        normal_max=float(thresholds.get("normal_max", 0.42) or 0.42),
        mixed_delta_max=float(thresholds.get("mixed_delta_max", 0.08) or 0.08),
    )
    risk_state = "RISK_OFF" if scores.get("risk_off_score", 0.0) >= float(thresholds.get("risk_off_min", 0.58) or 0.58) else "NORMAL"
    liquidity_state = assign_liquidity_state(
        max(
            scores.get("volatility_shock_score", 0.0),
            scores.get("usd_liquidity_stress_score", 0.0),
            scores.get("cross_asset_dislocation_score", 0.0),
        ),
        caution_min=float(thresholds.get("liquidity_caution_min", 0.48) or 0.48),
        stressed_min=float(thresholds.get("liquidity_stressed_min", 0.66) or 0.66),
    )
    reasons: list[str] = list(label_reasons)
    if scores.get("rates_repricing_score", 0.0) >= 0.58:
        reasons.append("FRONT_END_RATES_DIVERGING")
    if scores.get("risk_off_score", 0.0) >= 0.58:
        reasons.append("EQUITY_RISK_PROXY_WEAK")
    if scores.get("commodity_shock_score", 0.0) >= 0.52:
        reasons.append("COMMODITY_SHOCK_ELEVATED")
    if scores.get("volatility_shock_score", 0.0) >= 0.58:
        reasons.append("VOLATILITY_STRESS_ELEVATED")
    if scores.get("usd_liquidity_stress_score", 0.0) >= 0.56:
        reasons.append("USD_LIQUIDITY_PRESSURE_RISING")
    if scores.get("cross_asset_dislocation_score", 0.0) >= 0.58:
        reasons.append("CROSS_ASSET_DISLOCATION_ELEVATED")
    return (
        {
            "macro_state": macro_label,
            "risk_state": risk_state,
            "liquidity_state": liquidity_state,
        },
        list(dict.fromkeys(reasons)),
    )


def _build_pair_states(
    *,
    config: dict[str, Any],
    scores: dict[str, float],
    labels: dict[str, str],
    source_status: dict[str, Any],
    global_reasons: list[str],
) -> dict[str, dict[str, Any]]:
    thresholds = dict(config.get("label_thresholds", {}))
    profiles = dict(config.get("currency_profiles", {}))
    weights = dict(config.get("pair_component_weights", {}))
    stale = any(
        bool(dict(source_status.get(source_name, {})).get("stale", True))
        for source_name in list(config.get("critical_sources", []))
    )
    pairs: dict[str, dict[str, Any]] = {}
    for pair in list(dict(config.get("market_universe", {})).get("tradable_pairs", [])):
        if len(pair) != 6:
            continue
        base_currency = pair[:3]
        quote_currency = pair[3:]
        pair_components = build_pair_cross_asset_risk(
            base_currency=base_currency,
            quote_currency=quote_currency,
            currency_profiles=profiles,
            weights=weights,
            rates_repricing_score=float(scores.get("rates_repricing_score", 0.0) or 0.0),
            risk_off_score=float(scores.get("risk_off_score", 0.0) or 0.0),
            commodity_shock_score=float(scores.get("commodity_shock_score", 0.0) or 0.0),
            volatility_shock_score=float(scores.get("volatility_shock_score", 0.0) or 0.0),
            usd_liquidity_stress_score=float(scores.get("usd_liquidity_stress_score", 0.0) or 0.0),
        )
        pair_reasons = list(global_reasons)
        if pair_components["commodity_component"] >= 0.12:
            pair_reasons.append("COMMODITY_SENSITIVE_PAIR")
        if pair_components["risk_component"] >= 0.14:
            pair_reasons.append("RISK_SENTIMENT_SENSITIVE_PAIR")
        if pair_components["liquidity_component"] >= 0.12:
            pair_reasons.append("USD_LIQUIDITY_SENSITIVE_PAIR")
        if stale:
            pair_reasons.append("cross-asset state stale or incomplete")

        trade_gate = "ALLOW"
        if stale:
            trade_gate = "BLOCK"
        elif pair_components["risk_score"] >= float(thresholds.get("pair_block_min", 0.82) or 0.82):
            trade_gate = "BLOCK"
        elif pair_components["risk_score"] >= float(thresholds.get("pair_caution_min", 0.56) or 0.56):
            trade_gate = "CAUTION"

        pairs[pair] = {
            "pair": pair,
            "base_currency": base_currency,
            "quote_currency": quote_currency,
            "macro_state": labels["macro_state"],
            "risk_state": labels["risk_state"],
            "liquidity_state": labels["liquidity_state"],
            "pair_cross_asset_risk_score": round(pair_components["risk_score"], 6),
            "pair_sensitivity": round(pair_components["sensitivity"], 6),
            "trade_gate": trade_gate,
            "stale": stale,
            "reasons": list(dict.fromkeys(pair_reasons))[:8],
        }
    return pairs


def _build_source_status(
    *,
    config: dict[str, Any],
    probe_payload: dict[str, Any],
    rates_payload: dict[str, Any],
    selected: dict[str, dict[str, Any]],
    now_dt,
) -> dict[str, Any]:
    probe_generated_at = sanitize_utc_timestamp(
        probe_payload.get("generated_at") or json_load(COMMON_CROSS_ASSET_PROBE_STATUS).get("generated_at"),
        now_dt=now_dt,
    )
    rates_generated_at = sanitize_utc_timestamp(rates_payload.get("generated_at"), now_dt=now_dt)
    probe_stale_after = int(config.get("probe_stale_after_sec", 300) or 300)
    rates_stale_after = int(config.get("rates_stale_after_sec", 1200) or 1200)
    probe_stale = True
    rates_stale = True
    if probe_generated_at is not None:
        probe_stale = (now_dt - probe_generated_at).total_seconds() > probe_stale_after
    if rates_generated_at is not None:
        rates_stale = (now_dt - rates_generated_at).total_seconds() > rates_stale_after

    equities_ok = bool(selected.get("equities", {}).get("symbol"))
    commodities_ok = bool(selected.get("oil", {}).get("symbol") or selected.get("gold", {}).get("symbol"))
    volatility_ok = bool(selected.get("volatility", {}).get("symbol") or equities_ok)
    liquidity_ok = bool(selected.get("dollar_liquidity", {}).get("symbol"))

    return {
        "rates": _source_status(
            bool(rates_payload),
            rates_stale,
            last_update_at=isoformat_utc(rates_generated_at) if rates_generated_at else "",
        ),
        "context_service": _source_status(
            bool(probe_payload),
            probe_stale,
            last_update_at=isoformat_utc(probe_generated_at) if probe_generated_at else "",
            available_symbols=len(_symbol_records(probe_payload)),
            configured_symbols=len(list(config.get("market_universe", {}).get("indicator_symbols", []))),
        ),
        "equities": _source_status(equities_ok, probe_stale or not equities_ok, proxy_symbol=selected.get("equities", {}).get("symbol", "")),
        "commodities": _source_status(commodities_ok, probe_stale or not commodities_ok, oil_proxy=selected.get("oil", {}).get("symbol", ""), gold_proxy=selected.get("gold", {}).get("symbol", "")),
        "volatility": _source_status(volatility_ok, probe_stale or not volatility_ok, proxy_symbol=selected.get("volatility", {}).get("symbol", "")),
        "liquidity": _source_status(liquidity_ok, probe_stale or not liquidity_ok, proxy_symbol=selected.get("dollar_liquidity", {}).get("symbol", "")),
    }


def _recent_transitions(state: dict[str, Any], labels: dict[str, str], pairs: dict[str, dict[str, Any]], generated_at: str, limit: int) -> list[dict[str, Any]]:
    previous = state.get("last_snapshot_summary", {})
    transitions = list(state.get("recent_transitions", []))
    if not isinstance(previous, dict):
        previous = {}
    if not isinstance(transitions, list):
        transitions = []

    def maybe_append(kind: str, old: str, new: str, target: str = "global") -> None:
        if old and old != new:
            transitions.append({"type": kind, "target": target, "from": old, "to": new, "observed_at": generated_at})

    maybe_append("macro_state", str(previous.get("macro_state", "")), labels["macro_state"])
    maybe_append("risk_state", str(previous.get("risk_state", "")), labels["risk_state"])
    maybe_append("liquidity_state", str(previous.get("liquidity_state", "")), labels["liquidity_state"])

    previous_pair_gates = previous.get("pair_gates", {})
    if not isinstance(previous_pair_gates, dict):
        previous_pair_gates = {}
    for pair, pair_state in pairs.items():
        maybe_append("pair_gate", str(previous_pair_gates.get(pair, "")), str(pair_state.get("trade_gate", "")), pair)

    state["recent_transitions"] = transitions[-limit:]
    state["last_snapshot_summary"] = {
        "macro_state": labels["macro_state"],
        "risk_state": labels["risk_state"],
        "liquidity_state": labels["liquidity_state"],
        "pair_gates": {pair: spec.get("trade_gate", "") for pair, spec in pairs.items()},
    }
    return list(state["recent_transitions"])


def _write_runtime_flat(
    *,
    generated_at: str,
    features: dict[str, float],
    scores: dict[str, float],
    labels: dict[str, str],
    reasons: list[str],
    pair_states: dict[str, dict[str, Any]],
) -> None:
    generated_at_unix = int((parse_iso8601(generated_at) or utc_now()).timestamp())
    lines = [
        f"meta\tglobal\tgenerated_at\t{generated_at}",
        f"meta\tglobal\tgenerated_at_unix\t{generated_at_unix}",
        f"meta\tglobal\tmacro_state\t{labels['macro_state']}",
        f"meta\tglobal\trisk_state\t{labels['risk_state']}",
        f"meta\tglobal\tliquidity_state\t{labels['liquidity_state']}",
    ]
    for key, value in sorted(features.items()):
        lines.append(f"feature\tglobal\t{key}\t{value}")
    for key, value in sorted(scores.items()):
        lines.append(f"score\tglobal\t{key}\t{value}")
    for reason in reasons:
        lines.append(f"reason\tglobal\treason\t{reason}")
    symbol_map_lines = []
    for pair, state in sorted(pair_states.items()):
        symbol_map_lines.append(f"symbol\t{pair}\t{pair}")
        for key in (
            "macro_state",
            "risk_state",
            "liquidity_state",
            "pair_cross_asset_risk_score",
            "pair_sensitivity",
            "trade_gate",
            "stale",
        ):
            value = state.get(key, "")
            if isinstance(value, bool):
                value = 1 if value else 0
            lines.append(f"pair\t{pair}\t{key}\t{value}")
        for reason in list(state.get("reasons", [])):
            lines.append(f"pair_reason\t{pair}\treason\t{reason}")
    COMMON_CROSS_ASSET_FLAT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    COMMON_CROSS_ASSET_SYMBOL_MAP.write_text("\n".join(symbol_map_lines) + ("\n" if symbol_map_lines else ""), encoding="utf-8")


def run_cross_asset_cycle(daemon_context: dict[str, Any] | None = None) -> dict[str, Any]:
    ensure_cross_asset_dirs()
    config = load_config()
    now_dt = utc_now()
    generated_at = isoformat_utc(now_dt)
    probe_payload = json_load(COMMON_CROSS_ASSET_PROBE_JSON)
    rates_payload = json_load(COMMON_RATES_JSON)
    state = _load_state()

    raw_metrics, rates_context, selected, fallback_reasons = _compute_raw_metrics(
        config=config,
        probe_payload=probe_payload,
        rates_payload=rates_payload,
    )
    features = _compute_features(
        generated_at=generated_at,
        raw_metrics=raw_metrics,
        state=state,
        history_points=int(config.get("history_points", 192) or 192),
    )
    source_status = _build_source_status(
        config=config,
        probe_payload=probe_payload,
        rates_payload=rates_payload,
        selected=selected,
        now_dt=now_dt,
    )
    scores = _compute_state_scores(config=config, features=features, rates_context=rates_context)
    labels, global_reasons = _assign_labels(config=config, scores=scores)

    partial_data = any(not bool(spec.get("ok", False)) for name, spec in source_status.items() if name not in {"rates", "context_service"})
    stale = any(
        bool(dict(source_status.get(source_name, {})).get("stale", True))
        for source_name in list(config.get("critical_sources", []))
    )
    reasons = list(dict.fromkeys(global_reasons + fallback_reasons))
    if partial_data:
        reasons.append("PARTIAL_CROSS_ASSET_SOURCE_COVERAGE")
    if stale:
        reasons.append("CRITICAL_CROSS_ASSET_SOURCE_STALE")

    pair_states = _build_pair_states(
        config=config,
        scores=scores,
        labels=labels,
        source_status=source_status,
        global_reasons=reasons,
    )
    transitions = _recent_transitions(
        state,
        labels,
        pair_states,
        generated_at,
        int(config.get("max_recent_transitions", 24) or 24),
    )
    selected_proxies = {
        group: {
            "symbol": selection.get("symbol", ""),
            "fallback_used": bool(selection.get("fallback_used", False)),
            "available": bool(selection.get("record", {}).get("available", False)),
            "change_pct_1d": _safe_float(selection.get("record", {}).get("change_pct_1d")),
            "range_ratio_1d": _safe_float(selection.get("record", {}).get("range_ratio_1d")),
        }
        for group, selection in selected.items()
    }

    snapshot = {
        "schema_version": CROSS_ASSET_SCHEMA_VERSION,
        "generated_at": generated_at,
        "source_status": source_status,
        "features": features,
        "state_scores": scores,
        "state_labels": labels,
        "selected_proxies": selected_proxies,
        "pair_states": pair_states,
        "recent_transitions": transitions,
        "reason_codes": reasons[:12],
        "quality_flags": {
            "fallback_proxy_used": any(bool(selection.get("fallback_used", False)) for selection in selected.values()),
            "partial_data": partial_data,
            "data_stale": stale,
        },
    }
    status_payload = {
        "generated_at": generated_at,
        "daemon": dict(daemon_context or {}),
        "source_status": source_status,
        "health": {
            "pair_count": len(pair_states),
            "feature_count": len(features),
            "snapshot_stale_after_sec": int(config.get("snapshot_stale_after_sec", 900) or 900),
            "partial_data": partial_data,
            "data_stale": stale,
        },
        "artifacts": {
            "snapshot_json": str(COMMON_CROSS_ASSET_JSON),
            "snapshot_flat": str(COMMON_CROSS_ASSET_FLAT),
            "history_ndjson": str(COMMON_CROSS_ASSET_HISTORY),
            "symbol_map_tsv": str(COMMON_CROSS_ASSET_SYMBOL_MAP),
            "probe_snapshot_json": str(COMMON_CROSS_ASSET_PROBE_JSON),
            "probe_status_json": str(COMMON_CROSS_ASSET_PROBE_STATUS),
        },
    }

    COMMON_CROSS_ASSET_JSON.write_text(json.dumps(snapshot, indent=2, sort_keys=True), encoding="utf-8")
    COMMON_CROSS_ASSET_STATUS.write_text(json.dumps(status_payload, indent=2, sort_keys=True), encoding="utf-8")
    CROSS_ASSET_STATUS_PATH.write_text(json.dumps(status_payload, indent=2, sort_keys=True), encoding="utf-8")
    state["generated_at"] = generated_at
    state["snapshot"] = {
        "state_labels": labels,
        "state_scores": scores,
        "quality_flags": snapshot["quality_flags"],
    }
    json_dump(CROSS_ASSET_STATE_PATH, state)
    _write_runtime_flat(
        generated_at=generated_at,
        features=features,
        scores=scores,
        labels=labels,
        reasons=reasons,
        pair_states=pair_states,
    )

    history_record = {
        "record_type": "snapshot",
        "generated_at": generated_at,
        "snapshot": snapshot,
    }
    ndjson_append(COMMON_CROSS_ASSET_HISTORY, history_record)
    ndjson_append(CROSS_ASSET_LOCAL_HISTORY_PATH, history_record)
    build_cross_asset_replay_report(hours_back=72)

    return {
        "generated_at": generated_at,
        "snapshot_path": str(COMMON_CROSS_ASSET_JSON),
        "flat_path": str(COMMON_CROSS_ASSET_FLAT),
        "history_path": str(COMMON_CROSS_ASSET_HISTORY),
        "pair_count": len(pair_states),
        "partial_data": partial_data,
        "data_stale": stale,
        "macro_state": labels["macro_state"],
    }


def validate_cross_asset_config() -> dict[str, Any]:
    config = load_config()
    probe_status = json_load(COMMON_CROSS_ASSET_PROBE_STATUS)
    return {
        "ok": True,
        "config_path": str(CROSS_ASSET_CONFIG_PATH),
        "resolved_probe_symbol_count": len(resolve_probe_symbols(config)),
        "probe_config_path": str(COMMON_CROSS_ASSET_CONFIG),
        "runtime_snapshot_path": str(COMMON_CROSS_ASSET_JSON),
        "probe_snapshot_path": str(COMMON_CROSS_ASSET_PROBE_JSON),
        "market_pairs": list(dict(config.get("market_universe", {})).get("tradable_pairs", [])),
        "probe_generated_at": str(probe_status.get("generated_at", "")),
    }


def cross_asset_health_snapshot() -> dict[str, Any]:
    runtime_status = json_load(COMMON_CROSS_ASSET_STATUS)
    local_status = json_load(CROSS_ASSET_STATUS_PATH)
    probe_status = json_load(COMMON_CROSS_ASSET_PROBE_STATUS)
    generated_at = (
        str(runtime_status.get("generated_at", "") or local_status.get("generated_at", "") or probe_status.get("generated_at", ""))
    )
    return {
        "generated_at": generated_at,
        "status_path": str(CROSS_ASSET_STATUS_PATH),
        "runtime_status_path": str(COMMON_CROSS_ASSET_STATUS),
        "runtime_snapshot_path": str(COMMON_CROSS_ASSET_JSON),
        "probe_status_path": str(COMMON_CROSS_ASSET_PROBE_STATUS),
        "probe_snapshot_path": str(COMMON_CROSS_ASSET_PROBE_JSON),
        "source_status": runtime_status.get("source_status", {}),
        "health": runtime_status.get("health", {}),
        "probe": probe_status.get("service", {}),
    }


def run_cross_asset_once() -> dict[str, Any]:
    return run_cross_asset_cycle()


def run_cross_asset_daemon(iterations: int = 0, interval_seconds: int | None = None) -> dict[str, Any]:
    config = load_config()
    interval = int(interval_seconds or config.get("poll_interval_sec", 120) or 120)
    if interval < 30:
        interval = 30
    attempts = 0
    successful_cycles = 0
    consecutive_failures = 0
    last_payload: dict[str, Any] | None = None
    while iterations <= 0 or attempts < iterations:
        started_at = utc_now()
        attempts += 1
        try:
            last_payload = run_cross_asset_cycle(
                daemon_context={
                    "mode": "daemon",
                    "heartbeat_at": isoformat_utc(started_at),
                    "interval_seconds": interval,
                    "cycles_completed": successful_cycles,
                    "consecutive_failures": consecutive_failures,
                    "degraded": consecutive_failures > 0,
                    "last_error": "",
                }
            )
            successful_cycles += 1
            consecutive_failures = 0
        except Exception as exc:
            consecutive_failures += 1
            failure_time = utc_now()
            COMMON_CROSS_ASSET_STATUS.write_text(
                json.dumps(
                    {
                        "generated_at": isoformat_utc(failure_time),
                        "daemon": {
                            "mode": "daemon",
                            "heartbeat_at": isoformat_utc(failure_time),
                            "interval_seconds": interval,
                            "cycles_completed": successful_cycles,
                            "consecutive_failures": consecutive_failures,
                            "degraded": True,
                            "last_error": str(exc),
                        },
                    },
                    indent=2,
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
        if iterations > 0 and attempts >= iterations:
            break
        for _ in range(interval):
            time.sleep(1.0)
    return {
        "iterations": attempts,
        "interval_seconds": interval,
        "successful_iterations": successful_cycles,
        "consecutive_failures": consecutive_failures,
        "last_payload": last_payload or {},
    }
