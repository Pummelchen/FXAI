from __future__ import annotations

import math
from statistics import mean, pstdev
from typing import Any


def clamp(value: float, minimum: float, maximum: float) -> float:
    if value < minimum:
        return minimum
    if value > maximum:
        return maximum
    return value


def clamp01(value: float) -> float:
    return clamp(float(value), 0.0, 1.0)


def safe_mean(values: list[float]) -> float:
    usable = [float(value) for value in values]
    if not usable:
        return 0.0
    return float(mean(usable))


def safe_pstdev(values: list[float]) -> float:
    usable = [float(value) for value in values]
    if len(usable) < 2:
        return 0.0
    return float(pstdev(usable))


def bounded_zscore(current: float, history: list[float], *, cap: float = 4.0) -> float:
    if len(history) < 8:
        return clamp(float(current), -cap, cap)
    sigma = safe_pstdev(history)
    if sigma <= 1e-9:
        return clamp(float(current), -cap, cap)
    z_value = (float(current) - safe_mean(history)) / sigma
    return clamp(z_value, -cap, cap)


def positive_score_from_z(z_value: float, *, cap: float = 3.0) -> float:
    return clamp01(max(float(z_value), 0.0) / cap)


def magnitude_score_from_z(z_value: float, *, cap: float = 3.0) -> float:
    return clamp01(abs(float(z_value)) / cap)


def mean_numeric(records: list[dict[str, Any]], key: str) -> float:
    values = [float(record.get(key, 0.0) or 0.0) for record in records if isinstance(record, dict)]
    return safe_mean(values)


def select_proxy(symbols: dict[str, Any], candidates: list[str]) -> dict[str, Any]:
    for raw_symbol in candidates:
        symbol = str(raw_symbol or "").strip().upper()
        if not symbol:
            continue
        record = symbols.get(symbol)
        if not isinstance(record, dict):
            continue
        if bool(record.get("available", False)):
            return {"symbol": symbol, "record": record, "fallback_used": symbol != str(candidates[0]).strip().upper()}
    return {"symbol": "", "record": {}, "fallback_used": False}


def top_label_from_scores(
    scores: dict[str, float],
    *,
    normal_max: float,
    mixed_delta_max: float,
) -> tuple[str, list[str]]:
    ranked = sorted(
        ((name, float(value or 0.0)) for name, value in scores.items()),
        key=lambda item: (-item[1], item[0]),
    )
    if not ranked:
        return "NORMAL", []
    top_name, top_value = ranked[0]
    second_value = ranked[1][1] if len(ranked) > 1 else 0.0
    if top_value <= normal_max:
        return "NORMAL", []
    if abs(top_value - second_value) <= mixed_delta_max:
        return "MIXED", [top_name]
    return top_name, [top_name]


def assign_liquidity_state(score: float, *, caution_min: float, stressed_min: float) -> str:
    if score >= stressed_min:
        return "STRESSED"
    if score >= caution_min:
        return "CAUTION"
    return "NORMAL"


def build_pair_cross_asset_risk(
    *,
    base_currency: str,
    quote_currency: str,
    currency_profiles: dict[str, dict[str, float]],
    weights: dict[str, float],
    rates_repricing_score: float,
    risk_off_score: float,
    commodity_shock_score: float,
    volatility_shock_score: float,
    usd_liquidity_stress_score: float,
) -> dict[str, float]:
    base_profile = dict(currency_profiles.get(base_currency, {}))
    quote_profile = dict(currency_profiles.get(quote_currency, {}))

    def profile_value(key: str) -> float:
        return max(float(base_profile.get(key, 0.0) or 0.0), float(quote_profile.get(key, 0.0) or 0.0))

    usd_multiplier = 1.0 if "USD" in {base_currency, quote_currency} else max(profile_value("liquidity"), 0.55)

    rates_component = float(weights.get("rates", 0.0) or 0.0) * rates_repricing_score * profile_value("rates")
    risk_component = float(weights.get("risk_off", 0.0) or 0.0) * risk_off_score * max(profile_value("risk"), profile_value("safe_haven"))
    commodity_component = float(weights.get("commodity", 0.0) or 0.0) * commodity_shock_score * profile_value("commodity")
    volatility_component = float(weights.get("volatility", 0.0) or 0.0) * volatility_shock_score * profile_value("liquidity")
    liquidity_component = float(weights.get("liquidity", 0.0) or 0.0) * usd_liquidity_stress_score * usd_multiplier

    risk_score = clamp01(rates_component + risk_component + commodity_component + volatility_component + liquidity_component)
    sensitivity = clamp01(profile_value("rates") * 0.24 + profile_value("risk") * 0.22 + profile_value("commodity") * 0.18 + profile_value("liquidity") * 0.18 + profile_value("safe_haven") * 0.18)
    return {
        "risk_score": risk_score,
        "sensitivity": sensitivity,
        "rates_component": rates_component,
        "risk_component": risk_component,
        "commodity_component": commodity_component,
        "volatility_component": volatility_component,
        "liquidity_component": liquidity_component,
    }
