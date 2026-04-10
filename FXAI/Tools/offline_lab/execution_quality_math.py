from __future__ import annotations

from typing import Any


def clamp(value: float, low: float, high: float) -> float:
    return min(max(float(value), float(low)), float(high))


def classify_session_thinness(session_label: str, handoff_flag: bool) -> float:
    session = str(session_label or "").upper()
    thinness = 0.18
    if "ASIA" in session:
        thinness = 0.42
    if "OVERLAP" in session:
        thinness = 0.22
    if "ROLLOVER" in session or "OFF" in session:
        thinness = 0.60
    if handoff_flag:
        thinness = max(thinness, 0.55)
    return clamp(thinness, 0.0, 1.0)


def select_execution_quality_tier(
    records: list[dict[str, Any]],
    *,
    symbol: str,
    session: str,
    regime: str,
    soft_support_floor: int,
    hard_support_floor: int,
) -> dict[str, Any]:
    symbol = str(symbol or "").upper()
    session = str(session or "").upper()
    regime = str(regime or "").upper()
    hierarchy = [
        "PAIR_SESSION_REGIME",
        "PAIR_REGIME",
        "SESSION_REGIME",
        "REGIME",
        "GLOBAL",
    ]

    def _matches(record: dict[str, Any]) -> bool:
        kind = str(record.get("kind", "") or "").upper()
        rec_symbol = str(record.get("symbol", "*") or "*").upper()
        rec_session = str(record.get("session", "*") or "*").upper()
        rec_regime = str(record.get("regime", "*") or "*").upper()
        if kind == "PAIR_SESSION_REGIME":
            return rec_symbol == symbol and rec_session == session and rec_regime == regime
        if kind == "PAIR_REGIME":
            return rec_symbol == symbol and rec_regime == regime
        if kind == "SESSION_REGIME":
            return rec_session == session and rec_regime == regime
        if kind == "REGIME":
            return rec_regime == regime
        if kind == "GLOBAL":
            return True
        return False

    for kind in hierarchy:
        kind_matches = [
            dict(record)
            for record in records
            if str(record.get("kind", "") or "").upper() == kind and _matches(record)
        ]
        if not kind_matches:
            continue
        preferred = [
            record for record in kind_matches
            if int(record.get("support", 0) or 0) >= int(soft_support_floor)
        ]
        fallback = [
            record for record in kind_matches
            if int(record.get("support", 0) or 0) >= int(hard_support_floor)
        ]
        candidates = preferred if preferred else fallback
        if not candidates:
            continue
        candidates.sort(
            key=lambda record: (
                int(record.get("support", 0) or 0),
                float(record.get("quality", 0.0) or 0.0),
            ),
            reverse=True,
        )
        selected = candidates[0]
        selected["fallback_used"] = not preferred
        selected["support_usable"] = int(selected.get("support", 0) or 0) >= int(hard_support_floor)
        return selected

    return {
        "kind": "GLOBAL",
        "symbol": "*",
        "session": "*",
        "regime": "*",
        "support": 0,
        "quality": 0.34,
        "spread_mult": 1.08,
        "slippage_mult": 1.12,
        "fill_quality_bias": -0.06,
        "latency_mult": 1.08,
        "fragility_mult": 1.10,
        "deviation_mult": 1.06,
        "fallback_used": True,
        "support_usable": False,
    }


def compute_execution_quality_forecast(
    *,
    config: dict[str, Any],
    tier: dict[str, Any],
    symbol: str,
    session_label: str,
    regime_label: str,
    current_spread_points: float,
    broker_slippage_points: float,
    broker_latency_points: float,
    broker_reject_prob: float,
    broker_partial_fill_prob: float,
    broker_fill_ratio_mean: float,
    broker_event_burst_penalty: float,
    micro_spread_zscore: float,
    micro_hostile_execution: float,
    micro_liquidity_stress: float,
    micro_vol_burst: float,
    micro_tick_rate_zscore: float,
    micro_tick_imbalance: float,
    news_risk_score: float,
    rates_risk_score: float,
    stale_context_count: int = 0,
    news_window_active: bool = False,
    rates_repricing_active: bool = False,
    handoff_flag: bool = False,
    support_usable: bool = True,
    data_stale: bool = False,
    base_allowed_deviation_points: float = 4.0,
) -> dict[str, Any]:
    thresholds = dict(config.get("state_thresholds", {}))
    lot_scales = dict(config.get("lot_scales", {}))
    enter_prob_buffers = dict(config.get("enter_prob_buffers", {}))
    forecast_caps = dict(config.get("forecast_caps", {}))
    weights = dict(config.get("weights", {}))

    support_soft_floor = max(int(config.get("support_soft_floor", 64) or 64), 1)
    tier_quality = clamp(float(tier.get("quality", 0.0) or 0.0), 0.0, 1.0)
    tier_support = max(int(tier.get("support", 0) or 0), 0)
    support_shortfall = clamp((support_soft_floor - tier_support) / support_soft_floor, 0.0, 1.0)

    spread_z_norm = clamp(float(micro_spread_zscore or 0.0) / 4.0, 0.0, 1.0)
    hostile_norm = clamp(float(micro_hostile_execution or 0.0), 0.0, 1.0)
    liquidity_norm = clamp(float(micro_liquidity_stress or 0.0), 0.0, 1.0)
    vol_burst_norm = clamp(float(micro_vol_burst or 0.0) / 3.0, 0.0, 1.0)
    tick_rate_norm = clamp(float(micro_tick_rate_zscore or 0.0) / 3.0, 0.0, 1.0)
    tick_imbalance_norm = clamp(abs(float(micro_tick_imbalance or 0.0)), 0.0, 1.0)
    news_risk = clamp(float(news_risk_score or 0.0), 0.0, 1.0)
    rates_risk = clamp(float(rates_risk_score or 0.0), 0.0, 1.0)
    reject_norm = clamp(float(broker_reject_prob or 0.0), 0.0, 1.0)
    partial_norm = clamp(
        max(float(broker_partial_fill_prob or 0.0), 1.0 - clamp(float(broker_fill_ratio_mean or 1.0), 0.0, 1.0)),
        0.0,
        1.0,
    )
    latency_norm = clamp(float(broker_latency_points or 0.0) / 5.0, 0.0, 1.0)
    broker_event_norm = clamp(float(broker_event_burst_penalty or 0.0), 0.0, 1.0)
    session_thinness = classify_session_thinness(session_label, handoff_flag)
    stale_norm = clamp(stale_context_count / 3.0, 0.0, 1.0)

    spread_widening_risk = clamp(
        0.10
        + float(weights.get("spread_zscore", 0.22) or 0.22) * spread_z_norm
        + float(weights.get("news_risk", 0.18) or 0.18) * news_risk
        + float(weights.get("rates_risk", 0.10) or 0.10) * rates_risk
        + float(weights.get("micro_liquidity", 0.18) or 0.18) * liquidity_norm
        + float(weights.get("volatility_burst", 0.14) or 0.14) * vol_burst_norm
        + float(weights.get("session_thinness", 0.10) or 0.10) * session_thinness
        + float(weights.get("broker_reject", 0.16) or 0.16) * reject_norm * 0.45
        + float(weights.get("broker_partial", 0.14) or 0.14) * partial_norm * 0.40
        + float(weights.get("broker_event_burst", 0.12) or 0.12) * broker_event_norm
        + float(weights.get("stale_context", 0.10) or 0.10) * stale_norm
        + (0.08 if news_window_active else 0.0)
        + (0.05 if rates_repricing_active else 0.0)
        - 0.10 * tier_quality,
        0.0,
        1.0,
    )

    spread_expected_mult = clamp(
        0.96
        + 0.38 * float(tier.get("spread_mult", 1.0) or 1.0)
        + 0.64 * spread_widening_risk
        + 0.14 * spread_z_norm
        + 0.06 * session_thinness,
        1.0,
        float(forecast_caps.get("spread_expected_mult", 4.5) or 4.5),
    )
    spread_now = max(float(current_spread_points or 0.0), 0.0)
    spread_expected_points = max(
        spread_now,
        spread_now * spread_expected_mult + 0.12 * max(float(broker_slippage_points or 0.0), 0.0),
    )

    expected_slippage_points = clamp(
        max(float(broker_slippage_points or 0.0), 0.0) * float(tier.get("slippage_mult", 1.0) or 1.0)
        + 0.16 * spread_expected_points
        + 0.55 * hostile_norm
        + 0.38 * vol_burst_norm
        + 0.26 * session_thinness
        + 0.24 * news_risk
        + 0.18 * rates_risk
        + 0.28 * broker_event_norm
        + 0.32 * reject_norm
        + 0.30 * partial_norm
        + 0.18 * latency_norm * float(tier.get("latency_mult", 1.0) or 1.0),
        0.0,
        float(forecast_caps.get("expected_slippage_points", 18.0) or 18.0),
    )

    slippage_risk = clamp(
        0.12
        + 0.24 * clamp(expected_slippage_points / max(spread_expected_points + 0.5, 1.0), 0.0, 3.0) / 3.0
        + 0.18 * hostile_norm
        + 0.12 * vol_burst_norm
        + 0.12 * news_risk
        + 0.08 * rates_risk
        + 0.10 * broker_event_norm
        + 0.12 * reject_norm
        + 0.10 * partial_norm,
        0.0,
        1.0,
    )

    latency_sensitivity_score = clamp(
        0.14
        + 0.22 * tick_rate_norm
        + 0.18 * vol_burst_norm
        + 0.16 * news_risk
        + 0.10 * rates_risk
        + 0.12 * latency_norm * float(tier.get("latency_mult", 1.0) or 1.0)
        + 0.08 * hostile_norm
        + 0.08 * session_thinness
        + 0.06 * tick_imbalance_norm,
        0.0,
        1.0,
    )

    liquidity_fragility_score = clamp(
        0.10
        + 0.26 * liquidity_norm * float(tier.get("fragility_mult", 1.0) or 1.0)
        + 0.16 * hostile_norm
        + 0.12 * spread_z_norm
        + 0.08 * news_risk
        + 0.08 * rates_risk
        + 0.12 * partial_norm
        + 0.10 * reject_norm
        + 0.10 * session_thinness
        + 0.06 * broker_event_norm
        - 0.08 * tier_quality,
        0.0,
        1.0,
    )

    fill_quality_score = clamp(
        0.86
        + float(tier.get("fill_quality_bias", 0.0) or 0.0)
        - 0.28 * slippage_risk
        - 0.24 * latency_sensitivity_score
        - 0.22 * liquidity_fragility_score
        - 0.14 * reject_norm
        - 0.12 * partial_norm
        - 0.08 * session_thinness,
        0.0,
        1.0,
    )

    execution_quality_score = clamp(
        0.40 * fill_quality_score
        + 0.18 * (1.0 - spread_widening_risk)
        + 0.18 * (1.0 - slippage_risk)
        + 0.12 * (1.0 - latency_sensitivity_score)
        + 0.12 * (1.0 - liquidity_fragility_score)
        - 0.08 * stale_norm
        - 0.06 * support_shortfall,
        0.0,
        1.0,
    )

    block_threshold = float(thresholds.get("stressed_min", 0.36) or 0.36) * 0.72
    if data_stale and bool(config.get("block_on_unknown", True)):
        execution_state = "BLOCKED"
    elif (
        bool(config.get("allow_block_state", True))
        and (
            execution_quality_score < block_threshold
            or spread_widening_risk >= 0.90
            or slippage_risk >= 0.90
            or fill_quality_score <= 0.20
        )
    ):
        execution_state = "BLOCKED"
    elif execution_quality_score < float(thresholds.get("stressed_min", 0.36) or 0.36):
        execution_state = "STRESSED"
    elif execution_quality_score < float(thresholds.get("caution_min", 0.54) or 0.54):
        execution_state = "CAUTION"
    else:
        execution_state = "NORMAL"

    allowed_deviation_points = clamp(
        float(base_allowed_deviation_points or 0.0)
        * float(tier.get("deviation_mult", 1.0) or 1.0)
        * (1.0 + 0.14 * spread_widening_risk + 0.18 * slippage_risk + 0.10 * latency_sensitivity_score),
        float(forecast_caps.get("allowed_deviation_points_min", 2.0) or 2.0),
        float(forecast_caps.get("allowed_deviation_points_max", 25.0) or 25.0),
    )

    reasons: list[str] = []
    if data_stale:
        reasons.append("DATA_STALE")
    if not support_usable:
        reasons.append("SUPPORT_TOO_LOW")
    if news_window_active or news_risk >= 0.68:
        reasons.append("NEWS_WINDOW_ACTIVE")
    if rates_repricing_active or rates_risk >= 0.68:
        reasons.append("RATES_REPRICING_ACTIVE")
    if spread_z_norm >= 0.55:
        reasons.append("SPREAD_ALREADY_ELEVATED")
    if hostile_norm >= 0.62:
        reasons.append("MICROSTRUCTURE_HOSTILE")
    if liquidity_norm >= 0.62:
        reasons.append("LIQUIDITY_STRESS_ELEVATED")
    if vol_burst_norm >= 0.58:
        reasons.append("VOLATILITY_BURST")
    if session_thinness >= 0.52:
        reasons.append("LOW_LIQUIDITY_SESSION")
    if slippage_risk >= 0.66:
        reasons.append("SLIPPAGE_RISK_ELEVATED")
    if latency_sensitivity_score >= 0.66:
        reasons.append("LATENCY_SENSITIVITY_HIGH")
    if reject_norm >= 0.40:
        reasons.append("BROKER_REJECT_RISK_ELEVATED")
    if partial_norm >= 0.42:
        reasons.append("BROKER_PARTIAL_FILL_RISK_ELEVATED")
    if execution_state == "BLOCKED":
        reasons.append("EXECUTION_STATE_BLOCKED")
    elif execution_state == "STRESSED":
        reasons.append("EXECUTION_STATE_STRESSED")
    elif execution_state == "CAUTION":
        reasons.append("EXECUTION_STATE_CAUTION")

    unique_reasons: list[str] = []
    for reason in reasons:
        if reason not in unique_reasons:
            unique_reasons.append(reason)

    caution_lot_scale = float(lot_scales.get(execution_state.lower(), 1.0 if execution_state == "NORMAL" else 0.82) or 1.0)
    caution_enter_prob_buffer = float(enter_prob_buffers.get(execution_state.lower(), 0.0) or 0.0)

    return {
        "symbol": str(symbol or "").upper(),
        "session": str(session_label or "UNKNOWN").upper(),
        "regime": str(regime_label or "UNKNOWN").upper(),
        "spread_now_points": round(spread_now, 6),
        "spread_expected_points": round(spread_expected_points, 6),
        "spread_widening_risk": round(spread_widening_risk, 6),
        "expected_slippage_points": round(expected_slippage_points, 6),
        "slippage_risk": round(slippage_risk, 6),
        "fill_quality_score": round(fill_quality_score, 6),
        "latency_sensitivity_score": round(latency_sensitivity_score, 6),
        "liquidity_fragility_score": round(liquidity_fragility_score, 6),
        "execution_quality_score": round(execution_quality_score, 6),
        "execution_state": execution_state,
        "allowed_deviation_points": round(allowed_deviation_points, 6),
        "caution_lot_scale": round(caution_lot_scale, 6),
        "caution_enter_prob_buffer": round(caution_enter_prob_buffer, 6),
        "quality_flags": {
            "fallback_model_used": bool(tier.get("fallback_used", False)),
            "data_stale": bool(data_stale),
            "support_low": not bool(support_usable),
        },
        "reason_codes": unique_reasons,
    }
