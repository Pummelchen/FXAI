from __future__ import annotations

import math
from collections.abc import Sequence
from datetime import datetime, timezone
from typing import Any


def clamp(value: float, min_v: float, max_v: float) -> float:
    return max(min_v, min(max_v, value))


def tick_imbalance_from_changes(changes_points: Sequence[float]) -> float:
    up = sum(1 for value in changes_points if value > 0)
    down = sum(1 for value in changes_points if value < 0)
    total = max(up + down, 1)
    return clamp((up - down) / total, -1.0, 1.0)


def directional_efficiency_from_changes(changes_points: Sequence[float]) -> float:
    total_abs = sum(abs(value) for value in changes_points)
    if total_abs <= 1e-9:
        return 0.0
    return clamp(abs(sum(changes_points)) / total_abs, 0.0, 1.0)


def spread_metrics(
    spreads: Sequence[float],
    *,
    wide_spread_zscore: float,
    wide_spread_absolute_points_floor: float,
) -> dict[str, float]:
    if not spreads:
        return {
            "spread_current": 0.0,
            "spread_mean": 0.0,
            "spread_std": 0.0,
            "spread_zscore": 0.0,
            "spread_widen_events": 0.0,
            "wide_spread_fraction": 0.0,
            "spread_instability": 0.0,
        }

    mean = sum(spreads) / len(spreads)
    variance = max(sum(value * value for value in spreads) / len(spreads) - mean * mean, 0.0)
    std = math.sqrt(variance)
    current = float(spreads[-1])
    zscore = (current - mean) / std if std > 1e-9 else 0.0
    widen_events = 0
    for previous, current_value in zip(spreads, spreads[1:]):
        if (current_value - previous) > max(0.35 * max(previous, 1.0), 0.5):
            widen_events += 1
    wide_threshold = max(mean + wide_spread_zscore * std, wide_spread_absolute_points_floor)
    wide_fraction = sum(1 for value in spreads if value >= wide_threshold) / max(len(spreads), 1)
    instability = clamp(
        0.38 * clamp((std / max(mean, 0.25)) / 2.0, 0.0, 4.0)
        + 0.34 * wide_fraction
        + 0.28 * clamp(widen_events / max(len(spreads) - 1, 1), 0.0, 1.0),
        0.0,
        1.0,
    )
    return {
        "spread_current": current,
        "spread_mean": mean,
        "spread_std": std,
        "spread_zscore": zscore,
        "spread_widen_events": float(widen_events),
        "wide_spread_fraction": wide_fraction,
        "spread_instability": instability,
    }


def burst_scores(current_value: float, baseline_values: Sequence[float]) -> dict[str, float]:
    if not baseline_values:
        return {"mean": current_value, "std": 0.0, "zscore": 0.0, "ratio": 1.0}

    mean = sum(baseline_values) / len(baseline_values)
    variance = max(sum(value * value for value in baseline_values) / len(baseline_values) - mean * mean, 0.0)
    std = math.sqrt(variance)
    denominator = max(std, mean * 0.25, 1e-6)
    return {
        "mean": mean,
        "std": std,
        "zscore": clamp((current_value - mean) / denominator, -8.0, 8.0),
        "ratio": clamp(current_value / max(mean, 1e-6), 0.0, 8.0),
    }


def detect_sweep_and_reject(
    midpoints: Sequence[float],
    *,
    point_value: float,
    spread_current: float,
    shock_move_points_factor: float,
    stop_run_reversal_fraction: float,
    directional_efficiency: float,
    shock_move_count: int = 0,
) -> dict[str, float | bool]:
    if len(midpoints) < 2 or point_value <= 0.0:
        return {
            "local_extrema_breach_score": 0.0,
            "sweep_and_reject_flag": False,
            "breakout_reversal_score": 0.0,
            "exhaustion_proxy": 0.0,
        }

    maximum = max(midpoints)
    minimum = min(midpoints)
    last = midpoints[-1]
    max_index = max(range(len(midpoints)), key=midpoints.__getitem__)
    min_index = min(range(len(midpoints)), key=midpoints.__getitem__)
    previous_high = max(midpoints[:max_index]) if max_index > 0 else maximum
    previous_low = min(midpoints[:min_index]) if min_index > 0 else minimum
    range_expansion = max((maximum - minimum) / point_value, 0.0)
    breach_up = max(0.0, (maximum - previous_high) / point_value)
    breach_down = max(0.0, (previous_low - minimum) / point_value)
    breach_norm = clamp(max(breach_up, breach_down) / max(range_expansion, 1.0), 0.0, 1.0)

    rejection_score = 0.0
    rejection_flag = False
    snapback = clamp(stop_run_reversal_fraction, 0.10, 0.90)
    threshold = max(shock_move_points_factor * max(spread_current, 1.0), 2.0)

    if breach_up > threshold:
        rejection = max(0.0, (maximum - last) / point_value)
        if rejection >= breach_up * snapback:
            rejection_flag = True
            rejection_score = max(rejection_score, clamp(rejection / max(breach_up, 1.0), 0.0, 1.0))
    if breach_down > threshold:
        rejection = max(0.0, (last - minimum) / point_value)
        if rejection >= breach_down * snapback:
            rejection_flag = True
            rejection_score = max(rejection_score, clamp(rejection / max(breach_down, 1.0), 0.0, 1.0))

    breakout_reversal = clamp(0.55 * breach_norm + 0.45 * rejection_score, 0.0, 1.0)
    exhaustion = clamp(
        0.46 * breakout_reversal
        + 0.30 * (1.0 - directional_efficiency)
        + 0.24 * clamp(shock_move_count / max(len(midpoints), 1), 0.0, 1.0),
        0.0,
        1.0,
    )
    return {
        "local_extrema_breach_score": breach_norm,
        "sweep_and_reject_flag": rejection_flag,
        "breakout_reversal_score": breakout_reversal,
        "exhaustion_proxy": exhaustion,
    }


def resolve_session(now_dt: datetime, session_model: dict[str, Any]) -> dict[str, Any]:
    if now_dt.tzinfo is None:
        now_dt = now_dt.replace(tzinfo=timezone.utc)
    minute_of_day = now_dt.hour * 60 + now_dt.minute
    handoff_minutes = int(session_model.get("handoff_minutes", 20) or 20)
    sessions = session_model.get("sessions", {})
    for label, spec in sessions.items():
        if not isinstance(spec, dict):
            continue
        start_min = int(spec.get("start_hour", 0) or 0) * 60
        end_min = int(spec.get("end_hour", 0) or 0) * 60
        if end_min > start_min:
            in_session = start_min <= minute_of_day < end_min
        else:
            in_session = minute_of_day >= start_min or minute_of_day < end_min
        if not in_session:
            continue
        minutes_since_open = minute_of_day - start_min
        if minutes_since_open < 0:
            minutes_since_open += 1440
        minutes_to_close = end_min - minute_of_day
        if minutes_to_close < 0:
            minutes_to_close += 1440
        return {
            "session_tag": label,
            "handoff_flag": minutes_since_open <= max(handoff_minutes, 5) or minutes_to_close <= max(handoff_minutes, 5),
            "minutes_since_session_open": minutes_since_open,
            "minutes_to_session_close": minutes_to_close,
        }
    return {
        "session_tag": "UNKNOWN",
        "handoff_flag": False,
        "minutes_since_session_open": None,
        "minutes_to_session_close": None,
    }


def classify_microstructure_state(
    *,
    spread_instability: float,
    spread_zscore_60s: float,
    wide_spread_fraction_60s: float,
    session_spread_behavior_score: float,
    vol_burst_score_5m: float,
    intensity_burst_score_30s: float,
    silent_gap_seconds_current: float,
    handoff_flag: bool,
    local_extrema_breach_score_60s: float,
    breakout_reversal_score_60s: float,
    exhaustion_proxy_30s: float,
    directional_efficiency_60s: float,
    tick_imbalance_30s: float,
    session_open_burst_score: float,
    thresholds: dict[str, float],
) -> dict[str, Any]:
    sweep_risk = max(breakout_reversal_score_60s, exhaustion_proxy_30s)
    spread_level_stress = clamp(max(spread_zscore_60s, 0.0) / 3.0, 0.0, 1.0)
    thin_and_wide_flag = (
        spread_instability >= thresholds["spread_instability_block"]
        or (spread_level_stress >= 0.45 and wide_spread_fraction_60s >= 0.10)
        or (silent_gap_seconds_current >= 8.0 and spread_level_stress >= 0.35)
    )
    liquidity_stress_score = clamp(
        0.24 * spread_instability
        + 0.18 * spread_level_stress
        + 0.20 * session_spread_behavior_score
        + 0.15 * clamp(vol_burst_score_5m - 1.0, 0.0, 3.0) / 3.0
        + 0.13 * clamp(intensity_burst_score_30s - 1.0, 0.0, 3.0) / 3.0
        + 0.10 * clamp(silent_gap_seconds_current / 10.0, 0.0, 1.0),
        0.0,
        1.0,
    )
    hostile_execution_score = clamp(
        0.34 * liquidity_stress_score
        + 0.18 * spread_instability
        + 0.12 * spread_level_stress
        + 0.18 * sweep_risk
        + 0.10 * (1.0 if handoff_flag else 0.0)
        + 0.08 * clamp(vol_burst_score_5m - 1.0, 0.0, 3.0) / 3.0,
        0.0,
        1.0,
    )

    if hostile_execution_score >= thresholds["hostile_execution_block"] or liquidity_stress_score >= thresholds["liquidity_stress_block"]:
        trade_gate = "BLOCK"
    elif hostile_execution_score >= thresholds["hostile_execution_caution"] or liquidity_stress_score >= thresholds["liquidity_stress_caution"]:
        trade_gate = "CAUTION"
    else:
        trade_gate = "ALLOW"
    if thin_and_wide_flag and trade_gate == "ALLOW":
        trade_gate = "CAUTION"

    trend_eff = directional_efficiency_60s
    trend_imbalance = abs(tick_imbalance_30s)
    reasons: list[str] = []

    if thin_and_wide_flag:
        regime = "THIN_AND_WIDE"
    elif sweep_risk >= thresholds["stop_run_rejection_score_flag"]:
        regime = "STOP_RUN_RISK"
    elif (
        trend_eff >= thresholds["clean_trend_efficiency_floor"]
        and trend_imbalance >= thresholds["clean_trend_imbalance_floor"]
        and hostile_execution_score < 0.40
    ):
        regime = "TRENDING_CLEAN"
    elif trend_eff >= thresholds["clean_trend_efficiency_floor"] and hostile_execution_score >= 0.40:
        regime = "TRENDING_FRAGILE"
    elif (
        intensity_burst_score_30s >= thresholds["tick_burst_ratio_caution"] and trend_eff < 0.45
    ) or (
        vol_burst_score_5m >= thresholds["vol_burst_ratio_caution"] and trend_eff < 0.45
    ):
        regime = "CHOPPY_HIGH_ACTIVITY"
    elif (
        intensity_burst_score_30s >= thresholds["tick_burst_ratio_block"]
        and vol_burst_score_5m >= thresholds["vol_burst_ratio_caution"]
    ) or session_open_burst_score >= 0.65:
        regime = "VOLATILE_NEWSLIKE"
    else:
        regime = "NORMAL"

    if trade_gate == "BLOCK":
        reasons.append("hostile execution block threshold exceeded")
    elif trade_gate == "CAUTION":
        reasons.append("hostile execution caution threshold exceeded")
    if spread_instability >= thresholds["spread_instability_caution"]:
        reasons.append("spread instability elevated")
    if thin_and_wide_flag:
        reasons.append("current spread regime is abnormally wide")
    if intensity_burst_score_30s >= thresholds["tick_burst_ratio_caution"]:
        reasons.append("tick intensity burst above baseline")
    if vol_burst_score_5m >= thresholds["vol_burst_ratio_caution"]:
        reasons.append("realized volatility burst above baseline")
    if breakout_reversal_score_60s >= thresholds["stop_run_rejection_score_flag"]:
        reasons.append("recent breakout rejection detected")
    if handoff_flag and session_open_burst_score >= 0.45:
        reasons.append("session handoff burst active")

    return {
        "liquidity_stress_score": liquidity_stress_score,
        "hostile_execution_score": hostile_execution_score,
        "microstructure_regime": regime,
        "trade_gate": trade_gate,
        "reasons": reasons,
        "sweep_risk": sweep_risk,
        "breakout_pressure": local_extrema_breach_score_60s,
    }
