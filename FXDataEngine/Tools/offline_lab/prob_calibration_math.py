from __future__ import annotations

import math
from typing import Any


def clamp(value: float, low: float, high: float) -> float:
    return min(max(float(value), float(low)), float(high))


def sigmoid(value: float) -> float:
    if value >= 0.0:
        exp_neg = math.exp(-value)
        return 1.0 / (1.0 + exp_neg)
    exp_pos = math.exp(value)
    return exp_pos / (1.0 + exp_pos)


def logit(probability: float) -> float:
    p = clamp(probability, 1e-6, 1.0 - 1e-6)
    return math.log(p / (1.0 - p))


def select_calibration_tier(
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
        if kind == "REGIME":
            return rec_regime == regime
        if kind == "GLOBAL":
            return True
        return False

    for kind in hierarchy:
        kind_matches = [
            dict(record)
            for record in records
            if _matches(record)
            and str(record.get("kind", "") or "").upper() == kind
        ]
        if not kind_matches:
            continue

        preferred = [
            record
            for record in kind_matches
            if int(record.get("support", 0) or 0) >= int(soft_support_floor)
        ]
        fallback = [
            record
            for record in kind_matches
            if int(record.get("support", 0) or 0) >= int(hard_support_floor)
        ]

        candidates = preferred if preferred else fallback
        if not candidates:
            continue

        candidates.sort(
            key=lambda record: (
                int(record.get("support", 0) or 0),
                float(record.get("calibration_quality", 0.0) or 0.0),
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
        "prob_scale": 1.6,
        "prob_bias": 0.0,
        "skip_bias": 0.08,
        "move_mean_scale": 0.78,
        "move_q25_scale": 0.60,
        "move_q50_scale": 0.72,
        "move_q75_scale": 0.88,
        "calibration_quality": 0.34,
        "uncertainty_mult": 1.30,
        "confidence_cap": 0.58,
        "fallback_used": True,
        "support_usable": False,
    }


def calibrate_probabilities(
    *,
    raw_buy_prob: float,
    raw_sell_prob: float,
    raw_skip_prob: float,
    tier: dict[str, Any],
    config: dict[str, Any],
    uncertainty_score: float,
) -> dict[str, float]:
    soft = dict(config.get("soft_fallback", {}))
    raw_buy = clamp(raw_buy_prob, 0.0, 1.0)
    raw_sell = clamp(raw_sell_prob, 0.0, 1.0)
    raw_skip = clamp(raw_skip_prob, 0.0, 1.0)
    raw_mass = raw_buy + raw_sell + raw_skip
    if raw_mass <= 0.0:
        raw_buy, raw_sell, raw_skip = 0.0, 0.0, 1.0
    else:
        raw_buy /= raw_mass
        raw_sell /= raw_mass
        raw_skip /= raw_mass

    directional_mass = max(raw_buy + raw_sell, 1e-6)
    directional_share = raw_buy / directional_mass
    prob_scale = float(tier.get("prob_scale", soft.get("prob_scale", 1.6)) or 1.6)
    prob_bias = float(tier.get("prob_bias", 0.0) or 0.0)
    calibrated_dir_buy = sigmoid(prob_bias + prob_scale * logit(directional_share))

    calibration_quality = clamp(float(tier.get("calibration_quality", 0.0) or 0.0), 0.0, 1.0)
    neutral_blend = clamp(
        float(config.get("neutral_blend_gain", 0.65) or 0.65) * (1.0 - calibration_quality),
        0.0,
        0.85,
    )
    calibrated_dir_buy = neutral_blend * 0.5 + (1.0 - neutral_blend) * calibrated_dir_buy

    skip_bias = float(tier.get("skip_bias", soft.get("skip_bias", 0.08)) or 0.08)
    skip_uncertainty_gain = float(config.get("skip_uncertainty_gain", 0.12) or 0.12)
    skip_calibration_credit = float(config.get("skip_calibration_credit", 0.05) or 0.05)
    calibrated_skip = clamp(
        raw_skip + skip_bias + skip_uncertainty_gain * uncertainty_score - skip_calibration_credit * calibration_quality,
        float(config.get("skip_floor", 0.02) or 0.02),
        float(config.get("skip_cap", 0.96) or 0.96),
    )
    calibrated_directional_mass = max(1.0 - calibrated_skip, 1e-6)

    confidence_cap = clamp(
        float(tier.get("confidence_cap", soft.get("confidence_cap", 0.58)) or 0.58),
        0.50,
        0.95,
    )
    directional_distance = calibrated_dir_buy - 0.5
    max_distance = max(confidence_cap - 0.5, 0.0)
    calibrated_dir_buy = 0.5 + clamp(directional_distance, -max_distance, max_distance)

    calibrated_buy = calibrated_directional_mass * calibrated_dir_buy
    calibrated_sell = calibrated_directional_mass * (1.0 - calibrated_dir_buy)
    calibrated_skip = clamp(1.0 - calibrated_buy - calibrated_sell, 0.0, 1.0)
    confidence = max(calibrated_buy, calibrated_sell)
    directional_advantage = abs(calibrated_buy - calibrated_sell)
    raw_score = raw_buy - raw_sell

    return {
        "raw_score": raw_score,
        "calibrated_buy_prob": calibrated_buy,
        "calibrated_sell_prob": calibrated_sell,
        "calibrated_skip_prob": calibrated_skip,
        "calibrated_confidence": confidence,
        "directional_advantage": directional_advantage,
    }


def scale_move_distribution(
    *,
    move_mean_points: float,
    move_q25_points: float,
    move_q50_points: float,
    move_q75_points: float,
    tier: dict[str, Any],
    uncertainty_score: float,
) -> dict[str, float]:
    mean_scale = float(tier.get("move_mean_scale", 0.78) or 0.78)
    q25_scale = float(tier.get("move_q25_scale", 0.60) or 0.60)
    q50_scale = float(tier.get("move_q50_scale", 0.72) or 0.72)
    q75_scale = float(tier.get("move_q75_scale", 0.88) or 0.88)

    uncertainty_mean = clamp(1.0 - 0.18 * uncertainty_score, 0.35, 1.0)
    uncertainty_q25 = clamp(1.0 - 0.24 * uncertainty_score, 0.20, 1.0)
    uncertainty_q50 = clamp(1.0 - 0.16 * uncertainty_score, 0.25, 1.0)
    uncertainty_q75 = clamp(1.0 - 0.10 * uncertainty_score, 0.35, 1.0)

    q25 = max(float(move_q25_points or 0.0) * q25_scale * uncertainty_q25, 0.0)
    q50 = max(float(move_q50_points or 0.0) * q50_scale * uncertainty_q50, q25)
    q75 = max(float(move_q75_points or 0.0) * q75_scale * uncertainty_q75, q50)
    mean = max(float(move_mean_points or 0.0) * mean_scale * uncertainty_mean, q50)

    return {
        "expected_move_mean_points": mean,
        "expected_move_q25_points": q25,
        "expected_move_q50_points": q50,
        "expected_move_q75_points": q75,
    }


def compute_uncertainty_score(
    *,
    config: dict[str, Any],
    tier: dict[str, Any],
    min_move_points: float,
    expected_move_mean_points: float,
    expected_move_q25_points: float,
    expected_move_q75_points: float,
    agreement_score: float,
    news_risk_score: float,
    rates_risk_score: float,
    micro_risk_score: float,
    dynamic_abstain_bias: float,
    adaptive_abstain_bias: float,
    stale_context_count: int,
) -> dict[str, float]:
    penalties = dict(config.get("uncertainty_penalties", {}))
    quality = clamp(float(tier.get("calibration_quality", 0.0) or 0.0), 0.0, 1.0)
    support = max(int(tier.get("support", 0) or 0), 0)
    soft_floor = max(int(config.get("support_soft_floor", 64) or 64), 1)
    support_shortfall = clamp((soft_floor - support) / soft_floor, 0.0, 1.0)
    quality_shortfall = clamp(float(config.get("min_calibration_quality", 0.44) or 0.44) - quality, 0.0, 1.0)
    agreement_penalty = 1.0 - clamp(agreement_score, 0.0, 1.0)
    width = max(expected_move_q75_points - expected_move_q25_points, 0.0)
    width_ratio = clamp(width / max(expected_move_mean_points, min_move_points, 0.25), 0.0, 3.0) / 3.0

    uncertainty_score = (
        float(config.get("base_uncertainty_score", 0.18) or 0.18)
        + float(penalties.get("support", 0.34) or 0.34) * support_shortfall
        + float(penalties.get("quality", 0.28) or 0.28) * quality_shortfall
        + float(penalties.get("disagreement", 0.26) or 0.26) * agreement_penalty
        + float(penalties.get("distribution_width", 0.22) or 0.22) * width_ratio
        + float(penalties.get("news", 0.18) or 0.18) * clamp(news_risk_score, 0.0, 1.0)
        + float(penalties.get("rates", 0.14) or 0.14) * clamp(rates_risk_score, 0.0, 1.0)
        + float(penalties.get("micro", 0.24) or 0.24) * clamp(micro_risk_score, 0.0, 1.0)
        + float(penalties.get("dynamic_abstain", 0.20) or 0.20) * clamp(dynamic_abstain_bias, 0.0, 1.0)
        + float(penalties.get("adaptive_abstain", 0.22) or 0.22) * clamp(adaptive_abstain_bias, 0.0, 1.0)
        + float(penalties.get("stale_context", 0.16) or 0.16) * clamp(stale_context_count / 3.0, 0.0, 1.0)
    )
    uncertainty_score *= clamp(float(tier.get("uncertainty_mult", 1.0) or 1.0), 0.40, 2.50)
    uncertainty_penalty_points = max(min_move_points, 0.25) * uncertainty_score
    return {
        "uncertainty_score": uncertainty_score,
        "uncertainty_penalty_points": uncertainty_penalty_points,
        "support_shortfall": support_shortfall,
        "quality_shortfall": quality_shortfall,
        "distribution_width_ratio": width_ratio,
    }


def decide_action(
    *,
    upstream_action: str,
    calibrated_buy_prob: float,
    calibrated_sell_prob: float,
    expected_gross_edge_points: float,
    edge_after_costs_points: float,
    edge_floor_points: float,
    uncertainty_score: float,
    uncertainty_limit: float,
    calibration_quality: float,
    min_calibration_quality: float,
    support: int,
    support_hard_floor: int,
    raw_score: float,
    signal_zero_band: float,
    expected_move_q25_points: float,
    cost_floor_points: float,
    context_flags: dict[str, bool],
) -> dict[str, Any]:
    reasons: list[str] = []
    raw_direction = "BUY" if calibrated_buy_prob >= calibrated_sell_prob else "SELL"
    upstream = str(upstream_action or "SKIP").upper()

    if context_flags.get("calibration_stale"):
        reasons.append("CALIBRATION_STALE")
    if context_flags.get("input_stale"):
        reasons.append("INPUT_STALE")
    if support < support_hard_floor:
        reasons.append("SUPPORT_TOO_LOW")
    if calibration_quality < min_calibration_quality:
        reasons.append("CALIBRATION_WEAK")
    if abs(raw_score) < signal_zero_band:
        reasons.append("SIGNAL_TOO_CLOSE_TO_ZERO")
    if expected_move_q25_points <= cost_floor_points:
        reasons.append("MOVE_DISTRIBUTION_TOO_WEAK")
    if expected_gross_edge_points <= cost_floor_points:
        reasons.append("COST_TOO_HIGH")
    if uncertainty_score >= uncertainty_limit:
        reasons.append("UNCERTAINTY_TOO_HIGH")
    if edge_after_costs_points <= edge_floor_points:
        reasons.append("EDGE_TOO_SMALL")
    if context_flags.get("news_risk_block"):
        reasons.append("NEWS_RISK_BLOCK")
    if context_flags.get("microstructure_stress"):
        reasons.append("MICROSTRUCTURE_STRESS")
    if context_flags.get("rates_risk_block"):
        reasons.append("RATES_RISK_BLOCK")

    final_action = upstream
    abstain = False
    if upstream not in {"BUY", "SELL"}:
        final_action = "SKIP"
        abstain = True
    elif upstream != raw_direction and abs(calibrated_buy_prob - calibrated_sell_prob) >= 0.08:
        reasons.append("CALIBRATED_DIRECTION_CONFLICT")
        final_action = "SKIP"
        abstain = True
    elif reasons:
        final_action = "SKIP"
        abstain = True

    if not reasons and final_action == upstream:
        final_action = upstream
    return {
        "final_action": final_action,
        "abstain": abstain,
        "reason_codes": reasons,
    }
