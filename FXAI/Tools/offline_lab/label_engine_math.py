from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from statistics import median
from typing import Any


@dataclass(frozen=True)
class BarRecord:
    index: int
    time_unix: int
    open: float
    high: float
    low: float
    close: float
    spread_points: float
    tick_volume: int
    real_volume: int


@dataclass(frozen=True)
class HorizonSpec:
    horizon_id: str
    bars: int


@dataclass(frozen=True)
class CostSpec:
    spread_cost_points: float
    slippage_points: float
    fill_penalty_points: float
    commission_points: float
    safety_margin_points: float
    execution_penalty_points: float = 0.0
    news_penalty_points: float = 0.0

    @property
    def total_cost_points(self) -> float:
        return (
            self.spread_cost_points
            + self.slippage_points
            + self.fill_penalty_points
            + self.commission_points
            + self.safety_margin_points
            + self.execution_penalty_points
            + self.news_penalty_points
        )


@dataclass(frozen=True)
class SignalCandidate:
    signal_id: str
    sample_index: int
    bar_time_unix: int
    side: str
    raw_score: float
    source: str
    horizon_id: str | None = None
    execution_penalty_points: float = 0.0
    news_penalty_points: float = 0.0
    diagnostics: dict[str, Any] = field(default_factory=dict)


def clamp(value: float, low: float, high: float) -> float:
    return min(max(float(value), float(low)), float(high))


def safe_median(values: list[float]) -> float:
    filtered = [float(value) for value in values]
    return float(median(filtered)) if filtered else 0.0


def safe_mean(values: list[float]) -> float:
    filtered = [float(value) for value in values]
    return float(sum(filtered) / len(filtered)) if filtered else 0.0


def normalize_side(value: str | None) -> str:
    text = str(value or "").strip().upper()
    if text in {"LONG", "BUY", "1"}:
        return "LONG"
    if text in {"SHORT", "SELL", "-1"}:
        return "SHORT"
    return ""


def infer_point_size(symbol: str, last_price: float, config: dict[str, Any]) -> float:
    overrides = dict(config.get("symbol_point_overrides", {}))
    default_sizes = dict(config.get("default_point_sizes", {}))
    lookup_key = str(symbol or "").strip().upper()
    if lookup_key in overrides:
        return max(float(overrides[lookup_key] or 0.0), 1e-8)

    if len(lookup_key) == 6 and lookup_key.isalpha():
        quote = lookup_key[3:]
        if quote == "JPY" or float(last_price or 0.0) >= 20.0:
            return max(float(default_sizes.get("fx_jpy", 0.001) or 0.001), 1e-8)
        return max(float(default_sizes.get("fx", 0.00001) or 0.00001), 1e-8)

    if lookup_key.startswith(("XAU", "XAG", "XPT", "XPD")):
        return max(float(default_sizes.get("metal", 0.01) or 0.01), 1e-8)
    if lookup_key.endswith("USD") and lookup_key.startswith(("BTC", "ETH", "LTC")):
        return max(float(default_sizes.get("crypto", 0.01) or 0.01), 1e-8)
    if "." in lookup_key:
        return max(float(default_sizes.get("share", 0.01) or 0.01), 1e-8)
    if float(last_price or 0.0) >= 1000.0:
        return max(float(default_sizes.get("index", 0.10) or 0.10), 1e-8)
    return max(float(default_sizes.get("other", 0.0001) or 0.0001), 1e-8)


def build_horizon_specs(config: dict[str, Any]) -> list[HorizonSpec]:
    return [
        HorizonSpec(horizon_id=str(item["id"]), bars=int(item["bars"]))
        for item in list(config.get("horizons", []))
    ]


def session_label_from_unix(bar_time_unix: int) -> str:
    hour = int(bar_time_unix // 3600) % 24
    if 12 <= hour < 16:
        return "LONDON_NY_OVERLAP"
    if 7 <= hour < 12:
        return "LONDON"
    if 16 <= hour < 21:
        return "NEW_YORK"
    if 21 <= hour or hour < 1:
        return "ROLLOVER"
    return "ASIA"


def _side_price_delta(entry_price: float, price: float, side: str, point_size: float) -> float:
    if side == "LONG":
        return (float(price) - float(entry_price)) / point_size
    return (float(entry_price) - float(price)) / point_size


def _threshold_times(
    *,
    future_bars: list[BarRecord],
    entry_price: float,
    side: str,
    point_size: float,
    favorable_threshold_points: float,
    adverse_threshold_points: float,
) -> tuple[int | None, int | None, str]:
    favorable_at: int | None = None
    adverse_at: int | None = None
    collision = False
    for bar in future_bars:
        favorable_move = _side_price_delta(entry_price, bar.high if side == "LONG" else bar.low, side, point_size)
        adverse_move = _side_price_delta(entry_price, bar.low if side == "LONG" else bar.high, side, point_size)
        favorable_hit = favorable_move >= favorable_threshold_points
        adverse_hit = adverse_move <= -adverse_threshold_points
        if favorable_hit and favorable_at is None:
            favorable_at = bar.time_unix
        if adverse_hit and adverse_at is None:
            adverse_at = bar.time_unix
        if favorable_hit and adverse_hit:
            collision = True
        if favorable_at is not None and adverse_at is not None:
            break
    if favorable_at is None and adverse_at is None:
        return None, None, "NONE"
    if favorable_at is not None and adverse_at is None:
        return favorable_at, None, "FAVORABLE_ONLY"
    if favorable_at is None and adverse_at is not None:
        return None, adverse_at, "ADVERSE_ONLY"
    if favorable_at == adverse_at:
        return favorable_at, adverse_at, "COLLISION" if collision else "SAME_BAR"
    assert favorable_at is not None and adverse_at is not None
    return (
        favorable_at,
        adverse_at,
        "FAVORABLE_FIRST" if favorable_at < adverse_at else "ADVERSE_FIRST",
    )


def build_label_row(
    *,
    dataset_key: str,
    symbol: str,
    point_size: float,
    bar: BarRecord,
    future_bars: list[BarRecord],
    side: str,
    horizon: HorizonSpec,
    cost_spec: CostSpec,
    config: dict[str, Any],
    label_version: int,
) -> dict[str, Any]:
    entry_price = float(bar.close)
    terminal_bar = future_bars[-1]
    terminal_price = float(terminal_bar.close)
    forward_return_points = _side_price_delta(entry_price, terminal_price, side, point_size)
    forward_return_bps = ((terminal_price - entry_price) / max(entry_price, point_size)) * 10_000.0
    if side == "SHORT":
        forward_return_bps *= -1.0

    favorable_prices = [candidate.high if side == "LONG" else candidate.low for candidate in future_bars]
    adverse_prices = [candidate.low if side == "LONG" else candidate.high for candidate in future_bars]
    mfe_points = max(_side_price_delta(entry_price, price, side, point_size) for price in favorable_prices)
    mae_points = min(_side_price_delta(entry_price, price, side, point_size) for price in adverse_prices)
    mfe_bps = (mfe_points * point_size / max(entry_price, point_size)) * 10_000.0
    mae_bps = (mae_points * point_size / max(entry_price, point_size)) * 10_000.0
    total_cost_points = float(cost_spec.total_cost_points)
    cost_adjusted_return_points = forward_return_points - total_cost_points

    tradeability_cfg = dict(config.get("tradeability", {}))
    favorable_barrier_points = max(
        total_cost_points * float(tradeability_cfg.get("favorable_cost_mult", 1.25) or 1.25),
        float(tradeability_cfg.get("min_favorable_points", 2.0) or 2.0),
    )
    adverse_barrier_points = max(
        total_cost_points * float(tradeability_cfg.get("adverse_cost_mult", 1.0) or 1.0),
        float(tradeability_cfg.get("min_adverse_points", 2.0) or 2.0),
    )
    tradeability_threshold_points = max(
        total_cost_points * float(tradeability_cfg.get("tradeability_cost_mult", 1.10) or 1.10),
        favorable_barrier_points,
    )
    favorable_at, adverse_at, barrier_order_label = _threshold_times(
        future_bars=future_bars,
        entry_price=entry_price,
        side=side,
        point_size=point_size,
        favorable_threshold_points=favorable_barrier_points,
        adverse_threshold_points=adverse_barrier_points,
    )
    horizon_seconds = horizon.bars * 60
    time_to_favorable_hit_sec = max(favorable_at - bar.time_unix, 0) if favorable_at is not None else None
    time_to_adverse_hit_sec = max(adverse_at - bar.time_unix, 0) if adverse_at is not None else None
    favorable_before_timeout = (
        time_to_favorable_hit_sec is not None
        and time_to_favorable_hit_sec <= horizon_seconds * float(tradeability_cfg.get("max_time_to_favorable_ratio", 0.65) or 0.65)
    )
    require_favorable_before_adverse = bool(tradeability_cfg.get("require_favorable_before_adverse", True))
    favorable_before_adverse = (
        favorable_at is not None and (adverse_at is None or favorable_at < adverse_at)
    )
    max_mae_points = max(
        total_cost_points * float(tradeability_cfg.get("max_mae_cost_mult", 1.35) or 1.35),
        adverse_barrier_points,
    )
    direction_zero_band = float(config.get("direction_zero_band_points", 0.50) or 0.50)
    direction_label = 1 if forward_return_points > direction_zero_band else 0

    reason_codes: list[str] = []
    if cost_adjusted_return_points <= 0.0:
        reason_codes.append("MOVE_TOO_SMALL_AFTER_COSTS")
    if time_to_favorable_hit_sec is None:
        reason_codes.append("FAVORABLE_THRESHOLD_NOT_HIT")
    if time_to_favorable_hit_sec is not None and not favorable_before_timeout:
        reason_codes.append("FAVORABLE_HIT_TOO_SLOW")
    if require_favorable_before_adverse and adverse_at is not None and not favorable_before_adverse:
        reason_codes.append("ADVERSE_HIT_FIRST")
    if abs(mae_points) > max_mae_points:
        reason_codes.append("ADVERSE_EXCURSION_TOO_LARGE")
    if barrier_order_label in {"COLLISION", "SAME_BAR"}:
        reason_codes.append("PATH_ORDER_AMBIGUOUS")

    tradeability_label = 0 if reason_codes else 1
    label_quality_flags = {
        "path_approximation_used": True,
        "execution_proxy_used": bool(cost_spec.execution_penalty_points > 0.0 or cost_spec.news_penalty_points > 0.0),
        "partial_cost_model": bool(cost_spec.commission_points <= 0.0),
    }
    return {
        "sample_id": f"{dataset_key}:{bar.time_unix}:{side}:{horizon.horizon_id}",
        "timestamp_unix": int(bar.time_unix),
        "symbol": symbol,
        "dataset_key": dataset_key,
        "side": side,
        "session_label": session_label_from_unix(bar.time_unix),
        "horizon_id": horizon.horizon_id,
        "horizon_bars": int(horizon.bars),
        "direction_label": int(direction_label),
        "forward_return_points": round(forward_return_points, 6),
        "forward_return_bps": round(forward_return_bps, 6),
        "mfe_points": round(mfe_points, 6),
        "mae_points": round(mae_points, 6),
        "mfe_bps": round(mfe_bps, 6),
        "mae_bps": round(mae_bps, 6),
        "cost_adjusted_return_points": round(cost_adjusted_return_points, 6),
        "spread_cost_points": round(float(cost_spec.spread_cost_points), 6),
        "slippage_cost_points": round(float(cost_spec.slippage_points), 6),
        "fill_penalty_points": round(float(cost_spec.fill_penalty_points), 6),
        "commission_points": round(float(cost_spec.commission_points), 6),
        "execution_penalty_points": round(float(cost_spec.execution_penalty_points), 6),
        "news_penalty_points": round(float(cost_spec.news_penalty_points), 6),
        "safety_margin_points": round(float(cost_spec.safety_margin_points), 6),
        "total_cost_points": round(total_cost_points, 6),
        "favorable_barrier_points": round(favorable_barrier_points, 6),
        "adverse_barrier_points": round(adverse_barrier_points, 6),
        "tradeability_threshold_points": round(tradeability_threshold_points, 6),
        "time_to_favorable_hit_sec": time_to_favorable_hit_sec,
        "time_to_adverse_hit_sec": time_to_adverse_hit_sec,
        "favorable_hit_before_timeout": bool(favorable_before_timeout),
        "favorable_hit_before_adverse": bool(favorable_before_adverse),
        "barrier_order_label": barrier_order_label,
        "tradeability_label": int(tradeability_label),
        "reason_codes": reason_codes,
        "label_quality_flags": label_quality_flags,
        "label_version": int(label_version),
    }


def generate_baseline_candidates(
    *,
    dataset_key: str,
    bars: list[BarRecord],
    point_size: float,
    config: dict[str, Any],
) -> list[SignalCandidate]:
    meta_cfg = dict(config.get("meta_labeling", {}))
    lookback = max(int(meta_cfg.get("baseline_lookback_bars", 5) or 5), 1)
    threshold_points = float(meta_cfg.get("baseline_signal_threshold_points", 6.0) or 6.0)
    score_scale = float(meta_cfg.get("raw_score_scale_points", 20.0) or 20.0)
    candidates: list[SignalCandidate] = []
    for idx in range(lookback, len(bars)):
        bar = bars[idx]
        momentum_points = (float(bar.close) - float(bars[idx - lookback].close)) / point_size
        if abs(momentum_points) < threshold_points:
            continue
        side = "LONG" if momentum_points > 0 else "SHORT"
        raw_score = clamp(momentum_points / score_scale, -1.0, 1.0)
        candidates.append(
            SignalCandidate(
                signal_id=f"{dataset_key}:{bar.time_unix}:{side}:baseline",
                sample_index=idx,
                bar_time_unix=bar.time_unix,
                side=side,
                raw_score=raw_score,
                source="baseline_momentum",
                diagnostics={"lookback_bars": lookback, "momentum_points": round(momentum_points, 6)},
            )
        )
    return candidates


def load_external_candidates(
    *,
    dataset_key: str,
    path: Path,
    time_to_index: dict[int, int],
) -> list[SignalCandidate]:
    suffix = path.suffix.lower()
    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        records = payload if isinstance(payload, list) else []
    else:
        records = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            records.append(json.loads(line))
    candidates: list[SignalCandidate] = []
    for idx, record in enumerate(records):
        if not isinstance(record, dict):
            continue
        timestamp_unix = int(record.get("bar_time_unix", record.get("timestamp_unix", 0)) or 0)
        if timestamp_unix <= 0:
            continue
        sample_index = time_to_index.get(timestamp_unix)
        if sample_index is None:
            continue
        side = normalize_side(record.get("side") or record.get("signal"))
        if not side:
            continue
        diagnostics = record.get("diagnostics", {})
        if not isinstance(diagnostics, dict):
            diagnostics = {}
        candidates.append(
            SignalCandidate(
                signal_id=str(record.get("signal_id", "") or f"{dataset_key}:{timestamp_unix}:{side}:external:{idx}"),
                sample_index=sample_index,
                bar_time_unix=timestamp_unix,
                side=side,
                raw_score=clamp(float(record.get("raw_score", 0.0) or 0.0), -1.0, 1.0),
                source=str(record.get("source", "external_file") or "external_file"),
                horizon_id=str(record.get("horizon_id", "") or "").strip().upper() or None,
                execution_penalty_points=float(record.get("execution_penalty_points", 0.0) or 0.0),
                news_penalty_points=float(record.get("news_penalty_points", 0.0) or 0.0),
                diagnostics=diagnostics,
            )
        )
    return candidates


def summarize_horizon_rows(
    *,
    label_rows: list[dict[str, Any]],
    meta_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    long_rows = [row for row in label_rows if str(row.get("side")) == "LONG"]
    short_rows = [row for row in label_rows if str(row.get("side")) == "SHORT"]
    candidate_acceptance_rate = safe_mean([float(row.get("meta_label_trade", 0) or 0.0) for row in meta_rows])
    median_time = safe_median(
        [
            float(row["time_to_favorable_hit_sec"])
            for row in label_rows
            if row.get("time_to_favorable_hit_sec") is not None
        ]
    )
    return {
        "sample_count": len(label_rows),
        "long_sample_count": len(long_rows),
        "short_sample_count": len(short_rows),
        "long_positive_rate": round(safe_mean([float(row.get("direction_label", 0) or 0.0) for row in long_rows]), 6),
        "short_positive_rate": round(safe_mean([float(row.get("direction_label", 0) or 0.0) for row in short_rows]), 6),
        "long_tradeability_rate": round(safe_mean([float(row.get("tradeability_label", 0) or 0.0) for row in long_rows]), 6),
        "short_tradeability_rate": round(safe_mean([float(row.get("tradeability_label", 0) or 0.0) for row in short_rows]), 6),
        "mean_cost_adjusted_return_points": round(
            safe_mean([float(row.get("cost_adjusted_return_points", 0.0) or 0.0) for row in label_rows]),
            6,
        ),
        "median_mfe_points": round(safe_median([float(row.get("mfe_points", 0.0) or 0.0) for row in label_rows]), 6),
        "median_mae_points": round(safe_median([float(row.get("mae_points", 0.0) or 0.0) for row in label_rows]), 6),
        "median_time_to_favorable_hit_sec": round(median_time, 6) if median_time else None,
        "candidate_count": len(meta_rows),
        "candidate_acceptance_rate": round(candidate_acceptance_rate, 6),
    }


def top_reason_counts(rows: list[dict[str, Any]], *, limit: int = 8) -> list[dict[str, Any]]:
    counts: Counter[str] = Counter()
    for row in rows:
        for reason in list(row.get("reason_codes", [])):
            counts[str(reason)] += 1
    return [
        {"reason": reason, "count": int(count)}
        for reason, count in counts.most_common(limit)
    ]
