from __future__ import annotations

import math
from collections import Counter
from typing import Iterable, Sequence


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(value)))


def mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / float(len(values)))


def stddev(values: Sequence[float]) -> float:
    if len(values) <= 1:
        return 0.0
    avg = mean(values)
    variance = sum((float(value) - avg) ** 2 for value in values) / float(len(values))
    return math.sqrt(max(variance, 0.0))


def downside_shift_score(reference_values: Sequence[float],
                         recent_values: Sequence[float],
                         warn_delta: float,
                         degrade_delta: float) -> float:
    if not reference_values or not recent_values:
        return 0.0
    delta = mean(reference_values) - mean(recent_values)
    if delta <= 0.0:
        return 0.0
    scale = max(float(degrade_delta) - float(warn_delta), 1e-9)
    return clamp((delta - float(warn_delta)) / scale, 0.0, 1.0)


def upside_shift_score(reference_values: Sequence[float],
                       recent_values: Sequence[float],
                       warn_delta: float,
                       degrade_delta: float) -> float:
    if not reference_values or not recent_values:
        return 0.0
    delta = mean(recent_values) - mean(reference_values)
    if delta <= 0.0:
        return 0.0
    scale = max(float(degrade_delta) - float(warn_delta), 1e-9)
    return clamp((delta - float(warn_delta)) / scale, 0.0, 1.0)


def normalized_gap(reference_value: float,
                   recent_value: float,
                   warn_delta: float,
                   degrade_delta: float) -> float:
    delta = abs(float(recent_value) - float(reference_value))
    scale = max(float(degrade_delta) - float(warn_delta), 1e-9)
    return clamp((delta - float(warn_delta)) / scale, 0.0, 1.0)


def _histogram(values: Sequence[float], edges: Sequence[float]) -> list[int]:
    if len(edges) < 2:
        return [len(values)]
    counts = [0 for _ in range(len(edges) - 1)]
    for raw_value in values:
        value = float(raw_value)
        bucket_index = len(counts) - 1
        for index in range(len(edges) - 1):
            lower = float(edges[index])
            upper = float(edges[index + 1])
            is_last = index == len(edges) - 2
            if value < upper or is_last:
                if value >= lower or index == 0:
                    bucket_index = index
                    break
        counts[bucket_index] += 1
    return counts


def population_stability_index(reference_values: Sequence[float],
                               recent_values: Sequence[float],
                               *,
                               bucket_count: int = 6,
                               epsilon: float = 1e-6) -> float:
    if len(reference_values) < 2 or len(recent_values) < 2:
        return 0.0
    ref_sorted = sorted(float(value) for value in reference_values)
    if ref_sorted[0] == ref_sorted[-1]:
        return normalized_gap(ref_sorted[0], mean(recent_values), 0.05, 0.20)

    bucket_count = max(int(bucket_count), 3)
    edges: list[float] = [ref_sorted[0]]
    for step in range(1, bucket_count):
        quantile = int(round((len(ref_sorted) - 1) * step / float(bucket_count)))
        edges.append(ref_sorted[quantile])
    edges.append(ref_sorted[-1])

    # Enforce monotonic bucket edges so repeated values do not create zero-width bins.
    for index in range(1, len(edges)):
        if edges[index] <= edges[index - 1]:
            edges[index] = edges[index - 1] + epsilon

    ref_counts = _histogram(reference_values, edges)
    recent_counts = _histogram(recent_values, edges)
    ref_total = max(sum(ref_counts), 1)
    recent_total = max(sum(recent_counts), 1)
    psi = 0.0
    for ref_count, recent_count in zip(ref_counts, recent_counts):
        ref_pct = max(ref_count / float(ref_total), epsilon)
        recent_pct = max(recent_count / float(recent_total), epsilon)
        psi += (recent_pct - ref_pct) * math.log(recent_pct / ref_pct)
    return max(float(psi), 0.0)


def regime_mix_shift_score(reference_regimes: Sequence[int], recent_regimes: Sequence[int]) -> float:
    if not reference_regimes or not recent_regimes:
        return 0.0
    ref_counts = Counter(int(item) for item in reference_regimes)
    recent_counts = Counter(int(item) for item in recent_regimes)
    keys = sorted(set(ref_counts) | set(recent_counts))
    ref_total = max(sum(ref_counts.values()), 1)
    recent_total = max(sum(recent_counts.values()), 1)
    return clamp(
        0.5 * sum(
            abs(ref_counts.get(key, 0) / float(ref_total) - recent_counts.get(key, 0) / float(recent_total))
            for key in keys
        ),
        0.0,
        1.0,
    )


def weighted_average(values: Iterable[tuple[float, float]]) -> float:
    numerator = 0.0
    denominator = 0.0
    for value, weight in values:
        weight_f = max(float(weight), 0.0)
        numerator += float(value) * weight_f
        denominator += weight_f
    if denominator <= 0.0:
        return 0.0
    return numerator / denominator


def aggregate_drift_score(scores: dict[str, float], weights: dict[str, float]) -> float:
    return clamp(
        weighted_average(
            (float(scores.get(key, 0.0) or 0.0), float(weight or 0.0))
            for key, weight in weights.items()
        ),
        0.0,
        1.0,
    )


def severity_rank(state: str) -> int:
    ordering = {
        "HEALTHY": 0,
        "CAUTION": 1,
        "DEGRADED": 2,
        "SHADOW_ONLY": 3,
        "DEMOTED": 4,
        "DISABLED": 5,
    }
    return ordering.get(str(state or "").upper(), 0)
