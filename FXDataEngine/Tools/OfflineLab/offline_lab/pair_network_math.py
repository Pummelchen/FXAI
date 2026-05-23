from __future__ import annotations

import math
from collections.abc import Mapping


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(value)))


def clamp01(value: float) -> float:
    return clamp(value, 0.0, 1.0)


def safe_mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values)) / float(len(values))


def safe_pstdev(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    mean = safe_mean(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / float(len(values)))


def map_dot(lhs: Mapping[str, float], rhs: Mapping[str, float]) -> float:
    keys = set(lhs.keys()) | set(rhs.keys())
    return sum(float(lhs.get(key, 0.0)) * float(rhs.get(key, 0.0)) for key in keys)


def map_l2_norm(values: Mapping[str, float]) -> float:
    return math.sqrt(sum(float(value) ** 2 for value in values.values()))


def cosine_similarity(lhs: Mapping[str, float], rhs: Mapping[str, float]) -> float:
    left_norm = map_l2_norm(lhs)
    right_norm = map_l2_norm(rhs)
    if left_norm <= 0.0 or right_norm <= 0.0:
        return 0.0
    return clamp(map_dot(lhs, rhs) / (left_norm * right_norm), -1.0, 1.0)


def top_share(values: Mapping[str, float]) -> float:
    magnitudes = [abs(float(value)) for value in values.values() if abs(float(value)) > 0.0]
    if not magnitudes:
        return 0.0
    total = sum(magnitudes)
    if total <= 0.0:
        return 0.0
    return clamp01(max(magnitudes) / total)


def herfindahl_concentration(values: Mapping[str, float]) -> float:
    magnitudes = [abs(float(value)) for value in values.values() if abs(float(value)) > 0.0]
    if not magnitudes:
        return 0.0
    total = sum(magnitudes)
    if total <= 0.0:
        return 0.0
    return clamp01(sum((value / total) ** 2 for value in magnitudes))


def pearson_correlation(lhs: list[float], rhs: list[float]) -> tuple[float, int]:
    size = min(len(lhs), len(rhs))
    if size <= 1:
        return 0.0, size
    left = lhs[-size:]
    right = rhs[-size:]
    mean_left = safe_mean(left)
    mean_right = safe_mean(right)
    var_left = sum((value - mean_left) ** 2 for value in left)
    var_right = sum((value - mean_right) ** 2 for value in right)
    if var_left <= 0.0 or var_right <= 0.0:
        return 0.0, size
    cov = sum((a - mean_left) * (b - mean_right) for a, b in zip(left, right))
    return clamp(cov / math.sqrt(var_left * var_right), -1.0, 1.0), size
