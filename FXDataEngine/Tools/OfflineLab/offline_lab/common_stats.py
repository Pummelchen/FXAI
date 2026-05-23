#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping

def plugin_family_name(family_id: int) -> str:
    mapping = {
        0: "linear",
        1: "tree",
        2: "recurrent",
        3: "convolutional",
        4: "transformer",
        5: "state_space",
        6: "distribution",
        7: "mixture",
        8: "memory",
        9: "world",
        10: "rule",
        11: "other",
    }
    return mapping.get(int(family_id), "other")


def mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    mean_v = sum(values) / float(len(values))
    if len(values) <= 1:
        return mean_v, 0.0
    var = sum((v - mean_v) * (v - mean_v) for v in values) / float(len(values))
    return mean_v, math.sqrt(max(var, 0.0))


def row_float(row: Mapping[str, object] | None, key: str, default: float = 0.0) -> float:
    if row is None:
        return float(default)
    try:
        raw = row.get(key, default) if hasattr(row, "get") else row[key]
    except Exception:
        raw = default
    try:
        return float(raw)
    except Exception:
        return float(default)


def solve_linear_system(matrix: list[list[float]], vector: list[float]) -> list[float] | None:
    n = len(matrix)
    if n <= 0 or n != len(vector):
        return None
    a = [[float(matrix[r][c]) for c in range(n)] for r in range(n)]
    b = [float(vector[r]) for r in range(n)]
    for pivot in range(n):
        best_row = pivot
        best_abs = abs(a[pivot][pivot])
        for row in range(pivot + 1, n):
            cand = abs(a[row][pivot])
            if cand > best_abs:
                best_abs = cand
                best_row = row
        if best_abs <= 1e-12:
            return None
        if best_row != pivot:
            a[pivot], a[best_row] = a[best_row], a[pivot]
            b[pivot], b[best_row] = b[best_row], b[pivot]
        pivot_val = a[pivot][pivot]
        inv_pivot = 1.0 / pivot_val
        for col in range(pivot, n):
            a[pivot][col] *= inv_pivot
        b[pivot] *= inv_pivot
        for row in range(n):
            if row == pivot:
                continue
            factor = a[row][pivot]
            if abs(factor) <= 1e-12:
                continue
            for col in range(pivot, n):
                a[row][col] -= factor * a[pivot][col]
            b[row] -= factor * b[pivot]
    return b


def fit_weighted_linear_model(rows: list[Mapping[str, object]],
                              feature_names: list[str],
                              target_name: str,
                              weight_name: str | None = None,
                              ridge: float = 1e-6) -> dict:
    dim = len(feature_names) + 1
    ata = [[0.0 for _ in range(dim)] for _ in range(dim)]
    atb = [0.0 for _ in range(dim)]
    used = 0
    total_weight = 0.0
    samples: list[tuple[list[float], float, float]] = []

    for row in rows:
        target = row_float(row, target_name, math.nan)
        if not math.isfinite(target):
            continue
        weight = row_float(row, weight_name, 1.0) if weight_name else 1.0
        weight = max(weight, 1e-9)
        vec = [1.0]
        valid = True
        for name in feature_names:
            value = row_float(row, name, math.nan)
            if not math.isfinite(value):
                valid = False
                break
            vec.append(value)
        if not valid:
            continue
        samples.append((vec, target, weight))
        used += 1
        total_weight += weight
        for i in range(dim):
            atb[i] += weight * vec[i] * target
            for j in range(i, dim):
                ata[i][j] += weight * vec[i] * vec[j]

    if used <= 0:
        return {
            "feature_names": list(feature_names),
            "intercept": 0.0,
            "coefficients": {name: 0.0 for name in feature_names},
            "mae": 0.0,
            "rmse": 0.0,
            "r2": 0.0,
            "used_rows": 0,
            "total_weight": 0.0,
        }

    for i in range(dim):
        ata[i][i] += float(ridge)
        for j in range(i):
            ata[i][j] = ata[j][i]

    coeff_vec = solve_linear_system(ata, atb)
    if coeff_vec is None:
        coeff_vec = [0.0 for _ in range(dim)]

    predictions: list[tuple[float, float, float]] = []
    weighted_target_sum = 0.0
    for _vec, target, weight in samples:
        weighted_target_sum += weight * target
    target_mean = weighted_target_sum / max(total_weight, 1e-9)
    for vec, target, weight in samples:
        pred = float(sum(coeff_vec[idx] * vec[idx] for idx in range(dim)))
        predictions.append((target, pred, weight))

    abs_err = 0.0
    sq_err = 0.0
    total_var = 0.0
    for target, pred, weight in predictions:
        diff = pred - target
        abs_err += weight * abs(diff)
        sq_err += weight * diff * diff
        centered = target - target_mean
        total_var += weight * centered * centered

    mae = abs_err / max(total_weight, 1e-9)
    rmse = math.sqrt(max(sq_err / max(total_weight, 1e-9), 0.0))
    r2 = 0.0
    if total_var > 1e-12:
        r2 = 1.0 - sq_err / total_var

    return {
        "feature_names": list(feature_names),
        "intercept": float(coeff_vec[0]),
        "coefficients": {
            feature_names[idx]: float(coeff_vec[idx + 1])
            for idx in range(len(feature_names))
        },
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2),
        "used_rows": int(used),
        "total_weight": float(total_weight),
    }


def predict_linear_model(model: dict, feature_values: dict[str, float], default: float = 0.0) -> float:
    if not isinstance(model, dict):
        return float(default)
    total = row_float(model, "intercept", default)
    coeffs = model.get("coefficients", {})
    if not isinstance(coeffs, dict):
        return float(total)
    for key, coeff in coeffs.items():
        try:
            value = float(feature_values.get(str(key), 0.0))
            total += float(coeff) * value
        except Exception:
            continue
    return float(total)


def param_identity_hash(row: dict) -> str:
    profile = str(row.get("profile_name", ""))
    symbol = str(row.get("symbol", ""))
    plugin = str(row.get("plugin_name", ""))
    params_json = str(row.get("parameters_json", "{}"))
    return sha256_text(f"{profile}|{symbol}|{plugin}|{params_json}")


def family_distillation_profile(family_id: int) -> dict:
    fam = int(family_id)
    if fam in (2, 3, 4, 5):
        return {
            "temperature": 1.35,
            "teacher_weight": 0.70,
            "student_weight": 0.30,
            "self_supervised_weight": 0.28,
            "analog_weight": 0.18,
            "foundation_weight": 0.32,
        }
    if fam in (0, 1, 6):
        return {
            "temperature": 1.15,
            "teacher_weight": 0.62,
            "student_weight": 0.38,
            "self_supervised_weight": 0.16,
            "analog_weight": 0.12,
            "foundation_weight": 0.18,
        }
    if fam in (7, 8, 9):
        return {
            "temperature": 1.28,
            "teacher_weight": 0.66,
            "student_weight": 0.34,
            "self_supervised_weight": 0.22,
            "analog_weight": 0.24,
            "foundation_weight": 0.24,
        }
    return {
        "temperature": 1.10,
        "teacher_weight": 0.58,
        "student_weight": 0.42,
        "self_supervised_weight": 0.10,
        "analog_weight": 0.08,
        "foundation_weight": 0.14,
    }


