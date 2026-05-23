#!/usr/bin/env python3
"""Generic FXAI PyTorch/TensorFlow backend entrypoint.

The Swift side sends JSON on stdin and expects a JSON response on stdout. This
module intentionally has a pure-Python fallback so contract tests can run on
machines before PyTorch or TensorFlow are installed. When those frameworks are
available, it uses Apple Silicon acceleration paths (`mps` for PyTorch and the
TensorFlow Metal plugin when installed by TensorFlow).
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import re
import sys
from typing import Any


VOLUME_FEATURE_INDEXES = (6, 68, 69, 70, 71, 74, 75, 76, 77, 78, 80, 81, 82, 83)
FXAI_PLUGIN_API_VERSION = 4
FXAI_TOKENIZER_API_VERSION = "fxai-tokenizer-v1"


def _safe_float(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(result):
        return 0.0
    return result


def _require_latest_api(command: dict[str, Any]) -> None:
    if int(command.get("apiVersion", -1)) != FXAI_PLUGIN_API_VERSION:
        raise ValueError(f"unsupported FXAI plugin API version; expected {FXAI_PLUGIN_API_VERSION}")
    payload = command.get("inference") or (command.get("training") or {}).get("inference") or {}
    if int(payload.get("apiVersion", -1)) != FXAI_PLUGIN_API_VERSION:
        raise ValueError(f"unsupported FXAI inference API version; expected {FXAI_PLUGIN_API_VERSION}")
    tokenizer = payload.get("tokenizerContract") or {}
    if str(tokenizer.get("version", "")) != FXAI_TOKENIZER_API_VERSION:
        raise ValueError(f"unsupported FXAI tokenizer API version; expected {FXAI_TOKENIZER_API_VERSION}")


def _feature(values: list[Any], index: int) -> float:
    if index < 0 or index >= len(values):
        return 0.0
    return _safe_float(values[index])


def _sanitize_features(values: list[Any], data_has_volume: bool) -> list[float]:
    features = [max(-8.0, min(_safe_float(value), 8.0)) for value in values]
    if not data_has_volume:
        for index in VOLUME_FEATURE_INDEXES:
            if index < len(features):
                features[index] = 0.0
    return features


def _framework_edge(framework: str, values: list[Any], data_has_volume: bool) -> float:
    if framework == "pyTorch":
        edge = _torch_edge(values)
        if edge is not None:
            return edge + (0.08 * _feature(values, 6) if data_has_volume else 0.0)
    if framework == "tensorFlow":
        edge = _tensorflow_edge(values)
        if edge is not None:
            return edge + (0.08 * _feature(values, 6) if data_has_volume else 0.0)
    return (
        0.36 * _feature(values, 0)
        + 0.28 * _feature(values, 3)
        + 0.22 * (_feature(values, 7) - _feature(values, 8))
        + (0.10 * _feature(values, 6) if data_has_volume else 0.0)
        + 0.04 * _feature(values, 12)
    )


def _state_dir() -> Path:
    raw = os.environ.get("FXAI_PLUGIN_STATE_DIR")
    if raw:
        return Path(raw).expanduser()
    return Path.home() / ".fxai" / "plugins" / "state"


def _safe_token(value: Any) -> str:
    token = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "model"))
    return token.strip("._") or "model"


def _state_path(model_identifier: Any, framework: Any) -> Path:
    return _state_dir() / f"{_safe_token(model_identifier)}.{_safe_token(framework)}.json"


def _initial_state() -> dict[str, Any]:
    return {
        "steps": 0,
        "moveEMA": 1.0,
        "classMass": [1.0e-6, 1.0e-6, 1.0e-6],
        "classCentroids": [],
    }


def _load_state(model_identifier: Any, framework: Any) -> dict[str, Any]:
    path = _state_path(model_identifier, framework)
    if not path.exists():
        return _initial_state()
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return _initial_state()
    if not isinstance(state, dict):
        return _initial_state()
    base = _initial_state()
    base.update(state)
    return base


def _save_state(model_identifier: Any, framework: Any, state: dict[str, Any]) -> None:
    path = _state_path(model_identifier, framework)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, separators=(",", ":"), sort_keys=True), encoding="utf-8")


def _ensure_centroids(state: dict[str, Any], feature_count: int) -> list[list[float]]:
    centroids = state.get("classCentroids")
    if not isinstance(centroids, list) or len(centroids) != 3:
        centroids = []
    normalized: list[list[float]] = []
    for class_index in range(3):
        source = centroids[class_index] if class_index < len(centroids) and isinstance(centroids[class_index], list) else []
        row = [_safe_float(source[index]) if index < len(source) else 0.0 for index in range(feature_count)]
        normalized.append(row)
    state["classCentroids"] = normalized
    return normalized


def _label_index(value: Any) -> int:
    if isinstance(value, int):
        return max(0, min(value, 2))
    if isinstance(value, str):
        lowered = value.lower()
        if "sell" in lowered:
            return 0
        if "buy" in lowered:
            return 1
        if "skip" in lowered:
            return 2
    return 2


def _state_edge(state: dict[str, Any], values: list[Any]) -> float:
    if int(_safe_float(state.get("steps", 0))) <= 0:
        return 0.0
    features = [_safe_float(value) for value in values]
    centroids = _ensure_centroids(state, len(features))
    buy = _cosine(features, centroids[1])
    sell = _cosine(features, centroids[0])
    skip = max(_cosine(features, centroids[2]), 0.0)
    return max(-0.35, min(0.35, 0.25 * (buy - sell) * (1.0 - 0.35 * skip)))


def _cosine(lhs: list[float], rhs: list[float]) -> float:
    dot = 0.0
    lhs_norm = 0.0
    rhs_norm = 0.0
    for left, right in zip(lhs, rhs):
        dot += left * right
        lhs_norm += left * left
        rhs_norm += right * right
    if lhs_norm <= 1.0e-12 or rhs_norm <= 1.0e-12:
        return 0.0
    return max(-1.0, min(1.0, dot / math.sqrt(lhs_norm * rhs_norm)))


def _torch_edge(values: list[Any]) -> float | None:
    try:
        import torch
    except Exception:
        return None
    device = "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu"
    indexes = [0, 3, 7, 8, 12]
    weights = torch.tensor([0.36, 0.28, 0.22, -0.22, 0.04], dtype=torch.float32, device=device)
    tensor = torch.tensor([_feature(values, idx) for idx in indexes], dtype=torch.float32, device=device)
    return float(torch.sum(tensor * weights).detach().cpu().item())


def _tensorflow_edge(values: list[Any]) -> float | None:
    try:
        import tensorflow as tf
    except Exception:
        return None
    indexes = [0, 3, 7, 8, 12]
    tensor = tf.constant([_feature(values, idx) for idx in indexes], dtype=tf.float32)
    weights = tf.constant([0.36, 0.28, 0.22, -0.22, 0.04], dtype=tf.float32)
    return float(tf.reduce_sum(tensor * weights).numpy())


def _prediction(edge: float, min_move: float, price_cost: float, move_ema: float = 1.0) -> dict[str, Any]:
    strength = max(0.0, min(abs(edge) * 3.5, 1.0))
    if strength < 0.08:
        return {
            "apiVersion": FXAI_PLUGIN_API_VERSION,
            "classProbabilities": [0.09, 0.09, 0.82],
            "moveMeanPoints": 0.0,
            "moveQ25Points": 0.0,
            "moveQ50Points": 0.0,
            "moveQ75Points": 0.0,
            "mfeMeanPoints": 0.0,
            "maeMeanPoints": 0.0,
            "hitTimeFraction": 1.0,
            "pathRisk": 0.82,
            "fillRisk": 0.0,
            "confidence": 0.09,
            "reliability": 0.48,
        }

    directional = max(0.54, min(0.54 + 0.38 * strength, 0.92))
    opposite = 0.06
    skip = max(0.06, 1.0 - directional - opposite)
    if edge >= 0.0:
        probs = [opposite, directional, skip]
    else:
        probs = [directional, opposite, skip]
    total = sum(probs)
    probs = [p / total for p in probs]
    move = max(1.0, min_move, price_cost, move_ema, abs(edge) * 100.0)
    sigma = max(0.10, 0.32 * move)
    return {
        "apiVersion": FXAI_PLUGIN_API_VERSION,
        "classProbabilities": probs,
        "moveMeanPoints": move,
        "moveQ25Points": max(0.0, move - 0.55 * sigma),
        "moveQ50Points": move,
        "moveQ75Points": move + 0.55 * sigma,
        "mfeMeanPoints": move,
        "maeMeanPoints": max(0.0, 0.35 * move),
        "hitTimeFraction": 1.0,
        "pathRisk": probs[2],
        "fillRisk": 0.0,
        "confidence": max(probs[0], probs[1]),
        "reliability": 0.52,
    }


def _handle_predict(command: dict[str, Any]) -> dict[str, Any]:
    payload = command.get("inference") or {}
    context_min_move = max(0.0, _safe_float(payload.get("minMovePoints", 0.0)))
    context_price_cost = max(0.0, _safe_float(payload.get("priceCostPoints", 0.0)))
    values = payload.get("x") or []
    data_has_volume = bool(payload.get("dataHasVolume", False))
    features = _sanitize_features(values, data_has_volume)
    framework = str(payload.get("framework", ""))
    state = _load_state(payload.get("modelIdentifier"), framework)
    edge = _framework_edge(framework, features, data_has_volume) + _state_edge(state, features)
    move_ema = max(1.0, _safe_float(state.get("moveEMA", 1.0)))
    return {
        "apiVersion": FXAI_PLUGIN_API_VERSION,
        "ok": True,
        "prediction": _prediction(edge, context_min_move, context_price_cost, move_ema),
        "error": None,
    }


def _handle_train(command: dict[str, Any]) -> dict[str, Any]:
    training = command.get("training") or {}
    inference = training.get("inference") or {}
    values = inference.get("x") or []
    features = _sanitize_features(values, bool(inference.get("dataHasVolume", False)))
    framework = str(inference.get("framework", ""))
    state = _load_state(inference.get("modelIdentifier"), framework)
    centroids = _ensure_centroids(state, len(features))
    masses = state.get("classMass")
    if not isinstance(masses, list) or len(masses) != 3:
        masses = [1.0e-6, 1.0e-6, 1.0e-6]
    masses = [max(_safe_float(value), 1.0e-6) for value in masses[:3]]
    label = _label_index(training.get("labelClass"))
    sample_weight = max(0.0, min(_safe_float(training.get("sampleWeight", 1.0)), 8.0))
    if sample_weight > 0.0:
        alpha = max(0.005, min(sample_weight / (masses[label] + sample_weight), 0.25))
        for index, feature in enumerate(features):
            centroids[label][index] = (1.0 - alpha) * centroids[label][index] + alpha * feature
        masses[label] = min(masses[label] + sample_weight, 1_000_000.0)
        move_target = max(abs(_safe_float(training.get("movePoints", 0.0))), _safe_float(inference.get("minMovePoints", 0.0)), 1.0)
        move_alpha = max(0.005, min(0.02 * sample_weight, 0.20))
        state["moveEMA"] = (1.0 - move_alpha) * _safe_float(state.get("moveEMA", 1.0)) + move_alpha * move_target
        state["steps"] = int(_safe_float(state.get("steps", 0))) + 1
    state["classMass"] = masses
    state["classCentroids"] = centroids
    _save_state(inference.get("modelIdentifier"), framework, state)
    return {"apiVersion": FXAI_PLUGIN_API_VERSION, "ok": True, "prediction": None, "error": None}


def main() -> int:
    operation = sys.argv[1] if len(sys.argv) > 1 else "predict"
    try:
        command = json.loads(sys.stdin.read() or "{}")
        _require_latest_api(command)
        if operation == "train" or command.get("operation") == "train":
            response = _handle_train(command)
        else:
            response = _handle_predict(command)
        print(json.dumps(response, separators=(",", ":")))
        return 0
    except Exception as exc:
        print(json.dumps({"apiVersion": FXAI_PLUGIN_API_VERSION, "ok": False, "prediction": None, "error": str(exc)}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
