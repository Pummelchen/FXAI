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
import sys
from typing import Any


def _safe_float(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(result):
        return 0.0
    return result


def _feature(values: list[Any], index: int) -> float:
    if index < 0 or index >= len(values):
        return 0.0
    return _safe_float(values[index])


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


def _prediction(edge: float, min_move: float, price_cost: float) -> dict[str, Any]:
    strength = max(0.0, min(abs(edge) * 3.5, 1.0))
    if strength < 0.08:
        return {
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
    move = max(1.0, min_move, price_cost, abs(edge) * 100.0)
    sigma = max(0.10, 0.32 * move)
    return {
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
    edge = _framework_edge(str(payload.get("framework", "")), values, bool(payload.get("dataHasVolume", False)))
    return {
        "ok": True,
        "prediction": _prediction(edge, context_min_move, context_price_cost),
        "error": None,
    }


def _handle_train(command: dict[str, Any]) -> dict[str, Any]:
    _ = command.get("training") or {}
    return {"ok": True, "prediction": None, "error": None}


def main() -> int:
    operation = sys.argv[1] if len(sys.argv) > 1 else "predict"
    try:
        command = json.loads(sys.stdin.read() or "{}")
        if operation == "train" or command.get("operation") == "train":
            response = _handle_train(command)
        else:
            response = _handle_predict(command)
        print(json.dumps(response, separators=(",", ":")))
        return 0
    except Exception as exc:
        print(json.dumps({"ok": False, "prediction": None, "error": str(exc)}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
