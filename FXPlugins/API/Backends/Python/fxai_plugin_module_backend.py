#!/usr/bin/env python3
"""FXAI plugin-local Python backend dispatcher.

The Swift bridge sends the same JSON contract used by `fxai_plugin_backend.py`.
This dispatcher loads the plugin's own PyTorch, TensorFlow, or NLP module from
its folder and converts the module result back into `PredictionV4` JSON.
"""

from __future__ import annotations

import importlib.util
import json
import math
import os
from pathlib import Path
import pickle
import re
import sys
from typing import Any


VOLUME_FEATURE_INDEXES = (6, 68, 69, 70, 71, 74, 75, 76, 77, 78, 80, 81, 82, 83)


def _safe_float(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(result):
        return 0.0
    return result


def _safe_token(value: Any) -> str:
    token = re.sub(r"[^A-Za-z0-9_.:-]+", "_", str(value or "model"))
    return token.strip("._:-") or "model"


def _plugin_candidates(model_identifier: Any) -> list[str]:
    token = _safe_token(model_identifier)
    candidates = [token]
    for separator in (":", "."):
        if separator in token:
            candidates.append(token.split(separator, 1)[0])
    return list(dict.fromkeys(candidates))


def _plugins_root() -> Path:
    raw = os.environ.get("FXAI_PLUGIN_ROOT")
    if raw:
        return Path(raw).expanduser().resolve()
    return Path(__file__).resolve().parents[3]


def _state_dir() -> Path:
    raw = os.environ.get("FXAI_PLUGIN_STATE_DIR")
    if raw:
        return Path(raw).expanduser()
    return Path.home() / ".fxai" / "plugins" / "state"


def _backend_path(plugin_name: str, framework: str) -> Path:
    root = _plugins_root()
    if framework == "pyTorch":
        return root / plugin_name / "PyTorch" / f"{plugin_name}_torch.py"
    if framework == "tensorFlow":
        return root / plugin_name / "TensorFlow" / f"{plugin_name}_tensorflow.py"
    if framework == "foundationNLP":
        return root / plugin_name / "NLP" / f"{plugin_name}_nlp.py"
    raise ValueError(f"unsupported framework {framework}")


def _resolve_backend_path(model_identifier: Any, framework: str) -> tuple[str, Path]:
    for candidate in _plugin_candidates(model_identifier):
        path = _backend_path(candidate, framework)
        if path.exists():
            return candidate, path
    raise FileNotFoundError(f"no plugin-local {framework} backend for {model_identifier}")


def _load_module(plugin_name: str, path: Path) -> Any:
    module_name = f"fxai_{plugin_name}_{path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _prepare_framework(framework: str) -> None:
    if framework == "pyTorch" and os.environ.get("FXAI_FORCE_PYTORCH_CPU") == "1":
        import torch

        if getattr(torch.backends, "mps", None):
            torch.backends.mps.is_available = lambda: False


def _state_path(plugin_name: str, framework: str, model_identifier: Any) -> Path:
    return _state_dir() / f"{plugin_name}.{framework}.{_safe_token(model_identifier)}.pkl"


def _load_state(plugin_name: str, framework: str, model_identifier: Any) -> Any | None:
    path = _state_path(plugin_name, framework, model_identifier)
    if not path.exists():
        return None
    try:
        if framework == "pyTorch":
            import torch

            return torch.load(path, map_location="cpu", weights_only=False)
        with path.open("rb") as handle:
            return pickle.load(handle)
    except Exception:
        return None


def _save_state(plugin_name: str, framework: str, model_identifier: Any, state: Any) -> None:
    if state is None:
        return
    path = _state_path(plugin_name, framework, model_identifier)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        if framework == "pyTorch":
            import torch

            torch.save(state, path)
            return
        with path.open("wb") as handle:
            pickle.dump(state, handle)
    except Exception:
        # Persistence is best-effort because TensorFlow objects are not always pickle-safe.
        return


def _features(payload: dict[str, Any]) -> list[float]:
    data_has_volume = bool(payload.get("dataHasVolume", False))
    raw = payload.get("x") or []
    values = [max(-8.0, min(_safe_float(value), 8.0)) for value in raw]
    if not data_has_volume:
        for index in VOLUME_FEATURE_INDEXES:
            if index < len(values):
                values[index] = 0.0
    return values


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


def _first_row(value: Any) -> list[float]:
    if not isinstance(value, list):
        return []
    if value and isinstance(value[0], list):
        return [_safe_float(item) for item in value[0]]
    return [_safe_float(item) for item in value]


def _normalize_probabilities(value: Any) -> list[float]:
    raw = _first_row(value)[:3]
    if len(raw) < 3:
        raw.extend([0.0] * (3 - len(raw)))
    raw = [max(0.0, item) for item in raw]
    total = sum(raw)
    if total <= 1.0e-12:
        return [0.09, 0.09, 0.82]
    return [item / total for item in raw]


def _first_number(value: Any, default: float = 0.0) -> float:
    if isinstance(value, list):
        if not value:
            return default
        if isinstance(value[0], list):
            return _safe_float(value[0][0] if value[0] else default)
        return _safe_float(value[0])
    return _safe_float(value)


def _quantiles(value: Any, move: float) -> tuple[float, float, float]:
    row = _first_row(value)
    if len(row) >= 4:
        return max(0.0, row[1]), max(0.0, row[2]), max(0.0, row[3])
    sigma = max(0.10, 0.32 * move)
    return max(0.0, move - 0.55 * sigma), move, move + 0.55 * sigma


def _edge_prediction(edge: float, payload: dict[str, Any]) -> dict[str, Any]:
    strength = max(0.0, min(abs(edge) * 3.5, 1.0))
    if strength < 0.08:
        return _prediction([0.09, 0.09, 0.82], 0.0, [0.0, 0.0, 0.0])
    directional = max(0.54, min(0.54 + 0.38 * strength, 0.92))
    opposite = 0.06
    skip = max(0.06, 1.0 - directional - opposite)
    probabilities = [opposite, directional, skip] if edge >= 0.0 else [directional, opposite, skip]
    total = sum(probabilities)
    probabilities = [item / total for item in probabilities]
    move = max(
        1.0,
        abs(edge) * 100.0,
        _safe_float(payload.get("minMovePoints", 0.0)),
        _safe_float(payload.get("priceCostPoints", 0.0)),
    )
    sigma = max(0.10, 0.32 * move)
    return _prediction(probabilities, move, [max(0.0, move - 0.55 * sigma), move, move + 0.55 * sigma])


def _prediction(probabilities: list[float], move: float, quantiles: list[float]) -> dict[str, Any]:
    q25, q50, q75 = max(0.0, quantiles[0]), max(0.0, quantiles[1]), max(0.0, quantiles[2])
    q50 = max(q50, q25)
    q75 = max(q75, q50)
    return {
        "classProbabilities": probabilities,
        "moveMeanPoints": max(0.0, move),
        "moveQ25Points": q25,
        "moveQ50Points": q50,
        "moveQ75Points": q75,
        "mfeMeanPoints": max(0.0, move),
        "maeMeanPoints": max(0.0, 0.35 * move),
        "hitTimeFraction": 1.0,
        "pathRisk": max(0.0, min(probabilities[2], 1.0)),
        "fillRisk": 0.0,
        "confidence": max(probabilities[0], probabilities[1]),
        "reliability": 0.58,
    }


def _prediction_from_module_result(result: dict[str, Any]) -> dict[str, Any]:
    probabilities = _normalize_probabilities(result.get("class_probabilities"))
    move = max(0.0, _first_number(result.get("move_mean_points"), 0.0))
    q25, q50, q75 = _quantiles(result.get("move_quantiles"), move)
    return _prediction(probabilities, move, [q25, q50, q75])


def _nlp_prediction(module: Any, features: list[float], payload: dict[str, Any]) -> dict[str, Any]:
    texts = payload.get("texts") or payload.get("eventTexts") or []
    if isinstance(texts, str):
        texts = [texts]
    merged = module.merge_into_numeric_features(features, texts)
    edge = (
        0.32 * _safe_float(merged[55] if len(merged) > 55 else 0.0)
        + 0.18 * _safe_float(merged[56] if len(merged) > 56 else 0.0)
        + 0.18 * _safe_float(merged[57] if len(merged) > 57 else 0.0)
        + 0.12 * _safe_float(merged[59] if len(merged) > 59 else 0.0)
        + 0.08 * _safe_float(merged[60] if len(merged) > 60 else 0.0)
    )
    return _edge_prediction(edge, payload)


def _call_train_step(module: Any, features: list[float], label: int, move: float, state: Any, data_has_volume: bool) -> Any:
    try:
        return module.train_step([features], [label], [move], state=state, data_has_volume=data_has_volume)
    except TypeError:
        return module.train_step([features], [label], [move], state=state)


def _call_predict_batch(module: Any, features: list[float], state: Any, data_has_volume: bool) -> dict[str, Any]:
    try:
        return module.predict_batch([features], state=state, data_has_volume=data_has_volume)
    except TypeError:
        return module.predict_batch([features], state=state)


def _handle_predict(command: dict[str, Any]) -> dict[str, Any]:
    payload = command.get("inference") or {}
    framework = str(payload.get("framework", ""))
    model_identifier = payload.get("modelIdentifier")
    plugin_name, backend_path = _resolve_backend_path(model_identifier, framework)
    _prepare_framework(framework)
    module = _load_module(plugin_name, backend_path)
    features = _features(payload)
    if framework == "foundationNLP":
        prediction = _nlp_prediction(module, features, payload)
    else:
        state = _load_state(plugin_name, framework, model_identifier)
        result = _call_predict_batch(module, features, state, bool(payload.get("dataHasVolume", False)))
        prediction = _prediction_from_module_result(result)
    return {"ok": True, "prediction": prediction, "error": None}


def _handle_train(command: dict[str, Any]) -> dict[str, Any]:
    training = command.get("training") or {}
    inference = training.get("inference") or {}
    framework = str(inference.get("framework", ""))
    model_identifier = inference.get("modelIdentifier")
    plugin_name, backend_path = _resolve_backend_path(model_identifier, framework)
    if framework == "foundationNLP":
        return {"ok": True, "prediction": None, "error": None}
    _prepare_framework(framework)
    module = _load_module(plugin_name, backend_path)
    features = _features(inference)
    label = _label_index(training.get("labelClass"))
    move = abs(_safe_float(training.get("movePoints", 0.0)))
    state = _load_state(plugin_name, framework, model_identifier)
    state = _call_train_step(module, features, label, move, state, bool(inference.get("dataHasVolume", False)))
    _save_state(plugin_name, framework, model_identifier, state)
    return {"ok": True, "prediction": None, "error": None}


def main() -> int:
    operation = sys.argv[1] if len(sys.argv) > 1 else "predict"
    try:
        command = json.loads(sys.stdin.read() or "{}")
        response = _handle_train(command) if operation == "train" or command.get("operation") == "train" else _handle_predict(command)
        print(json.dumps(response, separators=(",", ":")))
        return 0
    except Exception as exc:
        print(json.dumps({"ok": False, "prediction": None, "error": str(exc)}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
