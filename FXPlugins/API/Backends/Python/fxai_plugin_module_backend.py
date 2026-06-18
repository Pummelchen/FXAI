#!/usr/bin/env python3
"""FXAI plugin-local Python backend dispatcher.

The Swift bridge sends the same JSON contract used by `fxai_plugin_backend.py`.
This dispatcher loads the plugin's own PyTorch, TensorFlow, NLP, or ONNX model
from its folder and converts the result back into `PredictionV4` JSON.
"""

from __future__ import annotations

import importlib.util
import io
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import pickle
import random
import re
import sys
from typing import Any


VOLUME_FEATURE_INDEXES = (6, 68, 69, 70, 71, 74, 75, 76, 77, 78, 80, 81, 82, 83)
FXAI_PLUGIN_API_VERSION = 4
FXAI_TOKENIZER_API_VERSION = "fxai-tokenizer-v1"
MAX_SEQUENCE_BARS = 512
CHECKPOINT_MANIFEST_VERSION = "fxai_backend_checkpoint_v1"


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


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
    if framework == "onnxRuntime":
        return root / plugin_name / "ONNX" / f"{plugin_name}.onnx"
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
    if framework == "pyTorch":
        import torch

        mps_backend = getattr(torch.backends, "mps", None)
        mps_available = bool(mps_backend and torch.backends.mps.is_available())
        if _env_flag("FXAI_FORCE_PYTORCH_CPU") and mps_backend:
            torch.backends.mps.is_available = lambda: False
            return
        if _env_flag("FXAI_REQUIRE_PYTORCH_MPS") and not mps_available:
            raise RuntimeError("PyTorch MPS is required for this FXAI accelerator runtime")
        if mps_available:
            os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    elif framework == "tensorFlow":
        import tensorflow as tf

        devices = [] if _env_flag("FXAI_FORCE_TENSORFLOW_CPU") else tf.config.list_physical_devices("GPU")
        for device in devices:
            try:
                tf.config.experimental.set_memory_growth(device, True)
            except Exception as _e:
                import logging
                logging.warning(f"TensorFlow memory growth failed for {device.device_type}:{device.name}: {_e}")
        if _env_flag("FXAI_REQUIRE_TENSORFLOW_METAL") and not devices:
            raise RuntimeError("TensorFlow Metal GPU device is required for this FXAI accelerator runtime")


def _stable_seed(plugin_name: str, framework: str, model_identifier: Any) -> int:
    material = f"{plugin_name}|{framework}|{_safe_token(model_identifier)}".encode("utf-8")
    digest = hashlib.sha256(material).digest()
    seed = int.from_bytes(digest[:8], "big") % 2_147_483_647
    return seed if seed > 0 else 1


def _configure_deterministic_environment(plugin_name: str, framework: str, model_identifier: Any) -> None:
    seed = _stable_seed(plugin_name, framework, model_identifier)
    os.environ.setdefault("FXAI_BACKEND_STABLE_SEED", str(seed))
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    if framework == "pyTorch":
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    elif framework == "tensorFlow":
        os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
        os.environ.setdefault("TF_CUDNN_DETERMINISTIC", "1")


def _seed_framework(plugin_name: str, framework: str, model_identifier: Any) -> None:
    seed = _stable_seed(plugin_name, framework, model_identifier)
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed % (2**32 - 1))
    except Exception:
        pass

    if framework == "pyTorch":
        try:
            import torch

            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            if hasattr(torch, "use_deterministic_algorithms"):
                try:
                    torch.use_deterministic_algorithms(True, warn_only=True)
                except TypeError:
                    torch.use_deterministic_algorithms(True)
        except Exception:
            pass
    elif framework == "tensorFlow":
        try:
            import tensorflow as tf

            tf.random.set_seed(seed)
        except Exception:
            pass


def _state_path(plugin_name: str, framework: str, model_identifier: Any) -> Path:
    suffix = "tfstate.pkl" if framework == "tensorFlow" else "pkl"
    return _state_dir() / f"{plugin_name}.{framework}.{_safe_token(model_identifier)}.{suffix}"


def _checkpoint_manifest_path(path: Path) -> Path:
    return path.with_name(f"{path.name}.manifest.json")


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str | None:
    try:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return None


def _checkpoint_manifest(
    path: Path,
    payload: bytes,
    plugin_name: str,
    framework: str,
    model_identifier: Any,
    state_format: str,
) -> dict[str, Any]:
    backend_path = _backend_path(plugin_name, framework)
    return {
        "schemaVersion": CHECKPOINT_MANIFEST_VERSION,
        "pluginName": plugin_name,
        "framework": framework,
        "modelIdentifier": _safe_token(model_identifier),
        "stateFileName": path.name,
        "stateFormat": state_format,
        "stateBytes": len(payload),
        "stateSha256": _sha256_bytes(payload),
        "backendPath": str(backend_path),
        "backendSha256": _sha256_file(backend_path),
        "stableSeed": _stable_seed(plugin_name, framework, model_identifier),
        "deterministic": True,
        "pythonVersion": sys.version.split()[0],
    }


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary_path.open("wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        temporary_path.replace(path)
    finally:
        try:
            temporary_path.unlink()
        except FileNotFoundError:
            pass


def _write_checkpoint(
    path: Path,
    payload: bytes,
    plugin_name: str,
    framework: str,
    model_identifier: Any,
    state_format: str,
) -> None:
    _atomic_write(path, payload)
    manifest = _checkpoint_manifest(path, payload, plugin_name, framework, model_identifier, state_format)
    manifest_payload = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
    _atomic_write(_checkpoint_manifest_path(path), manifest_payload)


def _checkpoint_manifest_valid(path: Path, plugin_name: str, framework: str, model_identifier: Any) -> bool:
    manifest_path = _checkpoint_manifest_path(path)
    if not manifest_path.exists():
        return not _env_flag("FXAI_REQUIRE_CHECKPOINT_MANIFEST")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if manifest.get("schemaVersion") != CHECKPOINT_MANIFEST_VERSION:
        return False
    if manifest.get("pluginName") != plugin_name:
        return False
    if manifest.get("framework") != framework:
        return False
    if manifest.get("modelIdentifier") != _safe_token(model_identifier):
        return False
    if manifest.get("stateFileName") != path.name:
        return False
    if not path.exists():
        return False
    state_hash = _sha256_file(path)
    if state_hash is None or state_hash != manifest.get("stateSha256"):
        return False
    try:
        state_bytes = path.stat().st_size
    except OSError:
        return False
    try:
        manifest_bytes = int(manifest.get("stateBytes", -1))
    except (TypeError, ValueError):
        return False
    if manifest_bytes != state_bytes:
        return False
    if _env_flag("FXAI_REQUIRE_BACKEND_SOURCE_MATCH"):
        backend_hash = _sha256_file(_backend_path(plugin_name, framework))
        if backend_hash is None or backend_hash != manifest.get("backendSha256"):
            return False
    return True


def _tensorflow_state_class(module: Any, class_name: Any) -> Any | None:
    if module is None:
        return None
    if isinstance(class_name, str):
        candidate = getattr(module, class_name, None)
        if candidate is not None and hasattr(candidate, "create"):
            return candidate
    for name in dir(module):
        if name.endswith("TensorFlowState"):
            candidate = getattr(module, name, None)
            if candidate is not None and hasattr(candidate, "create"):
                return candidate
    return None


def _optimizer_learning_rate(state: Any) -> float:
    optimizer = getattr(state, "optimizer", None)
    learning_rate = getattr(optimizer, "learning_rate", None)
    try:
        if hasattr(learning_rate, "numpy"):
            return _safe_float(learning_rate.numpy())
        return _safe_float(learning_rate)
    except Exception:
        return 3.0e-4


def _load_tensorflow_state(
    path: Path,
    module: Any,
    sequence: list[list[float]] | None,
    data_has_volume: bool
) -> Any | None:
    if module is None:
        return None
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, dict) or payload.get("kind") != "fxai_tensorflow_state_v1":
        return None
    state_class = _tensorflow_state_class(module, payload.get("stateClass"))
    if state_class is None:
        return None
    learning_rate = _safe_float(payload.get("learningRate", 3.0e-4)) or 3.0e-4
    state = state_class.create(lr=learning_rate)
    weights = payload.get("modelWeights")
    if isinstance(weights, list) and hasattr(getattr(state, "model", None), "set_weights"):
        feature_count = int(getattr(module, "FEATURE_COUNT", 32))
        build_sequence = sequence if sequence else [[0.0] * feature_count]
        _call_predict_batch(module, build_sequence, state, data_has_volume)
        state.model.set_weights(weights)
    return state


def _load_state(
    plugin_name: str,
    framework: str,
    model_identifier: Any,
    module: Any | None = None,
    sequence: list[list[float]] | None = None,
    data_has_volume: bool = True
) -> Any | None:
    path = _state_path(plugin_name, framework, model_identifier)
    if not path.exists():
        return None
    if not _checkpoint_manifest_valid(path, plugin_name, framework, model_identifier):
        return None
    try:
        if framework == "pyTorch":
            import torch

            mps_backend = getattr(torch.backends, "mps", None)
            use_mps = bool(mps_backend and torch.backends.mps.is_available() and not _env_flag("FXAI_FORCE_PYTORCH_CPU"))
            map_location = torch.device("mps") if use_mps else torch.device("cpu")
            return torch.load(path, map_location=map_location, weights_only=False)
        if framework == "tensorFlow":
            return _load_tensorflow_state(path, module, sequence, data_has_volume)
        with path.open("rb") as handle:
            return pickle.load(handle)
    except Exception:
        return None


def _tensorflow_state_payload(plugin_name: str, model_identifier: Any, state: Any) -> bytes:
    model = getattr(state, "model", None)
    weights = model.get_weights() if model is not None and hasattr(model, "get_weights") else []
    payload = {
        "kind": "fxai_tensorflow_state_v1",
        "plugin": plugin_name,
        "modelIdentifier": _safe_token(model_identifier),
        "stateClass": state.__class__.__name__,
        "architecture": getattr(model, "architecture", None),
        "learningRate": _optimizer_learning_rate(state),
        "modelWeights": weights,
    }
    return pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)


def _save_state(plugin_name: str, framework: str, model_identifier: Any, state: Any) -> None:
    if state is None:
        return
    path = _state_path(plugin_name, framework, model_identifier)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        if framework == "pyTorch":
            import torch

            buffer = io.BytesIO()
            torch.save(state, buffer)
            _write_checkpoint(path, buffer.getvalue(), plugin_name, framework, model_identifier, "torch_pickle_state_v1")
            return
        if framework == "tensorFlow":
            payload = _tensorflow_state_payload(plugin_name, model_identifier, state)
            _write_checkpoint(path, payload, plugin_name, framework, model_identifier, "tensorflow_weights_pickle_v1")
            return
        payload = pickle.dumps(state, protocol=pickle.HIGHEST_PROTOCOL)
        _write_checkpoint(path, payload, plugin_name, framework, model_identifier, "python_pickle_state_v1")
    except Exception:
        # Persistence is best-effort because TensorFlow objects are not always pickle-safe.
        return


def _sanitized_feature_row(raw: Any, data_has_volume: bool, target_length: int | None = None) -> list[float]:
    if not isinstance(raw, list):
        raw = []
    values = [max(-8.0, min(_safe_float(value), 8.0)) for value in raw]
    if target_length is not None:
        if len(values) < target_length:
            values.extend([0.0] * (target_length - len(values)))
        elif len(values) > target_length:
            values = values[:target_length]
    if not data_has_volume:
        for index in VOLUME_FEATURE_INDEXES:
            if index < len(values):
                values[index] = 0.0
    return values


def _features(payload: dict[str, Any]) -> list[float]:
    return _sanitized_feature_row(payload.get("x") or [], bool(payload.get("dataHasVolume", False)))


def _sequence(payload: dict[str, Any]) -> list[list[float]]:
    data_has_volume = bool(payload.get("dataHasVolume", False))
    current = _features(payload)
    target_length = len(current)
    sequence: list[list[float]] = []
    for row in payload.get("xWindow") or []:
        if isinstance(row, list):
            sequence.append(_sanitized_feature_row(row, data_has_volume, target_length))
    sequence.append(current)

    try:
        requested_bars = int(payload.get("sequenceBars", len(sequence)))
    except (TypeError, ValueError):
        requested_bars = len(sequence)
    limit = max(1, min(requested_bars, MAX_SEQUENCE_BARS))
    return sequence[-limit:]


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
        "apiVersion": FXAI_PLUGIN_API_VERSION,
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


def _resolve_onnx_model_path(model_identifier: Any) -> tuple[str, Path]:
    override = os.environ.get("FXAI_ONNX_MODEL_PATH")
    if override:
        path = Path(override).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"FXAI_ONNX_MODEL_PATH does not exist: {path}")
        return _plugin_candidates(model_identifier)[0], path
    return _resolve_backend_path(model_identifier, "onnxRuntime")


def _onnx_manifest_path(model_path: Path) -> Path:
    override = os.environ.get("FXAI_ONNX_MANIFEST_PATH")
    if override:
        return Path(override).expanduser().resolve()
    return model_path.with_suffix(".manifest.json")


def _manifest_string(manifest: dict[str, Any], *keys: str) -> str | None:
    for key in keys:
        value = manifest.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _load_onnx_manifest(plugin_name: str, model_identifier: Any, model_path: Path) -> dict[str, Any]:
    manifest_path = _onnx_manifest_path(model_path)
    if not manifest_path.exists():
        if os.environ.get("FXAI_ONNX_MANIFEST_PATH"):
            raise FileNotFoundError(f"FXAI_ONNX_MANIFEST_PATH does not exist: {manifest_path}")
        return {}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError(f"ONNX manifest must be a JSON object: {manifest_path}")
    declared_plugin = _manifest_string(manifest, "pluginName")
    if declared_plugin is not None and declared_plugin != plugin_name:
        raise ValueError(f"ONNX manifest pluginName {declared_plugin} does not match {plugin_name}")
    declared_model = _manifest_string(manifest, "modelIdentifier")
    if declared_model is not None and declared_model not in {str(model_identifier), _safe_token(model_identifier)}:
        raise ValueError(f"ONNX manifest modelIdentifier {declared_model} does not match {model_identifier}")
    declared_sha = _manifest_string(manifest, "modelSha256", "modelSHA256", "sha256")
    if declared_sha is not None:
        actual_sha = _sha256_file(model_path)
        if actual_sha is None or actual_sha.lower() != declared_sha.lower():
            raise ValueError("ONNX model SHA-256 does not match manifest")
    return manifest


def _onnx_providers(manifest: dict[str, Any], ort: Any) -> list[str]:
    raw = os.environ.get("FXAI_ONNX_PROVIDERS")
    if raw:
        requested = [item.strip() for item in raw.split(",") if item.strip()]
    else:
        manifest_providers = manifest.get("providers")
        requested = manifest_providers if isinstance(manifest_providers, list) else ["CPUExecutionProvider"]
    available = set(ort.get_available_providers())
    selected = [str(provider) for provider in requested if str(provider) in available]
    if selected:
        return selected
    if "CPUExecutionProvider" in available:
        return ["CPUExecutionProvider"]
    return list(available)


def _onnx_input_tensor(payload: dict[str, Any], input_shape: Any) -> Any:
    import numpy as np

    sequence = _sequence(payload)
    rank = len(input_shape) if isinstance(input_shape, (list, tuple)) else 3
    if rank <= 1:
        return np.asarray(sequence[-1], dtype=np.float32)
    if rank == 2:
        return np.asarray([sequence[-1]], dtype=np.float32)
    return np.asarray([sequence], dtype=np.float32)


def _to_python_value(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, tuple):
        return [_to_python_value(item) for item in value]
    if isinstance(value, list):
        return [_to_python_value(item) for item in value]
    return value


def _output_by_name(outputs: list[Any], output_names: list[str], manifest: dict[str, Any], *keys: str) -> Any | None:
    name = _manifest_string(manifest, *keys)
    if name is None:
        return None
    try:
        index = output_names.index(name)
    except ValueError:
        return None
    return outputs[index]


def _onnx_probabilities(value: Any) -> list[float]:
    raw = _first_row(value)[:3]
    if len(raw) < 3:
        raw.extend([0.0] * (3 - len(raw)))
    if any(item < 0.0 for item in raw) or sum(raw) > 1.5:
        high = max(raw)
        exps = [math.exp(max(-60.0, min(item - high, 60.0))) for item in raw]
        total = sum(exps)
        if total > 1.0e-12:
            return [item / total for item in exps]
    return _normalize_probabilities(raw)


def _onnx_prediction_from_outputs(outputs: list[Any], output_names: list[str], manifest: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    probabilities_source = (
        _output_by_name(outputs, output_names, manifest, "classProbabilitiesOutputName", "probabilitiesOutputName")
        or outputs[0]
    )
    move_source = _output_by_name(outputs, output_names, manifest, "moveMeanOutputName", "moveMeanPointsOutputName")
    quantile_source = _output_by_name(outputs, output_names, manifest, "moveQuantilesOutputName", "quantilesOutputName")

    if move_source is None and len(outputs) > 1:
        move_source = outputs[1]
    if quantile_source is None and len(outputs) > 2:
        quantile_source = outputs[2]

    probabilities = _onnx_probabilities(probabilities_source)
    row = _first_row(probabilities_source)
    if move_source is not None:
        move = max(0.0, _first_number(move_source, 0.0))
    elif len(row) >= 4:
        move = max(0.0, _safe_float(row[3]))
    else:
        move = max(
            _safe_float(payload.get("minMovePoints", 0.0)),
            _safe_float(payload.get("priceCostPoints", 0.0)),
            0.0,
        )

    if quantile_source is not None:
        q25, q50, q75 = _quantiles(quantile_source, move)
    elif len(row) >= 6:
        q25, q50, q75 = max(0.0, row[3]), max(0.0, row[4]), max(0.0, row[5])
    else:
        q25, q50, q75 = _quantiles([], move)
    return _prediction(probabilities, move, [q25, q50, q75])


def _onnx_prediction(plugin_name: str, model_identifier: Any, model_path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    import onnxruntime as ort

    manifest = _load_onnx_manifest(plugin_name, model_identifier, model_path)
    # Create onnxruntime.InferenceSession via the ort alias for ONNX model inference
    session = ort.InferenceSession(str(model_path), providers=_onnx_providers(manifest, ort))
    inputs = session.get_inputs()
    if not inputs:
        raise ValueError("ONNX model has no inputs")
    input_name = _manifest_string(manifest, "inputName") or inputs[0].name
    input_meta = next((item for item in inputs if item.name == input_name), inputs[0])
    tensor = _onnx_input_tensor(payload, getattr(input_meta, "shape", None))
    output_values = [_to_python_value(item) for item in session.run(None, {input_name: tensor})]
    output_names = [item.name for item in session.get_outputs()]
    if not output_values:
        raise ValueError("ONNX model returned no outputs")
    return _onnx_prediction_from_outputs(output_values, output_names, manifest, payload)


def _call_train_step(
    module: Any,
    sequence: list[list[float]],
    label: int,
    move: float,
    state: Any,
    data_has_volume: bool,
    transaction_cost_points: float | None = None,
    financial_targets: dict[str, Any] | None = None,
    financial_loss_config: dict[str, Any] | None = None,
) -> Any:
    train_step = module.train_step
    parameters = inspect.signature(train_step).parameters
    kwargs: dict[str, Any] = {}
    if "state" in parameters:
        kwargs["state"] = state
    if "data_has_volume" in parameters:
        kwargs["data_has_volume"] = data_has_volume
    if transaction_cost_points is not None and "transaction_cost_points" in parameters:
        kwargs["transaction_cost_points"] = transaction_cost_points
    if financial_targets is not None and "financial_targets" in parameters:
        kwargs["financial_targets"] = financial_targets
    if financial_loss_config is not None and "financial_loss_config" in parameters:
        kwargs["financial_loss_config"] = financial_loss_config
    if kwargs:
        return train_step([sequence], [label], [move], **kwargs)
    try:
        return train_step([sequence], [label], [move], state=state, data_has_volume=data_has_volume)
    except TypeError:
        return train_step([sequence], [label], [move], state=state)


def _call_predict_batch(
    module: Any,
    sequence: list[list[float]],
    state: Any,
    data_has_volume: bool
) -> dict[str, Any]:
    try:
        return module.predict_batch([sequence], state=state, data_has_volume=data_has_volume)
    except TypeError:
        return module.predict_batch([sequence], state=state)


def _handle_predict(command: dict[str, Any]) -> dict[str, Any]:
    payload = command.get("inference") or {}
    framework = str(payload.get("framework", ""))
    model_identifier = payload.get("modelIdentifier")
    if framework == "onnxRuntime":
        plugin_name, model_path = _resolve_onnx_model_path(model_identifier)
        _configure_deterministic_environment(plugin_name, framework, model_identifier)
        _seed_framework(plugin_name, framework, model_identifier)
        prediction = _onnx_prediction(plugin_name, model_identifier, model_path, payload)
        return {"apiVersion": FXAI_PLUGIN_API_VERSION, "ok": True, "prediction": prediction, "error": None}
    plugin_name, backend_path = _resolve_backend_path(model_identifier, framework)
    _configure_deterministic_environment(plugin_name, framework, model_identifier)
    _prepare_framework(framework)
    _seed_framework(plugin_name, framework, model_identifier)
    module = _load_module(plugin_name, backend_path)
    features = _features(payload)
    if framework == "foundationNLP":
        prediction = _nlp_prediction(module, features, payload)
    else:
        sequence = _sequence(payload)
        data_has_volume = bool(payload.get("dataHasVolume", False))
        state = _load_state(
            plugin_name,
            framework,
            model_identifier,
            module=module,
            sequence=sequence,
            data_has_volume=data_has_volume
        )
        result = _call_predict_batch(module, sequence, state, data_has_volume)
        prediction = _prediction_from_module_result(result)
    return {"apiVersion": FXAI_PLUGIN_API_VERSION, "ok": True, "prediction": prediction, "error": None}


def _handle_train(command: dict[str, Any]) -> dict[str, Any]:
    training = command.get("training") or {}
    inference = training.get("inference") or {}
    framework = str(inference.get("framework", ""))
    model_identifier = inference.get("modelIdentifier")
    if framework == "onnxRuntime":
        return {"apiVersion": FXAI_PLUGIN_API_VERSION, "ok": True, "prediction": None, "error": None}
    plugin_name, backend_path = _resolve_backend_path(model_identifier, framework)
    if framework == "foundationNLP":
        return {"apiVersion": FXAI_PLUGIN_API_VERSION, "ok": True, "prediction": None, "error": None}
    _configure_deterministic_environment(plugin_name, framework, model_identifier)
    _prepare_framework(framework)
    _seed_framework(plugin_name, framework, model_identifier)
    module = _load_module(plugin_name, backend_path)
    label = _label_index(training.get("labelClass"))
    raw_move = _safe_float(training.get("movePoints", 0.0))
    move = raw_move if plugin_name == "rl_ppo" else abs(raw_move)
    sequence = _sequence(inference)
    data_has_volume = bool(inference.get("dataHasVolume", False))
    transaction_cost_points = _safe_float(inference.get("priceCostPoints", 0.0)) if plugin_name == "rl_ppo" else None
    financial_targets = training.get("financialTargets")
    if not isinstance(financial_targets, dict):
        financial_targets = None
    financial_loss_config = training.get("financialLossSpec")
    if not isinstance(financial_loss_config, dict):
        financial_loss_config = None
    state = _load_state(
        plugin_name,
        framework,
        model_identifier,
        module=module,
        sequence=sequence,
        data_has_volume=data_has_volume
    )
    state = _call_train_step(
        module,
        sequence,
        label,
        move,
        state,
        data_has_volume,
        transaction_cost_points=transaction_cost_points,
        financial_targets=financial_targets,
        financial_loss_config=financial_loss_config,
    )
    _save_state(plugin_name, framework, model_identifier, state)
    return {"apiVersion": FXAI_PLUGIN_API_VERSION, "ok": True, "prediction": None, "error": None}


def main() -> int:
    operation = sys.argv[1] if len(sys.argv) > 1 else "predict"
    try:
        command = json.loads(sys.stdin.read() or "{}")
        _require_latest_api(command)
        response = _handle_train(command) if operation == "train" or command.get("operation") == "train" else _handle_predict(command)
        print(json.dumps(response, separators=(",", ":")))
        return 0
    except Exception as exc:
        print(json.dumps({"apiVersion": FXAI_PLUGIN_API_VERSION, "ok": False, "prediction": None, "error": str(exc)}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
