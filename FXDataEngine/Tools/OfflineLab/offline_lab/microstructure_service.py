from __future__ import annotations

import json
from contextlib import suppress
from pathlib import Path
from typing import Any

from .common import OfflineLabError
from .microstructure_config import ensure_default_files, load_config, validate_config_payload
from .microstructure_contracts import (
    COMMON_MICROSTRUCTURE_CONFIG,
    COMMON_MICROSTRUCTURE_JSON,
    COMMON_MICROSTRUCTURE_STATUS,
    MICROSTRUCTURE_CONFIG_PATH,
    MICROSTRUCTURE_LOCAL_HISTORY_PATH,
    MICROSTRUCTURE_STATUS_PATH,
    REPO_ROOT,
    ensure_microstructure_dirs,
    isoformat_utc,
    json_dump,
    json_load,
    sanitize_utc_timestamp,
    utc_now,
)


def _portable_artifact_path(path: Path) -> str:
    resolved = path.resolve()
    repo_root = REPO_ROOT.resolve()
    runtime_dir = COMMON_MICROSTRUCTURE_JSON.parent.resolve()
    with suppress(ValueError):
        return str(resolved.relative_to(repo_root))
    try:
        return "FILE_COMMON/FXAI/Runtime/" + str(resolved.relative_to(runtime_dir))
    except ValueError:
        return path.name


def _status_payload_from_runtime(now_dt=None) -> dict[str, Any]:
    now_dt = now_dt or utc_now()
    config = load_config()
    runtime_snapshot = json_load(COMMON_MICROSTRUCTURE_JSON)
    runtime_status = json_load(COMMON_MICROSTRUCTURE_STATUS)
    generated_at = sanitize_utc_timestamp(
        runtime_status.get("generated_at") or runtime_snapshot.get("generated_at"),
        now_dt=now_dt,
    )
    if not runtime_status and not runtime_snapshot:
        return {
            "generated_at": isoformat_utc(now_dt),
            "service": {
                "ok": False,
                "stale": True,
                "enabled": bool(config.get("enabled", True)),
            "collector_mode": str(config.get("collector_mode", "fxdatabase_api")),
            "configured_pairs": len(list(dict(config.get("symbol_universe", {})).get("canonical_pairs", []))),
            "last_error": "microstructure collector has not produced a snapshot yet",
            },
            "symbols": {},
            "health": {"snapshot_stale_after_sec": int(config.get("snapshot_stale_after_sec", 45) or 45)},
            "artifacts": {
                "snapshot_json": _portable_artifact_path(COMMON_MICROSTRUCTURE_JSON),
                "service_config_tsv": _portable_artifact_path(COMMON_MICROSTRUCTURE_CONFIG),
                "status_json": _portable_artifact_path(COMMON_MICROSTRUCTURE_STATUS),
                "history_ndjson": _portable_artifact_path(MICROSTRUCTURE_LOCAL_HISTORY_PATH),
            },
        }

    payload = dict(runtime_status if runtime_status else runtime_snapshot)
    payload.setdefault("generated_at", isoformat_utc(generated_at or now_dt))
    payload.setdefault("symbols", runtime_snapshot.get("symbols", {}))
    payload.setdefault("health", runtime_status.get("health", {}))
    payload.setdefault("service", runtime_status.get("service", runtime_snapshot.get("service", {})))
    service = payload.get("service")
    if not isinstance(service, dict):
        service = {}
        payload["service"] = service
    service.setdefault("enabled", bool(config.get("enabled", True)))
    service.setdefault("collector_mode", str(config.get("collector_mode", "fxdatabase_api")))
    service.setdefault("configured_pairs", len(list(dict(config.get("symbol_universe", {})).get("canonical_pairs", []))))
    payload["artifacts"] = {
        "snapshot_json": _portable_artifact_path(COMMON_MICROSTRUCTURE_JSON),
        "service_config_tsv": _portable_artifact_path(COMMON_MICROSTRUCTURE_CONFIG),
        "status_json": _portable_artifact_path(COMMON_MICROSTRUCTURE_STATUS),
        "history_ndjson": _portable_artifact_path(MICROSTRUCTURE_LOCAL_HISTORY_PATH),
    }
    return payload


def sync_local_status_from_runtime() -> dict[str, Any]:
    ensure_microstructure_dirs()
    payload = _status_payload_from_runtime()
    health = payload.get("health")
    if not isinstance(health, dict):
        health = {}
        payload["health"] = health
    if COMMON_MICROSTRUCTURE_STATUS.exists():
        try:
            health["runtime_status_mtime"] = COMMON_MICROSTRUCTURE_STATUS.stat().st_mtime
        except OSError as exc:
            health["runtime_status_error"] = str(exc)
    json_dump(MICROSTRUCTURE_STATUS_PATH, payload)
    return payload


def validate_microstructure_config() -> dict[str, Any]:
    config = ensure_default_files()
    validate_config_payload(config)
    status = sync_local_status_from_runtime()
    return {
        "ok": True,
        "config_path": str(MICROSTRUCTURE_CONFIG_PATH),
        "service_config_path": str(COMMON_MICROSTRUCTURE_CONFIG),
        "status_path": str(MICROSTRUCTURE_STATUS_PATH),
        "pair_count": len(config["symbol_universe"]["canonical_pairs"]),
        "windows_sec": config["windows_sec"],
        "runtime_status_generated_at": status.get("generated_at", ""),
    }


def microstructure_health_snapshot() -> dict[str, Any]:
    local_status = sync_local_status_from_runtime()
    return {
        "status_path": str(MICROSTRUCTURE_STATUS_PATH),
        "runtime_status_path": str(COMMON_MICROSTRUCTURE_STATUS),
        "snapshot_path": str(COMMON_MICROSTRUCTURE_JSON),
        "generated_at": local_status.get("generated_at", ""),
        "service": local_status.get("service", {}),
        "health": local_status.get("health", {}),
        "symbols": local_status.get("symbols", {}),
    }


def install_microstructure_service(compile_service: bool = True) -> dict[str, object]:
    config = ensure_default_files()
    return {
        "installed": True,
        "compiled": False,
        "compile_requested": bool(compile_service),
        "collector": str(config.get("collector_mode", "fxdatabase_api")),
        "service_config_path": str(COMMON_MICROSTRUCTURE_CONFIG),
        "pair_count": len(config["symbol_universe"]["canonical_pairs"]),
        "note": "Microstructure proxy snapshots are produced by the Swift/Python offline lab pipeline and FXDatabase API inputs.",
    }


def compile_microstructure_service(timeout_sec: int = 600) -> dict[str, object]:
    _ = timeout_sec
    return {
        "compiled": False,
        "collector": "fxdatabase_api",
        "note": "No separate terminal service is compiled for the Swift-era microstructure collector.",
    }
