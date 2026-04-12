from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import time
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
    MICROSTRUCTURE_SERVICE_SOURCE,
    MICROSTRUCTURE_STATUS_PATH,
    TERMINAL_SERVICE_BINARY,
    TERMINAL_SERVICE_SOURCE,
    ensure_microstructure_dirs,
    isoformat_utc,
    json_dump,
    json_load,
    sanitize_utc_timestamp,
    utc_now,
)
from testlab.shared import METAEDITOR, TERMINAL_ROOT, WINE, read_utf16_or_text, to_wine_path


def _status_payload_from_runtime(now_dt=None) -> dict[str, Any]:
    now_dt = now_dt or utc_now()
    runtime_snapshot = json_load(COMMON_MICROSTRUCTURE_JSON)
    runtime_status = json_load(COMMON_MICROSTRUCTURE_STATUS)
    generated_at = sanitize_utc_timestamp(
        runtime_status.get("generated_at") or runtime_snapshot.get("generated_at"),
        now_dt=now_dt,
    )
    if not runtime_status and not runtime_snapshot:
        return {
            "generated_at": isoformat_utc(now_dt),
            "service": {"ok": False, "stale": True, "enabled": False, "last_error": "microstructure service has not produced a snapshot yet"},
            "symbols": {},
            "health": {"snapshot_stale_after_sec": int(load_config().get("snapshot_stale_after_sec", 45) or 45)},
            "artifacts": {
                "snapshot_json": str(COMMON_MICROSTRUCTURE_JSON),
                "service_config_tsv": str(COMMON_MICROSTRUCTURE_CONFIG),
                "status_json": str(COMMON_MICROSTRUCTURE_STATUS),
                "history_ndjson": str(MICROSTRUCTURE_LOCAL_HISTORY_PATH),
            },
        }

    payload = dict(runtime_status if runtime_status else runtime_snapshot)
    payload.setdefault("generated_at", isoformat_utc(generated_at or now_dt))
    payload.setdefault("symbols", runtime_snapshot.get("symbols", {}))
    payload.setdefault("health", runtime_status.get("health", {}))
    payload.setdefault("service", runtime_status.get("service", runtime_snapshot.get("service", {})))
    payload["artifacts"] = {
        "snapshot_json": str(COMMON_MICROSTRUCTURE_JSON),
        "service_config_tsv": str(COMMON_MICROSTRUCTURE_CONFIG),
        "status_json": str(COMMON_MICROSTRUCTURE_STATUS),
        "history_ndjson": str(MICROSTRUCTURE_LOCAL_HISTORY_PATH),
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
        except Exception:
            pass
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
    ensure_default_files()
    if not MICROSTRUCTURE_SERVICE_SOURCE.exists():
        raise OfflineLabError(f"Microstructure service source is missing: {MICROSTRUCTURE_SERVICE_SOURCE}")
    TERMINAL_SERVICE_SOURCE.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(MICROSTRUCTURE_SERVICE_SOURCE, TERMINAL_SERVICE_SOURCE)
    payload: dict[str, object] = {
        "source_path": str(MICROSTRUCTURE_SERVICE_SOURCE),
        "installed_path": str(TERMINAL_SERVICE_SOURCE),
        "terminal_root": str(TERMINAL_ROOT),
        "compiled": False,
        "binary_path": str(TERMINAL_SERVICE_BINARY),
        "service_config_path": str(COMMON_MICROSTRUCTURE_CONFIG),
    }
    if compile_service:
        payload.update(compile_microstructure_service())
    return payload


def compile_microstructure_service(timeout_sec: int = 600) -> dict[str, object]:
    if not TERMINAL_SERVICE_SOURCE.exists():
        raise OfflineLabError(
            "Microstructure service is not installed into MQL5/Services. "
            "Run microstructure-install-service first."
        )
    with tempfile.TemporaryDirectory(prefix="fxai_microstructure_service_") as tmp_dir:
        stage_root = Path(tmp_dir)
        stage_source = stage_root / TERMINAL_SERVICE_SOURCE.name
        stage_log = stage_root / "compile_microstructure_service.log"
        shutil.copy2(TERMINAL_SERVICE_SOURCE, stage_source)
        cmd = [
            str(WINE),
            str(METAEDITOR),
            f"/compile:{to_wine_path(stage_source)}",
            f"/log:{to_wine_path(stage_log)}",
        ]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        deadline = time.time() + float(timeout_sec)
        built_ex5 = stage_source.with_suffix(".ex5")
        last_log = ""

        while time.time() < deadline:
            rc = proc.poll()
            if stage_log.exists():
                last_log = read_utf16_or_text(stage_log)
            if "0 errors, 0 warnings" in last_log and built_ex5.exists():
                if proc.poll() is None:
                    proc.kill()
                    proc.wait(timeout=5)
                TERMINAL_SERVICE_BINARY.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(built_ex5, TERMINAL_SERVICE_BINARY)
                log_path = TERMINAL_SERVICE_SOURCE.with_suffix(".compile.log")
                shutil.copy2(stage_log, log_path)
                return {
                    "compiled": True,
                    "binary_path": str(TERMINAL_SERVICE_BINARY),
                    "log_path": str(log_path),
                }
            if rc is not None:
                if proc.stdout is not None:
                    _ = proc.stdout.read()
                raise OfflineLabError(
                    "Microstructure service compile failed. "
                    f"Last log line: {(last_log.splitlines()[-1] if last_log.splitlines() else 'no log output')}"
                )
            time.sleep(1.5)

        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=5)
        raise OfflineLabError("Microstructure service compile timed out")
