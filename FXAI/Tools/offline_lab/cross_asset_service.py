from __future__ import annotations

import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

from .common import OfflineLabError
from .cross_asset_config import ensure_default_files, load_config, resolve_probe_symbols
from .cross_asset_contracts import (
    COMMON_CROSS_ASSET_CONFIG,
    COMMON_CROSS_ASSET_PROBE_JSON,
    COMMON_CROSS_ASSET_PROBE_STATUS,
    CROSS_ASSET_CONFIG_PATH,
    CROSS_ASSET_PROBE_SERVICE_SOURCE,
    TERMINAL_CROSS_ASSET_SERVICE_BINARY,
    TERMINAL_CROSS_ASSET_SERVICE_SOURCE,
    ensure_cross_asset_dirs,
    isoformat_utc,
    json_dump,
    json_load,
    sanitize_utc_timestamp,
    utc_now,
)
from testlab.shared import METAEDITOR, TERMINAL_ROOT, WINE, read_utf16_or_text, to_wine_path


def cross_asset_probe_health_snapshot() -> dict[str, Any]:
    ensure_cross_asset_dirs()
    status = json_load(COMMON_CROSS_ASSET_PROBE_STATUS)
    snapshot = json_load(COMMON_CROSS_ASSET_PROBE_JSON)
    generated_at = str(status.get("generated_at", "") or snapshot.get("generated_at", ""))
    return {
        "generated_at": generated_at,
        "probe_status_path": str(COMMON_CROSS_ASSET_PROBE_STATUS),
        "probe_snapshot_path": str(COMMON_CROSS_ASSET_PROBE_JSON),
        "service": dict(status.get("service", {})),
        "symbols": dict(snapshot.get("symbols", {})),
    }


def validate_cross_asset_service_config() -> dict[str, Any]:
    config = ensure_default_files()
    status = cross_asset_probe_health_snapshot()
    return {
        "ok": True,
        "config_path": str(CROSS_ASSET_CONFIG_PATH),
        "service_config_path": str(COMMON_CROSS_ASSET_CONFIG),
        "probe_symbol_count": len(resolve_probe_symbols(config)),
        "probe_runtime_generated_at": status.get("generated_at", ""),
    }


def install_cross_asset_service(compile_service: bool = True) -> dict[str, object]:
    ensure_default_files()
    if not CROSS_ASSET_PROBE_SERVICE_SOURCE.exists():
        raise OfflineLabError(f"Cross-asset service source is missing: {CROSS_ASSET_PROBE_SERVICE_SOURCE}")
    TERMINAL_CROSS_ASSET_SERVICE_SOURCE.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(CROSS_ASSET_PROBE_SERVICE_SOURCE, TERMINAL_CROSS_ASSET_SERVICE_SOURCE)
    payload: dict[str, object] = {
        "source_path": str(CROSS_ASSET_PROBE_SERVICE_SOURCE),
        "installed_path": str(TERMINAL_CROSS_ASSET_SERVICE_SOURCE),
        "terminal_root": str(TERMINAL_ROOT),
        "compiled": False,
        "binary_path": str(TERMINAL_CROSS_ASSET_SERVICE_BINARY),
        "service_config_path": str(COMMON_CROSS_ASSET_CONFIG),
    }
    if compile_service:
        payload.update(compile_cross_asset_service())
    return payload


def compile_cross_asset_service(timeout_sec: int = 600) -> dict[str, object]:
    if not TERMINAL_CROSS_ASSET_SERVICE_SOURCE.exists():
        raise OfflineLabError(
            "Cross-asset service is not installed into MQL5/Services. "
            "Run cross-asset-install-service first."
        )
    with tempfile.TemporaryDirectory(prefix="fxai_cross_asset_service_") as tmp_dir:
        stage_root = Path(tmp_dir)
        stage_source = stage_root / TERMINAL_CROSS_ASSET_SERVICE_SOURCE.name
        stage_log = stage_root / "compile_cross_asset_service.log"
        shutil.copy2(TERMINAL_CROSS_ASSET_SERVICE_SOURCE, stage_source)
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
                TERMINAL_CROSS_ASSET_SERVICE_BINARY.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(built_ex5, TERMINAL_CROSS_ASSET_SERVICE_BINARY)
                log_path = TERMINAL_CROSS_ASSET_SERVICE_SOURCE.with_suffix(".compile.log")
                shutil.copy2(stage_log, log_path)
                return {
                    "compiled": True,
                    "binary_path": str(TERMINAL_CROSS_ASSET_SERVICE_BINARY),
                    "log_path": str(log_path),
                }
            if rc is not None:
                if proc.stdout is not None:
                    _ = proc.stdout.read()
                raise OfflineLabError(
                    "Cross-asset service compile failed. "
                    f"Last log line: {(last_log.splitlines()[-1] if last_log.splitlines() else 'no log output')}"
                )
            time.sleep(1.5)

        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=5)
        raise OfflineLabError("Cross-asset service compile timed out")
