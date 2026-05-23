from __future__ import annotations

from typing import Any

from .cross_asset_config import ensure_default_files, load_config, resolve_probe_symbols
from .cross_asset_contracts import (
    COMMON_CROSS_ASSET_CONFIG,
    COMMON_CROSS_ASSET_PROBE_JSON,
    COMMON_CROSS_ASSET_PROBE_STATUS,
    CROSS_ASSET_CONFIG_PATH,
    ensure_cross_asset_dirs,
    json_load,
)


def cross_asset_probe_health_snapshot() -> dict[str, Any]:
    ensure_cross_asset_dirs()
    config = load_config()
    status = json_load(COMMON_CROSS_ASSET_PROBE_STATUS)
    snapshot = json_load(COMMON_CROSS_ASSET_PROBE_JSON)
    generated_at = str(status.get("generated_at", "") or snapshot.get("generated_at", ""))
    service = dict(status.get("service", {}))
    if not service:
        service = {
            "ok": False,
            "stale": True,
            "enabled": bool(config.get("enabled", True)),
            "configured_symbols": len(resolve_probe_symbols(config)),
            "last_error": "cross-asset probe has not produced a snapshot yet",
        }
    return {
        "generated_at": generated_at,
        "probe_status_path": str(COMMON_CROSS_ASSET_PROBE_STATUS),
        "probe_snapshot_path": str(COMMON_CROSS_ASSET_PROBE_JSON),
        "service": service,
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
    config = ensure_default_files()
    return {
        "installed": True,
        "compiled": False,
        "compile_requested": bool(compile_service),
        "collector": "fxdatabase_api",
        "service_config_path": str(COMMON_CROSS_ASSET_CONFIG),
        "probe_symbol_count": len(resolve_probe_symbols(config)),
        "note": "Cross-asset snapshots are produced by the Swift/Python offline lab pipeline and FXDatabase API inputs.",
    }


def compile_cross_asset_service(timeout_sec: int = 600) -> dict[str, object]:
    _ = timeout_sec
    return {
        "compiled": False,
        "collector": "fxdatabase_api",
        "note": "No separate terminal service is compiled for the Swift-era cross-asset collector.",
    }
