from __future__ import annotations

import importlib.util
import json
import os
import platform
import shutil
import sys
from pathlib import Path

from .common import (
    COMMON_EXPORT_DIR,
    COMMON_PROMOTION_DIR,
    DEFAULT_DB,
    DISTILL_DIR,
    OFFLINE_DIR,
    OFFLINE_ARTIFACT_SCHEMA_VERSION,
    OFFLINE_MACRO_SCHEMA_MIN,
    OFFLINE_SCHEMA_VERSION,
    PROFILES_DIR,
    RESEARCH_DIR,
    RUNS_DIR,
    connect_db,
    close_db,
    current_lab_versions,
    ensure_dir,
    turso_environment_status,
)
from testlab.shared import (
    COMMON_FILES,
    METAEDITOR,
    MT5_LOG_DIR,
    ROOT,
    TERMINAL,
    TERMINAL_ROOT,
    TESTER_PRESET_DIR,
    TOOLCHAIN_PROFILE,
    WINE,
    supported_toolchain_profiles,
    toolchain_summary,
)


def _path_state(path: Path, *, required: bool) -> dict[str, object]:
    parent = path if path.is_dir() else path.parent
    exists = path.exists()
    parent_exists = parent.exists()
    state = {
        "path": str(path),
        "exists": exists,
        "is_dir": path.is_dir(),
        "parent_exists": parent_exists,
        "writable_parent": os.access(parent if parent.exists() else parent.parent, os.W_OK),
        "required": required,
    }
    state["ok"] = bool(parent_exists and (exists or not required))
    return state


def _required_path_names(profile: str) -> set[str]:
    required = {
        "root",
        "common_files",
        "offline_dir",
        "profiles_dir",
        "research_dir",
        "distill_dir",
        "runs_dir",
        "common_promotion_dir",
        "common_export_dir",
        "tester_preset_dir",
    }
    if profile in {"macos_wine", "windows_native"}:
        required.update({"terminal_root", "metaeditor", "terminal", "mt5_log_dir"})
    if profile == "macos_wine":
        required.add("wine")
    return required


def doctor_report() -> dict[str, object]:
    profile = TOOLCHAIN_PROFILE
    required_paths = _required_path_names(profile)

    python_ok = sys.version_info >= (3, 11)
    pytest_ok = bool(importlib.util.find_spec("pytest"))
    libsql_ok = bool(importlib.util.find_spec("libsql"))
    turso_cli = shutil.which("turso") or ""
    turso_status = turso_environment_status(DEFAULT_DB)
    partial_sync_config = bool(turso_status.get("config_error"))
    report = {
        "python": {
            "version": sys.version.split()[0],
            "ok": python_ok,
            "platform": platform.platform(),
        },
        "dependencies": {
            "pytest": pytest_ok,
            "libsql": libsql_ok,
            "turso_cli": bool(turso_cli),
        },
        "database": {
            **turso_status,
            "turso_cli_path": turso_cli,
        },
        "versions": {
            "expected_offline_schema_version": OFFLINE_SCHEMA_VERSION,
            "expected_artifact_schema_version": OFFLINE_ARTIFACT_SCHEMA_VERSION,
            "expected_macro_schema_min": OFFLINE_MACRO_SCHEMA_MIN,
        },
        "paths": {
            "root": _path_state(ROOT, required="root" in required_paths),
            "terminal_root": _path_state(TERMINAL_ROOT, required="terminal_root" in required_paths),
            "metaeditor": _path_state(METAEDITOR, required="metaeditor" in required_paths),
            "terminal": _path_state(TERMINAL, required="terminal" in required_paths),
            "wine": _path_state(WINE, required="wine" in required_paths),
            "common_files": _path_state(COMMON_FILES, required="common_files" in required_paths),
            "offline_dir": _path_state(OFFLINE_DIR, required="offline_dir" in required_paths),
            "profiles_dir": _path_state(PROFILES_DIR, required="profiles_dir" in required_paths),
            "research_dir": _path_state(RESEARCH_DIR, required="research_dir" in required_paths),
            "distill_dir": _path_state(DISTILL_DIR, required="distill_dir" in required_paths),
            "runs_dir": _path_state(RUNS_DIR, required="runs_dir" in required_paths),
            "common_promotion_dir": _path_state(COMMON_PROMOTION_DIR, required="common_promotion_dir" in required_paths),
            "common_export_dir": _path_state(COMMON_EXPORT_DIR, required="common_export_dir" in required_paths),
            "tester_preset_dir": _path_state(TESTER_PRESET_DIR, required="tester_preset_dir" in required_paths),
            "mt5_log_dir": _path_state(MT5_LOG_DIR, required="mt5_log_dir" in required_paths),
        },
        "toolchain": {
            **toolchain_summary(),
            "supported_profiles": list(supported_toolchain_profiles()),
        },
    }
    required_paths_ok = all(bool(item["ok"]) for item in report["paths"].values() if item.get("required"))
    mt5_compile_ready = (
        bool(report["paths"]["metaeditor"]["exists"]) and
        (profile != "macos_wine" or bool(report["paths"]["wine"]["exists"]))
    )
    mt5_runtime_ready = (
        bool(report["paths"]["terminal"]["exists"]) and
        (profile != "macos_wine" or bool(report["paths"]["wine"]["exists"]))
    )
    ok = python_ok and pytest_ok and libsql_ok and not partial_sync_config and required_paths_ok
    if report["database"].get("platform_api_enabled"):
        ok = ok and bool(report["database"].get("organization_slug")) and bool(report["database"].get("api_token_configured"))
    report["checks"] = {
        "required_paths_ok": required_paths_ok,
        "mt5_compile_ready": mt5_compile_ready,
        "mt5_runtime_ready": mt5_runtime_ready,
        "profile": profile,
        "profile_requires_wine": profile == "macos_wine",
    }
    report["ok"] = bool(ok)
    return report


def validate_environment() -> dict[str, object]:
    return doctor_report()


def bootstrap_environment(db_path: Path = DEFAULT_DB,
                          init_db: bool = True) -> dict[str, object]:
    created = []
    for path in [OFFLINE_DIR, PROFILES_DIR, RESEARCH_DIR, DISTILL_DIR, RUNS_DIR, COMMON_PROMOTION_DIR, COMMON_EXPORT_DIR, TESTER_PRESET_DIR]:
        ensure_dir(path)
        created.append(str(path))

    payload = {
        "created_dirs": created,
        "db_path": str(db_path),
        "validated_environment": validate_environment(),
    }
    if init_db:
        conn = connect_db(db_path)
        payload["db_versions"] = current_lab_versions(conn)
        close_db(conn)
        payload["db_initialized"] = True
    else:
        payload["db_initialized"] = False
    return payload


def write_environment_report(output_path: Path) -> dict[str, object]:
    payload = validate_environment()
    ensure_dir(output_path.parent)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload
