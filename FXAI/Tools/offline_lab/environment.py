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
    current_lab_versions,
    ensure_dir,
    turso_environment_status,
)
from testlab.shared import COMMON_FILES, METAEDITOR, MT5_LOG_DIR, ROOT, TERMINAL, TERMINAL_ROOT, TESTER_PRESET_DIR, WINE


def _path_state(path: Path) -> dict[str, object]:
    parent = path if path.is_dir() else path.parent
    return {
        "path": str(path),
        "exists": path.exists(),
        "is_dir": path.is_dir(),
        "parent_exists": parent.exists(),
        "writable_parent": os.access(parent if parent.exists() else parent.parent, os.W_OK),
    }


def validate_environment() -> dict[str, object]:
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
            "root": _path_state(ROOT),
            "terminal_root": _path_state(TERMINAL_ROOT),
            "metaeditor": _path_state(METAEDITOR),
            "terminal": _path_state(TERMINAL),
            "wine": _path_state(WINE),
            "common_files": _path_state(COMMON_FILES),
            "offline_dir": _path_state(OFFLINE_DIR),
            "profiles_dir": _path_state(PROFILES_DIR),
            "research_dir": _path_state(RESEARCH_DIR),
            "distill_dir": _path_state(DISTILL_DIR),
            "runs_dir": _path_state(RUNS_DIR),
            "common_promotion_dir": _path_state(COMMON_PROMOTION_DIR),
            "common_export_dir": _path_state(COMMON_EXPORT_DIR),
            "tester_preset_dir": _path_state(TESTER_PRESET_DIR),
            "mt5_log_dir": _path_state(MT5_LOG_DIR),
        },
    }
    ok = python_ok and pytest_ok and libsql_ok and not partial_sync_config
    for item in report["paths"].values():
        ok = ok and bool(item["parent_exists"])
    report["ok"] = bool(ok)
    return report


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
        conn.close()
        payload["db_initialized"] = True
    else:
        payload["db_initialized"] = False
    return payload


def write_environment_report(output_path: Path) -> dict[str, object]:
    payload = validate_environment()
    ensure_dir(output_path.parent)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload
