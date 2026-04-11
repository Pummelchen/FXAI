from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import fxai_testlab as testlab

from .common_schema import OFFLINE_DIR

LABEL_ENGINE_SCHEMA_VERSION = 1
LABEL_ENGINE_CONFIG_VERSION = 1
LABEL_ENGINE_REPORT_VERSION = 1
LABEL_ENGINE_ALLOWED_SIDES = ["LONG", "SHORT"]
LABEL_ENGINE_ALLOWED_CANDIDATE_MODES = ["BASELINE_MOMENTUM", "EXTERNAL_FILE"]

LABEL_ENGINE_DIR = OFFLINE_DIR / "LabelEngine"
LABEL_ENGINE_REPORT_DIR = LABEL_ENGINE_DIR / "Reports"
LABEL_ENGINE_ARTIFACT_DIR = LABEL_ENGINE_DIR / "Artifacts"
LABEL_ENGINE_CONFIG_PATH = LABEL_ENGINE_DIR / "label_engine_config.json"
LABEL_ENGINE_STATUS_PATH = LABEL_ENGINE_DIR / "label_engine_status.json"
LABEL_ENGINE_REPORT_PATH = LABEL_ENGINE_REPORT_DIR / "label_engine_report.json"
LABEL_ENGINE_RUNTIME_SUMMARY_PATH = testlab.RUNTIME_DIR / "label_engine_summary.json"


def ensure_label_engine_dirs() -> dict[str, Path]:
    for path in (LABEL_ENGINE_DIR, LABEL_ENGINE_REPORT_DIR, LABEL_ENGINE_ARTIFACT_DIR, testlab.RUNTIME_DIR):
        path.mkdir(parents=True, exist_ok=True)
    return {
        "label_engine_dir": LABEL_ENGINE_DIR,
        "report_dir": LABEL_ENGINE_REPORT_DIR,
        "artifact_dir": LABEL_ENGINE_ARTIFACT_DIR,
        "runtime_dir": testlab.RUNTIME_DIR,
    }


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def isoformat_utc(dt: datetime | None = None) -> str:
    value = dt or utc_now()
    return value.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def json_dump(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def json_load(path: Path) -> dict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}
