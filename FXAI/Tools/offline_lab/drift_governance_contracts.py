from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import fxai_testlab as testlab

from .common_schema import OFFLINE_DIR

DRIFT_GOVERNANCE_SCHEMA_VERSION = 1
DRIFT_GOVERNANCE_CONFIG_VERSION = 1
DRIFT_GOVERNANCE_REPORT_VERSION = 1

DRIFT_GOVERNANCE_HEALTH_STATES = [
    "HEALTHY",
    "CAUTION",
    "DEGRADED",
    "SHADOW_ONLY",
    "DISABLED",
]
DRIFT_GOVERNANCE_STATES = [
    "HEALTHY",
    "CAUTION",
    "DEGRADED",
    "SHADOW_ONLY",
    "DEMOTED",
    "DISABLED",
    "CHALLENGER",
    "CHAMPION_CANDIDATE",
    "CHAMPION",
]
DRIFT_GOVERNANCE_ACTIONS = [
    "NONE",
    "DOWNWEIGHT",
    "RESTRICT",
    "SHADOW_ONLY",
    "DEMOTE",
    "DISABLE",
    "PROMOTION_REVIEW",
    "PROMOTE",
    "ROLLBACK",
]
DRIFT_GOVERNANCE_MODES = [
    "RECOMMEND_ONLY",
    "AUTO_APPLY_PROTECTIVE",
    "AUTO_APPLY_ALL",
]

DRIFT_GOVERNANCE_DIR = OFFLINE_DIR / "DriftGovernance"
DRIFT_GOVERNANCE_REPORT_DIR = DRIFT_GOVERNANCE_DIR / "Reports"
DRIFT_GOVERNANCE_ARTIFACT_DIR = DRIFT_GOVERNANCE_DIR / "Artifacts"
DRIFT_GOVERNANCE_CONFIG_PATH = DRIFT_GOVERNANCE_DIR / "drift_governance_config.json"
DRIFT_GOVERNANCE_STATUS_PATH = DRIFT_GOVERNANCE_DIR / "drift_governance_status.json"
DRIFT_GOVERNANCE_REPORT_PATH = DRIFT_GOVERNANCE_REPORT_DIR / "drift_governance_report.json"
DRIFT_GOVERNANCE_HISTORY_PATH = DRIFT_GOVERNANCE_DIR / "drift_governance_history.ndjson"
DRIFT_GOVERNANCE_RUNTIME_SUMMARY_PATH = testlab.RUNTIME_DIR / "drift_governance_summary.json"


def ensure_drift_governance_dirs() -> dict[str, Path]:
    for path in (
        DRIFT_GOVERNANCE_DIR,
        DRIFT_GOVERNANCE_REPORT_DIR,
        DRIFT_GOVERNANCE_ARTIFACT_DIR,
        testlab.RUNTIME_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)
    return {
        "drift_governance_dir": DRIFT_GOVERNANCE_DIR,
        "report_dir": DRIFT_GOVERNANCE_REPORT_DIR,
        "artifact_dir": DRIFT_GOVERNANCE_ARTIFACT_DIR,
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
