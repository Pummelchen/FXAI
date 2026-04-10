from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import fxai_testlab as testlab

from .common_schema import OFFLINE_DIR

EXECUTION_QUALITY_SCHEMA_VERSION = 1
EXECUTION_QUALITY_CONFIG_VERSION = 1
EXECUTION_QUALITY_MEMORY_VERSION = 1
EXECUTION_QUALITY_METHODS = ["SCORECARD_V1"]
EXECUTION_QUALITY_TIER_KINDS = [
    "PAIR_SESSION_REGIME",
    "PAIR_REGIME",
    "SESSION_REGIME",
    "REGIME",
    "GLOBAL",
]
EXECUTION_QUALITY_STATES = ["NORMAL", "CAUTION", "STRESSED", "BLOCKED"]

EXECUTION_QUALITY_DIR = OFFLINE_DIR / "ExecutionQuality"
EXECUTION_QUALITY_REPORT_DIR = EXECUTION_QUALITY_DIR / "Reports"
EXECUTION_QUALITY_CONFIG_PATH = EXECUTION_QUALITY_DIR / "execution_quality_config.json"
EXECUTION_QUALITY_MEMORY_PATH = EXECUTION_QUALITY_DIR / "execution_quality_memory.json"
EXECUTION_QUALITY_REPLAY_REPORT_PATH = EXECUTION_QUALITY_REPORT_DIR / "execution_quality_replay_report.json"
EXECUTION_QUALITY_RUNTIME_CONFIG_PATH = testlab.RUNTIME_DIR / "execution_quality_config.tsv"
EXECUTION_QUALITY_RUNTIME_MEMORY_PATH = testlab.RUNTIME_DIR / "execution_quality_memory.tsv"

COMMON_EXECUTION_QUALITY_STATE_PREFIX = "fxai_execution_quality_"
COMMON_EXECUTION_QUALITY_HISTORY_PREFIX = "fxai_execution_quality_history_"


def ensure_execution_quality_dirs() -> dict[str, Path]:
    for path in (EXECUTION_QUALITY_DIR, EXECUTION_QUALITY_REPORT_DIR, testlab.RUNTIME_DIR):
        path.mkdir(parents=True, exist_ok=True)
    return {
        "execution_quality_dir": EXECUTION_QUALITY_DIR,
        "report_dir": EXECUTION_QUALITY_REPORT_DIR,
        "runtime_dir": testlab.RUNTIME_DIR,
    }


def execution_quality_runtime_state_path(symbol: str) -> Path:
    return testlab.RUNTIME_DIR / f"{COMMON_EXECUTION_QUALITY_STATE_PREFIX}{symbol.upper()}.tsv"


def execution_quality_runtime_history_path(symbol: str) -> Path:
    return testlab.RUNTIME_DIR / f"{COMMON_EXECUTION_QUALITY_HISTORY_PREFIX}{symbol.upper()}.ndjson"


def json_dump(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def json_load(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def isoformat_utc(dt: datetime | None = None) -> str:
    value = dt or utc_now()
    return value.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
