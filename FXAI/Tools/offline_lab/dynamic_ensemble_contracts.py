from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import fxai_testlab as testlab

from .common_schema import OFFLINE_DIR

DYNAMIC_ENSEMBLE_SCHEMA_VERSION = 1
DYNAMIC_ENSEMBLE_CONFIG_VERSION = 1
DYNAMIC_ENSEMBLE_FAMILIES = [
    "linear",
    "tree",
    "recurrent",
    "convolutional",
    "transformer",
    "state_space",
    "distribution",
    "mixture",
    "memory",
    "world",
    "rule",
    "other",
]
DYNAMIC_ENSEMBLE_STATUSES = [
    "ACTIVE",
    "DOWNWEIGHTED",
    "SUPPRESSED",
    "EXCLUDED",
]
DYNAMIC_ENSEMBLE_POSTURES = [
    "NORMAL",
    "CAUTION",
    "ABSTAIN_BIAS",
    "BLOCK",
]

DYNAMIC_ENSEMBLE_DIR = OFFLINE_DIR / "DynamicEnsemble"
DYNAMIC_ENSEMBLE_REPORT_DIR = DYNAMIC_ENSEMBLE_DIR / "Reports"
DYNAMIC_ENSEMBLE_CONFIG_PATH = DYNAMIC_ENSEMBLE_DIR / "dynamic_ensemble_config.json"
DYNAMIC_ENSEMBLE_REPLAY_REPORT_PATH = DYNAMIC_ENSEMBLE_REPORT_DIR / "dynamic_ensemble_replay_report.json"
DYNAMIC_ENSEMBLE_RUNTIME_CONFIG_PATH = testlab.RUNTIME_DIR / "dynamic_ensemble_config.tsv"

COMMON_DYNAMIC_ENSEMBLE_STATE_PREFIX = "fxai_dynamic_ensemble_"
COMMON_DYNAMIC_ENSEMBLE_HISTORY_PREFIX = "fxai_dynamic_ensemble_history_"


def ensure_dynamic_ensemble_dirs() -> dict[str, Path]:
    for path in (DYNAMIC_ENSEMBLE_DIR, DYNAMIC_ENSEMBLE_REPORT_DIR, testlab.RUNTIME_DIR):
        path.mkdir(parents=True, exist_ok=True)
    return {
        "dynamic_ensemble_dir": DYNAMIC_ENSEMBLE_DIR,
        "report_dir": DYNAMIC_ENSEMBLE_REPORT_DIR,
        "runtime_dir": testlab.RUNTIME_DIR,
    }


def dynamic_ensemble_runtime_state_path(symbol: str) -> Path:
    return testlab.RUNTIME_DIR / f"{COMMON_DYNAMIC_ENSEMBLE_STATE_PREFIX}{symbol.upper()}.tsv"


def dynamic_ensemble_runtime_history_path(symbol: str) -> Path:
    return testlab.RUNTIME_DIR / f"{COMMON_DYNAMIC_ENSEMBLE_HISTORY_PREFIX}{symbol.upper()}.ndjson"


def json_dump(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def json_load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def isoformat_utc(dt: datetime | None = None) -> str:
    value = dt or utc_now()
    return value.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
