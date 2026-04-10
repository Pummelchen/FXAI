from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import fxai_testlab as testlab

from .common_schema import OFFLINE_DIR

PROB_CALIBRATION_SCHEMA_VERSION = 1
PROB_CALIBRATION_CONFIG_VERSION = 1
PROB_CALIBRATION_MEMORY_VERSION = 1
PROB_CALIBRATION_METHODS = ["LOGISTIC_AFFINE"]
PROB_CALIBRATION_TIER_KINDS = [
    "PAIR_SESSION_REGIME",
    "PAIR_REGIME",
    "REGIME",
    "GLOBAL",
]
PROB_CALIBRATION_ACTIONS = ["BUY", "SELL", "SKIP"]

PROB_CALIBRATION_DIR = OFFLINE_DIR / "ProbabilisticCalibration"
PROB_CALIBRATION_REPORT_DIR = PROB_CALIBRATION_DIR / "Reports"
PROB_CALIBRATION_CONFIG_PATH = PROB_CALIBRATION_DIR / "prob_calibration_config.json"
PROB_CALIBRATION_MEMORY_PATH = PROB_CALIBRATION_DIR / "prob_calibration_memory.json"
PROB_CALIBRATION_REPLAY_REPORT_PATH = PROB_CALIBRATION_REPORT_DIR / "prob_calibration_replay_report.json"
PROB_CALIBRATION_RUNTIME_CONFIG_PATH = testlab.RUNTIME_DIR / "prob_calibration_config.tsv"
PROB_CALIBRATION_RUNTIME_MEMORY_PATH = testlab.RUNTIME_DIR / "prob_calibration_memory.tsv"

COMMON_PROB_CALIBRATION_STATE_PREFIX = "fxai_prob_calibration_"
COMMON_PROB_CALIBRATION_HISTORY_PREFIX = "fxai_prob_calibration_history_"


def ensure_prob_calibration_dirs() -> dict[str, Path]:
    for path in (PROB_CALIBRATION_DIR, PROB_CALIBRATION_REPORT_DIR, testlab.RUNTIME_DIR):
        path.mkdir(parents=True, exist_ok=True)
    return {
        "prob_calibration_dir": PROB_CALIBRATION_DIR,
        "report_dir": PROB_CALIBRATION_REPORT_DIR,
        "runtime_dir": testlab.RUNTIME_DIR,
    }


def prob_calibration_runtime_state_path(symbol: str) -> Path:
    return testlab.RUNTIME_DIR / f"{COMMON_PROB_CALIBRATION_STATE_PREFIX}{symbol.upper()}.tsv"


def prob_calibration_runtime_history_path(symbol: str) -> Path:
    return testlab.RUNTIME_DIR / f"{COMMON_PROB_CALIBRATION_HISTORY_PREFIX}{symbol.upper()}.ndjson"


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
