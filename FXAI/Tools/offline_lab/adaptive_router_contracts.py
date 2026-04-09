from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import fxai_testlab as testlab

from .common_schema import OFFLINE_DIR, RESEARCH_DIR

ADAPTIVE_ROUTER_SCHEMA_VERSION = 1
ADAPTIVE_ROUTER_CONFIG_VERSION = 1
ADAPTIVE_ROUTER_REGIMES = [
    "TREND_PERSISTENT",
    "RANGE_MEAN_REVERTING",
    "BREAKOUT_TRANSITION",
    "HIGH_VOL_EVENT",
    "RISK_ON_OFF_MACRO",
    "LIQUIDITY_STRESS",
    "SESSION_FLOW",
]
ADAPTIVE_ROUTER_SESSIONS = [
    "ASIA",
    "LONDON",
    "NEWYORK",
    "LONDON_NY_OVERLAP",
    "ROLLOVER",
]

ADAPTIVE_ROUTER_DIR = OFFLINE_DIR / "AdaptiveRouter"
ADAPTIVE_ROUTER_REPORT_DIR = ADAPTIVE_ROUTER_DIR / "Reports"
ADAPTIVE_ROUTER_CONFIG_PATH = ADAPTIVE_ROUTER_DIR / "adaptive_router_config.json"
ADAPTIVE_ROUTER_REPLAY_REPORT_PATH = ADAPTIVE_ROUTER_REPORT_DIR / "adaptive_router_replay_report.json"

COMMON_ADAPTIVE_ROUTER_HISTORY_PREFIX = "fxai_regime_router_history_"
COMMON_ADAPTIVE_ROUTER_STATE_PREFIX = "fxai_regime_router_"


def ensure_adaptive_router_dirs() -> dict[str, Path]:
    for path in (ADAPTIVE_ROUTER_DIR, ADAPTIVE_ROUTER_REPORT_DIR, testlab.RUNTIME_DIR):
        path.mkdir(parents=True, exist_ok=True)
    return {
        "adaptive_router_dir": ADAPTIVE_ROUTER_DIR,
        "report_dir": ADAPTIVE_ROUTER_REPORT_DIR,
        "runtime_dir": testlab.RUNTIME_DIR,
    }


def adaptive_router_runtime_state_path(symbol: str) -> Path:
    return testlab.RUNTIME_DIR / f"{COMMON_ADAPTIVE_ROUTER_STATE_PREFIX}{symbol}.tsv"


def adaptive_router_runtime_history_path(symbol: str) -> Path:
    return testlab.RUNTIME_DIR / f"{COMMON_ADAPTIVE_ROUTER_HISTORY_PREFIX}{symbol}.ndjson"


def research_profile_dir(profile_name: str) -> Path:
    safe = (profile_name or "continuous").strip() or "continuous"
    return RESEARCH_DIR / safe


def json_dump(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def isoformat_utc(dt: datetime | None = None) -> str:
    value = dt or utc_now()
    return value.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
