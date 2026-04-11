from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import fxai_testlab as testlab

from .common_schema import OFFLINE_DIR

PAIR_NETWORK_SCHEMA_VERSION = 1
PAIR_NETWORK_CONFIG_VERSION = 1
PAIR_NETWORK_FACTOR_KEYS = [
    "usd_bloc",
    "eur_rates",
    "safe_haven",
    "commodity_fx",
    "risk_on",
    "liquidity_stress",
    "macro_shock",
]
PAIR_NETWORK_ACTIONS = [
    "ALLOW",
    "ALLOW_REDUCED",
    "SUPPRESS_REDUNDANT",
    "BLOCK_CONTRADICTORY",
    "BLOCK_CONCENTRATION",
    "PREFER_ALTERNATIVE_EXPRESSION",
]

PAIR_NETWORK_DIR = OFFLINE_DIR / "PairNetwork"
PAIR_NETWORK_REPORT_DIR = PAIR_NETWORK_DIR / "Reports"
PAIR_NETWORK_CONFIG_PATH = PAIR_NETWORK_DIR / "pair_network_config.json"
PAIR_NETWORK_STATUS_PATH = PAIR_NETWORK_DIR / "pair_network_status.json"
PAIR_NETWORK_REPORT_PATH = PAIR_NETWORK_REPORT_DIR / "pair_network_report.json"
PAIR_NETWORK_HISTORY_PATH = PAIR_NETWORK_DIR / "pair_network_history.ndjson"

PAIR_NETWORK_RUNTIME_CONFIG_PATH = testlab.RUNTIME_DIR / "pair_network_config.tsv"
PAIR_NETWORK_RUNTIME_STATUS_PATH = testlab.RUNTIME_DIR / "pair_network_status.tsv"
COMMON_PAIR_NETWORK_STATE_PREFIX = "fxai_pair_network_"
COMMON_PAIR_NETWORK_HISTORY_PREFIX = "fxai_pair_network_history_"


def ensure_pair_network_dirs() -> dict[str, Path]:
    for path in (PAIR_NETWORK_DIR, PAIR_NETWORK_REPORT_DIR, testlab.RUNTIME_DIR):
        path.mkdir(parents=True, exist_ok=True)
    return {
        "pair_network_dir": PAIR_NETWORK_DIR,
        "report_dir": PAIR_NETWORK_REPORT_DIR,
        "runtime_dir": testlab.RUNTIME_DIR,
    }


def pair_network_runtime_state_path(symbol: str) -> Path:
    return testlab.RUNTIME_DIR / f"{COMMON_PAIR_NETWORK_STATE_PREFIX}{symbol.upper()}.tsv"


def pair_network_runtime_history_path(symbol: str) -> Path:
    return testlab.RUNTIME_DIR / f"{COMMON_PAIR_NETWORK_HISTORY_PREFIX}{symbol.upper()}.ndjson"


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


def ndjson_append(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")
