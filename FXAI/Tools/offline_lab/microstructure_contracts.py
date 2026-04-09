from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import fxai_testlab as testlab

from .common_schema import OFFLINE_DIR, RESEARCH_DIR

MICROSTRUCTURE_SCHEMA_VERSION = 1
MICROSTRUCTURE_CONFIG_VERSION = 1

REPO_ROOT = Path(__file__).resolve().parents[2]
MICROSTRUCTURE_DIR = OFFLINE_DIR / "Microstructure"
MICROSTRUCTURE_STATE_DIR = MICROSTRUCTURE_DIR / "State"
MICROSTRUCTURE_REPORT_DIR = MICROSTRUCTURE_DIR / "Reports"
MICROSTRUCTURE_CONFIG_PATH = MICROSTRUCTURE_DIR / "microstructure_config.json"
MICROSTRUCTURE_STATUS_PATH = MICROSTRUCTURE_DIR / "microstructure_status.json"
MICROSTRUCTURE_STATE_PATH = MICROSTRUCTURE_STATE_DIR / "microstructure_state.json"
MICROSTRUCTURE_REPLAY_REPORT_PATH = MICROSTRUCTURE_REPORT_DIR / "microstructure_replay_report.json"
MICROSTRUCTURE_LOCAL_HISTORY_PATH = MICROSTRUCTURE_DIR / "microstructure_history.ndjson"

MICROSTRUCTURE_SERVICE_SOURCE = REPO_ROOT / "Services/FXAI_MicrostructureProbe.mq5"
TERMINAL_SERVICE_SOURCE = testlab.TERMINAL_ROOT / "MQL5/Services/FXAI_MicrostructureProbe.mq5"
TERMINAL_SERVICE_BINARY = TERMINAL_SERVICE_SOURCE.with_suffix(".ex5")

COMMON_MICROSTRUCTURE_JSON = testlab.RUNTIME_DIR / "microstructure_snapshot.json"
COMMON_MICROSTRUCTURE_FLAT = testlab.RUNTIME_DIR / "microstructure_snapshot_flat.tsv"
COMMON_MICROSTRUCTURE_HISTORY = testlab.RUNTIME_DIR / "microstructure_history.ndjson"
COMMON_MICROSTRUCTURE_STATUS = testlab.RUNTIME_DIR / "microstructure_status.json"
COMMON_MICROSTRUCTURE_SYMBOL_MAP = testlab.RUNTIME_DIR / "microstructure_symbol_map.tsv"
COMMON_MICROSTRUCTURE_CONFIG = testlab.RUNTIME_DIR / "microstructure_service_config.tsv"

COMMON_MICROSTRUCTURE_ARTIFACTS = {
    "snapshot_json": COMMON_MICROSTRUCTURE_JSON,
    "snapshot_flat": COMMON_MICROSTRUCTURE_FLAT,
    "history_ndjson": COMMON_MICROSTRUCTURE_HISTORY,
    "status_json": COMMON_MICROSTRUCTURE_STATUS,
    "symbol_map_tsv": COMMON_MICROSTRUCTURE_SYMBOL_MAP,
    "service_config_tsv": COMMON_MICROSTRUCTURE_CONFIG,
}


def ensure_microstructure_dirs() -> dict[str, Path]:
    for path in (
        MICROSTRUCTURE_DIR,
        MICROSTRUCTURE_STATE_DIR,
        MICROSTRUCTURE_REPORT_DIR,
        testlab.RUNTIME_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)
    return {
        "microstructure_dir": MICROSTRUCTURE_DIR,
        "state_dir": MICROSTRUCTURE_STATE_DIR,
        "report_dir": MICROSTRUCTURE_REPORT_DIR,
        "runtime_dir": testlab.RUNTIME_DIR,
    }


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def isoformat_utc(dt: datetime | None = None) -> str:
    value = dt or utc_now()
    return value.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def parse_iso8601(value: str | None) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).astimezone(timezone.utc)
    except ValueError:
        return None


def sanitize_utc_timestamp(
    value: str | None,
    *,
    now_dt: datetime | None = None,
    max_future_seconds: int = 120,
) -> datetime | None:
    dt = parse_iso8601(value)
    if dt is None:
        return None
    reference = now_dt or utc_now()
    if dt > reference + timedelta(seconds=max_future_seconds):
        return None
    return dt


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


def research_profile_dir(profile_name: str) -> Path:
    safe = (profile_name or "continuous").strip() or "continuous"
    return RESEARCH_DIR / safe
