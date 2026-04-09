from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import fxai_testlab as testlab

from .common_schema import OFFLINE_DIR, RESEARCH_DIR

RATES_ENGINE_SCHEMA_VERSION = 1
RATES_ENGINE_CONFIG_VERSION = 1
RATES_ENGINE_INPUTS_VERSION = 1
RATES_ENGINE_POLICY_VERSION = 1

RATES_ENGINE_DIR = OFFLINE_DIR / "RatesEngine"
RATES_ENGINE_STATE_DIR = RATES_ENGINE_DIR / "State"
RATES_ENGINE_REPORT_DIR = RATES_ENGINE_DIR / "Reports"
RATES_ENGINE_CONFIG_PATH = RATES_ENGINE_DIR / "rates_engine_config.json"
RATES_ENGINE_INPUTS_PATH = RATES_ENGINE_DIR / "rates_provider_inputs.json"
RATES_ENGINE_STATUS_PATH = RATES_ENGINE_DIR / "rates_engine_status.json"
RATES_ENGINE_STATE_PATH = RATES_ENGINE_STATE_DIR / "rates_engine_state.json"
RATES_ENGINE_REPLAY_REPORT_PATH = RATES_ENGINE_REPORT_DIR / "rates_replay_report.json"
RATES_ENGINE_LOCAL_HISTORY_PATH = RATES_ENGINE_DIR / "rates_history.ndjson"

COMMON_RATES_JSON = testlab.RUNTIME_DIR / "rates_snapshot.json"
COMMON_RATES_FLAT = testlab.RUNTIME_DIR / "rates_snapshot_flat.tsv"
COMMON_RATES_HISTORY = testlab.RUNTIME_DIR / "rates_history.ndjson"
COMMON_RATES_SYMBOL_MAP = testlab.RUNTIME_DIR / "rates_symbol_map.tsv"

COMMON_RATES_ARTIFACTS = {
    "snapshot_json": COMMON_RATES_JSON,
    "snapshot_flat": COMMON_RATES_FLAT,
    "history_ndjson": COMMON_RATES_HISTORY,
    "symbol_map_tsv": COMMON_RATES_SYMBOL_MAP,
}


def ensure_rates_engine_dirs() -> dict[str, Path]:
    for path in (RATES_ENGINE_DIR, RATES_ENGINE_STATE_DIR, RATES_ENGINE_REPORT_DIR, testlab.RUNTIME_DIR):
        path.mkdir(parents=True, exist_ok=True)
    return {
        "rates_engine_dir": RATES_ENGINE_DIR,
        "state_dir": RATES_ENGINE_STATE_DIR,
        "report_dir": RATES_ENGINE_REPORT_DIR,
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
