from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import fxai_testlab as testlab

from .common_schema import OFFLINE_DIR, RESEARCH_DIR

NEWSPULSE_SCHEMA_VERSION = 1
NEWSPULSE_CONFIG_VERSION = 1

REPO_ROOT = Path(__file__).resolve().parents[2]
NEWSPULSE_DIR = OFFLINE_DIR / "NewsPulse"
NEWSPULSE_STATE_DIR = NEWSPULSE_DIR / "State"
NEWSPULSE_REPORT_DIR = NEWSPULSE_DIR / "Reports"
NEWSPULSE_CONFIG_PATH = NEWSPULSE_DIR / "newspulse_config.json"
NEWSPULSE_SOURCES_PATH = NEWSPULSE_DIR / "newspulse_sources.json"
NEWSPULSE_STATUS_PATH = NEWSPULSE_DIR / "newspulse_status.json"
NEWSPULSE_LOCAL_HISTORY_PATH = NEWSPULSE_DIR / "news_history.ndjson"
NEWSPULSE_STATE_PATH = NEWSPULSE_STATE_DIR / "newspulse_state.json"
NEWSPULSE_SERVICE_SOURCE = REPO_ROOT / "Services/FXAI_NewsPulseCalendar.mq5"
TERMINAL_SERVICE_SOURCE = testlab.TERMINAL_ROOT / "MQL5/Services/FXAI_NewsPulseCalendar.mq5"
TERMINAL_SERVICE_BINARY = TERMINAL_SERVICE_SOURCE.with_suffix(".ex5")

COMMON_NEWSPULSE_JSON = testlab.RUNTIME_DIR / "news_snapshot.json"
COMMON_NEWSPULSE_FLAT = testlab.RUNTIME_DIR / "news_snapshot_flat.tsv"
COMMON_NEWSPULSE_HISTORY = testlab.RUNTIME_DIR / "news_history.ndjson"
COMMON_NEWSPULSE_CALENDAR_FEED = testlab.RUNTIME_DIR / "news_calendar_feed.tsv"
COMMON_NEWSPULSE_CALENDAR_STATE = testlab.RUNTIME_DIR / "news_calendar_state.tsv"
COMMON_NEWSPULSE_CALENDAR_HISTORY = testlab.RUNTIME_DIR / "news_calendar_history.ndjson"

COMMON_NEWSPULSE_ARTIFACTS = {
    "snapshot_json": COMMON_NEWSPULSE_JSON,
    "snapshot_flat": COMMON_NEWSPULSE_FLAT,
    "history_ndjson": COMMON_NEWSPULSE_HISTORY,
    "calendar_feed_tsv": COMMON_NEWSPULSE_CALENDAR_FEED,
    "calendar_state_tsv": COMMON_NEWSPULSE_CALENDAR_STATE,
    "calendar_history_ndjson": COMMON_NEWSPULSE_CALENDAR_HISTORY,
}


def ensure_newspulse_dirs() -> dict[str, Path]:
    for path in (NEWSPULSE_DIR, NEWSPULSE_STATE_DIR, NEWSPULSE_REPORT_DIR, testlab.RUNTIME_DIR):
        path.mkdir(parents=True, exist_ok=True)
    return {
        "newspulse_dir": NEWSPULSE_DIR,
        "state_dir": NEWSPULSE_STATE_DIR,
        "report_dir": NEWSPULSE_REPORT_DIR,
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


def sanitize_utc_timestamp(value: str | None,
                           *,
                           now_dt: datetime | None = None,
                           max_future_seconds: int = 120) -> datetime | None:
    dt = parse_iso8601(value)
    if dt is None:
        return None
    reference = now_dt or utc_now()
    if dt > reference + timedelta(seconds=max_future_seconds):
        return None
    return dt


def unix_to_iso8601(value: int | float | None) -> str:
    if value is None:
        return ""
    return isoformat_utc(datetime.fromtimestamp(float(value), tz=timezone.utc))


def json_dump(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def ndjson_append(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def research_profile_dir(profile_name: str) -> Path:
    safe = (profile_name or "continuous").strip() or "continuous"
    return RESEARCH_DIR / safe
