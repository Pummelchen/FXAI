from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import fxai_testlab as testlab

from .common_schema import OFFLINE_DIR, RESEARCH_DIR

CROSS_ASSET_SCHEMA_VERSION = 1
CROSS_ASSET_CONFIG_VERSION = 1
CROSS_ASSET_PROBE_SCHEMA_VERSION = 1

REPO_ROOT = Path(__file__).resolve().parents[2]
CROSS_ASSET_DIR = OFFLINE_DIR / "CrossAsset"
CROSS_ASSET_STATE_DIR = CROSS_ASSET_DIR / "State"
CROSS_ASSET_REPORT_DIR = CROSS_ASSET_DIR / "Reports"
CROSS_ASSET_CONFIG_PATH = CROSS_ASSET_DIR / "cross_asset_config.json"
CROSS_ASSET_STATUS_PATH = CROSS_ASSET_DIR / "cross_asset_status.json"
CROSS_ASSET_STATE_PATH = CROSS_ASSET_STATE_DIR / "cross_asset_state.json"
CROSS_ASSET_REPLAY_REPORT_PATH = CROSS_ASSET_REPORT_DIR / "cross_asset_replay_report.json"
CROSS_ASSET_LOCAL_HISTORY_PATH = CROSS_ASSET_DIR / "cross_asset_history.ndjson"

CROSS_ASSET_PROBE_SERVICE_SOURCE = REPO_ROOT / "Services/FXAI_CrossAssetProbe.mq5"
TERMINAL_CROSS_ASSET_SERVICE_SOURCE = testlab.TERMINAL_ROOT / "MQL5/Services/FXAI_CrossAssetProbe.mq5"
TERMINAL_CROSS_ASSET_SERVICE_BINARY = TERMINAL_CROSS_ASSET_SERVICE_SOURCE.with_suffix(".ex5")

COMMON_CROSS_ASSET_JSON = testlab.RUNTIME_DIR / "cross_asset_snapshot.json"
COMMON_CROSS_ASSET_FLAT = testlab.RUNTIME_DIR / "cross_asset_snapshot_flat.tsv"
COMMON_CROSS_ASSET_HISTORY = testlab.RUNTIME_DIR / "cross_asset_history.ndjson"
COMMON_CROSS_ASSET_STATUS = testlab.RUNTIME_DIR / "cross_asset_status.json"
COMMON_CROSS_ASSET_SYMBOL_MAP = testlab.RUNTIME_DIR / "cross_asset_symbol_map.tsv"
COMMON_CROSS_ASSET_CONFIG = testlab.RUNTIME_DIR / "cross_asset_probe_config.tsv"

COMMON_CROSS_ASSET_PROBE_JSON = testlab.RUNTIME_DIR / "cross_asset_probe_snapshot.json"
COMMON_CROSS_ASSET_PROBE_STATUS = testlab.RUNTIME_DIR / "cross_asset_probe_status.json"
COMMON_CROSS_ASSET_PROBE_HISTORY = testlab.RUNTIME_DIR / "cross_asset_probe_history.ndjson"

COMMON_CROSS_ASSET_ARTIFACTS = {
    "snapshot_json": COMMON_CROSS_ASSET_JSON,
    "snapshot_flat": COMMON_CROSS_ASSET_FLAT,
    "history_ndjson": COMMON_CROSS_ASSET_HISTORY,
    "status_json": COMMON_CROSS_ASSET_STATUS,
    "symbol_map_tsv": COMMON_CROSS_ASSET_SYMBOL_MAP,
    "probe_snapshot_json": COMMON_CROSS_ASSET_PROBE_JSON,
    "probe_status_json": COMMON_CROSS_ASSET_PROBE_STATUS,
    "probe_history_ndjson": COMMON_CROSS_ASSET_PROBE_HISTORY,
    "probe_config_tsv": COMMON_CROSS_ASSET_CONFIG,
}


def ensure_cross_asset_dirs() -> dict[str, Path]:
    for path in (
        CROSS_ASSET_DIR,
        CROSS_ASSET_STATE_DIR,
        CROSS_ASSET_REPORT_DIR,
        testlab.RUNTIME_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)
    return {
        "cross_asset_dir": CROSS_ASSET_DIR,
        "state_dir": CROSS_ASSET_STATE_DIR,
        "report_dir": CROSS_ASSET_REPORT_DIR,
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
