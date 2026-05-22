from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from .newspulse_contracts import (
    COMMON_NEWSPULSE_CALENDAR_FEED,
    COMMON_NEWSPULSE_CALENDAR_STATE,
    sanitize_utc_timestamp,
    utc_now,
)


@dataclass(frozen=True)
class CalendarRecord:
    event_id: str
    event_key: str
    title: str
    country_code: str
    country_name: str
    currency: str
    event_time_utc_unix: int
    event_time_trade_server: str
    importance: int
    actual: float | None
    forecast: float | None
    previous: float | None
    revised_previous: float | None
    surprise_proxy: float | None
    collector_seen_utc_unix: int
    collector_seen_trade_server: str
    trade_server_offset_sec: int
    change_id: int


def _maybe_float(value: str) -> float | None:
    text = str(value or "").strip()
    if text == "":
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _maybe_int(value: str) -> int:
    try:
        return int(float(str(value or "0").strip() or "0"))
    except ValueError:
        return 0


def load_calendar_state(path: Path = COMMON_NEWSPULSE_CALENDAR_STATE) -> dict[str, str]:
    if not path.exists():
        return {}
    out: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = line.split("\t", 1)
        if len(parts) == 2:
            out[parts[0]] = parts[1]
    return out


def load_calendar_records(path: Path = COMMON_NEWSPULSE_CALENDAR_FEED) -> list[CalendarRecord]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        out: list[CalendarRecord] = []
        for row in reader:
            if not row:
                continue
            out.append(
                CalendarRecord(
                    event_id=str(row.get("event_id", "") or ""),
                    event_key=str(row.get("event_key", "") or ""),
                    title=str(row.get("title", "") or ""),
                    country_code=str(row.get("country_code", "") or "").upper(),
                    country_name=str(row.get("country_name", "") or ""),
                    currency=str(row.get("currency", "") or "").upper(),
                    event_time_utc_unix=_maybe_int(row.get("event_time_utc_unix") or row.get("event_time_unix", "0")),
                    event_time_trade_server=str(row.get("event_time_trade_server", "") or ""),
                    importance=_maybe_int(row.get("importance", "0")),
                    actual=_maybe_float(row.get("actual", "")),
                    forecast=_maybe_float(row.get("forecast", "")),
                    previous=_maybe_float(row.get("previous", "")),
                    revised_previous=_maybe_float(row.get("revised_previous", "")),
                    surprise_proxy=_maybe_float(row.get("surprise_proxy", "")),
                    collector_seen_utc_unix=_maybe_int(row.get("collector_seen_utc_unix") or row.get("collector_seen_unix", "0")),
                    collector_seen_trade_server=str(row.get("collector_seen_trade_server", "") or ""),
                    trade_server_offset_sec=_maybe_int(row.get("trade_server_offset_sec", "0")),
                    change_id=_maybe_int(row.get("change_id", "0")),
                )
            )
    return out


def calendar_source_status() -> dict[str, object]:
    raw = load_calendar_state()
    now_dt = utc_now()
    collector_generated_dt = sanitize_utc_timestamp(
        raw.get("collector_generated_at") or raw.get("last_update_at", ""),
        now_dt=now_dt,
    )
    raw_ok = str(raw.get("ok", "0")) == "1"
    raw_stale = str(raw.get("stale", "1")) == "1"
    effective_stale = raw_stale or collector_generated_dt is None
    return {
        "ok": raw_ok and collector_generated_dt is not None,
        "stale": effective_stale,
        "time_basis": str(raw.get("time_basis", "trade_server") or "trade_server"),
        "last_update_at": collector_generated_dt.isoformat().replace("+00:00", "Z") if collector_generated_dt else "",
        "last_update_trade_server": str(raw.get("last_update_trade_server", "") or ""),
        "cursor": raw.get("change_id", ""),
        "last_error": raw.get("last_error", ""),
        "collector_generated_at": collector_generated_dt.isoformat().replace("+00:00", "Z") if collector_generated_dt else "",
        "collector_generated_dt": collector_generated_dt,
        "collector_generated_unix": int(collector_generated_dt.timestamp()) if collector_generated_dt else 0,
        "trade_server_offset_sec": _maybe_int(raw.get("trade_server_offset_sec", "0")),
        "record_count": _maybe_int(raw.get("record_count", "0")),
    }
