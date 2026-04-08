from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from .newspulse_contracts import (
    COMMON_NEWSPULSE_HISTORY,
    COMMON_NEWSPULSE_REPLAY_FLAT,
    NEWSPULSE_LOCAL_HISTORY_PATH,
    NEWSPULSE_REPLAY_REPORT_PATH,
    json_dump,
)


def _parse_iso8601(value: str | None) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).astimezone(timezone.utc)
    except ValueError:
        return None


def _isoformat(value: datetime | None) -> str:
    if value is None:
        return ""
    return value.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _prune(entries: list[dict[str, Any]], observed_at: datetime, max_age_hours: int, max_entries: int) -> list[dict[str, Any]]:
    cutoff = observed_at - timedelta(hours=max(max_age_hours, 1))
    kept = [entry for entry in entries if (_parse_iso8601(entry.get("observed_at")) or observed_at) >= cutoff]
    if len(kept) > max_entries:
        kept = kept[-max_entries:]
    return kept


def update_pair_gate_history(state: dict[str, Any],
                             pairs: dict[str, dict[str, Any]],
                             observed_at: datetime,
                             *,
                             max_age_hours: int = 72,
                             max_entries_per_pair: int = 64,
                             keepalive_minutes: int = 60) -> dict[str, list[dict[str, Any]]]:
    pair_history = state.setdefault("pair_gate_history", {})
    if not isinstance(pair_history, dict):
        pair_history = {}
        state["pair_gate_history"] = pair_history

    for pair_id, pair_state in pairs.items():
        history = pair_history.setdefault(pair_id, [])
        if not isinstance(history, list):
            history = []
            pair_history[pair_id] = history
        last_entry = history[-1] if history else None
        current_signature = (
            str(pair_state.get("trade_gate", "")),
            bool(pair_state.get("stale", True)),
            round(float(pair_state.get("news_risk_score", 0.0) or 0.0), 4),
            int(pair_state.get("event_eta_min", -1) if pair_state.get("event_eta_min") is not None else -1),
            str(pair_state.get("session_profile", "")),
            tuple(str(reason) for reason in pair_state.get("reasons", [])[:3]),
        )
        previous_signature = None
        elapsed_minutes = keepalive_minutes + 1
        if isinstance(last_entry, dict):
            previous_signature = (
                str(last_entry.get("trade_gate", "")),
                bool(last_entry.get("stale", True)),
                round(float(last_entry.get("news_risk_score", 0.0) or 0.0), 4),
                int(last_entry.get("event_eta_min", -1) if last_entry.get("event_eta_min") is not None else -1),
                str(last_entry.get("session_profile", "")),
                tuple(str(reason) for reason in last_entry.get("reasons", [])[:3]),
            )
            last_dt = _parse_iso8601(last_entry.get("observed_at"))
            if last_dt is not None:
                elapsed_minutes = int((observed_at - last_dt).total_seconds() / 60.0)
        if previous_signature == current_signature and elapsed_minutes < keepalive_minutes:
            pair_history[pair_id] = _prune(history, observed_at, max_age_hours, max_entries_per_pair)
            continue
        history.append(
            {
                "observed_at": _isoformat(observed_at),
                "trade_gate": str(pair_state.get("trade_gate", "UNKNOWN")),
                "news_risk_score": float(pair_state.get("news_risk_score", 0.0) or 0.0),
                "news_pressure": float(pair_state.get("news_pressure", 0.0) or 0.0),
                "stale": bool(pair_state.get("stale", True)),
                "event_eta_min": pair_state.get("event_eta_min"),
                "session_profile": str(pair_state.get("session_profile", "")),
                "calibration_profile": str(pair_state.get("calibration_profile", "")),
                "watchlist_tags": list(pair_state.get("watchlist_tags", [])),
                "reasons": list(pair_state.get("reasons", [])),
                "story_ids": list(pair_state.get("story_ids", [])),
            }
        )
        pair_history[pair_id] = _prune(history, observed_at, max_age_hours, max_entries_per_pair)
    return {str(key): list(value) for key, value in pair_history.items() if isinstance(value, list)}


def update_source_health_history(state: dict[str, Any],
                                 source_status: dict[str, Any],
                                 observed_at: datetime,
                                 *,
                                 max_entries: int = 96,
                                 keepalive_minutes: int = 30) -> list[dict[str, Any]]:
    history = state.setdefault("source_health_history", [])
    if not isinstance(history, list):
        history = []
        state["source_health_history"] = history
    last_entry = history[-1] if history else None
    current = {
        "observed_at": _isoformat(observed_at),
        "calendar_ok": bool(source_status.get("calendar", {}).get("ok", False)),
        "calendar_stale": bool(source_status.get("calendar", {}).get("stale", True)),
        "gdelt_ok": bool(source_status.get("gdelt", {}).get("ok", False)),
        "gdelt_stale": bool(source_status.get("gdelt", {}).get("stale", True)),
        "official_ok": bool(source_status.get("official", {}).get("ok", True)),
        "official_stale": bool(source_status.get("official", {}).get("stale", False)),
    }
    should_append = True
    if isinstance(last_entry, dict):
        last_dt = _parse_iso8601(last_entry.get("observed_at"))
        elapsed_minutes = int((observed_at - last_dt).total_seconds() / 60.0) if last_dt else keepalive_minutes + 1
        comparable = dict(current)
        comparable.pop("observed_at", None)
        last_comparable = dict(last_entry)
        last_comparable.pop("observed_at", None)
        should_append = comparable != last_comparable or elapsed_minutes >= keepalive_minutes
    if should_append:
        history.append(current)
    if len(history) > max_entries:
        history = history[-max_entries:]
        state["source_health_history"] = history
    return list(history)


def build_replay_report(snapshot: dict[str, Any],
                        pair_history: dict[str, list[dict[str, Any]]],
                        source_history: list[dict[str, Any]]) -> dict[str, Any]:
    pairs = snapshot.get("pairs", {})
    currencies = snapshot.get("currencies", {})
    top_pairs = sorted(
        (dict(pair=pair_id, **pair_state) for pair_id, pair_state in pairs.items()),
        key=lambda row: (float(row.get("news_risk_score", 0.0) or 0.0), row["pair"]),
        reverse=True,
    )[:18]
    return {
        "generated_at": snapshot.get("generated_at", ""),
        "pair_timelines": {
            pair_id: list(entries[-12:])
            for pair_id, entries in pair_history.items()
            if pair_id in {row["pair"] for row in top_pairs}
        },
        "source_health_timeline": list(source_history[-24:]),
        "top_pairs": top_pairs,
        "top_currencies": sorted(
            (dict(currency=currency, **payload) for currency, payload in currencies.items()),
            key=lambda row: (float(row.get("risk_score", 0.0) or 0.0), row["currency"]),
            reverse=True,
        )[:10],
        "stories": list(snapshot.get("stories", []))[:24],
    }


def write_replay_artifacts(snapshot: dict[str, Any],
                           pair_history: dict[str, list[dict[str, Any]]],
                           source_history: list[dict[str, Any]]) -> dict[str, Any]:
    report = build_replay_report(snapshot, pair_history, source_history)
    lines = [
        "pair_id\tobserved_at\tobserved_at_unix\ttrade_gate\tnews_risk_score\tnews_pressure\tstale\tevent_eta_min\tsession_profile\tcalibration_profile\twatchlist_tags\tstory_ids\treasons"
    ]
    for pair_id in sorted(pair_history):
        for entry in pair_history.get(pair_id, []):
            observed_at = str(entry.get("observed_at", ""))
            observed_dt = _parse_iso8601(observed_at)
            lines.append(
                "\t".join(
                    [
                        pair_id,
                        observed_at,
                        str(int(observed_dt.timestamp()) if observed_dt else 0),
                        str(entry.get("trade_gate", "UNKNOWN")),
                        str(entry.get("news_risk_score", 0.0)),
                        str(entry.get("news_pressure", 0.0)),
                        "1" if bool(entry.get("stale", True)) else "0",
                        "" if entry.get("event_eta_min") is None else str(entry.get("event_eta_min")),
                        str(entry.get("session_profile", "")),
                        str(entry.get("calibration_profile", "")),
                        ",".join(str(tag) for tag in entry.get("watchlist_tags", [])),
                        ",".join(str(tag) for tag in entry.get("story_ids", [])),
                        " | ".join(str(reason) for reason in entry.get("reasons", [])),
                    ]
                )
            )
    COMMON_NEWSPULSE_REPLAY_FLAT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    json_dump(NEWSPULSE_REPLAY_REPORT_PATH, report)
    return report


def rebuild_replay_report_from_history(history_path: Path | None = None,
                                       *,
                                       pair_filter: str = "",
                                       hours_back: int = 48,
                                       persist: bool | None = None) -> dict[str, Any]:
    target_path = history_path or NEWSPULSE_LOCAL_HISTORY_PATH
    should_persist = (not pair_filter) if persist is None else bool(persist)
    if not target_path.exists():
        report = {
            "generated_at": "",
            "pair_timelines": {},
            "source_health_timeline": [],
            "top_pairs": [],
            "top_currencies": [],
            "stories": [],
        }
        if should_persist:
            json_dump(NEWSPULSE_REPLAY_REPORT_PATH, report)
        return report

    now_dt = datetime.now(timezone.utc)
    cutoff = now_dt - timedelta(hours=max(hours_back, 1))
    snapshots: list[dict[str, Any]] = []
    with target_path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if payload.get("record_type") != "snapshot":
                continue
            snapshot = payload.get("snapshot", {})
            if not isinstance(snapshot, dict):
                continue
            generated_at = _parse_iso8601(snapshot.get("generated_at"))
            if generated_at is None or generated_at < cutoff:
                continue
            snapshots.append(snapshot)

    if not snapshots:
        report = {
            "generated_at": "",
            "pair_timelines": {},
            "source_health_timeline": [],
            "top_pairs": [],
            "top_currencies": [],
            "stories": [],
        }
        if should_persist:
            json_dump(NEWSPULSE_REPLAY_REPORT_PATH, report)
        return report

    snapshots.sort(key=lambda row: str(row.get("generated_at", "")))
    synthetic_state: dict[str, Any] = {}
    for snapshot in snapshots:
        observed_at = _parse_iso8601(snapshot.get("generated_at")) or now_dt
        update_pair_gate_history(synthetic_state, snapshot.get("pairs", {}), observed_at, max_age_hours=hours_back, keepalive_minutes=999999)
        update_source_health_history(synthetic_state, snapshot.get("source_status", {}), observed_at, keepalive_minutes=999999)

    pair_history = synthetic_state.get("pair_gate_history", {})
    if pair_filter:
        normalized = pair_filter.strip().upper()
        pair_history = {key: value for key, value in pair_history.items() if key == normalized}
    source_history = synthetic_state.get("source_health_history", [])
    report = build_replay_report(snapshots[-1], pair_history, source_history)
    if should_persist:
        json_dump(NEWSPULSE_REPLAY_REPORT_PATH, report)
    return report


def write_history_mirror(record: dict[str, Any]) -> None:
    for path in (COMMON_NEWSPULSE_HISTORY, NEWSPULSE_LOCAL_HISTORY_PATH):
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
