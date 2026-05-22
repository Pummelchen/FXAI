from __future__ import annotations

import json
import time
from typing import Any

from .newspulse_config import ensure_default_files, load_config, validate_config_payload
from .newspulse_contracts import (
    NEWSPULSE_CONFIG_PATH,
    NEWSPULSE_SOURCES_PATH,
    NEWSPULSE_STATE_PATH,
    NEWSPULSE_STATUS_PATH,
    isoformat_utc,
    utc_now,
)
from .newspulse_policy import ensure_default_policy_file, load_policy, validate_policy_payload
from .newspulse_fusion import run_newspulse_cycle


def validate_newspulse_config() -> dict[str, Any]:
    ensure_default_files()
    config, sources = load_config()
    policy = load_policy()
    validate_config_payload(config, sources)
    validate_policy_payload(policy)
    return {
        "ok": True,
        "config_path": str(NEWSPULSE_CONFIG_PATH),
        "sources_path": str(NEWSPULSE_SOURCES_PATH),
        "policy_path": str(ensure_default_policy_file()),
        "currencies": sorted(config["currencies"].keys()),
        "topic_groups": sorted(config["topic_groups"].keys()),
    }


def run_newspulse_once() -> dict[str, Any]:
    return run_newspulse_cycle()


def _load_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _write_daemon_health(payload: dict[str, Any]) -> None:
    state = _load_json(NEWSPULSE_STATE_PATH)
    if not isinstance(state, dict):
        state = {}
    state["daemon_health"] = dict(payload)
    NEWSPULSE_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    NEWSPULSE_STATE_PATH.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")

    status = _load_json(NEWSPULSE_STATUS_PATH)
    if not isinstance(status, dict):
        status = {}
    status.setdefault("generated_at", isoformat_utc())
    status["daemon"] = dict(payload)
    NEWSPULSE_STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    NEWSPULSE_STATUS_PATH.write_text(json.dumps(status, indent=2, sort_keys=True), encoding="utf-8")


def newspulse_health_snapshot() -> dict[str, Any]:
    status = _load_json(NEWSPULSE_STATUS_PATH)
    state = _load_json(NEWSPULSE_STATE_PATH)
    daemon = status.get("daemon", {})
    if not isinstance(daemon, dict) or not daemon:
        daemon = state.get("daemon_health", {})
    return {
        "status_path": str(NEWSPULSE_STATUS_PATH),
        "state_path": str(NEWSPULSE_STATE_PATH),
        "generated_at": status.get("generated_at", ""),
        "daemon": daemon if isinstance(daemon, dict) else {},
        "source_status": status.get("source_status", {}),
        "health": status.get("health", {}),
    }


def run_newspulse_daemon(iterations: int = 0, interval_seconds: int | None = None) -> dict[str, Any]:
    config, _sources = load_config()
    interval = int(interval_seconds or config.get("poll_interval_sec", 60) or 60)
    if interval < 15:
        interval = 15
    attempts = 0
    successful_cycles = 0
    consecutive_failures = 0
    last_payload: dict[str, Any] | None = None
    while iterations <= 0 or attempts < iterations:
        started_at = utc_now()
        started_iso = isoformat_utc(started_at)
        attempts += 1
        try:
            last_payload = run_newspulse_cycle(
                daemon_context={
                    "mode": "daemon",
                    "heartbeat_at": started_iso,
                    "last_cycle_started_at": started_iso,
                    "interval_seconds": interval,
                    "cycles_completed": successful_cycles,
                    "consecutive_failures": consecutive_failures,
                    "degraded": consecutive_failures > 0,
                    "degraded_reasons": ["previous_cycle_failures"] if consecutive_failures > 0 else [],
                    "last_error": "",
                }
            )
            successful_cycles += 1
            consecutive_failures = 0
            finished_at = utc_now()
            degraded_reasons = []
            source_status = dict(last_payload.get("source_status", {}))
            source_reason_map = {
                "calendar": "calendar_source_not_ready",
                "gdelt": "gdelt_source_not_ready",
                "official": "official_feed_not_ready",
            }
            for source_name, degraded_reason in source_reason_map.items():
                status = dict(source_status.get(source_name, {}))
                if not bool(status.get("required", False)):
                    continue
                if not bool(status.get("ok", False)) or bool(status.get("stale", True)):
                    degraded_reasons.append(degraded_reason)
            _write_daemon_health(
                {
                    "mode": "daemon",
                    "heartbeat_at": isoformat_utc(finished_at),
                    "last_cycle_started_at": started_iso,
                    "last_cycle_finished_at": isoformat_utc(finished_at),
                    "last_cycle_duration_sec": round((finished_at - started_at).total_seconds(), 6),
                    "interval_seconds": interval,
                    "cycles_completed": successful_cycles,
                    "consecutive_failures": consecutive_failures,
                    "degraded": bool(degraded_reasons),
                    "degraded_reasons": degraded_reasons,
                    "backoff_until": "",
                    "last_error": "",
                }
            )
        except Exception as exc:
            consecutive_failures += 1
            failure_time = utc_now()
            _write_daemon_health(
                {
                    "mode": "daemon",
                    "heartbeat_at": isoformat_utc(failure_time),
                    "last_cycle_started_at": started_iso,
                    "last_cycle_finished_at": isoformat_utc(failure_time),
                    "last_cycle_duration_sec": round((failure_time - started_at).total_seconds(), 6),
                    "interval_seconds": interval,
                    "cycles_completed": successful_cycles,
                    "consecutive_failures": consecutive_failures,
                    "degraded": True,
                    "degraded_reasons": ["cycle_exception"],
                    "backoff_until": "",
                    "last_error": str(exc),
                }
            )
        if iterations > 0 and attempts >= iterations:
            break
        for _ in range(interval):
            time.sleep(1.0)
    return {
        "iterations": attempts,
        "interval_seconds": interval,
        "successful_iterations": successful_cycles,
        "consecutive_failures": consecutive_failures,
        "last_payload": last_payload or {},
    }
