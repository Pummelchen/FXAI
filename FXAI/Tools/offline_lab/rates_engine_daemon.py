from __future__ import annotations

import json
import time
from typing import Any

from .rates_engine import run_rates_engine_cycle
from .rates_engine_config import ensure_default_files, load_config, validate_config_payload
from .rates_engine_contracts import (
    RATES_ENGINE_CONFIG_PATH,
    RATES_ENGINE_INPUTS_PATH,
    RATES_ENGINE_STATE_PATH,
    RATES_ENGINE_STATUS_PATH,
    isoformat_utc,
    utc_now,
)
from .rates_engine_inputs import ensure_default_inputs_file, load_inputs, validate_inputs_payload


def validate_rates_engine_config() -> dict[str, Any]:
    ensure_default_files()
    ensure_default_inputs_file()
    config = load_config()
    inputs = load_inputs()
    validate_config_payload(config)
    validate_inputs_payload(inputs)
    return {
        "ok": True,
        "config_path": str(RATES_ENGINE_CONFIG_PATH),
        "inputs_path": str(RATES_ENGINE_INPUTS_PATH),
        "currencies": sorted(dict(config.get("currencies", {})).keys()),
        "manual_input_currencies": sorted(
            code
            for code, spec in dict(inputs.get("currencies", {})).items()
            if isinstance(spec, dict) and any(spec.get(field) not in (None, "", "null") for field in ("front_end_level", "expected_path_level", "curve_slope_2s10s"))
        ),
    }


def run_rates_engine_once() -> dict[str, Any]:
    return run_rates_engine_cycle()


def _load_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _write_daemon_health(payload: dict[str, Any]) -> None:
    state = _load_json(RATES_ENGINE_STATE_PATH)
    if not isinstance(state, dict):
        state = {}
    state["daemon_health"] = dict(payload)
    RATES_ENGINE_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    RATES_ENGINE_STATE_PATH.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")

    status = _load_json(RATES_ENGINE_STATUS_PATH)
    if not isinstance(status, dict):
        status = {}
    status.setdefault("generated_at", isoformat_utc())
    status["daemon"] = dict(payload)
    RATES_ENGINE_STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RATES_ENGINE_STATUS_PATH.write_text(json.dumps(status, indent=2, sort_keys=True), encoding="utf-8")


def rates_engine_health_snapshot() -> dict[str, Any]:
    status = _load_json(RATES_ENGINE_STATUS_PATH)
    state = _load_json(RATES_ENGINE_STATE_PATH)
    daemon = status.get("daemon", {})
    if not isinstance(daemon, dict) or not daemon:
        daemon = state.get("daemon_health", {})
    return {
        "status_path": str(RATES_ENGINE_STATUS_PATH),
        "state_path": str(RATES_ENGINE_STATE_PATH),
        "generated_at": status.get("generated_at", ""),
        "daemon": daemon if isinstance(daemon, dict) else {},
        "source_status": status.get("source_status", {}),
        "health": status.get("health", {}),
    }


def run_rates_engine_daemon(iterations: int = 0, interval_seconds: int | None = None) -> dict[str, Any]:
    config = load_config()
    interval = int(interval_seconds or config.get("poll_interval_sec", 120) or 120)
    if interval < 30:
        interval = 30
    attempts = 0
    successful_cycles = 0
    consecutive_failures = 0
    last_payload: dict[str, Any] | None = None
    while iterations <= 0 or attempts < iterations:
        started_at = utc_now()
        started_iso = isoformat_utc(started_at)
        attempts += 1
        try:
            last_payload = run_rates_engine_cycle(
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
            if not bool(last_payload.get("proxy_ok", False)):
                degraded_reasons.append("proxy_source_not_ready")
            if int(last_payload.get("manual_inputs_used", 0) or 0) <= 0:
                degraded_reasons.append("manual_numeric_rates_absent")
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
