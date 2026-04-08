from __future__ import annotations

import time
from typing import Any

from .newspulse_config import ensure_default_files, load_config, validate_config_payload
from .newspulse_contracts import NEWSPULSE_CONFIG_PATH, NEWSPULSE_SOURCES_PATH
from .newspulse_fusion import run_newspulse_cycle


def validate_newspulse_config() -> dict[str, Any]:
    ensure_default_files()
    config, sources = load_config()
    validate_config_payload(config, sources)
    return {
        "ok": True,
        "config_path": str(NEWSPULSE_CONFIG_PATH),
        "sources_path": str(NEWSPULSE_SOURCES_PATH),
        "currencies": sorted(config["currencies"].keys()),
        "topic_groups": sorted(config["topic_groups"].keys()),
    }


def run_newspulse_once() -> dict[str, Any]:
    return run_newspulse_cycle()


def run_newspulse_daemon(iterations: int = 0, interval_seconds: int | None = None) -> dict[str, Any]:
    config, _sources = load_config()
    interval = int(interval_seconds or config.get("poll_interval_sec", 60) or 60)
    if interval < 15:
        interval = 15
    count = 0
    last_payload: dict[str, Any] | None = None
    while iterations <= 0 or count < iterations:
        last_payload = run_newspulse_cycle()
        count += 1
        if iterations > 0 and count >= iterations:
            break
        for _ in range(interval):
            time.sleep(1.0)
    return {
        "iterations": count,
        "interval_seconds": interval,
        "last_payload": last_payload or {},
    }
