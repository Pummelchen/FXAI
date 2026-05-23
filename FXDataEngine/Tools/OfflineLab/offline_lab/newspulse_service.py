from __future__ import annotations

from .newspulse_config import ensure_default_files


def install_calendar_service(compile_service: bool = True) -> dict[str, object]:
    ensure_default_files()
    return {
        "installed": True,
        "compiled": False,
        "compile_requested": bool(compile_service),
        "collector": "offline_news_sources",
        "note": "NewsPulse is populated by the Swift/Python offline lab data sources; no separate terminal service is installed.",
    }


def compile_calendar_service(timeout_sec: int = 600) -> dict[str, object]:
    _ = timeout_sec
    return {
        "compiled": False,
        "collector": "offline_news_sources",
        "note": "No separate terminal service is compiled for the Swift-era NewsPulse collector.",
    }
