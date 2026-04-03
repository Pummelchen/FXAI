from __future__ import annotations

RUNTIME_MODES: dict[str, dict[str, object]] = {
    "research": {
        "runtime_mode": "research",
        "telemetry_level": "full",
        "shadow_enabled": 1,
        "snapshot_detail": "full",
        "max_runtime_models": 12,
        "performance_budget_ms": 12.0,
    },
    "production": {
        "runtime_mode": "production",
        "telemetry_level": "lean",
        "shadow_enabled": 0,
        "snapshot_detail": "lean",
        "max_runtime_models": 6,
        "performance_budget_ms": 6.0,
    },
}


def _norm_mode(raw: str | None) -> str:
    text = str(raw or "").strip().lower()
    return text if text in RUNTIME_MODES else "research"


def resolve_runtime_mode(raw: str | None) -> dict[str, object]:
    return dict(RUNTIME_MODES[_norm_mode(raw)])


def runtime_mode_name(raw: str | None) -> str:
    return str(resolve_runtime_mode(raw)["runtime_mode"])
