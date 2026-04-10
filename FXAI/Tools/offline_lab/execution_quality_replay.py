from __future__ import annotations

import json
from collections import Counter
from typing import Any

from .execution_quality_contracts import (
    EXECUTION_QUALITY_REPLAY_REPORT_PATH,
    execution_quality_runtime_history_path,
    ensure_execution_quality_dirs,
    isoformat_utc,
    json_dump,
)
from .microstructure_contracts import parse_iso8601, utc_now


def _load_history_records(symbol: str, hours_back: int) -> list[dict[str, Any]]:
    now_dt = utc_now()
    cutoff = now_dt.timestamp() - float(max(hours_back, 1) * 3600)
    paths = []
    symbol_filter = str(symbol or "").strip().upper()
    if symbol_filter:
        paths.append(execution_quality_runtime_history_path(symbol_filter))
    else:
        runtime_dir = execution_quality_runtime_history_path("EURUSD").parent
        paths.extend(sorted(runtime_dir.glob("fxai_execution_quality_history_*.ndjson")))

    out: list[dict[str, Any]] = []
    for path in paths:
        if not path.exists():
            continue
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except Exception:
                continue
            if not isinstance(payload, dict):
                continue
            observed_at = parse_iso8601(str(payload.get("generated_at", "") or payload.get("observed_at", "")))
            if observed_at is None or observed_at.timestamp() < cutoff:
                continue
            out.append(payload)
    return out


def build_execution_quality_replay_report(*, symbol: str = "", hours_back: int = 72) -> dict[str, Any]:
    ensure_execution_quality_dirs()
    records = _load_history_records(symbol, hours_back)
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        current_symbol = str(record.get("symbol", "") or record.get("state", {}).get("symbol", "")).strip().upper()
        if not current_symbol:
            continue
        grouped.setdefault(current_symbol, []).append(record)

    symbols_payload: list[dict[str, Any]] = []
    for current_symbol, items in sorted(grouped.items()):
        state_counts = Counter()
        tier_counts = Counter()
        reason_counts = Counter()
        max_spread_risk = 0.0
        max_slippage_risk = 0.0
        min_quality = 1.0
        recent_transitions: list[dict[str, Any]] = []
        previous_state = ""
        latest: dict[str, Any] | None = None

        for record in sorted(items, key=lambda item: str(item.get("generated_at", ""))):
            state = record.get("state", {})
            if not isinstance(state, dict):
                continue
            latest = record
            execution_state = str(state.get("execution_state", "UNKNOWN") or "UNKNOWN")
            tier_kind = str(state.get("selected_tier_kind", "GLOBAL") or "GLOBAL")
            state_counts[execution_state] += 1
            tier_counts[tier_kind] += 1
            for reason in state.get("reason_codes", []):
                reason_counts[str(reason)] += 1
            max_spread_risk = max(max_spread_risk, float(state.get("spread_widening_risk", 0.0) or 0.0))
            max_slippage_risk = max(max_slippage_risk, float(state.get("slippage_risk", 0.0) or 0.0))
            min_quality = min(min_quality, float(state.get("execution_quality_score", 1.0) or 1.0))
            if previous_state and execution_state != previous_state:
                recent_transitions.append(
                    {"type": "execution_state", "from": previous_state, "to": execution_state, "at": record.get("generated_at", "")}
                )
            previous_state = execution_state

        symbols_payload.append(
            {
                "symbol": current_symbol,
                "observations": len(items),
                "state_counts": dict(sorted(state_counts.items())),
                "tier_counts": dict(sorted(tier_counts.items())),
                "max_spread_widening_risk": round(max_spread_risk, 6),
                "max_slippage_risk": round(max_slippage_risk, 6),
                "min_execution_quality_score": round(min_quality if items else 0.0, 6),
                "top_reasons": [{"reason": name, "count": count} for name, count in reason_counts.most_common(8)],
                "recent_transitions": recent_transitions[-12:],
                "latest": latest or {},
            }
        )

    payload = {
        "generated_at": isoformat_utc(),
        "hours_back": int(hours_back),
        "symbol_count": len(symbols_payload),
        "symbols": symbols_payload,
        "history_path": str(execution_quality_runtime_history_path(symbol.strip().upper() or "EURUSD").parent),
    }
    json_dump(EXECUTION_QUALITY_REPLAY_REPORT_PATH, payload)
    return payload
