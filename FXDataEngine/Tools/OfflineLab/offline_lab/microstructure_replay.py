from __future__ import annotations

import json
from collections import Counter
from typing import Any

from .microstructure_contracts import (
    COMMON_MICROSTRUCTURE_HISTORY,
    MICROSTRUCTURE_REPLAY_REPORT_PATH,
    ensure_microstructure_dirs,
    isoformat_utc,
    json_dump,
    parse_iso8601,
    utc_now,
)


def _load_history_records(hours_back: int) -> list[dict[str, Any]]:
    now_dt = utc_now()
    cutoff = now_dt.timestamp() - float(max(hours_back, 1) * 3600)
    out: list[dict[str, Any]] = []
    if not COMMON_MICROSTRUCTURE_HISTORY.exists():
        return out
    for raw_line in COMMON_MICROSTRUCTURE_HISTORY.read_text(encoding="utf-8").splitlines():
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


def build_microstructure_replay_report(*, symbol: str = "", hours_back: int = 24) -> dict[str, Any]:
    ensure_microstructure_dirs()
    records = _load_history_records(hours_back)
    symbol_filter = str(symbol or "").strip().upper()
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        state = record.get("state", {})
        if not isinstance(state, dict):
            continue
        current_symbol = str(record.get("symbol", "") or state.get("symbol", "")).strip().upper()
        if not current_symbol:
            continue
        if symbol_filter and current_symbol != symbol_filter:
            continue
        grouped.setdefault(current_symbol, []).append(record)

    symbols_payload: list[dict[str, Any]] = []
    for current_symbol, items in sorted(grouped.items()):
        regime_counts = Counter()
        gate_counts = Counter()
        reason_counts = Counter()
        max_hostile = 0.0
        max_liquidity = 0.0
        recent_transitions: list[dict[str, Any]] = []
        previous_regime = ""
        previous_gate = ""
        latest: dict[str, Any] | None = None

        for record in sorted(items, key=lambda item: str(item.get("generated_at", ""))):
            state = record.get("state", {})
            if not isinstance(state, dict):
                continue
            latest = record
            regime = str(state.get("microstructure_regime", "UNKNOWN") or "UNKNOWN")
            gate = str(state.get("trade_gate", "UNKNOWN") or "UNKNOWN")
            regime_counts[regime] += 1
            gate_counts[gate] += 1
            max_hostile = max(max_hostile, float(state.get("hostile_execution_score", 0.0) or 0.0))
            max_liquidity = max(max_liquidity, float(state.get("liquidity_stress_score", 0.0) or 0.0))
            for reason in state.get("reasons", []):
                reason_counts[str(reason)] += 1
            if previous_regime and regime != previous_regime:
                recent_transitions.append({"type": "regime", "from": previous_regime, "to": regime, "at": record.get("generated_at", "")})
            if previous_gate and gate != previous_gate:
                recent_transitions.append({"type": "gate", "from": previous_gate, "to": gate, "at": record.get("generated_at", "")})
            previous_regime = regime
            previous_gate = gate

        symbols_payload.append(
            {
                "symbol": current_symbol,
                "observations": len(items),
                "regime_counts": dict(sorted(regime_counts.items())),
                "gate_counts": dict(sorted(gate_counts.items())),
                "max_hostile_execution_score": round(max_hostile, 6),
                "max_liquidity_stress_score": round(max_liquidity, 6),
                "top_reasons": [
                    {"reason": name, "count": count}
                    for name, count in reason_counts.most_common(8)
                ],
                "recent_transitions": recent_transitions[-12:],
                "latest": latest or {},
            }
        )

    payload = {
        "generated_at": isoformat_utc(),
        "hours_back": int(hours_back),
        "symbol_count": len(symbols_payload),
        "symbols": symbols_payload,
        "history_path": str(COMMON_MICROSTRUCTURE_HISTORY),
    }
    json_dump(MICROSTRUCTURE_REPLAY_REPORT_PATH, payload)
    return payload
