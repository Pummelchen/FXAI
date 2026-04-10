from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import timedelta
from pathlib import Path
from typing import Any

from .cross_asset_contracts import (
    COMMON_CROSS_ASSET_HISTORY,
    CROSS_ASSET_REPLAY_REPORT_PATH,
    isoformat_utc,
    json_dump,
    parse_iso8601,
    utc_now,
)


def _load_snapshot_records(path: Path, hours_back: int) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    now_dt = utc_now()
    cutoff = now_dt - timedelta(hours=max(hours_back, 1))
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except Exception:
            continue
        if not isinstance(payload, dict) or payload.get("record_type") != "snapshot":
            continue
        generated_at = parse_iso8601(str(payload.get("generated_at", "")))
        if generated_at is None or generated_at < cutoff:
            continue
        snapshot = payload.get("snapshot")
        if isinstance(snapshot, dict):
            records.append(snapshot)
    return records


def build_cross_asset_replay_report(symbol: str = "", hours_back: int = 72) -> dict[str, Any]:
    records = _load_snapshot_records(COMMON_CROSS_ASSET_HISTORY, hours_back)
    symbol_filter = str(symbol or "").strip().upper()
    pair_timelines: dict[str, list[dict[str, Any]]] = defaultdict(list)
    global_timeline: list[dict[str, Any]] = []

    for snapshot in records:
        generated_at = str(snapshot.get("generated_at", ""))
        state_labels = dict(snapshot.get("state_labels", {}))
        global_timeline.append(
            {
                "observed_at": generated_at,
                "macro_state": str(state_labels.get("macro_state", "")),
                "risk_state": str(state_labels.get("risk_state", "")),
                "liquidity_state": str(state_labels.get("liquidity_state", "")),
                "reason_codes": list(snapshot.get("reason_codes", [])),
            }
        )
        for pair_id, state in dict(snapshot.get("pair_states", {})).items():
            pair_name = str(pair_id).upper()
            if symbol_filter and pair_name != symbol_filter:
                continue
            if not isinstance(state, dict):
                continue
            pair_timelines[pair_name].append(
                {
                    "observed_at": generated_at,
                    "trade_gate": str(state.get("trade_gate", "")),
                    "macro_state": str(state.get("macro_state", "")),
                    "risk_state": str(state.get("risk_state", "")),
                    "liquidity_state": str(state.get("liquidity_state", "")),
                    "pair_cross_asset_risk_score": float(state.get("pair_cross_asset_risk_score", 0.0) or 0.0),
                    "reasons": list(state.get("reasons", [])),
                }
            )

    symbols_payload = []
    for pair_id, timeline in sorted(pair_timelines.items()):
        if not timeline:
            continue
        gate_counts = Counter(item["trade_gate"] for item in timeline)
        macro_counts = Counter(item["macro_state"] for item in timeline)
        top_reasons = Counter()
        transitions: list[dict[str, Any]] = []
        prev_gate = ""
        prev_macro = ""
        for item in timeline:
            for reason in item.get("reasons", []):
                top_reasons[str(reason)] += 1
            gate = str(item["trade_gate"])
            macro = str(item["macro_state"])
            if prev_gate and gate != prev_gate:
                transitions.append({"type": "trade_gate", "from": prev_gate, "to": gate, "observed_at": item["observed_at"]})
            if prev_macro and macro != prev_macro:
                transitions.append({"type": "macro_state", "from": prev_macro, "to": macro, "observed_at": item["observed_at"]})
            prev_gate = gate
            prev_macro = macro
        symbols_payload.append(
            {
                "symbol": pair_id,
                "observations": len(timeline),
                "latest": timeline[-1],
                "gate_counts": dict(gate_counts),
                "macro_counts": dict(macro_counts),
                "top_reasons": [{"reason": reason, "count": count} for reason, count in top_reasons.most_common(8)],
                "recent_transitions": transitions[-12:],
                "timeline": timeline[-48:],
            }
        )

    global_counts = Counter(item["macro_state"] for item in global_timeline)
    risk_counts = Counter(item["risk_state"] for item in global_timeline)
    liquidity_counts = Counter(item["liquidity_state"] for item in global_timeline)
    payload = {
        "generated_at": isoformat_utc(),
        "hours_back": int(hours_back),
        "symbol_count": len(symbols_payload),
        "symbols": symbols_payload,
        "global_state_counts": dict(global_counts),
        "risk_state_counts": dict(risk_counts),
        "liquidity_state_counts": dict(liquidity_counts),
        "history_path": str(COMMON_CROSS_ASSET_HISTORY),
        "report_path": str(CROSS_ASSET_REPLAY_REPORT_PATH),
    }
    json_dump(CROSS_ASSET_REPLAY_REPORT_PATH, payload)
    return payload
