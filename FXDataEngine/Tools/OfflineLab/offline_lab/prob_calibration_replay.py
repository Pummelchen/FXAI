from __future__ import annotations

import json
from collections import defaultdict
from datetime import timedelta
from pathlib import Path
from typing import Any

from .common import parse_iso8601
from .prob_calibration_contracts import (
    PROB_CALIBRATION_REPLAY_REPORT_PATH,
    PROB_CALIBRATION_SCHEMA_VERSION,
    ensure_prob_calibration_dirs,
    isoformat_utc,
    json_dump,
    prob_calibration_runtime_history_path,
    utc_now,
)


def _iter_history_files(symbol: str = "") -> list[Path]:
    ensure_prob_calibration_dirs()
    if symbol:
        path = prob_calibration_runtime_history_path(symbol)
        return [path] if path.exists() else []
    runtime_dir = prob_calibration_runtime_history_path("EURUSD").parent
    return sorted(runtime_dir.glob("fxai_prob_calibration_history_*.ndjson"))


def _load_history_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        text = path.read_text(encoding="utf-8")
    except Exception:
        return rows
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except Exception:
            continue
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def build_prob_calibration_replay_report(*, symbol: str = "", hours_back: int = 72) -> dict[str, Any]:
    now_dt = utc_now()
    cutoff = now_dt - timedelta(hours=max(int(hours_back), 1))
    by_symbol: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for path in _iter_history_files(symbol):
        for row in _load_history_rows(path):
            generated_at = parse_iso8601(str(row.get("generated_at", "")))
            if generated_at is not None and generated_at < cutoff:
                continue
            row_symbol = str(row.get("symbol", "") or "").upper()
            if not row_symbol:
                continue
            by_symbol[row_symbol].append(row)

    symbols: list[dict[str, Any]] = []
    for row_symbol, rows in sorted(by_symbol.items()):
        rows.sort(key=lambda item: str(item.get("generated_at", "")))
        action_counts: dict[str, int] = defaultdict(int)
        reason_counts: dict[str, int] = defaultdict(int)
        tier_counts: dict[str, int] = defaultdict(int)
        fallback_count = 0
        abstain_count = 0
        avg_confidence_sum = 0.0
        avg_edge_sum = 0.0
        avg_uncertainty_sum = 0.0
        min_edge = None
        max_edge = None
        transitions: list[dict[str, Any]] = []
        previous_action = ""

        for row in rows:
            state = dict(row.get("state", {}))
            final_action = str(state.get("final_action", "") or "SKIP").upper()
            action_counts[final_action] += 1
            if bool(state.get("abstain", False)):
                abstain_count += 1
            if bool(state.get("fallback_used", False)):
                fallback_count += 1
            tier_counts[str(state.get("selected_tier_kind", "") or "UNKNOWN")] += 1
            avg_confidence_sum += float(state.get("calibrated_confidence", 0.0) or 0.0)
            avg_edge = float(state.get("edge_after_costs_points", 0.0) or 0.0)
            avg_edge_sum += avg_edge
            avg_uncertainty_sum += float(state.get("uncertainty_score", 0.0) or 0.0)
            min_edge = avg_edge if min_edge is None else min(min_edge, avg_edge)
            max_edge = avg_edge if max_edge is None else max(max_edge, avg_edge)

            for reason in list(state.get("reason_codes", []))[:12]:
                reason_counts[str(reason)] += 1

            if previous_action and previous_action != final_action:
                transitions.append(
                    {
                        "type": "action_change",
                        "from": previous_action,
                        "to": final_action,
                        "at": row.get("generated_at", ""),
                    }
                )
            previous_action = final_action

        latest = rows[-1]
        observation_count = max(len(rows), 1)
        symbols.append(
            {
                "symbol": row_symbol,
                "observations": len(rows),
                "abstain_count": abstain_count,
                "fallback_count": fallback_count,
                "average_confidence": round(avg_confidence_sum / observation_count, 6),
                "average_edge_after_costs_points": round(avg_edge_sum / observation_count, 6),
                "average_uncertainty_score": round(avg_uncertainty_sum / observation_count, 6),
                "min_edge_after_costs_points": round(min_edge or 0.0, 6),
                "max_edge_after_costs_points": round(max_edge or 0.0, 6),
                "latest": latest,
                "action_counts": dict(sorted(action_counts.items())),
                "tier_counts": dict(sorted(tier_counts.items())),
                "top_reasons": [
                    {"reason": reason, "count": count}
                    for reason, count in sorted(reason_counts.items(), key=lambda item: (-item[1], item[0]))[:10]
                ],
                "recent_transitions": transitions[-16:],
            }
        )

    payload = {
        "schema_version": PROB_CALIBRATION_SCHEMA_VERSION,
        "generated_at": isoformat_utc(now_dt),
        "hours_back": max(int(hours_back), 1),
        "symbol_filter": symbol.upper() if symbol else "",
        "symbol_count": len(symbols),
        "symbols": symbols,
    }
    json_dump(PROB_CALIBRATION_REPLAY_REPORT_PATH, payload)
    return payload
