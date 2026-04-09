from __future__ import annotations

import json
from collections import defaultdict
from datetime import timedelta
from pathlib import Path
from typing import Any

from .common import parse_iso8601
from .dynamic_ensemble_contracts import (
    DYNAMIC_ENSEMBLE_REPLAY_REPORT_PATH,
    DYNAMIC_ENSEMBLE_SCHEMA_VERSION,
    dynamic_ensemble_runtime_history_path,
    ensure_dynamic_ensemble_dirs,
    isoformat_utc,
    json_dump,
    utc_now,
)


def _iter_history_files(symbol: str = "") -> list[Path]:
    ensure_dynamic_ensemble_dirs()
    if symbol:
        path = dynamic_ensemble_runtime_history_path(symbol)
        return [path] if path.exists() else []
    runtime_dir = dynamic_ensemble_runtime_history_path("EURUSD").parent
    return sorted(runtime_dir.glob("fxai_dynamic_ensemble_history_*.ndjson"))


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


def build_dynamic_ensemble_replay_report(*, symbol: str = "", hours_back: int = 72) -> dict[str, Any]:
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
        posture_counts: dict[str, int] = defaultdict(int)
        action_counts: dict[str, int] = defaultdict(int)
        reason_counts: dict[str, int] = defaultdict(int)
        dominant_plugins: dict[str, int] = defaultdict(int)
        status_counts: dict[str, int] = defaultdict(int)
        quality_sum = 0.0
        abstain_max = 0.0
        transitions: list[dict[str, Any]] = []
        previous_posture = ""
        previous_action = ""

        for row in rows:
            ensemble = dict(row.get("ensemble", {}))
            posture = str(ensemble.get("trade_posture", "") or "UNKNOWN")
            action = str(ensemble.get("final_action", "") or "UNKNOWN")
            posture_counts[posture] += 1
            action_counts[action] += 1
            quality_sum += float(ensemble.get("ensemble_quality", 0.0) or 0.0)
            abstain_max = max(abstain_max, float(ensemble.get("abstain_bias", 0.0) or 0.0))

            for reason in list(ensemble.get("reasons", []))[:8]:
                reason_counts[str(reason)] += 1

            plugins = list(row.get("plugins", []))
            for plugin in plugins:
                status = str(plugin.get("status", "") or "UNKNOWN").upper()
                status_counts[status] += 1
            active_plugins = [plugin for plugin in plugins if str(plugin.get("status", "")).upper() == "ACTIVE"]
            if active_plugins:
                leader = max(
                    active_plugins,
                    key=lambda item: (
                        float(item.get("weight", 0.0) or 0.0),
                        float(item.get("trust", 0.0) or 0.0),
                        str(item.get("name", "")),
                    ),
                )
                dominant_plugins[str(leader.get("name", ""))] += 1

            if previous_posture and previous_posture != posture:
                transitions.append(
                    {
                        "type": "posture_change",
                        "from": previous_posture,
                        "to": posture,
                        "at": row.get("generated_at", ""),
                    }
                )
            if previous_action and previous_action != action:
                transitions.append(
                    {
                        "type": "action_change",
                        "from": previous_action,
                        "to": action,
                        "at": row.get("generated_at", ""),
                    }
                )
            previous_posture = posture
            previous_action = action

        latest = rows[-1]
        symbols.append(
            {
                "symbol": row_symbol,
                "observations": len(rows),
                "average_quality": round(quality_sum / max(len(rows), 1), 6),
                "max_abstain_bias": round(abstain_max, 6),
                "latest": latest,
                "posture_counts": dict(sorted(posture_counts.items())),
                "action_counts": dict(sorted(action_counts.items())),
                "plugin_status_counts": dict(sorted(status_counts.items())),
                "top_reasons": [
                    {"reason": reason, "count": count}
                    for reason, count in sorted(reason_counts.items(), key=lambda item: (-item[1], item[0]))[:8]
                ],
                "top_dominant_plugins": [
                    {"plugin": plugin, "count": count}
                    for plugin, count in sorted(dominant_plugins.items(), key=lambda item: (-item[1], item[0]))[:8]
                ],
                "recent_transitions": transitions[-16:],
            }
        )

    payload = {
        "schema_version": DYNAMIC_ENSEMBLE_SCHEMA_VERSION,
        "generated_at": isoformat_utc(now_dt),
        "hours_back": max(int(hours_back), 1),
        "symbol_filter": symbol.upper() if symbol else "",
        "symbol_count": len(symbols),
        "symbols": symbols,
    }
    json_dump(DYNAMIC_ENSEMBLE_REPLAY_REPORT_PATH, payload)
    return payload
