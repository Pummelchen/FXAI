from __future__ import annotations

import json
from collections import defaultdict
from datetime import timedelta
from pathlib import Path
from typing import Any

from .adaptive_router_contracts import (
    ADAPTIVE_ROUTER_REPLAY_REPORT_PATH,
    ADAPTIVE_ROUTER_SCHEMA_VERSION,
    adaptive_router_runtime_history_path,
    ensure_adaptive_router_dirs,
    isoformat_utc,
    json_dump,
    utc_now,
)
from .common import parse_iso8601


def _iter_history_files(symbol: str = "") -> list[Path]:
    ensure_adaptive_router_dirs()
    if symbol:
        path = adaptive_router_runtime_history_path(symbol)
        return [path] if path.exists() else []
    runtime_dir = adaptive_router_runtime_history_path("EURUSD").parent
    return sorted(runtime_dir.glob("fxai_regime_router_history_*.ndjson"))


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


def build_adaptive_router_replay_report(*,
                                        symbol: str = "",
                                        hours_back: int = 72) -> dict[str, Any]:
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
        regime_counts: dict[str, int] = defaultdict(int)
        posture_counts: dict[str, int] = defaultdict(int)
        reason_counts: dict[str, int] = defaultdict(int)
        plugin_wins: dict[str, int] = defaultdict(int)
        transitions: list[dict[str, Any]] = []
        previous_regime = ""
        previous_posture = ""
        latest = rows[-1]

        for row in rows:
            regime = dict(row.get("regime", {}))
            router = dict(row.get("router", {}))
            top_label = str(regime.get("top_label", "") or "UNKNOWN")
            posture = str(router.get("trade_posture", "") or "UNKNOWN")
            regime_counts[top_label] += 1
            posture_counts[posture] += 1
            for reason in list(router.get("reasons", []))[:6]:
                reason_counts[str(reason)] += 1
            plugins = list(row.get("plugins", []))
            if plugins:
                top_plugin = max(
                    plugins,
                    key=lambda item: (
                        float(item.get("weight", 0.0) or 0.0),
                        float(item.get("suitability", 0.0) or 0.0),
                        str(item.get("name", "")),
                    ),
                )
                plugin_wins[str(top_plugin.get("name", ""))] += 1
            if previous_regime and previous_regime != top_label:
                transitions.append(
                    {
                        "type": "regime_change",
                        "from": previous_regime,
                        "to": top_label,
                        "at": row.get("generated_at", ""),
                    }
                )
            if previous_posture and previous_posture != posture:
                transitions.append(
                    {
                        "type": "posture_change",
                        "from": previous_posture,
                        "to": posture,
                        "at": row.get("generated_at", ""),
                    }
                )
            previous_regime = top_label
            previous_posture = posture

        symbols.append(
            {
                "symbol": row_symbol,
                "observations": len(rows),
                "latest": latest,
                "regime_counts": dict(sorted(regime_counts.items())),
                "posture_counts": dict(sorted(posture_counts.items())),
                "top_reasons": [
                    {"reason": reason, "count": count}
                    for reason, count in sorted(reason_counts.items(), key=lambda item: (-item[1], item[0]))[:8]
                ],
                "top_plugins": [
                    {"plugin": plugin, "count": count}
                    for plugin, count in sorted(plugin_wins.items(), key=lambda item: (-item[1], item[0]))[:8]
                ],
                "recent_transitions": transitions[-16:],
            }
        )

    payload = {
        "schema_version": ADAPTIVE_ROUTER_SCHEMA_VERSION,
        "generated_at": isoformat_utc(now_dt),
        "hours_back": max(int(hours_back), 1),
        "symbol_filter": symbol.upper() if symbol else "",
        "symbol_count": len(symbols),
        "symbols": symbols,
    }
    json_dump(ADAPTIVE_ROUTER_REPLAY_REPORT_PATH, payload)
    return payload
