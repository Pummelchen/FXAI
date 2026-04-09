from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import timedelta
from pathlib import Path
from typing import Any

from .rates_engine_contracts import (
    COMMON_RATES_HISTORY,
    RATES_ENGINE_REPLAY_REPORT_PATH,
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


def build_rates_replay_report(symbol: str = "", hours_back: int = 72) -> dict[str, Any]:
    records = _load_snapshot_records(COMMON_RATES_HISTORY, hours_back)
    symbol_filter = str(symbol or "").strip().upper()
    pair_timelines: dict[str, list[dict[str, Any]]] = defaultdict(list)
    currency_timelines: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for snapshot in records:
        generated_at = str(snapshot.get("generated_at", ""))
        for pair_id, state in dict(snapshot.get("pairs", {})).items():
            pair_name = str(pair_id).upper()
            if symbol_filter and pair_name != symbol_filter:
                continue
            if not isinstance(state, dict):
                continue
            pair_timelines[pair_name].append(
                {
                    "observed_at": generated_at,
                    "trade_gate": str(state.get("trade_gate", "")),
                    "rates_regime": str(state.get("rates_regime", "")),
                    "rates_risk_score": float(state.get("rates_risk_score", 0.0) or 0.0),
                    "policy_divergence_score": float(state.get("policy_divergence_score", 0.0) or 0.0),
                    "meeting_path_reprice_now": bool(state.get("meeting_path_reprice_now", False)),
                    "reasons": list(state.get("reasons", [])),
                }
            )
        if not symbol_filter:
            for currency, state in dict(snapshot.get("currencies", {})).items():
                if not isinstance(state, dict):
                    continue
                currency_timelines[str(currency).upper()].append(
                    {
                        "observed_at": generated_at,
                        "policy_repricing_score": float(state.get("policy_repricing_score", 0.0) or 0.0),
                        "policy_surprise_score": float(state.get("policy_surprise_score", 0.0) or 0.0),
                        "policy_uncertainty_score": float(state.get("policy_uncertainty_score", 0.0) or 0.0),
                        "curve_shape_regime": str(state.get("curve_shape_regime", "")),
                    }
                )

    symbols_payload = []
    for pair_id, timeline in sorted(pair_timelines.items()):
        if not timeline:
            continue
        gate_counts = Counter(item["trade_gate"] for item in timeline)
        regime_counts = Counter(item["rates_regime"] for item in timeline)
        top_reasons = Counter()
        recent_transitions: list[dict[str, Any]] = []
        previous_gate = ""
        previous_regime = ""
        for item in timeline:
            for reason in item.get("reasons", []):
                top_reasons[str(reason)] += 1
            gate = str(item["trade_gate"])
            regime = str(item["rates_regime"])
            if previous_gate and gate != previous_gate:
                recent_transitions.append(
                    {
                        "type": "trade_gate",
                        "from": previous_gate,
                        "to": gate,
                        "observed_at": item["observed_at"],
                    }
                )
            if previous_regime and regime != previous_regime:
                recent_transitions.append(
                    {
                        "type": "rates_regime",
                        "from": previous_regime,
                        "to": regime,
                        "observed_at": item["observed_at"],
                    }
                )
            previous_gate = gate
            previous_regime = regime
        symbols_payload.append(
            {
                "symbol": pair_id,
                "observations": len(timeline),
                "latest": timeline[-1],
                "gate_counts": dict(gate_counts),
                "regime_counts": dict(regime_counts),
                "top_reasons": [
                    {"reason": reason, "count": count}
                    for reason, count in top_reasons.most_common(8)
                ],
                "recent_transitions": recent_transitions[-12:],
                "timeline": timeline[-48:],
            }
        )

    currencies_payload = []
    for currency, timeline in sorted(currency_timelines.items()):
        if not timeline:
            continue
        latest = timeline[-1]
        currencies_payload.append(
            {
                "currency": currency,
                "observations": len(timeline),
                "latest": latest,
                "mean_policy_repricing_score": round(
                    sum(item["policy_repricing_score"] for item in timeline) / max(len(timeline), 1),
                    6,
                ),
                "mean_policy_uncertainty_score": round(
                    sum(item["policy_uncertainty_score"] for item in timeline) / max(len(timeline), 1),
                    6,
                ),
            }
        )

    payload = {
        "generated_at": isoformat_utc(),
        "hours_back": int(hours_back),
        "symbols": symbols_payload,
        "currencies": currencies_payload,
        "history_path": str(COMMON_RATES_HISTORY),
        "report_path": str(RATES_ENGINE_REPLAY_REPORT_PATH),
    }
    json_dump(RATES_ENGINE_REPLAY_REPORT_PATH, payload)
    return payload
