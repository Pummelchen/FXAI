from __future__ import annotations

from copy import deepcopy
from typing import Any

from .rates_engine_contracts import COMMON_RATES_JSON, json_load


def load_rates_snapshot() -> dict[str, Any]:
    return json_load(COMMON_RATES_JSON)


def apply_rates_enrichment(snapshot: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(snapshot, dict):
        return snapshot
    rates_snapshot = load_rates_snapshot()
    if not isinstance(rates_snapshot, dict) or not rates_snapshot:
        return snapshot

    enriched = deepcopy(snapshot)
    rates_currencies = rates_snapshot.get("currencies", {})
    rates_pairs = rates_snapshot.get("pairs", {})
    source_status = rates_snapshot.get("source_status", {})
    enriched["rates_enrichment"] = {
        "available": bool(rates_pairs or rates_currencies),
        "generated_at": str(rates_snapshot.get("generated_at", "")),
        "source_status": source_status,
    }

    currencies = enriched.get("currencies", {})
    if isinstance(currencies, dict):
        for currency, state in list(currencies.items()):
            rates_state = dict(rates_currencies.get(currency, {}))
            if not rates_state or not isinstance(state, dict):
                continue
            state["rates_context"] = {
                "policy_relevance_score": rates_state.get("policy_relevance_score"),
                "policy_repricing_score": rates_state.get("policy_repricing_score"),
                "policy_surprise_score": rates_state.get("policy_surprise_score"),
                "policy_uncertainty_score": rates_state.get("policy_uncertainty_score"),
                "curve_shape_regime": rates_state.get("curve_shape_regime"),
                "front_end_basis": rates_state.get("front_end_basis"),
                "expected_path_basis": rates_state.get("expected_path_basis"),
                "stale": rates_state.get("stale"),
                "reasons": list(rates_state.get("reasons", []))[:3],
            }

    pairs = enriched.get("pairs", {})
    if isinstance(pairs, dict):
        for pair_id, state in list(pairs.items()):
            rates_state = dict(rates_pairs.get(pair_id, {}))
            if not rates_state or not isinstance(state, dict):
                continue
            state["rates_context"] = {
                "rates_risk_score": rates_state.get("rates_risk_score"),
                "trade_gate": rates_state.get("trade_gate"),
                "rates_regime": rates_state.get("rates_regime"),
                "policy_divergence_score": rates_state.get("policy_divergence_score"),
                "curve_divergence_score": rates_state.get("curve_divergence_score"),
                "meeting_path_reprice_now": rates_state.get("meeting_path_reprice_now"),
                "macro_to_rates_transmission_score": rates_state.get("macro_to_rates_transmission_score"),
                "policy_alignment": rates_state.get("policy_alignment"),
                "stale": rates_state.get("stale"),
                "reasons": list(rates_state.get("reasons", []))[:3],
            }

    recent_items = enriched.get("recent_items", [])
    if isinstance(recent_items, list):
        for item in recent_items:
            if not isinstance(item, dict):
                continue
            currencies = [str(code).upper() for code in list(item.get("currency_tags", []))]
            contexts = [dict(rates_currencies.get(code, {})) for code in currencies if isinstance(rates_currencies.get(code), dict)]
            if not contexts:
                continue
            policy_relevance = max(float(context.get("policy_relevance_score", 0.0) or 0.0) for context in contexts)
            transmission = max(float(context.get("macro_to_rates_transmission_score", 0.0) or 0.0) for context in contexts)
            surprise = max(float(context.get("policy_surprise_score", 0.0) or 0.0) for context in contexts)
            path_reprice = any(bool(context.get("meeting_path_reprice_now", False)) for context in contexts)
            item["policy_relevance_score"] = round(policy_relevance, 6)
            item["rates_confirmation_score"] = round(max(transmission, surprise * 0.85), 6)
            item["path_repricing_after_event"] = path_reprice
            item["cb_surprise_context"] = round(surprise, 6)
            item["macro_release_rates_impact"] = round(transmission, 6)
    return enriched
