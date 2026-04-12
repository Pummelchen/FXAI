from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import libsql

from .common import OfflineLabError, query_all
from .pair_network_config import export_runtime_config, export_runtime_status, load_config, validate_config_payload
from .pair_network_contracts import (
    PAIR_NETWORK_ACTIONS,
    PAIR_NETWORK_CONFIG_PATH,
    PAIR_NETWORK_HISTORY_PATH,
    PAIR_NETWORK_REPORT_PATH,
    PAIR_NETWORK_RUNTIME_CONFIG_PATH,
    PAIR_NETWORK_RUNTIME_STATUS_PATH,
    PAIR_NETWORK_SCHEMA_VERSION,
    PAIR_NETWORK_STATUS_PATH,
    ensure_pair_network_dirs,
    isoformat_utc,
    json_dump,
    json_load,
    ndjson_append,
    utc_now,
)
from .pair_network_math import clamp, clamp01, cosine_similarity, herfindahl_concentration, pearson_correlation, safe_mean, top_share


FACTOR_KEYS = [
    "usd_bloc",
    "eur_rates",
    "safe_haven",
    "commodity_fx",
    "risk_on",
    "liquidity_stress",
    "macro_shock",
]


def pair_legs(symbol: str) -> tuple[str, str]:
    clean = "".join(ch for ch in str(symbol or "").upper() if "A" <= ch <= "Z")
    if len(clean) < 6:
        return "", ""
    return clean[:3], clean[3:6]


def normalize_action(action: str | int) -> str:
    if isinstance(action, int):
        return "BUY" if action == 1 else ("SELL" if action == 0 else "SKIP")
    text = str(action or "").strip().upper()
    if text in {"BUY", "LONG", "1"}:
        return "BUY"
    if text in {"SELL", "SHORT", "0"}:
        return "SELL"
    return "SKIP"


def pair_currency_exposure(symbol: str, action: str | int, size_units: float = 1.0) -> dict[str, float]:
    base, quote = pair_legs(symbol)
    direction = normalize_action(action)
    if not base or not quote or direction == "SKIP":
        return {}
    signed = abs(float(size_units or 0.0)) * (1.0 if direction == "BUY" else -1.0)
    return {base: signed, quote: -signed}


def factor_exposure_from_currency_exposure(
    currency_exposure: dict[str, float],
    currency_profiles: dict[str, dict[str, float]],
) -> dict[str, float]:
    exposures: dict[str, float] = {factor: 0.0 for factor in FACTOR_KEYS}
    for currency, value in currency_exposure.items():
        profile = dict(currency_profiles.get(currency, {}))
        for factor in FACTOR_KEYS:
            exposures[factor] += float(value) * float(profile.get(factor, 0.0) or 0.0)
    return exposures


def pair_factor_exposure(
    symbol: str,
    action: str | int,
    currency_profiles: dict[str, dict[str, float]],
    size_units: float = 1.0,
) -> dict[str, float]:
    return factor_exposure_from_currency_exposure(
        pair_currency_exposure(symbol, action, size_units=size_units),
        currency_profiles,
    )


def pair_factor_signature(symbol: str, currency_profiles: dict[str, dict[str, float]]) -> dict[str, float]:
    exposure = pair_factor_exposure(symbol, "BUY", currency_profiles, size_units=1.0)
    return {key: abs(float(value)) for key, value in exposure.items()}


def aggregate_currency_exposure(trades: list[dict[str, Any]]) -> dict[str, float]:
    exposure: dict[str, float] = {}
    for trade in trades:
        for currency, value in pair_currency_exposure(
            str(trade.get("symbol", "") or ""),
            trade.get("action", "SKIP"),
            float(trade.get("size_units", 1.0) or 0.0),
        ).items():
            exposure[currency] = exposure.get(currency, 0.0) + value
    return {key: round(value, 8) for key, value in exposure.items() if abs(value) > 1e-12}


def aggregate_factor_exposure(
    trades: list[dict[str, Any]],
    currency_profiles: dict[str, dict[str, float]],
) -> dict[str, float]:
    return {
        key: round(value, 8)
        for key, value in factor_exposure_from_currency_exposure(
            aggregate_currency_exposure(trades),
            currency_profiles,
        ).items()
        if abs(value) > 1e-12
    }


def structural_dependency(
    lhs: str,
    rhs: str,
    currency_profiles: dict[str, dict[str, float]],
) -> dict[str, Any]:
    base_l, quote_l = pair_legs(lhs)
    base_r, quote_r = pair_legs(rhs)
    if not base_l or not quote_l or not base_r or not quote_r:
        return {
            "pair": rhs,
            "combined_score": 0.0,
            "structural_score": 0.0,
            "shared_currencies": [],
            "relation": "UNKNOWN",
        }

    lhs_key = f"{base_l}{quote_l}"
    rhs_key = f"{base_r}{quote_r}"
    shared = sorted({base_l, quote_l} & {base_r, quote_r})
    if lhs_key == rhs_key:
        currency_score = 1.0
        relation = "IDENTICAL"
    elif base_l == quote_r and quote_l == base_r:
        currency_score = 0.96
        relation = "INVERSE"
    elif base_l == base_r or quote_l == quote_r:
        currency_score = 0.84
        relation = "PARALLEL_LEG"
    elif base_l == quote_r or quote_l == base_r:
        currency_score = 0.72
        relation = "CROSS_LEG"
    elif shared:
        currency_score = 0.56
        relation = "SHARED_CURRENCY"
    else:
        currency_score = 0.0
        relation = "DISJOINT"

    lhs_signature = pair_factor_signature(lhs_key, currency_profiles)
    rhs_signature = pair_factor_signature(rhs_key, currency_profiles)
    factor_score = clamp01((cosine_similarity(lhs_signature, rhs_signature) + 1.0) * 0.5)
    same_cluster = 0.0
    if any(code in shared for code in ("AUD", "CAD", "NZD", "NOK")):
        same_cluster += 0.12
    if any(code in shared for code in ("JPY", "CHF")):
        same_cluster += 0.10
    if "USD" in shared:
        same_cluster += 0.08
    structural_score = clamp01(0.64 * currency_score + 0.28 * factor_score + same_cluster)
    return {
        "pair": rhs_key,
        "combined_score": round(structural_score, 6),
        "structural_score": round(structural_score, 6),
        "shared_currencies": shared,
        "relation": relation,
    }


def _latest_dataset_ids(conn: libsql.Connection, tradable_pairs: list[str]) -> dict[str, int]:
    rows = query_all(
        conn,
        """
        SELECT id, symbol
          FROM datasets
         ORDER BY created_at DESC, id DESC
        """,
    )
    selected: dict[str, int] = {}
    pair_set = set(tradable_pairs)
    for row in rows:
        symbol = str(row["symbol"] or "").upper()
        if symbol not in pair_set or symbol in selected:
            continue
        selected[symbol] = int(row["id"])
    return selected


def _dataset_returns(conn: libsql.Connection, dataset_id: int, lookback_bars: int) -> list[float]:
    rows = query_all(
        conn,
        """
        SELECT close
          FROM dataset_bars
         WHERE dataset_id = ?
         ORDER BY bar_time_unix DESC
         LIMIT ?
        """,
        (int(dataset_id), int(lookback_bars) + 1),
    )
    closes = [float(row["close"]) for row in reversed(rows) if float(row["close"]) > 0.0]
    returns: list[float] = []
    for left, right in zip(closes, closes[1:]):
        if left <= 0.0:
            continue
        returns.append((right / left) - 1.0)
    return returns


def empirical_dependency_matrix(
    conn: libsql.Connection | None,
    tradable_pairs: list[str],
    *,
    lookback_bars: int,
    min_overlap: int,
) -> tuple[dict[tuple[str, str], dict[str, Any]], bool]:
    if conn is None:
        return {}, True
    try:
        dataset_ids = _latest_dataset_ids(conn, tradable_pairs)
    except Exception:
        return {}, True
    if len(dataset_ids) <= 1:
        return {}, True
    series = {
        symbol: _dataset_returns(conn, dataset_id, lookback_bars)
        for symbol, dataset_id in dataset_ids.items()
    }
    edges: dict[tuple[str, str], dict[str, Any]] = {}
    pair_list = sorted(series.keys())
    any_empirical = False
    for index, lhs in enumerate(pair_list):
        for rhs in pair_list[index + 1 :]:
            corr, support = pearson_correlation(series.get(lhs, []), series.get(rhs, []))
            if support < min_overlap:
                continue
            any_empirical = True
            edges[(lhs, rhs)] = {
                "correlation": round(corr, 6),
                "abs_correlation": round(abs(corr), 6),
                "support": int(support),
            }
    return edges, not any_empirical


def _symbol_quality_score(
    trade: dict[str, Any],
    *,
    weights: dict[str, float],
    portfolio_overlap: float,
) -> float:
    edge = clamp01(float(trade.get("edge_after_costs", 0.0) or 0.0))
    execution = clamp01(float(trade.get("execution_quality_score", 0.0) or 0.0))
    calibration = clamp01(float(trade.get("calibration_quality", 0.0) or 0.0))
    portfolio_fit = clamp01(float(trade.get("portfolio_fit", 0.0) or 0.0))
    diversification = clamp01(1.0 - portfolio_overlap)
    macro_fit = clamp01(float(trade.get("macro_fit", 0.5) or 0.5))
    return clamp01(
        float(weights.get("edge_after_costs", 0.0) or 0.0) * edge
        + float(weights.get("execution_quality", 0.0) or 0.0) * execution
        + float(weights.get("calibration_quality", 0.0) or 0.0) * calibration
        + float(weights.get("portfolio_fit", 0.0) or 0.0) * portfolio_fit
        + float(weights.get("diversification", 0.0) or 0.0) * diversification
        + float(weights.get("macro_fit", 0.0) or 0.0) * macro_fit
    )


def resolve_candidate(
    *,
    config: dict[str, Any],
    candidate_trade: dict[str, Any],
    open_positions: list[dict[str, Any]] | None = None,
    pending_trades: list[dict[str, Any]] | None = None,
    peer_candidates: list[dict[str, Any]] | None = None,
    news_risk_score: float = 0.0,
    execution_stress_score: float = 0.0,
) -> dict[str, Any]:
    normalized = validate_config_payload(config)
    open_positions = list(open_positions or [])
    pending_trades = list(pending_trades or [])
    peer_candidates = list(peer_candidates or [])

    currency_profiles = dict(normalized.get("currency_profiles", {}))
    portfolio_trades = open_positions + pending_trades
    portfolio_currency = aggregate_currency_exposure(portfolio_trades)
    portfolio_factor = factor_exposure_from_currency_exposure(portfolio_currency, currency_profiles)

    candidate = {
        "symbol": str(candidate_trade.get("symbol", "") or "").upper(),
        "action": normalize_action(candidate_trade.get("action", "SKIP")),
        "size_units": abs(float(candidate_trade.get("size_units", 1.0) or 0.0)),
        "edge_after_costs": float(candidate_trade.get("edge_after_costs", 0.0) or 0.0),
        "execution_quality_score": float(candidate_trade.get("execution_quality_score", 0.0) or 0.0),
        "calibration_quality": float(candidate_trade.get("calibration_quality", 0.0) or 0.0),
        "portfolio_fit": float(candidate_trade.get("portfolio_fit", 0.0) or 0.0),
        "macro_fit": float(candidate_trade.get("macro_fit", 0.5) or 0.5),
    }
    candidate_currency = pair_currency_exposure(candidate["symbol"], candidate["action"], candidate["size_units"])
    candidate_factor = factor_exposure_from_currency_exposure(candidate_currency, currency_profiles)
    after_currency = aggregate_currency_exposure(portfolio_trades + [candidate])
    after_factor = factor_exposure_from_currency_exposure(after_currency, currency_profiles)

    currency_alignment = cosine_similarity(candidate_currency, portfolio_currency)
    factor_alignment = cosine_similarity(candidate_factor, portfolio_factor)
    overlap_score = clamp01(0.58 * max(currency_alignment, 0.0) + 0.42 * max(factor_alignment, 0.0))
    contradiction_core = clamp01(0.64 * max(-currency_alignment, 0.0) + 0.36 * max(-factor_alignment, 0.0))

    direct_contradiction = any(
        str(item.get("symbol", "") or "").upper() == candidate["symbol"]
        and normalize_action(item.get("action", "SKIP")) != candidate["action"]
        for item in portfolio_trades + peer_candidates
    )
    contradiction_score = clamp01(1.0 if direct_contradiction else contradiction_core)

    before_currency_conc = top_share(portfolio_currency)
    before_factor_conc = top_share(portfolio_factor)
    after_currency_conc = max(top_share(after_currency), herfindahl_concentration(after_currency))
    after_factor_conc = max(top_share(after_factor), herfindahl_concentration(after_factor))
    concentration_score = clamp01(
        0.54 * max(after_currency_conc, after_factor_conc)
        + 0.24 * max(after_currency_conc - before_currency_conc, 0.0)
        + 0.22 * max(after_factor_conc - before_factor_conc, 0.0)
    )

    candidate_overlap_with_portfolio = clamp01(max(overlap_score, concentration_score))
    candidate_quality = _symbol_quality_score(
        candidate,
        weights=dict(normalized.get("selection_weights", {})),
        portfolio_overlap=candidate_overlap_with_portfolio,
    )

    preferred_expression = ""
    preferred_quality = candidate_quality
    for peer in peer_candidates:
        peer_symbol = str(peer.get("symbol", "") or "").upper()
        peer_action = normalize_action(peer.get("action", "SKIP"))
        peer_currency = pair_currency_exposure(peer_symbol, peer_action, float(peer.get("size_units", 1.0) or 0.0))
        peer_factor = factor_exposure_from_currency_exposure(peer_currency, currency_profiles)
        same_view = clamp01(
            0.62 * max(cosine_similarity(candidate_currency, peer_currency), 0.0)
            + 0.38 * max(cosine_similarity(candidate_factor, peer_factor), 0.0)
        )
        same_view = max(
            same_view,
            float(structural_dependency(candidate["symbol"], peer_symbol, currency_profiles).get("structural_score", 0.0) or 0.0),
        )
        if same_view < 0.60:
            continue
        peer_quality = _symbol_quality_score(
            {
                "edge_after_costs": float(peer.get("edge_after_costs", 0.0) or 0.0),
                "execution_quality_score": float(peer.get("execution_quality_score", 0.0) or 0.0),
                "calibration_quality": float(peer.get("calibration_quality", 0.0) or 0.0),
                "portfolio_fit": float(peer.get("portfolio_fit", 0.0) or 0.0),
                "macro_fit": float(peer.get("macro_fit", 0.5) or 0.5),
            },
            weights=dict(normalized.get("selection_weights", {})),
            portfolio_overlap=clamp01(same_view),
        )
        if peer_quality > preferred_quality + float(normalized.get("preferred_expression_margin", 0.04) or 0.04):
            preferred_quality = peer_quality
            preferred_expression = peer_symbol

    redundancy_score = clamp01(
        0.56 * overlap_score
        + 0.18 * max(0.0, news_risk_score) * overlap_score
        + 0.16 * max(0.0, execution_stress_score) * overlap_score
        + 0.10 * (1.0 if preferred_expression else 0.0)
    )
    execution_overlap_score = clamp01(execution_stress_score * overlap_score)
    conflict_score = clamp01(max(contradiction_score, redundancy_score, concentration_score))

    decision = "ALLOW"
    size_multiplier = 1.0
    reason_codes: list[str] = []
    if direct_contradiction:
        reason_codes.append("DIRECT_SYMBOL_CONTRADICTION")
    if contradiction_score >= float(normalized.get("contradiction_threshold", 0.74) or 0.74):
        decision = "BLOCK_CONTRADICTORY"
        size_multiplier = 0.0
        reason_codes.append("CURRENCY_EXPOSURE_CONFLICT")
    elif preferred_expression and max(overlap_score, redundancy_score) >= 0.55:
        decision = "PREFER_ALTERNATIVE_EXPRESSION"
        size_multiplier = 0.0
        reason_codes.append("BETTER_ALTERNATIVE_EXPRESSION")
    elif (
        concentration_score >= float(normalized.get("concentration_block_threshold", 0.80) or 0.80)
        and candidate_quality < 0.72
    ):
        decision = "BLOCK_CONCENTRATION"
        size_multiplier = 0.0
        reason_codes.append("HIDDEN_CURRENCY_CONCENTRATION")
    elif (
        redundancy_score >= float(normalized.get("redundancy_threshold", 0.68) or 0.68)
        and candidate_quality < float(normalized.get("min_incremental_edge_score", 0.12) or 0.12)
    ):
        decision = "SUPPRESS_REDUNDANT"
        size_multiplier = 0.0
        reason_codes.append("LOW_INCREMENTAL_PORTFOLIO_EDGE")
    elif (
        concentration_score >= float(normalized.get("concentration_reduce_threshold", 0.58) or 0.58)
        or redundancy_score >= float(normalized.get("redundancy_threshold", 0.68) or 0.68)
        or execution_overlap_score >= float(normalized.get("execution_overlap_threshold", 0.62) or 0.62)
    ):
        decision = "ALLOW_REDUCED"
        size_multiplier = clamp(
            1.0 - 0.55 * max(redundancy_score, concentration_score, execution_overlap_score),
            float(normalized.get("reduced_size_multiplier_floor", 0.45) or 0.45),
            0.95,
        )

    dominant_currency = ""
    if candidate_currency:
        dominant_currency = max(candidate_currency.items(), key=lambda item: abs(item[1]))[0]
    if redundancy_score >= float(normalized.get("redundancy_threshold", 0.68) or 0.68):
        direction_label = "LONG" if float(candidate_currency.get(dominant_currency, 0.0)) >= 0.0 else "SHORT"
        if dominant_currency:
            reason_codes.append(f"DUPLICATES_EXISTING_{dominant_currency}_{direction_label}_EXPOSURE")
    if abs(float(candidate_factor.get("commodity_fx", 0.0) or 0.0)) >= 0.35 and redundancy_score >= 0.52:
        reason_codes.append("HIGH_COMMODITY_BLOC_OVERLAP")
    if concentration_score >= float(normalized.get("concentration_reduce_threshold", 0.58) or 0.58):
        reason_codes.append("FACTOR_CONCENTRATION_ELEVATED")
    if execution_overlap_score >= float(normalized.get("execution_overlap_threshold", 0.62) or 0.62):
        reason_codes.append("EXECUTION_STRESS_OVERLAP")
    if max(news_risk_score, 0.0) >= 0.55 and max(redundancy_score, concentration_score) >= 0.45:
        reason_codes.append("EVENT_STACKING_RISK")
    if preferred_expression:
        reason_codes.append(f"PREFERRED_EXPRESSION_{preferred_expression}")

    reason_codes = list(dict.fromkeys(reason_codes))
    if decision not in PAIR_NETWORK_ACTIONS:
        raise OfflineLabError(f"Unsupported pair-network decision computed: {decision}")

    return {
        "timestamp": isoformat_utc(),
        "portfolio_state": {
            "currency_net_exposure": {key: round(value, 6) for key, value in after_currency.items()},
            "factor_exposure": {key: round(value, 6) for key, value in after_factor.items()},
            "concentration_scores": {
                "currency_concentration": round(after_currency_conc, 6),
                "factor_concentration": round(after_factor_conc, 6),
                "portfolio_redundancy": round(redundancy_score, 6),
            },
        },
        "candidate_trade": {
            "symbol": candidate["symbol"],
            "action": candidate["action"],
            "size_units": candidate["size_units"],
        },
        "resolution": {
            "decision": decision,
            "conflict_score": round(conflict_score, 6),
            "redundancy_score": round(redundancy_score, 6),
            "contradiction_score": round(contradiction_score, 6),
            "concentration_score": round(concentration_score, 6),
            "preferred_expression": preferred_expression,
            "recommended_size_multiplier": round(size_multiplier, 6),
        },
        "reason_codes": reason_codes,
        "quality_flags": {
            "fallback_graph_used": False,
            "partial_dependency_data": False,
            "graph_stale": False,
        },
    }


def _pair_summary(
    pair: str,
    *,
    config: dict[str, Any],
    empirical_edges: dict[tuple[str, str], dict[str, Any]],
    tradable_pairs: list[str],
) -> dict[str, Any]:
    base, quote = pair_legs(pair)
    profiles = dict(config.get("currency_profiles", {}))
    factor_signature = pair_factor_signature(pair, profiles)
    dependencies: list[dict[str, Any]] = []
    for other in tradable_pairs:
        if other == pair:
            continue
        edge = structural_dependency(pair, other, profiles)
        empirical = empirical_edges.get((pair, other)) or empirical_edges.get((other, pair)) or {}
        structural_weight = float(config.get("structural_weight", 0.72) or 0.72)
        empirical_weight = float(config.get("empirical_weight", 0.28) or 0.28)
        combined = clamp01(
            structural_weight * float(edge.get("structural_score", 0.0) or 0.0)
            + empirical_weight * float(empirical.get("abs_correlation", 0.0) or 0.0)
        )
        dependencies.append(
            {
                "pair": other,
                "structural_score": round(float(edge.get("structural_score", 0.0) or 0.0), 6),
                "empirical_score": round(float(empirical.get("abs_correlation", 0.0) or 0.0), 6),
                "combined_score": round(combined, 6),
                "correlation": round(float(empirical.get("correlation", 0.0) or 0.0), 6),
                "support": int(empirical.get("support", 0) or 0),
                "shared_currencies": list(edge.get("shared_currencies", [])),
                "relation": str(edge.get("relation", "UNKNOWN") or "UNKNOWN"),
            }
        )
    dependencies.sort(key=lambda item: (-float(item["combined_score"]), item["pair"]))
    return {
        "pair": pair,
        "base_currency": base,
        "quote_currency": quote,
        "factor_signature": {key: round(float(value), 6) for key, value in factor_signature.items()},
        "top_dependencies": dependencies[: int(config.get("max_edges_per_pair", 10) or 10)],
    }


def build_pair_network_graph(conn: libsql.Connection | None, *, config: dict[str, Any]) -> dict[str, Any]:
    normalized = validate_config_payload(config)
    tradable_pairs = list(dict(normalized.get("market_universe", {})).get("tradable_pairs", []))
    empirical_edges, structural_only_mode = empirical_dependency_matrix(
        conn,
        tradable_pairs,
        lookback_bars=int(normalized.get("empirical_lookback_bars", 512) or 512),
        min_overlap=int(normalized.get("min_empirical_overlap", 128) or 128),
    )
    summaries = [
        _pair_summary(
            pair,
            config=normalized,
            empirical_edges=empirical_edges,
            tradable_pairs=tradable_pairs,
        )
        for pair in tradable_pairs
    ]
    edge_count = sum(len(item["top_dependencies"]) for item in summaries)
    top_edges = sorted(
        (
            {
                "source_pair": item["pair"],
                "target_pair": dependency.get("pair", ""),
                **dependency,
            }
            for item in summaries
            for dependency in item["top_dependencies"]
        ),
        key=lambda item: (-float(item["combined_score"]), str(item.get("source_pair", "")), str(item.get("target_pair", ""))),
    )[:20]
    reason_codes = []
    if structural_only_mode:
        reason_codes.append("STRUCTURAL_ONLY_MODE")
        reason_codes.append("EMPIRICAL_DEPENDENCY_SUPPORT_LOW")
    return {
        "schema_version": PAIR_NETWORK_SCHEMA_VERSION,
        "generated_at": isoformat_utc(),
        "graph_mode": "STRUCTURAL_ONLY" if structural_only_mode else "STRUCTURAL_PLUS_EMPIRICAL",
        "pair_count": len(tradable_pairs),
        "currency_count": len(list(dict(normalized.get("market_universe", {})).get("currencies", []))),
        "edge_count": edge_count,
        "pairs": summaries,
        "top_edges": top_edges,
        "reason_codes": reason_codes,
        "quality_flags": {
            "fallback_graph_used": structural_only_mode,
            "partial_dependency_data": structural_only_mode,
            "graph_stale": False,
        },
    }


def _status_payload(*, report: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    return {
        "ok": True,
        "generated_at": report.get("generated_at", ""),
        "graph_mode": report.get("graph_mode", "STRUCTURAL_ONLY"),
        "pair_count": int(report.get("pair_count", 0) or 0),
        "currency_count": int(report.get("currency_count", 0) or 0),
        "edge_count": int(report.get("edge_count", 0) or 0),
        "fallback_graph_used": bool(dict(report.get("quality_flags", {})).get("fallback_graph_used", False)),
        "partial_dependency_data": bool(dict(report.get("quality_flags", {})).get("partial_dependency_data", False)),
        "graph_stale": bool(dict(report.get("quality_flags", {})).get("graph_stale", False)),
        "action_mode": str(config.get("action_mode", "AUTO_APPLY")),
        "config_path": str(Path(PAIR_NETWORK_CONFIG_PATH).resolve()),
        "runtime_config_path": str(PAIR_NETWORK_RUNTIME_CONFIG_PATH),
        "runtime_status_path": str(PAIR_NETWORK_RUNTIME_STATUS_PATH),
        "report_path": str(PAIR_NETWORK_REPORT_PATH),
    }


def _default_unbuilt_status(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "ok": True,
        "generated_at": "",
        "graph_mode": "UNBUILT",
        "pair_count": len(list(dict(config.get("market_universe", {})).get("tradable_pairs", []))),
        "currency_count": len(list(dict(config.get("market_universe", {})).get("currencies", []))),
        "edge_count": 0,
        "fallback_graph_used": False,
        "partial_dependency_data": False,
        "graph_stale": True,
        "action_mode": config.get("action_mode", "AUTO_APPLY"),
    }


def _load_runtime_status_tsv(path: Path) -> dict[str, Any]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except Exception:
        return {}
    payload: dict[str, Any] = {}
    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue
        key, sep, value = line.partition("\t")
        if not sep or not key:
            continue
        payload[key] = value
    return payload


def validate_pair_network_config() -> dict[str, Any]:
    ensure_pair_network_dirs()
    config = load_config()
    export_runtime_config(config)
    runtime_status = _load_runtime_status_tsv(PAIR_NETWORK_RUNTIME_STATUS_PATH)
    local_status = json_load(PAIR_NETWORK_STATUS_PATH)
    status = runtime_status if runtime_status else local_status
    if not status:
        status = _default_unbuilt_status(config)
        export_runtime_status(status)
        json_dump(PAIR_NETWORK_STATUS_PATH, status)
    else:
        status = dict(status)
        status.setdefault("ok", True)
        status.setdefault("action_mode", config.get("action_mode", "AUTO_APPLY"))
        status.setdefault("config_path", str(Path(PAIR_NETWORK_CONFIG_PATH).resolve()))
        status.setdefault("runtime_config_path", str(PAIR_NETWORK_RUNTIME_CONFIG_PATH))
        status.setdefault("runtime_status_path", str(PAIR_NETWORK_RUNTIME_STATUS_PATH))
        status.setdefault("report_path", str(PAIR_NETWORK_REPORT_PATH))
        if not runtime_status:
            export_runtime_status(status)
        if not local_status:
            json_dump(PAIR_NETWORK_STATUS_PATH, status)
    return {
        "ok": True,
        "config_path": str(PAIR_NETWORK_CONFIG_PATH),
        "runtime_config_path": str(PAIR_NETWORK_RUNTIME_CONFIG_PATH),
        "runtime_status_path": str(PAIR_NETWORK_RUNTIME_STATUS_PATH),
        "config": config,
        "status": status,
    }


def build_pair_network_artifacts(
    conn: libsql.Connection | None,
    *,
    profile_name: str = "continuous",
    append_history: bool = True,
) -> dict[str, Any]:
    ensure_pair_network_dirs()
    config = load_config()
    report = build_pair_network_graph(conn, config=config)
    json_dump(PAIR_NETWORK_REPORT_PATH, report)
    status = _status_payload(report=report, config=config)
    json_dump(PAIR_NETWORK_STATUS_PATH, status)
    export_runtime_config(config)
    export_runtime_status(status)
    if append_history:
        ndjson_append(
            PAIR_NETWORK_HISTORY_PATH,
            {
                "record_type": "graph_snapshot",
                "profile_name": profile_name or "continuous",
                "generated_at": report.get("generated_at", ""),
                "graph_mode": report.get("graph_mode", "STRUCTURAL_ONLY"),
                "pair_count": report.get("pair_count", 0),
                "edge_count": report.get("edge_count", 0),
                "reason_codes": list(report.get("reason_codes", [])),
                "quality_flags": dict(report.get("quality_flags", {})),
            },
        )
    return {
        "ok": True,
        "profile_name": profile_name or "continuous",
        "graph_mode": report.get("graph_mode", "STRUCTURAL_ONLY"),
        "pair_count": report.get("pair_count", 0),
        "edge_count": report.get("edge_count", 0),
        "fallback_graph_used": bool(dict(report.get("quality_flags", {})).get("fallback_graph_used", False)),
        "config_path": str(PAIR_NETWORK_CONFIG_PATH),
        "runtime_config_path": str(PAIR_NETWORK_RUNTIME_CONFIG_PATH),
        "status_path": str(PAIR_NETWORK_STATUS_PATH),
        "report_path": str(PAIR_NETWORK_REPORT_PATH),
    }


def build_pair_network_report(
    conn: libsql.Connection | None,
    *,
    profile_name: str = "continuous",
) -> dict[str, Any]:
    payload = build_pair_network_artifacts(conn, profile_name=profile_name, append_history=False)
    return json_load(PAIR_NETWORK_REPORT_PATH) or payload
