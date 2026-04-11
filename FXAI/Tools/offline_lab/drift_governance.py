from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import libsql

from .adaptive_router_contracts import ADAPTIVE_ROUTER_REPLAY_REPORT_PATH
from .common import *
from .cross_asset_contracts import CROSS_ASSET_REPLAY_REPORT_PATH
from .drift_governance_config import load_config, validate_config_payload
from .drift_governance_contracts import (
    DRIFT_GOVERNANCE_CONFIG_PATH,
    DRIFT_GOVERNANCE_HISTORY_PATH,
    DRIFT_GOVERNANCE_REPORT_PATH,
    DRIFT_GOVERNANCE_REPORT_VERSION,
    DRIFT_GOVERNANCE_RUNTIME_SUMMARY_PATH,
    DRIFT_GOVERNANCE_SCHEMA_VERSION,
    DRIFT_GOVERNANCE_STATUS_PATH,
    ensure_drift_governance_dirs,
    isoformat_utc,
    json_dump,
    json_load,
)
from .drift_governance_math import (
    aggregate_drift_score,
    clamp,
    downside_shift_score,
    mean,
    normalized_gap,
    population_stability_index,
    regime_mix_shift_score,
    severity_rank,
    upside_shift_score,
)
from .dynamic_ensemble_contracts import DYNAMIC_ENSEMBLE_REPLAY_REPORT_PATH
from .execution_quality_contracts import EXECUTION_QUALITY_REPLAY_REPORT_PATH
from .microstructure_contracts import MICROSTRUCTURE_REPLAY_REPORT_PATH
from .newspulse_contracts import NEWSPULSE_REPLAY_REPORT_PATH, NEWSPULSE_STATUS_PATH
from .prob_calibration_contracts import PROB_CALIBRATION_REPLAY_REPORT_PATH


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except Exception:
        return int(default)


def _safe_json(raw: str, default: Any) -> Any:
    try:
        return json.loads(raw or "")
    except Exception:
        return default


def _append_history(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def _load_shadow_rows(conn: libsql.Connection,
                      profile_name: str) -> tuple[dict[tuple[str, str], list[dict[str, Any]]], dict[str, list[dict[str, Any]]]]:
    rows = query_all(
        conn,
        """
        SELECT *
          FROM shadow_fleet_observations
         WHERE profile_name = ?
         ORDER BY captured_at DESC, id DESC
        """,
        (profile_name,),
    )
    by_scope: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    by_plugin: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        payload = dict(row)
        symbol = str(payload.get("symbol", "") or "")
        plugin_name = str(payload.get("plugin_name", "") or "")
        by_scope[(symbol, plugin_name)].append(payload)
        by_plugin[plugin_name].append(payload)
    return by_scope, by_plugin


def _load_registry_rows(conn: libsql.Connection,
                        profile_name: str) -> dict[tuple[str, str], dict[str, Any]]:
    return {
        (str(row["symbol"]), str(row["plugin_name"])): dict(row)
        for row in query_all(
            conn,
            """
            SELECT *
              FROM champion_registry
             WHERE profile_name = ?
             ORDER BY symbol, plugin_name
            """,
            (profile_name,),
        )
    }


def _load_previous_states(conn: libsql.Connection,
                          profile_name: str) -> dict[tuple[str, str], dict[str, Any]]:
    return {
        (str(row["symbol"]), str(row["plugin_name"])): dict(row)
        for row in query_all(
            conn,
            """
            SELECT *
              FROM plugin_governance_states
             WHERE profile_name = ?
             ORDER BY symbol, plugin_name
            """,
            (profile_name,),
        )
    }


def _load_symbol_report_map(path: Path) -> dict[str, dict[str, Any]]:
    payload = json_load(path)
    rows = payload.get("symbols", [])
    if not isinstance(rows, list):
        return {}
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        symbol = str(row.get("symbol", "") or "").strip().upper()
        if not symbol:
            continue
        out[symbol] = row
    return out


def _load_newspulse_context() -> dict[str, dict[str, Any]]:
    report = json_load(NEWSPULSE_REPLAY_REPORT_PATH)
    status = json_load(NEWSPULSE_STATUS_PATH)
    out: dict[str, dict[str, Any]] = {}
    top_pairs = report.get("top_pairs", [])
    if isinstance(top_pairs, list):
        for row in top_pairs:
            if not isinstance(row, dict):
                continue
            pair = str(row.get("pair", "") or row.get("pair_id", "")).strip().upper()
            if pair:
                out[pair] = row
    pair_timelines = report.get("pair_timelines", {})
    if isinstance(pair_timelines, dict):
        for pair, entries in pair_timelines.items():
            pair_name = str(pair).strip().upper()
            if not pair_name:
                continue
            latest = entries[-1] if isinstance(entries, list) and entries else {}
            current = dict(out.get(pair_name, {}))
            if isinstance(latest, dict):
                current.update(latest)
            out[pair_name] = current
    if isinstance(status.get("pairs"), dict):
        for pair, state in dict(status.get("pairs", {})).items():
            pair_name = str(pair).strip().upper()
            if not pair_name or not isinstance(state, dict):
                continue
            current = dict(out.get(pair_name, {}))
            current.update(state)
            out[pair_name] = current
    return out


def _collect_scopes(conn: libsql.Connection,
                    profile_name: str,
                    config: dict[str, Any],
                    registry_rows: dict[tuple[str, str], dict[str, Any]]) -> list[dict[str, Any]]:
    rows = query_all(
        conn,
        """
        SELECT symbol, plugin_name, MAX(family_id) AS family_id
          FROM (
                SELECT symbol, plugin_name, family_id
                  FROM champion_registry
                 WHERE profile_name = ?
                UNION ALL
                SELECT symbol, plugin_name, family_id
                  FROM student_deployment_bundles
                 WHERE profile_name = ?
                UNION ALL
                SELECT symbol, plugin_name, family_id
                  FROM shadow_fleet_observations
                 WHERE profile_name = ?
               )
         GROUP BY symbol, plugin_name
         ORDER BY symbol, plugin_name
        """,
        (profile_name, profile_name, profile_name),
    )
    scopes: dict[tuple[str, str], dict[str, Any]] = {
        (str(row["symbol"]), str(row["plugin_name"])): {
            "symbol": str(row["symbol"]),
            "plugin_name": str(row["plugin_name"]),
            "family_id": int(row.get("family_id", 11) or 11),
            "origin": "runtime",
        }
        for row in rows
    }

    if bool(config.get("challenger_policy", {}).get("discover_from_best_configs", True)):
        candidate_limit = int(config["challenger_policy"]["max_candidates_per_symbol"])
        best_rows = query_all(
            conn,
            """
            SELECT symbol, plugin_name, family_id, run_id, ranking_score, score
              FROM best_configs
             WHERE profile_name = ?
             ORDER BY symbol, ranking_score DESC, score DESC, plugin_name ASC
            """,
            (profile_name,),
        )
        per_symbol = 0
        last_symbol = ""
        for row in best_rows:
            symbol = str(row["symbol"])
            plugin_name = str(row["plugin_name"])
            key = (symbol, plugin_name)
            if symbol != last_symbol:
                per_symbol = 0
                last_symbol = symbol
            if key in scopes or key in registry_rows:
                continue
            if per_symbol >= candidate_limit:
                continue
            scopes[key] = {
                "symbol": symbol,
                "plugin_name": plugin_name,
                "family_id": int(row.get("family_id", 11) or 11),
                "origin": "discovered_challenger",
                "challenger_run_id": int(row.get("run_id", 0) or 0),
                "ranking_score": float(row.get("ranking_score", 0.0) or 0.0),
                "score": float(row.get("score", 0.0) or 0.0),
            }
            per_symbol += 1
    return list(scopes.values())


def _select_windows(rows: list[dict[str, Any]],
                    *,
                    max_recent: int,
                    max_recent_age_hours: int,
                    max_reference: int,
                    max_reference_age_hours: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not rows:
        return [], []
    latest_at = int(rows[0].get("captured_at", 0) or 0)
    recent_cutoff = latest_at - max(int(max_recent_age_hours), 1) * 3600
    reference_cutoff = latest_at - max(int(max_reference_age_hours), 1) * 3600

    recent = [row for row in rows if int(row.get("captured_at", 0) or 0) >= recent_cutoff][:max_recent]
    recent_keys = {(int(row.get("captured_at", 0) or 0), int(row.get("id", 0) or 0)) for row in recent}
    reference = [
        row
        for row in rows
        if int(row.get("captured_at", 0) or 0) >= reference_cutoff
        and (int(row.get("captured_at", 0) or 0), int(row.get("id", 0) or 0)) not in recent_keys
    ][:max_reference]
    return recent, reference


def _series(rows: list[dict[str, Any]], key: str) -> list[float]:
    return [_safe_float(row.get(key, 0.0), 0.0) for row in rows]


def _mean_map(rows: list[dict[str, Any]], keys: list[str]) -> dict[str, float]:
    return {key: mean(_series(rows, key)) for key in keys}


def _resolve_reference_rows(symbol: str,
                            plugin_name: str,
                            by_scope: dict[tuple[str, str], list[dict[str, Any]]],
                            by_plugin: dict[str, list[dict[str, Any]]],
                            config: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str]:
    recent_cfg = dict(config.get("recent_window", {}))
    reference_cfg = dict(config.get("reference_window", {}))
    symbol_rows = by_scope.get((symbol, plugin_name), [])
    recent, reference = _select_windows(
        symbol_rows,
        max_recent=int(recent_cfg.get("max_observations", 24) or 24),
        max_recent_age_hours=int(recent_cfg.get("max_age_hours", 96) or 96),
        max_reference=int(reference_cfg.get("max_observations", 120) or 120),
        max_reference_age_hours=int(reference_cfg.get("max_age_hours", 1440) or 1440),
    )
    if len(reference) >= int(config["support"]["min_reference_observations"]):
        return recent, reference, "SYMBOL_PLUGIN"

    plugin_rows = by_plugin.get(plugin_name, [])
    recent_keys = {(int(row.get("captured_at", 0) or 0), int(row.get("id", 0) or 0)) for row in recent}
    fallback_reference = [
        row
        for row in plugin_rows
        if (int(row.get("captured_at", 0) or 0), int(row.get("id", 0) or 0)) not in recent_keys
    ][: int(reference_cfg.get("max_observations", 120) or 120)]
    return recent, fallback_reference, "PLUGIN_GLOBAL"


def _feature_drift_details(recent_rows: list[dict[str, Any]],
                           reference_rows: list[dict[str, Any]],
                           config: dict[str, Any]) -> tuple[float, dict[str, float]]:
    thresholds = config["thresholds"]
    details: dict[str, float] = {}
    for key in list(config["metric_keys"]["feature_keys"]):
        psi = population_stability_index(_series(reference_rows, key), _series(recent_rows, key))
        details[key] = round(psi, 6)
    if not details:
        return 0.0, details
    score = mean([
        clamp(
            (float(details[key]) - float(thresholds["feature_psi_warn"])) /
            max(float(thresholds["feature_psi_degrade"]) - float(thresholds["feature_psi_warn"]), 1e-9),
            0.0,
            1.0,
        )
        for key in details
    ])
    return clamp(score, 0.0, 1.0), details


def _regime_drift_score(recent_rows: list[dict[str, Any]],
                        reference_rows: list[dict[str, Any]],
                        config: dict[str, Any]) -> tuple[float, dict[str, Any]]:
    thresholds = config["thresholds"]
    recent_regimes = [_safe_int(row.get("regime_id", 0), 0) for row in recent_rows]
    reference_regimes = [_safe_int(row.get("regime_id", 0), 0) for row in reference_rows]
    mix_shift = regime_mix_shift_score(reference_regimes, recent_regimes)
    score = clamp(
        (mix_shift - float(thresholds["regime_mix_warn"])) /
        max(float(thresholds["regime_mix_degrade"]) - float(thresholds["regime_mix_warn"]), 1e-9),
        0.0,
        1.0,
    )
    return score, {
        "mix_shift": round(mix_shift, 6),
        "recent_regime_counts": dict(Counter(recent_regimes)),
        "reference_regime_counts": dict(Counter(reference_regimes)),
    }


def _calibration_drift_score(recent_rows: list[dict[str, Any]],
                             reference_rows: list[dict[str, Any]],
                             prob_context: dict[str, Any],
                             config: dict[str, Any]) -> tuple[float, dict[str, float]]:
    thresholds = config["thresholds"]
    reliability_shift = downside_shift_score(
        _series(reference_rows, "reliability"),
        _series(recent_rows, "reliability"),
        thresholds["calibration_warn_delta"],
        thresholds["calibration_degrade_delta"],
    )
    ref_overconfidence = [max(meta - rel, 0.0) for meta, rel in zip(_series(reference_rows, "meta_weight"), _series(reference_rows, "reliability"))]
    recent_overconfidence = [max(meta - rel, 0.0) for meta, rel in zip(_series(recent_rows, "meta_weight"), _series(recent_rows, "reliability"))]
    overconfidence_shift = upside_shift_score(
        ref_overconfidence,
        recent_overconfidence,
        thresholds["calibration_warn_delta"] * 0.5,
        thresholds["calibration_degrade_delta"] * 0.5,
    )
    uncertainty_penalty = 0.0
    if prob_context:
        uncertainty_floor = float(config["context_modifiers"]["elevated_uncertainty_floor"])
        uncertainty_penalty = clamp(
            (float(prob_context.get("average_uncertainty_score", 0.0) or 0.0) - uncertainty_floor) /
            max(1.0 - uncertainty_floor, 1e-9),
            0.0,
            1.0,
        )
    score = clamp(0.50 * reliability_shift + 0.30 * overconfidence_shift + 0.20 * uncertainty_penalty, 0.0, 1.0)
    return score, {
        "reliability_shift": round(reliability_shift, 6),
        "overconfidence_shift": round(overconfidence_shift, 6),
        "uncertainty_penalty": round(uncertainty_penalty, 6),
    }


def _pair_decay_score(recent_rows: list[dict[str, Any]],
                      reference_rows: list[dict[str, Any]],
                      config: dict[str, Any]) -> tuple[float, dict[str, float]]:
    thresholds = config["thresholds"]
    metrics = {}
    scores = []
    for key in list(config["metric_keys"]["performance_keys"]):
        score = downside_shift_score(
            _series(reference_rows, key),
            _series(recent_rows, key),
            thresholds["performance_warn_delta"],
            thresholds["performance_degrade_delta"],
        )
        metrics[key] = round(score, 6)
        scores.append(score)
    route_regret_score = upside_shift_score(
        _series(reference_rows, "route_regret"),
        _series(recent_rows, "route_regret"),
        thresholds["performance_warn_delta"],
        thresholds["performance_degrade_delta"],
    )
    metrics["route_regret"] = round(route_regret_score, 6)
    scores.append(route_regret_score)
    return clamp(mean(scores), 0.0, 1.0), metrics


def _post_event_drift_score(recent_rows: list[dict[str, Any]],
                            reference_rows: list[dict[str, Any]],
                            adaptive_context: dict[str, Any],
                            newspulse_context: dict[str, Any],
                            config: dict[str, Any]) -> tuple[float, dict[str, float]]:
    thresholds = config["thresholds"]
    modifiers = config["context_modifiers"]
    route_regret = upside_shift_score(
        _series(reference_rows, "route_regret"),
        _series(recent_rows, "route_regret"),
        thresholds["performance_warn_delta"],
        thresholds["performance_degrade_delta"],
    )
    no_trade = upside_shift_score(
        _series(reference_rows, "policy_no_trade_prob"),
        _series(recent_rows, "policy_no_trade_prob"),
        thresholds["performance_warn_delta"],
        thresholds["performance_degrade_delta"],
    )
    event_active = False
    if adaptive_context:
        latest = adaptive_context.get("latest", {})
        if isinstance(latest, dict):
            regime = str(latest.get("regime", {}).get("top_label", "") or latest.get("top_regime", "")).upper()
            posture = str(latest.get("router", {}).get("trade_posture", "") or latest.get("trade_posture", "")).upper()
            gates = set(str(item).upper() for item in modifiers.get("event_or_posture_risk_gates", []))
            event_active = regime in gates or posture in gates
    if newspulse_context:
        gate = str(newspulse_context.get("trade_gate", "") or "").upper()
        event_active = event_active or gate in set(str(item).upper() for item in modifiers.get("event_or_posture_risk_gates", []))

    base = 0.55 * route_regret + 0.45 * no_trade
    if event_active:
        base *= float(modifiers["event_score_multiplier"])
    return clamp(base, 0.0, 1.0), {
        "route_regret_shift": round(route_regret, 6),
        "no_trade_shift": round(no_trade, 6),
        "event_active": 1.0 if event_active else 0.0,
    }


def _execution_drift_score(recent_rows: list[dict[str, Any]],
                           reference_rows: list[dict[str, Any]],
                           execution_context: dict[str, Any],
                           micro_context: dict[str, Any],
                           config: dict[str, Any]) -> tuple[float, dict[str, float]]:
    thresholds = config["thresholds"]
    modifiers = config["context_modifiers"]
    scores = []
    details: dict[str, float] = {}
    for key in list(config["metric_keys"]["execution_keys"]):
        score = upside_shift_score(
            _series(reference_rows, key),
            _series(recent_rows, key),
            thresholds["execution_warn_delta"],
            thresholds["execution_degrade_delta"],
        )
        details[key] = round(score, 6)
        scores.append(score)
    context_penalty = 0.0
    if execution_context:
        min_quality = float(execution_context.get("min_execution_quality_score", 1.0) or 1.0)
        floor = float(modifiers["stressed_execution_quality_floor"])
        context_penalty = clamp((floor - min_quality) / max(floor, 1e-9), 0.0, 1.0)
        latest_state = str(dict(execution_context.get("latest", {})).get("state", {}).get("execution_state", "")).upper()
        if latest_state in {"STRESSED", "BLOCKED"}:
            context_penalty = max(context_penalty, 0.65)
    hostile_penalty = 0.0
    if micro_context:
        hostile_penalty = clamp(float(micro_context.get("max_hostile_execution_score", 0.0) or 0.0), 0.0, 1.0)
    details["context_penalty"] = round(context_penalty, 6)
    details["hostile_penalty"] = round(hostile_penalty, 6)
    scores.extend([context_penalty, hostile_penalty])
    return clamp(mean(scores), 0.0, 1.0), details


def _performance_drift_score(recent_rows: list[dict[str, Any]],
                             reference_rows: list[dict[str, Any]],
                             ensemble_context: dict[str, Any],
                             config: dict[str, Any]) -> tuple[float, dict[str, float]]:
    thresholds = config["thresholds"]
    score_components = []
    details: dict[str, float] = {}
    for key in ("shadow_score", "route_value", "portfolio_objective", "portfolio_stability"):
        score = downside_shift_score(
            _series(reference_rows, key),
            _series(recent_rows, key),
            thresholds["performance_warn_delta"],
            thresholds["performance_degrade_delta"],
        )
        score_components.append(score)
        details[key] = round(score, 6)
    if ensemble_context:
        avg_quality = float(ensemble_context.get("average_quality", 1.0) or 1.0)
        abstain_bias = float(ensemble_context.get("max_abstain_bias", 0.0) or 0.0)
        quality_penalty = clamp((0.55 - avg_quality) / 0.55, 0.0, 1.0)
        abstain_penalty = clamp((abstain_bias - 0.35) / 0.65, 0.0, 1.0)
        score_components.extend([quality_penalty, abstain_penalty])
        details["ensemble_quality_penalty"] = round(quality_penalty, 6)
        details["abstain_penalty"] = round(abstain_penalty, 6)
    return clamp(mean(score_components), 0.0, 1.0), details


def _load_tuning_reference(conn: libsql.Connection,
                           profile_name: str,
                           symbol: str,
                           plugin_name: str) -> dict[str, Any]:
    run_rows = query_all(
        conn,
        """
        SELECT score, market_recent_score, walkforward_score, adversarial_score, macro_event_score, issue_count
          FROM tuning_runs
         WHERE profile_name = ? AND symbol = ? AND plugin_name = ? AND status = 'ok'
         ORDER BY finished_at DESC
         LIMIT 48
        """,
        (profile_name, symbol, plugin_name),
    )
    scenario_rows = query_all(
        conn,
        """
        SELECT rs.scenario, AVG(rs.calibration_error) AS mean_calibration_error, AVG(rs.path_quality_error) AS mean_path_quality_error
          FROM run_scenarios rs
          JOIN tuning_runs tr ON tr.id = rs.run_id
         WHERE tr.profile_name = ? AND tr.symbol = ? AND tr.plugin_name = ? AND tr.status = 'ok'
         GROUP BY rs.scenario
        """,
        (profile_name, symbol, plugin_name),
    )
    if not run_rows:
        return {
            "support_count": 0,
            "mean_score": 0.0,
            "mean_recent_score": 0.0,
            "mean_walkforward_score": 0.0,
            "mean_adversarial_score": 0.0,
            "mean_macro_event_score": 0.0,
            "mean_issue_count": 0.0,
            "mean_calibration_error": 0.0,
            "mean_path_quality_error": 0.0,
        }
    return {
        "support_count": len(run_rows),
        "mean_score": round(mean([_safe_float(row["score"]) for row in run_rows]), 6),
        "mean_recent_score": round(mean([_safe_float(row["market_recent_score"]) for row in run_rows]), 6),
        "mean_walkforward_score": round(mean([_safe_float(row["walkforward_score"]) for row in run_rows]), 6),
        "mean_adversarial_score": round(mean([_safe_float(row["adversarial_score"]) for row in run_rows]), 6),
        "mean_macro_event_score": round(mean([_safe_float(row["macro_event_score"]) for row in run_rows]), 6),
        "mean_issue_count": round(mean([_safe_float(row["issue_count"]) for row in run_rows]), 6),
        "mean_calibration_error": round(mean([_safe_float(row["mean_calibration_error"]) for row in scenario_rows]), 6),
        "mean_path_quality_error": round(mean([_safe_float(row["mean_path_quality_error"]) for row in scenario_rows]), 6),
    }


def _evaluate_challenger(scope: dict[str, Any],
                         tuning_reference: dict[str, Any],
                         recent_rows: list[dict[str, Any]],
                         champion_score: float,
                         config: dict[str, Any]) -> dict[str, Any]:
    challenger_cfg = config["challenger_policy"]
    support_count = _safe_int(tuning_reference.get("support_count", 0), 0)
    shadow_support = len(recent_rows)
    live_shadow_score = mean(_series(recent_rows, "shadow_score"))
    live_reliability = mean(_series(recent_rows, "reliability"))
    mean_walkforward_score = _safe_float(tuning_reference.get("mean_walkforward_score", 0.0), 0.0)
    mean_recent_score = _safe_float(tuning_reference.get("mean_recent_score", 0.0), 0.0)
    mean_adversarial_score = _safe_float(tuning_reference.get("mean_adversarial_score", 0.0), 0.0)
    mean_macro_event_score = _safe_float(tuning_reference.get("mean_macro_event_score", 0.0), 0.0)
    mean_calibration_error = _safe_float(tuning_reference.get("mean_calibration_error", 1.0), 1.0)
    mean_issue_count = _safe_float(tuning_reference.get("mean_issue_count", 99.0), 99.0)
    mean_score = _safe_float(tuning_reference.get("mean_score", 0.0), 0.0)
    portfolio_score = 0.65 * mean_walkforward_score / 100.0 + 0.35 * live_shadow_score
    promotion_margin = mean_score - float(champion_score or 0.0)
    qualifies = (
        support_count >= int(config["support"]["min_challenger_runs"]) and
        shadow_support >= int(config["support"]["min_challenger_shadow_observations"]) and
        mean_walkforward_score >= float(challenger_cfg["min_walkforward_score"]) and
        mean_recent_score >= float(challenger_cfg["min_recent_score"]) and
        mean_adversarial_score >= float(challenger_cfg["min_adversarial_score"]) and
        mean_macro_event_score >= float(challenger_cfg["min_macro_event_score"]) and
        mean_calibration_error <= float(challenger_cfg["max_calibration_error"]) and
        mean_issue_count <= float(challenger_cfg["max_issue_count"]) and
        portfolio_score >= float(challenger_cfg["min_portfolio_score"]) and
        live_shadow_score >= float(challenger_cfg["min_shadow_score"]) and
        live_reliability >= float(challenger_cfg["min_reliability"]) and
        promotion_margin >= float(challenger_cfg["promotion_margin"])
    )
    return {
        "plugin_name": str(scope["plugin_name"]),
        "symbol": str(scope["symbol"]),
        "challenger_run_id": int(scope.get("challenger_run_id", 0) or 0),
        "support_count": support_count,
        "shadow_support": shadow_support,
        "walkforward_score": round(mean_walkforward_score, 6),
        "recent_score": round(mean_recent_score, 6),
        "adversarial_score": round(mean_adversarial_score, 6),
        "macro_event_score": round(mean_macro_event_score, 6),
        "calibration_error": round(mean_calibration_error, 6),
        "issue_count": round(mean_issue_count, 6),
        "live_shadow_score": round(live_shadow_score, 6),
        "live_reliability": round(live_reliability, 6),
        "portfolio_score": round(portfolio_score, 6),
        "promotion_margin": round(promotion_margin, 6),
        "eligibility_state": "QUALIFIED" if qualifies else "INSUFFICIENT",
        "qualifies": bool(qualifies),
    }


def _default_operational_state(base_status: str) -> str:
    status = str(base_status or "").lower()
    if status == "champion":
        return "CHAMPION"
    if status in {"challenger", "candidate"}:
        return "CHALLENGER"
    return "HEALTHY"


def _recommended_state(base_status: str,
                       aggregate_score: float,
                       health_state: str,
                       challenger_eval: dict[str, Any],
                       registry_row: dict[str, Any],
                       config: dict[str, Any]) -> tuple[str, str]:
    if health_state == "DISABLED":
        if str(base_status or "").lower() == "champion":
            return "DISABLED", "DISABLE"
        return "DISABLED", "DISABLE"
    if health_state == "SHADOW_ONLY":
        if str(base_status or "").lower() == "champion":
            return "DEMOTED", "DEMOTE"
        return "SHADOW_ONLY", "SHADOW_ONLY"
    if health_state == "DEGRADED":
        return "DEGRADED", "DOWNWEIGHT"
    if health_state == "CAUTION":
        return "CAUTION", "DOWNWEIGHT"
    if challenger_eval.get("qualifies"):
        return "CHAMPION_CANDIDATE", "PROMOTION_REVIEW"
    if str(base_status or "").lower() == "champion":
        rollback_hours = int(config["challenger_policy"]["rollback_hours"])
        promoted_at = int(registry_row.get("promoted_at", 0) or 0)
        if promoted_at > 0 and now_unix() - promoted_at <= rollback_hours * 3600 and aggregate_score >= float(config["thresholds"]["shadow_only_score"]):
            return "DEMOTED", "ROLLBACK"
        return "CHAMPION", "NONE"
    if str(base_status or "").lower() in {"challenger", "candidate"}:
        return "CHALLENGER", "NONE"
    return "HEALTHY", "NONE"


def _health_state(aggregate_score: float, config: dict[str, Any]) -> str:
    thresholds = config["thresholds"]
    if aggregate_score >= float(thresholds["disable_score"]):
        return "DISABLED"
    if aggregate_score >= float(thresholds["shadow_only_score"]):
        return "SHADOW_ONLY"
    if aggregate_score >= float(thresholds["degraded_score"]):
        return "DEGRADED"
    if aggregate_score >= float(thresholds["caution_score"]):
        return "CAUTION"
    return "HEALTHY"


def _should_apply_action(action: str, config: dict[str, Any]) -> bool:
    mode = str(config.get("action_mode", "AUTO_APPLY_PROTECTIVE") or "AUTO_APPLY_PROTECTIVE")
    actions_cfg = dict(config.get("actions", {}))
    action_upper = str(action or "NONE").upper()
    if action_upper == "NONE":
        return False
    if mode == "RECOMMEND_ONLY":
        return False
    if action_upper in {"PROMOTION_REVIEW"}:
        return False
    if action_upper == "PROMOTE":
        return mode == "AUTO_APPLY_ALL" and bool(actions_cfg.get("auto_promote", False))
    if action_upper in {"DISABLE"}:
        return bool(actions_cfg.get("auto_disable", True))
    if action_upper in {"DEMOTE", "ROLLBACK"}:
        return bool(actions_cfg.get("auto_demote", True))
    return mode in {"AUTO_APPLY_PROTECTIVE", "AUTO_APPLY_ALL"}


def _apply_state_machine(previous_state: dict[str, Any] | None,
                         base_status: str,
                         recommended_state: str,
                         action: str,
                         health_state: str,
                         aggregate_score: float,
                         low_support: bool,
                         config: dict[str, Any]) -> dict[str, Any]:
    thresholds = config["thresholds"]
    hysteresis = config["hysteresis"]
    actions_cfg = config["actions"]
    prior_state = str(previous_state.get("governance_state", "") or _default_operational_state(base_status)) if previous_state else _default_operational_state(base_status)
    prior_updated_at = int(previous_state.get("updated_at", 0) or 0) if previous_state else 0
    apply_action = _should_apply_action(action, config)
    target_state = recommended_state
    operational_state = prior_state

    if low_support:
        if severity_rank(prior_state) > severity_rank("CAUTION"):
            target_state = prior_state
        else:
            target_state = _default_operational_state(base_status)
        apply_action = False
    elif apply_action:
        if severity_rank(target_state) < severity_rank(prior_state):
            cooldown_seconds = int(hysteresis["recovery_hold_hours"]) * 3600
            if prior_updated_at > 0 and now_unix() - prior_updated_at < cooldown_seconds:
                target_state = prior_state
                apply_action = False
            elif aggregate_score > float(thresholds["recover_score"]):
                target_state = prior_state
                apply_action = False
            elif target_state in {"CHAMPION", "CHALLENGER", "HEALTHY"} and aggregate_score > float(thresholds["recover_to_healthy_score"]):
                target_state = prior_state
                apply_action = False

    if apply_action:
        operational_state = target_state

    weight_multiplier = 1.0
    if operational_state in {"CAUTION"}:
        weight_multiplier = float(actions_cfg["caution_weight_multiplier"])
    elif operational_state in {"DEGRADED"}:
        weight_multiplier = float(actions_cfg["degraded_weight_multiplier"])
    elif operational_state in {"SHADOW_ONLY", "DEMOTED"}:
        weight_multiplier = float(actions_cfg["shadow_only_weight_multiplier"])
    elif operational_state in {"DISABLED"}:
        weight_multiplier = float(actions_cfg["disabled_weight_multiplier"])

    restrict_live = apply_action and operational_state in set(str(item).upper() for item in actions_cfg.get("restrict_live_states", []))
    shadow_only = apply_action and operational_state == "SHADOW_ONLY"
    disabled = apply_action and operational_state == "DISABLED"
    return {
        "operational_state": operational_state,
        "action_applied": bool(apply_action and operational_state != prior_state),
        "weight_multiplier": clamp(weight_multiplier, 0.0, 2.0),
        "restrict_live": bool(restrict_live),
        "shadow_only": bool(shadow_only),
        "disabled": bool(disabled),
    }


def _upsert_drift_snapshot(conn: libsql.Connection,
                           *,
                           profile_name: str,
                           scope: dict[str, Any],
                           reference_scope: str,
                           scores: dict[str, float],
                           aggregate_score: float,
                           recent_rows: list[dict[str, Any]],
                           reference_rows: list[dict[str, Any]],
                           quality_flags: dict[str, Any],
                           payload: dict[str, Any]) -> None:
    conn.execute(
        """
        INSERT INTO plugin_drift_snapshots(
            profile_name, symbol, plugin_name, family_id, context_scope, reference_scope,
            feature_drift_score, regime_drift_score, calibration_drift_score, pair_decay_score,
            post_event_drift_score, execution_drift_score, performance_drift_score, aggregate_risk_score,
            recent_support, reference_support, recent_window_start, recent_window_end,
            reference_window_start, reference_window_end, quality_flags_json, payload_json, created_at
        )
        VALUES(?, ?, ?, ?, 'symbol_plugin', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            profile_name,
            str(scope["symbol"]),
            str(scope["plugin_name"]),
            int(scope.get("family_id", 11) or 11),
            reference_scope,
            float(scores["feature_drift_score"]),
            float(scores["regime_drift_score"]),
            float(scores["calibration_drift_score"]),
            float(scores["pair_decay_score"]),
            float(scores["post_event_drift_score"]),
            float(scores["execution_drift_score"]),
            float(scores["performance_drift_score"]),
            float(aggregate_score),
            len(recent_rows),
            len(reference_rows),
            int(recent_rows[-1]["captured_at"]) if recent_rows else 0,
            int(recent_rows[0]["captured_at"]) if recent_rows else 0,
            int(reference_rows[-1]["captured_at"]) if reference_rows else 0,
            int(reference_rows[0]["captured_at"]) if reference_rows else 0,
            json.dumps(quality_flags, sort_keys=True),
            json.dumps(payload, sort_keys=True),
            now_unix(),
        ),
    )


def _upsert_governance_state(conn: libsql.Connection,
                             *,
                             profile_name: str,
                             scope: dict[str, Any],
                             base_registry_status: str,
                             health_state: str,
                             operational_state: str,
                             action: str,
                             action_applied: bool,
                             weight_multiplier: float,
                             restrict_live: bool,
                             shadow_only: bool,
                             disabled: bool,
                             candidate_eligible: bool,
                             champion_eligible: bool,
                             aggregate_score: float,
                             reason_codes: list[str],
                             quality_flags: dict[str, Any],
                             payload: dict[str, Any],
                             policy_version: int) -> None:
    conn.execute(
        """
        INSERT INTO plugin_governance_states(
            profile_name, symbol, plugin_name, family_id, base_registry_status, health_state, governance_state,
            action_recommendation, action_applied, weight_multiplier, restrict_live, shadow_only, disabled,
            candidate_eligible, champion_eligible, aggregate_risk_score, reason_codes_json, quality_flags_json,
            payload_json, policy_version, updated_at
        )
        VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(profile_name, symbol, plugin_name) DO UPDATE SET
            family_id=excluded.family_id,
            base_registry_status=excluded.base_registry_status,
            health_state=excluded.health_state,
            governance_state=excluded.governance_state,
            action_recommendation=excluded.action_recommendation,
            action_applied=excluded.action_applied,
            weight_multiplier=excluded.weight_multiplier,
            restrict_live=excluded.restrict_live,
            shadow_only=excluded.shadow_only,
            disabled=excluded.disabled,
            candidate_eligible=excluded.candidate_eligible,
            champion_eligible=excluded.champion_eligible,
            aggregate_risk_score=excluded.aggregate_risk_score,
            reason_codes_json=excluded.reason_codes_json,
            quality_flags_json=excluded.quality_flags_json,
            payload_json=excluded.payload_json,
            policy_version=excluded.policy_version,
            updated_at=excluded.updated_at
        """,
        (
            profile_name,
            str(scope["symbol"]),
            str(scope["plugin_name"]),
            int(scope.get("family_id", 11) or 11),
            str(base_registry_status),
            str(health_state),
            str(operational_state),
            str(action),
            1 if action_applied else 0,
            float(weight_multiplier),
            1 if restrict_live else 0,
            1 if shadow_only else 0,
            1 if disabled else 0,
            1 if candidate_eligible else 0,
            1 if champion_eligible else 0,
            float(aggregate_score),
            json.dumps(reason_codes, sort_keys=True),
            json.dumps(quality_flags, sort_keys=True),
            json.dumps(payload, sort_keys=True),
            int(policy_version),
            now_unix(),
        ),
    )


def _insert_governance_action(conn: libsql.Connection,
                              *,
                              profile_name: str,
                              scope: dict[str, Any],
                              previous_state: str,
                              new_state: str,
                              action_kind: str,
                              action_applied: bool,
                              reason_codes: list[str],
                              payload: dict[str, Any],
                              policy_version: int) -> None:
    conn.execute(
        """
        INSERT INTO plugin_governance_actions(
            profile_name, symbol, plugin_name, family_id, previous_state, new_state, action_kind,
            action_applied, policy_version, reason_codes_json, payload_json, created_at
        )
        VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            profile_name,
            str(scope["symbol"]),
            str(scope["plugin_name"]),
            int(scope.get("family_id", 11) or 11),
            str(previous_state),
            str(new_state),
            str(action_kind),
            1 if action_applied else 0,
            int(policy_version),
            json.dumps(reason_codes, sort_keys=True),
            json.dumps(payload, sort_keys=True),
            now_unix(),
        ),
    )


def _upsert_challenger_eval(conn: libsql.Connection,
                            *,
                            profile_name: str,
                            scope: dict[str, Any],
                            challenger_eval: dict[str, Any]) -> None:
    conn.execute(
        """
        INSERT INTO plugin_challenger_evaluations(
            profile_name, symbol, plugin_name, family_id, challenger_run_id, evaluation_scope, eligibility_state,
            walkforward_score, recent_score, adversarial_score, macro_event_score, calibration_error,
            issue_count, live_shadow_score, live_reliability, portfolio_score, support_count,
            promotion_margin, qualifies, payload_json, created_at
        )
        VALUES(?, ?, ?, ?, ?, 'symbol', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(profile_name, symbol, plugin_name, evaluation_scope, challenger_run_id) DO UPDATE SET
            eligibility_state=excluded.eligibility_state,
            walkforward_score=excluded.walkforward_score,
            recent_score=excluded.recent_score,
            adversarial_score=excluded.adversarial_score,
            macro_event_score=excluded.macro_event_score,
            calibration_error=excluded.calibration_error,
            issue_count=excluded.issue_count,
            live_shadow_score=excluded.live_shadow_score,
            live_reliability=excluded.live_reliability,
            portfolio_score=excluded.portfolio_score,
            support_count=excluded.support_count,
            promotion_margin=excluded.promotion_margin,
            qualifies=excluded.qualifies,
            payload_json=excluded.payload_json,
            created_at=excluded.created_at
        """,
        (
            profile_name,
            str(scope["symbol"]),
            str(scope["plugin_name"]),
            int(scope.get("family_id", 11) or 11),
            int(challenger_eval.get("challenger_run_id", 0) or 0),
            str(challenger_eval.get("eligibility_state", "INSUFFICIENT")),
            float(challenger_eval.get("walkforward_score", 0.0) or 0.0),
            float(challenger_eval.get("recent_score", 0.0) or 0.0),
            float(challenger_eval.get("adversarial_score", 0.0) or 0.0),
            float(challenger_eval.get("macro_event_score", 0.0) or 0.0),
            float(challenger_eval.get("calibration_error", 0.0) or 0.0),
            float(challenger_eval.get("issue_count", 0.0) or 0.0),
            float(challenger_eval.get("live_shadow_score", 0.0) or 0.0),
            float(challenger_eval.get("live_reliability", 0.0) or 0.0),
            float(challenger_eval.get("portfolio_score", 0.0) or 0.0),
            int(challenger_eval.get("support_count", 0) or 0),
            float(challenger_eval.get("promotion_margin", 0.0) or 0.0),
            1 if bool(challenger_eval.get("qualifies", False)) else 0,
            json.dumps(challenger_eval, sort_keys=True),
            now_unix(),
        ),
    )


def latest_governance_state_map(conn: libsql.Connection,
                                profile_name: str,
                                symbol: str = "") -> dict[str, dict[str, Any]] | dict[tuple[str, str], dict[str, Any]]:
    params: list[Any] = [profile_name]
    clauses = ["profile_name = ?"]
    if symbol:
        clauses.append("symbol = ?")
        params.append(str(symbol))
    rows = query_all(
        conn,
        f"""
        SELECT *
          FROM plugin_governance_states
         WHERE {' AND '.join(clauses)}
         ORDER BY symbol, plugin_name
        """,
        params,
    )
    if symbol:
        return {str(row["plugin_name"]): dict(row) for row in rows}
    return {(str(row["symbol"]), str(row["plugin_name"])): dict(row) for row in rows}


def validate_drift_governance_config() -> dict[str, Any]:
    payload = load_config()
    return {
        "ok": True,
        "config_path": str(DRIFT_GOVERNANCE_CONFIG_PATH),
        "config": payload,
    }


def build_drift_governance_report(conn: libsql.Connection,
                                  profile_name: str) -> dict[str, Any]:
    ensure_drift_governance_dirs()
    config = load_config()
    state_rows = query_all(
        conn,
        """
        SELECT *
          FROM plugin_governance_states
         WHERE profile_name = ?
         ORDER BY symbol, aggregate_risk_score DESC, plugin_name ASC
        """,
        (profile_name,),
    )
    action_rows = query_all(
        conn,
        """
        SELECT *
          FROM plugin_governance_actions
         WHERE profile_name = ?
         ORDER BY created_at DESC, id DESC
         LIMIT 256
        """,
        (profile_name,),
    )
    symbol_actions: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in action_rows:
        symbol_actions[str(row["symbol"])].append(dict(row))

    by_symbol: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in state_rows:
        by_symbol[str(row["symbol"])].append(dict(row))

    symbols: list[dict[str, Any]] = []
    health_counts = Counter()
    governance_counts = Counter()
    action_counts = Counter()
    for symbol, rows in sorted(by_symbol.items()):
        plugins: list[dict[str, Any]] = []
        symbol_health = Counter()
        symbol_governance = Counter()
        symbol_actions_counter = Counter()
        for row in rows:
            payload = _safe_json(str(row.get("payload_json", "{}") or "{}"), {})
            plugin = {
                "plugin_name": str(row["plugin_name"]),
                "family_id": int(row.get("family_id", 11) or 11),
                "family_name": plugin_family_name(int(row.get("family_id", 11) or 11)),
                "base_registry_status": str(row.get("base_registry_status", "")),
                "health_state": str(row.get("health_state", "HEALTHY")),
                "governance_state": str(row.get("governance_state", "HEALTHY")),
                "recommended_governance_state": str(payload.get("recommended_governance_state", row.get("governance_state", "HEALTHY"))),
                "action_recommendation": str(row.get("action_recommendation", "NONE")),
                "action_applied": bool(int(row.get("action_applied", 0) or 0)),
                "weight_multiplier": round(float(row.get("weight_multiplier", 1.0) or 1.0), 6),
                "restrict_live": bool(int(row.get("restrict_live", 0) or 0)),
                "shadow_only": bool(int(row.get("shadow_only", 0) or 0)),
                "disabled": bool(int(row.get("disabled", 0) or 0)),
                "aggregate_risk_score": round(float(row.get("aggregate_risk_score", 0.0) or 0.0), 6),
                "drift_scores": dict(payload.get("drift_scores", {})),
                "support": dict(payload.get("support", {})),
                "reason_codes": list(payload.get("reason_codes", [])),
                "quality_flags": dict(payload.get("quality_flags", {})),
                "challenger_evaluation": dict(payload.get("challenger_evaluation", {})),
                "context": dict(payload.get("context", {})),
            }
            plugins.append(plugin)
            symbol_health[plugin["health_state"]] += 1
            symbol_governance[plugin["governance_state"]] += 1
            symbol_actions_counter[plugin["action_recommendation"]] += 1
            health_counts[plugin["health_state"]] += 1
            governance_counts[plugin["governance_state"]] += 1
            action_counts[plugin["action_recommendation"]] += 1
        latest_context = {}
        if plugins:
            latest_context = dict(plugins[0].get("context", {}))
        symbols.append(
            {
                "symbol": symbol,
                "plugin_count": len(plugins),
                "health_counts": dict(sorted(symbol_health.items())),
                "governance_counts": dict(sorted(symbol_governance.items())),
                "action_counts": dict(sorted(symbol_actions_counter.items())),
                "plugins": plugins,
                "recent_actions": [
                    {
                        "plugin_name": str(item.get("plugin_name", "")),
                        "previous_state": str(item.get("previous_state", "")),
                        "new_state": str(item.get("new_state", "")),
                        "action_kind": str(item.get("action_kind", "")),
                        "action_applied": bool(int(item.get("action_applied", 0) or 0)),
                        "created_at": int(item.get("created_at", 0) or 0),
                    }
                    for item in symbol_actions.get(symbol, [])[:16]
                ],
                "latest_context": latest_context,
            }
        )

    payload = {
        "schema_version": DRIFT_GOVERNANCE_SCHEMA_VERSION,
        "report_version": DRIFT_GOVERNANCE_REPORT_VERSION,
        "generated_at": isoformat_utc(),
        "profile_name": profile_name,
        "policy_version": int(config.get("policy_version", 1) or 1),
        "action_mode": str(config.get("action_mode", "AUTO_APPLY_PROTECTIVE")),
        "symbol_count": len(symbols),
        "plugin_count": len(state_rows),
        "health_counts": dict(sorted(health_counts.items())),
        "governance_counts": dict(sorted(governance_counts.items())),
        "action_counts": dict(sorted(action_counts.items())),
        "latest_action_count": len(action_rows),
        "symbols": symbols,
        "artifacts": {
            "report_path": str(DRIFT_GOVERNANCE_REPORT_PATH),
            "status_path": str(DRIFT_GOVERNANCE_STATUS_PATH),
            "history_path": str(DRIFT_GOVERNANCE_HISTORY_PATH),
        },
    }
    json_dump(DRIFT_GOVERNANCE_REPORT_PATH, payload)
    return payload


def run_drift_governance_cycle(conn: libsql.Connection,
                               args,
                               cycle_group_key: str = "") -> dict[str, Any]:
    ensure_drift_governance_dirs()
    config = load_config()
    if not bool(config.get("enabled", True)):
        payload = {
            "schema_version": DRIFT_GOVERNANCE_SCHEMA_VERSION,
            "generated_at": isoformat_utc(),
            "profile_name": args.profile,
            "enabled": False,
            "action_mode": str(config.get("action_mode", "RECOMMEND_ONLY")),
            "symbols": [],
            "actions": [],
        }
        json_dump(DRIFT_GOVERNANCE_STATUS_PATH, payload)
        json_dump(DRIFT_GOVERNANCE_RUNTIME_SUMMARY_PATH, payload)
        return payload

    by_scope, by_plugin = _load_shadow_rows(conn, args.profile)
    registry_rows = _load_registry_rows(conn, args.profile)
    previous_states = _load_previous_states(conn, args.profile)
    scopes = _collect_scopes(conn, args.profile, config, registry_rows)

    prob_context_map = _load_symbol_report_map(PROB_CALIBRATION_REPLAY_REPORT_PATH)
    execution_context_map = _load_symbol_report_map(EXECUTION_QUALITY_REPLAY_REPORT_PATH)
    adaptive_context_map = _load_symbol_report_map(ADAPTIVE_ROUTER_REPLAY_REPORT_PATH)
    cross_asset_context_map = _load_symbol_report_map(CROSS_ASSET_REPLAY_REPORT_PATH)
    micro_context_map = _load_symbol_report_map(MICROSTRUCTURE_REPLAY_REPORT_PATH)
    ensemble_context_map = _load_symbol_report_map(DYNAMIC_ENSEMBLE_REPLAY_REPORT_PATH)
    newspulse_context_map = _load_newspulse_context()

    actions: list[dict[str, Any]] = []
    symbol_plugins: dict[str, list[dict[str, Any]]] = defaultdict(list)
    policy_version = int(config.get("policy_version", 1) or 1)

    for scope in scopes:
        symbol = str(scope["symbol"])
        plugin_name = str(scope["plugin_name"])
        family_id = int(scope.get("family_id", 11) or 11)
        registry_row = registry_rows.get((symbol, plugin_name), {})
        previous_state = previous_states.get((symbol, plugin_name))
        base_status = str(registry_row.get("status", "") or scope.get("origin", "candidate")).lower()
        if base_status not in {"champion", "challenger", "candidate"}:
            base_status = "candidate" if str(scope.get("origin", "")).startswith("discovered") else "champion"

        recent_rows, reference_rows, reference_scope = _resolve_reference_rows(symbol, plugin_name, by_scope, by_plugin, config)
        low_support = (
            len(recent_rows) < int(config["support"]["min_recent_observations"]) or
            len(reference_rows) < int(config["support"]["min_reference_observations"])
        )

        prob_context = prob_context_map.get(symbol, {})
        execution_context = execution_context_map.get(symbol, {})
        adaptive_context = adaptive_context_map.get(symbol, {})
        cross_asset_context = cross_asset_context_map.get(symbol, {})
        micro_context = micro_context_map.get(symbol, {})
        ensemble_context = ensemble_context_map.get(symbol, {})
        newspulse_context = newspulse_context_map.get(symbol, {})

        feature_score, feature_details = _feature_drift_details(recent_rows, reference_rows, config)
        regime_score, regime_details = _regime_drift_score(recent_rows, reference_rows, config)
        calibration_score, calibration_details = _calibration_drift_score(recent_rows, reference_rows, prob_context, config)
        pair_decay_score, pair_decay_details = _pair_decay_score(recent_rows, reference_rows, config)
        post_event_score, post_event_details = _post_event_drift_score(recent_rows, reference_rows, adaptive_context, newspulse_context, config)
        execution_score, execution_details = _execution_drift_score(recent_rows, reference_rows, execution_context, micro_context, config)
        performance_score, performance_details = _performance_drift_score(recent_rows, reference_rows, ensemble_context, config)
        drift_scores = {
            "feature_drift_score": round(feature_score, 6),
            "regime_drift_score": round(regime_score, 6),
            "calibration_drift_score": round(calibration_score, 6),
            "pair_decay_score": round(pair_decay_score, 6),
            "post_event_drift_score": round(post_event_score, 6),
            "execution_drift_score": round(execution_score, 6),
            "performance_drift_score": round(performance_score, 6),
        }
        aggregate_score = aggregate_drift_score(drift_scores, config["aggregation_weights"])
        modifiers = config["context_modifiers"]
        if execution_context and float(execution_context.get("min_execution_quality_score", 1.0) or 1.0) < float(modifiers["stressed_execution_quality_floor"]):
            aggregate_score = clamp(aggregate_score * float(modifiers["execution_score_multiplier"]), 0.0, 1.0)
        latest_cross_asset = dict(cross_asset_context.get("latest", {}))
        if latest_cross_asset and float(latest_cross_asset.get("pair_cross_asset_risk_score", 0.0) or 0.0) > float(modifiers["elevated_cross_asset_risk"]):
            aggregate_score = clamp(aggregate_score * float(modifiers["cross_asset_score_multiplier"]), 0.0, 1.0)
        if prob_context and float(prob_context.get("average_uncertainty_score", 0.0) or 0.0) > float(modifiers["elevated_uncertainty_floor"]):
            aggregate_score = clamp(aggregate_score * float(modifiers["calibration_score_multiplier"]), 0.0, 1.0)
        health_state = _health_state(aggregate_score, config)
        tuning_reference = _load_tuning_reference(conn, args.profile, symbol, plugin_name)
        challenger_eval = _evaluate_challenger(scope, tuning_reference, recent_rows, float(registry_row.get("champion_score", 0.0) or 0.0), config)
        recommended_state, action = _recommended_state(base_status, aggregate_score, health_state, challenger_eval, registry_row, config)
        state_result = _apply_state_machine(previous_state, base_status, recommended_state, action, health_state, aggregate_score, low_support, config)

        reason_codes: list[str] = []
        if feature_score >= 0.55:
            reason_codes.append("FEATURE_DRIFT_ELEVATED")
        if regime_score >= 0.55:
            reason_codes.append("REGIME_DRIFT_ELEVATED")
        if calibration_score >= 0.55:
            reason_codes.append("CALIBRATION_DRIFT_ELEVATED")
        if pair_decay_score >= 0.55:
            reason_codes.append(f"PAIR_DECAY_{symbol}")
        if post_event_score >= 0.55:
            reason_codes.append("POST_EVENT_BEHAVIOR_DRIFT")
        if execution_score >= 0.55:
            reason_codes.append("EXECUTION_DRIFT_ELEVATED")
        if performance_score >= 0.55:
            reason_codes.append("RECENT_COST_ADJUSTED_UTILITY_WEAK")
        if low_support:
            reason_codes.append("LOW_SUPPORT")
        if bool(challenger_eval.get("qualifies", False)):
            reason_codes.append("CHALLENGER_PROMOTION_ELIGIBLE")
        if action == "ROLLBACK":
            reason_codes.append("POST_PROMOTION_DECAY")

        quality_flags = {
            "low_support": bool(low_support),
            "reference_fallback_scope": reference_scope,
            "fallback_thresholds_used": bool(low_support or reference_scope != "SYMBOL_PLUGIN"),
            "shadow_evaluation_only": bool(scope.get("origin", "") == "discovered_challenger" and not recent_rows),
        }
        payload = {
            "timestamp": isoformat_utc(),
            "plugin_id": plugin_name,
            "pair_scope": symbol,
            "base_registry_status": base_status,
            "health_state": health_state,
            "governance_state": state_result["operational_state"],
            "recommended_governance_state": recommended_state,
            "drift_scores": drift_scores,
            "drift_details": {
                "feature": feature_details,
                "regime": regime_details,
                "calibration": calibration_details,
                "pair_decay": pair_decay_details,
                "post_event": post_event_details,
                "execution": execution_details,
                "performance": performance_details,
            },
            "support": {
                "sample_count_recent": len(recent_rows),
                "sample_count_reference": len(reference_rows),
                "reference_scope": reference_scope,
            },
            "action_recommendation": action,
            "action_applied": bool(state_result["action_applied"]),
            "weight_multiplier": round(float(state_result["weight_multiplier"]), 6),
            "restrict_live": bool(state_result["restrict_live"]),
            "shadow_only": bool(state_result["shadow_only"]),
            "disabled": bool(state_result["disabled"]),
            "reason_codes": reason_codes,
            "quality_flags": quality_flags,
            "metadata": {
                "reference_window": {
                    "start": int(reference_rows[-1]["captured_at"]) if reference_rows else 0,
                    "end": int(reference_rows[0]["captured_at"]) if reference_rows else 0,
                },
                "live_window": {
                    "start": int(recent_rows[-1]["captured_at"]) if recent_rows else 0,
                    "end": int(recent_rows[0]["captured_at"]) if recent_rows else 0,
                },
                "policy_version": policy_version,
                "cycle_group_key": cycle_group_key,
            },
            "challenger_evaluation": challenger_eval,
            "tuning_reference": tuning_reference,
            "context": {
                "prob_calibration": prob_context,
                "execution_quality": execution_context,
                "adaptive_router": adaptive_context,
                "cross_asset": cross_asset_context,
                "microstructure": micro_context,
                "dynamic_ensemble": ensemble_context,
                "newspulse": newspulse_context,
            },
        }

        _upsert_drift_snapshot(
            conn,
            profile_name=args.profile,
            scope=scope,
            reference_scope=reference_scope,
            scores=drift_scores,
            aggregate_score=aggregate_score,
            recent_rows=recent_rows,
            reference_rows=reference_rows,
            quality_flags=quality_flags,
            payload=payload,
        )
        _upsert_governance_state(
            conn,
            profile_name=args.profile,
            scope=scope,
            base_registry_status=base_status,
            health_state=health_state,
            operational_state=state_result["operational_state"],
            action=action,
            action_applied=bool(state_result["action_applied"]),
            weight_multiplier=float(state_result["weight_multiplier"]),
            restrict_live=bool(state_result["restrict_live"]),
            shadow_only=bool(state_result["shadow_only"]),
            disabled=bool(state_result["disabled"]),
            candidate_eligible=bool(challenger_eval.get("qualifies", False)),
            champion_eligible=bool(challenger_eval.get("qualifies", False)),
            aggregate_score=aggregate_score,
            reason_codes=reason_codes,
            quality_flags=quality_flags,
            payload=payload,
            policy_version=policy_version,
        )
        _upsert_challenger_eval(
            conn,
            profile_name=args.profile,
            scope=scope,
            challenger_eval=challenger_eval,
        )

        previous_operational_state = str(previous_state.get("governance_state", "") or _default_operational_state(base_status)) if previous_state else _default_operational_state(base_status)
        if action != "NONE" or previous_operational_state != state_result["operational_state"]:
            _insert_governance_action(
                conn,
                profile_name=args.profile,
                scope=scope,
                previous_state=previous_operational_state,
                new_state=state_result["operational_state"],
                action_kind=action,
                action_applied=bool(state_result["action_applied"]),
                reason_codes=reason_codes,
                payload=payload,
                policy_version=policy_version,
            )

        symbol_plugins[symbol].append(payload)
        actions.append(
            {
                "symbol": symbol,
                "plugin_name": plugin_name,
                "health_state": health_state,
                "governance_state": state_result["operational_state"],
                "recommended_governance_state": recommended_state,
                "action": action,
                "action_applied": bool(state_result["action_applied"]),
                "aggregate_risk_score": round(float(aggregate_score), 6),
                "reason_codes": reason_codes,
                "weight_multiplier": round(float(state_result["weight_multiplier"]), 6),
            }
        )

    commit_db(conn)
    report = build_drift_governance_report(conn, args.profile)
    status = {
        "schema_version": DRIFT_GOVERNANCE_SCHEMA_VERSION,
        "generated_at": isoformat_utc(),
        "profile_name": args.profile,
        "policy_version": policy_version,
        "action_mode": str(config.get("action_mode", "AUTO_APPLY_PROTECTIVE")),
        "symbol_count": len(symbol_plugins),
        "plugin_count": sum(len(items) for items in symbol_plugins.values()),
        "applied_action_count": sum(1 for item in actions if bool(item.get("action_applied", False))),
        "latest_actions": actions[:24],
        "artifacts": {
            "report_path": str(DRIFT_GOVERNANCE_REPORT_PATH),
            "status_path": str(DRIFT_GOVERNANCE_STATUS_PATH),
            "history_path": str(DRIFT_GOVERNANCE_HISTORY_PATH),
        },
    }
    json_dump(DRIFT_GOVERNANCE_STATUS_PATH, status)
    json_dump(DRIFT_GOVERNANCE_RUNTIME_SUMMARY_PATH, status)
    _append_history(
        DRIFT_GOVERNANCE_HISTORY_PATH,
        {
            "record_type": "snapshot",
            "generated_at": isoformat_utc(),
            "profile_name": args.profile,
            "status": status,
            "report_summary": {
                "symbol_count": int(report.get("symbol_count", 0)),
                "plugin_count": int(report.get("plugin_count", 0)),
                "health_counts": dict(report.get("health_counts", {})),
                "governance_counts": dict(report.get("governance_counts", {})),
                "action_counts": dict(report.get("action_counts", {})),
            },
        },
    )
    return {
        "schema_version": DRIFT_GOVERNANCE_SCHEMA_VERSION,
        "generated_at": isoformat_utc(),
        "profile_name": args.profile,
        "policy_version": policy_version,
        "action_mode": str(config.get("action_mode", "AUTO_APPLY_PROTECTIVE")),
        "symbols": [
            {
                "symbol": symbol,
                "plugin_count": len(items),
                "plugins": items,
            }
            for symbol, items in sorted(symbol_plugins.items())
        ],
        "actions": actions,
        "report": report,
        "status": status,
    }
