from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
import libsql

from .common import *


def _shadow_float(row: dict, key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default) or default)
    except Exception:
        return float(default)


def _shadow_int(row: dict, key: str, default: int = 0) -> int:
    try:
        return int(float(row.get(key, default) or default))
    except Exception:
        return int(default)


def iter_shadow_ledger_files() -> list[Path]:
    if not SHADOW_LEDGER_DIR.exists():
        return []
    return sorted(SHADOW_LEDGER_DIR.glob("fxai_shadow_*.tsv"))


def ingest_shadow_fleet_ledgers(conn: libsql.Connection,
                                profile_name: str) -> dict:
    files = iter_shadow_ledger_files()
    rows_ingested = 0
    symbols: set[str] = set()
    plugins: set[str] = set()
    summary_rows: list[dict] = []

    for path in files:
        if not path.exists() or path.stat().st_size <= 0:
            continue
        captured_at = int(path.stat().st_mtime)
        source_sha = testlab.sha256_path(path)
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            for raw in reader:
                symbol = str(raw.get("symbol", "") or "").strip().upper()
                plugin_name = str(raw.get("ai_name", "") or "").strip()
                if not symbol or not plugin_name:
                    continue
                row = {
                    "profile_name": profile_name,
                    "symbol": symbol,
                    "plugin_name": plugin_name,
                    "family_id": _shadow_int(raw, "family_id", 11),
                    "captured_at": captured_at,
                    "source_path": str(path),
                    "source_sha256": source_sha,
                    "meta_weight": _shadow_float(raw, "meta_weight"),
                    "reliability": _shadow_float(raw, "reliability"),
                    "global_edge": _shadow_float(raw, "global_edge"),
                    "context_edge": _shadow_float(raw, "context_edge"),
                    "context_regret": _shadow_float(raw, "context_regret"),
                    "portfolio_objective": _shadow_float(raw, "portfolio_objective"),
                    "portfolio_stability": _shadow_float(raw, "portfolio_stability"),
                    "portfolio_corr": _shadow_float(raw, "portfolio_corr"),
                    "portfolio_div": _shadow_float(raw, "portfolio_div"),
                    "route_value": _shadow_float(raw, "route_value"),
                    "route_regret": _shadow_float(raw, "route_regret"),
                    "route_counterfactual": _shadow_float(raw, "route_counterfactual"),
                    "shadow_score": _shadow_float(raw, "shadow_score"),
                    "regime_id": _shadow_int(raw, "regime_id", 0),
                    "horizon_minutes": _shadow_int(raw, "horizon_minutes", 5),
                    "obs_count": _shadow_int(raw, "obs", 0),
                    "policy_enter_prob": _shadow_float(raw, "policy_enter_prob"),
                    "policy_no_trade_prob": _shadow_float(raw, "policy_no_trade_prob"),
                    "policy_exit_prob": _shadow_float(raw, "policy_exit_prob"),
                    "policy_add_prob": _shadow_float(raw, "policy_add_prob"),
                    "policy_reduce_prob": _shadow_float(raw, "policy_reduce_prob"),
                    "policy_timeout_prob": _shadow_float(raw, "policy_timeout_prob"),
                    "policy_tighten_prob": _shadow_float(raw, "policy_tighten_prob"),
                    "policy_portfolio_fit": _shadow_float(raw, "policy_portfolio_fit"),
                    "policy_capital_efficiency": _shadow_float(raw, "policy_capital_efficiency"),
                    "policy_lifecycle_action": _shadow_int(raw, "policy_lifecycle_action", 0),
                    "portfolio_pressure": _shadow_float(raw, "portfolio_pressure"),
                    "control_plane_score": _shadow_float(raw, "control_plane_score"),
                    "portfolio_supervisor_score": _shadow_float(raw, "portfolio_supervisor_score"),
                    "payload_json": json.dumps(raw, indent=2, sort_keys=True),
                }
                conn.execute(
                    """
                    INSERT INTO shadow_fleet_observations(profile_name, symbol, plugin_name, family_id, captured_at,
                                                          source_path, source_sha256, meta_weight, reliability,
                                                          global_edge, context_edge, context_regret, portfolio_objective,
                                                          portfolio_stability, portfolio_corr, portfolio_div, route_value,
                                                          route_regret, route_counterfactual, shadow_score, regime_id,
                                                          horizon_minutes, obs_count, policy_enter_prob, policy_no_trade_prob,
                                                          policy_exit_prob, policy_add_prob, policy_reduce_prob,
                                                          policy_timeout_prob, policy_tighten_prob,
                                                          policy_portfolio_fit, policy_capital_efficiency,
                                                          policy_lifecycle_action,
                                                          portfolio_pressure, control_plane_score, portfolio_supervisor_score,
                                                          payload_json)
                    VALUES(:profile_name, :symbol, :plugin_name, :family_id, :captured_at,
                           :source_path, :source_sha256, :meta_weight, :reliability,
                           :global_edge, :context_edge, :context_regret, :portfolio_objective,
                           :portfolio_stability, :portfolio_corr, :portfolio_div, :route_value,
                           :route_regret, :route_counterfactual, :shadow_score, :regime_id,
                           :horizon_minutes, :obs_count, :policy_enter_prob, :policy_no_trade_prob,
                           :policy_exit_prob, :policy_add_prob, :policy_reduce_prob,
                           :policy_timeout_prob, :policy_tighten_prob,
                           :policy_portfolio_fit, :policy_capital_efficiency,
                           :policy_lifecycle_action,
                           :portfolio_pressure, :control_plane_score, :portfolio_supervisor_score,
                           :payload_json)
                    ON CONFLICT(profile_name, symbol, plugin_name, captured_at, source_sha256) DO UPDATE SET
                        family_id=excluded.family_id,
                        source_path=excluded.source_path,
                        meta_weight=excluded.meta_weight,
                        reliability=excluded.reliability,
                        global_edge=excluded.global_edge,
                        context_edge=excluded.context_edge,
                        context_regret=excluded.context_regret,
                        portfolio_objective=excluded.portfolio_objective,
                        portfolio_stability=excluded.portfolio_stability,
                        portfolio_corr=excluded.portfolio_corr,
                        portfolio_div=excluded.portfolio_div,
                        route_value=excluded.route_value,
                        route_regret=excluded.route_regret,
                        route_counterfactual=excluded.route_counterfactual,
                        shadow_score=excluded.shadow_score,
                        regime_id=excluded.regime_id,
                        horizon_minutes=excluded.horizon_minutes,
                        obs_count=excluded.obs_count,
                        policy_enter_prob=excluded.policy_enter_prob,
                        policy_no_trade_prob=excluded.policy_no_trade_prob,
                        policy_exit_prob=excluded.policy_exit_prob,
                        policy_add_prob=excluded.policy_add_prob,
                        policy_reduce_prob=excluded.policy_reduce_prob,
                        policy_timeout_prob=excluded.policy_timeout_prob,
                        policy_tighten_prob=excluded.policy_tighten_prob,
                        policy_portfolio_fit=excluded.policy_portfolio_fit,
                        policy_capital_efficiency=excluded.policy_capital_efficiency,
                        policy_lifecycle_action=excluded.policy_lifecycle_action,
                        portfolio_pressure=excluded.portfolio_pressure,
                        control_plane_score=excluded.control_plane_score,
                        portfolio_supervisor_score=excluded.portfolio_supervisor_score,
                        payload_json=excluded.payload_json
                    """,
                    row,
                )
                rows_ingested += 1
                symbols.add(symbol)
                plugins.add(plugin_name)
                summary_rows.append({
                    "symbol": symbol,
                    "plugin_name": plugin_name,
                    "shadow_score": row["shadow_score"],
                    "route_value": row["route_value"],
                    "route_regret": row["route_regret"],
                    "portfolio_objective": row["portfolio_objective"],
                    "portfolio_stability": row["portfolio_stability"],
                    "policy_enter_prob": row["policy_enter_prob"],
                    "policy_no_trade_prob": row["policy_no_trade_prob"],
                    "policy_add_prob": row["policy_add_prob"],
                    "policy_reduce_prob": row["policy_reduce_prob"],
                    "policy_timeout_prob": row["policy_timeout_prob"],
                    "portfolio_pressure": row["portfolio_pressure"],
                    "captured_at": captured_at,
                })
    commit_db(conn)

    out_dir = RESEARCH_DIR / safe_token(profile_name)
    ensure_dir(out_dir)
    summary_json = out_dir / "shadow_fleet_ingest.json"
    summary_md = out_dir / "shadow_fleet_ingest.md"
    summary_payload = {
        "profile": profile_name,
        "files": [str(path) for path in files],
        "rows_ingested": rows_ingested,
        "symbols": sorted(symbols),
        "plugin_count": len(plugins),
        "latest_rows": sorted(summary_rows, key=lambda item: (item["symbol"], -int(item["captured_at"]), -float(item["shadow_score"])))[:64],
    }
    summary_json.write_text(json.dumps(summary_payload, indent=2, sort_keys=True), encoding="utf-8")
    md_lines = [
        "# FXAI Shadow Fleet Ingest",
        "",
        f"profile: {profile_name}",
        f"rows_ingested: {rows_ingested}",
        f"symbols: {', '.join(sorted(symbols)) if symbols else '(none)'}",
        "",
    ]
    for item in summary_payload["latest_rows"][:24]:
        md_lines.append(
            f"- {item['symbol']} | {item['plugin_name']} | shadow {float(item['shadow_score']):.4f} | "
            f"route {float(item['route_value']):.4f} | regret {float(item['route_regret']):.4f} | "
            f"portfolio {float(item['portfolio_objective']):.4f} | stability {float(item['portfolio_stability']):.4f} | "
            f"enter {float(item['policy_enter_prob']):.4f} | add {float(item['policy_add_prob']):.4f} | "
            f"reduce {float(item['policy_reduce_prob']):.4f} | timeout {float(item['policy_timeout_prob']):.4f} | "
            f"no-trade {float(item['policy_no_trade_prob']):.4f}"
        )
    summary_md.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    return summary_payload


def latest_shadow_rows(conn: libsql.Connection,
                       profile_name: str) -> dict[tuple[str, str], dict]:
    sql = """
        SELECT s.*
          FROM shadow_fleet_observations s
         WHERE s.profile_name = ?
           AND NOT EXISTS (
                SELECT 1
                  FROM shadow_fleet_observations newer
                 WHERE newer.profile_name = s.profile_name
                   AND newer.symbol = s.symbol
                   AND newer.plugin_name = s.plugin_name
                   AND (
                        newer.captured_at > s.captured_at OR
                        (newer.captured_at = s.captured_at AND newer.id > s.id)
                   )
          )
    """
    rows = query_all(conn, sql, (profile_name,))
    return {
        (str(row["symbol"]), str(row["plugin_name"])): dict(row)
        for row in rows
    }


def symbol_shadow_summary(conn: libsql.Connection,
                          profile_name: str,
                          symbol: str) -> dict:
    rows = list(latest_shadow_rows(conn, profile_name).values())
    rows = [row for row in rows if str(row["symbol"]) == symbol]
    if not rows:
        return {
            "obs_count": 0,
            "mean_shadow_score": 0.0,
            "mean_route_value": 0.0,
            "mean_route_regret": 0.0,
            "mean_counterfactual": 0.0,
            "mean_portfolio_objective": 0.0,
            "mean_portfolio_stability": 0.0,
            "mean_portfolio_corr": 0.0,
            "mean_portfolio_div": 0.0,
            "mean_policy_enter_prob": 0.0,
            "mean_policy_no_trade_prob": 0.0,
            "mean_policy_exit_prob": 0.0,
            "mean_policy_add_prob": 0.0,
            "mean_policy_reduce_prob": 0.0,
            "mean_policy_timeout_prob": 0.0,
            "mean_policy_tighten_prob": 0.0,
            "mean_policy_portfolio_fit": 0.0,
            "mean_policy_capital_efficiency": 0.0,
            "mean_policy_lifecycle_action": 0.0,
            "mean_portfolio_pressure": 0.0,
            "mean_control_plane_score": 0.0,
            "mean_portfolio_supervisor_score": 0.0,
        }
    n = float(len(rows))
    return {
        "obs_count": int(len(rows)),
        "mean_shadow_score": sum(float(row["shadow_score"]) for row in rows) / n,
        "mean_route_value": sum(float(row["route_value"]) for row in rows) / n,
        "mean_route_regret": sum(float(row["route_regret"]) for row in rows) / n,
        "mean_counterfactual": sum(float(row["route_counterfactual"]) for row in rows) / n,
        "mean_portfolio_objective": sum(float(row["portfolio_objective"]) for row in rows) / n,
        "mean_portfolio_stability": sum(float(row["portfolio_stability"]) for row in rows) / n,
        "mean_portfolio_corr": sum(float(row["portfolio_corr"]) for row in rows) / n,
        "mean_portfolio_div": sum(float(row["portfolio_div"]) for row in rows) / n,
        "mean_policy_enter_prob": sum(float(row["policy_enter_prob"]) for row in rows) / n,
        "mean_policy_no_trade_prob": sum(float(row["policy_no_trade_prob"]) for row in rows) / n,
        "mean_policy_exit_prob": sum(float(row["policy_exit_prob"]) for row in rows) / n,
        "mean_policy_add_prob": sum(float(row["policy_add_prob"]) for row in rows) / n,
        "mean_policy_reduce_prob": sum(float(row["policy_reduce_prob"]) for row in rows) / n,
        "mean_policy_timeout_prob": sum(float(row["policy_timeout_prob"]) for row in rows) / n,
        "mean_policy_tighten_prob": sum(float(row["policy_tighten_prob"]) for row in rows) / n,
        "mean_policy_portfolio_fit": sum(float(row["policy_portfolio_fit"]) for row in rows) / n,
        "mean_policy_capital_efficiency": sum(float(row["policy_capital_efficiency"]) for row in rows) / n,
        "mean_policy_lifecycle_action": sum(float(row["policy_lifecycle_action"]) for row in rows) / n,
        "mean_portfolio_pressure": sum(float(row["portfolio_pressure"]) for row in rows) / n,
        "mean_control_plane_score": sum(float(row["control_plane_score"]) for row in rows) / n,
        "mean_portfolio_supervisor_score": sum(float(row["portfolio_supervisor_score"]) for row in rows) / n,
    }
