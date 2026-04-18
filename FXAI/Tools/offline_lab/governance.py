from __future__ import annotations

import json
from pathlib import Path
import libsql

from .common import *
from .drift_governance import run_drift_governance_cycle
from .shadow_fleet import symbol_shadow_summary
from .supervisor_service import (
    write_supervisor_command_artifacts,
    write_supervisor_service_artifacts,
)
from .world_simulator import write_world_model_artifacts


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(v)))


def _unlink_if_exists(raw_path: str) -> None:
    path_text = str(raw_path or "").strip()
    if not path_text:
        return
    try:
        path = Path(path_text)
        if path.exists() and path.is_file():
            path.unlink()
    except Exception:
        pass


def write_portfolio_supervisor_profile(conn: libsql.Connection,
                                       args) -> dict:
    deployments = query_all(
        conn,
        """
        SELECT *
          FROM live_deployment_profiles
         WHERE profile_name = ?
         ORDER BY created_at DESC
        """,
        (args.profile,),
    )
    if not deployments:
        stale_rows = query_all(
            conn,
            "SELECT artifact_path FROM portfolio_supervisor_profiles WHERE profile_name = ?",
            (args.profile,),
        )
        for row in stale_rows:
            _unlink_if_exists(str(row["artifact_path"] or ""))
        conn.execute(
            "DELETE FROM portfolio_supervisor_profiles WHERE profile_name = ?",
            (args.profile,),
        )
        out_dir = RESEARCH_DIR / safe_token(args.profile)
        stale_json = out_dir / "portfolio_supervisor.json"
        if stale_json.exists():
            stale_json.unlink()
        commit_db(conn)
        return {}

    mean_shadow = 0.0
    mean_pressure = 0.0
    mean_no_trade = 0.0
    mean_route_regret = 0.0
    mean_corr = 0.0
    mean_div = 0.0
    total = 0.0
    symbol_payloads: list[dict] = []
    for row in deployments:
        symbol = str(row["symbol"])
        shadow = symbol_shadow_summary(conn, args.profile, symbol)
        weight = max(float(shadow.get("obs_count", 0) or 0), 1.0)
        mean_shadow += float(shadow.get("mean_shadow_score", 0.0)) * weight
        mean_pressure += float(shadow.get("mean_portfolio_pressure", 0.0)) * weight
        mean_no_trade += float(shadow.get("mean_policy_no_trade_prob", 0.0)) * weight
        mean_route_regret += float(shadow.get("mean_route_regret", 0.0)) * weight
        mean_corr += float(shadow.get("mean_portfolio_corr", 0.0)) * weight
        mean_div += float(shadow.get("mean_portfolio_div", 0.0)) * weight
        total += weight
        symbol_payloads.append({
            "symbol": symbol,
            "shadow_summary": shadow,
            "deployment": row,
        })

    total = max(total, 1.0)
    mean_shadow /= total
    mean_pressure /= total
    mean_no_trade /= total
    mean_route_regret /= total
    mean_corr /= total
    mean_div /= total

    payload = {
        "profile_name": args.profile,
        "gross_budget_bias": _clamp(0.96 + 0.16 * mean_div - 0.10 * mean_pressure, 0.40, 1.60),
        "correlated_budget_bias": _clamp(0.92 + 0.12 * mean_div - 0.16 * mean_corr - 0.08 * mean_pressure, 0.40, 1.60),
        "directional_budget_bias": _clamp(0.92 + 0.10 * mean_div - 0.12 * mean_no_trade - 0.08 * mean_pressure, 0.40, 1.60),
        "capital_risk_cap_pct": _clamp(1.10 + 0.80 * mean_div + 0.40 * mean_shadow - 0.30 * mean_pressure, 0.10, 10.0),
        "macro_overlap_cap": _clamp(0.86 + 0.18 * mean_shadow - 0.10 * mean_route_regret, 0.10, 2.0),
        "concentration_cap": _clamp(0.72 + 0.20 * mean_div - 0.08 * mean_corr, 0.10, 2.0),
        "supervisor_weight": _clamp(0.28 + 0.20 * mean_pressure + 0.18 * mean_no_trade + 0.12 * mean_route_regret, 0.0, 1.0),
        "hard_block_score": _clamp(0.92 + 0.40 * mean_pressure + 0.20 * mean_no_trade - 0.12 * mean_shadow, 0.20, 3.0),
        "policy_enter_floor": _clamp(0.36 + 0.08 * mean_pressure + 0.10 * mean_route_regret, 0.10, 0.95),
        "policy_no_trade_ceiling": _clamp(0.68 + 0.12 * mean_no_trade + 0.08 * mean_pressure, 0.10, 0.99),
        "symbols": symbol_payloads,
    }

    ensure_dir(COMMON_PROMOTION_DIR)
    out_dir = RESEARCH_DIR / safe_token(args.profile)
    ensure_dir(out_dir)
    tsv_path = COMMON_PROMOTION_DIR / "fxai_portfolio_supervisor.tsv"
    tsv_path.write_text(
        "".join(
            f"{key}\t{value:.6f}\n" if isinstance(value, float) else f"{key}\t{value}\n"
            for key, value in [
                ("profile_name", args.profile),
                ("gross_budget_bias", payload["gross_budget_bias"]),
                ("correlated_budget_bias", payload["correlated_budget_bias"]),
                ("directional_budget_bias", payload["directional_budget_bias"]),
                ("capital_risk_cap_pct", payload["capital_risk_cap_pct"]),
                ("macro_overlap_cap", payload["macro_overlap_cap"]),
                ("concentration_cap", payload["concentration_cap"]),
                ("supervisor_weight", payload["supervisor_weight"]),
                ("hard_block_score", payload["hard_block_score"]),
                ("policy_enter_floor", payload["policy_enter_floor"]),
                ("policy_no_trade_ceiling", payload["policy_no_trade_ceiling"]),
            ]
        ),
        encoding="utf-8",
    )
    json_path = out_dir / "portfolio_supervisor.json"
    json_path.write_text(
        json.dumps(portableize_payload_paths(payload), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    sha = testlab.sha256_path(tsv_path)
    conn.execute(
        """
        INSERT INTO portfolio_supervisor_profiles(profile_name, artifact_path, artifact_sha256,
                                                  gross_budget_bias, correlated_budget_bias, directional_budget_bias,
                                                  capital_risk_cap_pct, macro_overlap_cap, concentration_cap,
                                                  supervisor_weight, hard_block_score, policy_enter_floor,
                                                  policy_no_trade_ceiling, payload_json, created_at)
        VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(profile_name) DO UPDATE SET
            artifact_path=excluded.artifact_path,
            artifact_sha256=excluded.artifact_sha256,
            gross_budget_bias=excluded.gross_budget_bias,
            correlated_budget_bias=excluded.correlated_budget_bias,
            directional_budget_bias=excluded.directional_budget_bias,
            capital_risk_cap_pct=excluded.capital_risk_cap_pct,
            macro_overlap_cap=excluded.macro_overlap_cap,
            concentration_cap=excluded.concentration_cap,
            supervisor_weight=excluded.supervisor_weight,
            hard_block_score=excluded.hard_block_score,
            policy_enter_floor=excluded.policy_enter_floor,
            policy_no_trade_ceiling=excluded.policy_no_trade_ceiling,
            payload_json=excluded.payload_json,
            created_at=excluded.created_at
        """,
        (
            args.profile,
            str(tsv_path),
            sha,
            payload["gross_budget_bias"],
            payload["correlated_budget_bias"],
            payload["directional_budget_bias"],
            payload["capital_risk_cap_pct"],
            payload["macro_overlap_cap"],
            payload["concentration_cap"],
            payload["supervisor_weight"],
            payload["hard_block_score"],
            payload["policy_enter_floor"],
            payload["policy_no_trade_ceiling"],
            json.dumps(payload, indent=2, sort_keys=True),
            now_unix(),
        ),
    )
    commit_db(conn)
    return {
        "artifact_path": str(tsv_path),
        "artifact_sha256": sha,
        "profile_name": args.profile,
    }


def write_world_simulator_plans(conn: libsql.Connection,
                                args) -> list[dict]:
    symbols = [
        str(row["symbol"])
        for row in query_all(
            conn,
            "SELECT DISTINCT symbol FROM live_deployment_profiles WHERE profile_name = ? ORDER BY symbol",
            (args.profile,),
        )
    ]
    out_dir = RESEARCH_DIR / safe_token(args.profile)
    ensure_dir(out_dir)
    ensure_dir(COMMON_PROMOTION_DIR)
    world_models = {
        item["symbol"]: item["payload"]
        for item in write_world_model_artifacts(conn, args, symbols)
    }
    plans: list[dict] = []
    created_at = now_unix()
    active_symbols = set(symbols)

    for symbol in symbols:
        shadow = symbol_shadow_summary(conn, args.profile, symbol)
        model = dict(world_models.get(symbol, {}))
        redteam = query_one(
            conn,
            """
            SELECT plan_json, weak_scenarios_json
              FROM redteam_cycles
             WHERE profile_name = ? AND symbol = ?
             ORDER BY created_at DESC
            LIMIT 1
            """,
            (args.profile, symbol),
        )
        weak_scenarios = []
        if redteam is not None:
            try:
                weak_scenarios = json.loads(redteam["weak_scenarios_json"] or "[]")
            except Exception:
                weak_scenarios = []
        weak_names = {str(item.get("scenario", "")) for item in weak_scenarios}
        macro_flag = 1.0 if "market_macro_event" in weak_names else 0.0
        adversarial_flag = 1.0 if "market_adversarial" in weak_names else 0.0
        spread_flag = 1.0 if "market_spread_shock" in weak_names else 0.0
        session_flag = 1.0 if "market_session_edges" in weak_names else 0.0
        walkforward_flag = 1.0 if "market_walkforward" in weak_names else 0.0

        payload = {
            "profile_name": args.profile,
            "symbol": symbol,
            "sigma_scale": _clamp(
                0.62 * float(model.get("sigma_scale", 1.0)) +
                0.20 * (1.0 + 0.30 * adversarial_flag + 0.20 * walkforward_flag) +
                0.18 * float(shadow.get("mean_route_regret", 0.0)),
                0.50,
                3.00,
            ),
            "drift_bias": _clamp(
                0.70 * float(model.get("drift_bias", 0.0)) +
                0.00004 * float(shadow.get("mean_route_value", 0.0)) -
                0.00005 * float(shadow.get("mean_policy_no_trade_prob", 0.0)),
                -0.001,
                0.001,
            ),
            "spread_scale": _clamp(
                0.62 * float(model.get("spread_scale", 1.0)) +
                0.20 * (1.0 + 0.25 * spread_flag + 0.18 * session_flag) +
                0.12 * float(shadow.get("mean_portfolio_pressure", 0.0)),
                0.50,
                4.00,
            ),
            "gap_prob": _clamp(
                0.65 * float(model.get("gap_prob", 0.0)) +
                0.02 * adversarial_flag + 0.02 * walkforward_flag +
                0.05 * float(shadow.get("mean_route_regret", 0.0)),
                0.0,
                0.30,
            ),
            "gap_scale": _clamp(
                0.55 * float(model.get("gap_scale", 1.2)) +
                0.45 * (1.2 + 1.8 * adversarial_flag + 1.5 * float(shadow.get("mean_portfolio_pressure", 0.0))),
                0.0,
                8.0,
            ),
            "flip_prob": _clamp(
                0.65 * float(model.get("flip_prob", 0.0)) +
                0.04 * adversarial_flag + 0.03 * walkforward_flag +
                0.06 * float(shadow.get("mean_policy_no_trade_prob", 0.0)),
                0.0,
                0.50,
            ),
            "context_corr_bias": _clamp(
                0.55 * float(model.get("context_corr_bias", 0.0)) +
                0.35 * float(shadow.get("mean_portfolio_div", 0.0)) -
                0.30 * float(shadow.get("mean_portfolio_corr", 0.0)),
                -1.0,
                1.0,
            ),
            "liquidity_stress": _clamp(
                0.58 * float(model.get("liquidity_stress", 0.0)) +
                0.30 * spread_flag + 0.22 * session_flag +
                0.25 * float(shadow.get("mean_portfolio_supervisor_score", 0.0)),
                0.0,
                3.0,
            ),
            "macro_focus": _clamp(
                0.58 * float(model.get("macro_focus", 0.0)) +
                0.25 * macro_flag + 0.15 * float(shadow.get("mean_policy_no_trade_prob", 0.0)),
                0.0,
                1.5,
            ),
            "session_edge_focus": _clamp(
                0.75 * float(model.get("session_edge_focus", 0.0)) + 0.25 * session_flag,
                0.0,
                1.5,
            ),
            "trend_persistence": _clamp(
                0.70 * float(model.get("trend_persistence", 0.5)) + 0.15 * (1.0 - adversarial_flag) + 0.15 * walkforward_flag,
                0.0,
                1.0,
            ),
            "shock_memory": _clamp(
                0.75 * float(model.get("shock_memory", 0.0)) + 0.25 * adversarial_flag,
                0.0,
                1.0,
            ),
            "recovery_bias": _clamp(
                0.75 * float(model.get("recovery_bias", 0.0)) - 0.10 * float(shadow.get("mean_route_regret", 0.0)),
                -1.0,
                1.0,
            ),
            "spread_shock_prob": _clamp(
                0.72 * float(model.get("spread_shock_prob", 0.0)) + 0.28 * spread_flag,
                0.0,
                0.50,
            ),
            "spread_shock_scale": _clamp(
                0.72 * float(model.get("spread_shock_scale", 1.0)) + 0.28 * (1.0 + spread_flag),
                1.0,
                8.0,
            ),
            "regime_transition_burst": _clamp(
                0.70 * float(model.get("regime_transition_burst", 0.0)) +
                0.18 * adversarial_flag + 0.12 * walkforward_flag,
                0.0,
                1.0,
            ),
            "transition_entropy": _clamp(
                0.72 * float(model.get("transition_entropy", 0.0)) +
                0.14 * adversarial_flag + 0.10 * walkforward_flag,
                0.0,
                1.0,
            ),
            "mean_revert_bias": _clamp(
                0.72 * float(model.get("mean_revert_bias", 0.0)) +
                0.16 * adversarial_flag + 0.12 * float(shadow.get("mean_policy_no_trade_prob", 0.0)),
                0.0,
                1.0,
            ),
            "vol_cluster_bias": _clamp(
                0.72 * float(model.get("vol_cluster_bias", 0.0)) +
                0.16 * spread_flag + 0.12 * float(shadow.get("mean_portfolio_pressure", 0.0)),
                0.0,
                1.0,
            ),
            "shock_decay": _clamp(
                0.72 * float(model.get("shock_decay", 0.6)) +
                0.16 * (1.0 - spread_flag) + 0.12 * walkforward_flag,
                0.0,
                1.5,
            ),
            "asia_sigma_scale": _clamp(float(model.get("asia_sigma_scale", 1.0)), 0.50, 3.00),
            "london_sigma_scale": _clamp(float(model.get("london_sigma_scale", 1.0)), 0.50, 3.00),
            "newyork_sigma_scale": _clamp(float(model.get("newyork_sigma_scale", 1.0)), 0.50, 3.00),
            "asia_spread_scale": _clamp(float(model.get("asia_spread_scale", 1.0)), 0.50, 4.00),
            "london_spread_scale": _clamp(float(model.get("london_spread_scale", 1.0)), 0.50, 4.00),
            "newyork_spread_scale": _clamp(float(model.get("newyork_spread_scale", 1.0)), 0.50, 4.00),
            "weak_scenarios": weak_scenarios,
            "shadow_summary": shadow,
            "world_model": model,
        }

        tsv_path = COMMON_PROMOTION_DIR / f"fxai_world_plan_{safe_token(symbol)}.tsv"
        tsv_path.write_text(
            "".join(
                f"{key}\t{value:.6f}\n" if isinstance(value, float) else f"{key}\t{value}\n"
                for key, value in [
                    ("profile_name", args.profile),
                    ("symbol", symbol),
                    ("sigma_scale", payload["sigma_scale"]),
                    ("drift_bias", payload["drift_bias"]),
                    ("spread_scale", payload["spread_scale"]),
                    ("gap_prob", payload["gap_prob"]),
                    ("gap_scale", payload["gap_scale"]),
                    ("flip_prob", payload["flip_prob"]),
                    ("context_corr_bias", payload["context_corr_bias"]),
                    ("liquidity_stress", payload["liquidity_stress"]),
                    ("macro_focus", payload["macro_focus"]),
                    ("session_edge_focus", payload["session_edge_focus"]),
                    ("trend_persistence", payload["trend_persistence"]),
                    ("shock_memory", payload["shock_memory"]),
                    ("recovery_bias", payload["recovery_bias"]),
                    ("spread_shock_prob", payload["spread_shock_prob"]),
                    ("spread_shock_scale", payload["spread_shock_scale"]),
                    ("regime_transition_burst", payload["regime_transition_burst"]),
                    ("transition_entropy", payload["transition_entropy"]),
                    ("mean_revert_bias", payload["mean_revert_bias"]),
                    ("vol_cluster_bias", payload["vol_cluster_bias"]),
                    ("shock_decay", payload["shock_decay"]),
                    ("asia_sigma_scale", payload["asia_sigma_scale"]),
                    ("london_sigma_scale", payload["london_sigma_scale"]),
                    ("newyork_sigma_scale", payload["newyork_sigma_scale"]),
                    ("asia_spread_scale", payload["asia_spread_scale"]),
                    ("london_spread_scale", payload["london_spread_scale"]),
                    ("newyork_spread_scale", payload["newyork_spread_scale"]),
                ]
            ),
            encoding="utf-8",
        )
        sha = testlab.sha256_path(tsv_path)
        json_path = out_dir / f"world_simulator_{safe_token(symbol)}.json"
        json_path.write_text(
            json.dumps(portableize_payload_paths(payload), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        conn.execute(
            """
            INSERT INTO world_simulator_plans(profile_name, symbol, artifact_path, artifact_sha256,
                                              sigma_scale, drift_bias, spread_scale, gap_prob, gap_scale, flip_prob,
                                              context_corr_bias, liquidity_stress, macro_focus, payload_json, created_at)
            VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(profile_name, symbol) DO UPDATE SET
                artifact_path=excluded.artifact_path,
                artifact_sha256=excluded.artifact_sha256,
                sigma_scale=excluded.sigma_scale,
                drift_bias=excluded.drift_bias,
                spread_scale=excluded.spread_scale,
                gap_prob=excluded.gap_prob,
                gap_scale=excluded.gap_scale,
                flip_prob=excluded.flip_prob,
                context_corr_bias=excluded.context_corr_bias,
                liquidity_stress=excluded.liquidity_stress,
                macro_focus=excluded.macro_focus,
                payload_json=excluded.payload_json,
                created_at=excluded.created_at
            """,
            (
                args.profile,
                symbol,
                str(tsv_path),
                sha,
                payload["sigma_scale"],
                payload["drift_bias"],
                payload["spread_scale"],
                payload["gap_prob"],
                payload["gap_scale"],
                payload["flip_prob"],
                payload["context_corr_bias"],
                payload["liquidity_stress"],
                payload["macro_focus"],
                json.dumps(payload, indent=2, sort_keys=True),
                created_at,
            ),
        )
        plans.append({"symbol": symbol, "artifact_path": str(tsv_path), "artifact_sha256": sha})

    stale_rows = query_all(
        conn,
        """
        SELECT symbol, artifact_path
          FROM world_simulator_plans
         WHERE profile_name = ?
        """,
        (args.profile,),
    )
    for row in stale_rows:
        symbol = str(row["symbol"])
        if symbol in active_symbols:
            continue
        _unlink_if_exists(str(row["artifact_path"] or ""))
        stale_json = out_dir / f"world_simulator_{safe_token(symbol)}.json"
        if stale_json.exists():
            stale_json.unlink()
        conn.execute(
            "DELETE FROM world_simulator_plans WHERE profile_name = ? AND symbol = ?",
            (args.profile, symbol),
        )

    commit_db(conn)
    summary_path = out_dir / "world_simulator_plans.json"
    summary_path.write_text(
        json.dumps(portableize_payload_paths(plans), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return plans


def run_autonomous_governance(conn: libsql.Connection,
                              args,
                              cycle_group_key: str = "") -> dict:
    created_at = now_unix()
    drift_payload = run_drift_governance_cycle(conn, args, cycle_group_key)
    decisions: list[dict] = [
        {
            "symbol": str(item.get("symbol", "")),
            "plugin_name": str(item.get("plugin_name", "")),
            "status": str(item.get("governance_state", "")),
            "action": str(item.get("action", "NONE")),
            "notes": list(item.get("reason_codes", [])),
            "action_applied": bool(item.get("action_applied", False)),
            "aggregate_risk_score": float(item.get("aggregate_risk_score", 0.0) or 0.0),
        }
        for item in list(drift_payload.get("actions", []))
    ]
    promoted = sum(1 for item in decisions if str(item.get("action", "")).upper() in {"PROMOTE", "PROMOTION_REVIEW"})
    challengers = sum(
        1
        for symbol_payload in list(drift_payload.get("symbols", []))
        for item in list(symbol_payload.get("plugins", []))
        if str(dict(item).get("challenger_evaluation", {}).get("eligibility_state", "")).upper() in {"QUALIFIED", "INSUFFICIENT"}
    )
    review = sum(1 for item in decisions if str(item.get("action", "")).upper() in {"DOWNWEIGHT", "DEMOTE", "PROMOTION_REVIEW"})
    rollback = sum(1 for item in decisions if str(item.get("action", "")).upper() == "ROLLBACK")

    for item in decisions:
        symbol = str(item["symbol"])
        plugin_name = str(item["plugin_name"])
        shadow = symbol_shadow_summary(conn, args.profile, symbol)
        conn.execute(
            """
            UPDATE champion_registry
               SET reviewed_at = ?, notes = ?
             WHERE profile_name = ? AND symbol = ? AND plugin_name = ?
            """,
            (
                created_at,
                json.dumps(
                    {
                        "action": item["action"],
                        "notes": item["notes"],
                        "action_applied": bool(item.get("action_applied", False)),
                        "aggregate_risk_score": float(item.get("aggregate_risk_score", 0.0) or 0.0),
                        "shadow": shadow,
                    },
                    indent=2,
                    sort_keys=True,
                ),
                args.profile,
                symbol,
                plugin_name,
            ),
        )

    supervisor = write_portfolio_supervisor_profile(conn, args)
    supervisor_service = write_supervisor_service_artifacts(conn, args)
    supervisor_commands = write_supervisor_command_artifacts(conn, args)
    world_plans = write_world_simulator_plans(conn, args)
    artifact_dir = RESEARCH_DIR / safe_token(args.profile)
    ensure_dir(artifact_dir)
    payload = {
        "profile_name": args.profile,
        "cycle_group_key": cycle_group_key,
        "promoted_count": promoted,
        "challenger_count": challengers,
        "review_count": review,
        "rollback_count": rollback,
        "decisions": decisions,
        "drift_governance": {
            "action_mode": str(drift_payload.get("action_mode", "")),
            "report_path": str(dict(drift_payload.get("status", {})).get("artifacts", {}).get("report_path", "")),
            "status_path": str(dict(drift_payload.get("status", {})).get("artifacts", {}).get("status_path", "")),
            "symbol_count": int(dict(drift_payload.get("status", {})).get("symbol_count", 0) or 0),
            "plugin_count": int(dict(drift_payload.get("status", {})).get("plugin_count", 0) or 0),
            "applied_action_count": int(dict(drift_payload.get("status", {})).get("applied_action_count", 0) or 0),
        },
        "portfolio_supervisor": supervisor,
        "supervisor_service": supervisor_service,
        "supervisor_commands": supervisor_commands,
        "world_plans": world_plans,
    }
    json_path = artifact_dir / "autonomous_governance.json"
    md_path = artifact_dir / "autonomous_governance.md"
    json_path.write_text(json.dumps(portableize_payload_paths(payload), indent=2, sort_keys=True), encoding="utf-8")
    md_lines = [
        "# FXAI Autonomous Governance",
        "",
        f"profile: {args.profile}",
        f"cycle_group_key: {cycle_group_key or '(none)'}",
        f"promoted_count: {promoted}",
        f"challenger_count: {challengers}",
        f"review_count: {review}",
        f"rollback_count: {rollback}",
        "",
    ]
    for item in decisions[:64]:
        md_lines.append(f"- {item['symbol']} | {item['plugin_name']} | {item['status']} -> {item['action']}")
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    conn.execute(
        """
        INSERT INTO autonomous_governance_runs(profile_name, cycle_group_key, governance_status, promoted_count,
                                               challenger_count, review_count, rollback_count, artifact_dir,
                                               payload_json, created_at)
        VALUES(?, ?, 'ok', ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            args.profile,
            cycle_group_key,
            promoted,
            challengers,
            review,
            rollback,
            str(artifact_dir),
            json.dumps(payload, indent=2, sort_keys=True),
            created_at,
        ),
    )
    commit_db(conn)
    return payload
