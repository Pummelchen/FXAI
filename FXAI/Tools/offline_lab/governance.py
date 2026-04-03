from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from .common import *
from .shadow_fleet import symbol_shadow_summary


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(v)))


def write_portfolio_supervisor_profile(conn: sqlite3.Connection,
                                       args) -> dict:
    rows = conn.execute(
        """
        SELECT *
          FROM live_deployment_profiles
         WHERE profile_name = ?
         ORDER BY created_at DESC
        """,
        (args.profile,),
    ).fetchall()
    deployments = [dict(row) for row in rows]
    if not deployments:
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
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
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
    conn.commit()
    return {
        "artifact_path": str(tsv_path),
        "artifact_sha256": sha,
        "profile_name": args.profile,
    }


def write_world_simulator_plans(conn: sqlite3.Connection,
                                args) -> list[dict]:
    symbols = [
        str(row["symbol"])
        for row in conn.execute(
            "SELECT DISTINCT symbol FROM live_deployment_profiles WHERE profile_name = ? ORDER BY symbol",
            (args.profile,),
        ).fetchall()
    ]
    out_dir = RESEARCH_DIR / safe_token(args.profile)
    ensure_dir(out_dir)
    ensure_dir(COMMON_PROMOTION_DIR)
    plans: list[dict] = []
    created_at = now_unix()

    for symbol in symbols:
        shadow = symbol_shadow_summary(conn, args.profile, symbol)
        redteam = conn.execute(
            """
            SELECT plan_json, weak_scenarios_json
              FROM redteam_cycles
             WHERE profile_name = ? AND symbol = ?
             ORDER BY created_at DESC
             LIMIT 1
            """,
            (args.profile, symbol),
        ).fetchone()
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
                1.0 + 0.30 * adversarial_flag + 0.20 * walkforward_flag +
                0.18 * float(shadow.get("mean_route_regret", 0.0)),
                0.50,
                3.00,
            ),
            "drift_bias": _clamp(
                0.00004 * float(shadow.get("mean_route_value", 0.0)) -
                0.00005 * float(shadow.get("mean_policy_no_trade_prob", 0.0)),
                -0.001,
                0.001,
            ),
            "spread_scale": _clamp(
                1.0 + 0.25 * spread_flag + 0.18 * session_flag +
                0.12 * float(shadow.get("mean_portfolio_pressure", 0.0)),
                0.50,
                4.00,
            ),
            "gap_prob": _clamp(
                0.02 * adversarial_flag + 0.02 * walkforward_flag +
                0.05 * float(shadow.get("mean_route_regret", 0.0)),
                0.0,
                0.30,
            ),
            "gap_scale": _clamp(
                1.2 + 1.8 * adversarial_flag + 1.5 * float(shadow.get("mean_portfolio_pressure", 0.0)),
                0.0,
                8.0,
            ),
            "flip_prob": _clamp(
                0.04 * adversarial_flag + 0.03 * walkforward_flag +
                0.06 * float(shadow.get("mean_policy_no_trade_prob", 0.0)),
                0.0,
                0.50,
            ),
            "context_corr_bias": _clamp(
                0.35 * float(shadow.get("mean_portfolio_div", 0.0)) -
                0.30 * float(shadow.get("mean_portfolio_corr", 0.0)),
                -1.0,
                1.0,
            ),
            "liquidity_stress": _clamp(
                0.30 * spread_flag + 0.22 * session_flag +
                0.25 * float(shadow.get("mean_portfolio_supervisor_score", 0.0)),
                0.0,
                3.0,
            ),
            "macro_focus": _clamp(
                0.25 * macro_flag + 0.15 * float(shadow.get("mean_policy_no_trade_prob", 0.0)),
                0.0,
                1.5,
            ),
            "weak_scenarios": weak_scenarios,
            "shadow_summary": shadow,
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
                ]
            ),
            encoding="utf-8",
        )
        sha = testlab.sha256_path(tsv_path)
        json_path = out_dir / f"world_simulator_{safe_token(symbol)}.json"
        json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
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

    conn.commit()
    summary_path = out_dir / "world_simulator_plans.json"
    summary_path.write_text(json.dumps(plans, indent=2, sort_keys=True), encoding="utf-8")
    return plans


def run_autonomous_governance(conn: sqlite3.Connection,
                              args,
                              cycle_group_key: str = "") -> dict:
    champion_rows = conn.execute(
        """
        SELECT *
          FROM champion_registry
         WHERE profile_name = ?
         ORDER BY symbol, plugin_name
        """,
        (args.profile,),
    ).fetchall()
    promoted = 0
    challengers = 0
    review = 0
    rollback = 0
    decisions: list[dict] = []
    created_at = now_unix()
    for row in champion_rows:
        item = dict(row)
        symbol = str(item["symbol"])
        shadow = symbol_shadow_summary(conn, args.profile, symbol)
        action = "keep"
        notes = []
        if float(shadow.get("mean_shadow_score", 0.0)) < -0.10 or float(shadow.get("mean_portfolio_supervisor_score", 0.0)) > 0.95:
            action = "review"
            review += 1
            notes.append("shadow weakness or supervisor stress")
        if item["status"] == "challenger":
            challengers += 1
            margin = float(item.get("challenger_score", 0.0)) - float(item.get("champion_score", 0.0))
            if margin > 0.50 and float(shadow.get("mean_shadow_score", 0.0)) > 0.05:
                action = "promote"
                promoted += 1
                notes.append("challenger margin with positive shadow score")
        if item["status"] == "champion" and float(shadow.get("mean_policy_no_trade_prob", 0.0)) > 0.82:
            action = "rollback_watch"
            rollback += 1
            notes.append("live no-trade dominance")

        conn.execute(
            """
            UPDATE champion_registry
               SET reviewed_at = ?, notes = ?
             WHERE profile_name = ? AND symbol = ? AND plugin_name = ?
            """,
            (
                created_at,
                json.dumps({"action": action, "notes": notes, "shadow": shadow}, indent=2, sort_keys=True),
                args.profile,
                symbol,
                str(item["plugin_name"]),
            ),
        )
        decisions.append({
            "symbol": symbol,
            "plugin_name": str(item["plugin_name"]),
            "status": str(item["status"]),
            "action": action,
            "notes": notes,
        })

    supervisor = write_portfolio_supervisor_profile(conn, args)
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
        "portfolio_supervisor": supervisor,
        "world_plans": world_plans,
    }
    json_path = artifact_dir / "autonomous_governance.json"
    md_path = artifact_dir / "autonomous_governance.md"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
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
    conn.commit()
    return payload
