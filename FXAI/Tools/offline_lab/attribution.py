from __future__ import annotations

import json
from pathlib import Path
import libsql

from .common import *
from .mode import resolve_runtime_mode
from .shadow_fleet import latest_shadow_rows, symbol_shadow_summary


FEATURE_GROUPS: list[tuple[int, str]] = [
    (0, "price"),
    (1, "multi_timeframe"),
    (2, "volatility"),
    (3, "time_calendar"),
    (4, "context"),
    (5, "cost"),
    (6, "microstructure"),
    (7, "filters"),
]


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(v)))


def _safe_json(raw: str, default):
    try:
        return json.loads(raw or "")
    except Exception:
        return default


def _latest_family_rows(conn: libsql.Connection,
                        profile_name: str,
                        symbol: str) -> dict[int, dict]:
    rows = query_all(
        conn,
        """
        SELECT *
          FROM family_scorecards
         WHERE profile_name = ? AND symbol = ?
        ORDER BY family_id ASC, created_at DESC
        """,
        (profile_name, symbol),
    )
    out: dict[int, dict] = {}
    for row in rows:
        family_id = int(row["family_id"])
        if family_id not in out:
            out[family_id] = dict(row)
    return out


def _champion_rows(conn: libsql.Connection,
                   profile_name: str,
                   symbol: str) -> list[dict]:
    return query_all(
        conn,
        """
        SELECT *
          FROM champion_registry
         WHERE profile_name = ? AND symbol = ?
         ORDER BY
               CASE status WHEN 'champion' THEN 0 WHEN 'challenger' THEN 1 ELSE 2 END,
               champion_score DESC,
               challenger_score DESC,
               plugin_name ASC
        """,
        (profile_name, symbol),
    )


def _latest_plugin_shadow_rows(conn: libsql.Connection,
                               profile_name: str,
                               symbol: str) -> dict[str, dict]:
    rows = latest_shadow_rows(conn, profile_name)
    out: dict[str, dict] = {}
    for (row_symbol, plugin_name), row in rows.items():
        if str(row_symbol) != symbol:
            continue
        out[str(plugin_name)] = dict(row)
    return out


def _feature_group_weights(conn: libsql.Connection,
                           profile_name: str,
                           symbol: str,
                           shadow: dict) -> dict[str, float]:
    baseline_rows = query_all(
        conn,
        """
        SELECT plugin_name, MAX(score) AS best_score
          FROM tuning_runs
         WHERE profile_name = ? AND symbol = ? AND status = 'ok' AND experiment_name = 'baseline_all'
         GROUP BY plugin_name
        """,
        (profile_name, symbol),
    )
    baseline = {str(row["plugin_name"]): float(row["best_score"]) for row in baseline_rows}
    loss_sum = {name: 0.0 for _, name in FEATURE_GROUPS}
    gain_sum = {name: 0.0 for _, name in FEATURE_GROUPS}
    obs = {name: 0 for _, name in FEATURE_GROUPS}

    rows = query_all(
        conn,
        """
        SELECT plugin_name, score, parameters_json, experiment_name
          FROM tuning_runs
         WHERE profile_name = ? AND symbol = ? AND status = 'ok'
           AND experiment_name IN ('feature_mask_ablation', 'schema_ablation')
         ORDER BY finished_at DESC, started_at DESC
         LIMIT 512
        """,
        (profile_name, symbol),
    )
    for row in rows:
        params = _safe_json(str(row["parameters_json"] or "{}"), {})
        plugin_name = str(row["plugin_name"])
        ref_score = float(baseline.get(plugin_name, row["score"]))
        delta = ref_score - float(row["score"])
        mask = int(float(params.get("feature_mask", 0) or 0))
        if mask <= 0:
            continue
        for group_id, group_name in FEATURE_GROUPS:
            enabled = ((mask & (1 << group_id)) != 0)
            if enabled:
                continue
            obs[group_name] += 1
            if delta >= 0.0:
                loss_sum[group_name] += delta
            else:
                gain_sum[group_name] += -delta

    pressure = _clamp(float(shadow.get("mean_portfolio_pressure", 0.0)) / 1.5, 0.0, 1.0)
    macro_guard = _clamp(float(shadow.get("mean_policy_no_trade_prob", 0.0)), 0.0, 1.0)
    weights: dict[str, float] = {}
    for group_id, group_name in FEATURE_GROUPS:
        group_loss = loss_sum[group_name] / float(max(obs[group_name], 1))
        group_gain = gain_sum[group_name] / float(max(obs[group_name], 1))
        base = 1.0
        if group_name in ("price", "multi_timeframe", "volatility"):
            base += 0.08
        if group_name in ("cost", "microstructure"):
            base += 0.04 * pressure
        if group_name == "time_calendar":
            base += 0.06 * macro_guard
        if group_name == "context":
            base += 0.05 * _clamp(float(shadow.get("mean_portfolio_div", 0.0)), 0.0, 1.0)
        weights[group_name] = _clamp(base + 0.18 * group_loss - 0.10 * group_gain, 0.25, 1.50)
    return weights


def _family_weights(family_rows: dict[int, dict],
                    shadow: dict) -> dict[str, float]:
    route_regret = _clamp(float(shadow.get("mean_route_regret", 0.0)), 0.0, 1.0)
    portfolio_div = _clamp(float(shadow.get("mean_portfolio_div", 0.0)), 0.0, 1.0)
    out: dict[str, float] = {}
    for family_id in range(0, 12):
        row = family_rows.get(family_id)
        family_name = plugin_family_name(family_id)
        if row is None:
            out[family_name] = 0.90 if family_name != "rule" else 0.70
            continue
        mean_score = _clamp(float(row.get("mean_score", 0.0)) / 100.0, -1.0, 1.0)
        stability = _clamp(float(row.get("stability_score", 0.0)), 0.0, 1.0)
        mean_adv = _clamp(float(row.get("mean_adversarial_score", 0.0)) / 100.0, -1.0, 1.0)
        champions = _clamp(float(row.get("champion_count", 0.0)) / 4.0, 0.0, 1.0)
        promotions = _clamp(float(row.get("promotion_count", 0.0)) / 6.0, 0.0, 1.0)
        issue_penalty = _clamp(float(row.get("mean_issue_count", 0.0)) / 5.0, 0.0, 1.0)
        weight = 0.78
        weight += 0.26 * mean_score
        weight += 0.22 * stability
        weight += 0.16 * mean_adv
        weight += 0.16 * champions
        weight += 0.10 * promotions
        weight += 0.06 * portfolio_div
        weight -= 0.14 * issue_penalty
        weight -= 0.10 * route_regret
        out[family_name] = _clamp(weight, 0.10, 1.40)
    return out


def _plugin_weights(conn: libsql.Connection,
                    profile_name: str,
                    symbol: str,
                    champion_rows: list[dict],
                    family_weights: dict[str, float],
                    shadow: dict) -> tuple[dict[str, float], dict[str, str], list[str]]:
    latest_rows = _latest_plugin_shadow_rows(conn, profile_name, symbol)
    shadow_pressure = _clamp(float(shadow.get("mean_portfolio_pressure", 0.0)) / 1.5, 0.0, 1.0)
    route_regret = _clamp(float(shadow.get("mean_route_regret", 0.0)), 0.0, 1.0)
    plugin_weights: dict[str, float] = {}
    prune_reasons: dict[str, str] = {}
    pruned_plugins: list[str] = []

    for row in champion_rows:
        plugin_name = str(row["plugin_name"])
        family_name = plugin_family_name(int(row.get("family_id", 11)))
        family_weight = _clamp(float(family_weights.get(family_name, 0.90)), 0.05, 1.50)
        shadow_row = latest_rows.get(plugin_name, {})
        status = str(row.get("status", "")).lower()
        champion_score = _clamp(float(row.get("champion_score", 0.0)) / 100.0, -1.0, 1.0)
        challenger_score = _clamp(float(row.get("challenger_score", 0.0)) / 100.0, -1.0, 1.0)
        shadow_score = _clamp(float(shadow_row.get("shadow_score", 0.0)), -1.0, 1.0)
        route_value = _clamp(float(shadow_row.get("route_value", 0.0)), -1.0, 1.0)
        context_regret = _clamp(float(shadow_row.get("context_regret", 0.0)), 0.0, 1.0)
        route_row_regret = _clamp(float(shadow_row.get("route_regret", route_regret)), 0.0, 1.0)
        portfolio_objective = _clamp(float(shadow_row.get("portfolio_objective", 0.0)), -1.0, 1.0)
        reliability = _clamp(float(shadow_row.get("reliability", 0.0)), 0.0, 1.0)
        base = 0.62
        if status == "champion":
            base += 0.18
        elif status == "challenger":
            base += 0.06
        score = base
        score += 0.18 * family_weight
        score += 0.16 * champion_score
        score += 0.10 * challenger_score
        score += 0.16 * shadow_score
        score += 0.12 * route_value
        score += 0.08 * portfolio_objective
        score += 0.06 * reliability
        score -= 0.14 * route_row_regret
        score -= 0.08 * context_regret
        score -= 0.08 * shadow_pressure
        score = _clamp(score, 0.05, 1.60)
        plugin_weights[plugin_name] = score

        reason = ""
        if score <= 0.22:
            reason = "router_weight_too_low"
        elif shadow_score < -0.18 and route_row_regret > 0.55:
            reason = "negative_live_shadow_and_high_regret"
        elif reliability < 0.20 and route_value < -0.10:
            reason = "low_reliability_and_negative_route_value"
        if reason:
            prune_reasons[plugin_name] = reason
            pruned_plugins.append(plugin_name)

    pruned_plugins = sorted(set(pruned_plugins))
    return plugin_weights, prune_reasons, pruned_plugins


def write_attribution_profiles(conn: libsql.Connection,
                               args) -> list[dict]:
    mode_cfg = resolve_runtime_mode(getattr(args, "runtime_mode", None))
    symbols = sorted({
        str(row["symbol"])
        for row in query_all(
            conn,
            """
            SELECT DISTINCT symbol
              FROM (
                  SELECT symbol FROM live_deployment_profiles WHERE profile_name = ?
                  UNION ALL
                  SELECT symbol FROM champion_registry WHERE profile_name = ?
                  UNION ALL
                  SELECT symbol FROM shadow_fleet_observations WHERE profile_name = ?
              )
             ORDER BY symbol
            """,
            (args.profile, args.profile, args.profile),
        )
    })
    out_dir = RESEARCH_DIR / safe_token(args.profile)
    ensure_dir(out_dir)
    ensure_dir(COMMON_PROMOTION_DIR)
    artifacts: list[dict] = []
    created_at = now_unix()
    active_symbols = set(symbols)

    for symbol in symbols:
        shadow = symbol_shadow_summary(conn, args.profile, symbol)
        family_rows = _latest_family_rows(conn, args.profile, symbol)
        champion_rows = _champion_rows(conn, args.profile, symbol)
        family_weights = _family_weights(family_rows, shadow)
        feature_weights = _feature_group_weights(conn, args.profile, symbol, shadow)
        plugin_weights, prune_reasons, pruned_plugins = _plugin_weights(
            conn,
            args.profile,
            symbol,
            champion_rows,
            family_weights,
            shadow,
        )

        champion_plugins = [str(row["plugin_name"]) for row in champion_rows if str(row.get("status", "")) == "champion"]
        challenger_plugins = [str(row["plugin_name"]) for row in champion_rows if str(row.get("status", "")) == "challenger"]
        champion_only = (
            _clamp(float(shadow.get("mean_shadow_score", 0.0)), -1.0, 1.0) < -0.04 or
            _clamp(float(shadow.get("mean_route_regret", 0.0)), 0.0, 1.0) > 0.48
        )
        if str(mode_cfg["runtime_mode"]) == "production":
            champion_only = True
        max_active_models = int(round(_clamp(
            6.0 +
            8.0 * _clamp(float(shadow.get("mean_portfolio_div", 0.0)), 0.0, 1.0) +
            4.0 * _clamp(float(shadow.get("mean_shadow_score", 0.0)) + 0.5, 0.0, 1.0) -
            5.0 * _clamp(float(shadow.get("mean_route_regret", 0.0)), 0.0, 1.0),
            4.0,
            20.0,
        )))
        max_active_models = min(max_active_models, int(mode_cfg["max_runtime_models"]))
        min_meta_weight = _clamp(
            0.02 +
            0.08 * _clamp(float(shadow.get("mean_route_regret", 0.0)), 0.0, 1.0) +
            0.04 * _clamp(float(shadow.get("mean_policy_no_trade_prob", 0.0)), 0.0, 1.0) -
            0.03 * _clamp(float(shadow.get("mean_portfolio_div", 0.0)), 0.0, 1.0),
            0.0,
            0.25,
        )
        if str(mode_cfg["runtime_mode"]) == "production":
            min_meta_weight = _clamp(min_meta_weight + 0.03, 0.0, 0.25)
        payload = {
            "profile_name": args.profile,
            "symbol": symbol,
            "family_weights": family_weights,
            "feature_group_weights": feature_weights,
            "plugin_weights": plugin_weights,
            "plugin_prune_reasons": prune_reasons,
            "pruned_plugins": pruned_plugins,
            "champion_only": bool(champion_only),
            "max_active_models": int(max_active_models),
            "min_meta_weight": float(min_meta_weight),
            "champion_plugins": champion_plugins,
            "challenger_plugins": challenger_plugins,
            "shadow_summary": shadow,
        }
        tsv_path = COMMON_PROMOTION_DIR / f"fxai_attribution_{safe_token(symbol)}.tsv"
        lines = [
            ("profile_name", args.profile),
            ("symbol", symbol),
            ("champion_only", "1" if champion_only else "0"),
            ("max_active_models", str(int(max_active_models))),
            ("min_meta_weight", f"{min_meta_weight:.6f}"),
            ("champion_plugins_csv", ",".join(champion_plugins)),
        ]
        for family_name in sorted(family_weights):
            lines.append((f"family_weight_{family_name}", f"{float(family_weights[family_name]):.6f}"))
        for group_name in sorted(feature_weights):
            lines.append((f"feature_weight_{group_name}", f"{float(feature_weights[group_name]):.6f}"))
        tsv_path.write_text("".join(f"{key}\t{value}\n" for key, value in lines), encoding="utf-8")
        artifact_sha = testlab.sha256_path(tsv_path)
        json_path = out_dir / f"attribution_{safe_token(symbol)}.json"
        json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        conn.execute(
            """
            INSERT INTO attribution_profiles(profile_name, symbol, artifact_path, artifact_sha256,
                                             champion_only, max_active_models, min_meta_weight,
                                             payload_json, created_at)
            VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(profile_name, symbol) DO UPDATE SET
                artifact_path=excluded.artifact_path,
                artifact_sha256=excluded.artifact_sha256,
                champion_only=excluded.champion_only,
                max_active_models=excluded.max_active_models,
                min_meta_weight=excluded.min_meta_weight,
                payload_json=excluded.payload_json,
                created_at=excluded.created_at
            """,
            (
                args.profile,
                symbol,
                str(tsv_path),
                artifact_sha,
                1 if champion_only else 0,
                int(max_active_models),
                float(min_meta_weight),
                json.dumps(payload, indent=2, sort_keys=True),
                created_at,
            ),
        )
        artifacts.append({
            "symbol": symbol,
            "artifact_path": str(tsv_path),
            "artifact_sha256": artifact_sha,
            "champion_only": bool(champion_only),
            "max_active_models": int(max_active_models),
            "min_meta_weight": float(min_meta_weight),
        })

    stale_rows = query_all(
        conn,
        "SELECT symbol, artifact_path FROM attribution_profiles WHERE profile_name = ?",
        (args.profile,),
    )
    for row in stale_rows:
        symbol = str(row["symbol"])
        if symbol in active_symbols:
            continue
        stale_path = Path(str(row["artifact_path"] or "").strip())
        if stale_path.exists() and stale_path.is_file():
            stale_path.unlink()
        stale_json = out_dir / f"attribution_{safe_token(symbol)}.json"
        if stale_json.exists():
            stale_json.unlink()
        conn.execute(
            "DELETE FROM attribution_profiles WHERE profile_name = ? AND symbol = ?",
            (args.profile, symbol),
        )

    commit_db(conn)
    summary_path = out_dir / "attribution_profiles.json"
    summary_path.write_text(json.dumps(portableize_payload_paths(artifacts), indent=2, sort_keys=True), encoding="utf-8")
    return artifacts
