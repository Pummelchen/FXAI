from __future__ import annotations

import json
from pathlib import Path
import libsql

from .common import *
from .mode import resolve_runtime_mode
from .shadow_fleet import symbol_shadow_summary


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(v)))


def _safe_json(raw: str, default):
    try:
        return json.loads(raw or "")
    except Exception:
        return default


def _latest_attribution_payload(conn: libsql.Connection,
                                profile_name: str) -> dict[str, dict]:
    rows = query_all(
        conn,
        """
        SELECT symbol, payload_json
          FROM attribution_profiles
         WHERE profile_name = ?
         ORDER BY created_at DESC
        """,
        (profile_name,),
    )
    out: dict[str, dict] = {}
    for row in rows:
        symbol = str(row["symbol"])
        if symbol not in out:
            out[symbol] = _safe_json(str(row["payload_json"] or "{}"), {})
    return out


def _family_strengths(conn: libsql.Connection,
                      profile_name: str,
                      symbol: str) -> dict[str, float]:
    rows = query_all(
        conn,
        """
        SELECT family_id, MAX(mean_score) AS best_score, MAX(stability_score) AS best_stability
          FROM family_scorecards
         WHERE profile_name = ? AND symbol = ?
         GROUP BY family_id
        """,
        (profile_name, symbol),
    )
    out: dict[str, float] = {}
    for row in rows:
        family_name = plugin_family_name(int(row["family_id"]))
        out[family_name] = _clamp(
            0.75 +
            0.30 * _clamp(float(row["best_score"]) / 100.0, -1.0, 1.0) +
            0.20 * _clamp(float(row["best_stability"]), 0.0, 1.0),
            0.10,
            1.40,
        )
    return out


def _parse_plugin_weights(payload: dict) -> dict[str, float]:
    raw = payload.get("plugin_weights", {})
    out: dict[str, float] = {}
    if not isinstance(raw, dict):
        return out
    for key, value in raw.items():
        try:
            out[str(key)] = float(value)
        except Exception:
            continue
    return out


def write_student_router_profiles(conn: libsql.Connection,
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
                  SELECT symbol FROM student_deployment_bundles WHERE profile_name = ?
                  UNION ALL
                  SELECT symbol FROM champion_registry WHERE profile_name = ?
              )
             ORDER BY symbol
            """,
            (args.profile, args.profile, args.profile),
        )
    })
    out_dir = RESEARCH_DIR / safe_token(args.profile)
    ensure_dir(out_dir)
    ensure_dir(COMMON_PROMOTION_DIR)
    attribution_map = _latest_attribution_payload(conn, args.profile)
    artifacts: list[dict] = []
    created_at = now_unix()
    active_symbols = set(symbols)

    for symbol in symbols:
        shadow = symbol_shadow_summary(conn, args.profile, symbol)
        attr = dict(attribution_map.get(symbol, {}))
        bundle_rows = query_all(
            conn,
            """
            SELECT plugin_name, family_id, deployment_json
              FROM student_deployment_bundles
             WHERE profile_name = ? AND symbol = ?
             ORDER BY created_at DESC, plugin_name ASC
            """,
            (args.profile, symbol),
        )
        champions = [
            dict(row)
            for row in query_all(
                conn,
                """
                SELECT plugin_name, status, champion_score, challenger_score, family_id
                  FROM champion_registry
                 WHERE profile_name = ? AND symbol = ?
                 ORDER BY
                       CASE status WHEN 'champion' THEN 0 WHEN 'challenger' THEN 1 ELSE 2 END,
                       champion_score DESC,
                       challenger_score DESC,
                       plugin_name ASC
                """,
                (args.profile, symbol),
            )
        ]
        family_weights = {plugin_family_name(fid): 0.90 for fid in range(0, 12)}
        family_weights.update(_family_strengths(conn, args.profile, symbol))
        for family_name, value in dict(attr.get("family_weights", {})).items():
            family_weights[str(family_name)] = _clamp(
                0.55 * family_weights.get(str(family_name), 0.90) + 0.45 * float(value),
                0.05,
                1.50,
            )
        attr_plugin_weights = _parse_plugin_weights(attr)
        pruned_plugins = {str(item) for item in list(attr.get("pruned_plugins", []))}

        top_plugins: list[str] = []
        plugin_weights: dict[str, float] = {}
        for row in champions:
            plugin_name = str(row["plugin_name"])
            if plugin_name not in top_plugins:
                top_plugins.append(plugin_name)
            family_name = plugin_family_name(int(row.get("family_id", 11)))
            status = str(row.get("status", "")).lower()
            base = 0.78
            if status == "champion":
                base += 0.14
            elif status == "challenger":
                base += 0.06
            base *= family_weights.get(family_name, 0.90)
            plugin_weights[plugin_name] = _clamp(base, 0.05, 1.60)
        for row in bundle_rows:
            plugin_name = str(row["plugin_name"])
            if plugin_name not in top_plugins:
                top_plugins.append(plugin_name)
            payload = _safe_json(str(row["deployment_json"] or "{}"), {})
            deployable = _clamp(float(payload.get("deployable_student_floor", 0.42)), 0.0, 1.0)
            lifecycle = dict(payload.get("lifecycle_policy", {}))
            lifecycle_quality = _clamp(
                0.22 * _clamp(float(lifecycle.get("policy_hold_floor", 0.48)), 0.0, 1.0) +
                0.18 * (1.0 - _clamp(float(lifecycle.get("policy_exit_floor", 0.58)), 0.0, 1.0)) +
                0.18 * _clamp(float(lifecycle.get("policy_add_floor", 0.68)), 0.0, 1.0) +
                0.16 * (1.0 - _clamp(float(lifecycle.get("policy_reduce_floor", 0.56)), 0.0, 1.0)) +
                0.12 * (1.0 - _clamp(float(lifecycle.get("policy_timeout_floor", 0.72)), 0.0, 1.0)) +
                0.14 * _clamp(float(lifecycle.get("max_add_fraction", 0.50)), 0.0, 1.0),
                0.0,
                1.0,
            )
            current = plugin_weights.get(plugin_name, 0.78)
            plugin_weights[plugin_name] = _clamp(
                0.62 * current + 0.22 * deployable + 0.16 * lifecycle_quality,
                0.05,
                1.60,
            )
        top_plugins = top_plugins[:24]

        for plugin_name, value in attr_plugin_weights.items():
            current = plugin_weights.get(plugin_name, 0.78)
            plugin_weights[plugin_name] = _clamp(0.55 * current + 0.45 * value, 0.02, 1.60)
        for plugin_name in pruned_plugins:
            plugin_weights[plugin_name] = min(plugin_weights.get(plugin_name, 0.05), 0.02)

        portfolio_div = _clamp(float(shadow.get("mean_portfolio_div", 0.0)), 0.0, 1.0)
        route_regret = _clamp(float(shadow.get("mean_route_regret", 0.0)), 0.0, 1.0)
        champion_only = bool(attr.get("champion_only", False)) and any(str(row.get("status", "")) == "champion" for row in champions)
        max_active_models = int(attr.get("max_active_models", 0) or 0)
        if max_active_models <= 0:
            max_active_models = int(round(_clamp(5.0 + 10.0 * portfolio_div - 4.0 * route_regret, 4.0, 18.0)))
        max_active_models = min(max_active_models, int(mode_cfg["max_runtime_models"]))
        min_meta_weight = _clamp(float(attr.get("min_meta_weight", 0.0) or 0.0), 0.0, 0.25)
        if str(mode_cfg["runtime_mode"]) == "production":
            champion_only = True
            min_meta_weight = _clamp(min_meta_weight + 0.03, 0.0, 0.25)

        payload = {
            "profile_name": args.profile,
            "symbol": symbol,
            "champion_only": champion_only,
            "max_active_models": max_active_models,
            "min_meta_weight": min_meta_weight,
            "allow_plugins_csv": ",".join(top_plugins),
            "family_weights": family_weights,
            "plugin_weights": plugin_weights,
            "pruned_plugins": sorted(pruned_plugins),
            "attribution": attr,
            "shadow_summary": shadow,
        }
        tsv_path = COMMON_PROMOTION_DIR / f"fxai_student_router_{safe_token(symbol)}.tsv"
        lines = [
            ("profile_name", args.profile),
            ("symbol", symbol),
            ("champion_only", "1" if champion_only else "0"),
            ("max_active_models", str(int(max_active_models))),
            ("min_meta_weight", f"{min_meta_weight:.6f}"),
            ("allow_plugins_csv", ",".join(top_plugins)),
            ("plugin_weights_csv", ",".join(
                f"{name}={float(plugin_weights[name]):.6f}"
                for name in sorted(plugin_weights)
            )),
        ]
        for family_name in sorted(family_weights):
            lines.append((f"family_weight_{family_name}", f"{float(family_weights[family_name]):.6f}"))
        tsv_path.write_text("".join(f"{key}\t{value}\n" for key, value in lines), encoding="utf-8")
        artifact_sha = testlab.sha256_path(tsv_path)
        json_path = out_dir / f"student_router_{safe_token(symbol)}.json"
        json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        conn.execute(
            """
            INSERT INTO student_router_profiles(profile_name, symbol, artifact_path, artifact_sha256,
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
            "champion_only": champion_only,
            "max_active_models": int(max_active_models),
            "min_meta_weight": float(min_meta_weight),
        })

    stale_rows = query_all(
        conn,
        "SELECT symbol, artifact_path FROM student_router_profiles WHERE profile_name = ?",
        (args.profile,),
    )
    for row in stale_rows:
        symbol = str(row["symbol"])
        if symbol in active_symbols:
            continue
        path = Path(str(row["artifact_path"] or "").strip())
        if path.exists() and path.is_file():
            path.unlink()
        stale_json = out_dir / f"student_router_{safe_token(symbol)}.json"
        if stale_json.exists():
            stale_json.unlink()
        conn.execute(
            "DELETE FROM student_router_profiles WHERE profile_name = ? AND symbol = ?",
            (args.profile, symbol),
        )

    commit_db(conn)
    summary_path = out_dir / "student_router_profiles.json"
    summary_path.write_text(json.dumps(artifacts, indent=2, sort_keys=True), encoding="utf-8")
    return artifacts
