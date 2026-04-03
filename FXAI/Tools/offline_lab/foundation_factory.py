from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
import libsql

from .common import *
from .shadow_fleet import latest_shadow_rows, symbol_shadow_summary


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(v)))


def _weighted_mean(items: list[tuple[float, float]], default: float) -> float:
    total_w = 0.0
    total_v = 0.0
    for value, weight in items:
        w = max(float(weight), 0.0)
        total_w += w
        total_v += float(value) * w
    if total_w <= 1e-9:
        return float(default)
    return total_v / total_w


FOUNDATION_MODEL_FEATURES = [
    "shadow_score",
    "route_value",
    "route_regret",
    "portfolio_objective",
    "portfolio_stability",
    "portfolio_corr",
    "portfolio_div",
    "policy_enter_prob",
    "policy_no_trade_prob",
    "policy_add_prob",
    "policy_reduce_prob",
    "policy_timeout_prob",
    "policy_capital_efficiency",
    "portfolio_pressure",
    "control_plane_score",
    "portfolio_supervisor_score",
]


def _symbol_shadow_rows(conn: libsql.Connection,
                       profile_name: str,
                       symbol: str,
                       limit: int = 512) -> list[dict]:
    return query_all(
        conn,
        """
        SELECT *
          FROM shadow_fleet_observations
         WHERE profile_name = ? AND symbol = ?
         ORDER BY captured_at DESC, id DESC
         LIMIT ?
        """,
        (profile_name, symbol, limit),
    )


def _shadow_feature_snapshot(rows: list[dict]) -> dict[str, float]:
    if not rows:
        return {name: 0.0 for name in FOUNDATION_MODEL_FEATURES}
    out: dict[str, float] = {}
    for name in FOUNDATION_MODEL_FEATURES:
        out[name] = sum(row_float(row, name, 0.0) for row in rows) / float(len(rows))
    return out


def _latest_family_cards(conn: libsql.Connection,
                         profile_name: str) -> dict[tuple[str, int], dict]:
    rows = query_all(
        conn,
        """
        SELECT fs.*
          FROM family_scorecards fs
          JOIN (
                SELECT profile_name, symbol, family_id, MAX(created_at) AS max_created_at
                  FROM family_scorecards
                 WHERE profile_name = ?
                 GROUP BY profile_name, symbol, family_id
          ) latest
            ON latest.profile_name = fs.profile_name
           AND latest.symbol = fs.symbol
           AND latest.family_id = fs.family_id
           AND latest.max_created_at = fs.created_at
         WHERE fs.profile_name = ?
        """,
        (profile_name, profile_name),
    )
    return {
        (str(row["symbol"]), int(row["family_id"])): dict(row)
        for row in rows
    }


def _latest_distill_map(conn: libsql.Connection,
                        profile_name: str) -> dict[tuple[str, str], dict]:
    rows = query_all(
        conn,
        """
        SELECT *
          FROM distillation_artifacts
         WHERE profile_name = ? AND dataset_scope = 'aggregate'
         ORDER BY created_at DESC
        """,
        (profile_name,),
    )
    out: dict[tuple[str, str], dict] = {}
    for row in rows:
        key = (str(row["symbol"]), str(row["plugin_name"]))
        if key not in out:
            out[key] = dict(row)
    return out


def _latest_dataset_windows(conn: libsql.Connection,
                            symbol: str,
                            limit: int = 8) -> list[dict]:
    return query_all(
        conn,
        """
        SELECT dataset_key, group_key, months, bars, start_unix, end_unix, source_sha256
          FROM datasets
         WHERE symbol = ?
         ORDER BY end_unix DESC, created_at DESC
         LIMIT ?
        """,
        (symbol, limit),
    )


def write_foundation_model_bundles(conn: libsql.Connection,
                                   args,
                                   promoted_rows: list[dict]) -> list[dict]:
    out_dir = DISTILL_DIR / safe_token(args.profile) / "FoundationFactory"
    ensure_dir(out_dir)
    family_cards = _latest_family_cards(conn, args.profile)
    shadow_map = latest_shadow_rows(conn, args.profile)
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in promoted_rows:
        grouped[str(row["symbol"])].append(dict(row))

    artifacts: list[dict] = []
    created_at = now_unix()
    active_symbols = set(grouped.keys())
    for symbol, items in sorted(grouped.items()):
        shadow = symbol_shadow_summary(conn, args.profile, symbol)
        shadow_rows = _symbol_shadow_rows(conn, args.profile, symbol)
        shadow_features = _shadow_feature_snapshot(shadow_rows)
        foundation_fit_rows: list[dict] = []
        for row in shadow_rows:
            enriched = dict(row)
            enriched["_teacher_floor_target"] = (
                0.46 * row_float(row, "shadow_score", 0.0) +
                0.22 * row_float(row, "portfolio_stability", 0.0) +
                0.18 * row_float(row, "policy_capital_efficiency", 0.0) -
                0.16 * row_float(row, "route_regret", 0.0)
            )
            enriched["_student_floor_target"] = (
                0.34 * row_float(row, "route_value", 0.0) +
                0.22 * row_float(row, "portfolio_div", 0.0) +
                0.18 * row_float(row, "policy_enter_prob", 0.0) -
                0.16 * row_float(row, "policy_no_trade_prob", 0.0) -
                0.10 * row_float(row, "portfolio_pressure", 0.0)
            )
            foundation_fit_rows.append(enriched)
        foundation_models = {
            "teacher_floor": fit_weighted_linear_model(foundation_fit_rows, FOUNDATION_MODEL_FEATURES, "_teacher_floor_target", "obs_count"),
            "student_floor": fit_weighted_linear_model(foundation_fit_rows, FOUNDATION_MODEL_FEATURES, "_student_floor_target", "obs_count"),
        }
        family_payload = []
        family_weights: list[tuple[int, float]] = []
        for row in items:
            family_id = int(row.get("family_id", 11))
            card = family_cards.get((symbol, family_id), {})
            rank = max(float(row.get("ranking_score", 0.0)), 1.0)
            stability = float(card.get("stability_score", 0.0))
            macro = float(card.get("mean_macro_score", 0.0)) / 100.0
            family_weight = rank * (0.55 + 0.25 * stability + 0.20 * _clamp(macro, 0.0, 1.0))
            family_weights.append((family_id, family_weight))
            family_payload.append({
                "family_id": family_id,
                "family_name": plugin_family_name(family_id),
                "weight": family_weight,
                "scorecard": card,
            })

        total_family = max(sum(weight for _family, weight in family_weights), 1.0)
        for item in family_payload:
            item["weight"] = _clamp(float(item["weight"]) / total_family, 0.0, 1.0)

        windows = _latest_dataset_windows(conn, symbol)
        curriculum = [{
            "dataset_key": str(item["dataset_key"]),
            "group_key": str(item.get("group_key", "") or ""),
            "months": int(item["months"]),
            "bars": int(item["bars"]),
            "start_unix": int(item["start_unix"]),
            "end_unix": int(item["end_unix"]),
            "source_sha256": str(item.get("source_sha256", "") or ""),
        } for item in windows]

        payload = {
            "profile_name": args.profile,
            "symbol": symbol,
            "bundle_scope": "symbol",
            "family_mix": family_payload,
            "curriculum": curriculum,
            "shadow_summary": shadow,
            "training_models": foundation_models,
            "training_features": shadow_features,
            "teacher_floor": _clamp(
                0.48 +
                0.16 * _clamp(shadow.get("mean_shadow_score", 0.0), 0.0, 1.0) +
                0.12 * _clamp(shadow.get("mean_portfolio_stability", 0.0), 0.0, 1.0) -
                0.10 * _clamp(shadow.get("mean_route_regret", 0.0), 0.0, 1.0) +
                0.20 * predict_linear_model(foundation_models["teacher_floor"], shadow_features, 0.0),
                0.20,
                0.95,
            ),
            "student_floor": _clamp(
                0.34 +
                0.14 * _clamp(shadow.get("mean_policy_enter_prob", 0.0), 0.0, 1.0) +
                0.10 * _clamp(shadow.get("mean_portfolio_div", 0.0), 0.0, 1.0) -
                0.08 * _clamp(shadow.get("mean_portfolio_pressure", 0.0) / 1.5, 0.0, 1.0) +
                0.22 * predict_linear_model(foundation_models["student_floor"], shadow_features, 0.0),
                0.10,
                0.90,
            ),
            "macro_state_bias": _clamp(
                0.25 * _clamp(shadow.get("mean_shadow_score", 0.0), 0.0, 1.0) +
                0.35 * _clamp(shadow.get("mean_portfolio_objective", 0.0), 0.0, 1.0) +
                0.40 * _clamp(shadow.get("mean_policy_capital_efficiency", 0.0), 0.0, 1.0),
                0.0,
                1.0,
            ),
            "regime_transition_bias": _clamp(
                0.20 +
                0.24 * _clamp(shadow.get("mean_counterfactual", 0.0), 0.0, 1.0) +
                0.16 * _clamp(1.0 - shadow.get("mean_route_regret", 0.0), 0.0, 1.0),
                0.0,
                1.0,
            ),
        }

        symbol_dir = out_dir / safe_token(symbol)
        ensure_dir(symbol_dir)
        artifact_path = symbol_dir / "foundation_bundle.json"
        artifact_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        artifact_sha = testlab.sha256_path(artifact_path)
        conn.execute(
            """
            INSERT INTO foundation_model_bundles(profile_name, symbol, bundle_scope, artifact_path, artifact_sha256,
                                                 payload_json, created_at)
            VALUES(?, ?, 'symbol', ?, ?, ?, ?)
            ON CONFLICT(profile_name, symbol, bundle_scope) DO UPDATE SET
                artifact_path=excluded.artifact_path,
                artifact_sha256=excluded.artifact_sha256,
                payload_json=excluded.payload_json,
                created_at=excluded.created_at
            """,
            (
                args.profile,
                symbol,
                str(artifact_path),
                artifact_sha,
                json.dumps(payload, indent=2, sort_keys=True),
                created_at,
            ),
        )
        artifacts.append({
            "symbol": symbol,
            "artifact_path": str(artifact_path),
            "artifact_sha256": artifact_sha,
        })

    stale_rows = query_all(
        conn,
        """
        SELECT symbol, artifact_path
          FROM foundation_model_bundles
         WHERE profile_name = ? AND bundle_scope = 'symbol'
        """,
        (args.profile,),
    )
    for row in stale_rows:
        symbol = str(row["symbol"])
        if symbol in active_symbols:
            continue
        path = Path(str(row["artifact_path"] or "").strip())
        if path.exists() and path.is_file():
            path.unlink()
        conn.execute(
            "DELETE FROM foundation_model_bundles WHERE profile_name = ? AND symbol = ? AND bundle_scope = 'symbol'",
            (args.profile, symbol),
        )

    commit_db(conn)
    summary_path = out_dir / "foundation_bundles.json"
    summary_path.write_text(json.dumps(artifacts, indent=2, sort_keys=True), encoding="utf-8")
    return artifacts


def write_student_deployment_bundles(conn: libsql.Connection,
                                     args,
                                     promoted_rows: list[dict]) -> list[dict]:
    out_dir = DISTILL_DIR / safe_token(args.profile) / "StudentFactory"
    ensure_dir(out_dir)
    distill_map = _latest_distill_map(conn, args.profile)
    shadow_map = latest_shadow_rows(conn, args.profile)
    foundation_rows = query_all(
        conn,
        """
        SELECT symbol, payload_json
          FROM foundation_model_bundles
         WHERE profile_name = ? AND bundle_scope = 'symbol'
        """,
        (args.profile,),
    )
    foundation_map = {
        str(row["symbol"]): json.loads(row["payload_json"] or "{}")
        for row in foundation_rows
    }

    artifacts: list[dict] = []
    active_keys: set[tuple[str, str]] = set()
    created_at = now_unix()
    for row in promoted_rows:
        symbol = str(row["symbol"])
        plugin_name = str(row["plugin_name"])
        family_id = int(row.get("family_id", 11))
        active_keys.add((symbol, plugin_name))
        distill = distill_map.get((symbol, plugin_name), {})
        shadow = shadow_map.get((symbol, plugin_name), {})
        foundation = foundation_map.get(symbol, {})
        fit_features = {
            name: row_float(shadow, name, 0.0)
            for name in FOUNDATION_MODEL_FEATURES
        }
        lifecycle_fit_rows = [dict(shadow)] if shadow else []
        for row_payload in lifecycle_fit_rows:
            row_payload["_hold_target"] = (
                0.42 * row_float(row_payload, "policy_capital_efficiency", 0.0) +
                0.24 * row_float(row_payload, "shadow_score", 0.0) -
                0.16 * row_float(row_payload, "route_regret", 0.0) -
                0.12 * row_float(row_payload, "portfolio_pressure", 0.0)
            )
        lifecycle_model = fit_weighted_linear_model(
            lifecycle_fit_rows,
            FOUNDATION_MODEL_FEATURES,
            "_hold_target",
            "obs_count",
        )
        target = family_distillation_profile(family_id)
        if distill:
            try:
                target.update(json.loads(distill.get("student_target_json", "{}") or "{}"))
            except Exception:
                pass

        mean_shadow = _clamp(shadow.get("shadow_score", 0.0), 0.0, 1.0)
        route_regret = _clamp(shadow.get("route_regret", 0.0), 0.0, 1.0)
        port_div = _clamp(shadow.get("portfolio_div", 0.0), 0.0, 1.0)
        port_pressure = _clamp(shadow.get("portfolio_pressure", 0.0) / 1.5, 0.0, 1.0)
        capital_eff = _clamp(shadow.get("policy_capital_efficiency", 0.0), 0.0, 1.0)
        enter_prob = _clamp(shadow.get("policy_enter_prob", 0.0), 0.0, 1.0)
        no_trade_prob = _clamp(shadow.get("policy_no_trade_prob", 0.0), 0.0, 1.0)
        foundation_teacher_floor = _clamp(foundation.get("teacher_floor", 0.58), 0.20, 0.95)
        foundation_student_floor = _clamp(foundation.get("student_floor", 0.42), 0.10, 0.90)

        lifecycle_policy = {
            "policy_hold_floor": _clamp(0.42 + 0.16 * capital_eff + 0.10 * mean_shadow - 0.08 * route_regret, 0.20, 0.95),
            "policy_exit_floor": _clamp(0.48 + 0.18 * no_trade_prob + 0.12 * port_pressure + 0.08 * route_regret, 0.20, 0.99),
            "policy_add_floor": _clamp(0.58 + 0.18 * enter_prob + 0.14 * capital_eff + 0.10 * port_div - 0.14 * port_pressure - 0.10 * route_regret, 0.30, 0.99),
            "policy_reduce_floor": _clamp(0.44 + 0.18 * no_trade_prob + 0.14 * port_pressure + 0.10 * route_regret, 0.25, 0.99),
            "policy_timeout_floor": _clamp(0.56 + 0.14 * no_trade_prob + 0.12 * route_regret + 0.10 * port_pressure, 0.30, 0.99),
            "max_add_fraction": _clamp(0.18 + 0.30 * port_div + 0.18 * capital_eff + 0.10 * mean_shadow - 0.18 * port_pressure, 0.05, 1.00),
            "reduce_fraction": _clamp(0.18 + 0.24 * port_pressure + 0.16 * no_trade_prob + 0.10 * route_regret, 0.10, 0.90),
            "soft_timeout_bars": int(round(_clamp(6.0 + 6.0 * foundation_student_floor + 4.0 * capital_eff - 2.0 * route_regret, 4.0, 48.0))),
            "hard_timeout_bars": int(round(_clamp(12.0 + 10.0 * foundation_teacher_floor + 6.0 * capital_eff - 2.0 * route_regret, 8.0, 96.0))),
        }
        if lifecycle_policy["hard_timeout_bars"] <= lifecycle_policy["soft_timeout_bars"]:
            lifecycle_policy["hard_timeout_bars"] = lifecycle_policy["soft_timeout_bars"] + 4

        payload = {
            "profile_name": args.profile,
            "symbol": symbol,
            "plugin_name": plugin_name,
            "family_id": family_id,
            "family_name": plugin_family_name(family_id),
            "ranking_score": float(row.get("ranking_score", 0.0)),
            "score": float(row.get("score", 0.0)),
            "student_target": target,
            "foundation_bundle": foundation,
            "shadow_live": shadow,
            "lifecycle_policy": lifecycle_policy,
            "lifecycle_model": lifecycle_model,
            "fit_features": fit_features,
            "deployable_student_floor": _clamp(
                0.40 * foundation_student_floor +
                0.30 * target.get("student_weight", 0.42) +
                0.30 * capital_eff,
                0.10,
                0.95,
            ),
        }

        symbol_dir = out_dir / safe_token(symbol)
        ensure_dir(symbol_dir)
        artifact_path = symbol_dir / f"{plugin_name}__student_bundle.json"
        artifact_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        artifact_sha = testlab.sha256_path(artifact_path)
        conn.execute(
            """
            INSERT INTO student_deployment_bundles(profile_name, symbol, plugin_name, family_id,
                                                   artifact_path, artifact_sha256, deployment_json, created_at)
            VALUES(?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(profile_name, symbol, plugin_name) DO UPDATE SET
                family_id=excluded.family_id,
                artifact_path=excluded.artifact_path,
                artifact_sha256=excluded.artifact_sha256,
                deployment_json=excluded.deployment_json,
                created_at=excluded.created_at
            """,
            (
                args.profile,
                symbol,
                plugin_name,
                family_id,
                str(artifact_path),
                artifact_sha,
                json.dumps(payload, indent=2, sort_keys=True),
                created_at,
            ),
        )
        artifacts.append({
            "symbol": symbol,
            "plugin_name": plugin_name,
            "artifact_path": str(artifact_path),
            "artifact_sha256": artifact_sha,
        })

    stale_rows = query_all(
        conn,
        """
        SELECT symbol, plugin_name, artifact_path
          FROM student_deployment_bundles
         WHERE profile_name = ?
        """,
        (args.profile,),
    )
    for row in stale_rows:
        key = (str(row["symbol"]), str(row["plugin_name"]))
        if key in active_keys:
            continue
        path = Path(str(row["artifact_path"] or "").strip())
        if path.exists() and path.is_file():
            path.unlink()
        conn.execute(
            "DELETE FROM student_deployment_bundles WHERE profile_name = ? AND symbol = ? AND plugin_name = ?",
            (args.profile, key[0], key[1]),
        )

    commit_db(conn)
    summary_path = out_dir / "student_bundles.json"
    summary_path.write_text(json.dumps(artifacts, indent=2, sort_keys=True), encoding="utf-8")
    return artifacts
