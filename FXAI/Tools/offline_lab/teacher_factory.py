from __future__ import annotations

import json
import sqlite3
from collections import defaultdict
from pathlib import Path

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


def _latest_family_scorecards(conn: sqlite3.Connection,
                              profile_name: str) -> dict[tuple[str, int], dict]:
    rows = conn.execute(
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
    ).fetchall()
    return {
        (str(row["symbol"]), int(row["family_id"])): dict(row)
        for row in rows
    }


def _latest_distill_artifacts(conn: sqlite3.Connection,
                              profile_name: str) -> dict[tuple[str, str], dict]:
    rows = conn.execute(
        """
        SELECT *
          FROM distillation_artifacts
         WHERE profile_name = ? AND dataset_scope = 'aggregate'
         ORDER BY created_at DESC
        """,
        (profile_name,),
    ).fetchall()
    out: dict[tuple[str, str], dict] = {}
    for row in rows:
        key = (str(row["symbol"]), str(row["plugin_name"]))
        if key not in out:
            out[key] = dict(row)
    return out


def write_foundation_teacher_artifacts(conn: sqlite3.Connection,
                                       args,
                                       promoted_rows: list[dict]) -> list[dict]:
    out_dir = DISTILL_DIR / safe_token(args.profile) / "Foundations"
    ensure_dir(out_dir)
    family_cards = _latest_family_scorecards(conn, args.profile)
    distill_map = _latest_distill_artifacts(conn, args.profile)
    shadow_map = latest_shadow_rows(conn, args.profile)
    grouped: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for row in promoted_rows:
        grouped[(str(row["symbol"]), int(row.get("family_id", 11)))].append(dict(row))

    artifacts: list[dict] = []
    created_at = now_unix()
    for (symbol, family_id), items in sorted(grouped.items()):
        weights = [max(float(item.get("ranking_score", 0.0)), 1.0) for item in items]
        total_w = max(sum(weights), 1.0)
        teacher_weight = 0.0
        student_weight = 0.0
        self_sup_weight = 0.0
        analog_weight = 0.0
        foundation_weight = 0.0
        champions: list[dict] = []
        for item, weight in zip(items, weights):
            plugin_name = str(item["plugin_name"])
            distill = distill_map.get((symbol, plugin_name), {})
            student_target = family_distillation_profile(family_id)
            if distill:
                try:
                    student_target.update(json.loads(distill.get("student_target_json", "{}") or "{}"))
                except Exception:
                    pass
            shadow = shadow_map.get((symbol, plugin_name), {})
            teacher_weight += float(student_target.get("teacher_weight", 0.58)) * weight
            student_weight += float(student_target.get("student_weight", 0.42)) * weight
            self_sup_weight += float(student_target.get("self_supervised_weight", 0.10)) * weight
            analog_weight += float(student_target.get("analog_weight", 0.08)) * weight
            foundation_weight += float(student_target.get("foundation_weight", 0.14)) * weight
            champions.append({
                "plugin_name": plugin_name,
                "ai_id": int(item["ai_id"]),
                "ranking_score": float(item["ranking_score"]),
                "support_count": int(item["support_count"]),
                "shadow_score": float(shadow.get("shadow_score", 0.0)),
                "route_value": float(shadow.get("route_value", 0.0)),
                "student_target": student_target,
            })

        family_card = family_cards.get((symbol, family_id), {})
        payload = {
            "profile_name": args.profile,
            "symbol": symbol,
            "scope": "symbol_family",
            "family_id": family_id,
            "family_name": plugin_family_name(family_id),
            "teacher_weight": _clamp(teacher_weight / total_w, 0.05, 0.95),
            "student_weight": _clamp(student_weight / total_w, 0.05, 0.95),
            "self_supervised_weight": _clamp(self_sup_weight / total_w, 0.0, 1.0),
            "analog_weight": _clamp(analog_weight / total_w, 0.0, 0.80),
            "foundation_weight": _clamp(foundation_weight / total_w, 0.0, 0.90),
            "family_scorecard": family_card,
            "champions": champions,
        }
        symbol_dir = out_dir / safe_token(symbol)
        ensure_dir(symbol_dir)
        artifact_path = symbol_dir / f"foundation_family_{family_id}.json"
        artifact_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        artifact_sha = testlab.sha256_path(artifact_path)
        conn.execute(
            """
            INSERT INTO foundation_teacher_artifacts(profile_name, symbol, scope, family_id, artifact_path, artifact_sha256,
                                                     teacher_payload_json, student_profile_json, status, created_at)
            VALUES(?, ?, 'symbol_family', ?, ?, ?, ?, ?, 'ready', ?)
            ON CONFLICT(profile_name, symbol, scope, family_id) DO UPDATE SET
                artifact_path=excluded.artifact_path,
                artifact_sha256=excluded.artifact_sha256,
                teacher_payload_json=excluded.teacher_payload_json,
                student_profile_json=excluded.student_profile_json,
                status=excluded.status,
                created_at=excluded.created_at
            """,
            (
                args.profile,
                symbol,
                family_id,
                str(artifact_path),
                artifact_sha,
                json.dumps(payload, indent=2, sort_keys=True),
                json.dumps({
                    "teacher_weight": payload["teacher_weight"],
                    "student_weight": payload["student_weight"],
                    "self_supervised_weight": payload["self_supervised_weight"],
                    "analog_weight": payload["analog_weight"],
                    "foundation_weight": payload["foundation_weight"],
                }, indent=2, sort_keys=True),
                created_at,
            ),
        )
        artifacts.append({
            "symbol": symbol,
            "family_id": family_id,
            "artifact_path": str(artifact_path),
            "artifact_sha256": artifact_sha,
        })

    conn.commit()
    summary_path = out_dir / "foundation_teachers.json"
    summary_path.write_text(json.dumps(artifacts, indent=2, sort_keys=True), encoding="utf-8")
    return artifacts


def write_teacher_factory_artifacts(conn: sqlite3.Connection,
                                    args,
                                    promoted_rows: list[dict]) -> list[dict]:
    out_dir = DISTILL_DIR / safe_token(args.profile) / "TeacherFactory"
    ensure_dir(out_dir)
    family_cards = _latest_family_scorecards(conn, args.profile)
    distill_map = _latest_distill_artifacts(conn, args.profile)
    shadow_map = latest_shadow_rows(conn, args.profile)
    artifacts: list[dict] = []
    created_at = now_unix()

    for row in promoted_rows:
        symbol = str(row["symbol"])
        plugin_name = str(row["plugin_name"])
        family_id = int(row.get("family_id", 11))
        params = json.loads(row["parameters_json"])
        distill = distill_map.get((symbol, plugin_name))
        family_card = family_cards.get((symbol, family_id), {})
        shadow = shadow_map.get((symbol, plugin_name), {})
        base_profile = family_distillation_profile(family_id)
        student_target = dict(base_profile)
        if distill:
            try:
                student_target.update(json.loads(distill.get("student_target_json", "{}") or "{}"))
            except Exception:
                pass

        teacher_payload = {
            "profile_name": args.profile,
            "symbol": symbol,
            "plugin_name": plugin_name,
            "ai_id": int(row["ai_id"]),
            "family_id": family_id,
            "family_name": plugin_family_name(family_id),
            "ranking_score": float(row["ranking_score"]),
            "score": float(row["score"]),
            "support_count": int(row["support_count"]),
            "parameters": params,
            "student_target": student_target,
            "family_scorecard": family_card,
            "shadow_live": shadow,
            "support": json.loads(row.get("support_json", "[]") or "[]"),
        }

        symbol_dir = out_dir / safe_token(symbol)
        ensure_dir(symbol_dir)
        teacher_path = symbol_dir / f"{plugin_name}__teacher_factory.json"
        teacher_path.write_text(json.dumps(teacher_payload, indent=2, sort_keys=True), encoding="utf-8")
        teacher_sha = testlab.sha256_path(teacher_path)

        best_cfg = row_to_dict(conn.execute(
            "SELECT id FROM best_configs WHERE profile_name = ? AND dataset_scope = 'aggregate' AND symbol = ? AND plugin_name = ?",
            (args.profile, symbol, plugin_name),
        ).fetchone())
        best_config_id = int(best_cfg["id"]) if best_cfg else 0

        conn.execute(
            """
            INSERT INTO teacher_factories(profile_name, symbol, plugin_name, family_id, champion_best_config_id, source_run_id,
                                          teacher_artifact_path, teacher_artifact_sha256, student_artifact_path, student_artifact_sha256,
                                          deployment_profile_path, deployment_profile_sha256, teacher_score, student_score,
                                          live_shadow_score, portfolio_score, policy_score, payload_json, status, created_at)
            VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, '', '', ?, ?, ?, ?, ?, ?, 'ready', ?)
            ON CONFLICT(profile_name, symbol, plugin_name) DO UPDATE SET
                family_id=excluded.family_id,
                champion_best_config_id=excluded.champion_best_config_id,
                source_run_id=excluded.source_run_id,
                teacher_artifact_path=excluded.teacher_artifact_path,
                teacher_artifact_sha256=excluded.teacher_artifact_sha256,
                student_artifact_path=excluded.student_artifact_path,
                student_artifact_sha256=excluded.student_artifact_sha256,
                teacher_score=excluded.teacher_score,
                student_score=excluded.student_score,
                live_shadow_score=excluded.live_shadow_score,
                portfolio_score=excluded.portfolio_score,
                policy_score=excluded.policy_score,
                payload_json=excluded.payload_json,
                status=excluded.status,
                created_at=excluded.created_at
            """,
            (
                args.profile,
                symbol,
                plugin_name,
                family_id,
                best_config_id,
                int(row["run_id"]),
                str(teacher_path),
                teacher_sha,
                str(distill.get("artifact_path", "")) if distill else "",
                str(distill.get("artifact_sha256", "")) if distill else "",
                float(row["ranking_score"]),
                float(row["score"]),
                float(shadow.get("shadow_score", 0.0)),
                float(shadow.get("portfolio_objective", family_card.get("stability_score", 0.0) or 0.0)),
                float(shadow.get("route_value", 0.0)),
                json.dumps(teacher_payload, indent=2, sort_keys=True),
                created_at,
            ),
        )
        artifacts.append({
            "symbol": symbol,
            "plugin_name": plugin_name,
            "family_id": family_id,
            "teacher_artifact_path": str(teacher_path),
            "teacher_artifact_sha256": teacher_sha,
        })

    conn.commit()
    summary_path = out_dir / "teacher_factories.json"
    summary_path.write_text(json.dumps(artifacts, indent=2, sort_keys=True), encoding="utf-8")
    return artifacts


def write_live_deployment_profiles(conn: sqlite3.Connection,
                                   args) -> list[dict]:
    ensure_dir(COMMON_PROMOTION_DIR)
    out_dir = RESEARCH_DIR / safe_token(args.profile)
    ensure_dir(out_dir)
    family_cards = _latest_family_scorecards(conn, args.profile)
    distill_map = _latest_distill_artifacts(conn, args.profile)

    champion_rows = conn.execute(
        """
        SELECT cr.*, bc.parameters_json
          FROM champion_registry cr
          LEFT JOIN best_configs bc ON bc.id = cr.champion_best_config_id
         WHERE cr.profile_name = ? AND cr.status = 'champion'
         ORDER BY cr.symbol, cr.champion_score DESC, cr.portfolio_score DESC
        """,
        (args.profile,),
    ).fetchall()

    by_symbol: dict[str, list[dict]] = defaultdict(list)
    for row in champion_rows:
        by_symbol[str(row["symbol"])].append(dict(row))

    existing_rows = conn.execute(
        """
        SELECT symbol, artifact_path
          FROM live_deployment_profiles
         WHERE profile_name = ?
        """,
        (args.profile,),
    ).fetchall()
    existing_symbols = {str(row["symbol"]): str(row["artifact_path"] or "") for row in existing_rows}

    deployments: list[dict] = []
    created_at = now_unix()
    for symbol, items in sorted(by_symbol.items()):
        shadow_summary = symbol_shadow_summary(conn, args.profile, symbol)
        teacher_weights: list[tuple[float, float]] = []
        student_weights: list[tuple[float, float]] = []
        analog_weights: list[tuple[float, float]] = []
        foundation_weights: list[tuple[float, float]] = []
        policy_trade_floor = 0.52
        policy_no_trade_cap = 0.62
        policy_size_bias = 1.0
        portfolio_budget_bias = 1.0
        challenger_margin = 1.0
        regime_transition_weight = 0.35
        macro_quality_floor = 0.24
        capital_efficiency_bias = 1.0
        supervisor_blend = 0.45

        deployment_champions: list[dict] = []
        mean_macro_score = 0.0
        mean_stability = 0.0
        mean_route_value = 0.0
        mean_route_regret = 0.0

        for item in items:
            family_id = int(item.get("family_id", 11))
            champion_weight = max(float(item.get("champion_score", 0.0)), 1.0)
            distill = distill_map.get((symbol, str(item["plugin_name"])))
            distill_target = family_distillation_profile(family_id)
            if distill:
                try:
                    distill_target.update(json.loads(distill.get("student_target_json", "{}") or "{}"))
                except Exception:
                    pass
            family_card = family_cards.get((symbol, family_id), {})
            macro_score = float(family_card.get("mean_macro_score", 0.0))
            stability = float(family_card.get("stability_score", 0.0))

            teacher_weights.append((distill_target.get("teacher_weight", 0.58), champion_weight))
            student_weights.append((distill_target.get("student_weight", 0.42), champion_weight))
            analog_weight = float(distill_target.get("analog_weight", 0.08))
            if family_id in (7, 8, 9):
                analog_weight += 0.06
            analog_weights.append((analog_weight, champion_weight))
            foundation_weights.append((distill_target.get("foundation_weight", 0.14), champion_weight))
            mean_macro_score += macro_score * champion_weight
            mean_stability += stability * champion_weight
            deployment_champions.append({
                "plugin_name": item["plugin_name"],
                "family_id": family_id,
                "champion_score": float(item.get("champion_score", 0.0)),
                "portfolio_score": float(item.get("portfolio_score", 0.0)),
                "distillation": distill_target,
                "family_scorecard": family_card,
            })

        total_weight = max(sum(weight for _value, weight in teacher_weights), 1.0)
        mean_macro_score /= total_weight
        mean_stability /= total_weight
        mean_route_value = float(shadow_summary.get("mean_route_value", 0.0))
        mean_route_regret = float(shadow_summary.get("mean_route_regret", 0.0))

        teacher_weight = _clamp(_weighted_mean(teacher_weights, 0.58), 0.05, 0.95)
        student_weight = _clamp(_weighted_mean(student_weights, 0.42), 0.05, 0.95)
        analog_weight = _clamp(_weighted_mean(analog_weights, 0.18) + 0.08 * shadow_summary.get("mean_portfolio_div", 0.0), 0.0, 0.80)
        foundation_weight = _clamp(_weighted_mean(foundation_weights, 0.24) + 0.06 * shadow_summary.get("mean_shadow_score", 0.0), 0.0, 0.90)
        policy_trade_floor = _clamp(
            0.48 +
            0.10 * mean_stability +
            0.08 * _clamp(mean_macro_score / 100.0, 0.0, 1.0) +
            0.06 * _clamp(shadow_summary.get("mean_shadow_score", 0.0), 0.0, 1.0) -
            0.06 * _clamp(mean_route_regret, 0.0, 1.0),
            0.20,
            0.90,
        )
        policy_no_trade_cap = _clamp(
            0.72 -
            0.10 * _clamp(shadow_summary.get("mean_policy_enter_prob", 0.0), 0.0, 1.0) +
            0.12 * _clamp(shadow_summary.get("mean_policy_no_trade_prob", 0.0), 0.0, 1.0) +
            0.08 * _clamp(shadow_summary.get("mean_portfolio_pressure", 0.0) / 1.5, 0.0, 1.0),
            0.25,
            0.95,
        )
        policy_size_bias = _clamp(
            0.88 +
            0.16 * _clamp(shadow_summary.get("mean_portfolio_objective", 0.0), 0.0, 1.0) +
            0.08 * mean_stability -
            0.08 * _clamp(mean_route_regret, 0.0, 1.0),
            0.40,
            1.60,
        )
        capital_efficiency_bias = _clamp(
            0.86 +
            0.20 * _clamp(shadow_summary.get("mean_policy_capital_efficiency", 0.0), 0.0, 1.0) +
            0.08 * _clamp(shadow_summary.get("mean_route_value", 0.0), 0.0, 1.0) -
            0.06 * _clamp(mean_route_regret, 0.0, 1.0),
            0.40,
            1.80,
        )
        portfolio_budget_bias = _clamp(
            0.86 +
            0.18 * _clamp(shadow_summary.get("mean_portfolio_stability", 0.0), 0.0, 1.0) +
            0.10 * _clamp(shadow_summary.get("mean_portfolio_div", 0.0), 0.0, 1.0) -
            0.08 * _clamp(shadow_summary.get("mean_portfolio_corr", 0.0), 0.0, 1.0),
            0.40,
            1.60,
        )
        challenger_margin = _clamp(
            0.90 +
            0.40 * mean_stability +
            0.30 * _clamp(shadow_summary.get("mean_shadow_score", 0.0), 0.0, 1.0) -
            0.20 * _clamp(mean_route_regret, 0.0, 1.0),
            0.50,
            3.00,
        )
        regime_transition_weight = _clamp(
            0.22 +
            0.20 * _clamp(mean_macro_score / 100.0, 0.0, 1.0) +
            0.18 * _clamp(1.0 - mean_route_regret, 0.0, 1.0) +
            0.10 * _clamp(shadow_summary.get("mean_counterfactual", 0.0), 0.0, 1.0),
            0.0,
            1.0,
        )
        macro_quality_floor = _clamp(
            0.18 +
            0.30 * _clamp(mean_macro_score / 100.0, 0.0, 1.0) +
            0.08 * _clamp(shadow_summary.get("mean_shadow_score", 0.0), 0.0, 1.0),
            0.0,
            1.0,
        )
        supervisor_blend = _clamp(
            0.24 +
            0.16 * _clamp(shadow_summary.get("mean_portfolio_pressure", 0.0) / 1.5, 0.0, 1.0) +
            0.14 * _clamp(shadow_summary.get("mean_portfolio_supervisor_score", 0.0), 0.0, 1.0) +
            0.12 * _clamp(shadow_summary.get("mean_policy_no_trade_prob", 0.0), 0.0, 1.0),
            0.0,
            1.0,
        )

        payload = {
            "profile_name": args.profile,
            "symbol": symbol,
            "teacher_weight": teacher_weight,
            "student_weight": student_weight,
            "analog_weight": analog_weight,
            "foundation_weight": foundation_weight,
            "policy_trade_floor": policy_trade_floor,
            "policy_no_trade_cap": policy_no_trade_cap,
            "policy_size_bias": policy_size_bias,
            "portfolio_budget_bias": portfolio_budget_bias,
            "challenger_promote_margin": challenger_margin,
            "regime_transition_weight": regime_transition_weight,
            "macro_quality_floor": macro_quality_floor,
            "capital_efficiency_bias": capital_efficiency_bias,
            "supervisor_blend": supervisor_blend,
            "champions": deployment_champions,
            "shadow_summary": shadow_summary,
        }

        tsv_path = COMMON_PROMOTION_DIR / f"fxai_live_deploy_{safe_token(symbol)}.tsv"
        lines = [
            ("profile_name", args.profile),
            ("symbol", symbol),
            ("teacher_weight", f"{teacher_weight:.6f}"),
            ("student_weight", f"{student_weight:.6f}"),
            ("analog_weight", f"{analog_weight:.6f}"),
            ("foundation_weight", f"{foundation_weight:.6f}"),
            ("policy_trade_floor", f"{policy_trade_floor:.6f}"),
            ("policy_no_trade_cap", f"{policy_no_trade_cap:.6f}"),
            ("policy_size_bias", f"{policy_size_bias:.6f}"),
            ("portfolio_budget_bias", f"{portfolio_budget_bias:.6f}"),
            ("challenger_promote_margin", f"{challenger_margin:.6f}"),
            ("regime_transition_weight", f"{regime_transition_weight:.6f}"),
            ("macro_quality_floor", f"{macro_quality_floor:.6f}"),
            ("capital_efficiency_bias", f"{capital_efficiency_bias:.6f}"),
            ("supervisor_blend", f"{supervisor_blend:.6f}"),
        ]
        tsv_path.write_text("".join(f"{key}\t{value}\n" for key, value in lines), encoding="utf-8")
        tsv_sha = testlab.sha256_path(tsv_path)
        json_path = out_dir / f"live_deploy_{safe_token(symbol)}.json"
        json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

        conn.execute(
            """
            INSERT INTO live_deployment_profiles(profile_name, symbol, deployment_scope, artifact_path, artifact_sha256,
                                                 teacher_weight, student_weight, analog_weight, foundation_weight,
                                                 policy_trade_floor, policy_size_bias, portfolio_budget_bias,
                                                 challenger_promote_margin, regime_transition_weight, macro_quality_floor,
                                                 policy_no_trade_cap, capital_efficiency_bias, supervisor_blend,
                                                 payload_json, created_at)
            VALUES(?, ?, 'symbol', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(profile_name, symbol, deployment_scope) DO UPDATE SET
                artifact_path=excluded.artifact_path,
                artifact_sha256=excluded.artifact_sha256,
                teacher_weight=excluded.teacher_weight,
                student_weight=excluded.student_weight,
                analog_weight=excluded.analog_weight,
                foundation_weight=excluded.foundation_weight,
                policy_trade_floor=excluded.policy_trade_floor,
                policy_size_bias=excluded.policy_size_bias,
                portfolio_budget_bias=excluded.portfolio_budget_bias,
                challenger_promote_margin=excluded.challenger_promote_margin,
                regime_transition_weight=excluded.regime_transition_weight,
                macro_quality_floor=excluded.macro_quality_floor,
                policy_no_trade_cap=excluded.policy_no_trade_cap,
                capital_efficiency_bias=excluded.capital_efficiency_bias,
                supervisor_blend=excluded.supervisor_blend,
                payload_json=excluded.payload_json,
                created_at=excluded.created_at
            """,
            (
                args.profile,
                symbol,
                str(tsv_path),
                tsv_sha,
                teacher_weight,
                student_weight,
                analog_weight,
                foundation_weight,
                policy_trade_floor,
                policy_size_bias,
                portfolio_budget_bias,
                challenger_margin,
                regime_transition_weight,
                macro_quality_floor,
                policy_no_trade_cap,
                capital_efficiency_bias,
                supervisor_blend,
                json.dumps(payload, indent=2, sort_keys=True),
                created_at,
            ),
        )

        conn.execute(
            """
            UPDATE teacher_factories
               SET deployment_profile_path = ?, deployment_profile_sha256 = ?
             WHERE profile_name = ? AND symbol = ?
            """,
            (str(tsv_path), tsv_sha, args.profile, symbol),
        )
        deployments.append({
            "symbol": symbol,
            "artifact_path": str(tsv_path),
            "artifact_sha256": tsv_sha,
            "teacher_weight": teacher_weight,
            "student_weight": student_weight,
            "analog_weight": analog_weight,
            "foundation_weight": foundation_weight,
            "policy_no_trade_cap": policy_no_trade_cap,
            "capital_efficiency_bias": capital_efficiency_bias,
            "supervisor_blend": supervisor_blend,
        })

    stale_symbols = sorted(set(existing_symbols) - set(by_symbol))
    for symbol in stale_symbols:
        artifact_raw = str(existing_symbols.get(symbol, "") or "").strip()
        artifact_path = Path(artifact_raw) if artifact_raw else None
        if artifact_path is not None and artifact_path.exists() and artifact_path.is_file():
            artifact_path.unlink()
        stale_json = out_dir / f"live_deploy_{safe_token(symbol)}.json"
        if stale_json.exists():
            stale_json.unlink()
        conn.execute(
            "DELETE FROM live_deployment_profiles WHERE profile_name = ? AND symbol = ?",
            (args.profile, symbol),
        )
        conn.execute(
            """
            UPDATE teacher_factories
               SET deployment_profile_path = '', deployment_profile_sha256 = ''
             WHERE profile_name = ? AND symbol = ?
            """,
            (args.profile, symbol),
        )

    conn.commit()
    summary_path = out_dir / "live_deployments.json"
    summary_path.write_text(json.dumps(deployments, indent=2, sort_keys=True), encoding="utf-8")
    return deployments
