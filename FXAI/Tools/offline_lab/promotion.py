from __future__ import annotations

import json
import shutil
from collections import defaultdict
from pathlib import Path
import libsql

from .common import *


EA_SET_LINE_DEFAULTS: dict[str, tuple[object, object, object, str]] = {
    "AI_Type": (0, 0, 31, "N"),
    "AI_Ensemble": (False, 0, True, "N"),
    "AI_M1SyncBars": (3, 1, 12, "N"),
    "PredictionTargetMinutes": (5, 1, 720, "N"),
    "AI_MultiHorizon": (True, 0, True, "N"),
    "AI_Horizons": (0, 0, 0, "N"),
    "AI_FeatureNormalization": (0, 0, 16, "N"),
    "AI_ExecutionProfile": (0, 0, 4, "N"),
    "AI_CommissionPerLotSide": (0, 0, 100, "N"),
    "AI_CostBufferPoints": (2, 0, 100, "N"),
    "AI_ExecutionSlippageOverride": (-1, -1, 100, "N"),
    "AI_ExecutionFillPenaltyOverride": (-1, -1, 100, "N"),
    "TradeKiller": (0, 0, 10000, "N"),
}


def execution_profile_enum(name: str) -> int:
    profile = (name or "default").strip().lower()
    mapping = {
        "default": 0,
        "tight-fx": 1,
        "prime-ecn": 2,
        "retail-fx": 3,
        "stress": 4,
    }
    return int(mapping.get(profile, 0))


def _portable_artifact_path(value: str | Path) -> str:
    raw = str(value or "")
    if not raw:
        return raw
    candidate = Path(raw)
    replacements = (
        (testlab.TESTER_PRESET_DIR, "<FXAI_TESTER_PRESET_DIR>"),
        (COMMON_PROMOTION_DIR, "<FXAI_PROMOTION_DIR>"),
        (testlab.ROOT, "<FXAI_ROOT>"),
    )
    for base, token in replacements:
        try:
            relative = candidate.relative_to(base)
        except ValueError:
            continue
        return token if str(relative) == "." else f"{token}/{relative.as_posix()}"
    return raw


def _portableize_rows(rows: list[dict], *, path_keys: set[str]) -> list[dict]:
    portable_rows: list[dict] = []
    for row in rows:
        converted = dict(row)
        for key in path_keys:
            if key in converted and converted[key]:
                converted[key] = _portable_artifact_path(converted[key])
        portable_rows.append(converted)
    return portable_rows


def compile_strategy_profile_for_row(row: dict, params: dict) -> dict:
    return testlab.compile_strategy_profile(
        strategy_profile=str(params.get("strategy_profile", "default") or "default"),
        symbol=str(row["symbol"]),
        broker_profile=str(params.get("broker_profile", "") or ""),
        server=str(params.get("server", "") or ""),
        runtime_mode=str(params.get("runtime_mode", "research") or "research"),
        overrides=params,
        plugin_name=str(row["plugin_name"]),
        ai_id=int(row["ai_id"]),
    )


def write_ea_set(path: Path, row: dict, compiled_profile: dict) -> None:
    content = testlab.render_mt5_set(compiled_profile, line_defaults=EA_SET_LINE_DEFAULTS)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def write_audit_set_generic(path: Path, row: dict, compiled_profile: dict) -> None:
    audit_values = dict(compiled_profile["compiled"]["audit_values"])
    scenario_items = list(audit_values.get("scenario_list") or parse_csv_tokens(SERIOUS_SCENARIOS))
    ns = argparse.Namespace(
        all_plugins=False,
        plugin_id=int(row["ai_id"]),
        plugin_list="{" + str(row["plugin_name"]) + "}",
        scenario_list="{" + ", ".join(scenario_items) + "}",
        bars=int(audit_values.get("bars", 20000)),
        horizon=int(audit_values.get("horizon", 5)),
        m1sync_bars=int(audit_values.get("m1sync_bars", 3)),
        normalization=int(audit_values.get("normalization", 0)),
        sequence_bars=int(audit_values.get("sequence_bars", 0)),
        schema_id=int(audit_values.get("schema_id", 0)),
        feature_mask=int(audit_values.get("feature_mask", 0)),
        commission_per_lot_side=float(audit_values.get("commission_per_lot_side", 0.0)),
        cost_buffer_points=float(audit_values.get("cost_buffer_points", 2.0)),
        slippage_points=float(audit_values.get("slippage_points", 0.0)),
        fill_penalty_points=float(audit_values.get("fill_penalty_points", 0.0)),
        wf_train_bars=int(audit_values.get("wf_train_bars", 256)),
        wf_test_bars=int(audit_values.get("wf_test_bars", 64)),
        wf_purge_bars=int(audit_values.get("wf_purge_bars", 32)),
        wf_embargo_bars=int(audit_values.get("wf_embargo_bars", 24)),
        wf_folds=int(audit_values.get("wf_folds", 6)),
        window_start_unix=int(audit_values.get("window_start_unix", 0)),
        window_end_unix=int(audit_values.get("window_end_unix", 0)),
        seed=int(audit_values.get("seed", 42)),
        output="",
        compare_output=None,
        symbol=str(row["symbol"]),
        symbol_list="{" + str(row["symbol"]) + "}",
        symbol_pack="",
        execution_profile=str(audit_values.get("execution_profile", "default")),
        login=None,
        server=None,
        password=None,
        timeout=180,
        baseline=None,
        skip_compile=True,
    )
    testlab.write_audit_set(path, ns)


def load_completed_runs(conn: libsql.Connection, args) -> list[dict]:
    clauses = ["tr.status = 'ok'", "tr.profile_name = ?"]
    params: list[object] = [args.profile]
    dataset_keys = parse_csv_tokens(getattr(args, "dataset_keys", ""))
    group_key = (getattr(args, "group_key", "") or "").strip()
    symbols = resolve_symbols(args)
    if dataset_keys:
        clauses.append("d.dataset_key IN (%s)" % ",".join("?" for _ in dataset_keys))
        params.extend(dataset_keys)
    elif group_key:
        clauses.append("tr.group_key = ?")
        params.append(group_key)
    if symbols:
        clauses.append("tr.symbol IN (%s)" % ",".join("?" for _ in symbols))
        params.extend(symbols)
    sql = f"""
        SELECT tr.*, d.dataset_key, d.group_key, d.months, d.start_unix, d.end_unix
        FROM tuning_runs tr
        JOIN datasets d ON d.id = tr.dataset_id
        WHERE {' AND '.join(clauses)}
        ORDER BY tr.symbol, tr.plugin_name, tr.score DESC, tr.finished_at DESC
    """
    return query_all(conn, sql, params)


def aggregate_best_candidates(rows: list[dict]) -> tuple[list[dict], dict[str, int]]:
    grouped: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    dataset_counts: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        grouped[(row["symbol"], row["plugin_name"], row["parameters_json"])].append(row)
        dataset_counts[row["symbol"]].add(row["dataset_key"])

    winners: list[dict] = []
    total_datasets_per_symbol = {symbol: len(keys) for symbol, keys in dataset_counts.items()}
    per_symbol_plugin: dict[tuple[str, str], list[dict]] = defaultdict(list)

    for (symbol, plugin_name, _param_hash), group in grouped.items():
        weights = [math.sqrt(max(int(item.get("months", 1)) or 1, 1)) for item in group]
        weight_sum = sum(weights) or 1.0
        mean_score = sum(float(item["score"]) * w for item, w in zip(group, weights)) / weight_sum
        min_score = min(float(item["score"]) for item in group)
        mean_recent = sum(float(item["market_recent_score"]) * w for item, w in zip(group, weights)) / weight_sum
        mean_wf = sum(float(item["walkforward_score"]) * w for item, w in zip(group, weights)) / weight_sum
        mean_adv = sum(float(item["adversarial_score"]) * w for item, w in zip(group, weights)) / weight_sum
        mean_macro = sum(float(item["macro_event_score"]) * w for item, w in zip(group, weights)) / weight_sum
        mean_issues = sum(float(item["issue_count"]) * w for item, w in zip(group, weights)) / weight_sum
        support = len({item["dataset_key"] for item in group})
        coverage = float(support) / float(max(total_datasets_per_symbol.get(symbol, 1), 1))
        ranking = (
            0.48 * mean_score +
            0.18 * min_score +
            0.12 * mean_wf +
            0.10 * mean_adv +
            0.05 * mean_recent +
            0.03 * mean_macro +
            4.0 * coverage -
            0.75 * mean_issues
        )
        best_row = max(group, key=lambda item: (float(item["score"]), -float(item["issue_count"]), float(item["adversarial_score"]), float(item["walkforward_score"])))
        aggregated = {
            "symbol": symbol,
            "plugin_name": plugin_name,
            "ai_id": int(best_row["ai_id"]),
            "family_id": int(best_row.get("family_id", 11)),
            "run_id": int(best_row["id"]),
            "score": mean_score,
            "ranking_score": ranking,
            "support_count": support,
            "support_json": json.dumps([
                {
                    "dataset_key": item["dataset_key"],
                    "months": int(item.get("months", 0)),
                    "score": float(item["score"]),
                    "walkforward_score": float(item["walkforward_score"]),
                    "adversarial_score": float(item["adversarial_score"]),
                }
                for item in sorted(group, key=lambda x: (int(x.get("months", 0)), x["dataset_key"]))
            ], indent=2, sort_keys=True),
            "parameters_json": best_row["parameters_json"],
            "dataset_scope": "aggregate",
            "dataset_id": None,
        }
        per_symbol_plugin[(symbol, plugin_name)].append(aggregated)

    for (symbol, plugin_name), candidates in per_symbol_plugin.items():
        winner = max(
            candidates,
            key=lambda item: (
                float(item["ranking_score"]),
                float(item["score"]),
                int(item["support_count"]),
            ),
        )
        winners.append(winner)
    return winners, total_datasets_per_symbol


def render_family_scorecards(conn: libsql.Connection,
                             profile_name: str,
                             group_key: str,
                             rows: list[dict],
                             promoted_rows: list[dict]) -> list[dict]:
    promoted_by_family: dict[tuple[str, int], int] = defaultdict(int)
    for row in promoted_rows:
        promoted_by_family[(str(row["symbol"]), int(row.get("family_id", 11)))] += 1

    champion_rows = query_all(
        conn,
        "SELECT symbol, family_id, COUNT(*) AS champion_count "
        "FROM champion_registry WHERE profile_name = ? AND status = 'champion' "
        "GROUP BY symbol, family_id",
        (profile_name,),
    )
    champion_by_family = {
        (str(row["symbol"]), int(row["family_id"])): int(row["champion_count"])
        for row in champion_rows
    }

    grouped: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["symbol"]), int(row.get("family_id", 11)))].append(row)

    scorecards: list[dict] = []
    for (symbol, family_id), items in sorted(grouped.items()):
        scores = [float(item["score"]) for item in items]
        mean_score, score_std = mean_std(scores)
        stability = 1.0
        if mean_score > 1e-9:
            stability = max(0.0, 1.0 - min(score_std / max(mean_score, 10.0), 1.0))
        mean_recent = sum(float(item["market_recent_score"]) for item in items) / float(len(items))
        mean_wf = sum(float(item["walkforward_score"]) for item in items) / float(len(items))
        mean_adv = sum(float(item["adversarial_score"]) for item in items) / float(len(items))
        mean_macro = sum(float(item["macro_event_score"]) for item in items) / float(len(items))
        mean_issues = sum(float(item["issue_count"]) for item in items) / float(len(items))
        top_plugins = []
        ranked = sorted(
            items,
            key=lambda item: (float(item["score"]), float(item["walkforward_score"]), -float(item["issue_count"])),
            reverse=True,
        )
        seen_plugins = set()
        for item in ranked:
            name = str(item["plugin_name"])
            if name in seen_plugins:
                continue
            seen_plugins.add(name)
            top_plugins.append({
                "plugin_name": name,
                "score": float(item["score"]),
                "walkforward_score": float(item["walkforward_score"]),
                "adversarial_score": float(item["adversarial_score"]),
                "macro_event_score": float(item["macro_event_score"]),
            })
            if len(top_plugins) >= 5:
                break
        payload = {
            "symbol": symbol,
            "family_id": family_id,
            "family_name": plugin_family_name(family_id),
            "run_count": len(items),
            "score_std": score_std,
            "top_plugins": top_plugins,
        }
        scorecards.append({
            "profile_name": profile_name,
            "group_key": group_key,
            "symbol": symbol,
            "family_id": family_id,
            "family_name": plugin_family_name(family_id),
            "run_count": len(items),
            "mean_score": mean_score,
            "mean_recent_score": mean_recent,
            "mean_walkforward_score": mean_wf,
            "mean_adversarial_score": mean_adv,
            "mean_macro_score": mean_macro,
            "mean_issue_count": mean_issues,
            "stability_score": stability,
            "promotion_count": promoted_by_family.get((symbol, family_id), 0),
            "champion_count": champion_by_family.get((symbol, family_id), 0),
            "payload_json": json.dumps(payload, indent=2, sort_keys=True),
        })
    return scorecards


def persist_family_scorecards(conn: libsql.Connection,
                              args,
                              rows: list[dict],
                              promoted_rows: list[dict]) -> list[dict]:
    group_key = (getattr(args, "group_key", "") or "").strip()
    scorecards = render_family_scorecards(conn, args.profile, group_key, rows, promoted_rows)
    out_dir = RESEARCH_DIR / safe_token(args.profile)
    ensure_dir(out_dir)
    now_ts = now_unix()
    for row in scorecards:
        conn.execute(
            """
            INSERT INTO family_scorecards(profile_name, group_key, symbol, family_id, family_name,
                                         run_count, mean_score, mean_recent_score, mean_walkforward_score,
                                         mean_adversarial_score, mean_macro_score, mean_issue_count,
                                         stability_score, promotion_count, champion_count, payload_json, created_at)
            VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(profile_name, group_key, symbol, family_id) DO UPDATE SET
                family_name=excluded.family_name,
                run_count=excluded.run_count,
                mean_score=excluded.mean_score,
                mean_recent_score=excluded.mean_recent_score,
                mean_walkforward_score=excluded.mean_walkforward_score,
                mean_adversarial_score=excluded.mean_adversarial_score,
                mean_macro_score=excluded.mean_macro_score,
                mean_issue_count=excluded.mean_issue_count,
                stability_score=excluded.stability_score,
                promotion_count=excluded.promotion_count,
                champion_count=excluded.champion_count,
                payload_json=excluded.payload_json,
                created_at=excluded.created_at
            """,
            (
                row["profile_name"],
                row["group_key"],
                row["symbol"],
                int(row["family_id"]),
                row["family_name"],
                int(row["run_count"]),
                float(row["mean_score"]),
                float(row["mean_recent_score"]),
                float(row["mean_walkforward_score"]),
                float(row["mean_adversarial_score"]),
                float(row["mean_macro_score"]),
                float(row["mean_issue_count"]),
                float(row["stability_score"]),
                int(row["promotion_count"]),
                int(row["champion_count"]),
                row["payload_json"],
                now_ts,
            ),
        )
    commit_db(conn)

    score_json = out_dir / "family_scorecards.json"
    score_tsv = out_dir / "family_scorecards.tsv"
    score_md = out_dir / "family_scorecards.md"
    score_json.write_text(json.dumps(scorecards, indent=2, sort_keys=True), encoding="utf-8")
    with score_tsv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow([
            "symbol", "family_id", "family_name", "run_count", "mean_score", "mean_walkforward_score",
            "mean_adversarial_score", "mean_macro_score", "stability_score", "promotion_count", "champion_count",
        ])
        for row in scorecards:
            writer.writerow([
                row["symbol"],
                row["family_id"],
                row["family_name"],
                row["run_count"],
                f"{float(row['mean_score']):.4f}",
                f"{float(row['mean_walkforward_score']):.4f}",
                f"{float(row['mean_adversarial_score']):.4f}",
                f"{float(row['mean_macro_score']):.4f}",
                f"{float(row['stability_score']):.4f}",
                row["promotion_count"],
                row["champion_count"],
            ])
    md_lines = ["# FXAI Family Scorecards", "", f"profile: {args.profile}", ""]
    for row in scorecards:
        md_lines.append(
            f"- {row['symbol']} | {row['family_name']} | score {float(row['mean_score']):.2f} | "
            f"wf {float(row['mean_walkforward_score']):.2f} | adv {float(row['mean_adversarial_score']):.2f} | "
            f"macro {float(row['mean_macro_score']):.2f} | stability {float(row['stability_score']):.2f} | "
            f"promoted {int(row['promotion_count'])} | champions {int(row['champion_count'])}"
        )
    score_md.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    return scorecards


def persist_lineage_entry(conn: libsql.Connection,
                          profile_name: str,
                          symbol: str,
                          plugin_name: str,
                          family_id: int,
                          source_run_id: int,
                          best_config_id: int,
                          relation: str,
                          payload: dict) -> None:
    lineage_hash = sha256_text(
        f"{profile_name}|{symbol}|{plugin_name}|{family_id}|{source_run_id}|{best_config_id}|{relation}|{json_compact(payload)}"
    )
    conn.execute(
        """
        INSERT INTO config_lineage(profile_name, symbol, plugin_name, family_id, source_run_id, best_config_id, relation, lineage_hash, payload_json, created_at)
        VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            profile_name,
            symbol,
            plugin_name,
            int(family_id),
            int(source_run_id) if source_run_id else None,
            int(best_config_id) if best_config_id else None,
            relation,
            lineage_hash,
            json.dumps(payload, indent=2, sort_keys=True),
            now_unix(),
        ),
    )


def write_strategy_profile_manifest_file(path: Path,
                                         compiled_profile: dict,
                                         *,
                                         artifact_kind: str,
                                         artifact_reference: str | Path,
                                         metadata: dict | None = None) -> tuple[str, str]:
    payload = testlab.build_strategy_profile_manifest(
        compiled_profile,
        artifact_kind=artifact_kind,
        artifact_path=artifact_reference,
        metadata=metadata or {},
    )
    testlab.write_json(path, payload)
    return str(path), testlab.sha256_path(path)


def copy_artifact_if_exists(src: str | Path, dst: str | Path) -> None:
    src_path = Path(str(src))
    if src_path.exists():
        dst_path = Path(str(dst))
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_path, dst_path)


def update_champion_registry(conn: libsql.Connection,
                             args,
                             promoted_rows: list[dict]) -> list[dict]:
    profile_dir = PROFILES_DIR / safe_token(args.profile)
    ensure_dir(profile_dir)
    decisions: list[dict] = []
    for row in promoted_rows:
        symbol = str(row["symbol"])
        plugin_name = str(row["plugin_name"])
        family_id = int(row.get("family_id", 11))
        candidate_rank = float(row["ranking_score"])
        candidate_score = float(row["score"])
        registry = query_one(
            conn,
            "SELECT * FROM champion_registry WHERE profile_name = ? AND symbol = ? AND plugin_name = ?",
            (args.profile, symbol, plugin_name),
        )
        best_cfg = query_one(
            conn,
            "SELECT id FROM best_configs WHERE profile_name = ? AND dataset_scope = 'aggregate' AND symbol = ? AND plugin_name = ?",
            (args.profile, symbol, plugin_name),
        )
        best_config_id = int(best_cfg["id"]) if best_cfg else 0
        champion_dir = profile_dir / safe_token(symbol)
        ensure_dir(champion_dir)
        champion_audit = champion_dir / f"{plugin_name}__champion__audit.set"
        champion_ea = champion_dir / f"{plugin_name}__champion__ea.set"
        champion_strategy_manifest = champion_dir / f"{plugin_name}__champion__strategy_profile.json"

        promote = False
        note = ""
        if registry is None or int(registry.get("champion_best_config_id") or 0) <= 0:
            promote = True
            note = "bootstrap_champion"
        else:
            current_rank = float(registry.get("champion_score", 0.0))
            current_portfolio = float(registry.get("portfolio_score", 0.0))
            challenger_portfolio = 0.65 * candidate_rank + 0.35 * float(row.get("support_count", 0))
            if candidate_rank > current_rank + 1.10 or challenger_portfolio > current_portfolio + 0.90:
                promote = True
                note = "challenger_promoted"
            else:
                note = "challenger_held_out"

        if promote:
            shutil.copy2(row["audit_set_path"], champion_audit)
            shutil.copy2(row["ea_set_path"], champion_ea)
            copy_artifact_if_exists(row.get("strategy_profile_manifest_path", ""), champion_strategy_manifest)
            conn.execute(
                """
                INSERT INTO champion_registry(profile_name, symbol, plugin_name, family_id, champion_best_config_id, challenger_run_id,
                                              status, champion_score, challenger_score, portfolio_score, promoted_at, reviewed_at,
                                              champion_set_path, notes)
                VALUES(?, ?, ?, ?, ?, ?, 'champion', ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(profile_name, symbol, plugin_name) DO UPDATE SET
                    family_id=excluded.family_id,
                    champion_best_config_id=excluded.champion_best_config_id,
                    challenger_run_id=excluded.challenger_run_id,
                    status=excluded.status,
                    champion_score=excluded.champion_score,
                    challenger_score=excluded.challenger_score,
                    portfolio_score=excluded.portfolio_score,
                    promoted_at=excluded.promoted_at,
                    reviewed_at=excluded.reviewed_at,
                    champion_set_path=excluded.champion_set_path,
                    notes=excluded.notes
                """,
                (
                    args.profile,
                    symbol,
                    plugin_name,
                    family_id,
                    best_config_id,
                    int(row["run_id"]),
                    candidate_rank,
                    candidate_score,
                    0.65 * candidate_rank + 0.35 * float(row.get("support_count", 0)),
                    now_unix(),
                    now_unix(),
                    str(champion_ea),
                    note,
                ),
            )
            persist_lineage_entry(
                conn,
                args.profile,
                symbol,
                plugin_name,
                family_id,
                int(row["run_id"]),
                best_config_id,
                "champion",
                {"note": note, "ranking_score": candidate_rank, "score": candidate_score},
            )
        else:
            conn.execute(
                """
                INSERT INTO champion_registry(profile_name, symbol, plugin_name, family_id, champion_best_config_id, challenger_run_id,
                                              status, champion_score, challenger_score, portfolio_score, promoted_at, reviewed_at,
                                              champion_set_path, notes)
                VALUES(?, ?, ?, ?, ?, ?, 'champion', ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(profile_name, symbol, plugin_name) DO UPDATE SET
                    family_id=excluded.family_id,
                    challenger_run_id=excluded.challenger_run_id,
                    challenger_score=excluded.challenger_score,
                    reviewed_at=excluded.reviewed_at,
                    notes=excluded.notes
                """,
                (
                    args.profile,
                    symbol,
                    plugin_name,
                    family_id,
                    int(registry.get("champion_best_config_id", 0)),
                    int(row["run_id"]),
                    float(registry.get("champion_score", 0.0)),
                    candidate_score,
                    float(registry.get("portfolio_score", 0.0)),
                    int(registry.get("promoted_at", 0)),
                    now_unix(),
                    str(registry.get("champion_set_path", "")),
                    note,
                ),
            )
            persist_lineage_entry(
                conn,
                args.profile,
                symbol,
                plugin_name,
                family_id,
                int(row["run_id"]),
                best_config_id,
                "challenger",
                {"note": note, "ranking_score": candidate_rank, "score": candidate_score},
            )

        decisions.append({
            "symbol": symbol,
            "plugin_name": plugin_name,
            "family_id": family_id,
            "status": ("champion" if promote else "challenger"),
            "note": note,
            "ranking_score": candidate_rank,
            "score": candidate_score,
        })

    commit_db(conn)

    champion_rows = query_all(
        conn,
        "SELECT * FROM champion_registry WHERE profile_name = ? AND status = 'champion' ORDER BY symbol, champion_score DESC",
        (args.profile,),
    )
    champion_rows_dict = [dict(row) for row in champion_rows]
    by_symbol: dict[str, list[dict]] = defaultdict(list)
    for row in champion_rows_dict:
        by_symbol[str(row["symbol"])].append(row)
    for symbol, items in by_symbol.items():
        top = max(items, key=lambda item: (float(item["champion_score"]), float(item["portfolio_score"])))
        src_ea = Path(str(top["champion_set_path"]))
        src_audit = src_ea.with_name(src_ea.name.replace("__ea.set", "__audit.set"))
        src_strategy_manifest = src_ea.with_name(src_ea.name.replace("__ea.set", "__strategy_profile.json"))
        dst_dir = profile_dir / safe_token(symbol)
        ensure_dir(dst_dir)
        if src_audit.exists():
            shutil.copy2(src_audit, dst_dir / "__TOP__audit.set")
            shutil.copy2(src_audit, testlab.TESTER_PRESET_DIR / f"fxai_offline_{safe_token(args.profile)}__{safe_token(symbol)}__top__audit.set")
        if src_ea.exists():
            shutil.copy2(src_ea, dst_dir / "__TOP__ea.set")
            shutil.copy2(src_ea, testlab.TESTER_PRESET_DIR / f"fxai_offline_{safe_token(args.profile)}__{safe_token(symbol)}__top__ea.set")
        copy_artifact_if_exists(src_strategy_manifest, dst_dir / "__TOP__strategy_profile.json")
        copy_artifact_if_exists(
            src_strategy_manifest,
            testlab.TESTER_PRESET_DIR / f"fxai_offline_{safe_token(args.profile)}__{safe_token(symbol)}__top__strategy_profile.json",
        )

    for row in champion_rows_dict:
        champion_set_path = str(row.get("champion_set_path", "") or "")
        if champion_set_path.endswith("__ea.set"):
            row["strategy_profile_manifest_path"] = champion_set_path.replace("__ea.set", "__strategy_profile.json")
    summary_rows = _portableize_rows(champion_rows_dict, path_keys={"champion_set_path", "strategy_profile_manifest_path"})
    summary_path = RESEARCH_DIR / safe_token(args.profile) / "champions.json"
    ensure_dir(summary_path.parent)
    summary_path.write_text(json.dumps(summary_rows, indent=2, sort_keys=True), encoding="utf-8")
    return decisions


def write_distillation_artifacts(conn: libsql.Connection,
                                 args,
                                 promoted_rows: list[dict]) -> list[dict]:
    out_dir = DISTILL_DIR / safe_token(args.profile)
    ensure_dir(out_dir)
    artifacts: list[dict] = []
    created_at = now_unix()
    for row in promoted_rows:
        params = json.loads(row["parameters_json"])
        family_id = int(row.get("family_id", 11))
        distill_profile = family_distillation_profile(family_id)
        support_items = json.loads(row.get("support_json", "[]") or "[]")
        teacher_summary = {
            "plugin_name": row["plugin_name"],
            "symbol": row["symbol"],
            "ai_id": int(row["ai_id"]),
            "family_id": family_id,
            "family_name": plugin_family_name(family_id),
            "ranking_score": float(row["ranking_score"]),
            "score": float(row["score"]),
            "support_count": int(row["support_count"]),
            "support": support_items,
            "parameters": params,
        }
        student_target = dict(distill_profile)
        student_target.update({
            "target_horizon": int(params.get("horizon", 5)),
            "target_execution_profile": str(params.get("execution_profile", "default")),
            "target_sequence_bars": int(params.get("sequence_bars", 0)),
            "target_reliability_floor": round(max(0.42, min(0.86, 0.42 + 0.0035 * float(row["score"]))), 4),
            "target_trade_gate_floor": round(max(0.44, min(0.88, 0.44 + 0.0030 * float(row["ranking_score"]))), 4),
            "support_weight_floor": round(max(0.20, min(0.90, 0.18 + 0.08 * int(row["support_count"]))), 4),
        })
        symbol_dir = out_dir / safe_token(str(row["symbol"]))
        ensure_dir(symbol_dir)
        artifact_path = symbol_dir / f"{row['plugin_name']}__distill.json"
        payload = {
            "teacher_summary": teacher_summary,
            "student_target": student_target,
            "strategy_profile_manifest_path": _portable_artifact_path(str(row.get("strategy_profile_manifest_path", "") or "")),
        }
        artifact_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        artifact_sha = testlab.sha256_path(artifact_path)
        best_cfg = query_one(
            conn,
            "SELECT id FROM best_configs WHERE profile_name = ? AND dataset_scope = 'aggregate' AND symbol = ? AND plugin_name = ?",
            (args.profile, row["symbol"], row["plugin_name"]),
        )
        best_config_id = int(best_cfg["id"]) if best_cfg else 0
        conn.execute(
            """
            INSERT INTO distillation_artifacts(profile_name, symbol, plugin_name, family_id, source_run_id, best_config_id,
                                               dataset_scope, artifact_path, artifact_sha256, teacher_summary_json,
                                               student_target_json, status, created_at)
            VALUES(?, ?, ?, ?, ?, ?, 'aggregate', ?, ?, ?, ?, 'ready', ?)
            ON CONFLICT(profile_name, symbol, plugin_name, dataset_scope) DO UPDATE SET
                family_id=excluded.family_id,
                source_run_id=excluded.source_run_id,
                best_config_id=excluded.best_config_id,
                artifact_path=excluded.artifact_path,
                artifact_sha256=excluded.artifact_sha256,
                teacher_summary_json=excluded.teacher_summary_json,
                student_target_json=excluded.student_target_json,
                status=excluded.status,
                created_at=excluded.created_at
            """,
            (
                args.profile,
                row["symbol"],
                row["plugin_name"],
                family_id,
                int(row["run_id"]),
                best_config_id,
                str(artifact_path),
                artifact_sha,
                json.dumps(teacher_summary, indent=2, sort_keys=True),
                json.dumps(student_target, indent=2, sort_keys=True),
                created_at,
            ),
        )
        artifacts.append({
            "symbol": row["symbol"],
            "plugin_name": row["plugin_name"],
            "family_id": family_id,
            "artifact_path": _portable_artifact_path(artifact_path),
            "artifact_sha256": artifact_sha,
            "strategy_profile_manifest_path": _portable_artifact_path(str(row.get("strategy_profile_manifest_path", "") or "")),
        })
    commit_db(conn)
    summary = out_dir / "distillation_artifacts.json"
    summary.write_text(json.dumps(artifacts, indent=2, sort_keys=True), encoding="utf-8")
    return artifacts


def persist_best_configs(conn: libsql.Connection, args, winners: list[dict]) -> list[dict]:
    profile_dir = PROFILES_DIR / safe_token(args.profile)
    ensure_dir(profile_dir)
    ensure_dir(COMMON_PROMOTION_DIR)
    ensure_dir(testlab.TESTER_PRESET_DIR)

    promoted_rows = []
    by_symbol: dict[str, list[dict]] = defaultdict(list)
    now_ts = now_unix()

    for winner in sorted(winners, key=lambda item: (item["symbol"], -float(item["ranking_score"]), item["plugin_name"])):
        params = json.loads(winner["parameters_json"])
        compiled_profile = compile_strategy_profile_for_row(winner, params)
        symbol_dir = profile_dir / safe_token(winner["symbol"])
        ensure_dir(symbol_dir)
        audit_set_path = symbol_dir / f"{winner['plugin_name']}__audit.set"
        ea_set_path = symbol_dir / f"{winner['plugin_name']}__ea.set"
        strategy_manifest_path = symbol_dir / f"{winner['plugin_name']}__strategy_profile.json"
        write_audit_set_generic(audit_set_path, winner, compiled_profile)
        write_ea_set(ea_set_path, winner, compiled_profile)
        manifest_written_path, manifest_sha = write_strategy_profile_manifest_file(
            strategy_manifest_path,
            compiled_profile,
            artifact_kind="promotion_profile",
            artifact_reference=ea_set_path,
            metadata={
                "profile_name": args.profile,
                "plugin_name": winner["plugin_name"],
                "symbol": winner["symbol"],
                "run_id": int(winner["run_id"]),
                "parameters_json": json.loads(winner["parameters_json"]),
            },
        )

        tester_audit_path = testlab.TESTER_PRESET_DIR / f"fxai_offline_{safe_token(args.profile)}__{safe_token(winner['symbol'])}__{winner['plugin_name']}__audit.set"
        tester_ea_path = testlab.TESTER_PRESET_DIR / f"fxai_offline_{safe_token(args.profile)}__{safe_token(winner['symbol'])}__{winner['plugin_name']}__ea.set"
        tester_strategy_manifest_path = testlab.TESTER_PRESET_DIR / f"fxai_offline_{safe_token(args.profile)}__{safe_token(winner['symbol'])}__{winner['plugin_name']}__strategy_profile.json"
        shutil.copy2(audit_set_path, tester_audit_path)
        shutil.copy2(ea_set_path, tester_ea_path)
        shutil.copy2(strategy_manifest_path, tester_strategy_manifest_path)

        conn.execute(
            """
            INSERT INTO best_configs(dataset_scope, dataset_id, profile_name, symbol, plugin_name, ai_id, family_id, run_id, promoted_at,
                                     score, ranking_score, support_count, parameters_json, audit_set_path, ea_set_path, support_json)
            VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(dataset_scope, profile_name, symbol, plugin_name) DO UPDATE SET
                dataset_id=excluded.dataset_id,
                ai_id=excluded.ai_id,
                family_id=excluded.family_id,
                run_id=excluded.run_id,
                promoted_at=excluded.promoted_at,
                score=excluded.score,
                ranking_score=excluded.ranking_score,
                support_count=excluded.support_count,
                parameters_json=excluded.parameters_json,
                audit_set_path=excluded.audit_set_path,
                ea_set_path=excluded.ea_set_path,
                support_json=excluded.support_json
            """,
            (
                winner["dataset_scope"],
                winner["dataset_id"],
                args.profile,
                winner["symbol"],
                winner["plugin_name"],
                int(winner["ai_id"]),
                int(winner.get("family_id", 11)),
                int(winner["run_id"]),
                now_ts,
                float(winner["score"]),
                float(winner["ranking_score"]),
                int(winner["support_count"]),
                winner["parameters_json"],
                str(audit_set_path),
                str(ea_set_path),
                winner["support_json"],
            ),
        )
        promoted = dict(winner)
        promoted["audit_set_path"] = str(audit_set_path)
        promoted["ea_set_path"] = str(ea_set_path)
        promoted["tester_audit_set_path"] = str(tester_audit_path)
        promoted["tester_ea_set_path"] = str(tester_ea_path)
        promoted["strategy_profile_manifest_path"] = manifest_written_path
        promoted["strategy_profile_manifest_sha256"] = manifest_sha
        promoted["tester_strategy_profile_manifest_path"] = str(tester_strategy_manifest_path)
        promoted["strategy_profile_id"] = compiled_profile["strategy_profile_id"]
        promoted["strategy_profile_version"] = int(compiled_profile["strategy_profile_version"])
        by_symbol[winner["symbol"]].append(promoted)
        promoted_rows.append(promoted)

    commit_db(conn)

    for symbol, rows in by_symbol.items():
        top = max(rows, key=lambda item: (float(item["ranking_score"]), float(item["score"])))
        top_audit_path = profile_dir / safe_token(symbol) / "__TOP__audit.set"
        top_ea_path = profile_dir / safe_token(symbol) / "__TOP__ea.set"
        top_strategy_manifest_path = profile_dir / safe_token(symbol) / "__TOP__strategy_profile.json"
        shutil.copy2(top["audit_set_path"], top_audit_path)
        shutil.copy2(top["ea_set_path"], top_ea_path)
        copy_artifact_if_exists(top.get("strategy_profile_manifest_path", ""), top_strategy_manifest_path)
        shutil.copy2(top["audit_set_path"], testlab.TESTER_PRESET_DIR / f"fxai_offline_{safe_token(args.profile)}__{safe_token(symbol)}__top__audit.set")
        shutil.copy2(top["ea_set_path"], testlab.TESTER_PRESET_DIR / f"fxai_offline_{safe_token(args.profile)}__{safe_token(symbol)}__top__ea.set")
        copy_artifact_if_exists(
            top.get("strategy_profile_manifest_path", ""),
            testlab.TESTER_PRESET_DIR / f"fxai_offline_{safe_token(args.profile)}__{safe_token(symbol)}__top__strategy_profile.json",
        )

    summary_json = profile_dir / "promoted_best.json"
    summary_tsv = profile_dir / "promoted_best.tsv"
    summary_md = profile_dir / "promoted_best.md"
    portable_rows = _portableize_rows(
        promoted_rows,
        path_keys={
            "audit_set_path",
            "ea_set_path",
            "tester_audit_set_path",
            "tester_ea_set_path",
            "strategy_profile_manifest_path",
            "tester_strategy_profile_manifest_path",
        },
    )
    summary_json.write_text(json.dumps(portable_rows, indent=2, sort_keys=True), encoding="utf-8")
    with summary_tsv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow([
            "symbol", "plugin_name", "ai_id", "score", "ranking_score", "support_count",
            "audit_set_path", "ea_set_path", "strategy_profile_manifest_path",
        ])
        for row in sorted(portable_rows, key=lambda item: (item["symbol"], -float(item["ranking_score"]), item["plugin_name"])):
            writer.writerow([
                row["symbol"],
                row["plugin_name"],
                row["ai_id"],
                f"{float(row['score']):.4f}",
                f"{float(row['ranking_score']):.4f}",
                int(row["support_count"]),
                row["audit_set_path"],
                row["ea_set_path"],
                row["strategy_profile_manifest_path"],
            ])
    md_lines = ["# FXAI Offline Lab Promoted Best", "", f"profile: {args.profile}", ""]
    for symbol in sorted(by_symbol.keys()):
        md_lines.append(f"## {symbol}")
        ranked = sorted(by_symbol[symbol], key=lambda item: (float(item["ranking_score"]), float(item["score"])), reverse=True)
        for row in ranked[:8]:
            params = json.loads(row["parameters_json"])
            md_lines.append(
                f"- {row['plugin_name']} | score {float(row['score']):.2f} | rank {float(row['ranking_score']):.2f} | "
                f"H={int(params.get('horizon', 5))} | M1Sync={int(params.get('m1sync_bars', 3))} | "
                f"Norm={int(params.get('normalization', 0))} | Seq={int(params.get('sequence_bars', 0))} | "
                f"Schema={int(params.get('schema_id', 0))} | Mask={int(params.get('feature_mask', 0))} | "
                f"Strategy={row.get('strategy_profile_id', 'strategy/default')}@v{int(row.get('strategy_profile_version', 1))}"
            )
        md_lines.append("")
    summary_md.write_text("\n".join(md_lines), encoding="utf-8")

    common_json = COMMON_PROMOTION_DIR / f"fxai_offline_best_{safe_token(args.profile)}.json"
    common_tsv = COMMON_PROMOTION_DIR / f"fxai_offline_best_{safe_token(args.profile)}.tsv"
    shutil.copy2(summary_json, common_json)
    shutil.copy2(summary_tsv, common_tsv)
    return promoted_rows
