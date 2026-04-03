from __future__ import annotations

import argparse
import json
from pathlib import Path
import libsql

from .attribution import *
from .bundle import *
from .common import *
from .dashboard import live_state_snapshot, write_profile_dashboard
from .environment import bootstrap_environment, validate_environment, write_environment_report
from .exporter import *
from .fixtures import seed_profile_fixture
from .foundation_factory import *
from .governance import *
from .lineage import *
from .mode import RUNTIME_MODES, resolve_runtime_mode
from .performance import write_performance_reports
from .promotion import *
from .shadow_fleet import *
from .student_router import *
from .supervisor_service import *
from .teacher_factory import *
from .turso_platform import (
    branch_inventory,
    create_branch_database,
    destroy_database,
    persist_branch_result,
    resolve_platform_config,
    sync_audit_logs,
)
from .vector_store import latest_symbol_shadow_neighbors, refresh_research_vectors
from .verification import verify_deterministic_outputs

def serious_base_args(args, dataset: dict, output_path: Path) -> argparse.Namespace:
    bars = int(getattr(args, "bars", 0) or 0)
    if bars <= 0 or bars > int(dataset["bars"]):
        bars = int(dataset["bars"])
    return argparse.Namespace(
        all_plugins=False,
        plugin_id=28,
        plugin_list="{all}",
        scenario_list=getattr(args, "scenario_list", SERIOUS_SCENARIOS),
        bars=bars,
        horizon=getattr(args, "horizon", 5),
        m1sync_bars=getattr(args, "m1sync_bars", 3),
        normalization=getattr(args, "normalization", 0),
        sequence_bars=getattr(args, "sequence_bars", 0),
        schema_id=getattr(args, "schema_id", 0),
        feature_mask=getattr(args, "feature_mask", 0),
        commission_per_lot_side=getattr(args, "commission_per_lot_side", None),
        cost_buffer_points=getattr(args, "cost_buffer_points", None),
        slippage_points=getattr(args, "slippage_points", None),
        fill_penalty_points=getattr(args, "fill_penalty_points", None),
        wf_train_bars=getattr(args, "wf_train_bars", 256),
        wf_test_bars=getattr(args, "wf_test_bars", 64),
        wf_purge_bars=getattr(args, "wf_purge_bars", 32),
        wf_embargo_bars=getattr(args, "wf_embargo_bars", 24),
        wf_folds=getattr(args, "wf_folds", 6),
        seed=getattr(args, "seed", 42),
        symbol=str(dataset["symbol"]),
        symbol_list="{" + str(dataset["symbol"]) + "}",
        symbol_pack="",
        window_start_unix=int(dataset["start_unix"]),
        window_end_unix=int(dataset["end_unix"]),
        execution_profile=getattr(args, "execution_profile", "default"),
        login=getattr(args, "login", None),
        server=getattr(args, "server", None),
        password=getattr(args, "password", None),
        timeout=getattr(args, "timeout", 300),
        baseline=None,
        output=str(output_path),
        compare_output=None,
        skip_compile=True,
    )


def extend_campaign(campaign: dict, base_args) -> dict:
    horizon_base = max(int(getattr(base_args, "horizon", 5)), 1)
    m1sync_base = max(int(getattr(base_args, "m1sync_bars", 3)), 1)
    exec_base = str(getattr(base_args, "execution_profile", "default") or "default")
    for _name, info in campaign.get("plugins", {}).items():
        experiments = info.setdefault("experiments", [])
        horizon_candidates = []
        for candidate in [max(1, horizon_base - 2), horizon_base] + list(DEFAULT_HORIZON_CANDIDATES):
            if candidate not in horizon_candidates:
                horizon_candidates.append(candidate)
        m1sync_candidates = []
        for candidate in [m1sync_base] + list(DEFAULT_M1SYNC_CANDIDATES):
            if candidate not in m1sync_candidates:
                m1sync_candidates.append(candidate)
        execution_candidates = []
        for candidate in [exec_base] + list(DEFAULT_EXECUTION_PROFILES):
            if candidate not in execution_candidates:
                execution_candidates.append(candidate)
        market_focus = ["market_recent", "market_trend", "market_chop", "market_session_edges", "market_spread_shock", "market_walkforward", "market_macro_event", "market_adversarial"]
        experiments.append({"name": "horizon_sweep", "horizons": horizon_candidates[:6], "focus": market_focus})
        experiments.append({"name": "m1sync_sweep", "m1sync_bars": m1sync_candidates[:5], "focus": market_focus})
        experiments.append({"name": "execution_profile_sweep", "execution_profiles": execution_candidates[:5], "focus": market_focus})
    return campaign


def campaign_runs_extended(campaign: dict, limit_plugins: int = 0, limit_experiments: int = 0) -> list[dict]:
    runs: list[dict] = []
    plugin_items = sorted(
        campaign.get("plugins", {}).items(),
        key=lambda item: float(item[1].get("score", 0.0)),
        reverse=True,
    )
    if limit_plugins > 0:
        plugin_items = plugin_items[:limit_plugins]

    for name, info in plugin_items:
        exp_count = 0
        for exp in info.get("experiments", []):
            focus = exp.get("focus", [])
            if exp["name"] == "schema_ablation":
                for schema in exp.get("schemas", []):
                    runs.append({"plugin": name, "experiment": exp["name"], "scenario_list": focus, "schema_id": schema})
            elif exp["name"] == "normalization_sweep":
                for norm in exp.get("normalizations", []):
                    runs.append({"plugin": name, "experiment": exp["name"], "scenario_list": focus, "normalization": norm})
            elif exp["name"] == "sequence_sweep":
                for seq in exp.get("sequence_bars", []):
                    runs.append({"plugin": name, "experiment": exp["name"], "scenario_list": focus, "sequence_bars": seq})
            elif exp["name"] == "feature_mask_ablation":
                for mask in exp.get("feature_masks", []):
                    runs.append({"plugin": name, "experiment": exp["name"], "scenario_list": focus, "feature_mask": mask})
            elif exp["name"] == "execution_sweep":
                for slip in exp.get("slippage_points", []):
                    for fillp in exp.get("fill_penalty_points", []):
                        runs.append({
                            "plugin": name,
                            "experiment": exp["name"],
                            "scenario_list": focus,
                            "slippage_points": slip,
                            "fill_penalty_points": fillp,
                        })
            elif exp["name"] == "walkforward_gate":
                for train_bars, test_bars in exp.get("train_test_pairs", []):
                    runs.append({
                        "plugin": name,
                        "experiment": exp["name"],
                        "scenario_list": focus,
                        "wf_train_bars": train_bars,
                        "wf_test_bars": test_bars,
                    })
            elif exp["name"] == "horizon_sweep":
                for horizon in exp.get("horizons", []):
                    runs.append({"plugin": name, "experiment": exp["name"], "scenario_list": focus, "horizon": horizon})
            elif exp["name"] == "m1sync_sweep":
                for sync_bars in exp.get("m1sync_bars", []):
                    runs.append({"plugin": name, "experiment": exp["name"], "scenario_list": focus, "m1sync_bars": sync_bars})
            elif exp["name"] == "execution_profile_sweep":
                for profile in exp.get("execution_profiles", []):
                    runs.append({"plugin": name, "experiment": exp["name"], "scenario_list": focus, "execution_profile": profile})
            else:
                runs.append({"plugin": name, "experiment": exp["name"], "scenario_list": focus})
            exp_count += 1
            if limit_experiments > 0 and exp_count >= limit_experiments:
                break
    return runs


def historical_scenario_weaknesses(conn: libsql.Connection,
                                   profile_name: str,
                                   symbol: str,
                                   plugin_name: str,
                                   exclude_group_key: str = "") -> list[dict]:
    clauses = [
        "tr.status = 'ok'",
        "tr.profile_name = ?",
        "tr.symbol = ?",
        "tr.plugin_name = ?",
    ]
    params: list[object] = [profile_name, symbol, plugin_name]
    if exclude_group_key:
        clauses.append("COALESCE(tr.group_key, '') <> ?")
        params.append(exclude_group_key)
    sql = f"""
        SELECT rs.scenario,
               AVG(rs.score) AS mean_score,
               AVG(rs.calibration_error) AS mean_calibration_error,
               AVG(rs.path_quality_error) AS mean_path_quality_error,
               AVG(rs.wf_pbo) AS mean_wf_pbo,
               AVG(rs.wf_dsr) AS mean_wf_dsr,
               AVG(rs.issue_flags) AS mean_issue_flags,
               COUNT(*) AS obs_count
          FROM run_scenarios rs
          JOIN tuning_runs tr ON tr.id = rs.run_id
         WHERE {' AND '.join(clauses)}
         GROUP BY rs.scenario
         ORDER BY mean_score ASC, mean_calibration_error DESC, mean_path_quality_error DESC
    """
    return query_all(conn, sql, params)


def build_redteam_runs_for_plugin(plugin_name: str,
                                  family_id: int,
                                  weakness_rows: list[dict],
                                  base_args) -> tuple[list[dict], dict]:
    weak = []
    for row in weakness_rows:
        if float(row.get("mean_score", 0.0)) < 74.0 or float(row.get("mean_issue_flags", 0.0)) > 0.35:
            weak.append(dict(row))
    weak = weak[:3]
    if not weak:
        return [], {"plugin": plugin_name, "family_id": int(family_id), "weak_scenarios": [], "runs": []}

    weak_names = [str(row["scenario"]) for row in weak]
    focus = []
    for name in weak_names:
        if name not in focus:
            focus.append(name)
    if "market_adversarial" not in focus:
        focus.append("market_adversarial")
    if "market_walkforward" not in focus:
        focus.append("market_walkforward")

    base_horizon = max(int(getattr(base_args, "horizon", 5)), 1)
    base_seq = max(int(getattr(base_args, "sequence_bars", 0)), 0)
    base_sync = max(int(getattr(base_args, "m1sync_bars", 3)), 1)
    family = int(family_id)
    seq_target = base_seq
    if family in (2, 3, 4, 5):
        seq_target = max(seq_target, 32)
    elif family in (7, 8, 9):
        seq_target = max(seq_target, 16)
    else:
        seq_target = max(seq_target, 8)

    runs: list[dict] = []
    rationale: list[str] = []
    if any(name in ("market_adversarial", "market_walkforward") for name in weak_names):
        runs.append({
            "plugin": plugin_name,
            "experiment": "redteam_stability",
            "scenario_list": focus,
            "schema_id": 6 if family != 10 else 4,
            "sequence_bars": seq_target,
            "normalization": 14 if family in (2, 3, 4, 5) else 9,
            "execution_profile": "stress",
            "wf_train_bars": max(int(getattr(base_args, "wf_train_bars", 256)), 384),
            "wf_test_bars": max(int(getattr(base_args, "wf_test_bars", 64)), 96),
        })
        rationale.append("stability and walk-forward weakness triggered deeper sequence and stress execution replay")

    if "market_macro_event" in weak_names:
        runs.append({
            "plugin": plugin_name,
            "experiment": "redteam_macro",
            "scenario_list": focus,
            "schema_id": 6 if family in (2, 3, 4, 5, 7, 8, 9) else 5,
            "normalization": 13 if family != 10 else 0,
            "horizon": max(base_horizon, 8),
            "feature_mask": 0x7F if family != 10 else 0x29,
        })
        rationale.append("macro weakness triggered macro-heavy schema and full feature coverage")

    if any(name in ("market_session_edges", "market_spread_shock") for name in weak_names):
        runs.append({
            "plugin": plugin_name,
            "experiment": "redteam_execution",
            "scenario_list": focus,
            "execution_profile": "stress",
            "slippage_points": 1.0,
            "fill_penalty_points": 0.50,
            "m1sync_bars": max(base_sync, 5),
        })
        rationale.append("session/spread weakness triggered harsher execution stress and stricter M1 sync")

    if not runs:
        runs.append({
            "plugin": plugin_name,
            "experiment": "redteam_general",
            "scenario_list": focus,
            "schema_id": 6 if family != 10 else 4,
            "sequence_bars": seq_target,
            "horizon": max(base_horizon, 5),
        })
        rationale.append("general weakness triggered a broad adversarial certification pass")

    deduped: list[dict] = []
    seen = set()
    for run in runs:
        sig = json_compact(run)
        if sig in seen:
            continue
        seen.add(sig)
        deduped.append(run)

    plan = {
        "plugin": plugin_name,
        "family_id": int(family_id),
        "family_name": plugin_family_name(family_id),
        "weak_scenarios": weak,
        "rationale": rationale,
        "runs": deduped,
    }
    return deduped, plan


def persist_redteam_plan(conn: libsql.Connection,
                         profile_name: str,
                         group_key: str,
                         symbol: str,
                         plugin_name: str,
                         family_id: int,
                         plan: dict,
                         report_path: Path) -> None:
    ensure_dir(report_path.parent)
    md_lines = [
        f"# FXAI Red-Team Plan: {plugin_name}",
        "",
        f"profile: {profile_name}",
        f"symbol: {symbol}",
        f"family: {plugin_family_name(family_id)} ({int(family_id)})",
        "",
        "Weak scenarios:",
    ]
    weak = plan.get("weak_scenarios", [])
    if weak:
        for row in weak:
            md_lines.append(
                f"- {row.get('scenario', 'unknown')} | score {float(row.get('mean_score', 0.0)):.2f} | "
                f"cal {float(row.get('mean_calibration_error', 0.0)):.3f} | "
                f"path {float(row.get('mean_path_quality_error', 0.0)):.3f} | "
                f"obs {int(row.get('obs_count', 0))}"
            )
    else:
        md_lines.append("- none")
    md_lines.append("")
    md_lines.append("Planned targeted runs:")
    for run in plan.get("runs", []):
        md_lines.append(f"- {run.get('experiment', 'unknown')}: {json_compact(run)}")
    report_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    conn.execute(
        """
        INSERT INTO redteam_cycles(profile_name, group_key, symbol, plugin_name, family_id, weak_scenarios_json, plan_json, report_path, status, created_at)
        VALUES(?, ?, ?, ?, ?, ?, ?, ?, 'ready', ?)
        ON CONFLICT(profile_name, group_key, symbol, plugin_name) DO UPDATE SET
            family_id=excluded.family_id,
            weak_scenarios_json=excluded.weak_scenarios_json,
            plan_json=excluded.plan_json,
            report_path=excluded.report_path,
            status=excluded.status,
            created_at=excluded.created_at
        """,
        (
            profile_name,
            group_key,
            symbol,
            plugin_name,
            int(family_id),
            json.dumps(plan.get("weak_scenarios", []), indent=2, sort_keys=True),
            json.dumps(plan, indent=2, sort_keys=True),
            str(report_path),
            now_unix(),
        ),
    )
    commit_db(conn)


def generate_redteam_runs(conn: libsql.Connection,
                          profile_name: str,
                          group_key: str,
                          dataset: dict,
                          baseline_summary: dict,
                          base_args,
                          out_dir: Path) -> list[dict]:
    generated: list[dict] = []
    for plugin_name, info in sorted(baseline_summary.get("plugins", {}).items()):
        family_id = int(info.get("family", 11))
        weakness_rows = historical_scenario_weaknesses(
            conn,
            profile_name,
            str(dataset["symbol"]),
            plugin_name,
            group_key,
        )
        runs, plan = build_redteam_runs_for_plugin(plugin_name, family_id, weakness_rows, base_args)
        if not runs:
            continue
        report_path = out_dir / "redteam" / f"{safe_token(plugin_name)}__redteam.md"
        persist_redteam_plan(conn, profile_name, group_key, str(dataset["symbol"]), plugin_name, family_id, plan, report_path)
        generated.extend(runs)
    return generated


def normalize_namespace_parameters(args, plugin_name: str, experiment_name: str, dataset: dict) -> dict:
    return {
        "plugin": plugin_name,
        "experiment": experiment_name,
        "scenario_list": parse_csv_tokens(getattr(args, "scenario_list", SERIOUS_SCENARIOS)),
        "bars": int(getattr(args, "bars", dataset["bars"])),
        "horizon": int(getattr(args, "horizon", 5)),
        "m1sync_bars": int(getattr(args, "m1sync_bars", 3)),
        "normalization": int(getattr(args, "normalization", 0)),
        "sequence_bars": int(getattr(args, "sequence_bars", 0)),
        "schema_id": int(getattr(args, "schema_id", 0)),
        "feature_mask": int(getattr(args, "feature_mask", 0)),
        "commission_per_lot_side": float(getattr(args, "commission_per_lot_side", 0.0)),
        "cost_buffer_points": float(getattr(args, "cost_buffer_points", 0.0)),
        "slippage_points": float(getattr(args, "slippage_points", 0.0)),
        "fill_penalty_points": float(getattr(args, "fill_penalty_points", 0.0)),
        "wf_train_bars": int(getattr(args, "wf_train_bars", 256)),
        "wf_test_bars": int(getattr(args, "wf_test_bars", 64)),
        "wf_purge_bars": int(getattr(args, "wf_purge_bars", 32)),
        "wf_embargo_bars": int(getattr(args, "wf_embargo_bars", 24)),
        "wf_folds": int(getattr(args, "wf_folds", 6)),
        "execution_profile": str(getattr(args, "execution_profile", "default")),
        "symbol": str(dataset["symbol"]),
        "window_start_unix": int(dataset["start_unix"]),
        "window_end_unix": int(dataset["end_unix"]),
    }


def grouped_rows_by_plugin(report_tsv: Path) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = defaultdict(list)
    if not report_tsv.exists():
        return out
    for row in testlab.load_rows(report_tsv):
        out[row.get("ai_name", "unknown")].append(row)
    return out


def upsert_tuning_run(conn: libsql.Connection,
                      dataset: dict,
                      profile_name: str,
                      group_key: str,
                      plugin_name: str,
                      ai_id: int,
                      family_id: int,
                      experiment_name: str,
                      parameters: dict,
                      report_path: Path,
                      raw_report_path: Path,
                      summary_path: Path,
                      manifest_path: Path,
                      status: str,
                      started_at: int,
                      finished_at: int,
                      summary_plugin: dict | None) -> int:
    issues = []
    score = 0.0
    grade = "F"
    market_recent_score = 0.0
    walkforward_score = 0.0
    adversarial_score = 0.0
    macro_event_score = 0.0
    if summary_plugin:
        score = float(summary_plugin.get("score", 0.0))
        grade = str(summary_plugin.get("grade", "F"))
        issues = list(summary_plugin.get("issues", [])) + list(summary_plugin.get("findings", []))
        scenarios = summary_plugin.get("scenarios", {})
        market_recent_score = float(scenarios.get("market_recent", {}).get("score", 0.0))
        walkforward_score = float(scenarios.get("market_walkforward", {}).get("score", 0.0))
        adversarial_score = float(scenarios.get("market_adversarial", {}).get("score", 0.0))
        macro_event_score = float(scenarios.get("market_macro_event", {}).get("score", 0.0))
    parameters_json = json_compact(parameters)
    param_hash = sha256_text(f"{dataset['id']}|{profile_name}|{plugin_name}|{parameters_json}")

    conn.execute(
        """
        INSERT INTO tuning_runs(
            dataset_id, profile_name, group_key, symbol, plugin_name, ai_id, family_id, experiment_name, param_hash, parameters_json,
            report_path, raw_report_path, summary_path, manifest_path,
            score, grade, issue_count, issues_json,
            market_recent_score, walkforward_score, adversarial_score, macro_event_score,
            status, started_at, finished_at
        )
        VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(param_hash) DO UPDATE SET
            group_key=excluded.group_key,
            family_id=excluded.family_id,
            report_path=excluded.report_path,
            raw_report_path=excluded.raw_report_path,
            summary_path=excluded.summary_path,
            manifest_path=excluded.manifest_path,
            score=excluded.score,
            grade=excluded.grade,
            issue_count=excluded.issue_count,
            issues_json=excluded.issues_json,
            market_recent_score=excluded.market_recent_score,
            walkforward_score=excluded.walkforward_score,
            adversarial_score=excluded.adversarial_score,
            macro_event_score=excluded.macro_event_score,
            status=excluded.status,
            started_at=excluded.started_at,
            finished_at=excluded.finished_at
        """,
        (
            int(dataset["id"]),
            profile_name,
            (group_key or ""),
            str(dataset["symbol"]),
            plugin_name,
            ai_id,
            int(family_id),
            experiment_name,
            param_hash,
            parameters_json,
            str(report_path),
            str(raw_report_path),
            str(summary_path),
            str(manifest_path),
            score,
            grade,
            len(issues),
            json.dumps(issues, sort_keys=True),
            market_recent_score,
            walkforward_score,
            adversarial_score,
            macro_event_score,
            status,
            started_at,
            finished_at,
        ),
    )
    run_id = int(query_scalar(conn, "SELECT id FROM tuning_runs WHERE param_hash = ?", (param_hash,), 0))
    conn.execute("DELETE FROM run_scenarios WHERE run_id = ?", (run_id,))
    if summary_plugin:
        for scenario, metrics in summary_plugin.get("scenarios", {}).items():
            conn.execute(
                """
                INSERT INTO run_scenarios(run_id, scenario, score, calibration_error, path_quality_error, wf_pbo, wf_dsr, wf_pass_rate, net_signal, issue_flags)
                VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    scenario,
                    float(metrics.get("score", 0.0)),
                    float(metrics.get("calibration_error", 0.0)),
                    float(metrics.get("path_quality_error", 0.0)),
                    float(metrics.get("wf_pbo", 0.0)),
                    float(metrics.get("wf_dsr", 0.0)),
                    float(metrics.get("wf_pass_rate", 0.0)),
                    float(metrics.get("trend_align", 0.0)),
                    int(metrics.get("issue_flags", 0)),
                ),
            )
    commit_db(conn)
    return run_id


def store_baseline_run_bundle(conn: libsql.Connection,
                              dataset: dict,
                              profile_name: str,
                              group_key: str,
                              base_args,
                              report_path: Path,
                              raw_report_path: Path,
                              summary_path: Path,
                              manifest_path: Path,
                              started_at: int,
                              finished_at: int) -> None:
    summary = testlab.load_json(summary_path)
    grouped_rows = grouped_rows_by_plugin(raw_report_path)
    for plugin_name, plugin_summary in sorted(summary.get("plugins", {}).items()):
        rows = grouped_rows.get(plugin_name, [])
        ai_id = int(float(rows[0]["ai_id"])) if rows else -1
        family_id = int(plugin_summary.get("family", rows[0].get("family", 11) if rows else 11))
        params = normalize_namespace_parameters(base_args, plugin_name, "baseline_all", dataset)
        upsert_tuning_run(
            conn,
            dataset,
            profile_name,
            group_key,
            plugin_name,
            ai_id,
            family_id,
            "baseline_all",
            params,
            report_path,
            raw_report_path,
            summary_path,
            manifest_path,
            "ok",
            started_at,
            finished_at,
            plugin_summary,
        )


def run_dataset_baseline(conn: libsql.Connection, dataset: dict, profile_name: str, args, out_dir: Path) -> dict:
    baseline_path = out_dir / "baseline_all.md"
    base_args = testlab.build_effective_audit_args(serious_base_args(args, dataset, baseline_path))
    started_at = now_unix()
    rc = testlab.cmd_run_audit(base_args)
    finished_at = now_unix()
    if rc != 0:
        raise OfflineLabError(f"baseline audit failed for {dataset['dataset_key']}")
    raw_report_path = baseline_path.with_suffix(".tsv")
    shutil.copy2(testlab.DEFAULT_REPORT, raw_report_path)
    summary_path = baseline_path.with_suffix(".summary.json")
    manifest_path = baseline_path.with_suffix(".manifest.json")
    group_key = str(getattr(args, "group_key", "") or dataset.get("group_key", "") or "")
    store_baseline_run_bundle(conn, dataset, profile_name, group_key, base_args, baseline_path, raw_report_path, summary_path, manifest_path, started_at, finished_at)
    return {
        "report_path": baseline_path,
        "raw_report_path": raw_report_path,
        "summary_path": summary_path,
        "manifest_path": manifest_path,
        "summary": testlab.load_json(summary_path),
        "base_args": base_args,
    }


def run_dataset_campaign(conn: libsql.Connection, dataset: dict, profile_name: str, args, out_dir: Path, baseline_summary: dict, base_args) -> list[dict]:
    oracles = testlab.load_oracles()
    campaign = extend_campaign(testlab.build_optimization_campaign(baseline_summary, oracles), base_args)
    (out_dir / "campaign.json").write_text(json.dumps(campaign, indent=2, sort_keys=True), encoding="utf-8")
    runs = campaign_runs_extended(campaign, getattr(args, "top_plugins", 0), getattr(args, "limit_experiments", 0))
    group_key = str(getattr(args, "group_key", "") or dataset.get("group_key", "") or "")
    redteam_runs = generate_redteam_runs(conn, profile_name, group_key, dataset, baseline_summary, base_args, out_dir)
    if redteam_runs:
        runs.extend(redteam_runs)
    if getattr(args, "limit_runs", 0) > 0:
        runs = runs[: args.limit_runs]
    results = []
    for idx, run in enumerate(runs, start=1):
        run["bars"] = int(base_args.bars)
        run["window_start_unix"] = int(dataset["start_unix"])
        run["window_end_unix"] = int(dataset["end_unix"])
        if "execution_profile" not in run:
            run["execution_profile"] = str(base_args.execution_profile)
        stem = f"{idx:03d}_{run['plugin']}_{safe_token(run['experiment'])}"
        report_path = out_dir / "runs" / f"{stem}.md"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        started_at = now_unix()
        run_args = testlab.build_effective_audit_args(testlab.build_run_audit_namespace(base_args, run, report_path))
        parameters = normalize_namespace_parameters(run_args, run["plugin"], run["experiment"], dataset)
        rc = testlab.cmd_run_audit(run_args)
        finished_at = now_unix()
        raw_report_path = report_path.with_suffix(".tsv")
        summary_path = report_path.with_suffix(".summary.json")
        manifest_path = report_path.with_suffix(".manifest.json")
        plugin_summary = None
        ai_id = -1
        family_id = 11
        status = "failed"
        if rc == 0:
            shutil.copy2(testlab.DEFAULT_REPORT, raw_report_path)
            summary = testlab.load_json(summary_path)
            plugin_summary = summary.get("plugins", {}).get(run["plugin"], {})
            rows = grouped_rows_by_plugin(raw_report_path).get(run["plugin"], [])
            if rows:
                ai_id = int(float(rows[0]["ai_id"]))
                family_id = int(float(rows[0].get("family", plugin_summary.get("family", 11))))
            else:
                family_id = int(plugin_summary.get("family", 11))
            status = "ok"
        run_id = upsert_tuning_run(
            conn,
            dataset,
            profile_name,
            group_key,
            run["plugin"],
            ai_id,
            family_id,
            run["experiment"],
            parameters,
            report_path,
            raw_report_path,
            summary_path,
            manifest_path,
            status,
            started_at,
            finished_at,
            plugin_summary,
        )
        result = {
            "run_id": run_id,
            "status": status,
            "plugin": run["plugin"],
            "experiment": run["experiment"],
            "parameters": parameters,
            "report_path": str(report_path),
            "summary_path": str(summary_path),
        }
        results.append(result)
        (out_dir / "results.json").write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
    return results


def cmd_init_db(args) -> int:
    conn = connect_db(Path(args.db))
    close_db(conn)
    print(f"initialized Turso/libSQL lab: {args.db}")
    return 0


def cmd_validate_env(_args) -> int:
    payload = validate_environment()
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if bool(payload.get("ok")) else 1


def _default_turso_branch_name(profile_name: str, branch_kind: str) -> str:
    token = safe_token(profile_name or "fxai")
    return f"{token}-{safe_token(branch_kind)}-{now_unix()}"


def cmd_turso_branch_create(args) -> int:
    conn = connect_db(Path(args.db))
    try:
        config = resolve_platform_config(Path(args.db))
        source_database = str(getattr(args, "source_database", "") or config.database_name or "").strip()
        if not source_database:
            raise OfflineLabError("source database is required; set --source-database or TURSO_DATABASE_NAME")
        target_database = str(getattr(args, "target_database", "") or _default_turso_branch_name(args.profile, "campaign")).strip()
        result = create_branch_database(
            config=config,
            source_database=source_database,
            target_database=target_database,
            profile_name=args.profile,
            branch_kind="campaign",
            timestamp=str(getattr(args, "timestamp", "") or ""),
            group_name=str(getattr(args, "group_name", "") or ""),
            location_name=str(getattr(args, "location_name", "") or ""),
            token_expiration=str(getattr(args, "token_expiration", "7d") or "7d"),
            read_only_token=bool(getattr(args, "read_only_token", False)),
        )
        payload = persist_branch_result(conn, args.profile, result)
        commit_db(conn)
        print(json.dumps({
            "profile_name": args.profile,
            "branch": payload,
            "env_artifact_path": str(result.env_artifact_path),
        }, indent=2, sort_keys=True))
        return 0
    finally:
        close_db(conn)


def cmd_turso_pitr_restore(args) -> int:
    if not str(getattr(args, "timestamp", "") or "").strip():
        raise OfflineLabError("point-in-time restore requires --timestamp in RFC3339 format")
    conn = connect_db(Path(args.db))
    try:
        config = resolve_platform_config(Path(args.db))
        source_database = str(getattr(args, "source_database", "") or config.database_name or "").strip()
        if not source_database:
            raise OfflineLabError("source database is required; set --source-database or TURSO_DATABASE_NAME")
        target_database = str(getattr(args, "target_database", "") or _default_turso_branch_name(args.profile, "pitr")).strip()
        result = create_branch_database(
            config=config,
            source_database=source_database,
            target_database=target_database,
            profile_name=args.profile,
            branch_kind="pitr_restore",
            timestamp=str(args.timestamp),
            group_name=str(getattr(args, "group_name", "") or ""),
            location_name=str(getattr(args, "location_name", "") or ""),
            token_expiration=str(getattr(args, "token_expiration", "7d") or "7d"),
            read_only_token=bool(getattr(args, "read_only_token", False)),
        )
        payload = persist_branch_result(conn, args.profile, result)
        commit_db(conn)
        print(json.dumps({
            "profile_name": args.profile,
            "restore_branch": payload,
            "env_artifact_path": str(result.env_artifact_path),
        }, indent=2, sort_keys=True))
        return 0
    finally:
        close_db(conn)


def cmd_turso_branch_inventory(args) -> int:
    conn = connect_db(Path(args.db))
    try:
        config = resolve_platform_config(Path(args.db))
        source_database = str(getattr(args, "source_database", "") or config.database_name or "")
        if config.platform_api_enabled:
            branches = branch_inventory(config, source_database)
            source = "platform_api"
        else:
            branches = query_all(
                conn,
                """
                SELECT profile_name, source_database, target_database, branch_kind, source_timestamp,
                       group_name, location_name, sync_url, env_artifact_path, status, created_at
                  FROM turso_branch_runs
                 WHERE (? = '' OR source_database = ?)
                 ORDER BY created_at DESC
                """,
                (source_database, source_database),
            )
            source = "local_registry"
        payload = {
            "profile_name": args.profile,
            "source_database": source_database,
            "source": source,
            "branches": branches,
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    finally:
        close_db(conn)


def cmd_turso_branch_destroy(args) -> int:
    conn = connect_db(Path(args.db))
    try:
        config = resolve_platform_config(Path(args.db))
        target_database = str(getattr(args, "target_database", "") or "").strip()
        if not target_database:
            raise OfflineLabError("target database is required for destroy")
        row = query_one(
            conn,
            "SELECT env_artifact_path FROM turso_branch_runs WHERE target_database = ?",
            (target_database,),
        )
        destroy_database(config, target_database)
        conn.execute(
            "UPDATE turso_branch_runs SET status = 'destroyed' WHERE target_database = ?",
            (target_database,),
        )
        artifact_path = Path(str((row or {}).get("env_artifact_path", "") or "").strip())
        if artifact_path and artifact_path.exists() and artifact_path.is_file():
            artifact_path.unlink()
        commit_db(conn)
        print(json.dumps({"target_database": target_database, "status": "destroyed"}, indent=2, sort_keys=True))
        return 0
    finally:
        close_db(conn)


def cmd_turso_audit_sync(args) -> int:
    conn = connect_db(Path(args.db))
    try:
        config = resolve_platform_config(Path(args.db))
        payload = sync_audit_logs(conn, config, limit=int(getattr(args, "limit", 50) or 50), pages=int(getattr(args, "pages", 1) or 1))
        commit_db(conn)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    finally:
        close_db(conn)


def cmd_turso_vector_reindex(args) -> int:
    conn = connect_db(Path(args.db))
    try:
        payload = refresh_research_vectors(conn, args.profile, str(getattr(args, "symbol", "") or ""))
        commit_db(conn)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    finally:
        close_db(conn)


def cmd_turso_vector_neighbors(args) -> int:
    conn = connect_db(Path(args.db))
    try:
        refresh_research_vectors(conn, args.profile, args.symbol)
        payload = {
            "profile_name": args.profile,
            "symbol": args.symbol,
            "neighbors": latest_symbol_shadow_neighbors(conn, args.profile, args.symbol, limit=int(getattr(args, "limit", 5) or 5)),
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    finally:
        close_db(conn)


def cmd_bootstrap(args) -> int:
    payload = bootstrap_environment(Path(args.db), init_db=(not getattr(args, "no_init_db", False)))
    if getattr(args, "seed_demo", False):
        conn = connect_db(Path(args.db))
        try:
            fixture = seed_profile_fixture(conn, profile_name="smoke")
            demo_args = argparse.Namespace(profile="smoke", runtime_mode="research")
            attribution_payload = write_attribution_profiles(conn, demo_args)
            router_payload = write_student_router_profiles(conn, demo_args)
            deployment_payload = write_live_deployment_profiles(conn, demo_args)
            supervisor_payload = write_portfolio_supervisor_profile(conn, demo_args)
            supervisor_service = write_supervisor_service_artifacts(conn, demo_args)
            supervisor_commands = write_supervisor_command_artifacts(conn, demo_args)
            world_plans = write_world_simulator_plans(conn, demo_args)
            dashboard = write_profile_dashboard(conn, "smoke")
            lineage = write_lineage_report(conn, "smoke")
            bundle = write_minimal_live_bundle(conn, "smoke")
            payload["demo_fixture"] = {
                "fixture": fixture,
                "attribution_profiles": len(attribution_payload),
                "student_router_profiles": len(router_payload),
                "deployments": len(deployment_payload),
                "portfolio_supervisor_artifact": str(supervisor_payload.get("artifact_path", "")),
                "supervisor_service_artifacts": len(supervisor_service),
                "supervisor_command_artifacts": len(supervisor_commands),
                "world_plan_artifacts": len(world_plans),
                "dashboard": dashboard,
                "lineage": lineage,
                "minimal_bundle": bundle,
            }
        finally:
            close_db(conn)
    if getattr(args, "report", ""):
        report_path = Path(args.report)
        write_environment_report(report_path)
        payload["report_path"] = str(report_path)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if bool(payload.get("validated_environment", {}).get("ok")) else 1


def cmd_compile_export(_args) -> int:
    return compile_export_runner()


def cmd_export_dataset(args) -> int:
    conn = connect_db(Path(args.db))
    group_key = getattr(args, "group_key", "") or safe_token(f"manual_export_{now_unix()}")
    symbols = resolve_symbols(args)
    months_list = resolve_months_list(args.months_list)
    if not getattr(args, "skip_compile", False):
        if compile_export_runner() != 0:
            raise OfflineLabError("failed to compile FXAI_OfflineExportRunner.mq5")
        args = argparse.Namespace(**vars(args))
        args.skip_compile = True
    datasets = []
    for symbol in symbols:
        for months in months_list:
            datasets.append(export_single_dataset(conn, args, symbol, months, group_key))
    print(json.dumps({"group_key": group_key, "datasets": datasets}, indent=2, sort_keys=True))
    close_db(conn)
    return 0


def cmd_tune_zoo(args) -> int:
    conn = connect_db(Path(args.db))
    group_key = getattr(args, "group_key", "") or safe_token(f"{args.profile}_{now_unix()}")
    resolve_args = args
    if getattr(args, "auto_export", False) and not getattr(args, "skip_compile", False):
        if compile_export_runner() != 0:
            raise OfflineLabError("failed to compile FXAI_OfflineExportRunner.mq5")
        resolve_args = argparse.Namespace(**vars(args))
        resolve_args.skip_compile = True
    datasets = resolve_dataset_rows(conn, resolve_args, getattr(args, "auto_export", False), group_key)
    if not datasets:
        raise OfflineLabError("no datasets resolved for tune-zoo")

    if not getattr(args, "skip_compile", False):
        rc = compile_audit_runner()
        if rc != 0:
            raise OfflineLabError("failed to compile FXAI_AuditRunner.mq5")

    all_results = []
    for dataset in datasets:
        dataset_out_dir = RUNS_DIR / safe_token(args.profile) / safe_token(dataset["dataset_key"])
        ensure_dir(dataset_out_dir)
        baseline = run_dataset_baseline(conn, dataset, args.profile, args, dataset_out_dir)
        results = run_dataset_campaign(conn, dataset, args.profile, args, dataset_out_dir, baseline["summary"], baseline["base_args"])
        all_results.append({
            "dataset_key": dataset["dataset_key"],
            "symbol": dataset["symbol"],
            "baseline_report": str(baseline["report_path"]),
            "run_count": len(results),
        })
    print(json.dumps({"profile": args.profile, "datasets": all_results}, indent=2, sort_keys=True))
    close_db(conn)
    return 0


def cmd_best_params(args) -> int:
    conn = connect_db(Path(args.db))
    rows = load_completed_runs(conn, args)
    if not rows:
        raise OfflineLabError("no completed tuning runs found for best-params")
    winners, _dataset_counts = aggregate_best_candidates(rows)
    promoted = persist_best_configs(conn, args, winners)
    champion_decisions = update_champion_registry(conn, args, promoted)
    family_scorecards = persist_family_scorecards(conn, args, rows, promoted)
    distill_artifacts = write_distillation_artifacts(conn, args, promoted)
    shadow_ingest = ingest_shadow_fleet_ledgers(conn, args.profile)
    teacher_factories = write_teacher_factory_artifacts(conn, args, promoted)
    foundation_teachers = write_foundation_teacher_artifacts(conn, args, promoted)
    foundation_bundles = write_foundation_model_bundles(conn, args, promoted)
    student_bundles = write_student_deployment_bundles(conn, args, promoted)
    attribution_profiles = write_attribution_profiles(conn, args)
    student_router_profiles = write_student_router_profiles(conn, args)
    live_deployments = write_live_deployment_profiles(conn, args)
    governance_payload = run_autonomous_governance(conn, args, str(getattr(args, "group_key", "") or ""))
    print(json.dumps({
        "profile": args.profile,
        "promoted_count": len(promoted),
        "champion_count": sum(1 for item in champion_decisions if item["status"] == "champion"),
        "challenger_count": sum(1 for item in champion_decisions if item["status"] == "challenger"),
        "family_scorecards": len(family_scorecards),
        "distillation_artifacts": len(distill_artifacts),
        "teacher_factories": len(teacher_factories),
        "foundation_teachers": len(foundation_teachers),
        "foundation_bundles": len(foundation_bundles),
        "student_bundles": len(student_bundles),
        "attribution_profiles": len(attribution_profiles),
        "student_router_profiles": len(student_router_profiles),
        "live_deployments": len(live_deployments),
        "shadow_rows_ingested": int(shadow_ingest.get("rows_ingested", 0)),
        "governance_decisions": len(governance_payload.get("decisions", [])),
        "world_plans": len(governance_payload.get("world_plans", [])),
        "portfolio_supervisor_artifact": str(governance_payload.get("portfolio_supervisor", {}).get("artifact_path", "")),
        "supervisor_service_artifacts": len(governance_payload.get("supervisor_service", [])),
        "supervisor_command_artifacts": len(governance_payload.get("supervisor_commands", [])),
    }, indent=2, sort_keys=True))
    close_db(conn)
    return 0


def cmd_shadow_sync(args) -> int:
    conn = connect_db(Path(args.db))
    payload = ingest_shadow_fleet_ledgers(conn, args.profile)
    close_db(conn)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def cmd_dashboard(args) -> int:
    conn = connect_db(Path(args.db))
    try:
        refresh_research_vectors(conn, args.profile)
        payload = write_profile_dashboard(conn, args.profile)
        write_performance_reports(list(payload.get("symbols", [])), args.profile)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    finally:
        close_db(conn)


def cmd_live_state(args) -> int:
    payload = live_state_snapshot(args.profile, args.symbol)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def cmd_lineage_report(args) -> int:
    conn = connect_db(Path(args.db))
    try:
        payload = write_lineage_report(conn, args.profile, getattr(args, "symbol", ""))
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    finally:
        close_db(conn)


def cmd_minimal_bundle(args) -> int:
    conn = connect_db(Path(args.db))
    try:
        output_dir = Path(args.output_dir) if getattr(args, "output_dir", "") else None
        payload = write_minimal_live_bundle(conn, args.profile, output_dir)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    finally:
        close_db(conn)


def cmd_seed_demo(args) -> int:
    conn = connect_db(Path(args.db))
    try:
        fixture = seed_profile_fixture(conn, profile_name=args.profile, symbol=args.symbol)
        demo_args = argparse.Namespace(
            profile=args.profile,
            runtime_mode=str(resolve_runtime_mode(getattr(args, "runtime_mode", None))["runtime_mode"]),
        )
        attribution_payload = write_attribution_profiles(conn, demo_args)
        router_payload = write_student_router_profiles(conn, demo_args)
        deployment_payload = write_live_deployment_profiles(conn, demo_args)
        supervisor_payload = write_portfolio_supervisor_profile(conn, demo_args)
        supervisor_service = write_supervisor_service_artifacts(conn, demo_args)
        supervisor_commands = write_supervisor_command_artifacts(conn, demo_args)
        world_plans = write_world_simulator_plans(conn, demo_args)
        vector_payload = refresh_research_vectors(conn, args.profile, args.symbol)
        write_performance_reports([args.symbol], args.profile)
        dashboard = write_profile_dashboard(conn, args.profile)
        lineage = write_lineage_report(conn, args.profile, args.symbol)
        bundle = write_minimal_live_bundle(conn, args.profile)
        print(
            json.dumps(
                {
                    "profile": args.profile,
                    "runtime_mode": demo_args.runtime_mode,
                    "fixture": fixture,
                    "attribution_profiles": len(attribution_payload),
                    "student_router_profiles": len(router_payload),
                    "deployments": len(deployment_payload),
                    "portfolio_supervisor_artifact": str(supervisor_payload.get("artifact_path", "")),
                    "supervisor_service_artifacts": len(supervisor_service),
                    "supervisor_command_artifacts": len(supervisor_commands),
                    "world_plan_artifacts": len(world_plans),
                    "vector_refresh": vector_payload,
                    "dashboard": dashboard,
                    "lineage": lineage,
                    "minimal_bundle": bundle,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    finally:
        close_db(conn)


def cmd_verify_deterministic(args) -> int:
    payload = verify_deterministic_outputs(refresh_golden=bool(getattr(args, "refresh_golden", False)))
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if bool(payload.get("ok")) else 1


def cmd_deploy_profiles(args) -> int:
    conn = connect_db(Path(args.db))
    try:
        attribution_payload = write_attribution_profiles(conn, args)
        router_payload = write_student_router_profiles(conn, args)
        payload = write_live_deployment_profiles(conn, args)
        vector_payload = refresh_research_vectors(conn, args.profile)
        symbols = [str(item["symbol"]) for item in payload]
        write_performance_reports(symbols, args.profile)
        dashboard = write_profile_dashboard(conn, args.profile)
        lineage = write_lineage_report(conn, args.profile)
        bundle = write_minimal_live_bundle(conn, args.profile)
        print(json.dumps({
            "profile": args.profile,
            "runtime_mode": resolve_runtime_mode(getattr(args, "runtime_mode", None)),
            "attribution_profiles": attribution_payload,
            "student_router_profiles": router_payload,
            "deployments": payload,
            "vector_refresh": vector_payload,
            "dashboard": dashboard,
            "lineage": lineage,
            "minimal_bundle": bundle,
        }, indent=2, sort_keys=True))
        return 0
    finally:
        close_db(conn)


def cmd_autonomous_governance(args) -> int:
    conn = connect_db(Path(args.db))
    try:
        shadow_ingest = ingest_shadow_fleet_ledgers(conn, args.profile)
        attribution_payload = write_attribution_profiles(conn, args)
        router_payload = write_student_router_profiles(conn, args)
        vector_payload = refresh_research_vectors(conn, args.profile)
        payload = run_autonomous_governance(conn, args, str(getattr(args, "group_key", "") or ""))
        write_performance_reports(sorted({str(item["symbol"]) for item in payload.get("decisions", [])}), args.profile)
        dashboard = write_profile_dashboard(conn, args.profile)
        lineage = write_lineage_report(conn, args.profile)
        bundle = write_minimal_live_bundle(conn, args.profile)
        print(json.dumps({
            "profile": args.profile,
            "runtime_mode": resolve_runtime_mode(getattr(args, "runtime_mode", None)),
            "shadow_rows_ingested": int(shadow_ingest.get("rows_ingested", 0)),
            "attribution_profiles": len(attribution_payload),
            "student_router_profiles": len(router_payload),
            "vector_refresh": vector_payload,
            "governance_decisions": len(payload.get("decisions", [])),
            "world_plans": len(payload.get("world_plans", [])),
            "portfolio_supervisor_artifact": str(payload.get("portfolio_supervisor", {}).get("artifact_path", "")),
            "supervisor_service_artifacts": len(payload.get("supervisor_service", [])),
            "supervisor_command_artifacts": len(payload.get("supervisor_commands", [])),
            "dashboard": dashboard,
            "lineage": lineage,
            "minimal_bundle": bundle,
        }, indent=2, sort_keys=True))
        return 0
    finally:
        close_db(conn)


def cmd_supervisor_sync(args) -> int:
    conn = connect_db(Path(args.db))
    payload = write_supervisor_service_artifacts(conn, args)
    commands = write_supervisor_command_artifacts(conn, args)
    close_db(conn)
    print(json.dumps({"profile": args.profile, "artifacts": payload, "commands": commands}, indent=2, sort_keys=True))
    return 0


def cmd_attribution_prune(args) -> int:
    conn = connect_db(Path(args.db))
    try:
        attribution_payload = write_attribution_profiles(conn, args)
        router_payload = write_student_router_profiles(conn, args)
        print(json.dumps({
            "profile": args.profile,
            "runtime_mode": resolve_runtime_mode(getattr(args, "runtime_mode", None)),
            "attribution_profiles": attribution_payload,
            "student_router_profiles": router_payload,
        }, indent=2, sort_keys=True))
        return 0
    finally:
        close_db(conn)


def cmd_recover_artifacts(args) -> int:
    conn = connect_db(Path(args.db))
    try:
        attribution_payload = write_attribution_profiles(conn, args)
        router_payload = write_student_router_profiles(conn, args)
        deploy_payload = write_live_deployment_profiles(conn, args)
        vector_payload = refresh_research_vectors(conn, args.profile)
        supervisor_payload = write_portfolio_supervisor_profile(conn, args)
        supervisor_service = write_supervisor_service_artifacts(conn, args)
        supervisor_commands = write_supervisor_command_artifacts(conn, args)
        world_payload = write_world_simulator_plans(conn, args)
        symbols = sorted({str(item["symbol"]) for item in deploy_payload})
        performance_payload = write_performance_reports(symbols, args.profile)
        dashboard = write_profile_dashboard(conn, args.profile)
        lineage = write_lineage_report(conn, args.profile)
        bundle = write_minimal_live_bundle(conn, args.profile)
        print(json.dumps({
            "profile": args.profile,
            "runtime_mode": resolve_runtime_mode(getattr(args, "runtime_mode", None)),
            "attribution_profiles": len(attribution_payload),
            "student_router_profiles": len(router_payload),
            "deployments": len(deploy_payload),
            "vector_refresh": vector_payload,
            "portfolio_supervisor_artifact": str(supervisor_payload.get("artifact_path", "")),
            "supervisor_service_artifacts": len(supervisor_service),
            "supervisor_command_artifacts": len(supervisor_commands),
            "world_plan_artifacts": len(world_payload),
            "performance_reports": len(performance_payload),
            "dashboard": dashboard,
            "lineage": lineage,
            "minimal_bundle": bundle,
        }, indent=2, sort_keys=True))
        return 0
    finally:
        close_db(conn)


def cmd_supervisor_daemon(args) -> int:
    conn = connect_db(Path(args.db))
    payload = run_supervisor_daemon(conn, args)
    close_db(conn)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def cmd_control_loop(args) -> int:
    conn = connect_db(Path(args.db))
    cycles = int(getattr(args, "cycles", 1) or 0)
    cycle_idx = 0
    while True:
        cycle_idx += 1
        started_at = now_unix()
        group_key = safe_token(f"{args.profile}_{started_at}")
        conn.execute(
            "INSERT INTO control_cycles(profile_name, group_key, started_at, status, notes) VALUES(?, ?, ?, 'running', ?)",
            (args.profile, group_key, started_at, f"cycle={cycle_idx}"),
        )
        cycle_id = int(query_scalar(conn, "SELECT last_insert_rowid()", default=0))
        commit_db(conn)
        try:
            cycle_args = argparse.Namespace(**vars(args))
            cycle_args.group_key = group_key
            resolve_args = cycle_args
            if not getattr(args, "skip_compile", False):
                if compile_export_runner() != 0:
                    raise OfflineLabError("failed to compile FXAI_OfflineExportRunner.mq5")
                resolve_args = argparse.Namespace(**vars(cycle_args))
                resolve_args.skip_compile = True
            datasets = resolve_dataset_rows(conn, resolve_args, True, group_key)
            if not getattr(args, "skip_compile", False):
                if compile_audit_runner() != 0:
                    raise OfflineLabError("failed to compile FXAI_AuditRunner.mq5")
            summary_items = []
            for dataset in datasets:
                dataset_out_dir = RUNS_DIR / safe_token(args.profile) / safe_token(dataset["dataset_key"])
                ensure_dir(dataset_out_dir)
                baseline = run_dataset_baseline(conn, dataset, args.profile, cycle_args, dataset_out_dir)
                results = run_dataset_campaign(conn, dataset, args.profile, cycle_args, dataset_out_dir, baseline["summary"], baseline["base_args"])
                summary_items.append({"dataset_key": dataset["dataset_key"], "symbol": dataset["symbol"], "run_count": len(results)})
            best_args = argparse.Namespace(**vars(cycle_args))
            best_args.group_key = group_key
            cmd_best_params(best_args)
            conn.execute(
                "UPDATE control_cycles SET finished_at = ?, status = 'ok', datasets_json = ? WHERE id = ?",
                (now_unix(), json.dumps(summary_items, indent=2, sort_keys=True), cycle_id),
            )
            commit_db(conn)
        except Exception as exc:
            conn.execute(
                "UPDATE control_cycles SET finished_at = ?, status = 'failed', notes = ? WHERE id = ?",
                (now_unix(), str(exc), cycle_id),
            )
            commit_db(conn)
            close_db(conn)
            raise

        if cycles > 0 and cycle_idx >= cycles:
            break
        sleep_seconds = int(getattr(args, "sleep_seconds", 0) or 0)
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)
    close_db(conn)
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="FXAI Turso-backed offline tuning and control lab")
    ap.add_argument("--db", default=str(DEFAULT_DB))
    sub = ap.add_subparsers(dest="cmd", required=True)

    init_db = sub.add_parser("init-db", help="Initialize the Turso offline lab schema")
    init_db.set_defaults(func=cmd_init_db)

    val = sub.add_parser("validate-env", help="Validate Python, MT5, FILE_COMMON, and Offline Lab path assumptions")
    val.set_defaults(func=cmd_validate_env)

    boot = sub.add_parser("bootstrap", help="Create required lab folders, validate the environment, and initialize Turso")
    boot.add_argument("--report", default="")
    boot.add_argument("--no-init-db", action="store_true")
    boot.add_argument("--seed-demo", action="store_true")
    boot.set_defaults(func=cmd_bootstrap)

    comp = sub.add_parser("compile-export", help="Compile the MT5 offline export runner")
    comp.set_defaults(func=cmd_compile_export)

    exp = sub.add_parser("export-dataset", help="Export exact-window M1 OHLC+spread history from MT5 into Turso")
    exp.add_argument("--symbol", default="EURUSD")
    exp.add_argument("--symbol-list", default="")
    exp.add_argument("--symbol-pack", default="", choices=[""] + sorted(testlab.SYMBOL_PACKS.keys()))
    exp.add_argument("--months-list", default="3,6,12")
    exp.add_argument("--start-unix", type=int, default=0)
    exp.add_argument("--end-unix", type=int, default=0)
    exp.add_argument("--max-bars", type=int, default=600000)
    exp.add_argument("--group-key", default="")
    exp.add_argument("--notes", default="")
    exp.add_argument("--replace", action="store_true")
    exp.add_argument("--skip-compile", action="store_true")
    exp.add_argument("--login")
    exp.add_argument("--server")
    exp.add_argument("--password")
    exp.add_argument("--timeout", type=int, default=300)
    exp.set_defaults(func=cmd_export_dataset)

    tune = sub.add_parser("tune-zoo", help="Run the full MT5 model-zoo tuning campaign on exact exported windows")
    tune.add_argument("--profile", default="continuous")
    tune.add_argument("--dataset-keys", default="")
    tune.add_argument("--group-key", default="")
    tune.add_argument("--auto-export", action="store_true")
    tune.add_argument("--symbol", default="EURUSD")
    tune.add_argument("--symbol-list", default="")
    tune.add_argument("--symbol-pack", default="", choices=[""] + sorted(testlab.SYMBOL_PACKS.keys()))
    tune.add_argument("--months-list", default="3,6,12")
    tune.add_argument("--start-unix", type=int, default=0)
    tune.add_argument("--end-unix", type=int, default=0)
    tune.add_argument("--replace", action="store_true")
    tune.add_argument("--skip-compile", action="store_true")
    tune.add_argument("--top-plugins", type=int, default=0)
    tune.add_argument("--limit-experiments", type=int, default=0)
    tune.add_argument("--limit-runs", type=int, default=0)
    tune.add_argument("--scenario-list", default=SERIOUS_SCENARIOS)
    tune.add_argument("--bars", type=int, default=0)
    tune.add_argument("--horizon", type=int, default=5)
    tune.add_argument("--m1sync-bars", type=int, default=3)
    tune.add_argument("--normalization", type=int, default=0)
    tune.add_argument("--sequence-bars", type=int, default=0)
    tune.add_argument("--schema-id", type=int, default=0)
    tune.add_argument("--feature-mask", type=int, default=0)
    tune.add_argument("--commission-per-lot-side", type=float, default=None)
    tune.add_argument("--cost-buffer-points", type=float, default=None)
    tune.add_argument("--slippage-points", type=float, default=None)
    tune.add_argument("--fill-penalty-points", type=float, default=None)
    tune.add_argument("--wf-train-bars", type=int, default=256)
    tune.add_argument("--wf-test-bars", type=int, default=64)
    tune.add_argument("--wf-purge-bars", type=int, default=32)
    tune.add_argument("--wf-embargo-bars", type=int, default=24)
    tune.add_argument("--wf-folds", type=int, default=6)
    tune.add_argument("--seed", type=int, default=42)
    tune.add_argument("--execution-profile", default="default", choices=sorted(testlab.EXECUTION_PROFILES.keys()))
    tune.add_argument("--login")
    tune.add_argument("--server")
    tune.add_argument("--password")
    tune.add_argument("--timeout", type=int, default=300)
    tune.set_defaults(func=cmd_tune_zoo)

    best = sub.add_parser("best-params", help="Promote the strongest parameter packs and emit MT5-ready presets")
    best.add_argument("--profile", default="continuous")
    best.add_argument("--dataset-keys", default="")
    best.add_argument("--group-key", default="")
    best.add_argument("--symbol", default="")
    best.add_argument("--symbol-list", default="")
    best.add_argument("--symbol-pack", default="", choices=[""] + sorted(testlab.SYMBOL_PACKS.keys()))
    best.set_defaults(func=cmd_best_params)

    shadow = sub.add_parser("shadow-sync", help="Ingest live shadow-fleet ledgers from FILE_COMMON into Turso")
    shadow.add_argument("--profile", default="continuous")
    shadow.set_defaults(func=cmd_shadow_sync)

    tbranch = sub.add_parser("turso-branch-create", help="Create a Turso branch database and emit a branch env artifact")
    tbranch.add_argument("--profile", default="continuous")
    tbranch.add_argument("--source-database", default="")
    tbranch.add_argument("--target-database", default="")
    tbranch.add_argument("--timestamp", default="")
    tbranch.add_argument("--group-name", default="")
    tbranch.add_argument("--location-name", default="")
    tbranch.add_argument("--token-expiration", default="7d")
    tbranch.add_argument("--read-only-token", action="store_true")
    tbranch.set_defaults(func=cmd_turso_branch_create)

    tpitr = sub.add_parser("turso-pitr-restore", help="Create a point-in-time restore branch from Turso")
    tpitr.add_argument("--profile", default="continuous")
    tpitr.add_argument("--source-database", default="")
    tpitr.add_argument("--target-database", default="")
    tpitr.add_argument("--timestamp", required=True)
    tpitr.add_argument("--group-name", default="")
    tpitr.add_argument("--location-name", default="")
    tpitr.add_argument("--token-expiration", default="7d")
    tpitr.add_argument("--read-only-token", action="store_true")
    tpitr.set_defaults(func=cmd_turso_pitr_restore)

    tinv = sub.add_parser("turso-branch-inventory", help="List Turso branch inventory from the Platform API")
    tinv.add_argument("--profile", default="continuous")
    tinv.add_argument("--source-database", default="")
    tinv.set_defaults(func=cmd_turso_branch_inventory)

    tkill = sub.add_parser("turso-branch-destroy", help="Destroy a Turso branch database and mark it destroyed locally")
    tkill.add_argument("--target-database", required=True)
    tkill.set_defaults(func=cmd_turso_branch_destroy)

    taudit = sub.add_parser("turso-audit-sync", help="Ingest Turso organization audit logs into the Offline Lab")
    taudit.add_argument("--limit", type=int, default=50)
    taudit.add_argument("--pages", type=int, default=1)
    taudit.set_defaults(func=cmd_turso_audit_sync)

    tvec = sub.add_parser("turso-vector-reindex", help="Refresh Turso native vectors for analog-state retrieval")
    tvec.add_argument("--profile", default="continuous")
    tvec.add_argument("--symbol", default="")
    tvec.set_defaults(func=cmd_turso_vector_reindex)

    tnn = sub.add_parser("turso-vector-neighbors", help="Show nearest analog-state neighbors from Turso vectors")
    tnn.add_argument("--profile", default="continuous")
    tnn.add_argument("--symbol", required=True)
    tnn.add_argument("--limit", type=int, default=5)
    tnn.set_defaults(func=cmd_turso_vector_neighbors)

    demo = sub.add_parser("seed-demo", help="Seed a deterministic smoke profile and emit MT5 runtime artifacts without broker data")
    demo.add_argument("--profile", default="smoke")
    demo.add_argument("--symbol", default="EURUSD")
    demo.add_argument("--runtime-mode", default="research", choices=sorted(RUNTIME_MODES.keys()))
    demo.set_defaults(func=cmd_seed_demo)

    deploy = sub.add_parser("deploy-profiles", help="Build live deployment profiles for MT5 runtime control plane")
    deploy.add_argument("--profile", default="continuous")
    deploy.add_argument("--runtime-mode", default="research", choices=sorted(RUNTIME_MODES.keys()))
    deploy.set_defaults(func=cmd_deploy_profiles)

    gov = sub.add_parser("autonomous-governance", help="Build portfolio supervisor and world-plan artifacts from live research telemetry")
    gov.add_argument("--profile", default="continuous")
    gov.add_argument("--group-key", default="")
    gov.add_argument("--runtime-mode", default="research", choices=sorted(RUNTIME_MODES.keys()))
    gov.set_defaults(func=cmd_autonomous_governance)

    sup = sub.add_parser("supervisor-sync", help="Build central supervisor-service artifacts from live control-plane snapshots")
    sup.add_argument("--profile", default="continuous")
    sup.set_defaults(func=cmd_supervisor_sync)

    supd = sub.add_parser("supervisor-daemon", help="Continuously refresh supervisor-service artifacts from live control-plane snapshots")
    supd.add_argument("--profile", default="continuous")
    supd.add_argument("--interval-seconds", type=int, default=30)
    supd.add_argument("--iterations", type=int, default=0, help="0 means run forever")
    supd.set_defaults(func=cmd_supervisor_daemon)

    attr = sub.add_parser("attribution-prune", help="Build attribution and live student-router pruning profiles")
    attr.add_argument("--profile", default="continuous")
    attr.add_argument("--runtime-mode", default="research", choices=sorted(RUNTIME_MODES.keys()))
    attr.set_defaults(func=cmd_attribution_prune)

    dash = sub.add_parser("dashboard", help="Render the operator dashboard for a profile")
    dash.add_argument("--profile", default="continuous")
    dash.set_defaults(func=cmd_dashboard)

    live = sub.add_parser("live-state", help="Show the current file-backed runtime state for one symbol")
    live.add_argument("--profile", default="continuous")
    live.add_argument("--symbol", default="EURUSD")
    live.set_defaults(func=cmd_live_state)

    lineage = sub.add_parser("lineage-report", help="Render config lineage and live deployment provenance for a profile")
    lineage.add_argument("--profile", default="continuous")
    lineage.add_argument("--symbol", default="")
    lineage.set_defaults(func=cmd_lineage_report)

    bundle = sub.add_parser("minimal-bundle", help="Build a minimal live artifact bundle for promoted symbols")
    bundle.add_argument("--profile", default="continuous")
    bundle.add_argument("--output-dir", default="")
    bundle.set_defaults(func=cmd_minimal_bundle)

    det = sub.add_parser("verify-deterministic", help="Run seeded deterministic artifact checks against golden fixtures")
    det.add_argument("--refresh-golden", action="store_true")
    det.set_defaults(func=cmd_verify_deterministic)

    rec = sub.add_parser("recover-artifacts", help="Rebuild generated promotions, dashboards, and summaries from Turso state")
    rec.add_argument("--profile", default="continuous")
    rec.add_argument("--runtime-mode", default="research", choices=sorted(RUNTIME_MODES.keys()))
    rec.set_defaults(func=cmd_recover_artifacts)

    loop = sub.add_parser("control-loop", help="Run the full export -> tune -> promote cycle continuously")
    loop.add_argument("--profile", default="continuous")
    loop.add_argument("--symbol", default="EURUSD")
    loop.add_argument("--symbol-list", default="")
    loop.add_argument("--symbol-pack", default="", choices=[""] + sorted(testlab.SYMBOL_PACKS.keys()))
    loop.add_argument("--months-list", default="3,6,12")
    loop.add_argument("--start-unix", type=int, default=0)
    loop.add_argument("--end-unix", type=int, default=0)
    loop.add_argument("--replace", action="store_true")
    loop.add_argument("--skip-compile", action="store_true")
    loop.add_argument("--top-plugins", type=int, default=0)
    loop.add_argument("--limit-experiments", type=int, default=0)
    loop.add_argument("--limit-runs", type=int, default=0)
    loop.add_argument("--scenario-list", default=SERIOUS_SCENARIOS)
    loop.add_argument("--bars", type=int, default=0)
    loop.add_argument("--horizon", type=int, default=5)
    loop.add_argument("--m1sync-bars", type=int, default=3)
    loop.add_argument("--normalization", type=int, default=0)
    loop.add_argument("--sequence-bars", type=int, default=0)
    loop.add_argument("--schema-id", type=int, default=0)
    loop.add_argument("--feature-mask", type=int, default=0)
    loop.add_argument("--commission-per-lot-side", type=float, default=None)
    loop.add_argument("--cost-buffer-points", type=float, default=None)
    loop.add_argument("--slippage-points", type=float, default=None)
    loop.add_argument("--fill-penalty-points", type=float, default=None)
    loop.add_argument("--wf-train-bars", type=int, default=256)
    loop.add_argument("--wf-test-bars", type=int, default=64)
    loop.add_argument("--wf-purge-bars", type=int, default=32)
    loop.add_argument("--wf-embargo-bars", type=int, default=24)
    loop.add_argument("--wf-folds", type=int, default=6)
    loop.add_argument("--seed", type=int, default=42)
    loop.add_argument("--execution-profile", default="default", choices=sorted(testlab.EXECUTION_PROFILES.keys()))
    loop.add_argument("--cycles", type=int, default=1, help="0 means run forever")
    loop.add_argument("--sleep-seconds", type=int, default=0)
    loop.add_argument("--runtime-mode", default="research", choices=sorted(RUNTIME_MODES.keys()))
    loop.add_argument("--login")
    loop.add_argument("--server")
    loop.add_argument("--password")
    loop.add_argument("--timeout", type=int, default=300)
    loop.set_defaults(func=cmd_control_loop)

    return ap


def main() -> int:
    ap = build_parser()
    args = ap.parse_args()
    try:
        return int(args.func(args))
    except OfflineLabError as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
