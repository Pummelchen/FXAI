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
        strategy_profile=getattr(args, "strategy_profile", "default"),
        broker_profile=getattr(args, "broker_profile", ""),
        runtime_mode=getattr(args, "runtime_mode", "research"),
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
        "strategy_profile": str(getattr(args, "strategy_profile", "default") or "default"),
        "broker_profile": str(getattr(args, "broker_profile", "") or ""),
        "runtime_mode": str(getattr(args, "runtime_mode", "research") or "research"),
        "server": str(getattr(args, "server", "") or ""),
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
