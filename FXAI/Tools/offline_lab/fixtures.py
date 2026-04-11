from __future__ import annotations

import contextlib
import importlib
import json
import shutil
from pathlib import Path
import libsql

from .common import commit_db, ensure_dir, json_compact, now_unix, query_one, query_scalar, safe_token, sha256_text


PATCH_MODULES = [
    "offline_lab.common",
    "offline_lab.attribution",
    "offline_lab.student_router",
    "offline_lab.teacher_factory",
    "offline_lab.foundation_factory",
    "offline_lab.governance",
    "offline_lab.supervisor_service",
    "offline_lab.world_simulator",
    "offline_lab.environment",
    "offline_lab.dashboard",
    "offline_lab.performance",
    "offline_lab.lineage",
    "offline_lab.bundle",
    "offline_lab.newspulse_contracts",
    "offline_lab.newspulse_config",
    "offline_lab.newspulse_calendar",
    "offline_lab.newspulse_policy",
    "offline_lab.newspulse_official",
    "offline_lab.newspulse_story",
    "offline_lab.newspulse_fusion",
    "offline_lab.newspulse_replay",
    "offline_lab.newspulse_service",
    "offline_lab.newspulse_daemon",
    "offline_lab.rates_engine_contracts",
    "offline_lab.rates_engine_config",
    "offline_lab.rates_engine_inputs",
    "offline_lab.rates_engine_newspulse",
    "offline_lab.rates_engine_replay",
    "offline_lab.rates_engine",
    "offline_lab.rates_engine_daemon",
    "offline_lab.cross_asset_contracts",
    "offline_lab.cross_asset_config",
    "offline_lab.cross_asset_service",
    "offline_lab.cross_asset_replay",
    "offline_lab.cross_asset_engine",
    "offline_lab.drift_governance_contracts",
    "offline_lab.drift_governance_config",
    "offline_lab.drift_governance_math",
    "offline_lab.drift_governance",
    "offline_lab.label_engine_contracts",
    "offline_lab.label_engine_config",
    "offline_lab.label_engine",
    "offline_lab.microstructure_contracts",
    "offline_lab.microstructure_config",
    "offline_lab.microstructure_service",
    "offline_lab.microstructure_replay",
    "offline_lab.adaptive_router_contracts",
    "offline_lab.adaptive_router_config",
    "offline_lab.adaptive_router",
    "offline_lab.adaptive_router_replay",
    "offline_lab.dynamic_ensemble_contracts",
    "offline_lab.dynamic_ensemble_config",
    "offline_lab.dynamic_ensemble_replay",
    "offline_lab.prob_calibration_contracts",
    "offline_lab.prob_calibration_config",
    "offline_lab.prob_calibration_replay",
    "offline_lab.execution_quality_contracts",
    "offline_lab.execution_quality_config",
    "offline_lab.execution_quality_replay",
    "offline_lab.market_universe",
    "offline_lab.verification",
    "testlab.shared",
    "testlab.reporting",
    "testlab.release_gate",
]


@contextlib.contextmanager
def patched_paths(base_dir: Path):
    base_dir = base_dir.resolve()
    offline_dir = base_dir / "OfflineLab"
    newspulse_dir = offline_dir / "NewsPulse"
    newspulse_state_dir = newspulse_dir / "State"
    newspulse_report_dir = newspulse_dir / "Reports"
    adaptive_router_dir = offline_dir / "AdaptiveRouter"
    adaptive_router_report_dir = adaptive_router_dir / "Reports"
    dynamic_ensemble_dir = offline_dir / "DynamicEnsemble"
    dynamic_ensemble_report_dir = dynamic_ensemble_dir / "Reports"
    prob_calibration_dir = offline_dir / "ProbabilisticCalibration"
    prob_calibration_report_dir = prob_calibration_dir / "Reports"
    execution_quality_dir = offline_dir / "ExecutionQuality"
    execution_quality_report_dir = execution_quality_dir / "Reports"
    rates_engine_dir = offline_dir / "RatesEngine"
    rates_engine_state_dir = rates_engine_dir / "State"
    rates_engine_report_dir = rates_engine_dir / "Reports"
    cross_asset_dir = offline_dir / "CrossAsset"
    cross_asset_state_dir = cross_asset_dir / "State"
    cross_asset_report_dir = cross_asset_dir / "Reports"
    drift_governance_dir = offline_dir / "DriftGovernance"
    drift_governance_report_dir = drift_governance_dir / "Reports"
    drift_governance_artifact_dir = drift_governance_dir / "Artifacts"
    label_engine_dir = offline_dir / "LabelEngine"
    label_engine_report_dir = label_engine_dir / "Reports"
    label_engine_artifact_dir = label_engine_dir / "Artifacts"
    microstructure_dir = offline_dir / "Microstructure"
    microstructure_state_dir = microstructure_dir / "State"
    microstructure_report_dir = microstructure_dir / "Reports"
    common_dir = base_dir / "FILE_COMMON"
    common_promotion_dir = common_dir / "FXAI/Offline/Promotions"
    common_export_dir = common_dir / "FXAI/Offline/Exports"
    runtime_dir = common_dir / "FXAI/Runtime"
    tester_dir = base_dir / "MT5/Profiles/Tester"
    research_dir = offline_dir / "ResearchOS"
    distill_dir = offline_dir / "Distillation"
    profiles_dir = offline_dir / "Profiles"
    runs_dir = offline_dir / "Runs"
    default_db = offline_dir / "fxai_offline_lab.turso.db"
    for path in [
        offline_dir,
        newspulse_dir,
        newspulse_state_dir,
        newspulse_report_dir,
        adaptive_router_dir,
        adaptive_router_report_dir,
        dynamic_ensemble_dir,
        dynamic_ensemble_report_dir,
        prob_calibration_dir,
        prob_calibration_report_dir,
        execution_quality_dir,
        execution_quality_report_dir,
        rates_engine_dir,
        rates_engine_state_dir,
        rates_engine_report_dir,
        cross_asset_dir,
        cross_asset_state_dir,
        cross_asset_report_dir,
        drift_governance_dir,
        drift_governance_report_dir,
        drift_governance_artifact_dir,
        label_engine_dir,
        label_engine_report_dir,
        label_engine_artifact_dir,
        microstructure_dir,
        microstructure_state_dir,
        microstructure_report_dir,
        common_promotion_dir,
        common_export_dir,
        runtime_dir,
        tester_dir,
        research_dir,
        distill_dir,
        profiles_dir,
        runs_dir,
    ]:
        ensure_dir(path)

    patched = {}
    for mod_name in PATCH_MODULES:
        try:
            mod = importlib.import_module(mod_name)
        except Exception:
            continue
        patched[mod_name] = {}
        for attr, value in {
            "OFFLINE_DIR": offline_dir,
            "DEFAULT_DB": default_db,
            "RUNS_DIR": runs_dir,
            "PROFILES_DIR": profiles_dir,
            "RESEARCH_DIR": research_dir,
            "DISTILL_DIR": distill_dir,
            "COMMON_PROMOTION_DIR": common_promotion_dir,
            "COMMON_EXPORT_DIR": common_export_dir,
            "RUNTIME_DIR": runtime_dir,
            "COMMON_FILES": common_dir,
            "TESTER_PRESET_DIR": tester_dir,
            "ROOT": base_dir,
            "REPO_ROOT": base_dir.parent,
            "NEWSPULSE_DIR": newspulse_dir,
            "NEWSPULSE_STATE_DIR": newspulse_state_dir,
            "NEWSPULSE_REPORT_DIR": newspulse_report_dir,
            "NEWSPULSE_CONFIG_PATH": newspulse_dir / "newspulse_config.json",
            "NEWSPULSE_SOURCES_PATH": newspulse_dir / "newspulse_sources.json",
            "NEWSPULSE_POLICY_PATH": offline_dir / "NewsPulse/newspulse_policy.json",
            "NEWSPULSE_STATUS_PATH": newspulse_dir / "newspulse_status.json",
            "NEWSPULSE_LOCAL_HISTORY_PATH": newspulse_dir / "news_history.ndjson",
            "NEWSPULSE_STATE_PATH": newspulse_state_dir / "newspulse_state.json",
            "NEWSPULSE_REPLAY_REPORT_PATH": offline_dir / "NewsPulse/Reports/newspulse_replay_report.json",
            "ADAPTIVE_ROUTER_DIR": adaptive_router_dir,
            "ADAPTIVE_ROUTER_REPORT_DIR": adaptive_router_report_dir,
            "ADAPTIVE_ROUTER_CONFIG_PATH": adaptive_router_dir / "adaptive_router_config.json",
            "ADAPTIVE_ROUTER_REPLAY_REPORT_PATH": adaptive_router_report_dir / "adaptive_router_replay_report.json",
            "DYNAMIC_ENSEMBLE_DIR": dynamic_ensemble_dir,
            "DYNAMIC_ENSEMBLE_REPORT_DIR": dynamic_ensemble_report_dir,
            "DYNAMIC_ENSEMBLE_CONFIG_PATH": dynamic_ensemble_dir / "dynamic_ensemble_config.json",
            "DYNAMIC_ENSEMBLE_REPLAY_REPORT_PATH": dynamic_ensemble_report_dir / "dynamic_ensemble_replay_report.json",
            "DYNAMIC_ENSEMBLE_RUNTIME_CONFIG_PATH": runtime_dir / "dynamic_ensemble_config.tsv",
            "PROB_CALIBRATION_DIR": prob_calibration_dir,
            "PROB_CALIBRATION_REPORT_DIR": prob_calibration_report_dir,
            "PROB_CALIBRATION_CONFIG_PATH": prob_calibration_dir / "prob_calibration_config.json",
            "PROB_CALIBRATION_MEMORY_PATH": prob_calibration_dir / "prob_calibration_memory.json",
            "PROB_CALIBRATION_REPLAY_REPORT_PATH": prob_calibration_report_dir / "prob_calibration_replay_report.json",
            "PROB_CALIBRATION_RUNTIME_CONFIG_PATH": runtime_dir / "prob_calibration_config.tsv",
            "PROB_CALIBRATION_RUNTIME_MEMORY_PATH": runtime_dir / "prob_calibration_memory.tsv",
            "EXECUTION_QUALITY_DIR": execution_quality_dir,
            "EXECUTION_QUALITY_REPORT_DIR": execution_quality_report_dir,
            "EXECUTION_QUALITY_CONFIG_PATH": execution_quality_dir / "execution_quality_config.json",
            "EXECUTION_QUALITY_MEMORY_PATH": execution_quality_dir / "execution_quality_memory.json",
            "EXECUTION_QUALITY_REPLAY_REPORT_PATH": execution_quality_report_dir / "execution_quality_replay_report.json",
            "EXECUTION_QUALITY_RUNTIME_CONFIG_PATH": runtime_dir / "execution_quality_config.tsv",
            "EXECUTION_QUALITY_RUNTIME_MEMORY_PATH": runtime_dir / "execution_quality_memory.tsv",
            "RATES_ENGINE_DIR": rates_engine_dir,
            "RATES_ENGINE_STATE_DIR": rates_engine_state_dir,
            "RATES_ENGINE_REPORT_DIR": rates_engine_report_dir,
            "RATES_ENGINE_CONFIG_PATH": rates_engine_dir / "rates_engine_config.json",
            "RATES_ENGINE_INPUTS_PATH": rates_engine_dir / "rates_provider_inputs.json",
            "RATES_ENGINE_STATUS_PATH": rates_engine_dir / "rates_engine_status.json",
            "RATES_ENGINE_STATE_PATH": rates_engine_state_dir / "rates_engine_state.json",
            "RATES_ENGINE_REPLAY_REPORT_PATH": rates_engine_report_dir / "rates_replay_report.json",
            "RATES_ENGINE_LOCAL_HISTORY_PATH": rates_engine_dir / "rates_history.ndjson",
            "CROSS_ASSET_DIR": cross_asset_dir,
            "CROSS_ASSET_STATE_DIR": cross_asset_state_dir,
            "CROSS_ASSET_REPORT_DIR": cross_asset_report_dir,
            "CROSS_ASSET_CONFIG_PATH": cross_asset_dir / "cross_asset_config.json",
            "CROSS_ASSET_STATUS_PATH": cross_asset_dir / "cross_asset_status.json",
            "CROSS_ASSET_STATE_PATH": cross_asset_state_dir / "cross_asset_state.json",
            "CROSS_ASSET_REPLAY_REPORT_PATH": cross_asset_report_dir / "cross_asset_replay_report.json",
            "CROSS_ASSET_LOCAL_HISTORY_PATH": cross_asset_dir / "cross_asset_history.ndjson",
            "DRIFT_GOVERNANCE_DIR": drift_governance_dir,
            "DRIFT_GOVERNANCE_REPORT_DIR": drift_governance_report_dir,
            "DRIFT_GOVERNANCE_ARTIFACT_DIR": drift_governance_artifact_dir,
            "DRIFT_GOVERNANCE_CONFIG_PATH": drift_governance_dir / "drift_governance_config.json",
            "DRIFT_GOVERNANCE_STATUS_PATH": drift_governance_dir / "drift_governance_status.json",
            "DRIFT_GOVERNANCE_REPORT_PATH": drift_governance_report_dir / "drift_governance_report.json",
            "DRIFT_GOVERNANCE_HISTORY_PATH": drift_governance_dir / "drift_governance_history.ndjson",
            "DRIFT_GOVERNANCE_RUNTIME_SUMMARY_PATH": runtime_dir / "drift_governance_summary.json",
            "LABEL_ENGINE_DIR": label_engine_dir,
            "LABEL_ENGINE_REPORT_DIR": label_engine_report_dir,
            "LABEL_ENGINE_ARTIFACT_DIR": label_engine_artifact_dir,
            "LABEL_ENGINE_CONFIG_PATH": label_engine_dir / "label_engine_config.json",
            "LABEL_ENGINE_STATUS_PATH": label_engine_dir / "label_engine_status.json",
            "LABEL_ENGINE_REPORT_PATH": label_engine_report_dir / "label_engine_report.json",
            "LABEL_ENGINE_RUNTIME_SUMMARY_PATH": runtime_dir / "label_engine_summary.json",
            "MICROSTRUCTURE_DIR": microstructure_dir,
            "MICROSTRUCTURE_STATE_DIR": microstructure_state_dir,
            "MICROSTRUCTURE_REPORT_DIR": microstructure_report_dir,
            "MICROSTRUCTURE_CONFIG_PATH": microstructure_dir / "microstructure_config.json",
            "MICROSTRUCTURE_STATUS_PATH": microstructure_dir / "microstructure_status.json",
            "MICROSTRUCTURE_STATE_PATH": microstructure_state_dir / "microstructure_state.json",
            "MICROSTRUCTURE_REPLAY_REPORT_PATH": microstructure_report_dir / "microstructure_replay_report.json",
            "MICROSTRUCTURE_LOCAL_HISTORY_PATH": microstructure_dir / "microstructure_history.ndjson",
            "COMMON_NEWSPULSE_JSON": runtime_dir / "news_snapshot.json",
            "COMMON_NEWSPULSE_FLAT": runtime_dir / "news_snapshot_flat.tsv",
            "COMMON_NEWSPULSE_HISTORY": runtime_dir / "news_history.ndjson",
            "COMMON_NEWSPULSE_REPLAY_FLAT": runtime_dir / "news_replay_timeline.tsv",
            "COMMON_NEWSPULSE_SYMBOL_MAP": runtime_dir / "news_symbol_map.tsv",
            "COMMON_NEWSPULSE_CALENDAR_FEED": runtime_dir / "news_calendar_feed.tsv",
            "COMMON_NEWSPULSE_CALENDAR_STATE": runtime_dir / "news_calendar_state.tsv",
            "COMMON_NEWSPULSE_CALENDAR_HISTORY": runtime_dir / "news_calendar_history.ndjson",
            "COMMON_RATES_JSON": runtime_dir / "rates_snapshot.json",
            "COMMON_RATES_FLAT": runtime_dir / "rates_snapshot_flat.tsv",
            "COMMON_RATES_HISTORY": runtime_dir / "rates_history.ndjson",
            "COMMON_RATES_SYMBOL_MAP": runtime_dir / "rates_symbol_map.tsv",
            "COMMON_CROSS_ASSET_JSON": runtime_dir / "cross_asset_snapshot.json",
            "COMMON_CROSS_ASSET_FLAT": runtime_dir / "cross_asset_snapshot_flat.tsv",
            "COMMON_CROSS_ASSET_HISTORY": runtime_dir / "cross_asset_history.ndjson",
            "COMMON_CROSS_ASSET_STATUS": runtime_dir / "cross_asset_status.json",
            "COMMON_CROSS_ASSET_SYMBOL_MAP": runtime_dir / "cross_asset_symbol_map.tsv",
            "COMMON_CROSS_ASSET_CONFIG": runtime_dir / "cross_asset_probe_config.tsv",
            "COMMON_CROSS_ASSET_PROBE_JSON": runtime_dir / "cross_asset_probe_snapshot.json",
            "COMMON_CROSS_ASSET_PROBE_STATUS": runtime_dir / "cross_asset_probe_status.json",
            "COMMON_CROSS_ASSET_PROBE_HISTORY": runtime_dir / "cross_asset_probe_history.ndjson",
            "COMMON_MICROSTRUCTURE_JSON": runtime_dir / "microstructure_snapshot.json",
            "COMMON_MICROSTRUCTURE_FLAT": runtime_dir / "microstructure_snapshot_flat.tsv",
            "COMMON_MICROSTRUCTURE_HISTORY": runtime_dir / "microstructure_history.ndjson",
            "COMMON_MICROSTRUCTURE_STATUS": runtime_dir / "microstructure_status.json",
            "COMMON_MICROSTRUCTURE_SYMBOL_MAP": runtime_dir / "microstructure_symbol_map.tsv",
            "COMMON_MICROSTRUCTURE_CONFIG": runtime_dir / "microstructure_service_config.tsv",
            "COMMON_PROB_CALIBRATION_RUNTIME_CONFIG": runtime_dir / "prob_calibration_config.tsv",
            "COMMON_PROB_CALIBRATION_RUNTIME_MEMORY": runtime_dir / "prob_calibration_memory.tsv",
            "COMMON_EXECUTION_QUALITY_RUNTIME_CONFIG": runtime_dir / "execution_quality_config.tsv",
            "COMMON_EXECUTION_QUALITY_RUNTIME_MEMORY": runtime_dir / "execution_quality_memory.tsv",
        }.items():
            if hasattr(mod, attr):
                patched[mod_name][attr] = getattr(mod, attr)
                setattr(mod, attr, value)
    try:
        yield {
            "base_dir": base_dir,
            "offline_dir": offline_dir,
            "default_db": default_db,
            "common_dir": common_dir,
            "common_promotion_dir": common_promotion_dir,
            "runtime_dir": runtime_dir,
            "research_dir": research_dir,
            "distill_dir": distill_dir,
            "profiles_dir": profiles_dir,
            "tester_dir": tester_dir,
            "adaptive_router_dir": adaptive_router_dir,
            "dynamic_ensemble_dir": dynamic_ensemble_dir,
            "prob_calibration_dir": prob_calibration_dir,
            "rates_engine_dir": rates_engine_dir,
            "cross_asset_dir": cross_asset_dir,
            "label_engine_dir": label_engine_dir,
            "drift_governance_dir": drift_governance_dir,
            "microstructure_dir": microstructure_dir,
        }
    finally:
        for mod_name, attrs in patched.items():
            try:
                mod = importlib.import_module(mod_name)
            except Exception:
                continue
            for attr, value in attrs.items():
                setattr(mod, attr, value)


def _artifact_file(path: Path, text: str) -> tuple[str, str]:
    ensure_dir(path.parent)
    path.write_text(text, encoding="utf-8")
    return str(path), sha256_text(text)


def seed_profile_fixture(conn: libsql.Connection,
                         profile_name: str = "fixture",
                         symbol: str = "EURUSD",
                         plugin_name: str = "ai_mlp",
                         family_id: int = 2) -> dict[str, object]:
    now = now_unix()
    start_unix = now - 90 * 86400
    end_unix = now
    dataset_key = f"{profile_name}:{symbol}:m1:fixture"
    group_key = f"{profile_name}_fixture"
    dataset_path = f"/fixture/{safe_token(symbol)}.csv"

    conn.execute("DELETE FROM datasets WHERE dataset_key = ?", (dataset_key,))
    conn.execute(
        """
        INSERT OR REPLACE INTO datasets(dataset_key, group_key, symbol, timeframe, start_unix, end_unix, months, bars,
                                        source_path, source_sha256, created_at, notes)
        VALUES(?, ?, ?, 'M1', ?, ?, 3, 1000, ?, ?, ?, 'fixture')
        """,
        (dataset_key, group_key, symbol, start_unix, end_unix, dataset_path, sha256_text(dataset_path), now),
    )
    dataset_id = int(query_scalar(conn, "SELECT id FROM datasets WHERE dataset_key = ?", (dataset_key,), 0))

    params = {"PredictionTargetMinutes": 5, "M1SyncBars": 3, "FeatureSchema": 6}
    params_json = json.dumps(params, sort_keys=True)
    audit_set_path = f"/fixture/{safe_token(symbol)}_{plugin_name}_audit.set"
    ea_set_path = f"/fixture/{safe_token(symbol)}_{plugin_name}_ea.set"
    conn.execute(
        """
        INSERT OR REPLACE INTO best_configs(dataset_scope, dataset_id, profile_name, symbol, plugin_name, ai_id, family_id,
                                            run_id, promoted_at, score, ranking_score, support_count, parameters_json,
                                            audit_set_path, ea_set_path, support_json)
        VALUES('aggregate', ?, ?, ?, ?, 4, ?, 1, ?, 83.5, 82.7, 3, ?, ?, ?, '[]')
        """,
        (dataset_id, profile_name, symbol, plugin_name, family_id, now, params_json, audit_set_path, ea_set_path),
    )
    best_config_id = int(
        query_scalar(
            conn,
            "SELECT id FROM best_configs WHERE dataset_scope='aggregate' AND profile_name=? AND symbol=? AND plugin_name=?",
            (profile_name, symbol, plugin_name),
            0,
        )
    )

    conn.execute(
        """
        INSERT OR REPLACE INTO champion_registry(profile_name, symbol, plugin_name, family_id, champion_best_config_id,
                                                 challenger_run_id, status, champion_score, challenger_score,
                                                 portfolio_score, promoted_at, reviewed_at, champion_set_path, notes)
        VALUES(?, ?, ?, ?, ?, NULL, 'champion', 83.5, 81.2, 0.72, ?, ?, ?, '{}')
        """,
        (profile_name, symbol, plugin_name, family_id, best_config_id, now, now, ea_set_path),
    )

    conn.execute(
        """
        INSERT OR REPLACE INTO family_scorecards(profile_name, group_key, symbol, family_id, family_name, run_count,
                                                 mean_score, mean_recent_score, mean_walkforward_score,
                                                 mean_adversarial_score, mean_macro_score, mean_issue_count,
                                                 stability_score, promotion_count, champion_count, payload_json, created_at)
        VALUES(?, ?, ?, ?, 'recurrent', 12, 82.0, 80.0, 78.0, 76.0, 74.0, 0.0, 0.71, 2, 1, '{}', ?)
        """,
        (profile_name, group_key, symbol, family_id, now),
    )

    payload_json = json.dumps(
        {
            "captured_from": "fixture",
            "policy": {"mode": "research"},
            "route": {"plugin": plugin_name},
        },
        sort_keys=True,
    )
    conn.execute(
        """
        INSERT OR REPLACE INTO shadow_fleet_observations(profile_name, symbol, plugin_name, family_id, captured_at,
                                                         source_path, source_sha256, meta_weight, reliability, global_edge,
                                                         context_edge, context_regret, portfolio_objective, portfolio_stability,
                                                         portfolio_corr, portfolio_div, route_value, route_regret,
                                                         route_counterfactual, shadow_score, regime_id, horizon_minutes, obs_count,
                                                         policy_enter_prob, policy_no_trade_prob, policy_exit_prob, policy_add_prob,
                                                         policy_reduce_prob, policy_timeout_prob, policy_tighten_prob,
                                                         policy_portfolio_fit, policy_capital_efficiency, policy_lifecycle_action,
                                                         portfolio_pressure, control_plane_score, portfolio_supervisor_score,
                                                         payload_json)
        VALUES(?, ?, ?, ?, ?, ?, ?, 1.12, 0.78, 0.34, 0.28, 0.18, 0.42, 0.68, 0.22, 0.71,
               0.26, 0.14, 0.18, 0.44, 1, 5, 32, 0.61, 0.32, 0.14, 0.27, 0.21, 0.19, 0.23,
               0.66, 0.72, 2, 0.28, 0.36, 0.44, ?)
        """,
        (
            profile_name,
            symbol,
            plugin_name,
            family_id,
            now,
            f"/fixture/{safe_token(symbol)}_shadow.tsv",
            sha256_text(f"{profile_name}:{symbol}:{plugin_name}:{now}"),
            payload_json,
        ),
    )
    commit_db(conn)
    return {
        "profile_name": profile_name,
        "symbol": symbol,
        "plugin_name": plugin_name,
        "family_id": family_id,
        "group_key": group_key,
        "dataset_key": dataset_key,
    }


def seed_completed_run_fixture(conn: libsql.Connection,
                               fixture: dict[str, object],
                               parameters: dict[str, object] | None = None,
                               score: float = 83.5) -> dict[str, object]:
    dataset_row = query_one(
        conn,
        "SELECT id, symbol, dataset_key FROM datasets WHERE dataset_key = ?",
        (str(fixture["dataset_key"]),),
    )
    if dataset_row is None:
        raise ValueError("fixture dataset not found")

    params = dict(parameters or {
        "bars": 1000,
        "horizon": 5,
        "m1sync_bars": 3,
        "normalization": 14,
        "sequence_bars": 16,
        "schema_id": 6,
        "feature_mask": 0,
        "execution_profile": "default",
        "wf_train_bars": 256,
        "wf_test_bars": 64,
        "wf_purge_bars": 32,
        "wf_embargo_bars": 24,
    })
    parameters_json = json_compact(params)
    param_hash = sha256_text(
        f"{int(dataset_row['id'])}|{fixture['profile_name']}|{fixture['plugin_name']}|{parameters_json}"
    )
    started_at = now_unix()
    finished_at = started_at + 1
    conn.execute(
        """
        INSERT OR REPLACE INTO tuning_runs(
            dataset_id, profile_name, group_key, symbol, plugin_name, ai_id, family_id, experiment_name, param_hash, parameters_json,
            report_path, raw_report_path, summary_path, manifest_path, score, grade, issue_count, issues_json,
            market_recent_score, walkforward_score, adversarial_score, macro_event_score, status, started_at, finished_at
        )
        VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'ok', ?, ?)
        """,
        (
            int(dataset_row["id"]),
            str(fixture["profile_name"]),
            str(fixture["group_key"]),
            str(fixture["symbol"]),
            str(fixture["plugin_name"]),
            4,
            int(fixture["family_id"]),
            "fixture_completed_run",
            param_hash,
            parameters_json,
            "/fixture/report.tsv",
            "/fixture/raw.tsv",
            "/fixture/summary.json",
            "/fixture/manifest.json",
            float(score),
            "A",
            0,
            "[]",
            float(score) - 2.0,
            float(score) - 4.0,
            float(score) - 6.0,
            float(score) - 9.0,
            started_at,
            finished_at,
        ),
    )
    run_row = query_one(
        conn,
        "SELECT id FROM tuning_runs WHERE param_hash = ?",
        (param_hash,),
    )
    if run_row is None:
        raise RuntimeError("failed to seed tuning run fixture")
    run_id = int(run_row["id"])
    conn.execute("DELETE FROM run_scenarios WHERE run_id = ?", (run_id,))
    for scenario_name, scenario_score in [
        ("market_recent", float(score) - 2.0),
        ("market_walkforward", float(score) - 4.0),
        ("market_adversarial", float(score) - 6.0),
        ("market_macro_event", float(score) - 9.0),
    ]:
        conn.execute(
            """
            INSERT INTO run_scenarios(run_id, scenario, score, calibration_error, path_quality_error, wf_pbo, wf_dsr, wf_pass_rate, net_signal, issue_flags)
            VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (run_id, scenario_name, scenario_score, 0.10, 0.20, 0.15, 0.55, 0.62, 0.0, 0),
        )
    commit_db(conn)
    return {
        "run_id": run_id,
        "dataset_id": int(dataset_row["id"]),
        "parameters_json": parameters_json,
        "param_hash": param_hash,
    }


def clear_generated_outputs(paths: dict[str, Path], profile_name: str) -> None:
    for root in [paths["research_dir"], paths["distill_dir"], paths["profiles_dir"]]:
        target = root / safe_token(profile_name)
        if target.exists():
            shutil.rmtree(target)
    for candidate in paths["common_promotion_dir"].glob("fxai_*"):
        if candidate.is_file():
            candidate.unlink()
