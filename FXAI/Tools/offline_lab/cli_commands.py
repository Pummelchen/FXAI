from __future__ import annotations

import argparse
import json
from pathlib import Path
import libsql

from .attribution import *
from .adaptive_router import write_adaptive_router_profiles
from .adaptive_router_config import load_config as load_adaptive_router_config
from .adaptive_router_contracts import ADAPTIVE_ROUTER_CONFIG_PATH
from .adaptive_router_replay import build_adaptive_router_replay_report
from .bundle import *
from .common import *
from .cross_asset_engine import (
    cross_asset_health_snapshot,
    run_cross_asset_daemon,
    run_cross_asset_once,
    validate_cross_asset_config,
)
from .cross_asset_replay import build_cross_asset_replay_report
from .cross_asset_service import install_cross_asset_service
from .drift_governance import (
    build_drift_governance_report,
    run_drift_governance_cycle,
    validate_drift_governance_config,
)
from .dynamic_ensemble_config import load_config as load_dynamic_ensemble_config
from .dynamic_ensemble_contracts import DYNAMIC_ENSEMBLE_CONFIG_PATH
from .dynamic_ensemble_replay import build_dynamic_ensemble_replay_report
from .dashboard import live_state_snapshot, write_profile_dashboard
from .environment import bootstrap_environment, validate_environment, write_environment_report
from .execution_quality_config import (
    EXECUTION_QUALITY_CONFIG_PATH,
    EXECUTION_QUALITY_MEMORY_PATH,
    load_config as load_execution_quality_config,
    load_memory as load_execution_quality_memory,
)
from .execution_quality_replay import build_execution_quality_replay_report
from .exporter import *
from .fixtures import seed_profile_fixture
from .foundation_factory import *
from .governance import *
from .label_engine import build_label_engine_artifacts, build_label_engine_report, validate_label_engine_config
from .lineage import *
from .market_universe import (
    export_market_universe_config,
    import_market_universe_config,
    load_market_universe_config,
    reset_market_universe_config,
    summarize_market_universe_config,
)
from .microstructure_config import load_config as load_microstructure_config
from .microstructure_contracts import MICROSTRUCTURE_CONFIG_PATH
from .microstructure_replay import build_microstructure_replay_report
from .microstructure_service import (
    install_microstructure_service,
    microstructure_health_snapshot,
    validate_microstructure_config,
)
from .mode import RUNTIME_MODES, resolve_runtime_mode
from .newspulse_daemon import (
    newspulse_health_snapshot,
    run_newspulse_daemon,
    run_newspulse_once,
    validate_newspulse_config,
)
from .newspulse_replay import rebuild_replay_report_from_history
from .newspulse_service import install_calendar_service
from .performance import write_performance_reports
from .promotion import *
from .prob_calibration_config import (
    PROB_CALIBRATION_CONFIG_PATH,
    PROB_CALIBRATION_MEMORY_PATH,
    load_config as load_prob_calibration_config,
    load_memory as load_prob_calibration_memory,
)
from .prob_calibration_replay import build_prob_calibration_replay_report
from .rates_engine_daemon import (
    rates_engine_health_snapshot,
    run_rates_engine_daemon,
    run_rates_engine_once,
    validate_rates_engine_config,
)
from .rates_engine_replay import build_rates_replay_report
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


from .cli_campaigns import *

def cmd_init_db(args) -> int:
    conn = connect_db(Path(args.db))
    try:
        print(f"initialized Turso/libSQL lab: {args.db}")
        return 0
    finally:
        close_db(conn)


def cmd_validate_env(_args) -> int:
    payload = validate_environment()
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if bool(payload.get("ok")) else 1


def cmd_newspulse_validate(_args) -> int:
    payload = validate_newspulse_config()
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if bool(payload.get("ok")) else 1


def cmd_newspulse_once(_args) -> int:
    payload = run_newspulse_once()
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def cmd_newspulse_daemon(args) -> int:
    payload = run_newspulse_daemon(
        iterations=int(getattr(args, "iterations", 0) or 0),
        interval_seconds=int(getattr(args, "interval_seconds", 0) or 0),
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def cmd_newspulse_health(_args) -> int:
    payload = newspulse_health_snapshot()
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def cmd_newspulse_replay_report(args) -> int:
    payload = rebuild_replay_report_from_history(
        pair_filter=str(getattr(args, "pair", "") or ""),
        hours_back=int(getattr(args, "hours_back", 48) or 48),
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def cmd_newspulse_install_service(args) -> int:
    payload = install_calendar_service(compile_service=not bool(getattr(args, "skip_compile", False)))
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def cmd_adaptive_router_validate(_args) -> int:
    payload = load_adaptive_router_config()
    print(json.dumps({
        "status": "ok",
        "config_path": str(ADAPTIVE_ROUTER_CONFIG_PATH),
        "config": payload,
    }, indent=2, sort_keys=True))
    return 0


def cmd_dynamic_ensemble_validate(_args) -> int:
    payload = load_dynamic_ensemble_config()
    print(json.dumps({
        "status": "ok",
        "config_path": str(DYNAMIC_ENSEMBLE_CONFIG_PATH),
        "config": payload,
    }, indent=2, sort_keys=True))
    return 0


def cmd_dynamic_ensemble_replay_report(args) -> int:
    payload = build_dynamic_ensemble_replay_report(
        symbol=str(getattr(args, "symbol", "") or ""),
        hours_back=int(getattr(args, "hours_back", 72) or 72),
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def cmd_prob_calibration_validate(_args) -> int:
    config_payload = load_prob_calibration_config()
    memory_payload = load_prob_calibration_memory()
    print(json.dumps({
        "status": "ok",
        "config_path": str(PROB_CALIBRATION_CONFIG_PATH),
        "memory_path": str(PROB_CALIBRATION_MEMORY_PATH),
        "config": config_payload,
        "memory": memory_payload,
    }, indent=2, sort_keys=True))
    return 0


def cmd_prob_calibration_replay_report(args) -> int:
    payload = build_prob_calibration_replay_report(
        symbol=str(getattr(args, "symbol", "") or ""),
        hours_back=int(getattr(args, "hours_back", 72) or 72),
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def cmd_execution_quality_validate(_args) -> int:
    config_payload = load_execution_quality_config()
    memory_payload = load_execution_quality_memory()
    print(json.dumps({
        "status": "ok",
        "config_path": str(EXECUTION_QUALITY_CONFIG_PATH),
        "memory_path": str(EXECUTION_QUALITY_MEMORY_PATH),
        "config": config_payload,
        "memory": memory_payload,
    }, indent=2, sort_keys=True))
    return 0


def cmd_execution_quality_replay_report(args) -> int:
    payload = build_execution_quality_replay_report(
        symbol=str(getattr(args, "symbol", "") or ""),
        hours_back=int(getattr(args, "hours_back", 72) or 72),
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def cmd_drift_governance_validate(_args) -> int:
    payload = validate_drift_governance_config()
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if bool(payload.get("ok")) else 1


def cmd_drift_governance_run(args) -> int:
    conn = connect_db(Path(args.db))
    try:
        payload = run_drift_governance_cycle(conn, args, str(getattr(args, "group_key", "") or ""))
        router_payload = write_student_router_profiles(conn, args)
        adaptive_router_payload = write_adaptive_router_profiles(conn, args)
        dashboard = write_profile_dashboard(conn, args.profile)
        print(json.dumps({
            "profile": args.profile,
            "drift_governance": payload,
            "student_router_profiles": len(router_payload),
            "adaptive_router_profiles": len(adaptive_router_payload),
            "dashboard": dashboard,
        }, indent=2, sort_keys=True))
        return 0
    finally:
        close_db(conn)


def cmd_drift_governance_report(args) -> int:
    conn = connect_db(Path(args.db))
    try:
        payload = build_drift_governance_report(conn, args.profile)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    finally:
        close_db(conn)


def cmd_label_engine_validate(_args) -> int:
    payload = validate_label_engine_config()
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if bool(payload.get("ok")) else 1


def cmd_label_engine_build(args) -> int:
    conn = connect_db(Path(args.db))
    try:
        payload = build_label_engine_artifacts(conn, args=args)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    finally:
        close_db(conn)


def cmd_label_engine_report(args) -> int:
    conn = connect_db(Path(args.db))
    try:
        payload = build_label_engine_report(conn, profile_name=str(getattr(args, "profile", "") or ""))
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    finally:
        close_db(conn)


def cmd_microstructure_validate(_args) -> int:
    payload = validate_microstructure_config()
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if bool(payload.get("ok")) else 1


def cmd_microstructure_install_service(args) -> int:
    payload = install_microstructure_service(compile_service=not bool(getattr(args, "skip_compile", False)))
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def cmd_microstructure_health(_args) -> int:
    payload = microstructure_health_snapshot()
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def cmd_microstructure_replay_report(args) -> int:
    payload = build_microstructure_replay_report(
        symbol=str(getattr(args, "symbol", "") or ""),
        hours_back=int(getattr(args, "hours_back", 24) or 24),
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def cmd_cross_asset_validate(_args) -> int:
    payload = validate_cross_asset_config()
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if bool(payload.get("ok")) else 1


def cmd_cross_asset_once(_args) -> int:
    payload = run_cross_asset_once()
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def cmd_cross_asset_daemon(args) -> int:
    payload = run_cross_asset_daemon(
        iterations=int(getattr(args, "iterations", 0) or 0),
        interval_seconds=int(getattr(args, "interval_seconds", 0) or 0),
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def cmd_cross_asset_health(_args) -> int:
    payload = cross_asset_health_snapshot()
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def cmd_cross_asset_install_service(args) -> int:
    payload = install_cross_asset_service(compile_service=not bool(getattr(args, "skip_compile", False)))
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def cmd_cross_asset_replay_report(args) -> int:
    payload = build_cross_asset_replay_report(
        symbol=str(getattr(args, "symbol", "") or ""),
        hours_back=int(getattr(args, "hours_back", 72) or 72),
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def cmd_rates_engine_validate(_args) -> int:
    payload = validate_rates_engine_config()
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if bool(payload.get("ok")) else 1


def cmd_rates_engine_once(_args) -> int:
    payload = run_rates_engine_once()
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def cmd_rates_engine_daemon(args) -> int:
    payload = run_rates_engine_daemon(
        iterations=int(getattr(args, "iterations", 0) or 0),
        interval_seconds=int(getattr(args, "interval_seconds", 0) or 0),
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def cmd_rates_engine_health(_args) -> int:
    payload = rates_engine_health_snapshot()
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def cmd_rates_engine_replay_report(args) -> int:
    payload = build_rates_replay_report(
        symbol=str(getattr(args, "symbol", "") or ""),
        hours_back=int(getattr(args, "hours_back", 72) or 72),
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def cmd_adaptive_router_profiles(args) -> int:
    conn = connect_db(Path(args.db))
    try:
        payload = write_adaptive_router_profiles(conn, args)
        commit_db(conn)
        print(json.dumps({
            "profile": args.profile,
            "runtime_mode": resolve_runtime_mode(getattr(args, "runtime_mode", None)),
            "adaptive_router_profiles": payload,
        }, indent=2, sort_keys=True))
        return 0
    finally:
        close_db(conn)


def cmd_adaptive_router_replay_report(args) -> int:
    payload = build_adaptive_router_replay_report(
        symbol=str(getattr(args, "symbol", "") or ""),
        hours_back=int(getattr(args, "hours_back", 72) or 72),
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def cmd_market_universe_show(args) -> int:
    conn = connect_db(Path(args.db))
    try:
        payload = load_market_universe_config(conn)
        output = {
            "summary": summarize_market_universe_config(payload),
            "config": payload,
        }
        if bool(getattr(args, "summary_only", False)):
            output.pop("config", None)
        print(json.dumps(output, indent=2, sort_keys=True))
        return 0
    finally:
        close_db(conn)


def cmd_market_universe_reset_defaults(args) -> int:
    conn = connect_db(Path(args.db))
    try:
        payload = reset_market_universe_config(conn)
        commit_db(conn)
        print(json.dumps({
            "status": "reset_to_defaults",
            "summary": summarize_market_universe_config(payload),
        }, indent=2, sort_keys=True))
        return 0
    finally:
        close_db(conn)


def cmd_market_universe_export(args) -> int:
    conn = connect_db(Path(args.db))
    try:
        payload = export_market_universe_config(
            conn,
            str(getattr(args, "output_path", "") or "").strip() or None,
        )
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    finally:
        close_db(conn)


def cmd_market_universe_import(args) -> int:
    conn = connect_db(Path(args.db))
    try:
        payload = import_market_universe_config(conn, str(getattr(args, "input_path", "") or ""))
        commit_db(conn)
        print(json.dumps({
            "status": "imported",
            **payload,
        }, indent=2, sort_keys=True))
        return 0
    finally:
        close_db(conn)


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
            adaptive_router_payload = write_adaptive_router_profiles(conn, demo_args)
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
                "adaptive_router_profiles": len(adaptive_router_payload),
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
    try:
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
        return 0
    finally:
        close_db(conn)


def cmd_tune_zoo(args) -> int:
    conn = connect_db(Path(args.db))
    try:
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
        return 0
    finally:
        close_db(conn)


def cmd_best_params(args) -> int:
    conn = connect_db(Path(args.db))
    try:
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
        governance_payload = run_autonomous_governance(conn, args, str(getattr(args, "group_key", "") or ""))
        student_router_profiles = write_student_router_profiles(conn, args)
        adaptive_router_profiles = write_adaptive_router_profiles(conn, args)
        live_deployments = write_live_deployment_profiles(conn, args)
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
            "adaptive_router_profiles": len(adaptive_router_profiles),
            "live_deployments": len(live_deployments),
            "shadow_rows_ingested": int(shadow_ingest.get("rows_ingested", 0)),
            "governance_decisions": len(governance_payload.get("decisions", [])),
            "drift_governance": governance_payload.get("drift_governance", {}),
            "world_plans": len(governance_payload.get("world_plans", [])),
            "portfolio_supervisor_artifact": str(governance_payload.get("portfolio_supervisor", {}).get("artifact_path", "")),
            "supervisor_service_artifacts": len(governance_payload.get("supervisor_service", [])),
            "supervisor_command_artifacts": len(governance_payload.get("supervisor_commands", [])),
        }, indent=2, sort_keys=True))
        return 0
    finally:
        close_db(conn)


def cmd_shadow_sync(args) -> int:
    conn = connect_db(Path(args.db))
    try:
        payload = ingest_shadow_fleet_ledgers(conn, args.profile)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    finally:
        close_db(conn)


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
        adaptive_router_payload = write_adaptive_router_profiles(conn, demo_args)
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
                    "adaptive_router_profiles": len(adaptive_router_payload),
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
        adaptive_router_payload = write_adaptive_router_profiles(conn, args)
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
            "adaptive_router_profiles": adaptive_router_payload,
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
        vector_payload = refresh_research_vectors(conn, args.profile)
        payload = run_autonomous_governance(conn, args, str(getattr(args, "group_key", "") or ""))
        router_payload = write_student_router_profiles(conn, args)
        adaptive_router_payload = write_adaptive_router_profiles(conn, args)
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
            "adaptive_router_profiles": len(adaptive_router_payload),
            "vector_refresh": vector_payload,
            "governance_decisions": len(payload.get("decisions", [])),
            "drift_governance": payload.get("drift_governance", {}),
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
    try:
        payload = write_supervisor_service_artifacts(conn, args)
        commands = write_supervisor_command_artifacts(conn, args)
        print(json.dumps({"profile": args.profile, "artifacts": payload, "commands": commands}, indent=2, sort_keys=True))
        return 0
    finally:
        close_db(conn)


def cmd_attribution_prune(args) -> int:
    conn = connect_db(Path(args.db))
    try:
        attribution_payload = write_attribution_profiles(conn, args)
        router_payload = write_student_router_profiles(conn, args)
        adaptive_router_payload = write_adaptive_router_profiles(conn, args)
        print(json.dumps({
            "profile": args.profile,
            "runtime_mode": resolve_runtime_mode(getattr(args, "runtime_mode", None)),
            "attribution_profiles": attribution_payload,
            "student_router_profiles": router_payload,
            "adaptive_router_profiles": adaptive_router_payload,
        }, indent=2, sort_keys=True))
        return 0
    finally:
        close_db(conn)


def cmd_recover_artifacts(args) -> int:
    conn = connect_db(Path(args.db))
    try:
        attribution_payload = write_attribution_profiles(conn, args)
        router_payload = write_student_router_profiles(conn, args)
        adaptive_router_payload = write_adaptive_router_profiles(conn, args)
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
            "adaptive_router_profiles": len(adaptive_router_payload),
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
    try:
        payload = run_supervisor_daemon(conn, args)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    finally:
        close_db(conn)


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
