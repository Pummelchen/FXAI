#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import sys
import tempfile
import time
from collections import defaultdict
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path

import fxai_testlab as testlab

from .db_backend import (
    TURSO_AUTH_TOKEN_ENV,
    TURSO_DATABASE_URL_ENV,
    LabConnection,
    TursoConfig,
    connect_backend,
)

OFFLINE_DIR = Path(__file__).resolve().parent.parent / "OfflineLab"
DEFAULT_DB = OFFLINE_DIR / "fxai_offline_lab.sqlite"
RUNS_DIR = OFFLINE_DIR / "Runs"
PROFILES_DIR = OFFLINE_DIR / "Profiles"
RESEARCH_DIR = OFFLINE_DIR / "ResearchOS"
DISTILL_DIR = OFFLINE_DIR / "Distillation"
COMMON_EXPORT_DIR = testlab.COMMON_FILES / "FXAI/Offline/Exports"
COMMON_PROMOTION_DIR = testlab.COMMON_FILES / "FXAI/Offline/Promotions"
SHADOW_LEDGER_DIR = testlab.RUNTIME_DIR

SERIOUS_SCENARIOS = "{market_recent, market_trend, market_chop, market_session_edges, market_spread_shock, market_walkforward, market_macro_event, market_adversarial}"
DEFAULT_MONTHS_LIST = [3, 6, 12]
DEFAULT_HORIZON_CANDIDATES = [3, 5, 8, 13, 21, 34]
DEFAULT_M1SYNC_CANDIDATES = [2, 3, 5, 8]
DEFAULT_EXECUTION_PROFILES = ["default", "tight-fx", "prime-ecn", "retail-fx", "stress"]
EXPORT_EXPERT = r"FXAI\Tests\FXAI_OfflineExportRunner.ex5"
OFFLINE_SCHEMA_VERSION = 3
OFFLINE_ARTIFACT_SCHEMA_VERSION = 2
OFFLINE_MACRO_SCHEMA_MIN = 2

SQL_SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS lab_metadata (
    meta_key TEXT PRIMARY KEY,
    meta_value TEXT NOT NULL DEFAULT '',
    updated_at INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS datasets (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    dataset_key TEXT NOT NULL UNIQUE,
    group_key TEXT NOT NULL DEFAULT '',
    symbol TEXT NOT NULL,
    timeframe TEXT NOT NULL,
    start_unix INTEGER NOT NULL,
    end_unix INTEGER NOT NULL,
    months INTEGER NOT NULL DEFAULT 0,
    bars INTEGER NOT NULL DEFAULT 0,
    source_path TEXT NOT NULL,
    source_sha256 TEXT NOT NULL DEFAULT '',
    created_at INTEGER NOT NULL,
    notes TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS dataset_bars (
    dataset_id INTEGER NOT NULL,
    bar_time_unix INTEGER NOT NULL,
    open REAL NOT NULL,
    high REAL NOT NULL,
    low REAL NOT NULL,
    close REAL NOT NULL,
    spread_points INTEGER NOT NULL,
    tick_volume INTEGER NOT NULL,
    real_volume INTEGER NOT NULL,
    PRIMARY KEY(dataset_id, bar_time_unix),
    FOREIGN KEY(dataset_id) REFERENCES datasets(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS tuning_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    dataset_id INTEGER NOT NULL,
    profile_name TEXT NOT NULL,
    group_key TEXT NOT NULL DEFAULT '',
    symbol TEXT NOT NULL,
    plugin_name TEXT NOT NULL,
    ai_id INTEGER NOT NULL,
    family_id INTEGER NOT NULL DEFAULT 11,
    experiment_name TEXT NOT NULL,
    param_hash TEXT NOT NULL UNIQUE,
    parameters_json TEXT NOT NULL,
    report_path TEXT NOT NULL DEFAULT '',
    raw_report_path TEXT NOT NULL DEFAULT '',
    summary_path TEXT NOT NULL DEFAULT '',
    manifest_path TEXT NOT NULL DEFAULT '',
    score REAL NOT NULL DEFAULT 0.0,
    grade TEXT NOT NULL DEFAULT 'F',
    issue_count INTEGER NOT NULL DEFAULT 0,
    issues_json TEXT NOT NULL DEFAULT '[]',
    market_recent_score REAL NOT NULL DEFAULT 0.0,
    walkforward_score REAL NOT NULL DEFAULT 0.0,
    adversarial_score REAL NOT NULL DEFAULT 0.0,
    macro_event_score REAL NOT NULL DEFAULT 0.0,
    status TEXT NOT NULL DEFAULT 'pending',
    started_at INTEGER NOT NULL,
    finished_at INTEGER NOT NULL DEFAULT 0,
    FOREIGN KEY(dataset_id) REFERENCES datasets(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS run_scenarios (
    run_id INTEGER NOT NULL,
    scenario TEXT NOT NULL,
    score REAL NOT NULL DEFAULT 0.0,
    calibration_error REAL NOT NULL DEFAULT 0.0,
    path_quality_error REAL NOT NULL DEFAULT 0.0,
    wf_pbo REAL NOT NULL DEFAULT 0.0,
    wf_dsr REAL NOT NULL DEFAULT 0.0,
    wf_pass_rate REAL NOT NULL DEFAULT 0.0,
    net_signal REAL NOT NULL DEFAULT 0.0,
    issue_flags INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY(run_id, scenario),
    FOREIGN KEY(run_id) REFERENCES tuning_runs(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS best_configs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    dataset_scope TEXT NOT NULL,
    dataset_id INTEGER,
    profile_name TEXT NOT NULL,
    symbol TEXT NOT NULL,
    plugin_name TEXT NOT NULL,
    ai_id INTEGER NOT NULL,
    family_id INTEGER NOT NULL DEFAULT 11,
    run_id INTEGER,
    promoted_at INTEGER NOT NULL,
    score REAL NOT NULL DEFAULT 0.0,
    ranking_score REAL NOT NULL DEFAULT 0.0,
    support_count INTEGER NOT NULL DEFAULT 0,
    parameters_json TEXT NOT NULL,
    audit_set_path TEXT NOT NULL,
    ea_set_path TEXT NOT NULL,
    support_json TEXT NOT NULL DEFAULT '[]',
    UNIQUE(dataset_scope, profile_name, symbol, plugin_name)
);

CREATE TABLE IF NOT EXISTS control_cycles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    profile_name TEXT NOT NULL,
    group_key TEXT NOT NULL,
    started_at INTEGER NOT NULL,
    finished_at INTEGER NOT NULL DEFAULT 0,
    status TEXT NOT NULL DEFAULT 'running',
    datasets_json TEXT NOT NULL DEFAULT '[]',
    notes TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS champion_registry (
    profile_name TEXT NOT NULL,
    symbol TEXT NOT NULL,
    plugin_name TEXT NOT NULL,
    family_id INTEGER NOT NULL DEFAULT 11,
    champion_best_config_id INTEGER,
    challenger_run_id INTEGER,
    status TEXT NOT NULL DEFAULT 'candidate',
    champion_score REAL NOT NULL DEFAULT 0.0,
    challenger_score REAL NOT NULL DEFAULT 0.0,
    portfolio_score REAL NOT NULL DEFAULT 0.0,
    promotion_tier TEXT NOT NULL DEFAULT 'experimental',
    promoted_at INTEGER NOT NULL DEFAULT 0,
    reviewed_at INTEGER NOT NULL DEFAULT 0,
    champion_set_path TEXT NOT NULL DEFAULT '',
    notes TEXT NOT NULL DEFAULT '',
    PRIMARY KEY(profile_name, symbol, plugin_name)
);

CREATE TABLE IF NOT EXISTS config_lineage (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    profile_name TEXT NOT NULL,
    symbol TEXT NOT NULL,
    plugin_name TEXT NOT NULL,
    family_id INTEGER NOT NULL DEFAULT 11,
    source_run_id INTEGER,
    best_config_id INTEGER,
    relation TEXT NOT NULL DEFAULT 'candidate',
    lineage_hash TEXT NOT NULL DEFAULT '',
    payload_json TEXT NOT NULL DEFAULT '{}',
    created_at INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS family_scorecards (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    profile_name TEXT NOT NULL,
    group_key TEXT NOT NULL DEFAULT '',
    symbol TEXT NOT NULL,
    family_id INTEGER NOT NULL,
    family_name TEXT NOT NULL,
    run_count INTEGER NOT NULL DEFAULT 0,
    mean_score REAL NOT NULL DEFAULT 0.0,
    mean_recent_score REAL NOT NULL DEFAULT 0.0,
    mean_walkforward_score REAL NOT NULL DEFAULT 0.0,
    mean_adversarial_score REAL NOT NULL DEFAULT 0.0,
    mean_macro_score REAL NOT NULL DEFAULT 0.0,
    mean_issue_count REAL NOT NULL DEFAULT 0.0,
    stability_score REAL NOT NULL DEFAULT 0.0,
    promotion_count INTEGER NOT NULL DEFAULT 0,
    champion_count INTEGER NOT NULL DEFAULT 0,
    payload_json TEXT NOT NULL DEFAULT '{}',
    created_at INTEGER NOT NULL,
    UNIQUE(profile_name, group_key, symbol, family_id)
);

CREATE TABLE IF NOT EXISTS distillation_artifacts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    profile_name TEXT NOT NULL,
    symbol TEXT NOT NULL,
    plugin_name TEXT NOT NULL,
    family_id INTEGER NOT NULL DEFAULT 11,
    source_run_id INTEGER,
    best_config_id INTEGER,
    dataset_scope TEXT NOT NULL DEFAULT 'aggregate',
    artifact_path TEXT NOT NULL,
    artifact_sha256 TEXT NOT NULL DEFAULT '',
    teacher_summary_json TEXT NOT NULL DEFAULT '{}',
    student_target_json TEXT NOT NULL DEFAULT '{}',
    status TEXT NOT NULL DEFAULT 'ready',
    created_at INTEGER NOT NULL,
    UNIQUE(profile_name, symbol, plugin_name, dataset_scope)
);

CREATE TABLE IF NOT EXISTS foundation_teacher_artifacts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    profile_name TEXT NOT NULL,
    symbol TEXT NOT NULL,
    scope TEXT NOT NULL DEFAULT 'symbol_family',
    family_id INTEGER NOT NULL DEFAULT 11,
    artifact_path TEXT NOT NULL DEFAULT '',
    artifact_sha256 TEXT NOT NULL DEFAULT '',
    teacher_payload_json TEXT NOT NULL DEFAULT '{}',
    student_profile_json TEXT NOT NULL DEFAULT '{}',
    status TEXT NOT NULL DEFAULT 'ready',
    created_at INTEGER NOT NULL,
    UNIQUE(profile_name, symbol, scope, family_id)
);

CREATE TABLE IF NOT EXISTS redteam_cycles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    profile_name TEXT NOT NULL,
    group_key TEXT NOT NULL DEFAULT '',
    symbol TEXT NOT NULL,
    plugin_name TEXT NOT NULL,
    family_id INTEGER NOT NULL DEFAULT 11,
    weak_scenarios_json TEXT NOT NULL DEFAULT '[]',
    plan_json TEXT NOT NULL DEFAULT '{}',
    report_path TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL DEFAULT 'ready',
    created_at INTEGER NOT NULL,
    UNIQUE(profile_name, group_key, symbol, plugin_name)
);

CREATE TABLE IF NOT EXISTS teacher_factories (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    profile_name TEXT NOT NULL,
    symbol TEXT NOT NULL,
    plugin_name TEXT NOT NULL,
    family_id INTEGER NOT NULL DEFAULT 11,
    champion_best_config_id INTEGER,
    source_run_id INTEGER,
    teacher_artifact_path TEXT NOT NULL DEFAULT '',
    teacher_artifact_sha256 TEXT NOT NULL DEFAULT '',
    student_artifact_path TEXT NOT NULL DEFAULT '',
    student_artifact_sha256 TEXT NOT NULL DEFAULT '',
    deployment_profile_path TEXT NOT NULL DEFAULT '',
    deployment_profile_sha256 TEXT NOT NULL DEFAULT '',
    teacher_score REAL NOT NULL DEFAULT 0.0,
    student_score REAL NOT NULL DEFAULT 0.0,
    live_shadow_score REAL NOT NULL DEFAULT 0.0,
    portfolio_score REAL NOT NULL DEFAULT 0.0,
    policy_score REAL NOT NULL DEFAULT 0.0,
    payload_json TEXT NOT NULL DEFAULT '{}',
    status TEXT NOT NULL DEFAULT 'ready',
    created_at INTEGER NOT NULL,
    UNIQUE(profile_name, symbol, plugin_name)
);

CREATE TABLE IF NOT EXISTS foundation_model_bundles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    profile_name TEXT NOT NULL,
    symbol TEXT NOT NULL,
    bundle_scope TEXT NOT NULL DEFAULT 'symbol',
    artifact_path TEXT NOT NULL DEFAULT '',
    artifact_sha256 TEXT NOT NULL DEFAULT '',
    payload_json TEXT NOT NULL DEFAULT '{}',
    created_at INTEGER NOT NULL,
    UNIQUE(profile_name, symbol, bundle_scope)
);

CREATE TABLE IF NOT EXISTS student_deployment_bundles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    profile_name TEXT NOT NULL,
    symbol TEXT NOT NULL,
    plugin_name TEXT NOT NULL,
    family_id INTEGER NOT NULL DEFAULT 11,
    artifact_path TEXT NOT NULL DEFAULT '',
    artifact_sha256 TEXT NOT NULL DEFAULT '',
    deployment_json TEXT NOT NULL DEFAULT '{}',
    created_at INTEGER NOT NULL,
    UNIQUE(profile_name, symbol, plugin_name)
);

CREATE TABLE IF NOT EXISTS live_deployment_profiles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    profile_name TEXT NOT NULL,
    symbol TEXT NOT NULL,
    deployment_scope TEXT NOT NULL DEFAULT 'symbol',
    artifact_path TEXT NOT NULL,
    artifact_sha256 TEXT NOT NULL DEFAULT '',
    teacher_weight REAL NOT NULL DEFAULT 0.58,
    student_weight REAL NOT NULL DEFAULT 0.42,
    analog_weight REAL NOT NULL DEFAULT 0.18,
    foundation_weight REAL NOT NULL DEFAULT 0.24,
    policy_trade_floor REAL NOT NULL DEFAULT 0.52,
    policy_size_bias REAL NOT NULL DEFAULT 1.0,
    portfolio_budget_bias REAL NOT NULL DEFAULT 1.0,
    challenger_promote_margin REAL NOT NULL DEFAULT 1.0,
    regime_transition_weight REAL NOT NULL DEFAULT 0.35,
    macro_quality_floor REAL NOT NULL DEFAULT 0.24,
    policy_no_trade_cap REAL NOT NULL DEFAULT 0.62,
    capital_efficiency_bias REAL NOT NULL DEFAULT 1.0,
    supervisor_blend REAL NOT NULL DEFAULT 0.45,
    policy_hold_floor REAL NOT NULL DEFAULT 0.48,
    policy_exit_floor REAL NOT NULL DEFAULT 0.58,
    policy_add_floor REAL NOT NULL DEFAULT 0.68,
    policy_reduce_floor REAL NOT NULL DEFAULT 0.56,
    policy_timeout_floor REAL NOT NULL DEFAULT 0.72,
    max_add_fraction REAL NOT NULL DEFAULT 0.50,
    reduce_fraction REAL NOT NULL DEFAULT 0.35,
    soft_timeout_bars INTEGER NOT NULL DEFAULT 8,
    hard_timeout_bars INTEGER NOT NULL DEFAULT 18,
    runtime_mode TEXT NOT NULL DEFAULT 'research',
    telemetry_level TEXT NOT NULL DEFAULT 'full',
    performance_budget_ms REAL NOT NULL DEFAULT 12.0,
    shadow_enabled INTEGER NOT NULL DEFAULT 1,
    snapshot_detail TEXT NOT NULL DEFAULT 'full',
    max_runtime_models INTEGER NOT NULL DEFAULT 12,
    promotion_tier TEXT NOT NULL DEFAULT 'experimental',
    payload_json TEXT NOT NULL DEFAULT '{}',
    created_at INTEGER NOT NULL,
    UNIQUE(profile_name, symbol, deployment_scope)
);

CREATE TABLE IF NOT EXISTS portfolio_supervisor_profiles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    profile_name TEXT NOT NULL,
    artifact_path TEXT NOT NULL DEFAULT '',
    artifact_sha256 TEXT NOT NULL DEFAULT '',
    gross_budget_bias REAL NOT NULL DEFAULT 1.0,
    correlated_budget_bias REAL NOT NULL DEFAULT 1.0,
    directional_budget_bias REAL NOT NULL DEFAULT 1.0,
    capital_risk_cap_pct REAL NOT NULL DEFAULT 1.2,
    macro_overlap_cap REAL NOT NULL DEFAULT 0.92,
    concentration_cap REAL NOT NULL DEFAULT 0.82,
    supervisor_weight REAL NOT NULL DEFAULT 0.45,
    hard_block_score REAL NOT NULL DEFAULT 1.08,
    policy_enter_floor REAL NOT NULL DEFAULT 0.42,
    policy_no_trade_ceiling REAL NOT NULL DEFAULT 0.74,
    payload_json TEXT NOT NULL DEFAULT '{}',
    created_at INTEGER NOT NULL,
    UNIQUE(profile_name)
);

CREATE TABLE IF NOT EXISTS shadow_fleet_observations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    profile_name TEXT NOT NULL,
    symbol TEXT NOT NULL,
    plugin_name TEXT NOT NULL,
    family_id INTEGER NOT NULL DEFAULT 11,
    captured_at INTEGER NOT NULL,
    source_path TEXT NOT NULL,
    source_sha256 TEXT NOT NULL DEFAULT '',
    meta_weight REAL NOT NULL DEFAULT 0.0,
    reliability REAL NOT NULL DEFAULT 0.0,
    global_edge REAL NOT NULL DEFAULT 0.0,
    context_edge REAL NOT NULL DEFAULT 0.0,
    context_regret REAL NOT NULL DEFAULT 0.0,
    portfolio_objective REAL NOT NULL DEFAULT 0.0,
    portfolio_stability REAL NOT NULL DEFAULT 0.0,
    portfolio_corr REAL NOT NULL DEFAULT 0.0,
    portfolio_div REAL NOT NULL DEFAULT 0.0,
    route_value REAL NOT NULL DEFAULT 0.0,
    route_regret REAL NOT NULL DEFAULT 0.0,
    route_counterfactual REAL NOT NULL DEFAULT 0.0,
    shadow_score REAL NOT NULL DEFAULT 0.0,
    regime_id INTEGER NOT NULL DEFAULT 0,
    horizon_minutes INTEGER NOT NULL DEFAULT 5,
    obs_count INTEGER NOT NULL DEFAULT 0,
    policy_enter_prob REAL NOT NULL DEFAULT 0.0,
    policy_no_trade_prob REAL NOT NULL DEFAULT 0.0,
    policy_exit_prob REAL NOT NULL DEFAULT 0.0,
    policy_add_prob REAL NOT NULL DEFAULT 0.0,
    policy_reduce_prob REAL NOT NULL DEFAULT 0.0,
    policy_timeout_prob REAL NOT NULL DEFAULT 0.0,
    policy_tighten_prob REAL NOT NULL DEFAULT 0.0,
    policy_portfolio_fit REAL NOT NULL DEFAULT 0.0,
    policy_capital_efficiency REAL NOT NULL DEFAULT 0.0,
    policy_lifecycle_action INTEGER NOT NULL DEFAULT 0,
    portfolio_pressure REAL NOT NULL DEFAULT 0.0,
    control_plane_score REAL NOT NULL DEFAULT 0.0,
    portfolio_supervisor_score REAL NOT NULL DEFAULT 0.0,
    payload_json TEXT NOT NULL DEFAULT '{}',
    UNIQUE(profile_name, symbol, plugin_name, captured_at, source_sha256)
);

CREATE TABLE IF NOT EXISTS supervisor_service_states (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    profile_name TEXT NOT NULL,
    symbol TEXT NOT NULL,
    artifact_path TEXT NOT NULL DEFAULT '',
    artifact_sha256 TEXT NOT NULL DEFAULT '',
    snapshot_count INTEGER NOT NULL DEFAULT 0,
    gross_pressure REAL NOT NULL DEFAULT 0.0,
    directional_long_pressure REAL NOT NULL DEFAULT 0.0,
    directional_short_pressure REAL NOT NULL DEFAULT 0.0,
    macro_pressure REAL NOT NULL DEFAULT 0.0,
    concentration_pressure REAL NOT NULL DEFAULT 0.0,
    budget_multiplier REAL NOT NULL DEFAULT 1.0,
    add_multiplier REAL NOT NULL DEFAULT 1.0,
    reduce_bias REAL NOT NULL DEFAULT 0.0,
    exit_bias REAL NOT NULL DEFAULT 0.0,
    entry_floor REAL NOT NULL DEFAULT 0.42,
    block_score REAL NOT NULL DEFAULT 1.10,
    payload_json TEXT NOT NULL DEFAULT '{}',
    created_at INTEGER NOT NULL,
    UNIQUE(profile_name, symbol)
);

CREATE TABLE IF NOT EXISTS supervisor_command_profiles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    profile_name TEXT NOT NULL,
    symbol TEXT NOT NULL,
    artifact_path TEXT NOT NULL DEFAULT '',
    artifact_sha256 TEXT NOT NULL DEFAULT '',
    entry_budget_mult REAL NOT NULL DEFAULT 1.0,
    hold_budget_mult REAL NOT NULL DEFAULT 1.0,
    add_cap_mult REAL NOT NULL DEFAULT 1.0,
    reduce_bias REAL NOT NULL DEFAULT 0.0,
    exit_bias REAL NOT NULL DEFAULT 0.0,
    tighten_bias REAL NOT NULL DEFAULT 0.0,
    timeout_bias REAL NOT NULL DEFAULT 0.0,
    long_block INTEGER NOT NULL DEFAULT 0,
    short_block INTEGER NOT NULL DEFAULT 0,
    block_score REAL NOT NULL DEFAULT 1.1,
    payload_json TEXT NOT NULL DEFAULT '{}',
    created_at INTEGER NOT NULL,
    UNIQUE(profile_name, symbol)
);

CREATE TABLE IF NOT EXISTS world_simulator_plans (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    profile_name TEXT NOT NULL,
    symbol TEXT NOT NULL,
    artifact_path TEXT NOT NULL DEFAULT '',
    artifact_sha256 TEXT NOT NULL DEFAULT '',
    sigma_scale REAL NOT NULL DEFAULT 1.0,
    drift_bias REAL NOT NULL DEFAULT 0.0,
    spread_scale REAL NOT NULL DEFAULT 1.0,
    gap_prob REAL NOT NULL DEFAULT 0.0,
    gap_scale REAL NOT NULL DEFAULT 0.0,
    flip_prob REAL NOT NULL DEFAULT 0.0,
    context_corr_bias REAL NOT NULL DEFAULT 0.0,
    liquidity_stress REAL NOT NULL DEFAULT 0.0,
    macro_focus REAL NOT NULL DEFAULT 0.0,
    payload_json TEXT NOT NULL DEFAULT '{}',
    created_at INTEGER NOT NULL,
    UNIQUE(profile_name, symbol)
);

CREATE TABLE IF NOT EXISTS attribution_profiles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    profile_name TEXT NOT NULL,
    symbol TEXT NOT NULL,
    artifact_path TEXT NOT NULL DEFAULT '',
    artifact_sha256 TEXT NOT NULL DEFAULT '',
    champion_only INTEGER NOT NULL DEFAULT 0,
    max_active_models INTEGER NOT NULL DEFAULT 12,
    min_meta_weight REAL NOT NULL DEFAULT 0.0,
    payload_json TEXT NOT NULL DEFAULT '{}',
    created_at INTEGER NOT NULL,
    UNIQUE(profile_name, symbol)
);

CREATE TABLE IF NOT EXISTS student_router_profiles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    profile_name TEXT NOT NULL,
    symbol TEXT NOT NULL,
    artifact_path TEXT NOT NULL DEFAULT '',
    artifact_sha256 TEXT NOT NULL DEFAULT '',
    champion_only INTEGER NOT NULL DEFAULT 0,
    max_active_models INTEGER NOT NULL DEFAULT 12,
    min_meta_weight REAL NOT NULL DEFAULT 0.0,
    payload_json TEXT NOT NULL DEFAULT '{}',
    created_at INTEGER NOT NULL,
    UNIQUE(profile_name, symbol)
);

CREATE TABLE IF NOT EXISTS autonomous_governance_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    profile_name TEXT NOT NULL,
    cycle_group_key TEXT NOT NULL DEFAULT '',
    governance_status TEXT NOT NULL DEFAULT 'ready',
    promoted_count INTEGER NOT NULL DEFAULT 0,
    challenger_count INTEGER NOT NULL DEFAULT 0,
    review_count INTEGER NOT NULL DEFAULT 0,
    rollback_count INTEGER NOT NULL DEFAULT 0,
    artifact_dir TEXT NOT NULL DEFAULT '',
    payload_json TEXT NOT NULL DEFAULT '{}',
    created_at INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_datasets_group ON datasets(group_key, symbol, months);
CREATE INDEX IF NOT EXISTS idx_tuning_runs_lookup ON tuning_runs(profile_name, group_key, symbol, plugin_name, status);
CREATE INDEX IF NOT EXISTS idx_tuning_runs_dataset ON tuning_runs(dataset_id, profile_name, plugin_name);
CREATE INDEX IF NOT EXISTS idx_best_configs_lookup ON best_configs(profile_name, symbol, plugin_name);
CREATE INDEX IF NOT EXISTS idx_control_cycles_lookup ON control_cycles(profile_name, started_at);
CREATE INDEX IF NOT EXISTS idx_lineage_lookup ON config_lineage(profile_name, symbol, plugin_name, created_at);
CREATE INDEX IF NOT EXISTS idx_family_scorecards_lookup ON family_scorecards(profile_name, group_key, symbol, family_id);
CREATE INDEX IF NOT EXISTS idx_champion_lookup ON champion_registry(profile_name, symbol, plugin_name, status);
CREATE INDEX IF NOT EXISTS idx_distill_lookup ON distillation_artifacts(profile_name, symbol, plugin_name, dataset_scope);
CREATE INDEX IF NOT EXISTS idx_foundation_teacher_lookup ON foundation_teacher_artifacts(profile_name, symbol, scope, family_id);
CREATE INDEX IF NOT EXISTS idx_redteam_lookup ON redteam_cycles(profile_name, group_key, symbol, plugin_name);
CREATE INDEX IF NOT EXISTS idx_teacher_factories_lookup ON teacher_factories(profile_name, symbol, plugin_name, status);
CREATE INDEX IF NOT EXISTS idx_foundation_bundle_lookup ON foundation_model_bundles(profile_name, symbol, bundle_scope, created_at);
CREATE INDEX IF NOT EXISTS idx_student_bundle_lookup ON student_deployment_bundles(profile_name, symbol, plugin_name, created_at);
CREATE INDEX IF NOT EXISTS idx_live_deploy_lookup ON live_deployment_profiles(profile_name, symbol, deployment_scope);
CREATE INDEX IF NOT EXISTS idx_portfolio_supervisor_lookup ON portfolio_supervisor_profiles(profile_name, created_at);
CREATE INDEX IF NOT EXISTS idx_shadow_fleet_lookup ON shadow_fleet_observations(profile_name, symbol, plugin_name, captured_at);
CREATE INDEX IF NOT EXISTS idx_supervisor_service_lookup ON supervisor_service_states(profile_name, symbol, created_at);
CREATE INDEX IF NOT EXISTS idx_supervisor_command_lookup ON supervisor_command_profiles(profile_name, symbol, created_at);
CREATE INDEX IF NOT EXISTS idx_world_sim_lookup ON world_simulator_plans(profile_name, symbol, created_at);
CREATE INDEX IF NOT EXISTS idx_attribution_lookup ON attribution_profiles(profile_name, symbol, created_at);
CREATE INDEX IF NOT EXISTS idx_student_router_lookup ON student_router_profiles(profile_name, symbol, created_at);
CREATE INDEX IF NOT EXISTS idx_governance_runs_lookup ON autonomous_governance_runs(profile_name, created_at);
"""


class OfflineLabError(RuntimeError):
    pass


def now_unix() -> int:
    return int(time.time())


def safe_token(raw: str) -> str:
    text = (raw or "").strip()
    if not text:
        return "default"
    for ch in "\\/:*?\"<>|{}[](),;= \t\r\n":
        text = text.replace(ch, "_")
    while "__" in text:
        text = text.replace("__", "_")
    return text.strip("_") or "default"


def json_compact(payload) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def months_back(anchor: datetime, months: int) -> datetime:
    if months <= 0:
        return anchor
    month_index = anchor.month - 1 - months
    year = anchor.year + month_index // 12
    month = month_index % 12 + 1
    day = anchor.day
    while day > 28:
        try:
            return anchor.replace(year=year, month=month, day=day)
        except ValueError:
            day -= 1
    return anchor.replace(year=year, month=month, day=day)


def resolve_window(months: int, start_unix: int, end_unix: int) -> tuple[int, int]:
    if start_unix > 0 and end_unix > start_unix:
        return int(start_unix), int(end_unix)
    end_dt = datetime.fromtimestamp((end_unix if end_unix > 0 else now_unix()), tz=timezone.utc)
    start_dt = months_back(end_dt, max(months, 1))
    return int(start_dt.timestamp()), int(end_dt.timestamp())


def parse_csv_tokens(raw: str) -> list[str]:
    text = (raw or "").strip()
    if not text:
        return []
    text = text.replace("{", "").replace("}", "").replace(";", ",").replace("|", ",")
    out: list[str] = []
    for part in text.split(","):
        token = part.strip()
        if token and token not in out:
            out.append(token)
    return out


def resolve_symbols(args) -> list[str]:
    pack_name = (getattr(args, "symbol_pack", "") or "").strip().lower()
    if pack_name:
        return list(testlab.SYMBOL_PACKS.get(pack_name, [str(getattr(args, "symbol", "EURUSD")).upper()]))
    symbols = parse_csv_tokens(getattr(args, "symbol_list", ""))
    if not symbols:
        symbol = str(getattr(args, "symbol", "EURUSD") or "").strip()
        if symbol:
            symbols = [symbol]
    return [s.upper() for s in symbols]


def resolve_months_list(raw: str) -> list[int]:
    items = parse_csv_tokens(raw)
    out: list[int] = []
    for item in items:
        try:
            months = int(item)
        except Exception:
            continue
        if months > 0 and months not in out:
            out.append(months)
    return out or list(DEFAULT_MONTHS_LIST)


def ensure_table_column(conn: LabConnection, table: str, column: str, spec: str) -> None:
    columns = {str(row["name"]).lower() for row in conn.execute(f"PRAGMA table_info({table})")}
    if column.lower() not in columns:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {spec}")


def set_metadata(conn: LabConnection, key: str, value: str) -> None:
    conn.execute(
        """
        INSERT INTO lab_metadata(meta_key, meta_value, updated_at)
        VALUES(?, ?, ?)
        ON CONFLICT(meta_key) DO UPDATE SET
            meta_value=excluded.meta_value,
            updated_at=excluded.updated_at
        """,
        (str(key), str(value), now_unix()),
    )


def get_metadata(conn: LabConnection, key: str, default: str = "") -> str:
    row = conn.execute(
        "SELECT meta_value FROM lab_metadata WHERE meta_key = ?",
        (str(key),),
    ).fetchone()
    if row is None:
        return str(default)
    try:
        return str(row["meta_value"])
    except Exception:
        return str(default)


def current_lab_versions(conn: LabConnection) -> dict[str, str]:
    return {
        "offline_schema_version": get_metadata(conn, "offline_schema_version", str(OFFLINE_SCHEMA_VERSION)),
        "artifact_schema_version": get_metadata(conn, "artifact_schema_version", str(OFFLINE_ARTIFACT_SCHEMA_VERSION)),
        "macro_schema_min": get_metadata(conn, "macro_schema_min", str(OFFLINE_MACRO_SCHEMA_MIN)),
        "db_backend": get_metadata(conn, "db_backend", "turso_local_libsql"),
        "turso_sync_mode": get_metadata(conn, "turso_sync_mode", "local_only"),
    }


def resolve_turso_config(db_path: Path) -> TursoConfig:
    sync_url = (os.getenv(TURSO_DATABASE_URL_ENV, "") or "").strip()
    auth_token = (os.getenv(TURSO_AUTH_TOKEN_ENV, "") or "").strip()
    return TursoConfig(database=Path(db_path), sync_url=sync_url, auth_token=auth_token)


def turso_environment_status(db_path: Path = DEFAULT_DB) -> dict[str, object]:
    config = resolve_turso_config(Path(db_path))
    return {
        "backend": config.backend_name,
        "database_path": str(config.database),
        "sync_enabled": config.sync_enabled,
        "sync_mode": config.sync_mode,
        "sync_url_configured": bool(config.sync_url),
        "auth_token_configured": bool(config.auth_token),
        "config_error": ("partial_sync_credentials" if config.partial_sync_config else ""),
    }


def _is_retryable_db_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return "locked" in text or "busy" in text


def connect_db(db_path: Path) -> LabConnection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    config = resolve_turso_config(db_path)
    try:
        config.validate()
    except ValueError as exc:
        raise OfflineLabError(str(exc)) from exc
    last_error: Exception | None = None
    for attempt in range(6):
        conn: LabConnection | None = None
        try:
            conn = connect_backend(config, timeout=30.0)
            conn.execute("PRAGMA busy_timeout=30000")
            if conn.sync_enabled:
                conn.sync()
            conn.executescript(SQL_SCHEMA)
            ensure_table_column(conn, "tuning_runs", "group_key", "TEXT NOT NULL DEFAULT ''")
            ensure_table_column(conn, "tuning_runs", "family_id", "INTEGER NOT NULL DEFAULT 11")
            ensure_table_column(conn, "best_configs", "family_id", "INTEGER NOT NULL DEFAULT 11")
            ensure_table_column(conn, "champion_registry", "family_id", "INTEGER NOT NULL DEFAULT 11")
            ensure_table_column(conn, "champion_registry", "promotion_tier", "TEXT NOT NULL DEFAULT 'experimental'")
            ensure_table_column(conn, "live_deployment_profiles", "policy_no_trade_cap", "REAL NOT NULL DEFAULT 0.62")
            ensure_table_column(conn, "live_deployment_profiles", "capital_efficiency_bias", "REAL NOT NULL DEFAULT 1.0")
            ensure_table_column(conn, "live_deployment_profiles", "supervisor_blend", "REAL NOT NULL DEFAULT 0.45")
            ensure_table_column(conn, "live_deployment_profiles", "policy_hold_floor", "REAL NOT NULL DEFAULT 0.48")
            ensure_table_column(conn, "live_deployment_profiles", "policy_exit_floor", "REAL NOT NULL DEFAULT 0.58")
            ensure_table_column(conn, "live_deployment_profiles", "policy_add_floor", "REAL NOT NULL DEFAULT 0.68")
            ensure_table_column(conn, "live_deployment_profiles", "policy_reduce_floor", "REAL NOT NULL DEFAULT 0.56")
            ensure_table_column(conn, "live_deployment_profiles", "policy_timeout_floor", "REAL NOT NULL DEFAULT 0.72")
            ensure_table_column(conn, "live_deployment_profiles", "max_add_fraction", "REAL NOT NULL DEFAULT 0.50")
            ensure_table_column(conn, "live_deployment_profiles", "reduce_fraction", "REAL NOT NULL DEFAULT 0.35")
            ensure_table_column(conn, "live_deployment_profiles", "soft_timeout_bars", "INTEGER NOT NULL DEFAULT 8")
            ensure_table_column(conn, "live_deployment_profiles", "hard_timeout_bars", "INTEGER NOT NULL DEFAULT 18")
            ensure_table_column(conn, "live_deployment_profiles", "runtime_mode", "TEXT NOT NULL DEFAULT 'research'")
            ensure_table_column(conn, "live_deployment_profiles", "telemetry_level", "TEXT NOT NULL DEFAULT 'full'")
            ensure_table_column(conn, "live_deployment_profiles", "performance_budget_ms", "REAL NOT NULL DEFAULT 12.0")
            ensure_table_column(conn, "live_deployment_profiles", "shadow_enabled", "INTEGER NOT NULL DEFAULT 1")
            ensure_table_column(conn, "live_deployment_profiles", "snapshot_detail", "TEXT NOT NULL DEFAULT 'full'")
            ensure_table_column(conn, "live_deployment_profiles", "max_runtime_models", "INTEGER NOT NULL DEFAULT 12")
            ensure_table_column(conn, "live_deployment_profiles", "promotion_tier", "TEXT NOT NULL DEFAULT 'experimental'")
            ensure_table_column(conn, "shadow_fleet_observations", "policy_enter_prob", "REAL NOT NULL DEFAULT 0.0")
            ensure_table_column(conn, "shadow_fleet_observations", "policy_no_trade_prob", "REAL NOT NULL DEFAULT 0.0")
            ensure_table_column(conn, "shadow_fleet_observations", "policy_exit_prob", "REAL NOT NULL DEFAULT 0.0")
            ensure_table_column(conn, "shadow_fleet_observations", "policy_add_prob", "REAL NOT NULL DEFAULT 0.0")
            ensure_table_column(conn, "shadow_fleet_observations", "policy_reduce_prob", "REAL NOT NULL DEFAULT 0.0")
            ensure_table_column(conn, "shadow_fleet_observations", "policy_timeout_prob", "REAL NOT NULL DEFAULT 0.0")
            ensure_table_column(conn, "shadow_fleet_observations", "policy_tighten_prob", "REAL NOT NULL DEFAULT 0.0")
            ensure_table_column(conn, "shadow_fleet_observations", "policy_portfolio_fit", "REAL NOT NULL DEFAULT 0.0")
            ensure_table_column(conn, "shadow_fleet_observations", "policy_capital_efficiency", "REAL NOT NULL DEFAULT 0.0")
            ensure_table_column(conn, "shadow_fleet_observations", "policy_lifecycle_action", "INTEGER NOT NULL DEFAULT 0")
            ensure_table_column(conn, "shadow_fleet_observations", "portfolio_pressure", "REAL NOT NULL DEFAULT 0.0")
            ensure_table_column(conn, "shadow_fleet_observations", "control_plane_score", "REAL NOT NULL DEFAULT 0.0")
            ensure_table_column(conn, "shadow_fleet_observations", "portfolio_supervisor_score", "REAL NOT NULL DEFAULT 0.0")
            conn.execute("DROP INDEX IF EXISTS idx_tuning_runs_lookup")
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_tuning_runs_lookup "
                "ON tuning_runs(profile_name, group_key, symbol, plugin_name, status)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_tuning_runs_family "
                "ON tuning_runs(profile_name, family_id, symbol, plugin_name, status)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_best_configs_family "
                "ON best_configs(profile_name, family_id, symbol)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_shadow_fleet_symbol "
                "ON shadow_fleet_observations(profile_name, symbol, captured_at)"
            )
            conn.execute(
                """
                UPDATE tuning_runs
                   SET group_key = COALESCE((
                       SELECT d.group_key
                         FROM datasets d
                        WHERE d.id = tuning_runs.dataset_id
                   ), '')
                 WHERE COALESCE(group_key, '') = ''
                """
            )
            set_metadata(conn, "offline_schema_version", str(OFFLINE_SCHEMA_VERSION))
            set_metadata(conn, "artifact_schema_version", str(OFFLINE_ARTIFACT_SCHEMA_VERSION))
            set_metadata(conn, "macro_schema_min", str(OFFLINE_MACRO_SCHEMA_MIN))
            set_metadata(conn, "db_backend", str(config.backend_name))
            set_metadata(conn, "turso_sync_mode", str(config.sync_mode))
            conn.commit()
            return conn
        except Exception as exc:
            last_error = exc
            if conn is not None:
                conn.close()
            if not _is_retryable_db_error(exc) or attempt >= 5:
                raise
            time.sleep(0.25 * float(attempt + 1))
    if last_error is not None:
        raise last_error
    raise OfflineLabError(f"failed to open Turso libSQL lab: {db_path}")


def plugin_family_name(family_id: int) -> str:
    mapping = {
        0: "linear",
        1: "tree",
        2: "recurrent",
        3: "convolutional",
        4: "transformer",
        5: "state_space",
        6: "distribution",
        7: "mixture",
        8: "memory",
        9: "world",
        10: "rule",
        11: "other",
    }
    return mapping.get(int(family_id), "other")


def mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    mean_v = sum(values) / float(len(values))
    if len(values) <= 1:
        return mean_v, 0.0
    var = sum((v - mean_v) * (v - mean_v) for v in values) / float(len(values))
    return mean_v, math.sqrt(max(var, 0.0))


def row_float(row: Mapping[str, object] | None, key: str, default: float = 0.0) -> float:
    if row is None:
        return float(default)
    try:
        raw = row.get(key, default) if hasattr(row, "get") else row[key]
    except Exception:
        raw = default
    try:
        return float(raw)
    except Exception:
        return float(default)


def solve_linear_system(matrix: list[list[float]], vector: list[float]) -> list[float] | None:
    n = len(matrix)
    if n <= 0 or n != len(vector):
        return None
    a = [[float(matrix[r][c]) for c in range(n)] for r in range(n)]
    b = [float(vector[r]) for r in range(n)]
    for pivot in range(n):
        best_row = pivot
        best_abs = abs(a[pivot][pivot])
        for row in range(pivot + 1, n):
            cand = abs(a[row][pivot])
            if cand > best_abs:
                best_abs = cand
                best_row = row
        if best_abs <= 1e-12:
            return None
        if best_row != pivot:
            a[pivot], a[best_row] = a[best_row], a[pivot]
            b[pivot], b[best_row] = b[best_row], b[pivot]
        pivot_val = a[pivot][pivot]
        inv_pivot = 1.0 / pivot_val
        for col in range(pivot, n):
            a[pivot][col] *= inv_pivot
        b[pivot] *= inv_pivot
        for row in range(n):
            if row == pivot:
                continue
            factor = a[row][pivot]
            if abs(factor) <= 1e-12:
                continue
            for col in range(pivot, n):
                a[row][col] -= factor * a[pivot][col]
            b[row] -= factor * b[pivot]
    return b


def fit_weighted_linear_model(rows: list[Mapping[str, object]],
                              feature_names: list[str],
                              target_name: str,
                              weight_name: str | None = None,
                              ridge: float = 1e-6) -> dict:
    dim = len(feature_names) + 1
    ata = [[0.0 for _ in range(dim)] for _ in range(dim)]
    atb = [0.0 for _ in range(dim)]
    used = 0
    total_weight = 0.0
    samples: list[tuple[list[float], float, float]] = []

    for row in rows:
        target = row_float(row, target_name, math.nan)
        if not math.isfinite(target):
            continue
        weight = row_float(row, weight_name, 1.0) if weight_name else 1.0
        weight = max(weight, 1e-9)
        vec = [1.0]
        valid = True
        for name in feature_names:
            value = row_float(row, name, math.nan)
            if not math.isfinite(value):
                valid = False
                break
            vec.append(value)
        if not valid:
            continue
        samples.append((vec, target, weight))
        used += 1
        total_weight += weight
        for i in range(dim):
            atb[i] += weight * vec[i] * target
            for j in range(i, dim):
                ata[i][j] += weight * vec[i] * vec[j]

    if used <= 0:
        return {
            "feature_names": list(feature_names),
            "intercept": 0.0,
            "coefficients": {name: 0.0 for name in feature_names},
            "mae": 0.0,
            "rmse": 0.0,
            "r2": 0.0,
            "used_rows": 0,
            "total_weight": 0.0,
        }

    for i in range(dim):
        ata[i][i] += float(ridge)
        for j in range(i):
            ata[i][j] = ata[j][i]

    coeff_vec = solve_linear_system(ata, atb)
    if coeff_vec is None:
        coeff_vec = [0.0 for _ in range(dim)]

    predictions: list[tuple[float, float, float]] = []
    weighted_target_sum = 0.0
    for _vec, target, weight in samples:
        weighted_target_sum += weight * target
    target_mean = weighted_target_sum / max(total_weight, 1e-9)
    for vec, target, weight in samples:
        pred = float(sum(coeff_vec[idx] * vec[idx] for idx in range(dim)))
        predictions.append((target, pred, weight))

    abs_err = 0.0
    sq_err = 0.0
    total_var = 0.0
    for target, pred, weight in predictions:
        diff = pred - target
        abs_err += weight * abs(diff)
        sq_err += weight * diff * diff
        centered = target - target_mean
        total_var += weight * centered * centered

    mae = abs_err / max(total_weight, 1e-9)
    rmse = math.sqrt(max(sq_err / max(total_weight, 1e-9), 0.0))
    r2 = 0.0
    if total_var > 1e-12:
        r2 = 1.0 - sq_err / total_var

    return {
        "feature_names": list(feature_names),
        "intercept": float(coeff_vec[0]),
        "coefficients": {
            feature_names[idx]: float(coeff_vec[idx + 1])
            for idx in range(len(feature_names))
        },
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2),
        "used_rows": int(used),
        "total_weight": float(total_weight),
    }


def predict_linear_model(model: dict, feature_values: dict[str, float], default: float = 0.0) -> float:
    if not isinstance(model, dict):
        return float(default)
    total = row_float(model, "intercept", default)
    coeffs = model.get("coefficients", {})
    if not isinstance(coeffs, dict):
        return float(total)
    for key, coeff in coeffs.items():
        try:
            value = float(feature_values.get(str(key), 0.0))
            total += float(coeff) * value
        except Exception:
            continue
    return float(total)


def param_identity_hash(row: dict) -> str:
    profile = str(row.get("profile_name", ""))
    symbol = str(row.get("symbol", ""))
    plugin = str(row.get("plugin_name", ""))
    params_json = str(row.get("parameters_json", "{}"))
    return sha256_text(f"{profile}|{symbol}|{plugin}|{params_json}")


def family_distillation_profile(family_id: int) -> dict:
    fam = int(family_id)
    if fam in (2, 3, 4, 5):
        return {
            "temperature": 1.35,
            "teacher_weight": 0.70,
            "student_weight": 0.30,
            "self_supervised_weight": 0.28,
            "analog_weight": 0.18,
            "foundation_weight": 0.32,
        }
    if fam in (0, 1, 6):
        return {
            "temperature": 1.15,
            "teacher_weight": 0.62,
            "student_weight": 0.38,
            "self_supervised_weight": 0.16,
            "analog_weight": 0.12,
            "foundation_weight": 0.18,
        }
    if fam in (7, 8, 9):
        return {
            "temperature": 1.28,
            "teacher_weight": 0.66,
            "student_weight": 0.34,
            "self_supervised_weight": 0.22,
            "analog_weight": 0.24,
            "foundation_weight": 0.24,
        }
    return {
        "temperature": 1.10,
        "teacher_weight": 0.58,
        "student_weight": 0.42,
        "self_supervised_weight": 0.10,
        "analog_weight": 0.08,
        "foundation_weight": 0.14,
    }


def row_to_dict(row: Mapping[str, object] | None) -> dict | None:
    return dict(row) if row is not None else None


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def dataset_data_path(dataset_key: str, symbol: str) -> Path:
    return COMMON_EXPORT_DIR / f"fxai_export_{safe_token(dataset_key)}_{safe_token(symbol)}.tsv"


def dataset_meta_path(dataset_key: str, symbol: str) -> Path:
    return COMMON_EXPORT_DIR / f"fxai_export_{safe_token(dataset_key)}_{safe_token(symbol)}.meta.tsv"


def load_kv_tsv(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not path.exists():
        return out
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            key = (row.get("key", "") or "").strip()
            value = (row.get("value", "") or "").strip()
            if key:
                out[key] = value
    return out
