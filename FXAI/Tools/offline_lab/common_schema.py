#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import fxai_testlab as testlab

OFFLINE_DIR = Path(__file__).resolve().parent.parent / "OfflineLab"
DEFAULT_DB = OFFLINE_DIR / "fxai_offline_lab.turso.db"
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
OFFLINE_SCHEMA_VERSION = 4
OFFLINE_ARTIFACT_SCHEMA_VERSION = 2
OFFLINE_MACRO_SCHEMA_MIN = 2
RESEARCH_VECTOR_DIMS = 16

SQL_SCHEMA = """
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

CREATE TABLE IF NOT EXISTS turso_branch_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    profile_name TEXT NOT NULL DEFAULT '',
    source_database TEXT NOT NULL,
    target_database TEXT NOT NULL,
    branch_kind TEXT NOT NULL DEFAULT 'campaign',
    source_timestamp TEXT NOT NULL DEFAULT '',
    group_name TEXT NOT NULL DEFAULT '',
    location_name TEXT NOT NULL DEFAULT '',
    sync_url TEXT NOT NULL DEFAULT '',
    auth_token_sha256 TEXT NOT NULL DEFAULT '',
    env_artifact_path TEXT NOT NULL DEFAULT '',
    payload_json TEXT NOT NULL DEFAULT '{}',
    status TEXT NOT NULL DEFAULT 'created',
    created_at INTEGER NOT NULL,
    UNIQUE(target_database)
);

CREATE TABLE IF NOT EXISTS turso_audit_log_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    organization_slug TEXT NOT NULL,
    event_id TEXT NOT NULL,
    event_type TEXT NOT NULL DEFAULT '',
    actor_name TEXT NOT NULL DEFAULT '',
    actor_email TEXT NOT NULL DEFAULT '',
    target_type TEXT NOT NULL DEFAULT '',
    target_name TEXT NOT NULL DEFAULT '',
    occurred_at TEXT NOT NULL DEFAULT '',
    source_page INTEGER NOT NULL DEFAULT 1,
    payload_json TEXT NOT NULL DEFAULT '{}',
    observed_at INTEGER NOT NULL,
    UNIQUE(organization_slug, event_id)
);

CREATE TABLE IF NOT EXISTS research_vectors (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    profile_name TEXT NOT NULL,
    symbol TEXT NOT NULL,
    vector_scope TEXT NOT NULL DEFAULT 'analog_shadow',
    source_type TEXT NOT NULL DEFAULT 'shadow_observation',
    source_key TEXT NOT NULL,
    dims INTEGER NOT NULL DEFAULT 16,
    vector_blob F32_BLOB(16) NOT NULL,
    score REAL NOT NULL DEFAULT 0.0,
    payload_json TEXT NOT NULL DEFAULT '{}',
    created_at INTEGER NOT NULL,
    UNIQUE(profile_name, symbol, vector_scope, source_type, source_key)
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
CREATE INDEX IF NOT EXISTS idx_turso_branch_lookup ON turso_branch_runs(profile_name, branch_kind, created_at);
CREATE INDEX IF NOT EXISTS idx_turso_audit_lookup ON turso_audit_log_events(organization_slug, occurred_at);
CREATE INDEX IF NOT EXISTS idx_research_vectors_lookup ON research_vectors(profile_name, symbol, vector_scope, source_type, created_at);
CREATE INDEX IF NOT EXISTS idx_research_vectors_ann ON research_vectors(libsql_vector_idx(vector_blob));
"""


