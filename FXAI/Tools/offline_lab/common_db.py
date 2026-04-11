#!/usr/bin/env python3
from __future__ import annotations

import os
import time
from collections.abc import Sequence
from pathlib import Path

import libsql

from .common_schema import DEFAULT_DB, OFFLINE_ARTIFACT_SCHEMA_VERSION, OFFLINE_MACRO_SCHEMA_MIN, OFFLINE_SCHEMA_VERSION, SQL_SCHEMA
from .common_utils import OfflineLabError, now_unix
from .db_backend import (
    TURSO_API_TOKEN_ENV,
    TURSO_AUTH_TOKEN_ENV,
    TURSO_CONFIG_PATH_ENV,
    TURSO_DATABASE_NAME_ENV,
    TURSO_DATABASE_URL_ENV,
    TURSO_ENCRYPTION_KEY_ENV,
    TURSO_GROUP_ENV,
    TURSO_LOCATION_ENV,
    TURSO_ORGANIZATION_ENV,
    TURSO_SYNC_INTERVAL_ENV,
    TursoConfig,
    close_backend,
    commit_backend,
    connect_backend,
    sync_backend,
)
from .market_universe import seed_market_universe_config

def _cursor_column_names(cursor: libsql.Cursor) -> list[str]:
    description = cursor.description or []
    return [str(item[0] or "") for item in description]


def _row_to_mapping(column_names: Sequence[str], row: Sequence[object]) -> dict[str, object]:
    return {
        str(column_names[idx]): row[idx]
        for idx in range(min(len(column_names), len(row)))
    }


def fetch_all_dicts(cursor: libsql.Cursor) -> list[dict[str, object]]:
    column_names = _cursor_column_names(cursor)
    return [_row_to_mapping(column_names, row) for row in cursor.fetchall()]


def fetch_one_dict(cursor: libsql.Cursor) -> dict[str, object] | None:
    column_names = _cursor_column_names(cursor)
    row = cursor.fetchone()
    if row is None:
        return None
    return _row_to_mapping(column_names, row)


def query_all(conn: libsql.Connection,
              sql: str,
              params: Sequence[object] | None = None) -> list[dict[str, object]]:
    cursor = conn.execute(sql, tuple(params or ()))
    return fetch_all_dicts(cursor)


def query_one(conn: libsql.Connection,
              sql: str,
              params: Sequence[object] | None = None) -> dict[str, object] | None:
    cursor = conn.execute(sql, tuple(params or ()))
    return fetch_one_dict(cursor)


def query_scalar(conn: libsql.Connection,
                 sql: str,
                 params: Sequence[object] | None = None,
                 default=None):
    row = conn.execute(sql, tuple(params or ())).fetchone()
    if row is None or len(row) <= 0:
        return default
    return row[0]


def commit_db(conn: libsql.Connection) -> None:
    commit_backend(conn)


def close_db(conn: libsql.Connection) -> None:
    close_backend(conn)


def ensure_table_column(conn: libsql.Connection, table: str, column: str, spec: str) -> None:
    columns = {str(row["name"]).lower() for row in query_all(conn, f"PRAGMA table_info({table})")}
    if column.lower() not in columns:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {spec}")


def set_metadata(conn: libsql.Connection, key: str, value: str) -> None:
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


def get_metadata(conn: libsql.Connection, key: str, default: str = "") -> str:
    row = query_one(
        conn,
        "SELECT meta_value FROM lab_metadata WHERE meta_key = ?",
        (str(key),),
    )
    if row is None:
        return str(default)
    return str(row.get("meta_value", default))


def current_lab_versions(conn: libsql.Connection) -> dict[str, str]:
    return {
        "offline_schema_version": get_metadata(conn, "offline_schema_version", str(OFFLINE_SCHEMA_VERSION)),
        "artifact_schema_version": get_metadata(conn, "artifact_schema_version", str(OFFLINE_ARTIFACT_SCHEMA_VERSION)),
        "macro_schema_min": get_metadata(conn, "macro_schema_min", str(OFFLINE_MACRO_SCHEMA_MIN)),
        "db_backend": get_metadata(conn, "db_backend", "turso_local"),
        "turso_sync_mode": get_metadata(conn, "turso_sync_mode", "local_only"),
        "turso_encryption": get_metadata(conn, "turso_encryption", "disabled"),
        "turso_database_name": get_metadata(conn, "turso_database_name", ""),
        "turso_organization": get_metadata(conn, "turso_organization", ""),
        "turso_sync_interval_seconds": get_metadata(conn, "turso_sync_interval_seconds", "0"),
    }


def resolve_turso_config(db_path: Path) -> TursoConfig:
    sync_url = (os.getenv(TURSO_DATABASE_URL_ENV, "") or "").strip()
    auth_token = (os.getenv(TURSO_AUTH_TOKEN_ENV, "") or "").strip()
    encryption_key = (os.getenv(TURSO_ENCRYPTION_KEY_ENV, "") or "").strip()
    database_name = (os.getenv(TURSO_DATABASE_NAME_ENV, "") or "").strip()
    organization_slug = (os.getenv(TURSO_ORGANIZATION_ENV, "") or "").strip()
    api_token = (os.getenv(TURSO_API_TOKEN_ENV, "") or "").strip()
    group_name = (os.getenv(TURSO_GROUP_ENV, "") or "").strip()
    location_name = (os.getenv(TURSO_LOCATION_ENV, "") or "").strip()
    cli_config_path = (os.getenv(TURSO_CONFIG_PATH_ENV, "") or "").strip()
    sync_interval_raw = (os.getenv(TURSO_SYNC_INTERVAL_ENV, "") or "").strip()
    sync_interval_seconds = 0.0
    if sync_interval_raw:
        try:
            sync_interval_seconds = float(sync_interval_raw)
        except ValueError as exc:
            raise OfflineLabError(
                f"{TURSO_SYNC_INTERVAL_ENV} must be numeric"
            ) from exc
    return TursoConfig(
        database=Path(db_path),
        sync_url=sync_url,
        auth_token=auth_token,
        encryption_key=encryption_key,
        sync_interval_seconds=sync_interval_seconds,
        database_name=database_name,
        organization_slug=organization_slug,
        api_token=api_token,
        group_name=group_name,
        location_name=location_name,
        cli_config_path=cli_config_path,
    )


def turso_environment_status(db_path: Path = DEFAULT_DB) -> dict[str, object]:
    config = resolve_turso_config(Path(db_path))
    return {
        "backend": config.backend_name,
        "database_path": str(config.database),
        "sync_enabled": config.sync_enabled,
        "sync_mode": config.sync_mode,
        "sync_url_configured": bool(config.sync_url),
        "auth_token_configured": bool(config.auth_token),
        "encryption_enabled": config.encryption_enabled,
        "sync_interval_seconds": config.sync_interval_seconds,
        "database_name": config.database_name,
        "organization_slug": config.organization_slug,
        "api_token_configured": bool(config.api_token),
        "group_name": config.group_name,
        "location_name": config.location_name,
        "cli_config_path": config.cli_config_path,
        "platform_api_enabled": config.platform_api_enabled,
        "config_error": (
            "partial_sync_credentials"
            if config.partial_sync_config
            else "partial_platform_api_credentials"
            if config.partial_platform_api_config
            else ""
        ),
    }


def _is_retryable_db_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return "locked" in text or "busy" in text


def connect_db(db_path: Path) -> libsql.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    config = resolve_turso_config(db_path)
    try:
        config.validate()
    except ValueError as exc:
        raise OfflineLabError(str(exc)) from exc
    last_error: Exception | None = None
    for attempt in range(6):
        conn: libsql.Connection | None = None
        try:
            conn = connect_backend(config, timeout=30.0)
            sync_backend(conn)
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
                "CREATE INDEX IF NOT EXISTS idx_label_engine_lookup "
                "ON label_engine_artifacts(profile_name, dataset_id, created_at)"
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
            set_metadata(conn, "turso_encryption", "enabled" if config.encryption_enabled else "disabled")
            set_metadata(conn, "turso_database_name", str(config.database_name))
            set_metadata(conn, "turso_organization", str(config.organization_slug))
            set_metadata(conn, "turso_sync_interval_seconds", str(config.sync_interval_seconds))
            seed_market_universe_config(conn)
            commit_db(conn)
            return conn
        except Exception as exc:
            last_error = exc
            if conn is not None:
                close_db(conn)
            if not _is_retryable_db_error(exc) or attempt >= 5:
                raise
            time.sleep(0.25 * float(attempt + 1))
    if last_error is not None:
        raise last_error
    raise OfflineLabError(f"failed to open Turso lab: {db_path}")
