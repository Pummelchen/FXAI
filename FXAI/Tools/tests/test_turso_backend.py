from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import Mock

from offline_lab.db_backend import (
    TURSO_ENCRYPTION_KEY_ENV,
    TURSO_SYNC_INTERVAL_ENV,
    TursoConfig,
    commit_backend,
    connect_backend,
    register_connection_config,
)
from offline_lab.common import OfflineLabError, close_db, commit_db, connect_db, query_all, query_one
from offline_lab.environment import bootstrap_environment, validate_environment
from offline_lab.exporter import insert_dataset_bars
from offline_lab.fixtures import patched_paths
from offline_lab.vector_store import latest_symbol_shadow_neighbors, refresh_research_vectors


def test_turso_queries_use_explicit_mapping_helpers():
    with tempfile.TemporaryDirectory(prefix="fxai_turso_") as tmp_dir:
        with patched_paths(Path(tmp_dir)) as paths:
            bootstrap_environment(paths["default_db"], init_db=True)
            conn = connect_db(paths["default_db"])
            conn.execute(
                """
                INSERT INTO lab_metadata(meta_key, meta_value, updated_at)
                VALUES('row_test', 'ok', 1700000000)
                ON CONFLICT(meta_key) DO UPDATE SET
                    meta_value=excluded.meta_value,
                    updated_at=excluded.updated_at
                """
            )
            commit_backend(conn)
            row = query_one(
                conn,
                "SELECT meta_key, meta_value FROM lab_metadata WHERE meta_key = ?",
                ("row_test",),
            )
            close_db(conn)

            assert row is not None
            assert row["meta_key"] == "row_test"
            assert row["meta_value"] == "ok"


def test_world_simulator_plan_schema_uses_fill_risk_scale_and_migrates_legacy_spread_scale():
    with tempfile.TemporaryDirectory(prefix="fxai_world_schema_") as tmp_dir:
        with patched_paths(Path(tmp_dir)) as paths:
            conn = connect_db(paths["default_db"])
            fresh_columns = {str(row["name"]) for row in query_all(conn, "PRAGMA table_info(world_simulator_plans)")}
            assert "fill_risk_scale" in fresh_columns
            assert "spread_scale" not in fresh_columns

            conn.execute("DROP TABLE world_simulator_plans")
            conn.execute(
                """
                CREATE TABLE world_simulator_plans (
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
                )
                """
            )
            conn.execute(
                """
                INSERT INTO world_simulator_plans(profile_name, symbol, spread_scale, created_at)
                VALUES('legacy', 'EURUSD', 2.4, 1700000000)
                """
            )
            commit_db(conn)
            close_db(conn)

            conn = connect_db(paths["default_db"])
            migrated = query_one(
                conn,
                "SELECT fill_risk_scale, spread_scale FROM world_simulator_plans WHERE profile_name = 'legacy'",
            )
            close_db(conn)

            assert migrated is not None
            assert float(migrated["fill_risk_scale"]) == 2.4
            assert float(migrated["spread_scale"]) == 2.4


def test_dataset_bars_schema_uses_price_cost_points_and_migrates_legacy_spread_points():
    with tempfile.TemporaryDirectory(prefix="fxai_dataset_schema_") as tmp_dir:
        with patched_paths(Path(tmp_dir)) as paths:
            conn = connect_db(paths["default_db"])
            fresh_columns = {str(row["name"]) for row in query_all(conn, "PRAGMA table_info(dataset_bars)")}
            assert "price_cost_points" in fresh_columns
            assert "spread_points" not in fresh_columns

            conn.execute("DROP TABLE dataset_bars")
            conn.execute(
                """
                CREATE TABLE dataset_bars (
                    dataset_id INTEGER NOT NULL,
                    bar_time_unix INTEGER NOT NULL,
                    open REAL NOT NULL,
                    high REAL NOT NULL,
                    low REAL NOT NULL,
                    close REAL NOT NULL,
                    spread_points INTEGER NOT NULL,
                    tick_volume INTEGER NOT NULL,
                    real_volume INTEGER NOT NULL,
                    PRIMARY KEY(dataset_id, bar_time_unix)
                )
                """
            )
            conn.execute(
                """
                INSERT INTO dataset_bars(dataset_id, bar_time_unix, open, high, low, close, spread_points, tick_volume, real_volume)
                VALUES(1, 1700000040, 1.1, 1.2, 1.0, 1.15, 13, 100, 200)
                """
            )
            commit_db(conn)
            close_db(conn)

            conn = connect_db(paths["default_db"])
            migrated = query_one(
                conn,
                "SELECT price_cost_points, spread_points FROM dataset_bars WHERE dataset_id = 1",
            )
            close_db(conn)

            assert migrated is not None
            assert float(migrated["price_cost_points"]) == 13.0
            assert int(migrated["spread_points"]) == 13


def test_insert_dataset_bars_accepts_canonical_price_cost_points_header():
    with tempfile.TemporaryDirectory(prefix="fxai_dataset_insert_") as tmp_dir:
        with patched_paths(Path(tmp_dir)) as paths:
            conn = connect_db(paths["default_db"])
            conn.execute(
                """
                INSERT INTO datasets(dataset_key, symbol, timeframe, start_unix, end_unix, source_path, created_at)
                VALUES('canonical-price-cost', 'EURUSD', 'M1', 1700000040, 1700000100, '/tmp/input.tsv', 1700000040)
                """
            )
            dataset = query_one(conn, "SELECT id FROM datasets WHERE dataset_key = 'canonical-price-cost'")
            assert dataset is not None
            data_path = Path(tmp_dir) / "bars.tsv"
            data_path.write_text(
                "time_unix\topen\thigh\tlow\tclose\tprice_cost_points\ttick_volume\treal_volume\n"
                "1700000040\t1.10\t1.20\t1.00\t1.15\t1.7\t101\t202\n",
                encoding="utf-8",
            )

            inserted = insert_dataset_bars(conn, int(dataset["id"]), data_path)
            row = query_one(conn, "SELECT price_cost_points, tick_volume, real_volume FROM dataset_bars")
            close_db(conn)

            assert inserted == 1
            assert row is not None
            assert float(row["price_cost_points"]) == 1.7
            assert int(row["tick_volume"]) == 101
            assert int(row["real_volume"]) == 202


def test_environment_reports_turso_dependencies():
    report = validate_environment()
    assert report["dependencies"]["libsql"] is True
    assert report["dependencies"]["turso_cli"] is True
    assert str(report["database"]["backend"]).startswith("turso_")


def test_embedded_replica_commit_triggers_sync():
    raw = Mock()
    register_connection_config(
        raw,
        TursoConfig(
            database=Path("/tmp/fxai-test.db"),
            sync_url="libsql://example.turso.io",
            auth_token="token",
        ),
    )
    commit_backend(raw)
    raw.commit.assert_called_once()
    raw.sync.assert_called_once()


def test_partial_turso_env_is_rejected(monkeypatch):
    monkeypatch.setenv("TURSO_DATABASE_URL", "libsql://example.turso.io")
    monkeypatch.delenv("TURSO_AUTH_TOKEN", raising=False)
    try:
        connect_db(Path("/tmp/fxai_partial_env.db"))
    except OfflineLabError as exc:
        assert "partial Turso configuration" in str(exc)
    else:
        raise AssertionError("expected partial Turso configuration to be rejected")


def test_partial_turso_platform_env_is_rejected(monkeypatch):
    monkeypatch.setenv("TURSO_ORGANIZATION", "fxai-org")
    monkeypatch.delenv("TURSO_API_TOKEN", raising=False)
    try:
        connect_db(Path("/tmp/fxai_partial_platform.db"))
    except OfflineLabError as exc:
        assert "partial Turso platform configuration" in str(exc)
    else:
        raise AssertionError("expected partial Turso platform configuration to be rejected")


def test_connect_backend_rejects_partial_config():
    try:
        connect_backend(
            TursoConfig(
                database=Path("/tmp/fxai-partial.db"),
                sync_url="libsql://example.turso.io",
                auth_token="",
            )
        )
    except ValueError as exc:
        assert "partial Turso configuration" in str(exc)
    else:
        raise AssertionError("expected connect_backend to reject partial Turso config")


def test_turso_encryption_and_sync_interval_are_applied(monkeypatch):
    monkeypatch.setenv(TURSO_ENCRYPTION_KEY_ENV, "enc-key")
    monkeypatch.setenv(TURSO_SYNC_INTERVAL_ENV, "2.5")
    with tempfile.TemporaryDirectory(prefix="fxai_turso_cfg_") as tmp_dir:
        with patched_paths(Path(tmp_dir)) as paths:
            conn = connect_db(paths["default_db"])
            versions = query_one(
                conn,
                "SELECT meta_value FROM lab_metadata WHERE meta_key = ?",
                ("turso_encryption",),
            )
            close_db(conn)
            assert versions is not None
            assert str(versions["meta_value"]) == "enabled"


def test_vector_refresh_and_neighbors_smoke():
    with tempfile.TemporaryDirectory(prefix="fxai_vec_") as tmp_dir:
        with patched_paths(Path(tmp_dir)) as paths:
            bootstrap_environment(paths["default_db"], init_db=True)
            conn = connect_db(paths["default_db"])
            profile = "vec"
            conn.execute(
                """
                INSERT INTO shadow_fleet_observations(profile_name, symbol, plugin_name, family_id, captured_at, source_path,
                                                      source_sha256, meta_weight, reliability, global_edge, context_edge,
                                                      context_regret, portfolio_objective, portfolio_stability, portfolio_corr,
                                                      portfolio_div, route_value, route_regret, route_counterfactual,
                                                      shadow_score, regime_id, horizon_minutes, obs_count,
                                                      policy_capital_efficiency, portfolio_pressure,
                                                      control_plane_score, portfolio_supervisor_score, payload_json)
                VALUES(?, ?, ?, 2, ?, 'fixture.tsv', '', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, 5, 1, ?, ?, ?, ?, '{}')
                """,
                (profile, "EURUSD", "AI_A", 1700000000, 0.6, 0.7, 0.2, 0.15, 0.05, 0.5, 0.55, 0.2, 0.45, 0.3, 0.1, 0.1, 0.4, 0.35, 0.6, 0.2, 0.25, 0.22),
            )
            conn.execute(
                """
                INSERT INTO shadow_fleet_observations(profile_name, symbol, plugin_name, family_id, captured_at, source_path,
                                                      source_sha256, meta_weight, reliability, global_edge, context_edge,
                                                      context_regret, portfolio_objective, portfolio_stability, portfolio_corr,
                                                      portfolio_div, route_value, route_regret, route_counterfactual,
                                                      shadow_score, regime_id, horizon_minutes, obs_count,
                                                      policy_capital_efficiency, portfolio_pressure,
                                                      control_plane_score, portfolio_supervisor_score, payload_json)
                VALUES(?, ?, ?, 2, ?, 'fixture.tsv', '', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, 5, 1, ?, ?, ?, ?, '{}')
                """,
                (profile, "GBPUSD", "AI_B", 1700000100, 0.58, 0.68, 0.18, 0.16, 0.04, 0.48, 0.52, 0.18, 0.42, 0.28, 0.11, 0.09, 0.38, 0.33, 0.58, 0.22, 0.24, 0.21),
            )
            payload = refresh_research_vectors(conn, profile)
            neighbors = latest_symbol_shadow_neighbors(conn, profile, "EURUSD", limit=3)
            close_db(conn)
            assert payload["shadow_vectors"] >= 2
            assert neighbors
            assert str(neighbors[0]["symbol"]) == "GBPUSD"


def test_vector_refresh_replaces_stale_shadow_vectors():
    with tempfile.TemporaryDirectory(prefix="fxai_vec_stale_") as tmp_dir:
        with patched_paths(Path(tmp_dir)) as paths:
            bootstrap_environment(paths["default_db"], init_db=True)
            conn = connect_db(paths["default_db"])
            profile = "vec-stale"
            conn.execute(
                """
                INSERT INTO shadow_fleet_observations(profile_name, symbol, plugin_name, family_id, captured_at, source_path,
                                                      source_sha256, meta_weight, reliability, global_edge, context_edge,
                                                      context_regret, portfolio_objective, portfolio_stability, portfolio_corr,
                                                      portfolio_div, route_value, route_regret, route_counterfactual,
                                                      shadow_score, regime_id, horizon_minutes, obs_count,
                                                      policy_capital_efficiency, portfolio_pressure,
                                                      control_plane_score, portfolio_supervisor_score, payload_json)
                VALUES(?, 'EURUSD', 'AI_A', 2, 1700000000, 'fixture.tsv', '', 0.6, 0.7, 0.2, 0.15, 0.05, 0.5, 0.55, 0.2,
                       0.45, 0.3, 0.1, 0.1, 0.4, 1, 5, 1, 0.6, 0.2, 0.25, 0.22, '{}')
                """,
                (profile,),
            )
            first = refresh_research_vectors(conn, profile, "EURUSD")
            assert first["shadow_vectors"] == 1
            count = query_one(
                conn,
                "SELECT COUNT(*) AS n FROM research_vectors WHERE profile_name = ? AND symbol = ? AND vector_scope = 'analog_shadow'",
                (profile, "EURUSD"),
            )
            assert count is not None
            assert int(count["n"]) == 1
            conn.execute(
                """
                INSERT INTO shadow_fleet_observations(profile_name, symbol, plugin_name, family_id, captured_at, source_path,
                                                      source_sha256, meta_weight, reliability, global_edge, context_edge,
                                                      context_regret, portfolio_objective, portfolio_stability, portfolio_corr,
                                                      portfolio_div, route_value, route_regret, route_counterfactual,
                                                      shadow_score, regime_id, horizon_minutes, obs_count,
                                                      policy_capital_efficiency, portfolio_pressure,
                                                      control_plane_score, portfolio_supervisor_score, payload_json)
                VALUES(?, 'EURUSD', 'AI_A', 2, 1700000100, 'fixture.tsv', '', 0.61, 0.71, 0.22, 0.17, 0.04, 0.51, 0.57, 0.19,
                       0.47, 0.32, 0.09, 0.12, 0.42, 1, 5, 1, 0.61, 0.21, 0.27, 0.24, '{}')
                """,
                (profile,),
            )
            second = refresh_research_vectors(conn, profile, "EURUSD")
            assert second["shadow_vectors"] == 1
            count = query_one(
                conn,
                "SELECT COUNT(*) AS n FROM research_vectors WHERE profile_name = ? AND symbol = ? AND vector_scope = 'analog_shadow'",
                (profile, "EURUSD"),
            )
            close_db(conn)
            assert count is not None
            assert int(count["n"]) == 1
