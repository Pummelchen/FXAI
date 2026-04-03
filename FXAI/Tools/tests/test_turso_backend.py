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
from offline_lab.common import OfflineLabError, close_db, connect_db, query_one
from offline_lab.environment import bootstrap_environment, validate_environment
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
