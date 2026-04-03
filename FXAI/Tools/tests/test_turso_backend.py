from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import Mock

from offline_lab.db_backend import TursoConfig, commit_backend, connect_backend, register_connection_config
from offline_lab.common import OfflineLabError, close_db, connect_db, query_one
from offline_lab.environment import bootstrap_environment, validate_environment
from offline_lab.fixtures import patched_paths


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
