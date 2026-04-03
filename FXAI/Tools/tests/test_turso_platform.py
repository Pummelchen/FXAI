from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

from offline_lab import cli as offline_cli
from offline_lab.common import close_db, connect_db, query_one
from offline_lab.environment import bootstrap_environment
from offline_lab.fixtures import patched_paths
from offline_lab.turso_platform import TursoBranchResult


def test_turso_branch_create_command_persists_metadata():
    with tempfile.TemporaryDirectory(prefix="fxai_branch_") as tmp_dir:
        with patched_paths(Path(tmp_dir)) as paths:
            bootstrap_environment(paths["default_db"], init_db=True)
            saved_resolve = offline_cli.resolve_platform_config
            saved_create = offline_cli.create_branch_database
            try:
                offline_cli.resolve_platform_config = lambda _path: type(
                    "Cfg",
                    (),
                    {
                        "database_name": "fxai-main",
                        "group_name": "",
                        "location_name": "",
                    },
                )()
                offline_cli.create_branch_database = lambda **kwargs: TursoBranchResult(
                    source_database="fxai-main",
                    target_database="fxai-main-branch",
                    branch_kind="campaign",
                    source_timestamp="",
                    group_name="",
                    location_name="",
                    sync_url="libsql://fxai-main-branch.turso.io",
                    auth_token="token",
                    env_artifact_path=Path(tmp_dir) / "branch.env",
                    created_at=1700000000,
                )
                args = argparse.Namespace(
                    db=str(paths["default_db"]),
                    profile="branch",
                    source_database="fxai-main",
                    target_database="fxai-main-branch",
                    timestamp="",
                    group_name="",
                    location_name="",
                    token_expiration="7d",
                    read_only_token=False,
                )
                rc = offline_cli.cmd_turso_branch_create(args)
                assert rc == 0
                conn = connect_db(paths["default_db"])
                row = query_one(conn, "SELECT target_database FROM turso_branch_runs WHERE target_database = ?", ("fxai-main-branch",))
                close_db(conn)
                assert row is not None
            finally:
                offline_cli.resolve_platform_config = saved_resolve
                offline_cli.create_branch_database = saved_create


def test_turso_audit_sync_command_ingests_rows():
    with tempfile.TemporaryDirectory(prefix="fxai_audit_") as tmp_dir:
        with patched_paths(Path(tmp_dir)) as paths:
            bootstrap_environment(paths["default_db"], init_db=True)
            saved_resolve = offline_cli.resolve_platform_config
            saved_sync = offline_cli.sync_audit_logs
            try:
                offline_cli.resolve_platform_config = lambda _path: type(
                    "Cfg",
                    (),
                    {
                        "organization_slug": "fxai-org",
                    },
                )()
                def fake_sync(conn, _config, limit=50, pages=1):
                    conn.execute(
                        """
                        INSERT INTO turso_audit_log_events(organization_slug, event_id, event_type, actor_name,
                                                           actor_email, target_type, target_name, occurred_at,
                                                           source_page, payload_json, observed_at)
                        VALUES('fxai-org', 'evt-1', 'db.branch', 'tester', 'tester@example.com',
                               'database', 'fxai-main-branch', '2026-04-04T00:00:00Z', 1, '{}', 1700000000)
                        """
                    )
                    return {"organization": "fxai-org", "seen": 1, "inserted": 1, "sample": [{"event_id": "evt-1"}]}
                offline_cli.sync_audit_logs = fake_sync
                args = argparse.Namespace(db=str(paths["default_db"]), limit=25, pages=1)
                rc = offline_cli.cmd_turso_audit_sync(args)
                assert rc == 0
                conn = connect_db(paths["default_db"])
                row = query_one(conn, "SELECT event_id FROM turso_audit_log_events WHERE event_id = ?", ("evt-1",))
                close_db(conn)
                assert row is not None
            finally:
                offline_cli.resolve_platform_config = saved_resolve
                offline_cli.sync_audit_logs = saved_sync
