from __future__ import annotations

import tempfile
from pathlib import Path

from offline_lab.common import close_db, connect_db
from offline_lab.dashboard import live_state_snapshot, write_profile_dashboard
from offline_lab.environment import bootstrap_environment
from offline_lab.fixtures import patched_paths, seed_profile_fixture
from offline_lab.governance import run_autonomous_governance
from offline_lab.teacher_factory import write_live_deployment_profiles
from offline_lab.attribution import write_attribution_profiles
from offline_lab.student_router import write_student_router_profiles


def test_dashboard_and_live_state_render():
    with tempfile.TemporaryDirectory(prefix="fxai_dash_") as tmp_dir:
        with patched_paths(Path(tmp_dir)) as paths:
            bootstrap_environment(paths["default_db"], init_db=True)
            conn = connect_db(paths["default_db"])
            fixture = seed_profile_fixture(conn)
            args = type("Args", (), {"profile": fixture["profile_name"], "db": str(paths["default_db"]), "runtime_mode": "research"})()
            write_attribution_profiles(conn, args)
            write_student_router_profiles(conn, args)
            write_live_deployment_profiles(conn, args)
            run_autonomous_governance(conn, args, "dash")
            payload = write_profile_dashboard(conn, fixture["profile_name"])
            live = live_state_snapshot(fixture["profile_name"], fixture["symbol"])
            close_db(conn)
            assert Path(payload["html_path"]).exists()
            assert live["deployment"]
            assert live["router"]
