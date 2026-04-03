from __future__ import annotations

import tempfile
from pathlib import Path

from offline_lab.attribution import write_attribution_profiles
from offline_lab.common import close_db, connect_db
from offline_lab.dashboard import write_profile_dashboard
from offline_lab.environment import bootstrap_environment
from offline_lab.fixtures import patched_paths, seed_profile_fixture
from offline_lab.governance import run_autonomous_governance
from offline_lab.lineage import write_lineage_report
from offline_lab.bundle import write_minimal_live_bundle
from offline_lab.performance import write_performance_reports
from offline_lab.student_router import write_student_router_profiles
from offline_lab.teacher_factory import write_live_deployment_profiles


def test_lineage_and_bundle_outputs_render():
    with tempfile.TemporaryDirectory(prefix="fxai_lineage_") as tmp_dir:
        with patched_paths(Path(tmp_dir)) as paths:
            bootstrap_environment(paths["default_db"], init_db=True)
            conn = connect_db(paths["default_db"])
            fixture = seed_profile_fixture(conn, profile_name="lineage")
            args = type("Args", (), {"profile": fixture["profile_name"], "db": str(paths["default_db"]), "runtime_mode": "research"})()
            write_attribution_profiles(conn, args)
            write_student_router_profiles(conn, args)
            write_live_deployment_profiles(conn, args)
            run_autonomous_governance(conn, args, "lineage")
            write_performance_reports([fixture["symbol"]], fixture["profile_name"])
            write_profile_dashboard(conn, fixture["profile_name"])
            lineage = write_lineage_report(conn, fixture["profile_name"], fixture["symbol"])
            bundle = write_minimal_live_bundle(conn, fixture["profile_name"])
            close_db(conn)

            assert Path(lineage["json_path"]).exists()
            assert Path(lineage["md_path"]).exists()
            assert Path(lineage["html_path"]).exists()
            assert fixture["symbol"] in lineage["symbols"]
            assert Path(bundle["manifest_path"]).exists()
            assert int(bundle["symbol_count"]) == 1
