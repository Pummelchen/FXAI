from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

from offline_lab import cli as offline_cli
from offline_lab.common import connect_db
from offline_lab.environment import bootstrap_environment
from offline_lab.fixtures import patched_paths, seed_completed_run_fixture, seed_profile_fixture


def test_best_params_smoke():
    with tempfile.TemporaryDirectory(prefix="fxai_best_") as tmp_dir:
        with patched_paths(Path(tmp_dir)) as paths:
            bootstrap_environment(paths["default_db"], init_db=True)
            conn = connect_db(paths["default_db"])
            fixture = seed_profile_fixture(conn, profile_name="bestparams")
            seed_completed_run_fixture(conn, fixture)
            conn.close()

            args = argparse.Namespace(
                db=str(paths["default_db"]),
                profile="bestparams",
                dataset_keys="",
                group_key="",
                symbol="",
                symbol_list="",
                symbol_pack="",
                runtime_mode="research",
            )
            rc = offline_cli.cmd_best_params(args)
            assert rc == 0


def test_control_loop_smoke():
    with tempfile.TemporaryDirectory(prefix="fxai_loop_") as tmp_dir:
        with patched_paths(Path(tmp_dir)) as paths:
            bootstrap_environment(paths["default_db"], init_db=True)
            conn = connect_db(paths["default_db"])
            fixture = seed_profile_fixture(conn, profile_name="loop")
            dataset = dict(
                conn.execute(
                    "SELECT * FROM datasets WHERE dataset_key = ?",
                    (fixture["dataset_key"],),
                ).fetchone()
            )

            saved = {
                "compile_export_runner": offline_cli.compile_export_runner,
                "compile_audit_runner": offline_cli.compile_audit_runner,
                "resolve_dataset_rows": offline_cli.resolve_dataset_rows,
                "run_dataset_baseline": offline_cli.run_dataset_baseline,
                "run_dataset_campaign": offline_cli.run_dataset_campaign,
                "cmd_best_params": offline_cli.cmd_best_params,
            }
            best_params_called = {"count": 0}

            def fake_resolve_dataset_rows(_conn, _args, *_rest):
                return [dataset]

            def fake_run_dataset_baseline(_conn, _dataset, _profile_name, cycle_args, out_dir):
                return {"summary": {}, "base_args": cycle_args, "out_dir": str(out_dir)}

            def fake_run_dataset_campaign(*_args, **_kwargs):
                return []

            def fake_best_params(_args):
                best_params_called["count"] += 1
                return 0

            offline_cli.compile_export_runner = lambda: 0
            offline_cli.compile_audit_runner = lambda: 0
            offline_cli.resolve_dataset_rows = fake_resolve_dataset_rows
            offline_cli.run_dataset_baseline = fake_run_dataset_baseline
            offline_cli.run_dataset_campaign = fake_run_dataset_campaign
            offline_cli.cmd_best_params = fake_best_params

            try:
                args = argparse.Namespace(
                    db=str(paths["default_db"]),
                    profile="loop",
                    cycles=1,
                    sleep_seconds=0,
                    skip_compile=False,
                    symbol="EURUSD",
                    symbol_list="",
                    symbol_pack="",
                    months_list="3",
                    dataset_keys="",
                    group_key="",
                    runtime_mode="research",
                )
                rc = offline_cli.cmd_control_loop(args)
                assert rc == 0
                assert best_params_called["count"] == 1
                row = conn.execute(
                    "SELECT status FROM control_cycles WHERE profile_name = ? ORDER BY id DESC LIMIT 1",
                    ("loop",),
                ).fetchone()
                assert row is not None
                assert str(row["status"]) == "ok"
            finally:
                offline_cli.compile_export_runner = saved["compile_export_runner"]
                offline_cli.compile_audit_runner = saved["compile_audit_runner"]
                offline_cli.resolve_dataset_rows = saved["resolve_dataset_rows"]
                offline_cli.run_dataset_baseline = saved["run_dataset_baseline"]
                offline_cli.run_dataset_campaign = saved["run_dataset_campaign"]
                offline_cli.cmd_best_params = saved["cmd_best_params"]
                conn.close()


def test_seed_demo_respects_runtime_mode():
    with tempfile.TemporaryDirectory(prefix="fxai_seed_") as tmp_dir:
        with patched_paths(Path(tmp_dir)) as paths:
            bootstrap_environment(paths["default_db"], init_db=True)
            args = argparse.Namespace(
                db=str(paths["default_db"]),
                profile="seedmode",
                symbol="EURUSD",
                runtime_mode="production",
            )
            rc = offline_cli.cmd_seed_demo(args)
            assert rc == 0
            conn = connect_db(paths["default_db"])
            row = conn.execute(
                "SELECT runtime_mode FROM live_deployment_profiles WHERE profile_name = ? AND symbol = ?",
                ("seedmode", "EURUSD"),
            ).fetchone()
            conn.close()
            assert row is not None
            assert str(row["runtime_mode"]) == "production"
