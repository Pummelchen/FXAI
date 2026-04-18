from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

from .common import close_db, connect_db, safe_token
from .adaptive_router import write_adaptive_router_profiles
from .adaptive_router_replay import build_adaptive_router_replay_report
from .dashboard import live_state_snapshot, write_profile_dashboard
from .environment import bootstrap_environment
from .fixtures import clear_generated_outputs, patched_paths, seed_profile_fixture
from .governance import run_autonomous_governance
from .lineage import write_lineage_report
from .bundle import write_minimal_live_bundle
from .performance import write_performance_reports
from .student_router import write_student_router_profiles
from .attribution import write_attribution_profiles
from .supervisor_service import write_supervisor_command_artifacts, write_supervisor_service_artifacts
from .teacher_factory import write_live_deployment_profiles


def _load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _golden_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "tests" / "golden"


def _write_golden(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _sanitize_fixture_text(text: str) -> str:
    from .common import COMMON_EXPORT_DIR, COMMON_PROMOTION_DIR, OFFLINE_DIR
    from testlab.shared import COMMON_FILES, ROOT, RUNTIME_DIR, TESTER_PRESET_DIR

    sanitized = str(text)
    exact_path_replacements = [
        (str(COMMON_PROMOTION_DIR), "<FXAI_PROMOTION_DIR>"),
        (str(RUNTIME_DIR), "<FXAI_RUNTIME_DIR>"),
        (str(COMMON_EXPORT_DIR), "<FXAI_EXPORT_DIR>"),
        (str(COMMON_FILES), "<FXAI_COMMON_FILES>"),
        (str(TESTER_PRESET_DIR), "<FXAI_TESTER_PRESET_DIR>"),
        (str(OFFLINE_DIR), "<FXAI_ROOT>/Tools/OfflineLab"),
        (str(ROOT), "<FXAI_ROOT>"),
    ]
    for source, target in exact_path_replacements:
        if source:
            sanitized = sanitized.replace(source, target)

    sanitized = re.sub(r"fxai_fixture_[A-Za-z0-9_]+", "fxai_fixture_FIXED", sanitized)
    sanitized = re.sub(r"\"(generated_at|expires_at|reviewed_at|created_at|promoted_at|started_at|finished_at)\":\s*\d+", r'"\1": 1700000000', sanitized)
    sanitized = re.sub(r"\"(generated_at|expires_at|reviewed_at|created_at|promoted_at|started_at|finished_at)\":\s*\"\d+\"", r'"\1": "1700000000"', sanitized)
    sanitized = re.sub(r"\"artifact_age_sec\":\s*\d+", '"artifact_age_sec": 0', sanitized)
    sanitized = re.sub(
        r"\"(generated_at|expires_at|reviewed_at|created_at|promoted_at|started_at|finished_at)\":\s*\"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9:]+Z\"",
        r'"\1": "2026-01-01T00:00:00Z"',
        sanitized,
    )
    sanitized = re.sub(
        r"\"at\":\s*\"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9:]+Z\"",
        '"at": "2026-01-01T00:00:00Z"',
        sanitized,
    )
    sanitized = re.sub(r"(\t(?:generated_at|expires_at|promoted_at|started_at|finished_at))\t\d+", r"\1\t1700000000", sanitized)
    sanitized = re.sub(
        r"/(?:[^/\s\"`]+/)*FILE_COMMON/FXAI/Offline/Promotions",
        "<FXAI_PROMOTION_DIR>",
        sanitized,
    )
    sanitized = re.sub(
        r"/[^\"`\n]+?/Common/Files/FXAI/Offline/Promotions",
        "<FXAI_PROMOTION_DIR>",
        sanitized,
    )
    sanitized = re.sub(
        r"/(?:[^/\s\"`]+/)*FILE_COMMON/FXAI/Runtime",
        "<FXAI_RUNTIME_DIR>",
        sanitized,
    )
    sanitized = re.sub(
        r"/[^\"`\n]+?/Common/Files/FXAI/Runtime",
        "<FXAI_RUNTIME_DIR>",
        sanitized,
    )
    sanitized = re.sub(
        r"/(?:[^/\s\"`]+/)*FILE_COMMON",
        "<FXAI_COMMON_FILES>",
        sanitized,
    )
    sanitized = re.sub(
        r"/(?:[^/\s\"`]+/)*MT5/Profiles/Tester",
        "<FXAI_TESTER_PRESET_DIR>",
        sanitized,
    )
    sanitized = re.sub(
        r"/(?:[^/\s\"`]+/){2,}OfflineLab",
        "<FXAI_ROOT>/Tools/OfflineLab",
        sanitized,
    )
    return sanitized


def verify_deterministic_outputs(refresh_golden: bool = False) -> dict[str, object]:
    with tempfile.TemporaryDirectory(prefix="fxai_fixture_") as tmp_dir:
        with patched_paths(Path(tmp_dir)) as paths:
            bootstrap_environment(paths["default_db"], init_db=True)
            conn = connect_db(paths["default_db"])
            fixture = seed_profile_fixture(conn)
            profile_name = str(fixture["profile_name"])
            clear_generated_outputs(paths, profile_name)
            args = type("Args", (), {"profile": profile_name, "db": str(paths["default_db"]), "runtime_mode": "research"})()
            write_attribution_profiles(conn, args)
            write_student_router_profiles(conn, args)
            write_adaptive_router_profiles(conn, args)
            write_live_deployment_profiles(conn, args)
            write_supervisor_service_artifacts(conn, args)
            write_supervisor_command_artifacts(conn, args)
            run_autonomous_governance(conn, args, "fixture_cycle")
            write_student_router_profiles(conn, args)
            write_adaptive_router_profiles(conn, args)
            build_adaptive_router_replay_report(symbol="EURUSD", hours_back=72)
            write_performance_reports(["EURUSD"], profile_name)
            dashboard = write_profile_dashboard(conn, profile_name)
            lineage = write_lineage_report(conn, profile_name, "EURUSD")
            bundle = write_minimal_live_bundle(conn, profile_name)
            close_db(conn)

            live = live_state_snapshot(profile_name, "EURUSD")
            checks = {
                "live_deploy_tsv": paths["common_promotion_dir"] / "fxai_live_deploy_EURUSD.tsv",
                "student_router_tsv": paths["common_promotion_dir"] / "fxai_student_router_EURUSD.tsv",
                "adaptive_router_tsv": paths["common_promotion_dir"] / "fxai_adaptive_router_EURUSD.tsv",
                "world_plan_tsv": paths["common_promotion_dir"] / "fxai_world_plan_EURUSD.tsv",
                "operator_dashboard_json": Path(dashboard["json_path"]),
                "lineage_json": Path(lineage["json_path"]),
                "bundle_manifest_json": Path(bundle["manifest_path"]),
                "live_state_json": paths["research_dir"] / safe_token(profile_name) / "live_state_EURUSD.json",
            }
            checks["live_state_json"].write_text(json.dumps(live, indent=2, sort_keys=True), encoding="utf-8")
            golden_dir = _golden_dir()
            mismatches = []
            for name, path in checks.items():
                current = _sanitize_fixture_text(_load_text(path))
                golden = golden_dir / f"{name}.golden"
                if refresh_golden or not golden.exists():
                    _write_golden(golden, current)
                expected = _sanitize_fixture_text(_load_text(golden))
                if current != expected:
                    mismatches.append(name)
            return {
                "ok": len(mismatches) == 0,
                "mismatches": mismatches,
                "fixture_profile": profile_name,
                "golden_dir": str(golden_dir),
            }


def run_pytest_suite(repo_root: Path) -> dict[str, object]:
    fxai_root = repo_root / "FXAI"
    cmd = [sys.executable, "-m", "pytest", "Tools/tests", "-q"]
    env = os.environ.copy()
    repo_tools = str(fxai_root / "Tools")
    existing_pythonpath = str(env.get("PYTHONPATH", "") or "").strip()
    env["PYTHONPATH"] = (
        repo_tools
        if not existing_pythonpath
        else os.pathsep.join([repo_tools, existing_pythonpath])
    )
    proc = subprocess.run(
        cmd,
        cwd=fxai_root,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return {"ok": proc.returncode == 0, "returncode": proc.returncode, "output": proc.stdout}
