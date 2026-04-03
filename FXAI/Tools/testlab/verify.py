from __future__ import annotations

from pathlib import Path

from .compile import compile_target
from .shared import REPO_ROOT


def run_verify_all(refresh_golden: bool = False) -> dict[str, object]:
    from offline_lab.exporter import compile_export_runner
    from offline_lab.verification import run_pytest_suite, verify_deterministic_outputs

    payload: dict[str, object] = {
        "compile_main": int(compile_target(Path("FXAI.mq5"), "main_ea")),
        "compile_audit": int(compile_target(Path("Tests/FXAI_AuditRunner.mq5"), "audit_runner")),
        "compile_export": int(compile_export_runner()),
    }
    payload["pytest"] = run_pytest_suite(REPO_ROOT)
    payload["deterministic"] = verify_deterministic_outputs(refresh_golden=refresh_golden)
    payload["ok"] = (
        payload["compile_main"] == 0 and
        payload["compile_audit"] == 0 and
        payload["compile_export"] == 0 and
        bool(payload["pytest"].get("ok")) and
        bool(payload["deterministic"].get("ok"))
    )
    return payload
