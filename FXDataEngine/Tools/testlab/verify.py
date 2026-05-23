from __future__ import annotations

from .compile import compile_swift_package
from .shared import REPO_ROOT


def run_verify_all(refresh_golden: bool = False) -> dict[str, object]:
    from offline_lab.verification import run_pytest_suite, verify_deterministic_outputs

    payload: dict[str, object] = {
        "build_data_engine": int(compile_swift_package("data_engine")),
        "build_plugins": int(compile_swift_package("plugins")),
        "build_database": int(compile_swift_package("database")),
        "build_importer": int(compile_swift_package("importer")),
    }
    payload["pytest"] = run_pytest_suite(REPO_ROOT)
    payload["deterministic"] = verify_deterministic_outputs(refresh_golden=refresh_golden)
    payload["ok"] = (
        payload["build_data_engine"] == 0 and
        payload["build_plugins"] == 0 and
        payload["build_database"] == 0 and
        payload["build_importer"] == 0 and
        bool(payload["pytest"].get("ok")) and
        bool(payload["deterministic"].get("ok"))
    )
    return payload
