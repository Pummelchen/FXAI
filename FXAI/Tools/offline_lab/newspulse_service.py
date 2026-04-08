from __future__ import annotations

import shutil
import subprocess
import tempfile
import time
from pathlib import Path

from .common import OfflineLabError
from .newspulse_contracts import (
    NEWSPULSE_SERVICE_SOURCE,
    TERMINAL_SERVICE_BINARY,
    TERMINAL_SERVICE_SOURCE,
)
from testlab.shared import METAEDITOR, TERMINAL_ROOT, WINE, read_utf16_or_text, to_wine_path


def install_calendar_service(compile_service: bool = True) -> dict[str, object]:
    if not NEWSPULSE_SERVICE_SOURCE.exists():
        raise OfflineLabError(f"NewsPulse calendar service source is missing: {NEWSPULSE_SERVICE_SOURCE}")
    TERMINAL_SERVICE_SOURCE.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(NEWSPULSE_SERVICE_SOURCE, TERMINAL_SERVICE_SOURCE)
    payload = {
        "source_path": str(NEWSPULSE_SERVICE_SOURCE),
        "installed_path": str(TERMINAL_SERVICE_SOURCE),
        "terminal_root": str(TERMINAL_ROOT),
        "compiled": False,
        "binary_path": str(TERMINAL_SERVICE_BINARY),
    }
    if compile_service:
        payload.update(compile_calendar_service())
    return payload


def compile_calendar_service(timeout_sec: int = 600) -> dict[str, object]:
    if not TERMINAL_SERVICE_SOURCE.exists():
        raise OfflineLabError(
            "NewsPulse calendar service is not installed into MQL5/Services. "
            "Run newspulse-install-service first."
        )
    with tempfile.TemporaryDirectory(prefix="fxai_newspulse_service_") as tmp_dir:
        stage_root = Path(tmp_dir)
        stage_source = stage_root / TERMINAL_SERVICE_SOURCE.name
        stage_log = stage_root / "compile_newspulse_service.log"
        shutil.copy2(TERMINAL_SERVICE_SOURCE, stage_source)

        cmd = [
            str(WINE),
            str(METAEDITOR),
            f"/compile:{to_wine_path(stage_source)}",
            f"/log:{to_wine_path(stage_log)}",
        ]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        deadline = time.time() + float(timeout_sec)
        built_ex5 = stage_source.with_suffix(".ex5")
        last_log = ""

        while time.time() < deadline:
            rc = proc.poll()
            if stage_log.exists():
                last_log = read_utf16_or_text(stage_log)
            if "0 errors, 0 warnings" in last_log and built_ex5.exists():
                if proc.poll() is None:
                    proc.kill()
                    proc.wait(timeout=5)
                TERMINAL_SERVICE_BINARY.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(built_ex5, TERMINAL_SERVICE_BINARY)
                log_path = TERMINAL_SERVICE_SOURCE.with_suffix(".compile.log")
                shutil.copy2(stage_log, log_path)
                return {
                    "compiled": True,
                    "binary_path": str(TERMINAL_SERVICE_BINARY),
                    "log_path": str(log_path),
                }
            if rc is not None:
                if proc.stdout is not None:
                    _ = proc.stdout.read()
                raise OfflineLabError(
                    f"NewsPulse calendar service compile failed. "
                    f"Last log line: {(last_log.splitlines()[-1] if last_log.splitlines() else 'no log output')}"
                )
            time.sleep(1.5)

        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=5)
        raise OfflineLabError("NewsPulse calendar service compile timed out")
