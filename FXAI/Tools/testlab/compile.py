from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from .shared import (
    COMMON_INI,
    ROOT,
    TERMINAL_INI,
    build_metaeditor_compile_command,
    read_utf16_or_text,
    terminal_running,
    write_utf16,
)


def _stage_copy_ignore(directory: str, names: list[str]) -> set[str]:
    ignored: set[str] = set()
    current = Path(directory)
    if current.name == "GUI":
        for candidate in (".build", ".swiftpm", "Package.resolved"):
            if candidate in names:
                ignored.add(candidate)
    return ignored

def read_common_account() -> tuple[str, str]:
    text = read_utf16_or_text(COMMON_INI)
    login = ""
    server = ""
    for line in text.splitlines():
        if line.startswith("Login="):
            login = line.split("=", 1)[1].strip()
        elif line.startswith("Server="):
            server = line.split("=", 1)[1].strip()
    return login, server


def resolve_credentials(args) -> tuple[str, str, str]:
    common_login, common_server = read_common_account()
    login = args.login or os.environ.get("FXAI_MT5_LOGIN", "") or common_login
    server = args.server or os.environ.get("FXAI_MT5_SERVER", "") or common_server
    password = args.password or os.environ.get("FXAI_MT5_PASSWORD", "")
    return login, server, password

def update_ini_section(path: Path, section_name: str, kv: dict[str, str], encoding: str = "utf-16le") -> None:
    text = read_utf16_or_text(path)
    marker = f"[{section_name}]"
    start = text.find(marker)
    lines = [f"{k}={v}" for k, v in kv.items()]
    new_section = marker + "\n" + "\n".join(lines) + "\n"
    if start < 0:
        if text and not text.endswith("\n"):
            text += "\n"
        text += new_section
    else:
        next_sec = text.find("\n[", start + 1)
        if next_sec < 0:
            next_sec = len(text)
        text = text[:start] + new_section + text[next_sec:]
    if encoding == "utf-16le":
        write_utf16(path, text)
    else:
        path.write_text(text, encoding=encoding)


def read_metaeditor_log(log_path: Path) -> str:
    return read_utf16_or_text(log_path)


def compile_target(relative_target: Path, stage_name: str) -> int:
    stage_dir = Path(tempfile.gettempdir()) / f"fxai_testlab_{stage_name}"
    if stage_dir.exists():
        shutil.rmtree(stage_dir)
    shutil.copytree(
        ROOT,
        stage_dir,
        dirs_exist_ok=True,
        ignore=_stage_copy_ignore,
    )

    stage_target = stage_dir / relative_target
    stage_log = stage_dir / f"compile_{relative_target.stem}.log"
    cmd = build_metaeditor_compile_command(stage_target, stage_log)
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    deadline = time.time() + 1200.0
    built_ex5 = stage_target.with_suffix(".ex5")
    live_ex5 = ROOT / relative_target.with_suffix(".ex5")
    last_log_text = ""

    while time.time() < deadline:
        rc = proc.poll()
        log_text = read_metaeditor_log(stage_log)
        if log_text:
            last_log_text = log_text

        if "0 errors, 0 warnings" in last_log_text and built_ex5.exists():
            if proc.poll() is None:
                proc.kill()
                proc.wait(timeout=5)
            live_ex5.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(built_ex5, live_ex5)
            lines = [line for line in last_log_text.splitlines() if line.strip()]
            if lines:
                print(lines[-1])
            return 0

        if rc is not None:
            if last_log_text:
                lines = [line for line in last_log_text.splitlines() if line.strip()]
                if lines:
                    print(lines[-1])
            if "0 errors, 0 warnings" in last_log_text and built_ex5.exists():
                live_ex5.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(built_ex5, live_ex5)
                return 0
            if proc.stdout is not None:
                sys.stdout.write(proc.stdout.read())
            return rc or 1

        time.sleep(2.0)

    if proc.poll() is None:
        proc.kill()
        proc.wait(timeout=5)
    if last_log_text:
        lines = [line for line in last_log_text.splitlines() if line.strip()]
        if lines:
            print(lines[-1])
    return 124
