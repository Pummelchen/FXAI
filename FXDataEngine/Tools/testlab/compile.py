from __future__ import annotations

import configparser
import os
import subprocess
import sys
from pathlib import Path

from .shared import REPO_ROOT, terminal_running


SWIFT_PACKAGES: dict[str, Path] = {
    "data_engine": Path("FXDataEngine"),
    "plugins": Path("FXPlugins"),
    "database": Path("FXDatabase"),
    "importer": Path("FXImporter"),
    "backtest": Path("FXBacktest"),
    "gui": Path("FXGUI"),
}


def compile_swift_package(package: str, configuration: str = "debug") -> int:
    package_key = package.strip().lower()
    if package_key not in SWIFT_PACKAGES:
        raise ValueError(f"unknown Swift package: {package}")
    package_path = REPO_ROOT / SWIFT_PACKAGES[package_key]
    if not (package_path / "Package.swift").exists():
        raise FileNotFoundError(f"Swift package manifest not found: {package_path / 'Package.swift'}")
    cmd = ["swift", "build", "--configuration", configuration]
    proc = subprocess.run(
        cmd,
        cwd=package_path,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    if proc.stdout:
        sys.stdout.write(proc.stdout)
    return int(proc.returncode)


def compile_target(relative_target: Path, stage_name: str) -> int:
    """Compatibility wrapper for older tool callers.

    `relative_target` now points at a Swift package directory under the FXAI repo.
    The old terminal-side compiler path is intentionally not present in FXDataEngine.
    """
    target_text = relative_target.as_posix().strip("/")
    package_name = Path(target_text).parts[0] if target_text else stage_name
    if package_name == "FXDataEngine":
        package_name = "data_engine"
    elif package_name == "FXPlugins":
        package_name = "plugins"
    elif package_name == "FXDatabase":
        package_name = "database"
    elif package_name == "FXImporter":
        package_name = "importer"
    elif package_name == "FXBacktest":
        package_name = "backtest"
    elif package_name == "FXGUI":
        package_name = "gui"
    else:
        package_name = stage_name
    return compile_swift_package(package_name)


def resolve_credentials(args) -> tuple[str, str, str]:
    """Compatibility helper for older audit commands that still accept account flags."""
    login = getattr(args, "login", "") or os.environ.get("FXAI_MT5_LOGIN", "") or os.environ.get("MT5_LOGIN", "")
    server = getattr(args, "server", "") or os.environ.get("FXAI_MT5_SERVER", "") or os.environ.get("MT5_SERVER", "")
    password = getattr(args, "password", "") or os.environ.get("FXAI_MT5_PASSWORD", "") or os.environ.get("MT5_PASSWORD", "")
    return str(login), str(server), str(password)


def update_ini_section(path: Path, section: str, values: dict[str, str]) -> None:
    """Small INI patcher kept for legacy audit launch compatibility."""
    parser = configparser.ConfigParser(strict=False)
    parser.optionxform = str
    if path.exists():
        try:
            parser.read(path, encoding="utf-8")
        except UnicodeError:
            parser.read(path, encoding="utf-16le")
    if not parser.has_section(section):
        parser.add_section(section)
    for key, value in values.items():
        parser.set(section, key, str(value))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        parser.write(handle)
