from __future__ import annotations

import ast
import json
import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


PRODUCTION_CODE_ROOTS = [
    ROOT / "FXBacktest/Sources",
    ROOT / "FXDataEngine/Sources",
    ROOT / "FXDatabase/Sources",
    ROOT / "FXImporter/Sources",
    ROOT / "FXPlugins",
    ROOT / "FXGUI/Sources",
]


UNFINISHED_MARKER_PATTERN = re.compile(
    r"\b(TODO|FIXME|HACK|XXX|WIP|NotImplemented|not implemented|unimplemented|stub|placeholder)\b",
    re.IGNORECASE,
)


def _read(path: Path) -> str:
    """Read a repository text file with the encoding used by source files."""
    return path.read_text(encoding="utf-8")


def _production_code_files() -> list[Path]:
    """Return source files that participate in production package code.

    Generated build products and Python caches are ignored so stale local
    artifacts cannot hide real source-quality failures.
    """
    extensions = {".swift", ".py", ".metal", ".sh"}
    files: list[Path] = []
    for root in PRODUCTION_CODE_ROOTS:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if ".build" in path.parts or "__pycache__" in path.parts:
                continue
            if path.is_file() and path.suffix in extensions:
                files.append(path)
    return sorted(files)


def _assert_swift_declaration_docs(path: Path, declaration_pattern: re.Pattern[str]) -> None:
    """Assert that selected Swift API declarations have doc comments.

    The check is intentionally scoped to boundary code touched by this cleanup so
    it can enforce documentation without rewriting unrelated implementation
    internals during a hygiene pass.
    """
    lines = _read(path).splitlines()
    missing: list[str] = []
    for index, line in enumerate(lines):
        if not declaration_pattern.match(line):
            continue
        previous = index - 1
        while previous >= 0 and not lines[previous].strip():
            previous -= 1
        if previous < 0 or not lines[previous].lstrip().startswith("///"):
            missing.append(f"{path.relative_to(ROOT)}:{index + 1}: {line.strip()}")
    assert not missing, "Public boundary declarations need doc comments:\n" + "\n".join(missing)


def _swiftpm_manifest_stderr(stderr: str) -> str:
    """Return SwiftPM manifest stderr that represents actionable warnings.

    SwiftPM serializes access to a package build directory and prints a short
    wait notice when another process already owns the lock. That notice is not a
    manifest warning, so the quality gate ignores it while preserving all other
    stderr as a failure signal.
    """
    ignored_fragments = (
        "Another instance of SwiftPM",
        "waiting until that process has finished execution",
    )
    lines = [
        line
        for line in stderr.splitlines()
        if line.strip() and not any(fragment in line for fragment in ignored_fragments)
    ]
    return "\n".join(lines)


def test_production_code_has_no_unfinished_markers() -> None:
    """Prevent stale unfinished markers from re-entering production source files."""
    offenders: list[str] = []
    for path in _production_code_files():
        for line_number, line in enumerate(_read(path).splitlines(), start=1):
            if UNFINISHED_MARKER_PATTERN.search(line):
                offenders.append(f"{path.relative_to(ROOT)}:{line_number}: {line.strip()}")
    assert not offenders, "Unfinished markers remain in production code:\n" + "\n".join(offenders)


def test_fxplugins_manifest_lists_only_existing_sources() -> None:
    """Ensure the dynamic FXPlugins SwiftPM manifest never references stale files."""
    result = subprocess.run(
        ["swift", "package", "--package-path", "FXPlugins", "dump-package"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    assert _swiftpm_manifest_stderr(result.stderr) == ""
    manifest = json.loads(result.stdout)
    missing: list[str] = []
    for target in manifest["targets"]:
        base = ROOT / "FXPlugins"
        if target["name"] == "FXAIPluginsTests":
            base = ROOT / "FXPlugins/API/Tests/FXAIPluginsTests"
        for source in target.get("sources") or []:
            if not (base / source).exists():
                missing.append(f"{target['name']}: {source}")
    assert not missing, "SwiftPM target source entries must resolve to real files:\n" + "\n".join(missing)


def test_sinetest_public_functions_are_documented() -> None:
    """Guard public SineTest helper APIs with operator-readable documentation."""
    _assert_swift_declaration_docs(
        ROOT / "FXDatabase/Sources/BacktestCore/SineWaveAgent.swift",
        re.compile(r"^\s+public\s+(static\s+)?(func|init)\b"),
    )


def test_history_data_error_public_description_is_documented() -> None:
    """Guard the public history error description boundary with doc comments."""
    _assert_swift_declaration_docs(
        ROOT / "FXDatabase/Sources/BacktestCore/HistoryDataError.swift",
        re.compile(r"^\s+public\s+var\s+description\b"),
    )


def test_fxdatabase_exporter_boundary_functions_have_docstrings() -> None:
    """Require docstrings on OfflineLab functions that cross the FXDatabase API."""
    module = ast.parse(_read(ROOT / "FXDataEngine/Tools/OfflineLab/offline_lab/exporter.py"))
    required = {
        "fxdatabase_api_base_url",
        "fetch_fxdatabase_m1_history",
        "validate_fxdatabase_m1_history_response",
        "write_fxdatabase_export_files",
    }
    missing = [
        node.name
        for node in module.body
        if isinstance(node, ast.FunctionDef)
        and node.name in required
        and not ast.get_docstring(node)
    ]
    assert not missing, "FXDatabase API boundary functions need docstrings: " + ", ".join(sorted(missing))
