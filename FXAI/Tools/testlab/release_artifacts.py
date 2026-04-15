from __future__ import annotations

import datetime as dt
import json
import platform
import re
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path

from .compile import compile_target
from .shared import METAEDITOR, REPO_ROOT, ROOT, git_dirty, git_head_commit, sha256_path, write_json


@dataclass(frozen=True)
class MT5BinaryTarget:
    role: str
    source: Path
    binary: Path
    stage_name: str
    description: str
    compatible_profiles: tuple[str, ...] = ("research", "production")


MT5_BINARY_TARGETS: tuple[MT5BinaryTarget, ...] = (
    MT5BinaryTarget(
        role="main_ea",
        source=Path("FXAI.mq5"),
        binary=Path("FXAI.ex5"),
        stage_name="release_main_ea",
        description="Main FXAI Expert Advisor for live trading and Strategy Tester runs.",
    ),
    MT5BinaryTarget(
        role="audit_runner",
        source=Path("Tests/FXAI_AuditRunner.mq5"),
        binary=Path("Tests/FXAI_AuditRunner.ex5"),
        stage_name="release_audit_runner",
        description="Audit Lab runner for MT5-side certification and regression scenarios.",
        compatible_profiles=("research",),
    ),
    MT5BinaryTarget(
        role="offline_export_runner",
        source=Path("Tests/FXAI_OfflineExportRunner.mq5"),
        binary=Path("Tests/FXAI_OfflineExportRunner.ex5"),
        stage_name="release_offline_export_runner",
        description="Offline Lab export runner for reproducible MT5 market-data extraction.",
        compatible_profiles=("research",),
    ),
)


def _safe_token(value: str) -> str:
    token = re.sub(r"[^A-Za-z0-9._-]+", "-", (value or "dev").strip())
    token = token.strip(".-")
    return token or "dev"


def _parse_profiles(raw: str | None) -> tuple[str, ...]:
    profiles: list[str] = []
    for item in (raw or "").replace(";", ",").replace("|", ",").split(","):
        token = item.strip().lower()
        if token and token not in profiles:
            profiles.append(token)
    return tuple(profiles or ["research", "production"])


def _copy_release_binary(root: Path, output_dir: Path, target: MT5BinaryTarget) -> Path:
    source_path = root / target.binary
    if not source_path.exists() or not source_path.is_file():
        raise FileNotFoundError(f"compiled MT5 binary not found: {source_path}")
    output_path = output_dir / target.binary.name
    shutil.copy2(source_path, output_path)
    return output_path


def _entry_for_binary(
    *,
    root: Path,
    release_path: Path,
    target: MT5BinaryTarget,
    version: str,
    release_profile: str,
    compatible_profiles: tuple[str, ...],
) -> dict[str, object]:
    source_path = root / target.source
    entry_profiles = tuple(profile for profile in compatible_profiles if profile in target.compatible_profiles)
    if not entry_profiles:
        entry_profiles = target.compatible_profiles
    return {
        "name": target.binary.name,
        "role": target.role,
        "description": target.description,
        "source": target.source.as_posix(),
        "source_sha256": sha256_path(source_path),
        "release_file": release_path.name,
        "release_sha256": sha256_path(release_path),
        "release_size_bytes": int(release_path.stat().st_size),
        "mt5_install_path": f"MQL5/Experts/FXAI/{target.binary.as_posix()}",
        "fxai_version": version,
        "release_profile": release_profile,
        "compatible_profiles": list(entry_profiles),
    }


def write_mt5_release_bundle(
    *,
    root: Path,
    output_dir: Path,
    version: str,
    release_profile: str,
    compatible_profiles: tuple[str, ...],
    targets: tuple[MT5BinaryTarget, ...] = MT5_BINARY_TARGETS,
    repo_root: Path = REPO_ROOT,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)

    version_token = _safe_token(version)
    manifest_path = output_dir / f"fxai-mt5-{version_token}.manifest.json"
    sums_path = output_dir / f"fxai-mt5-{version_token}.SHA256SUMS"
    zip_path = output_dir / f"fxai-mt5-{version_token}.zip"
    zip_sums_path = output_dir / f"{zip_path.name}.sha256"

    copied: list[tuple[MT5BinaryTarget, Path]] = []
    for target in targets:
        copied.append((target, _copy_release_binary(root, output_dir, target)))

    entries = [
        _entry_for_binary(
            root=root,
            release_path=release_path,
            target=target,
            version=version,
            release_profile=release_profile,
            compatible_profiles=compatible_profiles,
        )
        for target, release_path in copied
    ]

    built_at = dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    manifest: dict[str, object] = {
        "schema_version": 1,
        "artifact_type": "fxai_mt5_binaries",
        "version": version,
        "built_at_utc": built_at,
        "release_profile": release_profile,
        "compatible_profiles": list(compatible_profiles),
        "provenance": {
            "repo_root": str(repo_root),
            "repo_head": git_head_commit(repo_root),
            "repo_dirty": git_dirty(repo_root),
            "build_platform": platform.platform(),
            "python": platform.python_version(),
            "metaeditor": str(METAEDITOR),
        },
        "install": {
            "mt5_root_relative": "MQL5/Experts/FXAI",
            "notes": "Install .ex5 files under the matching FXAI source tree paths shown per artifact.",
        },
        "bundle": {
            "zip": zip_path.name,
            "zip_sha256_file": zip_sums_path.name,
            "sha256s": sums_path.name,
            "manifest": manifest_path.name,
        },
        "artifacts": entries,
    }
    write_json(manifest_path, manifest)

    checksum_lines = [f"{sha256_path(path)}  {path.name}" for _, path in copied]
    checksum_lines.append(f"{sha256_path(manifest_path)}  {manifest_path.name}")
    sums_path.write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")

    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for _, release_path in copied:
            archive.write(release_path, arcname=release_path.name)
        archive.write(manifest_path, arcname=manifest_path.name)
        archive.write(sums_path, arcname=sums_path.name)

    zip_sums_path.write_text(f"{sha256_path(zip_path)}  {zip_path.name}\n", encoding="utf-8")
    payload = dict(manifest)
    payload["generated_files"] = {
        "zip": str(zip_path),
        "zip_sha256": sha256_path(zip_path),
        "zip_sha256_file": str(zip_sums_path),
        "sha256s": str(sums_path),
        "manifest": str(manifest_path),
    }
    return payload


def cmd_package_mt5_release(args) -> int:
    version = str(args.version or "").strip()
    if not version:
        raise SystemExit("--version is required")

    targets = MT5_BINARY_TARGETS
    if not bool(getattr(args, "skip_compile", False)):
        for target in targets:
            rc = compile_target(target.source, target.stage_name)
            if rc != 0:
                return rc

    output_dir = Path(args.output_dir) if args.output_dir else ROOT / "Artifacts/Release"
    release_profile = str(getattr(args, "release_profile", "production") or "production").strip().lower()
    compatible_profiles = _parse_profiles(getattr(args, "compatible_profiles", "research,production"))
    manifest = write_mt5_release_bundle(
        root=ROOT,
        output_dir=output_dir,
        version=version,
        release_profile=release_profile,
        compatible_profiles=compatible_profiles,
        targets=targets,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0
