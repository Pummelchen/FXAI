from __future__ import annotations

import datetime as dt
import json
import platform
import re
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path

from .compile import compile_swift_package
from .shared import FXAI_CONFIG_PATH, REPO_ROOT, ROOT, TOOLCHAIN_PROFILE, git_dirty, git_head_commit, sha256_path, write_json


@dataclass(frozen=True)
class SwiftPackageTarget:
    role: str
    package: str
    source: Path
    build_artifact: Path
    stage_name: str
    description: str
    compatible_profiles: tuple[str, ...] = ("research", "production")


SWIFT_PACKAGE_TARGETS: tuple[SwiftPackageTarget, ...] = (
    SwiftPackageTarget(
        role="data_engine",
        package="data_engine",
        source=Path("FXDataEngine/Package.swift"),
        build_artifact=Path("FXDataEngine/.build/debug/FXDataEngineCLI"),
        stage_name="release_data_engine",
        description="FXDataEngine Swift package and command-line data pipeline verifier.",
    ),
    SwiftPackageTarget(
        role="plugins",
        package="plugins",
        source=Path("FXPlugins/Package.swift"),
        build_artifact=Path("FXPlugins/.build/debug/Modules/FXAIPlugins.swiftmodule"),
        stage_name="release_plugins",
        description="FXPlugins Swift module with CPU, Metal, Python bridge, and NLP plugin surfaces.",
        compatible_profiles=("research",),
    ),
    SwiftPackageTarget(
        role="database",
        package="database",
        source=Path("FXDatabase/Package.swift"),
        build_artifact=Path("FXDatabase/.build/debug/FXDatabase"),
        stage_name="release_database",
        description="FXDatabase Swift package and API gateway executable.",
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


def _copy_release_artifact(root: Path, output_dir: Path, target: SwiftPackageTarget) -> Path:
    source_path = root / target.build_artifact
    if not source_path.exists() or not source_path.is_file():
        raise FileNotFoundError(f"Swift build artifact not found: {source_path}")
    output_name = f"{target.role}-{source_path.name}"
    output_path = output_dir / output_name
    shutil.copy2(source_path, output_path)
    return output_path


def _entry_for_artifact(
    *,
    root: Path,
    release_path: Path,
    target: SwiftPackageTarget,
    version: str,
    release_profile: str,
    compatible_profiles: tuple[str, ...],
) -> dict[str, object]:
    source_path = root / target.source
    entry_profiles = tuple(profile for profile in compatible_profiles if profile in target.compatible_profiles)
    if not entry_profiles:
        entry_profiles = target.compatible_profiles
    return {
        "name": release_path.name,
        "role": target.role,
        "description": target.description,
        "source": target.source.as_posix(),
        "source_sha256": sha256_path(source_path),
        "release_file": release_path.name,
        "release_sha256": sha256_path(release_path),
        "release_size_bytes": int(release_path.stat().st_size),
        "fxai_version": version,
        "release_profile": release_profile,
        "compatible_profiles": list(entry_profiles),
    }


def write_swift_release_bundle(
    *,
    root: Path,
    output_dir: Path,
    version: str,
    release_profile: str,
    compatible_profiles: tuple[str, ...],
    targets: tuple[SwiftPackageTarget, ...] = SWIFT_PACKAGE_TARGETS,
    repo_root: Path = REPO_ROOT,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)

    version_token = _safe_token(version)
    manifest_path = output_dir / f"fxai-swift-{version_token}.manifest.json"
    sums_path = output_dir / f"fxai-swift-{version_token}.SHA256SUMS"
    zip_path = output_dir / f"fxai-swift-{version_token}.zip"
    zip_sums_path = output_dir / f"{zip_path.name}.sha256"

    copied: list[tuple[SwiftPackageTarget, Path]] = []
    for target in targets:
        copied.append((target, _copy_release_artifact(root, output_dir, target)))

    entries = [
        _entry_for_artifact(
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
        "artifact_type": "fxai_swift_artifacts",
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
            "toolchain_profile": TOOLCHAIN_PROFILE,
            "toolchain_config": (str(FXAI_CONFIG_PATH) if FXAI_CONFIG_PATH else ""),
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


def cmd_package_swift_release(args) -> int:
    version = str(args.version or "").strip()
    if not version:
        raise SystemExit("--version is required")

    targets = SWIFT_PACKAGE_TARGETS
    if not bool(getattr(args, "skip_build", False)):
        for target in targets:
            rc = compile_swift_package(target.package)
            if rc != 0:
                return rc

    output_dir = Path(args.output_dir) if args.output_dir else ROOT / "Artifacts/Release"
    release_profile = str(getattr(args, "release_profile", "production") or "production").strip().lower()
    compatible_profiles = _parse_profiles(getattr(args, "compatible_profiles", "research,production"))
    manifest = write_swift_release_bundle(
        root=ROOT,
        output_dir=output_dir,
        version=version,
        release_profile=release_profile,
        compatible_profiles=compatible_profiles,
        targets=targets,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0
