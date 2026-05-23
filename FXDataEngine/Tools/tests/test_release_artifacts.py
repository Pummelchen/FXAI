from __future__ import annotations

import json
import zipfile
from pathlib import Path

from testlab.release_artifacts import SWIFT_PACKAGE_TARGETS, write_swift_release_bundle


def _write_fixture_artifact(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def _write_fixture_source(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("// fixture source\n", encoding="utf-8")


def test_swift_release_bundle_contains_hashes_and_compatibility(tmp_path: Path):
    root = tmp_path / "FXAI"
    for index, target in enumerate(SWIFT_PACKAGE_TARGETS):
        _write_fixture_source(root / target.source)
        _write_fixture_artifact(root / target.build_artifact, f"swift-artifact-{index}".encode("utf-8"))

    output_dir = tmp_path / "release"
    manifest = write_swift_release_bundle(
        root=root,
        output_dir=output_dir,
        version="v0.7.0-test",
        release_profile="production",
        compatible_profiles=("research", "production"),
        repo_root=tmp_path,
    )

    bundle = manifest["bundle"]
    assert (output_dir / str(bundle["zip"])).exists()
    assert (output_dir / str(bundle["zip_sha256_file"])).exists()
    assert (output_dir / str(bundle["manifest"])).exists()
    assert (output_dir / str(bundle["sha256s"])).exists()

    persisted = json.loads((output_dir / str(bundle["manifest"])).read_text(encoding="utf-8"))
    assert persisted["artifact_type"] == "fxai_swift_artifacts"
    assert persisted["release_profile"] == "production"
    assert persisted["compatible_profiles"] == ["research", "production"]
    assert len(persisted["artifacts"]) == len(SWIFT_PACKAGE_TARGETS)
    assert all(item["release_sha256"] for item in persisted["artifacts"])
    assert all(item["source_sha256"] for item in persisted["artifacts"])
    by_role = {str(item["role"]): item for item in persisted["artifacts"]}
    assert by_role["data_engine"]["compatible_profiles"] == ["research", "production"]
    assert by_role["plugins"]["compatible_profiles"] == ["research"]
    assert by_role["database"]["compatible_profiles"] == ["research", "production"]

    sums = (output_dir / str(bundle["sha256s"])).read_text(encoding="utf-8")
    assert "data_engine-FXDataEngineCLI" in sums
    assert "plugins-FXAIPlugins.swiftmodule" in sums
    assert "database-FXDatabase" in sums
    assert str(bundle["zip"]) in (output_dir / str(bundle["zip_sha256_file"])).read_text(encoding="utf-8")
    with zipfile.ZipFile(output_dir / str(bundle["zip"]), "r") as archive:
        names = set(archive.namelist())
    assert "data_engine-FXDataEngineCLI" in names
    assert "plugins-FXAIPlugins.swiftmodule" in names
    assert "database-FXDatabase" in names
    assert str(bundle["manifest"]) in names
    assert str(bundle["sha256s"]) in names


def test_fxai_gitignore_blocks_swift_build_products():
    root = Path(__file__).resolve().parents[3]
    text = (root / ".gitignore").read_text(encoding="utf-8")
    assert ".build/" in text
