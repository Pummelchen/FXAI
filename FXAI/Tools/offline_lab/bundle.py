from __future__ import annotations

import json
import shutil
from pathlib import Path

from .common import COMMON_PROMOTION_DIR, RESEARCH_DIR, ensure_dir, query_all, safe_token
from .performance import runtime_performance_manifest_path


def _copy_if_exists(src: Path, dst: Path) -> bool:
    if not src.exists() or not src.is_file():
        return False
    ensure_dir(dst.parent)
    shutil.copy2(src, dst)
    return True


def write_minimal_live_bundle(conn, profile_name: str, output_dir: Path | None = None) -> dict[str, object]:
    rows = query_all(
        conn,
        """
        SELECT symbol, payload_json, artifact_path, artifact_sha256, created_at
          FROM live_deployment_profiles
         WHERE profile_name = ?
         ORDER BY symbol
        """,
        (profile_name,),
    )
    bundle_root = output_dir or (RESEARCH_DIR / safe_token(profile_name) / "minimal_live_bundle")
    ensure_dir(bundle_root)

    manifest: dict[str, object] = {
        "profile_name": profile_name,
        "bundle_root": str(bundle_root),
        "symbols": [],
    }
    for row in rows:
        symbol = str(row["symbol"])
        token = safe_token(symbol)
        symbol_dir = bundle_root / token
        ensure_dir(symbol_dir)
        copied: list[str] = []
        for src in [
            COMMON_PROMOTION_DIR / f"fxai_live_deploy_{token}.tsv",
            COMMON_PROMOTION_DIR / f"fxai_student_router_{token}.tsv",
            COMMON_PROMOTION_DIR / f"fxai_attribution_{token}.tsv",
            COMMON_PROMOTION_DIR / f"fxai_world_plan_{token}.tsv",
            COMMON_PROMOTION_DIR / f"fxai_supervisor_service_{token}.tsv",
            COMMON_PROMOTION_DIR / f"fxai_supervisor_command_{token}.tsv",
            runtime_performance_manifest_path(symbol),
        ]:
            dst = symbol_dir / src.name
            if _copy_if_exists(src, dst):
                copied.append(src.name)
        manifest["symbols"].append(
            {
                "symbol": symbol,
                "artifact_path": str(row["artifact_path"]),
                "artifact_sha256": str(row["artifact_sha256"]),
                "created_at": int(row["created_at"]),
                "copied_files": copied,
            }
        )

    manifest_path = bundle_root / "bundle_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return {
        "profile_name": profile_name,
        "bundle_root": str(bundle_root),
        "manifest_path": str(manifest_path),
        "symbol_count": len(manifest["symbols"]),
    }
