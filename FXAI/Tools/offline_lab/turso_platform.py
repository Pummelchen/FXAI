from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from .common import (
    RESEARCH_DIR,
    OfflineLabError,
    ensure_dir,
    json_compact,
    now_unix,
    query_all,
    resolve_turso_config,
    safe_token,
    sha256_text,
)
from .db_backend import TursoConfig


_TURSO_API_BASE = "https://api.turso.tech"


@dataclass(frozen=True)
class TursoBranchResult:
    source_database: str
    target_database: str
    branch_kind: str
    source_timestamp: str
    group_name: str
    location_name: str
    sync_url: str
    auth_token: str
    env_artifact_path: Path
    created_at: int


def _turso_cli_prefix(config: TursoConfig) -> list[str]:
    argv = ["turso"]
    if config.cli_config_path:
        argv.extend(["--config-path", str(config.cli_config_path)])
    return argv


def _run_turso_cli(config: TursoConfig, args: list[str]) -> str:
    argv = _turso_cli_prefix(config) + args
    proc = subprocess.run(
        argv,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip()
        raise OfflineLabError(f"Turso CLI failed ({' '.join(args)}): {detail}")
    return (proc.stdout or "").strip()


def _platform_request(config: TursoConfig,
                      path: str,
                      params: dict[str, object] | None = None) -> dict[str, Any]:
    if not config.platform_api_enabled:
        raise OfflineLabError(
            "Turso Platform API requires TURSO_ORGANIZATION and TURSO_API_TOKEN"
        )
    query = urlencode({key: value for key, value in (params or {}).items() if value not in (None, "")})
    url = f"{_TURSO_API_BASE}{path}"
    if query:
        url = f"{url}?{query}"
    request = Request(
        url,
        headers={
            "Authorization": f"Bearer {config.api_token}",
            "Accept": "application/json",
        },
    )
    try:
        with urlopen(request, timeout=30.0) as response:
            payload = response.read().decode("utf-8", errors="replace")
    except Exception as exc:
        raise OfflineLabError(f"Turso Platform API request failed: {path}: {exc}") from exc
    try:
        return json.loads(payload or "{}")
    except json.JSONDecodeError as exc:
        raise OfflineLabError(f"Turso Platform API returned invalid JSON for {path}") from exc


def list_platform_groups(config: TursoConfig) -> list[dict[str, Any]]:
    payload = _platform_request(config, f"/v1/organizations/{config.organization_slug}/groups")
    groups = payload.get("groups", payload.get("data", payload))
    if isinstance(groups, list):
        return [dict(item) for item in groups if isinstance(item, dict)]
    return []


def list_platform_databases(config: TursoConfig) -> list[dict[str, Any]]:
    payload = _platform_request(config, f"/v1/organizations/{config.organization_slug}/databases")
    databases = payload.get("databases", payload.get("data", payload))
    if isinstance(databases, list):
        return [dict(item) for item in databases if isinstance(item, dict)]
    return []


def list_platform_audit_logs(config: TursoConfig,
                             limit: int = 50,
                             page: int = 1) -> list[dict[str, Any]]:
    payload = _platform_request(
        config,
        f"/v1/organizations/{config.organization_slug}/audit-logs",
        {"limit": int(limit), "page": int(page)},
    )
    logs = payload.get("audit_logs", payload.get("logs", payload.get("data", payload)))
    if isinstance(logs, list):
        return [dict(item) for item in logs if isinstance(item, dict)]
    return []


def branch_inventory(config: TursoConfig, source_database: str = "") -> list[dict[str, Any]]:
    databases = list_platform_databases(config)
    source_name = (source_database or config.database_name or "").strip()
    out: list[dict[str, Any]] = []
    for item in databases:
        name = str(item.get("name", "") or item.get("dbName", "") or "")
        parent = item.get("parent") or item.get("parent_database") or {}
        parent_name = ""
        if isinstance(parent, dict):
            parent_name = str(parent.get("name", "") or parent.get("dbName", "") or "")
        elif isinstance(parent, str):
            parent_name = str(parent)
        if source_name and parent_name != source_name and name != source_name:
            continue
        out.append(
            {
                "name": name,
                "group": str(item.get("group", "") or item.get("groupName", "") or ""),
                "hostname": str(item.get("hostname", "") or item.get("host", "") or ""),
                "is_branch": bool(parent_name),
                "parent_name": parent_name,
                "raw": dict(item),
            }
        )
    out.sort(key=lambda item: (item["parent_name"], item["name"]))
    return out


def _branch_artifact_path(profile_name: str, target_database: str) -> Path:
    out_dir = RESEARCH_DIR / safe_token(profile_name) / "turso"
    ensure_dir(out_dir)
    return out_dir / f"branch_{safe_token(target_database)}.env"


def _write_branch_env_artifact(profile_name: str,
                               config: TursoConfig,
                               target_database: str,
                               sync_url: str,
                               auth_token: str) -> Path:
    path = _branch_artifact_path(profile_name, target_database)
    lines = [
        f"export TURSO_DATABASE_NAME={target_database}",
        f"export TURSO_DATABASE_URL={sync_url}",
        f"export TURSO_AUTH_TOKEN={auth_token}",
    ]
    if config.organization_slug:
        lines.append(f"export TURSO_ORGANIZATION={config.organization_slug}")
    if config.group_name:
        lines.append(f"export TURSO_GROUP={config.group_name}")
    if config.location_name:
        lines.append(f"export TURSO_LOCATION={config.location_name}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def create_branch_database(config: TursoConfig,
                           source_database: str,
                           target_database: str,
                           profile_name: str,
                           branch_kind: str = "campaign",
                           timestamp: str = "",
                           group_name: str = "",
                           location_name: str = "",
                           token_expiration: str = "7d",
                           read_only_token: bool = False) -> TursoBranchResult:
    if not source_database:
        raise OfflineLabError("source Turso database is required for branching")
    if not target_database:
        raise OfflineLabError("target Turso database is required for branching")
    args = ["db", "branch", source_database, target_database]
    group_value = (group_name or config.group_name or "").strip()
    location_value = (location_name or config.location_name or "").strip()
    if group_value:
        args.extend(["--group", group_value])
    if location_value:
        args.extend(["--location", location_value])
    if timestamp:
        args.extend(["--timestamp", timestamp])
    _run_turso_cli(config, args)
    sync_url = _run_turso_cli(config, ["db", "show", target_database, "--url"]).strip()
    token_args = ["db", "tokens", "create", target_database, "--expiration", token_expiration]
    if read_only_token:
        token_args.append("--read-only")
    auth_token = _run_turso_cli(config, token_args).strip()
    env_artifact_path = _write_branch_env_artifact(profile_name, config, target_database, sync_url, auth_token)
    return TursoBranchResult(
        source_database=source_database,
        target_database=target_database,
        branch_kind=branch_kind,
        source_timestamp=timestamp,
        group_name=group_value,
        location_name=location_value,
        sync_url=sync_url,
        auth_token=auth_token,
        env_artifact_path=env_artifact_path,
        created_at=now_unix(),
    )


def destroy_database(config: TursoConfig, database_name: str) -> None:
    if not database_name:
        raise OfflineLabError("database name is required for Turso destroy")
    _run_turso_cli(config, ["db", "destroy", database_name])


def persist_branch_result(conn,
                          profile_name: str,
                          result: TursoBranchResult) -> dict[str, Any]:
    payload = {
        "source_database": result.source_database,
        "target_database": result.target_database,
        "branch_kind": result.branch_kind,
        "source_timestamp": result.source_timestamp,
        "group_name": result.group_name,
        "location_name": result.location_name,
        "sync_url": result.sync_url,
        "env_artifact_path": str(result.env_artifact_path),
        "created_at": result.created_at,
    }
    conn.execute(
        """
        INSERT INTO turso_branch_runs(profile_name, source_database, target_database, branch_kind,
                                      source_timestamp, group_name, location_name, sync_url,
                                      auth_token_sha256, env_artifact_path, payload_json, status, created_at)
        VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'created', ?)
        ON CONFLICT(target_database) DO UPDATE SET
            profile_name=excluded.profile_name,
            source_database=excluded.source_database,
            branch_kind=excluded.branch_kind,
            source_timestamp=excluded.source_timestamp,
            group_name=excluded.group_name,
            location_name=excluded.location_name,
            sync_url=excluded.sync_url,
            auth_token_sha256=excluded.auth_token_sha256,
            env_artifact_path=excluded.env_artifact_path,
            payload_json=excluded.payload_json,
            status=excluded.status,
            created_at=excluded.created_at
        """,
        (
            profile_name,
            result.source_database,
            result.target_database,
            result.branch_kind,
            result.source_timestamp,
            result.group_name,
            result.location_name,
            result.sync_url,
            sha256_text(result.auth_token),
            str(result.env_artifact_path),
            json_compact(payload),
            result.created_at,
        ),
    )
    return payload


def sync_audit_logs(conn,
                    config: TursoConfig,
                    limit: int = 50,
                    pages: int = 1) -> dict[str, Any]:
    if not config.organization_slug:
        raise OfflineLabError("Turso audit-log sync requires TURSO_ORGANIZATION")
    inserted = 0
    seen = 0
    records: list[dict[str, Any]] = []
    for page in range(1, max(int(pages), 1) + 1):
        rows = list_platform_audit_logs(config, limit=limit, page=page)
        if not rows:
            break
        for row in rows:
            seen += 1
            event_id = str(
                row.get("id", "")
                or row.get("event_id", "")
                or row.get("eventId", "")
                or sha256_text(json_compact(row))
            )
            actor = row.get("actor") or {}
            target = row.get("target") or {}
            payload_json = json.dumps(row, indent=2, sort_keys=True)
            before = query_all(
                conn,
                "SELECT event_id FROM turso_audit_log_events WHERE organization_slug = ? AND event_id = ?",
                (config.organization_slug, event_id),
            )
            conn.execute(
                """
                INSERT INTO turso_audit_log_events(organization_slug, event_id, event_type,
                                                   actor_name, actor_email, target_type, target_name,
                                                   occurred_at, source_page, payload_json, observed_at)
                VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(organization_slug, event_id) DO UPDATE SET
                    event_type=excluded.event_type,
                    actor_name=excluded.actor_name,
                    actor_email=excluded.actor_email,
                    target_type=excluded.target_type,
                    target_name=excluded.target_name,
                    occurred_at=excluded.occurred_at,
                    source_page=excluded.source_page,
                    payload_json=excluded.payload_json,
                    observed_at=excluded.observed_at
                """,
                (
                    config.organization_slug,
                    event_id,
                    str(row.get("type", "") or row.get("event_type", "") or row.get("action", "")),
                    str(actor.get("name", "") or actor.get("displayName", "") or row.get("actor_name", "")),
                    str(actor.get("email", "") or row.get("actor_email", "")),
                    str(target.get("type", "") or row.get("target_type", "")),
                    str(target.get("name", "") or row.get("target_name", "")),
                    str(row.get("created_at", "") or row.get("occurred_at", "") or row.get("timestamp", "")),
                    page,
                    payload_json,
                    now_unix(),
                ),
            )
            if not before:
                inserted += 1
            records.append(
                {
                    "event_id": event_id,
                    "event_type": str(row.get("type", "") or row.get("event_type", "") or row.get("action", "")),
                    "target_name": str(target.get("name", "") or row.get("target_name", "")),
                }
            )
    return {
        "organization": config.organization_slug,
        "seen": seen,
        "inserted": inserted,
        "sample": records[:10],
    }


def resolve_platform_config(db_path: Path) -> TursoConfig:
    return resolve_turso_config(db_path)
