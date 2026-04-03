from __future__ import annotations

import json
import time
from html import escape
from pathlib import Path

from .common import (
    COMMON_PROMOTION_DIR,
    DEFAULT_DB,
    RESEARCH_DIR,
    current_lab_versions,
    ensure_dir,
    query_all,
    safe_token,
    turso_environment_status,
)
from .performance import build_symbol_performance_report
from .vector_store import latest_symbol_shadow_neighbors


def _load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _load_tsv_map(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    out: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        parts = line.split("\t", 1)
        if len(parts) >= 2:
            out[parts[0]] = parts[1]
    return out


def build_profile_dashboard(conn, profile_name: str) -> dict[str, object]:
    turso_status = turso_environment_status(DEFAULT_DB)
    symbols = sorted(
        {
            str(row["symbol"])
            for row in query_all(
                conn,
                """
                SELECT symbol FROM live_deployment_profiles WHERE profile_name = ?
                UNION
                SELECT symbol FROM champion_registry WHERE profile_name = ?
                UNION
                SELECT symbol FROM shadow_fleet_observations WHERE profile_name = ?
                ORDER BY symbol
                """,
                (profile_name, profile_name, profile_name),
            )
        }
    )
    champions = query_all(
        conn,
        """
        SELECT symbol, plugin_name, status, champion_score, challenger_score, portfolio_score, reviewed_at
          FROM champion_registry
         WHERE profile_name = ?
         ORDER BY symbol, plugin_name
        """,
        (profile_name,),
    )
    branch_rows = query_all(
        conn,
        """
        SELECT source_database, target_database, branch_kind, source_timestamp, status, created_at
          FROM turso_branch_runs
         WHERE profile_name IN (?, '')
         ORDER BY created_at DESC
         LIMIT 25
        """,
        (profile_name,),
    )
    audit_rows = query_all(
        conn,
        """
        SELECT organization_slug, event_id, event_type, target_name, occurred_at, observed_at
          FROM turso_audit_log_events
         ORDER BY observed_at DESC, id DESC
         LIMIT 25
        """
    )
    live_rows = query_all(
        conn,
        """
        SELECT symbol, payload_json, artifact_path, artifact_sha256, created_at
          FROM live_deployment_profiles
         WHERE profile_name = ?
         ORDER BY symbol
        """,
        (profile_name,),
    )
    deployments = []
    now_unix = int(time.time())
    for row in live_rows:
        payload = json.loads(row["payload_json"] or "{}")
        symbol = str(row["symbol"])
        perf = build_symbol_performance_report(symbol)
        artifact_path = Path(str(row["artifact_path"]))
        live_state = {
            "deployment_tsv": _load_tsv_map(COMMON_PROMOTION_DIR / f"fxai_live_deploy_{safe_token(symbol)}.tsv"),
            "router_tsv": _load_tsv_map(COMMON_PROMOTION_DIR / f"fxai_student_router_{safe_token(symbol)}.tsv"),
            "world_tsv": _load_tsv_map(COMMON_PROMOTION_DIR / f"fxai_world_plan_{safe_token(symbol)}.tsv"),
            "service_tsv": _load_tsv_map(COMMON_PROMOTION_DIR / f"fxai_supervisor_service_{safe_token(symbol)}.tsv"),
            "command_tsv": _load_tsv_map(COMMON_PROMOTION_DIR / f"fxai_supervisor_command_{safe_token(symbol)}.tsv"),
        }
        analog_neighbors = latest_symbol_shadow_neighbors(conn, profile_name, symbol, limit=5)
        artifact_age_sec = max(now_unix - int(row["created_at"]), 0)
        deployments.append(
            {
                "symbol": symbol,
                "artifact_path": str(artifact_path),
                "created_at": int(row["created_at"]),
                "payload": payload,
                "performance": perf,
                "live_state": live_state,
                "analog_neighbors": analog_neighbors,
                "artifact_health": {
                    "artifact_exists": artifact_path.exists(),
                    "artifact_age_sec": artifact_age_sec,
                    "stale_artifact": artifact_age_sec > 86400,
                    "missing_deployment": not bool(live_state["deployment_tsv"]),
                    "missing_router": not bool(live_state["router_tsv"]),
                    "missing_world_plan": not bool(live_state["world_tsv"]),
                    "missing_supervisor_service": not bool(live_state["service_tsv"]),
                    "missing_supervisor_command": not bool(live_state["command_tsv"]),
                    "performance_failures": list(perf.get("budget_failures", [])),
                    "artifact_size_failures": list(perf.get("artifact_report", {}).get("budget_failures", [])),
                },
            }
        )
    dashboard = {
        "profile_name": profile_name,
        "versions": current_lab_versions(conn),
        "turso": {
            "environment": turso_status,
            "branches": branch_rows,
            "recent_audit_logs": audit_rows,
        },
        "symbols": symbols,
        "champions": champions,
        "deployments": deployments,
        "source_of_truth": {
            "turso_libsql": "authoritative research and promotion state",
            "file_common_promotions": "authoritative MT5 runtime consumption layer",
            "repo_source": "versioned source and tooling only",
            "generated_outputs": "must be rebuilt from Turso/libSQL state or runtime artifacts, not edited manually",
        },
    }
    return dashboard


def write_profile_dashboard(conn, profile_name: str) -> dict[str, object]:
    out_dir = RESEARCH_DIR / safe_token(profile_name)
    ensure_dir(out_dir)
    payload = build_profile_dashboard(conn, profile_name)
    json_path = out_dir / "operator_dashboard.json"
    md_path = out_dir / "operator_dashboard.md"
    html_path = out_dir / "operator_dashboard.html"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    md_lines = [f"# FXAI Operator Dashboard: {profile_name}", ""]
    md_lines.append("## Turso")
    md_lines.append(f"- backend: {payload['turso']['environment'].get('backend', '')}")
    md_lines.append(f"- sync_mode: {payload['turso']['environment'].get('sync_mode', '')}")
    md_lines.append(f"- encryption_enabled: {payload['turso']['environment'].get('encryption_enabled', False)}")
    md_lines.append(f"- platform_api_enabled: {payload['turso']['environment'].get('platform_api_enabled', False)}")
    md_lines.append(f"- tracked_branches: {len(payload['turso']['branches'])}")
    md_lines.append(f"- recent_audit_events: {len(payload['turso']['recent_audit_logs'])}")
    md_lines.append("")
    for item in payload["deployments"]:
        md_lines.append(f"## {item['symbol']}")
        md_lines.append(f"- performance_failures: {len(item['performance'].get('budget_failures', []))}")
        md_lines.append(f"- deployment_missing: {item['artifact_health']['missing_deployment']}")
        md_lines.append(f"- router_missing: {item['artifact_health']['missing_router']}")
        md_lines.append(f"- world_plan_missing: {item['artifact_health']['missing_world_plan']}")
        md_lines.append(f"- supervisor_missing: {item['artifact_health']['missing_supervisor_service'] or item['artifact_health']['missing_supervisor_command']}")
        md_lines.append(f"- stale_artifact: {item['artifact_health']['stale_artifact']}")
        md_lines.append(f"- artifact_size_failures: {len(item['artifact_health']['artifact_size_failures'])}")
        md_lines.append(f"- analog_neighbors: {len(item.get('analog_neighbors', []))}")
        md_lines.append("")
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    body = [
        "<html><head><meta charset='utf-8'><title>FXAI Operator Dashboard</title>",
        "<style>body{font-family:Arial,sans-serif;margin:24px}table{border-collapse:collapse;width:100%}th,td{border:1px solid #ddd;padding:8px}th{background:#f3f3f3}.bad{color:#b00020;font-weight:700}.ok{color:#006b2d;font-weight:700}</style>",
        "</head><body>",
        f"<h1>FXAI Operator Dashboard: {escape(profile_name)}</h1>",
        "<table><thead><tr><th>Symbol</th><th>Runtime Mode</th><th>Telemetry</th><th>Shadow</th><th>Perf Failures</th><th>Artifact Size Failures</th><th>Missing Artifacts</th></tr></thead><tbody>",
    ]
    for item in payload["deployments"]:
        pl = item["payload"]
        health = item["artifact_health"]
        missing = []
        for key, flag in [("deploy", health["missing_deployment"]), ("router", health["missing_router"]), ("world", health["missing_world_plan"])]:
            if flag:
                missing.append(key)
        body.append(
            "<tr>"
            f"<td>{escape(item['symbol'])}</td>"
            f"<td>{escape(str(pl.get('runtime_mode', 'research')))}</td>"
            f"<td>{escape(str(pl.get('telemetry_level', 'full')))}</td>"
            f"<td>{'on' if int(pl.get('shadow_enabled', 1) or 0) else 'off'}</td>"
            f"<td class=\"{'bad' if health['performance_failures'] else 'ok'}\">{len(health['performance_failures'])}</td>"
            f"<td class=\"{'bad' if health['artifact_size_failures'] else 'ok'}\">{len(health['artifact_size_failures'])}</td>"
            f"<td class=\"{'bad' if missing else 'ok'}\">{escape(','.join(missing) if missing else 'none')}</td>"
            "</tr>"
        )
    body.append("</tbody></table></body></html>")
    html_path.write_text("\n".join(body), encoding="utf-8")
    return {
        "json_path": str(json_path),
        "md_path": str(md_path),
        "html_path": str(html_path),
        "profile_name": profile_name,
        "symbols": payload["symbols"],
    }


def live_state_snapshot(profile_name: str, symbol: str) -> dict[str, object]:
    symbol_token = safe_token(symbol)
    return {
        "profile_name": profile_name,
        "symbol": symbol,
        "deployment": _load_tsv_map(COMMON_PROMOTION_DIR / f"fxai_live_deploy_{symbol_token}.tsv"),
        "router": _load_tsv_map(COMMON_PROMOTION_DIR / f"fxai_student_router_{symbol_token}.tsv"),
        "attribution": _load_tsv_map(COMMON_PROMOTION_DIR / f"fxai_attribution_{symbol_token}.tsv"),
        "world_plan": _load_tsv_map(COMMON_PROMOTION_DIR / f"fxai_world_plan_{symbol_token}.tsv"),
        "supervisor_service": _load_tsv_map(COMMON_PROMOTION_DIR / f"fxai_supervisor_service_{symbol_token}.tsv"),
        "supervisor_command": _load_tsv_map(COMMON_PROMOTION_DIR / f"fxai_supervisor_command_{symbol_token}.tsv"),
        "portfolio_supervisor": _load_tsv_map(COMMON_PROMOTION_DIR / "fxai_portfolio_supervisor.tsv"),
        "performance": build_symbol_performance_report(symbol),
    }
