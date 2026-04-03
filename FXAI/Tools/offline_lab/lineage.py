from __future__ import annotations

import json
from html import escape
from pathlib import Path

from .common import RESEARCH_DIR, ensure_dir, safe_token
from .dashboard import live_state_snapshot


def build_lineage_report(conn, profile_name: str, symbol: str = "") -> dict[str, object]:
    params: list[object] = [profile_name]
    symbol_clause = ""
    if symbol:
        symbol_clause = " AND symbol = ?"
        params.append(symbol)

    champion_rows = conn.execute(
        f"""
        SELECT profile_name, symbol, plugin_name, family_id, status, champion_score, challenger_score,
               portfolio_score, champion_best_config_id, promoted_at, reviewed_at, notes
          FROM champion_registry
         WHERE profile_name = ?{symbol_clause}
         ORDER BY symbol, plugin_name
        """,
        params,
    ).fetchall()

    deploy_rows = conn.execute(
        f"""
        SELECT symbol, payload_json, artifact_path, artifact_sha256, created_at
          FROM live_deployment_profiles
         WHERE profile_name = ?{symbol_clause}
         ORDER BY symbol
        """,
        params,
    ).fetchall()

    lineage_rows = conn.execute(
        f"""
        SELECT symbol, plugin_name, family_id, source_run_id, best_config_id, relation, lineage_hash,
               payload_json, created_at
          FROM config_lineage
         WHERE profile_name = ?{symbol_clause}
         ORDER BY symbol, plugin_name, created_at DESC
        """,
        params,
    ).fetchall()

    champion_by_symbol: dict[str, list[dict[str, object]]] = {}
    for row in champion_rows:
        champion_by_symbol.setdefault(str(row["symbol"]), []).append(dict(row))

    lineage_by_symbol: dict[str, list[dict[str, object]]] = {}
    for row in lineage_rows:
        item = dict(row)
        try:
            item["payload_json"] = json.loads(str(item.get("payload_json") or "{}"))
        except Exception:
            item["payload_json"] = {}
        lineage_by_symbol.setdefault(str(row["symbol"]), []).append(item)

    deployments = []
    for row in deploy_rows:
        symbol_name = str(row["symbol"])
        payload = json.loads(row["payload_json"] or "{}")
        live_state = live_state_snapshot(profile_name, symbol_name)
        deployments.append(
            {
                "symbol": symbol_name,
                "artifact_path": str(row["artifact_path"]),
                "artifact_sha256": str(row["artifact_sha256"]),
                "created_at": int(row["created_at"]),
                "payload": payload,
                "champions": champion_by_symbol.get(symbol_name, []),
                "lineage": lineage_by_symbol.get(symbol_name, []),
                "live_state": live_state,
            }
        )

    return {
        "profile_name": profile_name,
        "symbol_filter": symbol,
        "symbols": [item["symbol"] for item in deployments],
        "deployments": deployments,
    }


def write_lineage_report(conn, profile_name: str, symbol: str = "") -> dict[str, object]:
    payload = build_lineage_report(conn, profile_name, symbol)
    suffix = safe_token(symbol) if symbol else "all"
    out_dir = RESEARCH_DIR / safe_token(profile_name)
    ensure_dir(out_dir)
    json_path = out_dir / f"lineage_{suffix}.json"
    md_path = out_dir / f"lineage_{suffix}.md"
    html_path = out_dir / f"lineage_{suffix}.html"

    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    md_lines = [f"# FXAI Lineage: {profile_name}", ""]
    for item in payload["deployments"]:
        md_lines.append(f"## {item['symbol']}")
        md_lines.append(f"- artifact: `{item['artifact_path']}`")
        md_lines.append(f"- runtime_mode: `{item['payload'].get('runtime_mode', 'research')}`")
        md_lines.append(f"- promotion_tier: `{item['payload'].get('promotion_tier', 'experimental')}`")
        md_lines.append(f"- loaded_router: `{bool(item['live_state'].get('router'))}`")
        md_lines.append(f"- loaded_supervisor_service: `{bool(item['live_state'].get('supervisor_service'))}`")
        md_lines.append(f"- loaded_supervisor_command: `{bool(item['live_state'].get('supervisor_command'))}`")
        champions = item["champions"]
        if champions:
            for champion in champions:
                md_lines.append(
                    f"- champion: `{champion['plugin_name']}` status=`{champion['status']}` "
                    f"champion_score=`{champion['champion_score']}` portfolio_score=`{champion['portfolio_score']}`"
                )
        else:
            md_lines.append("- champion: none")
        if item["lineage"]:
            for lineage in item["lineage"][:10]:
                md_lines.append(
                    f"- lineage: relation=`{lineage['relation']}` plugin=`{lineage['plugin_name']}` "
                    f"family=`{lineage['family_id']}` best_config_id=`{lineage['best_config_id']}` "
                    f"source_run_id=`{lineage['source_run_id']}`"
                )
        else:
            md_lines.append("- lineage: none")
        md_lines.append("")
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    html_lines = [
        "<html><head><meta charset='utf-8'><title>FXAI Lineage</title>",
        "<style>body{font-family:Arial,sans-serif;margin:24px}table{border-collapse:collapse;width:100%}th,td{border:1px solid #ddd;padding:8px}th{background:#f3f3f3}</style>",
        "</head><body>",
        f"<h1>FXAI Lineage: {escape(profile_name)}</h1>",
    ]
    for item in payload["deployments"]:
        html_lines.append(f"<h2>{escape(item['symbol'])}</h2>")
        html_lines.append("<table><thead><tr><th>Kind</th><th>Plugin</th><th>Status</th><th>Details</th></tr></thead><tbody>")
        for champion in item["champions"]:
            html_lines.append(
                "<tr>"
                "<td>champion</td>"
                f"<td>{escape(str(champion['plugin_name']))}</td>"
                f"<td>{escape(str(champion['status']))}</td>"
                f"<td>champion_score={escape(str(champion['champion_score']))}, portfolio_score={escape(str(champion['portfolio_score']))}</td>"
                "</tr>"
            )
        for lineage in item["lineage"][:20]:
            html_lines.append(
                "<tr>"
                "<td>lineage</td>"
                f"<td>{escape(str(lineage['plugin_name']))}</td>"
                f"<td>{escape(str(lineage['relation']))}</td>"
                f"<td>best_config_id={escape(str(lineage['best_config_id']))}, source_run_id={escape(str(lineage['source_run_id']))}</td>"
                "</tr>"
            )
        html_lines.append("</tbody></table>")
    html_lines.append("</body></html>")
    html_path.write_text("\n".join(html_lines), encoding="utf-8")

    return {
        "profile_name": profile_name,
        "symbol": symbol,
        "json_path": str(json_path),
        "md_path": str(md_path),
        "html_path": str(html_path),
        "symbols": payload["symbols"],
    }
