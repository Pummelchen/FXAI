from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import libsql

from .common import *
from .promotion import SERIOUS_SCENARIOS


DRIFT_RETRAINING_SCHEMA_VERSION = 1
DRIFT_RETRAINING_REPORT_VERSION = 1
DRIFT_RETRAINING_DIR = OFFLINE_DIR / "DriftRetraining"
DRIFT_RETRAINING_REPORT_DIR = DRIFT_RETRAINING_DIR / "Reports"
DRIFT_RETRAINING_STATUS_PATH = DRIFT_RETRAINING_DIR / "drift_retraining_status.json"
DRIFT_RETRAINING_REPORT_PATH = DRIFT_RETRAINING_REPORT_DIR / "drift_retraining_report.json"
DRIFT_RETRAINING_HISTORY_PATH = DRIFT_RETRAINING_DIR / "drift_retraining_history.ndjson"
DRIFT_RETRAINING_ALERTS_PATH = DRIFT_RETRAINING_DIR / "drift_retraining_alerts.jsonl"

DEFAULT_TRIGGER_ACTIONS = {"DOWNWEIGHT", "RESTRICT", "SHADOW_ONLY", "DEMOTE", "DISABLE", "ROLLBACK"}
DEFAULT_TRIGGER_STATES = {"DEGRADED", "SHADOW_ONLY", "DEMOTED", "DISABLED"}
DEFAULT_REQUIRED_GATES = [
    "walkforward_score",
    "recent_score",
    "adversarial_score",
    "macro_event_score",
    "calibration_error",
    "issue_count",
    "financial_utility_loss",
    "promotion_review",
]


def ensure_drift_retraining_dirs() -> dict[str, Path]:
    for path in (DRIFT_RETRAINING_DIR, DRIFT_RETRAINING_REPORT_DIR):
        path.mkdir(parents=True, exist_ok=True)
    return {
        "drift_retraining_dir": DRIFT_RETRAINING_DIR,
        "report_dir": DRIFT_RETRAINING_REPORT_DIR,
    }


def _safe_json(raw: Any, default: Any) -> Any:
    if isinstance(raw, (dict, list)):
        return raw
    try:
        payload = json.loads(str(raw or ""))
    except Exception:
        return default
    return payload


def _safe_float(raw: Any, default: float = 0.0) -> float:
    try:
        return float(raw)
    except Exception:
        return float(default)


def _safe_int(raw: Any, default: int = 0) -> int:
    try:
        return int(float(raw))
    except Exception:
        return int(default)


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(portableize_payload_paths(payload), sort_keys=True) + "\n")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(portableize_payload_paths(payload), indent=2, sort_keys=True), encoding="utf-8")


def _parse_reason_codes(row: dict[str, Any], payload: dict[str, Any]) -> list[str]:
    raw = _safe_json(row.get("reason_codes_json", "[]"), [])
    if not isinstance(raw, list):
        raw = payload.get("reason_codes", [])
    reasons: list[str] = []
    for item in raw if isinstance(raw, list) else []:
        token = str(item or "").strip().upper()
        if token and token not in reasons:
            reasons.append(token)
    return reasons


def _support_from_payload(payload: dict[str, Any]) -> dict[str, int | str]:
    support = payload.get("support", {})
    if not isinstance(support, dict):
        support = {}
    return {
        "sample_count_recent": _safe_int(support.get("sample_count_recent", 0), 0),
        "sample_count_reference": _safe_int(support.get("sample_count_reference", 0), 0),
        "reference_scope": str(support.get("reference_scope", "") or ""),
    }


def _request_key(row: dict[str, Any], payload: dict[str, Any], reason_codes: list[str]) -> str:
    metadata = payload.get("metadata", {}) if isinstance(payload.get("metadata", {}), dict) else {}
    basis = {
        "profile_name": str(row["profile_name"]),
        "symbol": str(row["symbol"]),
        "plugin_name": str(row["plugin_name"]),
        "family_id": _safe_int(row.get("family_id", 11), 11),
        "trigger_action": str(row.get("action_recommendation", "") or "").upper(),
        "trigger_state": str(row.get("governance_state", "") or "").upper(),
        "health_state": str(row.get("health_state", "") or "").upper(),
        "reason_codes": sorted(reason_codes),
        "reference_window": metadata.get("reference_window", {}),
        "live_window": metadata.get("live_window", {}),
        "policy_version": _safe_int(row.get("policy_version", 1), 1),
    }
    return sha256_text(json_compact(basis))


def _priority(trigger_action: str, trigger_state: str, risk_score: float) -> int:
    action = trigger_action.upper()
    state = trigger_state.upper()
    if action == "DISABLE" or state == "DISABLED" or risk_score >= 0.88:
        return 100
    if action in {"DEMOTE", "ROLLBACK"} or state in {"DEMOTED", "SHADOW_ONLY"} or risk_score >= 0.74:
        return 80
    if state == "DEGRADED" or risk_score >= 0.56:
        return 60
    return 40


def _requested_months_list(data_days: int, explicit_months: str) -> str:
    tokens = parse_csv_tokens(explicit_months)
    if tokens:
        months = []
        for token in tokens:
            value = _safe_int(token, 0)
            if value > 0 and value not in months:
                months.append(value)
        if months:
            return ",".join(str(value) for value in months)
    months = max(1, (max(int(data_days), 1) + 29) // 30)
    return str(months)


def _should_queue(row: dict[str, Any],
                  payload: dict[str, Any],
                  *,
                  min_risk_score: float,
                  include_low_support: bool) -> tuple[bool, str]:
    action = str(row.get("action_recommendation", "") or "").upper()
    state = str(row.get("governance_state", "") or "").upper()
    health = str(row.get("health_state", "") or "").upper()
    risk = _safe_float(row.get("aggregate_risk_score", 0.0), 0.0)
    quality_flags = payload.get("quality_flags", {}) if isinstance(payload.get("quality_flags", {}), dict) else {}
    if bool(quality_flags.get("low_support", False)) and not include_low_support:
        return False, "low_support"
    if action in DEFAULT_TRIGGER_ACTIONS and risk >= min_risk_score:
        return True, "action_threshold"
    if state in DEFAULT_TRIGGER_STATES and risk >= min_risk_score:
        return True, "state_threshold"
    if health in DEFAULT_TRIGGER_STATES and risk >= min_risk_score:
        return True, "health_threshold"
    return False, "below_threshold"


def _load_governance_rows(conn: libsql.Connection, profile_name: str, symbol: str = "", plugin_name: str = "") -> list[dict[str, Any]]:
    clauses = ["profile_name = ?"]
    params: list[Any] = [profile_name]
    if symbol:
        clauses.append("symbol = ?")
        params.append(symbol.upper())
    if plugin_name:
        clauses.append("plugin_name = ?")
        params.append(plugin_name)
    return query_all(
        conn,
        f"""
        SELECT *
          FROM plugin_governance_states
         WHERE {' AND '.join(clauses)}
         ORDER BY aggregate_risk_score DESC, symbol ASC, plugin_name ASC
        """,
        params,
    )


def _emit_webhook(payload: dict[str, Any]) -> dict[str, Any]:
    if os.getenv("FXAI_DRIFT_RETRAINING_ENABLE_WEBHOOKS", "").strip() != "1":
        return {"enabled": False}
    text = str(payload.get("message", "FXAI drift retraining alert"))
    generic_url = os.getenv("FXAI_DRIFT_RETRAINING_WEBHOOK_URL", "").strip()
    telegram_token = os.getenv("FXAI_DRIFT_RETRAINING_TELEGRAM_BOT_TOKEN", "").strip()
    telegram_chat = os.getenv("FXAI_DRIFT_RETRAINING_TELEGRAM_CHAT_ID", "").strip()
    targets: list[tuple[str, str, dict[str, Any]]] = []
    if generic_url:
        targets.append(("generic", generic_url, {"content": text, "text": text, "payload": payload}))
    if telegram_token and telegram_chat:
        url = f"https://api.telegram.org/bot{telegram_token}/sendMessage"
        targets.append(("telegram", url, {"chat_id": telegram_chat, "text": text}))
    deliveries = []
    for kind, url, body in targets:
        request = urllib.request.Request(
            url,
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=5.0) as response:
                deliveries.append({"kind": kind, "ok": True, "status": int(getattr(response, "status", 0) or 0)})
        except (urllib.error.URLError, OSError, TimeoutError) as exc:
            deliveries.append({"kind": kind, "ok": False, "error": str(exc)})
    return {"enabled": True, "deliveries": deliveries}


def _emit_alert(alert: dict[str, Any]) -> dict[str, Any]:
    ensure_drift_retraining_dirs()
    alert_payload = {
        "schema_version": DRIFT_RETRAINING_SCHEMA_VERSION,
        "level": "alert",
        "component": "drift_retraining",
        "created_at": now_unix(),
        **alert,
    }
    _append_jsonl(DRIFT_RETRAINING_ALERTS_PATH, alert_payload)
    external_alert_path = os.getenv("FXAI_ALERT_JSONL_PATH", "").strip()
    if external_alert_path:
        _append_jsonl(Path(external_alert_path), alert_payload)
    delivery = _emit_webhook(alert_payload)
    return {
        "local_alert_path": str(DRIFT_RETRAINING_ALERTS_PATH),
        "external_alert_path": external_alert_path,
        "webhook": delivery,
    }


def queue_drift_retraining_requests(conn: libsql.Connection, args) -> dict[str, Any]:
    ensure_drift_retraining_dirs()
    profile_name = str(getattr(args, "profile", "") or "continuous")
    symbol = str(getattr(args, "symbol", "") or "").strip().upper()
    plugin_name = str(getattr(args, "plugin", "") or "").strip()
    min_risk_score = _safe_float(getattr(args, "min_risk_score", 0.56), 0.56)
    include_low_support = bool(getattr(args, "include_low_support", False))
    data_days = max(_safe_int(getattr(args, "data_days", 90), 90), 1)
    months_list = _requested_months_list(data_days, str(getattr(args, "months_list", "") or ""))
    wf_train_years = str(getattr(args, "wf_train_years", "") or "1,2,3,5")
    scenario_list = str(getattr(args, "scenario_list", "") or SERIOUS_SCENARIOS)
    required_gates = list(DEFAULT_REQUIRED_GATES)
    rows = _load_governance_rows(conn, profile_name, symbol, plugin_name)
    now = now_unix()
    queued: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    alert_deliveries: list[dict[str, Any]] = []

    for row in rows:
        payload = _safe_json(row.get("payload_json", "{}"), {})
        if not isinstance(payload, dict):
            payload = {}
        should_queue, skip_reason = _should_queue(
            row,
            payload,
            min_risk_score=min_risk_score,
            include_low_support=include_low_support,
        )
        if not should_queue:
            skipped.append({
                "symbol": str(row.get("symbol", "")),
                "plugin_name": str(row.get("plugin_name", "")),
                "reason": skip_reason,
                "aggregate_risk_score": round(_safe_float(row.get("aggregate_risk_score", 0.0), 0.0), 6),
            })
            continue

        reason_codes = _parse_reason_codes(row, payload)
        key = _request_key(row, payload, reason_codes)
        existing = query_one(conn, "SELECT id, status FROM drift_retraining_requests WHERE request_key = ?", (key,))
        trigger_action = str(row.get("action_recommendation", "") or "").upper()
        trigger_state = str(row.get("governance_state", "") or "").upper()
        risk_score = _safe_float(row.get("aggregate_risk_score", 0.0), 0.0)
        support = _support_from_payload(payload)
        source_payload = {
            "governance_state": {
                "profile_name": profile_name,
                "symbol": str(row.get("symbol", "")),
                "plugin_name": str(row.get("plugin_name", "")),
                "family_id": _safe_int(row.get("family_id", 11), 11),
                "health_state": str(row.get("health_state", "")),
                "governance_state": trigger_state,
                "action_recommendation": trigger_action,
                "aggregate_risk_score": risk_score,
                "policy_version": _safe_int(row.get("policy_version", 1), 1),
                "updated_at": _safe_int(row.get("updated_at", 0), 0),
            },
            "drift_scores": payload.get("drift_scores", {}),
            "drift_details": payload.get("drift_details", {}),
            "support": support,
            "quality_flags": payload.get("quality_flags", {}),
            "reason_codes": reason_codes,
            "required_gates": required_gates,
            "promotion_allowed": False,
            "manual_approval_required": True,
        }
        source_json = json.dumps(source_payload, sort_keys=True)
        conn.execute(
            """
            INSERT INTO drift_retraining_requests(
                profile_name, symbol, plugin_name, family_id, request_key, trigger_source,
                trigger_action, trigger_state, aggregate_risk_score, priority, status,
                requested_data_days, requested_months_list, requested_wf_train_years, scenario_list,
                required_gates_json, reason_codes_json, source_payload_json, execution_payload_json,
                alert_path, report_path, created_at, updated_at, queued_at
            )
            VALUES(?, ?, ?, ?, ?, 'drift_governance', ?, ?, ?, ?, 'queued', ?, ?, ?, ?, ?, ?, ?, '{}', ?, ?, ?, ?, ?)
            ON CONFLICT(request_key) DO UPDATE SET
                aggregate_risk_score=excluded.aggregate_risk_score,
                priority=excluded.priority,
                requested_data_days=excluded.requested_data_days,
                requested_months_list=excluded.requested_months_list,
                requested_wf_train_years=excluded.requested_wf_train_years,
                scenario_list=excluded.scenario_list,
                required_gates_json=excluded.required_gates_json,
                reason_codes_json=excluded.reason_codes_json,
                source_payload_json=excluded.source_payload_json,
                alert_path=excluded.alert_path,
                report_path=excluded.report_path,
                updated_at=excluded.updated_at
            """,
            (
                profile_name,
                str(row.get("symbol", "")),
                str(row.get("plugin_name", "")),
                _safe_int(row.get("family_id", 11), 11),
                key,
                trigger_action,
                trigger_state,
                risk_score,
                _priority(trigger_action, trigger_state, risk_score),
                data_days,
                months_list,
                wf_train_years,
                scenario_list,
                json.dumps(required_gates, sort_keys=True),
                json.dumps(reason_codes, sort_keys=True),
                source_json,
                str(DRIFT_RETRAINING_ALERTS_PATH),
                str(DRIFT_RETRAINING_REPORT_PATH),
                now,
                now,
                now,
            ),
        )
        item = {
            "request_key": key,
            "symbol": str(row.get("symbol", "")),
            "plugin_name": str(row.get("plugin_name", "")),
            "trigger_action": trigger_action,
            "trigger_state": trigger_state,
            "aggregate_risk_score": round(risk_score, 6),
            "priority": _priority(trigger_action, trigger_state, risk_score),
            "status": "updated" if existing else "queued",
            "reason_codes": reason_codes,
            "requested_data_days": data_days,
            "requested_months_list": months_list,
            "requested_wf_train_years": wf_train_years,
        }
        queued.append(item)
        if existing is None:
            alert = {
                "message": (
                    f"FXAI drift retraining queued: {item['symbol']} {item['plugin_name']} "
                    f"risk={item['aggregate_risk_score']:.3f} action={trigger_action or 'NONE'} state={trigger_state or 'UNKNOWN'}"
                ),
                "request": item,
            }
            alert_deliveries.append(_emit_alert(alert))

    commit_db(conn)
    report = build_drift_retraining_report(conn, profile_name)
    status = {
        "schema_version": DRIFT_RETRAINING_SCHEMA_VERSION,
        "generated_at": now,
        "profile_name": profile_name,
        "queued_or_updated_count": len(queued),
        "new_alert_count": len(alert_deliveries),
        "skipped_count": len(skipped),
        "queued": queued,
        "skipped": skipped,
        "artifacts": {
            "status_path": str(DRIFT_RETRAINING_STATUS_PATH),
            "report_path": str(DRIFT_RETRAINING_REPORT_PATH),
            "history_path": str(DRIFT_RETRAINING_HISTORY_PATH),
            "alerts_path": str(DRIFT_RETRAINING_ALERTS_PATH),
        },
        "alert_deliveries": alert_deliveries,
        "report_summary": {
            "request_count": int(report.get("request_count", 0)),
            "queued_count": int(report.get("status_counts", {}).get("queued", 0)),
        },
    }
    _write_json(DRIFT_RETRAINING_STATUS_PATH, status)
    _append_jsonl(DRIFT_RETRAINING_HISTORY_PATH, {"record_type": "queue", **status})
    return status


def _request_to_report_item(row: dict[str, Any]) -> dict[str, Any]:
    source_payload = _safe_json(row.get("source_payload_json", "{}"), {})
    execution_payload = _safe_json(row.get("execution_payload_json", "{}"), {})
    return {
        "id": _safe_int(row.get("id", 0), 0),
        "request_key": str(row.get("request_key", "")),
        "profile_name": str(row.get("profile_name", "")),
        "symbol": str(row.get("symbol", "")),
        "plugin_name": str(row.get("plugin_name", "")),
        "family_id": _safe_int(row.get("family_id", 11), 11),
        "trigger_source": str(row.get("trigger_source", "")),
        "trigger_action": str(row.get("trigger_action", "")),
        "trigger_state": str(row.get("trigger_state", "")),
        "aggregate_risk_score": round(_safe_float(row.get("aggregate_risk_score", 0.0), 0.0), 6),
        "priority": _safe_int(row.get("priority", 0), 0),
        "status": str(row.get("status", "")),
        "requested_data_days": _safe_int(row.get("requested_data_days", 0), 0),
        "requested_months_list": str(row.get("requested_months_list", "")),
        "requested_wf_train_years": str(row.get("requested_wf_train_years", "")),
        "scenario_list": str(row.get("scenario_list", "")),
        "required_gates": _safe_json(row.get("required_gates_json", "[]"), []),
        "reason_codes": _safe_json(row.get("reason_codes_json", "[]"), []),
        "source": source_payload,
        "execution": execution_payload,
        "created_at": _safe_int(row.get("created_at", 0), 0),
        "updated_at": _safe_int(row.get("updated_at", 0), 0),
        "queued_at": _safe_int(row.get("queued_at", 0), 0),
        "started_at": _safe_int(row.get("started_at", 0), 0),
        "completed_at": _safe_int(row.get("completed_at", 0), 0),
    }


def build_drift_retraining_report(conn: libsql.Connection, profile_name: str, symbol: str = "") -> dict[str, Any]:
    ensure_drift_retraining_dirs()
    clauses = ["profile_name = ?"]
    params: list[Any] = [profile_name]
    if symbol:
        clauses.append("symbol = ?")
        params.append(symbol.upper())
    rows = query_all(
        conn,
        f"""
        SELECT *
          FROM drift_retraining_requests
         WHERE {' AND '.join(clauses)}
         ORDER BY priority DESC, updated_at DESC, symbol ASC, plugin_name ASC
        """,
        params,
    )
    status_counts: dict[str, int] = {}
    symbol_counts: dict[str, int] = {}
    items = []
    for row in rows:
        item = _request_to_report_item(row)
        items.append(item)
        status_counts[item["status"]] = status_counts.get(item["status"], 0) + 1
        symbol_counts[item["symbol"]] = symbol_counts.get(item["symbol"], 0) + 1
    payload = {
        "schema_version": DRIFT_RETRAINING_SCHEMA_VERSION,
        "report_version": DRIFT_RETRAINING_REPORT_VERSION,
        "generated_at": now_unix(),
        "profile_name": profile_name,
        "request_count": len(items),
        "status_counts": dict(sorted(status_counts.items())),
        "symbol_counts": dict(sorted(symbol_counts.items())),
        "requests": items,
        "safety": {
            "detector": "drift_governance",
            "auto_promote": False,
            "promotion_allowed_from_drift": False,
            "execution_mode": "research_campaign_only",
            "required_gates": DEFAULT_REQUIRED_GATES,
        },
        "artifacts": {
            "report_path": str(DRIFT_RETRAINING_REPORT_PATH),
            "status_path": str(DRIFT_RETRAINING_STATUS_PATH),
            "history_path": str(DRIFT_RETRAINING_HISTORY_PATH),
            "alerts_path": str(DRIFT_RETRAINING_ALERTS_PATH),
        },
    }
    _write_json(DRIFT_RETRAINING_REPORT_PATH, payload)
    return payload


def load_retraining_requests_for_execution(conn: libsql.Connection, args) -> list[dict[str, Any]]:
    profile_name = str(getattr(args, "profile", "") or "continuous")
    symbol = str(getattr(args, "symbol", "") or "").strip().upper()
    plugin_name = str(getattr(args, "plugin", "") or "").strip()
    request_key = str(getattr(args, "request_key", "") or "").strip()
    statuses = parse_csv_tokens(str(getattr(args, "status", "") or "queued"))
    clauses = ["profile_name = ?"]
    params: list[Any] = [profile_name]
    if request_key:
        clauses.append("request_key = ?")
        params.append(request_key)
    if symbol:
        clauses.append("symbol = ?")
        params.append(symbol)
    if plugin_name:
        clauses.append("plugin_name = ?")
        params.append(plugin_name)
    if statuses:
        placeholders = ", ".join("?" for _ in statuses)
        clauses.append(f"status IN ({placeholders})")
        params.extend(statuses)
    limit = max(_safe_int(getattr(args, "limit", 1), 1), 1)
    return query_all(
        conn,
        f"""
        SELECT *
          FROM drift_retraining_requests
         WHERE {' AND '.join(clauses)}
         ORDER BY priority DESC, queued_at ASC, id ASC
         LIMIT {limit}
        """,
        params,
    )


def build_retraining_campaign_plan(row: dict[str, Any], args, group_key: str) -> dict[str, Any]:
    symbol = str(row.get("symbol", "") or "")
    requested_months = str(row.get("requested_months_list", "") or "3")
    scenario_list = str(row.get("scenario_list", "") or SERIOUS_SCENARIOS)
    wf_years = str(row.get("requested_wf_train_years", "") or "1,2,3,5")
    return {
        "request_key": str(row.get("request_key", "")),
        "profile": str(row.get("profile_name", "")),
        "symbol": symbol,
        "plugin_name": str(row.get("plugin_name", "")),
        "plugin_filter": str(row.get("plugin_name", "")),
        "group_key": group_key,
        "months_list": requested_months,
        "scenario_list": scenario_list,
        "wf_year_presets": wf_years,
        "auto_export": True,
        "auto_promote": False,
        "promotion_command": "manual_best_params_only",
        "required_gates": _safe_json(row.get("required_gates_json", "[]"), []),
        "skip_compile": bool(getattr(args, "skip_compile", False)),
        "limit_experiments": _safe_int(getattr(args, "limit_experiments", 0), 0),
        "limit_runs": _safe_int(getattr(args, "limit_runs", 0), 0),
        "top_plugins": _safe_int(getattr(args, "top_plugins", 1), 1),
    }


def mark_retraining_execution(conn: libsql.Connection,
                              request_id: int,
                              *,
                              status: str,
                              payload: dict[str, Any],
                              started: bool = False,
                              completed: bool = False) -> None:
    now = now_unix()
    started_value = now if started else 0
    completed_value = now if completed else 0
    conn.execute(
        """
        UPDATE drift_retraining_requests
           SET status = ?,
               execution_payload_json = ?,
               updated_at = ?,
               started_at = CASE WHEN ? > 0 THEN ? ELSE started_at END,
               completed_at = CASE WHEN ? > 0 THEN ? ELSE completed_at END
         WHERE id = ?
        """,
        (
            status,
            json.dumps(payload, sort_keys=True),
            now,
            started_value,
            started_value,
            completed_value,
            completed_value,
            int(request_id),
        ),
    )
