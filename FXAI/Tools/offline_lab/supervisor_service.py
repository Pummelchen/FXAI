from __future__ import annotations

import json
import time
import sqlite3
from collections import defaultdict
from pathlib import Path

from .common import *
from .shadow_fleet import symbol_shadow_summary


CONTROL_PLANE_DIR = testlab.COMMON_FILES / "FXAI/ControlPlane"
SUPERVISOR_GLOBAL_FILE = COMMON_PROMOTION_DIR / "fxai_supervisor_service_global.tsv"
SUPERVISOR_COMMAND_GLOBAL_FILE = COMMON_PROMOTION_DIR / "fxai_supervisor_command_global.tsv"


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(v)))


def _safe_float(raw, default: float = 0.0) -> float:
    try:
        return float(raw)
    except Exception:
        return float(default)


def _safe_int(raw, default: int = 0) -> int:
    try:
        return int(float(raw))
    except Exception:
        return int(default)


def _safe_json(raw, default):
    try:
        return json.loads(raw or "")
    except Exception:
        return default


def _parse_symbol_legs(symbol: str) -> tuple[str, str]:
    clean = "".join(ch for ch in str(symbol or "").upper() if "A" <= ch <= "Z")
    if len(clean) >= 6:
        return clean[:3], clean[3:6]
    return "", ""


def _correlation_weight(anchor_symbol: str, other_symbol: str) -> float:
    if anchor_symbol == other_symbol:
        return 1.0
    base_a, quote_a = _parse_symbol_legs(anchor_symbol)
    base_b, quote_b = _parse_symbol_legs(other_symbol)
    if len(base_a) != 3 or len(quote_a) != 3 or len(base_b) != 3 or len(quote_b) != 3:
        return 0.0
    if base_a == quote_b and quote_a == base_b:
        return 1.0
    if base_a == base_b and quote_a == quote_b:
        return 1.0
    if base_a == base_b or quote_a == quote_b:
        return 0.85
    if base_a == quote_b or quote_a == base_b:
        return 0.70
    if base_a in (base_b, quote_b) or quote_a in (base_b, quote_b):
        return 0.55
    return 0.0


def _snapshot_file(symbol: str) -> Path:
    return COMMON_PROMOTION_DIR / f"fxai_supervisor_service_{safe_token(symbol)}.tsv"


def _command_file(symbol: str) -> Path:
    return COMMON_PROMOTION_DIR / f"fxai_supervisor_command_{safe_token(symbol)}.tsv"


def _load_snapshot(path: Path) -> dict | None:
    if not path.exists() or path.stat().st_size <= 0:
        return None
    rows: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not line or "\t" not in line:
            continue
        key, value = line.split("\t", 1)
        rows[key.strip()] = value.strip()
    symbol = str(rows.get("symbol", "") or "").upper()
    if not symbol:
        return None
    return {
        "symbol": symbol,
        "login": _safe_int(rows.get("login", 0)),
        "magic": _safe_int(rows.get("magic", 0)),
        "chart_id": _safe_int(rows.get("chart_id", 0)),
        "bar_time": _safe_int(rows.get("bar_time", 0)),
        "direction": _safe_int(rows.get("direction", -1)),
        "signal_intensity": _safe_float(rows.get("signal_intensity", 0.0)),
        "confidence": _safe_float(rows.get("confidence", 0.0)),
        "reliability": _safe_float(rows.get("reliability", 0.0)),
        "trade_gate": _safe_float(rows.get("trade_gate", 0.0)),
        "hierarchy_score": _safe_float(rows.get("hierarchy_score", 0.0)),
        "macro_quality": _safe_float(rows.get("macro_quality", 0.0)),
        "trade_edge_norm": _safe_float(rows.get("trade_edge_norm", 0.0)),
        "expected_move_norm": _safe_float(rows.get("expected_move_norm", 0.0)),
        "policy_trade_prob": _safe_float(rows.get("policy_trade_prob", 0.0)),
        "policy_no_trade_prob": _safe_float(rows.get("policy_no_trade_prob", 0.0)),
        "policy_enter_prob": _safe_float(rows.get("policy_enter_prob", 0.0)),
        "policy_exit_prob": _safe_float(rows.get("policy_exit_prob", 0.0)),
        "policy_add_prob": _safe_float(rows.get("policy_add_prob", 0.0)),
        "policy_reduce_prob": _safe_float(rows.get("policy_reduce_prob", 0.0)),
        "policy_tighten_prob": _safe_float(rows.get("policy_tighten_prob", 0.0)),
        "policy_timeout_prob": _safe_float(rows.get("policy_timeout_prob", 0.0)),
        "policy_size_mult": _safe_float(rows.get("policy_size_mult", 0.0)),
        "policy_portfolio_fit": _safe_float(rows.get("policy_portfolio_fit", 0.0)),
        "policy_capital_efficiency": _safe_float(rows.get("policy_capital_efficiency", 0.0)),
        "policy_lifecycle_action": _safe_int(rows.get("policy_lifecycle_action", 0)),
        "gross_exposure_lots": _safe_float(rows.get("gross_exposure_lots", 0.0)),
        "correlated_exposure_lots": _safe_float(rows.get("correlated_exposure_lots", 0.0)),
        "directional_cluster_lots": _safe_float(rows.get("directional_cluster_lots", 0.0)),
        "capital_risk_pct": _safe_float(rows.get("capital_risk_pct", 0.0)),
        "portfolio_pressure": _safe_float(rows.get("portfolio_pressure", 0.0)),
        "path": str(path),
    }


def iter_control_plane_snapshots(ttl_seconds: int = 7200) -> list[dict]:
    if not CONTROL_PLANE_DIR.exists():
        return []
    now = now_unix()
    rows: list[dict] = []
    for path in sorted(CONTROL_PLANE_DIR.glob("cp_*.tsv")):
        snap = _load_snapshot(path)
        if not snap:
            continue
        bar_time = int(snap.get("bar_time", 0) or 0)
        if bar_time > 0 and now > bar_time and (now - bar_time) > int(ttl_seconds):
            continue
        rows.append(snap)
    return rows


def _supervisor_payload(symbol: str,
                        profile_name: str,
                        snapshots: list[dict],
                        shadow: dict,
                        deploy: dict | None,
                        supervisor_row: dict | None,
                        previous_payload: dict | None = None,
                        ttl_seconds: int = 180) -> dict:
    if symbol == "__GLOBAL__":
        relevant = list(snapshots)
    else:
        relevant = []
        for snap in snapshots:
            weight = _correlation_weight(symbol, str(snap.get("symbol", "")))
            if weight <= 0.0:
                continue
            item = dict(snap)
            item["corr_weight"] = weight
            relevant.append(item)

    gross = 0.0
    long_pressure = 0.0
    short_pressure = 0.0
    macro_pressure = 0.0
    concentration = 0.0
    add_pressure = 0.0
    reduce_pressure = 0.0
    timeout_pressure = 0.0
    freshness_penalty = 0.0
    if relevant:
        symbol_intensity: dict[str, float] = defaultdict(float)
        now = now_unix()
        age_penalty_sum = 0.0
        for snap in relevant:
            corr_weight = float(snap.get("corr_weight", 1.0))
            intensity = _clamp(float(snap.get("signal_intensity", 0.0)), 0.0, 4.0) * corr_weight
            gross += intensity
            symbol_intensity[str(snap["symbol"])] += intensity
            if int(snap.get("direction", -1)) == 1:
                long_pressure += intensity
            elif int(snap.get("direction", -1)) == 0:
                short_pressure += intensity
            macro_pressure += intensity * (1.0 - _clamp(float(snap.get("macro_quality", 0.0)), 0.0, 1.0))
            add_pressure += intensity * _clamp(float(snap.get("policy_add_prob", 0.0)), 0.0, 1.0)
            reduce_pressure += intensity * _clamp(float(snap.get("policy_reduce_prob", 0.0)), 0.0, 1.0)
            timeout_pressure += intensity * _clamp(float(snap.get("policy_timeout_prob", 0.0)), 0.0, 1.0)
            bar_time = int(snap.get("bar_time", 0) or 0)
            if bar_time > 0 and now > bar_time:
                age_penalty_sum += corr_weight * _clamp((now - bar_time) / float(max(ttl_seconds, 30)), 0.0, 1.0)
        if gross > 1e-9:
            concentration = max(symbol_intensity.values()) / gross
            macro_pressure /= gross
            add_pressure /= gross
            reduce_pressure /= gross
            timeout_pressure /= gross
            freshness_penalty = _clamp(age_penalty_sum / float(len(relevant)), 0.0, 1.0)

    mean_pressure = _clamp(float(shadow.get("mean_portfolio_pressure", 0.0)) / 1.5, 0.0, 1.0)
    mean_div = _clamp(float(shadow.get("mean_portfolio_div", 0.0)), 0.0, 1.0)
    mean_corr = _clamp(float(shadow.get("mean_portfolio_corr", 0.0)), 0.0, 1.0)
    mean_no_trade = _clamp(float(shadow.get("mean_policy_no_trade_prob", 0.0)), 0.0, 1.0)
    mean_enter = _clamp(float(shadow.get("mean_policy_enter_prob", 0.0)), 0.0, 1.0)
    shadow_score = _clamp(float(shadow.get("mean_shadow_score", 0.0)), 0.0, 1.0)
    route_regret = _clamp(float(shadow.get("mean_route_regret", 0.0)), 0.0, 1.0)
    supervisor_weight = _clamp(float((supervisor_row or {}).get("supervisor_weight", 0.45)), 0.0, 1.0)
    deploy_blend = _clamp(float((deploy or {}).get("supervisor_blend", 0.45)), 0.0, 1.0)

    gross_pressure = _clamp(gross / 4.0, 0.0, 2.0)
    long_pressure = _clamp(long_pressure / 3.0, 0.0, 2.0)
    short_pressure = _clamp(short_pressure / 3.0, 0.0, 2.0)
    score = _clamp(
        0.34 * gross_pressure +
        0.16 * concentration +
        0.14 * macro_pressure +
        0.12 * mean_pressure +
        0.12 * mean_no_trade +
        0.10 * mean_corr -
        0.10 * mean_div -
        0.08 * shadow_score +
        0.10 * route_regret,
        0.0,
        3.0,
    )
    blend = _clamp(0.55 * supervisor_weight + 0.45 * deploy_blend, 0.0, 1.0)
    prev_score = _safe_float((previous_payload or {}).get("supervisor_score", 0.0), 0.0)
    prev_gross = _safe_float((previous_payload or {}).get("gross_pressure", 0.0), 0.0)
    pressure_velocity = _clamp(score - prev_score, -1.0, 1.0)
    gross_velocity = _clamp(gross_pressure - prev_gross, -1.0, 1.0)
    generated_at = now_unix()
    expires_at = generated_at + max(int(ttl_seconds), 60)
    long_entry_budget_mult = _clamp(
        (0.96 - 0.18 * long_pressure - 0.12 * macro_pressure - 0.08 * freshness_penalty + 0.06 * mean_div) *
        (0.92 + 0.08 * max(-pressure_velocity, 0.0)),
        0.10,
        1.20,
    )
    short_entry_budget_mult = _clamp(
        (0.96 - 0.18 * short_pressure - 0.12 * macro_pressure - 0.08 * freshness_penalty + 0.06 * mean_div) *
        (0.92 + 0.08 * max(-pressure_velocity, 0.0)),
        0.10,
        1.20,
    )

    return {
        "profile_name": profile_name,
        "symbol": symbol,
        "generated_at": generated_at,
        "expires_at": expires_at,
        "snapshot_count": len(relevant),
        "gross_pressure": gross_pressure,
        "directional_long_pressure": long_pressure,
        "directional_short_pressure": short_pressure,
        "macro_pressure": _clamp(macro_pressure, 0.0, 1.5),
        "concentration_pressure": _clamp(concentration, 0.0, 1.0),
        "freshness_penalty": freshness_penalty,
        "pressure_velocity": pressure_velocity,
        "gross_velocity": gross_velocity,
        "long_entry_budget_mult": long_entry_budget_mult,
        "short_entry_budget_mult": short_entry_budget_mult,
        "budget_multiplier": _clamp(
            1.04 +
            0.12 * mean_div +
            0.08 * mean_enter -
            0.24 * gross_pressure -
            0.20 * mean_pressure -
            0.10 * mean_corr -
            0.08 * freshness_penalty,
            0.20,
            1.20,
        ),
        "add_multiplier": _clamp(
            0.96 +
            0.12 * mean_enter +
            0.12 * shadow_score +
            0.10 * mean_div +
            0.08 * add_pressure -
            0.20 * mean_pressure -
            0.16 * concentration,
            0.10,
            1.40,
        ),
        "reduce_bias": _clamp(
            0.28 * mean_pressure +
            0.22 * mean_no_trade +
            0.18 * reduce_pressure +
            0.16 * route_regret +
            0.14 * concentration,
            0.0,
            1.0,
        ),
        "exit_bias": _clamp(
            0.20 * mean_pressure +
            0.20 * mean_no_trade +
            0.18 * timeout_pressure +
            0.18 * macro_pressure +
            0.16 * concentration +
            0.12 * gross_pressure,
            0.0,
            1.0,
        ),
        "entry_floor": _clamp(
            0.40 +
            0.08 * mean_pressure +
            0.08 * mean_no_trade +
            0.06 * concentration -
            0.05 * shadow_score,
            0.10,
            0.95,
        ),
        "block_score": _clamp(
            0.92 +
            0.36 * blend +
            0.18 * mean_pressure +
            0.14 * concentration +
            0.12 * macro_pressure,
            0.20,
            3.0,
        ),
        "supervisor_score": score,
        "blend": blend,
        "shadow_summary": shadow,
    }


def write_supervisor_service_artifacts(conn: sqlite3.Connection,
                                       args) -> list[dict]:
    ensure_dir(COMMON_PROMOTION_DIR)
    out_dir = RESEARCH_DIR / safe_token(args.profile)
    ensure_dir(out_dir)

    snapshots = iter_control_plane_snapshots()
    deploy_rows = conn.execute(
        "SELECT symbol, payload_json FROM live_deployment_profiles WHERE profile_name = ?",
        (args.profile,),
    ).fetchall()
    prev_rows = conn.execute(
        "SELECT symbol, payload_json FROM supervisor_service_states WHERE profile_name = ?",
        (args.profile,),
    ).fetchall()
    supervisor_row = row_to_dict(conn.execute(
        """
        SELECT *
          FROM portfolio_supervisor_profiles
         WHERE profile_name = ?
         ORDER BY created_at DESC
         LIMIT 1
        """,
        (args.profile,),
    ).fetchone())
    deploy_map = {
        str(row["symbol"]): json.loads(row["payload_json"] or "{}")
        for row in deploy_rows
    }
    prev_map = {
        str(row["symbol"]): _safe_json(str(row["payload_json"] or "{}"), {})
        for row in prev_rows
    }
    symbols = sorted({str(row["symbol"]) for row in deploy_rows} | {str(item["symbol"]) for item in snapshots})
    artifacts: list[dict] = []
    created_at = now_unix()

    global_payload = _supervisor_payload(
        "__GLOBAL__",
        args.profile,
        snapshots,
        {"mean_portfolio_div": 0.0, "mean_portfolio_corr": 0.0, "mean_portfolio_pressure": 0.0,
         "mean_policy_no_trade_prob": 0.0, "mean_policy_enter_prob": 0.0, "mean_shadow_score": 0.0,
         "mean_route_regret": 0.0},
        None,
        supervisor_row,
        prev_map.get("__GLOBAL__"),
    )
    SUPERVISOR_GLOBAL_FILE.write_text(
        "".join(
            f"{key}\t{value:.6f}\n" if isinstance(value, float) else f"{key}\t{value}\n"
            for key, value in [
                ("profile_name", args.profile),
                ("symbol", "__GLOBAL__"),
                ("generated_at", global_payload["generated_at"]),
                ("expires_at", global_payload["expires_at"]),
                ("snapshot_count", global_payload["snapshot_count"]),
                ("gross_pressure", global_payload["gross_pressure"]),
                ("directional_long_pressure", global_payload["directional_long_pressure"]),
                ("directional_short_pressure", global_payload["directional_short_pressure"]),
                ("macro_pressure", global_payload["macro_pressure"]),
                ("concentration_pressure", global_payload["concentration_pressure"]),
                ("freshness_penalty", global_payload["freshness_penalty"]),
                ("pressure_velocity", global_payload["pressure_velocity"]),
                ("gross_velocity", global_payload["gross_velocity"]),
                ("long_entry_budget_mult", global_payload["long_entry_budget_mult"]),
                ("short_entry_budget_mult", global_payload["short_entry_budget_mult"]),
                ("budget_multiplier", global_payload["budget_multiplier"]),
                ("add_multiplier", global_payload["add_multiplier"]),
                ("reduce_bias", global_payload["reduce_bias"]),
                ("exit_bias", global_payload["exit_bias"]),
                ("entry_floor", global_payload["entry_floor"]),
                ("block_score", global_payload["block_score"]),
                ("supervisor_score", global_payload["supervisor_score"]),
            ]
        ),
        encoding="utf-8",
    )

    active_symbols = set(symbols)
    for symbol in symbols:
        shadow = symbol_shadow_summary(conn, args.profile, symbol)
        payload = _supervisor_payload(symbol,
                                      args.profile,
                                      snapshots,
                                      shadow,
                                      deploy_map.get(symbol),
                                      supervisor_row,
                                      prev_map.get(symbol))
        tsv_path = _snapshot_file(symbol)
        tsv_path.write_text(
            "".join(
                f"{key}\t{value:.6f}\n" if isinstance(value, float) else f"{key}\t{value}\n"
                for key, value in [
                    ("profile_name", args.profile),
                    ("symbol", symbol),
                    ("generated_at", payload["generated_at"]),
                    ("expires_at", payload["expires_at"]),
                    ("snapshot_count", payload["snapshot_count"]),
                    ("gross_pressure", payload["gross_pressure"]),
                    ("directional_long_pressure", payload["directional_long_pressure"]),
                    ("directional_short_pressure", payload["directional_short_pressure"]),
                    ("macro_pressure", payload["macro_pressure"]),
                    ("concentration_pressure", payload["concentration_pressure"]),
                    ("freshness_penalty", payload["freshness_penalty"]),
                    ("pressure_velocity", payload["pressure_velocity"]),
                    ("gross_velocity", payload["gross_velocity"]),
                    ("long_entry_budget_mult", payload["long_entry_budget_mult"]),
                    ("short_entry_budget_mult", payload["short_entry_budget_mult"]),
                    ("budget_multiplier", payload["budget_multiplier"]),
                    ("add_multiplier", payload["add_multiplier"]),
                    ("reduce_bias", payload["reduce_bias"]),
                    ("exit_bias", payload["exit_bias"]),
                    ("entry_floor", payload["entry_floor"]),
                    ("block_score", payload["block_score"]),
                    ("supervisor_score", payload["supervisor_score"]),
                ]
            ),
            encoding="utf-8",
        )
        sha = testlab.sha256_path(tsv_path)
        json_path = out_dir / f"supervisor_service_{safe_token(symbol)}.json"
        json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        conn.execute(
            """
            INSERT INTO supervisor_service_states(profile_name, symbol, artifact_path, artifact_sha256,
                                                  snapshot_count, gross_pressure, directional_long_pressure,
                                                  directional_short_pressure, macro_pressure, concentration_pressure,
                                                  budget_multiplier, add_multiplier, reduce_bias, exit_bias,
                                                  entry_floor, block_score, payload_json, created_at)
            VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(profile_name, symbol) DO UPDATE SET
                artifact_path=excluded.artifact_path,
                artifact_sha256=excluded.artifact_sha256,
                snapshot_count=excluded.snapshot_count,
                gross_pressure=excluded.gross_pressure,
                directional_long_pressure=excluded.directional_long_pressure,
                directional_short_pressure=excluded.directional_short_pressure,
                macro_pressure=excluded.macro_pressure,
                concentration_pressure=excluded.concentration_pressure,
                budget_multiplier=excluded.budget_multiplier,
                add_multiplier=excluded.add_multiplier,
                reduce_bias=excluded.reduce_bias,
                exit_bias=excluded.exit_bias,
                entry_floor=excluded.entry_floor,
                block_score=excluded.block_score,
                payload_json=excluded.payload_json,
                created_at=excluded.created_at
            """,
            (
                args.profile,
                symbol,
                str(tsv_path),
                sha,
                int(payload["snapshot_count"]),
                payload["gross_pressure"],
                payload["directional_long_pressure"],
                payload["directional_short_pressure"],
                payload["macro_pressure"],
                payload["concentration_pressure"],
                payload["budget_multiplier"],
                payload["add_multiplier"],
                payload["reduce_bias"],
                payload["exit_bias"],
                payload["entry_floor"],
                payload["block_score"],
                json.dumps(payload, indent=2, sort_keys=True),
                created_at,
            ),
        )
        artifacts.append({
            "symbol": symbol,
            "artifact_path": str(tsv_path),
            "artifact_sha256": sha,
        })

    stale_rows = conn.execute(
        "SELECT symbol, artifact_path FROM supervisor_service_states WHERE profile_name = ?",
        (args.profile,),
    ).fetchall()
    for row in stale_rows:
        symbol = str(row["symbol"])
        if symbol in active_symbols:
            continue
        path = Path(str(row["artifact_path"] or "").strip())
        if path.exists() and path.is_file():
            path.unlink()
        stale_json = out_dir / f"supervisor_service_{safe_token(symbol)}.json"
        if stale_json.exists():
            stale_json.unlink()
        conn.execute(
            "DELETE FROM supervisor_service_states WHERE profile_name = ? AND symbol = ?",
            (args.profile, symbol),
        )

    conn.commit()
    summary_path = out_dir / "supervisor_service.json"
    summary_path.write_text(json.dumps({
        "profile_name": args.profile,
        "global_artifact_path": str(SUPERVISOR_GLOBAL_FILE),
        "snapshots_seen": len(snapshots),
        "symbols": artifacts,
    }, indent=2, sort_keys=True), encoding="utf-8")
    return artifacts


def write_supervisor_command_artifacts(conn: sqlite3.Connection,
                                       args) -> list[dict]:
    ensure_dir(COMMON_PROMOTION_DIR)
    out_dir = RESEARCH_DIR / safe_token(args.profile)
    ensure_dir(out_dir)
    service_rows = conn.execute(
        """
        SELECT symbol, payload_json
          FROM supervisor_service_states
         WHERE profile_name = ?
         ORDER BY created_at DESC
        """,
        (args.profile,),
    ).fetchall()
    deploy_rows = conn.execute(
        "SELECT symbol, payload_json FROM live_deployment_profiles WHERE profile_name = ?",
        (args.profile,),
    ).fetchall()
    router_rows = conn.execute(
        "SELECT symbol, payload_json FROM student_router_profiles WHERE profile_name = ?",
        (args.profile,),
    ).fetchall()
    service_map: dict[str, dict] = {}
    for row in service_rows:
        symbol = str(row["symbol"])
        if symbol not in service_map:
            service_map[symbol] = _safe_json(str(row["payload_json"] or "{}"), {})
    deploy_map = {str(row["symbol"]): _safe_json(str(row["payload_json"] or "{}"), {}) for row in deploy_rows}
    router_map = {str(row["symbol"]): _safe_json(str(row["payload_json"] or "{}"), {}) for row in router_rows}
    symbols = sorted(set(service_map) | set(deploy_map) | set(router_map))
    active_symbols = set(symbols)
    created_at = now_unix()
    artifacts: list[dict] = []

    global_payload = {
        "profile_name": args.profile,
        "symbol": "__GLOBAL__",
        "generated_at": created_at,
        "expires_at": created_at + 180,
        "entry_budget_mult": 1.0,
        "long_entry_budget_mult": 1.0,
        "short_entry_budget_mult": 1.0,
        "hold_budget_mult": 1.0,
        "add_cap_mult": 1.0,
        "reduce_bias": 0.0,
        "exit_bias": 0.0,
        "tighten_bias": 0.0,
        "timeout_bias": 0.0,
        "long_block": 0,
        "short_block": 0,
        "block_score": 1.10,
    }
    SUPERVISOR_COMMAND_GLOBAL_FILE.write_text(
        "".join(f"{key}\t{value}\n" for key, value in global_payload.items()),
        encoding="utf-8",
    )

    for symbol in symbols:
        shadow = symbol_shadow_summary(conn, args.profile, symbol)
        service = dict(service_map.get(symbol, {}))
        deploy = dict(deploy_map.get(symbol, {}))
        router = dict(router_map.get(symbol, {}))
        gross_pressure = _clamp(float(service.get("gross_pressure", 0.0)), 0.0, 2.0)
        macro_pressure = _clamp(float(service.get("macro_pressure", 0.0)), 0.0, 1.5)
        concentration = _clamp(float(service.get("concentration_pressure", 0.0)), 0.0, 1.0)
        long_pressure = _clamp(float(service.get("directional_long_pressure", 0.0)), 0.0, 2.0)
        short_pressure = _clamp(float(service.get("directional_short_pressure", 0.0)), 0.0, 2.0)
        route_regret = _clamp(float(shadow.get("mean_route_regret", 0.0)), 0.0, 1.0)
        shadow_score = _clamp(float(shadow.get("mean_shadow_score", 0.0)), -1.0, 1.0)
        no_trade = _clamp(float(shadow.get("mean_policy_no_trade_prob", 0.0)), 0.0, 1.0)
        champion_only = bool(router.get("champion_only", False))
        max_active_models = int(router.get("max_active_models", 12) or 12)
        payload = {
            "profile_name": args.profile,
            "symbol": symbol,
            "generated_at": created_at,
            "expires_at": created_at + 180,
            "entry_budget_mult": _clamp(
                float(service.get("budget_multiplier", 1.0)) *
                (0.96 - 0.18 * gross_pressure - 0.10 * concentration - 0.08 * route_regret),
                0.10,
                1.20,
            ),
            "long_entry_budget_mult": _clamp(float(service.get("long_entry_budget_mult", 1.0)), 0.10, 1.20),
            "short_entry_budget_mult": _clamp(float(service.get("short_entry_budget_mult", 1.0)), 0.10, 1.20),
            "hold_budget_mult": _clamp(
                0.96 - 0.12 * gross_pressure - 0.08 * macro_pressure + 0.06 * max(shadow_score, 0.0),
                0.20,
                1.20,
            ),
            "add_cap_mult": _clamp(
                float(service.get("add_multiplier", 1.0)) *
                (0.92 - 0.18 * concentration - 0.14 * gross_pressure - 0.10 * no_trade),
                0.05,
                1.20,
            ),
            "reduce_bias": _clamp(
                float(service.get("reduce_bias", 0.0)) +
                0.12 * concentration + 0.12 * route_regret + 0.10 * no_trade,
                0.0,
                1.0,
            ),
            "exit_bias": _clamp(
                float(service.get("exit_bias", 0.0)) +
                0.12 * macro_pressure + 0.12 * no_trade + 0.08 * max(-shadow_score, 0.0),
                0.0,
                1.0,
            ),
            "tighten_bias": _clamp(
                0.24 * gross_pressure + 0.22 * concentration + 0.18 * route_regret + 0.18 * no_trade,
                0.0,
                1.0,
            ),
            "timeout_bias": _clamp(
                0.26 * gross_pressure + 0.18 * macro_pressure + 0.18 * route_regret + 0.18 * no_trade,
                0.0,
                1.0,
            ),
            "long_block": 1 if (long_pressure > 0.92 and gross_pressure > 0.70 and (champion_only or no_trade > 0.64)) else 0,
            "short_block": 1 if (short_pressure > 0.92 and gross_pressure > 0.70 and (champion_only or no_trade > 0.64)) else 0,
            "block_score": _clamp(
                float(service.get("block_score", 1.10)) + 0.12 * route_regret,
                0.20,
                3.0,
            ),
            "max_active_models": max_active_models,
            "champion_only": 1 if champion_only else 0,
            "student_router_ready": 1 if router else 0,
            "deployment_ready": 1 if deploy else 0,
            "shadow_summary": shadow,
            "service_state": service,
        }
        tsv_path = _command_file(symbol)
        tsv_path.write_text(
            "".join(
                f"{key}\t{value:.6f}\n" if isinstance(value, float) else f"{key}\t{value}\n"
                for key, value in payload.items()
                if key not in ("shadow_summary", "service_state")
            ),
            encoding="utf-8",
        )
        artifact_sha = testlab.sha256_path(tsv_path)
        json_path = out_dir / f"supervisor_command_{safe_token(symbol)}.json"
        json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        conn.execute(
            """
            INSERT INTO supervisor_command_profiles(profile_name, symbol, artifact_path, artifact_sha256,
                                                    entry_budget_mult, hold_budget_mult, add_cap_mult,
                                                    reduce_bias, exit_bias, tighten_bias, timeout_bias,
                                                    long_block, short_block, block_score,
                                                    payload_json, created_at)
            VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(profile_name, symbol) DO UPDATE SET
                artifact_path=excluded.artifact_path,
                artifact_sha256=excluded.artifact_sha256,
                entry_budget_mult=excluded.entry_budget_mult,
                hold_budget_mult=excluded.hold_budget_mult,
                add_cap_mult=excluded.add_cap_mult,
                reduce_bias=excluded.reduce_bias,
                exit_bias=excluded.exit_bias,
                tighten_bias=excluded.tighten_bias,
                timeout_bias=excluded.timeout_bias,
                long_block=excluded.long_block,
                short_block=excluded.short_block,
                block_score=excluded.block_score,
                payload_json=excluded.payload_json,
                created_at=excluded.created_at
            """,
            (
                args.profile,
                symbol,
                str(tsv_path),
                artifact_sha,
                float(payload["entry_budget_mult"]),
                float(payload["hold_budget_mult"]),
                float(payload["add_cap_mult"]),
                float(payload["reduce_bias"]),
                float(payload["exit_bias"]),
                float(payload["tighten_bias"]),
                float(payload["timeout_bias"]),
                int(payload["long_block"]),
                int(payload["short_block"]),
                float(payload["block_score"]),
                json.dumps(payload, indent=2, sort_keys=True),
                created_at,
            ),
        )
        artifacts.append({
            "symbol": symbol,
            "artifact_path": str(tsv_path),
            "artifact_sha256": artifact_sha,
        })

    stale_rows = conn.execute(
        "SELECT symbol, artifact_path FROM supervisor_command_profiles WHERE profile_name = ?",
        (args.profile,),
    ).fetchall()
    for row in stale_rows:
        symbol = str(row["symbol"])
        if symbol in active_symbols:
            continue
        path = Path(str(row["artifact_path"] or "").strip())
        if path.exists() and path.is_file():
            path.unlink()
        stale_json = out_dir / f"supervisor_command_{safe_token(symbol)}.json"
        if stale_json.exists():
            stale_json.unlink()
        conn.execute(
            "DELETE FROM supervisor_command_profiles WHERE profile_name = ? AND symbol = ?",
            (args.profile, symbol),
        )

    conn.commit()
    summary_path = out_dir / "supervisor_commands.json"
    summary_path.write_text(json.dumps(artifacts, indent=2, sort_keys=True), encoding="utf-8")
    return artifacts


def run_supervisor_daemon(conn: sqlite3.Connection,
                          args) -> dict:
    interval = max(int(getattr(args, "interval_seconds", 30) or 0), 5)
    iterations = int(getattr(args, "iterations", 0) or 0)
    cycle = 0
    last_payload: list[dict] = []
    last_commands: list[dict] = []
    while True:
        cycle += 1
        last_payload = write_supervisor_service_artifacts(conn, args)
        last_commands = write_supervisor_command_artifacts(conn, args)
        if iterations > 0 and cycle >= iterations:
            break
        time.sleep(interval)
    return {
        "profile_name": args.profile,
        "iterations": cycle,
        "artifacts": last_payload,
        "commands": last_commands,
        "interval_seconds": interval,
    }
