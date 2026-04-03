from __future__ import annotations

import json
import math
from typing import Any

import libsql

from .common import RESEARCH_VECTOR_DIMS, now_unix, query_all


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _normalize(values: list[float]) -> list[float]:
    if not values:
        return [0.0] * RESEARCH_VECTOR_DIMS
    norm = math.sqrt(sum(value * value for value in values))
    if norm <= 1e-12:
        return [0.0] * len(values)
    return [value / norm for value in values]


def _vector_json(values: list[float]) -> str:
    return json.dumps([round(float(value), 8) for value in values], separators=(",", ":"))


def shadow_observation_embedding(row: dict[str, Any]) -> list[float]:
    vector = [
        float(row.get("meta_weight", 0.0)),
        float(row.get("reliability", 0.0)),
        float(row.get("global_edge", 0.0)),
        float(row.get("context_edge", 0.0)),
        float(row.get("context_regret", 0.0)),
        float(row.get("portfolio_objective", 0.0)),
        float(row.get("portfolio_stability", 0.0)),
        float(row.get("portfolio_corr", 0.0)),
        float(row.get("portfolio_div", 0.0)),
        float(row.get("route_value", 0.0)),
        float(row.get("route_regret", 0.0)),
        float(row.get("route_counterfactual", 0.0)),
        float(row.get("shadow_score", 0.0)),
        _clamp(float(row.get("portfolio_pressure", 0.0)) / 1.5, 0.0, 1.0),
        _clamp(float(row.get("policy_capital_efficiency", 0.0)), 0.0, 1.0),
        _clamp(float(row.get("portfolio_supervisor_score", 0.0)), 0.0, 1.0),
    ]
    return _normalize(vector)


def family_scorecard_embedding(row: dict[str, Any]) -> list[float]:
    vector = [
        _clamp(float(row.get("mean_score", 0.0)) / 100.0, 0.0, 1.0),
        _clamp(float(row.get("mean_recent_score", 0.0)) / 100.0, 0.0, 1.0),
        _clamp(float(row.get("mean_walkforward_score", 0.0)) / 100.0, 0.0, 1.0),
        _clamp(float(row.get("mean_adversarial_score", 0.0)) / 100.0, 0.0, 1.0),
        _clamp(float(row.get("mean_macro_score", 0.0)) / 100.0, 0.0, 1.0),
        _clamp(float(row.get("stability_score", 0.0)), 0.0, 1.0),
        _clamp(float(row.get("promotion_count", 0.0)) / 8.0, 0.0, 1.0),
        _clamp(float(row.get("champion_count", 0.0)) / 4.0, 0.0, 1.0),
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ]
    return _normalize(vector)


def _upsert_vector(conn: libsql.Connection,
                   profile_name: str,
                   symbol: str,
                   vector_scope: str,
                   source_type: str,
                   source_key: str,
                   embedding: list[float],
                   score: float,
                   payload: dict[str, Any]) -> None:
    conn.execute(
        """
        INSERT INTO research_vectors(profile_name, symbol, vector_scope, source_type, source_key,
                                     dims, vector_blob, score, payload_json, created_at)
        VALUES(?, ?, ?, ?, ?, ?, vector32(?), ?, ?, ?)
        ON CONFLICT(profile_name, symbol, vector_scope, source_type, source_key) DO UPDATE SET
            dims=excluded.dims,
            vector_blob=excluded.vector_blob,
            score=excluded.score,
            payload_json=excluded.payload_json,
            created_at=excluded.created_at
        """,
        (
            profile_name,
            symbol,
            vector_scope,
            source_type,
            source_key,
            RESEARCH_VECTOR_DIMS,
            _vector_json(embedding),
            float(score),
            json.dumps(payload, sort_keys=True),
            now_unix(),
        ),
    )


def refresh_research_vectors(conn: libsql.Connection,
                             profile_name: str,
                             symbol: str = "") -> dict[str, Any]:
    params: list[object] = [profile_name]
    symbol_sql = ""
    if symbol:
        symbol_sql = " AND symbol = ?"
        params.append(symbol)
    shadow_rows = query_all(
        conn,
        f"""
        SELECT *
          FROM shadow_fleet_observations
         WHERE profile_name = ?{symbol_sql}
         ORDER BY captured_at DESC
        """,
        params,
    )
    latest_shadow: dict[tuple[str, str], dict[str, Any]] = {}
    for row in shadow_rows:
        key = (str(row["symbol"]), str(row["plugin_name"]))
        if key not in latest_shadow:
            latest_shadow[key] = row
    shadow_count = 0
    for row in latest_shadow.values():
        embedding = shadow_observation_embedding(row)
        source_key = f"{row['plugin_name']}:{row['captured_at']}"
        _upsert_vector(
            conn,
            profile_name,
            str(row["symbol"]),
            "analog_shadow",
            "shadow_observation",
            source_key,
            embedding,
            float(row.get("shadow_score", 0.0)),
            {
                "plugin_name": row["plugin_name"],
                "family_id": int(row.get("family_id", 11)),
                "captured_at": int(row.get("captured_at", 0)),
            },
        )
        shadow_count += 1

    family_rows = query_all(
        conn,
        f"""
        SELECT *
          FROM family_scorecards
         WHERE profile_name = ?{symbol_sql}
         ORDER BY created_at DESC
        """,
        params,
    )
    latest_family: dict[tuple[str, int], dict[str, Any]] = {}
    for row in family_rows:
        key = (str(row["symbol"]), int(row["family_id"]))
        if key not in latest_family:
            latest_family[key] = row
    family_count = 0
    for row in latest_family.values():
        embedding = family_scorecard_embedding(row)
        source_key = f"family:{row['family_id']}"
        _upsert_vector(
            conn,
            profile_name,
            str(row["symbol"]),
            "family_state",
            "family_scorecard",
            source_key,
            embedding,
            float(row.get("mean_score", 0.0)),
            {
                "family_id": int(row["family_id"]),
                "family_name": str(row.get("family_name", "")),
                "created_at": int(row.get("created_at", 0)),
            },
        )
        family_count += 1
    return {
        "profile_name": profile_name,
        "symbol_filter": symbol,
        "shadow_vectors": shadow_count,
        "family_vectors": family_count,
    }


def latest_symbol_shadow_neighbors(conn: libsql.Connection,
                                   profile_name: str,
                                   symbol: str,
                                   limit: int = 5) -> list[dict[str, Any]]:
    latest = query_all(
        conn,
        """
        SELECT *
          FROM shadow_fleet_observations
         WHERE profile_name = ? AND symbol = ?
         ORDER BY captured_at DESC
         LIMIT 1
        """,
        (profile_name, symbol),
    )
    if not latest:
        return []
    query_vector = _vector_json(shadow_observation_embedding(latest[0]))
    limit_value = max(int(limit), 1)
    try:
        rows = query_all(
            conn,
            """
            SELECT rv.symbol,
                   rv.vector_scope,
                   rv.source_type,
                   rv.source_key,
                   rv.score,
                   rv.payload_json,
                   vector_distance_cos(rv.vector_blob, vector32(?)) AS cosine_distance
              FROM research_vectors rv
              JOIN vector_top_k('idx_research_vectors_ann', vector32(?), ?) topk
                ON topk.id = rv.id
             WHERE rv.profile_name = ?
               AND rv.vector_scope = 'analog_shadow'
               AND NOT (rv.symbol = ? AND rv.source_type = 'shadow_observation')
             ORDER BY cosine_distance ASC, rv.score DESC
             LIMIT ?
            """,
            (query_vector, query_vector, max(limit_value * 4, 16), profile_name, symbol, limit_value),
        )
    except Exception:
        rows = query_all(
            conn,
            """
            SELECT rv.symbol,
                   rv.vector_scope,
                   rv.source_type,
                   rv.source_key,
                   rv.score,
                   rv.payload_json,
                   vector_distance_cos(rv.vector_blob, vector32(?)) AS cosine_distance
              FROM research_vectors rv
             WHERE rv.profile_name = ?
               AND rv.vector_scope = 'analog_shadow'
               AND NOT (rv.symbol = ? AND rv.source_type = 'shadow_observation')
             ORDER BY cosine_distance ASC, rv.score DESC
             LIMIT ?
            """,
            (query_vector, profile_name, symbol, limit_value),
        )
    for row in rows:
        try:
            row["payload"] = json.loads(str(row.get("payload_json", "") or "{}"))
        except Exception:
            row["payload"] = {}
    return rows
