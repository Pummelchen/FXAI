from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import libsql

from .adaptive_router_config import load_config
from .adaptive_router_contracts import (
    ADAPTIVE_ROUTER_REGIMES,
    ADAPTIVE_ROUTER_SESSIONS,
    ADAPTIVE_ROUTER_SCHEMA_VERSION,
    isoformat_utc,
    json_dump,
    research_profile_dir,
)
from .common import *
from .drift_governance import latest_governance_state_map
from .mode import resolve_runtime_mode
from .shadow_fleet import latest_shadow_rows


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(value)))


def _safe_json(raw: str, default):
    try:
        return json.loads(raw or "")
    except Exception:
        return default


def _pair_legs(symbol: str) -> tuple[str, str]:
    clean = "".join(ch for ch in str(symbol or "").upper() if "A" <= ch <= "Z")
    if len(clean) < 6:
        return "", ""
    return clean[:3], clean[3:6]


def _pair_tags(symbol: str) -> list[str]:
    base, quote = _pair_legs(symbol)
    if not base or not quote:
        return []
    tags: list[str] = []
    if "USD" in (base, quote):
        tags.append("dollar_core")
    if "JPY" in (base, quote):
        tags.append("yen_cross")
    if any(code in (base, quote) for code in ("AUD", "NZD", "CAD")):
        tags.append("commodity_fx")
    if any(code in (base, quote) for code in ("EUR", "GBP", "CHF")):
        tags.append("europe_rates")
    return sorted(set(tags))


def _family_strengths(conn: libsql.Connection,
                      profile_name: str,
                      symbol: str) -> dict[str, float]:
    rows = query_all(
        conn,
        """
        SELECT family_id, MAX(mean_score) AS best_score, MAX(stability_score) AS best_stability
          FROM family_scorecards
         WHERE profile_name = ? AND symbol = ?
         GROUP BY family_id
        """,
        (profile_name, symbol),
    )
    out: dict[str, float] = {}
    for row in rows:
        family_name = plugin_family_name(int(row["family_id"]))
        out[family_name] = _clamp(
            0.82 +
            0.20 * _clamp(float(row["best_score"]) / 100.0, -1.0, 1.0) +
            0.18 * _clamp(float(row["best_stability"]), 0.0, 1.0),
            0.40,
            1.35,
        )
    return out


def _latest_shadow_map(conn: libsql.Connection,
                       profile_name: str,
                       symbol: str) -> dict[str, dict[str, Any]]:
    return {
        plugin_name: row
        for (row_symbol, plugin_name), row in latest_shadow_rows(conn, profile_name).items()
        if row_symbol == symbol
    }


def _champion_rows(conn: libsql.Connection,
                   profile_name: str,
                   symbol: str) -> list[dict[str, Any]]:
    return [
        dict(row)
        for row in query_all(
            conn,
            """
            SELECT plugin_name, family_id, status, champion_score, challenger_score, portfolio_score
              FROM champion_registry
             WHERE profile_name = ? AND symbol = ?
             ORDER BY
                   CASE status WHEN 'champion' THEN 0 WHEN 'challenger' THEN 1 ELSE 2 END,
                   champion_score DESC,
                   challenger_score DESC,
                   plugin_name ASC
            """,
            (profile_name, symbol),
        )
    ]


def _bundle_rows(conn: libsql.Connection,
                 profile_name: str,
                 symbol: str) -> list[dict[str, Any]]:
    return [
        dict(row)
        for row in query_all(
            conn,
            """
            SELECT plugin_name, family_id, deployment_json
              FROM student_deployment_bundles
             WHERE profile_name = ? AND symbol = ?
             ORDER BY created_at DESC, plugin_name ASC
            """,
            (profile_name, symbol),
        )
    ]


def _plugin_candidates(conn: libsql.Connection,
                       profile_name: str,
                       symbol: str) -> list[dict[str, Any]]:
    shadow_map = _latest_shadow_map(conn, profile_name, symbol)
    bundle_rows = _bundle_rows(conn, profile_name, symbol)
    champions = _champion_rows(conn, profile_name, symbol)

    candidates: dict[str, dict[str, Any]] = {}
    for row in champions:
        plugin_name = str(row["plugin_name"])
        candidates[plugin_name] = {
            "plugin_name": plugin_name,
            "family_id": int(row.get("family_id", 11) or 11),
            "status": str(row.get("status", "candidate") or "candidate"),
            "champion_score": float(row.get("champion_score", 0.0) or 0.0),
            "challenger_score": float(row.get("challenger_score", 0.0) or 0.0),
            "portfolio_score": float(row.get("portfolio_score", 0.0) or 0.0),
        }
    for row in bundle_rows:
        plugin_name = str(row["plugin_name"])
        current = candidates.setdefault(
            plugin_name,
            {
                "plugin_name": plugin_name,
                "family_id": int(row.get("family_id", 11) or 11),
                "status": "candidate",
                "champion_score": 0.0,
                "challenger_score": 0.0,
                "portfolio_score": 0.0,
            },
        )
        current["family_id"] = int(row.get("family_id", current["family_id"]) or current["family_id"])
        payload = _safe_json(str(row.get("deployment_json", "") or "{}"), {})
        current["deployable_student_floor"] = _clamp(float(payload.get("deployable_student_floor", 0.42) or 0.42), 0.0, 1.0)
    for plugin_name, row in shadow_map.items():
        current = candidates.setdefault(
            plugin_name,
            {
                "plugin_name": plugin_name,
                "family_id": int(row.get("family_id", 11) or 11),
                "status": "candidate",
                "champion_score": 0.0,
                "challenger_score": 0.0,
                "portfolio_score": 0.0,
            },
        )
        current["family_id"] = int(row.get("family_id", current["family_id"]) or current["family_id"])
        current["shadow"] = dict(row)
    return sorted(candidates.values(), key=lambda item: (item["status"] != "champion", -float(item.get("champion_score", 0.0)), item["plugin_name"]))


def _match_pattern(plugin_name: str,
                   spec: dict[str, Any]) -> bool:
    lowered = plugin_name.lower()
    tokens = [str(item).strip().lower() for item in list(spec.get("match_any", []))]
    return any(token and token in lowered for token in tokens)


def _plugin_pattern_effects(config: dict[str, Any],
                            plugin_name: str) -> dict[str, Any]:
    out: dict[str, Any] = {
        "global_weight_mult": 1.0,
        "regime_weights": {label: 1.0 for label in ADAPTIVE_ROUTER_REGIMES},
        "news_compatibility": 1.0,
        "liquidity_robustness": 1.0,
        "matched_patterns": [],
    }
    for spec in list(config.get("plugin_patterns", [])):
        if not _match_pattern(plugin_name, spec):
            continue
        out["matched_patterns"].append(str(spec.get("id", plugin_name)))
        out["global_weight_mult"] *= float(spec.get("global_weight_mult", 1.0) or 1.0)
        for label, value in dict(spec.get("regime_weights", {})).items():
            if label in out["regime_weights"]:
                out["regime_weights"][label] *= float(value)
        out["news_compatibility"] *= float(spec.get("news_compatibility", 1.0) or 1.0)
        out["liquidity_robustness"] *= float(spec.get("liquidity_robustness", 1.0) or 1.0)
    return out


def _plugin_override_effects(config: dict[str, Any],
                             plugin_name: str) -> dict[str, Any]:
    overrides = dict(config.get("plugin_overrides", {}))
    spec = dict(overrides.get(plugin_name, {}))
    out = {
        "global_weight_mult": float(spec.get("global_weight_mult", 1.0) or 1.0),
        "regime_weights": {label: float(dict(spec.get("regime_weights", {})).get(label, 1.0) or 1.0) for label in ADAPTIVE_ROUTER_REGIMES},
        "news_compatibility": float(spec.get("news_compatibility", 1.0) or 1.0),
        "liquidity_robustness": float(spec.get("liquidity_robustness", 1.0) or 1.0),
    }
    return out


def _governance_reason_codes(state: dict[str, Any]) -> list[str]:
    reasons = _safe_json(str(state.get("reason_codes_json", "") or "[]"), [])
    if isinstance(reasons, list):
        return [str(item) for item in reasons if str(item)]
    payload = _safe_json(str(state.get("payload_json", "") or "{}"), {})
    if isinstance(payload, dict):
        nested = payload.get("reason_codes", [])
        if isinstance(nested, list):
            return [str(item) for item in nested if str(item)]
    return []


def _plugin_global_weight(candidate: dict[str, Any],
                          family_strength: float) -> float:
    status = str(candidate.get("status", "candidate") or "candidate").lower()
    status_base = 0.96
    if status == "champion":
        status_base = 1.08
    elif status == "challenger":
        status_base = 1.01

    shadow = dict(candidate.get("shadow", {}))
    shadow_component = 0.82
    if shadow:
        shadow_component = _clamp(
            0.78 +
            0.18 * float(shadow.get("shadow_score", 0.0) or 0.0) +
            0.14 * float(shadow.get("reliability", 0.0) or 0.0) +
            0.10 * float(shadow.get("portfolio_objective", 0.0) or 0.0) +
            0.10 * (1.0 - _clamp(float(shadow.get("route_regret", 0.0) or 0.0), 0.0, 1.0)) +
            0.08 * float(shadow.get("policy_portfolio_fit", 0.0) or 0.0),
            0.30,
            1.45,
        )

    deployable = _clamp(float(candidate.get("deployable_student_floor", 0.42) or 0.42), 0.0, 1.0)
    return _clamp(
        0.38 * status_base +
        0.34 * shadow_component +
        0.18 * family_strength +
        0.10 * (0.70 + 0.30 * deployable),
        0.10,
        1.80,
    )


def _apply_pair_tag_overrides(config: dict[str, Any],
                              symbol: str) -> tuple[dict[str, float], dict[str, float], list[str]]:
    regime_bias = {label: 1.0 for label in ADAPTIVE_ROUTER_REGIMES}
    thresholds = {
        "caution_threshold": float(config["thresholds"]["caution_threshold"]),
        "abstain_threshold": float(config["thresholds"]["abstain_threshold"]),
        "block_threshold": float(config["thresholds"]["block_threshold"]),
    }
    tags = _pair_tags(symbol)
    overrides = dict(config.get("pair_tag_overrides", {}))
    for tag in tags:
        spec = dict(overrides.get(tag, {}))
        for label, value in dict(spec.get("regime_bias", {})).items():
            if label in regime_bias:
                regime_bias[label] *= float(value)
        for key in ("caution_threshold", "abstain_threshold", "block_threshold"):
            if key in spec:
                thresholds[key] = min(thresholds[key], float(spec[key]))
    return regime_bias, thresholds, tags


def _family_session_weights(config: dict[str, Any],
                            family_name: str) -> dict[str, float]:
    family_defaults = dict(config.get("family_defaults", {}))
    spec = dict(family_defaults.get(family_name, family_defaults.get("other", {})))
    return {
        label: _clamp(float(dict(spec.get("session_weights", {})).get(label, 1.0) or 1.0), 0.05, 2.50)
        for label in ADAPTIVE_ROUTER_SESSIONS
    }


def _family_regime_weights(config: dict[str, Any],
                           family_name: str) -> tuple[dict[str, float], float, float]:
    family_defaults = dict(config.get("family_defaults", {}))
    spec = dict(family_defaults.get(family_name, family_defaults.get("other", {})))
    regime_weights = {
        label: _clamp(float(dict(spec.get("regime_weights", {})).get(label, 1.0) or 1.0), 0.05, 2.50)
        for label in ADAPTIVE_ROUTER_REGIMES
    }
    news_compatibility = _clamp(float(spec.get("news_compatibility", 1.0) or 1.0), 0.05, 2.50)
    liquidity_robustness = _clamp(float(spec.get("liquidity_robustness", 1.0) or 1.0), 0.05, 2.50)
    return regime_weights, news_compatibility, liquidity_robustness


def write_adaptive_router_profiles(conn: libsql.Connection,
                                   args) -> list[dict[str, Any]]:
    config = load_config()
    mode_cfg = resolve_runtime_mode(getattr(args, "runtime_mode", None))
    symbols = sorted({
        str(row["symbol"])
        for row in query_all(
            conn,
            """
            SELECT DISTINCT symbol
              FROM (
                  SELECT symbol FROM live_deployment_profiles WHERE profile_name = ?
                  UNION ALL
                  SELECT symbol FROM student_deployment_bundles WHERE profile_name = ?
                  UNION ALL
                  SELECT symbol FROM champion_registry WHERE profile_name = ?
              )
             ORDER BY symbol
            """,
            (args.profile, args.profile, args.profile),
        )
    })
    out_dir = research_profile_dir(args.profile)
    ensure_dir(out_dir)
    ensure_dir(COMMON_PROMOTION_DIR)
    created_at = now_unix()
    active_symbols = set(symbols)
    artifacts: list[dict[str, Any]] = []
    governance_map = latest_governance_state_map(conn, args.profile)

    for symbol in symbols:
        regime_bias, thresholds, pair_tags = _apply_pair_tag_overrides(config, symbol)
        family_strengths = _family_strengths(conn, args.profile, symbol)
        candidates = _plugin_candidates(conn, args.profile, symbol)
        governance_rows = {
            plugin_name: row
            for (row_symbol, plugin_name), row in governance_map.items()
            if row_symbol == symbol
        }

        family_regime_rows: dict[str, dict[str, float]] = {}
        family_session_rows: dict[str, dict[str, float]] = {}
        family_news: dict[str, float] = {}
        family_liquidity: dict[str, float] = {}
        for family_id in range(0, 12):
            family_name = plugin_family_name(family_id)
            regime_weights, news_compatibility, liquidity_robustness = _family_regime_weights(config, family_name)
            strength = family_strengths.get(family_name, 0.95)
            family_regime_rows[family_name] = {
                label: _clamp(regime_weights[label] * strength, 0.05, 2.50)
                for label in ADAPTIVE_ROUTER_REGIMES
            }
            family_session_rows[family_name] = _family_session_weights(config, family_name)
            family_news[family_name] = news_compatibility
            family_liquidity[family_name] = liquidity_robustness

        plugin_global_weights: dict[str, float] = {}
        plugin_news_compatibility: dict[str, float] = {}
        plugin_liquidity_robustness: dict[str, float] = {}
        plugin_regime_weights: dict[str, dict[str, float]] = {
            label: {} for label in ADAPTIVE_ROUTER_REGIMES
        }
        plugin_session_weights: dict[str, dict[str, float]] = {
            label: {} for label in ADAPTIVE_ROUTER_SESSIONS
        }
        plugin_payloads: dict[str, dict[str, Any]] = {}
        governance_overrides: dict[str, dict[str, Any]] = {}

        for candidate in candidates:
            plugin_name = str(candidate["plugin_name"])
            family_id = int(candidate.get("family_id", 11) or 11)
            family_name = plugin_family_name(family_id)
            family_strength = family_strengths.get(family_name, 0.95)
            base_global = _plugin_global_weight(candidate, family_strength)
            pattern = _plugin_pattern_effects(config, plugin_name)
            override = _plugin_override_effects(config, plugin_name)
            global_weight = _clamp(
                base_global *
                float(pattern["global_weight_mult"]) *
                float(override["global_weight_mult"]),
                float(config["thresholds"]["min_plugin_weight"]),
                float(config["thresholds"]["max_plugin_weight"]),
            )
            plugin_global_weights[plugin_name] = global_weight

            news_compatibility = _clamp(
                family_news.get(family_name, 1.0) *
                float(pattern["news_compatibility"]) *
                float(override["news_compatibility"]),
                0.05,
                2.50,
            )
            liquidity_robustness = _clamp(
                family_liquidity.get(family_name, 1.0) *
                float(pattern["liquidity_robustness"]) *
                float(override["liquidity_robustness"]),
                0.05,
                2.50,
            )
            governance_row = dict(governance_rows.get(plugin_name, {}))
            governance_state = str(governance_row.get("governance_state", "HEALTHY") or "HEALTHY").upper()
            weight_multiplier = _clamp(float(governance_row.get("weight_multiplier", 1.0) or 1.0), 0.0, 2.0)
            restrict_live = bool(int(governance_row.get("restrict_live", 0) or 0))
            shadow_only = bool(int(governance_row.get("shadow_only", 0) or 0))
            disabled = bool(int(governance_row.get("disabled", 0) or 0))
            if governance_row and bool(int(governance_row.get("action_applied", 0) or 0)):
                global_weight = _clamp(
                    global_weight * weight_multiplier,
                    0.0,
                    float(config["thresholds"]["max_plugin_weight"]),
                )
                if governance_state in {"SHADOW_ONLY", "DEMOTED", "DISABLED"} or restrict_live or shadow_only or disabled:
                    global_weight = float(config["thresholds"]["min_plugin_weight"])
                    news_compatibility = min(news_compatibility, 0.25)
                    liquidity_robustness = min(liquidity_robustness, 0.25)
            plugin_news_compatibility[plugin_name] = news_compatibility
            plugin_liquidity_robustness[plugin_name] = liquidity_robustness
            governance_overrides[plugin_name] = {
                "governance_state": governance_state,
                "action_recommendation": str(governance_row.get("action_recommendation", "") or "NONE").upper(),
                "action_applied": bool(int(governance_row.get("action_applied", 0) or 0)),
                "weight_multiplier": round(weight_multiplier, 6),
                "restrict_live": restrict_live,
                "shadow_only": shadow_only,
                "disabled": disabled,
                "reason_codes": _governance_reason_codes(governance_row),
            }
            session_weights: dict[str, float] = {}
            for session in ADAPTIVE_ROUTER_SESSIONS:
                session_weight = _clamp(
                    float(family_session_rows[family_name][session]),
                    0.05,
                    2.50,
                )
                plugin_session_weights[session][plugin_name] = session_weight
                session_weights[session] = session_weight

            regime_weights: dict[str, float] = {}
            for label in ADAPTIVE_ROUTER_REGIMES:
                value = (
                    family_regime_rows[family_name][label] *
                    regime_bias[label] *
                    float(pattern["regime_weights"][label]) *
                    float(override["regime_weights"][label])
                )
                weight = _clamp(value, 0.05, 2.50)
                plugin_regime_weights[label][plugin_name] = weight
                regime_weights[label] = weight

            plugin_payloads[plugin_name] = {
                "plugin_name": plugin_name,
                "family_id": family_id,
                "family_name": family_name,
                "status": str(candidate.get("status", "candidate") or "candidate"),
                "global_weight": round(global_weight, 6),
                "news_compatibility": round(news_compatibility, 6),
                "liquidity_robustness": round(liquidity_robustness, 6),
                "session_weights": {key: round(value, 6) for key, value in session_weights.items()},
                "regime_weights": {key: round(value, 6) for key, value in regime_weights.items()},
                "matched_patterns": list(pattern["matched_patterns"]),
                "governance": governance_overrides[plugin_name],
                "shadow": dict(candidate.get("shadow", {})),
            }

        sorted_plugins = sorted(
            plugin_global_weights.items(),
            key=lambda item: (item[1], item[0]),
            reverse=True,
        )

        caution_threshold = _clamp(thresholds["caution_threshold"], 0.20, 0.90)
        abstain_threshold = _clamp(thresholds["abstain_threshold"], 0.05, caution_threshold)
        block_threshold = _clamp(thresholds["block_threshold"], 0.05, abstain_threshold)
        if str(mode_cfg["runtime_mode"]) == "production":
            caution_threshold = _clamp(caution_threshold - 0.02, 0.20, 0.90)
            abstain_threshold = _clamp(abstain_threshold - 0.02, 0.05, caution_threshold)
            block_threshold = _clamp(block_threshold - 0.01, 0.05, abstain_threshold)

        payload = {
            "schema_version": ADAPTIVE_ROUTER_SCHEMA_VERSION,
            "profile_name": args.profile,
            "symbol": symbol,
            "enabled": bool(config.get("enabled", True)),
            "router_mode": str(config.get("router_mode", "WEIGHTED_ENSEMBLE")),
            "fallback_to_student_router_only": bool(config.get("fallback_to_student_router_only", True)),
            "pair_tags": pair_tags,
            "regime_bias": {key: round(value, 6) for key, value in regime_bias.items()},
            "thresholds": {
                "caution_threshold": round(caution_threshold, 6),
                "abstain_threshold": round(abstain_threshold, 6),
                "block_threshold": round(block_threshold, 6),
                "confidence_floor": round(_clamp(float(config["thresholds"]["confidence_floor"]), 0.0, 1.0), 6),
                "suppression_threshold": round(_clamp(float(config["thresholds"]["suppression_threshold"]), 0.05, 1.25), 6),
                "downweight_threshold": round(_clamp(float(config["thresholds"]["downweight_threshold"]), 0.20, 2.50), 6),
                "stale_news_abstain_bias": round(_clamp(float(config["thresholds"]["stale_news_abstain_bias"]), 0.0, 1.0), 6),
                "stale_news_force_caution": bool(config["thresholds"]["stale_news_force_caution"]),
                "min_plugin_weight": round(_clamp(float(config["thresholds"]["min_plugin_weight"]), 0.01, 1.0), 6),
                "max_plugin_weight": round(_clamp(float(config["thresholds"]["max_plugin_weight"]), 0.50, 3.00), 6),
                "max_active_weight_share": round(_clamp(float(config["thresholds"]["max_active_weight_share"]), 0.10, 0.99), 6),
            },
            "family_regime_weights": family_regime_rows,
            "family_session_weights": family_session_rows,
            "family_news_compatibility": family_news,
            "family_liquidity_robustness": family_liquidity,
            "plugins": plugin_payloads,
            "governance_overrides": governance_overrides,
            "summary": {
                "plugin_count": len(plugin_payloads),
                "top_plugins": [name for name, _weight in sorted_plugins[:12]],
                "runtime_mode": str(mode_cfg["runtime_mode"]),
                "generated_at": isoformat_utc(),
            },
        }

        tsv_path = COMMON_PROMOTION_DIR / f"fxai_adaptive_router_{safe_token(symbol)}.tsv"
        lines = [
            ("schema_version", str(ADAPTIVE_ROUTER_SCHEMA_VERSION)),
            ("profile_name", args.profile),
            ("symbol", symbol),
            ("enabled", "1" if payload["enabled"] else "0"),
            ("router_mode", payload["router_mode"]),
            ("fallback_to_student_router_only", "1" if payload["fallback_to_student_router_only"] else "0"),
            ("pair_tags_csv", ",".join(pair_tags)),
            ("caution_threshold", f"{caution_threshold:.6f}"),
            ("abstain_threshold", f"{abstain_threshold:.6f}"),
            ("block_threshold", f"{block_threshold:.6f}"),
            ("confidence_floor", f"{float(payload['thresholds']['confidence_floor']):.6f}"),
            ("suppression_threshold", f"{float(payload['thresholds']['suppression_threshold']):.6f}"),
            ("downweight_threshold", f"{float(payload['thresholds']['downweight_threshold']):.6f}"),
            ("stale_news_abstain_bias", f"{float(payload['thresholds']['stale_news_abstain_bias']):.6f}"),
            ("stale_news_force_caution", "1" if payload["thresholds"]["stale_news_force_caution"] else "0"),
            ("min_plugin_weight", f"{float(payload['thresholds']['min_plugin_weight']):.6f}"),
            ("max_plugin_weight", f"{float(payload['thresholds']['max_plugin_weight']):.6f}"),
            ("max_active_weight_share", f"{float(payload['thresholds']['max_active_weight_share']):.6f}"),
            ("plugin_global_weights_csv", ",".join(f"{name}={float(weight):.6f}" for name, weight in sorted(plugin_global_weights.items()))),
            ("plugin_news_compatibility_csv", ",".join(f"{name}={float(value):.6f}" for name, value in sorted(plugin_news_compatibility.items()))),
            ("plugin_liquidity_robustness_csv", ",".join(f"{name}={float(value):.6f}" for name, value in sorted(plugin_liquidity_robustness.items()))),
            ("governance_state_csv", ",".join(
                f"{name}={governance_overrides[name]['governance_state']}"
                for name in sorted(governance_overrides)
            )),
        ]
        for label in ADAPTIVE_ROUTER_REGIMES:
            lines.append((f"regime_bias_{label}", f"{float(regime_bias[label]):.6f}"))
            lines.append((f"plugin_regime_{label}_csv", ",".join(
                f"{name}={float(plugin_regime_weights[label][name]):.6f}"
                for name in sorted(plugin_regime_weights[label])
            )))
        for session in ADAPTIVE_ROUTER_SESSIONS:
            lines.append((f"plugin_session_{session}_csv", ",".join(
                f"{name}={float(plugin_session_weights[session][name]):.6f}"
                for name in sorted(plugin_session_weights[session])
            )))
        for family_name in sorted(family_regime_rows):
            lines.append((f"family_news_compatibility_{family_name}", f"{float(family_news[family_name]):.6f}"))
            lines.append((f"family_liquidity_robustness_{family_name}", f"{float(family_liquidity[family_name]):.6f}"))
            for label in ADAPTIVE_ROUTER_REGIMES:
                lines.append((f"family_regime_{family_name}_{label}", f"{float(family_regime_rows[family_name][label]):.6f}"))
            for session in ADAPTIVE_ROUTER_SESSIONS:
                lines.append((f"family_session_{family_name}_{session}", f"{float(family_session_rows[family_name][session]):.6f}"))
        tsv_path.write_text("".join(f"{key}\t{value}\n" for key, value in lines), encoding="utf-8")

        json_path = out_dir / f"adaptive_router_{safe_token(symbol)}.json"
        json_dump(json_path, payload)
        artifact_sha = testlab.sha256_path(tsv_path)
        conn.execute(
            """
            INSERT INTO adaptive_router_profiles(profile_name, symbol, artifact_path, artifact_sha256,
                                                 router_mode, caution_threshold, abstain_threshold,
                                                 block_threshold, payload_json, created_at)
            VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(profile_name, symbol) DO UPDATE SET
                artifact_path=excluded.artifact_path,
                artifact_sha256=excluded.artifact_sha256,
                router_mode=excluded.router_mode,
                caution_threshold=excluded.caution_threshold,
                abstain_threshold=excluded.abstain_threshold,
                block_threshold=excluded.block_threshold,
                payload_json=excluded.payload_json,
                created_at=excluded.created_at
            """,
            (
                args.profile,
                symbol,
                str(tsv_path),
                artifact_sha,
                payload["router_mode"],
                float(caution_threshold),
                float(abstain_threshold),
                float(block_threshold),
                json.dumps(payload, indent=2, sort_keys=True),
                created_at,
            ),
        )
        artifacts.append(
            {
                "symbol": symbol,
                "artifact_path": str(tsv_path),
                "artifact_sha256": artifact_sha,
                "pair_tags": pair_tags,
                "plugin_count": len(plugin_payloads),
                "top_plugins": [name for name, _weight in sorted_plugins[:8]],
                "caution_threshold": round(caution_threshold, 6),
                "abstain_threshold": round(abstain_threshold, 6),
                "block_threshold": round(block_threshold, 6),
            }
        )

    stale_rows = query_all(
        conn,
        "SELECT symbol, artifact_path FROM adaptive_router_profiles WHERE profile_name = ?",
        (args.profile,),
    )
    for row in stale_rows:
        symbol = str(row["symbol"])
        if symbol in active_symbols:
            continue
        path = Path(str(row["artifact_path"] or "").strip())
        if path.exists() and path.is_file():
            path.unlink()
        stale_json = out_dir / f"adaptive_router_{safe_token(symbol)}.json"
        if stale_json.exists():
            stale_json.unlink()
        conn.execute(
            "DELETE FROM adaptive_router_profiles WHERE profile_name = ? AND symbol = ?",
            (args.profile, symbol),
        )

    commit_db(conn)
    summary_path = out_dir / "adaptive_router_profiles.json"
    json_dump(summary_path, artifacts)
    return artifacts
