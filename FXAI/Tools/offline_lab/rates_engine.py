from __future__ import annotations

import math
import json
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any

from .newspulse_contracts import COMMON_NEWSPULSE_JSON, NEWSPULSE_STATUS_PATH
from .newspulse_policy import active_pairs, broker_symbols_for_pair, load_policy
from .rates_engine_config import load_config
from .rates_engine_contracts import (
    COMMON_RATES_FLAT,
    COMMON_RATES_HISTORY,
    COMMON_RATES_JSON,
    COMMON_RATES_SYMBOL_MAP,
    RATES_ENGINE_LOCAL_HISTORY_PATH,
    RATES_ENGINE_SCHEMA_VERSION,
    RATES_ENGINE_STATUS_PATH,
    RATES_ENGINE_STATE_PATH,
    ensure_rates_engine_dirs,
    isoformat_utc,
    json_dump,
    json_load,
    ndjson_append,
    parse_iso8601,
    sanitize_utc_timestamp,
    utc_now,
)
from .rates_engine_inputs import SUPPORTED_CURRENCIES, load_inputs
from .rates_engine_replay import build_rates_replay_report


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _optional_float(value: Any) -> float | None:
    if value in (None, "", "null"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(value)))


def _sign(value: float) -> float:
    if value > 0.0:
        return 1.0
    if value < 0.0:
        return -1.0
    return 0.0


def _pair_legs(pair_id: str) -> tuple[str, str]:
    clean = "".join(ch for ch in str(pair_id or "").upper() if "A" <= ch <= "Z")
    if len(clean) != 6:
        return "", ""
    return clean[:3], clean[3:6]


def _empty_state(now_dt: datetime | None = None) -> dict[str, Any]:
    reference = now_dt or utc_now()
    return {
        "last_generated_at": isoformat_utc(reference),
        "proxy_levels": {},
        "daemon_health": {
            "mode": "standalone",
            "heartbeat_at": isoformat_utc(reference),
            "last_cycle_started_at": "",
            "last_cycle_finished_at": "",
            "last_cycle_duration_sec": 0.0,
            "cycles_completed": 0,
            "consecutive_failures": 0,
            "degraded": False,
            "degraded_reasons": [],
            "interval_seconds": 0,
            "last_error": "",
        },
    }


def _load_state() -> dict[str, Any]:
    now_dt = utc_now()
    payload = json_load(RATES_ENGINE_STATE_PATH)
    if not payload:
        return _empty_state(now_dt)
    state = _empty_state(now_dt)
    state.update(payload)
    state.setdefault("proxy_levels", {})
    state.setdefault("daemon_health", _empty_state(now_dt)["daemon_health"])
    daemon = dict(state.get("daemon_health", {}))
    daemon["heartbeat_at"] = isoformat_utc(sanitize_utc_timestamp(daemon.get("heartbeat_at"), now_dt=now_dt) or now_dt)
    daemon["last_cycle_started_at"] = isoformat_utc(sanitize_utc_timestamp(daemon.get("last_cycle_started_at"), now_dt=now_dt) or now_dt) if daemon.get("last_cycle_started_at") else ""
    daemon["last_cycle_finished_at"] = isoformat_utc(sanitize_utc_timestamp(daemon.get("last_cycle_finished_at"), now_dt=now_dt) or now_dt) if daemon.get("last_cycle_finished_at") else ""
    state["daemon_health"] = daemon
    return state


def _save_state(payload: dict[str, Any]) -> None:
    json_dump(RATES_ENGINE_STATE_PATH, payload)


def _load_newspulse_snapshot(now_dt: datetime) -> tuple[dict[str, Any], dict[str, Any]]:
    snapshot = json_load(COMMON_NEWSPULSE_JSON)
    status = json_load(NEWSPULSE_STATUS_PATH)
    source_status = {}
    if isinstance(snapshot.get("source_status"), dict):
        source_status = dict(snapshot.get("source_status", {}))
    elif isinstance(status.get("source_status"), dict):
        source_status = dict(status.get("source_status", {}))
    generated_at = sanitize_utc_timestamp(snapshot.get("generated_at"), now_dt=now_dt)
    snapshot_stale_after_sec = 900
    if isinstance(status.get("health"), dict):
        snapshot_stale_after_sec = int(status.get("health", {}).get("snapshot_stale_after_sec", snapshot_stale_after_sec) or snapshot_stale_after_sec)
    if generated_at is None or (now_dt - generated_at).total_seconds() > snapshot_stale_after_sec:
        if isinstance(snapshot, dict):
            snapshot["stale"] = True
    return snapshot, source_status


def _iter_recent_items(snapshot: dict[str, Any]) -> list[dict[str, Any]]:
    items = snapshot.get("recent_items", [])
    return [dict(item) for item in items if isinstance(item, dict)] if isinstance(items, list) else []


def _iter_currency_state(snapshot: dict[str, Any]) -> dict[str, dict[str, Any]]:
    currencies = snapshot.get("currencies", {})
    if isinstance(currencies, dict):
        return {str(key).upper(): dict(value) for key, value in currencies.items() if isinstance(value, dict)}
    return {}


def _title_tokens(text: str) -> str:
    lowered = "".join(ch.lower() if ch.isalnum() else " " for ch in str(text or ""))
    return " ".join(part for part in lowered.split() if part)


def _topic_weight(topic_tags: list[str], topic_relevance: dict[str, float]) -> float:
    weights = [float(topic_relevance.get(str(topic), 0.0) or 0.0) for topic in topic_tags]
    return max(weights, default=0.0)


def _has_keyword(text: str, keywords: list[str]) -> bool:
    lowered = _title_tokens(text)
    return any(str(keyword).strip().lower() in lowered for keyword in keywords if str(keyword).strip())


def _item_policy_signal(
    item: dict[str, Any],
    currency: str,
    config: dict[str, Any],
    news_currency: dict[str, Any],
) -> dict[str, Any]:
    proxy_cfg = dict(config.get("proxy_model", {}))
    title = str(item.get("title", "") or "")
    topic_tags = [str(topic) for topic in list(item.get("topic_tags", []))]
    source = str(item.get("source", "") or "")
    domain = str(item.get("domain", "") or "")
    relevance = _topic_weight(topic_tags, dict(proxy_cfg.get("base_topic_relevance", {})))
    hawkish = _has_keyword(title, list(proxy_cfg.get("hawkish_keywords", [])))
    dovish = _has_keyword(title, list(proxy_cfg.get("dovish_keywords", [])))
    cb_event = _has_keyword(title, list(proxy_cfg.get("cb_event_keywords", []))) or "monetary_policy" in topic_tags
    macro_policy = _has_keyword(title, list(proxy_cfg.get("macro_policy_keywords", [])))
    official_boost = 1.18 if source == "official" else 1.0
    tone = abs(_safe_float(item.get("tone", 0.0)))
    magnitude = _clamp((0.35 + relevance + min(tone / 8.0, 0.35)) * official_boost, 0.0, 1.75)

    direction = 0.0
    if hawkish and not dovish:
        direction = magnitude
    elif dovish and not hawkish:
        direction = -magnitude

    if direction == 0.0 and source == "calendar":
        surprise = _safe_float(news_currency.get("last_surprise_proxy", 0.0))
        if macro_policy or cb_event or "scheduled_macro" in topic_tags:
            bias = max(
                (float(dict(proxy_cfg.get("topic_direction_bias", {})).get(topic, 0.0) or 0.0) for topic in topic_tags),
                default=0.0,
            )
            direction = _clamp(surprise * (0.45 + abs(bias)), -1.5, 1.5)

    return {
        "currency": currency,
        "item_id": str(item.get("id", "")),
        "source": source,
        "domain": domain,
        "title": title,
        "published_at": str(item.get("published_at", "")),
        "url": str(item.get("url", "") or ""),
        "topic_tags": topic_tags,
        "policy_relevance_score": round(_clamp(relevance * official_boost, 0.0, 1.25), 6),
        "direction": round(_clamp(direction, -1.75, 1.75), 6),
        "magnitude": round(magnitude, 6),
        "central_bank_event": cb_event,
        "macro_policy_event": macro_policy or "scheduled_macro" in topic_tags,
    }


def _recent_history_snapshots(hours_back: int) -> list[dict[str, Any]]:
    if not COMMON_RATES_HISTORY.exists():
        return []
    now_dt = utc_now()
    cutoff = now_dt - timedelta(hours=max(hours_back, 1))
    out: list[dict[str, Any]] = []
    for line in COMMON_RATES_HISTORY.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except Exception:
            continue
        if not isinstance(payload, dict) or payload.get("record_type") != "snapshot":
            continue
        generated_at = parse_iso8601(str(payload.get("generated_at", "")))
        if generated_at is None or generated_at < cutoff:
            continue
        snapshot = payload.get("snapshot")
        if isinstance(snapshot, dict):
            out.append(snapshot)
    return out


def _historical_currency_value(
    snapshots: list[dict[str, Any]],
    currency: str,
    key: str,
    hours_back: int,
    now_dt: datetime,
) -> float | None:
    cutoff = now_dt - timedelta(hours=hours_back)
    latest_value: float | None = None
    latest_time: datetime | None = None
    for snapshot in snapshots:
        generated_at = parse_iso8601(str(snapshot.get("generated_at", "")))
        if generated_at is None or generated_at > cutoff:
            continue
        state = dict(snapshot.get("currencies", {})).get(currency, {})
        if not isinstance(state, dict):
            continue
        value = _optional_float(state.get(key))
        if value is None:
            continue
        if latest_time is None or generated_at > latest_time:
            latest_time = generated_at
            latest_value = value
    return latest_value


def _curve_shape_regime(level: float | None, change_1d: float | None, config: dict[str, Any]) -> str:
    if level is None:
        return "UNAVAILABLE"
    thresholds = dict(config.get("curve_regime_thresholds", {}))
    inversion_level = _safe_float(thresholds.get("inversion_level", -0.15), -0.15)
    steepening_change = _safe_float(thresholds.get("steepening_change", 0.18), 0.18)
    flattening_change = _safe_float(thresholds.get("flattening_change", -0.18), -0.18)
    if level <= inversion_level:
        return "INVERSION_LIKE"
    if change_1d is not None and change_1d >= steepening_change:
        return "STEEPENING"
    if change_1d is not None and change_1d <= flattening_change:
        return "FLATTENING"
    return "NEUTRAL"


def _append_reason(reasons: list[str], reason: str) -> None:
    text = str(reason or "").strip()
    if not text or text in reasons:
        return
    reasons.append(text)


def _manual_currency_values(inputs: dict[str, Any], config: dict[str, Any], now_dt: datetime) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    currencies = dict(inputs.get("currencies", {}))
    providers = dict(config.get("providers", {}))
    stale_after_hours = int(providers.get("manual_stale_after_hours", 48) or 48)
    stale_after = timedelta(hours=max(stale_after_hours, 1))
    out: dict[str, dict[str, Any]] = {}
    updated_currencies = 0
    last_update: datetime | None = None
    stale_count = 0
    for currency in SUPPORTED_CURRENCIES:
        spec = dict(currencies.get(currency, {}))
        updated_at = sanitize_utc_timestamp(spec.get("last_update_at"), now_dt=now_dt)
        manual_enabled = bool(spec.get("enabled", True))
        front_end_level = _optional_float(spec.get("front_end_level"))
        expected_path_level = _optional_float(spec.get("expected_path_level"))
        curve_2y_level = _optional_float(spec.get("curve_2y_level"))
        curve_10y_level = _optional_float(spec.get("curve_10y_level"))
        curve_slope_2s10s = _optional_float(spec.get("curve_slope_2s10s"))
        if curve_slope_2s10s is None and curve_2y_level is not None and curve_10y_level is not None:
            curve_slope_2s10s = curve_10y_level - curve_2y_level
        has_numeric = any(value is not None for value in (front_end_level, expected_path_level, curve_slope_2s10s))
        stale = bool(updated_at and (now_dt - updated_at) > stale_after)
        if manual_enabled and has_numeric:
            updated_currencies += 1
            if updated_at and (last_update is None or updated_at > last_update):
                last_update = updated_at
        if stale:
            stale_count += 1
        out[currency] = {
            "enabled": manual_enabled,
            "front_end_level": front_end_level,
            "expected_path_level": expected_path_level,
            "curve_2y_level": curve_2y_level,
            "curve_10y_level": curve_10y_level,
            "curve_slope_2s10s": curve_slope_2s10s,
            "stale": stale,
            "last_update_at": isoformat_utc(updated_at) if updated_at else "",
            "basis": str(spec.get("basis", "proxy_only") or "proxy_only"),
        }
    status = {
        "ok": True,
        "stale": bool(updated_currencies > 0 and stale_count >= updated_currencies),
        "enabled": bool(providers.get("manual_inputs_enabled", True)),
        "required": False,
        "last_update_at": isoformat_utc(last_update) if last_update else "",
        "updated_currencies": updated_currencies,
        "coverage_ratio": round(updated_currencies / max(len(SUPPORTED_CURRENCIES), 1), 6),
    }
    return out, status


def _policy_event_windows(items: list[dict[str, Any]], now_dt: datetime, config: dict[str, Any]) -> dict[str, bool]:
    event_cfg = dict(config.get("event_windows", {}))
    pre_cb_event_min = int(event_cfg.get("pre_cb_event_min", 90) or 90)
    post_cb_event_min = int(event_cfg.get("post_cb_event_min", 120) or 120)
    pre_macro_policy_min = int(event_cfg.get("pre_macro_policy_min", 45) or 45)
    post_macro_policy_min = int(event_cfg.get("post_macro_policy_min", 90) or 90)
    flags = {
        "pre_cb_event_window": False,
        "post_cb_event_window": False,
        "pre_macro_policy_window": False,
        "post_macro_policy_window": False,
    }
    for item in items:
        published_at = parse_iso8601(str(item.get("published_at", "")))
        if published_at is None:
            continue
        delta_min = int(round((published_at - now_dt).total_seconds() / 60.0))
        item_is_cb = bool(item.get("central_bank_event", False))
        item_is_macro = bool(item.get("macro_policy_event", False))
        if item_is_cb and delta_min >= 0 and delta_min <= pre_cb_event_min:
            flags["pre_cb_event_window"] = True
        if item_is_cb and delta_min < 0 and abs(delta_min) <= post_cb_event_min:
            flags["post_cb_event_window"] = True
        if item_is_macro and delta_min >= 0 and delta_min <= pre_macro_policy_min:
            flags["pre_macro_policy_window"] = True
        if item_is_macro and delta_min < 0 and abs(delta_min) <= post_macro_policy_min:
            flags["post_macro_policy_window"] = True
    return flags


def _build_currency_state(
    currency: str,
    manual_values: dict[str, Any],
    news_currency: dict[str, Any],
    policy_items: list[dict[str, Any]],
    proxy_snapshot_stale: bool,
    state: dict[str, Any],
    history_snapshots: list[dict[str, Any]],
    config: dict[str, Any],
    now_dt: datetime,
) -> dict[str, Any]:
    thresholds = dict(config.get("thresholds", {}))
    proxy_cfg = dict(config.get("proxy_model", {}))
    providers = dict(config.get("providers", {}))
    proxy_levels = dict(state.get("proxy_levels", {})).get(currency, {})
    decay = _clamp(_safe_float(providers.get("proxy_decay", 0.92), 0.92), 0.50, 0.995)
    front_gain = _safe_float(proxy_cfg.get("front_end_proxy_gain", 0.35), 0.35)
    path_gain = _safe_float(proxy_cfg.get("expected_path_proxy_gain", 0.52), 0.52)
    event_gain = _safe_float(proxy_cfg.get("event_impulse_gain", 0.40), 0.40)
    uncertainty_conflict_gain = _safe_float(proxy_cfg.get("uncertainty_conflict_gain", 0.30), 0.30)

    relevance_scores = [float(item.get("policy_relevance_score", 0.0) or 0.0) for item in policy_items]
    directions = [float(item.get("direction", 0.0) or 0.0) for item in policy_items]
    magnitudes = [abs(float(item.get("magnitude", 0.0) or 0.0)) for item in policy_items]
    hawkish_count = sum(1 for value in directions if value > 0.0)
    dovish_count = sum(1 for value in directions if value < 0.0)
    conflict_score = 1.0 if hawkish_count > 0 and dovish_count > 0 else 0.0
    last_surprise_proxy = _optional_float(news_currency.get("last_surprise_proxy"))
    surprise_direction = _sign(last_surprise_proxy or 0.0)
    surprise_abs = abs(last_surprise_proxy or 0.0)
    windows = _policy_event_windows(policy_items, now_dt, config)

    weighted_direction = 0.0
    if relevance_scores:
        denominator = sum(max(score, 0.01) for score in relevance_scores)
        weighted_direction = sum(direction * max(score, 0.01) for direction, score in zip(directions, relevance_scores)) / max(denominator, 1e-6)
    weighted_magnitude = sum(magnitudes) / max(len(magnitudes), 1)
    burst_score = _safe_float(news_currency.get("burst_score_15m", 0.0))
    story_severity = _safe_float(news_currency.get("story_severity_15m", 0.0))
    official_count = int(news_currency.get("official_count_24h", 0) or 0)
    news_stale = bool(proxy_snapshot_stale or news_currency.get("stale", True))
    news_risk = _safe_float(news_currency.get("risk_score", 0.0))

    policy_relevance_score = _clamp(
        0.44 * max(relevance_scores, default=0.0) +
        0.18 * weighted_magnitude +
        0.14 * surprise_abs +
        0.12 * (1.0 if windows["pre_cb_event_window"] or windows["post_cb_event_window"] else 0.0) +
        0.12 * min(official_count / 2.0, 1.0),
        0.0,
        1.0,
    )
    policy_direction_score = _clamp(
        weighted_direction +
        surprise_direction * surprise_abs * 0.42,
        -2.0,
        2.0,
    )
    policy_repricing_score = _clamp(
        0.36 * abs(policy_direction_score) +
        0.22 * surprise_abs +
        0.18 * min(burst_score / 2.0, 1.0) +
        0.14 * story_severity +
        0.10 * (1.0 if windows["post_cb_event_window"] or windows["post_macro_policy_window"] else 0.0),
        0.0,
        1.0,
    )
    policy_surprise_score = _clamp(
        0.62 * surprise_abs +
        0.18 * abs(weighted_direction) +
        0.20 * (1.0 if windows["post_cb_event_window"] else 0.0),
        0.0,
        1.0,
    )
    policy_uncertainty_score = _clamp(
        0.28 * min(burst_score / 2.0, 1.0) +
        0.22 * story_severity +
        0.18 * conflict_score * uncertainty_conflict_gain +
        0.16 * news_risk +
        0.16 * (1.0 if news_stale else 0.0),
        0.0,
        1.0,
    )
    macro_to_rates_transmission_score = _clamp(
        0.48 * policy_relevance_score +
        0.32 * policy_surprise_score +
        0.20 * (1.0 if windows["post_macro_policy_window"] else 0.0),
        0.0,
        1.0,
    )
    meeting_path_reprice_now = bool(
        windows["post_cb_event_window"] and (
            policy_repricing_score >= _safe_float(thresholds.get("meeting_path_reprice_now", 0.60), 0.60)
            or policy_surprise_score >= _safe_float(thresholds.get("policy_surprise_high", 0.55), 0.55)
        )
    )

    previous_front = _safe_float(proxy_levels.get("front_end_proxy_level", 0.0))
    previous_path = _safe_float(proxy_levels.get("expected_path_proxy_level", 0.0))
    front_proxy_level = _clamp(previous_front * decay + policy_direction_score * front_gain + surprise_direction * surprise_abs * event_gain, -4.0, 4.0)
    expected_proxy_level = _clamp(previous_path * decay + policy_direction_score * path_gain + surprise_direction * surprise_abs * event_gain, -5.0, 5.0)

    front_end_level = manual_values.get("front_end_level")
    front_end_basis = "manual_market_input" if front_end_level is not None else "policy_proxy_index"
    if front_end_level is None:
        front_end_level = front_proxy_level

    expected_path_level = manual_values.get("expected_path_level")
    expected_path_basis = "manual_market_input" if expected_path_level is not None else "policy_proxy_index"
    if expected_path_level is None:
        expected_path_level = expected_proxy_level

    curve_slope_2s10s = manual_values.get("curve_slope_2s10s")
    curve_basis = "manual_market_input" if curve_slope_2s10s is not None else "unavailable"

    history_1d_front = _historical_currency_value(history_snapshots, currency, "front_end_level", 24, now_dt)
    history_5d_front = _historical_currency_value(history_snapshots, currency, "front_end_level", 24 * 5, now_dt)
    history_1d_path = _historical_currency_value(history_snapshots, currency, "expected_path_level", 24, now_dt)
    history_5d_path = _historical_currency_value(history_snapshots, currency, "expected_path_level", 24 * 5, now_dt)
    history_1d_slope = _historical_currency_value(history_snapshots, currency, "curve_slope_2s10s", 24, now_dt)
    history_5d_slope = _historical_currency_value(history_snapshots, currency, "curve_slope_2s10s", 24 * 5, now_dt)

    front_end_change_1d = None if history_1d_front is None else round(front_end_level - history_1d_front, 6)
    front_end_change_5d = None if history_5d_front is None else round(front_end_level - history_5d_front, 6)
    expected_path_change_1d = None if history_1d_path is None else round(expected_path_level - history_1d_path, 6)
    expected_path_change_5d = None if history_5d_path is None else round(expected_path_level - history_5d_path, 6)
    curve_slope_change_1d = None if curve_slope_2s10s is None or history_1d_slope is None else round(curve_slope_2s10s - history_1d_slope, 6)
    curve_slope_change_5d = None if curve_slope_2s10s is None or history_5d_slope is None else round(curve_slope_2s10s - history_5d_slope, 6)
    curve_shape_regime = _curve_shape_regime(curve_slope_2s10s, curve_slope_change_1d, config)

    reasons: list[str] = []
    if windows["pre_cb_event_window"]:
        _append_reason(reasons, f"{currency} central-bank event window active")
    if windows["post_cb_event_window"]:
        _append_reason(reasons, f"{currency} central-bank repricing window active")
    if windows["pre_macro_policy_window"]:
        _append_reason(reasons, f"{currency} macro-policy event window active")
    if windows["post_macro_policy_window"]:
        _append_reason(reasons, f"{currency} post-macro rates transmission active")
    if policy_surprise_score >= _safe_float(thresholds.get("policy_surprise_high", 0.55), 0.55):
        _append_reason(reasons, f"{currency} policy surprise elevated")
    if policy_uncertainty_score >= _safe_float(thresholds.get("policy_uncertainty_caution", 0.48), 0.48):
        _append_reason(reasons, f"{currency} policy uncertainty elevated")
    if manual_values.get("front_end_level") is None and manual_values.get("expected_path_level") is None:
        _append_reason(reasons, f"{currency} running in proxy policy-path mode")
    if news_stale:
        _append_reason(reasons, f"{currency} NewsPulse context stale")

    stale = bool(news_stale and manual_values.get("front_end_level") is None and manual_values.get("expected_path_level") is None)
    return {
        "front_end_level": round(front_end_level, 6) if front_end_level is not None else None,
        "front_end_basis": front_end_basis,
        "front_end_change_1d": front_end_change_1d,
        "front_end_change_5d": front_end_change_5d,
        "expected_path_level": round(expected_path_level, 6) if expected_path_level is not None else None,
        "expected_path_basis": expected_path_basis,
        "expected_path_change_1d": expected_path_change_1d,
        "expected_path_change_5d": expected_path_change_5d,
        "curve_slope_2s10s": round(curve_slope_2s10s, 6) if curve_slope_2s10s is not None else None,
        "curve_basis": curve_basis,
        "curve_slope_change_1d": curve_slope_change_1d,
        "curve_slope_change_5d": curve_slope_change_5d,
        "curve_shape_regime": curve_shape_regime,
        "policy_repricing_score": round(policy_repricing_score, 6),
        "policy_surprise_score": round(policy_surprise_score, 6),
        "policy_uncertainty_score": round(policy_uncertainty_score, 6),
        "policy_direction_score": round(policy_direction_score, 6),
        "policy_relevance_score": round(policy_relevance_score, 6),
        "pre_cb_event_window": windows["pre_cb_event_window"],
        "post_cb_event_window": windows["post_cb_event_window"],
        "pre_macro_policy_window": windows["pre_macro_policy_window"],
        "post_macro_policy_window": windows["post_macro_policy_window"],
        "meeting_path_reprice_now": meeting_path_reprice_now,
        "macro_to_rates_transmission_score": round(macro_to_rates_transmission_score, 6),
        "stale": stale,
        "reasons": reasons[:6],
        "_proxy_front_end_level": round(front_proxy_level, 6),
        "_proxy_expected_path_level": round(expected_proxy_level, 6),
    }


def _pair_symbol_universe(config: dict[str, Any], policy: dict[str, Any]) -> list[str]:
    out = set(active_pairs(policy))
    configured_currencies = [code for code in SUPPORTED_CURRENCIES if code in dict(config.get("currencies", {}))]
    for base in configured_currencies:
        for quote in configured_currencies:
            if base == quote:
                continue
            out.add(base + quote)
    return sorted(out)


def _build_pair_state(
    pair_id: str,
    currencies: dict[str, dict[str, Any]],
    policy: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    thresholds = dict(config.get("thresholds", {}))
    pair_cfg = dict(config.get("pair_thresholds", {}))
    base, quote = _pair_legs(pair_id)
    base_state = currencies.get(base, {})
    quote_state = currencies.get(quote, {})
    if not base_state or not quote_state:
        return {}

    front_end_diff = None
    if base_state.get("front_end_level") is not None and quote_state.get("front_end_level") is not None:
        front_end_diff = round(float(base_state["front_end_level"]) - float(quote_state["front_end_level"]), 6)
    expected_path_diff = None
    if base_state.get("expected_path_level") is not None and quote_state.get("expected_path_level") is not None:
        expected_path_diff = round(float(base_state["expected_path_level"]) - float(quote_state["expected_path_level"]), 6)
    front_end_diff_change_1d = None
    if base_state.get("front_end_change_1d") is not None and quote_state.get("front_end_change_1d") is not None:
        front_end_diff_change_1d = round(float(base_state["front_end_change_1d"]) - float(quote_state["front_end_change_1d"]), 6)
    front_end_diff_change_5d = None
    if base_state.get("front_end_change_5d") is not None and quote_state.get("front_end_change_5d") is not None:
        front_end_diff_change_5d = round(float(base_state["front_end_change_5d"]) - float(quote_state["front_end_change_5d"]), 6)
    expected_path_diff_change_1d = None
    if base_state.get("expected_path_change_1d") is not None and quote_state.get("expected_path_change_1d") is not None:
        expected_path_diff_change_1d = round(float(base_state["expected_path_change_1d"]) - float(quote_state["expected_path_change_1d"]), 6)
    expected_path_diff_change_5d = None
    if base_state.get("expected_path_change_5d") is not None and quote_state.get("expected_path_change_5d") is not None:
        expected_path_diff_change_5d = round(float(base_state["expected_path_change_5d"]) - float(quote_state["expected_path_change_5d"]), 6)

    base_curve = _optional_float(base_state.get("curve_slope_2s10s"))
    quote_curve = _optional_float(quote_state.get("curve_slope_2s10s"))
    curve_divergence_score = 0.0
    if base_curve is not None and quote_curve is not None:
        curve_divergence_score = _clamp(abs(base_curve - quote_curve), 0.0, 2.0) / 2.0

    policy_divergence_score = _clamp(
        0.42 * abs(_safe_float(base_state.get("policy_direction_score")) - _safe_float(quote_state.get("policy_direction_score"))) +
        0.32 * abs(_safe_float(base_state.get("expected_path_level")) - _safe_float(quote_state.get("expected_path_level"))) +
        0.26 * abs(_safe_float(base_state.get("policy_repricing_score")) - _safe_float(quote_state.get("policy_repricing_score"))),
        0.0,
        1.0,
    )
    rates_risk_score = _clamp(
        _safe_float(pair_cfg.get("uncertainty_weight", 0.38)) * max(_safe_float(base_state.get("policy_uncertainty_score")), _safe_float(quote_state.get("policy_uncertainty_score"))) +
        _safe_float(pair_cfg.get("repricing_weight", 0.22)) * max(_safe_float(base_state.get("policy_repricing_score")), _safe_float(quote_state.get("policy_repricing_score"))) +
        _safe_float(pair_cfg.get("divergence_weight", 0.24)) * policy_divergence_score +
        _safe_float(pair_cfg.get("event_weight", 0.16)) * (1.0 if (
            bool(base_state.get("meeting_path_reprice_now", False)) or bool(quote_state.get("meeting_path_reprice_now", False))
        ) else 0.0),
        0.0,
        1.0,
    )
    stale = bool(base_state.get("stale", True) or quote_state.get("stale", True))
    meeting_path_reprice_now = bool(base_state.get("meeting_path_reprice_now", False) or quote_state.get("meeting_path_reprice_now", False))
    macro_to_rates_transmission_score = round(
        max(_safe_float(base_state.get("macro_to_rates_transmission_score", 0.0)),
            _safe_float(quote_state.get("macro_to_rates_transmission_score", 0.0))),
        6,
    )
    reasons: list[str] = []
    if stale:
        _append_reason(reasons, "rates state stale or incomplete")
    if meeting_path_reprice_now:
        _append_reason(reasons, "meeting path repricing active")
    if rates_risk_score >= _safe_float(thresholds.get("rates_risk_caution", 0.44), 0.44):
        _append_reason(reasons, "rates risk elevated")
    if policy_divergence_score >= _safe_float(thresholds.get("policy_divergence_meaningful", 0.42), 0.42):
        _append_reason(reasons, "policy divergence meaningful")
    if curve_divergence_score >= _safe_float(thresholds.get("curve_divergence_meaningful", 0.35), 0.35):
        _append_reason(reasons, "curve divergence meaningful")

    policy_alignment = "balanced"
    if expected_path_diff is not None and expected_path_diff > 0.0:
        policy_alignment = "base_hawkish"
    elif expected_path_diff is not None and expected_path_diff < 0.0:
        policy_alignment = "quote_hawkish"

    if stale and bool(pair_cfg.get("stale_block", True)):
        trade_gate = "BLOCK"
        rates_regime = "UNSTABLE"
    elif meeting_path_reprice_now or rates_risk_score >= _safe_float(thresholds.get("rates_risk_block", 0.78), 0.78):
        trade_gate = "BLOCK"
        rates_regime = "UNSTABLE"
    elif rates_risk_score >= _safe_float(thresholds.get("rates_risk_caution", 0.44), 0.44):
        trade_gate = "CAUTION"
        rates_regime = "CONFLICTING" if policy_divergence_score < _safe_float(pair_cfg.get("conflicting_caution_floor", 0.34), 0.34) else "UNSTABLE"
    elif policy_divergence_score >= _safe_float(thresholds.get("policy_divergence_meaningful", 0.42), 0.42):
        trade_gate = "ALLOW"
        rates_regime = "SUPPORTIVE"
    else:
        trade_gate = "ALLOW"
        rates_regime = "NEUTRAL"

    return {
        "base_currency": base,
        "quote_currency": quote,
        "front_end_diff": front_end_diff,
        "front_end_diff_change_1d": front_end_diff_change_1d,
        "front_end_diff_change_5d": front_end_diff_change_5d,
        "expected_path_diff": expected_path_diff,
        "expected_path_diff_change_1d": expected_path_diff_change_1d,
        "expected_path_diff_change_5d": expected_path_diff_change_5d,
        "curve_divergence_score": round(curve_divergence_score, 6),
        "policy_divergence_score": round(policy_divergence_score, 6),
        "rates_regime": rates_regime,
        "rates_risk_score": round(rates_risk_score, 6),
        "trade_gate": trade_gate,
        "policy_alignment": policy_alignment,
        "meeting_path_reprice_now": meeting_path_reprice_now,
        "macro_to_rates_transmission_score": macro_to_rates_transmission_score,
        "stale": stale,
        "reasons": reasons[:6],
        "broker_symbols": broker_symbols_for_pair(pair_id, policy),
    }


def _write_flat_snapshot(snapshot: dict[str, Any]) -> None:
    lines: list[str] = []
    generated_at = parse_iso8601(str(snapshot.get("generated_at", "")))
    generated_unix = int(generated_at.timestamp()) if generated_at is not None else 0
    lines.append(f"meta\tglobal\tgenerated_at_unix\t{generated_unix}")
    for source_name, status in dict(snapshot.get("source_status", {})).items():
        if not isinstance(status, dict):
            continue
        for key, value in status.items():
            if isinstance(value, bool):
                text = "1" if value else "0"
            else:
                text = "" if value is None else str(value)
            lines.append(f"source\t{source_name}\t{key}\t{text}")
    for currency, state in dict(snapshot.get("currencies", {})).items():
        if not isinstance(state, dict):
            continue
        for key, value in state.items():
            if key.startswith("_") or key == "reasons":
                continue
            if isinstance(value, bool):
                text = "1" if value else "0"
            else:
                text = "" if value is None else str(value)
            lines.append(f"currency\t{currency}\t{key}\t{text}")
        for reason in list(state.get("reasons", []))[:6]:
            lines.append(f"currency_reason\t{currency}\treason\t{str(reason)}")
    for pair_id, state in dict(snapshot.get("pairs", {})).items():
        if not isinstance(state, dict):
            continue
        for key, value in state.items():
            if key == "reasons":
                continue
            if isinstance(value, list):
                text = ",".join(str(item) for item in value)
            elif isinstance(value, bool):
                text = "1" if value else "0"
            else:
                text = "" if value is None else str(value)
            lines.append(f"pair\t{pair_id}\t{key}\t{text}")
        for reason in list(state.get("reasons", []))[:6]:
            lines.append(f"pair_reason\t{pair_id}\treason\t{str(reason)}")
    COMMON_RATES_FLAT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_symbol_map(policy: dict[str, Any], pairs: dict[str, Any]) -> None:
    lines: list[str] = []
    for pair_id in sorted(pairs.keys()):
        lines.append(f"symbol\t{pair_id}\t{pair_id}")
        for broker_symbol in broker_symbols_for_pair(pair_id, policy):
            lines.append(f"symbol\t{broker_symbol}\t{pair_id}")
    COMMON_RATES_SYMBOL_MAP.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_rates_engine_cycle(daemon_context: dict[str, Any] | None = None) -> dict[str, Any]:
    ensure_rates_engine_dirs()
    now_dt = utc_now()
    now_iso = isoformat_utc(now_dt)
    config = load_config()
    inputs = load_inputs()
    policy = load_policy()
    state = _load_state()
    history_snapshots = _recent_history_snapshots(24 * 8)
    news_snapshot, news_source_status = _load_newspulse_snapshot(now_dt)
    news_currencies = _iter_currency_state(news_snapshot)
    recent_items = _iter_recent_items(news_snapshot)

    manual_values, manual_status = _manual_currency_values(inputs, config, now_dt)
    proxy_levels = dict(state.get("proxy_levels", {}))
    currencies: dict[str, dict[str, Any]] = {}
    recent_policy_events: list[dict[str, Any]] = []
    snapshot_stale_after_sec = int(config.get("snapshot_stale_after_sec", 900) or 900)
    snapshot_generated_at = parse_iso8601(str(news_snapshot.get("generated_at", "")))
    proxy_ok = bool(news_snapshot) and not bool(news_snapshot.get("stale", False))
    if snapshot_generated_at is None or (now_dt - snapshot_generated_at).total_seconds() > snapshot_stale_after_sec:
        proxy_ok = False

    for currency in SUPPORTED_CURRENCIES:
        news_currency = dict(news_currencies.get(currency, {}))
        currency_items = []
        for item in recent_items:
            currency_tags = {str(code).upper() for code in list(item.get("currency_tags", []))}
            if currency not in currency_tags:
                continue
            signal = _item_policy_signal(item, currency, config, news_currency)
            if signal["policy_relevance_score"] <= 0.0:
                continue
            currency_items.append(signal)
            if signal["central_bank_event"] or signal["macro_policy_event"]:
                recent_policy_events.append(
                    {
                        "id": signal["item_id"],
                        "currency": currency,
                        "source": signal["source"],
                        "domain": signal["domain"],
                        "published_at": signal["published_at"],
                        "title": signal["title"],
                        "url": signal["url"],
                        "policy_relevance_score": signal["policy_relevance_score"],
                        "direction": signal["direction"],
                        "central_bank_event": signal["central_bank_event"],
                        "macro_policy_event": signal["macro_policy_event"],
                    }
                )
        currency_state = _build_currency_state(
            currency,
            manual_values.get(currency, {}),
            news_currency,
            currency_items,
            not proxy_ok,
            state,
            history_snapshots,
            config,
            now_dt,
        )
        proxy_levels[currency] = {
            "front_end_proxy_level": currency_state.pop("_proxy_front_end_level"),
            "expected_path_proxy_level": currency_state.pop("_proxy_expected_path_level"),
            "updated_at": now_iso,
        }
        currencies[currency] = currency_state

    pairs: dict[str, dict[str, Any]] = {}
    for pair_id in _pair_symbol_universe(config, policy):
        state_payload = _build_pair_state(pair_id, currencies, policy, config)
        if state_payload:
            pairs[pair_id] = state_payload

    source_status = {
        "manual_inputs": manual_status,
        "proxy_engine": {
            "ok": proxy_ok,
            "stale": not proxy_ok,
            "enabled": True,
            "required": False,
            "last_update_at": str(news_snapshot.get("generated_at", "")),
            "mode": "newspulse_policy_proxy",
            "currencies_with_news": sum(1 for state in currencies.values() if not state.get("stale", True)),
        },
        "newspulse": {
            "ok": bool(news_snapshot),
            "stale": bool(news_snapshot.get("stale", True)) if news_snapshot else True,
            "enabled": True,
            "required": False,
            "last_update_at": str(news_snapshot.get("generated_at", "")),
            "sources": news_source_status,
        },
    }

    snapshot = {
        "schema_version": RATES_ENGINE_SCHEMA_VERSION,
        "generated_at": now_iso,
        "source_status": source_status,
        "currencies": currencies,
        "pairs": pairs,
        "recent_policy_events": sorted(
            recent_policy_events,
            key=lambda item: str(item.get("published_at", "")),
            reverse=True,
        )[:32],
        "provider_notes": {
            "front_end_level": "Uses operator-supplied numeric levels when available, otherwise a policy-proxy index derived from NewsPulse event and official-feed context.",
            "expected_path_level": "Uses operator-supplied numeric path levels when available, otherwise a rolling path-proxy index.",
            "curve_slope_2s10s": "Only populated from operator-supplied numeric curve inputs in phase 1.",
        },
    }
    json_dump(COMMON_RATES_JSON, snapshot)
    _write_flat_snapshot(snapshot)
    _write_symbol_map(policy, pairs)

    ndjson_append(COMMON_RATES_HISTORY, {"record_type": "snapshot", "generated_at": now_iso, "snapshot": snapshot})
    ndjson_append(RATES_ENGINE_LOCAL_HISTORY_PATH, {"record_type": "snapshot", "generated_at": now_iso, "snapshot": snapshot})
    replay_report = build_rates_replay_report(hours_back=72)

    status_payload = {
        **snapshot,
        "daemon": dict(daemon_context or state.get("daemon_health", {})),
        "health": {
            "required_sources_stale": bool(source_status["manual_inputs"]["stale"] and source_status["proxy_engine"]["stale"]),
            "history_records_local": len(
                RATES_ENGINE_LOCAL_HISTORY_PATH.read_text(encoding="utf-8", errors="replace").splitlines()
            ) if RATES_ENGINE_LOCAL_HISTORY_PATH.exists() else 0,
            "pair_count": len(pairs),
            "currency_count": len(currencies),
            "snapshot_stale_after_sec": snapshot_stale_after_sec,
        },
        "replay": replay_report,
        "artifacts": {
            "snapshot_json": str(COMMON_RATES_JSON),
            "snapshot_flat": str(COMMON_RATES_FLAT),
            "history_ndjson": str(RATES_ENGINE_LOCAL_HISTORY_PATH),
            "status_json": str(RATES_ENGINE_STATUS_PATH),
            "replay_report_json": str(replay_report.get("report_path", "")),
            "symbol_map_tsv": str(COMMON_RATES_SYMBOL_MAP),
        },
    }
    json_dump(RATES_ENGINE_STATUS_PATH, status_payload)

    state["proxy_levels"] = proxy_levels
    state["last_generated_at"] = now_iso
    if daemon_context is not None:
        state["daemon_health"] = dict(daemon_context)
    _save_state(state)
    return {
        "snapshot_path": str(COMMON_RATES_JSON),
        "flat_path": str(COMMON_RATES_FLAT),
        "history_path": str(RATES_ENGINE_LOCAL_HISTORY_PATH),
        "pair_count": len(pairs),
        "currency_count": len(currencies),
        "proxy_ok": proxy_ok,
        "manual_inputs_used": int(manual_status.get("updated_currencies", 0) or 0),
    }
