from __future__ import annotations

import math
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any

from .newspulse_calendar import calendar_source_status, load_calendar_records
from .newspulse_config import build_query_specs, load_config
from .newspulse_contracts import (
    COMMON_NEWSPULSE_FLAT,
    COMMON_NEWSPULSE_JSON,
    COMMON_NEWSPULSE_REPLAY_FLAT,
    COMMON_NEWSPULSE_SYMBOL_MAP,
    NEWSPULSE_CONFIG_PATH,
    NEWSPULSE_LOCAL_HISTORY_PATH,
    NEWSPULSE_POLICY_PATH,
    NEWSPULSE_REPLAY_REPORT_PATH,
    NEWSPULSE_SCHEMA_VERSION,
    NEWSPULSE_SOURCES_PATH,
    NEWSPULSE_STATE_PATH,
    NEWSPULSE_STATUS_PATH,
    ensure_newspulse_dirs,
    isoformat_utc,
    json_dump,
    parse_iso8601,
    sanitize_utc_timestamp,
    utc_now,
)
from .newspulse_gdelt import query_gdelt
from .newspulse_official import query_official_feeds
from .newspulse_policy import active_pairs, broker_symbols_for_pair, load_policy, pair_calibration, watchlist_tags_for_pair
from .newspulse_replay import (
    build_replay_report,
    update_pair_gate_history,
    update_source_health_history,
    write_history_mirror,
    write_replay_artifacts,
)
from .newspulse_story import build_story_clusters
from .rates_engine_newspulse import apply_rates_enrichment


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _pair_universe(currencies: list[str], include_permutations: bool) -> list[str]:
    pairs: list[str] = []
    for base in currencies:
        for quote in currencies:
            if base == quote:
                continue
            if not include_permutations and base > quote:
                continue
            pairs.append(base + quote)
    return sorted(set(pairs))


def _state_timestamp_text(value: str | None, now_dt: datetime) -> str:
    dt = sanitize_utc_timestamp(value, now_dt=now_dt)
    return isoformat_utc(dt) if dt is not None else ""


def _empty_state(now_dt: datetime | None = None) -> dict[str, Any]:
    reference = now_dt or utc_now()
    return {
        "last_gdelt_success_at": "",
        "last_gdelt_poll_at": "",
        "gdelt_backoff_until": "",
        "last_official_success_at": "",
        "last_official_poll_at": "",
        "seen_items": {},
        "currency_baselines": {},
        "query_rotation_index": 0,
        "pair_gate_history": {},
        "source_health_history": [],
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
            "backoff_until": "",
            "last_error": "",
        },
    }


def _load_state() -> dict[str, Any]:
    now_dt = utc_now()
    if not NEWSPULSE_STATE_PATH.exists():
        return _empty_state(now_dt)
    try:
        import json

        payload = json.loads(NEWSPULSE_STATE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return _empty_state(now_dt)
    if not isinstance(payload, dict):
        return _empty_state(now_dt)

    state = _empty_state(now_dt)
    state.update(payload)
    state.setdefault("seen_items", {})
    state.setdefault("currency_baselines", {})
    state.setdefault("query_rotation_index", 0)
    state.setdefault("pair_gate_history", {})
    state.setdefault("source_health_history", [])
    state.setdefault("daemon_health", _empty_state(now_dt)["daemon_health"])
    state["last_gdelt_success_at"] = _state_timestamp_text(state.get("last_gdelt_success_at"), now_dt)
    state["last_gdelt_poll_at"] = _state_timestamp_text(state.get("last_gdelt_poll_at"), now_dt)
    state["gdelt_backoff_until"] = _state_timestamp_text(state.get("gdelt_backoff_until"), now_dt)
    state["last_official_success_at"] = _state_timestamp_text(state.get("last_official_success_at"), now_dt)
    state["last_official_poll_at"] = _state_timestamp_text(state.get("last_official_poll_at"), now_dt)
    daemon = dict(state.get("daemon_health", {}))
    daemon["heartbeat_at"] = _state_timestamp_text(daemon.get("heartbeat_at"), now_dt)
    daemon["last_cycle_started_at"] = _state_timestamp_text(daemon.get("last_cycle_started_at"), now_dt)
    daemon["last_cycle_finished_at"] = _state_timestamp_text(daemon.get("last_cycle_finished_at"), now_dt)
    daemon["backoff_until"] = _state_timestamp_text(daemon.get("backoff_until"), now_dt)
    state["daemon_health"] = daemon
    return state


def _save_state(payload: dict[str, Any]) -> None:
    json_dump(NEWSPULSE_STATE_PATH, payload)


def _prune_seen_items(seen_items: dict[str, Any], now_dt: datetime, max_age_hours: int = 72) -> dict[str, Any]:
    out: dict[str, Any] = {}
    cutoff = now_dt - timedelta(hours=max_age_hours)
    for key, value in seen_items.items():
        if not isinstance(value, dict):
            continue
        published_at = parse_iso8601(str(value.get("published_at", "")))
        seen_at = parse_iso8601(str(value.get("seen_at", "")))
        if published_at and published_at >= cutoff:
            out[key] = value
            continue
        if seen_at and seen_at >= cutoff:
            out[key] = value
    return out


def _select_query_specs(config: dict[str, Any],
                        state: dict[str, Any],
                        query_specs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not query_specs:
        state["query_rotation_index"] = 0
        return []

    limit = int(config.get("gdelt", {}).get("max_query_sets_per_cycle", len(query_specs)) or len(query_specs))
    limit = max(1, min(limit, len(query_specs)))
    if limit >= len(query_specs):
        state["query_rotation_index"] = 0
        return query_specs

    start = int(state.get("query_rotation_index", 0) or 0) % len(query_specs)
    selected = [query_specs[(start + offset) % len(query_specs)] for offset in range(limit)]
    state["query_rotation_index"] = (start + limit) % len(query_specs)
    return selected


def _weight_item(item: dict[str, Any], now_dt: datetime, half_life_min: float) -> float:
    published_at = parse_iso8601(str(item.get("published_at", "")))
    if not published_at:
        return 0.0
    age_min = max((now_dt - published_at).total_seconds() / 60.0, 0.0)
    decay = math.exp(-math.log(2.0) * age_min / max(half_life_min, 0.1))
    tone = abs(_safe_float(item.get("tone", 0.0)))
    tone_weight = 1.0 + min(tone / 10.0, 0.5)
    source_boost = 1.20 if str(item.get("source", "")) == "official" else 1.0
    return (
        _safe_float(item.get("tier_weight", 0.0)) *
        _safe_float(item.get("topic_weight", 1.0), 1.0) *
        decay * tone_weight * source_boost
    )


def _calendar_currency_state(config: dict[str, Any], now_dt: datetime) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    calendar_cfg = config["calendar"]
    pre_windows = {int(key): int(value) for key, value in dict(calendar_cfg["pre_window_min_by_importance"]).items()}
    post_windows = {int(key): int(value) for key, value in dict(calendar_cfg["post_window_min_by_importance"]).items()}
    high_impact_min = int(calendar_cfg.get("high_impact_min", 2) or 2)

    per_currency: dict[str, dict[str, Any]] = defaultdict(lambda: {
        "next_high_impact_eta_min": None,
        "time_since_last_high_impact_min": None,
        "in_pre_event_window": False,
        "in_post_event_window": False,
        "last_surprise_proxy": None,
        "calendar_risk": 0.0,
        "reasons": [],
    })
    recent_items: list[dict[str, Any]] = []
    for record in load_calendar_records():
        if not record.currency:
            continue
        currency_state = per_currency[record.currency]
        event_dt = datetime.fromtimestamp(record.event_time_utc_unix, tz=timezone.utc)
        eta_min = int(round((event_dt - now_dt).total_seconds() / 60.0))
        since_last_min = int(round((now_dt - event_dt).total_seconds() / 60.0))
        pre_window = pre_windows.get(record.importance, pre_windows.get(3, 45))
        post_window = post_windows.get(record.importance, post_windows.get(3, 45))
        if record.importance >= high_impact_min and eta_min >= 0:
            current_eta = currency_state["next_high_impact_eta_min"]
            if current_eta is None or eta_min < current_eta:
                currency_state["next_high_impact_eta_min"] = eta_min
            if eta_min <= pre_window:
                currency_state["in_pre_event_window"] = True
                imminence = 1.0 - (_clamp(float(eta_min), 0.0, float(pre_window)) / max(float(pre_window), 1.0))
                currency_state["calendar_risk"] = max(currency_state["calendar_risk"], 0.68 + 0.24 * imminence)
                currency_state["reasons"].append(f"{record.currency} high-impact event in {eta_min} minutes")
        if record.importance >= high_impact_min and since_last_min >= 0:
            current_since = currency_state["time_since_last_high_impact_min"]
            if current_since is None or since_last_min < current_since:
                currency_state["time_since_last_high_impact_min"] = since_last_min
            if since_last_min <= post_window:
                currency_state["in_post_event_window"] = True
                hot_minutes = int(config["scoring"].get("post_release_hot_minutes", 8) or 8)
                hotness = 1.0 - (_clamp(float(since_last_min), 0.0, float(max(post_window, 1))) / max(float(post_window), 1.0))
                risk_floor = 0.84 if since_last_min <= hot_minutes else 0.70
                currency_state["calendar_risk"] = max(currency_state["calendar_risk"], risk_floor + 0.12 * hotness)
                currency_state["reasons"].append(f"{record.currency} high-impact event printed {since_last_min} minutes ago")
                if record.surprise_proxy is not None:
                    currency_state["last_surprise_proxy"] = record.surprise_proxy

        recent_items.append(
            {
                "id": record.event_key,
                "source": "calendar",
                "published_at": isoformat_utc(event_dt),
                "seen_at": isoformat_utc(datetime.fromtimestamp(record.collector_seen_utc_unix, tz=timezone.utc)),
                "currency_tags": [record.currency],
                "topic_tags": ["scheduled_macro"],
                "domain": "mt5-calendar",
                "title": record.title,
                "url": "",
                "importance": record.importance,
                "tone": 0.0,
                "published_at_trade_server": record.event_time_trade_server,
                "seen_at_trade_server": record.collector_seen_trade_server,
                "time_basis": "trade_server",
            }
        )
    return per_currency, recent_items


def _merge_seen_items(existing_state: dict[str, Any],
                      fresh_items: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    seen_items = dict(existing_state.get("seen_items", {}))
    appended_history: list[dict[str, Any]] = []
    for item in fresh_items:
        item_id = str(item.get("id", ""))
        if not item_id:
            continue
        stored = seen_items.get(item_id)
        if not isinstance(stored, dict):
            stored = dict(item)
            appended_history.append({"record_type": "item", "item": dict(item)})
        else:
            merged = dict(stored)
            merged.update({key: value for key, value in item.items() if value not in ("", None, [])})
            merged["currency_tags"] = list(dict.fromkeys(list(stored.get("currency_tags", [])) + list(item.get("currency_tags", []))))
            merged["topic_tags"] = list(dict.fromkeys(list(stored.get("topic_tags", [])) + list(item.get("topic_tags", []))))
            stored = merged
        stored["seen_at"] = str(item.get("seen_at", stored.get("seen_at", "")))
        seen_items[item_id] = stored
    merged_items = []
    for item_id, stored in seen_items.items():
        if not isinstance(stored, dict):
            continue
        payload = dict(stored)
        payload["id"] = item_id
        merged_items.append(payload)
    return merged_items, seen_items, appended_history


def _news_currency_state(config: dict[str, Any],
                         state: dict[str, Any],
                         news_items: list[dict[str, Any]],
                         stories: list[dict[str, Any]],
                         now_dt: datetime) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    scoring = config["scoring"]
    half_life_min = float(scoring.get("intensity_half_life_min", 12.0) or 12.0)
    baseline_alpha = float(scoring.get("baseline_alpha", 0.18) or 0.18)
    currencies = list(config["currencies"].keys())
    window_cutoff = now_dt - timedelta(minutes=15)
    day_cutoff = now_dt - timedelta(hours=24)
    state.setdefault("currency_baselines", {})
    per_currency: dict[str, dict[str, Any]] = {}
    for currency in currencies:
        relevant = [item for item in news_items if currency in item.get("currency_tags", [])]
        recent = [item for item in relevant if (parse_iso8601(item.get("published_at")) or now_dt) >= window_cutoff]
        daily = [item for item in relevant if (parse_iso8601(item.get("published_at")) or now_dt) >= day_cutoff]
        weighted_items = [_weight_item(item, now_dt, half_life_min) for item in recent]
        weighted_recent = [pair for pair in zip(recent, weighted_items) if pair[1] > 0.0]
        intensity = sum(weight for _, weight in weighted_recent)
        count = len(recent)
        tone_mean = 0.0
        tone_abs_mean = 0.0
        if weighted_recent:
            total_weight = sum(weight for _, weight in weighted_recent) or 1.0
            tone_mean = sum(_safe_float(item.get("tone", 0.0)) * weight for item, weight in weighted_recent) / total_weight
            tone_abs_mean = sum(abs(_safe_float(item.get("tone", 0.0))) * weight for item, weight in weighted_recent) / total_weight
        baseline_entry = state["currency_baselines"].get(currency, {})
        baseline = _safe_float(baseline_entry.get("ema_intensity", intensity or 0.5), intensity or 0.5)
        if baseline <= 0.05:
            baseline = 0.5
        burst_score = _clamp((intensity / max(baseline, 0.05)) - 1.0, 0.0, 4.0)
        state["currency_baselines"][currency] = {
            "ema_intensity": (1.0 - baseline_alpha) * baseline + baseline_alpha * intensity,
            "updated_at": isoformat_utc(now_dt),
        }
        recent_stories = []
        for story in stories:
            if currency not in story.get("currency_tags", []):
                continue
            last_dt = parse_iso8601(str(story.get("last_published_at", "")))
            if last_dt is None or last_dt < window_cutoff:
                continue
            recent_stories.append(story)
        story_severity = max((float(story.get("severity_score", 0.0) or 0.0) for story in recent_stories), default=0.0)
        official_count = sum(1 for item in daily if str(item.get("source", "")) == "official")
        dominant_story_ids = [str(story.get("id", "")) for story in recent_stories[:3] if str(story.get("id", ""))]
        per_currency[currency] = {
            "breaking_count_15m": count,
            "intensity_15m": round(intensity, 6),
            "tone_mean_15m": round(tone_mean, 6),
            "tone_abs_mean_15m": round(tone_abs_mean, 6),
            "burst_score_15m": round(burst_score, 6),
            "story_count_15m": len(recent_stories),
            "story_severity_15m": round(story_severity, 6),
            "official_count_24h": official_count,
            "dominant_story_ids": dominant_story_ids,
            "gdelt_risk": _clamp(
                0.16 * burst_score +
                0.08 * min(tone_abs_mean / 4.0, 1.0) +
                0.26 * story_severity +
                0.14 * min(float(official_count), 2.0),
                0.0,
                0.88,
            ),
            "reasons": [],
        }
        if burst_score >= 1.0:
            per_currency[currency]["reasons"].append(f"{currency} burst score elevated at {burst_score:.2f}")
        if story_severity >= 0.45 and recent_stories:
            per_currency[currency]["reasons"].append(f"{currency} evolving story severity {story_severity:.2f}")
        if official_count > 0:
            per_currency[currency]["reasons"].append(f"{currency} official feed activity in the last 24h")
    return per_currency, state


def _required_stale(source_status: dict[str, Any]) -> bool:
    for status in source_status.values():
        if not isinstance(status, dict):
            continue
        if bool(status.get("required", False)) and bool(status.get("stale", True)):
            return True
    return False


def _combine_currency_state(config: dict[str, Any],
                            news_currency: dict[str, dict[str, Any]],
                            calendar_currency: dict[str, dict[str, Any]],
                            source_status: dict[str, Any]) -> dict[str, dict[str, Any]]:
    stale_sources = _required_stale(source_status)
    stale_score = float(config["scoring"].get("stale_risk_score", 0.92) or 0.92)
    out: dict[str, dict[str, Any]] = {}
    for currency in sorted(set(news_currency) | set(calendar_currency)):
        news_state = news_currency.get(currency, {})
        calendar = calendar_currency.get(currency, {})
        reasons = list(dict.fromkeys(list(calendar.get("reasons", [])) + list(news_state.get("reasons", []))))
        risk = max(_safe_float(news_state.get("gdelt_risk", 0.0)), _safe_float(calendar.get("calendar_risk", 0.0)))
        if stale_sources:
            risk = max(risk, stale_score)
            reasons.append(f"{currency} NewsPulse source stale")
        out[currency] = {
            "breaking_count_15m": _safe_int(news_state.get("breaking_count_15m", 0)),
            "intensity_15m": round(_safe_float(news_state.get("intensity_15m", 0.0)), 6),
            "tone_mean_15m": round(_safe_float(news_state.get("tone_mean_15m", 0.0)), 6),
            "tone_abs_mean_15m": round(_safe_float(news_state.get("tone_abs_mean_15m", 0.0)), 6),
            "burst_score_15m": round(_safe_float(news_state.get("burst_score_15m", 0.0)), 6),
            "story_count_15m": _safe_int(news_state.get("story_count_15m", 0)),
            "story_severity_15m": round(_safe_float(news_state.get("story_severity_15m", 0.0)), 6),
            "official_count_24h": _safe_int(news_state.get("official_count_24h", 0)),
            "dominant_story_ids": list(news_state.get("dominant_story_ids", []))[:3],
            "next_high_impact_eta_min": calendar.get("next_high_impact_eta_min"),
            "time_since_last_high_impact_min": calendar.get("time_since_last_high_impact_min"),
            "in_pre_event_window": bool(calendar.get("in_pre_event_window", False)),
            "in_post_event_window": bool(calendar.get("in_post_event_window", False)),
            "last_surprise_proxy": calendar.get("last_surprise_proxy"),
            "stale": stale_sources,
            "risk_score": round(_clamp(risk, 0.0, 1.0), 6),
            "reasons": reasons[:8],
        }
    return out


def _pressure_for_currency(currency_state: dict[str, Any]) -> float:
    surprise = _safe_float(currency_state.get("last_surprise_proxy", 0.0))
    tone = _safe_float(currency_state.get("tone_mean_15m", 0.0))
    intensity = _safe_float(currency_state.get("intensity_15m", 0.0))
    burst = _safe_float(currency_state.get("burst_score_15m", 0.0))
    story = _safe_float(currency_state.get("story_severity_15m", 0.0))
    return _clamp(0.18 * tone + 0.10 * surprise + 0.04 * intensity + 0.03 * burst + 0.08 * story, -1.0, 1.0)


def _event_window_gate(calibration: dict[str, Any],
                       base: dict[str, Any],
                       quote: dict[str, Any]) -> tuple[str, float, list[str]]:
    reasons: list[str] = []
    risk = 0.0
    gate = "ALLOW"

    event_eta = min(
        [value for value in (base.get("next_high_impact_eta_min"), quote.get("next_high_impact_eta_min")) if value is not None],
        default=None,
    )
    since_last = min(
        [value for value in (base.get("time_since_last_high_impact_min"), quote.get("time_since_last_high_impact_min")) if value is not None],
        default=None,
    )
    block_eta = max(1, int(round(_safe_float(calibration.get("event_block_eta_min", 8), 8.0) * _safe_float(calibration.get("pre_window_mult", 1.0), 1.0))))
    caution_eta = max(block_eta + 1, int(round(_safe_float(calibration.get("event_caution_eta_min", 28), 28.0) * _safe_float(calibration.get("pre_window_mult", 1.0), 1.0))))
    post_block = max(1, int(round(_safe_float(calibration.get("post_block_min", 6), 6.0) * _safe_float(calibration.get("post_window_mult", 1.0), 1.0))))
    post_caution = max(post_block + 1, int(round(_safe_float(calibration.get("post_caution_min", 24), 24.0) * _safe_float(calibration.get("post_window_mult", 1.0), 1.0))))

    if event_eta is not None and event_eta >= 0:
        if event_eta <= block_eta:
            gate = "BLOCK"
            risk = max(risk, 0.90)
            reasons.append(f"pair event block window active ({event_eta}m <= {block_eta}m)")
        elif event_eta <= caution_eta:
            gate = "CAUTION"
            risk = max(risk, 0.72)
            reasons.append(f"pair event caution window active ({event_eta}m <= {caution_eta}m)")
    if since_last is not None and since_last >= 0:
        if since_last <= post_block:
            gate = "BLOCK"
            risk = max(risk, 0.94)
            reasons.append(f"pair post-release block window active ({since_last}m <= {post_block}m)")
        elif since_last <= post_caution and gate != "BLOCK":
            gate = "CAUTION"
            risk = max(risk, 0.76)
            reasons.append(f"pair post-release caution window active ({since_last}m <= {post_caution}m)")
    return gate, risk, reasons


def _pair_gate(config: dict[str, Any],
               calibration: dict[str, Any],
               base: dict[str, Any],
               quote: dict[str, Any]) -> tuple[str, float, list[str]]:
    scoring = config["scoring"]
    risk = max(_safe_float(base.get("risk_score", 0.0)), _safe_float(quote.get("risk_score", 0.0)))
    if bool(base.get("stale")) or bool(quote.get("stale")):
        stale_score = float(scoring.get("stale_risk_score", 0.92) or 0.92)
        reasons = list(dict.fromkeys(list(base.get("reasons", [])) + list(quote.get("reasons", []))))[:8]
        return "BLOCK", max(risk, stale_score), reasons

    burst_mult = _safe_float(calibration.get("burst_risk_mult", 1.0), 1.0)
    adjusted_risk = _clamp(risk * burst_mult, 0.0, 1.0)
    caution_threshold = _clamp(
        float(scoring.get("caution_risk_threshold", 0.45) or 0.45) * _safe_float(calibration.get("caution_threshold_mult", 1.0), 1.0),
        0.20,
        0.90,
    )
    block_threshold = _clamp(
        float(scoring.get("block_risk_threshold", 0.78) or 0.78) * _safe_float(calibration.get("block_threshold_mult", 1.0), 1.0),
        caution_threshold + 0.05,
        0.98,
    )

    reasons = list(dict.fromkeys(list(base.get("reasons", [])) + list(quote.get("reasons", []))))[:10]
    event_gate, event_risk, event_reasons = _event_window_gate(calibration, base, quote)
    reasons = list(dict.fromkeys(event_reasons + reasons))
    adjusted_risk = max(adjusted_risk, event_risk)
    if event_gate == "BLOCK":
        return "BLOCK", adjusted_risk, reasons[:8]
    if event_gate == "CAUTION":
        return "CAUTION", max(adjusted_risk, caution_threshold), reasons[:8]
    if adjusted_risk >= block_threshold:
        return "BLOCK", adjusted_risk, reasons[:8]
    if adjusted_risk >= caution_threshold:
        return "CAUTION", adjusted_risk, reasons[:8]
    return "ALLOW", adjusted_risk, reasons[:8]


def _pair_story_ids(base_ccy: str, quote_ccy: str, stories: list[dict[str, Any]]) -> list[str]:
    relevant = []
    for story in stories:
        currencies = set(str(token).upper() for token in story.get("currency_tags", []))
        if base_ccy in currencies or quote_ccy in currencies:
            relevant.append(story)
    relevant.sort(key=lambda row: float(row.get("severity_score", 0.0) or 0.0), reverse=True)
    return [str(story.get("id", "")) for story in relevant[:4] if str(story.get("id", ""))]


def _build_pairs(config: dict[str, Any],
                 policy: dict[str, Any],
                 currencies: dict[str, dict[str, Any]],
                 stories: list[dict[str, Any]],
                 now_dt: datetime) -> dict[str, dict[str, Any]]:
    pair_cfg = config["pairs"]
    pair_ids = set(active_pairs(policy))
    if not pair_ids:
        pair_ids.update(_pair_universe(list(pair_cfg["currencies"]), bool(pair_cfg.get("include_permutations", True))))
    explicit_pairs = {
        str(pair_id).strip().upper()
        for pair_id in dict(policy.get("broker_symbol_map", {})).values()
        if str(pair_id).strip()
    }
    pair_ids.update(explicit_pairs)
    out: dict[str, dict[str, Any]] = {}
    for pair_id in sorted(pair_ids):
        if len(pair_id) != 6:
            continue
        base_ccy = pair_id[:3]
        quote_ccy = pair_id[3:]
        base = currencies.get(base_ccy, {})
        quote = currencies.get(quote_ccy, {})
        calibration = pair_calibration(pair_id, policy, now_dt)
        gate, risk, reasons = _pair_gate(config, calibration, base, quote)
        event_etas = [value for value in (base.get("next_high_impact_eta_min"), quote.get("next_high_impact_eta_min")) if value is not None]
        news_pressure = _clamp(_pressure_for_currency(base) - _pressure_for_currency(quote), -1.0, 1.0)
        out[pair_id] = {
            "base_currency": base_ccy,
            "quote_currency": quote_ccy,
            "event_eta_min": min(event_etas) if event_etas else None,
            "news_risk_score": round(_clamp(risk, 0.0, 1.0), 6),
            "trade_gate": gate,
            "news_pressure": round(news_pressure, 6),
            "stale": bool(base.get("stale")) or bool(quote.get("stale")),
            "reasons": reasons[:8],
            "story_ids": _pair_story_ids(base_ccy, quote_ccy, stories),
            "watchlist_tags": watchlist_tags_for_pair(pair_id, policy),
            "broker_symbols": broker_symbols_for_pair(pair_id, policy),
            "session_profile": str(calibration.get("session_profile", "default")),
            "calibration_profile": str(calibration.get("calibration_profile", "default")),
            "caution_lot_scale": round(_clamp(_safe_float(calibration.get("caution_lot_scale", 0.65), 0.65), 0.10, 1.0), 6),
            "caution_enter_prob_buffer": round(_clamp(_safe_float(calibration.get("enter_prob_buffer", 0.05), 0.05), 0.0, 0.25), 6),
        }
    return out


def _sorted_recent_items(config: dict[str, Any], items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    limit = int(config.get("history_recent_limit", 80) or 80)
    deduped: dict[str, dict[str, Any]] = {}
    for item in items:
        item_id = str(item.get("id", ""))
        if not item_id:
            continue
        current = deduped.get(item_id)
        if current is None or str(item.get("seen_at", "")) > str(current.get("seen_at", "")):
            deduped[item_id] = dict(item)
    return sorted(
        deduped.values(),
        key=lambda row: (str(row.get("published_at", "")), str(row.get("id", ""))),
        reverse=True,
    )[:limit]


def _source_status(config: dict[str, Any],
                   calendar_status: dict[str, Any],
                   state: dict[str, Any],
                   gdelt_meta: dict[str, Any],
                   official_meta: dict[str, Any],
                   now_dt: datetime) -> dict[str, Any]:
    last_poll = sanitize_utc_timestamp(str(state.get("last_gdelt_poll_at", "")), now_dt=now_dt)
    last_success = sanitize_utc_timestamp(str(state.get("last_gdelt_success_at", "")), now_dt=now_dt)
    gdelt_stale_after_sec = int(config.get("gdelt_stale_after_sec", 360) or 360)
    gdelt_stale = True
    if last_success is not None:
        gdelt_stale = (now_dt - last_success).total_seconds() > gdelt_stale_after_sec

    last_official_poll = sanitize_utc_timestamp(str(state.get("last_official_poll_at", "")), now_dt=now_dt)
    last_official_success = sanitize_utc_timestamp(str(state.get("last_official_success_at", "")), now_dt=now_dt)
    official_stale_after_sec = int(config.get("gdelt_stale_after_sec", 360) or 360)
    official_stale = False
    official_enabled = int(official_meta.get("query_count", 0) or 0) > 0
    if official_enabled:
        official_stale = True
        if last_official_success is not None:
            official_stale = (now_dt - last_official_success).total_seconds() > official_stale_after_sec

    calendar_stale_after_sec = int(config.get("calendar_stale_after_sec", 360) or 360)
    calendar_last = sanitize_utc_timestamp(str(calendar_status.get("last_update_at", "")), now_dt=now_dt)
    calendar_stale = bool(calendar_status.get("stale", True))
    if calendar_last is not None:
        calendar_stale = calendar_stale or (now_dt - calendar_last).total_seconds() > calendar_stale_after_sec
    return {
        "calendar": {
            "ok": bool(calendar_status.get("ok", False)),
            "stale": calendar_stale,
            "enabled": True,
            "required": True,
            "last_update_at": isoformat_utc(calendar_last) if calendar_last is not None else "",
            "last_update_trade_server": str(calendar_status.get("last_update_trade_server", "")),
            "time_basis": str(calendar_status.get("time_basis", "trade_server")),
            "cursor": str(calendar_status.get("cursor", "")),
            "last_error": str(calendar_status.get("last_error", "")),
        },
        "gdelt": {
            "ok": gdelt_meta.get("success_count", 0) > 0 or (last_success is not None and not gdelt_stale),
            "stale": gdelt_stale,
            "enabled": True,
            "required": True,
            "last_poll_at": isoformat_utc(last_poll) if last_poll is not None else "",
            "last_success_at": isoformat_utc(last_success) if last_success is not None else "",
            "last_error": gdelt_meta["errors"][-1] if gdelt_meta.get("errors") else "",
            "backoff_until": str(state.get("gdelt_backoff_until", "")),
            "budget_exhausted": bool(gdelt_meta.get("budget_exhausted", False)),
            "throttled": bool(gdelt_meta.get("throttled", False)),
        },
        "official": {
            "ok": (not official_enabled) or official_meta.get("success_count", 0) > 0 or (last_official_success is not None and not official_stale),
            "stale": official_stale,
            "enabled": official_enabled,
            "required": False,
            "last_poll_at": isoformat_utc(last_official_poll) if last_official_poll is not None else "",
            "last_success_at": isoformat_utc(last_official_success) if last_official_success is not None else "",
            "last_error": official_meta["errors"][-1] if official_meta.get("errors") else "",
        },
    }


def _gate_changed_at(entries: list[dict[str, Any]]) -> str:
    if not entries:
        return ""
    current_gate = str(entries[-1].get("trade_gate", "UNKNOWN"))
    current_stale = bool(entries[-1].get("stale", True))
    for entry in reversed(entries):
        if str(entry.get("trade_gate", "UNKNOWN")) != current_gate or bool(entry.get("stale", True)) != current_stale:
            break
        changed_at = str(entry.get("observed_at", ""))
    return changed_at


def _status_pair_timelines(pair_history: dict[str, list[dict[str, Any]]],
                           pairs: dict[str, dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    priority_pairs = sorted(
        pairs.keys(),
        key=lambda pair_id: (
            0 if str(pairs[pair_id].get("trade_gate", "ALLOW")).upper() == "BLOCK" else
            1 if str(pairs[pair_id].get("trade_gate", "ALLOW")).upper() == "CAUTION" else
            2,
            -float(pairs[pair_id].get("news_risk_score", 0.0) or 0.0),
            pair_id,
        ),
    )[:16]
    return {pair_id: list(pair_history.get(pair_id, [])[-10:]) for pair_id in priority_pairs if pair_history.get(pair_id)}


def _write_symbol_map(policy: dict[str, Any], pairs: dict[str, dict[str, Any]]) -> None:
    lines = ["kind\tsymbol\tpair_id"]
    for pair_id in sorted(pairs):
        lines.append(f"symbol\t{pair_id}\t{pair_id}")
    for raw_symbol, pair_id in sorted(dict(policy.get("broker_symbol_map", {})).items()):
        normalized_symbol = str(raw_symbol).strip()
        normalized_pair = str(pair_id).strip().upper()
        if normalized_symbol and len(normalized_pair) == 6:
            lines.append(f"symbol\t{normalized_symbol}\t{normalized_pair}")
    COMMON_NEWSPULSE_SYMBOL_MAP.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _history_record_count(path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        return sum(1 for _ in handle)


def _write_flat_snapshot(snapshot: dict[str, Any]) -> None:
    generated_at_dt = parse_iso8601(str(snapshot["generated_at"])) or utc_now()
    lines = [
        f"meta\tglobal\tschema_version\t{snapshot['schema_version']}",
        f"meta\tglobal\tgenerated_at\t{snapshot['generated_at']}",
        f"meta\tglobal\tgenerated_at_unix\t{int(generated_at_dt.timestamp())}",
    ]
    for source_name, status in snapshot["source_status"].items():
        for key in (
            "ok",
            "stale",
            "enabled",
            "required",
            "last_update_at",
            "last_poll_at",
            "last_success_at",
            "cursor",
            "last_error",
            "backoff_until",
            "budget_exhausted",
            "throttled",
        ):
            if key not in status:
                continue
            value = status[key]
            if isinstance(value, bool):
                value = "1" if value else "0"
            lines.append(f"source\t{source_name}\t{key}\t{value}")
    for currency, state in snapshot["currencies"].items():
        for key, value in state.items():
            if key == "reasons":
                for idx, reason in enumerate(value):
                    lines.append(f"currency_reason\t{currency}\t{idx}\t{reason}")
                continue
            if isinstance(value, list):
                value = ",".join(str(entry) for entry in value)
            if isinstance(value, bool):
                value = "1" if value else "0"
            lines.append(f"currency\t{currency}\t{key}\t{'' if value is None else value}")
    for pair_id, state in snapshot["pairs"].items():
        for key, value in state.items():
            if key == "reasons":
                for idx, reason in enumerate(value):
                    lines.append(f"pair_reason\t{pair_id}\t{idx}\t{reason}")
                continue
            if isinstance(value, list):
                value = ",".join(str(entry) for entry in value)
            if isinstance(value, bool):
                value = "1" if value else "0"
            lines.append(f"pair\t{pair_id}\t{key}\t{'' if value is None else value}")
    for item in snapshot["recent_items"]:
        item_id = str(item.get("id", ""))
        for key, value in item.items():
            if key == "id":
                continue
            if isinstance(value, list):
                value = ",".join(str(entry) for entry in value)
            lines.append(f"recent_item\t{item_id}\t{key}\t{value}")
    COMMON_NEWSPULSE_FLAT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_newspulse_cycle(daemon_context: dict[str, Any] | None = None) -> dict[str, Any]:
    ensure_newspulse_dirs()
    config, sources = load_config()
    policy = load_policy()
    state = _load_state()
    now_dt = utc_now()
    now_iso = isoformat_utc(now_dt)

    if isinstance(daemon_context, dict) and daemon_context:
        daemon_health = dict(state.get("daemon_health", {}))
        daemon_health.update(daemon_context)
        daemon_health["heartbeat_at"] = now_iso
        state["daemon_health"] = daemon_health

    backoff_until = sanitize_utc_timestamp(state.get("gdelt_backoff_until", ""), now_dt=now_dt)
    query_specs = _select_query_specs(config, state, build_query_specs(config))
    if backoff_until is not None and backoff_until > now_dt:
        gdelt_items = []
        gdelt_meta = {
            "errors": [f"gdelt_backoff_until:{isoformat_utc(backoff_until)}"],
            "query_count": 0,
            "success_count": 0,
            "throttled": True,
            "budget_exhausted": False,
        }
    else:
        gdelt_items, gdelt_meta = query_gdelt(config, sources, query_specs, now_iso)
        state["last_gdelt_poll_at"] = now_iso
        if gdelt_meta.get("success_count", 0) > 0:
            state["last_gdelt_success_at"] = now_iso
        if gdelt_meta.get("throttled", False):
            backoff_sec = int(config.get("gdelt", {}).get("rate_limit_backoff_sec", 300) or 300)
            state["gdelt_backoff_until"] = isoformat_utc(now_dt + timedelta(seconds=backoff_sec))
        else:
            state["gdelt_backoff_until"] = ""

    official_items, official_meta = query_official_feeds(config, sources, now_iso)
    state["last_official_poll_at"] = now_iso if int(official_meta.get("query_count", 0) or 0) > 0 else state.get("last_official_poll_at", "")
    if official_meta.get("success_count", 0) > 0:
        state["last_official_success_at"] = now_iso

    calendar_status = calendar_source_status()
    calendar_currency, calendar_items = _calendar_currency_state(config, now_dt)

    rolling_items, seen_items, item_history = _merge_seen_items(state, gdelt_items + official_items)
    state["seen_items"] = _prune_seen_items(seen_items, now_dt)
    rolling_items = [
        dict(item, id=item_id)
        for item_id, item in state["seen_items"].items()
        if isinstance(item, dict)
    ]
    stories, item_story_map = build_story_clusters(rolling_items, now_dt)
    for item in rolling_items:
        story_id = item_story_map.get(str(item.get("id", "")), "")
        if story_id:
            item["story_id"] = story_id

    news_currency, state = _news_currency_state(config, state, rolling_items, stories, now_dt)
    source_status = _source_status(config, calendar_status, state, gdelt_meta, official_meta, now_dt)
    currencies = _combine_currency_state(config, news_currency, calendar_currency, source_status)
    pairs = _build_pairs(config, policy, currencies, stories, now_dt)
    pair_history = update_pair_gate_history(state, pairs, now_dt)
    source_health_history = update_source_health_history(state, source_status, now_dt)
    for pair_id, pair_state in pairs.items():
        pair_state["gate_changed_at"] = _gate_changed_at(pair_history.get(pair_id, []))

    recent_items = _sorted_recent_items(config, rolling_items + calendar_items)
    snapshot = {
        "schema_version": NEWSPULSE_SCHEMA_VERSION,
        "generated_at": now_iso,
        "source_status": source_status,
        "currencies": currencies,
        "pairs": pairs,
        "stories": stories[:32],
        "recent_items": recent_items,
    }
    snapshot = apply_rates_enrichment(snapshot)
    json_dump(COMMON_NEWSPULSE_JSON, snapshot)
    replay_report = write_replay_artifacts(snapshot, pair_history, source_health_history)
    _write_flat_snapshot(snapshot)
    _write_symbol_map(policy, pairs)
    status_payload = {
        **snapshot,
        "query_count": int(gdelt_meta.get("query_count", 0) or 0),
        "official_query_count": int(official_meta.get("query_count", 0) or 0),
        "pair_timelines": _status_pair_timelines(pair_history, pairs),
        "source_health_timeline": list(source_health_history[-24:]),
        "daemon": dict(state.get("daemon_health", {})),
        "policy_summary": {
            "active_pairs": active_pairs(policy),
            "watchlists": {
                key: list(value)
                for key, value in dict(policy.get("watchlists", {})).items()
                if isinstance(value, list)
            },
            "broker_symbol_map_count": len(dict(policy.get("broker_symbol_map", {}))),
        },
        "health": {
            "required_sources_stale": _required_stale(source_status),
            "gdelt_backoff_until": str(state.get("gdelt_backoff_until", "")),
            "history_records_local": _history_record_count(NEWSPULSE_LOCAL_HISTORY_PATH),
            "story_count": len(stories),
        },
        "replay": replay_report,
        "artifacts": {
            "snapshot_json": str(COMMON_NEWSPULSE_JSON),
            "snapshot_flat": str(COMMON_NEWSPULSE_FLAT),
            "history_ndjson": str(NEWSPULSE_LOCAL_HISTORY_PATH),
            "status_json": str(NEWSPULSE_STATUS_PATH),
            "replay_timeline_tsv": str(COMMON_NEWSPULSE_REPLAY_FLAT),
            "replay_report_json": str(NEWSPULSE_REPLAY_REPORT_PATH),
            "symbol_map_tsv": str(COMMON_NEWSPULSE_SYMBOL_MAP),
            "policy_json": str(NEWSPULSE_POLICY_PATH),
            "config_json": str(NEWSPULSE_CONFIG_PATH),
            "sources_json": str(NEWSPULSE_SOURCES_PATH),
        },
    }
    json_dump(NEWSPULSE_STATUS_PATH, status_payload)

    for record in item_history:
        write_history_mirror(record)
    write_history_mirror({"record_type": "story_snapshot", "generated_at": now_iso, "stories": stories[:12]})
    write_history_mirror({"record_type": "snapshot", "snapshot": snapshot})

    _save_state(state)
    return {
        "snapshot_path": str(COMMON_NEWSPULSE_JSON),
        "flat_path": str(COMMON_NEWSPULSE_FLAT),
        "history_path": str(NEWSPULSE_LOCAL_HISTORY_PATH),
        "query_count": int(gdelt_meta.get("query_count", 0) or 0),
        "official_query_count": int(official_meta.get("query_count", 0) or 0),
        "gdelt_errors": list(gdelt_meta.get("errors", [])),
        "official_errors": list(official_meta.get("errors", [])),
        "calendar_ok": bool(source_status["calendar"]["ok"]),
        "gdelt_ok": bool(source_status["gdelt"]["ok"]),
        "official_ok": bool(source_status["official"]["ok"]),
        "pair_count": len(pairs),
        "currency_count": len(currencies),
        "story_count": len(stories),
    }
