from __future__ import annotations

import math
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any

from .newspulse_calendar import calendar_source_status, load_calendar_records
from .newspulse_config import build_query_specs, load_config
from .newspulse_contracts import (
    COMMON_NEWSPULSE_FLAT,
    COMMON_NEWSPULSE_HISTORY,
    COMMON_NEWSPULSE_JSON,
    NEWSPULSE_LOCAL_HISTORY_PATH,
    NEWSPULSE_SCHEMA_VERSION,
    NEWSPULSE_STATE_PATH,
    NEWSPULSE_STATUS_PATH,
    ensure_newspulse_dirs,
    isoformat_utc,
    json_dump,
    ndjson_append,
    parse_iso8601,
    sanitize_utc_timestamp,
    utc_now,
)
from .newspulse_gdelt import query_gdelt


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


def _load_state() -> dict[str, Any]:
    if not NEWSPULSE_STATE_PATH.exists():
        return {
            "last_gdelt_success_at": "",
            "last_gdelt_poll_at": "",
            "gdelt_backoff_until": "",
            "seen_items": {},
            "currency_baselines": {},
            "query_rotation_index": 0,
        }
    try:
        import json

        payload = json.loads(NEWSPULSE_STATE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {
            "last_gdelt_success_at": "",
            "last_gdelt_poll_at": "",
            "gdelt_backoff_until": "",
            "seen_items": {},
            "currency_baselines": {},
            "query_rotation_index": 0,
        }
    if not isinstance(payload, dict):
        return {
            "last_gdelt_success_at": "",
            "last_gdelt_poll_at": "",
            "gdelt_backoff_until": "",
            "seen_items": {},
            "currency_baselines": {},
        }
    now_dt = utc_now()
    payload.setdefault("seen_items", {})
    payload.setdefault("currency_baselines", {})
    payload["last_gdelt_success_at"] = _state_timestamp_text(payload.get("last_gdelt_success_at"), now_dt)
    payload["last_gdelt_poll_at"] = _state_timestamp_text(payload.get("last_gdelt_poll_at"), now_dt)
    payload["gdelt_backoff_until"] = _state_timestamp_text(payload.get("gdelt_backoff_until"), now_dt)
    payload.setdefault("query_rotation_index", 0)
    return payload


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
    return _safe_float(item.get("tier_weight", 0.0)) * _safe_float(item.get("topic_weight", 1.0), 1.0) * decay * tone_weight


def _state_timestamp_text(value: str | None, now_dt: datetime) -> str:
    dt = sanitize_utc_timestamp(value, now_dt=now_dt)
    return isoformat_utc(dt) if dt is not None else ""


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


def _merge_seen_items(existing_state: dict[str, Any], fresh_items: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    seen_items = dict(existing_state.get("seen_items", {}))
    appended_history: list[dict[str, Any]] = []
    merged_items: list[dict[str, Any]] = []
    for item in fresh_items:
        item_id = str(item["id"])
        stored = seen_items.get(item_id)
        if not isinstance(stored, dict):
            seen_items[item_id] = {
                "published_at": item["published_at"],
                "seen_at": item["seen_at"],
                "title": item["title"],
                "domain": item["domain"],
            }
            appended_history.append({"record_type": "item", "item": item})
        else:
            stored["seen_at"] = item["seen_at"]
        merged_items.append(item)
    return merged_items, seen_items, appended_history


def _gdelt_currency_state(config: dict[str, Any],
                          state: dict[str, Any],
                          gdelt_items: list[dict[str, Any]],
                          now_dt: datetime) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    scoring = config["scoring"]
    half_life_min = float(scoring.get("intensity_half_life_min", 12.0) or 12.0)
    baseline_alpha = float(scoring.get("baseline_alpha", 0.18) or 0.18)
    currencies = list(config["currencies"].keys())
    window_cutoff = now_dt - timedelta(minutes=15)
    state.setdefault("currency_baselines", {})
    per_currency: dict[str, dict[str, Any]] = {}
    recent_items = []
    for currency in currencies:
        relevant = [item for item in gdelt_items if currency in item.get("currency_tags", [])]
        recent = [item for item in relevant if (parse_iso8601(item.get("published_at")) or now_dt) >= window_cutoff]
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
        per_currency[currency] = {
            "breaking_count_15m": count,
            "intensity_15m": round(intensity, 6),
            "tone_mean_15m": round(tone_mean, 6),
            "tone_abs_mean_15m": round(tone_abs_mean, 6),
            "burst_score_15m": round(burst_score, 6),
            "gdelt_risk": _clamp(0.18 * burst_score + 0.08 * min(tone_abs_mean / 4.0, 1.0), 0.0, 0.85),
            "reasons": [],
        }
        if burst_score >= 1.0:
            per_currency[currency]["reasons"].append(f"{currency} burst score elevated at {burst_score:.2f}")
        recent_items.extend(recent)
    return per_currency, recent_items, state


def _combine_currency_state(config: dict[str, Any],
                            gdelt_currency: dict[str, dict[str, Any]],
                            calendar_currency: dict[str, dict[str, Any]],
                            source_status: dict[str, Any]) -> dict[str, dict[str, Any]]:
    stale_sources = bool(source_status["calendar"]["stale"]) or bool(source_status["gdelt"]["stale"])
    stale_score = float(config["scoring"].get("stale_risk_score", 0.92) or 0.92)
    out: dict[str, dict[str, Any]] = {}
    for currency in sorted(set(gdelt_currency) | set(calendar_currency)):
        gdelt = gdelt_currency.get(currency, {})
        calendar = calendar_currency.get(currency, {})
        reasons = list(dict.fromkeys(list(calendar.get("reasons", [])) + list(gdelt.get("reasons", []))))
        risk = max(_safe_float(gdelt.get("gdelt_risk", 0.0)), _safe_float(calendar.get("calendar_risk", 0.0)))
        if stale_sources:
            risk = max(risk, stale_score)
            reasons.append(f"{currency} NewsPulse source stale")
        out[currency] = {
            "breaking_count_15m": _safe_int(gdelt.get("breaking_count_15m", 0)),
            "intensity_15m": round(_safe_float(gdelt.get("intensity_15m", 0.0)), 6),
            "tone_mean_15m": round(_safe_float(gdelt.get("tone_mean_15m", 0.0)), 6),
            "tone_abs_mean_15m": round(_safe_float(gdelt.get("tone_abs_mean_15m", 0.0)), 6),
            "burst_score_15m": round(_safe_float(gdelt.get("burst_score_15m", 0.0)), 6),
            "next_high_impact_eta_min": calendar.get("next_high_impact_eta_min"),
            "time_since_last_high_impact_min": calendar.get("time_since_last_high_impact_min"),
            "in_pre_event_window": bool(calendar.get("in_pre_event_window", False)),
            "in_post_event_window": bool(calendar.get("in_post_event_window", False)),
            "last_surprise_proxy": calendar.get("last_surprise_proxy"),
            "stale": stale_sources,
            "risk_score": round(_clamp(risk, 0.0, 1.0), 6),
            "reasons": reasons[:6],
        }
    return out


def _pressure_for_currency(currency_state: dict[str, Any]) -> float:
    surprise = _safe_float(currency_state.get("last_surprise_proxy", 0.0))
    tone = _safe_float(currency_state.get("tone_mean_15m", 0.0))
    intensity = _safe_float(currency_state.get("intensity_15m", 0.0))
    burst = _safe_float(currency_state.get("burst_score_15m", 0.0))
    return _clamp(0.18 * tone + 0.10 * surprise + 0.04 * intensity + 0.03 * burst, -1.0, 1.0)


def _pair_gate(config: dict[str, Any], base: dict[str, Any], quote: dict[str, Any]) -> tuple[str, float, list[str]]:
    scoring = config["scoring"]
    risk = max(_safe_float(base.get("risk_score", 0.0)), _safe_float(quote.get("risk_score", 0.0)))
    if bool(base.get("stale")) or bool(quote.get("stale")):
        return "BLOCK", max(risk, float(scoring.get("stale_risk_score", 0.92) or 0.92)), list(dict.fromkeys(base.get("reasons", []) + quote.get("reasons", [])))
    reasons = list(dict.fromkeys(list(base.get("reasons", [])) + list(quote.get("reasons", []))))[:8]
    if risk >= float(scoring.get("block_risk_threshold", 0.78) or 0.78):
        return "BLOCK", risk, reasons
    if risk >= float(scoring.get("caution_risk_threshold", 0.45) or 0.45):
        return "CAUTION", risk, reasons
    return "ALLOW", risk, reasons


def _build_pairs(config: dict[str, Any], currencies: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    pair_cfg = config["pairs"]
    pair_ids = _pair_universe(list(pair_cfg["currencies"]), bool(pair_cfg.get("include_permutations", True)))
    out: dict[str, dict[str, Any]] = {}
    for pair_id in pair_ids:
        base_ccy = pair_id[:3]
        quote_ccy = pair_id[3:]
        base = currencies.get(base_ccy, {})
        quote = currencies.get(quote_ccy, {})
        event_etas = [value for value in (base.get("next_high_impact_eta_min"), quote.get("next_high_impact_eta_min")) if value is not None]
        gate, risk, reasons = _pair_gate(config, base, quote)
        news_pressure = _clamp(_pressure_for_currency(base) - _pressure_for_currency(quote), -1.0, 1.0)
        out[pair_id] = {
            "event_eta_min": min(event_etas) if event_etas else None,
            "news_risk_score": round(_clamp(risk, 0.0, 1.0), 6),
            "trade_gate": gate,
            "news_pressure": round(news_pressure, 6),
            "stale": bool(base.get("stale")) or bool(quote.get("stale")),
            "reasons": reasons[:6],
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
    return sorted(deduped.values(), key=lambda row: (str(row.get("published_at", "")), str(row.get("id", ""))), reverse=True)[:limit]


def _source_status(config: dict[str, Any],
                   calendar_status: dict[str, Any],
                   state: dict[str, Any],
                   gdelt_errors: list[str],
                   gdelt_success_count: int,
                   now_dt: datetime) -> dict[str, Any]:
    last_poll = sanitize_utc_timestamp(str(state.get("last_gdelt_poll_at", "")), now_dt=now_dt)
    last_success = sanitize_utc_timestamp(str(state.get("last_gdelt_success_at", "")), now_dt=now_dt)
    gdelt_stale_after_sec = int(config.get("gdelt_stale_after_sec", 360) or 360)
    gdelt_stale = True
    if last_success is not None:
        gdelt_stale = (now_dt - last_success).total_seconds() > gdelt_stale_after_sec
    calendar_stale_after_sec = int(config.get("calendar_stale_after_sec", 360) or 360)
    calendar_last = sanitize_utc_timestamp(str(calendar_status.get("last_update_at", "")), now_dt=now_dt)
    calendar_stale = bool(calendar_status.get("stale", True))
    if calendar_last is not None:
        calendar_stale = calendar_stale or (now_dt - calendar_last).total_seconds() > calendar_stale_after_sec
    return {
        "calendar": {
            "ok": bool(calendar_status.get("ok", False)),
            "stale": calendar_stale,
            "last_update_at": isoformat_utc(calendar_last) if calendar_last is not None else "",
            "last_update_trade_server": str(calendar_status.get("last_update_trade_server", "")),
            "time_basis": str(calendar_status.get("time_basis", "trade_server")),
            "cursor": str(calendar_status.get("cursor", "")),
            "last_error": str(calendar_status.get("last_error", "")),
        },
        "gdelt": {
            "ok": gdelt_success_count > 0 or (last_success is not None and not gdelt_stale),
            "stale": gdelt_stale,
            "last_poll_at": isoformat_utc(last_poll) if last_poll is not None else "",
            "last_success_at": isoformat_utc(last_success) if last_success is not None else "",
            "last_error": gdelt_errors[-1] if gdelt_errors else "",
        },
    }


def _write_flat_snapshot(snapshot: dict[str, Any]) -> None:
    generated_at_dt = parse_iso8601(str(snapshot["generated_at"])) or utc_now()
    lines = [
        f"meta\tglobal\tschema_version\t{snapshot['schema_version']}",
        f"meta\tglobal\tgenerated_at\t{snapshot['generated_at']}",
        f"meta\tglobal\tgenerated_at_unix\t{int(generated_at_dt.timestamp())}",
    ]
    for source_name, status in snapshot["source_status"].items():
        for key in ("ok", "stale", "last_update_at", "last_poll_at", "last_success_at", "cursor", "last_error"):
            if key in status:
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
            if isinstance(value, bool):
                value = "1" if value else "0"
            lines.append(f"currency\t{currency}\t{key}\t{'' if value is None else value}")
    for pair_id, state in snapshot["pairs"].items():
        for key, value in state.items():
            if key == "reasons":
                for idx, reason in enumerate(value):
                    lines.append(f"pair_reason\t{pair_id}\t{idx}\t{reason}")
                continue
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


def run_newspulse_cycle() -> dict[str, Any]:
    ensure_newspulse_dirs()
    config, sources = load_config()
    state = _load_state()
    now_dt = utc_now()
    now_iso = isoformat_utc(now_dt)
    backoff_until = sanitize_utc_timestamp(state.get("gdelt_backoff_until", ""), now_dt=now_dt)
    query_specs = _select_query_specs(config, state, build_query_specs(config))
    if backoff_until is not None and backoff_until > now_dt:
        gdelt_items = []
        gdelt_meta = {
            "errors": [f"gdelt_backoff_until:{isoformat_utc(backoff_until)}"],
            "query_count": 0,
            "success_count": 0,
            "throttled": True,
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
    calendar_status = calendar_source_status()
    calendar_currency, calendar_items = _calendar_currency_state(config, now_dt)
    gdelt_currency, gdelt_recent_items, state = _gdelt_currency_state(config, state, gdelt_items, now_dt)
    merged_items, seen_items, item_history = _merge_seen_items(state, gdelt_items)
    state["seen_items"] = _prune_seen_items(seen_items, now_dt)
    source_status = _source_status(
        config,
        calendar_status,
        state,
        gdelt_meta["errors"],
        int(gdelt_meta.get("success_count", 0) or 0),
        now_dt,
    )
    currencies = _combine_currency_state(config, gdelt_currency, calendar_currency, source_status)
    pairs = _build_pairs(config, currencies)
    recent_items = _sorted_recent_items(config, merged_items + calendar_items + gdelt_recent_items)
    snapshot = {
        "schema_version": NEWSPULSE_SCHEMA_VERSION,
        "generated_at": now_iso,
        "source_status": source_status,
        "currencies": currencies,
        "pairs": pairs,
        "recent_items": recent_items,
    }
    json_dump(COMMON_NEWSPULSE_JSON, snapshot)
    json_dump(NEWSPULSE_STATUS_PATH, {
        **snapshot,
        "query_count": gdelt_meta["query_count"],
        "artifacts": {
            "snapshot_json": str(COMMON_NEWSPULSE_JSON),
            "snapshot_flat": str(COMMON_NEWSPULSE_FLAT),
            "history_ndjson": str(COMMON_NEWSPULSE_HISTORY),
        },
    })
    _write_flat_snapshot(snapshot)
    for record in item_history:
        ndjson_append(COMMON_NEWSPULSE_HISTORY, record)
        ndjson_append(NEWSPULSE_LOCAL_HISTORY_PATH, record)
    snapshot_record = {"record_type": "snapshot", "snapshot": snapshot}
    ndjson_append(COMMON_NEWSPULSE_HISTORY, snapshot_record)
    ndjson_append(NEWSPULSE_LOCAL_HISTORY_PATH, snapshot_record)
    _save_state(state)
    return {
        "snapshot_path": str(COMMON_NEWSPULSE_JSON),
        "flat_path": str(COMMON_NEWSPULSE_FLAT),
        "history_path": str(COMMON_NEWSPULSE_HISTORY),
        "query_count": gdelt_meta["query_count"],
        "gdelt_errors": list(gdelt_meta["errors"]),
        "calendar_ok": bool(source_status["calendar"]["ok"]),
        "gdelt_ok": bool(source_status["gdelt"]["ok"]),
        "pair_count": len(pairs),
        "currency_count": len(currencies),
    }
