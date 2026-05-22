from __future__ import annotations

import hashlib
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any

_STOPWORDS = {
    "a",
    "an",
    "and",
    "ahead",
    "at",
    "for",
    "from",
    "in",
    "into",
    "of",
    "on",
    "says",
    "the",
    "to",
    "with",
}


def _parse_iso8601(value: str) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).astimezone(timezone.utc)
    except ValueError:
        return None


def _story_terms(title: str) -> list[str]:
    tokens = [
        token
        for token in "".join(ch.lower() if ch.isalnum() else " " for ch in str(title or "")).split()
        if len(token) >= 3 and token not in _STOPWORDS
    ]
    return tokens[:6]


def story_key(item: dict[str, Any]) -> str:
    currencies = ",".join(sorted({str(token).upper() for token in item.get("currency_tags", []) if str(token).strip()}))
    topics = ",".join(sorted({str(token) for token in item.get("topic_tags", []) if str(token).strip()}))
    title_terms = "-".join(_story_terms(str(item.get("title", "")))[:4]) or "story"
    key = f"{currencies}|{topics}|{title_terms}"
    return hashlib.sha256(key.encode("utf-8", errors="ignore")).hexdigest()[:16]


def build_story_clusters(items: list[dict[str, Any]],
                         now_dt: datetime,
                         *,
                         lookback_hours: int = 48,
                         active_minutes: int = 90) -> tuple[list[dict[str, Any]], dict[str, str]]:
    cutoff = now_dt - timedelta(hours=max(lookback_hours, 1))
    active_cutoff = now_dt - timedelta(minutes=max(active_minutes, 1))
    clusters: dict[str, list[dict[str, Any]]] = defaultdict(list)
    item_to_story: dict[str, str] = {}

    for item in items:
        published_dt = _parse_iso8601(str(item.get("published_at", "")))
        if published_dt is None or published_dt < cutoff:
            continue
        story_id = story_key(item)
        clusters[story_id].append(item)
        item_to_story[str(item.get("id", ""))] = story_id

    stories: list[dict[str, Any]] = []
    for story_id, story_items in clusters.items():
        story_items = sorted(story_items, key=lambda row: str(row.get("published_at", "")))
        latest = story_items[-1]
        latest_dt = _parse_iso8601(str(latest.get("published_at", ""))) or now_dt
        distinct_domains = sorted({str(item.get("domain", "")) for item in story_items if str(item.get("domain", "")).strip()})
        official_hits = sum(1 for item in story_items if str(item.get("source", "")) == "official")
        tier_strength = max(float(item.get("tier_weight", 0.0) or 0.0) for item in story_items)
        recency_weight = 1.0 if latest_dt >= active_cutoff else max(0.15, 1.0 - ((active_cutoff - latest_dt).total_seconds() / 3600.0))
        severity = min(
            1.0,
            0.18 * min(len(story_items), 5) +
            0.16 * min(len(distinct_domains), 4) +
            0.22 * min(official_hits, 2) +
            0.22 * min(tier_strength, 1.25) +
            0.22 * recency_weight,
        )
        stories.append(
            {
                "id": story_id,
                "latest_title": str(latest.get("title", "")),
                "first_published_at": str(story_items[0].get("published_at", "")),
                "last_published_at": str(latest.get("published_at", "")),
                "currency_tags": sorted({tag for item in story_items for tag in item.get("currency_tags", [])}),
                "topic_tags": sorted({tag for item in story_items for tag in item.get("topic_tags", [])}),
                "domains": distinct_domains,
                "representative_url": str(latest.get("url", "")),
                "item_count": len(story_items),
                "source_count": len(distinct_domains),
                "official_hits": official_hits,
                "severity_score": round(severity, 6),
                "active": latest_dt >= active_cutoff,
                "item_ids": [str(item.get("id", "")) for item in story_items if str(item.get("id", ""))],
            }
        )

    stories.sort(key=lambda row: (float(row.get("severity_score", 0.0)), str(row.get("last_published_at", ""))), reverse=True)
    return stories, item_to_story
