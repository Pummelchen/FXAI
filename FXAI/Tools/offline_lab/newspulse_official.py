from __future__ import annotations

import hashlib
import ssl
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from typing import Any
from urllib.parse import urlparse

from .newspulse_config import domain_weight, source_spec

_XML_NS = {
    "atom": "http://www.w3.org/2005/Atom",
    "dc": "http://purl.org/dc/elements/1.1/",
}


def _ssl_context() -> ssl.SSLContext:
    try:
        import certifi  # type: ignore

        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return ssl.create_default_context()


def _normalize_domain(value: str) -> str:
    domain = str(value or "").strip().lower()
    domain = domain.removeprefix("https://").removeprefix("http://")
    domain = domain.split("/", 1)[0]
    if domain.startswith("www."):
        domain = domain[4:]
    return domain


def _normalize_title(value: str) -> str:
    text = "".join(ch.lower() if ch.isalnum() else " " for ch in str(value or "").strip())
    return " ".join(token for token in text.split() if token)


def _parse_feed_datetime(value: str) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    except ValueError:
        pass
    try:
        return parsedate_to_datetime(text).astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    except (TypeError, ValueError):
        return ""


def _entry_text(node: ET.Element | None, names: list[str]) -> str:
    if node is None:
        return ""
    for name in names:
        try:
            found = node.find(name, _XML_NS)
        except SyntaxError:
            continue
        if found is not None and found.text:
            return found.text.strip()
    return ""


def _entry_link(node: ET.Element | None) -> str:
    if node is None:
        return ""
    link = node.find("link")
    if link is not None:
        href = link.get("href")
        if href:
            return href.strip()
        if link.text:
            return link.text.strip()
    atom_link = node.find("atom:link", _XML_NS)
    if atom_link is not None:
        href = atom_link.get("href")
        if href:
            return href.strip()
    return ""


def _feed_entries(xml_text: str) -> list[dict[str, str]]:
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return []

    out: list[dict[str, str]] = []
    for item in root.findall(".//item"):
        out.append(
            {
                "title": _entry_text(item, ["title"]),
                "url": _entry_link(item),
                "published_at": _parse_feed_datetime(
                    _entry_text(item, ["pubDate", "published", "updated", "dc:date"])
                ),
                "identifier": _entry_text(item, ["guid"]) or _entry_link(item),
            }
        )
    if out:
        return out

    for entry in root.findall(".//atom:entry", _XML_NS):
        out.append(
            {
                "title": _entry_text(entry, ["atom:title", "title"]),
                "url": _entry_link(entry),
                "published_at": _parse_feed_datetime(
                    _entry_text(entry, ["atom:updated", "atom:published", "updated", "published"])
                ),
                "identifier": _entry_text(entry, ["atom:id", "id"]) or _entry_link(entry),
            }
        )
    return out


def _feed_specs(sources: dict[str, Any]) -> list[dict[str, Any]]:
    feeds = sources.get("official_feeds", [])
    if not isinstance(feeds, list):
        return []
    return [dict(feed) for feed in feeds if isinstance(feed, dict) and bool(feed.get("enabled", False))]


def fetch_official_feed(feed_spec: dict[str, Any], request_timeout_sec: int) -> list[dict[str, str]]:
    request = urllib.request.Request(
        str(feed_spec.get("url", "")),
        headers={
            "Accept": "application/rss+xml, application/atom+xml, application/xml, text/xml",
            "User-Agent": "FXAI-NewsPulse/1.0 (+https://github.com/Pummelchen/FXAI)",
        },
    )
    with urllib.request.urlopen(request, timeout=request_timeout_sec, context=_ssl_context()) as response:
        payload = response.read().decode("utf-8", errors="ignore")
    return _feed_entries(payload)


def query_official_feeds(config: dict[str, Any],
                         sources: dict[str, Any],
                         seen_at_iso: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    feeds = _feed_specs(sources)
    if not feeds:
        return [], {"errors": [], "query_count": 0, "success_count": 0}

    request_timeout_sec = int(config.get("gdelt", {}).get("request_timeout_sec", 6) or 6)
    try:
        now_dt = datetime.fromisoformat(str(seen_at_iso).replace("Z", "+00:00")).astimezone(timezone.utc)
    except ValueError:
        now_dt = datetime.now(timezone.utc)
    items: list[dict[str, Any]] = []
    errors: list[str] = []
    success_count = 0
    for feed in feeds:
        try:
            raw_entries = fetch_official_feed(feed, request_timeout_sec)
            success_count += 1
        except Exception as exc:
            errors.append(f"{feed.get('id', 'official')}:{exc}")
            continue

        domain = _normalize_domain(str(feed.get("domain", "") or urlparse(str(feed.get("url", ""))).netloc))
        weight = domain_weight(sources, domain)
        if weight <= 0.0:
            continue
        domain_meta = source_spec(sources, domain)
        keywords = [str(item).strip().lower() for item in feed.get("title_keywords", []) if str(item).strip()]
        lookback_hours = int(feed.get("lookback_hours", 48) or 48)
        cutoff = now_dt - timedelta(hours=max(lookback_hours, 1))
        max_items = int(feed.get("max_items", 12) or 12)

        accepted = 0
        for entry in raw_entries:
            if accepted >= max_items:
                break
            title = str(entry.get("title", "") or "").strip()
            url = str(entry.get("url", "") or "").strip()
            if not title:
                continue
            normalized_title = _normalize_title(title)
            if keywords and not any(keyword in normalized_title for keyword in keywords):
                continue
            published_at = _parse_feed_datetime(entry.get("published_at", ""))
            if not published_at:
                continue
            try:
                published_dt = datetime.fromisoformat(published_at.replace("Z", "+00:00")).astimezone(timezone.utc)
            except ValueError:
                continue
            if published_dt < cutoff:
                continue
            stable_id = hashlib.sha256(
                (str(entry.get("identifier", "")) or f"{domain}|{normalized_title}|{published_at}").encode("utf-8", errors="ignore")
            ).hexdigest()
            accepted += 1
            items.append(
                {
                    "id": stable_id,
                    "source": "official",
                    "published_at": published_at,
                    "seen_at": seen_at_iso,
                    "currency_tags": [str(feed.get("currency", "")).upper()],
                    "topic_tags": [str(feed.get("topic", ""))],
                    "domain": domain,
                    "title": title,
                    "url": url,
                    "importance": "official",
                    "tone": 0.0,
                    "source_country": "",
                    "language": "en",
                    "tier_weight": max(weight, 1.0),
                    "topic_weight": 1.15,
                    "title_signature": normalized_title,
                    "official_feed_id": str(feed.get("id", "")),
                }
            )
    return items, {
        "errors": errors,
        "query_count": len(feeds),
        "success_count": success_count,
    }
