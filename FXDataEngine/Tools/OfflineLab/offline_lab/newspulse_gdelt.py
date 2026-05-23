from __future__ import annotations

import hashlib
import json
import re
import ssl
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse

from .common import OfflineLabError
from .newspulse_config import domain_weight, source_spec

_TITLE_SANITIZER = re.compile(r"[^a-z0-9]+")


def _normalize_domain(value: str) -> str:
    domain = str(value or "").strip().lower()
    domain = domain.removeprefix("https://").removeprefix("http://")
    domain = domain.split("/", 1)[0]
    if domain.startswith("www."):
        domain = domain[4:]
    return domain


def _normalize_title(value: str) -> str:
    text = _TITLE_SANITIZER.sub(" ", str(value or "").strip().lower())
    return " ".join(token for token in text.split() if token)


def _parse_seendate(value: str) -> str:
    text = str(value or "").strip()
    if not text:
        return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    for fmt in ("%Y%m%dT%H%M%SZ", "%Y%m%d%H%M%S", "%Y-%m-%dT%H:%M:%SZ"):
        try:
            dt = datetime.strptime(text, fmt).replace(tzinfo=timezone.utc)
            return dt.isoformat().replace("+00:00", "Z")
        except ValueError:
            continue
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _extract_articles(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict):
        for key in ("articles", "articles_results", "entries", "items"):
            value = payload.get(key)
            if isinstance(value, list):
                return [row for row in value if isinstance(row, dict)]
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    return []


def _tone_value(raw: dict[str, Any]) -> float:
    for key in ("tone", "Tone", "sentiment", "tone_score"):
        value = raw.get(key)
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return 0.0


def build_gdelt_url(query_spec: dict[str, Any], sources: dict[str, Any]) -> str:
    currency_clause = "(" + " OR ".join(
        f"\"{alias}\"" if " " in alias else alias for alias in query_spec.get("aliases", [])
    ) + ")"
    topic_clause = "(" + " OR ".join(str(term) for term in query_spec.get("query_terms", [])) + ")"
    domains = [
        f"domainis:{domain}"
        for domain, spec in (sources.get("domains", {}) or {}).items()
        if isinstance(spec, dict) and bool(spec.get("enabled", False))
    ]
    domain_clause = "(" + " OR ".join(domains) + ")" if domains else ""
    country_clause = ""
    source_countries = [str(item).strip() for item in query_spec.get("source_countries", []) if str(item).strip()]
    if source_countries:
        country_clause = "(" + " OR ".join(f"sourcecountry:{country}" for country in source_countries) + ")"
    filters = [currency_clause, topic_clause, domain_clause, country_clause]
    tone_floor = float(query_spec.get("tone_abs_floor", 0.0) or 0.0)
    if tone_floor > 0.0:
        filters.append(f"toneabs>{tone_floor:g}")
    query = " ".join(part for part in filters if part)
    return "https://api.gdeltproject.org/api/v2/doc/doc?" + urllib.parse.urlencode(
        {
            "query": query,
            "mode": "artlist",
            "maxrecords": str(int(query_spec.get("maxrecords", 25) or 25)),
            "timespan": str(query_spec.get("timespan", "30min") or "30min"),
            "sort": str(query_spec.get("sort", "datedesc") or "datedesc"),
            "format": "json",
        }
    )


def _ssl_context(allow_insecure_tls_fallback: bool):
    try:
        import certifi  # type: ignore

        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        if allow_insecure_tls_fallback:
            return ssl._create_unverified_context()
        return ssl.create_default_context()


def fetch_query_results(query_spec: dict[str, Any],
                        sources: dict[str, Any],
                        request_timeout_sec: int,
                        allow_insecure_tls_fallback: bool) -> list[dict[str, Any]]:
    url = build_gdelt_url(query_spec, sources)
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/json",
            "User-Agent": "FXAI-NewsPulse/1.0 (+https://github.com/Pummelchen/FXAI)",
        },
    )
    context = _ssl_context(allow_insecure_tls_fallback)
    with urllib.request.urlopen(request, timeout=request_timeout_sec, context=context) as response:
        payload = json.loads(response.read().decode("utf-8"))
    return _extract_articles(payload)


def normalize_articles(raw_articles: list[dict[str, Any]],
                       query_spec: dict[str, Any],
                       sources: dict[str, Any],
                       seen_at_iso: str) -> list[dict[str, Any]]:
    accepted: list[dict[str, Any]] = []
    for raw in raw_articles:
        url = str(raw.get("url") or raw.get("url_mobile") or raw.get("urlamp") or "").strip()
        domain = _normalize_domain(str(raw.get("domain") or urlparse(url).netloc or ""))
        if not domain:
            continue
        weight = domain_weight(sources, domain)
        if weight <= 0.0:
            continue
        domain_meta = source_spec(sources, domain)
        allowed_currencies = {str(item).upper() for item in domain_meta.get("currencies", []) if str(item).strip()}
        if allowed_currencies and str(query_spec["currency"]).upper() not in allowed_currencies:
            continue

        title = str(raw.get("title") or raw.get("title_english") or "").strip()
        if not title:
            continue

        source_country = str(raw.get("sourcecountry") or raw.get("sourceCountry") or "").strip().lower()
        allowed_countries = {str(item).lower() for item in domain_meta.get("source_countries", []) if str(item).strip()}
        if allowed_countries and source_country and source_country not in allowed_countries:
            continue

        language = str(raw.get("language") or raw.get("sourcelang") or "").strip().lower()
        allowed_languages = {str(item).lower() for item in domain_meta.get("languages", []) if str(item).strip()}
        if allowed_languages and language and language not in allowed_languages:
            continue

        published_at = _parse_seendate(str(raw.get("seendate") or raw.get("seendateutc") or raw.get("date") or ""))
        title_sig = _normalize_title(title)
        stable_id = hashlib.sha256(
            (url or f"{domain}|{title_sig}|{published_at}").encode("utf-8", errors="ignore")
        ).hexdigest()
        tone = _tone_value(raw)
        accepted.append(
            {
                "id": stable_id,
                "source": "gdelt",
                "published_at": published_at,
                "seen_at": seen_at_iso,
                "currency_tags": [str(query_spec["currency"]).upper()],
                "topic_tags": [str(query_spec["topic"])],
                "domain": domain,
                "title": title,
                "url": url,
                "importance": "",
                "tone": tone,
                "source_country": source_country,
                "language": language,
                "tier_weight": weight,
                "topic_weight": float(query_spec.get("weight", 1.0) or 1.0),
                "title_signature": title_sig,
            }
        )
    return accepted


def merge_normalized_articles(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    fallback_index: dict[tuple[str, str], str] = {}
    for item in items:
        key = str(item["id"])
        signature_key = (str(item.get("domain", "")), str(item.get("title_signature", "")))
        if key not in merged and signature_key in fallback_index:
            key = fallback_index[signature_key]
        current = merged.get(key)
        if current is None:
            current = dict(item)
            current["currency_tags"] = list(dict.fromkeys(item.get("currency_tags", [])))
            current["topic_tags"] = list(dict.fromkeys(item.get("topic_tags", [])))
            merged[key] = current
            fallback_index[signature_key] = key
            continue
        current["currency_tags"] = list(dict.fromkeys(list(current.get("currency_tags", [])) + list(item.get("currency_tags", []))))
        current["topic_tags"] = list(dict.fromkeys(list(current.get("topic_tags", [])) + list(item.get("topic_tags", []))))
        current["tier_weight"] = max(float(current.get("tier_weight", 0.0) or 0.0), float(item.get("tier_weight", 0.0) or 0.0))
        current["topic_weight"] = max(float(current.get("topic_weight", 0.0) or 0.0), float(item.get("topic_weight", 0.0) or 0.0))
        if str(item.get("seen_at", "")) > str(current.get("seen_at", "")):
            current["seen_at"] = item["seen_at"]
    return list(merged.values())


def query_gdelt(config: dict[str, Any],
                sources: dict[str, Any],
                query_specs: list[dict[str, Any]],
                seen_at_iso: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    request_timeout_sec = int(config["gdelt"].get("request_timeout_sec", 20) or 20)
    max_cycle_runtime_sec = int(config["gdelt"].get("max_cycle_runtime_sec", 12) or 12)
    allow_insecure_tls_fallback = bool(config["gdelt"].get("allow_insecure_tls_fallback", False))
    all_items: list[dict[str, Any]] = []
    errors: list[str] = []
    success_count = 0
    attempted_queries = 0
    throttled = False
    budget_exhausted = False
    started_at = time.monotonic()
    for spec in query_specs:
        if attempted_queries > 0 and (time.monotonic() - started_at) >= max_cycle_runtime_sec:
            budget_exhausted = True
            errors.append("gdelt_cycle_budget_exhausted")
            break
        attempted_queries += 1
        try:
            raw_items = fetch_query_results(spec, sources, request_timeout_sec, allow_insecure_tls_fallback)
            success_count += 1
        except HTTPError as exc:
            errors.append(f"{spec['currency']}:{spec['topic']}:HTTP {exc.code}")
            if exc.code == 429:
                throttled = True
                break
            continue
        except URLError as exc:
            errors.append(f"{spec['currency']}:{spec['topic']}:network:{exc.reason}")
            continue
        except json.JSONDecodeError:
            errors.append(f"{spec['currency']}:{spec['topic']}:invalid_json")
            continue
        except Exception as exc:
            errors.append(f"{spec['currency']}:{spec['topic']}:{exc}")
            continue
        all_items.extend(normalize_articles(raw_items, spec, sources, seen_at_iso))
    return merge_normalized_articles(all_items), {
        "errors": errors,
        "query_count": attempted_queries,
        "success_count": success_count,
        "throttled": throttled,
        "budget_exhausted": budget_exhausted,
    }
