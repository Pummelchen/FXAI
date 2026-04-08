from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

from .common import OfflineLabError
from .newspulse_contracts import (
    NEWSPULSE_CONFIG_PATH,
    NEWSPULSE_CONFIG_VERSION,
    NEWSPULSE_SOURCES_PATH,
    ensure_newspulse_dirs,
)

SUPPORTED_CURRENCIES = ["USD", "EUR", "GBP", "JPY", "CHF", "CAD", "AUD", "NZD", "SEK", "NOK"]


def default_config() -> dict[str, object]:
    return {
        "schema_version": NEWSPULSE_CONFIG_VERSION,
        "enabled": True,
        "poll_interval_sec": 60,
        "calendar_stale_after_sec": 360,
        "gdelt_stale_after_sec": 360,
        "snapshot_stale_after_sec": 360,
        "unknown_blocks": True,
        "history_recent_limit": 80,
        "calendar": {
            "high_impact_min": 2,
            "pre_window_min_by_importance": {"1": 10, "2": 25, "3": 45},
            "post_window_min_by_importance": {"1": 10, "2": 25, "3": 45},
            "bootstrap_lookback_hours": 36,
            "lookahead_days": 7,
        },
        "gdelt": {
            "request_timeout_sec": 6,
            "timespan": "30min",
            "maxrecords": 25,
            "sort": "datedesc",
            "tone_abs_floor": 0.0,
            "max_query_sets_per_cycle": 8,
            "max_cycle_runtime_sec": 12,
            "rate_limit_backoff_sec": 300,
            "allow_insecure_tls_fallback": False,
        },
        "scoring": {
            "intensity_half_life_min": 12.0,
            "baseline_alpha": 0.18,
            "burst_block_threshold": 1.8,
            "caution_risk_threshold": 0.45,
            "block_risk_threshold": 0.78,
            "stale_risk_score": 0.92,
            "post_release_hot_minutes": 8,
        },
        "pairs": {
            "currencies": SUPPORTED_CURRENCIES,
            "include_permutations": True,
        },
        "currencies": {
            "USD": {
                "aliases": ["USD", "dollar", "Federal Reserve", "Fed", "Treasury", "FOMC"],
                "topic_groups": ["monetary_policy", "inflation", "employment_growth", "banking_stress", "geopolitical_risk"],
                "source_countries": ["us"],
            },
            "EUR": {
                "aliases": ["EUR", "euro", "ECB", "Eurozone", "European Central Bank"],
                "topic_groups": ["monetary_policy", "inflation", "employment_growth", "banking_stress", "geopolitical_risk"],
                "source_countries": ["eu", "de", "fr", "it"],
            },
            "GBP": {
                "aliases": ["GBP", "sterling", "pound", "Bank of England", "BoE"],
                "topic_groups": ["monetary_policy", "inflation", "employment_growth", "banking_stress", "geopolitical_risk"],
                "source_countries": ["gb", "uk"],
            },
            "JPY": {
                "aliases": ["JPY", "yen", "Bank of Japan", "BoJ", "Japan"],
                "topic_groups": ["monetary_policy", "inflation", "employment_growth", "banking_stress", "geopolitical_risk"],
                "source_countries": ["jp"],
            },
            "CHF": {
                "aliases": ["CHF", "franc", "Swiss National Bank", "SNB", "Switzerland"],
                "topic_groups": ["monetary_policy", "inflation", "banking_stress", "geopolitical_risk"],
                "source_countries": ["ch"],
            },
            "CAD": {
                "aliases": ["CAD", "loonie", "Bank of Canada", "BoC", "Canada"],
                "topic_groups": ["monetary_policy", "inflation", "employment_growth", "commodity_energy_shock"],
                "source_countries": ["ca"],
            },
            "AUD": {
                "aliases": ["AUD", "aussie", "Reserve Bank of Australia", "RBA", "Australia"],
                "topic_groups": ["monetary_policy", "inflation", "employment_growth", "commodity_energy_shock"],
                "source_countries": ["au"],
            },
            "NZD": {
                "aliases": ["NZD", "kiwi", "Reserve Bank of New Zealand", "RBNZ", "New Zealand"],
                "topic_groups": ["monetary_policy", "inflation", "employment_growth", "commodity_energy_shock"],
                "source_countries": ["nz"],
            },
            "SEK": {
                "aliases": ["SEK", "krona", "Riksbank", "Sweden"],
                "topic_groups": ["monetary_policy", "inflation", "employment_growth"],
                "source_countries": ["se"],
            },
            "NOK": {
                "aliases": ["NOK", "krone", "Norges Bank", "Norway"],
                "topic_groups": ["monetary_policy", "inflation", "employment_growth", "commodity_energy_shock"],
                "source_countries": ["no"],
            },
        },
        "topic_groups": {
            "monetary_policy": {
                "query_terms": ['"central bank"', '"interest rate"', '"rate decision"', '"policy rate"', '"monetary policy"'],
                "weight": 1.0,
            },
            "inflation": {
                "query_terms": ['inflation', '"consumer prices"', 'CPI', 'PPI'],
                "weight": 0.9,
            },
            "employment_growth": {
                "query_terms": ['employment', '"labor market"', '"jobs report"', 'GDP', 'growth'],
                "weight": 0.82,
            },
            "banking_stress": {
                "query_terms": ['"banking stress"', '"bank failure"', '"financial stability"', '"liquidity crisis"', '"credit stress"'],
                "weight": 1.0,
            },
            "geopolitical_risk": {
                "query_terms": ['sanctions', '"trade war"', 'geopolitical', 'conflict', 'tariff'],
                "weight": 0.78,
            },
            "commodity_energy_shock": {
                "query_terms": ['oil', 'gas', 'energy', 'commodity', 'crude'],
                "weight": 0.74,
            },
        },
    }


def default_sources() -> dict[str, object]:
    return {
        "schema_version": NEWSPULSE_CONFIG_VERSION,
        "tier_weights": {
            "official": 1.25,
            "top_tier_macro_financial": 1.0,
            "secondary": 0.72,
            "disabled": 0.0,
        },
        "domains": {
            "reuters.com": {"tier": "top_tier_macro_financial", "enabled": True},
            "bloomberg.com": {"tier": "top_tier_macro_financial", "enabled": True},
            "ft.com": {"tier": "top_tier_macro_financial", "enabled": True},
            "wsj.com": {"tier": "secondary", "enabled": True},
            "cnbc.com": {"tier": "secondary", "enabled": True},
            "marketwatch.com": {"tier": "secondary", "enabled": True},
            "federalreserve.gov": {"tier": "official", "enabled": True, "currencies": ["USD"]},
            "ecb.europa.eu": {"tier": "official", "enabled": True, "currencies": ["EUR"]},
            "bankofengland.co.uk": {"tier": "official", "enabled": True, "currencies": ["GBP"]},
            "boj.or.jp": {"tier": "official", "enabled": True, "currencies": ["JPY"]},
            "snb.ch": {"tier": "official", "enabled": True, "currencies": ["CHF"]},
            "bankofcanada.ca": {"tier": "official", "enabled": True, "currencies": ["CAD"]},
            "rba.gov.au": {"tier": "official", "enabled": True, "currencies": ["AUD"]},
            "rbnz.govt.nz": {"tier": "official", "enabled": True, "currencies": ["NZD"]},
            "riksbank.se": {"tier": "official", "enabled": True, "currencies": ["SEK"]},
            "norges-bank.no": {"tier": "official", "enabled": True, "currencies": ["NOK"]},
            "bls.gov": {"tier": "official", "enabled": True, "currencies": ["USD"]},
            "bea.gov": {"tier": "official", "enabled": True, "currencies": ["USD"]},
            "statistics.govt.nz": {"tier": "official", "enabled": True, "currencies": ["NZD"]},
            "ons.gov.uk": {"tier": "official", "enabled": True, "currencies": ["GBP"]},
            "statcan.gc.ca": {"tier": "official", "enabled": True, "currencies": ["CAD"]},
            "destatis.de": {"tier": "official", "enabled": True, "currencies": ["EUR"]},
            "bfs.admin.ch": {"tier": "official", "enabled": True, "currencies": ["CHF"]},
        },
    }


def ensure_default_files() -> dict[str, Path]:
    ensure_newspulse_dirs()
    if not NEWSPULSE_CONFIG_PATH.exists():
        NEWSPULSE_CONFIG_PATH.write_text(json.dumps(default_config(), indent=2, sort_keys=True), encoding="utf-8")
    if not NEWSPULSE_SOURCES_PATH.exists():
        NEWSPULSE_SOURCES_PATH.write_text(json.dumps(default_sources(), indent=2, sort_keys=True), encoding="utf-8")
    return {
        "config_path": NEWSPULSE_CONFIG_PATH,
        "sources_path": NEWSPULSE_SOURCES_PATH,
    }


def load_json_file(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise OfflineLabError(f"NewsPulse config missing: {path}") from exc
    except json.JSONDecodeError as exc:
        raise OfflineLabError(f"NewsPulse config is not valid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise OfflineLabError(f"NewsPulse config must be a JSON object: {path}")
    return payload


def validate_config_payload(payload: dict[str, object], sources: dict[str, object] | None = None) -> dict[str, object]:
    schema_version = int(payload.get("schema_version", 0) or 0)
    if schema_version != NEWSPULSE_CONFIG_VERSION:
        raise OfflineLabError(
            f"NewsPulse config schema mismatch: expected {NEWSPULSE_CONFIG_VERSION}, got {schema_version}"
        )

    poll_interval = int(payload.get("poll_interval_sec", 0) or 0)
    if poll_interval < 15:
        raise OfflineLabError("NewsPulse poll_interval_sec must be at least 15 seconds")

    gdelt = payload.get("gdelt")
    if not isinstance(gdelt, dict):
        raise OfflineLabError("NewsPulse gdelt config is missing")
    if int(gdelt.get("maxrecords", 0) or 0) <= 0:
        raise OfflineLabError("NewsPulse gdelt.maxrecords must be positive")
    if not str(gdelt.get("timespan", "") or "").strip():
        raise OfflineLabError("NewsPulse gdelt.timespan must be configured")
    if int(gdelt.get("request_timeout_sec", 0) or 0) <= 0:
        raise OfflineLabError("NewsPulse gdelt.request_timeout_sec must be positive")
    if int(gdelt.get("max_query_sets_per_cycle", 0) or 0) <= 0:
        raise OfflineLabError("NewsPulse gdelt.max_query_sets_per_cycle must be positive")
    if int(gdelt.get("max_cycle_runtime_sec", 0) or 0) <= 0:
        raise OfflineLabError("NewsPulse gdelt.max_cycle_runtime_sec must be positive")
    if int(gdelt.get("rate_limit_backoff_sec", 0) or 0) <= 0:
        raise OfflineLabError("NewsPulse gdelt.rate_limit_backoff_sec must be positive")

    currencies = payload.get("currencies")
    if not isinstance(currencies, dict) or not currencies:
        raise OfflineLabError("NewsPulse currencies config is missing")
    for currency, spec in currencies.items():
        if currency not in SUPPORTED_CURRENCIES:
            raise OfflineLabError(f"NewsPulse currency not supported in phase 1: {currency}")
        if not isinstance(spec, dict):
            raise OfflineLabError(f"NewsPulse currency spec must be an object: {currency}")
        aliases = spec.get("aliases")
        topics = spec.get("topic_groups")
        if not isinstance(aliases, list) or not aliases:
            raise OfflineLabError(f"NewsPulse aliases missing for {currency}")
        if not isinstance(topics, list) or not topics:
            raise OfflineLabError(f"NewsPulse topic_groups missing for {currency}")

    topic_groups = payload.get("topic_groups")
    if not isinstance(topic_groups, dict) or not topic_groups:
        raise OfflineLabError("NewsPulse topic_groups config is missing")
    for topic_name, spec in topic_groups.items():
        if not isinstance(spec, dict):
            raise OfflineLabError(f"NewsPulse topic group must be an object: {topic_name}")
        if not isinstance(spec.get("query_terms"), list) or not spec["query_terms"]:
            raise OfflineLabError(f"NewsPulse topic group has no query_terms: {topic_name}")

    if sources is not None:
        tier_weights = sources.get("tier_weights")
        domains = sources.get("domains")
        if not isinstance(tier_weights, dict) or not isinstance(domains, dict):
            raise OfflineLabError("NewsPulse sources config is missing tier_weights or domains")
        if not domains:
            raise OfflineLabError("NewsPulse sources config must whitelist at least one domain")
        for domain, spec in domains.items():
            if not isinstance(spec, dict):
                raise OfflineLabError(f"NewsPulse source entry must be an object: {domain}")
            tier = str(spec.get("tier", "") or "")
            if tier not in tier_weights:
                raise OfflineLabError(f"NewsPulse source tier is unknown for {domain}: {tier}")

    return payload


def load_config() -> tuple[dict[str, object], dict[str, object]]:
    ensure_default_files()
    config = load_json_file(NEWSPULSE_CONFIG_PATH)
    sources = load_json_file(NEWSPULSE_SOURCES_PATH)
    validate_config_payload(config, sources)
    return config, sources


def build_query_specs(config: dict[str, object]) -> list[dict[str, object]]:
    currencies = config["currencies"]
    topics = config["topic_groups"]
    gdelt = config["gdelt"]
    specs: list[dict[str, object]] = []
    for currency, currency_spec in currencies.items():
        aliases = [str(item).strip() for item in currency_spec.get("aliases", []) if str(item).strip()]
        source_countries = [str(item).strip().lower() for item in currency_spec.get("source_countries", []) if str(item).strip()]
        for topic_name in currency_spec.get("topic_groups", []):
            topic_spec = topics.get(str(topic_name), {})
            query_terms = [str(item).strip() for item in topic_spec.get("query_terms", []) if str(item).strip()]
            if not aliases or not query_terms:
                continue
            specs.append(
                {
                    "currency": currency,
                    "topic": str(topic_name),
                    "aliases": aliases,
                    "source_countries": source_countries,
                    "query_terms": query_terms,
                    "timespan": str(gdelt.get("timespan", "30min")),
                    "maxrecords": int(gdelt.get("maxrecords", 25) or 25),
                    "sort": str(gdelt.get("sort", "datedesc")),
                    "tone_abs_floor": float(gdelt.get("tone_abs_floor", 0.0) or 0.0),
                    "weight": float(topic_spec.get("weight", 1.0) or 1.0),
                }
            )
    return specs


def domain_weight(sources: dict[str, object], domain: str) -> float:
    domains = sources.get("domains", {})
    if not isinstance(domains, dict):
        return 0.0
    spec = domains.get(domain.lower(), {})
    if not isinstance(spec, dict) or not bool(spec.get("enabled", False)):
        return 0.0
    tier = str(spec.get("tier", "disabled") or "disabled")
    tier_weights = sources.get("tier_weights", {})
    if not isinstance(tier_weights, dict):
        return 0.0
    return float(tier_weights.get(tier, 0.0) or 0.0)


def source_spec(sources: dict[str, object], domain: str) -> dict[str, object]:
    domains = sources.get("domains", {})
    if not isinstance(domains, dict):
        return {}
    spec = domains.get(domain.lower(), {})
    return deepcopy(spec) if isinstance(spec, dict) else {}
