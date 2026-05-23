from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from .common import OfflineLabError
from .rates_engine_contracts import (
    RATES_ENGINE_CONFIG_PATH,
    RATES_ENGINE_CONFIG_VERSION,
)
from .rates_engine_inputs import SUPPORTED_CURRENCIES


def default_config() -> dict[str, Any]:
    return {
        "schema_version": RATES_ENGINE_CONFIG_VERSION,
        "enabled": True,
        "poll_interval_sec": 120,
        "snapshot_stale_after_sec": 900,
        "history_recent_limit": 120,
        "providers": {
            "manual_inputs_enabled": True,
            "policy_proxy_enabled": True,
            "manual_stale_after_hours": 48,
            "proxy_decay": 0.92,
        },
        "currencies": {
            code: {
                "central_bank_aliases": {
                    "USD": ["federal reserve", "fed", "fomc", "us treasury"],
                    "EUR": ["ecb", "european central bank", "euro area"],
                    "GBP": ["bank of england", "boe", "sterling"],
                    "JPY": ["bank of japan", "boj", "yen"],
                    "CHF": ["snb", "swiss national bank", "franc"],
                    "CAD": ["bank of canada", "boc", "loonie"],
                    "AUD": ["reserve bank of australia", "rba", "aussie"],
                    "NZD": ["reserve bank of new zealand", "rbnz", "kiwi"],
                    "SEK": ["riksbank", "sweden", "krona"],
                    "NOK": ["norges bank", "norway", "krone"],
                }[code],
            }
            for code in SUPPORTED_CURRENCIES
        },
        "event_windows": {
            "pre_cb_event_min": 90,
            "post_cb_event_min": 120,
            "pre_macro_policy_min": 45,
            "post_macro_policy_min": 90,
        },
        "thresholds": {
            "policy_repricing_high": 0.65,
            "policy_surprise_high": 0.55,
            "policy_uncertainty_caution": 0.48,
            "policy_uncertainty_block": 0.78,
            "curve_divergence_meaningful": 0.35,
            "policy_divergence_meaningful": 0.42,
            "meeting_path_reprice_now": 0.60,
            "macro_to_rates_transmission": 0.50,
            "rates_risk_caution": 0.44,
            "rates_risk_block": 0.78,
        },
        "pair_thresholds": {
            "stale_block": True,
            "conflicting_caution_floor": 0.34,
            "event_block_window_min": 10,
            "event_caution_window_min": 35,
            "uncertainty_weight": 0.38,
            "repricing_weight": 0.22,
            "divergence_weight": 0.24,
            "event_weight": 0.16,
        },
        "proxy_model": {
            "base_topic_relevance": {
                "monetary_policy": 1.00,
                "inflation": 0.85,
                "employment_growth": 0.72,
                "banking_stress": 0.60,
                "geopolitical_risk": 0.38,
                "commodity_energy_shock": 0.34,
                "scheduled_macro": 0.55,
            },
            "hawkish_keywords": [
                "hawkish",
                "higher for longer",
                "rate hike",
                "tightening",
                "inflation persistent",
                "restrictive",
                "hot inflation",
                "sticky inflation",
                "strong payrolls",
                "upside surprise",
            ],
            "dovish_keywords": [
                "dovish",
                "rate cut",
                "easing",
                "disinflation",
                "weak payrolls",
                "growth slowdown",
                "lower for longer",
                "downside surprise",
                "soft inflation",
                "recession risk",
            ],
            "cb_event_keywords": [
                "rate decision",
                "policy statement",
                "minutes",
                "meeting",
                "forward guidance",
                "press conference",
                "speech",
            ],
            "macro_policy_keywords": [
                "inflation",
                "cpi",
                "pce",
                "employment",
                "payroll",
                "wages",
                "gdp",
                "retail sales",
            ],
            "topic_direction_bias": {
                "monetary_policy": 1.0,
                "inflation": 0.8,
                "employment_growth": 0.65,
                "banking_stress": -0.55,
                "geopolitical_risk": -0.35,
                "commodity_energy_shock": 0.20,
                "scheduled_macro": 0.0,
            },
            "surprise_scalars": {
                "monetary_policy": 1.0,
                "inflation": 0.9,
                "employment_growth": 0.75,
                "scheduled_macro": 0.65,
            },
            "front_end_proxy_gain": 0.35,
            "expected_path_proxy_gain": 0.52,
            "event_impulse_gain": 0.40,
            "uncertainty_conflict_gain": 0.30,
        },
        "curve_regime_thresholds": {
            "steepening_change": 0.18,
            "flattening_change": -0.18,
            "inversion_level": -0.15,
        },
    }


def _merge_defaults(default_value: Any, payload_value: Any) -> Any:
    if isinstance(default_value, dict):
        payload_dict = payload_value if isinstance(payload_value, dict) else {}
        merged = {key: _merge_defaults(default_value[key], payload_dict.get(key)) for key in default_value}
        for key, value in payload_dict.items():
            if key not in merged:
                merged[key] = deepcopy(value)
        return merged
    if isinstance(default_value, list):
        if isinstance(payload_value, list):
            return deepcopy(payload_value)
        return deepcopy(default_value)
    if payload_value is None:
        return deepcopy(default_value)
    return deepcopy(payload_value)


def ensure_default_files(path: Path | None = None) -> Path:
    path = path or RATES_ENGINE_CONFIG_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text(json.dumps(default_config(), indent=2, sort_keys=True), encoding="utf-8")
    return path


def validate_config_payload(payload: dict[str, Any]) -> dict[str, Any]:
    schema_version = int(payload.get("schema_version", 0) or 0)
    if schema_version != RATES_ENGINE_CONFIG_VERSION:
        raise OfflineLabError(
            f"Rates engine config schema mismatch: expected {RATES_ENGINE_CONFIG_VERSION}, got {schema_version}"
        )
    if int(payload.get("poll_interval_sec", 0) or 0) < 15:
        raise OfflineLabError("Rates engine poll_interval_sec must be at least 15 seconds")
    currencies = payload.get("currencies")
    if not isinstance(currencies, dict) or not currencies:
        raise OfflineLabError("Rates engine config currencies are missing")
    for code in currencies.keys():
        currency = str(code or "").strip().upper()
        if currency not in SUPPORTED_CURRENCIES:
            raise OfflineLabError(f"Unsupported rates-engine config currency: {code}")
    providers = payload.get("providers")
    if not isinstance(providers, dict):
        raise OfflineLabError("Rates engine providers config must be an object")
    thresholds = payload.get("thresholds")
    if not isinstance(thresholds, dict):
        raise OfflineLabError("Rates engine thresholds config must be an object")
    proxy_model = payload.get("proxy_model")
    if not isinstance(proxy_model, dict):
        raise OfflineLabError("Rates engine proxy_model config must be an object")
    return payload


def load_config(path: Path | None = None) -> dict[str, Any]:
    path = path or RATES_ENGINE_CONFIG_PATH
    ensure_default_files(path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise OfflineLabError(f"Rates engine config missing: {path}") from exc
    except json.JSONDecodeError as exc:
        raise OfflineLabError(f"Rates engine config is not valid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise OfflineLabError(f"Rates engine config must be a JSON object: {path}")
    merged = _merge_defaults(default_config(), payload)
    validate_config_payload(merged)
    return merged
