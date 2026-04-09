from __future__ import annotations

import json
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .common import OfflineLabError
from .rates_engine_contracts import (
    RATES_ENGINE_INPUTS_PATH,
    RATES_ENGINE_INPUTS_VERSION,
    isoformat_utc,
)

SUPPORTED_CURRENCIES = ["USD", "EUR", "GBP", "JPY", "CHF", "CAD", "AUD", "NZD", "SEK", "NOK"]


def default_inputs() -> dict[str, Any]:
    currencies = {}
    for code in SUPPORTED_CURRENCIES:
        currencies[code] = {
            "enabled": True,
            "front_end_level": None,
            "expected_path_level": None,
            "curve_2y_level": None,
            "curve_10y_level": None,
            "curve_slope_2s10s": None,
            "last_update_at": "",
            "basis": "proxy_only",
            "notes": "Leave numeric fields null to use the policy-proxy path only. Fill them with trusted rates inputs to upgrade this currency to true-market mode.",
        }
    return {
        "schema_version": RATES_ENGINE_INPUTS_VERSION,
        "updated_at": isoformat_utc(),
        "currencies": currencies,
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


def ensure_default_inputs_file(path: Path | None = None) -> Path:
    path = path or RATES_ENGINE_INPUTS_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text(json.dumps(default_inputs(), indent=2, sort_keys=True), encoding="utf-8")
    return path


def _safe_float_or_none(value: Any) -> float | None:
    if value in (None, "", "null"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _validate_timestamp(value: Any, field_name: str) -> None:
    text = str(value or "").strip()
    if not text:
        return
    try:
        datetime.fromisoformat(text.replace("Z", "+00:00")).astimezone(timezone.utc)
    except ValueError as exc:
        raise OfflineLabError(f"Rates engine input timestamp is not valid ISO8601: {field_name}") from exc


def validate_inputs_payload(payload: dict[str, Any]) -> dict[str, Any]:
    schema_version = int(payload.get("schema_version", 0) or 0)
    if schema_version != RATES_ENGINE_INPUTS_VERSION:
        raise OfflineLabError(
            f"Rates engine input schema mismatch: expected {RATES_ENGINE_INPUTS_VERSION}, got {schema_version}"
        )
    currencies = payload.get("currencies")
    if not isinstance(currencies, dict) or not currencies:
        raise OfflineLabError("Rates engine inputs must define currencies")
    for code, spec in currencies.items():
        currency = str(code or "").strip().upper()
        if currency not in SUPPORTED_CURRENCIES:
            raise OfflineLabError(f"Unsupported rates-engine input currency: {code}")
        if not isinstance(spec, dict):
            raise OfflineLabError(f"Rates engine currency input must be an object: {currency}")
        for field_name in ("front_end_level", "expected_path_level", "curve_2y_level", "curve_10y_level", "curve_slope_2s10s"):
            numeric_value = _safe_float_or_none(spec.get(field_name))
            if spec.get(field_name) not in (None, "", "null") and numeric_value is None:
                raise OfflineLabError(f"Rates engine input field must be numeric or null: {currency}.{field_name}")
        _validate_timestamp(spec.get("last_update_at"), f"{currency}.last_update_at")
    return payload


def load_inputs(path: Path | None = None) -> dict[str, Any]:
    path = path or RATES_ENGINE_INPUTS_PATH
    ensure_default_inputs_file(path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise OfflineLabError(f"Rates engine inputs missing: {path}") from exc
    except json.JSONDecodeError as exc:
        raise OfflineLabError(f"Rates engine inputs are not valid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise OfflineLabError(f"Rates engine inputs must be a JSON object: {path}")
    merged = _merge_defaults(default_inputs(), payload)
    validate_inputs_payload(merged)
    return merged
