from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import libsql

from .common_schema import PROFILES_DIR
from .common_utils import OfflineLabError, now_unix

MARKET_UNIVERSE_CONFIG_KEY = "market_universe_config_json"
MARKET_UNIVERSE_SCHEMA_VERSION = 1
MARKET_UNIVERSE_DEFAULT_EXPORT = PROFILES_DIR / "market_universe.json"

_DEFAULT_SYMBOL_ROWS: tuple[tuple[str, str, str, str], ...] = (
    ("AUDCAD", "FX", "tradable", ""),
    ("AUDCHF", "FX", "tradable", ""),
    ("AUDJPY", "FX", "tradable", ""),
    ("AUDNZD", "FX", "tradable", ""),
    ("AUDSGD", "FX", "tradable", ""),
    ("AUDUSD", "FX", "tradable", ""),
    ("CADCHF", "FX", "tradable", ""),
    ("CADJPY", "FX", "tradable", ""),
    ("CHFJPY", "FX", "tradable", ""),
    ("CHFSGD", "FX", "tradable", ""),
    ("EURAUD", "FX", "tradable", ""),
    ("EURCAD", "FX", "tradable", ""),
    ("EURCHF", "FX", "tradable", ""),
    ("EURGBP", "FX", "tradable", ""),
    ("EURHKD", "FX", "tradable", ""),
    ("EURNOK", "FX", "tradable", ""),
    ("EURPLN", "FX", "tradable", ""),
    ("EURSEK", "FX", "tradable", ""),
    ("EURSGD", "FX", "tradable", ""),
    ("EURUSD", "FX", "tradable", ""),
    ("EURZAR", "FX", "tradable", ""),
    ("GBPAUD", "FX", "tradable", ""),
    ("GBPCAD", "FX", "tradable", ""),
    ("GBPCHF", "FX", "tradable", ""),
    ("GBPDKK", "FX", "tradable", ""),
    ("GBPJPY", "FX", "tradable", ""),
    ("GBPNOK", "FX", "tradable", ""),
    ("GBPNZD", "FX", "tradable", ""),
    ("GBPSEK", "FX", "tradable", ""),
    ("GBPSGD", "FX", "tradable", ""),
    ("GBPUSD", "FX", "tradable", ""),
    ("NOKJPY", "FX", "tradable", ""),
    ("NOKSEK", "FX", "tradable", ""),
    ("NZDCAD", "FX", "tradable", ""),
    ("NZDCHF", "FX", "tradable", ""),
    ("NZDJPY", "FX", "tradable", ""),
    ("NZDUSD", "FX", "tradable", ""),
    ("SEKJPY", "FX", "tradable", ""),
    ("SGDJPY", "FX", "tradable", ""),
    ("USDCAD", "FX", "tradable", ""),
    ("USDCHF", "FX", "tradable", ""),
    ("USDCNH", "FX", "tradable", ""),
    ("USDCZK", "FX", "tradable", ""),
    ("USDDKK", "FX", "tradable", ""),
    ("USDHKD", "FX", "tradable", ""),
    ("USDHUF", "FX", "tradable", ""),
    ("USDJPY", "FX", "tradable", ""),
    ("USDMXN", "FX", "tradable", ""),
    ("USDNOK", "FX", "tradable", ""),
    ("USDPLN", "FX", "tradable", ""),
    ("USDSEK", "FX", "tradable", ""),
    ("USDSGD", "FX", "tradable", ""),
    ("USDTHB", "FX", "tradable", ""),
    ("USDZAR", "FX", "tradable", ""),
    ("US2000", "USA", "indicator_only", "US Small Cap 2000 Index"),
    ("US30", "USA", "indicator_only", "US Wall Street 30 Index"),
    ("US500", "USA", "indicator_only", "US SPX 500 Index"),
    ("USTEC", "USA", "indicator_only", "US Tech 100 Index"),
    ("XAGUSD", "Metal", "indicator_only", "Silver vs US Dollar"),
    ("XAUUSD", "Metal", "indicator_only", "Gold vs US Dollar"),
    ("XPDUSD", "Metal", "indicator_only", "Palladium vs US Dollar"),
    ("XPTUSD", "Metal", "indicator_only", "Platinum vs US Dollar"),
    ("MidDE50", "EU", "indicator_only", "Germany Mid 50 Index"),
    ("DE40", "EU", "indicator_only", "Germany 40 Index"),
    ("STOXX50", "EU", "indicator_only", "EU Stocks 50 Index"),
    ("TecDE30", "EU", "indicator_only", "Germany Tech 30 Index"),
    ("XBRUSD", "Oil", "indicator_only", "Brent Oil vs US Dollar"),
    ("XNGUSD", "Oil", "indicator_only", "Natrual Gas vs US Dollar"),
    ("XTIUSD", "Oil", "indicator_only", "Crude Oil vs US Dollar"),
    ("BTCUSD", "Crypto", "indicator_only", "Bitcoin"),
    ("LTCUSD", "Crypto", "indicator_only", "Litecoin"),
    ("ETHUSD", "Crypto", "indicator_only", "Ethereum"),
    ("IAU.NYSE", "ETF", "indicator_only", "iShares Gold Trust"),
    ("FXI.NYSE", "ETF", "indicator_only", "iShares China Large-Cap ETF CFD"),
    ("USO.NYSE", "ETF", "indicator_only", "United States Oil ETF CFD"),
    ("UNG.NYSE", "ETF", "indicator_only", "United States Natural Gas ETF CFD"),
    ("APPL.NAS", "Shares", "indicator_only", "Apple"),
    ("MSFT.NAS", "Shares", "indicator_only", "Microsoft"),
    ("NVDA.NAS", "Shares", "indicator_only", "Nvidia"),
    ("GOOG.NAS", "Shares", "indicator_only", "Google"),
    ("XOM.NYSE", "Shares", "indicator_only", "Exxon Mobile"),
    ("BP.LSE", "Shares", "indicator_only", "BP"),
    ("RDSB.LSE", "Shares", "indicator_only", "Shell"),
    ("HK50", "China", "indicator_only", "Hong Kong 50 Index"),
    ("CHINA50", "China", "indicator_only", "FTSE China A50 Index"),
    ("AUS200", "Index", "indicator_only", "Australia 200 Index"),
    ("CA60", "Index", "indicator_only", "Canada 60 Index"),
    ("JP225", "Index", "indicator_only", "Japan 225 Index"),
)


def _metadata_value(conn: libsql.Connection, key: str) -> str:
    row = conn.execute(
        "SELECT meta_value FROM lab_metadata WHERE meta_key = ?",
        (str(key),),
    ).fetchone()
    if row is None or len(row) <= 0:
        return ""
    return str(row[0] or "")


def _write_metadata_value(conn: libsql.Connection, key: str, value: str) -> None:
    conn.execute(
        """
        INSERT INTO lab_metadata(meta_key, meta_value, updated_at)
        VALUES(?, ?, ?)
        ON CONFLICT(meta_key) DO UPDATE SET
            meta_value=excluded.meta_value,
            updated_at=excluded.updated_at
        """,
        (str(key), str(value), now_unix()),
    )


def default_market_universe_config() -> dict[str, Any]:
    symbol_records = [
        {
            "symbol": symbol,
            "asset_class": asset_class,
            "role": role,
            "notes": notes,
        }
        for symbol, asset_class, role, notes in _DEFAULT_SYMBOL_ROWS
    ]
    return {
        "schema_version": MARKET_UNIVERSE_SCHEMA_VERSION,
        "config_id": "fxai_market_universe",
        "trading_scope": "FX_ONLY",
        "description": (
            "Trade FX pairs only. Use non-tradable MT5 securities as "
            "indicator/context symbols only."
        ),
        "symbol_records": symbol_records,
    }


def validate_market_universe_config(payload: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise OfflineLabError("market universe config must be a JSON object")
    schema_version = int(payload.get("schema_version", 0) or 0)
    if schema_version != MARKET_UNIVERSE_SCHEMA_VERSION:
        raise OfflineLabError(
            f"market universe schema_version must be {MARKET_UNIVERSE_SCHEMA_VERSION}"
        )
    trading_scope = str(payload.get("trading_scope", "") or "").strip().upper()
    if trading_scope != "FX_ONLY":
        raise OfflineLabError("market universe trading_scope must be FX_ONLY")
    description = str(payload.get("description", "") or "").strip()
    if not description:
        raise OfflineLabError("market universe description is required")
    symbol_records = payload.get("symbol_records")
    if not isinstance(symbol_records, list) or not symbol_records:
        raise OfflineLabError("market universe symbol_records must be a non-empty array")

    seen_symbols: dict[str, str] = {}
    tradable_count = 0
    indicator_count = 0
    normalized_records: list[dict[str, str]] = []
    for index, record in enumerate(symbol_records):
        if not isinstance(record, dict):
            raise OfflineLabError(f"market universe symbol_records[{index}] must be an object")
        symbol = str(record.get("symbol", "") or "").strip()
        symbol_key = symbol.upper()
        asset_class = str(record.get("asset_class", "") or "").strip()
        role = str(record.get("role", "") or "").strip().lower()
        notes = str(record.get("notes", "") or "").strip()
        if not symbol:
            raise OfflineLabError(f"market universe symbol_records[{index}].symbol is required")
        if symbol_key in seen_symbols:
            raise OfflineLabError(f"duplicate market universe symbol: {symbol}")
        if not asset_class:
            raise OfflineLabError(
                f"market universe symbol_records[{index}].asset_class is required"
            )
        if role not in {"tradable", "indicator_only"}:
            raise OfflineLabError(
                f"market universe symbol_records[{index}].role must be tradable or indicator_only"
            )
        if role == "tradable" and asset_class.upper() != "FX":
            raise OfflineLabError(
                f"tradable market universe symbols must use asset_class FX: {symbol}"
            )
        if role == "tradable":
            tradable_count += 1
        else:
            indicator_count += 1
        seen_symbols[symbol_key] = role
        normalized_records.append(
            {
                "symbol": symbol,
                "asset_class": asset_class,
                "role": role,
                "notes": notes,
            }
        )
    if tradable_count <= 0:
        raise OfflineLabError("market universe must contain at least one tradable symbol")
    if indicator_count <= 0:
        raise OfflineLabError("market universe must contain at least one indicator-only symbol")

    return {
        "schema_version": MARKET_UNIVERSE_SCHEMA_VERSION,
        "config_id": str(payload.get("config_id", "fxai_market_universe") or "fxai_market_universe").strip(),
        "trading_scope": "FX_ONLY",
        "description": description,
        "symbol_records": normalized_records,
    }


def summarize_market_universe_config(payload: dict[str, Any]) -> dict[str, Any]:
    normalized = validate_market_universe_config(payload)
    tradable_symbols = [
        str(record["symbol"])
        for record in normalized["symbol_records"]
        if str(record["role"]) == "tradable"
    ]
    indicator_symbols = [
        str(record["symbol"])
        for record in normalized["symbol_records"]
        if str(record["role"]) == "indicator_only"
    ]
    asset_counts = Counter(str(record["asset_class"]) for record in normalized["symbol_records"])
    role_counts = Counter(str(record["role"]) for record in normalized["symbol_records"])
    return {
        "schema_version": int(normalized["schema_version"]),
        "config_id": str(normalized["config_id"]),
        "trading_scope": str(normalized["trading_scope"]),
        "tradable_symbol_count": int(role_counts.get("tradable", 0)),
        "indicator_symbol_count": int(role_counts.get("indicator_only", 0)),
        "asset_class_counts": dict(sorted(asset_counts.items())),
        "tradable_symbols": tradable_symbols,
        "indicator_only_symbols": indicator_symbols,
    }


def load_market_universe_config(conn: libsql.Connection) -> dict[str, Any]:
    raw_value = _metadata_value(conn, MARKET_UNIVERSE_CONFIG_KEY).strip()
    if not raw_value:
        payload = default_market_universe_config()
        save_market_universe_config(conn, payload)
        return payload
    try:
        payload = json.loads(raw_value)
    except json.JSONDecodeError as exc:
        raise OfflineLabError("market universe config in lab_metadata is not valid JSON") from exc
    return validate_market_universe_config(payload)


def save_market_universe_config(conn: libsql.Connection, payload: dict[str, Any]) -> dict[str, Any]:
    normalized = validate_market_universe_config(payload)
    _write_metadata_value(
        conn,
        MARKET_UNIVERSE_CONFIG_KEY,
        json.dumps(normalized, indent=2, sort_keys=True),
    )
    return normalized


def seed_market_universe_config(conn: libsql.Connection) -> None:
    if _metadata_value(conn, MARKET_UNIVERSE_CONFIG_KEY).strip():
        return
    save_market_universe_config(conn, default_market_universe_config())


def reset_market_universe_config(conn: libsql.Connection) -> dict[str, Any]:
    return save_market_universe_config(conn, default_market_universe_config())


def export_market_universe_config(
    conn: libsql.Connection,
    output_path: Path | str | None = None,
) -> dict[str, Any]:
    payload = load_market_universe_config(conn)
    path = Path(output_path) if output_path is not None else MARKET_UNIVERSE_DEFAULT_EXPORT
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "output_path": str(path),
        "summary": summarize_market_universe_config(payload),
    }


def import_market_universe_config(conn: libsql.Connection, input_path: Path | str) -> dict[str, Any]:
    if not str(input_path or "").strip():
        raise OfflineLabError("market universe import requires a non-empty input path")
    path = Path(input_path)
    if not path.exists() or not path.is_file():
        raise OfflineLabError(f"market universe config file not found: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise OfflineLabError(f"market universe config file is not valid JSON: {path}") from exc
    normalized = save_market_universe_config(conn, payload)
    return {
        "input_path": str(path),
        "summary": summarize_market_universe_config(normalized),
    }
