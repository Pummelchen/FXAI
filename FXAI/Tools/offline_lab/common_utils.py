#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timezone

import fxai_testlab as testlab

from .common_schema import DEFAULT_MONTHS_LIST


class OfflineLabError(RuntimeError):
    pass


def now_unix() -> int:
    return int(time.time())


def safe_token(raw: str) -> str:
    text = (raw or "").strip()
    if not text:
        return "default"
    for ch in "\\/:*?\"<>|{}[](),;= \t\r\n":
        text = text.replace(ch, "_")
    while "__" in text:
        text = text.replace("__", "_")
    return text.strip("_") or "default"


def json_compact(payload) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def months_back(anchor: datetime, months: int) -> datetime:
    if months <= 0:
        return anchor
    month_index = anchor.month - 1 - months
    year = anchor.year + month_index // 12
    month = month_index % 12 + 1
    day = anchor.day
    while day > 28:
        try:
            return anchor.replace(year=year, month=month, day=day)
        except ValueError:
            day -= 1
    return anchor.replace(year=year, month=month, day=day)


def resolve_window(months: int, start_unix: int, end_unix: int) -> tuple[int, int]:
    if start_unix > 0 and end_unix > start_unix:
        return int(start_unix), int(end_unix)
    end_dt = datetime.fromtimestamp((end_unix if end_unix > 0 else now_unix()), tz=timezone.utc)
    start_dt = months_back(end_dt, max(months, 1))
    return int(start_dt.timestamp()), int(end_dt.timestamp())


def parse_csv_tokens(raw: str) -> list[str]:
    text = (raw or "").strip()
    if not text:
        return []
    text = text.replace("{", "").replace("}", "").replace(";", ",").replace("|", ",")
    out: list[str] = []
    for part in text.split(","):
        token = part.strip()
        if token and token not in out:
            out.append(token)
    return out


def resolve_symbols(args) -> list[str]:
    pack_name = (getattr(args, "symbol_pack", "") or "").strip().lower()
    if pack_name:
        return list(testlab.SYMBOL_PACKS.get(pack_name, [str(getattr(args, "symbol", "EURUSD")).upper()]))
    symbols = parse_csv_tokens(getattr(args, "symbol_list", ""))
    if not symbols:
        symbol = str(getattr(args, "symbol", "EURUSD") or "").strip()
        if symbol:
            symbols = [symbol]
    return [s.upper() for s in symbols]


def resolve_months_list(raw: str) -> list[int]:
    items = parse_csv_tokens(raw)
    out: list[int] = []
    for item in items:
        try:
            months = int(item)
        except Exception:
            continue
        if months > 0 and months not in out:
            out.append(months)
    return out or list(DEFAULT_MONTHS_LIST)
