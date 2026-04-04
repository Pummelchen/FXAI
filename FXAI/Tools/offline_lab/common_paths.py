#!/usr/bin/env python3
from __future__ import annotations

import csv
from pathlib import Path

from .common_schema import COMMON_EXPORT_DIR
from .common_utils import safe_token

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def dataset_data_path(dataset_key: str, symbol: str) -> Path:
    return COMMON_EXPORT_DIR / f"fxai_export_{safe_token(dataset_key)}_{safe_token(symbol)}.tsv"


def dataset_meta_path(dataset_key: str, symbol: str) -> Path:
    return COMMON_EXPORT_DIR / f"fxai_export_{safe_token(dataset_key)}_{safe_token(symbol)}.meta.tsv"


def load_kv_tsv(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not path.exists():
        return out
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            key = (row.get("key", "") or "").strip()
            value = (row.get("value", "") or "").strip()
            if key:
                out[key] = value
    return out
