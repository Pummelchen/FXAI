#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import sys
import tempfile
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path

import fxai_testlab as testlab
import libsql

from .db_backend import *
from .common_schema import *
from .common_utils import *
from .common_db import *
from .common_stats import *
from .common_paths import *
from .market_universe import *


def portable_artifact_path(value: str | Path) -> str:
    raw = str(value or "")
    if not raw:
        return raw
    candidate = Path(raw)
    replacements = (
        (COMMON_PROMOTION_DIR, "<FXAI_PROMOTION_DIR>"),
        (COMMON_EXPORT_DIR, "<FXAI_EXPORT_DIR>"),
        (SHADOW_LEDGER_DIR, "<FXAI_RUNTIME_DIR>"),
        (testlab.TESTER_PRESET_DIR, "<FXAI_TESTER_PRESET_DIR>"),
        (OFFLINE_DIR, "<FXAI_ROOT>/Tools/OfflineLab"),
        (testlab.ROOT, "<FXAI_ROOT>"),
    )
    for base, token in replacements:
        try:
            relative = candidate.relative_to(base)
        except ValueError:
            continue
        return token if str(relative) == "." else f"{token}/{relative.as_posix()}"
    return raw


def portableize_payload_paths(payload):
    if isinstance(payload, dict):
        return {key: portableize_payload_paths(value) for key, value in payload.items()}
    if isinstance(payload, list):
        return [portableize_payload_paths(item) for item in payload]
    if isinstance(payload, str) and ("/" in payload or "\\" in payload):
        return portable_artifact_path(payload)
    return payload
