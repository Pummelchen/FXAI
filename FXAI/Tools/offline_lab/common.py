#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import sqlite3
import sys
import tempfile
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import fxai_testlab as testlab

OFFLINE_DIR = Path(__file__).resolve().parent.parent / "OfflineLab"
DEFAULT_DB = OFFLINE_DIR / "fxai_offline_lab.sqlite"
RUNS_DIR = OFFLINE_DIR / "Runs"
PROFILES_DIR = OFFLINE_DIR / "Profiles"
RESEARCH_DIR = OFFLINE_DIR / "ResearchOS"
DISTILL_DIR = OFFLINE_DIR / "Distillation"
COMMON_EXPORT_DIR = testlab.COMMON_FILES / "FXAI/Offline/Exports"
COMMON_PROMOTION_DIR = testlab.COMMON_FILES / "FXAI/Offline/Promotions"

SERIOUS_SCENARIOS = "{market_recent, market_trend, market_chop, market_session_edges, market_spread_shock, market_walkforward, market_macro_event, market_adversarial}"
DEFAULT_MONTHS_LIST = [3, 6, 12]
DEFAULT_HORIZON_CANDIDATES = [3, 5, 8, 13, 21, 34]
DEFAULT_M1SYNC_CANDIDATES = [2, 3, 5, 8]
DEFAULT_EXECUTION_PROFILES = ["default", "tight-fx", "prime-ecn", "retail-fx", "stress"]
EXPORT_EXPERT = r"FXAI\Tests\FXAI_OfflineExportRunner.ex5"

SQL_SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS datasets (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    dataset_key TEXT NOT NULL UNIQUE,
    group_key TEXT NOT NULL DEFAULT '',
    symbol TEXT NOT NULL,
    timeframe TEXT NOT NULL,
    start_unix INTEGER NOT NULL,
    end_unix INTEGER NOT NULL,
    months INTEGER NOT NULL DEFAULT 0,
    bars INTEGER NOT NULL DEFAULT 0,
    source_path TEXT NOT NULL,
    source_sha256 TEXT NOT NULL DEFAULT '',
    created_at INTEGER NOT NULL,
    notes TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS dataset_bars (
    dataset_id INTEGER NOT NULL,
    bar_time_unix INTEGER NOT NULL,
    open REAL NOT NULL,
    high REAL NOT NULL,
    low REAL NOT NULL,
    close REAL NOT NULL,
    spread_points INTEGER NOT NULL,
    tick_volume INTEGER NOT NULL,
    real_volume INTEGER NOT NULL,
    PRIMARY KEY(dataset_id, bar_time_unix),
    FOREIGN KEY(dataset_id) REFERENCES datasets(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS tuning_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    dataset_id INTEGER NOT NULL,
    profile_name TEXT NOT NULL,
    group_key TEXT NOT NULL DEFAULT '',
    symbol TEXT NOT NULL,
    plugin_name TEXT NOT NULL,
    ai_id INTEGER NOT NULL,
    family_id INTEGER NOT NULL DEFAULT 11,
    experiment_name TEXT NOT NULL,
    param_hash TEXT NOT NULL UNIQUE,
    parameters_json TEXT NOT NULL,
    report_path TEXT NOT NULL DEFAULT '',
    raw_report_path TEXT NOT NULL DEFAULT '',
    summary_path TEXT NOT NULL DEFAULT '',
    manifest_path TEXT NOT NULL DEFAULT '',
    score REAL NOT NULL DEFAULT 0.0,
    grade TEXT NOT NULL DEFAULT 'F',
    issue_count INTEGER NOT NULL DEFAULT 0,
    issues_json TEXT NOT NULL DEFAULT '[]',
    market_recent_score REAL NOT NULL DEFAULT 0.0,
    walkforward_score REAL NOT NULL DEFAULT 0.0,
    adversarial_score REAL NOT NULL DEFAULT 0.0,
    macro_event_score REAL NOT NULL DEFAULT 0.0,
    status TEXT NOT NULL DEFAULT 'pending',
    started_at INTEGER NOT NULL,
    finished_at INTEGER NOT NULL DEFAULT 0,
    FOREIGN KEY(dataset_id) REFERENCES datasets(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS run_scenarios (
    run_id INTEGER NOT NULL,
    scenario TEXT NOT NULL,
    score REAL NOT NULL DEFAULT 0.0,
    calibration_error REAL NOT NULL DEFAULT 0.0,
    path_quality_error REAL NOT NULL DEFAULT 0.0,
    wf_pbo REAL NOT NULL DEFAULT 0.0,
    wf_dsr REAL NOT NULL DEFAULT 0.0,
    wf_pass_rate REAL NOT NULL DEFAULT 0.0,
    net_signal REAL NOT NULL DEFAULT 0.0,
    issue_flags INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY(run_id, scenario),
    FOREIGN KEY(run_id) REFERENCES tuning_runs(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS best_configs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    dataset_scope TEXT NOT NULL,
    dataset_id INTEGER,
    profile_name TEXT NOT NULL,
    symbol TEXT NOT NULL,
    plugin_name TEXT NOT NULL,
    ai_id INTEGER NOT NULL,
    family_id INTEGER NOT NULL DEFAULT 11,
    run_id INTEGER,
    promoted_at INTEGER NOT NULL,
    score REAL NOT NULL DEFAULT 0.0,
    ranking_score REAL NOT NULL DEFAULT 0.0,
    support_count INTEGER NOT NULL DEFAULT 0,
    parameters_json TEXT NOT NULL,
    audit_set_path TEXT NOT NULL,
    ea_set_path TEXT NOT NULL,
    support_json TEXT NOT NULL DEFAULT '[]',
    UNIQUE(dataset_scope, profile_name, symbol, plugin_name)
);

CREATE TABLE IF NOT EXISTS control_cycles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    profile_name TEXT NOT NULL,
    group_key TEXT NOT NULL,
    started_at INTEGER NOT NULL,
    finished_at INTEGER NOT NULL DEFAULT 0,
    status TEXT NOT NULL DEFAULT 'running',
    datasets_json TEXT NOT NULL DEFAULT '[]',
    notes TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS champion_registry (
    profile_name TEXT NOT NULL,
    symbol TEXT NOT NULL,
    plugin_name TEXT NOT NULL,
    family_id INTEGER NOT NULL DEFAULT 11,
    champion_best_config_id INTEGER,
    challenger_run_id INTEGER,
    status TEXT NOT NULL DEFAULT 'candidate',
    champion_score REAL NOT NULL DEFAULT 0.0,
    challenger_score REAL NOT NULL DEFAULT 0.0,
    portfolio_score REAL NOT NULL DEFAULT 0.0,
    promoted_at INTEGER NOT NULL DEFAULT 0,
    reviewed_at INTEGER NOT NULL DEFAULT 0,
    champion_set_path TEXT NOT NULL DEFAULT '',
    notes TEXT NOT NULL DEFAULT '',
    PRIMARY KEY(profile_name, symbol, plugin_name)
);

CREATE TABLE IF NOT EXISTS config_lineage (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    profile_name TEXT NOT NULL,
    symbol TEXT NOT NULL,
    plugin_name TEXT NOT NULL,
    family_id INTEGER NOT NULL DEFAULT 11,
    source_run_id INTEGER,
    best_config_id INTEGER,
    relation TEXT NOT NULL DEFAULT 'candidate',
    lineage_hash TEXT NOT NULL DEFAULT '',
    payload_json TEXT NOT NULL DEFAULT '{}',
    created_at INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS family_scorecards (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    profile_name TEXT NOT NULL,
    group_key TEXT NOT NULL DEFAULT '',
    symbol TEXT NOT NULL,
    family_id INTEGER NOT NULL,
    family_name TEXT NOT NULL,
    run_count INTEGER NOT NULL DEFAULT 0,
    mean_score REAL NOT NULL DEFAULT 0.0,
    mean_recent_score REAL NOT NULL DEFAULT 0.0,
    mean_walkforward_score REAL NOT NULL DEFAULT 0.0,
    mean_adversarial_score REAL NOT NULL DEFAULT 0.0,
    mean_macro_score REAL NOT NULL DEFAULT 0.0,
    mean_issue_count REAL NOT NULL DEFAULT 0.0,
    stability_score REAL NOT NULL DEFAULT 0.0,
    promotion_count INTEGER NOT NULL DEFAULT 0,
    champion_count INTEGER NOT NULL DEFAULT 0,
    payload_json TEXT NOT NULL DEFAULT '{}',
    created_at INTEGER NOT NULL,
    UNIQUE(profile_name, group_key, symbol, family_id)
);

CREATE TABLE IF NOT EXISTS distillation_artifacts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    profile_name TEXT NOT NULL,
    symbol TEXT NOT NULL,
    plugin_name TEXT NOT NULL,
    family_id INTEGER NOT NULL DEFAULT 11,
    source_run_id INTEGER,
    best_config_id INTEGER,
    dataset_scope TEXT NOT NULL DEFAULT 'aggregate',
    artifact_path TEXT NOT NULL,
    artifact_sha256 TEXT NOT NULL DEFAULT '',
    teacher_summary_json TEXT NOT NULL DEFAULT '{}',
    student_target_json TEXT NOT NULL DEFAULT '{}',
    status TEXT NOT NULL DEFAULT 'ready',
    created_at INTEGER NOT NULL,
    UNIQUE(profile_name, symbol, plugin_name, dataset_scope)
);

CREATE TABLE IF NOT EXISTS redteam_cycles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    profile_name TEXT NOT NULL,
    group_key TEXT NOT NULL DEFAULT '',
    symbol TEXT NOT NULL,
    plugin_name TEXT NOT NULL,
    family_id INTEGER NOT NULL DEFAULT 11,
    weak_scenarios_json TEXT NOT NULL DEFAULT '[]',
    plan_json TEXT NOT NULL DEFAULT '{}',
    report_path TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL DEFAULT 'ready',
    created_at INTEGER NOT NULL,
    UNIQUE(profile_name, group_key, symbol, plugin_name)
);

CREATE INDEX IF NOT EXISTS idx_datasets_group ON datasets(group_key, symbol, months);
CREATE INDEX IF NOT EXISTS idx_tuning_runs_lookup ON tuning_runs(profile_name, group_key, symbol, plugin_name, status);
CREATE INDEX IF NOT EXISTS idx_tuning_runs_dataset ON tuning_runs(dataset_id, profile_name, plugin_name);
CREATE INDEX IF NOT EXISTS idx_best_configs_lookup ON best_configs(profile_name, symbol, plugin_name);
CREATE INDEX IF NOT EXISTS idx_control_cycles_lookup ON control_cycles(profile_name, started_at);
CREATE INDEX IF NOT EXISTS idx_lineage_lookup ON config_lineage(profile_name, symbol, plugin_name, created_at);
CREATE INDEX IF NOT EXISTS idx_family_scorecards_lookup ON family_scorecards(profile_name, group_key, symbol, family_id);
CREATE INDEX IF NOT EXISTS idx_champion_lookup ON champion_registry(profile_name, symbol, plugin_name, status);
CREATE INDEX IF NOT EXISTS idx_distill_lookup ON distillation_artifacts(profile_name, symbol, plugin_name, dataset_scope);
CREATE INDEX IF NOT EXISTS idx_redteam_lookup ON redteam_cycles(profile_name, group_key, symbol, plugin_name);
"""


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


def ensure_sqlite_column(conn: sqlite3.Connection, table: str, column: str, spec: str) -> None:
    columns = {str(row["name"]).lower() for row in conn.execute(f"PRAGMA table_info({table})")}
    if column.lower() not in columns:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {spec}")


def connect_db(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    last_error: Exception | None = None
    for attempt in range(6):
        conn: sqlite3.Connection | None = None
        try:
            conn = sqlite3.connect(str(db_path), timeout=30.0)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA busy_timeout=30000")
            conn.executescript(SQL_SCHEMA)
            ensure_sqlite_column(conn, "tuning_runs", "group_key", "TEXT NOT NULL DEFAULT ''")
            ensure_sqlite_column(conn, "tuning_runs", "family_id", "INTEGER NOT NULL DEFAULT 11")
            ensure_sqlite_column(conn, "best_configs", "family_id", "INTEGER NOT NULL DEFAULT 11")
            conn.execute("DROP INDEX IF EXISTS idx_tuning_runs_lookup")
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_tuning_runs_lookup "
                "ON tuning_runs(profile_name, group_key, symbol, plugin_name, status)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_tuning_runs_family "
                "ON tuning_runs(profile_name, family_id, symbol, plugin_name, status)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_best_configs_family "
                "ON best_configs(profile_name, family_id, symbol)"
            )
            conn.execute(
                """
                UPDATE tuning_runs
                   SET group_key = COALESCE((
                       SELECT d.group_key
                         FROM datasets d
                        WHERE d.id = tuning_runs.dataset_id
                   ), '')
                 WHERE COALESCE(group_key, '') = ''
                """
            )
            conn.commit()
            return conn
        except sqlite3.OperationalError as exc:
            last_error = exc
            if conn is not None:
                conn.close()
            if "locked" not in str(exc).lower() or attempt >= 5:
                raise
            time.sleep(0.25 * float(attempt + 1))
        except Exception:
            if conn is not None:
                conn.close()
            raise
    if last_error is not None:
        raise last_error
    raise OfflineLabError(f"failed to open sqlite lab: {db_path}")


def plugin_family_name(family_id: int) -> str:
    mapping = {
        0: "linear",
        1: "tree",
        2: "recurrent",
        3: "convolutional",
        4: "transformer",
        5: "state_space",
        6: "distribution",
        7: "mixture",
        8: "memory",
        9: "world",
        10: "rule",
        11: "other",
    }
    return mapping.get(int(family_id), "other")


def mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    mean_v = sum(values) / float(len(values))
    if len(values) <= 1:
        return mean_v, 0.0
    var = sum((v - mean_v) * (v - mean_v) for v in values) / float(len(values))
    return mean_v, math.sqrt(max(var, 0.0))


def param_identity_hash(row: dict) -> str:
    profile = str(row.get("profile_name", ""))
    symbol = str(row.get("symbol", ""))
    plugin = str(row.get("plugin_name", ""))
    params_json = str(row.get("parameters_json", "{}"))
    return sha256_text(f"{profile}|{symbol}|{plugin}|{params_json}")


def family_distillation_profile(family_id: int) -> dict:
    fam = int(family_id)
    if fam in (2, 3, 4, 5):
        return {
            "temperature": 1.35,
            "teacher_weight": 0.70,
            "student_weight": 0.30,
            "self_supervised_weight": 0.28,
            "analog_weight": 0.18,
            "foundation_weight": 0.32,
        }
    if fam in (0, 1, 6):
        return {
            "temperature": 1.15,
            "teacher_weight": 0.62,
            "student_weight": 0.38,
            "self_supervised_weight": 0.16,
            "analog_weight": 0.12,
            "foundation_weight": 0.18,
        }
    if fam in (7, 8, 9):
        return {
            "temperature": 1.28,
            "teacher_weight": 0.66,
            "student_weight": 0.34,
            "self_supervised_weight": 0.22,
            "analog_weight": 0.24,
            "foundation_weight": 0.24,
        }
    return {
        "temperature": 1.10,
        "teacher_weight": 0.58,
        "student_weight": 0.42,
        "self_supervised_weight": 0.10,
        "analog_weight": 0.08,
        "foundation_weight": 0.14,
    }


def row_to_dict(row: sqlite3.Row | None) -> dict | None:
    return dict(row) if row is not None else None


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

