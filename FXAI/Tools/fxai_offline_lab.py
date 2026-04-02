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

OFFLINE_DIR = Path(__file__).resolve().parent / "OfflineLab"
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


def write_export_set(path: Path, output_key: str, start_unix: int, end_unix: int, max_bars: int, reset_output: bool) -> None:
    content = "\n".join([
        f"Export_OutputKey={output_key}||0||0||0||N",
        f"Export_WindowStartUnix={int(start_unix)}||0||0||2147483647||N",
        f"Export_WindowEndUnix={int(end_unix)}||0||0||2147483647||N",
        f"Export_MaxBars={int(max_bars)}||600000||1||2000000||N",
        f"Export_ResetOutput={'true' if reset_output else 'false'}||true||0||true||N",
        "TradeKiller=0||0||0||10000||N",
    ]) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def write_tester_ini(path: Path, expert_rel_path: str, preset_name: str, login: str, symbol: str, server: str = "", password: str = "") -> None:
    lines = [
        "[Common]",
        f"Login={login}" if login else "Login=",
        f"Server={server}" if server else "Server=",
        f"Password={password}" if password else "Password=",
        "KeepPrivate=1",
        "ProxyEnable=0",
        "CertInstall=0",
        "NewsEnable=0",
        "",
        "[Tester]",
        f"Expert={expert_rel_path}",
        f"ExpertParameters={preset_name}",
        f"Symbol={symbol}",
        "Period=M1",
        "Model=1",
        "ExecutionMode=0",
        "Optimization=0",
        "ForwardMode=0",
        "Visual=0",
        "Deposit=10000",
        "Currency=USD",
        "Leverage=100",
        "ReplaceReport=1",
        "ShutdownTerminal=1",
        "Report=fxai_offline_lab_auto",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_profile_tester_section(expert_rel_path: str, preset_name: str, symbol: str, login: str = "", server: str = "") -> dict[str, str]:
    return {
        "LastExpert": expert_rel_path,
        "LastIndicator": r"Indicators\Examples\Accelerator.ex5",
        "LastTicksMode": "1",
        "LastCriterion": "0",
        "LastForward": "0",
        "LastDelay": "100",
        "LastOptimization": "0",
        "Expert": expert_rel_path,
        "ExpertParameters": preset_name,
        "Login": login,
        "Server": server,
        "Symbol": symbol,
        "Period": "1",
        "DateRange": "0",
        "DateFrom": "1735689600",
        "DateTo": "1736035200",
        "Visualization": "0",
        "Execution": "100",
        "Currency": "USD",
        "CheckCurrencyDigits": "2",
        "Leverage": "100",
        "PipsCalculation": "0",
        "TicksMode": "1",
        "ProgramType": "0",
        "Deposit": "10000.00",
        "OptMode": "0",
        "OptForward": "0",
        "OptCrit": "0",
        "Report": "fxai_offline_lab_auto",
        "ReplaceReport": "1",
        "ShutdownTerminal": "1",
    }


def wait_for_paths(paths: list[Path], start_ts: float, timeout_sec: int) -> bool:
    deadline = time.time() + max(timeout_sec, 1)
    while time.time() < deadline:
        ready = True
        for path in paths:
            if not path.exists() or path.stat().st_mtime + 1e-9 < start_ts:
                ready = False
                break
        if ready:
            return True
        time.sleep(1.0)
    return False


def attempt_expert_launch(expert_rel_path: str,
                          preset_name: str,
                          symbol: str,
                          login: str,
                          server: str,
                          password: str,
                          timeout_sec: int,
                          success_paths: list[Path]) -> tuple[bool, str, str]:
    start_ts = time.time()
    config_path = Path(tempfile.gettempdir()) / f"{safe_token(preset_name)}.ini"
    write_tester_ini(config_path, expert_rel_path, preset_name, login, symbol, server, password)
    try:
        testlab.run_terminal_audit(config_path, timeout_sec)
    except testlab.AuditRunError as exc:
        if wait_for_paths(success_paths, start_ts, 10):
            return True, "config", ""
        return False, "config", str(exc)
    if wait_for_paths(success_paths, start_ts, 10):
        return True, "config", ""

    log_path = testlab.latest_terminal_log()
    log_text = testlab.read_utf16_or_text(log_path) if log_path else ""
    failure = testlab.extract_terminal_failure(log_text)
    if failure and "account is not specified" in failure.lower() and not password:
        return False, "config", failure
    if testlab.terminal_running():
        if not failure:
            failure = "profile fallback skipped because terminal64.exe is already running"
        return False, "config", failure

    common_backup = testlab.COMMON_INI.read_bytes()
    terminal_backup = testlab.TERMINAL_INI.read_bytes()
    try:
        if login or server:
            testlab.update_ini_section(
                testlab.COMMON_INI,
                "Common",
                {
                    "Login": login,
                    "Server": server,
                    "Password": password,
                    "KeepPrivate": "1",
                    "ProxyEnable": "0",
                    "CertInstall": "0",
                    "NewsEnable": "0",
                },
            )
        testlab.update_ini_section(
            testlab.TERMINAL_INI,
            "Tester",
            build_profile_tester_section(expert_rel_path, preset_name, symbol, login, server),
        )
        start_ts_profile = time.time()
        testlab.run_terminal_profile(timeout_sec)
        if wait_for_paths(success_paths, start_ts_profile, 10):
            return True, "profile", ""
        log_path = testlab.latest_terminal_log()
        log_text = testlab.read_utf16_or_text(log_path) if log_path else ""
        failure2 = testlab.extract_terminal_failure(log_text)
        if not failure2:
            failure2 = "profile launch exited without producing expected offline-lab artifacts"
        return False, "profile", failure2
    finally:
        testlab.COMMON_INI.write_bytes(common_backup)
        testlab.TERMINAL_INI.write_bytes(terminal_backup)


def compile_export_runner() -> int:
    return testlab.compile_target(Path("Tests/FXAI_OfflineExportRunner.mq5"), "offline_export")


def compile_audit_runner() -> int:
    return testlab.cmd_compile(argparse.Namespace())


def build_dataset_key(symbol: str, start_unix: int, end_unix: int, months: int) -> str:
    months_tag = (f"{months}m" if months > 0 else "window")
    return safe_token(f"{symbol}_m1_{months_tag}_{start_unix}_{end_unix}")


def load_dataset(conn: sqlite3.Connection, dataset_key: str) -> dict | None:
    row = conn.execute("SELECT * FROM datasets WHERE dataset_key = ?", (dataset_key,)).fetchone()
    return row_to_dict(row)


def load_dataset_by_id(conn: sqlite3.Connection, dataset_id: int) -> dict | None:
    row = conn.execute("SELECT * FROM datasets WHERE id = ?", (dataset_id,)).fetchone()
    return row_to_dict(row)


def insert_dataset_bars(conn: sqlite3.Connection, dataset_id: int, data_path: Path) -> int:
    conn.execute("DELETE FROM dataset_bars WHERE dataset_id = ?", (dataset_id,))
    batch: list[tuple] = []
    inserted = 0
    prev_time = -1
    with data_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            bar_time = int(float(row["time_unix"]))
            o = float(row["open"])
            h = float(row["high"])
            l = float(row["low"])
            c = float(row["close"])
            spread = int(float(row["spread_points"]))
            tick_volume = int(float(row["tick_volume"]))
            real_volume = int(float(row["real_volume"]))
            if bar_time <= prev_time:
                raise OfflineLabError(f"dataset bars not strictly ascending at {bar_time} in {data_path}")
            prev_time = bar_time
            if h + 1e-12 < max(o, c, l) or l - 1e-12 > min(o, c, h):
                raise OfflineLabError(f"invalid OHLC geometry at {bar_time} in {data_path}")
            batch.append((dataset_id, bar_time, o, h, l, c, spread, tick_volume, real_volume))
            if len(batch) >= 10000:
                conn.executemany(
                    "INSERT INTO dataset_bars(dataset_id, bar_time_unix, open, high, low, close, spread_points, tick_volume, real_volume) "
                    "VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    batch,
                )
                inserted += len(batch)
                batch = []
        if batch:
            conn.executemany(
                "INSERT INTO dataset_bars(dataset_id, bar_time_unix, open, high, low, close, spread_points, tick_volume, real_volume) "
                "VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)",
                batch,
            )
            inserted += len(batch)
    return inserted


def ingest_dataset(conn: sqlite3.Connection,
                   dataset_key: str,
                   group_key: str,
                   symbol: str,
                   months: int,
                   data_path: Path,
                   meta_path: Path,
                   notes: str = "") -> dict:
    meta = load_kv_tsv(meta_path)
    if not meta:
        raise OfflineLabError(f"offline export meta not found: {meta_path}")
    requested_start_unix = int(meta.get("window_start_unix", "0") or 0)
    requested_end_unix = int(meta.get("window_end_unix", "0") or 0)
    exported_start_unix = int(meta.get("first_time_unix", "0") or 0)
    exported_end_unix = int(meta.get("last_time_unix", "0") or 0)
    start_unix = (exported_start_unix if exported_start_unix > 0 else requested_start_unix)
    end_unix = (exported_end_unix if exported_end_unix > start_unix else requested_end_unix)
    if start_unix <= 0 or end_unix <= start_unix:
        raise OfflineLabError(f"offline export meta has invalid effective window: {meta_path}")
    source_sha = testlab.sha256_path(data_path)
    created_at = now_unix()
    conn.execute(
        """
        INSERT INTO datasets(dataset_key, group_key, symbol, timeframe, start_unix, end_unix, months, bars, source_path, source_sha256, created_at, notes)
        VALUES(?, ?, ?, 'M1', ?, ?, ?, 0, ?, ?, ?, ?)
        ON CONFLICT(dataset_key) DO UPDATE SET
            group_key=excluded.group_key,
            symbol=excluded.symbol,
            timeframe=excluded.timeframe,
            start_unix=excluded.start_unix,
            end_unix=excluded.end_unix,
            months=excluded.months,
            source_path=excluded.source_path,
            source_sha256=excluded.source_sha256,
            created_at=excluded.created_at,
            notes=excluded.notes
        """,
        (dataset_key, group_key, symbol, start_unix, end_unix, months, str(data_path), source_sha, created_at, notes),
    )
    dataset_id = int(conn.execute("SELECT id FROM datasets WHERE dataset_key = ?", (dataset_key,)).fetchone()[0])
    inserted = insert_dataset_bars(conn, dataset_id, data_path)
    conn.execute("UPDATE datasets SET bars = ? WHERE id = ?", (inserted, dataset_id))
    conn.commit()
    dataset = load_dataset_by_id(conn, dataset_id)
    if dataset is None:
        raise OfflineLabError(f"failed to reload dataset {dataset_key}")
    return dataset


def export_single_dataset(conn: sqlite3.Connection, args, symbol: str, months: int, group_key: str) -> dict:
    start_unix, end_unix = resolve_window(months, getattr(args, "start_unix", 0), getattr(args, "end_unix", 0))
    dataset_key = build_dataset_key(symbol, start_unix, end_unix, months)
    existing = load_dataset(conn, dataset_key)
    if existing and not getattr(args, "replace", False):
        return existing

    if not getattr(args, "skip_compile", False):
        rc = compile_export_runner()
        if rc != 0:
            raise OfflineLabError("failed to compile FXAI_OfflineExportRunner.mq5")

    output_key = dataset_key
    data_path = dataset_data_path(output_key, symbol)
    meta_path = dataset_meta_path(output_key, symbol)
    ensure_dir(testlab.TESTER_PRESET_DIR)
    ensure_dir(COMMON_EXPORT_DIR)
    preset_name = f"fxai_offline_export_{safe_token(symbol)}.set"
    preset_path = testlab.TESTER_PRESET_DIR / preset_name
    write_export_set(preset_path, output_key, start_unix, end_unix, getattr(args, "max_bars", 600000), True)

    login, server, password = testlab.resolve_credentials(args)
    if data_path.exists():
        data_path.unlink()
    if meta_path.exists():
        meta_path.unlink()

    success, mode, failure = attempt_expert_launch(
        EXPORT_EXPERT,
        preset_name,
        symbol,
        login,
        server,
        password,
        getattr(args, "timeout", 300),
        [data_path, meta_path],
    )
    if not success:
        raise OfflineLabError(f"{mode} launch failed for export {symbol}: {failure}")

    return ingest_dataset(conn, dataset_key, group_key, symbol, months, data_path, meta_path, getattr(args, "notes", ""))


def resolve_dataset_rows(conn: sqlite3.Connection, args, auto_export: bool, group_key: str) -> list[dict]:
    rows: list[dict] = []
    dataset_keys = parse_csv_tokens(getattr(args, "dataset_keys", ""))
    for key in dataset_keys:
        dataset = load_dataset(conn, key)
        if dataset is None:
            raise OfflineLabError(f"dataset not found: {key}")
        rows.append(dataset)
    if rows:
        return rows

    symbols = resolve_symbols(args)
    months_list = resolve_months_list(getattr(args, "months_list", ""))
    for symbol in symbols:
        for months in months_list:
            start_unix, end_unix = resolve_window(months, getattr(args, "start_unix", 0), getattr(args, "end_unix", 0))
            dataset_key = build_dataset_key(symbol, start_unix, end_unix, months)
            dataset = load_dataset(conn, dataset_key)
            if dataset is None and auto_export:
                dataset = export_single_dataset(conn, args, symbol, months, group_key)
            if dataset is None:
                raise OfflineLabError(f"dataset not found and auto-export disabled: {dataset_key}")
            rows.append(dataset)
    return rows


def serious_base_args(args, dataset: dict, output_path: Path) -> argparse.Namespace:
    bars = int(getattr(args, "bars", 0) or 0)
    if bars <= 0 or bars > int(dataset["bars"]):
        bars = int(dataset["bars"])
    return argparse.Namespace(
        all_plugins=False,
        plugin_id=28,
        plugin_list="{all}",
        scenario_list=getattr(args, "scenario_list", SERIOUS_SCENARIOS),
        bars=bars,
        horizon=getattr(args, "horizon", 5),
        m1sync_bars=getattr(args, "m1sync_bars", 3),
        normalization=getattr(args, "normalization", 0),
        sequence_bars=getattr(args, "sequence_bars", 0),
        schema_id=getattr(args, "schema_id", 0),
        feature_mask=getattr(args, "feature_mask", 0),
        commission_per_lot_side=getattr(args, "commission_per_lot_side", None),
        cost_buffer_points=getattr(args, "cost_buffer_points", None),
        slippage_points=getattr(args, "slippage_points", None),
        fill_penalty_points=getattr(args, "fill_penalty_points", None),
        wf_train_bars=getattr(args, "wf_train_bars", 256),
        wf_test_bars=getattr(args, "wf_test_bars", 64),
        wf_purge_bars=getattr(args, "wf_purge_bars", 32),
        wf_embargo_bars=getattr(args, "wf_embargo_bars", 24),
        wf_folds=getattr(args, "wf_folds", 6),
        seed=getattr(args, "seed", 42),
        symbol=str(dataset["symbol"]),
        symbol_list="{" + str(dataset["symbol"]) + "}",
        symbol_pack="",
        window_start_unix=int(dataset["start_unix"]),
        window_end_unix=int(dataset["end_unix"]),
        execution_profile=getattr(args, "execution_profile", "default"),
        login=getattr(args, "login", None),
        server=getattr(args, "server", None),
        password=getattr(args, "password", None),
        timeout=getattr(args, "timeout", 300),
        baseline=None,
        output=str(output_path),
        compare_output=None,
        skip_compile=True,
    )


def extend_campaign(campaign: dict, base_args) -> dict:
    horizon_base = max(int(getattr(base_args, "horizon", 5)), 1)
    m1sync_base = max(int(getattr(base_args, "m1sync_bars", 3)), 1)
    exec_base = str(getattr(base_args, "execution_profile", "default") or "default")
    for _name, info in campaign.get("plugins", {}).items():
        experiments = info.setdefault("experiments", [])
        horizon_candidates = []
        for candidate in [max(1, horizon_base - 2), horizon_base] + list(DEFAULT_HORIZON_CANDIDATES):
            if candidate not in horizon_candidates:
                horizon_candidates.append(candidate)
        m1sync_candidates = []
        for candidate in [m1sync_base] + list(DEFAULT_M1SYNC_CANDIDATES):
            if candidate not in m1sync_candidates:
                m1sync_candidates.append(candidate)
        execution_candidates = []
        for candidate in [exec_base] + list(DEFAULT_EXECUTION_PROFILES):
            if candidate not in execution_candidates:
                execution_candidates.append(candidate)
        market_focus = ["market_recent", "market_trend", "market_chop", "market_session_edges", "market_spread_shock", "market_walkforward", "market_macro_event", "market_adversarial"]
        experiments.append({"name": "horizon_sweep", "horizons": horizon_candidates[:6], "focus": market_focus})
        experiments.append({"name": "m1sync_sweep", "m1sync_bars": m1sync_candidates[:5], "focus": market_focus})
        experiments.append({"name": "execution_profile_sweep", "execution_profiles": execution_candidates[:5], "focus": market_focus})
    return campaign


def campaign_runs_extended(campaign: dict, limit_plugins: int = 0, limit_experiments: int = 0) -> list[dict]:
    runs: list[dict] = []
    plugin_items = sorted(
        campaign.get("plugins", {}).items(),
        key=lambda item: float(item[1].get("score", 0.0)),
        reverse=True,
    )
    if limit_plugins > 0:
        plugin_items = plugin_items[:limit_plugins]

    for name, info in plugin_items:
        exp_count = 0
        for exp in info.get("experiments", []):
            focus = exp.get("focus", [])
            if exp["name"] == "schema_ablation":
                for schema in exp.get("schemas", []):
                    runs.append({"plugin": name, "experiment": exp["name"], "scenario_list": focus, "schema_id": schema})
            elif exp["name"] == "normalization_sweep":
                for norm in exp.get("normalizations", []):
                    runs.append({"plugin": name, "experiment": exp["name"], "scenario_list": focus, "normalization": norm})
            elif exp["name"] == "sequence_sweep":
                for seq in exp.get("sequence_bars", []):
                    runs.append({"plugin": name, "experiment": exp["name"], "scenario_list": focus, "sequence_bars": seq})
            elif exp["name"] == "feature_mask_ablation":
                for mask in exp.get("feature_masks", []):
                    runs.append({"plugin": name, "experiment": exp["name"], "scenario_list": focus, "feature_mask": mask})
            elif exp["name"] == "execution_sweep":
                for slip in exp.get("slippage_points", []):
                    for fillp in exp.get("fill_penalty_points", []):
                        runs.append({
                            "plugin": name,
                            "experiment": exp["name"],
                            "scenario_list": focus,
                            "slippage_points": slip,
                            "fill_penalty_points": fillp,
                        })
            elif exp["name"] == "walkforward_gate":
                for train_bars, test_bars in exp.get("train_test_pairs", []):
                    runs.append({
                        "plugin": name,
                        "experiment": exp["name"],
                        "scenario_list": focus,
                        "wf_train_bars": train_bars,
                        "wf_test_bars": test_bars,
                    })
            elif exp["name"] == "horizon_sweep":
                for horizon in exp.get("horizons", []):
                    runs.append({"plugin": name, "experiment": exp["name"], "scenario_list": focus, "horizon": horizon})
            elif exp["name"] == "m1sync_sweep":
                for sync_bars in exp.get("m1sync_bars", []):
                    runs.append({"plugin": name, "experiment": exp["name"], "scenario_list": focus, "m1sync_bars": sync_bars})
            elif exp["name"] == "execution_profile_sweep":
                for profile in exp.get("execution_profiles", []):
                    runs.append({"plugin": name, "experiment": exp["name"], "scenario_list": focus, "execution_profile": profile})
            else:
                runs.append({"plugin": name, "experiment": exp["name"], "scenario_list": focus})
            exp_count += 1
            if limit_experiments > 0 and exp_count >= limit_experiments:
                break
    return runs


def historical_scenario_weaknesses(conn: sqlite3.Connection,
                                   profile_name: str,
                                   symbol: str,
                                   plugin_name: str,
                                   exclude_group_key: str = "") -> list[dict]:
    clauses = [
        "tr.status = 'ok'",
        "tr.profile_name = ?",
        "tr.symbol = ?",
        "tr.plugin_name = ?",
    ]
    params: list[object] = [profile_name, symbol, plugin_name]
    if exclude_group_key:
        clauses.append("COALESCE(tr.group_key, '') <> ?")
        params.append(exclude_group_key)
    sql = f"""
        SELECT rs.scenario,
               AVG(rs.score) AS mean_score,
               AVG(rs.calibration_error) AS mean_calibration_error,
               AVG(rs.path_quality_error) AS mean_path_quality_error,
               AVG(rs.wf_pbo) AS mean_wf_pbo,
               AVG(rs.wf_dsr) AS mean_wf_dsr,
               AVG(rs.issue_flags) AS mean_issue_flags,
               COUNT(*) AS obs_count
          FROM run_scenarios rs
          JOIN tuning_runs tr ON tr.id = rs.run_id
         WHERE {' AND '.join(clauses)}
         GROUP BY rs.scenario
         ORDER BY mean_score ASC, mean_calibration_error DESC, mean_path_quality_error DESC
    """
    rows = conn.execute(sql, params).fetchall()
    return [dict(row) for row in rows]


def build_redteam_runs_for_plugin(plugin_name: str,
                                  family_id: int,
                                  weakness_rows: list[dict],
                                  base_args) -> tuple[list[dict], dict]:
    weak = []
    for row in weakness_rows:
        if float(row.get("mean_score", 0.0)) < 74.0 or float(row.get("mean_issue_flags", 0.0)) > 0.35:
            weak.append(dict(row))
    weak = weak[:3]
    if not weak:
        return [], {"plugin": plugin_name, "family_id": int(family_id), "weak_scenarios": [], "runs": []}

    weak_names = [str(row["scenario"]) for row in weak]
    focus = []
    for name in weak_names:
        if name not in focus:
            focus.append(name)
    if "market_adversarial" not in focus:
        focus.append("market_adversarial")
    if "market_walkforward" not in focus:
        focus.append("market_walkforward")

    base_horizon = max(int(getattr(base_args, "horizon", 5)), 1)
    base_seq = max(int(getattr(base_args, "sequence_bars", 0)), 0)
    base_sync = max(int(getattr(base_args, "m1sync_bars", 3)), 1)
    family = int(family_id)
    seq_target = base_seq
    if family in (2, 3, 4, 5):
        seq_target = max(seq_target, 32)
    elif family in (7, 8, 9):
        seq_target = max(seq_target, 16)
    else:
        seq_target = max(seq_target, 8)

    runs: list[dict] = []
    rationale: list[str] = []
    if any(name in ("market_adversarial", "market_walkforward") for name in weak_names):
        runs.append({
            "plugin": plugin_name,
            "experiment": "redteam_stability",
            "scenario_list": focus,
            "schema_id": 6 if family != 10 else 4,
            "sequence_bars": seq_target,
            "normalization": 14 if family in (2, 3, 4, 5) else 9,
            "execution_profile": "stress",
            "wf_train_bars": max(int(getattr(base_args, "wf_train_bars", 256)), 384),
            "wf_test_bars": max(int(getattr(base_args, "wf_test_bars", 64)), 96),
        })
        rationale.append("stability and walk-forward weakness triggered deeper sequence and stress execution replay")

    if "market_macro_event" in weak_names:
        runs.append({
            "plugin": plugin_name,
            "experiment": "redteam_macro",
            "scenario_list": focus,
            "schema_id": 6 if family in (2, 3, 4, 5, 7, 8, 9) else 5,
            "normalization": 13 if family != 10 else 0,
            "horizon": max(base_horizon, 8),
            "feature_mask": 0x7F if family != 10 else 0x29,
        })
        rationale.append("macro weakness triggered macro-heavy schema and full feature coverage")

    if any(name in ("market_session_edges", "market_spread_shock") for name in weak_names):
        runs.append({
            "plugin": plugin_name,
            "experiment": "redteam_execution",
            "scenario_list": focus,
            "execution_profile": "stress",
            "slippage_points": 1.0,
            "fill_penalty_points": 0.50,
            "m1sync_bars": max(base_sync, 5),
        })
        rationale.append("session/spread weakness triggered harsher execution stress and stricter M1 sync")

    if not runs:
        runs.append({
            "plugin": plugin_name,
            "experiment": "redteam_general",
            "scenario_list": focus,
            "schema_id": 6 if family != 10 else 4,
            "sequence_bars": seq_target,
            "horizon": max(base_horizon, 5),
        })
        rationale.append("general weakness triggered a broad adversarial certification pass")

    deduped: list[dict] = []
    seen = set()
    for run in runs:
        sig = json_compact(run)
        if sig in seen:
            continue
        seen.add(sig)
        deduped.append(run)

    plan = {
        "plugin": plugin_name,
        "family_id": int(family_id),
        "family_name": plugin_family_name(family_id),
        "weak_scenarios": weak,
        "rationale": rationale,
        "runs": deduped,
    }
    return deduped, plan


def persist_redteam_plan(conn: sqlite3.Connection,
                         profile_name: str,
                         group_key: str,
                         symbol: str,
                         plugin_name: str,
                         family_id: int,
                         plan: dict,
                         report_path: Path) -> None:
    ensure_dir(report_path.parent)
    md_lines = [
        f"# FXAI Red-Team Plan: {plugin_name}",
        "",
        f"profile: {profile_name}",
        f"symbol: {symbol}",
        f"family: {plugin_family_name(family_id)} ({int(family_id)})",
        "",
        "Weak scenarios:",
    ]
    weak = plan.get("weak_scenarios", [])
    if weak:
        for row in weak:
            md_lines.append(
                f"- {row.get('scenario', 'unknown')} | score {float(row.get('mean_score', 0.0)):.2f} | "
                f"cal {float(row.get('mean_calibration_error', 0.0)):.3f} | "
                f"path {float(row.get('mean_path_quality_error', 0.0)):.3f} | "
                f"obs {int(row.get('obs_count', 0))}"
            )
    else:
        md_lines.append("- none")
    md_lines.append("")
    md_lines.append("Planned targeted runs:")
    for run in plan.get("runs", []):
        md_lines.append(f"- {run.get('experiment', 'unknown')}: {json_compact(run)}")
    report_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    conn.execute(
        """
        INSERT INTO redteam_cycles(profile_name, group_key, symbol, plugin_name, family_id, weak_scenarios_json, plan_json, report_path, status, created_at)
        VALUES(?, ?, ?, ?, ?, ?, ?, ?, 'ready', ?)
        ON CONFLICT(profile_name, group_key, symbol, plugin_name) DO UPDATE SET
            family_id=excluded.family_id,
            weak_scenarios_json=excluded.weak_scenarios_json,
            plan_json=excluded.plan_json,
            report_path=excluded.report_path,
            status=excluded.status,
            created_at=excluded.created_at
        """,
        (
            profile_name,
            group_key,
            symbol,
            plugin_name,
            int(family_id),
            json.dumps(plan.get("weak_scenarios", []), indent=2, sort_keys=True),
            json.dumps(plan, indent=2, sort_keys=True),
            str(report_path),
            now_unix(),
        ),
    )
    conn.commit()


def generate_redteam_runs(conn: sqlite3.Connection,
                          profile_name: str,
                          group_key: str,
                          dataset: dict,
                          baseline_summary: dict,
                          base_args,
                          out_dir: Path) -> list[dict]:
    generated: list[dict] = []
    for plugin_name, info in sorted(baseline_summary.get("plugins", {}).items()):
        family_id = int(info.get("family", 11))
        weakness_rows = historical_scenario_weaknesses(
            conn,
            profile_name,
            str(dataset["symbol"]),
            plugin_name,
            group_key,
        )
        runs, plan = build_redteam_runs_for_plugin(plugin_name, family_id, weakness_rows, base_args)
        if not runs:
            continue
        report_path = out_dir / "redteam" / f"{safe_token(plugin_name)}__redteam.md"
        persist_redteam_plan(conn, profile_name, group_key, str(dataset["symbol"]), plugin_name, family_id, plan, report_path)
        generated.extend(runs)
    return generated


def normalize_namespace_parameters(args, plugin_name: str, experiment_name: str, dataset: dict) -> dict:
    return {
        "plugin": plugin_name,
        "experiment": experiment_name,
        "scenario_list": parse_csv_tokens(getattr(args, "scenario_list", SERIOUS_SCENARIOS)),
        "bars": int(getattr(args, "bars", dataset["bars"])),
        "horizon": int(getattr(args, "horizon", 5)),
        "m1sync_bars": int(getattr(args, "m1sync_bars", 3)),
        "normalization": int(getattr(args, "normalization", 0)),
        "sequence_bars": int(getattr(args, "sequence_bars", 0)),
        "schema_id": int(getattr(args, "schema_id", 0)),
        "feature_mask": int(getattr(args, "feature_mask", 0)),
        "commission_per_lot_side": float(getattr(args, "commission_per_lot_side", 0.0)),
        "cost_buffer_points": float(getattr(args, "cost_buffer_points", 0.0)),
        "slippage_points": float(getattr(args, "slippage_points", 0.0)),
        "fill_penalty_points": float(getattr(args, "fill_penalty_points", 0.0)),
        "wf_train_bars": int(getattr(args, "wf_train_bars", 256)),
        "wf_test_bars": int(getattr(args, "wf_test_bars", 64)),
        "wf_purge_bars": int(getattr(args, "wf_purge_bars", 32)),
        "wf_embargo_bars": int(getattr(args, "wf_embargo_bars", 24)),
        "wf_folds": int(getattr(args, "wf_folds", 6)),
        "execution_profile": str(getattr(args, "execution_profile", "default")),
        "symbol": str(dataset["symbol"]),
        "window_start_unix": int(dataset["start_unix"]),
        "window_end_unix": int(dataset["end_unix"]),
    }


def grouped_rows_by_plugin(report_tsv: Path) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = defaultdict(list)
    if not report_tsv.exists():
        return out
    for row in testlab.load_rows(report_tsv):
        out[row.get("ai_name", "unknown")].append(row)
    return out


def upsert_tuning_run(conn: sqlite3.Connection,
                      dataset: dict,
                      profile_name: str,
                      group_key: str,
                      plugin_name: str,
                      ai_id: int,
                      family_id: int,
                      experiment_name: str,
                      parameters: dict,
                      report_path: Path,
                      raw_report_path: Path,
                      summary_path: Path,
                      manifest_path: Path,
                      status: str,
                      started_at: int,
                      finished_at: int,
                      summary_plugin: dict | None) -> int:
    issues = []
    score = 0.0
    grade = "F"
    market_recent_score = 0.0
    walkforward_score = 0.0
    adversarial_score = 0.0
    macro_event_score = 0.0
    if summary_plugin:
        score = float(summary_plugin.get("score", 0.0))
        grade = str(summary_plugin.get("grade", "F"))
        issues = list(summary_plugin.get("issues", [])) + list(summary_plugin.get("findings", []))
        scenarios = summary_plugin.get("scenarios", {})
        market_recent_score = float(scenarios.get("market_recent", {}).get("score", 0.0))
        walkforward_score = float(scenarios.get("market_walkforward", {}).get("score", 0.0))
        adversarial_score = float(scenarios.get("market_adversarial", {}).get("score", 0.0))
        macro_event_score = float(scenarios.get("market_macro_event", {}).get("score", 0.0))
    parameters_json = json_compact(parameters)
    param_hash = sha256_text(f"{dataset['id']}|{profile_name}|{plugin_name}|{parameters_json}")

    conn.execute(
        """
        INSERT INTO tuning_runs(
            dataset_id, profile_name, group_key, symbol, plugin_name, ai_id, family_id, experiment_name, param_hash, parameters_json,
            report_path, raw_report_path, summary_path, manifest_path,
            score, grade, issue_count, issues_json,
            market_recent_score, walkforward_score, adversarial_score, macro_event_score,
            status, started_at, finished_at
        )
        VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(param_hash) DO UPDATE SET
            group_key=excluded.group_key,
            family_id=excluded.family_id,
            report_path=excluded.report_path,
            raw_report_path=excluded.raw_report_path,
            summary_path=excluded.summary_path,
            manifest_path=excluded.manifest_path,
            score=excluded.score,
            grade=excluded.grade,
            issue_count=excluded.issue_count,
            issues_json=excluded.issues_json,
            market_recent_score=excluded.market_recent_score,
            walkforward_score=excluded.walkforward_score,
            adversarial_score=excluded.adversarial_score,
            macro_event_score=excluded.macro_event_score,
            status=excluded.status,
            started_at=excluded.started_at,
            finished_at=excluded.finished_at
        """,
        (
            int(dataset["id"]),
            profile_name,
            (group_key or ""),
            str(dataset["symbol"]),
            plugin_name,
            ai_id,
            int(family_id),
            experiment_name,
            param_hash,
            parameters_json,
            str(report_path),
            str(raw_report_path),
            str(summary_path),
            str(manifest_path),
            score,
            grade,
            len(issues),
            json.dumps(issues, sort_keys=True),
            market_recent_score,
            walkforward_score,
            adversarial_score,
            macro_event_score,
            status,
            started_at,
            finished_at,
        ),
    )
    run_id = int(conn.execute("SELECT id FROM tuning_runs WHERE param_hash = ?", (param_hash,)).fetchone()[0])
    conn.execute("DELETE FROM run_scenarios WHERE run_id = ?", (run_id,))
    if summary_plugin:
        for scenario, metrics in summary_plugin.get("scenarios", {}).items():
            conn.execute(
                """
                INSERT INTO run_scenarios(run_id, scenario, score, calibration_error, path_quality_error, wf_pbo, wf_dsr, wf_pass_rate, net_signal, issue_flags)
                VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    scenario,
                    float(metrics.get("score", 0.0)),
                    float(metrics.get("calibration_error", 0.0)),
                    float(metrics.get("path_quality_error", 0.0)),
                    float(metrics.get("wf_pbo", 0.0)),
                    float(metrics.get("wf_dsr", 0.0)),
                    float(metrics.get("wf_pass_rate", 0.0)),
                    float(metrics.get("trend_align", 0.0)),
                    int(metrics.get("issue_flags", 0)),
                ),
            )
    conn.commit()
    return run_id


def store_baseline_run_bundle(conn: sqlite3.Connection,
                              dataset: dict,
                              profile_name: str,
                              group_key: str,
                              base_args,
                              report_path: Path,
                              raw_report_path: Path,
                              summary_path: Path,
                              manifest_path: Path,
                              started_at: int,
                              finished_at: int) -> None:
    summary = testlab.load_json(summary_path)
    grouped_rows = grouped_rows_by_plugin(raw_report_path)
    for plugin_name, plugin_summary in sorted(summary.get("plugins", {}).items()):
        rows = grouped_rows.get(plugin_name, [])
        ai_id = int(float(rows[0]["ai_id"])) if rows else -1
        family_id = int(plugin_summary.get("family", rows[0].get("family", 11) if rows else 11))
        params = normalize_namespace_parameters(base_args, plugin_name, "baseline_all", dataset)
        upsert_tuning_run(
            conn,
            dataset,
            profile_name,
            group_key,
            plugin_name,
            ai_id,
            family_id,
            "baseline_all",
            params,
            report_path,
            raw_report_path,
            summary_path,
            manifest_path,
            "ok",
            started_at,
            finished_at,
            plugin_summary,
        )


def run_dataset_baseline(conn: sqlite3.Connection, dataset: dict, profile_name: str, args, out_dir: Path) -> dict:
    baseline_path = out_dir / "baseline_all.md"
    base_args = testlab.build_effective_audit_args(serious_base_args(args, dataset, baseline_path))
    started_at = now_unix()
    rc = testlab.cmd_run_audit(base_args)
    finished_at = now_unix()
    if rc != 0:
        raise OfflineLabError(f"baseline audit failed for {dataset['dataset_key']}")
    raw_report_path = baseline_path.with_suffix(".tsv")
    shutil.copy2(testlab.DEFAULT_REPORT, raw_report_path)
    summary_path = baseline_path.with_suffix(".summary.json")
    manifest_path = baseline_path.with_suffix(".manifest.json")
    group_key = str(getattr(args, "group_key", "") or dataset.get("group_key", "") or "")
    store_baseline_run_bundle(conn, dataset, profile_name, group_key, base_args, baseline_path, raw_report_path, summary_path, manifest_path, started_at, finished_at)
    return {
        "report_path": baseline_path,
        "raw_report_path": raw_report_path,
        "summary_path": summary_path,
        "manifest_path": manifest_path,
        "summary": testlab.load_json(summary_path),
        "base_args": base_args,
    }


def run_dataset_campaign(conn: sqlite3.Connection, dataset: dict, profile_name: str, args, out_dir: Path, baseline_summary: dict, base_args) -> list[dict]:
    oracles = testlab.load_oracles()
    campaign = extend_campaign(testlab.build_optimization_campaign(baseline_summary, oracles), base_args)
    (out_dir / "campaign.json").write_text(json.dumps(campaign, indent=2, sort_keys=True), encoding="utf-8")
    runs = campaign_runs_extended(campaign, getattr(args, "top_plugins", 0), getattr(args, "limit_experiments", 0))
    group_key = str(getattr(args, "group_key", "") or dataset.get("group_key", "") or "")
    redteam_runs = generate_redteam_runs(conn, profile_name, group_key, dataset, baseline_summary, base_args, out_dir)
    if redteam_runs:
        runs.extend(redteam_runs)
    if getattr(args, "limit_runs", 0) > 0:
        runs = runs[: args.limit_runs]
    results = []
    for idx, run in enumerate(runs, start=1):
        run["bars"] = int(base_args.bars)
        run["window_start_unix"] = int(dataset["start_unix"])
        run["window_end_unix"] = int(dataset["end_unix"])
        if "execution_profile" not in run:
            run["execution_profile"] = str(base_args.execution_profile)
        stem = f"{idx:03d}_{run['plugin']}_{safe_token(run['experiment'])}"
        report_path = out_dir / "runs" / f"{stem}.md"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        started_at = now_unix()
        run_args = testlab.build_effective_audit_args(testlab.build_run_audit_namespace(base_args, run, report_path))
        parameters = normalize_namespace_parameters(run_args, run["plugin"], run["experiment"], dataset)
        rc = testlab.cmd_run_audit(run_args)
        finished_at = now_unix()
        raw_report_path = report_path.with_suffix(".tsv")
        summary_path = report_path.with_suffix(".summary.json")
        manifest_path = report_path.with_suffix(".manifest.json")
        plugin_summary = None
        ai_id = -1
        family_id = 11
        status = "failed"
        if rc == 0:
            shutil.copy2(testlab.DEFAULT_REPORT, raw_report_path)
            summary = testlab.load_json(summary_path)
            plugin_summary = summary.get("plugins", {}).get(run["plugin"], {})
            rows = grouped_rows_by_plugin(raw_report_path).get(run["plugin"], [])
            if rows:
                ai_id = int(float(rows[0]["ai_id"]))
                family_id = int(float(rows[0].get("family", plugin_summary.get("family", 11))))
            else:
                family_id = int(plugin_summary.get("family", 11))
            status = "ok"
        run_id = upsert_tuning_run(
            conn,
            dataset,
            profile_name,
            group_key,
            run["plugin"],
            ai_id,
            family_id,
            run["experiment"],
            parameters,
            report_path,
            raw_report_path,
            summary_path,
            manifest_path,
            status,
            started_at,
            finished_at,
            plugin_summary,
        )
        result = {
            "run_id": run_id,
            "status": status,
            "plugin": run["plugin"],
            "experiment": run["experiment"],
            "parameters": parameters,
            "report_path": str(report_path),
            "summary_path": str(summary_path),
        }
        results.append(result)
        (out_dir / "results.json").write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
    return results


def cmd_init_db(args) -> int:
    conn = connect_db(Path(args.db))
    conn.close()
    print(f"initialized sqlite lab: {args.db}")
    return 0


def cmd_compile_export(_args) -> int:
    return compile_export_runner()


def cmd_export_dataset(args) -> int:
    conn = connect_db(Path(args.db))
    group_key = getattr(args, "group_key", "") or safe_token(f"manual_export_{now_unix()}")
    symbols = resolve_symbols(args)
    months_list = resolve_months_list(args.months_list)
    if not getattr(args, "skip_compile", False):
        if compile_export_runner() != 0:
            raise OfflineLabError("failed to compile FXAI_OfflineExportRunner.mq5")
        args = argparse.Namespace(**vars(args))
        args.skip_compile = True
    datasets = []
    for symbol in symbols:
        for months in months_list:
            datasets.append(export_single_dataset(conn, args, symbol, months, group_key))
    print(json.dumps({"group_key": group_key, "datasets": datasets}, indent=2, sort_keys=True))
    conn.close()
    return 0


def cmd_tune_zoo(args) -> int:
    conn = connect_db(Path(args.db))
    group_key = getattr(args, "group_key", "") or safe_token(f"{args.profile}_{now_unix()}")
    resolve_args = args
    if getattr(args, "auto_export", False) and not getattr(args, "skip_compile", False):
        if compile_export_runner() != 0:
            raise OfflineLabError("failed to compile FXAI_OfflineExportRunner.mq5")
        resolve_args = argparse.Namespace(**vars(args))
        resolve_args.skip_compile = True
    datasets = resolve_dataset_rows(conn, resolve_args, getattr(args, "auto_export", False), group_key)
    if not datasets:
        raise OfflineLabError("no datasets resolved for tune-zoo")

    if not getattr(args, "skip_compile", False):
        rc = compile_audit_runner()
        if rc != 0:
            raise OfflineLabError("failed to compile FXAI_AuditRunner.mq5")

    all_results = []
    for dataset in datasets:
        dataset_out_dir = RUNS_DIR / safe_token(args.profile) / safe_token(dataset["dataset_key"])
        ensure_dir(dataset_out_dir)
        baseline = run_dataset_baseline(conn, dataset, args.profile, args, dataset_out_dir)
        results = run_dataset_campaign(conn, dataset, args.profile, args, dataset_out_dir, baseline["summary"], baseline["base_args"])
        all_results.append({
            "dataset_key": dataset["dataset_key"],
            "symbol": dataset["symbol"],
            "baseline_report": str(baseline["report_path"]),
            "run_count": len(results),
        })
    print(json.dumps({"profile": args.profile, "datasets": all_results}, indent=2, sort_keys=True))
    conn.close()
    return 0


def execution_profile_enum(name: str) -> int:
    profile = (name or "default").strip().lower()
    mapping = {
        "default": 0,
        "tight-fx": 1,
        "prime-ecn": 2,
        "retail-fx": 3,
        "stress": 4,
    }
    return int(mapping.get(profile, 0))


def write_ea_set(path: Path, row: dict, params: dict) -> None:
    horizon = int(params.get("horizon", 5))
    content = "\n".join([
        f"AI_Type={int(row['ai_id'])}||0||0||31||N",
        "AI_Ensemble=false||false||0||true||N",
        f"AI_M1SyncBars={int(params.get('m1sync_bars', 3))}||3||1||12||N",
        f"PredictionTargetMinutes={horizon}||5||1||720||N",
        "AI_MultiHorizon=false||true||0||true||N",
        f"AI_Horizons={{{horizon}}}||0||0||0||N",
        f"AI_FeatureNormalization={int(params.get('normalization', 0))}||0||0||14||N",
        f"AI_ExecutionProfile={execution_profile_enum(str(params.get('execution_profile', 'default')))}||0||0||4||N",
        f"AI_CommissionPerLotSide={float(params.get('commission_per_lot_side', 0.0)):.6f}||0||0||100||N",
        f"AI_CostBufferPoints={float(params.get('cost_buffer_points', 2.0)):.6f}||2||0||100||N",
        f"AI_ExecutionSlippageOverride={float(params.get('slippage_points', 0.0)):.6f}||-1||-1||100||N",
        f"AI_ExecutionFillPenaltyOverride={float(params.get('fill_penalty_points', 0.0)):.6f}||-1||-1||100||N",
    ]) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def write_audit_set_generic(path: Path, row: dict, params: dict) -> None:
    ns = argparse.Namespace(
        all_plugins=False,
        plugin_id=int(row["ai_id"]),
        plugin_list="{" + str(row["plugin_name"]) + "}",
        scenario_list=SERIOUS_SCENARIOS,
        bars=int(params.get("bars", 20000)),
        horizon=int(params.get("horizon", 5)),
        m1sync_bars=int(params.get("m1sync_bars", 3)),
        normalization=int(params.get("normalization", 0)),
        sequence_bars=int(params.get("sequence_bars", 0)),
        schema_id=int(params.get("schema_id", 0)),
        feature_mask=int(params.get("feature_mask", 0)),
        commission_per_lot_side=float(params.get("commission_per_lot_side", 0.0)),
        cost_buffer_points=float(params.get("cost_buffer_points", 2.0)),
        slippage_points=float(params.get("slippage_points", 0.0)),
        fill_penalty_points=float(params.get("fill_penalty_points", 0.0)),
        wf_train_bars=int(params.get("wf_train_bars", 256)),
        wf_test_bars=int(params.get("wf_test_bars", 64)),
        wf_purge_bars=int(params.get("wf_purge_bars", 32)),
        wf_embargo_bars=int(params.get("wf_embargo_bars", 24)),
        wf_folds=int(params.get("wf_folds", 6)),
        window_start_unix=0,
        window_end_unix=0,
        seed=42,
        output="",
        compare_output=None,
        symbol=str(row["symbol"]),
        symbol_list="{" + str(row["symbol"]) + "}",
        symbol_pack="",
        execution_profile=str(params.get("execution_profile", "default")),
        login=None,
        server=None,
        password=None,
        timeout=180,
        baseline=None,
        skip_compile=True,
    )
    testlab.write_audit_set(path, ns)


def load_completed_runs(conn: sqlite3.Connection, args) -> list[dict]:
    clauses = ["tr.status = 'ok'", "tr.profile_name = ?"]
    params: list[object] = [args.profile]
    dataset_keys = parse_csv_tokens(getattr(args, "dataset_keys", ""))
    group_key = (getattr(args, "group_key", "") or "").strip()
    symbols = resolve_symbols(args)
    if dataset_keys:
        clauses.append("d.dataset_key IN (%s)" % ",".join("?" for _ in dataset_keys))
        params.extend(dataset_keys)
    elif group_key:
        clauses.append("tr.group_key = ?")
        params.append(group_key)
    if symbols:
        clauses.append("tr.symbol IN (%s)" % ",".join("?" for _ in symbols))
        params.extend(symbols)
    sql = f"""
        SELECT tr.*, d.dataset_key, d.group_key, d.months, d.start_unix, d.end_unix
        FROM tuning_runs tr
        JOIN datasets d ON d.id = tr.dataset_id
        WHERE {' AND '.join(clauses)}
        ORDER BY tr.symbol, tr.plugin_name, tr.score DESC, tr.finished_at DESC
    """
    rows = conn.execute(sql, params).fetchall()
    return [dict(row) for row in rows]


def aggregate_best_candidates(rows: list[dict]) -> tuple[list[dict], dict[str, int]]:
    grouped: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    dataset_counts: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        grouped[(row["symbol"], row["plugin_name"], row["parameters_json"])].append(row)
        dataset_counts[row["symbol"]].add(row["dataset_key"])

    winners: list[dict] = []
    total_datasets_per_symbol = {symbol: len(keys) for symbol, keys in dataset_counts.items()}
    per_symbol_plugin: dict[tuple[str, str], list[dict]] = defaultdict(list)

    for (symbol, plugin_name, _param_hash), group in grouped.items():
        weights = [math.sqrt(max(int(item.get("months", 1)) or 1, 1)) for item in group]
        weight_sum = sum(weights) or 1.0
        mean_score = sum(float(item["score"]) * w for item, w in zip(group, weights)) / weight_sum
        min_score = min(float(item["score"]) for item in group)
        mean_recent = sum(float(item["market_recent_score"]) * w for item, w in zip(group, weights)) / weight_sum
        mean_wf = sum(float(item["walkforward_score"]) * w for item, w in zip(group, weights)) / weight_sum
        mean_adv = sum(float(item["adversarial_score"]) * w for item, w in zip(group, weights)) / weight_sum
        mean_macro = sum(float(item["macro_event_score"]) * w for item, w in zip(group, weights)) / weight_sum
        mean_issues = sum(float(item["issue_count"]) * w for item, w in zip(group, weights)) / weight_sum
        support = len({item["dataset_key"] for item in group})
        coverage = float(support) / float(max(total_datasets_per_symbol.get(symbol, 1), 1))
        ranking = (
            0.48 * mean_score +
            0.18 * min_score +
            0.12 * mean_wf +
            0.10 * mean_adv +
            0.05 * mean_recent +
            0.03 * mean_macro +
            4.0 * coverage -
            0.75 * mean_issues
        )
        best_row = max(group, key=lambda item: (float(item["score"]), -float(item["issue_count"]), float(item["adversarial_score"]), float(item["walkforward_score"])))
        aggregated = {
            "symbol": symbol,
            "plugin_name": plugin_name,
            "ai_id": int(best_row["ai_id"]),
            "family_id": int(best_row.get("family_id", 11)),
            "run_id": int(best_row["id"]),
            "score": mean_score,
            "ranking_score": ranking,
            "support_count": support,
            "support_json": json.dumps([
                {
                    "dataset_key": item["dataset_key"],
                    "months": int(item.get("months", 0)),
                    "score": float(item["score"]),
                    "walkforward_score": float(item["walkforward_score"]),
                    "adversarial_score": float(item["adversarial_score"]),
                }
                for item in sorted(group, key=lambda x: (int(x.get("months", 0)), x["dataset_key"]))
            ], indent=2, sort_keys=True),
            "parameters_json": best_row["parameters_json"],
            "dataset_scope": "aggregate",
            "dataset_id": None,
        }
        per_symbol_plugin[(symbol, plugin_name)].append(aggregated)

    for (symbol, plugin_name), candidates in per_symbol_plugin.items():
        winner = max(
            candidates,
            key=lambda item: (
                float(item["ranking_score"]),
                float(item["score"]),
                int(item["support_count"]),
            ),
        )
        winners.append(winner)
    return winners, total_datasets_per_symbol


def render_family_scorecards(conn: sqlite3.Connection,
                             profile_name: str,
                             group_key: str,
                             rows: list[dict],
                             promoted_rows: list[dict]) -> list[dict]:
    promoted_by_family: dict[tuple[str, int], int] = defaultdict(int)
    for row in promoted_rows:
        promoted_by_family[(str(row["symbol"]), int(row.get("family_id", 11)))] += 1

    champion_rows = conn.execute(
        "SELECT symbol, family_id, COUNT(*) AS champion_count "
        "FROM champion_registry WHERE profile_name = ? AND status = 'champion' "
        "GROUP BY symbol, family_id",
        (profile_name,),
    ).fetchall()
    champion_by_family = {
        (str(row["symbol"]), int(row["family_id"])): int(row["champion_count"])
        for row in champion_rows
    }

    grouped: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["symbol"]), int(row.get("family_id", 11)))].append(row)

    scorecards: list[dict] = []
    for (symbol, family_id), items in sorted(grouped.items()):
        scores = [float(item["score"]) for item in items]
        mean_score, score_std = mean_std(scores)
        stability = 1.0
        if mean_score > 1e-9:
            stability = max(0.0, 1.0 - min(score_std / max(mean_score, 10.0), 1.0))
        mean_recent = sum(float(item["market_recent_score"]) for item in items) / float(len(items))
        mean_wf = sum(float(item["walkforward_score"]) for item in items) / float(len(items))
        mean_adv = sum(float(item["adversarial_score"]) for item in items) / float(len(items))
        mean_macro = sum(float(item["macro_event_score"]) for item in items) / float(len(items))
        mean_issues = sum(float(item["issue_count"]) for item in items) / float(len(items))
        top_plugins = []
        ranked = sorted(
            items,
            key=lambda item: (float(item["score"]), float(item["walkforward_score"]), -float(item["issue_count"])),
            reverse=True,
        )
        seen_plugins = set()
        for item in ranked:
            name = str(item["plugin_name"])
            if name in seen_plugins:
                continue
            seen_plugins.add(name)
            top_plugins.append({
                "plugin_name": name,
                "score": float(item["score"]),
                "walkforward_score": float(item["walkforward_score"]),
                "adversarial_score": float(item["adversarial_score"]),
                "macro_event_score": float(item["macro_event_score"]),
            })
            if len(top_plugins) >= 5:
                break
        payload = {
            "symbol": symbol,
            "family_id": family_id,
            "family_name": plugin_family_name(family_id),
            "run_count": len(items),
            "score_std": score_std,
            "top_plugins": top_plugins,
        }
        scorecards.append({
            "profile_name": profile_name,
            "group_key": group_key,
            "symbol": symbol,
            "family_id": family_id,
            "family_name": plugin_family_name(family_id),
            "run_count": len(items),
            "mean_score": mean_score,
            "mean_recent_score": mean_recent,
            "mean_walkforward_score": mean_wf,
            "mean_adversarial_score": mean_adv,
            "mean_macro_score": mean_macro,
            "mean_issue_count": mean_issues,
            "stability_score": stability,
            "promotion_count": promoted_by_family.get((symbol, family_id), 0),
            "champion_count": champion_by_family.get((symbol, family_id), 0),
            "payload_json": json.dumps(payload, indent=2, sort_keys=True),
        })
    return scorecards


def persist_family_scorecards(conn: sqlite3.Connection,
                              args,
                              rows: list[dict],
                              promoted_rows: list[dict]) -> list[dict]:
    group_key = (getattr(args, "group_key", "") or "").strip()
    scorecards = render_family_scorecards(conn, args.profile, group_key, rows, promoted_rows)
    out_dir = RESEARCH_DIR / safe_token(args.profile)
    ensure_dir(out_dir)
    now_ts = now_unix()
    for row in scorecards:
        conn.execute(
            """
            INSERT INTO family_scorecards(profile_name, group_key, symbol, family_id, family_name,
                                         run_count, mean_score, mean_recent_score, mean_walkforward_score,
                                         mean_adversarial_score, mean_macro_score, mean_issue_count,
                                         stability_score, promotion_count, champion_count, payload_json, created_at)
            VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(profile_name, group_key, symbol, family_id) DO UPDATE SET
                family_name=excluded.family_name,
                run_count=excluded.run_count,
                mean_score=excluded.mean_score,
                mean_recent_score=excluded.mean_recent_score,
                mean_walkforward_score=excluded.mean_walkforward_score,
                mean_adversarial_score=excluded.mean_adversarial_score,
                mean_macro_score=excluded.mean_macro_score,
                mean_issue_count=excluded.mean_issue_count,
                stability_score=excluded.stability_score,
                promotion_count=excluded.promotion_count,
                champion_count=excluded.champion_count,
                payload_json=excluded.payload_json,
                created_at=excluded.created_at
            """,
            (
                row["profile_name"],
                row["group_key"],
                row["symbol"],
                int(row["family_id"]),
                row["family_name"],
                int(row["run_count"]),
                float(row["mean_score"]),
                float(row["mean_recent_score"]),
                float(row["mean_walkforward_score"]),
                float(row["mean_adversarial_score"]),
                float(row["mean_macro_score"]),
                float(row["mean_issue_count"]),
                float(row["stability_score"]),
                int(row["promotion_count"]),
                int(row["champion_count"]),
                row["payload_json"],
                now_ts,
            ),
        )
    conn.commit()

    score_json = out_dir / "family_scorecards.json"
    score_tsv = out_dir / "family_scorecards.tsv"
    score_md = out_dir / "family_scorecards.md"
    score_json.write_text(json.dumps(scorecards, indent=2, sort_keys=True), encoding="utf-8")
    with score_tsv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow([
            "symbol", "family_id", "family_name", "run_count", "mean_score", "mean_walkforward_score",
            "mean_adversarial_score", "mean_macro_score", "stability_score", "promotion_count", "champion_count",
        ])
        for row in scorecards:
            writer.writerow([
                row["symbol"],
                row["family_id"],
                row["family_name"],
                row["run_count"],
                f"{float(row['mean_score']):.4f}",
                f"{float(row['mean_walkforward_score']):.4f}",
                f"{float(row['mean_adversarial_score']):.4f}",
                f"{float(row['mean_macro_score']):.4f}",
                f"{float(row['stability_score']):.4f}",
                row["promotion_count"],
                row["champion_count"],
            ])
    md_lines = ["# FXAI Family Scorecards", "", f"profile: {args.profile}", ""]
    for row in scorecards:
        md_lines.append(
            f"- {row['symbol']} | {row['family_name']} | score {float(row['mean_score']):.2f} | "
            f"wf {float(row['mean_walkforward_score']):.2f} | adv {float(row['mean_adversarial_score']):.2f} | "
            f"macro {float(row['mean_macro_score']):.2f} | stability {float(row['stability_score']):.2f} | "
            f"promoted {int(row['promotion_count'])} | champions {int(row['champion_count'])}"
        )
    score_md.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    return scorecards


def persist_lineage_entry(conn: sqlite3.Connection,
                          profile_name: str,
                          symbol: str,
                          plugin_name: str,
                          family_id: int,
                          source_run_id: int,
                          best_config_id: int,
                          relation: str,
                          payload: dict) -> None:
    lineage_hash = sha256_text(
        f"{profile_name}|{symbol}|{plugin_name}|{family_id}|{source_run_id}|{best_config_id}|{relation}|{json_compact(payload)}"
    )
    conn.execute(
        """
        INSERT INTO config_lineage(profile_name, symbol, plugin_name, family_id, source_run_id, best_config_id, relation, lineage_hash, payload_json, created_at)
        VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            profile_name,
            symbol,
            plugin_name,
            int(family_id),
            int(source_run_id) if source_run_id else None,
            int(best_config_id) if best_config_id else None,
            relation,
            lineage_hash,
            json.dumps(payload, indent=2, sort_keys=True),
            now_unix(),
        ),
    )


def update_champion_registry(conn: sqlite3.Connection,
                             args,
                             promoted_rows: list[dict]) -> list[dict]:
    profile_dir = PROFILES_DIR / safe_token(args.profile)
    ensure_dir(profile_dir)
    decisions: list[dict] = []
    for row in promoted_rows:
        symbol = str(row["symbol"])
        plugin_name = str(row["plugin_name"])
        family_id = int(row.get("family_id", 11))
        candidate_rank = float(row["ranking_score"])
        candidate_score = float(row["score"])
        registry = row_to_dict(conn.execute(
            "SELECT * FROM champion_registry WHERE profile_name = ? AND symbol = ? AND plugin_name = ?",
            (args.profile, symbol, plugin_name),
        ).fetchone())
        best_cfg = row_to_dict(conn.execute(
            "SELECT id FROM best_configs WHERE profile_name = ? AND dataset_scope = 'aggregate' AND symbol = ? AND plugin_name = ?",
            (args.profile, symbol, plugin_name),
        ).fetchone())
        best_config_id = int(best_cfg["id"]) if best_cfg else 0
        champion_dir = profile_dir / safe_token(symbol)
        ensure_dir(champion_dir)
        champion_audit = champion_dir / f"{plugin_name}__champion__audit.set"
        champion_ea = champion_dir / f"{plugin_name}__champion__ea.set"

        promote = False
        note = ""
        if registry is None or int(registry.get("champion_best_config_id") or 0) <= 0:
            promote = True
            note = "bootstrap_champion"
        else:
            current_rank = float(registry.get("champion_score", 0.0))
            current_portfolio = float(registry.get("portfolio_score", 0.0))
            challenger_portfolio = 0.65 * candidate_rank + 0.35 * float(row.get("support_count", 0))
            if candidate_rank > current_rank + 1.10 or challenger_portfolio > current_portfolio + 0.90:
                promote = True
                note = "challenger_promoted"
            else:
                note = "challenger_held_out"

        if promote:
            shutil.copy2(row["audit_set_path"], champion_audit)
            shutil.copy2(row["ea_set_path"], champion_ea)
            conn.execute(
                """
                INSERT INTO champion_registry(profile_name, symbol, plugin_name, family_id, champion_best_config_id, challenger_run_id,
                                              status, champion_score, challenger_score, portfolio_score, promoted_at, reviewed_at,
                                              champion_set_path, notes)
                VALUES(?, ?, ?, ?, ?, ?, 'champion', ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(profile_name, symbol, plugin_name) DO UPDATE SET
                    family_id=excluded.family_id,
                    champion_best_config_id=excluded.champion_best_config_id,
                    challenger_run_id=excluded.challenger_run_id,
                    status=excluded.status,
                    champion_score=excluded.champion_score,
                    challenger_score=excluded.challenger_score,
                    portfolio_score=excluded.portfolio_score,
                    promoted_at=excluded.promoted_at,
                    reviewed_at=excluded.reviewed_at,
                    champion_set_path=excluded.champion_set_path,
                    notes=excluded.notes
                """,
                (
                    args.profile,
                    symbol,
                    plugin_name,
                    family_id,
                    best_config_id,
                    int(row["run_id"]),
                    candidate_rank,
                    candidate_score,
                    0.65 * candidate_rank + 0.35 * float(row.get("support_count", 0)),
                    now_unix(),
                    now_unix(),
                    str(champion_ea),
                    note,
                ),
            )
            persist_lineage_entry(
                conn,
                args.profile,
                symbol,
                plugin_name,
                family_id,
                int(row["run_id"]),
                best_config_id,
                "champion",
                {"note": note, "ranking_score": candidate_rank, "score": candidate_score},
            )
        else:
            conn.execute(
                """
                INSERT INTO champion_registry(profile_name, symbol, plugin_name, family_id, champion_best_config_id, challenger_run_id,
                                              status, champion_score, challenger_score, portfolio_score, promoted_at, reviewed_at,
                                              champion_set_path, notes)
                VALUES(?, ?, ?, ?, ?, ?, 'champion', ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(profile_name, symbol, plugin_name) DO UPDATE SET
                    family_id=excluded.family_id,
                    challenger_run_id=excluded.challenger_run_id,
                    challenger_score=excluded.challenger_score,
                    reviewed_at=excluded.reviewed_at,
                    notes=excluded.notes
                """,
                (
                    args.profile,
                    symbol,
                    plugin_name,
                    family_id,
                    int(registry.get("champion_best_config_id", 0)),
                    int(row["run_id"]),
                    float(registry.get("champion_score", 0.0)),
                    candidate_score,
                    float(registry.get("portfolio_score", 0.0)),
                    int(registry.get("promoted_at", 0)),
                    now_unix(),
                    str(registry.get("champion_set_path", "")),
                    note,
                ),
            )
            persist_lineage_entry(
                conn,
                args.profile,
                symbol,
                plugin_name,
                family_id,
                int(row["run_id"]),
                best_config_id,
                "challenger",
                {"note": note, "ranking_score": candidate_rank, "score": candidate_score},
            )

        decisions.append({
            "symbol": symbol,
            "plugin_name": plugin_name,
            "family_id": family_id,
            "status": ("champion" if promote else "challenger"),
            "note": note,
            "ranking_score": candidate_rank,
            "score": candidate_score,
        })

    conn.commit()

    champion_rows = conn.execute(
        "SELECT * FROM champion_registry WHERE profile_name = ? AND status = 'champion' ORDER BY symbol, champion_score DESC",
        (args.profile,),
    ).fetchall()
    champion_rows_dict = [dict(row) for row in champion_rows]
    by_symbol: dict[str, list[dict]] = defaultdict(list)
    for row in champion_rows_dict:
        by_symbol[str(row["symbol"])].append(row)
    for symbol, items in by_symbol.items():
        top = max(items, key=lambda item: (float(item["champion_score"]), float(item["portfolio_score"])))
        src_ea = Path(str(top["champion_set_path"]))
        src_audit = src_ea.with_name(src_ea.name.replace("__ea.set", "__audit.set"))
        dst_dir = profile_dir / safe_token(symbol)
        ensure_dir(dst_dir)
        if src_audit.exists():
            shutil.copy2(src_audit, dst_dir / "__TOP__audit.set")
            shutil.copy2(src_audit, testlab.TESTER_PRESET_DIR / f"fxai_offline_{safe_token(args.profile)}__{safe_token(symbol)}__top__audit.set")
        if src_ea.exists():
            shutil.copy2(src_ea, dst_dir / "__TOP__ea.set")
            shutil.copy2(src_ea, testlab.TESTER_PRESET_DIR / f"fxai_offline_{safe_token(args.profile)}__{safe_token(symbol)}__top__ea.set")

    summary_path = RESEARCH_DIR / safe_token(args.profile) / "champions.json"
    ensure_dir(summary_path.parent)
    summary_path.write_text(json.dumps(champion_rows_dict, indent=2, sort_keys=True), encoding="utf-8")
    return decisions


def write_distillation_artifacts(conn: sqlite3.Connection,
                                 args,
                                 promoted_rows: list[dict]) -> list[dict]:
    out_dir = DISTILL_DIR / safe_token(args.profile)
    ensure_dir(out_dir)
    artifacts: list[dict] = []
    created_at = now_unix()
    for row in promoted_rows:
        params = json.loads(row["parameters_json"])
        family_id = int(row.get("family_id", 11))
        distill_profile = family_distillation_profile(family_id)
        support_items = json.loads(row.get("support_json", "[]") or "[]")
        teacher_summary = {
            "plugin_name": row["plugin_name"],
            "symbol": row["symbol"],
            "ai_id": int(row["ai_id"]),
            "family_id": family_id,
            "family_name": plugin_family_name(family_id),
            "ranking_score": float(row["ranking_score"]),
            "score": float(row["score"]),
            "support_count": int(row["support_count"]),
            "support": support_items,
            "parameters": params,
        }
        student_target = dict(distill_profile)
        student_target.update({
            "target_horizon": int(params.get("horizon", 5)),
            "target_execution_profile": str(params.get("execution_profile", "default")),
            "target_sequence_bars": int(params.get("sequence_bars", 0)),
            "target_reliability_floor": round(max(0.42, min(0.86, 0.42 + 0.0035 * float(row["score"]))), 4),
            "target_trade_gate_floor": round(max(0.44, min(0.88, 0.44 + 0.0030 * float(row["ranking_score"]))), 4),
            "support_weight_floor": round(max(0.20, min(0.90, 0.18 + 0.08 * int(row["support_count"]))), 4),
        })
        symbol_dir = out_dir / safe_token(str(row["symbol"]))
        ensure_dir(symbol_dir)
        artifact_path = symbol_dir / f"{row['plugin_name']}__distill.json"
        payload = {
            "teacher_summary": teacher_summary,
            "student_target": student_target,
        }
        artifact_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        artifact_sha = testlab.sha256_path(artifact_path)
        best_cfg = row_to_dict(conn.execute(
            "SELECT id FROM best_configs WHERE profile_name = ? AND dataset_scope = 'aggregate' AND symbol = ? AND plugin_name = ?",
            (args.profile, row["symbol"], row["plugin_name"]),
        ).fetchone())
        best_config_id = int(best_cfg["id"]) if best_cfg else 0
        conn.execute(
            """
            INSERT INTO distillation_artifacts(profile_name, symbol, plugin_name, family_id, source_run_id, best_config_id,
                                               dataset_scope, artifact_path, artifact_sha256, teacher_summary_json,
                                               student_target_json, status, created_at)
            VALUES(?, ?, ?, ?, ?, ?, 'aggregate', ?, ?, ?, ?, 'ready', ?)
            ON CONFLICT(profile_name, symbol, plugin_name, dataset_scope) DO UPDATE SET
                family_id=excluded.family_id,
                source_run_id=excluded.source_run_id,
                best_config_id=excluded.best_config_id,
                artifact_path=excluded.artifact_path,
                artifact_sha256=excluded.artifact_sha256,
                teacher_summary_json=excluded.teacher_summary_json,
                student_target_json=excluded.student_target_json,
                status=excluded.status,
                created_at=excluded.created_at
            """,
            (
                args.profile,
                row["symbol"],
                row["plugin_name"],
                family_id,
                int(row["run_id"]),
                best_config_id,
                str(artifact_path),
                artifact_sha,
                json.dumps(teacher_summary, indent=2, sort_keys=True),
                json.dumps(student_target, indent=2, sort_keys=True),
                created_at,
            ),
        )
        artifacts.append({
            "symbol": row["symbol"],
            "plugin_name": row["plugin_name"],
            "family_id": family_id,
            "artifact_path": str(artifact_path),
            "artifact_sha256": artifact_sha,
        })
    conn.commit()
    summary = out_dir / "distillation_artifacts.json"
    summary.write_text(json.dumps(artifacts, indent=2, sort_keys=True), encoding="utf-8")
    return artifacts


def persist_best_configs(conn: sqlite3.Connection, args, winners: list[dict]) -> list[dict]:
    profile_dir = PROFILES_DIR / safe_token(args.profile)
    ensure_dir(profile_dir)
    ensure_dir(COMMON_PROMOTION_DIR)
    ensure_dir(testlab.TESTER_PRESET_DIR)

    promoted_rows = []
    by_symbol: dict[str, list[dict]] = defaultdict(list)
    now_ts = now_unix()

    for winner in sorted(winners, key=lambda item: (item["symbol"], -float(item["ranking_score"]), item["plugin_name"])):
        params = json.loads(winner["parameters_json"])
        symbol_dir = profile_dir / safe_token(winner["symbol"])
        ensure_dir(symbol_dir)
        audit_set_path = symbol_dir / f"{winner['plugin_name']}__audit.set"
        ea_set_path = symbol_dir / f"{winner['plugin_name']}__ea.set"
        write_audit_set_generic(audit_set_path, winner, params)
        write_ea_set(ea_set_path, winner, params)

        tester_audit_path = testlab.TESTER_PRESET_DIR / f"fxai_offline_{safe_token(args.profile)}__{safe_token(winner['symbol'])}__{winner['plugin_name']}__audit.set"
        tester_ea_path = testlab.TESTER_PRESET_DIR / f"fxai_offline_{safe_token(args.profile)}__{safe_token(winner['symbol'])}__{winner['plugin_name']}__ea.set"
        shutil.copy2(audit_set_path, tester_audit_path)
        shutil.copy2(ea_set_path, tester_ea_path)

        conn.execute(
            """
            INSERT INTO best_configs(dataset_scope, dataset_id, profile_name, symbol, plugin_name, ai_id, family_id, run_id, promoted_at,
                                     score, ranking_score, support_count, parameters_json, audit_set_path, ea_set_path, support_json)
            VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(dataset_scope, profile_name, symbol, plugin_name) DO UPDATE SET
                dataset_id=excluded.dataset_id,
                ai_id=excluded.ai_id,
                family_id=excluded.family_id,
                run_id=excluded.run_id,
                promoted_at=excluded.promoted_at,
                score=excluded.score,
                ranking_score=excluded.ranking_score,
                support_count=excluded.support_count,
                parameters_json=excluded.parameters_json,
                audit_set_path=excluded.audit_set_path,
                ea_set_path=excluded.ea_set_path,
                support_json=excluded.support_json
            """,
            (
                winner["dataset_scope"],
                winner["dataset_id"],
                args.profile,
                winner["symbol"],
                winner["plugin_name"],
                int(winner["ai_id"]),
                int(winner.get("family_id", 11)),
                int(winner["run_id"]),
                now_ts,
                float(winner["score"]),
                float(winner["ranking_score"]),
                int(winner["support_count"]),
                winner["parameters_json"],
                str(audit_set_path),
                str(ea_set_path),
                winner["support_json"],
            ),
        )
        promoted = dict(winner)
        promoted["audit_set_path"] = str(audit_set_path)
        promoted["ea_set_path"] = str(ea_set_path)
        promoted["tester_audit_set_path"] = str(tester_audit_path)
        promoted["tester_ea_set_path"] = str(tester_ea_path)
        by_symbol[winner["symbol"]].append(promoted)
        promoted_rows.append(promoted)

    conn.commit()

    for symbol, rows in by_symbol.items():
        top = max(rows, key=lambda item: (float(item["ranking_score"]), float(item["score"])))
        top_audit_path = profile_dir / safe_token(symbol) / "__TOP__audit.set"
        top_ea_path = profile_dir / safe_token(symbol) / "__TOP__ea.set"
        shutil.copy2(top["audit_set_path"], top_audit_path)
        shutil.copy2(top["ea_set_path"], top_ea_path)
        shutil.copy2(top["audit_set_path"], testlab.TESTER_PRESET_DIR / f"fxai_offline_{safe_token(args.profile)}__{safe_token(symbol)}__top__audit.set")
        shutil.copy2(top["ea_set_path"], testlab.TESTER_PRESET_DIR / f"fxai_offline_{safe_token(args.profile)}__{safe_token(symbol)}__top__ea.set")

    summary_json = profile_dir / "promoted_best.json"
    summary_tsv = profile_dir / "promoted_best.tsv"
    summary_md = profile_dir / "promoted_best.md"
    summary_json.write_text(json.dumps(promoted_rows, indent=2, sort_keys=True), encoding="utf-8")
    with summary_tsv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["symbol", "plugin_name", "ai_id", "score", "ranking_score", "support_count", "audit_set_path", "ea_set_path"])
        for row in sorted(promoted_rows, key=lambda item: (item["symbol"], -float(item["ranking_score"]), item["plugin_name"])):
            writer.writerow([
                row["symbol"],
                row["plugin_name"],
                row["ai_id"],
                f"{float(row['score']):.4f}",
                f"{float(row['ranking_score']):.4f}",
                int(row["support_count"]),
                row["audit_set_path"],
                row["ea_set_path"],
            ])
    md_lines = ["# FXAI Offline Lab Promoted Best", "", f"profile: {args.profile}", ""]
    for symbol in sorted(by_symbol.keys()):
        md_lines.append(f"## {symbol}")
        ranked = sorted(by_symbol[symbol], key=lambda item: (float(item["ranking_score"]), float(item["score"])), reverse=True)
        for row in ranked[:8]:
            params = json.loads(row["parameters_json"])
            md_lines.append(
                f"- {row['plugin_name']} | score {float(row['score']):.2f} | rank {float(row['ranking_score']):.2f} | "
                f"H={int(params.get('horizon', 5))} | M1Sync={int(params.get('m1sync_bars', 3))} | "
                f"Norm={int(params.get('normalization', 0))} | Seq={int(params.get('sequence_bars', 0))} | "
                f"Schema={int(params.get('schema_id', 0))} | Mask={int(params.get('feature_mask', 0))}"
            )
        md_lines.append("")
    summary_md.write_text("\n".join(md_lines), encoding="utf-8")

    common_json = COMMON_PROMOTION_DIR / f"fxai_offline_best_{safe_token(args.profile)}.json"
    common_tsv = COMMON_PROMOTION_DIR / f"fxai_offline_best_{safe_token(args.profile)}.tsv"
    shutil.copy2(summary_json, common_json)
    shutil.copy2(summary_tsv, common_tsv)
    return promoted_rows


def cmd_best_params(args) -> int:
    conn = connect_db(Path(args.db))
    rows = load_completed_runs(conn, args)
    if not rows:
        raise OfflineLabError("no completed tuning runs found for best-params")
    winners, _dataset_counts = aggregate_best_candidates(rows)
    promoted = persist_best_configs(conn, args, winners)
    champion_decisions = update_champion_registry(conn, args, promoted)
    family_scorecards = persist_family_scorecards(conn, args, rows, promoted)
    distill_artifacts = write_distillation_artifacts(conn, args, promoted)
    print(json.dumps({
        "profile": args.profile,
        "promoted_count": len(promoted),
        "champion_count": sum(1 for item in champion_decisions if item["status"] == "champion"),
        "challenger_count": sum(1 for item in champion_decisions if item["status"] == "challenger"),
        "family_scorecards": len(family_scorecards),
        "distillation_artifacts": len(distill_artifacts),
    }, indent=2, sort_keys=True))
    conn.close()
    return 0


def cmd_control_loop(args) -> int:
    conn = connect_db(Path(args.db))
    cycles = int(getattr(args, "cycles", 1) or 0)
    cycle_idx = 0
    while True:
        cycle_idx += 1
        started_at = now_unix()
        group_key = safe_token(f"{args.profile}_{started_at}")
        conn.execute(
            "INSERT INTO control_cycles(profile_name, group_key, started_at, status, notes) VALUES(?, ?, ?, 'running', ?)",
            (args.profile, group_key, started_at, f"cycle={cycle_idx}"),
        )
        cycle_id = int(conn.execute("SELECT last_insert_rowid()").fetchone()[0])
        conn.commit()
        try:
            cycle_args = argparse.Namespace(**vars(args))
            cycle_args.group_key = group_key
            resolve_args = cycle_args
            if not getattr(args, "skip_compile", False):
                if compile_export_runner() != 0:
                    raise OfflineLabError("failed to compile FXAI_OfflineExportRunner.mq5")
                resolve_args = argparse.Namespace(**vars(cycle_args))
                resolve_args.skip_compile = True
            datasets = resolve_dataset_rows(conn, resolve_args, True, group_key)
            if not getattr(args, "skip_compile", False):
                if compile_audit_runner() != 0:
                    raise OfflineLabError("failed to compile FXAI_AuditRunner.mq5")
            summary_items = []
            for dataset in datasets:
                dataset_out_dir = RUNS_DIR / safe_token(args.profile) / safe_token(dataset["dataset_key"])
                ensure_dir(dataset_out_dir)
                baseline = run_dataset_baseline(conn, dataset, args.profile, cycle_args, dataset_out_dir)
                results = run_dataset_campaign(conn, dataset, args.profile, cycle_args, dataset_out_dir, baseline["summary"], baseline["base_args"])
                summary_items.append({"dataset_key": dataset["dataset_key"], "symbol": dataset["symbol"], "run_count": len(results)})
            best_args = argparse.Namespace(**vars(cycle_args))
            best_args.group_key = group_key
            cmd_best_params(best_args)
            conn.execute(
                "UPDATE control_cycles SET finished_at = ?, status = 'ok', datasets_json = ? WHERE id = ?",
                (now_unix(), json.dumps(summary_items, indent=2, sort_keys=True), cycle_id),
            )
            conn.commit()
        except Exception as exc:
            conn.execute(
                "UPDATE control_cycles SET finished_at = ?, status = 'failed', notes = ? WHERE id = ?",
                (now_unix(), str(exc), cycle_id),
            )
            conn.commit()
            conn.close()
            raise

        if cycles > 0 and cycle_idx >= cycles:
            break
        sleep_seconds = int(getattr(args, "sleep_seconds", 0) or 0)
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)
    conn.close()
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="FXAI SQLite-backed offline tuning and control lab")
    ap.add_argument("--db", default=str(DEFAULT_DB))
    sub = ap.add_subparsers(dest="cmd", required=True)

    init_db = sub.add_parser("init-db", help="Initialize the SQLite offline lab schema")
    init_db.set_defaults(func=cmd_init_db)

    comp = sub.add_parser("compile-export", help="Compile the MT5 offline export runner")
    comp.set_defaults(func=cmd_compile_export)

    exp = sub.add_parser("export-dataset", help="Export exact-window M1 OHLC+spread history from MT5 into SQLite")
    exp.add_argument("--symbol", default="EURUSD")
    exp.add_argument("--symbol-list", default="")
    exp.add_argument("--symbol-pack", default="", choices=[""] + sorted(testlab.SYMBOL_PACKS.keys()))
    exp.add_argument("--months-list", default="3,6,12")
    exp.add_argument("--start-unix", type=int, default=0)
    exp.add_argument("--end-unix", type=int, default=0)
    exp.add_argument("--max-bars", type=int, default=600000)
    exp.add_argument("--group-key", default="")
    exp.add_argument("--notes", default="")
    exp.add_argument("--replace", action="store_true")
    exp.add_argument("--skip-compile", action="store_true")
    exp.add_argument("--login")
    exp.add_argument("--server")
    exp.add_argument("--password")
    exp.add_argument("--timeout", type=int, default=300)
    exp.set_defaults(func=cmd_export_dataset)

    tune = sub.add_parser("tune-zoo", help="Run the full MT5 model-zoo tuning campaign on exact exported windows")
    tune.add_argument("--profile", default="continuous")
    tune.add_argument("--dataset-keys", default="")
    tune.add_argument("--group-key", default="")
    tune.add_argument("--auto-export", action="store_true")
    tune.add_argument("--symbol", default="EURUSD")
    tune.add_argument("--symbol-list", default="")
    tune.add_argument("--symbol-pack", default="", choices=[""] + sorted(testlab.SYMBOL_PACKS.keys()))
    tune.add_argument("--months-list", default="3,6,12")
    tune.add_argument("--start-unix", type=int, default=0)
    tune.add_argument("--end-unix", type=int, default=0)
    tune.add_argument("--replace", action="store_true")
    tune.add_argument("--skip-compile", action="store_true")
    tune.add_argument("--top-plugins", type=int, default=0)
    tune.add_argument("--limit-experiments", type=int, default=0)
    tune.add_argument("--limit-runs", type=int, default=0)
    tune.add_argument("--scenario-list", default=SERIOUS_SCENARIOS)
    tune.add_argument("--bars", type=int, default=0)
    tune.add_argument("--horizon", type=int, default=5)
    tune.add_argument("--m1sync-bars", type=int, default=3)
    tune.add_argument("--normalization", type=int, default=0)
    tune.add_argument("--sequence-bars", type=int, default=0)
    tune.add_argument("--schema-id", type=int, default=0)
    tune.add_argument("--feature-mask", type=int, default=0)
    tune.add_argument("--commission-per-lot-side", type=float, default=None)
    tune.add_argument("--cost-buffer-points", type=float, default=None)
    tune.add_argument("--slippage-points", type=float, default=None)
    tune.add_argument("--fill-penalty-points", type=float, default=None)
    tune.add_argument("--wf-train-bars", type=int, default=256)
    tune.add_argument("--wf-test-bars", type=int, default=64)
    tune.add_argument("--wf-purge-bars", type=int, default=32)
    tune.add_argument("--wf-embargo-bars", type=int, default=24)
    tune.add_argument("--wf-folds", type=int, default=6)
    tune.add_argument("--seed", type=int, default=42)
    tune.add_argument("--execution-profile", default="default", choices=sorted(testlab.EXECUTION_PROFILES.keys()))
    tune.add_argument("--login")
    tune.add_argument("--server")
    tune.add_argument("--password")
    tune.add_argument("--timeout", type=int, default=300)
    tune.set_defaults(func=cmd_tune_zoo)

    best = sub.add_parser("best-params", help="Promote the strongest parameter packs and emit MT5-ready presets")
    best.add_argument("--profile", default="continuous")
    best.add_argument("--dataset-keys", default="")
    best.add_argument("--group-key", default="")
    best.add_argument("--symbol", default="")
    best.add_argument("--symbol-list", default="")
    best.add_argument("--symbol-pack", default="", choices=[""] + sorted(testlab.SYMBOL_PACKS.keys()))
    best.set_defaults(func=cmd_best_params)

    loop = sub.add_parser("control-loop", help="Run the full export -> tune -> promote cycle continuously")
    loop.add_argument("--profile", default="continuous")
    loop.add_argument("--symbol", default="EURUSD")
    loop.add_argument("--symbol-list", default="")
    loop.add_argument("--symbol-pack", default="", choices=[""] + sorted(testlab.SYMBOL_PACKS.keys()))
    loop.add_argument("--months-list", default="3,6,12")
    loop.add_argument("--start-unix", type=int, default=0)
    loop.add_argument("--end-unix", type=int, default=0)
    loop.add_argument("--replace", action="store_true")
    loop.add_argument("--skip-compile", action="store_true")
    loop.add_argument("--top-plugins", type=int, default=0)
    loop.add_argument("--limit-experiments", type=int, default=0)
    loop.add_argument("--limit-runs", type=int, default=0)
    loop.add_argument("--scenario-list", default=SERIOUS_SCENARIOS)
    loop.add_argument("--bars", type=int, default=0)
    loop.add_argument("--horizon", type=int, default=5)
    loop.add_argument("--m1sync-bars", type=int, default=3)
    loop.add_argument("--normalization", type=int, default=0)
    loop.add_argument("--sequence-bars", type=int, default=0)
    loop.add_argument("--schema-id", type=int, default=0)
    loop.add_argument("--feature-mask", type=int, default=0)
    loop.add_argument("--commission-per-lot-side", type=float, default=None)
    loop.add_argument("--cost-buffer-points", type=float, default=None)
    loop.add_argument("--slippage-points", type=float, default=None)
    loop.add_argument("--fill-penalty-points", type=float, default=None)
    loop.add_argument("--wf-train-bars", type=int, default=256)
    loop.add_argument("--wf-test-bars", type=int, default=64)
    loop.add_argument("--wf-purge-bars", type=int, default=32)
    loop.add_argument("--wf-embargo-bars", type=int, default=24)
    loop.add_argument("--wf-folds", type=int, default=6)
    loop.add_argument("--seed", type=int, default=42)
    loop.add_argument("--execution-profile", default="default", choices=sorted(testlab.EXECUTION_PROFILES.keys()))
    loop.add_argument("--cycles", type=int, default=1, help="0 means run forever")
    loop.add_argument("--sleep-seconds", type=int, default=0)
    loop.add_argument("--login")
    loop.add_argument("--server")
    loop.add_argument("--password")
    loop.add_argument("--timeout", type=int, default=300)
    loop.set_defaults(func=cmd_control_loop)

    return ap


def main() -> int:
    ap = build_parser()
    args = ap.parse_args()
    try:
        return int(args.func(args))
    except OfflineLabError as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
