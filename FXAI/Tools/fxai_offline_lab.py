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
    symbol TEXT NOT NULL,
    plugin_name TEXT NOT NULL,
    ai_id INTEGER NOT NULL,
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

CREATE INDEX IF NOT EXISTS idx_datasets_group ON datasets(group_key, symbol, months);
CREATE INDEX IF NOT EXISTS idx_tuning_runs_lookup ON tuning_runs(profile_name, symbol, plugin_name, status);
CREATE INDEX IF NOT EXISTS idx_tuning_runs_dataset ON tuning_runs(dataset_id, profile_name, plugin_name);
CREATE INDEX IF NOT EXISTS idx_best_configs_lookup ON best_configs(profile_name, symbol, plugin_name);
CREATE INDEX IF NOT EXISTS idx_control_cycles_lookup ON control_cycles(profile_name, started_at);
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
        symbols = [str(getattr(args, "symbol", "EURUSD"))]
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


def connect_db(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.executescript(SQL_SCHEMA)
    return conn


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
    start_unix = int(meta.get("window_start_unix", "0") or 0)
    end_unix = int(meta.get("window_end_unix", "0") or 0)
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
                      plugin_name: str,
                      ai_id: int,
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
            dataset_id, profile_name, symbol, plugin_name, ai_id, experiment_name, param_hash, parameters_json,
            report_path, raw_report_path, summary_path, manifest_path,
            score, grade, issue_count, issues_json,
            market_recent_score, walkforward_score, adversarial_score, macro_event_score,
            status, started_at, finished_at
        )
        VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(param_hash) DO UPDATE SET
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
            str(dataset["symbol"]),
            plugin_name,
            ai_id,
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
        params = normalize_namespace_parameters(base_args, plugin_name, "baseline_all", dataset)
        upsert_tuning_run(
            conn,
            dataset,
            profile_name,
            plugin_name,
            ai_id,
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
    store_baseline_run_bundle(conn, dataset, profile_name, base_args, baseline_path, raw_report_path, summary_path, manifest_path, started_at, finished_at)
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
        status = "failed"
        if rc == 0:
            shutil.copy2(testlab.DEFAULT_REPORT, raw_report_path)
            summary = testlab.load_json(summary_path)
            plugin_summary = summary.get("plugins", {}).get(run["plugin"], {})
            rows = grouped_rows_by_plugin(raw_report_path).get(run["plugin"], [])
            if rows:
                ai_id = int(float(rows[0]["ai_id"]))
            status = "ok"
        run_id = upsert_tuning_run(
            conn,
            dataset,
            profile_name,
            run["plugin"],
            ai_id,
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
        clauses.append("d.group_key = ?")
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
            INSERT INTO best_configs(dataset_scope, dataset_id, profile_name, symbol, plugin_name, ai_id, run_id, promoted_at,
                                     score, ranking_score, support_count, parameters_json, audit_set_path, ea_set_path, support_json)
            VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(dataset_scope, profile_name, symbol, plugin_name) DO UPDATE SET
                dataset_id=excluded.dataset_id,
                ai_id=excluded.ai_id,
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
    print(json.dumps({"profile": args.profile, "promoted_count": len(promoted)}, indent=2, sort_keys=True))
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
            resolve_args = args
            if not getattr(args, "skip_compile", False):
                if compile_export_runner() != 0:
                    raise OfflineLabError("failed to compile FXAI_OfflineExportRunner.mq5")
                resolve_args = argparse.Namespace(**vars(args))
                resolve_args.skip_compile = True
            datasets = resolve_dataset_rows(conn, resolve_args, True, group_key)
            if not getattr(args, "skip_compile", False):
                if compile_audit_runner() != 0:
                    raise OfflineLabError("failed to compile FXAI_AuditRunner.mq5")
            summary_items = []
            for dataset in datasets:
                dataset_out_dir = RUNS_DIR / safe_token(args.profile) / safe_token(dataset["dataset_key"])
                ensure_dir(dataset_out_dir)
                baseline = run_dataset_baseline(conn, dataset, args.profile, args, dataset_out_dir)
                results = run_dataset_campaign(conn, dataset, args.profile, args, dataset_out_dir, baseline["summary"], baseline["base_args"])
                summary_items.append({"dataset_key": dataset["dataset_key"], "symbol": dataset["symbol"], "run_count": len(results)})
            best_args = argparse.Namespace(**vars(args))
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
    best.add_argument("--symbol", default="EURUSD")
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
