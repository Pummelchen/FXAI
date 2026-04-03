from __future__ import annotations

from pathlib import Path
import tempfile
import time
import libsql

from .common import *

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


def load_dataset(conn: libsql.Connection, dataset_key: str) -> dict | None:
    return query_one(conn, "SELECT * FROM datasets WHERE dataset_key = ?", (dataset_key,))


def load_dataset_by_id(conn: libsql.Connection, dataset_id: int) -> dict | None:
    return query_one(conn, "SELECT * FROM datasets WHERE id = ?", (dataset_id,))


def insert_dataset_bars(conn: libsql.Connection, dataset_id: int, data_path: Path) -> int:
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


def ingest_dataset(conn: libsql.Connection,
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
    dataset_id = int(query_scalar(conn, "SELECT id FROM datasets WHERE dataset_key = ?", (dataset_key,), 0))
    inserted = insert_dataset_bars(conn, dataset_id, data_path)
    conn.execute("UPDATE datasets SET bars = ? WHERE id = ?", (inserted, dataset_id))
    commit_db(conn)
    dataset = load_dataset_by_id(conn, dataset_id)
    if dataset is None:
        raise OfflineLabError(f"failed to reload dataset {dataset_key}")
    return dataset


def export_single_dataset(conn: libsql.Connection, args, symbol: str, months: int, group_key: str) -> dict:
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


def resolve_dataset_rows(conn: libsql.Connection, args, auto_export: bool, group_key: str) -> list[dict]:
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
