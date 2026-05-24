from __future__ import annotations

from pathlib import Path
import tempfile
import time
import urllib.error
import urllib.request
import libsql

from .common import *

FXDATABASE_FXBACKTEST_API_VERSION = "fxdatabase.fxbacktest.v1"
FXDATABASE_M1_HISTORY_PATH = "/v1/history/m1"


def _required_int(payload: dict, key: str) -> int:
    try:
        return _coerce_int(payload[key], key)
    except KeyError as exc:
        raise OfflineLabError(f"FXDatabase API response field {key} must be an integer") from exc


def _coerce_int(value: object, field: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise OfflineLabError(f"FXDatabase API response field {field} must be an integer") from exc


def _normalize_fxdatabase_source_origin(value: str) -> str:
    normalized = str(value or "").strip().upper()
    if not normalized:
        raise OfflineLabError("FXDatabase source_origin must not be empty")
    if not all(("A" <= char <= "Z") or char.isdigit() or char in "_-" for char in normalized):
        raise OfflineLabError("FXDatabase source_origin must use uppercase letters, numbers, underscore, or hyphen")
    return normalized


def validate_fxdatabase_m1_history_response(response: dict) -> None:
    if not isinstance(response, dict):
        raise OfflineLabError("FXDatabase API history response must be a JSON object")
    api_version = str(response.get("api_version", ""))
    if api_version != FXDATABASE_FXBACKTEST_API_VERSION:
        raise OfflineLabError(
            f"FXDatabase API version mismatch: got {api_version or '<missing>'}, "
            f"expected {FXDATABASE_FXBACKTEST_API_VERSION}"
        )

    metadata = response.get("metadata")
    if not isinstance(metadata, dict):
        raise OfflineLabError("FXDatabase API history response metadata must be a JSON object")
    for key in ("broker_source_id", "logical_symbol", "mt5_symbol"):
        if not str(metadata.get(key, "")).strip():
            raise OfflineLabError(f"FXDatabase API metadata.{key} must not be empty")
    _normalize_fxdatabase_source_origin(str(metadata.get("source_origin", "")))
    if str(metadata.get("timeframe", "")) != "M1":
        raise OfflineLabError("FXDatabase API metadata.timeframe must be M1")
    digits = _required_int(metadata, "digits")
    if digits < 0 or digits > 10:
        raise OfflineLabError("FXDatabase API metadata.digits must be in 0...10")
    requested_start = _required_int(metadata, "requested_utc_start")
    requested_end = _required_int(metadata, "requested_utc_end_exclusive")
    if requested_start >= requested_end or requested_start % 60 != 0 or requested_end % 60 != 0:
        raise OfflineLabError("FXDatabase API metadata requested UTC range must be minute-aligned and increasing")

    columns: dict[str, list] = {}
    for key in ("utc_timestamps", "open", "high", "low", "close", "volume"):
        value = response.get(key)
        if not isinstance(value, list):
            raise OfflineLabError(f"FXDatabase API response field {key} must be an array")
        columns[key] = value
    count = len(columns["utc_timestamps"])
    if _required_int(metadata, "row_count") != count:
        raise OfflineLabError("FXDatabase API metadata.row_count does not match utc_timestamps count")
    if any(len(values) != count for values in columns.values()):
        raise OfflineLabError("FXDatabase API returned mismatched OHLCV column lengths")

    if count == 0:
        if metadata.get("first_utc") is not None or metadata.get("last_utc") is not None:
            raise OfflineLabError("FXDatabase API first_utc/last_utc must be null for an empty history response")
        return

    first_utc = _required_int(metadata, "first_utc")
    last_utc = _required_int(metadata, "last_utc")
    first_timestamp = _coerce_int(columns["utc_timestamps"][0], "utc_timestamps[0]")
    last_timestamp = _coerce_int(columns["utc_timestamps"][-1], "utc_timestamps[-1]")
    if first_utc != first_timestamp or last_utc != last_timestamp:
        raise OfflineLabError("FXDatabase API first_utc/last_utc do not match timestamp columns")

    previous_timestamp: int | None = None
    for index, timestamp_value in enumerate(columns["utc_timestamps"]):
        timestamp = _coerce_int(timestamp_value, f"utc_timestamps[{index}]")
        if timestamp % 60 != 0 or timestamp < requested_start or timestamp >= requested_end:
            raise OfflineLabError("FXDatabase API timestamp columns must be minute-aligned inside the requested range")
        if previous_timestamp is not None and timestamp <= previous_timestamp:
            raise OfflineLabError("FXDatabase API utc_timestamps must be strictly increasing")
        previous_timestamp = timestamp

        open_price = _coerce_int(columns["open"][index], f"open[{index}]")
        high_price = _coerce_int(columns["high"][index], f"high[{index}]")
        low_price = _coerce_int(columns["low"][index], f"low[{index}]")
        close_price = _coerce_int(columns["close"][index], f"close[{index}]")
        volume = _coerce_int(columns["volume"][index], f"volume[{index}]")
        if min(open_price, high_price, low_price, close_price) <= 0:
            raise OfflineLabError("FXDatabase API OHLC values must be positive")
        if volume < 0:
            raise OfflineLabError("FXDatabase API volume values must be non-negative")
        if high_price < max(open_price, low_price, close_price) or low_price > min(open_price, high_price, close_price):
            raise OfflineLabError(f"FXDatabase API OHLC invariant failed at row {index}")


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
    return testlab.compile_target(Path("FXDataEngine"), "data_engine")


def compile_audit_runner() -> int:
    return testlab.cmd_compile(argparse.Namespace())


def fxdatabase_api_base_url(args) -> str:
    raw = (
        getattr(args, "fxdatabase_api_url", "")
        or os.environ.get("FXDATABASE_API_URL", "")
        or os.environ.get("FXAI_FXDATABASE_API_URL", "")
        or "http://127.0.0.1:8765"
    )
    return str(raw).rstrip("/")


def fetch_fxdatabase_m1_history(args, symbol: str, start_unix: int, end_unix: int) -> dict:
    broker_source_id = str(
        getattr(args, "broker_source_id", "") or os.environ.get("FXDATABASE_BROKER_SOURCE_ID", "default")
    ).strip()
    source_origin = _normalize_fxdatabase_source_origin(
        getattr(args, "source_origin", "") or os.environ.get("FXDATABASE_SOURCE_ORIGIN", "MT5")
    )
    logical_symbol = str(symbol or "").strip().upper()
    if not broker_source_id:
        raise OfflineLabError("FXDatabase broker_source_id must not be empty")
    if not logical_symbol:
        raise OfflineLabError("FXDatabase logical_symbol must not be empty")
    payload = {
        "api_version": FXDATABASE_FXBACKTEST_API_VERSION,
        "broker_source_id": broker_source_id,
        "source_origin": source_origin,
        "logical_symbol": logical_symbol,
        "utc_start_inclusive": int(start_unix),
        "utc_end_exclusive": int(end_unix),
        "maximum_rows": int(getattr(args, "max_bars", 600000) or 600000),
    }
    body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    request = urllib.request.Request(
        f"{fxdatabase_api_base_url(args)}{FXDATABASE_M1_HISTORY_PATH}",
        data=body,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=max(1, int(getattr(args, "timeout", 300) or 300))) as response:
            payload = json.loads(response.read().decode("utf-8"))
            validate_fxdatabase_m1_history_response(payload)
            return payload
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise OfflineLabError(f"FXDatabase API history request failed with HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise OfflineLabError(f"FXDatabase API history request failed: {exc}") from exc


def write_fxdatabase_export_files(response: dict, data_path: Path, meta_path: Path) -> None:
    validate_fxdatabase_m1_history_response(response)
    metadata = dict(response.get("metadata", {}))
    timestamps = list(response.get("utc_timestamps", []))
    opens = list(response.get("open", []))
    highs = list(response.get("high", []))
    lows = list(response.get("low", []))
    closes = list(response.get("close", []))
    volumes = list(response.get("volume", []))
    count = len(timestamps)
    if not (len(opens) == len(highs) == len(lows) == len(closes) == len(volumes) == count):
        raise OfflineLabError("FXDatabase API returned mismatched OHLCV column lengths")
    digits = int(metadata.get("digits", 0) or 0)
    scale = 10.0 ** max(digits, 0)
    ensure_dir(data_path.parent)
    with data_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["time_unix", "open", "high", "low", "close", "price_cost_points", "tick_volume", "real_volume"])
        for index in range(count):
            writer.writerow([
                int(timestamps[index]),
                float(opens[index]) / scale,
                float(highs[index]) / scale,
                float(lows[index]) / scale,
                float(closes[index]) / scale,
                0.0,
                int(volumes[index]),
                int(volumes[index]),
            ])
    meta = {
        "api_version": str(response.get("api_version", "")),
        "broker_source_id": str(metadata.get("broker_source_id", "")),
        "source_origin": str(metadata.get("source_origin", "")),
        "logical_symbol": str(metadata.get("logical_symbol", "")),
        "provider_symbol": str(metadata.get("mt5_symbol", "")),
        "timeframe": str(metadata.get("timeframe", "M1")),
        "digits": str(digits),
        "window_start_unix": str(metadata.get("requested_utc_start", "")),
        "window_end_unix": str(metadata.get("requested_utc_end_exclusive", "")),
        "first_time_unix": str(metadata.get("first_utc") or ""),
        "last_time_unix": str(metadata.get("last_utc") or ""),
        "row_count": str(metadata.get("row_count", count)),
        "volume_policy": "provider_volume_when_positive",
    }
    ensure_dir(meta_path.parent)
    meta_path.write_text("\n".join(f"{key}\t{value}" for key, value in meta.items()) + "\n", encoding="utf-8")


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
    dataset_bar_columns = table_columns(conn, "dataset_bars")
    has_legacy_spread_column = "spread_points" in dataset_bar_columns
    if has_legacy_spread_column:
        insert_sql = (
            "INSERT INTO dataset_bars(dataset_id, bar_time_unix, open, high, low, close, "
            "price_cost_points, spread_points, tick_volume, real_volume) "
            "VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
        )
    else:
        insert_sql = (
            "INSERT INTO dataset_bars(dataset_id, bar_time_unix, open, high, low, close, "
            "price_cost_points, tick_volume, real_volume) "
            "VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)"
        )
    with data_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            bar_time = int(float(row["time_unix"]))
            o = float(row["open"])
            h = float(row["high"])
            l = float(row["low"])
            c = float(row["close"])
            price_cost_points = float(row.get("price_cost_points", row.get("spread_points", 0.0)) or 0.0)
            tick_volume = int(float(row["tick_volume"]))
            real_volume = int(float(row["real_volume"]))
            if bar_time <= prev_time:
                raise OfflineLabError(f"dataset bars not strictly ascending at {bar_time} in {data_path}")
            prev_time = bar_time
            if h + 1e-12 < max(o, c, l) or l - 1e-12 > min(o, c, h):
                raise OfflineLabError(f"invalid OHLC geometry at {bar_time} in {data_path}")
            if has_legacy_spread_column:
                batch.append((dataset_id, bar_time, o, h, l, c, price_cost_points, price_cost_points, tick_volume, real_volume))
            else:
                batch.append((dataset_id, bar_time, o, h, l, c, price_cost_points, tick_volume, real_volume))
            if len(batch) >= 10000:
                conn.executemany(insert_sql, batch)
                inserted += len(batch)
                batch = []
        if batch:
            conn.executemany(insert_sql, batch)
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

    output_key = dataset_key
    data_path = dataset_data_path(output_key, symbol)
    meta_path = dataset_meta_path(output_key, symbol)
    ensure_dir(COMMON_EXPORT_DIR)
    if data_path.exists():
        data_path.unlink()
    if meta_path.exists():
        meta_path.unlink()

    response = fetch_fxdatabase_m1_history(args, symbol, start_unix, end_unix)
    write_fxdatabase_export_files(response, data_path, meta_path)

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
