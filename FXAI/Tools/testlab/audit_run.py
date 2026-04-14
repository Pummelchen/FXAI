from __future__ import annotations

import shutil
import subprocess
import tempfile
import time
from pathlib import Path

from .compile import resolve_credentials, terminal_running, update_ini_section
from .reporting import build_multisymbol_summary, build_summary, load_rows, render_report
from .shared import (
    AuditRunError,
    COMMON_INI,
    DEFAULT_REPORT,
    MT5_LOG_DIR,
    TERMINAL,
    TERMINAL_INI,
    TESTER_PRESET_DIR,
    WINE,
    clone_args,
    load_oracles,
    read_utf16_or_text,
    to_wine_path,
)

def latest_terminal_log() -> Path | None:
    logs = [
        p for p in MT5_LOG_DIR.glob("*.log")
        if p.stem.isdigit() and len(p.stem) == 8
    ]
    logs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return logs[0] if logs else None


def extract_terminal_failure(log_text: str) -> str:
    needles = [
        "tester not started because the account is not specified",
        "tester EX5 not found",
        "tester didn't start",
        "incorrect input parameters",
    ]
    lower = log_text.lower()
    for needle in needles:
        idx = lower.rfind(needle.lower())
        if idx >= 0:
            snippet = log_text[idx: idx + 220].splitlines()[0].strip()
            return snippet
    return ""


def write_audit_set(path: Path, args) -> None:
    all_plugins = args.all_plugins or args.plugin_list.strip().lower() == "{all}"
    content = "\n".join([
        f"Audit_AllPlugins={'true' if all_plugins else 'false'}||false||0||true||N",
        f"Audit_Plugin={args.plugin_id}||0||0||28||N",
        f"Audit_PluginList={args.plugin_list}||0||0||0||N",
        f"Audit_ScenarioList={args.scenario_list}||0||0||0||N",
        f"Audit_Bars={args.bars}||2048||1||1000000||N",
        f"PredictionTargetMinutes={args.horizon}||1||1||720||N",
        f"Audit_M1SyncBars={args.m1sync_bars}||2||1||12||N",
        f"Audit_Normalization={args.normalization}||0||0||16||N",
        f"Audit_SequenceBarsOverride={args.sequence_bars}||0||0||256||N",
        f"Audit_SchemaOverride={args.schema_id}||0||0||6||N",
        f"Audit_FeatureGroupsMaskOverride={args.feature_mask}||0||0||9223372036854775807||N",
        f"Audit_CommissionPerLotSide={args.commission_per_lot_side}||0||0||100||N",
        f"Audit_CostBufferPoints={args.cost_buffer_points}||0||0||100||N",
        f"Audit_SlippagePoints={args.slippage_points}||0||0||100||N",
        f"Audit_FillPenaltyPoints={args.fill_penalty_points}||0||0||100||N",
        f"Audit_WalkForwardTrainBars={args.wf_train_bars}||64||1||1000000||N",
        f"Audit_WalkForwardTestBars={args.wf_test_bars}||16||1||1000000||N",
        f"Audit_WalkForwardPurgeBars={args.wf_purge_bars}||0||0||1000000||N",
        f"Audit_WalkForwardEmbargoBars={args.wf_embargo_bars}||0||0||1000000||N",
        f"Audit_WalkForwardFolds={args.wf_folds}||2||1||64||N",
        f"Audit_WindowStartUnix={getattr(args, 'window_start_unix', 0)}||0||0||2147483647||N",
        f"Audit_WindowEndUnix={getattr(args, 'window_end_unix', 0)}||0||0||2147483647||N",
        f"Audit_Seed={args.seed}||0||1||1000000||N",
        "Audit_ResetOutput=true||false||0||true||N",
        "Audit_StopOnFailure=false||false||0||true||N",
        "TradeKiller=0||0||0||10000||N",
    ]) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def write_audit_ini(path: Path, preset_name: str, login: str, symbol: str, server: str = "", password: str = "") -> None:
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
        "Expert=FXAI\\Tests\\FXAI_AuditRunner.ex5",
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
        "Report=fxai_audit_runner_auto",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_terminal_audit(config_path: Path, timeout_sec: int) -> None:
    cmd = [str(WINE), str(TERMINAL), f"/config:{to_wine_path(config_path)}"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        proc.communicate(timeout=timeout_sec)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.communicate()
        raise AuditRunError(f"MT5 tester timed out after {timeout_sec}s")


def run_terminal_profile(timeout_sec: int) -> None:
    cmd = [str(WINE), str(TERMINAL), "/portable"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        proc.communicate(timeout=timeout_sec)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.communicate()
        raise AuditRunError(f"MT5 tester timed out after {timeout_sec}s")


def build_profile_tester_section(preset_name: str, symbol: str, login: str = "", server: str = "") -> dict[str, str]:
    return {
        "LastExpert": r"FXAI\Tests\FXAI_AuditRunner.ex5",
        "LastIndicator": r"Indicators\Examples\Accelerator.ex5",
        "LastTicksMode": "1",
        "LastCriterion": "0",
        "LastForward": "0",
        "LastDelay": "100",
        "LastOptimization": "0",
        "Expert": r"FXAI\Tests\FXAI_AuditRunner.ex5",
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
        "Report": "fxai_audit_runner_auto",
        "ReplaceReport": "1",
        "ShutdownTerminal": "1",
    }


def attempt_audit_launch(login: str, server: str, password: str, preset_name: str, args) -> tuple[bool, str, str]:
    start_ts = time.time()
    config_path = Path(tempfile.gettempdir()) / "fxai_audit_runner.ini"
    write_audit_ini(config_path, preset_name, login, args.symbol, server, password)
    try:
        run_terminal_audit(config_path, args.timeout)
    except AuditRunError as exc:
        return False, "config", str(exc)
    if DEFAULT_REPORT.exists() and DEFAULT_REPORT.stat().st_mtime >= start_ts:
        return True, "config", ""

    log_path = latest_terminal_log()
    log_text = read_utf16_or_text(log_path) if log_path else ""
    failure = extract_terminal_failure(log_text)
    if failure and "account is not specified" in failure.lower() and not password:
        return False, "config", failure

    if terminal_running():
        if not failure:
            failure = "profile fallback skipped because terminal64.exe is already running"
        return False, "config", failure

    common_backup = COMMON_INI.read_bytes()
    terminal_backup = TERMINAL_INI.read_bytes()
    try:
        if login or server:
            update_ini_section(
                COMMON_INI,
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
        update_ini_section(TERMINAL_INI, "Tester", build_profile_tester_section(preset_name, args.symbol, login, server))
        start_ts_profile = time.time()
        run_terminal_profile(args.timeout)
        if DEFAULT_REPORT.exists() and DEFAULT_REPORT.stat().st_mtime >= start_ts_profile:
            return True, "profile", ""
        log_path = latest_terminal_log()
        log_text = read_utf16_or_text(log_path) if log_path else ""
        failure2 = extract_terminal_failure(log_text)
        if not failure2:
            failure2 = "profile launch exited without producing the audit report"
        return False, "profile", failure2
    finally:
        COMMON_INI.write_bytes(common_backup)
        TERMINAL_INI.write_bytes(terminal_backup)


def run_single_symbol_audit(args, symbol: str, raw_report_path: Path | None = None) -> dict:
    run_args = clone_args(args)
    run_args.symbol = symbol
    run_args.symbol_list = "{" + symbol + "}"

    preset_name = "fxai_audit_runner.set"
    preset_path = TESTER_PRESET_DIR / preset_name
    write_audit_set(preset_path, run_args)

    login, server, password = resolve_credentials(run_args)

    if DEFAULT_REPORT.exists():
        DEFAULT_REPORT.unlink()
    success, mode, failure = attempt_audit_launch(login, server, password, preset_name, run_args)
    if not success:
        if not failure and not login:
            failure = "MT5 tester did not produce a report and no login was available from common.ini"
        elif not failure and not password:
            failure = "MT5 tester did not produce a report and no password was supplied; set FXAI_MT5_PASSWORD or use --password"
        elif not failure:
            failure = "MT5 tester exited without producing the audit report"
        raise AuditRunError(f"{mode} launch failed for {symbol}: {failure}")

    if raw_report_path is not None:
        raw_report_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(DEFAULT_REPORT, raw_report_path)

    oracles = load_oracles()
    rows = load_rows(DEFAULT_REPORT)
    summary = build_summary(rows, oracles)
    text = render_report(rows, oracles)
    return {
        "symbol": symbol,
        "rows": rows,
        "summary": summary,
        "text": text,
        "execution_profile": getattr(run_args, "execution_profile", "default"),
    }
