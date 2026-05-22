from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from pathlib import Path

from .toolchain import FXAIToolchainConfig, SUPPORTED_TOOLCHAIN_PROFILES, load_toolchain_config
from .strategy_profiles import compile_strategy_profile


ACTIVE_TOOLCHAIN: FXAIToolchainConfig = load_toolchain_config()
TOOLCHAIN_PROFILE = ACTIVE_TOOLCHAIN.profile
FXAI_CONFIG_PATH = ACTIVE_TOOLCHAIN.config_path
FXAI_ENV_PATH = ACTIVE_TOOLCHAIN.env_path

ROOT = ACTIVE_TOOLCHAIN.project_root
REPO_ROOT = ACTIVE_TOOLCHAIN.repo_root
TERMINAL_ROOT = ACTIVE_TOOLCHAIN.mt5_root
METAEDITOR = ACTIVE_TOOLCHAIN.metaeditor
TERMINAL = ACTIVE_TOOLCHAIN.terminal
WINE = ACTIVE_TOOLCHAIN.wine or (ROOT / ".fxai/wine-not-required")
COMMON_FILES = ACTIVE_TOOLCHAIN.common_files
RUNTIME_DIR = ACTIVE_TOOLCHAIN.runtime_dir
DEFAULT_DB_PATH = ACTIVE_TOOLCHAIN.default_db
DEFAULT_REPORT = COMMON_FILES / "FXAI/Audit/fxai_audit_report.tsv"
DEFAULT_TEXT_REPORT = ROOT / "Tools/latest_drill_report.md"
ORACLES_PATH = ROOT / "Tools/plugin_oracles.json"
BASELINES_DIR = ROOT / "Tools/Baselines"
EXPERIMENTS_DIR = ROOT / "Tools/Experiments"
TESTER_PRESET_DIR = ACTIVE_TOOLCHAIN.tester_preset_dir
COMMON_INI = ACTIVE_TOOLCHAIN.common_ini
TERMINAL_INI = ACTIVE_TOOLCHAIN.terminal_ini
MT5_LOG_DIR = ACTIVE_TOOLCHAIN.mt5_log_dir

ISSUE = {
    1: "invalid predictions",
    2: "overtrades noise",
    4: "misses trend",
    8: "calibration drift",
    16: "reset drift",
    32: "sequence contract weak",
    64: "dead move output",
    128: "side collapse",
    256: "walkforward overfit",
    512: "walkforward unstable",
    1024: "walkforward weak edge",
    2048: "macro-event blind",
    4096: "macro-event overreaction",
    8192: "macro-event data gap",
    16384: "adversarial audit weak",
}

EXECUTION_PROFILES = {
    "default": {
        "commission_per_lot_side": 0.0,
        "cost_buffer_points": 2.0,
        "slippage_points": 0.0,
        "fill_penalty_points": 0.0,
    },
    "tight-fx": {
        "commission_per_lot_side": 0.0,
        "cost_buffer_points": 1.5,
        "slippage_points": 0.1,
        "fill_penalty_points": 0.1,
    },
    "prime-ecn": {
        "commission_per_lot_side": 3.5,
        "cost_buffer_points": 1.5,
        "slippage_points": 0.2,
        "fill_penalty_points": 0.15,
    },
    "retail-fx": {
        "commission_per_lot_side": 0.0,
        "cost_buffer_points": 2.5,
        "slippage_points": 0.4,
        "fill_penalty_points": 0.25,
    },
    "stress": {
        "commission_per_lot_side": 5.0,
        "cost_buffer_points": 3.5,
        "slippage_points": 1.0,
        "fill_penalty_points": 0.5,
    },
}

SYMBOL_PACKS = {
    "majors": ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "NZDUSD", "USDCAD", "USDCHF"],
    "dollar-core": ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "USDCHF"],
    "yen-cross": ["USDJPY", "EURJPY", "GBPJPY", "AUDJPY", "CADJPY", "CHFJPY"],
    "commodity-fx": ["AUDUSD", "NZDUSD", "USDCAD", "XAUUSD", "XAGUSD"],
    "metals-risk": ["XAUUSD", "XAGUSD", "US500", "NAS100", "USDX"],
}


class AuditRunError(RuntimeError):
    pass


def active_toolchain() -> FXAIToolchainConfig:
    return ACTIVE_TOOLCHAIN


def toolchain_profile() -> str:
    return TOOLCHAIN_PROFILE


def supported_toolchain_profiles() -> tuple[str, ...]:
    return SUPPORTED_TOOLCHAIN_PROFILES


def toolchain_summary() -> dict[str, object]:
    return ACTIVE_TOOLCHAIN.describe()


def to_wine_path(path: Path) -> str:
    return ACTIVE_TOOLCHAIN.path_argument(path)


def build_metaeditor_compile_command(source_path: Path, log_path: Path) -> list[str]:
    return ACTIVE_TOOLCHAIN.metaeditor_compile_command(source_path, log_path)


def build_terminal_launch_command(*, config_path: Path | None = None, portable: bool = False) -> list[str]:
    return ACTIVE_TOOLCHAIN.terminal_launch_command(config_path=config_path, portable=portable)


def terminal_process_name() -> str:
    return TERMINAL.name or "terminal64.exe"


def terminal_running() -> bool:
    token = terminal_process_name()
    if ACTIVE_TOOLCHAIN.profile == "windows_native":
        try:
            proc = subprocess.run(
                ["tasklist", "/FI", f"IMAGENAME eq {token}"],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                check=False,
            )
            return proc.returncode == 0 and token.lower() in proc.stdout.lower()
        except Exception:
            return False
    proc = subprocess.run(
        ["pgrep", "-f", token],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        check=False,
    )
    return proc.returncode == 0 and bool(proc.stdout.strip())


def read_utf16_or_text(path: Path) -> str:
    if not path.exists():
        return ""
    for enc in ("utf-16le", "utf-8"):
        try:
            return path.read_text(encoding=enc)
        except Exception:
            pass
    return path.read_text(errors="replace")


def write_utf16(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-16le")


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def deep_merge(base, overlay):
    if not isinstance(base, dict) or not isinstance(overlay, dict):
        return overlay
    out = dict(base)
    for k, v in overlay.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def parse_brace_list(raw: str) -> list[str]:
    text = (raw or "").strip()
    text = text.replace("{", "").replace("}", "").replace(";", ",").replace("|", ",")
    items = []
    for part in text.split(","):
        token = part.strip()
        if token and token not in items:
            items.append(token)
    return items


def runtime_artifact_safe_symbol(symbol: str) -> str:
    clean = symbol or "UNKNOWN"
    for ch in "\\/:*?\"<>|":
        clean = clean.replace(ch, "_")
    return clean


def runtime_persistence_manifest_path(symbol: str) -> Path:
    return RUNTIME_DIR / f"fxai_persistence_{runtime_artifact_safe_symbol(symbol)}.tsv"


def runtime_feature_manifest_path(symbol: str) -> Path:
    return RUNTIME_DIR / f"fxai_features_{runtime_artifact_safe_symbol(symbol)}.tsv"


def runtime_macro_manifest_path(symbol: str) -> Path:
    return RUNTIME_DIR / f"fxai_macro_{runtime_artifact_safe_symbol(symbol)}.tsv"


def runtime_performance_manifest_path(symbol: str) -> Path:
    return RUNTIME_DIR / f"fxai_perf_{runtime_artifact_safe_symbol(symbol)}.tsv"


def mean_std_ci(values: list[float]) -> tuple[float, float, float]:
    if not values:
        return 0.0, 0.0, 0.0
    mean = sum(values) / len(values)
    if len(values) <= 1:
        return mean, 0.0, 0.0
    var = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    std = math.sqrt(max(var, 0.0))
    ci = 1.96 * std / math.sqrt(len(values))
    return mean, std, ci


def resolve_execution_profile(name: str | None) -> dict:
    profile = (name or "default").strip().lower()
    return dict(EXECUTION_PROFILES.get(profile, EXECUTION_PROFILES["default"]))


def clone_args(args):
    return argparse.Namespace(**vars(args))


def build_effective_audit_args(args):
    out = clone_args(args)
    out.strategy_profile = (getattr(args, "strategy_profile", None) or "default").strip() or "default"
    out.broker_profile = (getattr(args, "broker_profile", None) or "").strip()
    out.runtime_mode = (getattr(args, "runtime_mode", None) or "research").strip().lower() or "research"
    out.execution_profile = (getattr(args, "execution_profile", None) or "default").strip().lower()
    profile = resolve_execution_profile(out.execution_profile)
    if getattr(args, "commission_per_lot_side", None) is None:
        out.commission_per_lot_side = float(profile["commission_per_lot_side"])
    if getattr(args, "cost_buffer_points", None) is None:
        out.cost_buffer_points = float(profile["cost_buffer_points"])
    if getattr(args, "slippage_points", None) is None:
        out.slippage_points = float(profile["slippage_points"])
    if getattr(args, "fill_penalty_points", None) is None:
        out.fill_penalty_points = float(profile["fill_penalty_points"])
    if not getattr(args, "symbol_list", None):
        out.symbol_list = "{" + str(getattr(args, "symbol", "EURUSD")) + "}"
    out.strategy_profile_compiled = compile_strategy_profile(
        strategy_profile=out.strategy_profile,
        symbol=str(getattr(out, "symbol", "EURUSD") or "EURUSD"),
        broker_profile=out.broker_profile,
        server=getattr(out, "server", None),
        runtime_mode=out.runtime_mode,
        overrides={
            "bars": getattr(out, "bars", None),
            "horizon": getattr(out, "horizon", None),
            "m1sync_bars": getattr(out, "m1sync_bars", None),
            "normalization": getattr(out, "normalization", None),
            "sequence_bars": getattr(out, "sequence_bars", None),
            "schema_id": getattr(out, "schema_id", None),
            "feature_mask": getattr(out, "feature_mask", None),
            "commission_per_lot_side": getattr(out, "commission_per_lot_side", None),
            "cost_buffer_points": getattr(out, "cost_buffer_points", None),
            "slippage_points": getattr(out, "slippage_points", None),
            "fill_penalty_points": getattr(out, "fill_penalty_points", None),
            "wf_train_bars": getattr(out, "wf_train_bars", None),
            "wf_test_bars": getattr(out, "wf_test_bars", None),
            "wf_purge_bars": getattr(out, "wf_purge_bars", None),
            "wf_embargo_bars": getattr(out, "wf_embargo_bars", None),
            "wf_folds": getattr(out, "wf_folds", None),
            "seed": getattr(out, "seed", None),
            "window_start_unix": getattr(out, "window_start_unix", None),
            "window_end_unix": getattr(out, "window_end_unix", None),
            "execution_profile": getattr(out, "execution_profile", None),
            "scenario_list": parse_brace_list(getattr(out, "scenario_list", "")),
        },
        plugin_name="",
        ai_id=getattr(out, "plugin_id", 28),
    )
    return out


def resolve_symbol_list(args) -> list[str]:
    pack_name = (getattr(args, "symbol_pack", "") or "").strip().lower()
    if pack_name:
        return list(SYMBOL_PACKS.get(pack_name, [str(getattr(args, "symbol", "EURUSD"))]))
    symbols = parse_brace_list(getattr(args, "symbol_list", ""))
    if not symbols:
        symbols = [str(getattr(args, "symbol", "EURUSD"))]
    return symbols


def render_summary_report(summary: dict, execution_profile: str = "default", manifest_path: Path | None = None) -> str:
    if summary.get("_aggregate", {}).get("type") == "multi_symbol":
        from .reporting import render_multisymbol_report

        return render_multisymbol_report(summary, execution_profile, manifest_path)

    out = ["# FXAI Audit Summary", ""]
    out.append(f"execution_profile: {execution_profile}")
    if manifest_path is not None:
        out.append(f"manifest: {manifest_path}")
    out.append("")
    ranked = sorted(summary.get("plugins", {}).items(), key=lambda item: float(item[1].get("score", 0.0)), reverse=True)
    for name, info in ranked:
        out.append(f"## {name} | {float(info.get('score', 0.0)):.1f}/100 | Grade {info.get('grade', 'F')}")
        issues = list(info.get("issues", [])) + list(info.get("findings", []))
        if issues:
            out.append("Issues: " + "; ".join(issues[:8]))
        else:
            out.append("Issues: none flagged.")
        scenarios = sorted(info.get("scenarios", {}).items())
        if scenarios:
            top = []
            for scenario, metrics in scenarios[:6]:
                top.append(f"{scenario}={float(metrics.get('score', 0.0)):.1f}")
            out.append("Scenarios: " + ", ".join(top))
        out.append("")
    return "\n".join(out)


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def sha256_path(path: Path) -> str:
    if not path.exists():
        return ""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def git_head_commit(repo_root: Path) -> str:
    try:
        out = subprocess.check_output(["git", "-C", str(repo_root), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL)
        return out.strip()
    except Exception:
        return ""


def git_dirty(repo_root: Path) -> bool:
    try:
        out = subprocess.check_output(["git", "-C", str(repo_root), "status", "--porcelain"], text=True, stderr=subprocess.DEVNULL)
        return bool(out.strip())
    except Exception:
        return False


def load_oracles():
    return load_json(ORACLES_PATH)


def get_oracle(plugin_name: str, family: int, oracles: dict) -> dict:
    base = dict(oracles.get("global_defaults", {}))
    fam = oracles.get("family_defaults", {}).get(str(family), {})
    plug = oracles.get("plugins", {}).get(plugin_name, {})
    return deep_merge(deep_merge(base, fam), plug)
