#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path("/Users/andreborchert/Library/Application Support/net.metaquotes.wine.metatrader5/drive_c/Program Files/MetaTrader 5/MQL5/Experts/FXAI")
REPO_ROOT = Path(__file__).resolve().parents[2]
TERMINAL_ROOT = Path("/Users/andreborchert/Library/Application Support/net.metaquotes.wine.metatrader5/drive_c/Program Files/MetaTrader 5")
METAEDITOR = TERMINAL_ROOT / "MetaEditor64.exe"
TERMINAL = TERMINAL_ROOT / "terminal64.exe"
WINE = Path("/Applications/MetaTrader 5.app/Contents/SharedSupport/wine/bin/wine64")
COMMON_FILES = Path("/Users/andreborchert/Library/Application Support/net.metaquotes.wine.metatrader5/drive_c/users/andreborchert/AppData/Roaming/MetaQuotes/Terminal/Common/Files")
RUNTIME_DIR = COMMON_FILES / "FXAI/Runtime"
DEFAULT_REPORT = COMMON_FILES / "FXAI/Audit/fxai_audit_report.tsv"
DEFAULT_TEXT_REPORT = ROOT / "Tools/latest_drill_report.md"
ORACLES_PATH = ROOT / "Tools/plugin_oracles.json"
BASELINES_DIR = ROOT / "Tools/Baselines"
EXPERIMENTS_DIR = ROOT / "Tools/Experiments"
TESTER_PRESET_DIR = TERMINAL_ROOT / "MQL5/Profiles/Tester"
COMMON_INI = TERMINAL_ROOT / "config/common.ini"
TERMINAL_INI = TERMINAL_ROOT / "config/terminal.ini"
MT5_LOG_DIR = TERMINAL_ROOT / "logs"

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


def to_wine_path(path: Path) -> str:
    return "Z:\\" + str(path).replace("/", "\\").lstrip("\\")


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


def load_current_summary(path: Path, oracles: dict) -> dict:
    if path.suffix.lower() == ".tsv":
        return build_summary(load_rows(path), oracles)
    return load_json(path)


def resolve_execution_profile(name: str | None) -> dict:
    profile = (name or "default").strip().lower()
    return dict(EXECUTION_PROFILES.get(profile, EXECUTION_PROFILES["default"]))


def clone_args(args):
    return argparse.Namespace(**vars(args))


def build_effective_audit_args(args):
    out = clone_args(args)
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


def read_common_account() -> tuple[str, str]:
    text = read_utf16_or_text(COMMON_INI)
    login = ""
    server = ""
    for line in text.splitlines():
        if line.startswith("Login="):
            login = line.split("=", 1)[1].strip()
        elif line.startswith("Server="):
            server = line.split("=", 1)[1].strip()
    return login, server


def resolve_credentials(args) -> tuple[str, str, str]:
    common_login, common_server = read_common_account()
    login = args.login or os.environ.get("FXAI_MT5_LOGIN", "") or common_login
    server = args.server or os.environ.get("FXAI_MT5_SERVER", "") or common_server
    password = args.password or os.environ.get("FXAI_MT5_PASSWORD", "")
    return login, server, password


def terminal_running() -> bool:
    proc = subprocess.run(
        ["pgrep", "-f", "terminal64.exe"],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    return proc.returncode == 0 and bool(proc.stdout.strip())


def update_ini_section(path: Path, section_name: str, kv: dict[str, str], encoding: str = "utf-16le") -> None:
    text = read_utf16_or_text(path)
    marker = f"[{section_name}]"
    start = text.find(marker)
    lines = [f"{k}={v}" for k, v in kv.items()]
    new_section = marker + "\n" + "\n".join(lines) + "\n"
    if start < 0:
        if text and not text.endswith("\n"):
            text += "\n"
        text += new_section
    else:
        next_sec = text.find("\n[", start + 1)
        if next_sec < 0:
            next_sec = len(text)
        text = text[:start] + new_section + text[next_sec:]
    if encoding == "utf-16le":
        write_utf16(path, text)
    else:
        path.write_text(text, encoding=encoding)


def read_metaeditor_log(log_path: Path) -> str:
    return read_utf16_or_text(log_path)


def compile_target(relative_target: Path, stage_name: str) -> int:
    stage_dir = Path(tempfile.gettempdir()) / f"fxai_testlab_{stage_name}"
    if stage_dir.exists():
        shutil.rmtree(stage_dir)
    stage_dir.mkdir(parents=True, exist_ok=True)

    subprocess.run(["rsync", "-a", "--delete", f"{ROOT}/", f"{stage_dir}/"], check=True)

    stage_target = stage_dir / relative_target
    stage_log = stage_dir / f"compile_{relative_target.stem}.log"
    cmd = [
        str(WINE),
        str(METAEDITOR),
        f"/compile:{to_wine_path(stage_target)}",
        f"/log:{to_wine_path(stage_log)}",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    deadline = time.time() + 1200.0
    built_ex5 = stage_target.with_suffix(".ex5")
    live_ex5 = ROOT / relative_target.with_suffix(".ex5")
    last_log_text = ""

    while time.time() < deadline:
        rc = proc.poll()
        log_text = read_metaeditor_log(stage_log)
        if log_text:
            last_log_text = log_text

        if "0 errors, 0 warnings" in last_log_text and built_ex5.exists():
            if proc.poll() is None:
                proc.kill()
                proc.wait(timeout=5)
            live_ex5.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(built_ex5, live_ex5)
            lines = [line for line in last_log_text.splitlines() if line.strip()]
            if lines:
                print(lines[-1])
            return 0

        if rc is not None:
            if last_log_text:
                lines = [line for line in last_log_text.splitlines() if line.strip()]
                if lines:
                    print(lines[-1])
            if "0 errors, 0 warnings" in last_log_text and built_ex5.exists():
                live_ex5.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(built_ex5, live_ex5)
                return 0
            if proc.stdout is not None:
                sys.stdout.write(proc.stdout.read())
            return rc or 1

        time.sleep(2.0)

    if proc.poll() is None:
        proc.kill()
        proc.wait(timeout=5)
    if last_log_text:
        lines = [line for line in last_log_text.splitlines() if line.strip()]
        if lines:
            print(lines[-1])
    return 124


def load_rows(report: Path):
    with report.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def load_tsv_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def manifest_path(raw_path: str) -> Path | None:
    text = (raw_path or "").strip()
    if not text:
        return None
    return Path(text)


def parse_issue_flags(v: str):
    flags = int(v)
    return [name for bit, name in ISSUE.items() if flags & bit]


def fnum(row, key):
    try:
        return float(row[key])
    except Exception:
        return 0.0


def inum(row, key):
    try:
        return int(float(row[key]))
    except Exception:
        return 0


def aggregate(rows):
    by_plugin = defaultdict(list)
    for row in rows:
        by_plugin[row["ai_name"]].append(row)
    return by_plugin


def plugin_family(rows):
    return int(rows[0]["family"]) if rows else 11


def scenario_row(rows, name):
    for r in rows:
        if r["scenario"] == name:
            return r
    return None


def ratio(num: float, den: float) -> float:
    if den <= 0.0:
        return 0.0
    return num / den


def scenario_metric(row, metric: str) -> float:
    if row is None:
        return 0.0
    if metric in row:
        return fnum(row, metric)
    if metric == "directional_hit_ratio":
        return ratio(inum(row, "directional_correct_count"), inum(row, "directional_eval_count"))
    if metric == "exact_match_ratio":
        return ratio(inum(row, "exact_match_count"), inum(row, "samples_total"))
    return 0.0


def evaluate_oracle(plugin_name: str, family: int, rows, oracle: dict):
    findings = []
    penalties = 0.0

    reset_limit = float(oracle.get("reset_delta_max", 0.30))
    sequence_min = float(oracle.get("sequence_delta_min", 0.005))
    avg_move_min = float(oracle.get("avg_move_min", 0.10))

    avg_move = 0.0
    if rows:
        avg_move = sum(fnum(r, "avg_move") for r in rows) / len(rows)
    if avg_move < avg_move_min:
        findings.append(f"dead move amplitude: avg_move {avg_move:.3f} < required {avg_move_min:.3f}")
        penalties += 8.0

    reset_candidates = [fnum(r, "reset_delta") for r in rows if fnum(r, "reset_delta") >= 0.0]
    if reset_candidates:
        reset_max = max(reset_candidates)
        if reset_max > reset_limit:
            findings.append(f"reset drift too high: {reset_max:.3f} > {reset_limit:.3f}")
            penalties += 10.0

    sequence_candidates = [fnum(r, "sequence_delta") for r in rows if fnum(r, "sequence_delta") >= 0.0]
    if sequence_candidates:
        seq_max = max(sequence_candidates)
        if seq_max < sequence_min:
            findings.append(f"sequence response too weak: {seq_max:.3f} < {sequence_min:.3f}")
            penalties += 8.0

    scenario_rules = oracle.get("scenario_rules", {})
    for scenario, rules in scenario_rules.items():
        row = scenario_row(rows, scenario)
        if row is None:
            findings.append(f"missing required scenario: {scenario}")
            penalties += 6.0
            continue
        for metric, threshold in rules.items():
            value = scenario_metric(row, metric)
            if metric.endswith("_min"):
                base_metric = metric[:-4]
                value = scenario_metric(row, base_metric)
                if value < float(threshold):
                    findings.append(f"{scenario} {base_metric} too low: {value:.3f} < {float(threshold):.3f}")
                    penalties += 5.0
            elif metric.endswith("_max"):
                base_metric = metric[:-4]
                value = scenario_metric(row, base_metric)
                if value > float(threshold):
                    findings.append(f"{scenario} {base_metric} too high: {value:.3f} > {float(threshold):.3f}")
                    penalties += 5.0

    return findings, penalties


def score_plugin(rows, oracle):
    scores = [fnum(r, "score") for r in rows]
    if not scores:
        return 0.0
    avg = sum(scores) / len(scores)
    invalid = sum(inum(r, "invalid_preds") for r in rows)
    findings, penalties = evaluate_oracle(rows[0]["ai_name"], plugin_family(rows), rows, oracle)
    if invalid > 0:
        penalties += 20.0
    worst = min(scores)
    penalties += max(0.0, 70.0 - worst) * 0.35
    return max(0.0, avg - penalties)


def grade(score):
    if score >= 92:
        return "A"
    if score >= 84:
        return "B"
    if score >= 74:
        return "C"
    if score >= 64:
        return "D"
    return "F"


def build_suggestions(name, family, rows, oracle, findings):
    suggestions = []
    family_note = oracle.get("family_note")
    if family_note:
        suggestions.append(family_note)
    identity = oracle.get("identity")
    if identity:
        suggestions.append(identity)

    issue_names = sorted({issue for r in rows for issue in parse_issue_flags(r["issue_flags"])})
    if "invalid predictions" in issue_names:
        suggestions.append("Fix numerical stability first: clamp logits, normalize probability sums, and harden move-head outputs against NaN/Inf.")
    if "overtrades noise" in issue_names:
        suggestions.append("Raise discipline on random walk: stronger skip behavior, harsher low-edge rejection, and better uncertainty gating are required.")
    if "misses trend" in issue_names:
        suggestions.append("Trend response is too weak. Strengthen horizon-conditioned trend learning and make directional heads react faster to clean drift regimes.")
    if "calibration drift" in issue_names:
        suggestions.append("Calibration is lying. Tighten context-bank updates and add stronger post-hoc calibration or reliability damping under drift.")
    if "reset drift" in issue_names:
        suggestions.append("Reset behavior is not clean enough. Hidden state, replay state, or calibration state is surviving when it should be zeroed.")
    if "sequence contract weak" in issue_names:
        suggestions.append("The declared sequence/window contract is too weak. The model must prove that longer context changes outputs in meaningful ways.")
    if "dead move output" in issue_names:
        suggestions.append("The expected-move head is too flat. It needs stronger amplitude learning and clearer separation between edge and noise.")
    if "side collapse" in issue_names:
        suggestions.append("The plugin is collapsing to one side. Rebalance the class objective and strengthen skip when evidence is symmetric.")
    if "walkforward overfit" in issue_names:
        suggestions.append("The walk-forward gap is too wide. Tighten promotion gates, reduce model variance, and prefer configurations that hold up after purge and embargo.")
    if "walkforward unstable" in issue_names:
        suggestions.append("Walk-forward stability is weak. Improve fold consistency, reduce sensitivity to regime edges, and harden the model against execution and spread shifts.")
    if "walkforward weak edge" in issue_names:
        suggestions.append("Out-of-sample edge is not robust enough. The live candidate needs stronger post-cost edge or more aggressive rejection of marginal trades.")

    suggestions.extend(findings)
    suggestions.extend(oracle.get("recommendations", []))

    seen = []
    for item in suggestions:
        if item and item not in seen:
            seen.append(item)
    return seen[:8]


def build_summary(rows, oracles: dict):
    groups = aggregate(rows)
    summary = {"plugins": {}}
    for name, items in groups.items():
        family = plugin_family(items)
        oracle = get_oracle(name, family, oracles)
        findings, _ = evaluate_oracle(name, family, items, oracle)
        score = score_plugin(items, oracle)
        issues = sorted({issue for r in items for issue in parse_issue_flags(r["issue_flags"])})
        scenarios = {}
        for row in items:
            scenarios[row["scenario"]] = {
                "score": fnum(row, "score"),
                "skip_ratio": fnum(row, "skip_ratio"),
                "active_ratio": fnum(row, "active_ratio"),
                "bias_abs": fnum(row, "bias_abs"),
                "conf_drift": fnum(row, "conf_drift"),
                "brier_score": fnum(row, "brier_score"),
                "calibration_error": fnum(row, "calibration_error"),
                "path_quality_error": fnum(row, "path_quality_error"),
                "reset_delta": fnum(row, "reset_delta"),
                "sequence_delta": fnum(row, "sequence_delta"),
                "wf_folds": inum(row, "wf_folds"),
                "wf_train_score": fnum(row, "wf_train_score"),
                "wf_test_score": fnum(row, "wf_test_score"),
                "wf_test_score_std": fnum(row, "wf_test_score_std"),
                "wf_gap": fnum(row, "wf_gap"),
                "wf_pbo": fnum(row, "wf_pbo"),
                "wf_dsr": fnum(row, "wf_dsr"),
                "wf_pass_rate": fnum(row, "wf_pass_rate"),
                "avg_move": fnum(row, "avg_move"),
                "trend_align": fnum(row, "trend_align"),
                "invalid_preds": inum(row, "invalid_preds"),
                "issue_flags": inum(row, "issue_flags")
            }
        summary["plugins"][name] = {
            "family": family,
            "score": score,
            "grade": grade(score),
            "issues": issues,
            "findings": findings,
            "scenarios": scenarios,
        }
    return summary


def build_multisymbol_summary(symbol_runs: list[dict]) -> dict:
    summary = {
        "_aggregate": {
            "type": "multi_symbol",
            "symbols": [run["symbol"] for run in symbol_runs],
            "run_count": len(symbol_runs),
        },
        "plugins": {},
    }
    plugin_names = sorted({
        name
        for run in symbol_runs
        for name in run.get("summary", {}).get("plugins", {}).keys()
    })

    for name in plugin_names:
        per_symbol = []
        issues = set()
        findings = set()
        family = 11
        scenario_values: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
        for run in symbol_runs:
            info = run.get("summary", {}).get("plugins", {}).get(name)
            if not info:
                continue
            family = int(info.get("family", family))
            issues.update(info.get("issues", []))
            findings.update(info.get("findings", []))
            per_symbol.append((run["symbol"], float(info.get("score", 0.0))))
            for scenario, metrics in info.get("scenarios", {}).items():
                for key, value in metrics.items():
                    if isinstance(value, (int, float)):
                        scenario_values[scenario][key].append(float(value))

        scores = [score for _, score in per_symbol]
        mean_score, std_score, ci_score = mean_std_ci(scores)
        stability = 1.0
        if mean_score > 1e-9:
            stability = max(0.0, 1.0 - min(std_score / max(mean_score, 10.0), 1.0))

        scenarios = {}
        for scenario, metric_map in scenario_values.items():
            metrics_out = {}
            for key, vals in metric_map.items():
                mean_v, std_v, ci_v = mean_std_ci(vals)
                metrics_out[key] = mean_v
                metrics_out[f"{key}_std"] = std_v
                metrics_out[f"{key}_ci95"] = ci_v
            scenarios[scenario] = metrics_out

        summary["plugins"][name] = {
            "family": family,
            "score": mean_score,
            "score_std": std_score,
            "score_ci95": ci_score,
            "stability": stability,
            "n_symbols": len(per_symbol),
            "grade": grade(mean_score),
            "issues": sorted(issues),
            "findings": sorted(findings),
            "symbol_scores": {symbol: score for symbol, score in per_symbol},
            "scenarios": scenarios,
        }
    return summary


def summary_has_market_replay(summary: dict) -> bool:
    required = {"market_recent", "market_trend", "market_chop", "market_session_edges", "market_spread_shock", "market_walkforward", "market_macro_event", "market_adversarial"}
    seen = set()
    for plugin in summary.get("plugins", {}).values():
        seen.update(plugin.get("scenarios", {}).keys())
    return required.issubset(seen)


def render_multisymbol_report(summary: dict, execution_profile: str, manifest_path: Path | None = None) -> str:
    out = ["# FXAI Statistical Audit Certification", ""]
    agg = summary.get("_aggregate", {})
    symbols = agg.get("symbols", [])
    out.append(f"symbols: {', '.join(symbols)}")
    out.append(f"execution_profile: {execution_profile}")
    if manifest_path is not None:
        out.append(f"manifest: {manifest_path}")
    out.append("")
    out.append("This report aggregates per-symbol audit runs into a portfolio-style certification view with cross-symbol stability and confidence intervals.")
    out.append("")

    ranked = sorted(summary.get("plugins", {}).items(), key=lambda item: float(item[1].get("score", 0.0)), reverse=True)
    for name, info in ranked:
        out.append(
            f"## {name} | {float(info.get('score', 0.0)):.1f}/100 | "
            f"CI95 ±{float(info.get('score_ci95', 0.0)):.2f} | "
            f"Stability {float(info.get('stability', 0.0)):.2f} | "
            f"Grade {info.get('grade', 'F')}"
        )
        recent = info.get("scenarios", {}).get("market_recent", {})
        if recent:
            out.append(
                "Market recent: "
                f"Brier {float(recent.get('brier_score', 0.0)):.3f} | "
                f"CalErr {float(recent.get('calibration_error', 0.0)):.3f} | "
                f"PathErr {float(recent.get('path_quality_error', 0.0)):.3f}"
            )
        wf = info.get("scenarios", {}).get("market_walkforward", {})
        if wf:
            out.append(
                "Walkforward: "
                f"Test {float(wf.get('wf_test_score', 0.0)):.1f} | "
                f"PBO {float(wf.get('wf_pbo', 0.0)):.2f} | "
                f"DSR(proxy) {float(wf.get('wf_dsr', 0.0)):.2f} | "
                f"Pass {float(wf.get('wf_pass_rate', 0.0)):.2f}"
            )
        macro = info.get("scenarios", {}).get("market_macro_event", {})
        if macro:
            out.append(
                "Macro events: "
                f"Score {float(macro.get('score', 0.0)):.1f} | "
                f"Coverage {float(macro.get('macro_data_coverage', 0.0)):.2f} | "
                f"EventRate {float(macro.get('macro_event_rate', 0.0)):.2f} | "
                f"Imp {float(macro.get('macro_importance_mean', 0.0)):.2f}"
            )
        adversarial = info.get("scenarios", {}).get("market_adversarial", {})
        if adversarial:
            out.append(
                "Adversarial: "
                f"Score {float(adversarial.get('score', 0.0)):.1f} | "
                f"Brier {float(adversarial.get('brier_score', 0.0)):.3f} | "
                f"CalErr {float(adversarial.get('calibration_error', 0.0)):.3f} | "
                f"PathErr {float(adversarial.get('path_quality_error', 0.0)):.3f}"
            )
        issues = list(info.get("issues", [])) + list(info.get("findings", []))
        if issues:
            out.append("Issues: " + "; ".join(issues[:8]))
        else:
            out.append("Issues: none flagged across the aggregated pack.")
        symbol_scores = info.get("symbol_scores", {})
        if symbol_scores:
            score_line = ", ".join(f"{sym}={score:.1f}" for sym, score in sorted(symbol_scores.items()))
            out.append("Per-symbol scores: " + score_line)
        out.append("")
    return "\n".join(out)


def render_report(rows, oracles: dict):
    groups = aggregate(rows)
    ranked = []
    for name, items in groups.items():
        fam = plugin_family(items)
        oracle = get_oracle(name, fam, oracles)
        ranked.append((score_plugin(items, oracle), name, items, fam, oracle))
    ranked.sort(reverse=True)

    out = []
    out.append("# FXAI Test Lab Drill Report")
    out.append("")
    out.append("This report is intentionally harsh. A plugin only earns credit when it behaves like its claimed model family under synthetic pressure and real-market replay certification.")
    out.append("")
    for score, name, items, fam, oracle in ranked:
        issues = sorted({issue for r in items for issue in parse_issue_flags(r["issue_flags"])})
        findings, _ = evaluate_oracle(name, fam, items, oracle)
        out.append(f"## {name} | {score:.1f}/100 | Grade {grade(score)}")
        out.append("")
        if issues or findings:
            all_issues = issues + findings
            out.append("Issues: " + "; ".join(all_issues))
        else:
            out.append("Issues: none flagged by the drill harness.")
        out.append("")
        out.append("Required improvements:")
        for s in build_suggestions(name, fam, items, oracle, findings):
            out.append(f"- {s}")
        out.append("")
    return "\n".join(out)


def build_optimization_campaign(summary: dict, oracles: dict) -> dict:
    campaign = {"plugins": {}}
    for name, info in sorted(summary.get("plugins", {}).items()):
        family = int(info.get("family", 11))
        oracle = get_oracle(name, family, oracles)
        issues = set(info.get("issues", [])) | set(info.get("findings", []))
        experiments = []

        schema_candidates = []
        norm_candidates = []
        seq_candidates = []
        scenario_focus = ["random_walk", "drift_up", "drift_down", "market_trend", "market_chop", "market_session_edges", "market_spread_shock", "market_walkforward", "market_macro_event", "market_adversarial"]

        if family in (2, 3, 4, 5):  # recurrent/conv/transformer/state-space
            schema_candidates.extend([3, 6])
            seq_candidates.extend([8, 16, 32, 64])
            norm_candidates.extend([0, 7, 8, 9, 13, 14])
        elif family in (0, 6):  # linear/distributional
            schema_candidates.extend([2, 6])
            seq_candidates.extend([1, 4, 8])
            norm_candidates.extend([0, 7, 8, 9, 10, 11])
        elif family == 1:  # tree
            schema_candidates.extend([5, 6, 1])
            seq_candidates.extend([1, 4, 8])
            norm_candidates.extend([0, 7, 8, 9, 10])
        elif family == 10:  # rule-based
            schema_candidates.extend([4, 1])
            seq_candidates.extend([1, 2, 3, 4])
            norm_candidates.extend([0])
        else:
            schema_candidates.extend([1, 6, 3])
            seq_candidates.extend([1, 8, 16])
            norm_candidates.extend([0, 7, 8, 9, 10])

        if any("overtrades noise" in x for x in issues):
            norm_candidates.extend([7, 8, 9, 10])
            schema_candidates.extend([2, 4])
        if any("misses trend" in x for x in issues):
            seq_candidates.extend([16, 32, 64])
            schema_candidates.extend([3, 6])
        if any("calibration drift" in x for x in issues):
            norm_candidates.extend([9, 10, 13, 14])
        if any("sequence contract weak" in x for x in issues):
            seq_candidates.extend([16, 32, 64, 96])
        if any("dead move output" in x for x in issues):
            schema_candidates.extend([6, 1])
            norm_candidates.extend([7, 8, 9])

        def uniq(seq):
            seen = set()
            out = []
            for item in seq:
                if item in seen:
                    continue
                seen.add(item)
                out.append(item)
            return out

        experiments.append({
            "name": "schema_ablation",
            "schemas": uniq(schema_candidates)[:4],
            "focus": scenario_focus,
        })
        experiments.append({
            "name": "normalization_sweep",
            "normalizations": uniq(norm_candidates)[:6],
            "focus": scenario_focus,
        })
        experiments.append({
            "name": "sequence_sweep",
            "sequence_bars": uniq(seq_candidates)[:6],
            "focus": scenario_focus,
        })
        feature_masks = []
        if family == 10:
            feature_masks.extend([0x29, 0x21])
        elif family in (0, 1, 6):
            feature_masks.extend([0x37, 0x3F, 0x77])
        else:
            feature_masks.extend([0x7F, 0x3F, 0x5F])
        experiments.append({
            "name": "feature_mask_ablation",
            "feature_masks": uniq(feature_masks)[:4],
            "focus": scenario_focus,
        })
        experiments.append({
            "name": "market_replay_cert",
            "focus": ["market_recent", "market_trend", "market_chop", "market_session_edges", "market_spread_shock", "market_walkforward", "market_macro_event", "market_adversarial"],
        })
        experiments.append({
            "name": "execution_sweep",
            "slippage_points": [0.0, 0.5, 1.0],
            "fill_penalty_points": [0.0, 0.25, 0.50],
            "focus": ["market_recent", "market_session_edges", "market_spread_shock", "market_walkforward", "market_macro_event", "market_adversarial"],
        })
        experiments.append({
            "name": "walkforward_gate",
            "focus": ["market_walkforward", "market_session_edges", "market_spread_shock", "market_macro_event", "market_adversarial"],
            "train_test_pairs": [(256, 64), (384, 96), (512, 128)],
        })

        campaign["plugins"][name] = {
            "score": float(info.get("score", 0.0)),
            "grade": info.get("grade", "F"),
            "issues": sorted(issues),
            "oracle_identity": oracle.get("identity", ""),
            "experiments": experiments,
        }
    return campaign


def render_optimization_campaign(campaign: dict) -> str:
    out = ["# FXAI Optimization Campaign", ""]
    out.append("This plan is generated from the latest audit summary. It proposes targeted schema, normalization, sequence, feature-mask, and market-replay certification sweeps instead of blind brute force.")
    out.append("")
    for name, info in sorted(campaign.get("plugins", {}).items()):
        out.append(f"## {name} | {info.get('score', 0.0):.1f}/100 | Grade {info.get('grade', 'F')}")
        issues = info.get("issues", [])
        if issues:
            out.append("Issues: " + "; ".join(issues))
        identity = info.get("oracle_identity", "")
        if identity:
            out.append(identity)
        out.append("")
        for exp in info.get("experiments", []):
            if exp["name"] == "schema_ablation":
                out.append(f"- Schema sweep: {exp['schemas']} | focus={exp['focus']}")
                for schema in exp['schemas']:
                    out.append(f"  run: run-audit --plugin-list '{{{name}}}' --scenario-list '{{{', '.join(exp['focus'])}}}' --schema-id {schema}")
            elif exp["name"] == "normalization_sweep":
                out.append(f"- Normalization sweep: {exp['normalizations']} | focus={exp['focus']}")
                for norm in exp['normalizations']:
                    out.append(f"  run: run-audit --plugin-list '{{{name}}}' --scenario-list '{{{', '.join(exp['focus'])}}}' --normalization {norm}")
            elif exp["name"] == "sequence_sweep":
                out.append(f"- Sequence sweep: {exp['sequence_bars']} | focus={exp['focus']}")
                for seq in exp['sequence_bars']:
                    out.append(f"  run: run-audit --plugin-list '{{{name}}}' --scenario-list '{{{', '.join(exp['focus'])}}}' --sequence-bars {seq}")
            elif exp["name"] == "feature_mask_ablation":
                out.append(f"- Feature-mask sweep: {exp['feature_masks']} | focus={exp['focus']}")
                for mask in exp['feature_masks']:
                    out.append(f"  run: run-audit --plugin-list '{{{name}}}' --scenario-list '{{{', '.join(exp['focus'])}}}' --feature-mask {mask}")
            elif exp["name"] == "market_replay_cert":
                out.append(f"- Market replay certification: {exp['focus']}")
                out.append(f"  run: run-audit --plugin-list '{{{name}}}' --scenario-list '{{{', '.join(exp['focus'])}}}'")
            elif exp["name"] == "execution_sweep":
                out.append(f"- Execution sweep: slippage={exp['slippage_points']} fill_penalty={exp['fill_penalty_points']} | focus={exp['focus']}")
                for slip in exp["slippage_points"]:
                    for fillp in exp["fill_penalty_points"]:
                        out.append(f"  run: run-audit --plugin-list '{{{name}}}' --scenario-list '{{{', '.join(exp['focus'])}}}' --slippage-points {slip} --fill-penalty-points {fillp}")
            elif exp["name"] == "walkforward_gate":
                out.append(f"- Walk-forward release gate: {exp['focus']} | windows={exp['train_test_pairs']}")
                for train_bars, test_bars in exp["train_test_pairs"]:
                    out.append(f"  run: run-audit --plugin-list '{{{name}}}' --scenario-list '{{{', '.join(exp['focus'])}}}' --wf-train-bars {train_bars} --wf-test-bars {test_bars}")
                out.append("  run: release-gate --baseline <baseline-name> --require-market-replay")
        out.append("")
    return "\n".join(out)


def campaign_runs(campaign: dict, limit_plugins: int | None = None, limit_experiments: int | None = None) -> list[dict]:
    runs: list[dict] = []
    plugin_items = sorted(campaign.get("plugins", {}).items())
    if limit_plugins and limit_plugins > 0:
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
            else:
                runs.append({"plugin": name, "experiment": exp["name"], "scenario_list": focus})

            exp_count += 1
            if limit_experiments and limit_experiments > 0 and exp_count >= limit_experiments:
                break
    return runs


def build_run_audit_namespace(base_args, run: dict, output_path: Path):
    symbol_list = run.get("symbol_list", getattr(base_args, "symbol_list", ""))
    symbols = parse_brace_list(symbol_list)
    symbol = run.get("symbol", (symbols[0] if symbols else getattr(base_args, "symbol", "EURUSD")))
    return argparse.Namespace(
        all_plugins=False,
        plugin_id=28,
        plugin_list=f"{{{run['plugin']}}}",
        scenario_list="{" + ", ".join(run.get("scenario_list", [])) + "}",
        bars=base_args.bars,
        horizon=base_args.horizon,
        m1sync_bars=base_args.m1sync_bars,
        normalization=run.get("normalization", base_args.normalization),
        sequence_bars=run.get("sequence_bars", base_args.sequence_bars),
        schema_id=run.get("schema_id", base_args.schema_id),
        feature_mask=run.get("feature_mask", base_args.feature_mask),
        commission_per_lot_side=run.get("commission_per_lot_side", base_args.commission_per_lot_side),
        cost_buffer_points=run.get("cost_buffer_points", base_args.cost_buffer_points),
        slippage_points=run.get("slippage_points", base_args.slippage_points),
        fill_penalty_points=run.get("fill_penalty_points", base_args.fill_penalty_points),
        wf_train_bars=run.get("wf_train_bars", base_args.wf_train_bars),
        wf_test_bars=run.get("wf_test_bars", base_args.wf_test_bars),
        seed=base_args.seed,
        symbol=symbol,
        symbol_list=("{" + ", ".join(symbols) + "}" if symbols else "{" + symbol + "}"),
        execution_profile=getattr(base_args, "execution_profile", "default"),
        login=base_args.login,
        server=base_args.server,
        password=base_args.password,
        timeout=base_args.timeout,
        baseline=None,
        output=str(output_path),
        compare_output=None,
    )


def execute_optimization_campaign(campaign: dict, args) -> int:
    args = build_effective_audit_args(args)
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir) if args.output_dir else (EXPERIMENTS_DIR / ts)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "campaign.json").write_text(json.dumps(campaign, indent=2, sort_keys=True), encoding="utf-8")

    runs = campaign_runs(campaign, args.limit_plugins, args.limit_experiments)
    results = []
    for idx, run in enumerate(runs, start=1):
        stem = f"{idx:03d}_{run['plugin']}_{run['experiment']}"
        report_path = out_dir / f"{stem}.md"
        run_args = build_run_audit_namespace(args, run, report_path)
        rc = cmd_run_audit(run_args)
        result = {
            "index": idx,
            "plugin": run["plugin"],
            "experiment": run["experiment"],
            "parameters": run,
            "status": ("ok" if rc == 0 else "failed"),
            "report": str(report_path),
        }
        summary_path = report_path.with_suffix(".summary.json")
        result["summary_path"] = str(summary_path)
        if rc == 0 and summary_path.exists():
            oracles = load_oracles()
            summary = load_current_summary(summary_path, oracles)
            result["summary"] = summary.get("plugins", {}).get(run["plugin"], {})
        results.append(result)
        (out_dir / "results.json").write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
        if rc != 0:
            # Keep running, but make failures explicit in the ledger.
            continue

    print(f"\n# FXAI Optimization Execution Ledger\n")
    print(f"output_dir: {out_dir}")
    print(f"runs: {len(results)}")
    failed = sum(1 for item in results if item["status"] != "ok")
    print(f"failed: {failed}")
    return 0 if failed == 0 else 1


def resolve_baseline_path(name_or_path: str) -> Path:
    candidate = Path(name_or_path)
    if candidate.exists():
        return candidate
    return BASELINES_DIR / f"{name_or_path}.summary.json"


def load_baseline_summary(path: Path, oracles: dict) -> dict:
    if path.suffix.lower() == ".tsv":
        return build_summary(load_rows(path), oracles)
    return load_json(path)


def compare_summaries(current: dict, baseline: dict) -> str:
    data = compare_summary_data(current, baseline)
    out = ["# FXAI Baseline Comparison", ""]
    regressions = data["regressions"]
    improvements = data["improvements"]

    if regressions:
        out.append("## Regressions")
        out.extend(f"- {item}" for item in regressions)
        out.append("")
    else:
        out.append("## Regressions")
        out.append("- none")
        out.append("")

    if improvements:
        out.append("## Improvements")
        out.extend(f"- {item}" for item in improvements)
        out.append("")
    else:
        out.append("## Improvements")
        out.append("- none")
        out.append("")

    return "\n".join(out)


def compare_summary_data(current: dict, baseline: dict) -> dict:
    current_plugins = current.get("plugins", {})
    base_plugins = baseline.get("plugins", {})
    regressions = []
    improvements = []

    for name, cur in sorted(current_plugins.items()):
        base = base_plugins.get(name)
        if base is None:
            improvements.append(f"{name}: new plugin in current audit set.")
            continue

        cur_score = float(cur.get("score", 0.0))
        base_score = float(base.get("score", 0.0))
        delta = cur_score - base_score
        cur_ci = float(cur.get("score_ci95", 0.0))
        base_ci = float(base.get("score_ci95", 0.0))
        cur_issues = set(cur.get("issues", [])) | set(cur.get("findings", []))
        base_issues = set(base.get("issues", [])) | set(base.get("findings", []))
        new_issues = sorted(cur_issues - base_issues)
        fixed_issues = sorted(base_issues - cur_issues)

        plugin_notes = []
        if delta <= -(2.0 + cur_ci + base_ci):
            plugin_notes.append(f"statistically significant score down {delta:.1f}")
        elif delta <= -2.0:
            plugin_notes.append(f"score down {delta:.1f}")
        elif delta >= (2.0 + cur_ci + base_ci):
            plugin_notes.append(f"statistically significant score up +{delta:.1f}")
        elif delta >= 2.0:
            plugin_notes.append(f"score up +{delta:.1f}")
        if new_issues:
            plugin_notes.append("new issues: " + "; ".join(new_issues[:4]))
        if fixed_issues:
            plugin_notes.append("fixed issues: " + "; ".join(fixed_issues[:4]))
        if "stability" in cur and float(cur.get("stability", 1.0)) < 0.55:
            plugin_notes.append(f"cross-symbol stability weak {float(cur.get('stability', 0.0)):.2f}")
        if "score_std" in cur and "score_std" in base:
            std_delta = float(cur.get("score_std", 0.0)) - float(base.get("score_std", 0.0))
            if std_delta >= 1.50:
                plugin_notes.append(f"cross-symbol dispersion worse +{std_delta:.2f}")

        cur_rw = cur.get("scenarios", {}).get("random_walk", {})
        base_rw = base.get("scenarios", {}).get("random_walk", {})
        if cur_rw and base_rw:
            skip_delta = float(cur_rw.get("skip_ratio", 0.0)) - float(base_rw.get("skip_ratio", 0.0))
            if skip_delta <= -0.05:
                plugin_notes.append(f"random_walk skip down {skip_delta:.3f}")
            elif skip_delta >= 0.05:
                plugin_notes.append(f"random_walk skip up +{skip_delta:.3f}")

        cur_du = cur.get("scenarios", {}).get("drift_up", {})
        base_du = base.get("scenarios", {}).get("drift_up", {})
        if cur_du and base_du:
            trend_delta = float(cur_du.get("trend_align", 0.0)) - float(base_du.get("trend_align", 0.0))
            if trend_delta <= -0.08:
                plugin_notes.append(f"drift_up align down {trend_delta:.3f}")
            elif trend_delta >= 0.08:
                plugin_notes.append(f"drift_up align up +{trend_delta:.3f}")

        cur_dd = cur.get("scenarios", {}).get("drift_down", {})
        base_dd = base.get("scenarios", {}).get("drift_down", {})
        if cur_dd and base_dd:
            trend_delta = float(cur_dd.get("trend_align", 0.0)) - float(base_dd.get("trend_align", 0.0))
            if trend_delta <= -0.08:
                plugin_notes.append(f"drift_down align down {trend_delta:.3f}")
            elif trend_delta >= 0.08:
                plugin_notes.append(f"drift_down align up +{trend_delta:.3f}")

        cur_recent = cur.get("scenarios", {}).get("market_recent", {})
        base_recent = base.get("scenarios", {}).get("market_recent", {})
        if cur_recent and base_recent:
            brier_delta = float(cur_recent.get("brier_score", 0.0)) - float(base_recent.get("brier_score", 0.0))
            cal_delta = float(cur_recent.get("calibration_error", 0.0)) - float(base_recent.get("calibration_error", 0.0))
            pq_delta = float(cur_recent.get("path_quality_error", 0.0)) - float(base_recent.get("path_quality_error", 0.0))
            if brier_delta >= 0.04:
                plugin_notes.append(f"market_recent brier worse +{brier_delta:.3f}")
            if cal_delta >= 0.04:
                plugin_notes.append(f"market_recent calibration worse +{cal_delta:.3f}")
            if pq_delta >= 0.05:
                plugin_notes.append(f"market_recent path-quality worse +{pq_delta:.3f}")
            if float(cur_recent.get("brier_score_ci95", 0.0)) >= float(base_recent.get("brier_score_ci95", 0.0)) + 0.02:
                plugin_notes.append("market_recent brier stability weaker")

        cur_wf = cur.get("scenarios", {}).get("market_walkforward", {})
        base_wf = base.get("scenarios", {}).get("market_walkforward", {})
        if cur_wf and base_wf:
            wf_delta = float(cur_wf.get("score", 0.0)) - float(base_wf.get("score", 0.0))
            if wf_delta <= -3.0:
                plugin_notes.append(f"walkforward score down {wf_delta:.1f}")
            if float(cur_wf.get("score_ci95", 0.0)) >= float(base_wf.get("score_ci95", 0.0)) + 1.0:
                plugin_notes.append("walkforward stability weaker")
            pbo_delta = float(cur_wf.get("wf_pbo", 0.0)) - float(base_wf.get("wf_pbo", 0.0))
            dsr_delta = float(cur_wf.get("wf_dsr", 0.0)) - float(base_wf.get("wf_dsr", 0.0))
            pass_delta = float(cur_wf.get("wf_pass_rate", 0.0)) - float(base_wf.get("wf_pass_rate", 0.0))
            if pbo_delta >= 0.08:
                plugin_notes.append(f"walkforward PBO worse +{pbo_delta:.2f}")
            if dsr_delta <= -0.08:
                plugin_notes.append(f"walkforward DSR weaker {dsr_delta:.2f}")
            if pass_delta <= -0.08:
                plugin_notes.append(f"walkforward pass-rate down {pass_delta:.2f}")

        cur_macro = cur.get("scenarios", {}).get("market_macro_event", {})
        base_macro = base.get("scenarios", {}).get("market_macro_event", {})
        if cur_macro and base_macro:
            macro_score_delta = float(cur_macro.get("score", 0.0)) - float(base_macro.get("score", 0.0))
            macro_cov_delta = float(cur_macro.get("macro_data_coverage", 0.0)) - float(base_macro.get("macro_data_coverage", 0.0))
            macro_rate_delta = float(cur_macro.get("macro_event_rate", 0.0)) - float(base_macro.get("macro_event_rate", 0.0))
            if macro_score_delta <= -3.0:
                plugin_notes.append(f"macro-event score down {macro_score_delta:.1f}")
            if macro_cov_delta <= -0.08:
                plugin_notes.append(f"macro-event coverage down {macro_cov_delta:.2f}")
            if macro_rate_delta <= -0.08:
                plugin_notes.append(f"macro-event response down {macro_rate_delta:.2f}")

        if any(x.startswith(("score down", "statistically significant score down", "new issues", "random_walk skip down", "drift_up align down", "drift_down align down", "cross-symbol stability weak", "cross-symbol dispersion worse", "market_recent brier worse", "market_recent calibration worse", "market_recent path-quality worse", "market_recent brier stability weaker", "walkforward score down", "walkforward stability weaker", "walkforward PBO worse", "walkforward DSR weaker", "walkforward pass-rate down", "macro-event score down", "macro-event coverage down", "macro-event response down")) for x in plugin_notes):
            regressions.append(f"{name}: " + ", ".join(plugin_notes))
        elif plugin_notes:
            improvements.append(f"{name}: " + ", ".join(plugin_notes))

    return {
        "regressions": regressions,
        "improvements": improvements,
    }


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
        f"Audit_Bars={args.bars}||2048||1||100000||N",
        f"PredictionTargetMinutes={args.horizon}||1||1||720||N",
        f"Audit_M1SyncBars={args.m1sync_bars}||2||1||12||N",
        f"Audit_Normalization={args.normalization}||0||0||14||N",
        f"Audit_SequenceBarsOverride={args.sequence_bars}||0||0||256||N",
        f"Audit_SchemaOverride={args.schema_id}||0||0||6||N",
        f"Audit_FeatureGroupsMaskOverride={args.feature_mask}||0||0||9223372036854775807||N",
        f"Audit_CommissionPerLotSide={args.commission_per_lot_side}||0||0||100||N",
        f"Audit_CostBufferPoints={args.cost_buffer_points}||0||0||100||N",
        f"Audit_SlippagePoints={args.slippage_points}||0||0||100||N",
        f"Audit_FillPenaltyPoints={args.fill_penalty_points}||0||0||100||N",
        f"Audit_WalkForwardTrainBars={args.wf_train_bars}||64||1||50000||N",
        f"Audit_WalkForwardTestBars={args.wf_test_bars}||16||1||50000||N",
        f"Audit_WalkForwardPurgeBars={args.wf_purge_bars}||0||0||50000||N",
        f"Audit_WalkForwardEmbargoBars={args.wf_embargo_bars}||0||0||50000||N",
        f"Audit_WalkForwardFolds={args.wf_folds}||2||1||64||N",
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


def cmd_compile(_args):
    return compile_target(Path("Tests/FXAI_AuditRunner.mq5"), "audit_runner")


def cmd_compile_main(_args):
    return compile_target(Path("FXAI.mq5"), "main_ea")


def cmd_run_audit(args):
    args = build_effective_audit_args(args)
    if cmd_compile(args) != 0:
        return 1

    BASELINES_DIR.mkdir(parents=True, exist_ok=True)
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    TESTER_PRESET_DIR.mkdir(parents=True, exist_ok=True)
    output_path = Path(args.output) if args.output else DEFAULT_TEXT_REPORT
    output_path.parent.mkdir(parents=True, exist_ok=True)

    symbols = resolve_symbol_list(args)
    artifact_dir = output_path.parent / f"{output_path.stem}_artifacts"
    if len(symbols) > 1:
        artifact_dir.mkdir(parents=True, exist_ok=True)

    try:
        symbol_runs = []
        for symbol in symbols:
            raw_report_path = artifact_dir / f"{symbol}.tsv" if len(symbols) > 1 else None
            run_result = run_single_symbol_audit(args, symbol, raw_report_path)
            symbol_runs.append(run_result)
            if len(symbols) == 1:
                output_path.write_text(run_result["text"], encoding="utf-8")
                print(run_result["text"])
    except AuditRunError as exc:
        print(str(exc), file=sys.stderr)
        log_path = latest_terminal_log()
        if log_path:
            print(f"terminal log: {log_path}", file=sys.stderr)
        return 1

    current_summary = (build_multisymbol_summary(symbol_runs) if len(symbol_runs) > 1
                       else symbol_runs[0]["summary"])
    summary_path = output_path.with_suffix(".summary.json")
    manifest_path = output_path.with_suffix(".manifest.json")

    if len(symbol_runs) > 1:
        aggregate_text = render_multisymbol_report(current_summary, args.execution_profile, manifest_path)
        output_path.write_text(aggregate_text, encoding="utf-8")
        print(aggregate_text)

    manifest = {
        "type": ("multi_symbol" if len(symbol_runs) > 1 else "single_symbol"),
        "symbols": symbols,
        "symbol_pack": getattr(args, "symbol_pack", "") or "",
        "symbol_count": len(symbols),
        "plugin_list": args.plugin_list,
        "scenario_list": args.scenario_list,
        "execution_profile": args.execution_profile,
        "generated_at": int(time.time()),
        "costs": {
            "commission_per_lot_side": args.commission_per_lot_side,
            "cost_buffer_points": args.cost_buffer_points,
            "slippage_points": args.slippage_points,
            "fill_penalty_points": args.fill_penalty_points,
        },
        "walkforward": {
            "train_bars": args.wf_train_bars,
            "test_bars": args.wf_test_bars,
        },
        "artifacts": {
            "report": str(output_path),
            "summary": str(summary_path),
        },
        "reproducibility": {
            "repo_root": str(REPO_ROOT),
            "repo_head": git_head_commit(REPO_ROOT),
            "repo_dirty": git_dirty(REPO_ROOT),
            "tool_sha256": sha256_path(Path(__file__).resolve()),
            "oracle_sha256": sha256_path(ORACLES_PATH),
        },
        "per_symbol": [
            {
                "symbol": run["symbol"],
                "report_tsv": str(artifact_dir / f"{run['symbol']}.tsv") if len(symbol_runs) > 1 else str(DEFAULT_REPORT),
                "plugin_count": len(run["summary"].get("plugins", {})),
                "runtime_persistence_manifest": str(runtime_persistence_manifest_path(run["symbol"])),
                "runtime_feature_manifest": str(runtime_feature_manifest_path(run["symbol"])),
                "runtime_macro_manifest": str(runtime_macro_manifest_path(run["symbol"])),
            }
            for run in symbol_runs
        ],
    }
    write_json(summary_path, current_summary)
    manifest["artifacts"]["report_sha256"] = sha256_path(output_path)
    manifest["artifacts"]["summary_sha256"] = sha256_path(summary_path)
    write_json(manifest_path, manifest)

    if args.baseline:
        oracles = load_oracles()
        baseline_path = resolve_baseline_path(args.baseline)
        if baseline_path.exists():
            baseline_summary = load_baseline_summary(baseline_path, oracles)
            cmp_text = compare_summaries(current_summary, baseline_summary)
            print("\n" + cmp_text)
            if args.compare_output:
                Path(args.compare_output).write_text(cmp_text, encoding="utf-8")
        else:
            print(f"baseline not found: {baseline_path}", file=sys.stderr)
            return 1
    return 0


def cmd_analyze(args):
    report = Path(args.report) if args.report else DEFAULT_REPORT
    if not report.exists():
        print(f"report not found: {report}", file=sys.stderr)
        return 1
    oracles = load_oracles()
    summary = load_current_summary(report, oracles)
    if report.suffix.lower() == ".tsv":
        text = render_report(load_rows(report), oracles)
    else:
        manifest_path = report.with_suffix(".manifest.json")
        text = render_summary_report(summary, "default", (manifest_path if manifest_path.exists() else None))
    print(text)
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
    return 0


def cmd_baseline_save(args):
    report = Path(args.report) if args.report else DEFAULT_REPORT
    if not report.exists():
        print(f"report not found: {report}", file=sys.stderr)
        return 1
    BASELINES_DIR.mkdir(parents=True, exist_ok=True)
    oracles = load_oracles()
    summary = load_current_summary(report, oracles)
    name = args.name
    baseline_report = BASELINES_DIR / f"{name}{report.suffix.lower()}"
    baseline_summary = BASELINES_DIR / f"{name}.summary.json"
    shutil.copy2(report, baseline_report)
    write_json(baseline_summary, summary)
    print(f"saved baseline: {baseline_report}")
    print(f"saved summary: {baseline_summary}")
    return 0


def cmd_baseline_compare(args):
    report = Path(args.report) if args.report else DEFAULT_REPORT
    if not report.exists():
        print(f"report not found: {report}", file=sys.stderr)
        return 1
    baseline_path = resolve_baseline_path(args.baseline)
    if not baseline_path.exists():
        print(f"baseline not found: {baseline_path}", file=sys.stderr)
        return 1

    oracles = load_oracles()
    current_summary = load_current_summary(report, oracles)
    baseline_summary = load_baseline_summary(baseline_path, oracles)
    text = compare_summaries(current_summary, baseline_summary)
    print(text)
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
    return 0


def cmd_release_gate(args):
    report = Path(args.report) if args.report else DEFAULT_REPORT
    if not report.exists():
        print(f"report not found: {report}", file=sys.stderr)
        return 1
    baseline_path = resolve_baseline_path(args.baseline)
    if not baseline_path.exists():
        print(f"baseline not found: {baseline_path}", file=sys.stderr)
        return 1

    oracles = load_oracles()
    current_summary = load_current_summary(report, oracles)
    baseline_summary = load_baseline_summary(baseline_path, oracles)
    cmp = compare_summary_data(current_summary, baseline_summary)
    cmp_text = compare_summaries(current_summary, baseline_summary)
    manifest_json = report.with_suffix(".manifest.json")
    manifest = load_json(manifest_json) if manifest_json.exists() else {}

    gate_failures = []
    if cmp["regressions"]:
        gate_failures.extend(cmp["regressions"])

    if args.require_market_replay and not summary_has_market_replay(current_summary):
        gate_failures.append("current audit report does not contain the full required market replay certification pack")

    per_symbol = manifest.get("per_symbol", []) if isinstance(manifest, dict) else []
    seen_persistence = set()
    for item in per_symbol:
        p_path = manifest_path(item.get("runtime_persistence_manifest", ""))
        if p_path is None or p_path in seen_persistence:
            continue
        seen_persistence.add(p_path)
        for row in load_tsv_rows(p_path):
            if inum(row, "stateful_checkpoint") <= 0:
                continue
            if inum(row, "promotion_ready") > 0:
                continue
            depth = row.get("checkpoint_depth", row.get("coverage_tag", "unknown"))
            native_snapshot = inum(row, "native_snapshot")
            deterministic_replay = inum(row, "deterministic_replay")
            gate_failures.append(
                f"{row.get('ai_name', 'unknown')}: live promotion blocked by checkpoint coverage "
                f"({depth}, native_snapshot={native_snapshot}, deterministic_replay={deterministic_replay})"
            )

    macro_dataset_present = False
    seen_macro = set()
    for item in per_symbol:
        m_path = manifest_path(item.get("runtime_macro_manifest", ""))
        if m_path is None or m_path in seen_macro:
            continue
        seen_macro.add(m_path)
        rows = load_tsv_rows(m_path)
        if not rows:
            continue
        row = rows[0]
        if inum(row, "record_count") > 0:
            macro_dataset_present = True
        if inum(row, "record_count") > 0 and inum(row, "schema_version") < 2:
            gate_failures.append(
                f"{item.get('symbol', 'unknown')}: macro dataset schema {inum(row, 'schema_version')} is below required version 2"
            )
        if inum(row, "record_count") > 0 and inum(row, "leakage_safe") <= 0:
            gate_failures.append(
                f"{item.get('symbol', 'unknown')}: macro dataset failed leakage guard "
                f"(score={fnum(row, 'leakage_guard_score'):.2f}, parse_errors={inum(row, 'parse_errors')})"
            )
        if inum(row, "record_count") > 0 and fnum(row, "avg_source_trust") < 0.60:
            gate_failures.append(
                f"{item.get('symbol', 'unknown')}: macro dataset source trust {fnum(row, 'avg_source_trust'):.2f} below minimum 0.60"
            )
        if inum(row, "record_count") > 0 and inum(row, "distinct_revision_chains") <= 0:
            gate_failures.append(
                f"{item.get('symbol', 'unknown')}: macro dataset has no revision-chain coverage"
            )

    for name, info in sorted(current_summary.get("plugins", {}).items()):
        score = float(info.get("score", 0.0))
        if score < args.min_score:
            gate_failures.append(f"{name}: score {score:.1f} below minimum {args.min_score:.1f}")
        if "stability" in info and float(info.get("stability", 1.0)) < args.min_stability:
            gate_failures.append(f"{name}: cross-symbol stability {float(info.get('stability', 0.0)):.2f} below minimum {args.min_stability:.2f}")
        wf = info.get("scenarios", {}).get("market_walkforward", {})
        if wf:
            if float(wf.get("wf_pbo", 0.0)) > 0.45:
                gate_failures.append(f"{name}: walkforward PBO {float(wf.get('wf_pbo', 0.0)):.2f} above maximum 0.45")
            if float(wf.get("wf_dsr", 1.0)) < 0.35:
                gate_failures.append(f"{name}: walkforward DSR(proxy) {float(wf.get('wf_dsr', 0.0)):.2f} below minimum 0.35")
            if float(wf.get("wf_pass_rate", 1.0)) < 0.55:
                gate_failures.append(f"{name}: walkforward pass rate {float(wf.get('wf_pass_rate', 0.0)):.2f} below minimum 0.55")
        if args.fail_on_issues:
            issues = list(info.get("issues", []))
            findings = list(info.get("findings", []))
            if issues or findings:
                gate_failures.append(f"{name}: unresolved issues present")
        adversarial = info.get("scenarios", {}).get("market_adversarial", {})
        if not adversarial:
            gate_failures.append(f"{name}: missing market_adversarial certification")
        else:
            if float(adversarial.get("score", 0.0)) < 68.0:
                gate_failures.append(f"{name}: adversarial score {float(adversarial.get('score', 0.0)):.1f} below minimum 68.0")
            if float(adversarial.get("calibration_error", 0.0)) > 0.26:
                gate_failures.append(f"{name}: adversarial calibration error {float(adversarial.get('calibration_error', 0.0)):.3f} above maximum 0.260")
            if float(adversarial.get("path_quality_error", 0.0)) > 0.50:
                gate_failures.append(f"{name}: adversarial path-quality error {float(adversarial.get('path_quality_error', 0.0)):.3f} above maximum 0.500")
            if "adversarial audit weak" in set(info.get("issues", [])):
                gate_failures.append(f"{name}: adversarial certification issues present")
        if macro_dataset_present:
            macro = info.get("scenarios", {}).get("market_macro_event", {})
            if not macro:
                gate_failures.append(f"{name}: missing market_macro_event certification while macro dataset is active")
            else:
                if float(macro.get("macro_data_coverage", 0.0)) < 0.60:
                    gate_failures.append(
                        f"{name}: macro-event coverage {float(macro.get('macro_data_coverage', 0.0)):.2f} below minimum 0.60"
                    )
                if float(macro.get("macro_event_rate", 0.0)) < 0.06:
                    gate_failures.append(
                        f"{name}: macro-event activity {float(macro.get('macro_event_rate', 0.0)):.2f} below minimum 0.06"
                    )
                if float(macro.get("macro_provenance_trust_mean", 0.0)) < 0.55:
                    gate_failures.append(
                        f"{name}: macro provenance trust {float(macro.get('macro_provenance_trust_mean', 0.0)):.2f} below minimum 0.55"
                    )
                if float(macro.get("macro_currency_relevance_mean", 0.0)) < 0.45:
                    gate_failures.append(
                        f"{name}: macro currency relevance {float(macro.get('macro_currency_relevance_mean', 0.0)):.2f} below minimum 0.45"
                    )
                issues = set(info.get("issues", []))
                if {"macro-event blind", "macro-event overreaction", "macro-event data gap"} & issues:
                    gate_failures.append(f"{name}: macro-event certification issues present")

    if args.output:
        Path(args.output).write_text(cmp_text, encoding="utf-8")

    print(cmp_text)
    if gate_failures:
        print("\n# FXAI Release Gate: FAIL\n", file=sys.stderr)
        for item in gate_failures:
            print(f"- {item}", file=sys.stderr)
        return 1

    print("\n# FXAI Release Gate: PASS")
    return 0


def cmd_optimize_audit(args):
    report = Path(args.report) if args.report else DEFAULT_REPORT
    if not report.exists():
        print(f"report not found: {report}", file=sys.stderr)
        return 1
    oracles = load_oracles()
    summary = load_current_summary(report, oracles)
    campaign = build_optimization_campaign(summary, oracles)
    text = render_optimization_campaign(campaign)
    print(text)
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
    if args.json_output:
        Path(args.json_output).write_text(json.dumps(campaign, indent=2, sort_keys=True), encoding="utf-8")
    if args.execute:
        return execute_optimization_campaign(campaign, args)
    return 0


def main():
    ap = argparse.ArgumentParser(description="FXAI external drill-sergeant test lab")
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("compile-audit", help="Compile the MT5 audit runner")
    c.set_defaults(func=cmd_compile)

    cm = sub.add_parser("compile-main", help="Compile the main FXAI EA")
    cm.set_defaults(func=cmd_compile_main)

    ra = sub.add_parser("run-audit", help="Compile and attempt to run the MT5 audit runner, then analyze the report")
    ra.add_argument("--all-plugins", action="store_true")
    ra.add_argument("--plugin-id", type=int, default=28)
    ra.add_argument("--plugin-list", default="{all}")
    ra.add_argument("--scenario-list", default="{random_walk, drift_up, drift_down, mean_revert, vol_cluster, monotonic_up, monotonic_down, regime_shift, market_recent, market_trend, market_chop, market_session_edges, market_spread_shock, market_walkforward, market_macro_event, market_adversarial}")
    ra.add_argument("--bars", type=int, default=20000)
    ra.add_argument("--horizon", type=int, default=5)
    ra.add_argument("--m1sync-bars", type=int, default=3)
    ra.add_argument("--normalization", type=int, default=0)
    ra.add_argument("--sequence-bars", type=int, default=0)
    ra.add_argument("--schema-id", type=int, default=0)
    ra.add_argument("--feature-mask", type=int, default=0)
    ra.add_argument("--commission-per-lot-side", type=float, default=None)
    ra.add_argument("--cost-buffer-points", type=float, default=None)
    ra.add_argument("--slippage-points", type=float, default=None)
    ra.add_argument("--fill-penalty-points", type=float, default=None)
    ra.add_argument("--wf-train-bars", type=int, default=256)
    ra.add_argument("--wf-test-bars", type=int, default=64)
    ra.add_argument("--wf-purge-bars", type=int, default=32)
    ra.add_argument("--wf-embargo-bars", type=int, default=24)
    ra.add_argument("--wf-folds", type=int, default=6)
    ra.add_argument("--seed", type=int, default=42)
    ra.add_argument("--symbol", default="EURUSD")
    ra.add_argument("--symbol-list", default="")
    ra.add_argument("--symbol-pack", default="", choices=[""] + sorted(SYMBOL_PACKS.keys()))
    ra.add_argument("--execution-profile", default="default", choices=sorted(EXECUTION_PROFILES.keys()))
    ra.add_argument("--login")
    ra.add_argument("--server")
    ra.add_argument("--password")
    ra.add_argument("--timeout", type=int, default=180)
    ra.add_argument("--baseline")
    ra.add_argument("--output")
    ra.add_argument("--compare-output")
    ra.set_defaults(func=cmd_run_audit)

    a = sub.add_parser("analyze", help="Analyze an audit TSV report")
    a.add_argument("--report", default=str(DEFAULT_REPORT))
    a.add_argument("--output")
    a.set_defaults(func=cmd_analyze)

    bs = sub.add_parser("baseline-save", help="Save a report as a named regression baseline")
    bs.add_argument("--report", default=str(DEFAULT_REPORT))
    bs.add_argument("--name", required=True)
    bs.set_defaults(func=cmd_baseline_save)

    bc = sub.add_parser("baseline-compare", help="Compare a report to a saved regression baseline")
    bc.add_argument("--report", default=str(DEFAULT_REPORT))
    bc.add_argument("--baseline", required=True)
    bc.add_argument("--output")
    bc.set_defaults(func=cmd_baseline_compare)

    rg = sub.add_parser("release-gate", help="Fail if the current audit regresses against a baseline")
    rg.add_argument("--report", default=str(DEFAULT_REPORT))
    rg.add_argument("--baseline", required=True)
    rg.add_argument("--min-score", type=float, default=70.0)
    rg.add_argument("--min-stability", type=float, default=0.55)
    rg.add_argument("--fail-on-issues", action="store_true")
    rg.add_argument("--require-market-replay", action="store_true")
    rg.add_argument("--output")
    rg.set_defaults(func=cmd_release_gate)

    oa = sub.add_parser("optimize-audit", help="Generate an audit-guided optimization campaign for schemas, normalizers, and sequence lengths")
    oa.add_argument("--report", default=str(DEFAULT_REPORT))
    oa.add_argument("--output")
    oa.add_argument("--json-output")
    oa.add_argument("--execute", action="store_true")
    oa.add_argument("--output-dir")
    oa.add_argument("--limit-plugins", type=int, default=0)
    oa.add_argument("--limit-experiments", type=int, default=0)
    oa.add_argument("--bars", type=int, default=20000)
    oa.add_argument("--horizon", type=int, default=5)
    oa.add_argument("--m1sync-bars", type=int, default=3)
    oa.add_argument("--normalization", type=int, default=0)
    oa.add_argument("--sequence-bars", type=int, default=0)
    oa.add_argument("--schema-id", type=int, default=0)
    oa.add_argument("--feature-mask", type=int, default=0)
    oa.add_argument("--commission-per-lot-side", type=float, default=None)
    oa.add_argument("--cost-buffer-points", type=float, default=None)
    oa.add_argument("--slippage-points", type=float, default=None)
    oa.add_argument("--fill-penalty-points", type=float, default=None)
    oa.add_argument("--wf-train-bars", type=int, default=256)
    oa.add_argument("--wf-test-bars", type=int, default=64)
    oa.add_argument("--seed", type=int, default=42)
    oa.add_argument("--symbol", default="EURUSD")
    oa.add_argument("--symbol-list", default="")
    oa.add_argument("--symbol-pack", default="", choices=[""] + sorted(SYMBOL_PACKS.keys()))
    oa.add_argument("--execution-profile", default="default", choices=sorted(EXECUTION_PROFILES.keys()))
    oa.add_argument("--login")
    oa.add_argument("--server")
    oa.add_argument("--password")
    oa.add_argument("--timeout", type=int, default=180)
    oa.set_defaults(func=cmd_optimize_audit)

    args = ap.parse_args()
    raise SystemExit(args.func(args))


if __name__ == "__main__":
    main()
