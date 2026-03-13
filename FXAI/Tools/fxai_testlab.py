#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path("/Users/andreborchert/Library/Application Support/net.metaquotes.wine.metatrader5/drive_c/Program Files/MetaTrader 5/MQL5/Experts/FXAI")
TERMINAL_ROOT = Path("/Users/andreborchert/Library/Application Support/net.metaquotes.wine.metatrader5/drive_c/Program Files/MetaTrader 5")
METAEDITOR = TERMINAL_ROOT / "MetaEditor64.exe"
TERMINAL = TERMINAL_ROOT / "terminal64.exe"
WINE = Path("/Applications/MetaTrader 5.app/Contents/SharedSupport/wine/bin/wine64")
COMMON_FILES = Path("/Users/andreborchert/Library/Application Support/net.metaquotes.wine.metatrader5/drive_c/users/andreborchert/AppData/Roaming/MetaQuotes/Terminal/Common/Files")
DEFAULT_REPORT = COMMON_FILES / "FXAI/Audit/fxai_audit_report.tsv"
DEFAULT_TEXT_REPORT = ROOT / "Tools/latest_drill_report.md"
ORACLES_PATH = ROOT / "Tools/plugin_oracles.json"
BASELINES_DIR = ROOT / "Tools/Baselines"
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
                "reset_delta": fnum(row, "reset_delta"),
                "sequence_delta": fnum(row, "sequence_delta"),
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


def summary_has_market_replay(summary: dict) -> bool:
    required = {"market_recent", "market_trend", "market_chop", "market_session_edges", "market_spread_shock", "market_walkforward"}
    seen = set()
    for plugin in summary.get("plugins", {}).values():
        seen.update(plugin.get("scenarios", {}).keys())
    return required.issubset(seen)


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
        scenario_focus = ["random_walk", "drift_up", "drift_down", "market_trend", "market_chop", "market_session_edges", "market_spread_shock", "market_walkforward"]

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
            "focus": ["market_recent", "market_trend", "market_chop", "market_session_edges", "market_spread_shock", "market_walkforward"],
        })
        experiments.append({
            "name": "walkforward_gate",
            "focus": ["market_walkforward", "market_session_edges", "market_spread_shock"],
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
            elif exp["name"] == "walkforward_gate":
                out.append(f"- Walk-forward release gate: {exp['focus']}")
                out.append("  run: release-gate --baseline <baseline-name> --require-market-replay")
        out.append("")
    return "\n".join(out)


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
        cur_issues = set(cur.get("issues", [])) | set(cur.get("findings", []))
        base_issues = set(base.get("issues", [])) | set(base.get("findings", []))
        new_issues = sorted(cur_issues - base_issues)
        fixed_issues = sorted(base_issues - cur_issues)

        plugin_notes = []
        if delta <= -2.0:
            plugin_notes.append(f"score down {delta:.1f}")
        elif delta >= 2.0:
            plugin_notes.append(f"score up +{delta:.1f}")
        if new_issues:
            plugin_notes.append("new issues: " + "; ".join(new_issues[:4]))
        if fixed_issues:
            plugin_notes.append("fixed issues: " + "; ".join(fixed_issues[:4]))

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

        if any(x.startswith(("score down", "new issues", "random_walk skip down", "drift_up align down", "drift_down align down")) for x in plugin_notes):
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


def cmd_compile(_args):
    return compile_target(Path("Tests/FXAI_AuditRunner.mq5"), "audit_runner")


def cmd_compile_main(_args):
    return compile_target(Path("FXAI.mq5"), "main_ea")


def cmd_run_audit(args):
    if cmd_compile(args) != 0:
        return 1

    BASELINES_DIR.mkdir(parents=True, exist_ok=True)
    TESTER_PRESET_DIR.mkdir(parents=True, exist_ok=True)

    preset_name = "fxai_audit_runner.set"
    preset_path = TESTER_PRESET_DIR / preset_name
    write_audit_set(preset_path, args)

    login, server, password = resolve_credentials(args)

    if DEFAULT_REPORT.exists():
        DEFAULT_REPORT.unlink()
    success, mode, failure = attempt_audit_launch(login, server, password, preset_name, args)
    if not success:
        if not failure and not login:
            failure = "MT5 tester did not produce a report and no login was available from common.ini"
        elif not failure and not password:
            failure = "MT5 tester did not produce a report and no password was supplied; set FXAI_MT5_PASSWORD or use --password"
        elif not failure:
            failure = "MT5 tester exited without producing the audit report"
        print(f"{mode} launch failed: {failure}", file=sys.stderr)
        log_path = latest_terminal_log()
        if log_path:
            print(f"terminal log: {log_path}", file=sys.stderr)
        return 1

    oracles = load_oracles()
    rows = load_rows(DEFAULT_REPORT)
    text = render_report(rows, oracles)
    output_path = Path(args.output) if args.output else DEFAULT_TEXT_REPORT
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")
    print(text)

    if args.baseline:
        current_summary = build_summary(rows, oracles)
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
    rows = load_rows(report)
    text = render_report(rows, oracles)
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
    rows = load_rows(report)
    summary = build_summary(rows, oracles)
    name = args.name
    baseline_report = BASELINES_DIR / f"{name}.tsv"
    baseline_summary = BASELINES_DIR / f"{name}.summary.json"
    shutil.copy2(report, baseline_report)
    baseline_summary.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
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
    current_summary = build_summary(load_rows(report), oracles)
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
    current_summary = build_summary(load_rows(report), oracles)
    baseline_summary = load_baseline_summary(baseline_path, oracles)
    cmp = compare_summary_data(current_summary, baseline_summary)
    cmp_text = compare_summaries(current_summary, baseline_summary)

    gate_failures = []
    if cmp["regressions"]:
        gate_failures.extend(cmp["regressions"])

    if args.require_market_replay and not summary_has_market_replay(current_summary):
        gate_failures.append("current audit report does not contain the full required market replay certification pack")

    for name, info in sorted(current_summary.get("plugins", {}).items()):
        score = float(info.get("score", 0.0))
        if score < args.min_score:
            gate_failures.append(f"{name}: score {score:.1f} below minimum {args.min_score:.1f}")
        if args.fail_on_issues:
            issues = list(info.get("issues", []))
            findings = list(info.get("findings", []))
            if issues or findings:
                gate_failures.append(f"{name}: unresolved issues present")

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
    summary = build_summary(load_rows(report), oracles)
    campaign = build_optimization_campaign(summary, oracles)
    text = render_optimization_campaign(campaign)
    print(text)
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
    if args.json_output:
        Path(args.json_output).write_text(json.dumps(campaign, indent=2, sort_keys=True), encoding="utf-8")
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
    ra.add_argument("--scenario-list", default="{random_walk, drift_up, drift_down, mean_revert, vol_cluster, monotonic_up, monotonic_down, regime_shift, market_recent, market_trend, market_chop, market_session_edges, market_spread_shock, market_walkforward}")
    ra.add_argument("--bars", type=int, default=20000)
    ra.add_argument("--horizon", type=int, default=5)
    ra.add_argument("--m1sync-bars", type=int, default=3)
    ra.add_argument("--normalization", type=int, default=0)
    ra.add_argument("--sequence-bars", type=int, default=0)
    ra.add_argument("--schema-id", type=int, default=0)
    ra.add_argument("--feature-mask", type=int, default=0)
    ra.add_argument("--seed", type=int, default=42)
    ra.add_argument("--symbol", default="EURUSD")
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
    rg.add_argument("--fail-on-issues", action="store_true")
    rg.add_argument("--require-market-replay", action="store_true")
    rg.add_argument("--output")
    rg.set_defaults(func=cmd_release_gate)

    oa = sub.add_parser("optimize-audit", help="Generate an audit-guided optimization campaign for schemas, normalizers, and sequence lengths")
    oa.add_argument("--report", default=str(DEFAULT_REPORT))
    oa.add_argument("--output")
    oa.add_argument("--json-output")
    oa.set_defaults(func=cmd_optimize_audit)

    args = ap.parse_args()
    raise SystemExit(args.func(args))


if __name__ == "__main__":
    main()
