from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

from .shared import ISSUE, get_oracle, load_json

def load_current_summary(path: Path, oracles: dict) -> dict:
    if path.suffix.lower() == ".tsv":
        return build_summary(load_rows(path), oracles)
    return load_json(path)


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
