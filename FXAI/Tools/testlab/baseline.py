from __future__ import annotations

from pathlib import Path

from .shared import BASELINES_DIR

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


