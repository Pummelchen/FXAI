from __future__ import annotations

import sys
from pathlib import Path

from .baseline import compare_summaries, compare_summary_data, load_baseline_summary, resolve_baseline_path
from .reporting import load_current_summary, load_tsv_rows, manifest_path, summary_has_market_replay, fnum, inum
from .shared import COMMON_FILES, DEFAULT_REPORT, load_json, load_oracles, runtime_artifact_safe_symbol, runtime_performance_manifest_path

DEFAULT_RELEASE_GATE_MIN_SCORE = 70.0
DEFAULT_RELEASE_GATE_MIN_STABILITY = 0.55

ARTIFACT_SIZE_BUDGETS: dict[str, int] = {
    "fxai_live_deploy": 24 * 1024,
    "fxai_student_router": 24 * 1024,
    "fxai_attribution": 32 * 1024,
    "fxai_world_plan": 32 * 1024,
    "fxai_supervisor_service": 24 * 1024,
    "fxai_supervisor_command": 16 * 1024,
    "fxai_perf": 128 * 1024,
}

PERFORMANCE_THRESHOLDS: dict[str, float] = {
    "predict_mean_ms_max": 1.60,
    "update_mean_ms_max": 1.25,
    "working_set_kb_max": 4096.0,
    "runtime_total_mean_ms_max": 12.0,
}

WALKFORWARD_THRESHOLDS: dict[str, float] = {
    "pbo_max": 0.45,
    "dsr_min": 0.35,
    "pass_rate_min": 0.55,
}

ADVERSARIAL_THRESHOLDS: dict[str, float] = {
    "score_min": 68.0,
    "calibration_error_max": 0.26,
    "path_quality_error_max": 0.50,
}

MACRO_DATASET_THRESHOLDS: dict[str, float] = {
    "schema_version_min": 2.0,
    "avg_source_trust_min": 0.60,
    "coverage_min": 0.60,
    "event_rate_min": 0.06,
    "provenance_trust_min": 0.55,
    "currency_relevance_min": 0.45,
    "distinct_revision_chains_min": 1.0,
}


def release_gate_policy_snapshot(*,
                                 min_score: float = DEFAULT_RELEASE_GATE_MIN_SCORE,
                                 min_stability: float = DEFAULT_RELEASE_GATE_MIN_STABILITY,
                                 fail_on_issues: bool = False,
                                 require_market_replay: bool = False) -> dict[str, object]:
    return {
        "defaults": {
            "min_score": float(min_score),
            "min_stability": float(min_stability),
            "fail_on_issues": bool(fail_on_issues),
            "require_market_replay": bool(require_market_replay),
        },
        "artifact_size_budgets_bytes": dict(ARTIFACT_SIZE_BUDGETS),
        "performance_thresholds": dict(PERFORMANCE_THRESHOLDS),
        "walkforward_thresholds": dict(WALKFORWARD_THRESHOLDS),
        "adversarial_thresholds": dict(ADVERSARIAL_THRESHOLDS),
        "macro_dataset_thresholds": dict(MACRO_DATASET_THRESHOLDS),
    }


def _build_symbol_artifact_report(symbol: str) -> dict[str, object]:
    token = runtime_artifact_safe_symbol(symbol)
    files = {
        "fxai_live_deploy": COMMON_FILES / f"FXAI/Offline/Promotions/fxai_live_deploy_{token}.tsv",
        "fxai_student_router": COMMON_FILES / f"FXAI/Offline/Promotions/fxai_student_router_{token}.tsv",
        "fxai_attribution": COMMON_FILES / f"FXAI/Offline/Promotions/fxai_attribution_{token}.tsv",
        "fxai_world_plan": COMMON_FILES / f"FXAI/Offline/Promotions/fxai_world_plan_{token}.tsv",
        "fxai_supervisor_service": COMMON_FILES / f"FXAI/Offline/Promotions/fxai_supervisor_service_{token}.tsv",
        "fxai_supervisor_command": COMMON_FILES / f"FXAI/Offline/Promotions/fxai_supervisor_command_{token}.tsv",
        "fxai_perf": runtime_performance_manifest_path(symbol),
    }
    failures = []
    for key, path in files.items():
        size_bytes = int(path.stat().st_size) if path.exists() and path.is_file() else 0
        budget = int(ARTIFACT_SIZE_BUDGETS[key])
        if size_bytes > budget:
            failures.append(f"{key} {size_bytes} > {budget}")
    return {"budget_failures": failures}

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
    seen_perf = set()
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
        if inum(row, "record_count") > 0 and inum(row, "schema_version") < int(MACRO_DATASET_THRESHOLDS["schema_version_min"]):
            gate_failures.append(
                f"{item.get('symbol', 'unknown')}: macro dataset schema {inum(row, 'schema_version')} is below required version {int(MACRO_DATASET_THRESHOLDS['schema_version_min'])}"
            )
        if inum(row, "record_count") > 0 and inum(row, "leakage_safe") <= 0:
            gate_failures.append(
                f"{item.get('symbol', 'unknown')}: macro dataset failed leakage guard "
                f"(score={fnum(row, 'leakage_guard_score'):.2f}, parse_errors={inum(row, 'parse_errors')})"
            )
        if inum(row, "record_count") > 0 and fnum(row, "avg_source_trust") < float(MACRO_DATASET_THRESHOLDS["avg_source_trust_min"]):
            gate_failures.append(
                f"{item.get('symbol', 'unknown')}: macro dataset source trust {fnum(row, 'avg_source_trust'):.2f} below minimum {float(MACRO_DATASET_THRESHOLDS['avg_source_trust_min']):.2f}"
            )
        if inum(row, "record_count") > 0 and inum(row, "distinct_revision_chains") < int(MACRO_DATASET_THRESHOLDS["distinct_revision_chains_min"]):
            gate_failures.append(
                f"{item.get('symbol', 'unknown')}: macro dataset has no revision-chain coverage"
            )
        p_path = manifest_path(item.get("runtime_performance_manifest", "")) or runtime_performance_manifest_path(str(item.get("symbol", "")))
        if p_path is None or p_path in seen_perf:
            continue
        seen_perf.add(p_path)
        perf_rows = load_tsv_rows(p_path)
        if not perf_rows:
            continue
        runtime_total = 0.0
        for row_perf in perf_rows:
            if row_perf.get("row_type") == "stage":
                runtime_total += fnum(row_perf, "mean_ms")
            elif row_perf.get("row_type") == "plugin":
                predict_ms = fnum(row_perf, "predict_mean_ms")
                update_ms = fnum(row_perf, "update_mean_ms")
                working_set = fnum(row_perf, "working_set_kb")
                if predict_ms > float(PERFORMANCE_THRESHOLDS["predict_mean_ms_max"]):
                    gate_failures.append(
                        f"{row_perf.get('ai_name', 'unknown')}: predict mean {predict_ms:.3f}ms above maximum {float(PERFORMANCE_THRESHOLDS['predict_mean_ms_max']):.3f}ms"
                    )
                if update_ms > float(PERFORMANCE_THRESHOLDS["update_mean_ms_max"]):
                    gate_failures.append(
                        f"{row_perf.get('ai_name', 'unknown')}: update mean {update_ms:.3f}ms above maximum {float(PERFORMANCE_THRESHOLDS['update_mean_ms_max']):.3f}ms"
                    )
                if working_set > float(PERFORMANCE_THRESHOLDS["working_set_kb_max"]):
                    gate_failures.append(
                        f"{row_perf.get('ai_name', 'unknown')}: working set {working_set:.1f}KB above maximum {float(PERFORMANCE_THRESHOLDS['working_set_kb_max']):.1f}KB"
                    )
        if runtime_total > float(PERFORMANCE_THRESHOLDS["runtime_total_mean_ms_max"]):
            gate_failures.append(
                f"{item.get('symbol', 'unknown')}: runtime total mean {runtime_total:.3f}ms above maximum {float(PERFORMANCE_THRESHOLDS['runtime_total_mean_ms_max']):.3f}ms"
            )
        artifact_report = _build_symbol_artifact_report(str(item.get("symbol", "")))
        for failure in artifact_report.get("budget_failures", []):
            gate_failures.append(f"{item.get('symbol', 'unknown')}: artifact size gate failed ({failure})")

    for name, info in sorted(current_summary.get("plugins", {}).items()):
        score = float(info.get("score", 0.0))
        if score < args.min_score:
            gate_failures.append(f"{name}: score {score:.1f} below minimum {args.min_score:.1f}")
        if "stability" in info and float(info.get("stability", 1.0)) < args.min_stability:
            gate_failures.append(f"{name}: cross-symbol stability {float(info.get('stability', 0.0)):.2f} below minimum {args.min_stability:.2f}")
        wf = info.get("scenarios", {}).get("market_walkforward", {})
        if wf:
            if float(wf.get("wf_pbo", 0.0)) > float(WALKFORWARD_THRESHOLDS["pbo_max"]):
                gate_failures.append(f"{name}: walkforward PBO {float(wf.get('wf_pbo', 0.0)):.2f} above maximum {float(WALKFORWARD_THRESHOLDS['pbo_max']):.2f}")
            if float(wf.get("wf_dsr", 1.0)) < float(WALKFORWARD_THRESHOLDS["dsr_min"]):
                gate_failures.append(f"{name}: walkforward DSR(proxy) {float(wf.get('wf_dsr', 0.0)):.2f} below minimum {float(WALKFORWARD_THRESHOLDS['dsr_min']):.2f}")
            if float(wf.get("wf_pass_rate", 1.0)) < float(WALKFORWARD_THRESHOLDS["pass_rate_min"]):
                gate_failures.append(f"{name}: walkforward pass rate {float(wf.get('wf_pass_rate', 0.0)):.2f} below minimum {float(WALKFORWARD_THRESHOLDS['pass_rate_min']):.2f}")
        if args.fail_on_issues:
            issues = list(info.get("issues", []))
            findings = list(info.get("findings", []))
            if issues or findings:
                gate_failures.append(f"{name}: unresolved issues present")
        adversarial = info.get("scenarios", {}).get("market_adversarial", {})
        if not adversarial:
            gate_failures.append(f"{name}: missing market_adversarial certification")
        else:
            if float(adversarial.get("score", 0.0)) < float(ADVERSARIAL_THRESHOLDS["score_min"]):
                gate_failures.append(f"{name}: adversarial score {float(adversarial.get('score', 0.0)):.1f} below minimum {float(ADVERSARIAL_THRESHOLDS['score_min']):.1f}")
            if float(adversarial.get("calibration_error", 0.0)) > float(ADVERSARIAL_THRESHOLDS["calibration_error_max"]):
                gate_failures.append(f"{name}: adversarial calibration error {float(adversarial.get('calibration_error', 0.0)):.3f} above maximum {float(ADVERSARIAL_THRESHOLDS['calibration_error_max']):.3f}")
            if float(adversarial.get("path_quality_error", 0.0)) > float(ADVERSARIAL_THRESHOLDS["path_quality_error_max"]):
                gate_failures.append(f"{name}: adversarial path-quality error {float(adversarial.get('path_quality_error', 0.0)):.3f} above maximum {float(ADVERSARIAL_THRESHOLDS['path_quality_error_max']):.3f}")
            if "adversarial audit weak" in set(info.get("issues", [])):
                gate_failures.append(f"{name}: adversarial certification issues present")
        if macro_dataset_present:
            macro = info.get("scenarios", {}).get("market_macro_event", {})
            if not macro:
                gate_failures.append(f"{name}: missing market_macro_event certification while macro dataset is active")
            else:
                if float(macro.get("macro_data_coverage", 0.0)) < float(MACRO_DATASET_THRESHOLDS["coverage_min"]):
                    gate_failures.append(
                        f"{name}: macro-event coverage {float(macro.get('macro_data_coverage', 0.0)):.2f} below minimum {float(MACRO_DATASET_THRESHOLDS['coverage_min']):.2f}"
                    )
                if float(macro.get("macro_event_rate", 0.0)) < float(MACRO_DATASET_THRESHOLDS["event_rate_min"]):
                    gate_failures.append(
                        f"{name}: macro-event activity {float(macro.get('macro_event_rate', 0.0)):.2f} below minimum {float(MACRO_DATASET_THRESHOLDS['event_rate_min']):.2f}"
                    )
                if float(macro.get("macro_provenance_trust_mean", 0.0)) < float(MACRO_DATASET_THRESHOLDS["provenance_trust_min"]):
                    gate_failures.append(
                        f"{name}: macro provenance trust {float(macro.get('macro_provenance_trust_mean', 0.0)):.2f} below minimum {float(MACRO_DATASET_THRESHOLDS['provenance_trust_min']):.2f}"
                    )
                if float(macro.get("macro_currency_relevance_mean", 0.0)) < float(MACRO_DATASET_THRESHOLDS["currency_relevance_min"]):
                    gate_failures.append(
                        f"{name}: macro currency relevance {float(macro.get('macro_currency_relevance_mean', 0.0)):.2f} below minimum {float(MACRO_DATASET_THRESHOLDS['currency_relevance_min']):.2f}"
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
