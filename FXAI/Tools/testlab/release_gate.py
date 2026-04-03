from __future__ import annotations

import sys
from pathlib import Path

from .baseline import compare_summaries, compare_summary_data, load_baseline_summary, resolve_baseline_path
from .reporting import load_current_summary, load_tsv_rows, manifest_path, summary_has_market_replay, fnum, inum
from .shared import DEFAULT_REPORT, load_json, load_oracles

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

