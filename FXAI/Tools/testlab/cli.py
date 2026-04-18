from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

from .audit_run import latest_terminal_log, run_single_symbol_audit
from .baseline import compare_summaries, load_baseline_summary, resolve_baseline_path
from .compile import compile_target
from .optimize import build_optimization_campaign, execute_optimization_campaign, render_optimization_campaign
from .release_gate import cmd_release_gate
from .release_artifacts import cmd_package_mt5_release
from .reporting import build_multisymbol_summary, load_current_summary, load_rows, render_multisymbol_report, render_report, render_summary_report
from .shared import (
    AuditRunError,
    BASELINES_DIR,
    DEFAULT_REPORT,
    DEFAULT_TEXT_REPORT,
    EXECUTION_PROFILES,
    EXPERIMENTS_DIR,
    ORACLES_PATH,
    REPO_ROOT,
    SYMBOL_PACKS,
    TESTER_PRESET_DIR,
    build_effective_audit_args,
    git_dirty,
    git_head_commit,
    load_json,
    load_oracles,
    resolve_symbol_list,
    runtime_feature_manifest_path,
    runtime_macro_manifest_path,
    runtime_performance_manifest_path,
    runtime_persistence_manifest_path,
    sha256_path,
    write_json,
)
from .verify import run_verify_all

def cmd_compile(_args):
    return compile_target(Path("Tests/FXAI_AuditRunner.mq5"), "audit_runner")


def cmd_compile_main(_args):
    return compile_target(Path("FXAI.mq5"), "main_ea")


def cmd_verify_all(args):
    payload = run_verify_all(refresh_golden=bool(getattr(args, "refresh_golden", False)))
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if bool(payload.get("ok")) else 1


def cmd_doctor(_args):
    from offline_lab.environment import doctor_report

    payload = doctor_report()
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if bool(payload.get("ok")) else 1


def cmd_run_audit(args):
    args = build_effective_audit_args(args)
    if not getattr(args, "skip_compile", False) and cmd_compile(args) != 0:
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
            "purge_bars": args.wf_purge_bars,
            "embargo_bars": args.wf_embargo_bars,
            "folds": args.wf_folds,
        },
        "window": {
            "start_unix": int(getattr(args, "window_start_unix", 0) or 0),
            "end_unix": int(getattr(args, "window_end_unix", 0) or 0),
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
                "runtime_performance_manifest": str(runtime_performance_manifest_path(run["symbol"])),
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

    doctor = sub.add_parser("doctor", help="Run a profile-aware FXAI toolchain and environment self-check")
    doctor.set_defaults(func=cmd_doctor)

    pkg = sub.add_parser("package-mt5-release", help="Compile and package MT5 .ex5 binaries for GitHub Releases")
    pkg.add_argument("--version", required=True, help="Release version or tag used in artifact metadata")
    pkg.add_argument("--output-dir", default="")
    pkg.add_argument("--release-profile", default="production")
    pkg.add_argument("--compatible-profiles", default="research,production")
    pkg.add_argument("--skip-compile", action="store_true", help="Package existing local .ex5 files without recompiling")
    pkg.set_defaults(func=cmd_package_mt5_release)

    va = sub.add_parser("verify-all", help="Run Python tests, deterministic fixture checks, and clean MT5 compiles")
    va.add_argument("--refresh-golden", action="store_true")
    va.set_defaults(func=cmd_verify_all)

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
    ra.add_argument("--window-start-unix", type=int, default=0)
    ra.add_argument("--window-end-unix", type=int, default=0)
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
    ra.add_argument("--skip-compile", action="store_true", help=argparse.SUPPRESS)
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
    oa.add_argument("--wf-purge-bars", type=int, default=32)
    oa.add_argument("--wf-embargo-bars", type=int, default=24)
    oa.add_argument("--wf-folds", type=int, default=6)
    oa.add_argument("--window-start-unix", type=int, default=0)
    oa.add_argument("--window-end-unix", type=int, default=0)
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
