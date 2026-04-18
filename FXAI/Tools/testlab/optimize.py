from __future__ import annotations

import argparse
import json
from pathlib import Path

from .reporting import grade
from .shared import DEFAULT_TEXT_REPORT, EXPERIMENTS_DIR, build_effective_audit_args, deep_merge, load_oracles, write_json

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
        bars=run.get("bars", base_args.bars),
        horizon=run.get("horizon", base_args.horizon),
        m1sync_bars=run.get("m1sync_bars", base_args.m1sync_bars),
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
        wf_purge_bars=run.get("wf_purge_bars", getattr(base_args, "wf_purge_bars", 32)),
        wf_embargo_bars=run.get("wf_embargo_bars", getattr(base_args, "wf_embargo_bars", 24)),
        wf_folds=run.get("wf_folds", getattr(base_args, "wf_folds", 6)),
        seed=base_args.seed,
        symbol=symbol,
        symbol_list=("{" + ", ".join(symbols) + "}" if symbols else "{" + symbol + "}"),
        strategy_profile=getattr(base_args, "strategy_profile", "default"),
        broker_profile=getattr(base_args, "broker_profile", ""),
        runtime_mode=getattr(base_args, "runtime_mode", "research"),
        window_start_unix=run.get("window_start_unix", getattr(base_args, "window_start_unix", 0)),
        window_end_unix=run.get("window_end_unix", getattr(base_args, "window_end_unix", 0)),
        execution_profile=run.get("execution_profile", getattr(base_args, "execution_profile", "default")),
        login=base_args.login,
        server=base_args.server,
        password=base_args.password,
        timeout=base_args.timeout,
        baseline=None,
        output=str(output_path),
        compare_output=None,
        skip_compile=getattr(base_args, "skip_compile", False),
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

