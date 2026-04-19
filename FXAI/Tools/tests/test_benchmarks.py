from __future__ import annotations

import json
from pathlib import Path

from testlab.benchmarks import publish_benchmark_suite
from testlab.release_gate import (
    ADVERSARIAL_THRESHOLDS,
    ARTIFACT_SIZE_BUDGETS,
    DEFAULT_RELEASE_GATE_MIN_SCORE,
    DEFAULT_RELEASE_GATE_MIN_STABILITY,
    PERFORMANCE_THRESHOLDS,
    WALKFORWARD_THRESHOLDS,
    release_gate_policy_snapshot,
)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def test_release_gate_policy_snapshot_exports_public_thresholds():
    payload = release_gate_policy_snapshot()
    assert payload["defaults"]["min_score"] == DEFAULT_RELEASE_GATE_MIN_SCORE
    assert payload["defaults"]["min_stability"] == DEFAULT_RELEASE_GATE_MIN_STABILITY
    assert payload["artifact_size_budgets_bytes"] == ARTIFACT_SIZE_BUDGETS
    assert payload["performance_thresholds"] == PERFORMANCE_THRESHOLDS
    assert payload["walkforward_thresholds"] == WALKFORWARD_THRESHOLDS
    assert payload["adversarial_thresholds"] == ADVERSARIAL_THRESHOLDS


def test_publish_benchmark_suite_writes_matrix_reference_bundle_and_release_notes(tmp_path: Path):
    root = tmp_path / "FXAI"
    reference_report = root / "Tools/Baselines/sample_baseline.tsv"
    reference_summary = root / "Tools/Baselines/sample_baseline.summary.json"
    promoted_best = root / "Tools/OfflineLab/Profiles/bestparams/promoted_best.json"
    strategy_manifest = root / "Tools/OfflineLab/Profiles/bestparams/EURUSD/ai_mlp__strategy_profile.json"
    live_deploy = root / "Tools/OfflineLab/ResearchOS/bestparams/live_deploy_EURUSD.json"

    _write_text(
        reference_report,
        "\n".join(
            [
                "ai_id\tai_name\tfamily\tscenario\tbars_total\tsamples_total\tvalid_preds\tinvalid_preds\tbuy_count\tsell_count\tskip_count\ttrue_buy_count\ttrue_sell_count\ttrue_skip_count\texact_match_count\tdirectional_eval_count\tdirectional_correct_count\tskip_ratio\tactive_ratio\tbias_abs\tconf_drift\treset_delta\tsequence_delta\tscore\tissue_flags\tavg_conf\tavg_rel\tavg_move\ttrend_align",
                "28\trule_m1sync\t10\trandom_walk\t1000\t900\t900\t0\t20\t18\t862\t100\t100\t700\t720\t38\t22\t0.95\t0.04\t0.05\t0.05\t0.0\t-1.0\t97.0\t0\t0.93\t0.96\t2.1\t0.0",
            ]
        )
        + "\n",
    )
    _write_json(
        reference_summary,
        {
            "plugins": {
                "rule_m1sync": {
                    "family": 10,
                    "findings": [],
                    "grade": "D",
                    "issues": [],
                    "scenarios": {
                        "random_walk": {
                            "score": 97.0,
                        }
                    },
                    "score": 73.5,
                }
            }
        },
    )
    _write_json(
        promoted_best,
        [
            {
                "symbol": "EURUSD",
                "plugin_name": "ai_mlp",
                "score": 83.5,
                "ranking_score": 82.71,
                "support_count": 1,
                "support_json": json.dumps(
                    [
                        {
                            "score": 83.5,
                            "walkforward_score": 79.5,
                            "adversarial_score": 77.5,
                        }
                    ]
                ),
                "parameters_json": json.dumps(
                    {
                        "broker_profile": "",
                        "execution_profile": "default",
                        "runtime_mode": "research",
                        "horizon": 5,
                    }
                ),
                "strategy_profile_id": "strategy/default",
                "strategy_profile_version": 1,
                "strategy_profile_manifest_path": "<FXAI_ROOT>/Tools/OfflineLab/Profiles/bestparams/EURUSD/ai_mlp__strategy_profile.json",
                "audit_set_path": "<FXAI_ROOT>/Tools/OfflineLab/Profiles/bestparams/EURUSD/ai_mlp__audit.set",
                "ea_set_path": "<FXAI_ROOT>/Tools/OfflineLab/Profiles/bestparams/EURUSD/ai_mlp__ea.set",
            }
        ],
    )
    _write_json(
        strategy_manifest,
        {
            "context": {
                "resolved_broker": "default",
                "runtime_mode": "research",
            },
            "compiled": {
                "audit_values": {
                    "execution_profile": "default",
                    "horizon": 5,
                    "scenario_list": ["market_recent", "market_walkforward", "market_adversarial"],
                }
            },
        },
    )
    _write_json(
        live_deploy,
        {
            "promotion_tier": "audit-approved",
        },
    )

    payload = publish_benchmark_suite(
        root=root,
        output_dir=root / "Tools/Benchmarks",
        profile_name="bestparams",
        reference_report=reference_report,
        reference_summary=reference_summary,
        release_tag="vtest",
    )

    matrix_json = root / "Tools/Benchmarks/benchmark_matrix.json"
    release_notes_md = root / "Tools/Benchmarks/ReleaseNotes/vtest_release_notes.md"
    promotion_criteria_json = root / "Tools/Benchmarks/promotion_criteria.json"
    reference_manifest = root / "Tools/Benchmarks/ReferenceAudit/sample_audit.manifest.json"
    assert Path(payload["artifacts"]["benchmark_matrix_json"].replace("<FXAI_ROOT>/", f"{root.as_posix()}/")).exists()
    assert matrix_json.exists()
    assert release_notes_md.exists()
    assert promotion_criteria_json.exists()
    assert reference_manifest.exists()

    matrix = json.loads(matrix_json.read_text(encoding="utf-8"))
    assert len(matrix["rows"]) == 2
    assert {row["benchmark_source"] for row in matrix["rows"]} == {"reference_audit", "promoted_profile"}
    current_row = next(row for row in matrix["rows"] if row["benchmark_source"] == "promoted_profile")
    assert current_row["strategy_profile_label"] == "strategy/default@v1"
    assert current_row["walkforward_score"] == 79.5
    assert current_row["adversarial_score"] == 77.5

    release_notes = release_notes_md.read_text(encoding="utf-8")
    assert "Model change: rule_m1sync -> ai_mlp" in release_notes
    assert "Strategy profile change: reference/legacy-audit@v0 -> strategy/default@v1" in release_notes
    assert "Audit score: 73.5 -> 83.5 (+10.0)" in release_notes

    promotion_criteria = json.loads(promotion_criteria_json.read_text(encoding="utf-8"))
    assert promotion_criteria["walkforward_thresholds"]["pbo_max"] == 0.45
