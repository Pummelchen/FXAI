from __future__ import annotations

import argparse

from testlab.optimize import build_optimization_campaign, parse_walkforward_year_presets
from testlab.walkforward import (
    FX_M1_BARS_PER_TRADING_YEAR,
    WalkForwardPolicy,
    analyze_walkforward_report,
    analyze_walkforward_rows,
    apply_walkforward_policy,
    build_walkforward_windows,
    render_walkforward_analysis,
)


def _args(**overrides):
    values = {
        "bars": 20_000,
        "wf_train_bars": 256,
        "wf_test_bars": 64,
        "wf_purge_bars": 32,
        "wf_embargo_bars": 24,
        "wf_folds": 3,
        "wf_train_years": 0.0,
        "wf_test_years": 0.0,
        "wf_purge_days": 0.0,
        "wf_embargo_days": 0.0,
        "wf_window_mode": "rolling",
        "wf_bars_per_year": FX_M1_BARS_PER_TRADING_YEAR,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_year_policy_resolves_to_m1_bars_and_expands_audit_bars():
    args = apply_walkforward_policy(
        _args(
            wf_train_years=1.0,
            wf_test_years=0.25,
            wf_purge_days=2.0,
            wf_embargo_days=1.0,
        )
    )

    assert args.wf_train_bars == FX_M1_BARS_PER_TRADING_YEAR
    assert args.wf_test_bars == FX_M1_BARS_PER_TRADING_YEAR // 4
    assert args.wf_purge_bars == 2 * 24 * 60
    assert args.wf_embargo_bars == 24 * 60
    assert args.bars == args.walkforward_policy.minimum_bars()
    assert args.walkforward_policy.mode == "rolling"


def test_walkforward_windows_respect_purge_embargo_and_window_modes():
    rolling = WalkForwardPolicy(
        mode="rolling",
        train_bars=10,
        test_bars=3,
        purge_bars=2,
        embargo_bars=1,
        folds=3,
    )
    rolling_windows = build_walkforward_windows(50, rolling)
    assert [(w.train_start_bar, w.train_end_bar, w.purge_start_bar, w.purge_end_bar, w.test_start_bar, w.test_end_bar, w.embargo_start_bar, w.embargo_end_bar) for w in rolling_windows] == [
        (0, 9, 10, 11, 12, 14, 15, 15),
        (4, 13, 14, 15, 16, 18, 19, 19),
        (8, 17, 18, 19, 20, 22, 23, 23),
    ]

    anchored = WalkForwardPolicy(
        mode="anchored",
        train_bars=10,
        test_bars=3,
        purge_bars=2,
        embargo_bars=1,
        folds=3,
    )
    anchored_windows = build_walkforward_windows(50, anchored)
    assert [(w.train_start_bar, w.train_end_bar, w.test_start_bar, w.test_end_bar) for w in anchored_windows] == [
        (0, 9, 12, 14),
        (0, 13, 16, 18),
        (0, 17, 20, 22),
    ]


def test_walkforward_analyzer_flags_degrading_oos_edge():
    rows = [
        {
            "scenario": "market_walkforward",
            "ai_name": "ai_mlp",
            "window_index": index,
            "wf_train_score": train,
            "wf_test_score": test,
            "wf_pass_rate": pass_rate,
            "wf_pbo": pbo,
            "wf_dsr": dsr,
        }
        for index, train, test, pass_rate, pbo, dsr in [
            (1, 84.0, 82.0, 0.86, 0.18, 0.74),
            (2, 83.0, 75.0, 0.80, 0.25, 0.62),
            (3, 82.0, 62.0, 0.58, 0.41, 0.38),
        ]
    ]

    payload = analyze_walkforward_rows(rows, {"walkforward": {"mode": "rolling", "train_years": 1.0}})
    info = payload["plugins"]["ai_mlp"]

    assert info["degradation_flag"] is True
    assert info["test_score_slope"] < 0.0
    assert info["worst_test_score"] == 62.0
    rendered = render_walkforward_analysis(payload)
    assert "DEGRADING" in rendered
    assert "window 3" in rendered


def test_walkforward_report_analyzer_prefers_sidecar_window_evidence(tmp_path):
    report = tmp_path / "audit.tsv"
    report.write_text(
        "ai_id\tai_name\tfamily\tscenario\twf_train_score\twf_test_score\twf_pass_rate\twf_pbo\twf_dsr\n"
        "4\tai_mlp\t2\tmarket_walkforward\t90\t90\t1\t0\t1\n",
        encoding="utf-8",
    )
    report.with_suffix(".walkforward.json").write_text(
        """
{
  "schemaVersion": 1,
  "plugins": [
    {
      "aiID": 4,
      "aiName": "ai_mlp",
      "family": 2,
      "scenario": "market_walkforward",
      "folds": 2,
      "windows": [
        {"fold": 1, "trainSamples": 128, "testSamples": 64, "trainScore": 82.0, "testScore": 80.0, "gap": 2.0, "passed": true, "overfit": false},
        {"fold": 2, "trainSamples": 128, "testSamples": 64, "trainScore": 84.0, "testScore": 60.0, "gap": 24.0, "passed": false, "overfit": true}
      ]
    }
  ]
}
""",
        encoding="utf-8",
    )

    payload = analyze_walkforward_report(report)
    info = payload["plugins"]["ai_mlp"]

    assert info["window_count"] == 2
    assert info["worst_test_score"] == 60.0
    assert info["degradation_flag"] is True


def test_optimization_campaign_uses_configurable_year_presets():
    summary = {
        "plugins": {
            "ai_mlp": {
                "family": 2,
                "score": 72.0,
                "grade": "C",
                "issues": ["walkforward unstable"],
            }
        }
    }

    assert parse_walkforward_year_presets("1;2,3") == [1.0, 2.0, 3.0]
    campaign = build_optimization_campaign(summary, {}, walkforward_years=[1.0, 2.0])
    experiments = campaign["plugins"]["ai_mlp"]["experiments"]
    walkforward_gate = next(exp for exp in experiments if exp["name"] == "walkforward_gate")

    assert walkforward_gate["train_years"] == [1.0, 2.0]
    assert walkforward_gate["test_years"] == 0.25
    assert walkforward_gate["window_modes"] == ["rolling", "anchored"]
