from __future__ import annotations

import json
from pathlib import Path

from testlab.reporting import build_summary, load_rows, render_summary_report
from testlab.shared import load_oracles


ROOT = Path(__file__).resolve().parents[2]
REFERENCE_DIR = ROOT / "Tools/Benchmarks/ReferenceAudit"


def test_reference_audit_bundle_rebuilds_the_committed_summary():
    rows = load_rows(REFERENCE_DIR / "sample_audit.tsv")
    oracles = load_oracles()
    rebuilt = build_summary(rows, oracles)
    expected = json.loads((REFERENCE_DIR / "sample_audit.summary.json").read_text(encoding="utf-8"))
    assert rebuilt == expected


def test_reference_audit_bundle_renders_a_stable_summary_surface():
    rows = load_rows(REFERENCE_DIR / "sample_audit.tsv")
    summary = build_summary(rows, load_oracles())
    text = render_summary_report(summary, "default")
    assert "# FXAI Audit Summary" in text
    assert "## rule_m1sync | " in text
    assert "Scenarios: monotonic_up=" in text
    assert "random_walk=" in text
