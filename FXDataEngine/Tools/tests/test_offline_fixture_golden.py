from __future__ import annotations

from offline_lab.verification import verify_deterministic_outputs


def test_fixture_outputs_match_golden():
    result = verify_deterministic_outputs(refresh_golden=False)
    assert result["ok"], result["mismatches"]

