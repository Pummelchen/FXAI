from __future__ import annotations

from offline_lab.environment import validate_environment


def test_environment_report_has_expected_sections():
    report = validate_environment()
    assert "python" in report
    assert "dependencies" in report
    assert "paths" in report
    assert "root" in report["paths"]

