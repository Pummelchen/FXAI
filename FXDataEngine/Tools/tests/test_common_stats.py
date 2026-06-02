from __future__ import annotations

import pytest

from offline_lab.common_stats import mean_std, row_float


def test_mean_std_uses_sample_standard_deviation():
    mean, std = mean_std([1.0, 2.0, 3.0])

    assert mean == pytest.approx(2.0)
    assert std == pytest.approx(1.0)


def test_row_float_handles_expected_missing_and_conversion_failures():
    assert row_float(None, "value", 7.0) == pytest.approx(7.0)
    assert row_float({}, "value", 3.5) == pytest.approx(3.5)
    assert row_float({"value": "bad"}, "value", 2.0) == pytest.approx(2.0)
    assert row_float(["bad"], "value", 4.0) == pytest.approx(4.0)


def test_row_float_does_not_mask_unexpected_mapping_errors():
    class ExplodingMapping:
        def __getitem__(self, key: str) -> object:
            raise RuntimeError(f"unexpected lookup for {key}")

    with pytest.raises(RuntimeError):
        row_float(ExplodingMapping(), "value")
