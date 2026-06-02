from __future__ import annotations

import math

from offline_lab.drift_governance_math import population_stability_index


def test_population_stability_index_handles_repeated_narrow_edges():
    reference_values = [0.0] * 20 + [0.000002]
    recent_values = [0.0] * 5 + [0.000002] * 10

    score = population_stability_index(reference_values, recent_values, bucket_count=6)

    assert math.isfinite(score)
    assert score >= 0.0
