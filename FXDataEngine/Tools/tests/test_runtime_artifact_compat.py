from __future__ import annotations

import tempfile
from pathlib import Path

from offline_lab.dashboard import live_state_snapshot
from offline_lab.fixtures import patched_paths


def test_live_state_handles_missing_artifacts():
    with tempfile.TemporaryDirectory(prefix="fxai_live_state_") as tmp_dir:
        with patched_paths(Path(tmp_dir)):
            payload = live_state_snapshot("missing", "EURUSD")
            assert payload["symbol"] == "EURUSD"
            assert payload["deployment"] == {}
