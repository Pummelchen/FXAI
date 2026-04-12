from __future__ import annotations

import json
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

from offline_lab.dynamic_ensemble_config import load_config
import offline_lab.dynamic_ensemble_contracts as contracts
from offline_lab.dynamic_ensemble_replay import build_dynamic_ensemble_replay_report
from offline_lab.environment import bootstrap_environment
from offline_lab.fixtures import patched_paths


def _iso_hours_ago(hours_ago: float) -> str:
    return (datetime.now(timezone.utc) - timedelta(hours=hours_ago)).strftime("%Y-%m-%dT%H:%M:%SZ")


def test_dynamic_ensemble_validate_exports_runtime_config():
    with tempfile.TemporaryDirectory(prefix="fxai_dynamic_ensemble_cfg_") as tmp_dir:
        with patched_paths(Path(tmp_dir)) as paths:
            bootstrap_environment(paths["default_db"], init_db=True)
            payload = load_config()

            assert payload["enabled"] is True
            assert contracts.DYNAMIC_ENSEMBLE_RUNTIME_CONFIG_PATH.exists()
            runtime_text = contracts.DYNAMIC_ENSEMBLE_RUNTIME_CONFIG_PATH.read_text(encoding="utf-8")
            assert "threshold_suppress_trust_threshold" in runtime_text
            assert "family_news_compat_transformer" in runtime_text


def test_dynamic_ensemble_replay_report_summarizes_runtime_history():
    with tempfile.TemporaryDirectory(prefix="fxai_dynamic_ensemble_replay_") as tmp_dir:
        with patched_paths(Path(tmp_dir)) as paths:
            bootstrap_environment(paths["default_db"], init_db=True)
            history_path = contracts.dynamic_ensemble_runtime_history_path("EURUSD")
            history_path.parent.mkdir(parents=True, exist_ok=True)
            history_path.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "schema_version": 1,
                                "generated_at": _iso_hours_ago(2.0),
                                "symbol": "EURUSD",
                                "ensemble": {
                                    "trade_posture": "NORMAL",
                                    "ensemble_quality": 0.66,
                                    "abstain_bias": 0.08,
                                    "final_action": "BUY",
                                    "reasons": ["Strong plugin agreement"],
                                },
                                "plugins": [
                                    {"name": "ai_tft", "status": "ACTIVE", "weight": 0.44, "trust": 0.78},
                                    {"name": "ai_gha", "status": "DOWNWEIGHTED", "weight": 0.22, "trust": 0.52},
                                ],
                            }
                        ),
                        json.dumps(
                            {
                                "schema_version": 1,
                                "generated_at": _iso_hours_ago(1.5),
                                "symbol": "EURUSD",
                                "ensemble": {
                                    "trade_posture": "CAUTION",
                                    "ensemble_quality": 0.41,
                                    "abstain_bias": 0.24,
                                    "final_action": "SKIP",
                                    "reasons": ["Plugin disagreement elevated", "NewsPulse caution active"],
                                },
                                "plugins": [
                                    {"name": "ai_gha", "status": "ACTIVE", "weight": 0.39, "trust": 0.74},
                                    {"name": "lin_pa", "status": "SUPPRESSED", "weight": 0.0, "trust": 0.20},
                                ],
                            }
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            payload = build_dynamic_ensemble_replay_report(symbol="EURUSD", hours_back=48)

            assert payload["symbol_count"] == 1
            assert (paths["dynamic_ensemble_dir"] / "Reports/dynamic_ensemble_replay_report.json").exists()
            entry = payload["symbols"][0]
            assert entry["symbol"] == "EURUSD"
            assert entry["posture_counts"]["CAUTION"] == 1
            assert entry["action_counts"]["SKIP"] == 1
            assert entry["top_dominant_plugins"][0]["plugin"] == "ai_gha"
            assert entry["recent_transitions"]
