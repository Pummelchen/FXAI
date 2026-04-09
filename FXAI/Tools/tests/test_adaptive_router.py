from __future__ import annotations

import json
import tempfile
from pathlib import Path

from offline_lab.adaptive_router import write_adaptive_router_profiles
from offline_lab.adaptive_router_contracts import adaptive_router_runtime_history_path
from offline_lab.adaptive_router_replay import build_adaptive_router_replay_report
from offline_lab.common import close_db, connect_db, query_scalar
from offline_lab.environment import bootstrap_environment
from offline_lab.fixtures import patched_paths, seed_profile_fixture


def test_adaptive_router_profile_generation_writes_artifacts_and_db_rows():
    with tempfile.TemporaryDirectory(prefix="fxai_adaptive_router_") as tmp_dir:
        with patched_paths(Path(tmp_dir)) as paths:
            bootstrap_environment(paths["default_db"], init_db=True)
            conn = connect_db(paths["default_db"])
            fixture = seed_profile_fixture(conn, profile_name="adaptive")
            args = type("Args", (), {"profile": fixture["profile_name"], "db": str(paths["default_db"]), "runtime_mode": "research"})()

            artifacts = write_adaptive_router_profiles(conn, args)

            assert len(artifacts) == 1
            artifact = artifacts[0]
            assert Path(artifact["artifact_path"]).exists()
            json_path = paths["research_dir"] / fixture["profile_name"] / f"adaptive_router_{fixture['symbol']}.json"
            payload = json.loads(json_path.read_text(encoding="utf-8"))
            assert payload["symbol"] == fixture["symbol"]
            assert "plugins" in payload and payload["plugins"]
            db_count = int(query_scalar(conn, "SELECT COUNT(*) FROM adaptive_router_profiles WHERE profile_name = ?", (fixture["profile_name"],), 0))
            assert db_count == 1
            close_db(conn)


def test_adaptive_router_replay_report_summarizes_runtime_history():
    with tempfile.TemporaryDirectory(prefix="fxai_adaptive_router_replay_") as tmp_dir:
        with patched_paths(Path(tmp_dir)) as paths:
            bootstrap_environment(paths["default_db"], init_db=True)
            history_path = adaptive_router_runtime_history_path("EURUSD")
            history_path.parent.mkdir(parents=True, exist_ok=True)
            history_path.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "schema_version": 1,
                                "generated_at": "2026-04-09T08:00:00Z",
                                "symbol": "EURUSD",
                                "regime": {
                                    "top_label": "BREAKOUT_TRANSITION",
                                    "confidence": 0.61,
                                    "probabilities": {"BREAKOUT_TRANSITION": 0.61},
                                    "reasons": ["Volatility expansion detected"],
                                },
                                "router": {
                                    "mode": "WEIGHTED_ENSEMBLE",
                                    "trade_posture": "NORMAL",
                                    "reasons": ["Volatility expansion detected"],
                                },
                                "plugins": [
                                    {"name": "ai_tft", "weight": 0.44, "suitability": 1.12, "status": "ACTIVE", "reasons": ["Strong trend fit"]},
                                ],
                            }
                        ),
                        json.dumps(
                            {
                                "schema_version": 1,
                                "generated_at": "2026-04-09T08:30:00Z",
                                "symbol": "EURUSD",
                                "regime": {
                                    "top_label": "HIGH_VOL_EVENT",
                                    "confidence": 0.82,
                                    "probabilities": {"HIGH_VOL_EVENT": 0.82},
                                    "reasons": ["NewsPulse event window active"],
                                },
                                "router": {
                                    "mode": "WEIGHTED_ENSEMBLE",
                                    "trade_posture": "CAUTION",
                                    "reasons": ["NewsPulse event window active"],
                                },
                                "plugins": [
                                    {"name": "ai_gha", "weight": 0.46, "suitability": 1.28, "status": "UPWEIGHTED", "reasons": ["Strong macro/event regime fit"]},
                                ],
                            }
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            payload = build_adaptive_router_replay_report(symbol="EURUSD", hours_back=48)

            assert payload["symbol_count"] == 1
            assert (paths["adaptive_router_dir"] / "Reports/adaptive_router_replay_report.json").exists()
            entry = payload["symbols"][0]
            assert entry["symbol"] == "EURUSD"
            assert entry["regime_counts"]["HIGH_VOL_EVENT"] == 1
            assert entry["posture_counts"]["CAUTION"] == 1
            assert entry["recent_transitions"]
