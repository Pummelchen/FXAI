from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import fxai_testlab as testlab

from offline_lab import cli as offline_cli
from offline_lab.common import close_db, connect_db
from offline_lab.environment import bootstrap_environment
from offline_lab.fixtures import patched_paths, seed_completed_run_fixture, seed_profile_fixture


def test_compile_strategy_profile_layers_and_overrides():
    with tempfile.TemporaryDirectory(prefix="fxai_strategy_profiles_") as tmp_dir:
        with patched_paths(Path(tmp_dir)) as _paths:
            compiled = testlab.compile_strategy_profile(
                strategy_profile="default",
                symbol="EURUSD",
                broker_profile="retail-fx",
                runtime_mode="production",
                overrides={
                    "horizon": 13,
                    "normalization": 14,
                    "sequence_bars": 16,
                    "schema_id": 6,
                    "execution_profile": "stress",
                },
                plugin_name="ai_mlp",
                ai_id=4,
            )

            assert compiled["strategy_profile_id"] == "strategy/default"
            assert compiled["context"]["runtime_mode"] == "production"
            assert "symbol/EURUSD" in compiled["layer_ids"]
            assert "broker/retail-fx" in compiled["layer_ids"]
            assert "runtime/production" in compiled["layer_ids"]
            assert compiled["compiled"]["ea_inputs"]["AI_Type"] == 4
            assert compiled["compiled"]["ea_inputs"]["PredictionTargetMinutes"] == 13
            assert compiled["compiled"]["ea_inputs"]["AI_FeatureNormalization"] == 14
            assert compiled["compiled"]["ea_inputs"]["AI_ExecutionProfile"] == 4
            assert compiled["compiled"]["audit_values"]["sequence_bars"] == 16
            assert compiled["compiled"]["audit_values"]["schema_id"] == 6


def test_best_params_writes_strategy_profile_manifest():
    with tempfile.TemporaryDirectory(prefix="fxai_best_strategy_") as tmp_dir:
        with patched_paths(Path(tmp_dir)) as paths:
            bootstrap_environment(paths["default_db"], init_db=True)
            conn = connect_db(paths["default_db"])
            fixture = seed_profile_fixture(conn, profile_name="bestparams")
            seed_completed_run_fixture(conn, fixture)
            close_db(conn)

            args = argparse.Namespace(
                db=str(paths["default_db"]),
                profile="bestparams",
                dataset_keys="",
                group_key="",
                symbol="",
                symbol_list="",
                symbol_pack="",
                runtime_mode="research",
            )
            assert offline_cli.cmd_best_params(args) == 0

            manifest_path = paths["profiles_dir"] / "bestparams" / "EURUSD" / "ai_mlp__strategy_profile.json"
            tester_manifest_path = paths["tester_dir"] / "fxai_offline_bestparams__EURUSD__ai_mlp__strategy_profile.json"
            promoted_json_path = paths["profiles_dir"] / "bestparams" / "promoted_best.json"

            assert manifest_path.exists()
            assert tester_manifest_path.exists()
            promoted_rows = json.loads(promoted_json_path.read_text(encoding="utf-8"))
            assert promoted_rows
            first = promoted_rows[0]
            assert first["strategy_profile_id"] == "strategy/default"
            assert first["strategy_profile_version"] == 1
            assert first["strategy_profile_manifest_path"].endswith("ai_mlp__strategy_profile.json")

            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            assert manifest["strategy_profile_id"] == "strategy/default"
            assert manifest["context"]["symbol"] == "EURUSD"
            assert manifest["compiled"]["ea_inputs"]["AI_Type"] == 4
            assert manifest["compiled"]["ea_inputs"]["TradeKiller"] == 0
