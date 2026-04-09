from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import pytest

from offline_lab import cli_commands
from offline_lab.common import close_db, connect_db
from offline_lab.environment import bootstrap_environment
from offline_lab.fixtures import patched_paths
from offline_lab.market_universe import (
    MARKET_UNIVERSE_CONFIG_KEY,
    export_market_universe_config,
    load_market_universe_config,
    summarize_market_universe_config,
)


def test_market_universe_defaults_seed_into_lab_metadata():
    with tempfile.TemporaryDirectory(prefix="fxai_market_universe_") as tmp_dir:
        with patched_paths(Path(tmp_dir)) as paths:
            bootstrap_environment(paths["default_db"], init_db=True)
            conn = connect_db(paths["default_db"])
            payload = load_market_universe_config(conn)
            row = conn.execute(
                "SELECT meta_value FROM lab_metadata WHERE meta_key = ?",
                (MARKET_UNIVERSE_CONFIG_KEY,),
            ).fetchone()
            close_db(conn)

            assert row is not None
            summary = summarize_market_universe_config(payload)
            assert summary["trading_scope"] == "FX_ONLY"
            assert summary["tradable_symbol_count"] == 54
            assert summary["indicator_symbol_count"] == 34
            assert "EURUSD" in summary["tradable_symbols"]
            assert "XAUUSD" in summary["indicator_only_symbols"]
            assert "US500" in summary["indicator_only_symbols"]
            assert "MidDE50" in summary["indicator_only_symbols"]
            assert "TecDE30" in summary["indicator_only_symbols"]


def test_market_universe_export_and_import_roundtrip():
    with tempfile.TemporaryDirectory(prefix="fxai_market_universe_roundtrip_") as tmp_dir:
        with patched_paths(Path(tmp_dir)) as paths:
            bootstrap_environment(paths["default_db"], init_db=True)
            conn = connect_db(paths["default_db"])
            export_payload = export_market_universe_config(conn)
            export_path = Path(str(export_payload["output_path"]))
            payload = json.loads(export_path.read_text(encoding="utf-8"))
            payload["description"] = "Custom market universe description"
            import_path = paths["profiles_dir"] / "market_universe_custom.json"
            import_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
            close_db(conn)

            args = argparse.Namespace(db=str(paths["default_db"]), input_path=str(import_path))
            rc = cli_commands.cmd_market_universe_import(args)
            assert rc == 0

            conn = connect_db(paths["default_db"])
            reloaded = load_market_universe_config(conn)
            close_db(conn)
            assert reloaded["description"] == "Custom market universe description"


def test_market_universe_rejects_non_fx_tradable_symbol():
    with tempfile.TemporaryDirectory(prefix="fxai_market_universe_invalid_") as tmp_dir:
        with patched_paths(Path(tmp_dir)) as paths:
            bootstrap_environment(paths["default_db"], init_db=True)
            conn = connect_db(paths["default_db"])
            payload = load_market_universe_config(conn)
            close_db(conn)

            payload["symbol_records"].append(
                {
                    "symbol": "XAUUSD",
                    "asset_class": "Metal",
                    "role": "tradable",
                    "notes": "invalid on purpose",
                }
            )
            import_path = paths["profiles_dir"] / "market_universe_invalid.json"
            import_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
            args = argparse.Namespace(db=str(paths["default_db"]), input_path=str(import_path))
            with pytest.raises(cli_commands.OfflineLabError):
                cli_commands.cmd_market_universe_import(args)
