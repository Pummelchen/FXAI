from __future__ import annotations

import json
import tempfile
from pathlib import Path

from offline_lab.common import close_db, connect_db
from offline_lab.environment import bootstrap_environment
from offline_lab.fixtures import patched_paths
from offline_lab.pair_network import (
    aggregate_currency_exposure,
    build_pair_network_artifacts,
    pair_currency_exposure,
    pair_factor_exposure,
    resolve_candidate,
    validate_pair_network_config,
)
from offline_lab.pair_network_config import load_config
import offline_lab.pair_network_contracts as contracts


def _seed_dataset(conn, dataset_id: int, symbol: str, closes: list[float], created_at: int) -> None:
    dataset_key = f"{symbol}_fixture_{dataset_id}"
    conn.execute(
        """
        INSERT INTO datasets(
            id, dataset_key, group_key, symbol, timeframe, start_unix, end_unix,
            months, bars, source_path, source_sha256, created_at, notes
        ) VALUES (?, ?, '', ?, 'M1', ?, ?, 0, ?, '/tmp/dataset.csv', '', ?, '')
        """,
        (dataset_id, dataset_key, symbol, created_at, created_at + len(closes) * 60, len(closes), created_at),
    )
    for index, close in enumerate(closes):
        conn.execute(
            """
            INSERT INTO dataset_bars(
                dataset_id, bar_time_unix, open, high, low, close, spread_points, tick_volume, real_volume
            ) VALUES (?, ?, ?, ?, ?, ?, 12, 100, 100)
            """,
            (dataset_id, created_at + index * 60, close, close, close, close),
        )
    conn.commit()


def test_pair_network_validate_creates_default_files():
    with tempfile.TemporaryDirectory(prefix="fxai_pair_network_validate_") as tmp_dir:
        with patched_paths(Path(tmp_dir)):
            payload = validate_pair_network_config()
            config = load_config()
            assert payload["ok"] is True
            assert Path(payload["config_path"]).exists()
            assert contracts.PAIR_NETWORK_RUNTIME_CONFIG_PATH.exists()
            assert contracts.PAIR_NETWORK_RUNTIME_STATUS_PATH.exists()
            assert config["enabled"] is True
            assert "EURUSD" in config["market_universe"]["tradable_pairs"]


def test_pair_network_exposure_decomposition_and_factor_mapping():
    config = load_config()
    profiles = config["currency_profiles"]
    exposure = pair_currency_exposure("EURUSD", "BUY", 1.5)
    assert exposure == {"EUR": 1.5, "USD": -1.5}

    portfolio = aggregate_currency_exposure(
        [
            {"symbol": "EURUSD", "action": "BUY", "size_units": 1.0},
            {"symbol": "USDJPY", "action": "BUY", "size_units": 0.5},
        ]
    )
    assert portfolio["EUR"] == 1.0
    assert portfolio["JPY"] == -0.5

    factor = pair_factor_exposure("AUDUSD", "BUY", profiles, 1.0)
    assert factor["commodity_fx"] > 0.0
    assert factor["risk_on"] > 0.0


def test_pair_network_resolver_blocks_direct_contradiction():
    config = load_config()
    resolution = resolve_candidate(
        config=config,
        candidate_trade={"symbol": "EURUSD", "action": "SELL", "size_units": 1.0, "edge_after_costs": 0.30},
        open_positions=[{"symbol": "EURUSD", "action": "BUY", "size_units": 1.0}],
    )
    assert resolution["resolution"]["decision"] == "BLOCK_CONTRADICTORY"
    assert "DIRECT_SYMBOL_CONTRADICTION" in resolution["reason_codes"]


def test_pair_network_resolver_prefers_better_overlapping_expression():
    config = load_config()
    resolution = resolve_candidate(
        config=config,
        candidate_trade={
            "symbol": "NZDUSD",
            "action": "BUY",
            "size_units": 1.0,
            "edge_after_costs": 0.18,
            "execution_quality_score": 0.42,
            "calibration_quality": 0.44,
            "portfolio_fit": 0.38,
            "macro_fit": 0.55,
        },
        open_positions=[{"symbol": "AUDUSD", "action": "BUY", "size_units": 1.0}],
        peer_candidates=[
            {
                "symbol": "AUDUSD",
                "action": "BUY",
                "size_units": 1.0,
                "edge_after_costs": 0.81,
                "execution_quality_score": 0.76,
                "calibration_quality": 0.72,
                "portfolio_fit": 0.74,
                "macro_fit": 0.70,
            }
        ],
        execution_stress_score=0.62,
    )
    assert resolution["resolution"]["decision"] == "PREFER_ALTERNATIVE_EXPRESSION"
    assert resolution["resolution"]["preferred_expression"] == "AUDUSD"
    assert "BETTER_ALTERNATIVE_EXPRESSION" in resolution["reason_codes"]


def test_pair_network_build_artifacts_structural_only_when_no_dataset_support():
    with tempfile.TemporaryDirectory(prefix="fxai_pair_network_build_") as tmp_dir:
        with patched_paths(Path(tmp_dir)) as paths:
            bootstrap_environment(paths["default_db"], init_db=True)
            conn = connect_db(paths["default_db"])
            try:
                payload = build_pair_network_artifacts(conn, profile_name="continuous")
            finally:
                close_db(conn)

            assert payload["ok"] is True
            assert contracts.PAIR_NETWORK_STATUS_PATH.exists()
            assert contracts.PAIR_NETWORK_REPORT_PATH.exists()
            report = contracts.PAIR_NETWORK_REPORT_PATH.read_text(encoding="utf-8")
            assert "STRUCTURAL_ONLY" in report


def test_pair_network_build_artifacts_uses_empirical_edges_when_supported():
    with tempfile.TemporaryDirectory(prefix="fxai_pair_network_empirical_") as tmp_dir:
        with patched_paths(Path(tmp_dir)) as paths:
            bootstrap_environment(paths["default_db"], init_db=True)
            conn = connect_db(paths["default_db"])
            try:
                closes_eurusd = [1.10, 1.101, 1.103, 1.102, 1.105, 1.107, 1.106, 1.108, 1.110, 1.111] * 20
                closes_gbpusd = [1.28, 1.281, 1.284, 1.283, 1.287, 1.289, 1.288, 1.291, 1.293, 1.294] * 20
                _seed_dataset(conn, 101, "EURUSD", closes_eurusd, 1_700_000_000)
                _seed_dataset(conn, 102, "GBPUSD", closes_gbpusd, 1_700_000_000)

                config = load_config()
                config["min_empirical_overlap"] = 64
                config["empirical_lookback_bars"] = 180
                from offline_lab.pair_network_config import save_config

                save_config(config)
                payload = build_pair_network_artifacts(conn, profile_name="continuous")
            finally:
                close_db(conn)

            assert payload["ok"] is True
            report = contracts.PAIR_NETWORK_REPORT_PATH.read_text(encoding="utf-8")
            assert "STRUCTURAL_PLUS_EMPIRICAL" in report
            parsed = json.loads(report)
            assert parsed["top_edges"]
            assert "source_pair" in parsed["top_edges"][0]
            assert "target_pair" in parsed["top_edges"][0]
