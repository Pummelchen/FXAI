from __future__ import annotations

import json
import tempfile
from pathlib import Path

import offline_lab.label_engine_contracts as label_engine_contracts
from offline_lab.common import close_db, connect_db, query_one
from offline_lab.dashboard import build_profile_dashboard
from offline_lab.environment import bootstrap_environment
from offline_lab.fixtures import patched_paths
from offline_lab.label_engine import build_label_engine_artifacts, build_label_engine_bundle, build_label_engine_report, validate_label_engine_config
from offline_lab.label_engine_config import load_config, validate_config_payload


def _insert_dataset(conn, *, dataset_key: str = "label:eurusd:m1", profile_name: str = "labels", symbol: str = "EURUSD") -> dict:
    start_unix = 1_775_782_800
    end_unix = start_unix + 90 * 60
    conn.execute(
        """
        INSERT INTO datasets(dataset_key, group_key, symbol, timeframe, start_unix, end_unix, months, bars,
                             source_path, source_sha256, created_at, notes)
        VALUES(?, ?, ?, 'M1', ?, ?, 3, 0, ?, 'fixture', ?, 'label-test')
        """,
        (dataset_key, f"{profile_name}_fixture", symbol, start_unix, end_unix, f"/tmp/{dataset_key}.tsv", start_unix),
    )
    dataset = query_one(conn, "SELECT * FROM datasets WHERE dataset_key = ?", (dataset_key,))
    assert dataset is not None

    rows = []
    price = 1.10000
    for index in range(96):
        if index < 48:
            price += 0.00005
        else:
            price -= 0.00004
        open_price = price - 0.00001
        close_price = price
        high_price = max(open_price, close_price) + 0.00003
        low_price = min(open_price, close_price) - 0.00003
        rows.append(
            (
                int(dataset["id"]),
                start_unix + index * 60,
                round(open_price, 5),
                round(high_price, 5),
                round(low_price, 5),
                round(close_price, 5),
                12,
                100 + index,
                0,
            )
        )
    conn.executemany(
        """
        INSERT INTO dataset_bars(dataset_id, bar_time_unix, open, high, low, close, spread_points, tick_volume, real_volume)
        VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    conn.execute("UPDATE datasets SET bars = ? WHERE id = ?", (len(rows), int(dataset["id"])))
    return query_one(conn, "SELECT * FROM datasets WHERE dataset_key = ?", (dataset_key,)) or {}


def test_label_engine_validate_creates_default_files():
    with tempfile.TemporaryDirectory(prefix="fxai_label_engine_validate_") as tmp_dir:
        with patched_paths(Path(tmp_dir)):
            payload = validate_label_engine_config()
            config = load_config()
            assert payload["ok"] is True
            assert Path(payload["config_path"]).exists()
            assert config["execution_profile"] == "default"
            assert len(config["horizons"]) >= 3


def test_label_engine_validate_rejects_duplicate_horizon_ids():
    payload = load_config()
    payload["horizons"] = [{"id": "M5", "bars": 5}, {"id": "M5", "bars": 15}]
    try:
        validate_config_payload(payload)
    except Exception as exc:  # noqa: BLE001
        assert "Duplicate label-engine horizon id" in str(exc)
    else:
        raise AssertionError("duplicate label-engine horizon ids should fail validation")


def test_label_engine_bundle_writes_artifacts_and_dashboard_summary():
    with tempfile.TemporaryDirectory(prefix="fxai_label_engine_bundle_") as tmp_dir:
        with patched_paths(Path(tmp_dir)) as paths:
            bootstrap_environment(paths["default_db"], init_db=True)
            conn = connect_db(paths["default_db"])
            dataset = _insert_dataset(conn)
            bundle = build_label_engine_bundle(conn, dataset=dataset, profile_name="continuous")
            report = build_label_engine_report(conn, profile_name="continuous")
            dashboard = build_profile_dashboard(conn, "continuous")
            close_db(conn)

            assert bundle["summary_metrics"]["label_row_count"] > 0
            assert bundle["meta_summary"]["candidate_count"] > 0
            assert Path(bundle["artifact_paths"]["labels_ndjson"]).exists()
            assert Path(bundle["artifact_paths"]["meta_labels_ndjson"]).exists()
            assert report["artifact_count"] == 1
            assert dashboard["label_engine"]["report"]["artifact_count"] == 1
            assert dashboard["label_engine"]["report"]["builds"][0]["dataset_key"] == dataset["dataset_key"]


def test_label_engine_build_supports_external_candidates():
    with tempfile.TemporaryDirectory(prefix="fxai_label_engine_external_") as tmp_dir:
        with patched_paths(Path(tmp_dir)) as paths:
            bootstrap_environment(paths["default_db"], init_db=True)
            conn = connect_db(paths["default_db"])
            dataset = _insert_dataset(conn, dataset_key="label:external:eurusd")
            candidate_path = Path(tmp_dir) / "candidates.ndjson"
            candidate_path.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "signal_id": "sig-1",
                                "timestamp_unix": 1_775_782_800 + 32 * 60,
                                "side": "BUY",
                                "horizon_id": "M15",
                                "raw_score": 0.62,
                                "diagnostics": {"source": "unit-test"},
                            }
                        )
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            args = type(
                "Args",
                (),
                {
                    "db": str(paths["default_db"]),
                    "profile": "continuous",
                    "dataset_keys": dataset["dataset_key"],
                    "group_key": "",
                    "symbol": "EURUSD",
                    "symbol_list": "",
                    "symbol_pack": "",
                    "months_list": "3",
                    "start_unix": 0,
                    "end_unix": 0,
                    "execution_profile": "",
                    "candidate_path": str(candidate_path),
                    "candidate_mode": "EXTERNAL_FILE",
                    "limit_datasets": 0,
                },
            )()
            payload = build_label_engine_artifacts(conn, args=args)
            close_db(conn)

            assert payload["bundle_count"] == 1
            report = payload["report"]
            build = report["builds"][0]
            assert build["meta_summary"]["candidate_mode"] == "EXTERNAL_FILE"
            assert build["summary_metrics"]["meta_label_count"] >= 1


def test_label_engine_build_rejects_external_mode_without_candidate_path():
    with tempfile.TemporaryDirectory(prefix="fxai_label_engine_external_missing_") as tmp_dir:
        with patched_paths(Path(tmp_dir)) as paths:
            bootstrap_environment(paths["default_db"], init_db=True)
            conn = connect_db(paths["default_db"])
            dataset = _insert_dataset(conn, dataset_key="label:external-missing:eurusd")
            args = type(
                "Args",
                (),
                {
                    "db": str(paths["default_db"]),
                    "profile": "continuous",
                    "dataset_keys": dataset["dataset_key"],
                    "group_key": "",
                    "symbol": "EURUSD",
                    "symbol_list": "",
                    "symbol_pack": "",
                    "months_list": "3",
                    "start_unix": 0,
                    "end_unix": 0,
                    "execution_profile": "",
                    "candidate_path": "",
                    "candidate_mode": "EXTERNAL_FILE",
                    "limit_datasets": 0,
                },
            )()
            try:
                build_label_engine_artifacts(conn, args=args)
            except Exception as exc:  # noqa: BLE001
                assert "requires --candidate-path" in str(exc)
            else:
                raise AssertionError("external candidate mode without a candidate file should fail")
            finally:
                close_db(conn)


def test_label_engine_build_honors_meta_label_disable():
    with tempfile.TemporaryDirectory(prefix="fxai_label_engine_meta_disabled_") as tmp_dir:
        with patched_paths(Path(tmp_dir)) as paths:
            bootstrap_environment(paths["default_db"], init_db=True)
            config = load_config()
            config["meta_labeling"]["enabled"] = False
            label_engine_contracts.LABEL_ENGINE_CONFIG_PATH.write_text(
                json.dumps(config, indent=2, sort_keys=True),
                encoding="utf-8",
            )

            conn = connect_db(paths["default_db"])
            dataset = _insert_dataset(conn, dataset_key="label:meta-disabled:eurusd")
            bundle = build_label_engine_bundle(conn, dataset=dataset, profile_name="continuous")
            close_db(conn)

            assert bundle["meta_summary"]["enabled"] is False
            assert bundle["meta_summary"]["candidate_count"] == 0
            assert bundle["summary_metrics"]["meta_label_count"] == 0
