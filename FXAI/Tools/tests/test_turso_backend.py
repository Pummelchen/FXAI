from __future__ import annotations

import tempfile
from pathlib import Path

from offline_lab.common import connect_db
from offline_lab.environment import bootstrap_environment, validate_environment
from offline_lab.fixtures import patched_paths


def test_turso_row_compatibility_supports_mapping_and_index_access():
    with tempfile.TemporaryDirectory(prefix="fxai_turso_") as tmp_dir:
        with patched_paths(Path(tmp_dir)) as paths:
            bootstrap_environment(paths["default_db"], init_db=True)
            conn = connect_db(paths["default_db"])
            conn.execute(
                """
                INSERT INTO lab_metadata(meta_key, meta_value, updated_at)
                VALUES('row_test', 'ok', 1700000000)
                ON CONFLICT(meta_key) DO UPDATE SET
                    meta_value=excluded.meta_value,
                    updated_at=excluded.updated_at
                """
            )
            conn.commit()
            row = conn.execute(
                "SELECT meta_key, meta_value FROM lab_metadata WHERE meta_key = ?",
                ("row_test",),
            ).fetchone()
            conn.close()

            assert row is not None
            assert row["meta_key"] == "row_test"
            assert row["meta_value"] == "ok"
            assert row[0] == "row_test"
            assert row[1] == "ok"
            assert dict(row) == {"meta_key": "row_test", "meta_value": "ok"}


def test_environment_reports_turso_dependencies():
    report = validate_environment()
    assert report["dependencies"]["libsql"] is True
    assert report["dependencies"]["turso_cli"] is True
    assert str(report["database"]["backend"]).startswith("turso_")
