from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from offline_lab import exporter


def _history_response(api_version: str | None = None) -> dict:
    return {
        "api_version": api_version or exporter.FXDATABASE_FXBACKTEST_API_VERSION,
        "metadata": {
            "broker_source_id": "default",
            "source_origin": "MT5",
            "logical_symbol": "EURUSD",
            "mt5_symbol": "EURUSD",
            "timeframe": "M1",
            "digits": 5,
            "requested_utc_start": 1_704_067_200,
            "requested_utc_end_exclusive": 1_704_067_320,
            "first_utc": 1_704_067_200,
            "last_utc": 1_704_067_260,
            "row_count": 2,
        },
        "utc_timestamps": [1_704_067_200, 1_704_067_260],
        "open": [110000, 110010],
        "high": [110020, 110030],
        "low": [109990, 110000],
        "close": [110010, 110020],
        "volume": [100, 120],
    }


class _FakeHTTPResponse:
    def __init__(self, payload: dict) -> None:
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")


def test_fxdatabase_export_request_uses_latest_version_and_normalized_source_origin(monkeypatch):
    captured: dict[str, object] = {}

    def fake_urlopen(request, timeout):
        captured["timeout"] = timeout
        captured["url"] = request.full_url
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        return _FakeHTTPResponse(_history_response())

    monkeypatch.setattr(exporter.urllib.request, "urlopen", fake_urlopen)
    args = SimpleNamespace(
        fxdatabase_api_url="http://127.0.0.1:8765/",
        broker_source_id=" default ",
        source_origin=" yahoo_finance_history ",
        max_bars=25,
        timeout=5,
    )

    response = exporter.fetch_fxdatabase_m1_history(
        args,
        symbol=" eurusd ",
        start_unix=1_704_067_200,
        end_unix=1_704_067_320,
    )

    assert response["api_version"] == exporter.FXDATABASE_FXBACKTEST_API_VERSION
    assert captured["url"] == "http://127.0.0.1:8765/v1/history/m1"
    assert captured["timeout"] == 5
    assert captured["payload"] == {
        "api_version": exporter.FXDATABASE_FXBACKTEST_API_VERSION,
        "broker_source_id": "default",
        "source_origin": "YAHOO_FINANCE_HISTORY",
        "logical_symbol": "EURUSD",
        "utc_start_inclusive": 1_704_067_200,
        "utc_end_exclusive": 1_704_067_320,
        "maximum_rows": 25,
    }


def test_fxdatabase_exporter_rejects_stale_api_response_version():
    stale = _history_response(api_version="fxdatabase.fxbacktest.v0")

    with pytest.raises(Exception, match="FXDatabase API version mismatch"):
        exporter.validate_fxdatabase_m1_history_response(stale)


def test_fxdatabase_exporter_rejects_mismatched_ohlcv_columns(tmp_path: Path):
    malformed = _history_response()
    malformed["close"] = [110010]

    with pytest.raises(Exception, match="mismatched OHLCV column lengths"):
        exporter.write_fxdatabase_export_files(
            malformed,
            tmp_path / "dataset.tsv",
            tmp_path / "dataset.meta.tsv",
        )
