from __future__ import annotations

from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[2]
ALLOWED_RAW_MARKET_DATA_FILES = {
    "Engine/market_data_gateway.mqh",
}
FORBIDDEN_RAW_PATTERNS = (
    r"\bCopyRates\s*\(",
    r"\bCopyTicksRange\s*\(",
    r"\bCopyTicks\s*\(",
    r"\bCopyClose\s*\(",
    r"\bSymbolInfoTick\s*\(",
    r"\biBarShift\s*\(",
    r"\biTime\s*\(",
    r"\biOpen\s*\(",
    r"\biClose\s*\(",
)


def _read(rel_path: str) -> str:
    return (ROOT / rel_path).read_text(encoding="utf-8")


def _iter_code_files() -> list[str]:
    rel_paths: list[str] = []
    for rel_root in ("Engine", "Plugins", "Services", "Tests", "API"):
        root = ROOT / rel_root
        for path in root.rglob("*"):
            if path.suffix.lower() not in {".mq5", ".mqh"}:
                continue
            rel_paths.append(path.relative_to(ROOT).as_posix())
    rel_paths.append("FXAI.mq5")
    return sorted(set(rel_paths))


def test_raw_market_data_apis_are_isolated_to_gateway():
    offenders: list[str] = []
    for rel_path in _iter_code_files():
        if rel_path in ALLOWED_RAW_MARKET_DATA_FILES:
            continue
        text = _read(rel_path)
        for pattern in FORBIDDEN_RAW_PATTERNS:
            if re.search(pattern, text):
                offenders.append(f"{rel_path}: {pattern}")
    assert not offenders, "raw MT5 market-data API bypasses found:\n" + "\n".join(offenders)


def test_data_pipeline_includes_single_market_data_gateway():
    data_pipeline = _read("Engine/data_pipeline.mqh")
    gateway = _read("Engine/market_data_gateway.mqh")
    data_io = _read("Engine/data_io.mqh")

    assert '#include "market_data_gateway.mqh"' in data_pipeline
    assert "bool FXAI_MarketDataPull(" in gateway
    assert "bool FXAI_MarketDataCopyRatesByPos(" in gateway
    assert "bool FXAI_MarketDataGetLatestTick(" in gateway
    assert "FXAI_MarketDataCopyRatesByPos(symbol, tf, 1, needed, rates_arr)" in data_io
    assert "FXAI_MarketDataBarTime(symbol, tf, 1, cur_bar_time)" in data_io
    assert "FXAI_MarketDataBarShift(symbol, tf, last_bar_time, true, shift)" in data_io


def test_prediction_path_consumers_use_gateway_and_core_pipeline_only():
    runtime = _read("Engine/Runtime/runtime_feature_pipeline_block.mqh")
    rule_m1sync = _read("Plugins/Rule/rule_m1sync.mqh")
    factor_context = _read("Engine/Runtime/runtime_factor_context.mqh")
    feature_norm = _read("Engine/feature_norm.mqh")
    feature_build = _read("Engine/feature_build.mqh")

    assert "FXAI_DataCoreRefreshLiveBundle(" in runtime
    assert "FXAI_MarketDataCopyRatesByPos(symbol, PERIOD_M1, shift, bars, live_rates)" in rule_m1sync
    assert "FXAI_MarketDataBarShift(symbol, PERIOD_M1, ctx_time, true, shift)" in rule_m1sync
    assert "FXAI_MarketDataGetLatestTick(symbol, tick)" in rule_m1sync
    assert "FXAI_MarketDataBarClose(symbol, PERIOD_D1, shift_now, now_close)" in factor_context
    for text in (feature_norm, feature_build):
        for pattern in FORBIDDEN_RAW_PATTERNS:
            assert not re.search(pattern, text)


def test_normalization_pipeline_remains_causal_and_train_split_safe():
    feature_norm = _read("Engine/feature_norm.mqh")
    warmup_norm = _read("Engine/Warmup/warmup_normalization.mqh")
    build_loop = "for(int i=end; i>=start; i--)"
    out_assign = "out_features[f] = out_v;"
    hist_write = "g_fxai_norm_hist[method_idx][f][h] = cur;"

    assert build_loop in feature_norm
    assert "FXAI_FeatureCoreBuildFrame(bundle, feature_request, feature_frame)" in feature_norm
    assert out_assign in feature_norm
    assert hist_write in feature_norm
    assert feature_norm.index(out_assign) < feature_norm.index(hist_write)
    assert "bool FXAI_DeriveNormCandidateSplit(" in warmup_norm
    assert "train_start,\n                                                    train_end," in warmup_norm
