from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


NEW_PLUGINS = {
    "stat_msgarch": ("AI_STAT_MSGARCH", "Plugins/Stat/stat_msgarch.mqh"),
    "stat_arimax_garch": ("AI_STAT_ARIMAX_GARCH", "Plugins/Stat/stat_arimax_garch.mqh"),
    "tree_rf": ("AI_TREE_RF", "Plugins/Tree/tree_rf.mqh"),
    "stat_coint_vecm": ("AI_STAT_COINT_VECM", "Plugins/Stat/stat_coint_vecm.mqh"),
    "stat_ou_spread": ("AI_STAT_OU_SPREAD", "Plugins/Stat/stat_ou_spread.mqh"),
    "rl_ppo": ("AI_RL_PPO", "Plugins/RL/rl_ppo.mqh"),
    "stat_microflow_proxy": ("AI_STAT_MICROFLOW_PROXY", "Plugins/Stat/stat_microflow_proxy.mqh"),
    "stat_hmm_regime": ("AI_STAT_HMM_REGIME", "Plugins/Stat/stat_hmm_regime.mqh"),
    "lin_elastic_logit": ("AI_LIN_ELASTIC_LOGIT", "Plugins/Linear/lin_elastic_logit.mqh"),
    "lin_profit_logit": ("AI_LIN_PROFIT_LOGIT", "Plugins/Linear/lin_profit_logit.mqh"),
    "ai_cnn_lstm": ("AI_CNN_LSTM", "Plugins/Sequence/ai_cnn_lstm.mqh"),
    "ai_attn_cnn_bilstm": ("AI_ATTN_CNN_BILSTM", "Plugins/Sequence/ai_attn_cnn_bilstm.mqh"),
    "stat_emd_hht": ("AI_STAT_EMD_HHT", "Plugins/Stat/stat_emd_hht.mqh"),
    "stat_vmd": ("AI_STAT_VMD", "Plugins/Stat/stat_vmd.mqh"),
    "stat_tvp_kalman": ("AI_STAT_TVP_KALMAN", "Plugins/Stat/stat_tvp_kalman.mqh"),
    "factor_pca_panel": ("AI_FACTOR_PCA_PANEL", "Plugins/Factor/factor_pca_panel.mqh"),
    "factor_ppp_value": ("AI_FACTOR_PPP_VALUE", "Plugins/Factor/factor_ppp_value.mqh"),
    "factor_carry": ("AI_FACTOR_CARRY", "Plugins/Factor/factor_carry.mqh"),
    "factor_cmv_panel": ("AI_FACTOR_CMV_PANEL", "Plugins/Factor/factor_cmv_panel.mqh"),
    "trend_tsmom_vol": ("AI_TREND_TSMOM_VOL", "Plugins/Trend/trend_tsmom_vol.mqh"),
    "trend_xsmom_rank": ("AI_TREND_XSMOM_RANK", "Plugins/Trend/trend_xsmom_rank.mqh"),
    "trend_vol_breakout": ("AI_TREND_VOL_BREAKOUT", "Plugins/Trend/trend_vol_breakout.mqh"),
    "stat_xrate_consistency": ("AI_STAT_XRATE_CONSISTENCY", "Plugins/Stat/stat_xrate_consistency.mqh"),
    "ai_gru": ("AI_GRU", "Plugins/Sequence/ai_gru.mqh"),
    "ai_bilstm": ("AI_BILSTM", "Plugins/Sequence/ai_bilstm.mqh"),
    "ai_lstm_tcn": ("AI_LSTM_TCN", "Plugins/Sequence/ai_lstm_tcn.mqh"),
}


def _read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8")


def test_ai_count_matches_registry_expansion() -> None:
    core = _read("Engine/core.mqh")
    api = _read("API/api.mqh")
    count = int(re.search(r"#define FXAI_AI_COUNT\s+(\d+)", core).group(1))
    enum_names = re.findall(r"\bAI_[A-Z0-9_]+\b", core.split("enum ENUM_AI_TYPE", 1)[1].split("};", 1)[0])
    include_paths = re.findall(r'#include "\.\.\\(Plugins\\[^"]+\.mqh)"', api)
    create_cases = re.findall(r"case \(int\)AI_[A-Z0-9_]+: return new [A-Za-z0-9_]+\(\);", api)
    assert count == 62
    assert len(enum_names) == count
    assert len(include_paths) == count
    assert len(create_cases) == count


def test_new_plugins_are_registered_and_expose_contract_methods() -> None:
    core = _read("Engine/core.mqh")
    api = _read("API/api.mqh")
    for plugin_name, (enum_name, rel_path) in NEW_PLUGINS.items():
        source = ROOT / rel_path
        assert source.exists(), rel_path
        text = source.read_text(encoding="utf-8")
        include_path = rel_path.replace("/", "\\")
        assert enum_name in core
        assert f'#include "..\\{include_path}"' in api
        assert f"case (int){enum_name}: return new " in api
        assert f'return "{plugin_name}";' in text
        for token in ("AIId(", "AIName(", "Describe("):
            assert token in text, f"{rel_path} missing {token}"


def test_framework_common_contains_required_algorithmic_guards() -> None:
    text = _read("Plugins/Common/fxai_framework_model.mqh")
    required_tokens = [
        "HMMForward",
        "MSGARCHMargin",
        "RandomForestMargin",
        "KalmanMargin",
        "PCAMargin",
        "SequenceMargin",
        "FXAI_FW_KIND_GRU",
        "FXAI_FW_KIND_BILSTM",
        "FXAI_FW_KIND_LSTM_TCN",
        "NormalizeClassDistribution(out.class_probs)",
        "MathMax(h, 1e-8)",
        "rowsum",
        "cost_points",
    ]
    for token in required_tokens:
        assert token in text
    assert "ResolveCostPoints(z)" not in text


def test_new_plugin_oracles_exist() -> None:
    data = json.loads(_read("Tools/plugin_oracles.json"))
    plugins = data["plugins"]
    for plugin_name in NEW_PLUGINS:
        entry = plugins.get(plugin_name)
        assert entry, plugin_name
        assert entry.get("identity")
        assert entry.get("recommendations")


def test_router_config_knows_new_regime_and_trend_plugins() -> None:
    data = json.loads(_read("Tools/OfflineLab/AdaptiveRouter/adaptive_router_config.json"))
    overrides = data["plugin_overrides"]
    for plugin_name in (
        "stat_msgarch",
        "stat_hmm_regime",
        "stat_microflow_proxy",
        "stat_coint_vecm",
        "stat_ou_spread",
        "factor_carry",
        "trend_vol_breakout",
    ):
        assert plugin_name in overrides
    pattern_text = json.dumps(data["plugin_patterns"], sort_keys=True)
    for token in ("msgarch", "hmm", "coint", "vecm", "tsmom", "breakout", "xsmom"):
        assert token in pattern_text


def test_new_runtime_plugins_do_not_bypass_market_data_gateway() -> None:
    forbidden = (
        "CopyRates(",
        "CopyOpen(",
        "CopyHigh(",
        "CopyLow(",
        "CopyClose(",
        "iOpen(",
        "iHigh(",
        "iLow(",
        "iClose(",
        "MarketBookGet(",
    )
    for _, rel_path in NEW_PLUGINS.values():
        text = _read(rel_path)
        for token in forbidden:
            assert token not in text, f"{rel_path} bypasses DataCore/market gateway with {token}"
