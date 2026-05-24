from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


CONVERTED_MODEL_CASES = {
    "stat_msgarch": "statMSGARCH",
    "stat_arimax_garch": "statARIMAXGARCH",
    "tree_rf": "treeRF",
    "stat_coint_vecm": "statCointVECM",
    "stat_ou_spread": "statOUSpread",
    "rl_ppo": "rlPPO",
    "stat_microflow_proxy": "statMicroflowProxy",
    "stat_hmm_regime": "statHMMRegime",
    "lin_elastic_logit": "linElasticLogit",
    "lin_profit_logit": "linProfitLogit",
    "ai_cnn_lstm": "cnnLSTM",
    "ai_attn_cnn_bilstm": "attnCNNBiLSTM",
    "stat_emd_hht": "statEMDHHT",
    "stat_vmd": "statVMD",
    "stat_tvp_kalman": "statTVPKalman",
    "factor_pca_panel": "factorPCAPanel",
    "factor_ppp_value": "factorPPPValue",
    "factor_carry": "factorCarry",
    "factor_cmv_panel": "factorCMVPanel",
    "trend_tsmom_vol": "trendTSMOMVol",
    "trend_xsmom_rank": "trendXSMOMRank",
    "trend_vol_breakout": "trendVolBreakout",
    "stat_xrate_consistency": "statXRateConsistency",
    "ai_gru": "gru",
    "ai_bilstm": "bilstm",
    "ai_lstm_tcn": "lstmTCN",
    "ai_mythos_rdt": "mythosRDT",
    "fxbacktest_moving_average_cross": "demoMovingAverageCross",
    "fxbacktest_fxstupid": "demoFXStupid",
    "fx7": "demoFX7",
}


def _read(rel_path: str) -> str:
    return (ROOT / rel_path).read_text(encoding="utf-8")


def _assert_tokens(text: str, tokens: list[str]) -> None:
    for token in tokens:
        assert token in text


def test_ai_count_matches_swift_registry_expansion() -> None:
    constants = _read("FXDataEngine/Sources/FXDataEngine/Core/Constants.swift")
    core_types = _read("FXDataEngine/Sources/FXDataEngine/Core/CoreTypes.swift")
    count = int(re.search(r"public static let aiCount = (\d+)", constants).group(1))
    enum_body = core_types.split("public enum AIModelID", 1)[1].split("public var usesDeepNormalizationCandidates", 1)[0]
    enum_cases = re.findall(r"case ([A-Za-z0-9_]+)", enum_body)

    assert count == 66
    assert len(enum_cases) == count
    for swift_case in CONVERTED_MODEL_CASES.values():
        assert swift_case in enum_cases


def test_swift_plugin_contract_exposes_volume_and_backend_requirements() -> None:
    contracts = _read("FXDataEngine/Sources/FXDataEngine/Plugins/PluginContracts.swift")
    ml_backend = _read("FXDataEngine/Sources/FXDataEngine/Core/MLBackend.swift")
    fxplugins_readme = _read("FXPlugins/README.md")

    _assert_tokens(
        contracts,
        [
            "public struct PluginManifestV4",
            "public var requiresVolumeWhenAvailable: Bool",
            "requiresVolumeWhenAvailable: Bool = true",
            "public protocol FXAIPluginV4: Sendable",
            "var manifest: PluginManifestV4 { get }",
            "mutating func train(",
            "func predict(",
            "func selfTest() -> Bool",
        ],
    )
    _assert_tokens(
        ml_backend,
        [
            "case pyTorch",
            "case tensorFlow",
            "case foundationNLP",
            "public protocol ExternalMLBackend",
            "public struct PythonMLBackendBridge",
            "configurationError(",
            "framework == .pyTorch",
            "framework == .tensorFlow",
            "framework == .foundationNLP",
            "Python bridge does not support",
            "public let usesVolumeFeatures: Bool",
            "dataHasVolume: request.context.dataHasVolume",
        ],
    )
    _assert_tokens(
        fxplugins_readme,
        [
            "AI plugins own model execution.",
            "PyTorch or TensorFlow backend",
            "Swift FXDataEngine OHLCV contracts",
            "use volume-derived features whenever the loaded dataset has nonzero volume",
        ],
    )


def test_framework_common_contains_required_swift_algorithmic_guards() -> None:
    support = _read("FXDataEngine/Sources/FXDataEngine/Plugins/PluginSupport.swift")
    contracts = _read("FXDataEngine/Sources/FXDataEngine/Plugins/PluginContracts.swift")
    _assert_tokens(
        support,
        [
            "public static func moveEdgeWeight(movePoints: Double, priceCostPoints: Double) -> Double",
            "let edge = move - priceCost",
            "return fxClamp(0.50 + edge / denominator, 0.25, 4.00)",
            "public static func moveSampleWeight(",
            "qualityTargets: PluginQualityTargets? = nil",
            "return baseWeight * fxClamp(quality, 0.45, 1.85)",
            "public static func computeReplayPriority(",
            "let edge = max(abs(fxSafeFinite(movePoints)) - max(fxSafeFinite(priceCostPoints), 0.0), 0.0)",
        ],
    )
    _assert_tokens(
        contracts,
        [
            "guard priceCostPoints.isFinite, priceCostPoints >= 0 else",
            "guard minMovePoints.isFinite, minMovePoints >= 0 else",
            "public static func validateCompatibility(manifest: PluginManifestV4, context: PluginContextV4) throws",
        ],
    )
    assert "ResolveCostPoints" not in support


def test_new_plugin_oracles_exist() -> None:
    data = json.loads(_read("FXDataEngine/Tools/plugin_oracles.json"))
    plugins = data["plugins"]
    for plugin_name in CONVERTED_MODEL_CASES:
        entry = plugins.get(plugin_name)
        assert entry, plugin_name
        assert entry.get("identity")
        assert entry.get("recommendations")


def test_router_config_knows_new_regime_and_trend_plugins() -> None:
    data = json.loads(_read("FXDataEngine/Tools/OfflineLab/AdaptiveRouter/adaptive_router_config.json"))
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


def test_swift_model_framework_does_not_call_mt5_market_data() -> None:
    forbidden = (
        "Copy" + "Rates(",
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
    for path in (ROOT / "FXDataEngine/Sources/FXDataEngine").rglob("*.swift"):
        text = path.read_text(encoding="utf-8")
        for token in forbidden:
            assert token not in text, f"{path.relative_to(ROOT).as_posix()} bypasses FXDatabase with {token}"
