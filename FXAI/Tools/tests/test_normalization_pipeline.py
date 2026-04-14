from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _read(rel_path: str) -> str:
    return (ROOT / rel_path).read_text(encoding="utf-8")


def test_normalization_enum_exposes_exact_buffer_variants():
    core = _read("Engine/core.mqh")
    assert "#define FXAI_NORM_METHOD_COUNT 17" in core
    assert "FXAI_NORM_MINMAX_BUFFER2" in core
    assert "FXAI_NORM_MINMAX_BUFFER3" in core
    assert "#define FXAI_UNIT_RANGE_FLOOR 0.0001" in core
    assert "#define FXAI_UNIT_RANGE_CEIL 0.9999" in core
    assert "#define FXAI_SIGNED_UNIT_RANGE_FLOOR -0.9999" in core
    assert "#define FXAI_SIGNED_UNIT_RANGE_CEIL 0.9999" in core
    assert "double FXAI_ClampUnitOpen(const double v)" in core
    assert "double FXAI_ClampSignedUnitOpen(const double v)" in core


def test_feature_norm_implements_fitted_and_adaptive_layers():
    feature_norm = _read("Engine/feature_norm.mqh")
    required_tokens = [
        "#define FXAI_NORM_QUANTILE_KNOTS 17",
        "bool g_fxai_norm_fit_ready[FXAI_MAX_HORIZONS][FXAI_NORM_METHOD_COUNT];",
        "bool FXAI_MethodUsesFittedStats(",
        "bool FXAI_MethodUsesAdaptivePayloadNormalization(",
        "bool FXAI_FitFeatureNormalizationMethodForRange(",
        "void FXAI_ApplyFeatureNormalizationEx(",
        "void FXAI_ApplyPayloadAdaptiveNormalization(",
        "double FXAI_SignedLog1P(",
        "return FXAI_ClampUnitOpen((v - lo_b) / den);",
        "out_v = (has_prev && cur > prev ? FXAI_UNIT_RANGE_CEIL : FXAI_UNIT_RANGE_FLOOR);",
    ]
    for token in required_tokens:
        assert token in feature_norm


def test_bounded_feature_emitters_use_open_interval_clamps():
    feature_math = _read("Engine/feature_math.mqh")
    feature_build = _read("Engine/feature_build.mqh")
    event_macro = _read("Engine/event_macro.mqh")

    math_tokens = [
        "upper_wick_norm = FXAI_ClampUnitOpen(upper_wick / den_range);",
        "lower_wick_norm = FXAI_ClampUnitOpen(lower_wick / den_range);",
    ]
    for token in math_tokens:
        assert token in feature_math

    build_tokens = [
        "return FXAI_ClampUnitOpen(1.0 - d / radius);",
        "return FXAI_ClampSignedUnitOpen(0.60 * asia_to_eu + 0.80 * eu_to_us - 0.70 * us_to_roll);",
        "return FXAI_ClampUnitOpen(0.70 * overlap + 0.30 * FXAI_CyclicHourPulse(hour_value, 15.0, 2.0));",
        "triple_swap_bias = FXAI_ClampSignedUnitOpen(day_bias * (0.35 + 0.65 * roll_bias));",
        "spread_rank20 = FXAI_ClampSignedUnitOpen(2.0 * ((double)rank_le / (double)MathMax(used, 1)) - 1.0);",
        "features[12] = FXAI_ClampSignedUnitOpen((ctx_up_ratio - 0.5) * 2.0);",
        "features[15] = FXAI_ClampSignedUnitOpen(((double)weekday - 3.0) / 2.0);",
        "features[16] = FXAI_ClampSignedUnitOpen(((double)hh - 11.5) / 11.5);",
        "features[17] = FXAI_ClampSignedUnitOpen(((double)mm - 29.5) / 29.5);",
        "features[40] = FXAI_ClampSignedUnitOpen((rsi14 - 50.0) / 50.0);",
        "features[base_f + 3] = FXAI_ClampSignedUnitOpen(ctx_corr);",
        "features[62] = FXAI_ClampSignedUnitOpen(shared_util);",
        "features[63] = FXAI_ClampSignedUnitOpen(2.0 * shared_stability - 1.0);",
        "features[64] = FXAI_ClampSignedUnitOpen(2.0 * shared_lead - 1.0);",
        "features[65] = FXAI_ClampSignedUnitOpen(2.0 * shared_coverage - 1.0);",
        "features[FXAI_MACRO_EVENT_FEATURE_OFFSET + 14] = FXAI_ClampSignedUnitOpen(macro_state.policy_divergence);",
        "features[FXAI_MACRO_EVENT_FEATURE_OFFSET + 15] = FXAI_ClampSignedUnitOpen(macro_state.policy_pressure);",
        "features[FXAI_MACRO_EVENT_FEATURE_OFFSET + 16] = FXAI_ClampSignedUnitOpen(macro_state.inflation_pressure);",
        "features[FXAI_MACRO_EVENT_FEATURE_OFFSET + 17] = FXAI_ClampSignedUnitOpen(macro_state.labor_pressure);",
        "features[FXAI_MACRO_EVENT_FEATURE_OFFSET + 18] = FXAI_ClampSignedUnitOpen(macro_state.growth_pressure);",
        "features[FXAI_MACRO_EVENT_FEATURE_OFFSET + 19] = FXAI_ClampUnitOpen(macro_state.state_quality);",
        "features[83] = FXAI_ClampSignedUnitOpen(spread_rank20);",
    ]
    for token in build_tokens:
        assert token in feature_build

    macro_tokens = [
        "out.policy_divergence = FXAI_ClampSignedUnitOpen(policy_norm - 0.35 * inflation_norm + 0.20 * growth_norm);",
        "out.policy_pressure = FXAI_ClampSignedUnitOpen(0.70 * policy_norm + 0.30 * inflation_norm);",
        "out.inflation_pressure = FXAI_ClampSignedUnitOpen(inflation_norm);",
        "out.labor_pressure = FXAI_ClampSignedUnitOpen(labor_norm);",
        "out.growth_pressure = FXAI_ClampSignedUnitOpen(0.78 * growth_norm + 0.22 * trade_norm);",
        "out.carry_pressure = FXAI_ClampSignedUnitOpen(0.60 * out.policy_pressure +",
        "out.event_decay = FXAI_ClampUnitOpen(out.event_decay);",
        "double family_diversity = FXAI_ClampUnitOpen((double)family_hits / 5.0);",
        "double density = FXAI_ClampUnitOpen(coverage_weight / 2.0);",
        "out.state_quality = FXAI_ClampUnitOpen(0.34 * trust_mean +",
        "pre_embargo = FXAI_ClampUnitOpen(pre_embargo);",
        "post_embargo = FXAI_ClampUnitOpen(post_embargo);",
        "event_importance = FXAI_ClampUnitOpen(event_importance);",
        "event_class_bias = FXAI_ClampSignedUnitOpen(event_class_bias);",
        "currency_relevance = FXAI_ClampUnitOpen(currency_relevance);",
        "provenance_trust = FXAI_ClampUnitOpen(provenance_trust);",
        "rates_activity = FXAI_ClampUnitOpen(rates_activity);",
        "inflation_activity = FXAI_ClampUnitOpen(inflation_activity);",
        "labor_activity = FXAI_ClampUnitOpen(labor_activity);",
        "growth_activity = FXAI_ClampUnitOpen(growth_activity);",
    ]
    for token in macro_tokens:
        assert token in event_macro


def test_training_pipeline_uses_horizon_aware_fit_and_payload_transform():
    engine_training = _read("Engine/engine_training.mqh")
    assert "FXAI_FitFeatureNormalizationMethodForRange(method_id," in engine_training
    assert "FXAI_DataCoreBindArrayBundle(predict_snapshot," in engine_training
    assert "FXAI_FeatureCoreBuildFrameFromBundle(bundle, 0, horizon_minutes, norm_method, feature_frame)" in engine_training
    assert "FXAI_NormalizationCoreBuildInputFrameFromFeatureFrame(feature_frame, norm_frame)" in engine_training
    assert "FXAI_ApplyPayloadTransformPipelineEx(manifest.feature_schema_id," in engine_training
    assert "caches[sz].horizon_minutes = horizon_minutes;" in engine_training


def test_warmup_normalization_fits_candidates_on_train_only_split():
    warmup_norm = _read("Engine/Warmup/warmup_normalization.mqh")
    assert "bool FXAI_DeriveNormCandidateSplit(" in warmup_norm
    assert "train_start,\n                                                    train_end," in warmup_norm
    assert "FXAI_ScoreNormMethodCandidate(ai_idx,\n                                                      method_id," in warmup_norm


def test_runtime_artifacts_persist_normalization_fits():
    runtime_artifacts = _read("Engine/runtime_artifacts.mqh")
    assert "#define FXAI_RUNTIME_ARTIFACT_VERSION 15" in runtime_artifacts
    assert "FileWriteInteger(handle, (g_fxai_norm_fit_inited ? 1 : 0));" in runtime_artifacts
    assert "g_fxai_norm_fit_inited = (FileReadInteger(handle) != 0);" in runtime_artifacts


def test_tooling_accepts_new_normalization_id_range():
    promotion = _read("Tools/offline_lab/promotion.py")
    audit_run = _read("Tools/testlab/audit_run.py")
    assert "||0||0||16||N" in promotion
    assert "||0||0||16||N" in audit_run
