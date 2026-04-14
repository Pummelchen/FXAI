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
    ]
    for token in required_tokens:
        assert token in feature_norm


def test_training_pipeline_uses_horizon_aware_fit_and_payload_transform():
    engine_training = _read("Engine/engine_training.mqh")
    assert "FXAI_FitFeatureNormalizationMethodForRange(method_id," in engine_training
    assert "FXAI_ApplyFeatureNormalizationEx(norm_method," in engine_training
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
