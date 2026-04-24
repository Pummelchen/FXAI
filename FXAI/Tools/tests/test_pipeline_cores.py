from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _read(rel_path: str) -> str:
    return (ROOT / rel_path).read_text(encoding="utf-8")


def test_data_pipeline_exposes_dedicated_cores_and_contracts():
    data_pipeline = _read("Engine/data_pipeline.mqh")
    required_includes = [
        '#include "Core\\core_pipeline_contracts.mqh"',
        '#include "market_data_gateway.mqh"',
        '#include "Core\\core_data_core.mqh"',
        '#include "Core\\core_feature_core.mqh"',
        '#include "Core\\core_normalization_core.mqh"',
    ]
    for token in required_includes:
        assert token in data_pipeline


def test_core_pipeline_contracts_define_stage_structures():
    contracts = _read("Engine/Core/core_pipeline_contracts.mqh")
    required_tokens = [
        "struct FXAIDataCoreRequest",
        "struct FXAIDataCoreBundle",
        "struct FXAIFeatureCoreRequest",
        "struct FXAIFeatureCoreFrame",
        "struct FXAINormalizationCoreFrame",
        "struct FXAINormalizationPayloadRequest",
        "struct FXAINormalizationPayloadFrame",
        "struct FXAIContextSeries",
        "struct FXAIPreparedSample",
    ]
    for token in required_tokens:
        assert token in contracts


def test_core_stage_files_expose_unified_api():
    data_core = _read("Engine/Core/core_data_core.mqh")
    feature_core = _read("Engine/Core/core_feature_core.mqh")
    normalization_core = _read("Engine/Core/core_normalization_core.mqh")

    for token in [
        "void FXAI_DataCoreResetRequest(",
        "void FXAI_DataCoreInitRequest(",
        "bool FXAI_DataCoreAddContextSymbol(",
        "void FXAI_DataCoreCaptureGlobalContextSymbols(",
        "bool FXAI_DataCoreLoadBundleFromRequest(",
        "bool FXAI_DataCoreLoadHistoryBundle(",
        "bool FXAI_DataCoreRefreshLiveBundle(",
        "void FXAI_DataCoreBindArrayBundle(",
    ]:
        assert token in data_core

    assert "void FXAI_FeatureCoreResetRequest(" in feature_core
    assert "bool FXAI_FeatureCoreBuildFrame(" in feature_core
    assert "bool FXAI_FeatureCoreBuildFrameFromBundle(" in feature_core
    assert "bool FXAI_NormalizationCoreBuildInputFrameFromFeatureFrame(" in normalization_core
    assert "bool FXAI_NormalizationCoreBuildPayloadFrame(" in normalization_core
    assert "bool FXAI_NormalizationCoreFinalizePredictRequest(" in normalization_core
    assert "bool FXAI_NormalizationCoreFinalizeTrainRequest(" in normalization_core


def test_runtime_training_and_audit_paths_use_pipeline_cores():
    runtime_block = _read("Engine/Runtime/runtime_feature_pipeline_block.mqh")
    engine_samples = _read("Engine/engine_samples.mqh")
    engine_training = _read("Engine/engine_training.mqh")
    audit_samples = _read("Tests/audit_samples.mqh")
    warmup_entry = _read("Engine/Warmup/warmup_entrypoint.mqh")

    runtime_tokens = [
        "static FXAIDataCoreBundle live_bundle;",
        "FXAI_DataCoreRefreshLiveBundle(live_bundle,",
        "FXAI_FeatureCoreBuildFrameFromBundle(live_bundle, 0, H, norm_method, predict_feature_frame)",
    ]
    for token in runtime_tokens:
        assert token in runtime_block

    samples_tokens = [
        "bool FXAI_PrepareTrainingSampleFromBundle(",
        "FXAI_FeatureCoreBuildFrameFromBundle(bundle, i, H, norm_method, feature_frame)",
        "FXAI_NormalizationCoreBuildInputFrameFromFeatureFrame(feature_frame, norm_frame)",
        "FXAI_DataCoreBindArrayBundle(snapshot,",
    ]
    for token in samples_tokens:
        assert token in engine_samples

    training_tokens = [
        "FXAI_PrepareTrainingSampleFromBundle(bundle,",
        "FXAI_DataCoreBindArrayBundle(snapshot,",
        "FXAI_FeatureCoreBuildFrameFromBundle(bundle, 0, horizon_minutes, norm_method, feature_frame)",
        "FXAI_NormalizationCoreBuildInputFrameFromFeatureFrame(feature_frame, norm_frame)",
        "FXAI_NormalizationCoreFinalizeTrainRequest(manifest, s3)",
    ]
    for token in training_tokens:
        assert token in engine_training

    audit_tokens = [
        "FXAI_DataCoreBindArrayBundle(snapshot,",
        "FXAI_FeatureCoreBuildFrameFromBundle(bundle,",
        "FXAI_NormalizationCoreBuildInputFrameFromFeatureFrame(feature_frame, norm_frame)",
    ]
    for token in audit_tokens:
        assert token in audit_samples

    assert "FXAI_DataCoreLoadHistoryBundle(symbol," in warmup_entry


def test_shortcut_paths_are_removed_from_pipeline_callers():
    warmup_transfer = _read("Engine/Warmup/warmup_transfer.mqh")
    meta_reliability = _read("Engine/meta_reliability.mqh")
    feature_norm = _read("Engine/feature_norm.mqh")
    direct_payload_callers = [
        "Engine/engine_training.mqh",
        "Engine/Runtime/runtime_model_stage_block.mqh",
        "Engine/Warmup/warmup_scoring.mqh",
        "Engine/Warmup/warmup_transfer.mqh",
        "Engine/Warmup/warmup_normalization.mqh",
        "Engine/Warmup/warmup_portfolio.mqh",
        "Engine/Lifecycle/lifecycle_bootstrap.mqh",
        "Engine/Lifecycle/lifecycle_compliance.mqh",
        "Tests/Scoring/audit_scoring_run.mqh",
        "Tests/Scoring/audit_scoring_adversarial.mqh",
    ]

    assert "FXAI_LoadSeriesOptionalCached(" not in warmup_transfer
    assert "FXAI_LoadRatesOptional(" not in warmup_transfer
    assert "FXAI_UpdateRatesRolling(" not in meta_reliability
    assert "FXAI_ComputeFeatureVector(" not in feature_norm

    for rel_path in direct_payload_callers:
        assert "FXAI_ApplyPayloadTransformPipelineEx(" not in _read(rel_path)


def test_audit_harness_provides_context_shim_for_pipeline_cores():
    audit_core = _read("Tests/audit_core.mqh")
    audit_utils = _read("Tests/audit_utils.mqh")

    assert '#include "audit_utils.mqh"' in audit_core
    assert '#include "..\\Engine\\data_pipeline.mqh"' in audit_core
    assert audit_core.index('#include "audit_utils.mqh"') < audit_core.index('#include "..\\Engine\\data_pipeline.mqh"')
    assert "#define FXAI_DISABLE_DYNAMIC_CONTEXT_API 1" in audit_utils
