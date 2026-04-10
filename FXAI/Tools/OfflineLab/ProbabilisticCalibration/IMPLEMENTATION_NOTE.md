# Probabilistic Calibration + Abstention Layer

## Repo Mapping

This subsystem is integrated as a final decision-quality layer in the existing FXAI runtime path instead of replacing the existing routing stack.

### Runtime boundary

- Existing upstream decision sources already present in repo:
  - `Engine/Runtime/runtime_model_stage_block.mqh`
  - `Engine/Runtime/runtime_policy_stage_block.mqh`
  - `Engine/Runtime/runtime_dynamic_ensemble_stage.mqh`
  - `Engine/Runtime/runtime_adaptive_router_stage.mqh`
- Existing cost/execution path reused:
  - `Engine/Core/core_broker_execution.mqh`
- Existing shared context sources consumed, not duplicated:
  - `Engine/Runtime/Trade/runtime_trade_newspulse.mqh`
  - `Engine/Runtime/Trade/runtime_trade_rates_engine.mqh`
  - `Engine/Runtime/Trade/runtime_trade_microstructure.mqh`

### Chosen insertion point

The probabilistic calibration stage is inserted after:

1. raw plugin predictions
2. stack/meta-policy blending
3. dynamic ensemble weighting
4. adaptive router posture application

and before:

1. final runtime artifact emission
2. final signal finalization
3. execution/risk handling that consumes `g_ai_last_*` and `g_policy_last_*`

This preserves the current plugin and router APIs while letting the new subsystem:

- calibrate ensemble-level directional probabilities
- compute cost-aware expected edge
- add explicit uncertainty penalties
- force `SKIP` when edge does not clear cost and uncertainty

### Artifact pattern reused

This subsystem follows the existing control-plane subsystem convention already used by:

- NewsPulse
- Rates Engine
- Microstructure
- Adaptive Router
- Dynamic Ensemble

Artifacts added:

- `FILE_COMMON/FXAI/Runtime/prob_calibration_config.tsv`
- `FILE_COMMON/FXAI/Runtime/prob_calibration_memory.tsv`
- `FILE_COMMON/FXAI/Runtime/fxai_prob_calibration_<SYMBOL>.tsv`
- `FILE_COMMON/FXAI/Runtime/fxai_prob_calibration_history_<SYMBOL>.ndjson`
- `Tools/OfflineLab/ProbabilisticCalibration/prob_calibration_config.json`
- `Tools/OfflineLab/ProbabilisticCalibration/prob_calibration_memory.json`
- `Tools/OfflineLab/ProbabilisticCalibration/Reports/prob_calibration_replay_report.json`

### Runtime behavior

The runtime stage:

- consumes final ensemble probabilities and expected-move summaries
- selects the most specific reliable calibrator tier
- applies a conservative logistic-affine probability mapping
- shrinks move estimates using calibrator memory
- prices spread, slippage, fill/risk, and uncertainty explicitly
- abstains when edge after costs does not clear the configured floor
- writes machine-readable reason codes and fallback flags

### Offline-Lab scope in phase 1

Phase 1 ships:

- config validation
- default conservative calibrator memory
- runtime TSV export
- replay-report rebuilding from append-only history
- a Python reference math module used by tests

Deferred to later phases:

- empirical calibrator fitting from full realized live ledger joins
- champion/challenger calibrator training jobs
- per-plugin calibration artifacts
- richer move-distribution fitting

### GUI integration

The GUI gets a dedicated surface following the existing dedicated-page pattern used by Dynamic Ensemble:

- core models in `GUI/Sources/FXAIGUICore/Models`
- artifact reader in `GUI/Sources/FXAIGUICore/Services`
- page in `GUI/Sources/FXAIGUIApp/Features`

The page exposes:

- raw vs calibrated probabilities
- expected move distribution
- cost and uncertainty breakdown
- final action / abstain state
- fallback tier and support level
- top abstention reasons
- replay counts from append-only history
