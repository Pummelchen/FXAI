# Multi-Horizon Label Engine Implementation Note

## Repo Mapping

- Canonical training data source: `<FXAI_ROOT>/Tools/offline_lab/exporter.py`
- Canonical dataset tables: `datasets` and `dataset_bars` in `<FXAI_ROOT>/Tools/offline_lab/common_schema.py`
- Existing offline orchestration entry points: `<FXAI_ROOT>/Tools/offline_lab/cli_parser.py` and `<FXAI_ROOT>/Tools/offline_lab/cli_commands.py`
- Existing cost/execution context reused by label construction:
  - `<FXAI_ROOT>/Tools/testlab/shared.py`
  - `<FXAI_ROOT>/Tools/offline_lab/prob_calibration_math.py`
  - `<FXAI_ROOT>/Tools/offline_lab/execution_quality_math.py`
- Existing operator/reporting surfaces extended:
  - `<FXAI_ROOT>/Tools/offline_lab/dashboard.py`
  - `<FXAI_ROOT>/GUI/Sources/FXAIGUIApp`

## What Was Added

- A dedicated offline-lab subsystem rooted at `<FXAI_ROOT>/Tools/OfflineLab/LabelEngine`
- Deterministic multi-horizon side-aware labels over exported `dataset_bars`
- Cost-aware tradeability labels driven by spread, slippage, fill penalty, commission, and safety margin
- Path-aware barrier timing and MFE/MAE labeling using bar-path approximation
- Meta-label generation over either:
  - baseline momentum candidates
  - externally supplied candidate files
- Artifact persistence in both:
  - DB metadata via `label_engine_artifacts`
  - file artifacts under `Tools/OfflineLab/LabelEngine/Artifacts`
- Dashboard and GUI integration through a shared report artifact

## Repo-Specific Deviations From The Prompt

- There was no existing clean candidate-signal log tied to offline dataset bars, so phase 1 implements:
  - a deterministic baseline candidate generator
  - an external candidate ingest path
- The export runner currently stores M1 OHLC and `spread_points`, but not symbol digit metadata. Point-size resolution is therefore:
  - overrideable in config
  - heuristically inferred per symbol otherwise
- Path-aware labels are built from M1 bar geometry, not tick-level path, so every row records that a path approximation was used.

## Integration Boundary

- Phase 1 does **not** rewrite plugin training internals.
- Phase 1 does make the label engine available as a reusable artifact layer for:
  - offline evaluation
  - future plugin target specialization
  - calibration and abstention target research
  - GUI and dashboard visibility
