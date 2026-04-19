# FXAI Promotion Criteria

These are the exact release-gate thresholds exported from `Tools/testlab/release_gate.py`. Public benchmark cards and release notes should be interpreted against this policy, not against ad hoc screenshots.

## Core Admission Thresholds

- Minimum audit score: 70.0
- Minimum cross-symbol stability: 0.55
- Require market replay pack: False
- Fail on unresolved issues: False

## Walkforward Thresholds

- PBO max: 0.45
- DSR(proxy) min: 0.35
- Pass-rate min: 0.55

## Adversarial Thresholds

- Score min: 68.0
- Calibration error max: 0.260
- Path-quality error max: 0.500

## Macro Dataset Thresholds

- Schema version min: 2
- Source trust min: 0.60
- Macro coverage min: 0.60
- Macro event-rate min: 0.06
- Provenance trust min: 0.55
- Currency relevance min: 0.45
- Revision-chain coverage min: 1

## Runtime Performance Thresholds

- Predict mean max: 1.600 ms
- Update mean max: 1.250 ms
- Working set max: 4096.0 KB
- Runtime total mean max: 12.000 ms

## Artifact Size Budgets

- `fxai_attribution`: 32768 bytes max
- `fxai_live_deploy`: 24576 bytes max
- `fxai_perf`: 131072 bytes max
- `fxai_student_router`: 24576 bytes max
- `fxai_supervisor_command`: 16384 bytes max
- `fxai_supervisor_service`: 24576 bytes max
- `fxai_world_plan`: 32768 bytes max
