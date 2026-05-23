# Adaptive Regime Classifier + Plugin Router

This note maps the subsystem design onto the actual FXAI codebase.

## Real Integration Points

- Existing shadow-fleet / research telemetry: `<FXAI_ROOT>/Tools/OfflineLab/offline_lab/shadow_fleet.py`
- Existing student-router artifact generation: `<FXAI_ROOT>/Tools/OfflineLab/offline_lab/student_router.py`
- Existing operator dashboard export: `<FXAI_ROOT>/Tools/OfflineLab/offline_lab/dashboard.py`
- Existing GUI runtime reader and monitor: `<FXAI_ROOT>/FXGUI/Sources/FXGUICore/Services/RuntimeArtifactReader.swift` and `<FXAI_ROOT>/FXGUI/Sources/FXGUIApp/Features/Runtime/RuntimeMonitorView.swift`

## Repo-Aware Design Choice

FXAI already has two useful routing layers:

1. the `student_router` control-plane profile emitted by Offline Lab

The new subsystem extends those rails instead of replacing them.

## What Will Be Added

### Offline Lab / Turso side

- A new Adaptive Router config and contract layer under `Tools/OfflineLab/offline_lab`
- A new `adaptive_router_profiles` Turso table
- Profile generation that combines:
  - static family/plugin regime priors
  - pair-aware priors
  - rolling empirical adaptation from `shadow_fleet_observations`
  - promotion/champion state
- Promotion artifacts:
  - `fxai_adaptive_router_<symbol>.tsv` in `FILE_COMMON/FXAI/Offline/Promotions`
  - `adaptive_router_<symbol>.json` in `Tools/OfflineLab/ResearchOS/<profile>/`
- Replay/report generation from append-only runtime history

### MQL runtime side

- A new live regime state object with:
  - compact regime taxonomy
  - normalized probabilities
  - top label
  - confidence
  - machine-readable reasons
- A new adaptive router profile loader beside the existing student-router loader
- Router weighting that multiplies:
  - existing meta score
  - student-router family/plugin gates
  - adaptive-router regime/plugin suitability
- Posture output:
  - `NORMAL`
  - `CAUTION`
  - `ABSTAIN_BIAS`
  - `BLOCK`
- NewsPulse-aware abstention and suppression
- Live runtime artifacts:
  - `fxai_regime_router_<symbol>.tsv`
  - `fxai_regime_router_history_<symbol>.ndjson`

### GUI side

- A new regime/router surface inside the existing GUI shell
- Runtime monitor enrichment using:
  - current regime
  - confidence/probabilities
  - active plugins and weights
  - suppressed plugins
  - current posture
  - routing reasons
  - live history/replay visibility

## Intentional Deviations From The Generic Plan

- The live regime classifier currently runs inside the runtime MQL runtime instead of Python. During the Swift migration, its broker-cost input is represented as price-cost state alongside volatility, session, and runtime NewsPulse consumption.
- The offline Python side will not classify live regime directly. It will generate the adaptive priors, thresholds, and replay reports that the runtime consumes.
- Existing plugin interfaces will remain unchanged. Routing is applied above plugin inference at the current ensemble/meta score stage.
- The existing 12 internal `FXAI_GetRegimeId(...)` buckets are retained as low-level calibration state. The new subsystem adds a higher-level operator-facing regime taxonomy on top of them.

## Phase-1 Safety

- Feature-flagged and fallback-safe
- No forced retraining of plugin internals
- No change to canonical plugin request payloads
- Can fall back to current `student_router + meta score + risk gating` behavior if disabled
