# Market Microstructure and Order-Flow Proxy Layer

## Repo Mapping

- Existing shared control-plane subsystems already follow a consistent FXAI pattern:
  - producer under `Tools/OfflineLab/offline_lab/*` or `FXDataEngine/Services/*`
  - runtime artifact in `FILE_COMMON/FXAI/Runtime`
  - MQL runtime adapter under `FXDataEngine/Engine/Runtime/Trade/*`
  - GUI reader under `FXGUI/Sources/FXGUICore/Services/*`
  - GUI surface under `FXGUI/Sources/FXGUIApp/Features/*`
- The FX-only tradable universe is already stored in Offline Lab metadata and modeled in `Tools/OfflineLab/offline_lab/market_universe.py`.

## Phase-1 Design Choice

The microstructure engine is implemented as an **FXDatabase Service** during the runtime transition because only FXDatabase currently has direct, low-latency access to broker-visible tick and price-cost behavior. Python is used for:

- config generation and validation
- service installation / compile automation
- replay report generation
- local operator status artifacts

## Files Added / Changed

  Terminal-wide FXDatabase collector that samples tradable FX symbols, computes rolling proxy features, and writes shared runtime artifacts.
- `Tools/OfflineLab/offline_lab/microstructure_contracts.py`
  Paths, schema constants, runtime artifact locations, and status/report paths.
- `Tools/OfflineLab/offline_lab/microstructure_config.py`
  Default config, threshold validation, and runtime TSV export for the FXDatabase service.
- `Tools/OfflineLab/offline_lab/microstructure_service.py`
  Config validation, local status sync, and FXDatabase service install/compile automation.
- `Tools/OfflineLab/offline_lab/microstructure_replay.py`
  History replay summarizer for operator review and audit alignment.
  MQL runtime reader for the shared microstructure flat artifact plus risk-side globals.
- GUI reader/model/view additions under `FXGUI/Sources/.../Microstructure*`
  Operator-visible microstructure diagnostics integrated into the existing GUI shell.

## Repo-Specific Deviations From the Original Concept

- The phase-1 collector focuses on **proxy features computed from broker-visible FXDatabase tick/quote behavior**. It explicitly does **not** claim true institutional order-book or dealer inventory visibility.
- The GUI can read either:
  - the project-local microstructure status artifact under `Tools/OfflineLab/Microstructure`, or
  - the live runtime artifact path in `FILE_COMMON/FXAI/Runtime` when running in the live FXDatabase tree.
- Phase 1 integrates microstructure primarily as:
  - execution/risk gating
  - adaptive-router regime enrichment
  - operator diagnostics

It does **not** force a canonical model-vector expansion.
