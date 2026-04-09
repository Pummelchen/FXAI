# Market Microstructure and Order-Flow Proxy Layer

## Repo Mapping

- Live broker-side tick access already exists only on the MT5 side, primarily through `SymbolInfoTick()` and the `MqlTick` / `CopyTicksRange()` APIs.
- Existing shared control-plane subsystems already follow a consistent FXAI pattern:
  - producer under `Tools/offline_lab/*` or `Services/*`
  - runtime artifact in `FILE_COMMON/FXAI/Runtime`
  - MQL runtime adapter under `Engine/Runtime/Trade/*`
  - GUI reader under `GUI/Sources/FXAIGUICore/Services/*`
  - GUI surface under `GUI/Sources/FXAIGUIApp/Features/*`
- Trade gating composition already lives in `Engine/Runtime/Trade/runtime_trade_risk.mqh`.
- Adaptive regime routing already computes a coarse `liquidity_stress` heuristic in `Engine/Runtime/runtime_adaptive_router_stage.mqh`, so the new microstructure layer can upgrade that context instead of inventing a parallel risk concept.
- The FX-only tradable universe is already stored in Offline Lab metadata and modeled in `Tools/offline_lab/market_universe.py`.

## Phase-1 Design Choice

The microstructure engine is implemented as an **MT5 Service** instead of a Python daemon because only MT5 has direct, low-latency access to broker-visible tick and spread behavior. Python is used for:

- config generation and validation
- service installation / compile automation
- replay report generation
- local operator status artifacts

## Files Added / Changed

- `Services/FXAI_MicrostructureProbe.mq5`
  Terminal-wide MT5 collector that samples tradable FX symbols, computes rolling proxy features, and writes shared runtime artifacts.
- `Tools/offline_lab/microstructure_contracts.py`
  Paths, schema constants, runtime artifact locations, and status/report paths.
- `Tools/offline_lab/microstructure_config.py`
  Default config, threshold validation, and runtime TSV export for the MT5 service.
- `Tools/offline_lab/microstructure_service.py`
  Config validation, local status sync, and MT5 service install/compile automation.
- `Tools/offline_lab/microstructure_replay.py`
  History replay summarizer for operator review and audit alignment.
- `Engine/Runtime/Trade/runtime_trade_microstructure.mqh`
  MQL runtime reader for the shared microstructure flat artifact plus risk-side globals.
- GUI reader/model/view additions under `GUI/Sources/.../Microstructure*`
  Operator-visible microstructure diagnostics integrated into the existing GUI shell.

## Repo-Specific Deviations From the Original Concept

- The phase-1 collector focuses on **proxy features computed from broker-visible MT5 tick/quote behavior**. It explicitly does **not** claim true institutional order-book or dealer inventory visibility.
- The GUI can read either:
  - the project-local microstructure status artifact under `Tools/OfflineLab/Microstructure`, or
  - the live runtime artifact path in `FILE_COMMON/FXAI/Runtime` when running in the live MT5 tree.
- Phase 1 integrates microstructure primarily as:
  - execution/risk gating
  - adaptive-router regime enrichment
  - operator diagnostics

It does **not** force a canonical model-vector expansion.
