# Rates / Term-Structure / Policy-Path Engine: Implementation Note

## Repo Mapping

- Python artifact engine lives under `<FXAI_ROOT>/Tools/offline_lab`.
- Shared runtime artifacts live under `FILE_COMMON/FXAI/Runtime`, matching NewsPulse and Adaptive Router.
- Runtime consumption lands in `<FXAI_ROOT>/Engine/Runtime/Trade`.
- GUI integration follows the existing artifact-reader pattern under `<FXAI_ROOT>/GUI/Sources/FXAIGUICore/Services` and `<FXAI_ROOT>/GUI/Sources/FXAIGUIApp/Features`.
- Operator/dashboard integration follows `<FXAI_ROOT>/Tools/offline_lab/dashboard.py`.

## Phase-1 Repo Reality

The repo does not currently contain a native live rates, bond, swap, or OIS ingestion layer. It does already contain:

- NewsPulse event timing, official central-bank feed coverage, and scheduled macro state
- runtime gating hooks that already consume shared flat TSV artifacts
- append-only history and replay patterns
- GUI readers and operator surfaces for shared control-plane subsystems

That means phase 1 must be honest about two input modes:

1. **True-market numeric mode**
   - operator-supplied front-end / expected-path / curve inputs via a local JSON file
   - auditable, simple, and swappable later for better providers

2. **Policy-proxy mode**
   - uses NewsPulse event timing, official-feed items, scheduled surprise proxies, and policy-topic clustering
   - computes stable policy-repricing, policy-surprise, uncertainty, divergence, and risk states even when true numeric rates inputs are unavailable

## What This Implementation Adds

- a new Rates Engine artifact subsystem:
  - `rates_snapshot.json`
  - `rates_snapshot_flat.tsv`
  - `rates_history.ndjson`
  - `rates_symbol_map.tsv`
- config and operator-editable numeric provider inputs under `Tools/OfflineLab/RatesEngine`
- a rates daemon / validate / once / replay command set in the Offline Lab CLI
- runtime pair-state gating through a lightweight MQL adapter
- NewsPulse enrichment using rates-aware context
- a dedicated macOS GUI Rates Engine surface

## Repo-Specific Deviations From The Idealized Prompt

- Because the repo does not already have a supported live OIS/yield feed, phase 1 uses a provider abstraction and defaults to proxy mode when operator-supplied numeric inputs are absent.
- Phase 1 does not expand the canonical model feature vector. It integrates through runtime gating, NewsPulse enrichment, GUI observability, and replay history.
- Phase 1 keeps curve-shape fields nullable when true curve inputs are unavailable instead of inventing false precision.
