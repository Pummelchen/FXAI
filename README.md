# FXAI

FXAI is a professional MetaTrader 5 framework for building, testing, and operating AI-driven FX trading systems fully inside MT5 and MQL5.

It is not a single strategy EA. It is a research and deployment framework with:
- a plugin-based prediction layer
- a shared `M1 OHLC + spread` data contract
- an internal `TensorCore` runtime for stronger neural and sequence models
- an Audit Lab for certification, regression checks, and release gating
- one workflow for research, backtesting, audit, and live operation

FXAI stays MT5-native. There are no external inference services and no DLL dependency in live trading.

## Why It Matters

### For Traders
- Compare many model families under one execution shell instead of running isolated EA experiments.
- Test under realistic FX costs, skip logic, and execution stress instead of optimistic toy assumptions.
- Use Audit Lab and release gates to reject weak models before cloud optimization or live deployment.
- Keep live behavior closer to research with shared persistence, broker replay, macro-data guards, and execution controls.

### For Trade System Architects
- One codebase for data, features, normalization, model plugins, routing, audit, and live execution.
- Clear contracts for plugins, persistence, checkpoint depth, and promotion readiness.
- Shared TensorCore, transfer backbone, contextual routing, and portfolio-aware meta scoring reduce duplicated model infrastructure.
- Runtime manifests, feature governance, and macro-data leakage guards make the framework auditable and reproducible.

## Current Architecture Highlights
- Canonical market input is `M1 OHLC + spread`.
- Shared transfer warmup now uses a deeper temporal backbone across symbols, horizons, sessions, and rolling windows.
- Stateful plugin promotion is gated by native checkpoint coverage and runtime persistence manifests.
- Ensemble routing now uses contextual regret, counterfactual state, and portfolio-objective signals.
- Broker execution replay persists richer trace state for runtime and audit reuse.
- Macro-event data uses the hardened schema v2 contract with provenance and leakage checks.

## Quick Start

Source of truth for runnable code:
- live MT5 tree: `MQL5/Experts/FXAI`

Versioned mirror:
- git repo: `FXAI/`

Compile from repo root:

```bash
cd /Users/andreborchert/FXAI-main2
python3 FXAI/Tools/fxai_testlab.py compile-main
python3 FXAI/Tools/fxai_testlab.py compile-audit
```

Focused audit example:

```bash
python3 FXAI/Tools/fxai_testlab.py run-audit \
  --plugin-list "{ai_mlp}" \
  --scenario-list "{market_recent, market_walkforward, market_macro_event}" \
  --symbol EURUSD
```

## Documentation

Detailed documentation is kept in the wiki:
- [Home](https://github.com/Pummelchen/FXAI/wiki)
- [Getting Started](https://github.com/Pummelchen/FXAI/wiki/Getting-Started)
- [FXAI Framework](https://github.com/Pummelchen/FXAI/wiki/FXAI-Framework)
- [Audit Lab](https://github.com/Pummelchen/FXAI/wiki/Audit-Lab)
- [Project Structure](https://github.com/Pummelchen/FXAI/wiki/Project-Structure)
- [Data Policy](https://github.com/Pummelchen/FXAI/wiki/Data-Policy)

The synced MT5 subtree also includes its own local operator guide at [FXAI/README.md](/Users/andreborchert/FXAI-main2/FXAI/README.md).
