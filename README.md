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

## Documentation

Detailed documentation is kept in the wiki:
- [Home](https://github.com/Pummelchen/FXAI/wiki)
- [Getting Started](https://github.com/Pummelchen/FXAI/wiki/Getting-Started)
- [FXAI Framework](https://github.com/Pummelchen/FXAI/wiki/FXAI-Framework)
- [Audit Lab](https://github.com/Pummelchen/FXAI/wiki/Audit-Lab)
- [Offline Lab](https://github.com/Pummelchen/FXAI/wiki/Offline-Lab)
- [Project Structure](https://github.com/Pummelchen/FXAI/wiki/Project-Structure)
- [Data Policy](https://github.com/Pummelchen/FXAI/wiki/Data-Policy)

Quick start instructions and current architecture highlights now live in the wiki so the repo front page can stay focused on project value and positioning.

The synced MT5 subtree also includes its own local operator guide at [FXAI/README.md](/Users/andreborchert/FXAI-main2/FXAI/README.md).
