# FXAI

FXAI is a professional MetaTrader 5 framework for building, testing, and operating AI-driven FX trading systems fully inside MT5 and MQL5.

It is not a single strategy EA. It is a research and deployment framework with:
- a plugin-based prediction layer
- a shared `M1 OHLC + spread` data contract
- an internal `TensorCore` runtime for stronger neural and sequence models
- an Audit Lab for certification, regression checks, and release gating
- a Turso-backed Offline Lab for export, tuning, promotion, champion-challenger control loops, branch/PITR workflows, audit-log ingestion, and vector-backed analog retrieval
- foundation-teacher and student-deployment bundles, autonomous governance, supervisor-service artifacts, and live deployment profiles for promoted MT5 runtime behavior
- one workflow for research, backtesting, audit, and live operation

FXAI stays MT5-native. There are no external inference services and no DLL dependency in live trading.

An optional macOS 26 operator GUI now lives under [FXAI/GUI](/Users/andreborchert/FXAI-main2/FXAI/GUI). It is a SwiftUI-based helper surface for dashboards, plugin-zoo browsing, report discovery, Audit/Offline/backtest run building, and terminal-first workflow guidance. The terminal remains the primary control path.

## Why It Matters

### For Traders
- Compare many model families under one execution shell instead of running isolated EA experiments.
- Test under realistic FX costs, skip logic, and execution stress instead of optimistic toy assumptions.
- Use Audit Lab and release gates to reject weak models before cloud optimization or live deployment.
- Keep live behavior closer to research with shared persistence, broker replay, macro-data guards, portfolio-aware control-plane signals, policy-driven lifecycle control, and promoted supervisor-service constraints.
- Promote stronger parameter packs, router-pruning profiles, supervisor-service budgets, and live deployment profiles from the Offline Lab without manual copy and paste.

### For Trade System Architects
- One codebase for data, features, normalization, model plugins, routing, audit, and live execution.
- Clear contracts for plugins, persistence, checkpoint depth, and promotion readiness.
- Shared TensorCore, transfer backbone, contextual routing, policy-first gating, and portfolio-aware meta scoring reduce duplicated model infrastructure.
- Runtime manifests, feature governance, and macro-data leakage guards make the framework auditable and reproducible.
- Turso/libSQL experiment ledgers, teacher-factory artifacts, fitted deployment models, shadow-fleet telemetry, and deployment profiles provide a serious research OS around MT5 instead of ad hoc backtest folders.
- Portfolio-supervisor artifacts, supervisor-service snapshots, causal attribution and pruning profiles, world-simulator plans, and foundation or student payloads let the research OS steer live gating, peer-pressure handling, lifecycle actions, and future distillation without manual copy and paste.
- The Offline Lab can run local-only through libSQL or as a Turso embedded replica when `TURSO_DATABASE_URL` and `TURSO_AUTH_TOKEN` are configured, with optional local database encryption, sync interval control, branch workflows, and audit-log ingestion.

## Documentation

Detailed documentation is kept in the wiki:
- [Home](https://github.com/Pummelchen/FXAI/wiki)
- [Getting Started](https://github.com/Pummelchen/FXAI/wiki/Getting-Started)
- [FXAI Framework](https://github.com/Pummelchen/FXAI/wiki/FXAI-Framework)
- [Audit Lab](https://github.com/Pummelchen/FXAI/wiki/Audit-Lab)
- [Offline Lab](https://github.com/Pummelchen/FXAI/wiki/Offline-Lab)
- [GUI](https://github.com/Pummelchen/FXAI/wiki/GUI)
- [Project Structure](https://github.com/Pummelchen/FXAI/wiki/Project-Structure)
- [Data Policy](https://github.com/Pummelchen/FXAI/wiki/Data-Policy)

Quick start instructions and current architecture highlights now live in the wiki so the repo front page can stay focused on project value and positioning.

The synced MT5 subtree also includes its own local operator guide at [FXAI/README.md](/Users/andreborchert/FXAI-main2/FXAI/README.md).
