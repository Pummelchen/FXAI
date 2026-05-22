# FXAI

FXAI is a trading-system framework that has moved its data-engine, plugin, and offline backtest surface from the original MT5/MQL5 runtime into native Swift and Metal packages. The remaining MT5 bridge is the FXDatabase exporter EA, which exists only to collect M1 OHLCV data for the Swift stack.

FXAI is not a single black-box strategy. It is a governed decision framework: shared data contracts, model plugins, runtime risk layers, audit gates, promotion artifacts, and an optional GUI all work from the same source of truth.

FXBacktest and FXDatabase are now first-class FXAI subprojects included as ordinary tracked source in `FXBacktest/` and `FXDatabase/`. Clone FXAI normally; no submodule initialization is required for the Swift projects:

```bash
git clone https://github.com/Pummelchen/FXAI.git
```

## Swift Standard

All Swift projects and subprojects in FXAI use the current local Apple toolchain standard: Swift tools 6.3, Swift language mode 6, Xcode 26.5, and macOS 26 as the deployment floor. New Swift packages should follow that baseline unless the repo standard is intentionally upgraded.

## User Matrix Benefits

| User | Core Benefit | Start Here |
|---|---|---|
| Live Trader | See whether a trade is allowed, cautioned, blocked, or abstained before trusting a signal. | [FXDataEngineGUI](FXDataEngineGUI/README.md) |
| Demo Trader | Learn how the full control plane behaves under real market conditions without risking capital. | [FXDataEngine](FXDataEngine/README.md) |
| Backtester | Run scenario-aware Swift/Metal evaluations instead of comparing isolated tester runs by headline score only. | [FXBacktest](FXBacktest/README.md) |
| EA Researcher | Improve models, labels, calibration, routing, and promotion decisions with lineage and evidence. | [Offline Lab](FXDataEngine/Tools/OfflineLab/README.md) |
| System Architect | Operate services, artifacts, release gates, recovery, and platform health without hidden machine assumptions. | [FXDataEngine](FXDataEngine/README.md) |

## Core Benefits

- Swift-first runtime direction: FXDataEngine, FXPlugins, FXBacktest, FXDatabase, and the agent folders are the active source surfaces for the migration away from MT5 execution.
- One canonical historical market contract: verified `M1 OHLCV` from FXDatabase, with MT5 raw data access isolated behind the FXDatabase bridge and FXAI data pipeline. Spread is not part of this offline contract; volume is used by plugins whenever the dataset provides nonzero values.
- Native Swift backtesting: FXBacktest consumes FXDatabase history and can run CPU, Metal, or hybrid CPU+Metal optimization paths for converted plugins.
- Plugin-based model layer: statistical, tree, linear, sequence, factor, trend, regime, policy, and ensemble families share one prediction contract.
- Runtime control plane: NewsPulse, Rates Engine, Cross Asset, Microstructure, Adaptive Router, Dynamic Ensemble, Probabilistic Calibration, Execution Quality, Drift Governance, Pair Network, and System Health layers can explain or suppress unsafe trades.
- Audit and promotion discipline: candidates are checked through repeatable compile, deterministic, pytest, audit, benchmark, and release-gate workflows before promotion.
- Practical operator surfaces: terminal commands remain first-class, and the optional GUI provides role-based dashboards, report browsing, run builders, promotion review, and recovery guidance.

## What Users Can Do With FXAI

- Check the live state of a symbol and understand the reason behind `ALLOW`, `CAUTION`, `BLOCK`, or `ABSTAIN`.
- Run Swift-native FXBacktest jobs and audit scenarios with shared assumptions instead of manually comparing inconsistent settings.
- Research and promote better candidates with Offline Lab, benchmark cards, model-family scorecards, and strategy-profile manifests.
- Inspect service health for news, rates, cross-asset, microstructure, calendar, factor context, calibration, execution, drift, and portfolio-conflict layers.
- Use the GUI to arrange dashboards, save layouts, start common workflows, and recover from missing or stale artifacts.

## Documentation

The old in-repo handbook has been retired. Documentation now lives with the project that owns the code or artifact flow.

Recommended first pages:

- [FXDataEngine](FXDataEngine/README.md)
- [FXDataEngine MQL5 Port Plan](FXDataEngine/Docs/MQL5PortPlan.md)
- [FXDataEngineGUI](FXDataEngineGUI/README.md)
- [FXPlugins](FXPlugins/README.md)
- [FXBacktest Subproject](FXBacktest/README.md)
- [FXDatabase Subproject](FXDatabase/README.md)

## Swift And Metal Migration

FXBacktest and FXDatabase are the foundation for the pure Swift stack. FXDatabase owns verified historical M1 OHLCV data and the one remaining MT5 exporter bridge. FXDataEngine owns the ported data-engine contracts and the former FXAI toolchain under `FXDataEngine/Tools`, FXPlugins owns converted plugin execution adapters, and FXBacktest owns strategy simulation, optimization, plugin evaluation, and Metal acceleration. Legacy FXAI MQL5 source has been retired from this repository.

Subsystem guides:

- [Offline Lab](FXDataEngine/Tools/OfflineLab/README.md)
- [Benchmarks](FXDataEngine/Tools/Benchmarks/benchmark_matrix.md)
- [Promotion Criteria](FXDataEngine/Tools/Benchmarks/promotion_criteria.md)
- [Release Notes](FXDataEngine/Tools/Benchmarks/ReleaseNotes/reference_release_notes.md)
- [NewsPulse](FXDataEngine/Tools/OfflineLab/NewsPulse/README.md)
- [Rates Engine](FXDataEngine/Tools/OfflineLab/RatesEngine/README.md)
- [Cross Asset](FXDataEngine/Tools/OfflineLab/CrossAsset/README.md)
- [Microstructure](FXDataEngine/Tools/OfflineLab/Microstructure/README.md)
- [Adaptive Router](FXDataEngine/Tools/OfflineLab/AdaptiveRouter/README.md)
- [Dynamic Ensemble](FXDataEngine/Tools/OfflineLab/DynamicEnsemble/README.md)
- [Probabilistic Calibration](FXDataEngine/Tools/OfflineLab/ProbabilisticCalibration/README.md)
- [Execution Quality](FXDataEngine/Tools/OfflineLab/ExecutionQuality/README.md)
- [Label Engine](FXDataEngine/Tools/OfflineLab/LabelEngine/README.md)
- [Drift Governance](FXDataEngine/Tools/OfflineLab/DriftGovernance/README.md)
- [Pair Network](FXDataEngine/Tools/OfflineLab/PairNetwork/README.md)

## Operating Boundary

FXAI can improve decision quality, auditability, and operational discipline. It does not guarantee profit. A model score is not a trade by itself; FXAI is designed to evaluate the score against costs, uncertainty, event risk, liquidity, regime, execution quality, drift, and portfolio conflict before action.
