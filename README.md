<p align="center">
  <img src="FXAI/Wiki/assets/fxai-overview.png" alt="FXAI architecture overview">
</p>

# FXAI

FXAI is an MT5-native framework for researching, testing, auditing, and operating AI-assisted FX trading systems. Live trading stays inside MetaTrader 5 and MQL5; offline Python tooling is used for research, calibration, export, audit, and release evidence.

FXAI is not a single black-box strategy. It is a governed decision framework: shared data contracts, model plugins, runtime risk layers, audit gates, promotion artifacts, and an optional GUI all work from the same source of truth.

## User Matrix Benefits

| User | Core Benefit | Start Here |
|---|---|---|
| Live Trader | See whether a trade is allowed, cautioned, blocked, or abstained before trusting a signal. | [Quick Start By Role](FXAI/Wiki/Quick%20Start%20By%20Role.md) |
| Demo Trader | Learn how the full control plane behaves under real market conditions without risking capital. | [Getting Started](FXAI/Wiki/Getting%20Started.md) |
| Backtester | Run scenario-aware evaluations instead of comparing isolated MT5 tester runs by headline score only. | [Audit Lab](FXAI/Wiki/Audit%20Lab.md) |
| EA Researcher | Improve models, labels, calibration, routing, and promotion decisions with lineage and evidence. | [Offline Lab](FXAI/Wiki/Offline%20Lab.md) |
| System Architect | Operate services, artifacts, release gates, recovery, and platform health without hidden machine assumptions. | [Project Structure](FXAI/Wiki/Project%20Structure.md) |

## Core Benefits

- MT5-native live runtime: no external live inference service and no DLL dependency for trade decisions.
- One canonical market contract: `M1 OHLC + spread`, with raw MT5 data access isolated behind the FXAI data pipeline.
- Plugin-based model layer: statistical, tree, linear, sequence, factor, trend, regime, policy, and ensemble families share one prediction contract.
- Runtime control plane: NewsPulse, Rates Engine, Cross Asset, Microstructure, Adaptive Router, Dynamic Ensemble, Probabilistic Calibration, Execution Quality, Drift Governance, and Pair Network layers can explain or suppress unsafe trades.
- Audit and promotion discipline: candidates are checked through repeatable compile, deterministic, pytest, audit, benchmark, and release-gate workflows before promotion.
- Practical operator surfaces: terminal commands remain first-class, and the optional GUI provides role-based dashboards, report browsing, run builders, promotion review, and recovery guidance.

## What Users Can Do With FXAI

- Check the live state of a symbol and understand the reason behind `ALLOW`, `CAUTION`, `BLOCK`, or `ABSTAIN`.
- Run backtests and Audit Lab scenarios with shared assumptions instead of manually comparing inconsistent settings.
- Research and promote better candidates with Offline Lab, benchmark cards, model-family scorecards, and strategy-profile manifests.
- Inspect service health for news, rates, cross-asset, microstructure, calibration, execution, drift, and portfolio-conflict layers.
- Use the GUI to arrange dashboards, save layouts, start common workflows, and recover from missing or stale artifacts.

## Documentation

Use the versioned handbook in [FXAI/Wiki](FXAI/Wiki/Home.md). The public GitHub wiki should mirror these pages.

Recommended first pages:

- [Home](FXAI/Wiki/Home.md)
- [Quick Start By Role](FXAI/Wiki/Quick%20Start%20By%20Role.md)
- [Getting Started](FXAI/Wiki/Getting%20Started.md)
- [Runtime Control Plane](FXAI/Wiki/Runtime%20Control%20Plane.md)
- [GUI](FXAI/Wiki/GUI.md)
- [Data Policy](FXAI/Wiki/Data%20Policy.md)
- [Project Structure](FXAI/Wiki/Project%20Structure.md)

Subsystem guides:

- [Audit Lab](FXAI/Wiki/Audit%20Lab.md)
- [Offline Lab](FXAI/Wiki/Offline%20Lab.md)
- [Benchmarks](FXAI/Wiki/Benchmarks.md)
- [Model Zoo](FXAI/Wiki/Model%20Zoo.md)
- [Promotion Criteria](FXAI/Wiki/Promotion%20Criteria.md)
- [Release Notes](FXAI/Wiki/Release%20Notes.md)
- [NewsPulse](FXAI/Wiki/NewsPulse.md)
- [Rates Engine](FXAI/Wiki/Rates%20Engine.md)
- [Cross Asset](FXAI/Wiki/Cross%20Asset.md)
- [Microstructure](FXAI/Wiki/Microstructure.md)
- [Adaptive Router](FXAI/Wiki/Adaptive%20Router.md)
- [Dynamic Ensemble](FXAI/Wiki/Dynamic%20Ensemble.md)
- [Probabilistic Calibration](FXAI/Wiki/Probabilistic%20Calibration.md)
- [Execution Quality](FXAI/Wiki/Execution%20Quality.md)
- [Label Engine](FXAI/Wiki/Label%20Engine.md)
- [Drift Governance](FXAI/Wiki/Drift%20Governance.md)
- [Pair Network](FXAI/Wiki/Pair%20Network.md)

## Operating Boundary

FXAI can improve decision quality, auditability, and operational discipline. It does not guarantee profit. A model score is not a trade by itself; FXAI is designed to evaluate the score against costs, uncertainty, event risk, liquidity, regime, execution quality, drift, and portfolio conflict before action.
