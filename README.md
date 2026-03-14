# FXAI

FXAI is a professional MetaTrader 5 framework for building, testing, and operating AI-driven FX trading systems inside MT5.

It is designed for serious traders and researchers who want one disciplined workflow for:
- comparing different model families
- backtesting on realistic FX costs
- filtering weak models before cloud optimization
- running release-style audit checks before live use

FXAI runs fully inside MT5 and MQL5. There are no external inference servers, no DLL dependency, and no separate Python model runtime in live trading.

Core benefits at a glance:
- Many model plugins under one execution and risk framework
- Cost-aware `BUY / SELL / SKIP` decisions
- Built-in equity and execution protection logic
- MT5-native backtesting and cloud optimization compatibility
- A dedicated Audit Lab for plugin certification, regression checks, and release gating
- One workflow for research, backtesting, audit, and deployment

## Table of Contents

- [What It Is](#what-it-is)
- [Core Benefits](#core-benefits)
- [Quick Start](#quick-start)
- [How Traders Use FXAI](#how-traders-use-fxai)
- [Project Structure](#project-structure)
- [Typical Workflow](#typical-workflow)
- [Audit Lab](#audit-lab)
- [Reference Guides](#reference-guides)
- [Notes](#notes)

## What It Is

FXAI is an MT5 Expert Advisor project with a plugin-based model layer.

In practical trading terms:
- the EA is the execution shell
- a plugin is the prediction engine or "model brain"
- the data pipeline builds the market inputs
- the meta layer decides how to combine or route model outputs
- the risk layer decides whether the signal is safe enough to trade
- the Audit Lab checks whether a plugin behaves correctly before you trust it

You do not need to be an MQL5 programmer to use FXAI as an operator. For normal usage, the main tasks are:
- choose the symbol and tester settings
- choose the plugin or plugin set you want to test
- run backtests and focused audits
- save baselines for stable behavior
- reject weak model changes before live deployment

## Core Benefits

- **One framework, many models**
  - Swap or benchmark AI plugins without rewriting EA logic.
- **Cost-aware signals**
  - Labels and thresholds account for trading friction, improving realistic expectancy.
- **Safer execution**
  - Built-in equity controls, skip class, and conservative calibration reduce overtrading.
- **Backtest efficiency**
  - Lightweight online updates and shared data pipeline support large optimization runs.
- **Audit discipline**
  - Synthetic pressure tests and real-market replay certification catch weak plugins before large cloud backtests.
- **Extensible by design**
  - New models can be added through the plugin API with consistent train/predict flow.
- **Production-oriented workflow**
  - MT5 Experts folder remains source-of-truth, GitHub is used as the synchronized repository copy.

## Quick Start

If you are an MT5 trader with basic AI understanding, the shortest correct path is:
1. Compile the EA and the Audit Runner
2. Run a simple Strategy Tester pass on one symbol
3. Run a focused Audit Lab pass on the plugin you care about
4. Save a baseline once behavior is acceptable
5. Only then scale to optimization or live evaluation

Runtime source-of-truth:
- Live MT5 tree: `MQL5/Experts/FXAI`

Versioned mirror:
- Git repo copy: `FXAI/`

Important clarification:
- `fxai_testlab.py` compiles and launches the live MT5 tree.
- If you edit only the git mirror, sync those changes into the MT5 Experts folder before compiling.

Compile from the repo root:

```bash
cd /Users/andreborchert/Documents/New\ project/FXAI
python3 FXAI/Tools/fxai_testlab.py compile-main
python3 FXAI/Tools/fxai_testlab.py compile-audit
```

Focused Audit Lab example:

```bash
cd /Users/andreborchert/Documents/New\ project/FXAI
python3 FXAI/Tools/fxai_testlab.py run-audit \
  --plugin-list "{ai_mlp}" \
  --scenario-list "{market_recent, market_walkforward}" \
  --symbol EURUSD
```

Typical baseline workflow:

```bash
python3 FXAI/Tools/fxai_testlab.py baseline-save --name eurusd_smoke
python3 FXAI/Tools/fxai_testlab.py release-gate --baseline eurusd_smoke --require-market-replay
```

## How Traders Use FXAI

Typical operator use cases:
- **Compare model candidates**
  - Example: test `ai_mlp` versus `tree_lgbm` on `EURUSD`
- **Check whether a model is still trustworthy**
  - Use the Audit Lab before a long optimization or before promoting a configuration
- **Validate execution realism**
  - Re-run audits with spread, slippage, and fill penalties that match your broker conditions
- **Create a baseline**
  - Save a known-good audit result so future changes can be compared against it
- **Reject weak changes early**
  - Use `release-gate` to stop regressions before they reach large backtests or live trading

What FXAI is not:
- not a guaranteed-profit black box
- not a one-click live trading promise
- not a substitute for realistic cost settings and disciplined validation

## Project Structure

- `FXAI/FXAI.mq5`  
  Main EA entry point and live trading shell
- `FXAI/API/api.mqh`  
  API v4 registry, validation, and plugin wiring
- `FXAI/API/plugin_base.mqh`  
  Shared plugin base contract and common model services
- `FXAI/Engine/core.mqh`  
  Shared types, enums, manifest helpers, feature-schema helpers, and common math
- `FXAI/Engine/data_pipeline.mqh`  
  Aggregated data-pipeline include that pulls in the split data and feature modules
- `FXAI/Engine/*.mqh`  
  Runtime, training, warmup, lifecycle, sample, and meta orchestration layers
- `FXAI/Engine/data_*.mqh`, `FXAI/Engine/feature_*.mqh`  
  Split market-data, alignment, normalization-window, feature-math, normalization, and feature-build modules
- `FXAI/Engine/meta_*.mqh`  
  Split horizon, stacker, calibration, reliability, threshold, and support subsystems
- `FXAI/Plugins/*.mqh`  
  Individual AI model implementations
- `FXAI/Tests/FXAI_AuditRunner.mq5`  
  MT5-side synthetic and market-replay plugin audit runner
- `FXAI/Tests/audit_*.mqh`  
  Split scenario, sample-build, scoring, report, and audit utility modules
- `FXAI/Tools/fxai_testlab.py`  
  External drill-sergeant analyzer, optimization planner, baseline comparator, and release gate

If you are not a programmer, the most important folders are:
- `FXAI/` for the main EA
- `FXAI/Tests/` for the Audit Runner
- `FXAI/Tools/` for the command-line helper and audit outputs

## Typical Workflow

1. Compile in the live MT5 Experts folder.
2. Run a clean single-symbol Strategy Tester pass with realistic spread and commission settings.
3. Run a focused Audit Lab pass on the plugin you want to trust.
4. Save a baseline once the plugin behavior is stable.
5. Only then move to larger optimization, walk-forward, or live-demo evaluation.
6. Sync MT5 project state into GitHub only after the live code compiles cleanly.

## Audit Lab

FXAI includes a drill-sergeant Audit Lab for plugin stress testing, market replay certification, regression checks, optimization planning, and release gating.

Recent Audit Lab capabilities include:
- release-grade market replay packs alongside synthetic stress tests
- schema and feature-mask override testing
- audit-guided optimization campaign generation for schemas, normalizers, sequence lengths, and feature groups

The full operator handbook now lives in the GitHub wiki:
- [Audit Lab Wiki](https://github.com/Pummelchen/FXAI/wiki/Audit-Lab)

Use that guide for:
- what the Audit Lab does
- how the MT5 audit runner and external analyzer work
- unattended tester launch and credentials
- baseline, comparison, optimization, and release-gate workflows
- output files, troubleshooting, and recommended usage

Short version:
- Use Strategy Tester to measure trading behavior
- Use Audit Lab to decide whether a plugin deserves further trust
- A plugin that backtests well but fails Audit Lab is not production-ready

## Reference Guides

The main framework handbook explains the architecture, naming convention, model families, and core design choices. The Audit Lab handbook explains the plugin certification toolchain and how to operate it.

- [FXAI Wiki Home](https://github.com/Pummelchen/FXAI/wiki)
- [FXAI Framework](https://github.com/Pummelchen/FXAI/wiki/FXAI-Framework)
- [Getting Started](https://github.com/Pummelchen/FXAI/wiki/Getting-Started)
- [Audit Lab](https://github.com/Pummelchen/FXAI/wiki/Audit-Lab)
- [Project Structure](https://github.com/Pummelchen/FXAI/wiki/Project-Structure)

## Notes

- This project is for research and systematic strategy development.
- No model guarantees profit; robust validation and risk control are required.
- Use realistic spread/commission/slippage settings in tester for meaningful results.
