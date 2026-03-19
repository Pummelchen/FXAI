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
- A dedicated internal `TensorCore` runtime for the serious neural and math-heavy models
- Cost-aware `BUY / SELL / SKIP` decisions
- Built-in equity and execution protection logic
- MT5-native backtesting and cloud optimization compatibility
- A dedicated Audit Lab for plugin certification, regression checks, and release gating
- One workflow for research, backtesting, audit, and deployment

## Table of Contents

- [What It Is](#what-it-is)
- [Core Benefits](#core-benefits)
- [TensorCore](#tensorcore)
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
- **Internal neural runtime**
  - Serious neural plugins share one native `TensorCore` layer for tensor math, sequence handling, sequence blocks, optimizers, and numeric verification.
- **Cross-symbol transfer warmup**
  - Shared transfer adapters now pretrain across configured horizons and capped context-symbol sample packs before live trading begins, with a deeper shared latent backbone that is conditioned by symbol-domain, horizon, session context, and explicit sequence-window tokens.
- **Checkpoint recovery for stateful plugins**
  - Persistent plugins now carry replay-backed checkpoint reconstruction metadata, and runtime manifests plus release gating now block live promotion of stateful plugins until they provide full native checkpoint coverage.
- **Cost-aware signals**
  - Labels and thresholds account for trading friction, improving realistic expectancy.
- **Feature governance**
  - A feature registry now tracks provenance, leakage guards, session-transition and rollover states, swap/carry factors, macro-event features, live feature-family drift diagnostics, and emits a runtime feature manifest for auditability.
- **M1-first research data**
  - FXAI now explicitly treats M1 OHLC plus spread as the canonical execution and feature source, with integrity checks shared by warmup, runtime, and Audit Lab. Higher-timeframe candle and spread states for the traded symbol and the top context FX pairs are derived from that same M1 source, so every plugin sees one consistent `OHLC + spread` contract instead of mixed shortcuts.
- **Safer execution**
  - Built-in equity controls, skip class, and conservative calibration reduce overtrading.
- **Execution parity controls**
  - Shared execution profiles, M1 replay-trace penalties, persistent broker-event trace replay, risk-aware sizing, correlated exposure caps, and `OrderCheck` preflight keep live trading closer to Audit Lab and tester assumptions.
- **Contextual model routing**
  - The meta layer now persists regime and horizon-specific contextual edge and regret state so ensemble routing can adapt to where each plugin is actually earning or losing edge.
- **Persistent research state**
  - Runtime artifacts now persist feature-drift diagnostics and emit per-plugin checkpoint coverage plus feature-registry manifests so restart behavior and checkpoint depth are auditable.
- **Backtest efficiency**
  - Lightweight online updates and shared data pipeline support large optimization runs.
- **Audit discipline**
  - Synthetic pressure tests, real-market replay certification, macro-event scenario checks, and manifest-based release gates catch weak plugins before large cloud backtests.
- **Extensible by design**
  - New models can be added through the plugin API with consistent train/predict flow.
- **Production-oriented workflow**
  - MT5 Experts folder remains source-of-truth, GitHub is used as the synchronized repository copy.

## TensorCore

If you are familiar with TensorFlow or PyTorch, think of `TensorCore` as FXAI's focused internal neural runtime.

It exists so the stronger FXAI plugins do not each carry their own private version of:
- sequence packing and masking
- attention and causal convolution logic
- normalization and residual sequence blocks
- optimizer and learning-rate schedule math
- numeric self-tests and serialization sanity checks

What `TensorCore` now provides inside pure MQL5:
- fixed-shape tensor kernels for the dimensions FXAI actually uses
- shared sequence-to-sequence attention and causal convolution blocks
- dimension-aware runtime configs for model width, heads, stride, patching, and sequence caps
- shared parameter-group training utilities for vector and matrix updates
- sequence runtime state helpers for packed windows, resampling, and positional bias handling
- Audit Lab sanity coverage for the runtime itself, not only for the plugins built on top of it

Why that matters:
- stronger plugins can use a more realistic shared neural foundation
- model upgrades land once in TensorCore and benefit multiple plugins
- TensorCore regressions can be caught by Audit Lab before they leak into live plugin behavior

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
cd /Users/andreborchert/FXAI-main2
python3 FXAI/Tools/fxai_testlab.py compile-main
python3 FXAI/Tools/fxai_testlab.py compile-audit
```

Focused Audit Lab example:

```bash
cd /Users/andreborchert/FXAI-main2
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

Optional macro-event input file:

- `FXAI\\Runtime\\macro_events.tsv` in the MT5 common-files area
- required tab-separated columns: `symbol`, `event_time`, `pre_window_min`, `post_window_min`, `importance`, `surprise`, `actual_delta`, `forecast_delta`, `class`
- optional appended columns supported by the current contract: `event_id`, `country`, `currency`, `source`, `revision_delta`, `prior_delta`, `surprise_z`
- if the file is absent, FXAI falls back to zeroed macro-event features and keeps the macro-event audit scenario neutral instead of penalizing plugins for missing optional data
- if the file is present but fails the leakage guard, release-gate checks fail until the dataset is fixed

Portfolio-style audit pack example:

```bash
python3 FXAI/Tools/fxai_testlab.py run-audit \
  --plugin-list "{ai_tft, tree_lgbm}" \
  --scenario-list "{market_recent, market_walkforward}" \
  --symbol-pack majors \
  --execution-profile prime-ecn
```

## How Traders Use FXAI

Typical operator use cases:
- **Compare model candidates**
  - Example: test `ai_mlp` versus `tree_lgbm` on `EURUSD`
- **Check whether a model is still trustworthy**
  - Use the Audit Lab before a long optimization or before promoting a configuration
- **Validate execution realism**
  - Re-run audits with M1 spread and bar-trace-aware slippage, partial-fill, reject, and session-edge penalties that match your broker conditions
- **Deploy with stricter live controls**
  - Configure execution profile, position sizing mode, and portfolio exposure caps before moving from audit into live evaluation
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
  Public plugin-base aggregator
- `FXAI/API/plugin_contract.mqh`, `plugin_context.mqh`, `plugin_tensor_bridge.mqh`, `plugin_quality_heads.mqh`
  Split plugin contract, context/runtime helpers, TensorCore bridge, and quality-head support
- `FXAI/Engine/core.mqh`  
  Shared types, enums, manifest helpers, feature-schema helpers, and common math
- `FXAI/TensorCore/TensorCore.mqh`  
  Public entry point for the internal neural runtime used by serious AI plugins
- `FXAI/TensorCore/tensor_*.mqh`  
  Split tensor kernels, sequence buffers, modules, losses, and optimizers
- `FXAI/Engine/data_pipeline.mqh`  
  Aggregated data-pipeline include that pulls in the split data and feature modules
- `FXAI/Engine/*.mqh`  
  Runtime, training, warmup, lifecycle, sample, and meta orchestration layers
- `FXAI/Engine/data_*.mqh`, `FXAI/Engine/feature_*.mqh`  
  Split market-data, alignment, normalization-window, feature-math, normalization, feature-build, and feature-registry modules
- `FXAI/Engine/meta_*.mqh`  
  Split horizon, stacker, calibration, reliability, threshold, and support subsystems
- `FXAI/Plugins/Sequence/*.mqh`
  Sequence and transformer-family plugins
- `FXAI/Plugins/Linear/*.mqh`, `Tree/*.mqh`, `Rule/*.mqh`
  Linear, tree, and rule-based families
- `FXAI/Plugins/World/*.mqh`, `Memory/*.mqh`, `Mixture/*.mqh`, `Distribution/*.mqh`
  World, retrieval/memory, mixture, and distribution plugins
- `FXAI/Plugins/Sequence/ai_tft/`, `ai_chronos/`, `ai_autoformer/`
  Internal state/forward/public/training sections for the largest evolving sequence models
- `FXAI/Tests/FXAI_AuditRunner.mq5`  
  MT5-side synthetic and market-replay plugin audit runner
- `FXAI/Tests/audit_*.mqh`  
  Split scenario, sample-build, scoring, report, and audit utility modules
- `FXAI/Tests/audit_tensor.mqh`  
  TensorCore numeric sanity and drift checks used by the Audit Runner
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
- named multi-symbol packs such as `majors` and `yen-cross` for portfolio-style certification
- schema and feature-mask override testing
- reproducible `.summary.json` and `.manifest.json` artifacts with repo and file hashes
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
