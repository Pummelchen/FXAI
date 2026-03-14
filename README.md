# FXAI

FXAI is the most advanced, state-of-the-art AI framework for Forex trading with MetaTrader 5.

Engineered for serious FX & AI research, it combines institutional-grade multi-model architecture, online learning, strict risk controls, and high-speed backtesting workflows in one unified framework.

Built with zero external libraries or DLLs, FXAI runs entirely on highly optimized, pure MQL5 CPU code. Fully compatible with backtesting on the MQL5 Cloud Network, it enables massive parallel speedups and much faster optimization results.

Core benefits at a glance:
- Multi-model plugin architecture with one unified API and shared data pipeline.
- Manifest-driven feature schemas and schema-specific projectors so each plugin family can consume a more appropriate view of the data.
- Cost-aware `BUY / SELL / SKIP` decisions tuned for realistic FX execution conditions.
- Strong equity-level protection logic to reduce overtrading and uncontrolled drawdowns.
- High-speed MT5-native backtesting, richer meta-policy routing, dynamic market-context selection, market replay certification, and optimization without external libraries or DLLs.

## Table of Contents

- [What It Is](#what-it-is)
- [Core Benefits](#core-benefits)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Typical Workflow](#typical-workflow)
- [Audit Lab](#audit-lab)
- [Reference Guides](#reference-guides)
- [Notes](#notes)

## What It Is

FXAI is an MT5 Expert Advisor project that combines:
- A plugin-based AI architecture (many models, one unified API)
- Baseline control plugins such as `rule_buyonly`, `rule_sellonly`, and `rule_random`
- 3-class decision logic: `BUY / SELL / SKIP`
- Cost-aware training (spread/commission-aware labeling)
- Online model updates during backtest/live runtime
- Ensemble support to compare and combine models
- Dynamic market-context selection from a larger symbol candidate pool
- Richer meta-policy inputs for horizon routing and ensemble blending
- Equity-level risk management and trade protection logic

The project is designed to keep MT5 execution practical while enabling advanced model experimentation.

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

## Project Structure

- `FXAI/FXAI.mq5`  
  Main EA entry point
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

## Typical Workflow

1. Develop and compile in MT5 (`MQL5/Experts/FXAI`).
2. Backtest/optimize model and risk parameters in Strategy Tester.
3. Sync MT5 project state into this GitHub repo after a clean MT5 compile.
4. Repeat with walk-forward validation before live deployment.

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
