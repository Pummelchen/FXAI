# FXAI

FXAI is the most advanced, state-of-the-art AI framework for Forex trading with MetaTrader 5.

Engineered for serious FX & AI research, it combines institutional-grade multi-model architecture, online learning, strict risk controls, and high-speed backtesting workflows in one unified framework.

Built with zero external libraries or DLLs, FXAI runs entirely on highly optimized, pure MQL5 CPU code. Fully compatible with backtesting on the MQL5 Cloud Network, it enables massive parallel speedups and much faster optimization results.

Core benefits at a glance:
- Multi-model plugin architecture with one unified API and shared data pipeline.
- Cost-aware `BUY / SELL / SKIP` decisions tuned for realistic FX execution conditions.
- Strong equity-level protection logic to reduce overtrading and uncontrolled drawdowns.
- High-speed MT5-native backtesting and optimization without external libraries or DLLs.

## Table of Contents

- [What It Is](#what-it-is)
- [Core Benefits](#core-benefits)
- [Project Structure](#project-structure)
- [Typical Workflow](#typical-workflow)
- [Reference Appendix](#reference-appendix)
- [Notes](#notes)

## What It Is

FXAI is an MT5 Expert Advisor project that combines:
- A plugin-based AI architecture (many models, one unified API)
- Baseline control plugins such as `rule_buyonly`, `rule_sellonly`, and `rule_random`
- 3-class decision logic: `BUY / SELL / SKIP`
- Cost-aware training (spread/commission-aware labeling)
- Online model updates during backtest/live runtime
- Ensemble support to compare and combine models
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
- **Extensible by design**
  - New models can be added through the plugin API with consistent train/predict flow.
- **Production-oriented workflow**
  - MT5 Experts folder remains source-of-truth, GitHub is used as the synchronized repository copy.

## Project Structure

- `FXAI/FXAI.mq5`  
  Main EA entry point
- `FXAI/API/api.mqh`  
  API v4 registry, validation, and plugin wiring
- `FXAI/API/plugin_base.mqh`  
  Shared plugin base contract and common model services
- `FXAI/Engine/core.mqh`  
  Shared types, enums, constants, and common helpers
- `FXAI/Engine/data_pipeline.mqh`  
  Feature generation, normalization, and data/context pipeline
- `FXAI/Engine/*.mqh`  
  Runtime, training, warmup, lifecycle, sample, and meta orchestration layers
- `FXAI/Plugins/*.mqh`  
  Individual AI model implementations
- `FXAI/Tests/FXAI_AuditRunner.mq5`  
  MT5-side synthetic plugin audit runner
- `FXAI/Tools/fxai_testlab.py`  
  External drill-sergeant analyzer, baseline comparator, and release gate

## Typical Workflow

1. Develop and compile in MT5 (`MQL5/Experts/FXAI`).
2. Backtest/optimize model and risk parameters in Strategy Tester.
3. Sync MT5 project state into this GitHub repo after a clean MT5 compile.
4. Repeat with walk-forward validation before live deployment.

## Audit Lab

FXAI includes a synthetic audit framework that pressure-tests plugins outside normal market backtests.

- Compile the main EA:
  - `python3 "FXAI/Tools/fxai_testlab.py" compile-main`
- Compile the audit runner:
  - `python3 "FXAI/Tools/fxai_testlab.py" compile-audit`
- Analyze an existing audit report:
  - `python3 "FXAI/Tools/fxai_testlab.py" analyze`
- Save a regression baseline:
  - `python3 "FXAI/Tools/fxai_testlab.py" baseline-save --name nightly_a`
- Compare against a baseline:
  - `python3 "FXAI/Tools/fxai_testlab.py" baseline-compare --baseline nightly_a`
- Enforce the release gate:
  - `python3 "FXAI/Tools/fxai_testlab.py" release-gate --baseline nightly_a`

For unattended MT5 tester launch, `run-audit` supports:
- CLI: `--login`, `--server`, `--password`
- Environment: `FXAI_MT5_LOGIN`, `FXAI_MT5_SERVER`, `FXAI_MT5_PASSWORD`

Typical unattended example:

```bash
export FXAI_MT5_LOGIN=11013759
export FXAI_MT5_SERVER=ICMarketsSC-MT5-4
export FXAI_MT5_PASSWORD='YOUR_PASSWORD'
python3 "FXAI/Tools/fxai_testlab.py" run-audit --plugin-list "{rule_m1sync}" --scenario-list "{random_walk,monotonic_up}"
```

## Reference Appendix

The compact reference below explains the framework, naming convention, model families, and core design choices without overloading the main README.

- [FXAI Reference](FXAI/FXAI_Reference.txt)

## Notes

- This project is for research and systematic strategy development.
- No model guarantees profit; robust validation and risk control are required.
- Use realistic spread/commission/slippage settings in tester for meaningful results.
