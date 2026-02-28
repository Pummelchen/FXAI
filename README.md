# FXAI

FXAI is a modular, multi-model AI Expert Advisor framework for MetaTrader 5, focused on online learning for FX trading with strict risk controls and fast backtesting workflows.

## What It Is

FXAI is an MT5 Expert Advisor project that combines:
- A plugin-based AI architecture (many models, one unified API)
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
  - MT5 Experts folder remains source-of-truth, GitHub is used for versioned snapshots.

## Project Structure

- `FXAI/FXAI.mq5`  
  Main EA entry point
- `FXAI/api.mqh`  
  Plugin registry and model wiring
- `FXAI/plugin_base.mqh`  
  Shared plugin interface and training/prediction contracts
- `FXAI/data.mqh`  
  Feature generation and data/context pipeline
- `FXAI/shared.mqh`  
  Shared types, constants, and utility math
- `FXAI/Plugins/*.mqh`  
  Individual AI model implementations

## Typical Workflow

1. Develop and compile in MT5 (`MQL5/Experts/FXAI`).
2. Backtest/optimize model and risk parameters in Strategy Tester.
3. Sync MT5 project state into this GitHub repo for version history.
4. Repeat with walk-forward validation before live deployment.

## Notes

- This project is for research and systematic strategy development.
- No model guarantees profit; robust validation and risk control are required.
- Use realistic spread/commission/slippage settings in tester for meaningful results.
