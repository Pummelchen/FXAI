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
- [Reference PDF](#reference-pdf)
- [Notes](#notes)

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
  - MT5 Experts folder remains source-of-truth, GitHub is used as the synchronized repository copy.

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
3. Sync MT5 project state into this GitHub repo after a clean MT5 compile.
4. Repeat with walk-forward validation before live deployment.

## Reference PDF

The detailed appendix material was moved into a dedicated reference document so the main README stays focused. Use the links below for the full long-form lists and descriptions.

- [FXAI Detailed Reference (PDF)](docs/FXAI_Detailed_Reference.pdf)
- [FXAI Detailed Reference (Text Source)](docs/FXAI_Detailed_Reference.txt)

## Notes

- This project is for research and systematic strategy development.
- No model guarantees profit; robust validation and risk control are required.
- Use realistic spread/commission/slippage settings in tester for meaningful results.
