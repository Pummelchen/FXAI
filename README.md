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

## Feature Set (A-Z)

Engineered model input features currently used by the project:

- 1-bar return normalized (M1)
- 3-bar return normalized (M1)
- 5-bar return normalized (M1)
- ATR(14) normalized (M1)
- Close-to-Low edge (M1 OHLC)
- Close-to-Open edge (M1 OHLC)
- Context return dispersion (cross-symbol)
- Context return mean (cross-symbol)
- Context up-breadth ratio (cross-symbol)
- Ehlers Super Smoother (2-pole, period 20) edge
- EMA(100) edge (H1)
- EMA(100) edge (M15)
- EMA(100) edge (M30)
- EMA(100) edge (M5)
- EMA(200) edge (H1)
- EMA(200) edge (M15)
- EMA(200) edge (M30)
- EMA(200) edge (M5)
- Garman-Klass volatility (OHLC)
- H1 return (aligned)
- H1 slope (aligned)
- Hampel robust score (rolling median/MAD, window 21)
- High-Low range normalized (M1 OHLC)
- High-to-Close edge (M1 OHLC)
- Hour of day (MT5 bar time)
- Kalman filter estimate edge (window 34)
- M15 return (aligned)
- M15 SMA(100) edge
- M15 SMA(200) edge
- M30 SMA(100) edge
- M30 SMA(200) edge
- M5 return (aligned)
- M5 slope (aligned)
- M5 SMA(100) edge
- M5 SMA(200) edge
- Minute of hour (MT5 bar time)
- NATR(14) (M1)
- Parkinson volatility (HL-only, window 20)
- Quadruple-smoothed DEMA(100) edge
- Quadruple-smoothed DEMA(200) edge
- Return variance proxy normalized (M1, window 10)
- Rogers-Satchell volatility (OHLC)
- Rolling median edge (window 21)
- RSI(14) normalized (M1)
- Spread normalized
- Weekday (MT5 bar time)
- Z-score vs rolling mean/std (M1, window 10)

## AI Models (A-Z)

### Autoformer
Autoformer is strong at separating trend and seasonality, which helps when FX regimes drift over weeks or months. It is useful for time-series prediction because decomposition can reduce noise before forecasting direction or expected move.

### CatBoost
CatBoost handles nonlinear interactions and mixed feature scales with minimal manual feature engineering. It is useful for FX because tree ensembles often capture threshold effects in volatility and session behavior.

### Chronos
Chronos-style sequence modeling is designed for robust forecasting over diverse temporal patterns. It is useful for FX prediction because it can transfer structure across changing market states better than purely linear models.

### ENHash
ENHash combines elastic-net regularization with hashed interactions for a sparse but expressive online model. It is useful for FX because it keeps CPU usage low while still modeling nonlinear feature combinations.

### FTRL Logit
FTRL is an online optimizer built for sparse, streaming updates and stable incremental learning. It is useful for FX/time-series because it adapts quickly bar by bar without expensive retraining.

### GeodesicAttention
GeodesicAttention emphasizes relational structure in feature space, not only raw distance in Euclidean terms. It is useful for FX because regime similarity can be more informative than absolute price level proximity.

### LightGBM
LightGBM-style gradient boosting is effective at handling nonlinear boundaries and feature importance ranking. It is useful for FX prediction because it can model asymmetric responses to volatility, spread, and trend context.

### LSTM
LSTM captures temporal dependencies and delayed effects in sequential data. It is useful for FX/time-series because momentum, mean reversion, and volatility clustering are often path dependent.

### LSTM-G
LSTM-G extends recurrent modeling with additional gating refinements for sequence stability. It is useful for FX because improved gating can reduce noisy state transitions in fast-changing intraday data.

### MLP Tiny
MLP Tiny is a compact neural baseline that is fast to run in MT5 environments. It is useful for FX as a low-latency nonlinear model and as a robust ensemble member against over-specialized models.

### PA Linear
Passive-Aggressive linear learning updates strongly on mistakes while staying conservative otherwise. It is useful for FX streaming prediction where fast reaction to fresh market shifts is critical.

### PatchTST
PatchTST processes time-series in patches to improve efficiency and local pattern extraction. It is useful for FX because short recurring micro-structures can matter for near-horizon trade decisions.

### Quantile
Quantile modeling estimates move distribution rather than only direction. It is useful for FX because risk-aware entries depend on expected range and tail behavior, not just up/down probability.

### S4
S4 (state-space sequence model) is designed for long-context temporal modeling with efficient recurrence. It is useful for FX/time-series where distant context can still influence current volatility and direction.

### SGD Logit
SGD logistic regression is a simple, transparent online classifier with fast updates. It is useful for FX as a stable baseline and quick-adapting model under strict CPU constraints.

### STMN
STMN targets spatio-temporal structure, combining time dynamics with cross-context relationships. It is useful for FX because multi-symbol co-movement and lead-lag effects are naturally spatio-temporal.

### TCN
TCN uses causal dilated convolutions to model long effective history without recurrent loops. It is useful for FX prediction because it captures multi-scale temporal motifs with good parallel efficiency.

### TFT
TFT combines gating, variable selection, and attention for interpretable sequence forecasting. It is useful for FX because feature relevance can shift by regime, and TFT can adapt weighting over time.

### TimesFM
TimesFM-style forecasting emphasizes broad time-series generalization and stable sequence representations. It is useful for FX as a high-capacity model for regime-robust predictions across diverse market conditions.

### TST
Time-Series Transformer (TST) applies transformer attention directly to temporal tokens. It is useful for FX because attention can detect nonlocal dependencies that fixed-window methods may miss.

### XGB Fast
XGB Fast is a lightweight boosting variant optimized for speed in constrained runtimes. It is useful for FX backtesting and live execution when you need tree-style nonlinear power with low overhead.

### XGBoost
XGBoost is a strong gradient-boosted tree framework with robust regularization and split optimization. It is useful for FX/time-series because it often performs well on tabular engineered features and noisy signals.

## Notes

- This project is for research and systematic strategy development.
- No model guarantees profit; robust validation and risk control are required.
- Use realistic spread/commission/slippage settings in tester for meaningful results.
