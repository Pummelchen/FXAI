# FXAI

FXAI is the most advanced, professional, and state-of-the-art AI framework for MetaTrader 5, engineered for serious FX research and deployment. It combines institutional-grade multi-model architecture, online learning, strict risk controls, and high-speed backtesting workflows in one unified MT5-native system. Built with zero external libraries or DLLs, FXAI runs entirely on highly optimized, pure MQL5 CPU code. Fully compatible with backtesting on the MQL5 Cloud Network, it enables massive parallel speedups and much faster optimization results.

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
- [AI Models (A-Z)](#ai-models-a-z)
- [Feature Set (A-Z)](#feature-set-a-z)
- [Normalization Techniques](#normalization-techniques)
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

## AI Models (A-Z)

### Autoformer
Autoformer decomposes time series into trend and seasonal components before forecasting. This helps FX workflows where regime drift can hide signal in raw returns. In this project it is useful for separating slower structure from intraday noise before decision scoring. It works best when the training window includes multiple volatility regimes and not only a single market phase.

### CatBoost
CatBoost is a gradient-boosted decision tree method that handles nonlinear feature interactions well. It is useful in FX because threshold effects in spread, volatility, and session context are often nonlinear. The model is robust with mixed feature scales and can perform strongly on engineered tabular features. Further reading: CatBoost.

### Chronos
Chronos-style forecasting targets robust sequence modeling across changing temporal patterns. It is useful for FX when intraday behavior shifts with sessions and macro events. In ensemble mode it can add diversity against purely linear or tree-based models. Background on the domain: Time series.

### ENHash
ENHash combines elastic-net regularization with hashed interaction features for sparse online learning. It is useful for FX because it stays CPU-light while still modeling nonlinear feature crosses. This makes it a practical model for frequent M1 updates in MT5 environments. Further reading: Elastic net regularization.

### FTRL Logit
FTRL Logit is designed for streaming optimization with sparse updates and good stability. It is useful for FX because it can adapt quickly bar by bar without full retraining. The approach is strong when feature distributions drift during session transitions. Background: Online machine learning.

### GeodesicAttention
GeodesicAttention emphasizes relational structure between observations instead of only raw Euclidean distance. It is useful for FX because regime similarity can matter more than absolute price levels. This can improve signal consistency when markets rotate between risk-on and risk-off states. Related concept: Attention (machine learning).

### LightGBM
LightGBM is a gradient-boosted tree framework optimized for speed and strong nonlinear modeling. It is useful for FX because it can capture asymmetric responses to volatility and liquidity conditions. The model also provides interpretable split behavior across engineered feature sets. Further reading: LightGBM.

### LSTM
LSTM is a recurrent neural architecture built to retain useful sequence memory over time. It is useful in FX where momentum, mean reversion, and volatility clustering are path dependent. In this framework it complements tabular models by modeling temporal state dynamics directly. Further reading: Long short-term memory.

### LSTM-G
LSTM-G extends recurrent modeling with additional gating refinements for sequence stability. It is useful for FX because better gating can reduce noisy state transitions on fast intraday data. This helps when signal quality changes quickly around session opens and news spikes. Related background: Recurrent neural network.

### MLP Tiny
MLP Tiny is a compact feed-forward neural baseline with low compute overhead. It is useful for FX as a fast nonlinear learner that can run efficiently in MT5. The model also serves as a resilient ensemble member against over-specialized plugins. Further reading: Multilayer perceptron.

### PA Linear
PA Linear uses passive-aggressive updates to react strongly to mistakes and stay conservative otherwise. It is useful for FX streaming where fast adaptation to regime changes is critical. The linear form keeps latency low and behavior easy to diagnose during backtests. Background: Online machine learning.

### PatchTST
PatchTST processes sequences in patches to improve local pattern extraction efficiency. It is useful for FX because short recurring micro-structures can matter on M1 horizons. This model can complement recurrent and boosting approaches by focusing on patch-level context. Related concept: Transformer (deep learning architecture).

### Quantile
Quantile modeling predicts distribution bands instead of only a single directional probability. It is useful for FX because trade quality depends on expected move size and tail risk. In this project it supports cost-aware decision thresholds and skip logic. Further reading: Quantile regression.

### S4
S4 is a state-space sequence model designed for long-context temporal dependencies. It is useful for FX because distant context can still influence current volatility and direction. The architecture is attractive when you want long memory with controlled runtime cost. Related background: State-space representation.

### SGD Logit
SGD Logit is a transparent online logistic classifier with very fast incremental updates. It is useful for FX as a stable baseline under strict CPU and latency constraints. The model is easy to calibrate and monitor during long optimization runs. Further reading: Logistic regression and Stochastic gradient descent.

### STMN
STMN targets spatio-temporal dynamics by combining time evolution with cross-context structure. It is useful for FX where multi-symbol co-movement and lead-lag effects are common. This helps the ensemble exploit information beyond a single-symbol price stream. Background on the domain: Spatiotemporal.

### TCN
TCN uses causal dilated convolutions to model long effective history without recurrent loops. It is useful for FX because it can capture multi-scale temporal motifs with good efficiency. The convolutional structure is often stable for online inference in constrained runtimes. Related concepts: Convolutional neural network and Dilated convolution.

### TFT
TFT combines gating, variable selection, and attention for interpretable sequence forecasting. It is useful for FX because feature relevance shifts across sessions and volatility regimes. This architecture can dynamically reweight inputs instead of relying on static feature importance. Related concept: Attention (machine learning).

### TimesFM
TimesFM-style modeling emphasizes broad time-series generalization with strong sequence representations. It is useful for FX as a high-capacity model when market structure changes across periods. In ensemble settings it can contribute robust regime-level forecasts. Background: Foundation model.

### TST
TST applies transformer attention directly to time-series tokens. It is useful for FX because attention can detect nonlocal dependencies across the lookback window. This makes it a strong complement to local-window linear or tree-based methods. Further reading: Transformer (deep learning architecture).

### XGB Fast
XGB Fast is a speed-optimized boosting variant for low-latency runtime environments. It is useful for FX backtesting and live execution when throughput is a primary constraint. The model retains nonlinear decision power while reducing computational overhead. Related background: Gradient boosting.

### XGBoost
XGBoost is a widely used gradient-boosted tree method with strong regularization controls. It is useful for FX because it performs well on noisy engineered tabular features. The algorithm is often a strong benchmark for directional and expected-move tasks. Further reading: XGBoost.

## Feature Set (A-Z)

Engineered model input features are grouped below by predictive role.

### Price Structure and Return Dynamics
These features encode near-term directional behavior, momentum decay, and micro mean-reversion signatures from M1 price action. They are useful in FX because short-horizon returns and candle geometry often react quickly to order-flow imbalance. This group forms the core directional signal layer before higher-level context is applied.

- 1-bar return normalized (M1)
- 3-bar return normalized (M1)
- 5-bar return normalized (M1)
- Close-to-Low edge (M1 OHLC)
- Close-to-Open edge (M1 OHLC)
- High-to-Close edge (M1 OHLC)
- Return variance proxy normalized (M1, window 10)
- RSI(14) normalized (M1)
- Rolling median edge (window 21)
- Z-score vs rolling mean/std (M1, window 10)

### Trend and Multi-Timeframe Structure
This group captures trend persistence and slope alignment across M1 through H1 horizons. It is useful in FX prediction because many profitable intraday setups depend on higher-timeframe drift and local pullback structure agreeing. Smoothing filters in this group also reduce noise sensitivity and improve model stability in volatile sessions.

- Ehlers Super Smoother (2-pole, period 20) edge
- EMA(100) edge (H1)
- EMA(100) edge (M15)
- EMA(100) edge (M30)
- EMA(100) edge (M5)
- EMA(200) edge (H1)
- EMA(200) edge (M15)
- EMA(200) edge (M30)
- EMA(200) edge (M5)
- H1 return (aligned)
- H1 slope (aligned)
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
- Quadruple-smoothed DEMA(100) edge
- Quadruple-smoothed DEMA(200) edge

### Volatility and Range Regime
These features estimate current market variability and effective move potential under different volatility definitions. They are useful for FX because entry quality depends not only on direction but also on whether expected movement can cover trading costs. This group improves risk-aware classification, skip decisions, and expected-value filtering.

- ATR(14) normalized (M1)
- Garman-Klass volatility (OHLC)
- Hampel robust score (rolling median/MAD, window 21)
- High-Low range normalized (M1 OHLC)
- NATR(14) (M1)
- Parkinson volatility (HL-only, window 20)
- Rogers-Satchell volatility (OHLC)

### Cross-Symbol Market Context
These features summarize behavior from related instruments to capture correlation and breadth effects. They are useful in FX because major pairs and macro-sensitive symbols often provide early context for risk-on and risk-off rotations. This group adds regime information that single-symbol models usually miss.

- Context return dispersion (cross-symbol)
- Context return mean (cross-symbol)
- Context up-breadth ratio (cross-symbol)

### Time and Execution Friction Context
This group captures temporal structure and direct transaction-cost pressure on signals. It is useful in FX because trade quality varies strongly by weekday, hour, and minute due to liquidity cycles. Including spread and clock-time context helps models avoid low-quality entries during expensive or structurally noisy periods.

- Hour of day (MT5 bar time)
- Minute of hour (MT5 bar time)
- Spread normalized
- Weekday (MT5 bar time)

## Normalization Techniques

Normalization is handled by a shared pipeline and can be selected through `AI_FeatureNormalization`. The project uses both market-structure-aware scaling and statistical transforms so models can remain stable across changing volatility regimes. Grouping methods this way makes backtest optimization easier because you can search by family before fine-tuning model-specific hyperparameters. The goal is to maximize signal consistency while preserving economically meaningful price structure.

### Market-Structure and Volatility-Aware Scaling
These methods preserve FX microstructure context by scaling signals with volatility or candle geometry rather than treating all bars as statistically identical. They are useful when session liquidity, spread pressure, and volatility clustering shift quickly across the day. In practice, they help reduce false positives during noisy periods while keeping directional moves tradable during expansions. This group is often the strongest first choice for intraday FX where cost-adjusted edge matters more than pure statistical fit.

- `Regime-Scaled Z + Clipped` (`FXAI_NORM_EXISTING`)
- `Candle Geometry` (`FXAI_NORM_CANDLE_GEOMETRY`)
- `Volatility-Scaled Returns` (`FXAI_NORM_VOL_STD_RETURNS`)
- `ATR/NATR Unit Scaling` (`FXAI_NORM_ATR_NATR_UNIT`)

### Statistical Distribution and Robust Shape Transforms
These methods reshape feature distributions so models can train on smoother, more comparable inputs across symbols and periods. They are useful in FX because engineered features often have heavy tails, skew, and regime-dependent dispersion that can destabilize learning. Robust and quantile-based transforms also reduce sensitivity to outliers from spikes and thin-liquidity prints. This group is valuable when you want better calibration and more stable model behavior in long optimization runs.

- `Min/Max with 5% Buffer` (`FXAI_NORM_MINMAX_BUFFER5`)
- `Z-Score Standardization` (`FXAI_NORM_ZSCORE`)
- `Robust Median/IQR` (`FXAI_NORM_ROBUST_MEDIAN_IQR`)
- `Quantile to Normal` (`FXAI_NORM_QUANTILE_TO_NORMAL`)
- `Power Transform (Yeo-Johnson) + Standardize` (`FXAI_NORM_POWER_YEOJOHNSON`)

### Relative Change and Directional Encodings
These methods emphasize feature movement dynamics from one bar to the next, which is critical for short-horizon decisioning. They are useful for FX because momentum shifts and micro mean-reversion often appear first as relative or signed change rather than absolute level. Binary directional encoding can also simplify noisy features for lightweight models and ensemble voting. This group is especially practical when targeting high trade frequency with tight latency and robust online adaptation.

- `Change in %` (`FXAI_NORM_CHANGE_PERCENT`)
- `Relative Change %` (`FXAI_NORM_RELATIVE_CHANGE_PERCENT`)
- `Log Return Transform` (`FXAI_NORM_LOG_RETURN`)
- `Binary 0/1 Direction` (`FXAI_NORM_BINARY_01`)

### Adaptive Window/Instance Normalization for Deep Models
These methods adapt normalization to current-window context rather than relying only on static global assumptions. They are useful in FX because deep sequence models can drift when the mean and scale of features change across sessions and regimes. RevIN improves invariance by normalizing per instance/window, while DAIN adds lightweight learnable adaptation for dynamic rescaling and gating. This group is most useful for transformer/recurrent plugins where distribution drift can strongly impact inference quality.

- `RevIN` (`FXAI_NORM_REVIN`)
- `DAIN` (`FXAI_NORM_DAIN`)

## Notes

- This project is for research and systematic strategy development.
- No model guarantees profit; robust validation and risk control are required.
- Use realistic spread/commission/slippage settings in tester for meaningful results.
