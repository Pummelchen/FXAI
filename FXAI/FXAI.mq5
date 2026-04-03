//+------------------------------------------------------------------+
//|                                                         FXAI.mq5 |
//| FXAI modular EA: plugin-based AI + equity trailing + equity SL   |
//+------------------------------------------------------------------+
#property strict

#include <Trade\Trade.mqh>
#include "Engine\core.mqh"
#include "Engine\data_pipeline.mqh"
#include "API\api.mqh"

CTrade trade;

enum ENUM_FXAI_POSITION_SIZING
{
   FXAI_SIZE_FIXED_LOT = 0,
   FXAI_SIZE_CONVICTION = 1,
   FXAI_SIZE_VOL_TARGET = 2
};

//-------------------------- INPUTS ---------------------------------
input ENUM_AI_TYPE AI_Type = AI_SGD_LOGIT;
// Models: all (selector for which plugin runs in single-model mode).
// Purpose: chooses the active prediction model implementation.
// Importance/Range: enum value 0..FXAI_AI_COUNT-1; use backtests to pick best per symbol.
input bool   AI_Ensemble          = false;
// Models: all (ensemble controller across all plugins).
// Purpose: enables multi-model voting instead of one selected model.
// Importance/Range: false/true; true is usually more stable but may reduce trade count.
input double Ensemble_AgreePct    = 70.0;
// Models: all (only used when AI_Ensemble=true).
// Purpose: sets the minimum vote percentage needed for BUY or SELL.
// Importance/Range: practical 50..90; higher means fewer but stricter entries.
input double Ensemble_ExplorePct  = 12.0;
// Models: all (ensemble candidate selection only).
// Purpose: with this probability per bar, add one non-top model for exploration.
// Importance/Range: practical 0..30; higher speeds adaptation but can add noisy votes.
input int    Ensemble_ShadowEveryBars = 3;
// Models: all (ensemble maintenance path only).
// Purpose: cadence for shadow-training non-active models to avoid stale/pruned starvation.
// Importance/Range: practical 2..10; lower adapts faster but costs more CPU.
input int    Ensemble_ShadowSamples = 20;
// Models: all (ensemble maintenance path only).
// Purpose: number of recent prepared samples used for each shadow update.
// Importance/Range: practical 8..80; small values keep CPU low.
input int    Ensemble_ShadowEpochs = 1;
// Models: all (ensemble maintenance path only).
// Purpose: epochs used by shadow updates for non-active models.
// Importance/Range: practical 1..2; higher may overfit and increase runtime.


input double AI_BuyThreshold         = 0.60;
// Models: all (3-class head + adaptive entry gate).
// Purpose: base minimum BUY confidence; runtime logic tightens it by cost/volatility regime.
// Importance/Range: common 0.55..0.80; must stay in (0,1) and above AI_SellThreshold.
input double AI_SellThreshold        = 0.40;
// Models: all (3-class head + adaptive entry gate).
// Purpose: base SELL side threshold input (lower than buy); runtime maps it to sell confidence.
// Importance/Range: common 0.20..0.45; must stay in (0,1) and below AI_BuyThreshold.
input int    AI_M1SyncBars          = 3;
// Models: m1sync only (strict micro-momentum filter).
// Purpose: number of completed M1 closes that must align with the current price in one direction.
// Importance/Range: practical 2..8; higher is stricter and lowers trade count, code clamps to 2..12.

input bool   AI_Warmup        = false;
// Models: all (startup trainer/tuner for each plugin).
// Purpose: pre-trade warmup using 10,000 M1 bars and 100 tuning loops/model.
// Importance/Range: false/true; true improves initial calibration but adds startup latency/CPU.
input int    AI_WarmupSamples = 10000;
// Models: all (warmup optimizer/trainer input).
// Purpose: number of historical M1 samples used for warmup dataset build.
// Importance/Range: practical 3000..30000; larger helps stability but increases startup time.
input int    AI_WarmupLoops   = 100;
// Models: all (warmup optimizer search budget).
// Purpose: number of hyperparameter trials per model during warmup.
// Importance/Range: practical 20..300; higher can improve fit but costs CPU.
input int    AI_WarmupFolds   = 3;
// Models: all (warmup validation robustness).
// Purpose: walk-forward validation folds used to rank warmup trials.
// Importance/Range: practical 2..5; higher is more robust but slower.
input int    AI_WarmupSeed    = 42;
// Models: all (warmup reproducibility).
// Purpose: deterministic random seed for warmup trial sampling.
// Importance/Range: any non-negative integer; same seed gives repeatable tuning.
input int    AI_WarmupMinTrades = 120;
// Models: all (warmup objective guardrail).
// Purpose: minimum validation trade count target while selecting warmup params.
// Importance/Range: practical 50..400; higher avoids overfitting to tiny sample counts.


input double TP_USD = 100;
// Models: all (trade/risk layer, independent of model math).
// Purpose: equity target for cycle take-profit handling.
// Importance/Range: practical 20..300 on small lots; too low closes winners early.
input double SL_USD = 5.0;
// Models: all (trade/risk layer, independent of model math).
// Purpose: hard equity stop for open-cycle drawdown.
// Importance/Range: practical 3..30; lower protects capital but can cut valid setups.
input double Lot    = 0.01;
// Models: all (execution sizing for all model signals).
// Purpose: order volume before broker min/max/step normalization.
// Importance/Range: broker-dependent, often 0.01..1.00; main driver of PnL volatility.
input ENUM_FXAI_POSITION_SIZING AI_PositionSizing = FXAI_SIZE_CONVICTION;
// Models: all (live sizing and allocation layer).
// Purpose: selects fixed, conviction-scaled, or volatility-targeted live position sizing.
// Importance/Range: enum 0..2; conviction or vol-target sizing is safer for serious deployment.
input double RiskPerTradePct = 0.35;
// Models: all (portfolio/risk layer).
// Purpose: caps per-trade capital at risk as a percent of account equity.
// Importance/Range: practical 0.05..2.00; 0 disables risk-budget capping.
input double RiskTargetMovePoints = 12.0;
// Models: all (portfolio/risk layer).
// Purpose: fallback adverse-move estimate used when sizing from volatility/risk budget.
// Importance/Range: symbol-dependent, practical 3..60 points.
input double MaxPortfolioExposureLots = 0.30;
// Models: all (portfolio/risk layer).
// Purpose: caps total managed gross exposure across all symbols for this magic number.
// Importance/Range: 0 disables; otherwise set to account/broker appropriate max lots.
input double MaxCorrelatedExposureLots = 0.20;
// Models: all (portfolio/risk layer).
// Purpose: caps FXAI exposure in symbols sharing major currency risk with the active symbol.
// Importance/Range: 0 disables; practical value is usually below portfolio cap.
input double MaxDirectionalClusterLots = 0.18;
// Models: all (portfolio/risk layer).
// Purpose: caps aligned directional exposure across correlated currency clusters.
// Importance/Range: 0 disables; use below correlated exposure cap for multi-symbol portfolios.
input double RiskMinConfidence = 0.52;
// Models: all (trade admission layer).
// Purpose: blocks entries when model confidence is below the live quality floor.
// Importance/Range: 0..1; higher is stricter.
input double RiskMinReliability = 0.48;
// Models: all (trade admission layer).
// Purpose: blocks entries when recent realized reliability is too weak.
// Importance/Range: 0..1; higher is stricter.
input double RiskMaxPathRisk = 0.72;
// Models: all (trade admission layer).
// Purpose: blocks entries when path-risk forecasts imply poor MAE/hit-time shape.
// Importance/Range: 0..1; lower is stricter.
input double RiskMaxFillRisk = 0.68;
// Models: all (trade admission layer).
// Purpose: blocks entries when fill-risk forecasts imply poor execution quality.
// Importance/Range: 0..1; lower is stricter.
input double RiskMinTradeGate = 0.52;
// Models: all (trade admission layer).
// Purpose: blocks entries when the learned trade gate is below the live acceptance floor.
// Importance/Range: 0..1; higher is stricter.
input double RiskMinHierarchyScore = 0.46;
// Models: all (hierarchical decision layer).
// Purpose: blocks entries when the multi-head hierarchy score is too weak.
// Importance/Range: 0..1; higher is stricter.
input double RiskMinHierarchyConsistency = 0.40;
// Models: all (hierarchical decision layer).
// Purpose: blocks entries when direction, move, path, and execution heads disagree too much.
// Importance/Range: 0..1; higher is stricter.
input double RiskMinHierarchyTradability = 0.38;
// Models: all (hierarchical decision layer).
// Purpose: blocks entries when tradability quality from the hierarchy is too weak.
// Importance/Range: 0..1; higher is stricter.
input double RiskMinHierarchyExecution = 0.34;
// Models: all (hierarchical decision layer).
// Purpose: blocks entries when execution viability from the hierarchy is too weak.
// Importance/Range: 0..1; higher is stricter.
input double RiskMinMacroStateQuality = 0.24;
// Models: all (macro-state layer).
// Purpose: scales or blocks entries when macro-state coverage is too weak for event-aware routing.
// Importance/Range: 0..1; only active when a leakage-safe macro dataset is present.
input double RiskMaxPortfolioPressure = 0.78;
// Models: all (portfolio-native runtime).
// Purpose: blocks entries when gross, correlated, and directional cluster pressure are too high.
// Importance/Range: 0..1; lower is stricter.
input double RiskKillTradeGate = 0.24;
// Models: all (open-position safety layer).
// Purpose: forces exit when the live gate collapses under the post-entry kill threshold.
// Importance/Range: 0..1; lower reduces unnecessary churn.
input double RiskKillPathRisk = 0.92;
// Models: all (open-position safety layer).
// Purpose: forces exit when path risk rises into a severe adverse-selection regime.
// Importance/Range: 0..1; lower is more defensive.
input double RiskKillFillRisk = 0.90;
// Models: all (open-position safety layer).
// Purpose: forces exit when fill-risk shifts into a severe execution-stress regime.
// Importance/Range: 0..1; lower is more defensive.
input ulong  TradeMagic = 6206001;
// Models: all (execution ownership / trade isolation).
// Purpose: magic number attached to FXAI orders and used to manage only this EA's own trades.
// Importance/Range: any positive integer; change it when multiple EAs share one account.
input double MaxDD  = 50;
// Models: all (EA-level safety brake).
// Purpose: stops EA when equity drops below allowed drawdown fraction.
// Importance/Range: 5..60 typical; high values increase ruin risk.
input int    TradeKiller = 0;
// Models: all (trade lifecycle safety gate).
// Purpose: if >0, force-close all open trades after X minutes in market.
// Importance/Range: 0 disables; practical 5..240 depending on strategy horizon.

//--------------------- EQUITY TRAIL INPUTS --------------------------
input bool   TrailEnabled     = true;
// Models: all (post-entry equity management).
// Purpose: turns equity trailing stop logic on/off.
// Importance/Range: false/true; true can preserve gains in volatile periods.
input double TrailStartUSD    = 5.0;
// Models: all (post-entry equity management).
// Purpose: profit level where trailing lock starts.
// Importance/Range: practical 2..30; smaller starts earlier but can over-tighten.
input double TrailGivebackPct = 30.0;
// Models: all (post-entry equity management).
// Purpose: allowed giveback from peak before forced exit.
// Importance/Range: 10..60 common; lower locks more, higher lets trades breathe.
input double TrailTPBreathUSD = 5.0;
// Models: all (post-entry equity management).
// Purpose: lifts TP baseline above peak to avoid premature cap.
// Importance/Range: 1..20 typical; too high can delay realized profits.

// AI core
input int    AI_Window        = 60;
// Models: all (core trainer for every plugin).
// Purpose: bootstrap training sample count on first model fit.
// Importance/Range: clamped 50..500; larger is stabler but heavier CPU/latency.
input int    AI_Epochs        = 8;
// Models: all (core trainer for every plugin).
// Purpose: number of full passes over the bootstrap window.
// Importance/Range: clamped 1..20 in code; values above 20 are auto-limited.
input double AI_LearningRate  = 0.01;
// Models: all shared-linear updates (at minimum SGD and compatible plugins).
// Purpose: update step size for gradient-style parameter updates.
// Importance/Range: clamped 0.001..0.200; common 0.005..0.05 to avoid instability.
input double AI_L2            = 0.010;
// Models: all shared-linear updates (at minimum SGD and compatible plugins).
// Purpose: L2 shrinkage to reduce overfitting/noise sensitivity.
// Importance/Range: clamped 0..0.1; common 0.0005..0.02 for FX.

input int    PredictionTargetMinutes = 5;
// Models: all (label horizon and expected-move horizon).
// Purpose: forecast distance in minutes for training labels and signal EV.
// Importance/Range: clamped 1..720; common 3..30 for intraday FX.
input bool   AI_MultiHorizon = true;
// Models: all (training + signal router).
// Purpose: enables runtime horizon routing over a configured horizon set.
// Importance/Range: false/true; true helps regime adaptation.
input string AI_Horizons = "{3, 5, 8, 13}";
// Models: all (training + signal router).
// Purpose: candidate minute horizons used by the router (base horizon auto-included).
// Importance/Range: 1..8 unique horizons in [1..720], ordered automatically.
input double AI_HorizonPenaltyPerMinute = 0.0015;
// Models: all (training + signal router).
// Purpose: holding-time penalty in horizon scoring to avoid over-long routes.
// Importance/Range: practical 0.0000..0.0100; higher favors shorter horizons.


input double AI_CommissionPerLotSide = 0.0;
// Models: all (cost-aware labeling + EV gating).
// Purpose: adds per-side commission into roundtrip point cost.
// Importance/Range: 0..10+ broker-dependent; accurate value is critical for realism.
input ENUM_FXAI_EXECUTION_PROFILE AI_ExecutionProfile = FXAI_EXEC_DEFAULT;
// Models: all (execution-parity layer across live, warmup, and audit paths).
// Purpose: selects a broker or venue style preset for cost/slippage/fill assumptions.
// Importance/Range: enum 0..4; use the preset closest to the intended deployment venue.
input double AI_CostBufferPoints     = 2.0;
// Models: all (cost-aware labeling + EV gating).
// Purpose: extra safety buffer above spread/commission for slippage/noise.
// Importance/Range: common 0..5 points; higher reduces false-positive entries.
input double AI_ExecutionSlippageOverride = -1.0;
// Models: all (execution-parity layer).
// Purpose: overrides the preset slippage points when set >= 0.
// Importance/Range: -1 keeps preset; otherwise use measured live/tester median slippage.
input double AI_ExecutionFillPenaltyOverride = -1.0;
// Models: all (execution-parity layer).
// Purpose: overrides the preset fill penalty points when set >= 0.
// Importance/Range: -1 keeps preset; otherwise use measured partial-fill/requote drag.
input double AI_EVThresholdPoints    = 0.30;
// Models: all (label classing + final EV decision gate).
// Purpose: minimum expected-value edge (in points) required to trade.
// Importance/Range: common 0.1..1.5; higher means fewer but higher-margin signals.
input int    AI_EVLookbackSamples    = 80;
// Models: all (fallback expected-move estimator; key for non-quantile plugins).
// Purpose: sample length for average absolute move estimation.
// Importance/Range: clamped 20..400; common 50..150 for adaptive but stable EV.

input int    AI_OnlineSamples = 60;
// Models: all (incremental learning stage).
// Purpose: bars used each new closed M1 bar for online updates.
// Importance/Range: clamped 5..200; higher adapts slower but is less noisy.
input int    AI_OnlineEpochs  = 1;
// Models: all (incremental learning stage).
// Purpose: number of passes over online sample window per update.
// Importance/Range: clamped 1..5; common 1..2 to keep CPU low and avoid overfit.

input bool   AI_DebugFlow     = false;
// Models: all (diagnostics only, no model math impact).
// Purpose: prints one-per-bar reasons for trade blocks/no-signal paths.
// Importance/Range: false/true; enable during debugging, disable for production speed.
input bool   AI_ComplianceHarness = false;
// Models: all (framework/API validation only).
// Purpose: runs plugin API v4 compliance checks on init before trading starts.
// Importance/Range: false/true; enable when validating framework changes, disable for production speed.
input ENUM_FXAI_FEATURE_NORMALIZATION AI_FeatureNormalization = FXAI_NORM_EXISTING;
// Models: all (shared feature pipeline before all plugin train/predict calls).
// Purpose: selects the feature normalization method applied to prepared feature vectors.
// Importance/Range: enum 0..14 (adds 11=QuantileToNormal, 12=PowerYeoJohnson, 13=RevIN, 14=DAIN).

// Session/liquidity filter
input bool SessionFilterEnabled        = false;
// Models: all (pre-trade execution gate).
// Purpose: blocks trading outside configured liquid session windows.
// Importance/Range: false/true; true can avoid dead hours and poor fills.
input int  SessionMinAfterOpenMinutes  = 60;
// Models: all (used when SessionFilterEnabled=true).
// Purpose: delay after session open before allowing entries.
// Importance/Range: common 5..60; larger avoids open spikes but skips early moves.
input int  SessionMinBeforeCloseMinutes= 60;
// Models: all (used when SessionFilterEnabled=true).
// Purpose: stop entering shortly before session close.
// Importance/Range: common 5..60; larger reduces close-liquidity risk.

// Multi-symbol context features
input string AI_ContextSymbols = "{EURUSD, USDJPY, AUDUSD, EURAUD, EURGBP, GBPUSD}";
// Models: all (feature-engine input for cross-symbol context).
// Purpose: symbol list used to compute market breadth/dispersion context features.
// Importance/Range: 1..32 symbols parsed; use liquid, structurally related instruments.

// FTRL-Proximal
input double FTRL_Alpha = 0.08;
// Models: FTRL only.
// Purpose: base learning-rate control for FTRL-Proximal updates.
// Importance/Range: clamped 0.001..1.0; common 0.03..0.2.
input double FTRL_Beta  = 1.00;
// Models: FTRL only.
// Purpose: smoothing term in adaptive denominator for FTRL.
// Importance/Range: clamped 0..5.0; common 0.5..2.0.
input double FTRL_L1    = 0.0005;
// Models: FTRL only.
// Purpose: L1 sparsity pressure on FTRL weights.
// Importance/Range: clamped 0..0.1; common 0..0.005.
input double FTRL_L2    = 0.0100;
// Models: FTRL only.
// Purpose: L2 regularization for FTRL stability/generalization.
// Importance/Range: clamped 0..1.0; common 0.001..0.05.

// Passive-Aggressive
input double PA_C       = 4.00;
// Models: PA only.
// Purpose: aggressiveness cap on PA correction step size.
// Importance/Range: clamped 0.01..10; common 0.5..6.0.
input double PA_Margin  = 1.20;
// Models: PA only.
// Purpose: target margin before no further PA update is needed.
// Importance/Range: clamped 0.1..2.0; common 0.6..1.6.

// XGB-like split learners
input double XGB_FastLearningRate = 0.03;
// Models: XGB_Fast, XGBoost, LightGBM, CatBoost.
// Purpose: shrinkage factor for split-learner updates.
// Importance/Range: clamped 0.001..0.3; CatBoost common 0.02..0.05.
input double XGB_FastL2           = 4.0;
// Models: XGB_Fast, XGBoost, LightGBM, CatBoost.
// Purpose: L2 regularization for split learner coefficients.
// Importance/Range: clamped 0..10; CatBoost common 3..8.
input double XGB_SplitThreshold   = 0.00;
// Models: XGB_Fast, XGBoost, LightGBM, CatBoost.
// Purpose: feature split pivot used by lightweight tree-like logic.
// Importance/Range: clamped -2..2; common -0.5..0.5.

// Tiny MLP
input double MLP_LearningRate = 0.010;
// Models: MLP_TINY only.
// Purpose: gradient step size for tiny neural network updates.
// Importance/Range: clamped 0.0005..0.05; common 0.003..0.02.
input double MLP_L2           = 0.0005;
// Models: MLP_TINY only.
// Purpose: L2 regularization for MLP weights.
// Importance/Range: clamped 0..0.05; common 0.0001..0.01.
input double MLP_InitScale    = 0.10;
// Models: MLP_TINY only.
// Purpose: initial random weight amplitude.
// Importance/Range: clamped 0.01..0.5; common 0.05..0.2 for stable starts.

// Deep TCN stack
input int TCN_Layers = 4;
// Models: TCN only.
// Purpose: number of dilated residual blocks in the temporal stack.
// Importance/Range: clamped 2..8; common 3..6, deeper increases context and CPU.
input int TCN_KernelSize = 3;
// Models: TCN only.
// Purpose: causal kernel width per block (how many lag taps each block sees).
// Importance/Range: clamped 2..5; common 2..4, larger widens receptive field.
input int TCN_DilationBase = 2;
// Models: TCN only.
// Purpose: per-layer dilation growth base (1=linear, 2=exponential).
// Importance/Range: clamped 1..3; common 2, higher grows context faster.

// Quantile regressor
input double Quantile_LearningRate = 0.015;
// Models: QUANTILE only.
// Purpose: update speed for quantile heads (q20/q50/q80).
// Importance/Range: clamped 0.0001..0.1; common 0.005..0.03.
input double Quantile_L2           = 0.0005;
// Models: QUANTILE only.
// Purpose: L2 regularization for quantile weight vectors.
// Importance/Range: clamped 0..0.1; common 0.0001..0.01.

// Elastic-net hashed interactions
input double ENHash_LearningRate = 0.020;
// Models: ENHASH only.
// Purpose: update speed for linear and hashed interaction weights.
// Importance/Range: clamped 0.0005..0.1; common 0.005..0.03.
input double ENHash_L1           = 0.0001;
// Models: ENHASH only.
// Purpose: L1 term to promote sparse robust interaction weights.
// Importance/Range: clamped 0..0.1; common 0..0.005.
input double ENHash_L2           = 0.0020;
// Models: ENHASH only.
// Purpose: L2 term for smoothness and overfit control.
// Importance/Range: clamped 0..0.1; common 0.0005..0.02.


//------------------------ GLOBAL VARS -------------------------------
double InitialEquity;
double TP_Value = 0;
double EquiMax  = 0;
double RealizedManagedProfit = 0.0;

int CloseCounter = 1;

bool   CycleActive      = false;
double CycleEntryEquity = 0.0;
double CycleEntryManagedPnl = 0.0;
double CycleEntryRealizedProfit = 0.0;
datetime CycleStartTime = 0;

bool   TrailTracking   = false;
double TrailPeakProfit = 0.0;

CFXAIAIRegistry g_plugins;
bool g_plugins_ready = false;

string   g_ai_last_symbol = "";
bool     g_ai_trained[FXAI_AI_COUNT];
datetime g_ai_last_train_bar[FXAI_AI_COUNT];
datetime g_ai_last_signal_bar = 0;
int      g_ai_last_signal = -1;
int      g_ai_last_signal_key = -1;
string   g_ai_last_reason = "init";
double   g_ai_last_expected_move_points = 0.0;
double   g_ai_last_trade_edge_points = 0.0;
double   g_ai_last_confidence = 0.0;
double   g_ai_last_reliability = 0.0;
double   g_ai_last_path_risk = 1.0;
double   g_ai_last_fill_risk = 1.0;
double   g_ai_last_trade_gate = 0.0;
double   g_ai_last_hierarchy_score = 0.0;
double   g_ai_last_hierarchy_consistency = 0.0;
double   g_ai_last_hierarchy_tradability = 0.0;
double   g_ai_last_hierarchy_execution = 0.0;
double   g_ai_last_hierarchy_horizon_fit = 0.0;
double   g_ai_last_macro_state_quality = 0.0;
double   g_ai_last_portfolio_pressure = 0.0;
double   g_ai_last_context_quality = 0.0;
double   g_ai_last_context_strength = 0.0;
double   g_ai_last_min_move_points = 0.0;
int      g_ai_last_horizon_minutes = 0;
int      g_ai_last_regime_id = 0;
datetime g_last_debug_bar = 0;

#define FXAI_MAX_CONTEXT_SYMBOLS 48
#define FXAI_REL_MAX_PENDING 2048
#define FXAI_REGIME_COUNT 12
#define FXAI_MAX_HORIZONS 8
#define FXAI_STACK_FEATS 84
#define FXAI_STACK_HIDDEN 28
#define FXAI_TRADE_GATE_FEATS FXAI_STACK_FEATS
#define FXAI_TRADE_GATE_HIDDEN 16
#define FXAI_HPOL_FEATS 48
#define FXAI_HPOL_HIDDEN 16
#define FXAI_NORM_CAND_MAX 8
#define FXAI_REPLAY_CAPACITY 384
#define FXAI_REPLAY_DRAWS 12
#define FXAI_PATHFLAG_DUAL_HIT 1
#define FXAI_PATHFLAG_KILLED_EARLY 2
#define FXAI_PATHFLAG_SPREAD_STRESS 4
#define FXAI_PATHFLAG_SLOW_HIT 8
#define FXAI_REPLAYFLAG_FALSE_POS 16
#define FXAI_REPLAYFLAG_MISSED_MOVE 32
#define FXAI_REPLAYFLAG_WRONG_DIR 64

string g_context_symbols[];
double g_context_symbol_utility[FXAI_MAX_CONTEXT_SYMBOLS];
bool   g_context_symbol_utility_ready[FXAI_MAX_CONTEXT_SYMBOLS];
double g_context_symbol_stability[FXAI_MAX_CONTEXT_SYMBOLS];
double g_context_symbol_lead[FXAI_MAX_CONTEXT_SYMBOLS];
double g_context_symbol_coverage[FXAI_MAX_CONTEXT_SYMBOLS];
int    g_horizon_minutes[];
double g_model_reliability[FXAI_AI_COUNT];
double g_model_abs_move_ema[FXAI_AI_COUNT];
bool   g_model_abs_move_ready[FXAI_AI_COUNT];
double g_model_meta_weight[FXAI_AI_COUNT];
double g_model_global_edge_ema[FXAI_AI_COUNT];
bool   g_model_global_edge_ready[FXAI_AI_COUNT];
double g_model_regime_edge_ema[FXAI_AI_COUNT][FXAI_REGIME_COUNT];
bool   g_model_regime_edge_ready[FXAI_AI_COUNT][FXAI_REGIME_COUNT];
int    g_model_regime_obs[FXAI_AI_COUNT][FXAI_REGIME_COUNT];
FXAIAIHyperParams g_model_hp[FXAI_AI_COUNT];
bool g_model_hp_ready[FXAI_AI_COUNT];
FXAIAIHyperParams g_model_hp_horizon[FXAI_AI_COUNT][FXAI_MAX_HORIZONS];
bool g_model_hp_horizon_ready[FXAI_AI_COUNT][FXAI_MAX_HORIZONS];
FXAIAIHyperParams g_model_hp_bank[FXAI_AI_COUNT][FXAI_REGIME_COUNT][FXAI_MAX_HORIZONS];
bool g_model_hp_bank_ready[FXAI_AI_COUNT][FXAI_REGIME_COUNT][FXAI_MAX_HORIZONS];
int g_model_norm_method[FXAI_AI_COUNT];
bool g_model_norm_ready[FXAI_AI_COUNT];
int g_model_norm_method_horizon[FXAI_AI_COUNT][FXAI_MAX_HORIZONS];
bool g_model_norm_horizon_ready[FXAI_AI_COUNT][FXAI_MAX_HORIZONS];
int g_model_norm_method_bank[FXAI_AI_COUNT][FXAI_REGIME_COUNT][FXAI_MAX_HORIZONS];
bool g_model_norm_bank_ready[FXAI_AI_COUNT][FXAI_REGIME_COUNT][FXAI_MAX_HORIZONS];
double g_model_buy_thr[FXAI_AI_COUNT];
double g_model_sell_thr[FXAI_AI_COUNT];
bool g_model_thr_ready[FXAI_AI_COUNT];
double g_model_buy_thr_horizon[FXAI_AI_COUNT][FXAI_MAX_HORIZONS];
double g_model_sell_thr_horizon[FXAI_AI_COUNT][FXAI_MAX_HORIZONS];
bool   g_model_thr_horizon_ready[FXAI_AI_COUNT][FXAI_MAX_HORIZONS];
double g_model_buy_thr_regime[FXAI_AI_COUNT][FXAI_REGIME_COUNT];
double g_model_sell_thr_regime[FXAI_AI_COUNT][FXAI_REGIME_COUNT];
bool   g_model_thr_regime_ready[FXAI_AI_COUNT][FXAI_REGIME_COUNT];
double g_model_buy_thr_bank[FXAI_AI_COUNT][FXAI_REGIME_COUNT][FXAI_MAX_HORIZONS];
double g_model_sell_thr_bank[FXAI_AI_COUNT][FXAI_REGIME_COUNT][FXAI_MAX_HORIZONS];
bool   g_model_thr_bank_ready[FXAI_AI_COUNT][FXAI_REGIME_COUNT][FXAI_MAX_HORIZONS];
double g_model_horizon_edge_ema[FXAI_AI_COUNT][FXAI_MAX_HORIZONS];
bool   g_model_horizon_edge_ready[FXAI_AI_COUNT][FXAI_MAX_HORIZONS];
int    g_model_horizon_obs[FXAI_AI_COUNT][FXAI_MAX_HORIZONS];
double g_model_context_edge_ema[FXAI_AI_COUNT][FXAI_REGIME_COUNT][FXAI_MAX_HORIZONS];
double g_model_context_regret_ema[FXAI_AI_COUNT][FXAI_REGIME_COUNT][FXAI_MAX_HORIZONS];
bool   g_model_context_ready[FXAI_AI_COUNT][FXAI_REGIME_COUNT][FXAI_MAX_HORIZONS];
int    g_model_context_obs[FXAI_AI_COUNT][FXAI_REGIME_COUNT][FXAI_MAX_HORIZONS];
double g_model_portfolio_mean_edge[FXAI_AI_COUNT];
double g_model_portfolio_stability[FXAI_AI_COUNT];
double g_model_portfolio_corr_penalty[FXAI_AI_COUNT];
double g_model_portfolio_diversification[FXAI_AI_COUNT];
double g_model_portfolio_objective[FXAI_AI_COUNT];
bool   g_model_portfolio_ready[FXAI_AI_COUNT];
int    g_model_portfolio_symbol_count[FXAI_AI_COUNT];
double g_model_plugin_route_value[FXAI_AI_COUNT][FXAI_REGIME_COUNT][FXAI_PLUGIN_SESSION_BUCKETS][FXAI_MAX_HORIZONS];
double g_model_plugin_route_regret[FXAI_AI_COUNT][FXAI_REGIME_COUNT][FXAI_PLUGIN_SESSION_BUCKETS][FXAI_MAX_HORIZONS];
double g_model_plugin_route_counterfactual[FXAI_AI_COUNT][FXAI_REGIME_COUNT][FXAI_PLUGIN_SESSION_BUCKETS][FXAI_MAX_HORIZONS];
bool   g_model_plugin_route_ready[FXAI_AI_COUNT][FXAI_REGIME_COUNT][FXAI_PLUGIN_SESSION_BUCKETS][FXAI_MAX_HORIZONS];
int    g_model_plugin_route_obs[FXAI_AI_COUNT][FXAI_REGIME_COUNT][FXAI_PLUGIN_SESSION_BUCKETS][FXAI_MAX_HORIZONS];
double g_horizon_regime_edge_ema[FXAI_REGIME_COUNT][FXAI_MAX_HORIZONS];
bool   g_horizon_regime_edge_ready[FXAI_REGIME_COUNT][FXAI_MAX_HORIZONS];
int    g_horizon_regime_obs[FXAI_REGIME_COUNT][FXAI_MAX_HORIZONS];
double g_horizon_regime_total_obs[FXAI_REGIME_COUNT];
bool g_ai_warmup_done = false;
int      g_rel_pending_seq[FXAI_AI_COUNT][FXAI_REL_MAX_PENDING];
double   g_rel_pending_prob[FXAI_AI_COUNT][FXAI_REL_MAX_PENDING][3];
int      g_rel_pending_signal[FXAI_AI_COUNT][FXAI_REL_MAX_PENDING];
int      g_rel_pending_regime[FXAI_AI_COUNT][FXAI_REL_MAX_PENDING];
int      g_rel_pending_session[FXAI_AI_COUNT][FXAI_REL_MAX_PENDING];
double   g_rel_pending_expected_move[FXAI_AI_COUNT][FXAI_REL_MAX_PENDING];
int      g_rel_pending_horizon[FXAI_AI_COUNT][FXAI_REL_MAX_PENDING];
int      g_rel_pending_head[FXAI_AI_COUNT];
int      g_rel_pending_tail[FXAI_AI_COUNT];
double   g_conf_class_score[FXAI_AI_COUNT][FXAI_REGIME_COUNT][FXAI_MAX_HORIZONS][FXAI_CONFORMAL_DEPTH];
double   g_conf_move_score[FXAI_AI_COUNT][FXAI_REGIME_COUNT][FXAI_MAX_HORIZONS][FXAI_CONFORMAL_DEPTH];
double   g_conf_path_score[FXAI_AI_COUNT][FXAI_REGIME_COUNT][FXAI_MAX_HORIZONS][FXAI_CONFORMAL_DEPTH];
int      g_conf_count[FXAI_AI_COUNT][FXAI_REGIME_COUNT][FXAI_MAX_HORIZONS];
int      g_conf_head[FXAI_AI_COUNT][FXAI_REGIME_COUNT][FXAI_MAX_HORIZONS];
int      g_conf_pending_seq[FXAI_AI_COUNT][FXAI_REL_MAX_PENDING];
int      g_conf_pending_regime[FXAI_AI_COUNT][FXAI_REL_MAX_PENDING];
int      g_conf_pending_horizon[FXAI_AI_COUNT][FXAI_REL_MAX_PENDING];
double   g_conf_pending_prob[FXAI_AI_COUNT][FXAI_REL_MAX_PENDING][3];
double   g_conf_pending_move_q25[FXAI_AI_COUNT][FXAI_REL_MAX_PENDING];
double   g_conf_pending_move_q50[FXAI_AI_COUNT][FXAI_REL_MAX_PENDING];
double   g_conf_pending_move_q75[FXAI_AI_COUNT][FXAI_REL_MAX_PENDING];
double   g_conf_pending_path_risk[FXAI_AI_COUNT][FXAI_REL_MAX_PENDING];
int      g_conf_pending_head[FXAI_AI_COUNT];
int      g_conf_pending_tail[FXAI_AI_COUNT];
int      g_stack_pending_seq[FXAI_REL_MAX_PENDING];
int      g_stack_pending_signal[FXAI_REL_MAX_PENDING];
int      g_stack_pending_regime[FXAI_REL_MAX_PENDING];
int      g_stack_pending_horizon[FXAI_REL_MAX_PENDING];
double   g_stack_pending_prob[FXAI_REL_MAX_PENDING][3];
double   g_stack_pending_feat[FXAI_REL_MAX_PENDING][FXAI_STACK_FEATS];
double   g_stack_pending_expected_move[FXAI_REL_MAX_PENDING];
int      g_stack_pending_head = 0;
int      g_stack_pending_tail = 0;
double   g_stack_w1[FXAI_REGIME_COUNT][FXAI_STACK_HIDDEN][FXAI_STACK_FEATS];
double   g_stack_b1[FXAI_REGIME_COUNT][FXAI_STACK_HIDDEN];
double   g_stack_w2[FXAI_REGIME_COUNT][3][FXAI_STACK_HIDDEN];
double   g_stack_b2[FXAI_REGIME_COUNT][3];
bool     g_stack_ready[FXAI_REGIME_COUNT];
int      g_stack_obs[FXAI_REGIME_COUNT];
double   g_router_action_value[FXAI_REGIME_COUNT][FXAI_MAX_HORIZONS][3];
double   g_router_action_regret[FXAI_REGIME_COUNT][FXAI_MAX_HORIZONS][3];
double   g_router_action_counterfactual[FXAI_REGIME_COUNT][FXAI_MAX_HORIZONS][3];
bool     g_router_action_ready[FXAI_REGIME_COUNT][FXAI_MAX_HORIZONS][3];
int      g_router_action_obs[FXAI_REGIME_COUNT][FXAI_MAX_HORIZONS][3];
double   g_trade_gate_w1[FXAI_REGIME_COUNT][FXAI_TRADE_GATE_HIDDEN][FXAI_TRADE_GATE_FEATS];
double   g_trade_gate_b1[FXAI_REGIME_COUNT][FXAI_TRADE_GATE_HIDDEN];
double   g_trade_gate_w2[FXAI_REGIME_COUNT][FXAI_TRADE_GATE_HIDDEN];
double   g_trade_gate_b2[FXAI_REGIME_COUNT];
bool     g_trade_gate_ready[FXAI_REGIME_COUNT];
int      g_trade_gate_obs[FXAI_REGIME_COUNT];
double   g_hpolicy_w1[FXAI_REGIME_COUNT][FXAI_HPOL_HIDDEN][FXAI_HPOL_FEATS];
double   g_hpolicy_b1[FXAI_REGIME_COUNT][FXAI_HPOL_HIDDEN];
double   g_hpolicy_w2[FXAI_REGIME_COUNT][FXAI_HPOL_HIDDEN];
double   g_hpolicy_b2[FXAI_REGIME_COUNT];
bool     g_hpolicy_ready[FXAI_REGIME_COUNT];
int      g_hpolicy_obs[FXAI_REGIME_COUNT];
double   g_meta_oof_score_ema[FXAI_REGIME_COUNT][FXAI_MAX_HORIZONS];
double   g_meta_oof_edge_ema[FXAI_REGIME_COUNT][FXAI_MAX_HORIZONS];
double   g_meta_oof_quality_ema[FXAI_REGIME_COUNT][FXAI_MAX_HORIZONS];
double   g_meta_oof_trade_rate_ema[FXAI_REGIME_COUNT][FXAI_MAX_HORIZONS];
bool     g_meta_oof_ready[FXAI_REGIME_COUNT][FXAI_MAX_HORIZONS];
int      g_meta_oof_obs[FXAI_REGIME_COUNT][FXAI_MAX_HORIZONS];
int      g_hpolicy_pending_seq[FXAI_REL_MAX_PENDING];
int      g_hpolicy_pending_regime[FXAI_REL_MAX_PENDING];
int      g_hpolicy_pending_horizon[FXAI_REL_MAX_PENDING];
double   g_hpolicy_pending_min_move[FXAI_REL_MAX_PENDING];
double   g_hpolicy_pending_feat[FXAI_REL_MAX_PENDING][FXAI_HPOL_FEATS];
int      g_hpolicy_pending_head = 0;
int      g_hpolicy_pending_tail = 0;
datetime g_rel_clock_bar_time = 0;
int      g_rel_clock_seq = 0;
double   g_regime_class_mass[FXAI_AI_COUNT][FXAI_REGIME_COUNT][3];
double   g_regime_total[FXAI_AI_COUNT][FXAI_REGIME_COUNT];
double   g_regime_spread_ema = 0.0;
double   g_regime_vol_ema = 0.0;
bool     g_regime_ema_ready = false;
int      g_norm_feature_windows[FXAI_AI_FEATURES];
int      g_norm_default_window = FXAI_NORM_ROLL_WINDOW_DEFAULT;
bool     g_norm_windows_ready = false;
bool     g_meta_artifacts_dirty = false;
datetime g_meta_last_save_time = 0;
bool     g_runtime_artifacts_dirty = false;
datetime g_runtime_last_save_time = 0;

struct FXAIContextSeries
{
   bool loaded;
   string symbol;
   datetime last_bar_time;
   MqlRates rates[];
   double open[];
   double high[];
   double low[];
   double close[];
   datetime time[];
   int spread[];
   int aligned_idx[];
};

struct FXAIPreparedSample
{
   bool valid;
   int label_class;
   int regime_id;
   int horizon_minutes;
   int horizon_slot;
   double move_points;
   double min_move_points;
   double cost_points;
   double sample_weight;
   double quality_score;
   double mfe_points;
   double mae_points;
   double spread_stress;
   double trace_spread_mean_ratio;
   double trace_spread_peak_ratio;
   double trace_range_mean_ratio;
   double trace_body_efficiency;
   double trace_gap_ratio;
   double trace_reversal_ratio;
   double trace_session_transition;
   double trace_rollover;
   double time_to_hit_frac;
   int path_flags;
   double masked_step_target;
   double next_vol_target;
   double regime_shift_target;
   double context_lead_target;
   double point_value;
   double domain_hash;
   datetime sample_time;
   double x[FXAI_AI_WEIGHTS];
};

struct FXAINormSampleCache
{
   int method_id;
   bool ready;
   FXAIPreparedSample samples[];
};

struct FXAINormInputCache
{
   int method_id;
   bool ready;
   double x[FXAI_AI_WEIGHTS];
};

FXAIPreparedSample g_replay_samples[FXAI_REPLAY_CAPACITY];
bool     g_replay_used[FXAI_REPLAY_CAPACITY];
int      g_replay_bucket_count[FXAI_REGIME_COUNT][FXAI_MAX_HORIZONS];
double   g_replay_priority[FXAI_REPLAY_CAPACITY];
int      g_replay_flags[FXAI_REPLAY_CAPACITY];
int      g_replay_count = 0;
int      g_replay_cursor = 0;
datetime g_replay_last_sample_time[FXAI_MAX_HORIZONS];
datetime g_last_order_request_time = 0;
ulong    g_last_order_request_us = 0;
double   g_last_order_request_price = 0.0;
double   g_last_order_request_volume = 0.0;
double   g_last_order_request_filled_volume = 0.0;
int      g_last_order_request_horizon = 0;
int      g_last_order_request_side = 0;
int      g_last_order_request_type = 0;
bool     g_last_order_request_pending = false;

void FXAI_ClearLastOrderRequestState()
{
   g_last_order_request_time = 0;
   g_last_order_request_us = 0;
   g_last_order_request_price = 0.0;
   g_last_order_request_volume = 0.0;
   g_last_order_request_filled_volume = 0.0;
   g_last_order_request_horizon = 0;
   g_last_order_request_side = 0;
   g_last_order_request_type = 0;
   g_last_order_request_pending = false;
}


#include "Engine\engine_all.mqh"

int OnInit()
{
   FXAI_SetRandomSeed((ulong)((long)TimeLocal() > 0 ? (long)TimeLocal() : 1));
   trade.SetExpertMagicNumber((long)TradeMagic);

   double buy_init = AI_BuyThreshold;
   double sell_init = AI_SellThreshold;
   FXAI_SanitizeThresholdPair(buy_init, sell_init);
   if(MathAbs(buy_init - AI_BuyThreshold) > 1e-12 || MathAbs(sell_init - AI_SellThreshold) > 1e-12)
   {
      // Optimizer-safe behavior: keep running and sanitize thresholds in runtime path.
      Print("FXAI warning: threshold inputs are outside recommended relation/range. ",
            "Runtime threshold sanitization will be applied.");
   }

   InitialEquity = AccountInfoDouble(ACCOUNT_EQUITY);
   EquiMax       = InitialEquity;
   RealizedManagedProfit = 0.0;
   TP_Value      = 0.0;

   g_plugins_ready = g_plugins.Initialize();
   if(!g_plugins_ready)
      return(INIT_FAILED);

   if(!FXAI_ValidateNativePluginAPI())
      return(INIT_PARAMETERS_INCORRECT);

   if(AI_ComplianceHarness && !FXAI_RunPluginComplianceHarness())
      return(INIT_PARAMETERS_INCORRECT);

   FXAI_ParseContextSymbols(AI_ContextSymbols, g_context_symbols);
   FXAI_FilterContextSymbols(_Symbol, g_context_symbols);
   FXAI_ExtendContextSymbolsFromMarketWatch(_Symbol, g_context_symbols);

   ResetAIState(_Symbol);
   FXAI_RecoverManagedCycleState();
   return(INIT_SUCCEEDED);
}

void OnDeinit(const int reason)
{
   FXAI_SaveRuntimeArtifacts(_Symbol);
   FXAI_SaveMetaArtifacts(_Symbol);
   g_plugins.Release();
   g_plugins_ready = false;
}

int FXAI_GetM1SyncBars(void)
{
   int n = AI_M1SyncBars;
   if(n < 2) n = 2;
   if(n > 12) n = 12;
   return n;
}

void FXAI_ResolveExecutionProfile(FXAIExecutionProfile &profile)
{
   FXAI_SetExecutionProfilePreset((int)AI_ExecutionProfile, profile);

   if(AI_CommissionPerLotSide > profile.commission_per_lot_side)
      profile.commission_per_lot_side = AI_CommissionPerLotSide;

   if(AI_ExecutionSlippageOverride >= 0.0)
      profile.slippage_points = AI_ExecutionSlippageOverride;
   if(AI_ExecutionFillPenaltyOverride >= 0.0)
      profile.fill_penalty_points = AI_ExecutionFillPenaltyOverride;
}

//------------------------- UTILS ------------------------------------
void Calc_TP()
{
   double tp_usd = (TP_USD > 0.0 ? TP_USD : 0.0);
   TP_Value = tp_usd * CloseCounter;
}

void ResetCycleState()
{
   CycleActive      = false;
   CycleEntryEquity = 0.0;
   CycleEntryManagedPnl = 0.0;
   CycleEntryRealizedProfit = 0.0;
   CycleStartTime   = 0;

   TrailTracking    = false;
   TrailPeakProfit  = 0.0;
}

void FXAI_RecoverManagedCycleState()
{
   int total = FXAI_ManagedOrdersTotal(_Symbol) + FXAI_ManagedPositionsTotal(_Symbol);
   if(total <= 0)
      return;

   double open_profit = FXAI_ManagedOpenProfit(_Symbol);
   CycleActive = true;
   CycleEntryEquity = AccountInfoDouble(ACCOUNT_EQUITY);
   CycleEntryManagedPnl = 0.0;
   CycleEntryRealizedProfit = RealizedManagedProfit;
   CycleStartTime = FXAI_GetOldestPositionTime();
   if(CycleStartTime <= 0) CycleStartTime = TimeCurrent();
   if(CycleStartTime <= 0) CycleStartTime = TimeTradeServer();
   TrailTracking = true;
   TrailPeakProfit = MathMax(open_profit, 0.0);
   Calc_TP();
}

datetime FXAI_GetOldestPositionTime()
{
   datetime oldest = 0;
   for(int i=PositionsTotal() - 1; i>=0; i--)
   {
      ulong ticket = PositionGetTicket(i);
      if(ticket == 0) continue;
      if(!PositionSelectByTicket(ticket)) continue;
      if((ulong)PositionGetInteger(POSITION_MAGIC) != TradeMagic) continue;
      if(PositionGetString(POSITION_SYMBOL) != _Symbol) continue;

      datetime t = (datetime)PositionGetInteger(POSITION_TIME);
      if(t <= 0) continue;
      if(oldest == 0 || t < oldest) oldest = t;
   }
   return oldest;
}

double FXAI_ManagedOpenProfit(const string symbol = "")
{
   double total = 0.0;
   for(int i=PositionsTotal() - 1; i>=0; i--)
   {
      ulong ticket = PositionGetTicket(i);
      if(ticket == 0) continue;
      if(!PositionSelectByTicket(ticket)) continue;
      if((ulong)PositionGetInteger(POSITION_MAGIC) != TradeMagic) continue;
      if(StringLen(symbol) > 0 && PositionGetString(POSITION_SYMBOL) != symbol) continue;

      double pos_total = PositionGetDouble(POSITION_PROFIT) +
                         PositionGetDouble(POSITION_SWAP);

      ulong position_id = (ulong)PositionGetInteger(POSITION_IDENTIFIER);
      if(position_id != 0 && HistorySelectByPosition(position_id))
      {
         int deal_count = HistoryDealsTotal();
         for(int d=deal_count - 1; d>=0; d--)
         {
            ulong deal_ticket = HistoryDealGetTicket(d);
            if(deal_ticket == 0) continue;
            if((ulong)HistoryDealGetInteger(deal_ticket, DEAL_POSITION_ID) != position_id) continue;

            long entry = HistoryDealGetInteger(deal_ticket, DEAL_ENTRY);
            if(entry != DEAL_ENTRY_IN && entry != DEAL_ENTRY_INOUT) continue;

            pos_total += HistoryDealGetDouble(deal_ticket, DEAL_COMMISSION);
         }
      }

      total += pos_total;
   }
   return total;
}

double FXAI_CurrentCycleProfit()
{
   if(!CycleActive) return 0.0;
   return FXAI_ManagedOpenProfit(_Symbol) - CycleEntryManagedPnl;
}

double FXAI_CurrentCycleRealizedProfit()
{
   if(!CycleActive) return 0.0;
   double realized = RealizedManagedProfit - CycleEntryRealizedProfit;
   if(!MathIsValidNumber(realized)) return 0.0;
   return realized;
}

double FXAI_TotalManagedProfit()
{
   return FXAI_CurrentCycleRealizedProfit() + FXAI_CurrentCycleProfit();
}

int FXAI_ManagedPositionsTotal(const string symbol = "")
{
   int count = 0;
   for(int i=PositionsTotal() - 1; i>=0; i--)
   {
      ulong ticket = PositionGetTicket(i);
      if(ticket == 0) continue;
      if(!PositionSelectByTicket(ticket)) continue;
      if((ulong)PositionGetInteger(POSITION_MAGIC) != TradeMagic) continue;
      if(StringLen(symbol) > 0 && PositionGetString(POSITION_SYMBOL) != symbol) continue;
      count++;
   }
   return count;
}

int FXAI_ManagedOrdersTotal(const string symbol = "")
{
   int count = 0;
   for(int i=OrdersTotal() - 1; i>=0; i--)
   {
      ulong ticket = OrderGetTicket(i);
      if(ticket == 0) continue;
      if(!OrderSelect(ticket)) continue;
      if((ulong)OrderGetInteger(ORDER_MAGIC) != TradeMagic) continue;
      if(StringLen(symbol) > 0 && OrderGetString(ORDER_SYMBOL) != symbol) continue;
      count++;
   }
   return count;
}

#include "Engine\Runtime\runtime_trade_helpers.mqh"

//------------------------- EA HARD EXIT -----------------------------
void HardExit()
{
   if (MQLInfoInteger(MQL_TESTER)) TesterStop();
   else ExpertRemove();
}

bool EAStop()
{
   if(EquiMax <= 0.0) return false;

   double eq = AccountInfoDouble(ACCOUNT_EQUITY);
   if(eq > EquiMax) EquiMax = eq;

   double maxdd = FXAI_Clamp(MaxDD, 0.0, 99.9);
   if(maxdd <= 0.0) return false;

   if((eq / EquiMax) < ((100.0 - maxdd) / 100.0))
   {
      int managed_total = FXAI_ManagedOrdersTotal(_Symbol) + FXAI_ManagedPositionsTotal(_Symbol);
      if(managed_total > 0)
      {
         if(!CloseAll())
         {
            Print("FXAI warning: MaxDD stop triggered but managed exposure could not be flattened yet.");
            return true;
         }
         ResetCycleState();
         Calc_TP();
      }

      HardExit();
      return true;
   }
   return false;
}

//--------------------------- TP CHECK -------------------------------
void TPCheck()
{
   if(TP_USD <= 0.0) return;

   if(FXAI_TotalManagedProfit() >= TP_Value)
   {
      if(CloseAll())
      {
         ResetCycleState();
         CloseCounter++;
         Calc_TP();
      }
   }
}

void TradeKillerManage()
{
   if(TradeKiller <= 0) return;
   if(FXAI_ManagedPositionsTotal(_Symbol) <= 0) return;

   datetime start_t = CycleStartTime;
   if(start_t <= 0)
      start_t = FXAI_GetOldestPositionTime();
   if(start_t <= 0) return;

   datetime now_t = TimeCurrent();
   if(now_t <= 0) now_t = TimeTradeServer();
   if(now_t <= 0) return;

   int minutes_limit = TradeKiller;
   if(minutes_limit < 1) minutes_limit = 1;

   long held_sec = (long)(now_t - start_t);
   if(held_sec >= (long)minutes_limit * 60L)
   {
      if(CloseAll())
      {
         ResetCycleState();
         Calc_TP();
      }
   }
}

//--------------------- EQUITY SL MANAGER ----------------------------
void EquitySLManage()
{
   if(SL_USD <= 0.0) return;
   if(FXAI_ManagedPositionsTotal(_Symbol) <= 0) return;
   if(!CycleActive) return;

   double dd = FXAI_CurrentCycleProfit();

   if(dd <= -SL_USD)
   {
      if(CloseAll())
      {
         ResetCycleState();
         Calc_TP();
      }
   }
}

//--------------------- EQUITY TRAILING STOP -------------------------
void EquityTrailManage()
{
   if(!TrailEnabled) return;
   if(FXAI_ManagedPositionsTotal(_Symbol) <= 0) return;
   if(!CycleActive) return;

   if(!TrailTracking)
   {
      TrailTracking = true;
      TrailPeakProfit = 0.0;
   }

   double profit = FXAI_CurrentCycleProfit();

   if(profit > TrailPeakProfit) TrailPeakProfit = profit;
   double trail_start = (TrailStartUSD > 0.0 ? TrailStartUSD : 0.0);
   if(TrailPeakProfit < trail_start) return;

   double tp_breath = (TrailTPBreathUSD > 0.0 ? TrailTPBreathUSD : 0.0);
   double desiredTP = FXAI_CurrentCycleRealizedProfit() + TrailPeakProfit + tp_breath;
   if(desiredTP > TP_Value) TP_Value = desiredTP;

   double giveback = TrailGivebackPct;
   if(giveback < 0.0) giveback = 0.0;
   if(giveback > 99.0) giveback = 99.0;

   double lockedProfit = TrailPeakProfit * (1.0 - (giveback / 100.0));
   double stopProfit = lockedProfit;

   if(profit <= stopProfit)
   {
      if(CloseAll())
      {
         ResetCycleState();
         Calc_TP();
      }
   }
}

//--------------------------- AI SIGNAL -------------------------------

//-------------------------- SEND TRADE ------------------------------
void SendTrade(const int precomputed_direction = -2)
{
   datetime bar_t = iTime(_Symbol, PERIOD_M1, 1);
   bool emit_debug = (AI_DebugFlow && bar_t > 0 && bar_t != g_last_debug_bar);
   if(emit_debug) g_last_debug_bar = bar_t;

   string trade_reason = "ok";
   if(TradePossible(_Symbol, trade_reason) != 1)
   {
      if(emit_debug)
         Print("FXAI debug: Trade blocked. reason=", trade_reason);
      return;
   }

   Calc_TP();

   int direction = precomputed_direction;
   if(direction < -1 || direction > 1)
      direction = SpecialDirectionAI(_Symbol);
   if(direction == -1)
   {
      if(emit_debug)
         Print("FXAI debug: AI no-trade. reason=", g_ai_last_reason);
      return;
   }

   double trade_lot = FXAI_CalcRiskAwareLot(_Symbol, direction, trade_reason);
   if(trade_lot <= 0.0)
   {
      if(emit_debug)
         Print("FXAI debug: Trade sizing blocked. reason=", trade_reason,
               " conf=", DoubleToString(g_ai_last_confidence, 3),
               " rel=", DoubleToString(g_ai_last_reliability, 3),
               " gate=", DoubleToString(g_ai_last_trade_gate, 3),
               " hier=", DoubleToString(g_ai_last_hierarchy_score, 3),
               " hcons=", DoubleToString(g_ai_last_hierarchy_consistency, 3),
               " hexec=", DoubleToString(g_ai_last_hierarchy_execution, 3),
               " macroq=", DoubleToString(g_ai_last_macro_state_quality, 3),
               " ppress=", DoubleToString(g_ai_last_portfolio_pressure, 3),
               " path=", DoubleToString(g_ai_last_path_risk, 3),
               " fill=", DoubleToString(g_ai_last_fill_risk, 3));
      return;
   }

   uint retcode = 0;
   string ret_desc = "";
   bool exec_ok = FXAI_SendMarketOrderChecked(_Symbol,
                                              direction,
                                              trade_lot,
                                              trade_reason,
                                              retcode,
                                              ret_desc);

   if(!exec_ok && emit_debug)
      Print("FXAI debug: Order send failed. retcode=", (int)retcode,
            " desc=", ret_desc,
            " reason=", trade_reason);
   else if(exec_ok && emit_debug)
      Print("FXAI debug: Order sent. direction=", direction, " lot=", DoubleToString(trade_lot, 2));

   if(exec_ok)
   {
      ResetCycleState();
      CycleActive      = true;
      CycleEntryEquity = AccountInfoDouble(ACCOUNT_EQUITY);
      CycleEntryManagedPnl = FXAI_ManagedOpenProfit(_Symbol);
      CycleEntryRealizedProfit = RealizedManagedProfit;
      CycleStartTime   = TimeCurrent();
      if(CycleStartTime <= 0) CycleStartTime = TimeTradeServer();
      if(CycleStartTime <= 0) CycleStartTime = iTime(_Symbol, PERIOD_M1, 0);

      TrailTracking    = true;
      TrailPeakProfit  = 0.0;
   }
}

void OnTradeTransaction(const MqlTradeTransaction &trans,
                        const MqlTradeRequest &request,
                        const MqlTradeResult &result)
{
   if(trans.type != TRADE_TRANSACTION_DEAL_ADD)
      return;

   ulong deal_ticket = trans.deal;
   if(deal_ticket == 0 || !HistoryDealSelect(deal_ticket))
      return;

   if((ulong)HistoryDealGetInteger(deal_ticket, DEAL_MAGIC) != TradeMagic)
      return;
   if(HistoryDealGetString(deal_ticket, DEAL_SYMBOL) != _Symbol)
      return;

   long entry = HistoryDealGetInteger(deal_ticket, DEAL_ENTRY);
   if(entry == DEAL_ENTRY_IN || entry == DEAL_ENTRY_INOUT)
   {
      if(!g_last_order_request_pending)
         return;

      double point_value = SymbolInfoDouble(_Symbol, SYMBOL_POINT);
      if(point_value <= 0.0)
         point_value = (_Point > 0.0 ? _Point : 1.0);
      double deal_price = HistoryDealGetDouble(deal_ticket, DEAL_PRICE);
      double deal_volume = HistoryDealGetDouble(deal_ticket, DEAL_VOLUME);
      double slippage_points = 0.0;
      if(g_last_order_request_pending && point_value > 0.0 && g_last_order_request_price > 0.0)
         slippage_points = MathAbs(deal_price - g_last_order_request_price) / point_value;
      double elapsed_ms = 0.0;
      if(g_last_order_request_pending && g_last_order_request_us > 0)
         elapsed_ms = (double)(GetMicrosecondCount() - g_last_order_request_us) / 1000.0;
      double latency_points = 0.05 * MathLog(1.0 + MathMax(elapsed_ms, 0.0));
      double remaining_before = MathMax(g_last_order_request_volume - g_last_order_request_filled_volume, 0.0);
      bool partial_fill = (g_last_order_request_volume > 0.0 &&
                           remaining_before > 1e-9 &&
                           deal_volume + 1e-9 < remaining_before);
      double fill_ratio = 1.0;
      if(g_last_order_request_volume > 1e-9)
         fill_ratio = FXAI_Clamp(deal_volume / g_last_order_request_volume, 0.0, 1.0);
      FXAI_RecordBrokerExecutionEventEx((g_last_order_request_time > 0 ? g_last_order_request_time : TimeCurrent()),
                                        _Symbol,
                                        g_last_order_request_horizon,
                                        g_last_order_request_side,
                                        g_last_order_request_type,
                                        (partial_fill ? 1 : 2),
                                        slippage_points,
                                        latency_points,
                                        false,
                                        partial_fill,
                                        fill_ratio);
      FXAI_MarkRuntimeArtifactsDirty();
      g_last_order_request_filled_volume += MathMax(deal_volume, 0.0);
      if(g_last_order_request_volume <= 1e-9 ||
         g_last_order_request_filled_volume + 1e-9 >= g_last_order_request_volume)
         FXAI_ClearLastOrderRequestState();
      return;
   }

   if(entry != DEAL_ENTRY_OUT && entry != DEAL_ENTRY_OUT_BY)
      return;

   double net = HistoryDealGetDouble(deal_ticket, DEAL_PROFIT) +
                HistoryDealGetDouble(deal_ticket, DEAL_SWAP) +
                HistoryDealGetDouble(deal_ticket, DEAL_COMMISSION);
   if(MathIsValidNumber(net))
      RealizedManagedProfit += net;
}

//--------------------------- ON TICK --------------------------------
void OnTick()
{
   static datetime last_ai_bar = 0;
   datetime signal_bar = iTime(_Symbol, PERIOD_M1, 1);
   bool new_m1_bar = false;
   if(signal_bar > 0 && signal_bar != last_ai_bar)
   {
      last_ai_bar = signal_bar;
      new_m1_bar = true;
   }

   int total = FXAI_ManagedOrdersTotal(_Symbol) + FXAI_ManagedPositionsTotal(_Symbol);
   if(total > 0 && !CycleActive)
      FXAI_RecoverManagedCycleState();

   if(total > 0)
   {
      TradeKillerManage();
      if(FXAI_ManagedOrdersTotal(_Symbol) + FXAI_ManagedPositionsTotal(_Symbol) == 0) return;

      EquitySLManage();
      if(FXAI_ManagedOrdersTotal(_Symbol) + FXAI_ManagedPositionsTotal(_Symbol) == 0) return;

      EquityTrailManage();
      if(FXAI_ManagedOrdersTotal(_Symbol) + FXAI_ManagedPositionsTotal(_Symbol) == 0) return;

      if(MaxDD > 0 && EAStop()) return;
      TPCheck();
      if(FXAI_ManagedOrdersTotal(_Symbol) + FXAI_ManagedPositionsTotal(_Symbol) == 0) return;
   }

   if(!new_m1_bar) return;

   // Heavy model/reliability updates run only once per closed M1 bar.
   FXAI_ProcessReliabilityBar(_Symbol);
   int precomputed_signal = SpecialDirectionAI(_Symbol);

   int managed_total = FXAI_ManagedOrdersTotal(_Symbol) + FXAI_ManagedPositionsTotal(_Symbol);
   if(managed_total > 0)
   {
      string kill_reason = "ok";
      if(FXAI_RegimeKillSwitchTriggered(kill_reason))
      {
         if(AI_DebugFlow)
            Print("FXAI debug: kill switch exit. reason=", kill_reason,
                  " gate=", DoubleToString(g_ai_last_trade_gate, 3),
                  " path=", DoubleToString(g_ai_last_path_risk, 3),
                  " fill=", DoubleToString(g_ai_last_fill_risk, 3));
         if(CloseAll())
         {
            ResetCycleState();
            Calc_TP();
         }
         return;
      }
   }

   if(managed_total == 0)
      SendTrade(precomputed_signal);
}
//+------------------------------------------------------------------+
