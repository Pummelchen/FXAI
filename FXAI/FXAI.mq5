//+------------------------------------------------------------------+
//|                                                         FXAI.mq5 |
//| FXAI modular EA: plugin-based AI + equity trailing + equity SL   |
//+------------------------------------------------------------------+
#property strict

#include <Trade\Trade.mqh>
#include "shared.mqh"
#include "data.mqh"
#include "api.mqh"

CTrade trade;

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
input double AI_CostBufferPoints     = 2.0;
// Models: all (cost-aware labeling + EV gating).
// Purpose: extra safety buffer above spread/commission for slippage/noise.
// Importance/Range: common 0..5 points; higher reduces false-positive entries.
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
// Purpose: runs plugin V2 compliance checks on init before trading starts.
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

int CloseCounter = 1;

bool   CycleActive      = false;
double CycleEntryEquity = 0.0;
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
datetime g_last_debug_bar = 0;

#define FXAI_MAX_CONTEXT_SYMBOLS 32
#define FXAI_REL_MAX_PENDING 2048
#define FXAI_REGIME_COUNT 12
#define FXAI_MAX_HORIZONS 8
#define FXAI_STACK_FEATS 14
#define FXAI_STACK_HIDDEN 8
#define FXAI_HPOL_FEATS 12
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
double g_horizon_regime_edge_ema[FXAI_REGIME_COUNT][FXAI_MAX_HORIZONS];
bool   g_horizon_regime_edge_ready[FXAI_REGIME_COUNT][FXAI_MAX_HORIZONS];
int    g_horizon_regime_obs[FXAI_REGIME_COUNT][FXAI_MAX_HORIZONS];
double g_horizon_regime_total_obs[FXAI_REGIME_COUNT];
bool g_ai_warmup_done = false;
int      g_rel_pending_seq[FXAI_AI_COUNT][FXAI_REL_MAX_PENDING];
double   g_rel_pending_prob[FXAI_AI_COUNT][FXAI_REL_MAX_PENDING][3];
int      g_rel_pending_signal[FXAI_AI_COUNT][FXAI_REL_MAX_PENDING];
int      g_rel_pending_regime[FXAI_AI_COUNT][FXAI_REL_MAX_PENDING];
double   g_rel_pending_expected_move[FXAI_AI_COUNT][FXAI_REL_MAX_PENDING];
int      g_rel_pending_horizon[FXAI_AI_COUNT][FXAI_REL_MAX_PENDING];
int      g_rel_pending_head[FXAI_AI_COUNT];
int      g_rel_pending_tail[FXAI_AI_COUNT];
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
double   g_hpolicy_w[FXAI_REGIME_COUNT][FXAI_HPOL_FEATS];
bool     g_hpolicy_ready[FXAI_REGIME_COUNT];
int      g_hpolicy_obs[FXAI_REGIME_COUNT];
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

struct FXAIContextSeries
{
   bool loaded;
   string symbol;
   datetime last_bar_time;
   MqlRates rates[];
   double close[];
   datetime time[];
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
   double time_to_hit_frac;
   int path_flags;
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

double FXAI_GetArrayValue(const double &arr[], const int idx, const double def_value)
{
   if(idx >= 0 && idx < ArraySize(arr)) return arr[idx];
   return def_value;
}

double FXAI_GetIntArrayMean(const int &arr[],
                            const int start_idx,
                            const int width,
                            const double fallback)
{
   int n = ArraySize(arr);
   if(n <= 0 || start_idx < 0 || start_idx >= n || width <= 0)
      return MathMax(fallback, 0.10);

   int end = start_idx + width;
   if(end > n) end = n;
   double sum = 0.0;
   int used = 0;
   for(int i=start_idx; i<end; i++)
   {
      double v = (double)arr[i];
      if(v <= 0.0) continue;
      sum += v;
      used++;
   }
   if(used <= 0) return MathMax(fallback, 0.10);
   return sum / (double)used;
}

int FXAI_GetStaticRegimeId(const datetime sample_time,
                           const double spread_points,
                           const double spread_ref,
                           const double vol_proxy_abs,
                           const double vol_ref)
{
   MqlDateTime dt;
   TimeToStruct(sample_time > 0 ? sample_time : TimeCurrent(), dt);
   int hour = dt.hour;
   if(hour < 0) hour = 0;
   if(hour > 23) hour = 23;

   int sess = 0;
   if(hour < 8) sess = 0;
   else if(hour < 16) sess = 1;
   else sess = 2;

   double sp_ref = MathMax(spread_ref, 0.10);
   double vp_ref = MathMax(MathAbs(vol_ref), 1e-6);
   int spread_hi = (spread_points > (1.15 * sp_ref + 0.10) ? 1 : 0);
   int vol_hi = (MathAbs(vol_proxy_abs) > (1.15 * vp_ref + 0.02) ? 1 : 0);

   int regime = sess * 4 + vol_hi * 2 + spread_hi;
   if(regime < 0) regime = 0;
   if(regime >= FXAI_REGIME_COUNT) regime = FXAI_REGIME_COUNT - 1;
   return regime;
}

int FXAI_ClampHorizon(const int h_in)
{
   int h = h_in;
   if(h < 1) h = 1;
   if(h > 720) h = 720;
   return h;
}

void FXAI_SortIntAsc(int &arr[])
{
   int n = ArraySize(arr);
   for(int i=1; i<n; i++)
   {
      int key = arr[i];
      int j = i - 1;
      while(j >= 0 && arr[j] > key)
      {
         arr[j + 1] = arr[j];
         j--;
      }
      arr[j + 1] = key;
   }
}

void FXAI_ParseHorizonList(const string raw, const int base_h, int &out_horizons[])
{
   ArrayResize(out_horizons, 0);

   string clean = raw;
   StringReplace(clean, "{", "");
   StringReplace(clean, "}", "");
   StringReplace(clean, ";", ",");
   StringReplace(clean, "|", ",");

   string parts[];
   int n = StringSplit(clean, ',', parts);
   for(int i=0; i<n; i++)
   {
      string tok = parts[i];
      StringTrimLeft(tok);
      StringTrimRight(tok);
      if(StringLen(tok) <= 0) continue;

      int hv = (int)StringToInteger(tok);
      hv = FXAI_ClampHorizon(hv);

      bool exists = false;
      for(int j=0; j<ArraySize(out_horizons); j++)
      {
         if(out_horizons[j] == hv)
         {
            exists = true;
            break;
         }
      }
      if(exists) continue;

      int sz = ArraySize(out_horizons);
      ArrayResize(out_horizons, sz + 1);
      out_horizons[sz] = hv;
      if(ArraySize(out_horizons) >= FXAI_MAX_HORIZONS)
         break;
   }

   int b = FXAI_ClampHorizon(base_h);
   bool has_base = false;
   for(int j=0; j<ArraySize(out_horizons); j++)
   {
      if(out_horizons[j] == b)
      {
         has_base = true;
         break;
      }
   }
   if(!has_base)
   {
      int sz2 = ArraySize(out_horizons);
      ArrayResize(out_horizons, sz2 + 1);
      out_horizons[sz2] = b;
   }

   if(ArraySize(out_horizons) <= 0)
   {
      ArrayResize(out_horizons, 1);
      out_horizons[0] = b;
   }

   FXAI_SortIntAsc(out_horizons);
}

int FXAI_GetMaxConfiguredHorizon(const int fallback_h)
{
   int hmax = FXAI_ClampHorizon(fallback_h);
   for(int i=0; i<ArraySize(g_horizon_minutes); i++)
   {
      int h = FXAI_ClampHorizon(g_horizon_minutes[i]);
      if(h > hmax) hmax = h;
   }
   return hmax;
}

int FXAI_GetHorizonSlot(const int horizon_minutes)
{
   int n = ArraySize(g_horizon_minutes);
   if(n <= 0) return 0;
   if(n > FXAI_MAX_HORIZONS) n = FXAI_MAX_HORIZONS;

   int h = FXAI_ClampHorizon(horizon_minutes);
   int best = 0;
   int best_diff = MathAbs(FXAI_ClampHorizon(g_horizon_minutes[0]) - h);
   for(int i=1; i<n; i++)
   {
      int hv = FXAI_ClampHorizon(g_horizon_minutes[i]);
      int d = MathAbs(hv - h);
      if(d < best_diff)
      {
         best_diff = d;
         best = i;
      }
   }
   if(best < 0) best = 0;
   if(best >= FXAI_MAX_HORIZONS) best = FXAI_MAX_HORIZONS - 1;
   return best;
}

void FXAI_BuildHorizonPolicyFeatures(const int horizon_minutes,
                                     const int base_h,
                                     const double expected_abs_points,
                                     const double min_move_points,
                                     const FXAIDataSnapshot &snapshot,
                                     const double current_vol,
                                     const int regime_id,
                                     const int ai_hint,
                                     double &feat[])
{
   MqlDateTime dt;
   TimeToStruct(snapshot.bar_time, dt);
   double hold_penalty = FXAI_Clamp(AI_HorizonPenaltyPerMinute, 0.0, 0.02);

   feat[0] = 1.0;
   feat[1] = FXAI_Clamp((expected_abs_points - min_move_points) / MathMax(min_move_points, 0.50), -4.0, 6.0) / 4.0;
   feat[2] = FXAI_Clamp(expected_abs_points / MathMax(min_move_points, 0.50), 0.0, 8.0) / 4.0;
   feat[3] = 1.0 / MathSqrt((double)MathMax(horizon_minutes, 1));
   feat[4] = -hold_penalty * (double)horizon_minutes;
   feat[5] = FXAI_Clamp(FXAI_GetHorizonRegimeEdge(regime_id, horizon_minutes) / MathMax(min_move_points, 0.50), -3.0, 3.0) / 3.0;
   feat[6] = (ai_hint >= 0 ? FXAI_Clamp(FXAI_GetModelRegimeEdge(ai_hint, regime_id) / MathMax(min_move_points, 0.50), -3.0, 3.0) / 3.0 : 0.0);
   feat[7] = FXAI_Clamp(current_vol / MathMax(snapshot.point, 1e-6), 0.0, 50.0) / 25.0;
   feat[8] = FXAI_Clamp(snapshot.spread_points / MathMax(min_move_points, 0.50), 0.0, 2.0) - 0.5;
   feat[9] = ((double)dt.hour - 11.5) / 11.5;
   feat[10] = ((double)dt.min - 29.5) / 29.5;
   feat[11] = FXAI_Clamp(((double)horizon_minutes - (double)base_h) / (double)MathMax(base_h, 1), -2.0, 2.0) / 2.0;
}

int FXAI_SelectRoutedHorizon(const double &close_arr[],
                             const FXAIDataSnapshot &snapshot,
                             const double min_move_points,
                             const int ev_lookback,
                             const int fallback_h,
                             const int regime_id,
                             const int ai_hint)
{
   int base_h = FXAI_ClampHorizon(fallback_h);
   if(!AI_MultiHorizon) return base_h;
   if(ArraySize(g_horizon_minutes) <= 0) return base_h;

   double best_score = -1e18;
   int best_h = base_h;
   double hold_penalty = FXAI_Clamp(AI_HorizonPenaltyPerMinute, 0.0, 0.02);
   double current_vol = MathAbs(FXAI_SafeReturn(close_arr, 0, 1));

   for(int i=0; i<ArraySize(g_horizon_minutes); i++)
   {
      int h = FXAI_ClampHorizon(g_horizon_minutes[i]);
      double exp_abs = FXAI_EstimateExpectedAbsMovePoints(close_arr,
                                                          h,
                                                          ev_lookback,
                                                          snapshot.point);
      if(exp_abs <= 0.0) continue;

      double net = exp_abs - min_move_points;
      double score = (net / MathSqrt((double)h)) - (hold_penalty * (double)h);
      int slot = FXAI_GetHorizonSlot(h);

      // Learned global regime-aware horizon utility with UCB exploration.
      if(regime_id >= 0 && regime_id < FXAI_REGIME_COUNT &&
         slot >= 0 && slot < FXAI_MAX_HORIZONS &&
         g_horizon_regime_edge_ready[regime_id][slot])
      {
         double edge = g_horizon_regime_edge_ema[regime_id][slot];
         int obs = g_horizon_regime_obs[regime_id][slot];
         double total_obs = g_horizon_regime_total_obs[regime_id];
         if(total_obs < 1.0) total_obs = 1.0;
         double ucb = edge + (0.35 * MathSqrt(MathLog(1.0 + total_obs) / (1.0 + (double)obs)));
         score += 0.25 * (ucb / MathMax(min_move_points, 0.50));
      }

      // Optional model-specific horizon utility when single-model mode.
      if(ai_hint >= 0 && ai_hint < FXAI_AI_COUNT &&
         slot >= 0 && slot < FXAI_MAX_HORIZONS &&
         g_model_horizon_edge_ready[ai_hint][slot])
      {
         double medge = g_model_horizon_edge_ema[ai_hint][slot];
         int mobs = g_model_horizon_obs[ai_hint][slot];
         double mu = medge + (0.20 / MathSqrt(1.0 + (double)mobs));
         score += 0.15 * (mu / MathMax(min_move_points, 0.50));
      }

      if(regime_id >= 0 && regime_id < FXAI_REGIME_COUNT && g_hpolicy_ready[regime_id])
      {
         double feat[FXAI_HPOL_FEATS];
         FXAI_BuildHorizonPolicyFeatures(h,
                                         base_h,
                                         exp_abs,
                                         min_move_points,
                                         snapshot,
                                         current_vol,
                                         regime_id,
                                         ai_hint,
                                         feat);

         double learned = 0.0;
         for(int k=0; k<FXAI_HPOL_FEATS; k++)
            learned += g_hpolicy_w[regime_id][k] * feat[k];
         score += 0.35 * learned;
      }

      if(score > best_score)
      {
         best_score = score;
         best_h = h;
      }
   }

   return FXAI_ClampHorizon(best_h);
}

bool FXAI_IsModelInList(const int ai_idx, const int &ai_list[])
{
   for(int i=0; i<ArraySize(ai_list); i++)
      if(ai_list[i] == ai_idx) return true;
   return false;
}

void FXAI_StackBuildFeatures(const double buy_pct,
                             const double sell_pct,
                             const double skip_pct,
                             const double avg_buy_ev,
                             const double avg_sell_ev,
                             const double min_move_points,
                             const double expected_move_points,
                             const double vol_proxy,
                             const int horizon_minutes,
                             double &feat[])
{
   double mm = MathMax(min_move_points, 0.10);
   double em = MathMax(expected_move_points, mm);
   double pb = FXAI_Clamp(buy_pct / 100.0, 0.0, 1.0);
   double ps = FXAI_Clamp(sell_pct / 100.0, 0.0, 1.0);
   double pk = FXAI_Clamp(skip_pct / 100.0, 0.0, 1.0);
   double sump = pb + ps + pk;
   if(sump <= 0.0) sump = 1.0;
   pb /= sump; ps /= sump; pk /= sump;

   double entropy = 0.0;
   if(pb > 1e-9) entropy -= pb * MathLog(pb);
   if(ps > 1e-9) entropy -= ps * MathLog(ps);
   if(pk > 1e-9) entropy -= pk * MathLog(pk);
   double entropy_norm = 1.0 - FXAI_Clamp(entropy / MathLog(3.0), 0.0, 1.0);

   feat[0] = 1.0;
   feat[1] = FXAI_Clamp((pb - 0.5) * 2.0, -1.0, 1.0);
   feat[2] = FXAI_Clamp((ps - 0.5) * 2.0, -1.0, 1.0);
   feat[3] = FXAI_Clamp((pk - 0.5) * 2.0, -1.0, 1.0);
   feat[4] = FXAI_Clamp(avg_buy_ev / mm, -3.0, 3.0) / 3.0;
   feat[5] = FXAI_Clamp(avg_sell_ev / mm, -3.0, 3.0) / 3.0;
   feat[6] = FXAI_Clamp(pb - ps, -1.0, 1.0);
   feat[7] = FXAI_Clamp(MathMax(pb, ps), 0.0, 1.0);
   feat[8] = FXAI_Clamp(entropy_norm, 0.0, 1.0);
   feat[9] = FXAI_Clamp((avg_buy_ev - avg_sell_ev) / mm, -4.0, 4.0) / 4.0;
   feat[10] = FXAI_Clamp(min_move_points / em, 0.0, 2.0) - 0.5;
   feat[11] = FXAI_Clamp(vol_proxy / 4.0, 0.0, 1.0);
   feat[12] = FXAI_Clamp((double)horizon_minutes / (double)MathMax(FXAI_GetMaxConfiguredHorizon(horizon_minutes), 1), 0.0, 1.0);
   feat[13] = FXAI_Clamp(pk - MathMax(pb, ps), -1.0, 1.0);
}

void FXAI_StackPredict(const int regime_id, const double &feat[], double &probs[])
{
   if(ArraySize(probs) != 3) ArrayResize(probs, 3);
   probs[0] = 0.3333;
   probs[1] = 0.3333;
   probs[2] = 0.3334;

   int r = regime_id;
   if(r < 0 || r >= FXAI_REGIME_COUNT) r = 0;

   if(!g_stack_ready[r])
   {
      double p_sell = FXAI_Clamp(0.32 + (0.35 * feat[2]) - (0.12 * feat[3]) + (0.18 * feat[5]) - (0.08 * feat[10]), 0.01, 0.98);
      double p_buy  = FXAI_Clamp(0.32 + (0.35 * feat[1]) - (0.12 * feat[3]) + (0.18 * feat[4]) - (0.08 * feat[10]), 0.01, 0.98);
      double p_skip = FXAI_Clamp(0.26 + (0.40 * feat[3]) + (0.18 * feat[10]) - (0.10 * feat[8]), 0.01, 0.98);
      double s0 = p_sell + p_buy + p_skip;
      if(s0 <= 0.0) s0 = 1.0;
      probs[0] = p_sell / s0;
      probs[1] = p_buy / s0;
      probs[2] = p_skip / s0;
      return;
   }

   double hidden[FXAI_STACK_HIDDEN];
   for(int h=0; h<FXAI_STACK_HIDDEN; h++)
   {
      double z = g_stack_b1[r][h];
      for(int k=0; k<FXAI_STACK_FEATS; k++)
         z += g_stack_w1[r][h][k] * feat[k];
      hidden[h] = FXAI_Tanh(z);
   }

   double z[3];
   for(int c=0; c<3; c++)
   {
      z[c] = g_stack_b2[r][c];
      for(int h=0; h<FXAI_STACK_HIDDEN; h++)
         z[c] += g_stack_w2[r][c][h] * hidden[h];
   }

   double zmax = z[0];
   if(z[1] > zmax) zmax = z[1];
   if(z[2] > zmax) zmax = z[2];
   double e0 = MathExp(z[0] - zmax);
   double e1 = MathExp(z[1] - zmax);
   double e2 = MathExp(z[2] - zmax);
   double s = e0 + e1 + e2;
   if(s <= 0.0) s = 1.0;
   probs[0] = e0 / s;
   probs[1] = e1 / s;
   probs[2] = e2 / s;
}

void FXAI_StackUpdate(const int regime_id,
                      const int label_class,
                      const double &feat[],
                      const double sample_weight)
{
   if(label_class < 0 || label_class > 2) return;
   int r = regime_id;
   if(r < 0 || r >= FXAI_REGIME_COUNT) r = 0;

   double hidden[FXAI_STACK_HIDDEN];
   for(int h=0; h<FXAI_STACK_HIDDEN; h++)
   {
      double z = g_stack_b1[r][h];
      for(int k=0; k<FXAI_STACK_FEATS; k++)
         z += g_stack_w1[r][h][k] * feat[k];
      hidden[h] = FXAI_Tanh(z);
   }

   double z_out[3];
   double probs[3];
   for(int c=0; c<3; c++)
   {
      z_out[c] = g_stack_b2[r][c];
      for(int h=0; h<FXAI_STACK_HIDDEN; h++)
         z_out[c] += g_stack_w2[r][c][h] * hidden[h];
   }

   double zmax = z_out[0];
   if(z_out[1] > zmax) zmax = z_out[1];
   if(z_out[2] > zmax) zmax = z_out[2];
   double e0 = MathExp(z_out[0] - zmax);
   double e1 = MathExp(z_out[1] - zmax);
   double e2 = MathExp(z_out[2] - zmax);
   double s = e0 + e1 + e2;
   if(s <= 0.0) s = 1.0;
   probs[0] = e0 / s;
   probs[1] = e1 / s;
   probs[2] = e2 / s;

   double lr = 0.025 / MathSqrt(1.0 + 0.02 * (double)g_stack_obs[r]);
   lr = FXAI_Clamp(lr, 0.002, 0.025);
   double sw = FXAI_Clamp(sample_weight, 0.20, 7.50);

   double delta_out[3];
   for(int c=0; c<3; c++)
   {
      double target = (c == label_class ? 1.0 : 0.0);
      delta_out[c] = FXAI_Clamp((target - probs[c]) * sw, -3.0, 3.0);
   }

   double delta_hidden[FXAI_STACK_HIDDEN];
   for(int h=0; h<FXAI_STACK_HIDDEN; h++)
   {
      double back = 0.0;
      for(int c=0; c<3; c++)
         back += delta_out[c] * g_stack_w2[r][c][h];
      delta_hidden[h] = back * (1.0 - hidden[h] * hidden[h]);
      delta_hidden[h] = FXAI_Clamp(delta_hidden[h], -3.0, 3.0);
   }

   for(int c=0; c<3; c++)
   {
      g_stack_b2[r][c] += lr * delta_out[c];
      for(int h=0; h<FXAI_STACK_HIDDEN; h++)
      {
         double reg = 0.0005 * g_stack_w2[r][c][h];
         g_stack_w2[r][c][h] += lr * (delta_out[c] * hidden[h] - reg);
      }
   }

   for(int h=0; h<FXAI_STACK_HIDDEN; h++)
   {
      g_stack_b1[r][h] += lr * delta_hidden[h];
      for(int k=0; k<FXAI_STACK_FEATS; k++)
      {
         double reg = (k == 0 ? 0.0 : 0.0004 * g_stack_w1[r][h][k]);
         g_stack_w1[r][h][k] += lr * (delta_hidden[h] * feat[k] - reg);
      }
   }

   g_stack_obs[r]++;
   if(g_stack_obs[r] > 200000) g_stack_obs[r] = 200000;
   g_stack_ready[r] = true;
}

double FXAI_BarRandom01(const datetime bar_time, const int salt)
{
   uint x = (uint)(bar_time & 0x7FFFFFFF);
   uint s = (uint)(salt + 1);
   x ^= (s * 1103515245U + 12345U);
   x ^= (x << 13);
   x ^= (x >> 17);
   x ^= (x << 5);
   return (double)(x % 100000U) / 100000.0;
}

bool FXAI_ShouldSampleByPct(const datetime bar_time, const int salt, const double pct)
{
   double p = FXAI_Clamp(pct, 0.0, 100.0);
   if(p <= 0.0) return false;
   if(p >= 100.0) return true;
   return (FXAI_BarRandom01(bar_time, salt) < (p / 100.0));
}

bool FXAI_IsShadowBar(const int cadence_bars, const int bar_seq)
{
   int c = cadence_bars;
   if(c <= 0) return false;
   if(c == 1) return true;
   if(bar_seq < 0) return false;
   return ((bar_seq % c) == 0);
}

ENUM_FXAI_FEATURE_NORMALIZATION FXAI_GetFeatureNormalizationMethod()
{
   int v = (int)AI_FeatureNormalization;
   if(v < (int)FXAI_NORM_EXISTING || v > (int)FXAI_NORM_DAIN)
      return FXAI_NORM_EXISTING;
   return (ENUM_FXAI_FEATURE_NORMALIZATION)v;
}

ENUM_FXAI_FEATURE_NORMALIZATION FXAI_SanitizeNormMethod(const int method_id)
{
   int v = method_id;
   if(v < (int)FXAI_NORM_EXISTING || v > (int)FXAI_NORM_DAIN)
      v = (int)FXAI_NORM_EXISTING;
   return (ENUM_FXAI_FEATURE_NORMALIZATION)v;
}

ENUM_FXAI_FEATURE_NORMALIZATION FXAI_GetModelNormMethodRouted(const int ai_idx,
                                                             const int regime_id,
                                                             const int horizon_minutes)
{
   ENUM_FXAI_FEATURE_NORMALIZATION method = FXAI_GetFeatureNormalizationMethod();
   if(ai_idx < 0 || ai_idx >= FXAI_AI_COUNT)
      return method;

   if(g_model_norm_ready[ai_idx])
      method = FXAI_SanitizeNormMethod(g_model_norm_method[ai_idx]);

   int hslot = FXAI_GetHorizonSlot(horizon_minutes);
   if(hslot >= 0 && hslot < FXAI_MAX_HORIZONS && g_model_norm_horizon_ready[ai_idx][hslot])
      method = FXAI_SanitizeNormMethod(g_model_norm_method_horizon[ai_idx][hslot]);

   if(regime_id >= 0 && regime_id < FXAI_REGIME_COUNT &&
      hslot >= 0 && hslot < FXAI_MAX_HORIZONS &&
      g_model_norm_bank_ready[ai_idx][regime_id][hslot])
   {
      method = FXAI_SanitizeNormMethod(g_model_norm_method_bank[ai_idx][regime_id][hslot]);
   }
   return method;
}

void FXAI_BuildNormMethodCandidateList(const int ai_idx, int &methods[])
{
   ArrayResize(methods, 0);

   int seed_methods[FXAI_NORM_CAND_MAX];
   int n_seed = 0;
   seed_methods[n_seed++] = (int)FXAI_GetFeatureNormalizationMethod();

   bool deep_model = (ai_idx == (int)AI_LSTM || ai_idx == (int)AI_LSTMG ||
                      ai_idx == (int)AI_TCN || ai_idx == (int)AI_TFT ||
                      ai_idx == (int)AI_TST || ai_idx == (int)AI_AUTOFORMER ||
                      ai_idx == (int)AI_PATCHTST || ai_idx == (int)AI_STMN ||
                      ai_idx == (int)AI_S4 || ai_idx == (int)AI_CHRONOS ||
                      ai_idx == (int)AI_TIMESFM || ai_idx == (int)AI_GEODESICATTENTION);

   if(deep_model)
   {
      seed_methods[n_seed++] = (int)FXAI_NORM_EXISTING;
      seed_methods[n_seed++] = (int)FXAI_NORM_VOL_STD_RETURNS;
      seed_methods[n_seed++] = (int)FXAI_NORM_ATR_NATR_UNIT;
      seed_methods[n_seed++] = (int)FXAI_NORM_ZSCORE;
      seed_methods[n_seed++] = (int)FXAI_NORM_REVIN;
      seed_methods[n_seed++] = (int)FXAI_NORM_DAIN;
      seed_methods[n_seed++] = (int)FXAI_NORM_ROBUST_MEDIAN_IQR;
   }
   else
   {
      seed_methods[n_seed++] = (int)FXAI_NORM_EXISTING;
      seed_methods[n_seed++] = (int)FXAI_NORM_ZSCORE;
      seed_methods[n_seed++] = (int)FXAI_NORM_ROBUST_MEDIAN_IQR;
      seed_methods[n_seed++] = (int)FXAI_NORM_QUANTILE_TO_NORMAL;
      seed_methods[n_seed++] = (int)FXAI_NORM_CHANGE_PERCENT;
      seed_methods[n_seed++] = (int)FXAI_NORM_VOL_STD_RETURNS;
      seed_methods[n_seed++] = (int)FXAI_NORM_ATR_NATR_UNIT;
   }

   for(int i=0; i<n_seed; i++)
   {
      int m = seed_methods[i];
      if(m < (int)FXAI_NORM_EXISTING || m > (int)FXAI_NORM_DAIN) continue;
      bool exists = false;
      for(int j=0; j<ArraySize(methods); j++)
      {
         if(methods[j] == m)
         {
            exists = true;
            break;
         }
      }
      if(exists) continue;
      int sz = ArraySize(methods);
      ArrayResize(methods, sz + 1);
      methods[sz] = m;
      if(ArraySize(methods) >= FXAI_NORM_CAND_MAX)
         break;
   }
}

int FXAI_GetNormDefaultWindow()
{
   int w = FXAI_NORM_ROLL_WINDOW_DEFAULT;
   if(PredictionTargetMinutes <= 2) w = 128;
   else if(PredictionTargetMinutes >= 30) w = 256;
   if(w < 32) w = 32;
   if(w > FXAI_NORM_ROLL_WINDOW_MAX) w = FXAI_NORM_ROLL_WINDOW_MAX;
   return w;
}

void FXAI_BuildNormWindowsFromGroups(const int w_fast,
                                     const int w_mid,
                                     const int w_slow,
                                     const int w_regime,
                                     int &windows_out[])
{
   if(ArraySize(windows_out) != FXAI_AI_FEATURES)
      ArrayResize(windows_out, FXAI_AI_FEATURES);

   int wf = FXAI_NormalizationWindowClamp(w_fast);
   int wm = FXAI_NormalizationWindowClamp(w_mid);
   int ws = FXAI_NormalizationWindowClamp(w_slow);
   int wr = FXAI_NormalizationWindowClamp(w_regime);

   for(int f=0; f<FXAI_AI_FEATURES; f++)
   {
      int w = wm;
      if(f <= 6) w = wf;            // ultra-short momentum/cost features
      else if(f <= 14) w = wm;      // MTF trend/returns
      else if(f <= 21) w = wr;      // time/candle geometry
      else if(f <= 33) w = ws;      // MA/EMA trend structure
      else if(f <= 49) w = wm;      // volatility/statistical filters
      else w = wm;                  // detailed cross-symbol context
      windows_out[f] = w;
   }
}

void FXAI_ApplyNormWindows(const int &windows[], const int default_window)
{
   FXAI_SetNormalizationWindows(windows, default_window);
   int n = ArraySize(windows);
   for(int f=0; f<FXAI_AI_FEATURES; f++)
   {
      int w = default_window;
      if(f < n) w = windows[f];
      g_norm_feature_windows[f] = FXAI_NormalizationWindowClamp(w);
   }
   g_norm_default_window = FXAI_NormalizationWindowClamp(default_window);
   g_norm_windows_ready = true;
}

void FXAI_ResetRegimeCalibration()
{
   for(int ai=0; ai<FXAI_AI_COUNT; ai++)
   {
      for(int r=0; r<FXAI_REGIME_COUNT; r++)
      {
         g_regime_class_mass[ai][r][(int)FXAI_LABEL_SELL] = 1.0;
         g_regime_class_mass[ai][r][(int)FXAI_LABEL_BUY] = 1.0;
         g_regime_class_mass[ai][r][(int)FXAI_LABEL_SKIP] = 1.2;
         g_regime_total[ai][r] = 3.2;
      }
   }
   g_regime_spread_ema = 0.0;
   g_regime_vol_ema = 0.0;
   g_regime_ema_ready = false;
}

void FXAI_UpdateRegimeEMAs(const double spread_points, const double vol_proxy_abs)
{
   double sp = MathMax(0.0, spread_points);
   double vp = MathMax(0.0, vol_proxy_abs);
   if(!g_regime_ema_ready)
   {
      g_regime_spread_ema = sp;
      g_regime_vol_ema = vp;
      g_regime_ema_ready = true;
      return;
   }

   g_regime_spread_ema = (0.98 * g_regime_spread_ema) + (0.02 * sp);
   g_regime_vol_ema = (0.98 * g_regime_vol_ema) + (0.02 * vp);
}

int FXAI_GetRegimeId(const datetime sample_time,
                     const double spread_points,
                     const double vol_proxy_abs)
{
   MqlDateTime dt;
   TimeToStruct(sample_time > 0 ? sample_time : TimeCurrent(), dt);
   int hour = dt.hour;
   if(hour < 0) hour = 0;
   if(hour > 23) hour = 23;

   int sess = 0;
   if(hour < 8) sess = 0;
   else if(hour < 16) sess = 1;
   else sess = 2;

   double sp_ref = (g_regime_ema_ready ? MathMax(g_regime_spread_ema, 0.10) : MathMax(spread_points, 0.10));
   double vp_ref = (g_regime_ema_ready ? MathMax(g_regime_vol_ema, 1e-6) : MathMax(MathAbs(vol_proxy_abs), 1e-6));

   int spread_hi = (spread_points > (1.15 * sp_ref + 0.10) ? 1 : 0);
   int vol_hi = (MathAbs(vol_proxy_abs) > (1.15 * vp_ref + 0.02) ? 1 : 0);

   int regime = sess * 4 + vol_hi * 2 + spread_hi;
   if(regime < 0) regime = 0;
   if(regime >= FXAI_REGIME_COUNT) regime = FXAI_REGIME_COUNT - 1;
   return regime;
}

void FXAI_ApplyRegimeCalibration(const int ai_idx, const int regime_id, double &probs[])
{
   if(ai_idx < 0 || ai_idx >= FXAI_AI_COUNT) return;
   if(regime_id < 0 || regime_id >= FXAI_REGIME_COUNT) return;

   double total = g_regime_total[ai_idx][regime_id];
   if(total < 8.0) return;

   double prior[3];
   for(int c=0; c<3; c++)
      prior[c] = g_regime_class_mass[ai_idx][regime_id][c] / MathMax(total, 1e-9);

   double strength = FXAI_Clamp((total - 8.0) / 220.0, 0.0, 0.45);
   double s = 0.0;
   for(int c=0; c<3; c++)
   {
      probs[c] = FXAI_Clamp(((1.0 - strength) * probs[c]) + (strength * prior[c]), 0.0005, 0.9990);
      s += probs[c];
   }
   if(s <= 0.0) s = 1.0;
   for(int c=0; c<3; c++) probs[c] /= s;
}

void FXAI_UpdateRegimeCalibration(const int ai_idx,
                                  const int regime_id,
                                  const int label_class,
                                  const double &probs[])
{
   if(ai_idx < 0 || ai_idx >= FXAI_AI_COUNT) return;
   if(regime_id < 0 || regime_id >= FXAI_REGIME_COUNT) return;
   if(label_class < 0 || label_class > 2) return;

   double p_true = FXAI_Clamp(probs[label_class], 0.0, 1.0);
   double w = 1.0 + (0.5 * p_true);
   g_regime_class_mass[ai_idx][regime_id][label_class] += w;
   g_regime_total[ai_idx][regime_id] += w;

   if(g_regime_total[ai_idx][regime_id] > 20000.0)
   {
      for(int c=0; c<3; c++)
         g_regime_class_mass[ai_idx][regime_id][c] *= 0.5;
      g_regime_total[ai_idx][regime_id] *= 0.5;
   }
}

void FXAI_ResetModelPerformanceState(const int ai_idx)
{
   if(ai_idx < 0 || ai_idx >= FXAI_AI_COUNT) return;
   g_model_meta_weight[ai_idx] = 1.0;
   g_model_global_edge_ema[ai_idx] = 0.0;
   g_model_global_edge_ready[ai_idx] = false;
   for(int r=0; r<FXAI_REGIME_COUNT; r++)
   {
      g_model_regime_edge_ema[ai_idx][r] = 0.0;
      g_model_regime_edge_ready[ai_idx][r] = false;
      g_model_regime_obs[ai_idx][r] = 0;
   }
}

double FXAI_GetModelRegimeEdge(const int ai_idx, const int regime_id)
{
   if(ai_idx < 0 || ai_idx >= FXAI_AI_COUNT) return 0.0;
   if(regime_id >= 0 && regime_id < FXAI_REGIME_COUNT && g_model_regime_edge_ready[ai_idx][regime_id])
      return g_model_regime_edge_ema[ai_idx][regime_id];
   if(g_model_global_edge_ready[ai_idx]) return g_model_global_edge_ema[ai_idx];
   return 0.0;
}

double FXAI_GetHorizonRegimeEdge(const int regime_id, const int horizon_minutes)
{
   int hslot = FXAI_GetHorizonSlot(horizon_minutes);
   if(regime_id >= 0 && regime_id < FXAI_REGIME_COUNT &&
      hslot >= 0 && hslot < FXAI_MAX_HORIZONS &&
      g_horizon_regime_edge_ready[regime_id][hslot])
   {
      return g_horizon_regime_edge_ema[regime_id][hslot];
   }
   return 0.0;
}

bool FXAI_IsModelPruned(const int ai_idx, const int regime_id)
{
   if(ai_idx < 0 || ai_idx >= FXAI_AI_COUNT) return true;
   double rel = FXAI_Clamp(g_model_reliability[ai_idx], 0.0, 3.0);
   if(rel < 0.30) return true;

   if(regime_id >= 0 && regime_id < FXAI_REGIME_COUNT)
   {
      int obs = g_model_regime_obs[ai_idx][regime_id];
      if(obs >= 24)
      {
         double edge = g_model_regime_edge_ema[ai_idx][regime_id];
         if(edge < -0.35) return true;
      }
   }

   if(g_model_global_edge_ready[ai_idx] && g_model_global_edge_ema[ai_idx] < -0.45)
      return true;
   return false;
}

double FXAI_GetModelMetaScore(const int ai_idx,
                              const int regime_id,
                              const double min_move_points)
{
   if(ai_idx < 0 || ai_idx >= FXAI_AI_COUNT) return 0.0;
   double rel = FXAI_Clamp(g_model_reliability[ai_idx], 0.20, 3.00);
   double meta = FXAI_Clamp(g_model_meta_weight[ai_idx], 0.20, 3.00);
   double edge = FXAI_GetModelRegimeEdge(ai_idx, regime_id);
   double edge_scale = 1.0 + FXAI_Clamp(edge / MathMax(min_move_points, 0.50), -0.70, 1.20);
   if(edge_scale < 0.15) edge_scale = 0.15;
   return rel * meta * edge_scale;
}

void FXAI_UpdateModelPerformance(const int ai_idx,
                                 const int regime_id,
                                 const int label_class,
                                 const int signal,
                                 const double realized_move_points,
                                 const double min_move_points,
                                 const int horizon_minutes,
                                 const double expected_move_points,
                                 const double &probs[])
{
   if(ai_idx < 0 || ai_idx >= FXAI_AI_COUNT) return;
   if(label_class < 0 || label_class > 2) return;

   double min_mv = MathMax(min_move_points, 0.10);
   double realized_net = 0.0;
   double opportunity_penalty = 0.0;
   if(signal == 0 || signal == 1)
   {
      realized_net = FXAI_RealizedNetPointsForSignal(signal,
                                                     realized_move_points,
                                                     min_mv,
                                                     horizon_minutes);
   }
   else
   {
      if(label_class == (int)FXAI_LABEL_SKIP) realized_net = 0.05 * min_mv;
      else
      {
         opportunity_penalty = FXAI_Clamp((MathAbs(realized_move_points) - min_mv), 0.0, 8.0 * min_mv);
         realized_net = -0.25 * opportunity_penalty;
      }
   }

   double pred_edge = 0.0;
   if(signal == 1)
      pred_edge = ((2.0 * probs[(int)FXAI_LABEL_BUY]) - 1.0) * expected_move_points - min_mv;
   else if(signal == 0)
      pred_edge = ((2.0 * probs[(int)FXAI_LABEL_SELL]) - 1.0) * expected_move_points - min_mv;

   double err = FXAI_Clamp(realized_net - pred_edge, -8.0 * min_mv, 8.0 * min_mv);
   double alpha = 0.04;

   if(!g_model_global_edge_ready[ai_idx])
   {
      g_model_global_edge_ema[ai_idx] = realized_net;
      g_model_global_edge_ready[ai_idx] = true;
   }
   else
   {
      g_model_global_edge_ema[ai_idx] = (1.0 - alpha) * g_model_global_edge_ema[ai_idx] + alpha * realized_net;
   }

   if(regime_id >= 0 && regime_id < FXAI_REGIME_COUNT)
   {
      if(!g_model_regime_edge_ready[ai_idx][regime_id])
      {
         g_model_regime_edge_ema[ai_idx][regime_id] = realized_net;
         g_model_regime_edge_ready[ai_idx][regime_id] = true;
      }
      else
      {
         g_model_regime_edge_ema[ai_idx][regime_id] =
            (1.0 - alpha) * g_model_regime_edge_ema[ai_idx][regime_id] + alpha * realized_net;
      }
      g_model_regime_obs[ai_idx][regime_id]++;
      if(g_model_regime_obs[ai_idx][regime_id] > 200000)
         g_model_regime_obs[ai_idx][regime_id] = 200000;
   }

   int hslot = FXAI_GetHorizonSlot(horizon_minutes);
   if(hslot >= 0 && hslot < FXAI_MAX_HORIZONS)
   {
      if(!g_model_horizon_edge_ready[ai_idx][hslot])
      {
         g_model_horizon_edge_ema[ai_idx][hslot] = realized_net;
         g_model_horizon_edge_ready[ai_idx][hslot] = true;
      }
      else
      {
         g_model_horizon_edge_ema[ai_idx][hslot] =
            (1.0 - alpha) * g_model_horizon_edge_ema[ai_idx][hslot] + alpha * realized_net;
      }
      g_model_horizon_obs[ai_idx][hslot]++;
      if(g_model_horizon_obs[ai_idx][hslot] > 200000)
         g_model_horizon_obs[ai_idx][hslot] = 200000;

      if(regime_id >= 0 && regime_id < FXAI_REGIME_COUNT)
      {
         if(!g_horizon_regime_edge_ready[regime_id][hslot])
         {
            g_horizon_regime_edge_ema[regime_id][hslot] = realized_net;
            g_horizon_regime_edge_ready[regime_id][hslot] = true;
         }
         else
         {
            g_horizon_regime_edge_ema[regime_id][hslot] =
               (1.0 - alpha) * g_horizon_regime_edge_ema[regime_id][hslot] + alpha * realized_net;
         }
         g_horizon_regime_obs[regime_id][hslot]++;
         if(g_horizon_regime_obs[regime_id][hslot] > 200000)
            g_horizon_regime_obs[regime_id][hslot] = 200000;
         g_horizon_regime_total_obs[regime_id] += 1.0;
         if(g_horizon_regime_total_obs[regime_id] > 1e9)
            g_horizon_regime_total_obs[regime_id] = 1e9;
      }
   }

   // Online utility-driven threshold adaptation (model + regime + horizon aware).
   if(signal == 1 || signal == 0)
   {
      double utility = FXAI_Clamp(realized_net / min_mv, -2.5, 2.5);
      double mag = MathAbs(utility);
      double step = 0.004 + (0.004 * mag);
      step = FXAI_Clamp(step, 0.002, 0.02);

      double buy_u = g_model_buy_thr[ai_idx];
      double sell_u = g_model_sell_thr[ai_idx];
      if(signal == 1)
      {
         if(utility >= 0.0) buy_u -= step * MathMin(utility, 1.5) * 0.8;
         else               buy_u += step * mag;
      }
      else
      {
         // sell_thr is inverse-coded: higher sell_thr means looser sell gate.
         if(utility >= 0.0) sell_u += step * MathMin(utility, 1.5) * 0.8;
         else               sell_u -= step * mag;
      }
      g_model_buy_thr[ai_idx] = FXAI_Clamp(buy_u, 0.50, 0.95);
      g_model_sell_thr[ai_idx] = FXAI_Clamp(sell_u, 0.05, 0.50);
      FXAI_SanitizeThresholdPair(g_model_buy_thr[ai_idx], g_model_sell_thr[ai_idx]);
      g_model_thr_ready[ai_idx] = true;

      int hslot_thr = FXAI_GetHorizonSlot(horizon_minutes);
      if(hslot_thr >= 0 && hslot_thr < FXAI_MAX_HORIZONS)
      {
         double bh = g_model_buy_thr_horizon[ai_idx][hslot_thr];
         double sh = g_model_sell_thr_horizon[ai_idx][hslot_thr];
         if(!g_model_thr_horizon_ready[ai_idx][hslot_thr])
         {
            bh = g_model_buy_thr[ai_idx];
            sh = g_model_sell_thr[ai_idx];
            g_model_thr_horizon_ready[ai_idx][hslot_thr] = true;
         }
         if(signal == 1)
         {
            if(utility >= 0.0) bh -= step * MathMin(utility, 1.5);
            else               bh += step * mag;
         }
         else
         {
            if(utility >= 0.0) sh += step * MathMin(utility, 1.5);
            else               sh -= step * mag;
         }
         g_model_buy_thr_horizon[ai_idx][hslot_thr] = FXAI_Clamp(bh, 0.50, 0.95);
         g_model_sell_thr_horizon[ai_idx][hslot_thr] = FXAI_Clamp(sh, 0.05, 0.50);
         FXAI_SanitizeThresholdPair(g_model_buy_thr_horizon[ai_idx][hslot_thr],
                                    g_model_sell_thr_horizon[ai_idx][hslot_thr]);
      }

      if(regime_id >= 0 && regime_id < FXAI_REGIME_COUNT)
      {
         double br = g_model_buy_thr_regime[ai_idx][regime_id];
         double sr = g_model_sell_thr_regime[ai_idx][regime_id];
         if(!g_model_thr_regime_ready[ai_idx][regime_id])
         {
            br = g_model_buy_thr[ai_idx];
            sr = g_model_sell_thr[ai_idx];
            g_model_thr_regime_ready[ai_idx][regime_id] = true;
         }
         if(signal == 1)
         {
            if(utility >= 0.0) br -= step * MathMin(utility, 1.5);
            else               br += step * mag;
         }
         else
         {
            if(utility >= 0.0) sr += step * MathMin(utility, 1.5);
            else               sr -= step * mag;
         }
         g_model_buy_thr_regime[ai_idx][regime_id] = FXAI_Clamp(br, 0.50, 0.95);
         g_model_sell_thr_regime[ai_idx][regime_id] = FXAI_Clamp(sr, 0.05, 0.50);
         FXAI_SanitizeThresholdPair(g_model_buy_thr_regime[ai_idx][regime_id],
                                    g_model_sell_thr_regime[ai_idx][regime_id]);

         int hslot_bank = FXAI_GetHorizonSlot(horizon_minutes);
         if(hslot_bank >= 0 && hslot_bank < FXAI_MAX_HORIZONS)
         {
            double bb = g_model_buy_thr_bank[ai_idx][regime_id][hslot_bank];
            double sb = g_model_sell_thr_bank[ai_idx][regime_id][hslot_bank];
            if(!g_model_thr_bank_ready[ai_idx][regime_id][hslot_bank])
            {
               bb = g_model_buy_thr_regime[ai_idx][regime_id];
               sb = g_model_sell_thr_regime[ai_idx][regime_id];
               g_model_thr_bank_ready[ai_idx][regime_id][hslot_bank] = true;
            }
            if(signal == 1)
            {
               if(utility >= 0.0) bb -= step * MathMin(utility, 1.5);
               else               bb += step * mag;
            }
            else
            {
               if(utility >= 0.0) sb += step * MathMin(utility, 1.5);
               else               sb -= step * mag;
            }
            g_model_buy_thr_bank[ai_idx][regime_id][hslot_bank] = FXAI_Clamp(bb, 0.50, 0.95);
            g_model_sell_thr_bank[ai_idx][regime_id][hslot_bank] = FXAI_Clamp(sb, 0.05, 0.50);
            FXAI_SanitizeThresholdPair(g_model_buy_thr_bank[ai_idx][regime_id][hslot_bank],
                                       g_model_sell_thr_bank[ai_idx][regime_id][hslot_bank]);
         }
      }
   }

   double grad = FXAI_Clamp(err / min_mv, -2.0, 2.0);
   double lr = 0.015;
   double meta_new = g_model_meta_weight[ai_idx] + lr * grad;
   g_model_meta_weight[ai_idx] = FXAI_Clamp(meta_new, 0.20, 3.00);
}

void FXAI_ResetModelAuxState(const int ai_idx)
{
   if(ai_idx < 0 || ai_idx >= FXAI_AI_COUNT) return;

   g_model_reliability[ai_idx] = 1.0;
   g_model_abs_move_ema[ai_idx] = 0.0;
   g_model_abs_move_ready[ai_idx] = false;
   FXAI_ResetModelPerformanceState(ai_idx);
}

void FXAI_UpdateModelMoveStats(const int ai_idx, const double move_points)
{
   if(ai_idx < 0 || ai_idx >= FXAI_AI_COUNT) return;

   double abs_move = MathAbs(move_points);
   if(!MathIsValidNumber(abs_move) || abs_move <= 0.0) return;

   if(!g_model_abs_move_ready[ai_idx])
   {
      g_model_abs_move_ema[ai_idx] = abs_move;
      g_model_abs_move_ready[ai_idx] = true;
      return;
   }

   g_model_abs_move_ema[ai_idx] = (0.95 * g_model_abs_move_ema[ai_idx]) + (0.05 * abs_move);
}

double FXAI_GetModelExpectedMove(const int ai_idx, const double fallback_move)
{
   if(ai_idx >= 0 && ai_idx < FXAI_AI_COUNT && g_model_abs_move_ready[ai_idx] && g_model_abs_move_ema[ai_idx] > 0.0)
      return g_model_abs_move_ema[ai_idx];
   return fallback_move;
}

void FXAI_UpdateModelReliability(const int ai_idx,
                                 const int label_class,
                                 const int signal,
                                 const double realized_move_points,
                                 const double min_move_points,
                                 const double expected_move_points,
                                 const double &probs[])
{
   if(ai_idx < 0 || ai_idx >= FXAI_AI_COUNT) return;
   if(label_class < 0 || label_class > 2) return;

   int best = 0;
   for(int c=1; c<3; c++)
      if(probs[c] > probs[best]) best = c;

   double p_true = FXAI_Clamp(probs[label_class], 0.0, 1.0);
   double min_mv = MathMax(min_move_points, 0.10);
   double target = 1.0;

   if(signal == 1 || signal == 0)
   {
      double net_points = (signal == 1 ? realized_move_points : -realized_move_points) - min_mv;
      double edge_norm = FXAI_Clamp(net_points / min_mv, -2.5, 2.5);
      int pred_class = (signal == 1 ? (int)FXAI_LABEL_BUY : (int)FXAI_LABEL_SELL);
      double cls_bonus = (pred_class == label_class ? 0.20 : -0.20);
      if(label_class == (int)FXAI_LABEL_SKIP) cls_bonus -= 0.15;
      double exp_mv = MathMax(expected_move_points, min_mv);
      double exp_fit = 1.0 - FXAI_Clamp(MathAbs(MathAbs(realized_move_points) - exp_mv) / MathMax(exp_mv, 0.10), 0.0, 1.5);
      target = 1.0 + (0.35 * edge_norm) + cls_bonus + (0.10 * (p_true - 0.5) * 2.0) + (0.08 * exp_fit);
   }
   else
   {
      // Abstention-aware: reward correct skips, penalize missed opportunities.
      if(label_class == (int)FXAI_LABEL_SKIP)
      {
         target = 1.10 + (0.10 * p_true);
      }
      else
      {
         double opportunity = FXAI_Clamp((MathAbs(realized_move_points) - min_mv) / min_mv, 0.0, 3.0);
         target = 0.95 - (0.20 * opportunity);
      }
   }

   if(best == label_class) target += 0.05;
   target = FXAI_Clamp(target, 0.20, 2.80);
   g_model_reliability[ai_idx] = FXAI_Clamp((0.97 * g_model_reliability[ai_idx]) + (0.03 * target), 0.20, 3.00);
}

void FXAI_ResetReliabilityPending()
{
   int default_h = FXAI_ClampHorizon(PredictionTargetMinutes);
   for(int ai=0; ai<FXAI_AI_COUNT; ai++)
   {
      g_rel_pending_head[ai] = 0;
      g_rel_pending_tail[ai] = 0;
      for(int k=0; k<FXAI_REL_MAX_PENDING; k++)
      {
         g_rel_pending_seq[ai][k] = -1;
         g_rel_pending_signal[ai][k] = -1;
         g_rel_pending_regime[ai][k] = -1;
         g_rel_pending_expected_move[ai][k] = 0.0;
         g_rel_pending_horizon[ai][k] = default_h;
      }
   }
   g_rel_clock_bar_time = 0;
   g_rel_clock_seq = 0;
}

void FXAI_ResetHorizonPolicyPending()
{
   g_hpolicy_pending_head = 0;
   g_hpolicy_pending_tail = 0;
   for(int k=0; k<FXAI_REL_MAX_PENDING; k++)
   {
      g_hpolicy_pending_seq[k] = -1;
      g_hpolicy_pending_regime[k] = 0;
      g_hpolicy_pending_horizon[k] = FXAI_ClampHorizon(PredictionTargetMinutes);
      g_hpolicy_pending_min_move[k] = 0.0;
      for(int j=0; j<FXAI_HPOL_FEATS; j++)
         g_hpolicy_pending_feat[k][j] = 0.0;
   }
}

void FXAI_ResetStackPending()
{
   int default_h = FXAI_ClampHorizon(PredictionTargetMinutes);
   g_stack_pending_head = 0;
   g_stack_pending_tail = 0;
   for(int k=0; k<FXAI_REL_MAX_PENDING; k++)
   {
      g_stack_pending_seq[k] = -1;
      g_stack_pending_signal[k] = -1;
      g_stack_pending_regime[k] = -1;
      g_stack_pending_horizon[k] = default_h;
      g_stack_pending_prob[k][0] = 0.0;
      g_stack_pending_prob[k][1] = 0.0;
      g_stack_pending_prob[k][2] = 0.0;
      g_stack_pending_expected_move[k] = 0.0;
      for(int j=0; j<FXAI_STACK_FEATS; j++)
         g_stack_pending_feat[k][j] = 0.0;
   }
}

void FXAI_ResetAdaptiveRoutingState()
{
   for(int ai=0; ai<FXAI_AI_COUNT; ai++)
   {
      for(int r=0; r<FXAI_REGIME_COUNT; r++)
      {
         g_model_thr_regime_ready[ai][r] = false;
         g_model_buy_thr_regime[ai][r] = g_model_buy_thr[ai];
         g_model_sell_thr_regime[ai][r] = g_model_sell_thr[ai];
      }
      for(int h=0; h<FXAI_MAX_HORIZONS; h++)
      {
         g_model_horizon_edge_ema[ai][h] = 0.0;
         g_model_horizon_edge_ready[ai][h] = false;
         g_model_horizon_obs[ai][h] = 0;
      }
   }

   for(int r=0; r<FXAI_REGIME_COUNT; r++)
   {
      g_horizon_regime_total_obs[r] = 0.0;
      g_stack_ready[r] = false;
      g_stack_obs[r] = 0;
      g_hpolicy_ready[r] = false;
      g_hpolicy_obs[r] = 0;
      for(int h=0; h<FXAI_STACK_HIDDEN; h++)
      {
         for(int k=0; k<FXAI_STACK_FEATS; k++)
            g_stack_w1[r][h][k] = 0.0;
         g_stack_b1[r][h] = 0.0;
      }
      for(int c=0; c<3; c++)
      {
         g_stack_b2[r][c] = 0.0;
         for(int h=0; h<FXAI_STACK_HIDDEN; h++)
            g_stack_w2[r][c][h] = 0.0;
      }
      for(int k=0; k<FXAI_HPOL_FEATS; k++)
         g_hpolicy_w[r][k] = 0.0;
      for(int h=0; h<FXAI_MAX_HORIZONS; h++)
      {
         g_horizon_regime_edge_ema[r][h] = 0.0;
         g_horizon_regime_edge_ready[r][h] = false;
         g_horizon_regime_obs[r][h] = 0;
      }
   }
}

void FXAI_EnqueueStackPending(const int signal_seq,
                              const int signal,
                              const int regime_id,
                              const int horizon_minutes,
                              const double expected_move_points,
                              const double &probs[],
                              const double &feat[])
{
   if(signal_seq < 0) return;
   int h = FXAI_ClampHorizon(horizon_minutes);
   int head = g_stack_pending_head;
   int tail = g_stack_pending_tail;
   int prev = tail - 1;
   if(prev < 0) prev += FXAI_REL_MAX_PENDING;

   if(head != tail && g_stack_pending_seq[prev] == signal_seq)
   {
      g_stack_pending_signal[prev] = signal;
      g_stack_pending_regime[prev] = regime_id;
      g_stack_pending_horizon[prev] = h;
      g_stack_pending_expected_move[prev] = expected_move_points;
      g_stack_pending_prob[prev][0] = probs[0];
      g_stack_pending_prob[prev][1] = probs[1];
      g_stack_pending_prob[prev][2] = probs[2];
      for(int j=0; j<FXAI_STACK_FEATS; j++)
         g_stack_pending_feat[prev][j] = feat[j];
      return;
   }

   g_stack_pending_seq[tail] = signal_seq;
   g_stack_pending_signal[tail] = signal;
   g_stack_pending_regime[tail] = regime_id;
   g_stack_pending_horizon[tail] = h;
   g_stack_pending_expected_move[tail] = expected_move_points;
   g_stack_pending_prob[tail][0] = probs[0];
   g_stack_pending_prob[tail][1] = probs[1];
   g_stack_pending_prob[tail][2] = probs[2];
   for(int j=0; j<FXAI_STACK_FEATS; j++)
      g_stack_pending_feat[tail][j] = feat[j];

   int next_tail = tail + 1;
   if(next_tail >= FXAI_REL_MAX_PENDING) next_tail = 0;
   if(next_tail == head)
   {
      head++;
      if(head >= FXAI_REL_MAX_PENDING) head = 0;
      g_stack_pending_head = head;
   }
   g_stack_pending_tail = next_tail;
}

void FXAI_UpdateStackFromPending(const int current_signal_seq,
                                 const FXAIDataSnapshot &snapshot,
                                 const int &spread_m1[],
                                 const double &high_arr[],
                                 const double &low_arr[],
                                 const double &close_arr[],
                                 const double commission_points,
                                 const double cost_buffer_points,
                                 const double ev_threshold_points)
{
   if(current_signal_seq < 0) return;
   int head = g_stack_pending_head;
   int tail = g_stack_pending_tail;
   if(head == tail) return;

   int keep_seq[];
   int keep_signal[];
   int keep_regime[];
   int keep_horizon[];
   double keep_expected[];
   double keep_prob0[];
   double keep_prob1[];
   double keep_prob2[];
   double keep_feat[][FXAI_STACK_FEATS];
   ArrayResize(keep_seq, 0);
   ArrayResize(keep_signal, 0);
   ArrayResize(keep_regime, 0);
   ArrayResize(keep_horizon, 0);
   ArrayResize(keep_expected, 0);
   ArrayResize(keep_prob0, 0);
   ArrayResize(keep_prob1, 0);
   ArrayResize(keep_prob2, 0);
   ArrayResize(keep_feat, 0);

   int idx = head;
   while(idx != tail)
   {
      int seq_pred = g_stack_pending_seq[idx];
      int pending_signal = g_stack_pending_signal[idx];
      int pending_regime = g_stack_pending_regime[idx];
      int pending_h = FXAI_ClampHorizon(g_stack_pending_horizon[idx]);
      bool consumed = false;

      if(seq_pred < 0)
      {
         consumed = true;
      }
      else
      {
         int age = current_signal_seq - seq_pred;
         if(age >= pending_h)
         {
            int idx_pred = age;
            if(idx_pred >= 0 && idx_pred < ArraySize(close_arr) &&
               idx_pred < ArraySize(high_arr) &&
               idx_pred < ArraySize(low_arr))
            {
               double spread_i = FXAI_GetSpreadAtIndex(idx_pred, spread_m1, snapshot.spread_points);
               double min_move_i = spread_i + commission_points + cost_buffer_points;
               if(min_move_i < 0.0) min_move_i = 0.0;

               double move_points = 0.0;
               int label_class = FXAI_BuildTripleBarrierLabel(idx_pred,
                                                              pending_h,
                                                              min_move_i,
                                                              ev_threshold_points,
                                                              snapshot,
                                                              high_arr,
                                                              low_arr,
                                                              close_arr,
                                                              move_points);
               double feat[FXAI_STACK_FEATS];
               for(int j=0; j<FXAI_STACK_FEATS; j++)
                  feat[j] = g_stack_pending_feat[idx][j];
               double sw = FXAI_MoveEdgeWeight(move_points, min_move_i);
               if(pending_signal == -1 && label_class != (int)FXAI_LABEL_SKIP)
                  sw *= 0.80;
               FXAI_StackUpdate(pending_regime, label_class, feat, sw);
            }
            consumed = true;
         }
      }

      if(!consumed)
      {
         int ks = ArraySize(keep_seq);
         if(ks < FXAI_REL_MAX_PENDING)
         {
            ArrayResize(keep_seq, ks + 1);
            ArrayResize(keep_signal, ks + 1);
            ArrayResize(keep_regime, ks + 1);
            ArrayResize(keep_horizon, ks + 1);
            ArrayResize(keep_expected, ks + 1);
            ArrayResize(keep_prob0, ks + 1);
            ArrayResize(keep_prob1, ks + 1);
            ArrayResize(keep_prob2, ks + 1);
            ArrayResize(keep_feat, ks + 1);

            keep_seq[ks] = g_stack_pending_seq[idx];
            keep_signal[ks] = g_stack_pending_signal[idx];
            keep_regime[ks] = g_stack_pending_regime[idx];
            keep_horizon[ks] = g_stack_pending_horizon[idx];
            keep_expected[ks] = g_stack_pending_expected_move[idx];
            keep_prob0[ks] = g_stack_pending_prob[idx][0];
            keep_prob1[ks] = g_stack_pending_prob[idx][1];
            keep_prob2[ks] = g_stack_pending_prob[idx][2];
            for(int j=0; j<FXAI_STACK_FEATS; j++)
               keep_feat[ks][j] = g_stack_pending_feat[idx][j];
         }
      }

      idx++;
      if(idx >= FXAI_REL_MAX_PENDING) idx = 0;
   }

   FXAI_ResetStackPending();
   int keep_n = ArraySize(keep_seq);
   if(keep_n > FXAI_REL_MAX_PENDING) keep_n = FXAI_REL_MAX_PENDING;
   for(int k=0; k<keep_n; k++)
   {
      g_stack_pending_seq[k] = keep_seq[k];
      g_stack_pending_signal[k] = keep_signal[k];
      g_stack_pending_regime[k] = keep_regime[k];
      g_stack_pending_horizon[k] = keep_horizon[k];
      g_stack_pending_expected_move[k] = keep_expected[k];
      g_stack_pending_prob[k][0] = keep_prob0[k];
      g_stack_pending_prob[k][1] = keep_prob1[k];
      g_stack_pending_prob[k][2] = keep_prob2[k];
      for(int j=0; j<FXAI_STACK_FEATS; j++)
         g_stack_pending_feat[k][j] = keep_feat[k][j];
   }
   g_stack_pending_head = 0;
   g_stack_pending_tail = keep_n;
   if(g_stack_pending_tail >= FXAI_REL_MAX_PENDING)
      g_stack_pending_tail = FXAI_REL_MAX_PENDING - 1;
}

void FXAI_EnqueueHorizonPolicyPending(const int signal_seq,
                                      const int regime_id,
                                      const int horizon_minutes,
                                      const double min_move_points,
                                      const double &feat[])
{
   if(signal_seq < 0) return;
   int head = g_hpolicy_pending_head;
   int tail = g_hpolicy_pending_tail;
   int prev = tail - 1;
   if(prev < 0) prev += FXAI_REL_MAX_PENDING;

   if(head != tail && g_hpolicy_pending_seq[prev] == signal_seq)
   {
      g_hpolicy_pending_regime[prev] = regime_id;
      g_hpolicy_pending_horizon[prev] = FXAI_ClampHorizon(horizon_minutes);
      g_hpolicy_pending_min_move[prev] = min_move_points;
      for(int k=0; k<FXAI_HPOL_FEATS; k++)
         g_hpolicy_pending_feat[prev][k] = feat[k];
      return;
   }

   g_hpolicy_pending_seq[tail] = signal_seq;
   g_hpolicy_pending_regime[tail] = regime_id;
   g_hpolicy_pending_horizon[tail] = FXAI_ClampHorizon(horizon_minutes);
   g_hpolicy_pending_min_move[tail] = min_move_points;
   for(int k=0; k<FXAI_HPOL_FEATS; k++)
      g_hpolicy_pending_feat[tail][k] = feat[k];

   int next_tail = tail + 1;
   if(next_tail >= FXAI_REL_MAX_PENDING) next_tail = 0;
   if(next_tail == head)
   {
      head++;
      if(head >= FXAI_REL_MAX_PENDING) head = 0;
      g_hpolicy_pending_head = head;
   }
   g_hpolicy_pending_tail = next_tail;
}

void FXAI_UpdateHorizonPolicy(const int regime_id,
                              const double &feat[],
                              const double reward_scaled)
{
   int r = regime_id;
   if(r < 0 || r >= FXAI_REGIME_COUNT) r = 0;

   double pred = 0.0;
   for(int k=0; k<FXAI_HPOL_FEATS; k++)
      pred += g_hpolicy_w[r][k] * feat[k];

   double err = FXAI_Clamp(reward_scaled - pred, -4.0, 4.0);
   double lr = 0.020 / MathSqrt(1.0 + 0.02 * (double)g_hpolicy_obs[r]);
   lr = FXAI_Clamp(lr, 0.0015, 0.020);

   for(int k=0; k<FXAI_HPOL_FEATS; k++)
   {
      double reg = (k == 0 ? 0.0 : 0.0008 * g_hpolicy_w[r][k]);
      g_hpolicy_w[r][k] += lr * (err * feat[k] - reg);
   }

   g_hpolicy_obs[r]++;
   if(g_hpolicy_obs[r] > 200000) g_hpolicy_obs[r] = 200000;
   g_hpolicy_ready[r] = true;
}

void FXAI_UpdateHorizonPolicyFromPending(const int current_signal_seq,
                                         const FXAIDataSnapshot &snapshot,
                                         const int &spread_m1[],
                                         const double &high_arr[],
                                         const double &low_arr[],
                                         const double &close_arr[],
                                         const double commission_points,
                                         const double cost_buffer_points,
                                         const double ev_threshold_points)
{
   int head = g_hpolicy_pending_head;
   int tail = g_hpolicy_pending_tail;
   if(head == tail) return;

   int idx = head;
   int keep_seq[];
   int keep_regime[];
   int keep_horizon[];
   double keep_min_move[];
   double keep_feat[][FXAI_HPOL_FEATS];
   ArrayResize(keep_seq, 0);
   ArrayResize(keep_regime, 0);
   ArrayResize(keep_horizon, 0);
   ArrayResize(keep_min_move, 0);
   ArrayResize(keep_feat, 0);

   while(idx != tail)
   {
      bool consumed = false;
      int seq_pred = g_hpolicy_pending_seq[idx];
      int pending_h = FXAI_ClampHorizon(g_hpolicy_pending_horizon[idx]);
      if(seq_pred < 0)
      {
         consumed = true;
      }
      else
      {
         int age = current_signal_seq - seq_pred;
         if(age >= pending_h)
         {
            int idx_pred = age;
            if(idx_pred >= 0 && idx_pred < ArraySize(close_arr) &&
               idx_pred < ArraySize(high_arr) &&
               idx_pred < ArraySize(low_arr))
            {
               double spread_i = FXAI_GetSpreadAtIndex(idx_pred, spread_m1, snapshot.spread_points);
               double min_move_i = spread_i + commission_points + cost_buffer_points;
               if(min_move_i < 0.0) min_move_i = 0.0;
               double move_points = 0.0;
               double mfe_points = 0.0;
               double mae_points = 0.0;
               double time_to_hit_frac = 1.0;
               int path_flags = 0;
               int label_class = FXAI_BuildTripleBarrierLabelEx(idx_pred,
                                                                pending_h,
                                                                min_move_i,
                                                                ev_threshold_points,
                                                                snapshot,
                                                                high_arr,
                                                                low_arr,
                                                                close_arr,
                                                                move_points,
                                                                mfe_points,
                                                                mae_points,
                                                                time_to_hit_frac,
                                                                path_flags);
               double edge = MathMax(MathAbs(move_points) - min_move_i, 0.0);
               double reward = -0.25;
               if(label_class != (int)FXAI_LABEL_SKIP)
               {
                  double speed_bonus = 1.0 - FXAI_Clamp(time_to_hit_frac, 0.0, 1.0);
                  double quality = 1.0 + 0.20 * speed_bonus - 0.12 * FXAI_Clamp(mae_points / MathMax(mfe_points, min_move_i), 0.0, 3.0);
                  if((path_flags & FXAI_PATHFLAG_DUAL_HIT) != 0) quality -= 0.10;
                  reward = quality * edge / MathMax(min_move_i, 0.50);
               }
               reward = FXAI_Clamp(reward, -2.0, 6.0);
               double feat_local[FXAI_HPOL_FEATS];
               for(int k=0; k<FXAI_HPOL_FEATS; k++)
                  feat_local[k] = g_hpolicy_pending_feat[idx][k];
               FXAI_UpdateHorizonPolicy(g_hpolicy_pending_regime[idx], feat_local, reward);
            }
            consumed = true;
         }
      }

      if(!consumed)
      {
         int ks = ArraySize(keep_seq);
         ArrayResize(keep_seq, ks + 1);
         ArrayResize(keep_regime, ks + 1);
         ArrayResize(keep_horizon, ks + 1);
         ArrayResize(keep_min_move, ks + 1);
         ArrayResize(keep_feat, ks + 1);
         keep_seq[ks] = g_hpolicy_pending_seq[idx];
         keep_regime[ks] = g_hpolicy_pending_regime[idx];
         keep_horizon[ks] = g_hpolicy_pending_horizon[idx];
         keep_min_move[ks] = g_hpolicy_pending_min_move[idx];
         for(int k=0; k<FXAI_HPOL_FEATS; k++)
            keep_feat[ks][k] = g_hpolicy_pending_feat[idx][k];
      }

      idx++;
      if(idx >= FXAI_REL_MAX_PENDING) idx = 0;
   }

   FXAI_ResetHorizonPolicyPending();
   int keep_n = ArraySize(keep_seq);
   for(int k=0; k<keep_n && k<FXAI_REL_MAX_PENDING; k++)
   {
      g_hpolicy_pending_seq[k] = keep_seq[k];
      g_hpolicy_pending_regime[k] = keep_regime[k];
      g_hpolicy_pending_horizon[k] = keep_horizon[k];
      g_hpolicy_pending_min_move[k] = keep_min_move[k];
      for(int j=0; j<FXAI_HPOL_FEATS; j++)
         g_hpolicy_pending_feat[k][j] = keep_feat[k][j];
   }
   g_hpolicy_pending_head = 0;
   g_hpolicy_pending_tail = keep_n;
   if(g_hpolicy_pending_tail >= FXAI_REL_MAX_PENDING)
      g_hpolicy_pending_tail = FXAI_REL_MAX_PENDING - 1;
}

void FXAI_AdvanceReliabilityClock(const datetime signal_bar)
{
   if(signal_bar <= 0) return;
   if(g_rel_clock_bar_time == 0)
   {
      g_rel_clock_bar_time = signal_bar;
      g_rel_clock_seq = 0;
      return;
   }

   if(signal_bar != g_rel_clock_bar_time)
   {
      g_rel_clock_seq++;
      g_rel_clock_bar_time = signal_bar;
   }
}

void FXAI_EnqueueReliabilityPending(const int ai_idx,
                                    const int signal_seq,
                                    const int signal,
                                    const int regime_id,
                                    const double expected_move_points,
                                    const int horizon_minutes,
                                    const double &probs[])
{
   if(ai_idx < 0 || ai_idx >= FXAI_AI_COUNT) return;
   if(signal_seq < 0) return;
   int h = FXAI_ClampHorizon(horizon_minutes);

   int head = g_rel_pending_head[ai_idx];
   int tail = g_rel_pending_tail[ai_idx];

   int prev = tail - 1;
   if(prev < 0) prev += FXAI_REL_MAX_PENDING;
   if(head != tail && g_rel_pending_seq[ai_idx][prev] == signal_seq)
   {
      g_rel_pending_prob[ai_idx][prev][0] = probs[0];
      g_rel_pending_prob[ai_idx][prev][1] = probs[1];
      g_rel_pending_prob[ai_idx][prev][2] = probs[2];
      g_rel_pending_signal[ai_idx][prev] = signal;
      g_rel_pending_regime[ai_idx][prev] = regime_id;
      g_rel_pending_expected_move[ai_idx][prev] = expected_move_points;
      g_rel_pending_horizon[ai_idx][prev] = h;
      return;
   }

   g_rel_pending_seq[ai_idx][tail] = signal_seq;
   g_rel_pending_prob[ai_idx][tail][0] = probs[0];
   g_rel_pending_prob[ai_idx][tail][1] = probs[1];
   g_rel_pending_prob[ai_idx][tail][2] = probs[2];
   g_rel_pending_signal[ai_idx][tail] = signal;
   g_rel_pending_regime[ai_idx][tail] = regime_id;
   g_rel_pending_expected_move[ai_idx][tail] = expected_move_points;
   g_rel_pending_horizon[ai_idx][tail] = h;

   int next_tail = tail + 1;
   if(next_tail >= FXAI_REL_MAX_PENDING) next_tail = 0;
   if(next_tail == head)
   {
      head++;
      if(head >= FXAI_REL_MAX_PENDING) head = 0;
      g_rel_pending_head[ai_idx] = head;
   }
   g_rel_pending_tail[ai_idx] = next_tail;
}

void FXAI_UpdateReliabilityFromPending(const int ai_idx,
                                      const int current_signal_seq,
                                      const FXAIDataSnapshot &snapshot,
                                      const int &spread_m1[],
                                      const datetime &time_arr[],
                                      const double &high_arr[],
                                      const double &low_arr[],
                                      const double &close_arr[],
                                      const double commission_points,
                                      const double cost_buffer_points,
                                      const double ev_threshold_points)
{
   if(ai_idx < 0 || ai_idx >= FXAI_AI_COUNT) return;
   if(current_signal_seq < 0) return;

   int head = g_rel_pending_head[ai_idx];
   int tail = g_rel_pending_tail[ai_idx];
   if(head == tail) return;

   int keep_seq[];
   int keep_signal[];
   int keep_regime[];
   int keep_horizon[];
   double keep_expected[];
   double keep_prob0[];
   double keep_prob1[];
   double keep_prob2[];
   ArrayResize(keep_seq, 0);
   ArrayResize(keep_signal, 0);
   ArrayResize(keep_regime, 0);
   ArrayResize(keep_horizon, 0);
   ArrayResize(keep_expected, 0);
   ArrayResize(keep_prob0, 0);
   ArrayResize(keep_prob1, 0);
   ArrayResize(keep_prob2, 0);

   int idx = head;
   while(idx != tail)
   {
      int seq_pred = g_rel_pending_seq[ai_idx][idx];
      int pending_signal = g_rel_pending_signal[ai_idx][idx];
      int pending_regime = g_rel_pending_regime[ai_idx][idx];
      double pending_expected_move = g_rel_pending_expected_move[ai_idx][idx];
      int pending_h = FXAI_ClampHorizon(g_rel_pending_horizon[ai_idx][idx]);

      double p0 = g_rel_pending_prob[ai_idx][idx][0];
      double p1 = g_rel_pending_prob[ai_idx][idx][1];
      double p2 = g_rel_pending_prob[ai_idx][idx][2];

      bool consumed = false;
      if(seq_pred < 0)
      {
         consumed = true;
      }
      else
      {
         int age = current_signal_seq - seq_pred;
         if(age >= pending_h)
         {
            int idx_pred = age;
            int idx_future = age - pending_h;
            if(idx_pred >= 0 && idx_pred < ArraySize(close_arr) &&
               idx_pred < ArraySize(time_arr) &&
               idx_pred < ArraySize(high_arr) &&
               idx_pred < ArraySize(low_arr) &&
               idx_future >= 0 && idx_future < ArraySize(close_arr))
            {
               double spread_i = FXAI_GetSpreadAtIndex(idx_pred, spread_m1, snapshot.spread_points);
               double min_move_i = spread_i + commission_points + cost_buffer_points;
               if(min_move_i < 0.0) min_move_i = 0.0;

               double move_points = 0.0;
               int label_class = FXAI_BuildTripleBarrierLabel(idx_pred,
                                                              pending_h,
                                                              min_move_i,
                                                              ev_threshold_points,
                                                              snapshot,
                                                              high_arr,
                                                              low_arr,
                                                              close_arr,
                                                              move_points);

               double probs_eval[3];
               probs_eval[0] = p0;
               probs_eval[1] = p1;
               probs_eval[2] = p2;

               FXAI_UpdateModelReliability(ai_idx,
                                           label_class,
                                           pending_signal,
                                           move_points,
                                           min_move_i,
                                           pending_expected_move,
                                           probs_eval);
               FXAI_UpdateRegimeCalibration(ai_idx, pending_regime, label_class, probs_eval);
               FXAI_UpdateModelPerformance(ai_idx,
                                           pending_regime,
                                           label_class,
                                           pending_signal,
                                           move_points,
                                           min_move_i,
                                           pending_h,
                                           pending_expected_move,
                                           probs_eval);
               FXAI_BoostReplayPriorityByOutcome(time_arr[idx_pred],
                                                pending_h,
                                                pending_regime,
                                                label_class,
                                                pending_signal,
                                                move_points,
                                                min_move_i);
            }
            consumed = true;
         }
      }

      if(!consumed)
      {
         int ks = ArraySize(keep_seq);
         if(ks < FXAI_REL_MAX_PENDING)
         {
            ArrayResize(keep_seq, ks + 1);
            ArrayResize(keep_signal, ks + 1);
            ArrayResize(keep_regime, ks + 1);
            ArrayResize(keep_horizon, ks + 1);
            ArrayResize(keep_expected, ks + 1);
            ArrayResize(keep_prob0, ks + 1);
            ArrayResize(keep_prob1, ks + 1);
            ArrayResize(keep_prob2, ks + 1);

            keep_seq[ks] = seq_pred;
            keep_signal[ks] = pending_signal;
            keep_regime[ks] = pending_regime;
            keep_horizon[ks] = pending_h;
            keep_expected[ks] = pending_expected_move;
            keep_prob0[ks] = p0;
            keep_prob1[ks] = p1;
            keep_prob2[ks] = p2;
         }
      }

      idx++;
      if(idx >= FXAI_REL_MAX_PENDING) idx = 0;
   }

   int keep_n = ArraySize(keep_seq);
   if(keep_n > FXAI_REL_MAX_PENDING) keep_n = FXAI_REL_MAX_PENDING;
   for(int k=0; k<FXAI_REL_MAX_PENDING; k++)
   {
      g_rel_pending_seq[ai_idx][k] = -1;
      g_rel_pending_signal[ai_idx][k] = -1;
      g_rel_pending_regime[ai_idx][k] = -1;
      g_rel_pending_expected_move[ai_idx][k] = 0.0;
      g_rel_pending_horizon[ai_idx][k] = FXAI_ClampHorizon(PredictionTargetMinutes);
      g_rel_pending_prob[ai_idx][k][0] = 0.0;
      g_rel_pending_prob[ai_idx][k][1] = 0.0;
      g_rel_pending_prob[ai_idx][k][2] = 0.0;
   }

   for(int k=0; k<keep_n; k++)
   {
      g_rel_pending_seq[ai_idx][k] = keep_seq[k];
      g_rel_pending_signal[ai_idx][k] = keep_signal[k];
      g_rel_pending_regime[ai_idx][k] = keep_regime[k];
      g_rel_pending_expected_move[ai_idx][k] = keep_expected[k];
      g_rel_pending_horizon[ai_idx][k] = keep_horizon[k];
      g_rel_pending_prob[ai_idx][k][0] = keep_prob0[k];
      g_rel_pending_prob[ai_idx][k][1] = keep_prob1[k];
      g_rel_pending_prob[ai_idx][k][2] = keep_prob2[k];
   }

   g_rel_pending_head[ai_idx] = 0;
   g_rel_pending_tail[ai_idx] = keep_n;
   if(g_rel_pending_tail[ai_idx] >= FXAI_REL_MAX_PENDING)
      g_rel_pending_tail[ai_idx] = FXAI_REL_MAX_PENDING - 1;
}

int FXAI_GetMaxPendingHorizon(const int fallback_h)
{
   int hmax = FXAI_ClampHorizon(fallback_h);
   for(int ai=0; ai<FXAI_AI_COUNT; ai++)
   {
      int head = g_rel_pending_head[ai];
      int tail = g_rel_pending_tail[ai];
      int idx = head;
      while(idx != tail)
      {
         int seq_pred = g_rel_pending_seq[ai][idx];
         if(seq_pred >= 0)
         {
            int h = FXAI_ClampHorizon(g_rel_pending_horizon[ai][idx]);
            if(h > hmax) hmax = h;
         }
         idx++;
         if(idx >= FXAI_REL_MAX_PENDING) idx = 0;
      }
   }

   int idx = g_stack_pending_head;
   while(idx != g_stack_pending_tail)
   {
      int seq_pred = g_stack_pending_seq[idx];
      if(seq_pred >= 0)
      {
         int h = FXAI_ClampHorizon(g_stack_pending_horizon[idx]);
         if(h > hmax) hmax = h;
      }
      idx++;
      if(idx >= FXAI_REL_MAX_PENDING) idx = 0;
   }

   idx = g_hpolicy_pending_head;
   while(idx != g_hpolicy_pending_tail)
   {
      int seq_pred = g_hpolicy_pending_seq[idx];
      if(seq_pred >= 0)
      {
         int h = FXAI_ClampHorizon(g_hpolicy_pending_horizon[idx]);
         if(h > hmax) hmax = h;
      }
      idx++;
      if(idx >= FXAI_REL_MAX_PENDING) idx = 0;
   }
   return hmax;
}

void FXAI_ProcessReliabilityBar(const string symbol)
{
   if(StringLen(symbol) <= 0) return;

   int H = FXAI_ClampHorizon(PredictionTargetMinutes);
   H = FXAI_GetMaxConfiguredHorizon(H);
   H = FXAI_GetMaxPendingHorizon(H);

   datetime signal_bar = iTime(symbol, PERIOD_M1, 1);
   if(signal_bar <= 0) return;

   static string rel_symbol = "";
   static datetime rel_last_processed_bar = 0;
   static datetime rel_last_rates_bar = 0;
   static MqlRates rel_rates_m1[];
   static double rel_open_arr[];
   static double rel_high_arr[];
   static double rel_low_arr[];
   static double rel_close_arr[];
   static datetime rel_time_arr[];
   static int rel_spread_arr[];

   if(rel_symbol != symbol)
   {
      rel_symbol = symbol;
      rel_last_processed_bar = 0;
      rel_last_rates_bar = 0;
      ArrayResize(rel_rates_m1, 0);
      ArrayResize(rel_open_arr, 0);
      ArrayResize(rel_high_arr, 0);
      ArrayResize(rel_low_arr, 0);
      ArrayResize(rel_close_arr, 0);
      ArrayResize(rel_time_arr, 0);
      ArrayResize(rel_spread_arr, 0);
   }

   FXAI_AdvanceReliabilityClock(signal_bar);
   if(signal_bar == rel_last_processed_bar) return;
   rel_last_processed_bar = signal_bar;

   int needed = H + 64;
   if(needed < 128) needed = 128;
   if(needed > 1500) needed = 1500;

   if(!FXAI_UpdateRatesRolling(symbol, PERIOD_M1, needed, rel_last_rates_bar, rel_rates_m1))
      return;

   FXAI_ExtractRatesCloseTimeSpread(rel_rates_m1, rel_close_arr, rel_time_arr, rel_spread_arr);
   FXAI_ExtractRatesOHLC(rel_rates_m1, rel_open_arr, rel_high_arr, rel_low_arr, rel_close_arr);
   if(ArraySize(rel_close_arr) <= H || ArraySize(rel_spread_arr) <= H)
      return;

   FXAIDataSnapshot snapshot;
   if(!FXAI_ExportDataSnapshot(symbol, AI_CommissionPerLotSide, AI_CostBufferPoints, snapshot))
      return;
   snapshot.bar_time = signal_bar;

   double cost_buffer_points = (AI_CostBufferPoints < 0.0 ? 0.0 : AI_CostBufferPoints);
   double commission_points = snapshot.commission_points;
   double evThresholdPoints = FXAI_Clamp(AI_EVThresholdPoints, 0.0, 100.0);
   int signal_seq = g_rel_clock_seq;

   for(int ai_idx=0; ai_idx<FXAI_AI_COUNT; ai_idx++)
   {
      FXAI_UpdateReliabilityFromPending(ai_idx,
                                       signal_seq,
                                       snapshot,
                                       rel_spread_arr,
                                       rel_time_arr,
                                       rel_high_arr,
                                       rel_low_arr,
                                       rel_close_arr,
                                       commission_points,
                                       cost_buffer_points,
                                       evThresholdPoints);
   }

   FXAI_UpdateStackFromPending(signal_seq,
                               snapshot,
                               rel_spread_arr,
                               rel_high_arr,
                               rel_low_arr,
                               rel_close_arr,
                               commission_points,
                               cost_buffer_points,
                               evThresholdPoints);
   FXAI_UpdateHorizonPolicyFromPending(signal_seq,
                                       snapshot,
                                       rel_spread_arr,
                                       rel_high_arr,
                                       rel_low_arr,
                                       rel_close_arr,
                                       commission_points,
                                       cost_buffer_points,
                                       evThresholdPoints);
}

double FXAI_GetModelVoteWeight(const int ai_idx)
{
   if(ai_idx < 0 || ai_idx >= FXAI_AI_COUNT) return 1.0;
   return FXAI_Clamp(g_model_reliability[ai_idx], 0.20, 3.00);
}

void FXAI_DeriveAdaptiveThresholds(const double base_buy_threshold,
                                  const double base_sell_threshold,
                                  const double min_move_points,
                                  const double expected_move_points,
                                  const double vol_proxy,
                                  double &buy_min_prob,
                                  double &sell_min_prob,
                                  double &skip_min_prob)
{
   double buy_base = FXAI_Clamp(base_buy_threshold, 0.50, 0.95);
   double sell_base = FXAI_Clamp(1.0 - base_sell_threshold, 0.50, 0.95);

   double em = MathMax(expected_move_points, min_move_points + 0.10);
   double cost_ratio = FXAI_Clamp(min_move_points / em, 0.0, 2.0);
   double vol_ratio = FXAI_Clamp(vol_proxy / 4.0, 0.0, 1.0);

   double tighten = FXAI_Clamp(((cost_ratio - 0.35) * 0.35) + (0.10 * vol_ratio), 0.0, 0.25);

   buy_min_prob = FXAI_Clamp(buy_base + tighten, 0.50, 0.96);
   sell_min_prob = FXAI_Clamp(sell_base + tighten, 0.50, 0.96);
   skip_min_prob = FXAI_Clamp(0.45 + (0.20 * cost_ratio) + (0.10 * vol_ratio), 0.35, 0.85);
}

int FXAI_ClassSignalFromEV(const double &probs[],
                          const double buy_min_prob,
                          const double sell_min_prob,
                          const double skip_min_prob,
                          const double expected_move_points,
                          const double min_move_points,
                          const double ev_threshold_points)
{
   if(expected_move_points <= 0.0) return -1;

   double p_sell = probs[(int)FXAI_LABEL_SELL];
   double p_buy = probs[(int)FXAI_LABEL_BUY];
   double p_skip = probs[(int)FXAI_LABEL_SKIP];

   if(p_skip >= skip_min_prob) return -1;

   double buy_ev = ((2.0 * p_buy) - 1.0) * expected_move_points - min_move_points;
   double sell_ev = ((2.0 * p_sell) - 1.0) * expected_move_points - min_move_points;

   if(p_buy >= buy_min_prob && buy_ev >= ev_threshold_points && buy_ev > sell_ev)
      return 1;
   if(p_sell >= sell_min_prob && sell_ev >= ev_threshold_points && sell_ev > buy_ev)
      return 0;

   return -1;
}

void FXAI_SanitizeThresholdPair(double &buy_threshold, double &sell_threshold)
{
   buy_threshold = FXAI_Clamp(buy_threshold, 0.50, 0.95);
   sell_threshold = FXAI_Clamp(sell_threshold, 0.05, 0.50);

   if(sell_threshold >= buy_threshold)
   {
      sell_threshold = FXAI_Clamp(sell_threshold, 0.05, 0.49);
      buy_threshold = FXAI_Clamp(MathMax(buy_threshold, sell_threshold + 0.01), 0.50, 0.95);
      if(sell_threshold >= buy_threshold)
      {
         sell_threshold = 0.49;
         buy_threshold = 0.50;
      }
   }
}

int FXAI_BuildTripleBarrierLabelEx(const int i,
                                   const int H,
                                   const double roundtrip_cost_points,
                                   const double ev_threshold_points,
                                   const FXAIDataSnapshot &snapshot,
                                   const double &high_arr[],
                                   const double &low_arr[],
                                   const double &close_arr[],
                                   double &realized_move_points,
                                   double &mfe_points,
                                   double &mae_points,
                                   double &time_to_hit_frac,
                                   int &path_flags)
{
   realized_move_points = 0.0;
   mfe_points = 0.0;
   mae_points = 0.0;
   time_to_hit_frac = 1.0;
   path_flags = 0;
   if(i < 0 || H < 1) return (int)FXAI_LABEL_SKIP;
   if(i >= ArraySize(close_arr) || i >= ArraySize(high_arr) || i >= ArraySize(low_arr))
      return (int)FXAI_LABEL_SKIP;

   double entry = close_arr[i];
   if(entry <= 0.0 || snapshot.point <= 0.0)
      return (int)FXAI_LABEL_SKIP;

   int max_step = H;
   if(i - max_step < 0) max_step = i;
   if(max_step < 1) return (int)FXAI_LABEL_SKIP;

   double ev_min = (ev_threshold_points > 0.0 ? ev_threshold_points : 0.0);
   double barrier = roundtrip_cost_points + ev_min;
   if(barrier < 0.10) barrier = 0.10;

   // Asymmetric barrier shaping: blend short momentum and local range regime.
   // This reduces excessive SKIP labels while staying cost-aware.
   double mom = 0.0;
   if(i + 5 < ArraySize(close_arr) && snapshot.point > 0.0)
      mom = FXAI_MovePoints(close_arr[i + 5], close_arr[i], snapshot.point);
   double drift = FXAI_Clamp(mom / MathMax(barrier, 0.10), -1.0, 1.0);

   double range_sum = 0.0;
   int range_n = 0;
   for(int k=0; k<10; k++)
   {
      int ik = i + k;
      if(ik < 0 || ik >= ArraySize(high_arr) || ik >= ArraySize(low_arr)) break;
      range_sum += MathMax(0.0, (high_arr[ik] - low_arr[ik]) / snapshot.point);
      range_n++;
   }
   double range_avg = (range_n > 0 ? (range_sum / (double)range_n) : barrier);
   double vol_scale = FXAI_Clamp(range_avg / MathMax(barrier, 0.10), 0.7, 1.8);

   double buy_barrier = barrier * vol_scale * (1.0 - 0.10 * drift);
   double sell_barrier = barrier * vol_scale * (1.0 + 0.10 * drift);
   if(buy_barrier < 0.10) buy_barrier = 0.10;
   if(sell_barrier < 0.10) sell_barrier = 0.10;

   double best_up = 0.0;
   double best_dn = 0.0;

   for(int step=1; step<=max_step; step++)
   {
      int idx = i - step;
      if(idx < 0) break;
      if(idx >= ArraySize(high_arr) || idx >= ArraySize(low_arr)) break;

      double up_mv = FXAI_MovePoints(entry, high_arr[idx], snapshot.point);
      double dn_mv = FXAI_MovePoints(entry, low_arr[idx], snapshot.point);
      if(up_mv > best_up) best_up = up_mv;
      double dn_abs = MathAbs(dn_mv);
      if(dn_abs > best_dn) best_dn = dn_abs;
      bool hit_up = (up_mv >= buy_barrier);
      bool hit_dn = (dn_mv <= -sell_barrier);

      if(hit_up && !hit_dn)
      {
         realized_move_points = MathMax(up_mv, buy_barrier);
         mfe_points = best_up;
         mae_points = best_dn;
         time_to_hit_frac = FXAI_Clamp((double)step / (double)MathMax(max_step, 1), 0.0, 1.0);
         if(time_to_hit_frac > 0.75) path_flags |= FXAI_PATHFLAG_SLOW_HIT;
         return (int)FXAI_LABEL_BUY;
      }
      if(hit_dn && !hit_up)
      {
         realized_move_points = MathMin(dn_mv, -sell_barrier);
         mfe_points = best_dn;
         mae_points = best_up;
         time_to_hit_frac = FXAI_Clamp((double)step / (double)MathMax(max_step, 1), 0.0, 1.0);
         if(time_to_hit_frac > 0.75) path_flags |= FXAI_PATHFLAG_SLOW_HIT;
         return (int)FXAI_LABEL_SELL;
      }
      if(hit_up && hit_dn)
      {
         path_flags |= FXAI_PATHFLAG_DUAL_HIT;
         // Lower-timeframe disambiguation proxy: use close direction and
         // distance-to-barrier to reduce skip inflation on dual-hit bars.
         double close_mv = FXAI_MovePoints(entry, close_arr[idx], snapshot.point);
         double up_excess = up_mv - buy_barrier;
         double dn_excess = -dn_mv - sell_barrier;
         if(close_mv > 0.0 && up_excess >= dn_excess)
         {
            realized_move_points = MathMax(close_mv, buy_barrier);
            mfe_points = best_up;
            mae_points = best_dn;
            time_to_hit_frac = FXAI_Clamp((double)step / (double)MathMax(max_step, 1), 0.0, 1.0);
            return (int)FXAI_LABEL_BUY;
         }
         if(close_mv < 0.0 && dn_excess >= up_excess)
         {
            realized_move_points = MathMin(close_mv, -sell_barrier);
            mfe_points = best_dn;
            mae_points = best_up;
            time_to_hit_frac = FXAI_Clamp((double)step / (double)MathMax(max_step, 1), 0.0, 1.0);
            return (int)FXAI_LABEL_SELL;
         }
         realized_move_points = close_mv;
         mfe_points = MathMax(best_up, best_dn);
         mae_points = MathMin(best_up, best_dn);
         time_to_hit_frac = FXAI_Clamp((double)step / (double)MathMax(max_step, 1), 0.0, 1.0);
         return FXAI_BuildEVClassLabel(realized_move_points, roundtrip_cost_points, ev_threshold_points);
      }
   }

   int idx_term = i - max_step;
   if(idx_term < 0) idx_term = 0;
   realized_move_points = FXAI_MovePoints(entry, close_arr[idx_term], snapshot.point);
   mfe_points = MathMax(best_up, best_dn);
   mae_points = MathMin(best_up, best_dn);
   if(TradeKiller > 0 && H > TradeKiller)
      path_flags |= FXAI_PATHFLAG_KILLED_EARLY;
   return FXAI_BuildEVClassLabel(realized_move_points, roundtrip_cost_points, ev_threshold_points);
}

int FXAI_BuildTripleBarrierLabel(const int i,
                                 const int H,
                                 const double roundtrip_cost_points,
                                 const double ev_threshold_points,
                                 const FXAIDataSnapshot &snapshot,
                                 const double &high_arr[],
                                 const double &low_arr[],
                                 const double &close_arr[],
                                 double &realized_move_points)
{
   double mfe_points = 0.0;
   double mae_points = 0.0;
   double time_to_hit_frac = 1.0;
   int path_flags = 0;
   return FXAI_BuildTripleBarrierLabelEx(i,
                                         H,
                                         roundtrip_cost_points,
                                         ev_threshold_points,
                                         snapshot,
                                         high_arr,
                                         low_arr,
                                         close_arr,
                                         realized_move_points,
                                         mfe_points,
                                         mae_points,
                                         time_to_hit_frac,
                                         path_flags);
}

double FXAI_RealizedNetPointsForSignal(const int signal,
                                       const double realized_move_points,
                                       const double roundtrip_cost_points,
                                       const int horizon_minutes)
{
   if(signal != 0 && signal != 1) return 0.0;

   double slippage_points = 0.10 + (0.05 * MathMax(roundtrip_cost_points, 0.0));
   if(slippage_points > 5.0) slippage_points = 5.0;

   double kill_penalty = 0.0;
   if(TradeKiller > 0 && horizon_minutes > TradeKiller)
   {
      double frac_cut = 1.0 - ((double)TradeKiller / (double)horizon_minutes);
      kill_penalty = FXAI_Clamp(frac_cut * 0.10 * MathAbs(realized_move_points), 0.0, 10.0);
   }

   double gross = (signal == 1 ? realized_move_points : -realized_move_points);
   return gross - MathMax(roundtrip_cost_points, 0.0) - slippage_points - kill_penalty;
}

void FXAI_ResetPreparedSample(FXAIPreparedSample &sample)
{
   sample.valid = false;
   sample.label_class = (int)FXAI_LABEL_SKIP;
   sample.regime_id = 0;
   sample.horizon_minutes = FXAI_ClampHorizon(PredictionTargetMinutes);
   sample.horizon_slot = 0;
   sample.move_points = 0.0;
   sample.min_move_points = 0.0;
   sample.cost_points = 0.0;
   sample.sample_weight = 1.0;
   sample.quality_score = 1.0;
   sample.mfe_points = 0.0;
   sample.mae_points = 0.0;
   sample.spread_stress = 0.0;
   sample.time_to_hit_frac = 1.0;
   sample.path_flags = 0;
   sample.sample_time = 0;
   for(int k=0; k<FXAI_AI_WEIGHTS; k++)
      sample.x[k] = 0.0;
}

void FXAI_CopyPreparedSamples(const FXAIPreparedSample &src[], FXAIPreparedSample &dst[])
{
   int n = ArraySize(src);
   ArrayResize(dst, n);
   for(int i=0; i<n; i++)
      dst[i] = src[i];
}

bool FXAI_PrepareTrainingSample(const int i,
                               const int H,
                               const double commission_points,
                               const double cost_buffer_points,
                               const double ev_threshold_points,
                               const FXAIDataSnapshot &snapshot,
                               const int &spread_m1[],
                               const datetime &time_arr[],
                               const double &open_arr[],
                               const double &high_arr[],
                               const double &low_arr[],
                               const double &close_arr[],
                               const datetime &time_m5[],
                               const double &close_m5[],
                               const int &map_m5[],
                               const datetime &time_m15[],
                               const double &close_m15[],
                               const int &map_m15[],
                               const datetime &time_m30[],
                               const double &close_m30[],
                               const int &map_m30[],
                               const datetime &time_h1[],
                               const double &close_h1[],
                               const int &map_h1[],
                               const double &ctx_mean_arr[],
                               const double &ctx_std_arr[],
                               const double &ctx_up_arr[],
                               const double &ctx_extra_arr[],
                               const int norm_method_override,
                               FXAIPreparedSample &sample)
{
   FXAI_ResetPreparedSample(sample);

   if(i < 0 || i >= ArraySize(close_arr)) return false;
   bool has_label = (i - H >= 0 && i - H < ArraySize(close_arr));
   double move_points = 0.0;
   double mfe_points = 0.0;
   double mae_points = 0.0;
   double time_to_hit_frac = 1.0;
   int path_flags = 0;
   double spread_i = FXAI_GetSpreadAtIndex(i, spread_m1, snapshot.spread_points);
   double min_move_i = spread_i + commission_points + cost_buffer_points;
   if(min_move_i < 0.0) min_move_i = 0.0;

   int label_class = (int)FXAI_LABEL_SKIP;
   if(has_label)
      label_class = FXAI_BuildTripleBarrierLabelEx(i,
                                                   H,
                                                   min_move_i,
                                                   ev_threshold_points,
                                                   snapshot,
                                                   high_arr,
                                                   low_arr,
                                                   close_arr,
                                                   move_points,
                                                   mfe_points,
                                                   mae_points,
                                                   time_to_hit_frac,
                                                   path_flags);

   double ctx_mean_i = FXAI_GetArrayValue(ctx_mean_arr, i, 0.0);
   double ctx_std_i = FXAI_GetArrayValue(ctx_std_arr, i, 0.0);
   double ctx_up_i = FXAI_GetArrayValue(ctx_up_arr, i, 0.5);

   ENUM_FXAI_FEATURE_NORMALIZATION norm_method =
      (norm_method_override >= 0 ? FXAI_SanitizeNormMethod(norm_method_override)
                                 : FXAI_GetFeatureNormalizationMethod());
   double feat[FXAI_AI_FEATURES];
   if(!FXAI_ComputeFeatureVector(i,
                                spread_i,
                                time_arr,
                                open_arr,
                                high_arr,
                                low_arr,
                                close_arr,
                                time_m5,
                                close_m5,
                                map_m5,
                                time_m15,
                                close_m15,
                                map_m15,
                                time_m30,
                                close_m30,
                                map_m30,
                                time_h1,
                                close_h1,
                                map_h1,
                                ctx_mean_i,
                                ctx_std_i,
                                ctx_up_i,
                                ctx_extra_arr,
                                norm_method,
                                feat))
      return false;

   bool need_prev = FXAI_FeatureNormNeedsPrevious(norm_method);
   bool has_prev_feat = false;
   double feat_prev[FXAI_AI_FEATURES];
   for(int f=0; f<FXAI_AI_FEATURES; f++)
      feat_prev[f] = 0.0;

   if(need_prev && (i + 1) < ArraySize(close_arr))
   {
      double spread_prev = FXAI_GetSpreadAtIndex(i + 1, spread_m1, spread_i);
      double ctx_mean_prev = FXAI_GetArrayValue(ctx_mean_arr, i + 1, ctx_mean_i);
      double ctx_std_prev = FXAI_GetArrayValue(ctx_std_arr, i + 1, ctx_std_i);
      double ctx_up_prev = FXAI_GetArrayValue(ctx_up_arr, i + 1, ctx_up_i);

      has_prev_feat = FXAI_ComputeFeatureVector(i + 1,
                                               spread_prev,
                                               time_arr,
                                               open_arr,
                                               high_arr,
                                               low_arr,
                                               close_arr,
                                               time_m5,
                                               close_m5,
                                               map_m5,
                                               time_m15,
                                               close_m15,
                                               map_m15,
                                               time_m30,
                                               close_m30,
                                               map_m30,
                                               time_h1,
                                               close_h1,
                                               map_h1,
                                               ctx_mean_prev,
                                               ctx_std_prev,
                                               ctx_up_prev,
                                               ctx_extra_arr,
                                               norm_method,
                                               feat_prev);
   }

   double feat_norm[FXAI_AI_FEATURES];
   sample.sample_time = ((i >= 0 && i < ArraySize(time_arr)) ? time_arr[i] : 0);
   FXAI_ApplyFeatureNormalization(norm_method, feat, feat_prev, has_prev_feat, sample.sample_time, feat_norm);

   if(!has_label)
      return false;

   sample.label_class = label_class;
   sample.horizon_minutes = FXAI_ClampHorizon(H);
   sample.horizon_slot = FXAI_GetHorizonSlot(sample.horizon_minutes);
   sample.move_points = move_points;
   sample.min_move_points = min_move_i;
   sample.cost_points = min_move_i;
   sample.mfe_points = mfe_points;
   sample.mae_points = mae_points;
   sample.time_to_hit_frac = time_to_hit_frac;
   sample.path_flags = path_flags;
   double spread_ref = FXAI_GetIntArrayMean(spread_m1, i, 64, snapshot.spread_points);
   double vol_ref = FXAI_RollingAbsReturn(close_arr, i, 64);
   if(vol_ref < 1e-6) vol_ref = MathAbs(feat[5]);
   sample.regime_id = FXAI_GetStaticRegimeId(sample.sample_time, spread_i, spread_ref, MathAbs(feat[5]), vol_ref);
   double edge = MathMax(MathAbs(move_points) - min_move_i, 0.0);
   double spread_peak = spread_i;
   int spread_n = 0;
   double spread_sum = 0.0;
   int max_step = H;
   if(i - max_step < 0) max_step = i;
   for(int k=0; k<=max_step; k++)
   {
      int idx_sp = i - k;
      if(idx_sp < 0 || idx_sp >= ArraySize(spread_m1)) break;
      double sp = FXAI_GetSpreadAtIndex(idx_sp, spread_m1, spread_i);
      if(sp > spread_peak) spread_peak = sp;
      spread_sum += sp;
      spread_n++;
   }
   double spread_avg = (spread_n > 0 ? spread_sum / (double)spread_n : spread_i);
   double spread_stress = FXAI_Clamp((spread_peak - spread_avg) / MathMax(min_move_i, 0.50), 0.0, 3.0);
   if(spread_stress > 0.35) sample.path_flags |= FXAI_PATHFLAG_SPREAD_STRESS;
   sample.spread_stress = spread_stress;

   double quality = 1.0;
   if(label_class == (int)FXAI_LABEL_SKIP)
   {
      quality = 0.75 - (0.10 * spread_stress);
   }
   else
   {
      double mfe_ratio = mfe_points / MathMax(min_move_i, 0.50);
      double adverse_ratio = mae_points / MathMax(mfe_points, min_move_i);
      double speed_bonus = 1.0 - FXAI_Clamp(time_to_hit_frac, 0.0, 1.0);
      quality = 0.85 +
                0.20 * FXAI_Clamp(mfe_ratio, 0.0, 4.0) +
                0.20 * speed_bonus -
                0.15 * FXAI_Clamp(adverse_ratio, 0.0, 3.0) -
                0.10 * spread_stress;
      if((sample.path_flags & FXAI_PATHFLAG_DUAL_HIT) != 0) quality -= 0.12;
      if((sample.path_flags & FXAI_PATHFLAG_KILLED_EARLY) != 0) quality -= 0.10;
   }
   sample.quality_score = FXAI_Clamp(quality, 0.35, 2.20);

   double dir_bias = (label_class == (int)FXAI_LABEL_SKIP ? 0.85 : 1.20);
   double trade_quality_weight = sample.quality_score;
   sample.sample_weight = FXAI_Clamp(dir_bias *
                                     trade_quality_weight *
                                     (0.75 + edge / MathMax(min_move_i, 0.50)),
                                     0.25,
                                     7.50);
   FXAI_BuildInputVector(feat_norm, sample.x);
   sample.valid = true;
   return true;
}

double FXAI_ScoreNormalizationSetup(const int i_start,
                                    const int i_end,
                                    const int H,
                                    const int target_ai_id,
                                    const double commission_points,
                                    const double cost_buffer_points,
                                    const double ev_threshold_points,
                                    const FXAIDataSnapshot &snapshot,
                                    const int &spread_m1[],
                                    const datetime &time_arr[],
                                    const double &open_arr[],
                                    const double &high_arr[],
                                    const double &low_arr[],
                                    const double &close_arr[],
                                    const datetime &time_m5[],
                                    const double &close_m5[],
                                    const int &map_m5[],
                                    const datetime &time_m15[],
                                    const double &close_m15[],
                                    const int &map_m15[],
                                    const datetime &time_m30[],
                                    const double &close_m30[],
                                    const int &map_m30[],
                                    const datetime &time_h1[],
                                    const double &close_h1[],
                                    const int &map_h1[],
                                    const double &ctx_mean_arr[],
                                    const double &ctx_std_arr[],
                                    const double &ctx_up_arr[],
                                    const double &ctx_extra_arr[],
                                    const FXAIAIHyperParams &hp,
                                    const double buy_thr,
                                    const double sell_thr)
{
   FXAIPreparedSample samples[];
   FXAI_PrecomputeTrainingSamples(i_start,
                                 i_end,
                                 H,
                                 commission_points,
                                 cost_buffer_points,
                                 ev_threshold_points,
                                 snapshot,
                                 spread_m1,
                                 time_arr,
                                 open_arr,
                                 high_arr,
                                 low_arr,
                                 close_arr,
                                 time_m5,
                                 close_m5,
                                 map_m5,
                                 time_m15,
                                 close_m15,
                                 map_m15,
                                 time_m30,
                                 close_m30,
                                 map_m30,
                                 time_h1,
                                 close_h1,
                                 map_h1,
                                 ctx_mean_arr,
                                 ctx_std_arr,
                                 ctx_up_arr,
                                 ctx_extra_arr,
                                 -1,
                                 samples);

   int span = i_end - i_start + 1;
   if(span < 240) return -1e9;

   int val_len = span / 3;
   if(val_len < 80) val_len = 80;
   if(val_len > 220) val_len = 220;
   int val_start = i_start;
   int val_end = val_start + val_len - 1;
   if(val_end >= i_end) val_end = i_end - 1;
   if(val_end <= val_start) return -1e9;

   int purge = H + 240;
   if(purge < H + 40) purge = H + 40;
   int train_start = val_end + purge + 1;
   int train_end = i_end;
   if(train_end - train_start < 100) return -1e9;

   CFXAIAIPlugin *trial = g_plugins.CreateInstance(target_ai_id);
   if(trial == NULL) return -1e9;

   trial.Reset();
   trial.EnsureInitialized(hp);
   for(int epoch=0; epoch<2; epoch++)
   {
      for(int i=train_end; i>=train_start; i--)
      {
         if(i < 0 || i >= ArraySize(samples)) continue;
         if(!samples[i].valid) continue;
         FXAIAISampleV2 s2;
         s2.valid = samples[i].valid;
         s2.label_class = samples[i].label_class;
         s2.regime_id = samples[i].regime_id;
         s2.horizon_minutes = samples[i].horizon_minutes;
         s2.move_points = samples[i].move_points;
         s2.min_move_points = samples[i].min_move_points;
         s2.cost_points = samples[i].cost_points;
         s2.sample_time = samples[i].sample_time;
         for(int k=0; k<FXAI_AI_WEIGHTS; k++)
            s2.x[k] = samples[i].x[k];
         trial.TrainV2(s2, hp);
      }
   }

   int trades = 0;
   double regime_scores[];
   int regime_trades[];
   double score = FXAI_ScoreWarmupTrial(*trial,
                                        hp,
                                        H,
                                        val_start,
                                        val_end,
                                        buy_thr,
                                        sell_thr,
                                        samples,
                                        trades,
                                        regime_scores,
                                        regime_trades);
   delete trial;
   if(trades < 20) score -= 1.5;
   return score;
}

void FXAI_BuildNormScoringModelList(const int primary_ai, int &model_ids[])
{
   ArrayResize(model_ids, 0);

   int p = primary_ai;
   if(p < 0 || p >= FXAI_AI_COUNT) p = (int)AI_SGD_LOGIT;

   int anchors[4];
   anchors[0] = p;
   anchors[1] = (int)AI_FTRL_LOGIT;
   anchors[2] = (int)AI_XGB_FAST;
   anchors[3] = (int)AI_LIGHTGBM;

   int max_models = (AI_Ensemble ? 4 : 1);
   if(max_models < 1) max_models = 1;
   if(max_models > 4) max_models = 4;

   for(int i=0; i<4; i++)
   {
      int id = anchors[i];
      bool exists = false;
      for(int j=0; j<ArraySize(model_ids); j++)
      {
         if(model_ids[j] == id)
         {
            exists = true;
            break;
         }
      }
      if(exists) continue;
      int sz = ArraySize(model_ids);
      ArrayResize(model_ids, sz + 1);
      model_ids[sz] = id;
      if(ArraySize(model_ids) >= max_models) break;
   }
}

double FXAI_ScoreNormalizationSetupMulti(const int i_start,
                                         const int i_end,
                                         const int H,
                                         const int &model_ids[],
                                         const double commission_points,
                                         const double cost_buffer_points,
                                         const double ev_threshold_points,
                                         const FXAIDataSnapshot &snapshot,
                                         const int &spread_m1[],
                                         const datetime &time_arr[],
                                         const double &open_arr[],
                                         const double &high_arr[],
                                         const double &low_arr[],
                                         const double &close_arr[],
                                         const datetime &time_m5[],
                                         const double &close_m5[],
                                         const int &map_m5[],
                                         const datetime &time_m15[],
                                         const double &close_m15[],
                                         const int &map_m15[],
                                         const datetime &time_m30[],
                                         const double &close_m30[],
                                         const int &map_m30[],
                                         const datetime &time_h1[],
                                         const double &close_h1[],
                                         const int &map_h1[],
                                         const double &ctx_mean_arr[],
                                         const double &ctx_std_arr[],
                                         const double &ctx_up_arr[],
                                         const double &ctx_extra_arr[],
                                         const FXAIAIHyperParams &hp,
                                         const double buy_thr,
                                         const double sell_thr)
{
   if(ArraySize(model_ids) <= 0) return -1e9;

   double sum = 0.0;
   int used = 0;
   for(int m=0; m<ArraySize(model_ids); m++)
   {
      int model_id = model_ids[m];
      double s = FXAI_ScoreNormalizationSetup(i_start,
                                              i_end,
                                              H,
                                              model_id,
                                              commission_points,
                                              cost_buffer_points,
                                              ev_threshold_points,
                                              snapshot,
                                              spread_m1,
                                              time_arr,
                                              open_arr,
                                              high_arr,
                                              low_arr,
                                              close_arr,
                                              time_m5,
                                              close_m5,
                                              map_m5,
                                              time_m15,
                                              close_m15,
                                              map_m15,
                                              time_m30,
                                              close_m30,
                                              map_m30,
                                              time_h1,
                                              close_h1,
                                              map_h1,
                                              ctx_mean_arr,
                                              ctx_std_arr,
                                              ctx_up_arr,
                                              ctx_extra_arr,
                                              hp,
                                              buy_thr,
                                              sell_thr);
      if(s <= -1e8) continue;
      sum += s;
      used++;
   }
   if(used <= 0) return -1e9;
   return sum / (double)used;
}

void FXAI_OptimizeNormalizationWindows(const int i_start,
                                       const int i_end,
                                       const int H,
                                       const double commission_points,
                                       const double cost_buffer_points,
                                       const double ev_threshold_points,
                                       const FXAIDataSnapshot &snapshot,
                                       const int &spread_m1[],
                                       const datetime &time_arr[],
                                       const double &open_arr[],
                                       const double &high_arr[],
                                       const double &low_arr[],
                                       const double &close_arr[],
                                       const datetime &time_m5[],
                                       const double &close_m5[],
                                       const int &map_m5[],
                                       const datetime &time_m15[],
                                       const double &close_m15[],
                                       const int &map_m15[],
                                       const datetime &time_m30[],
                                       const double &close_m30[],
                                       const int &map_m30[],
                                       const datetime &time_h1[],
                                       const double &close_h1[],
                                       const int &map_h1[],
                                       const double &ctx_mean_arr[],
                                       const double &ctx_std_arr[],
                                       const double &ctx_up_arr[],
                                       const double &ctx_extra_arr[],
                                       const FXAIAIHyperParams &base_hp,
                                       const double buy_thr,
                                       const double sell_thr)
{
   int default_window = FXAI_GetNormDefaultWindow();
   int w_fast = 96;
   int w_mid = default_window;
   int w_slow = default_window + 64;
   if(w_slow > FXAI_NORM_ROLL_WINDOW_MAX) w_slow = FXAI_NORM_ROLL_WINDOW_MAX;
   int w_regime = 128;

   int windows_tmp[];
   FXAI_BuildNormWindowsFromGroups(w_fast, w_mid, w_slow, w_regime, windows_tmp);
   FXAI_ApplyNormWindows(windows_tmp, default_window);
   int model_ids[];
   FXAI_BuildNormScoringModelList((int)AI_Type, model_ids);

   double best_score = FXAI_ScoreNormalizationSetupMulti(i_start,
                                                         i_end,
                                                         H,
                                                         model_ids,
                                                         commission_points,
                                                         cost_buffer_points,
                                                         ev_threshold_points,
                                                         snapshot,
                                                         spread_m1,
                                                         time_arr,
                                                         open_arr,
                                                         high_arr,
                                                         low_arr,
                                                         close_arr,
                                                         time_m5,
                                                         close_m5,
                                                         map_m5,
                                                         time_m15,
                                                         close_m15,
                                                         map_m15,
                                                         time_m30,
                                                         close_m30,
                                                         map_m30,
                                                         time_h1,
                                                         close_h1,
                                                         map_h1,
                                                         ctx_mean_arr,
                                                         ctx_std_arr,
                                                         ctx_up_arr,
                                                         ctx_extra_arr,
                                                         base_hp,
                                                         buy_thr,
                                                         sell_thr);

   int candidates[6] = {64, 96, 128, 192, 256, 320};
   for(int group=0; group<4; group++)
   {
      int best_w = (group == 0 ? w_fast : (group == 1 ? w_mid : (group == 2 ? w_slow : w_regime)));
      for(int ci=0; ci<6; ci++)
      {
         int trial_w = FXAI_NormalizationWindowClamp(candidates[ci]);
         int tf = w_fast, tm = w_mid, ts = w_slow, tr = w_regime;
         if(group == 0) tf = trial_w;
         else if(group == 1) tm = trial_w;
         else if(group == 2) ts = trial_w;
         else tr = trial_w;

         FXAI_BuildNormWindowsFromGroups(tf, tm, ts, tr, windows_tmp);
         FXAI_ApplyNormWindows(windows_tmp, default_window);
         double score = FXAI_ScoreNormalizationSetupMulti(i_start,
                                                          i_end,
                                                          H,
                                                          model_ids,
                                                          commission_points,
                                                          cost_buffer_points,
                                                          ev_threshold_points,
                                                          snapshot,
                                                          spread_m1,
                                                          time_arr,
                                                          open_arr,
                                                          high_arr,
                                                          low_arr,
                                                          close_arr,
                                                          time_m5,
                                                          close_m5,
                                                          map_m5,
                                                          time_m15,
                                                          close_m15,
                                                          map_m15,
                                                          time_m30,
                                                          close_m30,
                                                          map_m30,
                                                          time_h1,
                                                          close_h1,
                                                          map_h1,
                                                          ctx_mean_arr,
                                                          ctx_std_arr,
                                                          ctx_up_arr,
                                                          ctx_extra_arr,
                                                          base_hp,
                                                          buy_thr,
                                                          sell_thr);
         if(score > best_score)
         {
            best_score = score;
            best_w = trial_w;
         }
      }

      if(group == 0) w_fast = best_w;
      else if(group == 1) w_mid = best_w;
      else if(group == 2) w_slow = best_w;
      else w_regime = best_w;
   }

   FXAI_BuildNormWindowsFromGroups(w_fast, w_mid, w_slow, w_regime, windows_tmp);
   FXAI_ApplyNormWindows(windows_tmp, default_window);
}

double FXAI_ScoreNormMethodCandidate(const int ai_idx,
                                     const int H,
                                     const int warmup_train_epochs,
                                     const int i_start,
                                     const int i_end,
                                     const FXAIAIHyperParams &hp,
                                     const double buy_thr,
                                     const double sell_thr,
                                     const FXAIPreparedSample &samples[],
                                     double &regime_scores[],
                                     int &regime_trades[])
{
   int span = i_end - i_start + 1;
   if(span < 240) return -1e9;

   int val_len = span / 3;
   if(val_len < 80) val_len = 80;
   if(val_len > 240) val_len = 240;
   int val_start = i_start;
   int val_end = val_start + val_len - 1;
   if(val_end >= i_end) val_end = i_end - 1;
   if(val_end <= val_start) return -1e9;

   int purge = H + 240;
   if(purge < H + 40) purge = H + 40;
   int train_start = val_end + purge + 1;
   int train_end = i_end;
   if(train_end - train_start < 100) return -1e9;

   CFXAIAIPlugin *trial = g_plugins.CreateInstance(ai_idx);
   if(trial == NULL) return -1e9;

   trial.Reset();
   trial.EnsureInitialized(hp);
   int train_epochs = warmup_train_epochs;
   if(train_epochs < 1) train_epochs = 1;
   if(train_epochs > 2) train_epochs = 2;
   FXAI_TrainModelWindowPrepared(ai_idx,
                                 *trial,
                                 train_start,
                                 train_end,
                                 train_epochs,
                                 hp,
                                 samples);

   int trades = 0;
   double score = FXAI_ScoreWarmupTrial(*trial,
                                        hp,
                                        H,
                                        val_start,
                                        val_end,
                                        buy_thr,
                                        sell_thr,
                                        samples,
                                        trades,
                                        regime_scores,
                                        regime_trades);
   delete trial;
   return score;
}

void FXAI_StoreNormBank(const int ai_idx,
                        const int regime_id,
                        const int horizon_minutes,
                        const int method_id)
{
   if(ai_idx < 0 || ai_idx >= FXAI_AI_COUNT) return;
   int hslot = FXAI_GetHorizonSlot(horizon_minutes);
   if(hslot < 0 || hslot >= FXAI_MAX_HORIZONS) return;

   int m = (int)FXAI_SanitizeNormMethod(method_id);
   if(regime_id < 0)
   {
      g_model_norm_method_horizon[ai_idx][hslot] = m;
      g_model_norm_horizon_ready[ai_idx][hslot] = true;
   }
   else if(regime_id >= 0 && regime_id < FXAI_REGIME_COUNT)
   {
      g_model_norm_method_bank[ai_idx][regime_id][hslot] = m;
      g_model_norm_bank_ready[ai_idx][regime_id][hslot] = true;
   }
}

void FXAI_WarmupSelectNormBanksForHorizon(const int H,
                                          const bool primary_horizon,
                                          const int warmup_train_epochs,
                                          const int i_start,
                                          const int i_end,
                                          const FXAIAIHyperParams &base_hp,
                                          const double base_buy_thr,
                                          const double base_sell_thr,
                                          FXAINormSampleCache &norm_caches[])
{
   for(int ai_idx=0; ai_idx<FXAI_AI_COUNT; ai_idx++)
   {
      int methods[];
      FXAI_BuildNormMethodCandidateList(ai_idx, methods);
      if(ArraySize(methods) <= 0) continue;

      double best_score = -1e18;
      int best_method = methods[0];
      double best_regime_score[FXAI_REGIME_COUNT];
      int best_regime_method[FXAI_REGIME_COUNT];
      for(int r=0; r<FXAI_REGIME_COUNT; r++)
      {
         best_regime_score[r] = -1e18;
         best_regime_method[r] = best_method;
      }

      for(int m=0; m<ArraySize(methods); m++)
      {
         int method_id = methods[m];
         int cache_idx = FXAI_FindNormSampleCache(method_id, norm_caches);
         if(cache_idx < 0) continue;

         double regime_scores[];
         int regime_trades[];
         double score = FXAI_ScoreNormMethodCandidate(ai_idx,
                                                      H,
                                                      warmup_train_epochs,
                                                      i_start,
                                                      i_end,
                                                      base_hp,
                                                      base_buy_thr,
                                                      base_sell_thr,
                                                      norm_caches[cache_idx].samples,
                                                      regime_scores,
                                                      regime_trades);
         if(score > best_score)
         {
            best_score = score;
            best_method = method_id;
         }

         for(int r=0; r<FXAI_REGIME_COUNT; r++)
         {
            if(r >= ArraySize(regime_scores) || r >= ArraySize(regime_trades)) continue;
            if(regime_trades[r] < 12) continue;
            if(regime_scores[r] > best_regime_score[r])
            {
               best_regime_score[r] = regime_scores[r];
               best_regime_method[r] = method_id;
            }
         }
      }

      FXAI_StoreNormBank(ai_idx, -1, H, best_method);
      if(primary_horizon)
      {
         g_model_norm_method[ai_idx] = best_method;
         g_model_norm_ready[ai_idx] = true;
      }
      for(int r=0; r<FXAI_REGIME_COUNT; r++)
      {
         int method_r = (best_regime_score[r] > -1e17 ? best_regime_method[r] : best_method);
         FXAI_StoreNormBank(ai_idx, r, H, method_r);
      }
   }
}

void FXAI_WarmupTrainHorizonPolicyForSamples(const int H,
                                             const int base_h,
                                             const int ev_lookback,
                                             const FXAIDataSnapshot &snapshot,
                                             const double &close_arr[],
                                             const int ai_hint,
                                             const int i_start,
                                             const int i_end,
                                             const FXAIPreparedSample &samples[])
{
   int n = ArraySize(samples);
   if(n <= 0) return;

   int start = i_start;
   int end = i_end;
   if(start < 0) start = 0;
   if(end >= n) end = n - 1;
   if(end < start) return;

   for(int i=end; i>=start; i--)
   {
      if(i < 0 || i >= ArraySize(samples)) continue;
      if(!samples[i].valid) continue;

      double exp_abs = FXAI_EstimateExpectedAbsMovePointsAtIndex(close_arr,
                                                                 i,
                                                                 H,
                                                                 ev_lookback,
                                                                 snapshot.point);
      if(exp_abs <= 0.0)
         exp_abs = MathMax(samples[i].min_move_points, MathAbs(samples[i].move_points));
      double current_vol = FXAI_RollingAbsReturn(close_arr, i, 20);
      if(current_vol < 1e-6) current_vol = MathAbs(samples[i].move_points) * snapshot.point;
      double feat[FXAI_HPOL_FEATS];
      FXAIDataSnapshot snap_i = snapshot;
      snap_i.bar_time = samples[i].sample_time;
      FXAI_BuildHorizonPolicyFeatures(H,
                                      base_h,
                                      exp_abs,
                                      samples[i].min_move_points,
                                      snap_i,
                                      current_vol,
                                      samples[i].regime_id,
                                      ai_hint,
                                      feat);

      double edge = MathMax(MathAbs(samples[i].move_points) - samples[i].min_move_points, 0.0);
      double reward = (samples[i].label_class == (int)FXAI_LABEL_SKIP ? -0.25 :
                       samples[i].quality_score * edge / MathMax(samples[i].min_move_points, 0.50));
      reward = FXAI_Clamp(reward, -2.0, 6.0);
      FXAI_UpdateHorizonPolicy(samples[i].regime_id, feat, reward);
   }
}

struct FXAIWarmupBucketStats
{
   int trades;
   int wins;
   double net_sum;
   double gross_pos;
   double gross_neg;
   double eq;
   double eq_peak;
   double max_dd;
};

void FXAI_ResetWarmupBucketStats(FXAIWarmupBucketStats &stats)
{
   stats.trades = 0;
   stats.wins = 0;
   stats.net_sum = 0.0;
   stats.gross_pos = 0.0;
   stats.gross_neg = 0.0;
   stats.eq = 0.0;
   stats.eq_peak = 0.0;
   stats.max_dd = 0.0;
}

void FXAI_UpdateWarmupBucketStats(FXAIWarmupBucketStats &stats, const double net_pts)
{
   stats.net_sum += net_pts;
   if(net_pts >= 0.0) stats.gross_pos += net_pts;
   else               stats.gross_neg += -net_pts;
   stats.eq += net_pts;
   if(stats.eq > stats.eq_peak) stats.eq_peak = stats.eq;
   double dd = stats.eq_peak - stats.eq;
   if(dd > stats.max_dd) stats.max_dd = dd;
   stats.trades++;
   if(net_pts > 0.0) stats.wins++;
}

double FXAI_ScoreWarmupBucketStats(const FXAIWarmupBucketStats &stats)
{
   if(stats.trades <= 0) return -1e9;
   double win_rate = (double)stats.wins / (double)stats.trades;
   double avg_net = stats.net_sum / (double)stats.trades;
   double profit_factor = stats.gross_pos / MathMax(stats.gross_neg, 1e-6);
   if(profit_factor > 8.0) profit_factor = 8.0;

   double dd_penalty = 0.0;
   if(stats.gross_pos > 0.0) dd_penalty = stats.max_dd / stats.gross_pos;
   else if(stats.max_dd > 0.0) dd_penalty = 2.0;

   return (avg_net * 5.0) + (win_rate * 1.75) + (0.80 * profit_factor) - (1.50 * dd_penalty);
}

double FXAI_ScoreWarmupTrial(CFXAIAIPlugin &plugin,
                            const FXAIAIHyperParams &hp,
                            const int horizon_minutes,
                            const int val_start,
                            const int val_end,
                            const double buyThr,
                            const double sellThr,
                            const FXAIPreparedSample &samples[],
                            int &trades_out,
                            double &regime_scores[],
                            int &regime_trades[])
{
   trades_out = 0;
   int n = ArraySize(samples);
   if(n <= 0 || val_end < val_start) return -1e9;

   if(ArraySize(regime_scores) != FXAI_REGIME_COUNT) ArrayResize(regime_scores, FXAI_REGIME_COUNT);
   if(ArraySize(regime_trades) != FXAI_REGIME_COUNT) ArrayResize(regime_trades, FXAI_REGIME_COUNT);
   FXAIWarmupBucketStats regime_stats[FXAI_REGIME_COUNT];
   for(int r=0; r<FXAI_REGIME_COUNT; r++)
   {
      regime_scores[r] = -1e9;
      regime_trades[r] = 0;
      FXAI_ResetWarmupBucketStats(regime_stats[r]);
   }

   int start = val_start;
   int end = val_end;
   if(start < 0) start = 0;
   if(end >= n) end = n - 1;
   if(end < start) return -1e9;

   FXAIWarmupBucketStats total_stats;
   FXAI_ResetWarmupBucketStats(total_stats);
   double fallback_move_ema = 0.0;
   bool fallback_move_ready = false;

   int score_h = FXAI_ClampHorizon(horizon_minutes);
   for(int i=end; i>=start; i--)
   {
      if(!samples[i].valid) continue;

      FXAIAIPredictV2 req;
      req.regime_id = samples[i].regime_id;
      req.horizon_minutes = score_h;
      req.min_move_points = samples[i].min_move_points;
      req.cost_points = samples[i].cost_points;
      req.sample_time = samples[i].sample_time;
      for(int k=0; k<FXAI_AI_WEIGHTS; k++)
         req.x[k] = samples[i].x[k];

      FXAIAIPredictionV2 pred;
      plugin.PredictV2(req, hp, pred);

      double probs_eval[3];
      probs_eval[(int)FXAI_LABEL_SELL] = pred.class_probs[(int)FXAI_LABEL_SELL];
      probs_eval[(int)FXAI_LABEL_BUY] = pred.class_probs[(int)FXAI_LABEL_BUY];
      probs_eval[(int)FXAI_LABEL_SKIP] = pred.class_probs[(int)FXAI_LABEL_SKIP];

      double expected_move = pred.expected_move_points;
      if(expected_move <= 0.0 && fallback_move_ready)
         expected_move = MathMax(fallback_move_ema, samples[i].min_move_points);
      if(expected_move <= 0.0) expected_move = samples[i].min_move_points;
      if(expected_move <= 0.0) expected_move = 0.10;

      double buyMinProb = buyThr;
      double sellMinProb = 1.0 - sellThr;
      double skipMinProb = 0.55;
      double vol_proxy = 0.0;
      if(FXAI_AI_WEIGHTS > 6) vol_proxy = MathAbs(samples[i].x[6]);
      FXAI_DeriveAdaptiveThresholds(buyThr,
                                   sellThr,
                                   samples[i].min_move_points,
                                   expected_move,
                                   vol_proxy,
                                   buyMinProb,
                                   sellMinProb,
                                   skipMinProb);

      int signal = FXAI_ClassSignalFromEV(probs_eval,
                                         buyMinProb,
                                         sellMinProb,
                                         skipMinProb,
                                         expected_move,
                                         samples[i].min_move_points,
                                         FXAI_Clamp(AI_EVThresholdPoints, 0.0, 100.0));
      if(signal == -1) continue;

      double net_pts = FXAI_RealizedNetPointsForSignal(signal,
                                                       samples[i].move_points,
                                                       samples[i].min_move_points,
                                                       score_h);
      FXAI_UpdateWarmupBucketStats(total_stats, net_pts);
      int regime_id = samples[i].regime_id;
      if(regime_id < 0 || regime_id >= FXAI_REGIME_COUNT) regime_id = 0;
      FXAI_UpdateWarmupBucketStats(regime_stats[regime_id], net_pts);
      FXAI_UpdateMoveEMA(fallback_move_ema, fallback_move_ready, samples[i].move_points, 0.08);
   }

   if(total_stats.trades <= 0) return -1e9;
   trades_out = total_stats.trades;
   for(int r=0; r<FXAI_REGIME_COUNT; r++)
   {
      regime_trades[r] = regime_stats[r].trades;
      if(regime_stats[r].trades > 0)
         regime_scores[r] = FXAI_ScoreWarmupBucketStats(regime_stats[r]);
   }
   return FXAI_ScoreWarmupBucketStats(total_stats);
}

double FXAI_ScoreWarmupTrialRouted(const int ai_idx,
                                   CFXAIAIPlugin &plugin,
                                   const FXAIAIHyperParams &hp,
                                   const int horizon_minutes,
                                   const int val_start,
                                   const int val_end,
                                   const double buyThr,
                                   const double sellThr,
                                   const FXAIPreparedSample &samples[],
                                   FXAINormSampleCache &caches[],
                                   int &trades_out,
                                   double &regime_scores[],
                                   int &regime_trades[])
{
   trades_out = 0;
   int n = ArraySize(samples);
   if(n <= 0 || val_end < val_start) return -1e9;

   if(ArraySize(regime_scores) != FXAI_REGIME_COUNT) ArrayResize(regime_scores, FXAI_REGIME_COUNT);
   if(ArraySize(regime_trades) != FXAI_REGIME_COUNT) ArrayResize(regime_trades, FXAI_REGIME_COUNT);
   FXAIWarmupBucketStats regime_stats[FXAI_REGIME_COUNT];
   for(int r=0; r<FXAI_REGIME_COUNT; r++)
   {
      regime_scores[r] = -1e9;
      regime_trades[r] = 0;
      FXAI_ResetWarmupBucketStats(regime_stats[r]);
   }

   int start = val_start;
   int end = val_end;
   if(start < 0) start = 0;
   if(end >= n) end = n - 1;
   if(end < start) return -1e9;

   FXAIWarmupBucketStats total_stats;
   FXAI_ResetWarmupBucketStats(total_stats);
   double fallback_move_ema = 0.0;
   bool fallback_move_ready = false;

   int score_h = FXAI_ClampHorizon(horizon_minutes);
   for(int i=end; i>=start; i--)
   {
      if(i < 0 || i >= n) continue;
      if(!samples[i].valid) continue;

      FXAIPreparedSample eval_sample;
      FXAI_GetCachedPreparedSample(ai_idx, samples[i], i, caches, eval_sample);
      if(!eval_sample.valid) continue;

      FXAIAIPredictV2 req;
      req.regime_id = eval_sample.regime_id;
      req.horizon_minutes = score_h;
      req.min_move_points = eval_sample.min_move_points;
      req.cost_points = eval_sample.cost_points;
      req.sample_time = eval_sample.sample_time;
      for(int k=0; k<FXAI_AI_WEIGHTS; k++)
         req.x[k] = eval_sample.x[k];

      FXAIAIPredictionV2 pred;
      plugin.PredictV2(req, hp, pred);

      double probs_eval[3];
      probs_eval[(int)FXAI_LABEL_SELL] = pred.class_probs[(int)FXAI_LABEL_SELL];
      probs_eval[(int)FXAI_LABEL_BUY] = pred.class_probs[(int)FXAI_LABEL_BUY];
      probs_eval[(int)FXAI_LABEL_SKIP] = pred.class_probs[(int)FXAI_LABEL_SKIP];

      double expected_move = pred.expected_move_points;
      if(expected_move <= 0.0 && fallback_move_ready)
         expected_move = MathMax(fallback_move_ema, eval_sample.min_move_points);
      if(expected_move <= 0.0) expected_move = eval_sample.min_move_points;
      if(expected_move <= 0.0) expected_move = 0.10;

      double buyMinProb = buyThr;
      double sellMinProb = 1.0 - sellThr;
      double skipMinProb = 0.55;
      double vol_proxy = 0.0;
      if(FXAI_AI_WEIGHTS > 6) vol_proxy = MathAbs(eval_sample.x[6]);
      FXAI_DeriveAdaptiveThresholds(buyThr,
                                    sellThr,
                                    eval_sample.min_move_points,
                                    expected_move,
                                    vol_proxy,
                                    buyMinProb,
                                    sellMinProb,
                                    skipMinProb);

      int signal = FXAI_ClassSignalFromEV(probs_eval,
                                          buyMinProb,
                                          sellMinProb,
                                          skipMinProb,
                                          expected_move,
                                          eval_sample.min_move_points,
                                          FXAI_Clamp(AI_EVThresholdPoints, 0.0, 100.0));
      if(signal == -1) continue;

      double net_pts = FXAI_RealizedNetPointsForSignal(signal,
                                                       eval_sample.move_points,
                                                       eval_sample.min_move_points,
                                                       score_h);
      FXAI_UpdateWarmupBucketStats(total_stats, net_pts);
      int regime_id = eval_sample.regime_id;
      if(regime_id < 0 || regime_id >= FXAI_REGIME_COUNT) regime_id = 0;
      FXAI_UpdateWarmupBucketStats(regime_stats[regime_id], net_pts);
      FXAI_UpdateMoveEMA(fallback_move_ema, fallback_move_ready, eval_sample.move_points, 0.08);
   }

   if(total_stats.trades <= 0) return -1e9;
   trades_out = total_stats.trades;
   for(int r=0; r<FXAI_REGIME_COUNT; r++)
   {
      regime_trades[r] = regime_stats[r].trades;
      if(regime_stats[r].trades > 0)
         regime_scores[r] = FXAI_ScoreWarmupBucketStats(regime_stats[r]);
   }
   return FXAI_ScoreWarmupBucketStats(total_stats);
}

void FXAI_StoreWarmupBank(const int ai_idx,
                          const int regime_id,
                          const int horizon_minutes,
                          const FXAIAIHyperParams &hp,
                          const double buy_thr,
                          const double sell_thr)
{
   if(ai_idx < 0 || ai_idx >= FXAI_AI_COUNT) return;
   int hslot = FXAI_GetHorizonSlot(horizon_minutes);
   if(hslot < 0 || hslot >= FXAI_MAX_HORIZONS) return;

   if(regime_id < 0)
   {
      g_model_hp_horizon[ai_idx][hslot] = hp;
      g_model_hp_horizon_ready[ai_idx][hslot] = true;
      g_model_buy_thr_horizon[ai_idx][hslot] = buy_thr;
      g_model_sell_thr_horizon[ai_idx][hslot] = sell_thr;
      FXAI_SanitizeThresholdPair(g_model_buy_thr_horizon[ai_idx][hslot],
                                 g_model_sell_thr_horizon[ai_idx][hslot]);
      g_model_thr_horizon_ready[ai_idx][hslot] = true;
   }
   else if(regime_id >= 0 && regime_id < FXAI_REGIME_COUNT)
   {
      g_model_hp_bank[ai_idx][regime_id][hslot] = hp;
      g_model_hp_bank_ready[ai_idx][regime_id][hslot] = true;
      g_model_buy_thr_bank[ai_idx][regime_id][hslot] = buy_thr;
      g_model_sell_thr_bank[ai_idx][regime_id][hslot] = sell_thr;
      FXAI_SanitizeThresholdPair(g_model_buy_thr_bank[ai_idx][regime_id][hslot],
                                 g_model_sell_thr_bank[ai_idx][regime_id][hslot]);
      g_model_thr_bank_ready[ai_idx][regime_id][hslot] = true;
   }
}

void FXAI_WarmupSelectBanksForHorizon(const int H,
                                      const bool primary_horizon,
                                      const int warmup_loops,
                                      const int warmup_folds,
                                      const int warmup_train_epochs,
                                      const int warmup_min_trades,
                                      const int seed,
                                      const datetime bar_time,
                                      const FXAIAIHyperParams &base_hp,
                                      const double base_buy_thr,
                                      const double base_sell_thr,
                                      const int i_start,
                                      const int i_end,
                                      const FXAIDataSnapshot &snapshot,
                                      const int &spread_m1[],
                                      const datetime &time_arr[],
                                      const double &open_arr[],
                                      const double &high_arr[],
                                      const double &low_arr[],
                                      const double &close_arr[],
                                      const datetime &time_m5[],
                                      const double &close_m5[],
                                      const int &map_m5[],
                                      const datetime &time_m15[],
                                      const double &close_m15[],
                                      const int &map_m15[],
                                      const datetime &time_m30[],
                                      const double &close_m30[],
                                      const int &map_m30[],
                                      const datetime &time_h1[],
                                      const double &close_h1[],
                                      const int &map_h1[],
                                      const double &ctx_mean_arr[],
                                      const double &ctx_std_arr[],
                                      const double &ctx_up_arr[],
                                      const double &ctx_extra_arr[],
                                      const FXAIPreparedSample &samples[],
                                      FXAINormSampleCache &norm_caches[])
{
   int sample_span = i_end - i_start + 1;
   int fold_len = sample_span / (warmup_folds + 1);
   if(fold_len < 40) fold_len = 40;
   if(fold_len > (sample_span / 2)) fold_len = sample_span / 2;
   if(fold_len < 20) return;

   for(int ai_idx=0; ai_idx<FXAI_AI_COUNT; ai_idx++)
   {
      FXAI_EnsureRoutedNormCachesForSamples(ai_idx,
                                            i_start,
                                            i_end,
                                            H,
                                            snapshot.commission_points,
                                            (AI_CostBufferPoints < 0.0 ? 0.0 : AI_CostBufferPoints),
                                            FXAI_Clamp(AI_EVThresholdPoints, 0.0, 100.0),
                                            snapshot,
                                            spread_m1,
                                            time_arr,
                                            open_arr,
                                            high_arr,
                                            low_arr,
                                            close_arr,
                                            time_m5,
                                            close_m5,
                                            map_m5,
                                            time_m15,
                                            close_m15,
                                            map_m15,
                                            time_m30,
                                            close_m30,
                                            map_m30,
                                            time_h1,
                                            close_h1,
                                            map_h1,
                                            ctx_mean_arr,
                                            ctx_std_arr,
                                            ctx_up_arr,
                                            ctx_extra_arr,
                                            samples,
                                            norm_caches);

      MathSrand((uint)(seed + (ai_idx + 1) * 104729 + (int)(bar_time % 65521) + H * 97));

      CFXAIAIPlugin *fold_pool[];
      ArrayResize(fold_pool, warmup_folds);
      for(int f=0; f<warmup_folds; f++)
         fold_pool[f] = g_plugins.CreateInstance(ai_idx);

      double best_score = -1e18;
      FXAIAIHyperParams best_hp = base_hp;
      double best_buy_thr = base_buy_thr;
      double best_sell_thr = base_sell_thr;

      double best_regime_score[FXAI_REGIME_COUNT];
      FXAIAIHyperParams best_regime_hp[FXAI_REGIME_COUNT];
      double best_regime_buy[FXAI_REGIME_COUNT];
      double best_regime_sell[FXAI_REGIME_COUNT];
      for(int r=0; r<FXAI_REGIME_COUNT; r++)
      {
         best_regime_score[r] = -1e18;
         best_regime_hp[r] = base_hp;
         best_regime_buy[r] = base_buy_thr;
         best_regime_sell[r] = base_sell_thr;
      }

      for(int loop=0; loop<warmup_loops; loop++)
      {
         FXAIAIHyperParams hp_trial;
         double buy_trial = base_buy_thr;
         double sell_trial = base_sell_thr;
         if(loop == 0)
            hp_trial = base_hp;
         else
         {
            FXAI_SampleModelHyperParams(ai_idx, base_hp, hp_trial);
            FXAI_SampleThresholdPair(base_buy_thr, base_sell_thr, buy_trial, sell_trial);
         }

         double score_sum = 0.0;
         int folds_used = 0;
         int trades_total = 0;
         double regime_score_sum[FXAI_REGIME_COUNT];
         int regime_score_used[FXAI_REGIME_COUNT];
         int regime_trade_total[FXAI_REGIME_COUNT];
         for(int r=0; r<FXAI_REGIME_COUNT; r++)
         {
            regime_score_sum[r] = 0.0;
            regime_score_used[r] = 0;
            regime_trade_total[r] = 0;
         }

         for(int f=0; f<warmup_folds; f++)
         {
            int val_start = i_start + (f * fold_len);
            int val_end = val_start + fold_len - 1;
            if(val_start < i_start) val_start = i_start;
            if(val_end >= i_end) val_end = i_end - 1;
            if(val_end <= val_start) continue;

            int purge = H + 240;
            if(purge < H + 40) purge = H + 40;
            int train_start = val_end + purge + 1;
            int train_end = i_end;
            if(train_end - train_start < 100) continue;

            CFXAIAIPlugin *trial = fold_pool[f];
            if(trial == NULL) continue;
            trial.Reset();
            trial.EnsureInitialized(hp_trial);

            for(int epoch=0; epoch<warmup_train_epochs; epoch++)
            {
               FXAI_TrainModelWindowPreparedRoutedCached(ai_idx,
                                                         *trial,
                                                         train_start,
                                                         train_end,
                                                         1,
                                                         samples,
                                                         norm_caches);
            }

            int trades_fold = 0;
            double regime_scores_fold[];
            int regime_trades_fold[];
            double score_fold = FXAI_ScoreWarmupTrialRouted(ai_idx,
                                                            *trial,
                                                            hp_trial,
                                                            H,
                                                            val_start,
                                                            val_end,
                                                            buy_trial,
                                                            sell_trial,
                                                            samples,
                                                            norm_caches,
                                                            trades_fold,
                                                            regime_scores_fold,
                                                            regime_trades_fold);

            if(score_fold <= -1e8 || trades_fold <= 0) continue;
            score_sum += score_fold;
            trades_total += trades_fold;
            folds_used++;
            for(int r=0; r<FXAI_REGIME_COUNT; r++)
            {
               if(r >= ArraySize(regime_scores_fold) || r >= ArraySize(regime_trades_fold)) continue;
               if(regime_trades_fold[r] <= 0 || regime_scores_fold[r] <= -1e8) continue;
               regime_score_sum[r] += regime_scores_fold[r];
               regime_score_used[r]++;
               regime_trade_total[r] += regime_trades_fold[r];
            }
         }

         if(folds_used <= 0) continue;

         double score = score_sum / (double)folds_used;
         if(trades_total < warmup_min_trades)
         {
            double miss = (double)(warmup_min_trades - trades_total) / (double)warmup_min_trades;
            score -= 1.5 * miss;
         }

         if(score > best_score)
         {
            best_score = score;
            best_hp = hp_trial;
            best_buy_thr = buy_trial;
            best_sell_thr = sell_trial;
         }

         for(int r=0; r<FXAI_REGIME_COUNT; r++)
         {
            if(regime_score_used[r] <= 0) continue;
            int min_regime_trades = warmup_min_trades / 8;
            if(min_regime_trades < 12) min_regime_trades = 12;
            if(regime_trade_total[r] < min_regime_trades) continue;
            double regime_score = regime_score_sum[r] / (double)regime_score_used[r];
            if(regime_score > best_regime_score[r])
            {
               best_regime_score[r] = regime_score;
               best_regime_hp[r] = hp_trial;
               best_regime_buy[r] = buy_trial;
               best_regime_sell[r] = sell_trial;
            }
         }
      }

      for(int f=0; f<warmup_folds; f++)
      {
         if(fold_pool[f] != NULL)
         {
            delete fold_pool[f];
            fold_pool[f] = NULL;
         }
      }

      FXAI_StoreWarmupBank(ai_idx, -1, H, best_hp, best_buy_thr, best_sell_thr);
      for(int r=0; r<FXAI_REGIME_COUNT; r++)
      {
         FXAIAIHyperParams hp_reg = (best_regime_score[r] > -1e17 ? best_regime_hp[r] : best_hp);
         double buy_reg = (best_regime_score[r] > -1e17 ? best_regime_buy[r] : best_buy_thr);
         double sell_reg = (best_regime_score[r] > -1e17 ? best_regime_sell[r] : best_sell_thr);
         FXAI_StoreWarmupBank(ai_idx, r, H, hp_reg, buy_reg, sell_reg);
         if(primary_horizon)
         {
            g_model_buy_thr_regime[ai_idx][r] = buy_reg;
            g_model_sell_thr_regime[ai_idx][r] = sell_reg;
            FXAI_SanitizeThresholdPair(g_model_buy_thr_regime[ai_idx][r],
                                       g_model_sell_thr_regime[ai_idx][r]);
            g_model_thr_regime_ready[ai_idx][r] = true;
         }
      }

      if(primary_horizon)
      {
         g_model_hp[ai_idx] = best_hp;
         g_model_hp_ready[ai_idx] = true;
         g_model_buy_thr[ai_idx] = best_buy_thr;
         g_model_sell_thr[ai_idx] = best_sell_thr;
         FXAI_SanitizeThresholdPair(g_model_buy_thr[ai_idx], g_model_sell_thr[ai_idx]);
         g_model_thr_ready[ai_idx] = true;
      }
   }
}

void FXAI_WarmupPretrainMetaForSamples(const int H,
                                       const int warmup_folds,
                                       const int warmup_train_epochs,
                                       const int i_start,
                                       const int i_end,
                                       const double base_buy_thr,
                                       const double base_sell_thr,
                                       const FXAIPreparedSample &samples[],
                                       FXAINormSampleCache &norm_caches[])
{
   int sample_span = i_end - i_start + 1;
   int fold_len = sample_span / (warmup_folds + 1);
   if(fold_len < 40) fold_len = 40;
   if(fold_len > (sample_span / 2)) fold_len = sample_span / 2;
   if(fold_len < 20) return;

   int warm_epochs = warmup_train_epochs;
   if(warm_epochs < 1) warm_epochs = 1;
   if(warm_epochs > 3) warm_epochs = 3;

   for(int f=0; f<warmup_folds; f++)
   {
      int val_start = i_start + (f * fold_len);
      int val_end = val_start + fold_len - 1;
      if(val_start < i_start) val_start = i_start;
      if(val_end >= i_end) val_end = i_end - 1;
      if(val_end <= val_start) continue;

      int purge = H + 240;
      if(purge < H + 40) purge = H + 40;
      int train_start = val_end + purge + 1;
      int train_end = i_end;
      if(train_end - train_start < 100) continue;

      CFXAIAIPlugin *pool[];
      ArrayResize(pool, FXAI_AI_COUNT);
      for(int ai_idx=0; ai_idx<FXAI_AI_COUNT; ai_idx++)
      {
         pool[ai_idx] = g_plugins.CreateInstance(ai_idx);
         if(pool[ai_idx] == NULL) continue;
         FXAIAIHyperParams hp_init;
         FXAI_GetModelHyperParamsRouted(ai_idx, 0, H, hp_init);
         pool[ai_idx].Reset();
         pool[ai_idx].EnsureInitialized(hp_init);
         FXAI_TrainModelWindowPreparedRoutedCached(ai_idx,
                                                   *pool[ai_idx],
                                                   train_start,
                                                   train_end,
                                                   warm_epochs,
                                                   samples,
                                                   norm_caches);
      }

      double fallback_move_ema = 0.0;
      bool fallback_move_ready = false;
      for(int i=val_end; i>=val_start; i--)
      {
         if(i < 0 || i >= ArraySize(samples)) continue;
         if(!samples[i].valid) continue;

         int regime_id = samples[i].regime_id;
         if(regime_id < 0 || regime_id >= FXAI_REGIME_COUNT) regime_id = 0;
         double ensemble_buy_ev_sum = 0.0;
         double ensemble_sell_ev_sum = 0.0;
         double ensemble_buy_support = 0.0;
         double ensemble_sell_support = 0.0;
         double ensemble_skip_support = 0.0;
         double ensemble_meta_total = 0.0;
         double ensemble_expected_sum = 0.0;

         for(int ai_idx=0; ai_idx<FXAI_AI_COUNT; ai_idx++)
         {
            CFXAIAIPlugin *plugin = pool[ai_idx];
            if(plugin == NULL) continue;

            FXAIAIHyperParams hp_model;
            FXAI_GetModelHyperParamsRouted(ai_idx, regime_id, H, hp_model);

            FXAIAIPredictV2 req;
            FXAIPreparedSample pred_sample;
            FXAI_GetCachedPreparedSample(ai_idx, samples[i], i, norm_caches, pred_sample);
            req.regime_id = pred_sample.regime_id;
            req.horizon_minutes = pred_sample.horizon_minutes;
            req.min_move_points = pred_sample.min_move_points;
            req.cost_points = pred_sample.cost_points;
            req.sample_time = pred_sample.sample_time;
            for(int k=0; k<FXAI_AI_WEIGHTS; k++)
               req.x[k] = pred_sample.x[k];

            FXAIAIPredictionV2 pred;
            plugin.PredictV2(req, hp_model, pred);

            double probs_eval[3];
            probs_eval[0] = pred.class_probs[0];
            probs_eval[1] = pred.class_probs[1];
            probs_eval[2] = pred.class_probs[2];
            FXAI_ApplyRegimeCalibration(ai_idx, regime_id, probs_eval);

            double expected_move = pred.expected_move_points;
            if(expected_move <= 0.0 && fallback_move_ready)
               expected_move = MathMax(fallback_move_ema, samples[i].min_move_points);
            if(expected_move <= 0.0) expected_move = samples[i].min_move_points;
            if(expected_move <= 0.0) expected_move = 0.10;

            double modelBuyThr = base_buy_thr;
            double modelSellThr = base_sell_thr;
            FXAI_GetModelThresholds(ai_idx, regime_id, H, base_buy_thr, base_sell_thr, modelBuyThr, modelSellThr);

            double buyMinProb = modelBuyThr;
            double sellMinProb = 1.0 - modelSellThr;
            double skipMinProb = 0.55;
            double vol_proxy = 0.0;
            if(FXAI_AI_WEIGHTS > 6) vol_proxy = MathAbs(pred_sample.x[6]);
            FXAI_DeriveAdaptiveThresholds(modelBuyThr,
                                         modelSellThr,
                                         samples[i].min_move_points,
                                         expected_move,
                                         vol_proxy,
                                         buyMinProb,
                                         sellMinProb,
                                         skipMinProb);

            int signal = FXAI_ClassSignalFromEV(probs_eval,
                                               buyMinProb,
                                               sellMinProb,
                                               skipMinProb,
                                               expected_move,
                                               samples[i].min_move_points,
                                               FXAI_Clamp(AI_EVThresholdPoints, 0.0, 100.0));

            FXAI_UpdateModelReliability(ai_idx,
                                        samples[i].label_class,
                                        signal,
                                        samples[i].move_points,
                                        samples[i].min_move_points,
                                        expected_move,
                                        probs_eval);
            FXAI_UpdateRegimeCalibration(ai_idx, regime_id, samples[i].label_class, probs_eval);
            FXAI_UpdateModelPerformance(ai_idx,
                                        regime_id,
                                        samples[i].label_class,
                                        signal,
                                        samples[i].move_points,
                                        samples[i].min_move_points,
                                        H,
                                        expected_move,
                                        probs_eval);

            double meta_w = FXAI_GetModelMetaScore(ai_idx, regime_id, samples[i].min_move_points);
            if(meta_w <= 0.0) meta_w = 1.0;
            double model_buy_ev = ((2.0 * probs_eval[(int)FXAI_LABEL_BUY]) - 1.0) * expected_move - samples[i].min_move_points;
            double model_sell_ev = ((2.0 * probs_eval[(int)FXAI_LABEL_SELL]) - 1.0) * expected_move - samples[i].min_move_points;
            model_buy_ev = FXAI_Clamp(model_buy_ev, -10.0 * samples[i].min_move_points, 10.0 * samples[i].min_move_points);
            model_sell_ev = FXAI_Clamp(model_sell_ev, -10.0 * samples[i].min_move_points, 10.0 * samples[i].min_move_points);

            ensemble_meta_total += meta_w;
            ensemble_buy_ev_sum += meta_w * model_buy_ev;
            ensemble_sell_ev_sum += meta_w * model_sell_ev;
            ensemble_expected_sum += meta_w * expected_move;
            if(signal == 1) ensemble_buy_support += meta_w;
            else if(signal == 0) ensemble_sell_support += meta_w;
            else ensemble_skip_support += meta_w;
         }

         FXAI_UpdateMoveEMA(fallback_move_ema, fallback_move_ready, samples[i].move_points, 0.08);

         if(ensemble_meta_total > 0.0)
         {
            double buyPct = 100.0 * (ensemble_buy_support / ensemble_meta_total);
            double sellPct = 100.0 * (ensemble_sell_support / ensemble_meta_total);
            double skipPct = 100.0 * (ensemble_skip_support / ensemble_meta_total);
            double avg_buy_ev = ensemble_buy_ev_sum / ensemble_meta_total;
            double avg_sell_ev = ensemble_sell_ev_sum / ensemble_meta_total;
            double avg_expected = ensemble_expected_sum / ensemble_meta_total;
            double feat[FXAI_STACK_FEATS];
            FXAI_StackBuildFeatures(buyPct,
                                    sellPct,
                                    skipPct,
                                    avg_buy_ev,
                                    avg_sell_ev,
                                    samples[i].min_move_points,
                                    avg_expected,
                                    (FXAI_AI_WEIGHTS > 6 ? MathAbs(samples[i].x[6]) : 0.0),
                                    H,
                                    feat);
            FXAI_StackUpdate(regime_id, samples[i].label_class, feat, samples[i].sample_weight);
         }
      }

      for(int ai_idx=0; ai_idx<FXAI_AI_COUNT; ai_idx++)
      {
         if(pool[ai_idx] != NULL)
         {
            delete pool[ai_idx];
            pool[ai_idx] = NULL;
         }
      }
   }
}

bool FXAI_WarmupTrainAndTune(const string symbol)
{
   const int FEATURE_LB = 10;

   int warmup_samples = AI_WarmupSamples;
   if(warmup_samples < 2000) warmup_samples = 2000;
   if(warmup_samples > 50000) warmup_samples = 50000;

   int warmup_loops = AI_WarmupLoops;
   if(warmup_loops < 10) warmup_loops = 10;
   if(warmup_loops > 500) warmup_loops = 500;

   int warmup_train_epochs = AI_Epochs;
   if(warmup_train_epochs < 1) warmup_train_epochs = 1;
   if(warmup_train_epochs > 5) warmup_train_epochs = 5;

   int warmup_folds = AI_WarmupFolds;
   if(warmup_folds < 2) warmup_folds = 2;
   if(warmup_folds > 5) warmup_folds = 5;

   int warmup_min_trades = AI_WarmupMinTrades;
   if(warmup_min_trades < 20) warmup_min_trades = 20;
   if(warmup_min_trades > 2000) warmup_min_trades = 2000;

   int base_h = FXAI_ClampHorizon(PredictionTargetMinutes);
   int horizons[];
   ArrayResize(horizons, 0);
   if(AI_MultiHorizon && ArraySize(g_horizon_minutes) > 0)
   {
      int hn = ArraySize(g_horizon_minutes);
      if(hn > FXAI_MAX_HORIZONS) hn = FXAI_MAX_HORIZONS;
      ArrayResize(horizons, hn);
      for(int i=0; i<hn; i++)
         horizons[i] = FXAI_ClampHorizon(g_horizon_minutes[i]);
   }
   if(ArraySize(horizons) <= 0)
   {
      ArrayResize(horizons, 1);
      horizons[0] = base_h;
   }
   bool have_primary = false;
   int max_h = base_h;
   for(int i=0; i<ArraySize(horizons); i++)
   {
      if(horizons[i] == base_h) have_primary = true;
      if(horizons[i] > max_h) max_h = horizons[i];
   }
   if(!have_primary && ArraySize(horizons) < FXAI_MAX_HORIZONS)
   {
      int hs = ArraySize(horizons);
      ArrayResize(horizons, hs + 1);
      horizons[hs] = base_h;
      if(base_h > max_h) max_h = base_h;
   }

   double base_buy_thr = AI_BuyThreshold;
   double base_sell_thr = AI_SellThreshold;
   FXAI_SanitizeThresholdPair(base_buy_thr, base_sell_thr);
   double evThresholdPoints = FXAI_Clamp(AI_EVThresholdPoints, 0.0, 100.0);

   int needed = warmup_samples + max_h + FEATURE_LB;

   FXAIDataSnapshot snapshot;
   if(!FXAI_ExportDataSnapshot(symbol, AI_CommissionPerLotSide, AI_CostBufferPoints, snapshot))
      return false;

   MqlRates rates_m1[];
   MqlRates rates_m5[];
   MqlRates rates_m15[];
   MqlRates rates_m30[];
   MqlRates rates_h1[];
   MqlRates rates_ctx_tmp[];

   double open_arr[];
   double high_arr[];
   double low_arr[];
   double close_arr[];
   datetime time_arr[];
   int spread_m1[];
   if(!FXAI_LoadSeriesWithSpread(symbol, needed, rates_m1, close_arr, time_arr, spread_m1))
      return false;

   FXAI_ExtractRatesOHLC(rates_m1, open_arr, high_arr, low_arr, close_arr);

   if(ArraySize(close_arr) < needed || ArraySize(time_arr) < needed)
      return false;

   int needed_m5 = (needed / 5) + 80;
   int needed_m15 = (needed / 15) + 80;
   int needed_m30 = (needed / 30) + 80;
   int needed_h1 = (needed / 60) + 80;
   if(needed_m5 < 220) needed_m5 = 220;
   if(needed_m15 < 220) needed_m15 = 220;
   if(needed_m30 < 220) needed_m30 = 220;
   if(needed_h1 < 220) needed_h1 = 220;

   double close_m5[];
   datetime time_m5[];
   double close_m15[];
   datetime time_m15[];
   double close_m30[];
   datetime time_m30[];
   double close_h1[];
   datetime time_h1[];
   int map_m5[];
   int map_m15[];
   int map_m30[];
   int map_h1[];

   FXAI_LoadSeriesOptionalCached(symbol, PERIOD_M5, needed_m5, rates_m5, close_m5, time_m5);
   FXAI_LoadSeriesOptionalCached(symbol, PERIOD_M15, needed_m15, rates_m15, close_m15, time_m15);
   FXAI_LoadSeriesOptionalCached(symbol, PERIOD_M30, needed_m30, rates_m30, close_m30, time_m30);
   FXAI_LoadSeriesOptionalCached(symbol, PERIOD_H1, needed_h1, rates_h1, close_h1, time_h1);

   int lag_m5 = 2 * PeriodSeconds(PERIOD_M5);
   int lag_m15 = 2 * PeriodSeconds(PERIOD_M15);
   int lag_m30 = 2 * PeriodSeconds(PERIOD_M30);
   int lag_h1 = 2 * PeriodSeconds(PERIOD_H1);
   if(lag_m5 <= 0) lag_m5 = 600;
   if(lag_m15 <= 0) lag_m15 = 1800;
   if(lag_m30 <= 0) lag_m30 = 3600;
   if(lag_h1 <= 0) lag_h1 = 7200;

   FXAI_BuildAlignedIndexMap(time_arr, time_m5, lag_m5, map_m5);
   FXAI_BuildAlignedIndexMap(time_arr, time_m15, lag_m15, map_m15);
   FXAI_BuildAlignedIndexMap(time_arr, time_m30, lag_m30, map_m30);
   FXAI_BuildAlignedIndexMap(time_arr, time_h1, lag_h1, map_h1);

   int ctx_count = ArraySize(g_context_symbols);
   if(ctx_count > FXAI_MAX_CONTEXT_SYMBOLS) ctx_count = FXAI_MAX_CONTEXT_SYMBOLS;
   FXAIContextSeries ctx_series[];
   ArrayResize(ctx_series, ctx_count);
   for(int s=0; s<ctx_count; s++)
   {
      ctx_series[s].loaded = FXAI_LoadSeriesOptionalCached(g_context_symbols[s],
                                                          PERIOD_M1,
                                                          needed,
                                                          rates_ctx_tmp,
                                                          ctx_series[s].close,
                                                          ctx_series[s].time);
   }

   int i_start = max_h;
   int i_end = max_h + warmup_samples - 1;
   int max_valid = needed - FEATURE_LB - 1;
   if(i_end > max_valid) i_end = max_valid;
   if(i_end <= i_start) return false;

   double ctx_mean_arr[];
   double ctx_std_arr[];
   double ctx_up_arr[];
   double ctx_extra_arr[];
   FXAI_PrecomputeContextAggregates(time_arr,
                                   close_arr,
                                   ctx_series,
                                   ctx_count,
                                   i_end,
                                   ctx_mean_arr,
                                   ctx_std_arr,
                                   ctx_up_arr,
                                   ctx_extra_arr);

   double cost_buffer_points = (AI_CostBufferPoints < 0.0 ? 0.0 : AI_CostBufferPoints);
   double commission_points = snapshot.commission_points;
   FXAIAIHyperParams base_hp;
   FXAI_BuildHyperParams(base_hp);

   // Warmup-stage feature-adaptive normalization window search.
   FXAI_OptimizeNormalizationWindows(i_start,
                                     i_end,
                                     base_h,
                                     commission_points,
                                     cost_buffer_points,
                                     evThresholdPoints,
                                     snapshot,
                                     spread_m1,
                                     time_arr,
                                     open_arr,
                                     high_arr,
                                     low_arr,
                                     close_arr,
                                     time_m5,
                                     close_m5,
                                     map_m5,
                                     time_m15,
                                     close_m15,
                                     map_m15,
                                     time_m30,
                                     close_m30,
                                     map_m30,
                                     time_h1,
                                     close_h1,
                                     map_h1,
                                     ctx_mean_arr,
                                     ctx_std_arr,
                                     ctx_up_arr,
                                     ctx_extra_arr,
                                     base_hp,
                                     base_buy_thr,
                                     base_sell_thr);

   datetime bar_time = iTime(symbol, PERIOD_M1, 1);
   if(bar_time <= 0) bar_time = TimeCurrent();
   int seed = AI_WarmupSeed;
   if(seed < 0) seed = -seed;
   int evLookbackWarm = AI_EVLookbackSamples;
   if(evLookbackWarm < 20) evLookbackWarm = 20;
   if(evLookbackWarm > 400) evLookbackWarm = 400;
   int ai_hint = (AI_Ensemble ? -1 : (int)AI_Type);
   if(ai_hint < -1 || ai_hint >= FXAI_AI_COUNT) ai_hint = -1;
   FXAIPreparedSample primary_samples[];
   for(int hi=0; hi<ArraySize(horizons); hi++)
   {
      int H = FXAI_ClampHorizon(horizons[hi]);
      FXAIPreparedSample samples_h[];
      FXAI_PrecomputeTrainingSamples(i_start,
                                    i_end,
                                    H,
                                    commission_points,
                                    cost_buffer_points,
                                    evThresholdPoints,
                                    snapshot,
                                    spread_m1,
                                    time_arr,
                                    open_arr,
                                    high_arr,
                                    low_arr,
                                    close_arr,
                                    time_m5,
                                    close_m5,
                                    map_m5,
                                    time_m15,
                                    close_m15,
                                    map_m15,
                                    time_m30,
                                    close_m30,
                                    map_m30,
                                    time_h1,
                                    close_h1,
                                    map_h1,
                                    ctx_mean_arr,
                                    ctx_std_arr,
                                    ctx_up_arr,
                                    ctx_extra_arr,
                                    -1,
                                    samples_h);
      FXAI_WarmupTrainHorizonPolicyForSamples(H,
                                              base_h,
                                              evLookbackWarm,
                                              snapshot,
                                              close_arr,
                                              ai_hint,
                                              i_start,
                                              i_end,
                                              samples_h);

      FXAINormSampleCache norm_caches[];
      ArrayResize(norm_caches, 0);
      for(int ai_idx=0; ai_idx<FXAI_AI_COUNT; ai_idx++)
      {
         int methods[];
         FXAI_BuildNormMethodCandidateList(ai_idx, methods);
         for(int m=0; m<ArraySize(methods); m++)
         {
            if(FXAI_FindNormSampleCache(methods[m], norm_caches) >= 0) continue;
            FXAI_EnsureNormSampleCache(methods[m],
                                       i_start,
                                       i_end,
                                       H,
                                       commission_points,
                                       cost_buffer_points,
                                       evThresholdPoints,
                                       snapshot,
                                       spread_m1,
                                       time_arr,
                                       open_arr,
                                       high_arr,
                                       low_arr,
                                       close_arr,
                                       time_m5,
                                       close_m5,
                                       map_m5,
                                       time_m15,
                                       close_m15,
                                       map_m15,
                                       time_m30,
                                       close_m30,
                                       map_m30,
                                       time_h1,
                                       close_h1,
                                       map_h1,
                                       ctx_mean_arr,
                                       ctx_std_arr,
                                       ctx_up_arr,
                                       ctx_extra_arr,
                                       norm_caches);
         }
      }

      FXAI_WarmupSelectNormBanksForHorizon(H,
                                           H == base_h,
                                           warmup_train_epochs,
                                           i_start,
                                           i_end,
                                           base_hp,
                                           base_buy_thr,
                                           base_sell_thr,
                                           norm_caches);

      FXAI_WarmupSelectBanksForHorizon(H,
                                       H == base_h,
                                       warmup_loops,
                                       warmup_folds,
                                       warmup_train_epochs,
                                       warmup_min_trades,
                                       seed,
                                       bar_time,
                                       base_hp,
                                       base_buy_thr,
                                       base_sell_thr,
                                       i_start,
                                       i_end,
                                       snapshot,
                                       spread_m1,
                                       time_arr,
                                       open_arr,
                                       high_arr,
                                       low_arr,
                                       close_arr,
                                       time_m5,
                                       close_m5,
                                       map_m5,
                                       time_m15,
                                       close_m15,
                                       map_m15,
                                       time_m30,
                                       close_m30,
                                       map_m30,
                                       time_h1,
                                       close_h1,
                                       map_h1,
                                       ctx_mean_arr,
                                       ctx_std_arr,
                                       ctx_up_arr,
                                       ctx_extra_arr,
                                       samples_h,
                                       norm_caches);
      if(H == base_h)
         FXAI_CopyPreparedSamples(samples_h, primary_samples);
   }

   for(int hi=0; hi<ArraySize(horizons); hi++)
   {
      int H = FXAI_ClampHorizon(horizons[hi]);
      FXAIPreparedSample samples_h[];
      FXAI_PrecomputeTrainingSamples(i_start,
                                    i_end,
                                    H,
                                    commission_points,
                                    cost_buffer_points,
                                    evThresholdPoints,
                                    snapshot,
                                    spread_m1,
                                    time_arr,
                                    open_arr,
                                    high_arr,
                                    low_arr,
                                    close_arr,
                                    time_m5,
                                    close_m5,
                                    map_m5,
                                    time_m15,
                                    close_m15,
                                    map_m15,
                                    time_m30,
                                    close_m30,
                                    map_m30,
                                    time_h1,
                                    close_h1,
                                    map_h1,
                                    ctx_mean_arr,
                                    ctx_std_arr,
                                    ctx_up_arr,
                                    ctx_extra_arr,
                                    -1,
                                    samples_h);
      FXAINormSampleCache norm_caches[];
      ArrayResize(norm_caches, 0);
      for(int ai_idx=0; ai_idx<FXAI_AI_COUNT; ai_idx++)
      {
         FXAI_EnsureRoutedNormCachesForSamples(ai_idx,
                                               i_start,
                                               i_end,
                                               H,
                                               commission_points,
                                               cost_buffer_points,
                                               evThresholdPoints,
                                               snapshot,
                                               spread_m1,
                                               time_arr,
                                               open_arr,
                                               high_arr,
                                               low_arr,
                                               close_arr,
                                               time_m5,
                                               close_m5,
                                               map_m5,
                                               time_m15,
                                               close_m15,
                                               map_m15,
                                               time_m30,
                                               close_m30,
                                               map_m30,
                                               time_h1,
                                               close_h1,
                                               map_h1,
                                               ctx_mean_arr,
                                               ctx_std_arr,
                                               ctx_up_arr,
                                               ctx_extra_arr,
                                               samples_h,
                                               norm_caches);
      }
      FXAI_WarmupPretrainMetaForSamples(H,
                                        warmup_folds,
                                        warmup_train_epochs,
                                        i_start,
                                        i_end,
                                        base_buy_thr,
                                        base_sell_thr,
                                        samples_h,
                                        norm_caches);
      if(H == base_h && ArraySize(primary_samples) <= 0)
         FXAI_CopyPreparedSamples(samples_h, primary_samples);
   }

   if(ArraySize(primary_samples) <= 0) return false;
   for(int ai_idx=0; ai_idx<FXAI_AI_COUNT; ai_idx++)
   {
      CFXAIAIPlugin *runtime = g_plugins.Get(ai_idx);
      if(runtime == NULL) continue;

      FXAI_ResetModelAuxState(ai_idx);
      runtime.Reset();
      FXAIAIHyperParams hp_init;
      FXAI_GetModelHyperParamsRouted(ai_idx, 0, base_h, hp_init);
      runtime.EnsureInitialized(hp_init);
   }

   // Warm the runtime models across every configured horizon. The online path
   // uses a single runtime instance per model, so base-horizon-only warmup can
   // leave routed non-base horizons effectively cold on the first live bars.
   for(int hi=0; hi<ArraySize(horizons); hi++)
   {
      int H = FXAI_ClampHorizon(horizons[hi]);
      FXAIPreparedSample runtime_samples[];
      FXAINormSampleCache runtime_norm_caches[];
      ArrayResize(runtime_norm_caches, 0);
      if(H == base_h)
      {
         FXAI_CopyPreparedSamples(primary_samples, runtime_samples);
      }
      else
      {
         FXAI_PrecomputeTrainingSamples(i_start,
                                       i_end,
                                       H,
                                       commission_points,
                                       cost_buffer_points,
                                       evThresholdPoints,
                                       snapshot,
                                       spread_m1,
                                       time_arr,
                                       open_arr,
                                       high_arr,
                                       low_arr,
                                       close_arr,
                                       time_m5,
                                       close_m5,
                                       map_m5,
                                       time_m15,
                                       close_m15,
                                       map_m15,
                                       time_m30,
                                       close_m30,
                                       map_m30,
                                       time_h1,
                                       close_h1,
                                       map_h1,
                                       ctx_mean_arr,
                                       ctx_std_arr,
                                       ctx_up_arr,
                                       ctx_extra_arr,
                                       -1,
                                       runtime_samples);
      }

      for(int ai_idx=0; ai_idx<FXAI_AI_COUNT; ai_idx++)
      {
         FXAI_EnsureRoutedNormCachesForSamples(ai_idx,
                                               i_start,
                                               i_end,
                                               H,
                                               commission_points,
                                               cost_buffer_points,
                                               evThresholdPoints,
                                               snapshot,
                                               spread_m1,
                                               time_arr,
                                               open_arr,
                                               high_arr,
                                               low_arr,
                                               close_arr,
                                               time_m5,
                                               close_m5,
                                               map_m5,
                                               time_m15,
                                               close_m15,
                                               map_m15,
                                               time_m30,
                                               close_m30,
                                               map_m30,
                                               time_h1,
                                               close_h1,
                                               map_h1,
                                               ctx_mean_arr,
                                               ctx_std_arr,
                                               ctx_up_arr,
                                               ctx_extra_arr,
                                               runtime_samples,
                                               runtime_norm_caches);
      }

      for(int ai_idx=0; ai_idx<FXAI_AI_COUNT; ai_idx++)
      {
         CFXAIAIPlugin *runtime = g_plugins.Get(ai_idx);
         if(runtime == NULL) continue;

         FXAI_TrainModelWindowPreparedRoutedCached(ai_idx,
                                                   *runtime,
                                                   i_start,
                                                   i_end,
                                                   warmup_train_epochs,
                                                   runtime_samples,
                                                   runtime_norm_caches);
      }
   }

   for(int ai_idx=0; ai_idx<FXAI_AI_COUNT; ai_idx++)
   {
      if(g_plugins.Get(ai_idx) == NULL) continue;
      g_ai_trained[ai_idx] = true;
      g_ai_last_train_bar[ai_idx] = bar_time;
   }

   g_ai_warmup_done = true;
   Print("FXAI warmup completed: symbol=", symbol,
         ", samples=", warmup_samples,
         ", loops=", warmup_loops,
         ", folds=", warmup_folds,
         ", horizons=", ArraySize(horizons));
   return true;
}

void FXAI_PrecomputeTrainingSamples(const int i_start,
                                   const int i_end,
                                   const int H,
                                   const double commission_points,
                                   const double cost_buffer_points,
                                   const double ev_threshold_points,
                                   const FXAIDataSnapshot &snapshot,
                                   const int &spread_m1[],
                                   const datetime &time_arr[],
                                   const double &open_arr[],
                                   const double &high_arr[],
                                   const double &low_arr[],
                                   const double &close_arr[],
                                   const datetime &time_m5[],
                                   const double &close_m5[],
                                   const int &map_m5[],
                                   const datetime &time_m15[],
                                   const double &close_m15[],
                                   const int &map_m15[],
                                   const datetime &time_m30[],
                                   const double &close_m30[],
                                   const int &map_m30[],
                                   const datetime &time_h1[],
                                   const double &close_h1[],
                                   const int &map_h1[],
                                   const double &ctx_mean_arr[],
                                   const double &ctx_std_arr[],
                                   const double &ctx_up_arr[],
                                   const double &ctx_extra_arr[],
                                   const int norm_method_override,
                                   FXAIPreparedSample &samples[])
{
   if(i_end < i_start) return;
   if(i_end < 0) return;

   int need_size = i_end + 1;
   if(ArraySize(samples) < need_size)
      ArrayResize(samples, need_size);

   // Build samples oldest -> newest (as-series: larger index is older) so
   // any stateful normalizer sees a causal timeline.
   for(int i=i_end; i>=i_start; i--)
   {
      if(i < 0 || i >= ArraySize(samples)) continue;
      FXAI_PrepareTrainingSample(i,
                                H,
                                commission_points,
                                cost_buffer_points,
                                ev_threshold_points,
                                snapshot,
                                spread_m1,
                                time_arr,
                                open_arr,
                                high_arr,
                                low_arr,
                                close_arr,
                                time_m5,
                                close_m5,
                                map_m5,
                                time_m15,
                                close_m15,
                                map_m15,
                                time_m30,
                                close_m30,
                                map_m30,
                                time_h1,
                                close_h1,
                                map_h1,
                                ctx_mean_arr,
                                ctx_std_arr,
                                ctx_up_arr,
                                ctx_extra_arr,
                                norm_method_override,
                                samples[i]);
   }
}

int FXAI_FindNormSampleCache(const int method_id,
                             FXAINormSampleCache &caches[])
{
   for(int i=0; i<ArraySize(caches); i++)
   {
      if(caches[i].ready && caches[i].method_id == method_id)
         return i;
   }
   return -1;
}

int FXAI_EnsureNormSampleCache(const int method_id,
                               const int i_start,
                               const int i_end,
                               const int H,
                               const double commission_points,
                               const double cost_buffer_points,
                               const double ev_threshold_points,
                               const FXAIDataSnapshot &snapshot,
                               const int &spread_m1[],
                               const datetime &time_arr[],
                               const double &open_arr[],
                               const double &high_arr[],
                               const double &low_arr[],
                               const double &close_arr[],
                               const datetime &time_m5[],
                               const double &close_m5[],
                               const int &map_m5[],
                               const datetime &time_m15[],
                               const double &close_m15[],
                               const int &map_m15[],
                               const datetime &time_m30[],
                               const double &close_m30[],
                               const int &map_m30[],
                               const datetime &time_h1[],
                               const double &close_h1[],
                               const int &map_h1[],
                               const double &ctx_mean_arr[],
                               const double &ctx_std_arr[],
                               const double &ctx_up_arr[],
                               const double &ctx_extra_arr[],
                               FXAINormSampleCache &caches[])
{
   int idx = FXAI_FindNormSampleCache(method_id, caches);
   if(idx >= 0) return idx;

   int sz = ArraySize(caches);
   ArrayResize(caches, sz + 1);
   caches[sz].method_id = method_id;
   caches[sz].ready = true;
   FXAI_PrecomputeTrainingSamples(i_start,
                                  i_end,
                                  H,
                                  commission_points,
                                  cost_buffer_points,
                                  ev_threshold_points,
                                  snapshot,
                                  spread_m1,
                                  time_arr,
                                  open_arr,
                                  high_arr,
                                  low_arr,
                                  close_arr,
                                  time_m5,
                                  close_m5,
                                  map_m5,
                                  time_m15,
                                  close_m15,
                                  map_m15,
                                  time_m30,
                                  close_m30,
                                  map_m30,
                                  time_h1,
                                  close_h1,
                                  map_h1,
                                  ctx_mean_arr,
                                  ctx_std_arr,
                                  ctx_up_arr,
                                  ctx_extra_arr,
                                  method_id,
                                  caches[sz].samples);
   return sz;
}

void FXAI_EnsureRoutedNormCachesForSamples(const int ai_idx,
                                           const int i_start,
                                           const int i_end,
                                           const int H,
                                           const double commission_points,
                                           const double cost_buffer_points,
                                           const double ev_threshold_points,
                                           const FXAIDataSnapshot &snapshot,
                                           const int &spread_m1[],
                                           const datetime &time_arr[],
                                           const double &open_arr[],
                                           const double &high_arr[],
                                           const double &low_arr[],
                                           const double &close_arr[],
                                           const datetime &time_m5[],
                                           const double &close_m5[],
                                           const int &map_m5[],
                                           const datetime &time_m15[],
                                           const double &close_m15[],
                                           const int &map_m15[],
                                           const datetime &time_m30[],
                                           const double &close_m30[],
                                           const int &map_m30[],
                                           const datetime &time_h1[],
                                           const double &close_h1[],
                                           const int &map_h1[],
                                           const double &ctx_mean_arr[],
                                           const double &ctx_std_arr[],
                                           const double &ctx_up_arr[],
                                           const double &ctx_extra_arr[],
                                           const FXAIPreparedSample &samples[],
                                           FXAINormSampleCache &caches[])
{
   if(ai_idx < 0 || ai_idx >= FXAI_AI_COUNT) return;
   int n = ArraySize(samples);
   if(n <= 0) return;

   int start = i_start;
   int end = i_end;
   if(start < 0) start = 0;
   if(end >= n) end = n - 1;
   if(end < start) return;

   bool needed_method[FXAI_NORM_METHOD_COUNT];
   for(int m=0; m<FXAI_NORM_METHOD_COUNT; m++)
      needed_method[m] = false;

   for(int i=end; i>=start; i--)
   {
      if(i < 0 || i >= n) continue;
      if(!samples[i].valid) continue;
      int method_id = (int)FXAI_GetModelNormMethodRouted(ai_idx,
                                                         samples[i].regime_id,
                                                         samples[i].horizon_minutes);
      if(method_id < 0 || method_id >= FXAI_NORM_METHOD_COUNT) continue;
      if(needed_method[method_id]) continue;
      needed_method[method_id] = true;

      FXAI_EnsureNormSampleCache(method_id,
                                 i_start,
                                 i_end,
                                 H,
                                 commission_points,
                                 cost_buffer_points,
                                 ev_threshold_points,
                                 snapshot,
                                 spread_m1,
                                 time_arr,
                                 open_arr,
                                 high_arr,
                                 low_arr,
                                 close_arr,
                                 time_m5,
                                 close_m5,
                                 map_m5,
                                 time_m15,
                                 close_m15,
                                 map_m15,
                                 time_m30,
                                 close_m30,
                                 map_m30,
                                 time_h1,
                                 close_h1,
                                 map_h1,
                                 ctx_mean_arr,
                                 ctx_std_arr,
                                 ctx_up_arr,
                                 ctx_extra_arr,
                                 caches);
   }

   int default_method = (int)FXAI_GetModelNormMethodRouted(ai_idx, 0, H);
   if(default_method >= 0 && default_method < FXAI_NORM_METHOD_COUNT && !needed_method[default_method])
   {
      FXAI_EnsureNormSampleCache(default_method,
                                 i_start,
                                 i_end,
                                 H,
                                 commission_points,
                                 cost_buffer_points,
                                 ev_threshold_points,
                                 snapshot,
                                 spread_m1,
                                 time_arr,
                                 open_arr,
                                 high_arr,
                                 low_arr,
                                 close_arr,
                                 time_m5,
                                 close_m5,
                                 map_m5,
                                 time_m15,
                                 close_m15,
                                 map_m15,
                                 time_m30,
                                 close_m30,
                                 map_m30,
                                 time_h1,
                                 close_h1,
                                 map_h1,
                                 ctx_mean_arr,
                                 ctx_std_arr,
                                 ctx_up_arr,
                                 ctx_extra_arr,
                                 caches);
   }
}

void FXAI_GetCachedPreparedSample(const int ai_idx,
                                  const FXAIPreparedSample &reference_sample,
                                  const int sample_index,
                                  FXAINormSampleCache &caches[],
                                  FXAIPreparedSample &out_sample)
{
   out_sample = reference_sample;
   int method_id = (int)FXAI_GetModelNormMethodRouted(ai_idx,
                                                      reference_sample.regime_id,
                                                      reference_sample.horizon_minutes);
   int cache_idx = FXAI_FindNormSampleCache(method_id, caches);
   if(cache_idx < 0) return;
   if(sample_index < 0 || sample_index >= ArraySize(caches[cache_idx].samples))
      return;
   out_sample = caches[cache_idx].samples[sample_index];
}

int FXAI_FindNormInputCache(const int method_id,
                            FXAINormInputCache &caches[])
{
   for(int i=0; i<ArraySize(caches); i++)
   {
      if(caches[i].ready && caches[i].method_id == method_id)
         return i;
   }
   return -1;
}

int FXAI_EnsureNormInputCache(const int method_id,
                              const double spread_pred,
                              const int &spread_m1[],
                              const FXAIDataSnapshot &snapshot,
                              const datetime &time_arr[],
                              const double &open_arr[],
                              const double &high_arr[],
                              const double &low_arr[],
                              const double &close_arr[],
                              const datetime &time_m5[],
                              const double &close_m5[],
                              const int &map_m5[],
                              const datetime &time_m15[],
                              const double &close_m15[],
                              const int &map_m15[],
                              const datetime &time_m30[],
                              const double &close_m30[],
                              const int &map_m30[],
                              const datetime &time_h1[],
                              const double &close_h1[],
                              const int &map_h1[],
                              const double &ctx_mean_arr[],
                              const double &ctx_std_arr[],
                              const double &ctx_up_arr[],
                              const double &ctx_extra_arr[],
                              FXAINormInputCache &caches[])
{
   int idx = FXAI_FindNormInputCache(method_id, caches);
   if(idx >= 0) return idx;

   ENUM_FXAI_FEATURE_NORMALIZATION norm_method = FXAI_SanitizeNormMethod(method_id);
   double ctx_mean_pred = FXAI_GetArrayValue(ctx_mean_arr, 0, 0.0);
   double ctx_std_pred = FXAI_GetArrayValue(ctx_std_arr, 0, 0.0);
   double ctx_up_pred = FXAI_GetArrayValue(ctx_up_arr, 0, 0.5);
   double feat_pred[FXAI_AI_FEATURES];
   if(!FXAI_ComputeFeatureVector(0,
                                 spread_pred,
                                 time_arr,
                                 open_arr,
                                 high_arr,
                                 low_arr,
                                 close_arr,
                                 time_m5,
                                 close_m5,
                                 map_m5,
                                 time_m15,
                                 close_m15,
                                 map_m15,
                                 time_m30,
                                 close_m30,
                                 map_m30,
                                 time_h1,
                                 close_h1,
                                 map_h1,
                                 ctx_mean_pred,
                                 ctx_std_pred,
                                 ctx_up_pred,
                                 ctx_extra_arr,
                                 norm_method,
                                 feat_pred))
      return -1;

   bool need_prev = FXAI_FeatureNormNeedsPrevious(norm_method);
   bool has_prev_feat = false;
   double feat_prev[FXAI_AI_FEATURES];
   for(int f=0; f<FXAI_AI_FEATURES; f++)
      feat_prev[f] = 0.0;

   if(need_prev && ArraySize(close_arr) > 1)
   {
      double spread_prev = FXAI_GetSpreadAtIndex(1, spread_m1, spread_pred);
      double ctx_mean_prev = FXAI_GetArrayValue(ctx_mean_arr, 1, ctx_mean_pred);
      double ctx_std_prev = FXAI_GetArrayValue(ctx_std_arr, 1, ctx_std_pred);
      double ctx_up_prev = FXAI_GetArrayValue(ctx_up_arr, 1, ctx_up_pred);
      has_prev_feat = FXAI_ComputeFeatureVector(1,
                                               spread_prev,
                                               time_arr,
                                               open_arr,
                                               high_arr,
                                               low_arr,
                                               close_arr,
                                               time_m5,
                                               close_m5,
                                               map_m5,
                                               time_m15,
                                               close_m15,
                                               map_m15,
                                               time_m30,
                                               close_m30,
                                               map_m30,
                                               time_h1,
                                               close_h1,
                                               map_h1,
                                               ctx_mean_prev,
                                               ctx_std_prev,
                                               ctx_up_prev,
                                               ctx_extra_arr,
                                               norm_method,
                                               feat_prev);
   }

   double feat_norm[FXAI_AI_FEATURES];
   FXAI_ApplyFeatureNormalization(norm_method,
                                  feat_pred,
                                  feat_prev,
                                  has_prev_feat,
                                  snapshot.bar_time,
                                  feat_norm);

   int sz = ArraySize(caches);
   ArrayResize(caches, sz + 1);
   caches[sz].method_id = method_id;
   caches[sz].ready = true;
   FXAI_BuildInputVector(feat_norm, caches[sz].x);
   return sz;
}

void FXAI_ApplyPreparedSampleToModel(const int ai_idx,
                                    CFXAIAIPlugin &plugin,
                                    const FXAIPreparedSample &sample,
                                    const FXAIAIHyperParams &hp)
{
   if(!sample.valid) return;

   FXAIAISampleV2 s2;
   s2.valid = sample.valid;
   s2.label_class = sample.label_class;
   s2.regime_id = sample.regime_id;
   s2.horizon_minutes = sample.horizon_minutes;
   s2.move_points = sample.move_points;
   s2.min_move_points = sample.min_move_points;
   s2.cost_points = sample.cost_points;
   s2.sample_time = sample.sample_time;
   for(int k=0; k<FXAI_AI_WEIGHTS; k++)
      s2.x[k] = sample.x[k];

   plugin.TrainV2(s2, hp);
   FXAI_UpdateModelMoveStats(ai_idx, sample.move_points);
}

void FXAI_TrainModelWindowPrepared(const int ai_idx,
                                  CFXAIAIPlugin &plugin,
                                  const int i_start,
                                  const int i_end,
                                  const int epochs,
                                  const FXAIAIHyperParams &hp,
                                  const FXAIPreparedSample &samples[])
{
   if(i_end < i_start || epochs <= 0) return;
   int n = ArraySize(samples);
   if(n <= 0) return;

   int start = i_start;
   int end = i_end;
   if(start < 0) start = 0;
   if(end >= n) end = n - 1;
   if(end < start) return;

   for(int epoch=0; epoch<epochs; epoch++)
   {
      for(int i=end; i>=start; i--)
         FXAI_ApplyPreparedSampleToModel(ai_idx, plugin, samples[i], hp);
   }
}

void FXAI_ApplyPreparedSampleToModelRouted(const int ai_idx,
                                           CFXAIAIPlugin &plugin,
                                           const FXAIPreparedSample &sample)
{
   if(!sample.valid) return;
   FXAIAIHyperParams hp_sample;
   FXAI_GetModelHyperParamsRouted(ai_idx, sample.regime_id, sample.horizon_minutes, hp_sample);
   FXAI_ApplyPreparedSampleToModel(ai_idx, plugin, sample, hp_sample);
}

void FXAI_TrainModelWindowPreparedRouted(const int ai_idx,
                                         CFXAIAIPlugin &plugin,
                                         const int i_start,
                                         const int i_end,
                                         const int epochs,
                                         const FXAIPreparedSample &samples[])
{
   if(i_end < i_start || epochs <= 0) return;
   int n = ArraySize(samples);
   if(n <= 0) return;

   int start = i_start;
   int end = i_end;
   if(start < 0) start = 0;
   if(end >= n) end = n - 1;
   if(end < start) return;

   for(int epoch=0; epoch<epochs; epoch++)
   {
      for(int i=end; i>=start; i--)
         FXAI_ApplyPreparedSampleToModelRouted(ai_idx, plugin, samples[i]);
   }
}

void FXAI_TrainModelWindowPreparedRoutedCached(const int ai_idx,
                                               CFXAIAIPlugin &plugin,
                                               const int i_start,
                                               const int i_end,
                                               const int epochs,
                                               const FXAIPreparedSample &samples[],
                                               FXAINormSampleCache &caches[])
{
   if(i_end < i_start || epochs <= 0) return;
   int n = ArraySize(samples);
   if(n <= 0) return;

   int start = i_start;
   int end = i_end;
   if(start < 0) start = 0;
   if(end >= n) end = n - 1;
   if(end < start) return;

   for(int epoch=0; epoch<epochs; epoch++)
   {
      for(int i=end; i>=start; i--)
      {
         if(i < 0 || i >= ArraySize(samples)) continue;
         if(!samples[i].valid) continue;
         FXAIPreparedSample train_sample;
         FXAI_GetCachedPreparedSample(ai_idx, samples[i], i, caches, train_sample);
         FXAI_ApplyPreparedSampleToModelRouted(ai_idx, plugin, train_sample);
      }
   }
}

double FXAI_CalcReplayPriority(const FXAIPreparedSample &sample)
{
   double p = sample.sample_weight;
   p += 0.35 * sample.quality_score;
   p += 0.10 * FXAI_Clamp(sample.spread_stress, 0.0, 3.0);
   if(sample.label_class != (int)FXAI_LABEL_SKIP) p += 0.20;
   if((sample.path_flags & FXAI_PATHFLAG_DUAL_HIT) != 0) p += 0.30;
   if((sample.path_flags & FXAI_PATHFLAG_SPREAD_STRESS) != 0) p += 0.25;
   if((sample.path_flags & FXAI_PATHFLAG_SLOW_HIT) != 0) p += 0.10;
   return FXAI_Clamp(p, 0.25, 12.0);
}

void FXAI_AddReplaySample(const FXAIPreparedSample &sample)
{
   if(!sample.valid) return;

   int regime_id = sample.regime_id;
   if(regime_id < 0 || regime_id >= FXAI_REGIME_COUNT) regime_id = 0;
   int hslot = sample.horizon_slot;
   if(hslot < 0 || hslot >= FXAI_MAX_HORIZONS)
      hslot = FXAI_GetHorizonSlot(sample.horizon_minutes);
   if(hslot < 0 || hslot >= FXAI_MAX_HORIZONS) hslot = 0;

   if(sample.sample_time > 0 && g_replay_last_sample_time[hslot] == sample.sample_time)
      return;

   int slot = -1;
   if(g_replay_count < FXAI_REPLAY_CAPACITY)
   {
      for(int i=0; i<FXAI_REPLAY_CAPACITY; i++)
      {
         if(!g_replay_used[i])
         {
            slot = i;
            break;
         }
      }
   }
   else
   {
      double new_bucket = (double)g_replay_bucket_count[regime_id][hslot];
      double best_evict = -1e18;
      for(int i=0; i<FXAI_REPLAY_CAPACITY; i++)
      {
         if(!g_replay_used[i]) continue;
         int er = g_replay_samples[i].regime_id;
         int eh = g_replay_samples[i].horizon_slot;
         if(er < 0 || er >= FXAI_REGIME_COUNT) er = 0;
         if(eh < 0 || eh >= FXAI_MAX_HORIZONS) eh = 0;
         double old_bucket = (double)g_replay_bucket_count[er][eh];
         double evict_score = old_bucket - (0.25 * g_replay_priority[i]);
         if(g_replay_samples[i].label_class == (int)FXAI_LABEL_SKIP)
            evict_score += 0.10;
         if(old_bucket > new_bucket) evict_score += 0.50;
         if(evict_score > best_evict)
         {
            best_evict = evict_score;
            slot = i;
         }
      }
   }

   if(slot < 0) return;
   if(g_replay_used[slot])
   {
      int old_r = g_replay_samples[slot].regime_id;
      int old_h = g_replay_samples[slot].horizon_slot;
      if(old_r >= 0 && old_r < FXAI_REGIME_COUNT &&
         old_h >= 0 && old_h < FXAI_MAX_HORIZONS &&
         g_replay_bucket_count[old_r][old_h] > 0)
      {
         g_replay_bucket_count[old_r][old_h]--;
      }
   }
   else
   {
      g_replay_count++;
   }

   g_replay_samples[slot] = sample;
   g_replay_samples[slot].regime_id = regime_id;
   g_replay_samples[slot].horizon_slot = hslot;
   g_replay_priority[slot] = FXAI_CalcReplayPriority(sample);
   g_replay_flags[slot] = sample.path_flags;
   g_replay_used[slot] = true;
   g_replay_bucket_count[regime_id][hslot]++;
   if(sample.sample_time > 0) g_replay_last_sample_time[hslot] = sample.sample_time;
}

void FXAI_BoostReplayPriorityByOutcome(const datetime sample_time,
                                       const int horizon_minutes,
                                       const int regime_id,
                                       const int label_class,
                                       const int signal,
                                       const double move_points,
                                       const double min_move_points)
{
   if(sample_time <= 0) return;

   int hslot = FXAI_GetHorizonSlot(horizon_minutes);
   if(hslot < 0 || hslot >= FXAI_MAX_HORIZONS) hslot = 0;

   double min_mv = MathMax(min_move_points, 0.50);
   double edge = MathMax(MathAbs(move_points) - min_mv, 0.0);
   double edge_ratio = FXAI_Clamp(edge / min_mv, 0.0, 4.0);

   bool false_positive = ((signal == 0 || signal == 1) && label_class == (int)FXAI_LABEL_SKIP);
   bool wrong_direction = ((signal == 1 && label_class == (int)FXAI_LABEL_SELL) ||
                           (signal == 0 && label_class == (int)FXAI_LABEL_BUY));
   bool missed_move = (signal == -1 && label_class != (int)FXAI_LABEL_SKIP && edge > 0.0);

   double base_boost = 0.0;
   int add_flags = 0;
   if(false_positive)
   {
      base_boost += 1.10 + 0.35 * edge_ratio;
      add_flags |= FXAI_REPLAYFLAG_FALSE_POS;
   }
   if(wrong_direction)
   {
      base_boost += 1.35 + 0.45 * edge_ratio;
      add_flags |= FXAI_REPLAYFLAG_WRONG_DIR;
   }
   if(missed_move)
   {
      base_boost += 1.00 + 0.50 * edge_ratio;
      add_flags |= FXAI_REPLAYFLAG_MISSED_MOVE;
   }
   if(base_boost <= 0.0) return;

   for(int i=0; i<FXAI_REPLAY_CAPACITY; i++)
   {
      if(!g_replay_used[i]) continue;
      if(g_replay_samples[i].sample_time != sample_time) continue;
      if(g_replay_samples[i].horizon_slot != hslot) continue;

      double boost = base_boost;
      if(regime_id >= 0 && regime_id < FXAI_REGIME_COUNT && g_replay_samples[i].regime_id == regime_id)
         boost += 0.20;
      if((g_replay_flags[i] & FXAI_PATHFLAG_DUAL_HIT) != 0)
         boost += 0.10;
      if((g_replay_flags[i] & FXAI_PATHFLAG_SPREAD_STRESS) != 0)
         boost += 0.10;

      g_replay_priority[i] = FXAI_Clamp(g_replay_priority[i] + boost, 0.25, 20.0);
      g_replay_flags[i] |= add_flags;
      break;
   }
}

void FXAI_TrainModelReplay(const int ai_idx,
                           CFXAIAIPlugin &plugin,
                           const int regime_id,
                           const int horizon_minutes,
                           const int epochs)
{
   if(epochs <= 0 || g_replay_count <= 0) return;

   int hslot = FXAI_GetHorizonSlot(horizon_minutes);
   if(hslot < 0 || hslot >= FXAI_MAX_HORIZONS) hslot = 0;
   int prefer_regime = regime_id;
   if(prefer_regime < 0 || prefer_regime >= FXAI_REGIME_COUNT) prefer_regime = 0;

   int probe_limit = g_replay_count;
   if(probe_limit > 64) probe_limit = 64;

   for(int epoch=0; epoch<epochs; epoch++)
   {
      for(int draw=0; draw<FXAI_REPLAY_DRAWS; draw++)
      {
         int best_idx = -1;
         double best_score = -1e18;
         int start = (g_replay_cursor + draw * 7 + epoch * 13) % FXAI_REPLAY_CAPACITY;
         for(int p=0; p<probe_limit; p++)
         {
            int idx = (start + p) % FXAI_REPLAY_CAPACITY;
            if(!g_replay_used[idx]) continue;

            FXAIPreparedSample sample = g_replay_samples[idx];
            double score = g_replay_priority[idx];
            if(sample.regime_id == prefer_regime) score += 1.00;
            if(sample.horizon_slot == hslot) score += 0.75;
            if(sample.label_class != (int)FXAI_LABEL_SKIP) score += 0.20;
            if((g_replay_flags[idx] & FXAI_PATHFLAG_DUAL_HIT) != 0) score += 0.20;
            if((g_replay_flags[idx] & FXAI_REPLAYFLAG_FALSE_POS) != 0) score += 0.35;
            if((g_replay_flags[idx] & FXAI_REPLAYFLAG_MISSED_MOVE) != 0) score += 0.45;
            if((g_replay_flags[idx] & FXAI_REPLAYFLAG_WRONG_DIR) != 0) score += 0.55;
            double recency_penalty = 0.05 * MathAbs((double)(g_replay_cursor - idx));
            score -= recency_penalty;
            if(score > best_score)
            {
               best_score = score;
               best_idx = idx;
            }
         }

         if(best_idx < 0) continue;
         FXAI_ApplyPreparedSampleToModelRouted(ai_idx, plugin, g_replay_samples[best_idx]);
         g_replay_cursor = (best_idx + 1) % FXAI_REPLAY_CAPACITY;
      }
   }
}

void FXAI_BuildHyperParams(FXAIAIHyperParams &hp)
{
   hp.lr = FXAI_Clamp(AI_LearningRate, 0.001, 0.200);
   hp.l2 = FXAI_Clamp(AI_L2, 0.0, 0.100);

   hp.ftrl_alpha = FXAI_Clamp(FTRL_Alpha, 0.001, 1.000);
   hp.ftrl_beta  = FXAI_Clamp(FTRL_Beta,  0.000, 5.000);
   hp.ftrl_l1    = FXAI_Clamp(FTRL_L1,    0.000, 0.100);
   hp.ftrl_l2    = FXAI_Clamp(FTRL_L2,    0.000, 1.000);

   hp.pa_c      = FXAI_Clamp(PA_C,      0.010, 10.000);
   hp.pa_margin = FXAI_Clamp(PA_Margin, 0.100, 2.000);

   hp.xgb_lr    = FXAI_Clamp(XGB_FastLearningRate, 0.001, 0.300);
   hp.xgb_l2    = FXAI_Clamp(XGB_FastL2,           0.000, 10.000);
   hp.xgb_split = FXAI_Clamp(XGB_SplitThreshold,  -2.000, 2.000);

   hp.mlp_lr   = FXAI_Clamp(MLP_LearningRate, 0.0005, 0.0500);
   hp.mlp_l2   = FXAI_Clamp(MLP_L2,           0.0000, 0.0500);
   hp.mlp_init = FXAI_Clamp(MLP_InitScale,    0.0100, 0.5000);

   hp.tcn_layers = (double)((int)FXAI_Clamp((double)TCN_Layers, 2.0, 8.0));
   hp.tcn_kernel = (double)((int)FXAI_Clamp((double)TCN_KernelSize, 2.0, 5.0));
   hp.tcn_dilation_base = (double)((int)FXAI_Clamp((double)TCN_DilationBase, 1.0, 3.0));

   hp.quantile_lr = FXAI_Clamp(Quantile_LearningRate, 0.0001, 0.1000);
   hp.quantile_l2 = FXAI_Clamp(Quantile_L2,           0.0000, 0.1000);

   hp.enhash_lr = FXAI_Clamp(ENHash_LearningRate, 0.0005, 0.1000);
   hp.enhash_l1 = FXAI_Clamp(ENHash_L1,           0.0000, 0.1000);
   hp.enhash_l2 = FXAI_Clamp(ENHash_L2,           0.0000, 0.1000);
}

double FXAI_RandRange(const double lo, const double hi)
{
   if(hi <= lo) return lo;
   double u = (double)MathRand() / 32767.0;
   return lo + (hi - lo) * FXAI_Clamp(u, 0.0, 1.0);
}

void FXAI_ResetModelHyperParams()
{
   FXAIAIHyperParams base;
   FXAI_BuildHyperParams(base);
   double base_buy = AI_BuyThreshold;
   double base_sell = AI_SellThreshold;
   FXAI_SanitizeThresholdPair(base_buy, base_sell);

   for(int i=0; i<FXAI_AI_COUNT; i++)
   {
      g_model_hp[i] = base;
      g_model_hp_ready[i] = false;
      g_model_norm_method[i] = (int)FXAI_GetFeatureNormalizationMethod();
      g_model_norm_ready[i] = false;
      g_model_buy_thr[i] = base_buy;
      g_model_sell_thr[i] = base_sell;
      g_model_thr_ready[i] = false;
      for(int r=0; r<FXAI_REGIME_COUNT; r++)
      {
         for(int h=0; h<FXAI_MAX_HORIZONS; h++)
         {
            g_model_norm_method_bank[i][r][h] = (int)FXAI_GetFeatureNormalizationMethod();
            g_model_norm_bank_ready[i][r][h] = false;
         }
         g_model_buy_thr_regime[i][r] = base_buy;
         g_model_sell_thr_regime[i][r] = base_sell;
         g_model_thr_regime_ready[i][r] = false;
         for(int h=0; h<FXAI_MAX_HORIZONS; h++)
         {
            g_model_hp_bank[i][r][h] = base;
            g_model_hp_bank_ready[i][r][h] = false;
            g_model_buy_thr_bank[i][r][h] = base_buy;
            g_model_sell_thr_bank[i][r][h] = base_sell;
            g_model_thr_bank_ready[i][r][h] = false;
         }
      }
      for(int h=0; h<FXAI_MAX_HORIZONS; h++)
      {
         g_model_hp_horizon[i][h] = base;
         g_model_hp_horizon_ready[i][h] = false;
         g_model_norm_method_horizon[i][h] = (int)FXAI_GetFeatureNormalizationMethod();
         g_model_norm_horizon_ready[i][h] = false;
         g_model_buy_thr_horizon[i][h] = base_buy;
         g_model_sell_thr_horizon[i][h] = base_sell;
         g_model_thr_horizon_ready[i][h] = false;
         g_model_horizon_edge_ema[i][h] = 0.0;
         g_model_horizon_edge_ready[i][h] = false;
         g_model_horizon_obs[i][h] = 0;
      }
   }

   for(int r=0; r<FXAI_REGIME_COUNT; r++)
   {
      g_horizon_regime_total_obs[r] = 0.0;
      g_stack_ready[r] = false;
      g_stack_obs[r] = 0;
      g_hpolicy_ready[r] = false;
      g_hpolicy_obs[r] = 0;
      for(int h=0; h<FXAI_STACK_HIDDEN; h++)
      {
         g_stack_b1[r][h] = 0.0;
         for(int k=0; k<FXAI_STACK_FEATS; k++)
            g_stack_w1[r][h][k] = 0.0;
      }
      for(int c=0; c<3; c++)
      {
         g_stack_b2[r][c] = 0.0;
         for(int h=0; h<FXAI_STACK_HIDDEN; h++)
            g_stack_w2[r][c][h] = 0.0;
      }
      for(int k=0; k<FXAI_HPOL_FEATS; k++)
         g_hpolicy_w[r][k] = 0.0;
      for(int h=0; h<FXAI_MAX_HORIZONS; h++)
      {
         g_horizon_regime_edge_ema[r][h] = 0.0;
         g_horizon_regime_edge_ready[r][h] = false;
         g_horizon_regime_obs[r][h] = 0;
      }
   }
}

void FXAI_ResetReplayReservoir()
{
   g_replay_count = 0;
   g_replay_cursor = 0;
   for(int r=0; r<FXAI_REGIME_COUNT; r++)
   {
      for(int h=0; h<FXAI_MAX_HORIZONS; h++)
         g_replay_bucket_count[r][h] = 0;
   }
   for(int h=0; h<FXAI_MAX_HORIZONS; h++)
      g_replay_last_sample_time[h] = 0;
   for(int i=0; i<FXAI_REPLAY_CAPACITY; i++)
   {
      g_replay_used[i] = false;
      g_replay_priority[i] = 0.0;
      g_replay_flags[i] = 0;
      FXAI_ResetPreparedSample(g_replay_samples[i]);
   }
}

void FXAI_GetModelHyperParams(const int ai_idx, FXAIAIHyperParams &hp)
{
   if(ai_idx >= 0 && ai_idx < FXAI_AI_COUNT && g_model_hp_ready[ai_idx])
   {
      hp = g_model_hp[ai_idx];
      return;
   }
   FXAI_BuildHyperParams(hp);
   // Recommended GeodesicAttention starting defaults.
   if(ai_idx == (int)AI_GEODESICATTENTION)
   {
      hp.lr = 0.0060;
      hp.l2 = 0.0030;
   }
   // Recommended LSTM starting defaults.
   if(ai_idx == (int)AI_LSTM)
   {
      hp.lr = 0.0080;
      hp.l2 = 0.0040;
   }
   // Recommended LightGBM starting defaults.
   if(ai_idx == (int)AI_LIGHTGBM)
   {
      hp.xgb_lr = 0.0300;
      hp.xgb_l2 = 4.0000;
      hp.xgb_split = 0.0000;
   }
   // Recommended PA_LINEAR starting defaults.
   if(ai_idx == (int)AI_PA_LINEAR)
   {
      hp.lr = 0.0600;
      hp.l2 = 0.0030;
      hp.pa_c = 4.0000;
      hp.pa_margin = 1.2000;
   }
   // Recommended CFX_WORLD starting defaults.
   if(ai_idx == (int)AI_CFX_WORLD)
   {
      hp.lr = 0.0100;
      hp.l2 = 0.0020;
   }
   // Recommended LOFFM starting defaults.
   if(ai_idx == (int)AI_LOFFM)
   {
      hp.lr = 0.0080;
      hp.l2 = 0.0030;
   }
   // Recommended TRR starting defaults.
   if(ai_idx == (int)AI_TRR)
   {
      hp.lr = 0.0090;
      hp.l2 = 0.0025;
   }
   // Recommended GRAPHWM starting defaults.
   if(ai_idx == (int)AI_GRAPHWM)
   {
      hp.lr = 0.0080;
      hp.l2 = 0.0020;
   }
   // Recommended MOE_CONFORMAL starting defaults.
   if(ai_idx == (int)AI_MOE_CONFORMAL)
   {
      hp.lr = 0.0060;
      hp.l2 = 0.0030;
   }
   // Recommended M1SYNC starting defaults.
   if(ai_idx == (int)AI_M1SYNC)
   {
      hp.lr = 0.0;
      hp.l2 = 0.0;
   }
}

void FXAI_GetModelHyperParamsRouted(const int ai_idx,
                                    const int regime_id,
                                    const int horizon_minutes,
                                    FXAIAIHyperParams &hp)
{
   FXAI_GetModelHyperParams(ai_idx, hp);
   if(ai_idx < 0 || ai_idx >= FXAI_AI_COUNT) return;

   int hslot = FXAI_GetHorizonSlot(horizon_minutes);
   if(hslot >= 0 && hslot < FXAI_MAX_HORIZONS && g_model_hp_horizon_ready[ai_idx][hslot])
      hp = g_model_hp_horizon[ai_idx][hslot];

   if(regime_id >= 0 && regime_id < FXAI_REGIME_COUNT &&
      hslot >= 0 && hslot < FXAI_MAX_HORIZONS &&
      g_model_hp_bank_ready[ai_idx][regime_id][hslot])
   {
      hp = g_model_hp_bank[ai_idx][regime_id][hslot];
   }
}

void FXAI_GetModelThresholds(const int ai_idx,
                            const int regime_id,
                            const int horizon_minutes,
                            const double base_buy,
                            const double base_sell,
                            double &buy_thr,
                            double &sell_thr)
{
   buy_thr = base_buy;
   sell_thr = base_sell;
   FXAI_SanitizeThresholdPair(buy_thr, sell_thr);

   if(ai_idx < 0 || ai_idx >= FXAI_AI_COUNT) return;
   if(g_model_thr_ready[ai_idx])
   {
      buy_thr = g_model_buy_thr[ai_idx];
      sell_thr = g_model_sell_thr[ai_idx];
   }

   int hslot = FXAI_GetHorizonSlot(horizon_minutes);
   if(hslot >= 0 && hslot < FXAI_MAX_HORIZONS && g_model_thr_horizon_ready[ai_idx][hslot])
   {
      buy_thr = 0.55 * buy_thr + 0.45 * g_model_buy_thr_horizon[ai_idx][hslot];
      sell_thr = 0.55 * sell_thr + 0.45 * g_model_sell_thr_horizon[ai_idx][hslot];
   }

   if(regime_id >= 0 && regime_id < FXAI_REGIME_COUNT && g_model_thr_regime_ready[ai_idx][regime_id])
   {
      buy_thr = 0.65 * buy_thr + 0.35 * g_model_buy_thr_regime[ai_idx][regime_id];
      sell_thr = 0.65 * sell_thr + 0.35 * g_model_sell_thr_regime[ai_idx][regime_id];
   }

   if(regime_id >= 0 && regime_id < FXAI_REGIME_COUNT &&
      hslot >= 0 && hslot < FXAI_MAX_HORIZONS &&
      g_model_thr_bank_ready[ai_idx][regime_id][hslot])
   {
      buy_thr = 0.35 * buy_thr + 0.65 * g_model_buy_thr_bank[ai_idx][regime_id][hslot];
      sell_thr = 0.35 * sell_thr + 0.65 * g_model_sell_thr_bank[ai_idx][regime_id][hslot];
   }

   if(hslot >= 0 && hslot < FXAI_MAX_HORIZONS && g_model_horizon_edge_ready[ai_idx][hslot])
   {
      double edge = g_model_horizon_edge_ema[ai_idx][hslot];
      double adj = FXAI_Clamp(edge / MathMax(0.50, MathAbs(g_model_global_edge_ema[ai_idx]) + 0.50), -0.08, 0.08);
      buy_thr = FXAI_Clamp(buy_thr - (0.35 * adj), 0.50, 0.95);
      sell_thr = FXAI_Clamp(sell_thr + (0.35 * adj), 0.05, 0.50);
   }

   FXAI_SanitizeThresholdPair(buy_thr, sell_thr);
}

void FXAI_SampleThresholdPair(const double base_buy,
                             const double base_sell,
                             double &buy_thr,
                             double &sell_thr)
{
   double b0 = base_buy;
   double s0 = base_sell;
   FXAI_SanitizeThresholdPair(b0, s0);

   buy_thr = FXAI_Clamp(FXAI_RandRange(MathMax(0.52, b0 - 0.08), MathMin(0.90, b0 + 0.08)), 0.50, 0.95);
   sell_thr = FXAI_Clamp(FXAI_RandRange(MathMax(0.08, s0 - 0.08), MathMin(0.48, s0 + 0.08)), 0.05, 0.50);
   FXAI_SanitizeThresholdPair(buy_thr, sell_thr);
}

void FXAI_SampleModelHyperParams(const int ai_idx,
                                const FXAIAIHyperParams &base,
                                FXAIAIHyperParams &hp)
{
   hp = base;

      switch(ai_idx)
      {
         case (int)AI_SGD_LOGIT:
      case (int)AI_LSTMG:
      case (int)AI_S4:
      case (int)AI_TFT:
      case (int)AI_AUTOFORMER:
      case (int)AI_STMN:
      case (int)AI_TST:
      case (int)AI_PATCHTST:
         case (int)AI_CHRONOS:
         case (int)AI_TIMESFM:
         case (int)AI_CFX_WORLD:
         case (int)AI_LOFFM:
         case (int)AI_TRR:
         case (int)AI_GRAPHWM:
         case (int)AI_MOE_CONFORMAL:
         case (int)AI_RETRDIFF:
         hp.lr = FXAI_RandRange(0.0030, 0.0600);
         hp.l2 = FXAI_RandRange(0.0000, 0.0300);
         break;

      case (int)AI_M1SYNC:
         break;

      case (int)AI_LSTM:
         hp.lr = FXAI_RandRange(0.0040, 0.0200);
         hp.l2 = FXAI_RandRange(0.0010, 0.0100);
         break;

      case (int)AI_GEODESICATTENTION:
         hp.lr = FXAI_RandRange(0.0030, 0.0150);
         hp.l2 = FXAI_RandRange(0.0010, 0.0080);
         break;

      case (int)AI_TCN:
         hp.lr = FXAI_RandRange(0.0030, 0.0500);
         hp.l2 = FXAI_RandRange(0.0000, 0.0200);
         hp.tcn_layers = (double)((int)MathRound(FXAI_RandRange(3.0, 6.0)));
         hp.tcn_kernel = (double)((int)MathRound(FXAI_RandRange(2.0, 4.0)));
         hp.tcn_dilation_base = (double)((int)MathRound(FXAI_RandRange(1.0, 3.0)));
         break;

      case (int)AI_FTRL_LOGIT:
         hp.ftrl_alpha = FXAI_RandRange(0.0100, 0.2500);
         hp.ftrl_beta = FXAI_RandRange(0.1000, 2.5000);
         hp.ftrl_l1 = FXAI_RandRange(0.0000, 0.0100);
         hp.ftrl_l2 = FXAI_RandRange(0.0000, 0.1000);
         break;

      case (int)AI_PA_LINEAR:
         hp.lr = FXAI_RandRange(0.0200, 0.0800);
         hp.l2 = FXAI_RandRange(0.0010, 0.0100);
         hp.pa_c = FXAI_RandRange(0.5000, 6.0000);
         hp.pa_margin = FXAI_RandRange(0.6000, 1.9000);
         break;

      case (int)AI_XGB_FAST:
      case (int)AI_XGBOOST:
         hp.xgb_lr = FXAI_RandRange(0.0050, 0.1200);
         hp.xgb_l2 = FXAI_RandRange(0.0000, 0.0300);
         hp.xgb_split = FXAI_RandRange(-0.8000, 0.8000);
         break;

      case (int)AI_LIGHTGBM:
         hp.xgb_lr = FXAI_RandRange(0.0200, 0.0400);
         hp.xgb_l2 = FXAI_RandRange(2.0000, 6.0000);
         hp.xgb_split = FXAI_RandRange(-0.2000, 0.2000);
         break;

      case (int)AI_CATBOOST:
         hp.xgb_lr = FXAI_RandRange(0.0200, 0.0500);
         hp.xgb_l2 = FXAI_RandRange(3.0000, 8.0000);
         hp.xgb_split = FXAI_RandRange(-0.2000, 0.2000);
         break;

      case (int)AI_MLP_TINY:
         hp.mlp_lr = FXAI_RandRange(0.0010, 0.0300);
         hp.mlp_l2 = FXAI_RandRange(0.0000, 0.0200);
         hp.mlp_init = FXAI_RandRange(0.0300, 0.2500);
         break;

      case (int)AI_QUANTILE:
         hp.quantile_lr = FXAI_RandRange(0.0010, 0.0500);
         hp.quantile_l2 = FXAI_RandRange(0.0000, 0.0200);
         break;

      case (int)AI_ENHASH:
         hp.enhash_lr = FXAI_RandRange(0.0020, 0.0500);
         hp.enhash_l1 = FXAI_RandRange(0.0000, 0.0100);
         hp.enhash_l2 = FXAI_RandRange(0.0000, 0.0200);
         break;

      default:
         hp.lr = FXAI_RandRange(0.0030, 0.0600);
         hp.l2 = FXAI_RandRange(0.0000, 0.0300);
         break;
   }
}

void FXAI_ParseContextSymbols(const string raw, string &symbols[])
{
   ArrayResize(symbols, 0);

   string clean = raw;
   StringReplace(clean, "{", "");
   StringReplace(clean, "}", "");
   StringReplace(clean, ";", ",");
   StringReplace(clean, "|", ",");

   string parts[];
   int n = StringSplit(clean, ',', parts);
   if(n <= 0) return;

   for(int i=0; i<n; i++)
   {
      string sym = parts[i];
      StringTrimLeft(sym);
      StringTrimRight(sym);
      if(StringLen(sym) <= 0) continue;

      bool exists = false;
      for(int j=0; j<ArraySize(symbols); j++)
      {
         if(StringCompare(symbols[j], sym, false) == 0)
         {
            exists = true;
            break;
         }
      }
      if(exists) continue;

      int sz = ArraySize(symbols);
      ArrayResize(symbols, sz + 1);
      symbols[sz] = sym;
      if(ArraySize(symbols) >= FXAI_MAX_CONTEXT_SYMBOLS)
         break;
   }
}

void FXAI_FilterContextSymbols(const string main_symbol, string &symbols[])
{
   int n = ArraySize(symbols);
   if(n <= 0) return;

   int w = 0;
   for(int i=0; i<n; i++)
   {
      string sym = symbols[i];
      StringTrimLeft(sym);
      StringTrimRight(sym);
      if(StringLen(sym) <= 0) continue;
      if(StringCompare(sym, main_symbol, false) == 0) continue;
      if(!SymbolSelect(sym, true)) continue;

      symbols[w] = sym;
      w++;
   }

   if(w < n)
      ArrayResize(symbols, w);
}

double FXAI_ContextAlignedCorr(const double &main_close[],
                               const int main_i,
                               const FXAIContextSeries &ctx,
                               const int window)
{
   int n_main = ArraySize(main_close);
   int n_map = ArraySize(ctx.aligned_idx);
   if(window < 4 || main_i < 0 || main_i >= n_main || main_i >= n_map)
      return 0.0;

   double sx = 0.0, sy = 0.0, sxx = 0.0, syy = 0.0, sxy = 0.0;
   int used = 0;
   for(int k=0; k<window; k++)
   {
      int im = main_i + k;
      if(im + 1 >= n_main || im >= n_map) break;
      int ic = ctx.aligned_idx[im];
      if(ic < 0 || ic + 1 >= ArraySize(ctx.close)) continue;

      double xr = FXAI_SafeReturn(main_close, im, im + 1);
      double yr = FXAI_SafeReturn(ctx.close, ic, ic + 1);
      sx += xr;
      sy += yr;
      sxx += xr * xr;
      syy += yr * yr;
      sxy += xr * yr;
      used++;
   }

   if(used < 4) return 0.0;
   double cov = sxy - (sx * sy) / (double)used;
   double vx = sxx - (sx * sx) / (double)used;
   double vy = syy - (sy * sy) / (double)used;
   if(vx <= 1e-12 || vy <= 1e-12) return 0.0;
   return FXAI_Clamp(cov / MathSqrt(vx * vy), -1.0, 1.0);
}

void FXAI_PrecomputeContextAggregates(const datetime &main_time[],
                                     const double &main_close[],
                                     FXAIContextSeries &ctx_series[],
                                     const int ctx_count,
                                     const int upto_index,
                                     double &ctx_mean_arr[],
                                     double &ctx_std_arr[],
                                     double &ctx_up_arr[],
                                     double &ctx_extra_arr[])
{
   int n = ArraySize(main_time);
   if(ArraySize(main_close) != n) return;
   ArrayResize(ctx_mean_arr, n);
   ArrayResize(ctx_std_arr, n);
   ArrayResize(ctx_up_arr, n);
   ArrayResize(ctx_extra_arr, n * FXAI_CONTEXT_EXTRA_FEATS);
   if(n <= 0) return;

   int lag_m1 = 2 * PeriodSeconds(PERIOD_M1);
   if(lag_m1 <= 0) lag_m1 = 120;

   int upto = upto_index;
   if(upto < 0) upto = 0;
   if(upto >= n) upto = n - 1;
   // Keep one extra index ready for normalizers that need previous-bar features (i+1).
   int upto_fill = upto;
   if(upto_fill < n - 1) upto_fill++;

   for(int i=0; i<n; i++)
   {
      ctx_mean_arr[i] = 0.0;
      ctx_std_arr[i] = 0.0;
      ctx_up_arr[i] = 0.5;
   }
   for(int i=0; i<ArraySize(ctx_extra_arr); i++)
      ctx_extra_arr[i] = 0.0;

   for(int s=0; s<ctx_count; s++)
   {
      if(!ctx_series[s].loaded)
      {
         ArrayResize(ctx_series[s].aligned_idx, 0);
         continue;
      }
      FXAI_BuildAlignedIndexMapRange(main_time,
                                    ctx_series[s].time,
                                    lag_m1,
                                    upto_fill,
                                    ctx_series[s].aligned_idx);
   }

   for(int i=0; i<=upto_fill; i++)
   {
      datetime t_ref = main_time[i];
      if(t_ref <= 0 || ctx_count <= 0) continue;

      double main_ret = FXAI_SafeReturn(main_close, i, i + 1);
      double main_vol = FXAI_RollingAbsReturn(main_close, i, 20);
      if(main_vol < 1e-6) main_vol = MathAbs(main_ret);
      if(main_vol < 1e-6) main_vol = 1e-4;

      double weighted_sum = 0.0;
      double weighted_sum2 = 0.0;
      double weight_total = 0.0;
      double up_weight = 0.0;
      int valid = 0;
      double top_score[FXAI_CONTEXT_TOP_SYMBOLS];
      int top_symbol_idx[FXAI_CONTEXT_TOP_SYMBOLS];
      double top_ctx_ret[FXAI_CONTEXT_TOP_SYMBOLS];
      double top_ctx_lag[FXAI_CONTEXT_TOP_SYMBOLS];
      double top_ctx_rel[FXAI_CONTEXT_TOP_SYMBOLS];
      double top_ctx_corr[FXAI_CONTEXT_TOP_SYMBOLS];
      for(int t=0; t<FXAI_CONTEXT_TOP_SYMBOLS; t++)
      {
         top_score[t] = -1e18;
         top_symbol_idx[t] = -1;
         top_ctx_ret[t] = 0.0;
         top_ctx_lag[t] = 0.0;
         top_ctx_rel[t] = 0.0;
         top_ctx_corr[t] = 0.0;
      }

      for(int s=0; s<ctx_count; s++)
      {
         if(!ctx_series[s].loaded) continue;
         int idx = -1;
         if(i >= 0 && i < ArraySize(ctx_series[s].aligned_idx))
            idx = ctx_series[s].aligned_idx[i];
         if(idx < 0) continue;

         double freshness = FXAI_AlignedFreshnessWeight(ctx_series[s].time, idx, t_ref, lag_m1);
         double ctx_ret_raw = FXAI_SafeReturn(ctx_series[s].close, idx, idx + 1);
         double ctx_lag_raw = FXAI_SafeReturn(ctx_series[s].close, idx + 1, idx + 2);
         double ctx_rel_raw = ctx_ret_raw - main_ret;
         double ctx_corr_raw = FXAI_ContextAlignedCorr(main_close, i, ctx_series[s], 20);
         double rel_edge = FXAI_Clamp(MathAbs(ctx_rel_raw) / main_vol, 0.0, 4.0);
         double ret_edge = FXAI_Clamp(MathAbs(ctx_ret_raw) / main_vol, 0.0, 4.0);
         double lag_edge = FXAI_Clamp(MathAbs(ctx_lag_raw) / main_vol, 0.0, 4.0);
         double corr_edge = MathAbs(ctx_corr_raw);
         double symbol_score = freshness * ((0.40 * corr_edge) +
                                            (0.30 * rel_edge) +
                                            (0.20 * lag_edge) +
                                            (0.10 * ret_edge));

         double w = 0.20 + symbol_score;
         weighted_sum += w * ctx_ret_raw;
         weighted_sum2 += w * ctx_ret_raw * ctx_ret_raw;
         weight_total += w;
         if(ctx_ret_raw > 0.0) up_weight += w;
         valid++;

         for(int slot=0; slot<FXAI_CONTEXT_TOP_SYMBOLS; slot++)
         {
            if(symbol_score <= top_score[slot]) continue;
            for(int shift=FXAI_CONTEXT_TOP_SYMBOLS - 1; shift>slot; shift--)
            {
               top_score[shift] = top_score[shift - 1];
               top_symbol_idx[shift] = top_symbol_idx[shift - 1];
               top_ctx_ret[shift] = top_ctx_ret[shift - 1];
               top_ctx_lag[shift] = top_ctx_lag[shift - 1];
               top_ctx_rel[shift] = top_ctx_rel[shift - 1];
               top_ctx_corr[shift] = top_ctx_corr[shift - 1];
            }
            top_score[slot] = symbol_score;
            top_symbol_idx[slot] = s;
            top_ctx_ret[slot] = ctx_ret_raw * freshness;
            top_ctx_lag[slot] = ctx_lag_raw * freshness;
            top_ctx_rel[slot] = ctx_rel_raw * freshness;
            top_ctx_corr[slot] = ctx_corr_raw * freshness;
            break;
         }
      }

      if(valid <= 0 || weight_total <= 0.0) continue;

      double mean = weighted_sum / weight_total;
      double var = (weighted_sum2 / weight_total) - (mean * mean);
      if(var < 0.0) var = 0.0;
      double up_ratio = up_weight / weight_total;

      double coverage = (ctx_count > 0 ? ((double)valid / (double)ctx_count) : 0.0);
      coverage = FXAI_Clamp(coverage, 0.0, 1.0);
      double conf = 0.30 + (0.70 * coverage);

      ctx_mean_arr[i] = mean * coverage;
      ctx_std_arr[i] = MathSqrt(var) * conf;
      ctx_up_arr[i] = 0.5 + ((up_ratio - 0.5) * coverage);

      for(int top_slot=0; top_slot<FXAI_CONTEXT_TOP_SYMBOLS; top_slot++)
      {
         if(top_symbol_idx[top_slot] < 0) continue;
         FXAI_SetContextExtraValue(ctx_extra_arr, i, top_slot * 4 + 0, top_ctx_ret[top_slot]);
         FXAI_SetContextExtraValue(ctx_extra_arr, i, top_slot * 4 + 1, top_ctx_lag[top_slot]);
         FXAI_SetContextExtraValue(ctx_extra_arr, i, top_slot * 4 + 2, top_ctx_rel[top_slot]);
         FXAI_SetContextExtraValue(ctx_extra_arr, i, top_slot * 4 + 3, top_ctx_corr[top_slot]);
      }
   }
}

//--------------------------- INIT -----------------------------------
void ResetAIState(const string symbol)
{
   g_ai_last_symbol = symbol;
   g_ai_last_signal_bar = 0;
   g_ai_last_signal = -1;
   g_ai_last_signal_key = -1;
   g_ai_warmup_done = (!AI_Warmup);
   FXAI_ParseHorizonList(AI_Horizons, PredictionTargetMinutes, g_horizon_minutes);
   FXAI_ResetModelHyperParams();
   FXAI_ResetReliabilityPending();
   FXAI_ResetHorizonPolicyPending();
   FXAI_ResetStackPending();
   FXAI_ResetAdaptiveRoutingState();
   FXAI_ResetRegimeCalibration();
   FXAI_ResetReplayReservoir();

   if(!g_norm_windows_ready)
   {
      int windows[];
      int default_w = FXAI_GetNormDefaultWindow();
      FXAI_BuildNormWindowsFromGroups(default_w, default_w, default_w, default_w, windows);
      FXAI_ApplyNormWindows(windows, default_w);
   }
   else
   {
      FXAI_ApplyNormWindows(g_norm_feature_windows, g_norm_default_window);
   }

   for(int i=0; i<FXAI_AI_COUNT; i++)
   {
      g_ai_trained[i] = false;
      g_ai_last_train_bar[i] = 0;
      FXAI_ResetModelAuxState(i);
   }

   if(g_plugins_ready)
      g_plugins.ResetAll();
}

bool FXAI_ValidateNativePluginAPI()
{
   double x_dummy[FXAI_AI_WEIGHTS];
   for(int k=0; k<FXAI_AI_WEIGHTS; k++) x_dummy[k] = 0.0;
   x_dummy[0] = 1.0;

   for(int ai_idx=0; ai_idx<FXAI_AI_COUNT; ai_idx++)
   {
      CFXAIAIPlugin *plugin = g_plugins.Get(ai_idx);
      if(plugin == NULL)
      {
         Print("FXAI error: API v2 plugin missing at id=", ai_idx);
         return false;
      }

      FXAIAIHyperParams hp;
      FXAI_GetModelHyperParams(ai_idx, hp);
      plugin.EnsureInitialized(hp);

      if(!plugin.SupportsNativeClassProbs())
      {
         Print("FXAI error: API v2 requires native 3-class support. model=", plugin.AIName(),
               " id=", ai_idx);
         return false;
      }

      double probs[3];
      probs[0] = 0.0; probs[1] = 0.0; probs[2] = 0.0;
      double expected_move = 0.0;
      if(!plugin.PredictNativeClassProbs(x_dummy, hp, probs, expected_move))
      {
         Print("FXAI error: model native 3-class path returned false. model=", plugin.AIName(),
               " id=", ai_idx);
         return false;
      }

      if(!MathIsValidNumber(probs[0]) || !MathIsValidNumber(probs[1]) || !MathIsValidNumber(probs[2]))
      {
         Print("FXAI error: model native probabilities invalid. model=", plugin.AIName(),
               " id=", ai_idx);
         return false;
      }
      if(probs[0] < 0.0 || probs[1] < 0.0 || probs[2] < 0.0)
      {
         Print("FXAI error: model native probabilities negative. model=", plugin.AIName(),
               " id=", ai_idx);
         return false;
      }
      double s = probs[0] + probs[1] + probs[2];
      if(!MathIsValidNumber(s) || s <= 0.0)
      {
         Print("FXAI error: model native probabilities degenerate. model=", plugin.AIName(),
               " id=", ai_idx);
         return false;
      }
   }

   return true;
}

void FXAI_FillComplianceSample(FXAIAISampleV2 &sample,
                               const int label_class,
                               const double move_points,
                               const double cost_points,
                               const double v1,
                               const double v2,
                               const double v3,
                               const datetime sample_time,
                               const int regime_id,
                               const int horizon_minutes)
{
   sample.valid = true;
   sample.label_class = label_class;
   sample.regime_id = regime_id;
   sample.horizon_minutes = horizon_minutes;
   sample.move_points = move_points;
   sample.min_move_points = MathMax(cost_points + 0.30, 0.50);
   sample.cost_points = MathMax(cost_points, 0.0);
   sample.sample_time = sample_time;
   for(int k=0; k<FXAI_AI_WEIGHTS; k++)
      sample.x[k] = 0.0;
   sample.x[0] = 1.0;
   sample.x[1] = v1;
   sample.x[2] = v2;
   sample.x[3] = v3;
   sample.x[4] = 0.35 * v2;
   sample.x[5] = 0.25 * v3;
   sample.x[6] = 0.15 * v1;
   sample.x[7] = sample.cost_points;
}

void FXAI_FillComplianceRequest(FXAIAIPredictV2 &req,
                                const double cost_points,
                                const double v1,
                                const double v2,
                                const double v3,
                                const datetime sample_time,
                                const int regime_id,
                                const int horizon_minutes)
{
   req.regime_id = regime_id;
   req.horizon_minutes = horizon_minutes;
   req.min_move_points = MathMax(cost_points + 0.30, 0.50);
   req.cost_points = MathMax(cost_points, 0.0);
   req.sample_time = sample_time;
   for(int k=0; k<FXAI_AI_WEIGHTS; k++)
      req.x[k] = 0.0;
   req.x[0] = 1.0;
   req.x[1] = v1;
   req.x[2] = v2;
   req.x[3] = v3;
   req.x[4] = 0.35 * v2;
   req.x[5] = 0.25 * v3;
   req.x[6] = 0.15 * v1;
   req.x[7] = req.cost_points;
}

bool FXAI_ValidatePredictionOutput(const CFXAIAIPlugin &plugin,
                                   const FXAIAIPredictionV2 &pred,
                                   const string tag)
{
   double s = pred.class_probs[(int)FXAI_LABEL_SELL]
            + pred.class_probs[(int)FXAI_LABEL_BUY]
            + pred.class_probs[(int)FXAI_LABEL_SKIP];
   if(!MathIsValidNumber(s) || MathAbs(s - 1.0) > 1e-3)
   {
      Print("FXAI compliance error: probability sum invalid. model=", plugin.AIName(),
            " tag=", tag, " sum=", DoubleToString(s, 6));
      return false;
   }

   for(int c=0; c<3; c++)
   {
      if(!MathIsValidNumber(pred.class_probs[c]) || pred.class_probs[c] < 0.0 || pred.class_probs[c] > 1.0)
      {
         Print("FXAI compliance error: probability range invalid. model=", plugin.AIName(),
               " tag=", tag, " class=", c, " value=", DoubleToString(pred.class_probs[c], 6));
         return false;
      }
   }

   if(!MathIsValidNumber(pred.expected_move_points) || pred.expected_move_points <= 0.0)
   {
      Print("FXAI compliance error: expected move invalid. model=", plugin.AIName(),
            " tag=", tag, " ev=", DoubleToString(pred.expected_move_points, 6));
      return false;
   }

   return true;
}

bool FXAI_RunPluginComplianceHarness()
{
   datetime now_t = TimeCurrent();

   for(int ai_idx=0; ai_idx<FXAI_AI_COUNT; ai_idx++)
   {
      CFXAIAIPlugin *plugin = g_plugins.CreateInstance(ai_idx);
      if(plugin == NULL)
      {
         Print("FXAI compliance error: could not create plugin id=", ai_idx);
         return false;
      }

      FXAIAIHyperParams hp;
      FXAI_GetModelHyperParams(ai_idx, hp);
      plugin.Reset();
      plugin.EnsureInitialized(hp);

      FXAIAISampleV2 buy_s, sell_s, skip_s, buy_big_s;
      FXAI_FillComplianceSample(buy_s, (int)FXAI_LABEL_BUY, 4.5, 0.8, 0.75, 0.40, 0.20, now_t - 180, 1, 5);
      FXAI_FillComplianceSample(sell_s, (int)FXAI_LABEL_SELL, -4.5, 0.8, -0.75, -0.40, -0.20, now_t - 120, 1, 5);
      FXAI_FillComplianceSample(skip_s, (int)FXAI_LABEL_SKIP, 0.2, 0.8, 0.02, 0.01, 0.00, now_t - 60, 1, 5);
      FXAI_FillComplianceSample(buy_big_s, (int)FXAI_LABEL_BUY, 8.0, 0.8, 1.20, 0.65, 0.35, now_t - 30, 1, 13);

      for(int rep=0; rep<10; rep++)
      {
         plugin.TrainV2(buy_s, hp);
         plugin.TrainV2(sell_s, hp);
         plugin.TrainV2(skip_s, hp);
         plugin.TrainV2(buy_big_s, hp);
      }

      FXAIAIPredictV2 req_buy_lo, req_buy_hi, req_sell_lo, req_skip_lo, req_buy_big;
      FXAI_FillComplianceRequest(req_buy_lo, 0.8, 0.75, 0.40, 0.20, now_t, 1, 5);
      FXAI_FillComplianceRequest(req_buy_hi, 3.5, 0.75, 0.40, 0.20, now_t, 1, 5);
      FXAI_FillComplianceRequest(req_sell_lo, 0.8, -0.75, -0.40, -0.20, now_t, 1, 5);
      FXAI_FillComplianceRequest(req_skip_lo, 0.8, 0.02, 0.01, 0.00, now_t, 1, 5);
      FXAI_FillComplianceRequest(req_buy_big, 0.8, 1.20, 0.65, 0.35, now_t, 1, 13);

      FXAIAIPredictionV2 pred_buy_lo, pred_buy_hi, pred_sell_lo, pred_skip_lo, pred_buy_big;
      plugin.PredictV2(req_buy_lo, hp, pred_buy_lo);
      plugin.PredictV2(req_buy_hi, hp, pred_buy_hi);
      plugin.PredictV2(req_sell_lo, hp, pred_sell_lo);
      plugin.PredictV2(req_skip_lo, hp, pred_skip_lo);
      plugin.PredictV2(req_buy_big, hp, pred_buy_big);

      bool ok = FXAI_ValidatePredictionOutput(*plugin, pred_buy_lo, "buy_lo")
             && FXAI_ValidatePredictionOutput(*plugin, pred_buy_hi, "buy_hi")
             && FXAI_ValidatePredictionOutput(*plugin, pred_sell_lo, "sell_lo")
             && FXAI_ValidatePredictionOutput(*plugin, pred_skip_lo, "skip_lo")
             && FXAI_ValidatePredictionOutput(*plugin, pred_buy_big, "buy_big");
      if(!ok)
      {
         delete plugin;
         return false;
      }

      if(pred_buy_lo.class_probs[(int)FXAI_LABEL_BUY] + 0.05 < pred_buy_lo.class_probs[(int)FXAI_LABEL_SELL])
      {
         Print("FXAI compliance error: buy ordering failed. model=", plugin.AIName());
         delete plugin;
         return false;
      }
      if(pred_sell_lo.class_probs[(int)FXAI_LABEL_SELL] + 0.05 < pred_sell_lo.class_probs[(int)FXAI_LABEL_BUY])
      {
         Print("FXAI compliance error: sell ordering failed. model=", plugin.AIName());
         delete plugin;
         return false;
      }
      if(pred_skip_lo.class_probs[(int)FXAI_LABEL_SKIP] < 0.20)
      {
         Print("FXAI compliance error: skip response too weak. model=", plugin.AIName());
         delete plugin;
         return false;
      }

      double actionable_lo = pred_buy_lo.class_probs[(int)FXAI_LABEL_BUY] + pred_buy_lo.class_probs[(int)FXAI_LABEL_SELL];
      double actionable_hi = pred_buy_hi.class_probs[(int)FXAI_LABEL_BUY] + pred_buy_hi.class_probs[(int)FXAI_LABEL_SELL];
      if(actionable_hi > actionable_lo + 0.20)
      {
         Print("FXAI compliance error: cost awareness failed. model=", plugin.AIName(),
               " low=", DoubleToString(actionable_lo, 4),
               " high=", DoubleToString(actionable_hi, 4));
         delete plugin;
         return false;
      }

      if(pred_buy_big.expected_move_points + 0.25 < pred_buy_lo.expected_move_points)
      {
         Print("FXAI compliance error: EV monotonicity failed. model=", plugin.AIName(),
               " big=", DoubleToString(pred_buy_big.expected_move_points, 4),
               " base=", DoubleToString(pred_buy_lo.expected_move_points, 4));
         delete plugin;
         return false;
      }

      if(plugin.NativePredictFailures() > 0)
      {
         Print("FXAI compliance error: native predict failures detected. model=", plugin.AIName(),
               " failures=", plugin.NativePredictFailures());
         delete plugin;
         return false;
      }

      delete plugin;
   }

   return true;
}

int OnInit()
{
   MathSrand((uint)TimeLocal());

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
   TP_Value      = InitialEquity + TP_USD;

   g_plugins_ready = g_plugins.Initialize();
   if(!g_plugins_ready)
      return(INIT_FAILED);

   if(!FXAI_ValidateNativePluginAPI())
      return(INIT_PARAMETERS_INCORRECT);

   if(AI_ComplianceHarness && !FXAI_RunPluginComplianceHarness())
      return(INIT_PARAMETERS_INCORRECT);

   FXAI_ParseContextSymbols(AI_ContextSymbols, g_context_symbols);
   FXAI_FilterContextSymbols(_Symbol, g_context_symbols);

   ResetAIState(_Symbol);
   return(INIT_SUCCEEDED);
}

void OnDeinit(const int reason)
{
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

//------------------------- UTILS ------------------------------------
void Calc_TP()
{
   double tp_usd = (TP_USD > 0.0 ? TP_USD : 0.0);
   TP_Value = InitialEquity + (tp_usd * CloseCounter);
}

void ResetCycleState()
{
   CycleActive      = false;
   CycleEntryEquity = 0.0;
   CycleStartTime   = 0;

   TrailTracking    = false;
   TrailPeakProfit  = 0.0;
}

datetime FXAI_GetOldestPositionTime()
{
   int total = PositionsTotal();
   if(total <= 0) return 0;

   datetime oldest = 0;
   for(int i=0; i<total; i++)
   {
      ulong ticket = PositionGetTicket(i);
      if(ticket == 0) continue;

      datetime t = (datetime)PositionGetInteger(POSITION_TIME);
      if(t <= 0) continue;
      if(oldest == 0 || t < oldest) oldest = t;
   }
   return oldest;
}

double FXAI_NormalizeLot(const string symbol, const double requested_lot)
{
   double vmin  = SymbolInfoDouble(symbol, SYMBOL_VOLUME_MIN);
   double vmax  = SymbolInfoDouble(symbol, SYMBOL_VOLUME_MAX);
   double vstep = SymbolInfoDouble(symbol, SYMBOL_VOLUME_STEP);

   if(vstep <= 0.0) vstep = 0.01;
   if(vmin <= 0.0) vmin = vstep;
   if(vmax < vmin) vmax = vmin;

   double lot = requested_lot;
   if(!MathIsValidNumber(lot) || lot <= 0.0)
      lot = vmin;

   lot = FXAI_Clamp(lot, vmin, vmax);
   lot = vmin + MathFloor(((lot - vmin) / vstep) + 1e-9) * vstep;

   if(lot < vmin) lot = vmin;
   if(lot > vmax) lot = vmax;

   return NormalizeDouble(lot, 8);
}

//------------------------- CLOSE ALL --------------------------------
void CloseAll()
{
   const int max_passes = 25;
   bool in_tester = (MQLInfoInteger(MQL_TESTER) != 0);
   int op_pause_ms = (in_tester ? 0 : 100);

   for(int pass=0; pass<max_passes; pass++)
   {
      int pos_before = PositionsTotal();
      int ord_before = OrdersTotal();

      for(int i=pos_before - 1; i>=0; i--)
      {
         ulong ticket = PositionGetTicket(i);
         if(ticket == 0) continue;
         trade.PositionClose(ticket);
         if(op_pause_ms > 0) Sleep(op_pause_ms);
      }

      for(int i=ord_before - 1; i>=0; i--)
      {
         ulong ticket = OrderGetTicket(i);
         if(ticket == 0) continue;

         long orderType = OrderGetInteger(ORDER_TYPE);
         if(orderType == ORDER_TYPE_BUY_LIMIT      || orderType == ORDER_TYPE_SELL_LIMIT ||
            orderType == ORDER_TYPE_BUY_STOP       || orderType == ORDER_TYPE_SELL_STOP  ||
            orderType == ORDER_TYPE_BUY_STOP_LIMIT || orderType == ORDER_TYPE_SELL_STOP_LIMIT)
         {
            trade.OrderDelete(ticket);
            if(op_pause_ms > 0) Sleep(op_pause_ms);
         }
      }

      int pos_after = PositionsTotal();
      int ord_after = OrdersTotal();
      if(pos_after == 0 && ord_after == 0)
         return;

      // no progress; avoid hard lock in OnTick if broker rejects close/delete
      if(pos_after >= pos_before && ord_after >= ord_before)
         break;
   }

   Print("FXAI warning: CloseAll incomplete. Remaining positions=", PositionsTotal(),
         ", orders=", OrdersTotal());
}

//---------------------- TRADE POSSIBLE ------------------------------
int TradePossible(const string symbol, string &reason)
{
   bool in_tester = (MQLInfoInteger(MQL_TESTER) != 0);
   reason = "ok";

   if(!SymbolSelect(symbol, true))
   {
      reason = "symbol_select_failed";
      return 0;
   }

   long tradeMode = SymbolInfoInteger(symbol, SYMBOL_TRADE_MODE);
   if(tradeMode == SYMBOL_TRADE_MODE_DISABLED)
   {
      reason = "trade_mode_disabled";
      return 0;
   }

   if(!in_tester && !TerminalInfoInteger(TERMINAL_TRADE_ALLOWED))
   {
      reason = "terminal_trade_not_allowed";
      return 0;
   }
   if(!in_tester && !TerminalInfoInteger(TERMINAL_CONNECTED))
   {
      reason = "terminal_not_connected";
      return 0;
   }

   double bid = SymbolInfoDouble(symbol, SYMBOL_BID);
   double ask = SymbolInfoDouble(symbol, SYMBOL_ASK);
   if(bid <= 0 || ask <= 0)
   {
      reason = "invalid_bid_ask";
      return 0;
   }

   MqlTick last_tick;
   if(!SymbolInfoTick(symbol, last_tick))
   {
      reason = "symbol_tick_failed";
      return 0;
   }

   datetime lastTickTime = last_tick.time;
   datetime currentTime  = TimeCurrent();
   if(!in_tester && currentTime - lastTickTime > 10)
   {
      reason = "stale_tick";
      return 0;
   }

   if(SessionFilterEnabled)
   {
      if(!FXAI_IsInLiquidSession(symbol,
                                currentTime,
                                SessionMinAfterOpenMinutes,
                                SessionMinBeforeCloseMinutes))
      {
         reason = "session_filter_block";
         return 0;
      }
   }

   return 1;
}

//------------------------- EA HARD EXIT -----------------------------
void HardExit()
{
   if (MQLInfoInteger(MQL_TESTER)) TesterStop();
   else ExpertRemove();
}

void EAStop()
{
   if(EquiMax <= 0.0) return;

   double eq = AccountInfoDouble(ACCOUNT_EQUITY);
   if(eq > EquiMax) EquiMax = eq;

   double maxdd = FXAI_Clamp(MaxDD, 0.0, 99.9);
   if(maxdd <= 0.0) return;

   if((eq / EquiMax) < ((100.0 - maxdd) / 100.0))
      HardExit();
}

//--------------------------- TP CHECK -------------------------------
void TPCheck()
{
   if(AccountInfoDouble(ACCOUNT_EQUITY) < 100) TP_Value = 110;

   if(AccountInfoDouble(ACCOUNT_EQUITY) > TP_Value)
   {
      CloseAll();
      ResetCycleState();

      CloseCounter++;
      Calc_TP();
   }
}

void TradeKillerManage()
{
   if(TradeKiller <= 0) return;
   if(PositionsTotal() <= 0) return;

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
      CloseAll();
      ResetCycleState();
      Calc_TP();
   }
}

//--------------------- EQUITY SL MANAGER ----------------------------
void EquitySLManage()
{
   if(SL_USD <= 0.0) return;
   if(PositionsTotal() <= 0) return;
   if(!CycleActive) return;

   double eq = AccountInfoDouble(ACCOUNT_EQUITY);
   double dd = eq - CycleEntryEquity;

   if(dd <= -SL_USD)
   {
      CloseAll();
      ResetCycleState();
      Calc_TP();
   }
}

//--------------------- EQUITY TRAILING STOP -------------------------
void EquityTrailManage()
{
   if(!TrailEnabled) return;
   if(PositionsTotal() <= 0) return;
   if(!CycleActive) return;

   if(!TrailTracking)
   {
      TrailTracking = true;
      TrailPeakProfit = 0.0;
   }

   double eq = AccountInfoDouble(ACCOUNT_EQUITY);
   double profit = eq - CycleEntryEquity;

   if(profit > TrailPeakProfit) TrailPeakProfit = profit;
   double trail_start = (TrailStartUSD > 0.0 ? TrailStartUSD : 0.0);
   if(TrailPeakProfit < trail_start) return;

   double tp_breath = (TrailTPBreathUSD > 0.0 ? TrailTPBreathUSD : 0.0);
   double desiredTP = CycleEntryEquity + TrailPeakProfit + tp_breath;
   if(desiredTP > TP_Value) TP_Value = desiredTP;

   double giveback = TrailGivebackPct;
   if(giveback < 0.0) giveback = 0.0;
   if(giveback > 99.0) giveback = 99.0;

   double lockedProfit = TrailPeakProfit * (1.0 - (giveback / 100.0));
   double stopEquity = CycleEntryEquity + lockedProfit;

   if(eq <= stopEquity)
   {
      CloseAll();
      ResetCycleState();
      Calc_TP();
   }
}

//--------------------------- AI SIGNAL -------------------------------
int SpecialDirectionAI(const string symbol)
{
   g_ai_last_reason = "start";
   if(!g_plugins_ready)
   {
      g_ai_last_reason = "plugins_not_ready";
      return -1;
   }

   int base_h = FXAI_ClampHorizon(PredictionTargetMinutes);

   int base = AI_Window;
   if(base < 50) base = 50;
   if(base > 500) base = 500;

   int K = AI_OnlineSamples;
   if(K < 5) K = 5;
   if(K > 200) K = 200;

   int onlineEpochs = AI_OnlineEpochs;
   if(onlineEpochs < 1) onlineEpochs = 1;
   if(onlineEpochs > 5) onlineEpochs = 5;

   int trainEpochs = AI_Epochs;
   if(trainEpochs < 1) trainEpochs = 1;
   if(trainEpochs > 20) trainEpochs = 20;

   int aiType = (int)AI_Type;
   if(aiType < 0 || aiType >= FXAI_AI_COUNT)
      aiType = (int)AI_SGD_LOGIT;

   bool ensembleMode = (bool)AI_Ensemble;
   double agreePct = FXAI_Clamp(Ensemble_AgreePct, 50.0, 100.0);

   double buyThr = AI_BuyThreshold;
   double sellThr = AI_SellThreshold;
   FXAI_SanitizeThresholdPair(buyThr, sellThr);

   double evThresholdPoints = FXAI_Clamp(AI_EVThresholdPoints, 0.0, 100.0);
   int evLookback = AI_EVLookbackSamples;
   if(evLookback < 20) evLookback = 20;
   if(evLookback > 400) evLookback = 400;

   if(g_ai_last_symbol != symbol)
      ResetAIState(symbol);

   if(AI_Warmup && !g_ai_warmup_done)
   {
      if(!FXAI_WarmupTrainAndTune(symbol))
      {
         g_ai_last_reason = "warmup_pending";
         return -1;
      }
   }

   datetime signal_bar = iTime(symbol, PERIOD_M1, 1);
   if(signal_bar == 0)
   {
      g_ai_last_reason = "bar_time_failed";
      return -1;
   }

   int pctKey = (int)MathRound(agreePct * 10.0);
   int decisionKey = (ensembleMode == 1 ? (100000 + pctKey) : aiType);
   if(g_ai_last_signal_bar == signal_bar && g_ai_last_signal_key == decisionKey)
   {
      g_ai_last_reason = "signal_cache_hit";
      return g_ai_last_signal;
   }

   FXAIDataSnapshot snapshot;
   if(!FXAI_ExportDataSnapshot(symbol, AI_CommissionPerLotSide, AI_CostBufferPoints, snapshot))
   {
      g_ai_last_reason = "snapshot_export_failed";
      return -1;
   }
   // Keep cache/training keyed to the same closed bar anchor.
   snapshot.bar_time = signal_bar;

   const int FEATURE_LB = 10;
   int horizon_load_max = FXAI_GetMaxConfiguredHorizon(base_h);
   int needed = (K > base ? K : base) + horizon_load_max + FEATURE_LB;
   if(needed < 128) needed = 128;
   int align_upto = needed - 1;

   static MqlRates rates_m1[];
   static MqlRates rates_m5[];
   static MqlRates rates_m15[];
   static MqlRates rates_m30[];
   static MqlRates rates_h1[];
   static string cache_symbol = "";
   static datetime last_bar_m1 = 0;
   static datetime last_bar_m5 = 0;
   static datetime last_bar_m15 = 0;
   static datetime last_bar_m30 = 0;
   static datetime last_bar_h1 = 0;

   static double open_arr[];
   static double high_arr[];
   static double low_arr[];
   static double close_arr[];
   static datetime time_arr[];
   static int spread_m1[];
   static FXAIContextSeries ctx_series[];
   static double ctx_mean_arr[];
   static double ctx_std_arr[];
   static double ctx_up_arr[];
   static double ctx_extra_arr[];

   if(cache_symbol != symbol)
   {
      cache_symbol = symbol;
      last_bar_m1 = 0;
      last_bar_m5 = 0;
      last_bar_m15 = 0;
      last_bar_m30 = 0;
      last_bar_h1 = 0;
      ArrayResize(rates_m1, 0);
      ArrayResize(rates_m5, 0);
      ArrayResize(rates_m15, 0);
      ArrayResize(rates_m30, 0);
      ArrayResize(rates_h1, 0);
      ArrayResize(ctx_series, 0);
   }

   FXAI_AdvanceReliabilityClock(signal_bar);
   int signal_seq = g_rel_clock_seq;

   if(!FXAI_UpdateRatesRolling(symbol, PERIOD_M1, needed, last_bar_m1, rates_m1))
   {
      g_ai_last_reason = "m1_series_load_failed";
      return -1;
   }
   FXAI_ExtractRatesCloseTimeSpread(rates_m1, close_arr, time_arr, spread_m1);
   FXAI_ExtractRatesOHLC(rates_m1, open_arr, high_arr, low_arr, close_arr);
   if(ArraySize(close_arr) < needed || ArraySize(time_arr) < needed || ArraySize(spread_m1) < needed)
   {
      g_ai_last_reason = "m1_series_size_failed";
      return -1;
   }

   int needed_m5 = (needed / 5) + 80;
   int needed_m15 = (needed / 15) + 80;
   int needed_m30 = (needed / 30) + 80;
   int needed_h1 = (needed / 60) + 80;
   if(needed_m5 < 220) needed_m5 = 220;
   if(needed_m15 < 220) needed_m15 = 220;
   if(needed_m30 < 220) needed_m30 = 220;
   if(needed_h1 < 220) needed_h1 = 220;

   static double close_m5[];
   static datetime time_m5[];
   static double close_m15[];
   static datetime time_m15[];
   static double close_m30[];
   static datetime time_m30[];
   static double close_h1[];
   static datetime time_h1[];
   static int map_m5[];
   static int map_m15[];
   static int map_m30[];
   static int map_h1[];
   if(FXAI_UpdateRatesRolling(symbol, PERIOD_M5, needed_m5, last_bar_m5, rates_m5))
      FXAI_ExtractRatesCloseTime(rates_m5, close_m5, time_m5);
   else
   {
      ArrayResize(close_m5, 0);
      ArrayResize(time_m5, 0);
      ArrayResize(map_m5, 0);
   }

   if(FXAI_UpdateRatesRolling(symbol, PERIOD_M15, needed_m15, last_bar_m15, rates_m15))
      FXAI_ExtractRatesCloseTime(rates_m15, close_m15, time_m15);
   else
   {
      ArrayResize(close_m15, 0);
      ArrayResize(time_m15, 0);
      ArrayResize(map_m15, 0);
   }

   if(FXAI_UpdateRatesRolling(symbol, PERIOD_M30, needed_m30, last_bar_m30, rates_m30))
      FXAI_ExtractRatesCloseTime(rates_m30, close_m30, time_m30);
   else
   {
      ArrayResize(close_m30, 0);
      ArrayResize(time_m30, 0);
      ArrayResize(map_m30, 0);
   }

   if(FXAI_UpdateRatesRolling(symbol, PERIOD_H1, needed_h1, last_bar_h1, rates_h1))
      FXAI_ExtractRatesCloseTime(rates_h1, close_h1, time_h1);
   else
   {
      ArrayResize(close_h1, 0);
      ArrayResize(time_h1, 0);
      ArrayResize(map_h1, 0);
   }

   int lag_m5 = 2 * PeriodSeconds(PERIOD_M5);
   int lag_m15 = 2 * PeriodSeconds(PERIOD_M15);
   int lag_m30 = 2 * PeriodSeconds(PERIOD_M30);
   int lag_h1 = 2 * PeriodSeconds(PERIOD_H1);
   if(lag_m5 <= 0) lag_m5 = 600;
   if(lag_m15 <= 0) lag_m15 = 1800;
   if(lag_m30 <= 0) lag_m30 = 3600;
   if(lag_h1 <= 0) lag_h1 = 7200;

   FXAI_BuildAlignedIndexMapRange(time_arr, time_m5, lag_m5, align_upto, map_m5);
   FXAI_BuildAlignedIndexMapRange(time_arr, time_m15, lag_m15, align_upto, map_m15);
   FXAI_BuildAlignedIndexMapRange(time_arr, time_m30, lag_m30, align_upto, map_m30);
   FXAI_BuildAlignedIndexMapRange(time_arr, time_h1, lag_h1, align_upto, map_h1);

   int ctx_count = ArraySize(g_context_symbols);
   if(ctx_count > FXAI_MAX_CONTEXT_SYMBOLS) ctx_count = FXAI_MAX_CONTEXT_SYMBOLS;
   if(ArraySize(ctx_series) != ctx_count)
   {
      ArrayResize(ctx_series, ctx_count);
      for(int s=0; s<ctx_count; s++)
      {
         ctx_series[s].loaded = false;
         ctx_series[s].symbol = "";
         ctx_series[s].last_bar_time = 0;
         ArrayResize(ctx_series[s].rates, 0);
         ArrayResize(ctx_series[s].close, 0);
         ArrayResize(ctx_series[s].time, 0);
         ArrayResize(ctx_series[s].aligned_idx, 0);
      }
   }
   for(int s=0; s<ctx_count; s++)
   {
      string ctx_symbol = g_context_symbols[s];
      if(ctx_series[s].symbol != ctx_symbol)
      {
         ctx_series[s].symbol = ctx_symbol;
         ctx_series[s].last_bar_time = 0;
         ArrayResize(ctx_series[s].rates, 0);
      }

      ctx_series[s].loaded = FXAI_UpdateRatesRolling(ctx_symbol,
                                                    PERIOD_M1,
                                                    needed,
                                                    ctx_series[s].last_bar_time,
                                                    ctx_series[s].rates);
      if(ctx_series[s].loaded)
      {
         FXAI_ExtractRatesCloseTime(ctx_series[s].rates,
                                   ctx_series[s].close,
                                   ctx_series[s].time);
      }
      else
      {
         ArrayResize(ctx_series[s].close, 0);
         ArrayResize(ctx_series[s].time, 0);
         ArrayResize(ctx_series[s].aligned_idx, 0);
      }
   }

   FXAI_PrecomputeContextAggregates(time_arr,
                                   close_arr,
                                   ctx_series,
                                   ctx_count,
                                   align_upto,
                                   ctx_mean_arr,
                                   ctx_std_arr,
                                   ctx_up_arr,
                                   ctx_extra_arr);

   double cost_buffer_points = (AI_CostBufferPoints < 0.0 ? 0.0 : AI_CostBufferPoints);
   double commission_points = snapshot.commission_points;
   double spread_pred = FXAI_GetSpreadAtIndex(0, spread_m1, snapshot.spread_points);
   double min_move_pred = spread_pred + commission_points + cost_buffer_points;
   if(min_move_pred < 0.0) min_move_pred = 0.0;
   double vol_hint = MathAbs(FXAI_SafeReturn(close_arr, 0, 1));
   int regime_hint = FXAI_GetRegimeId(snapshot.bar_time, spread_pred, vol_hint);
   int ai_hint = (ensembleMode ? -1 : aiType);

   int H = FXAI_SelectRoutedHorizon(close_arr,
                                    snapshot,
                                    min_move_pred,
                                    evLookback,
                                    base_h,
                                    regime_hint,
                                    ai_hint);
   int init_start = H;
   int init_end = H + base - 1;
   int online_start = H;
   int online_end = H + K - 1;
   int shadow_samples = Ensemble_ShadowSamples;
   if(shadow_samples < 8) shadow_samples = 8;
   if(shadow_samples > 200) shadow_samples = 200;
   int shadow_epochs = Ensemble_ShadowEpochs;
   if(shadow_epochs < 1) shadow_epochs = 1;
   if(shadow_epochs > 3) shadow_epochs = 3;
   int shadow_every = Ensemble_ShadowEveryBars;
   if(shadow_every < 1) shadow_every = 1;
   bool run_shadow = (ensembleMode && FXAI_IsShadowBar(shadow_every, signal_seq));
   int shadow_start = H;
   int shadow_end = H + shadow_samples - 1;

   int max_valid = needed - FEATURE_LB - 1;
   if(init_end > max_valid) init_end = max_valid;
   if(online_end > max_valid) online_end = max_valid;
   if(shadow_end > max_valid) shadow_end = max_valid;
   bool have_init_window = (init_end >= init_start);
   bool have_online_window = (online_end >= online_start);
   bool have_shadow_window = (shadow_end >= shadow_start);

   int precompute_end = -1;
   if(have_init_window) precompute_end = init_end;
   if(have_online_window && online_end > precompute_end) precompute_end = online_end;
   if(run_shadow && have_shadow_window && shadow_end > precompute_end) precompute_end = shadow_end;

   double ctx_mean_pred = FXAI_GetArrayValue(ctx_mean_arr, 0, 0.0);
   double ctx_std_pred = FXAI_GetArrayValue(ctx_std_arr, 0, 0.0);
   double ctx_up_pred = FXAI_GetArrayValue(ctx_up_arr, 0, 0.5);

   ENUM_FXAI_FEATURE_NORMALIZATION norm_method = FXAI_GetFeatureNormalizationMethod();
   double feat_pred[FXAI_AI_FEATURES];
   if(!FXAI_ComputeFeatureVector(0,
                                spread_pred,
                                time_arr,
                                open_arr,
                                high_arr,
                                low_arr,
                                close_arr,
                                time_m5,
                                close_m5,
                                map_m5,
                                time_m15,
                                close_m15,
                                map_m15,
                                time_m30,
                                close_m30,
                                map_m30,
                                time_h1,
                                close_h1,
                                map_h1,
                                ctx_mean_pred,
                                ctx_std_pred,
                                ctx_up_pred,
                                ctx_extra_arr,
                                norm_method,
                                feat_pred))
   {
      g_ai_last_signal_bar = signal_bar;
      g_ai_last_signal_key = decisionKey;
      g_ai_last_signal = -1;
      g_ai_last_reason = "predict_features_failed";
      return -1;
   }

   bool need_prev = FXAI_FeatureNormNeedsPrevious(norm_method);
   bool has_prev_feat = false;
   double feat_prev[FXAI_AI_FEATURES];
   for(int f=0; f<FXAI_AI_FEATURES; f++)
      feat_prev[f] = 0.0;

   if(need_prev && ArraySize(close_arr) > 1)
   {
      double spread_prev = FXAI_GetSpreadAtIndex(1, spread_m1, spread_pred);
      double ctx_mean_prev = FXAI_GetArrayValue(ctx_mean_arr, 1, ctx_mean_pred);
      double ctx_std_prev = FXAI_GetArrayValue(ctx_std_arr, 1, ctx_std_pred);
      double ctx_up_prev = FXAI_GetArrayValue(ctx_up_arr, 1, ctx_up_pred);

      has_prev_feat = FXAI_ComputeFeatureVector(1,
                                               spread_prev,
                                               time_arr,
                                               open_arr,
                                               high_arr,
                                               low_arr,
                                               close_arr,
                                               time_m5,
                                               close_m5,
                                               map_m5,
                                               time_m15,
                                               close_m15,
                                               map_m15,
                                               time_m30,
                                               close_m30,
                                               map_m30,
                                               time_h1,
                                               close_h1,
                                               map_h1,
                                               ctx_mean_prev,
                                               ctx_std_prev,
                                               ctx_up_prev,
                                               ctx_extra_arr,
                                               norm_method,
                                               feat_prev);
   }

   double feat_pred_norm[FXAI_AI_FEATURES];
   FXAI_ApplyFeatureNormalization(norm_method,
                                  feat_pred,
                                  feat_prev,
                                  has_prev_feat,
                                  snapshot.bar_time,
                                  feat_pred_norm);

   double x_pred[FXAI_AI_WEIGHTS];
   FXAI_BuildInputVector(feat_pred_norm, x_pred);

   double fallback_expected_move = FXAI_EstimateExpectedAbsMovePoints(close_arr,
                                                                      H,
                                                                      evLookback,
                                                                      snapshot.point);
   if(fallback_expected_move <= 0.0)
      fallback_expected_move = min_move_pred;
   double vol_proxy_abs = MathAbs(feat_pred[5]);
   FXAI_UpdateRegimeEMAs(spread_pred, vol_proxy_abs);
   int regime_id = FXAI_GetRegimeId(snapshot.bar_time, spread_pred, vol_proxy_abs);
   double hpolicy_feat[FXAI_HPOL_FEATS];
   FXAI_BuildHorizonPolicyFeatures(H,
                                   base_h,
                                   fallback_expected_move,
                                   min_move_pred,
                                   snapshot,
                                   MathAbs(FXAI_SafeReturn(close_arr, 0, 1)),
                                   regime_id,
                                   ai_hint,
                                   hpolicy_feat);
   FXAI_EnqueueHorizonPolicyPending(signal_seq, regime_id, H, min_move_pred, hpolicy_feat);

   static FXAIPreparedSample samples[];
   if(precompute_end >= 1)
   {
      // Start at 1 (not H) so rolling normalizers see the full recent past for
      // prediction-time feature scaling, even when horizon H is large.
      FXAI_PrecomputeTrainingSamples(1,
                                    precompute_end,
                                    H,
                                    commission_points,
                                    cost_buffer_points,
                                    evThresholdPoints,
                                    snapshot,
                                    spread_m1,
                                    time_arr,
                                    open_arr,
                                    high_arr,
                                    low_arr,
                                    close_arr,
                                    time_m5,
                                    close_m5,
                                    map_m5,
                                    time_m15,
                                    close_m15,
                                    map_m15,
                                    time_m30,
                                    close_m30,
                                    map_m30,
                                    time_h1,
                                    close_h1,
                                    map_h1,
                                    ctx_mean_arr,
                                    ctx_std_arr,
                                    ctx_up_arr,
                                    ctx_extra_arr,
                                    -1,
                                    samples);
   }

   if(have_online_window && online_start >= 0 && online_start < ArraySize(samples))
      FXAI_AddReplaySample(samples[online_start]);

   int active_ai_ids[];
   ArrayResize(active_ai_ids, 0);
   if(ensembleMode == 0)
   {
      if(g_plugins.Get(aiType) != NULL)
      {
         ArrayResize(active_ai_ids, 1);
         active_ai_ids[0] = aiType;
      }
   }
   else
   {
      int cand_ai_ids[];
      double cand_scores[];
      ArrayResize(cand_ai_ids, 0);
      ArrayResize(cand_scores, 0);

      for(int ai_idx=0; ai_idx<FXAI_AI_COUNT; ai_idx++)
      {
         if(g_plugins.Get(ai_idx) == NULL) continue;
         if(FXAI_IsModelPruned(ai_idx, regime_id)) continue;
         double score = FXAI_GetModelMetaScore(ai_idx, regime_id, min_move_pred);
         if(score <= 0.0) continue;
         int sz = ArraySize(cand_ai_ids);
         ArrayResize(cand_ai_ids, sz + 1);
         ArrayResize(cand_scores, sz + 1);
         cand_ai_ids[sz] = ai_idx;
         cand_scores[sz] = score;
      }

      int cand_n = ArraySize(cand_ai_ids);
      if(cand_n > 0)
      {
         int top_k = cand_n;
         if(top_k > 10) top_k = 10;
         if(top_k < 2 && cand_n >= 2) top_k = 2;

         bool used[];
         ArrayResize(used, cand_n);
         for(int j=0; j<cand_n; j++) used[j] = false;

         ArrayResize(active_ai_ids, top_k);
         int picked = 0;
         for(int pick=0; pick<top_k; pick++)
         {
            int best_j = -1;
            double best_sc = -1e18;
            for(int j=0; j<cand_n; j++)
            {
               if(used[j]) continue;
               if(cand_scores[j] > best_sc)
               {
                  best_sc = cand_scores[j];
                  best_j = j;
               }
            }
            if(best_j < 0) break;
            used[best_j] = true;
            active_ai_ids[picked] = cand_ai_ids[best_j];
            picked++;
         }
         if(picked < ArraySize(active_ai_ids))
            ArrayResize(active_ai_ids, picked);

         // Exploration arm: occasionally add one non-top model (cold-start biased)
         // to improve adaptation and avoid starvation under pruning.
         if(cand_n > picked && FXAI_ShouldSampleByPct(signal_bar, regime_id + 17, Ensemble_ExplorePct))
         {
            int explore_j = -1;
            double explore_sc = -1e18;
            for(int j=0; j<cand_n; j++)
            {
               if(used[j]) continue;
               int ai_id = cand_ai_ids[j];
               int obs = 0;
               if(regime_id >= 0 && regime_id < FXAI_REGIME_COUNT)
                  obs = g_model_regime_obs[ai_id][regime_id];
               double cold_bonus = 1.0 / MathSqrt(1.0 + (double)obs);
               double score = cand_scores[j] * (1.0 + (0.35 * cold_bonus));
               if(score > explore_sc)
               {
                  explore_sc = score;
                  explore_j = j;
               }
            }
            if(explore_j >= 0)
            {
               int add_id = cand_ai_ids[explore_j];
               if(!FXAI_IsModelInList(add_id, active_ai_ids))
               {
                  int sz = ArraySize(active_ai_ids);
                  ArrayResize(active_ai_ids, sz + 1);
                  active_ai_ids[sz] = add_id;
               }
            }
         }
      }
      else
      {
         // If all models are pruned for this regime, keep one conservative fallback.
         int fallback_id = -1;
         double fallback_rel = -1e9;
         for(int ai_idx=0; ai_idx<FXAI_AI_COUNT; ai_idx++)
         {
            if(g_plugins.Get(ai_idx) == NULL) continue;
            double rel = FXAI_GetModelVoteWeight(ai_idx);
            if(rel > fallback_rel)
            {
               fallback_rel = rel;
               fallback_id = ai_idx;
            }
         }
         if(fallback_id >= 0)
         {
            ArrayResize(active_ai_ids, 1);
            active_ai_ids[0] = fallback_id;
         }
      }
   }

   if(ArraySize(active_ai_ids) <= 0)
   {
      g_ai_last_signal_bar = signal_bar;
      g_ai_last_signal_key = decisionKey;
      g_ai_last_signal = -1;
      g_ai_last_reason = "no_active_models";
      return -1;
   }

   FXAINormSampleCache runtime_norm_caches[];
   ArrayResize(runtime_norm_caches, 0);
   FXAINormInputCache input_caches[];
   ArrayResize(input_caches, 0);

   for(int ai_pass=0; ai_pass<FXAI_AI_COUNT; ai_pass++)
   {
      bool needed_ai = false;
      if(FXAI_IsModelInList(ai_pass, active_ai_ids))
         needed_ai = true;
      else if(run_shadow && !FXAI_IsModelInList(ai_pass, active_ai_ids) && g_plugins.Get(ai_pass) != NULL)
         needed_ai = true;
      if(!needed_ai) continue;

      int method_id = (int)FXAI_GetModelNormMethodRouted(ai_pass, regime_id, H);
      if(FXAI_FindNormInputCache(method_id, input_caches) < 0)
      {
         FXAI_EnsureNormInputCache(method_id,
                                   spread_pred,
                                   spread_m1,
                                   snapshot,
                                   time_arr,
                                   open_arr,
                                   high_arr,
                                   low_arr,
                                   close_arr,
                                   time_m5,
                                   close_m5,
                                   map_m5,
                                   time_m15,
                                   close_m15,
                                   map_m15,
                                   time_m30,
                                   close_m30,
                                   map_m30,
                                   time_h1,
                                   close_h1,
                                   map_h1,
                                   ctx_mean_arr,
                                   ctx_std_arr,
                                   ctx_up_arr,
                                   ctx_extra_arr,
                                   input_caches);
      }

      if(precompute_end >= 1)
      {
         FXAI_EnsureRoutedNormCachesForSamples(ai_pass,
                                               1,
                                               precompute_end,
                                               H,
                                               commission_points,
                                               cost_buffer_points,
                                               evThresholdPoints,
                                               snapshot,
                                               spread_m1,
                                               time_arr,
                                               open_arr,
                                               high_arr,
                                               low_arr,
                                               close_arr,
                                               time_m5,
                                               close_m5,
                                               map_m5,
                                               time_m15,
                                               close_m15,
                                               map_m15,
                                               time_m30,
                                               close_m30,
                                               map_m30,
                                               time_h1,
                                               close_h1,
                                               map_h1,
                                               ctx_mean_arr,
                                               ctx_std_arr,
                                               ctx_up_arr,
                                               ctx_extra_arr,
                                               samples,
                                               runtime_norm_caches);
      }
   }

   if(run_shadow && have_shadow_window)
   {
      int warm_epochs = trainEpochs;
      if(warm_epochs > 4) warm_epochs = 4;
      if(warm_epochs < 1) warm_epochs = 1;

      for(int ai_idx=0; ai_idx<FXAI_AI_COUNT; ai_idx++)
      {
         if(FXAI_IsModelInList(ai_idx, active_ai_ids)) continue;
         CFXAIAIPlugin *plugin_shadow = g_plugins.Get(ai_idx);
         if(plugin_shadow == NULL) continue;

         FXAIAIHyperParams hp_shadow;
         FXAI_GetModelHyperParamsRouted(ai_idx, regime_id, H, hp_shadow);
         plugin_shadow.EnsureInitialized(hp_shadow);

         if(!g_ai_trained[ai_idx])
         {
            if(have_init_window)
               FXAI_TrainModelWindowPreparedRoutedCached(ai_idx,
                                                         *plugin_shadow,
                                                         init_start,
                                                         init_end,
                                                         warm_epochs,
                                                         samples,
                                                         runtime_norm_caches);
            FXAI_TrainModelReplay(ai_idx, *plugin_shadow, regime_id, H, 1);
            g_ai_trained[ai_idx] = true;
            g_ai_last_train_bar[ai_idx] = snapshot.bar_time;
         }

         if(snapshot.bar_time != g_ai_last_train_bar[ai_idx])
         {
            FXAI_TrainModelWindowPreparedRoutedCached(ai_idx,
                                                      *plugin_shadow,
                                                      shadow_start,
                                                      shadow_end,
                                                      shadow_epochs,
                                                      samples,
                                                      runtime_norm_caches);
            FXAI_TrainModelReplay(ai_idx, *plugin_shadow, regime_id, H, 1);
            g_ai_last_train_bar[ai_idx] = snapshot.bar_time;
         }
      }
   }

   int singleSignal = -1;
   double ensemble_buy_ev_sum = 0.0;
   double ensemble_sell_ev_sum = 0.0;
   double ensemble_buy_support = 0.0;
   double ensemble_sell_support = 0.0;
   double ensemble_skip_support = 0.0;
   double ensemble_meta_total = 0.0;
   double ensemble_expected_sum = 0.0;
   double ensemble_probs[3];
   ensemble_probs[0] = 0.3333;
   ensemble_probs[1] = 0.3333;
   ensemble_probs[2] = 0.3334;
   double stack_feat[FXAI_STACK_FEATS];
   for(int sf=0; sf<FXAI_STACK_FEATS; sf++) stack_feat[sf] = 0.0;

   for(int m=0; m<ArraySize(active_ai_ids); m++)
   {
      int ai_idx = active_ai_ids[m];

      CFXAIAIPlugin *plugin = g_plugins.Get(ai_idx);
      if(plugin == NULL)
         continue;

      FXAIAIHyperParams hp_model;
      FXAI_GetModelHyperParamsRouted(ai_idx, regime_id, H, hp_model);
      plugin.EnsureInitialized(hp_model);

      if(!g_ai_trained[ai_idx])
      {
         if(have_init_window)
         {
            FXAI_TrainModelWindowPreparedRoutedCached(ai_idx,
                                                      *plugin,
                                                      init_start,
                                                      init_end,
                                                      trainEpochs,
                                                      samples,
                                                      runtime_norm_caches);
         }
         FXAI_TrainModelReplay(ai_idx, *plugin, regime_id, H, 1);

         g_ai_trained[ai_idx] = true;
         g_ai_last_train_bar[ai_idx] = snapshot.bar_time;
      }
      else if(snapshot.bar_time != g_ai_last_train_bar[ai_idx])
      {
         if(have_online_window)
         {
            FXAI_TrainModelWindowPreparedRoutedCached(ai_idx,
                                                      *plugin,
                                                      online_start,
                                                      online_end,
                                                      onlineEpochs,
                                                      samples,
                                                      runtime_norm_caches);
         }
         FXAI_TrainModelReplay(ai_idx, *plugin, regime_id, H, 1);

         g_ai_last_train_bar[ai_idx] = snapshot.bar_time;
      }

      FXAIAIPredictV2 req;
      req.regime_id = regime_id;
      req.horizon_minutes = H;
      req.min_move_points = min_move_pred;
      req.cost_points = min_move_pred;
      req.sample_time = snapshot.bar_time;
      int method_id = (int)FXAI_GetModelNormMethodRouted(ai_idx, regime_id, H);
      int input_idx = FXAI_FindNormInputCache(method_id, input_caches);
      for(int k=0; k<FXAI_AI_WEIGHTS; k++)
         req.x[k] = (input_idx >= 0 ? input_caches[input_idx].x[k] : x_pred[k]);

      FXAIAIPredictionV2 pred;
      plugin.PredictV2(req, hp_model, pred);

      double class_probs_pred[3];
      class_probs_pred[0] = pred.class_probs[0];
      class_probs_pred[1] = pred.class_probs[1];
      class_probs_pred[2] = pred.class_probs[2];
      FXAI_ApplyRegimeCalibration(ai_idx, regime_id, class_probs_pred);

      double expected_move = pred.expected_move_points;
      if(expected_move <= 0.0)
         expected_move = FXAI_GetModelExpectedMove(ai_idx, fallback_expected_move);
      if(expected_move <= 0.0)
         expected_move = fallback_expected_move;

      double modelBuyThr = buyThr;
      double modelSellThr = sellThr;
      FXAI_GetModelThresholds(ai_idx, regime_id, H, buyThr, sellThr, modelBuyThr, modelSellThr);

      double buyMinProb = modelBuyThr;
      double sellMinProb = 1.0 - modelSellThr;
      double skipMinProb = 0.55;
      FXAI_DeriveAdaptiveThresholds(modelBuyThr,
                                   modelSellThr,
                                   min_move_pred,
                                   expected_move,
                                   feat_pred[5],
                                   buyMinProb,
                                   sellMinProb,
                                   skipMinProb);

      int signal = FXAI_ClassSignalFromEV(class_probs_pred,
                                         buyMinProb,
                                         sellMinProb,
                                         skipMinProb,
                                         expected_move,
                                         min_move_pred,
                                         evThresholdPoints);
      FXAI_EnqueueReliabilityPending(ai_idx,
                                     signal_seq,
                                     signal,
                                     regime_id,
                                     expected_move,
                                     H,
                                     class_probs_pred);

      if(ensembleMode == 0)
      {
         singleSignal = signal;
      }
      else
      {
         double meta_w = FXAI_GetModelMetaScore(ai_idx, regime_id, min_move_pred);
         if(meta_w <= 0.0) continue;

         double model_buy_ev = ((2.0 * class_probs_pred[(int)FXAI_LABEL_BUY]) - 1.0) * expected_move - min_move_pred;
         double model_sell_ev = ((2.0 * class_probs_pred[(int)FXAI_LABEL_SELL]) - 1.0) * expected_move - min_move_pred;
         model_buy_ev = FXAI_Clamp(model_buy_ev, -10.0 * min_move_pred, 10.0 * min_move_pred);
         model_sell_ev = FXAI_Clamp(model_sell_ev, -10.0 * min_move_pred, 10.0 * min_move_pred);

         ensemble_meta_total += meta_w;
         ensemble_buy_ev_sum += meta_w * model_buy_ev;
         ensemble_sell_ev_sum += meta_w * model_sell_ev;
         ensemble_expected_sum += meta_w * expected_move;

         if(signal == 1) ensemble_buy_support += meta_w;
         else if(signal == 0) ensemble_sell_support += meta_w;
         else ensemble_skip_support += meta_w;
      }
   }

   int decision = -1;
   if(ensembleMode == 0)
   {
      decision = singleSignal;
   }
   else
   {
      if(ensemble_meta_total > 0.0)
      {
         double buyPct = 100.0 * (ensemble_buy_support / ensemble_meta_total);
         double sellPct = 100.0 * (ensemble_sell_support / ensemble_meta_total);
         double skipPct = 100.0 * (ensemble_skip_support / ensemble_meta_total);
         double avg_buy_ev = ensemble_buy_ev_sum / ensemble_meta_total;
         double avg_sell_ev = ensemble_sell_ev_sum / ensemble_meta_total;
         double avg_expected = ensemble_expected_sum / ensemble_meta_total;
         double vote_probs[3];
         vote_probs[(int)FXAI_LABEL_SELL] = FXAI_Clamp(ensemble_sell_support / ensemble_meta_total, 0.0, 1.0);
         vote_probs[(int)FXAI_LABEL_BUY] = FXAI_Clamp(ensemble_buy_support / ensemble_meta_total, 0.0, 1.0);
         vote_probs[(int)FXAI_LABEL_SKIP] = FXAI_Clamp(ensemble_skip_support / ensemble_meta_total, 0.0, 1.0);
         double vs = vote_probs[0] + vote_probs[1] + vote_probs[2];
         if(vs <= 0.0) vs = 1.0;
         vote_probs[0] /= vs; vote_probs[1] /= vs; vote_probs[2] /= vs;

         FXAI_StackBuildFeatures(buyPct,
                                 sellPct,
                                 skipPct,
                                 avg_buy_ev,
                                 avg_sell_ev,
                                 min_move_pred,
                                 avg_expected,
                                 vol_proxy_abs,
                                 H,
                                 stack_feat);
         double stack_probs_dyn[];
         ArrayResize(stack_probs_dyn, 3);
         FXAI_StackPredict(regime_id, stack_feat, stack_probs_dyn);
         ensemble_probs[0] = FXAI_Clamp(0.65 * stack_probs_dyn[0] + 0.35 * vote_probs[0], 0.0005, 0.9990);
         ensemble_probs[1] = FXAI_Clamp(0.65 * stack_probs_dyn[1] + 0.35 * vote_probs[1], 0.0005, 0.9990);
         ensemble_probs[2] = FXAI_Clamp(0.65 * stack_probs_dyn[2] + 0.35 * vote_probs[2], 0.0005, 0.9990);
         double ps = ensemble_probs[0] + ensemble_probs[1] + ensemble_probs[2];
         if(ps <= 0.0) ps = 1.0;
         ensemble_probs[0] /= ps; ensemble_probs[1] /= ps; ensemble_probs[2] /= ps;

         double stack_buy_ev = ((2.0 * ensemble_probs[(int)FXAI_LABEL_BUY]) - 1.0) * MathMax(fallback_expected_move, min_move_pred) - min_move_pred;
         double stack_sell_ev = ((2.0 * ensemble_probs[(int)FXAI_LABEL_SELL]) - 1.0) * MathMax(fallback_expected_move, min_move_pred) - min_move_pred;

         if(ensemble_probs[(int)FXAI_LABEL_SKIP] >= 0.58 || skipPct >= 75.0)
            decision = -1;
         else if(ensemble_probs[(int)FXAI_LABEL_BUY] >= ensemble_probs[(int)FXAI_LABEL_SELL] &&
                 buyPct >= agreePct &&
                 stack_buy_ev >= evThresholdPoints &&
                 avg_buy_ev > avg_sell_ev)
            decision = 1;
         else if(ensemble_probs[(int)FXAI_LABEL_SELL] > ensemble_probs[(int)FXAI_LABEL_BUY] &&
                 sellPct >= agreePct &&
                 stack_sell_ev >= evThresholdPoints &&
                 avg_sell_ev > avg_buy_ev)
            decision = 0;
         else
         {
            // Conservative fallback if stack is uncertain.
            if(buyPct >= agreePct && avg_buy_ev >= evThresholdPoints && avg_buy_ev > avg_sell_ev)
               decision = 1;
            else if(sellPct >= agreePct && avg_sell_ev >= evThresholdPoints && avg_sell_ev > avg_buy_ev)
               decision = 0;
         }
      }
   }

   if(ensembleMode != 0 && ensemble_meta_total > 0.0)
   {
      double ens_expected = MathMax(min_move_pred,
                                    (ensemble_expected_sum > 0.0 ? ensemble_expected_sum / ensemble_meta_total :
                                     (MathAbs(ensemble_buy_ev_sum) + MathAbs(ensemble_sell_ev_sum)) / MathMax(ensemble_meta_total, 1.0)));
      FXAI_EnqueueStackPending(signal_seq,
                               decision,
                               regime_id,
                               H,
                               ens_expected,
                               ensemble_probs,
                               stack_feat);
   }

   g_ai_last_signal_bar = signal_bar;
   g_ai_last_signal_key = decisionKey;
   g_ai_last_signal = decision;
   if(decision == 1) g_ai_last_reason = "buy";
   else if(decision == 0) g_ai_last_reason = "sell";
   else if(ensembleMode != 0 && ensemble_meta_total <= 0.0) g_ai_last_reason = "no_meta_weight";
   else g_ai_last_reason = "no_consensus_or_ev";

   return decision;
}

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

   double trade_lot = FXAI_NormalizeLot(_Symbol, Lot);
   if(trade_lot <= 0.0) return;

   bool ok = false;
   if(direction == 1) ok = trade.Buy(trade_lot, _Symbol, 0, 0, 0, "Buy");
   else               ok = trade.Sell(trade_lot, _Symbol, 0, 0, 0, "Sell");

   if(!ok && emit_debug)
      Print("FXAI debug: Order send failed. retcode=", (int)trade.ResultRetcode(),
            " desc=", trade.ResultRetcodeDescription());
   else if(ok && emit_debug)
      Print("FXAI debug: Order sent. direction=", direction, " lot=", DoubleToString(trade_lot, 2));

   if(ok)
   {
      ResetCycleState();
      CycleActive      = true;
      CycleEntryEquity = AccountInfoDouble(ACCOUNT_EQUITY);
      CycleStartTime   = TimeCurrent();
      if(CycleStartTime <= 0) CycleStartTime = TimeTradeServer();
      if(CycleStartTime <= 0) CycleStartTime = iTime(_Symbol, PERIOD_M1, 0);

      TrailTracking    = true;
      TrailPeakProfit  = 0.0;
   }
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

   int total = OrdersTotal() + PositionsTotal();

   if(total > 0)
   {
      TradeKillerManage();
      if(OrdersTotal() + PositionsTotal() == 0) return;

      EquitySLManage();
      if(OrdersTotal() + PositionsTotal() == 0) return;

      EquityTrailManage();
      if(OrdersTotal() + PositionsTotal() == 0) return;

      if(MaxDD > 0) EAStop();
      TPCheck();
   }

   if(!new_m1_bar) return;

   // Heavy model/reliability updates run only once per closed M1 bar.
   FXAI_ProcessReliabilityBar(_Symbol);
   int precomputed_signal = SpecialDirectionAI(_Symbol);

   if(OrdersTotal() + PositionsTotal() == 0)
      SendTrade(precomputed_signal);
}
//+------------------------------------------------------------------+
