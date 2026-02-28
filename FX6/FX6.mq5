//+------------------------------------------------------------------+
//|                                                          FX6.mq5 |
//| FX6 modular EA: plugin-based AI + equity trailing + equity SL   |
//+------------------------------------------------------------------+
#property strict

#include <Trade\Trade.mqh>
#include "shared.mqh"
#include "data.mqh"
#include "api.mqh"

CTrade trade;

//-------------------------- INPUTS ---------------------------------
input ENUM_AI_TYPE AI_Type = AI_TYPE_SGD_LOGIT;
// Models: all (selector for which plugin runs in single-model mode).
// Purpose: chooses the active prediction model implementation.
// Importance/Range: enum value 0..FX6_AI_COUNT-1; use backtests to pick best per symbol.
input bool   AI_Ensemble          = false;  
// Models: all (ensemble controller across all plugins).
// Purpose: enables multi-model voting instead of one selected model.
// Importance/Range: false/true; true is usually more stable but may reduce trade count.
input double Ensemble_AgreePct    = 70.0; 
// Models: all (only used when AI_Ensemble=true).
// Purpose: sets the minimum vote percentage needed for BUY or SELL.
// Importance/Range: practical 50..90; higher means fewer but stricter entries.


input double AI_BuyThreshold         = 0.60;
// Models: all (3-class head + adaptive entry gate).
// Purpose: base minimum BUY confidence; runtime logic tightens it by cost/volatility regime.
// Importance/Range: common 0.55..0.80; must stay in (0,1) and above AI_SellThreshold.
input double AI_SellThreshold        = 0.40;
// Models: all (3-class head + adaptive entry gate).
// Purpose: base SELL side threshold input (lower than buy); runtime maps it to sell confidence.
// Importance/Range: common 0.20..0.45; must stay in (0,1) and below AI_BuyThreshold.

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
input double PA_C       = 0.50;
// Models: PA only.
// Purpose: aggressiveness cap on PA correction step size.
// Importance/Range: clamped 0.01..10; common 0.1..2.0.
input double PA_Margin  = 1.00;
// Models: PA only.
// Purpose: target margin before no further PA update is needed.
// Importance/Range: clamped 0.1..2.0; common 0.5..1.5.

// XGB-like split learners
input double XGB_FastLearningRate = 0.07;
// Models: XGB_Fast only.
// Purpose: shrinkage factor for split-learner updates.
// Importance/Range: clamped 0.001..0.2; common 0.02..0.1.
input double XGB_FastL2           = 0.005;
// Models: XGB_Fast only.
// Purpose: L2 regularization for split learner coefficients.
// Importance/Range: clamped 0..0.1; common 0.001..0.02.
input double XGB_SplitThreshold   = 0.00;
// Models: XGB_Fast only.
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

CFX6AIRegistry g_plugins;
bool g_plugins_ready = false;

string   g_ai_last_symbol = "";
bool     g_ai_trained[FX6_AI_COUNT];
datetime g_ai_last_train_bar[FX6_AI_COUNT];
datetime g_ai_last_signal_bar = 0;
int      g_ai_last_signal = -1;
int      g_ai_last_signal_key = -1;
string   g_ai_last_reason = "init";
datetime g_last_debug_bar = 0;

#define FX6_MAX_CONTEXT_SYMBOLS 32
#define FX6_REL_MAX_PENDING 2048

string g_context_symbols[];
double g_model_reliability[FX6_AI_COUNT];
double g_model_abs_move_ema[FX6_AI_COUNT];
bool   g_model_abs_move_ready[FX6_AI_COUNT];
FX6AIHyperParams g_model_hp[FX6_AI_COUNT];
bool g_model_hp_ready[FX6_AI_COUNT];
double g_model_buy_thr[FX6_AI_COUNT];
double g_model_sell_thr[FX6_AI_COUNT];
bool g_model_thr_ready[FX6_AI_COUNT];
bool g_ai_warmup_done = false;
int      g_rel_pending_seq[FX6_AI_COUNT][FX6_REL_MAX_PENDING];
double   g_rel_pending_prob[FX6_AI_COUNT][FX6_REL_MAX_PENDING][3];
int      g_rel_pending_head[FX6_AI_COUNT];
int      g_rel_pending_tail[FX6_AI_COUNT];
datetime g_rel_clock_bar_time = 0;
int      g_rel_clock_seq = 0;

struct FX6ContextSeries
{
   bool loaded;
   string symbol;
   datetime last_bar_time;
   MqlRates rates[];
   double close[];
   datetime time[];
   int aligned_idx[];
};

struct FX6PreparedSample
{
   bool valid;
   int label_class;
   double move_points;
   double min_move_points;
   double cost_points;
   datetime sample_time;
   double x[FX6_AI_WEIGHTS];
};

double FX6_GetArrayValue(const double &arr[], const int idx, const double def_value)
{
   if(idx >= 0 && idx < ArraySize(arr)) return arr[idx];
   return def_value;
}

void FX6_ResetModelAuxState(const int ai_idx)
{
   if(ai_idx < 0 || ai_idx >= FX6_AI_COUNT) return;

   g_model_reliability[ai_idx] = 1.0;
   g_model_abs_move_ema[ai_idx] = 0.0;
   g_model_abs_move_ready[ai_idx] = false;
}

void FX6_UpdateModelMoveStats(const int ai_idx, const double move_points)
{
   if(ai_idx < 0 || ai_idx >= FX6_AI_COUNT) return;

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

double FX6_GetModelExpectedMove(const int ai_idx, const double fallback_move)
{
   if(ai_idx >= 0 && ai_idx < FX6_AI_COUNT && g_model_abs_move_ready[ai_idx] && g_model_abs_move_ema[ai_idx] > 0.0)
      return g_model_abs_move_ema[ai_idx];
   return fallback_move;
}

void FX6_UpdateModelReliability(const int ai_idx, const int label_class, const double &probs[])
{
   if(ai_idx < 0 || ai_idx >= FX6_AI_COUNT) return;
   if(label_class < 0 || label_class > 2) return;

   int best = 0;
   for(int c=1; c<3; c++)
      if(probs[c] > probs[best]) best = c;

   double p_true = FX6_Clamp(probs[label_class], 0.0, 1.0);
   double score = p_true;
   if(best == label_class) score += 0.25;

   double target = FX6_Clamp(score * 1.4, 0.25, 1.75);
   g_model_reliability[ai_idx] = FX6_Clamp((0.98 * g_model_reliability[ai_idx]) + (0.02 * target), 0.25, 2.50);
}

void FX6_ResetReliabilityPending()
{
   for(int ai=0; ai<FX6_AI_COUNT; ai++)
   {
      g_rel_pending_head[ai] = 0;
      g_rel_pending_tail[ai] = 0;
      for(int k=0; k<FX6_REL_MAX_PENDING; k++)
         g_rel_pending_seq[ai][k] = -1;
   }
   g_rel_clock_bar_time = 0;
   g_rel_clock_seq = 0;
}

void FX6_AdvanceReliabilityClock(const datetime signal_bar)
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

void FX6_EnqueueReliabilityPending(const int ai_idx, const int signal_seq, const double &probs[])
{
   if(ai_idx < 0 || ai_idx >= FX6_AI_COUNT) return;
   if(signal_seq < 0) return;

   int head = g_rel_pending_head[ai_idx];
   int tail = g_rel_pending_tail[ai_idx];

   int prev = tail - 1;
   if(prev < 0) prev += FX6_REL_MAX_PENDING;
   if(head != tail && g_rel_pending_seq[ai_idx][prev] == signal_seq)
   {
      g_rel_pending_prob[ai_idx][prev][0] = probs[0];
      g_rel_pending_prob[ai_idx][prev][1] = probs[1];
      g_rel_pending_prob[ai_idx][prev][2] = probs[2];
      return;
   }

   g_rel_pending_seq[ai_idx][tail] = signal_seq;
   g_rel_pending_prob[ai_idx][tail][0] = probs[0];
   g_rel_pending_prob[ai_idx][tail][1] = probs[1];
   g_rel_pending_prob[ai_idx][tail][2] = probs[2];

   int next_tail = tail + 1;
   if(next_tail >= FX6_REL_MAX_PENDING) next_tail = 0;
   if(next_tail == head)
   {
      head++;
      if(head >= FX6_REL_MAX_PENDING) head = 0;
      g_rel_pending_head[ai_idx] = head;
   }
   g_rel_pending_tail[ai_idx] = next_tail;
}

void FX6_UpdateReliabilityFromPending(const int ai_idx,
                                      const int current_signal_seq,
                                      const int H,
                                      const FX6DataSnapshot &snapshot,
                                      const int &spread_m1[],
                                      const double &close_arr[],
                                      const double commission_points,
                                      const double cost_buffer_points,
                                      const double ev_threshold_points)
{
   if(ai_idx < 0 || ai_idx >= FX6_AI_COUNT) return;
   if(current_signal_seq < 0 || H < 1) return;

   int head = g_rel_pending_head[ai_idx];
   int tail = g_rel_pending_tail[ai_idx];

   while(head != tail)
   {
      int seq_pred = g_rel_pending_seq[ai_idx][head];
      int next_head = head + 1;
      if(next_head >= FX6_REL_MAX_PENDING) next_head = 0;

      if(seq_pred < 0)
      {
         head = next_head;
         continue;
      }

      int age = current_signal_seq - seq_pred;
      if(age < H)
         break; // queue is time-ordered, newer items are not ready either

      int idx_pred = age;
      int idx_future = age - H;
      if(idx_pred >= 0 && idx_pred < ArraySize(close_arr) &&
         idx_future >= 0 && idx_future < ArraySize(close_arr))
      {
         double spread_i = FX6_GetSpreadAtIndex(idx_pred, spread_m1, snapshot.spread_points);
         double min_move_i = spread_i + commission_points + cost_buffer_points;
         if(min_move_i < 0.0) min_move_i = 0.0;

         double move_points = FX6_MovePoints(close_arr[idx_pred], close_arr[idx_future], snapshot.point);
         int label_class = FX6_BuildEVClassLabel(move_points, min_move_i, ev_threshold_points);

         double probs_eval[3];
         probs_eval[0] = g_rel_pending_prob[ai_idx][head][0];
         probs_eval[1] = g_rel_pending_prob[ai_idx][head][1];
         probs_eval[2] = g_rel_pending_prob[ai_idx][head][2];
         FX6_UpdateModelReliability(ai_idx, label_class, probs_eval);
      }

      head = next_head;
   }

   g_rel_pending_head[ai_idx] = head;
}

void FX6_ProcessReliabilityBar(const string symbol)
{
   if(StringLen(symbol) <= 0) return;

   int H = PredictionTargetMinutes;
   if(H < 1) H = 1;
   if(H > 720) H = 720;

   datetime signal_bar = iTime(symbol, PERIOD_M1, 1);
   if(signal_bar <= 0) return;

   static string rel_symbol = "";
   static datetime rel_last_processed_bar = 0;
   static datetime rel_last_rates_bar = 0;
   static MqlRates rel_rates_m1[];
   static double rel_close_arr[];
   static datetime rel_time_arr[];
   static int rel_spread_arr[];

   if(rel_symbol != symbol)
   {
      rel_symbol = symbol;
      rel_last_processed_bar = 0;
      rel_last_rates_bar = 0;
      ArrayResize(rel_rates_m1, 0);
      ArrayResize(rel_close_arr, 0);
      ArrayResize(rel_time_arr, 0);
      ArrayResize(rel_spread_arr, 0);
   }

   FX6_AdvanceReliabilityClock(signal_bar);
   if(signal_bar == rel_last_processed_bar) return;
   rel_last_processed_bar = signal_bar;

   int needed = H + 64;
   if(needed < 128) needed = 128;
   if(needed > 1500) needed = 1500;

   if(!FX6_UpdateRatesRolling(symbol, PERIOD_M1, needed, rel_last_rates_bar, rel_rates_m1))
      return;

   FX6_ExtractRatesCloseTimeSpread(rel_rates_m1, rel_close_arr, rel_time_arr, rel_spread_arr);
   if(ArraySize(rel_close_arr) <= H || ArraySize(rel_spread_arr) <= H)
      return;

   FX6DataSnapshot snapshot;
   if(!FX6_ExportDataSnapshot(symbol, AI_CommissionPerLotSide, AI_CostBufferPoints, snapshot))
      return;
   snapshot.bar_time = signal_bar;

   double cost_buffer_points = (AI_CostBufferPoints < 0.0 ? 0.0 : AI_CostBufferPoints);
   double commission_points = snapshot.commission_points;
   double evThresholdPoints = FX6_Clamp(AI_EVThresholdPoints, 0.0, 100.0);
   int signal_seq = g_rel_clock_seq;

   for(int ai_idx=0; ai_idx<FX6_AI_COUNT; ai_idx++)
   {
      FX6_UpdateReliabilityFromPending(ai_idx,
                                       signal_seq,
                                       H,
                                       snapshot,
                                       rel_spread_arr,
                                       rel_close_arr,
                                       commission_points,
                                       cost_buffer_points,
                                       evThresholdPoints);
   }
}

double FX6_GetModelVoteWeight(const int ai_idx)
{
   if(ai_idx < 0 || ai_idx >= FX6_AI_COUNT) return 1.0;
   return FX6_Clamp(g_model_reliability[ai_idx], 0.25, 2.50);
}

void FX6_DeriveAdaptiveThresholds(const double base_buy_threshold,
                                  const double base_sell_threshold,
                                  const double min_move_points,
                                  const double expected_move_points,
                                  const double vol_proxy,
                                  double &buy_min_prob,
                                  double &sell_min_prob,
                                  double &skip_min_prob)
{
   double buy_base = FX6_Clamp(base_buy_threshold, 0.50, 0.95);
   double sell_base = FX6_Clamp(1.0 - base_sell_threshold, 0.50, 0.95);

   double em = MathMax(expected_move_points, min_move_points + 0.10);
   double cost_ratio = FX6_Clamp(min_move_points / em, 0.0, 2.0);
   double vol_ratio = FX6_Clamp(vol_proxy / 4.0, 0.0, 1.0);

   double tighten = FX6_Clamp(((cost_ratio - 0.35) * 0.35) + (0.10 * vol_ratio), 0.0, 0.25);

   buy_min_prob = FX6_Clamp(buy_base + tighten, 0.50, 0.96);
   sell_min_prob = FX6_Clamp(sell_base + tighten, 0.50, 0.96);
   skip_min_prob = FX6_Clamp(0.45 + (0.20 * cost_ratio) + (0.10 * vol_ratio), 0.35, 0.85);
}

int FX6_ClassSignalFromEV(const double &probs[],
                          const double buy_min_prob,
                          const double sell_min_prob,
                          const double skip_min_prob,
                          const double expected_move_points,
                          const double min_move_points,
                          const double ev_threshold_points)
{
   if(expected_move_points <= 0.0) return -1;

   double p_sell = probs[(int)FX6_LABEL_SELL];
   double p_buy = probs[(int)FX6_LABEL_BUY];
   double p_skip = probs[(int)FX6_LABEL_SKIP];

   if(p_skip >= skip_min_prob) return -1;

   double buy_ev = ((2.0 * p_buy) - 1.0) * expected_move_points - min_move_points;
   double sell_ev = ((2.0 * p_sell) - 1.0) * expected_move_points - min_move_points;

   if(p_buy >= buy_min_prob && buy_ev >= ev_threshold_points && buy_ev > sell_ev)
      return 1;
   if(p_sell >= sell_min_prob && sell_ev >= ev_threshold_points && sell_ev > buy_ev)
      return 0;

   return -1;
}

void FX6_SanitizeThresholdPair(double &buy_threshold, double &sell_threshold)
{
   buy_threshold = FX6_Clamp(buy_threshold, 0.50, 0.95);
   sell_threshold = FX6_Clamp(sell_threshold, 0.05, 0.50);

   if(sell_threshold >= buy_threshold)
   {
      sell_threshold = FX6_Clamp(sell_threshold, 0.05, 0.49);
      buy_threshold = FX6_Clamp(MathMax(buy_threshold, sell_threshold + 0.01), 0.50, 0.95);
      if(sell_threshold >= buy_threshold)
      {
         sell_threshold = 0.49;
         buy_threshold = 0.50;
      }
   }
}

void FX6_ResetPreparedSample(FX6PreparedSample &sample)
{
   sample.valid = false;
   sample.label_class = (int)FX6_LABEL_SKIP;
   sample.move_points = 0.0;
   sample.min_move_points = 0.0;
   sample.cost_points = 0.0;
   sample.sample_time = 0;
   for(int k=0; k<FX6_AI_WEIGHTS; k++)
      sample.x[k] = 0.0;
}

bool FX6_PrepareTrainingSample(const int i,
                               const int H,
                               const double commission_points,
                               const double cost_buffer_points,
                               const double ev_threshold_points,
                               const FX6DataSnapshot &snapshot,
                               const int &spread_m1[],
                               const datetime &time_arr[],
                               const double &close_arr[],
                               const datetime &time_m5[],
                               const double &close_m5[],
                               const int &map_m5[],
                               const datetime &time_m15[],
                               const double &close_m15[],
                               const int &map_m15[],
                               const datetime &time_h1[],
                               const double &close_h1[],
                               const int &map_h1[],
                               const double &ctx_mean_arr[],
                               const double &ctx_std_arr[],
                               const double &ctx_up_arr[],
                               FX6PreparedSample &sample)
{
   FX6_ResetPreparedSample(sample);

   if(i < 0 || i >= ArraySize(close_arr)) return false;
   if(i - H < 0 || i - H >= ArraySize(close_arr)) return false;

   double move_points = FX6_MovePoints(close_arr[i], close_arr[i - H], snapshot.point);
   double spread_i = FX6_GetSpreadAtIndex(i, spread_m1, snapshot.spread_points);
   double min_move_i = spread_i + commission_points + cost_buffer_points;
   if(min_move_i < 0.0) min_move_i = 0.0;

   double ctx_mean_i = FX6_GetArrayValue(ctx_mean_arr, i, 0.0);
   double ctx_std_i = FX6_GetArrayValue(ctx_std_arr, i, 0.0);
   double ctx_up_i = FX6_GetArrayValue(ctx_up_arr, i, 0.5);

   double feat[FX6_AI_FEATURES];
   if(!FX6_ComputeFeatureVector(i,
                                spread_i,
                                time_arr,
                                close_arr,
                                time_m5,
                                close_m5,
                                map_m5,
                                time_m15,
                                close_m15,
                                map_m15,
                                time_h1,
                                close_h1,
                                map_h1,
                                ctx_mean_i,
                                ctx_std_i,
                                ctx_up_i,
                                feat))
      return false;

   sample.label_class = FX6_BuildEVClassLabel(move_points, min_move_i, ev_threshold_points);
   sample.move_points = move_points;
   sample.min_move_points = min_move_i;
   sample.cost_points = min_move_i;
   sample.sample_time = ((i >= 0 && i < ArraySize(time_arr)) ? time_arr[i] : 0);
   FX6_BuildInputVector(feat, sample.x);
   sample.valid = true;
   return true;
}

double FX6_ScoreWarmupTrial(CFX6AIPlugin &plugin,
                            const FX6AIHyperParams &hp,
                            const int val_start,
                            const int val_end,
                            const double buyThr,
                            const double sellThr,
                            const FX6PreparedSample &samples[],
                            int &trades_out)
{
   trades_out = 0;
   int n = ArraySize(samples);
   if(n <= 0 || val_end < val_start) return -1e9;

   int start = val_start;
   int end = val_end;
   if(start < 0) start = 0;
   if(end >= n) end = n - 1;
   if(end < start) return -1e9;

   int trades = 0;
   int wins = 0;
   double ev_sum = 0.0;

   for(int i=end; i>=start; i--)
   {
      if(!samples[i].valid) continue;

      FX6AIPredictV2 req;
      req.min_move_points = samples[i].min_move_points;
      req.cost_points = samples[i].cost_points;
      req.sample_time = samples[i].sample_time;
      for(int k=0; k<FX6_AI_WEIGHTS; k++)
         req.x[k] = samples[i].x[k];

      FX6AIPredictionV2 pred;
      plugin.PredictV2(req, hp, pred);

      double p_buy = pred.class_probs[(int)FX6_LABEL_BUY];
      double p_sell = pred.class_probs[(int)FX6_LABEL_SELL];
      double p_skip = pred.class_probs[(int)FX6_LABEL_SKIP];

      int signal = -1;
      if(p_buy >= buyThr && p_buy > p_sell && p_buy > p_skip) signal = 1;
      else if(p_sell >= (1.0 - sellThr) && p_sell > p_buy && p_sell > p_skip) signal = 0;
      if(signal == -1) continue;

      double expected_move = pred.expected_move_points;
      if(expected_move <= 0.0) expected_move = MathAbs(samples[i].move_points);
      if(expected_move <= 0.0) expected_move = samples[i].min_move_points;

      double ev = 0.0;
      if(signal == 1) ev = ((2.0 * p_buy) - 1.0) * expected_move - samples[i].min_move_points;
      else            ev = ((2.0 * p_sell) - 1.0) * expected_move - samples[i].min_move_points;

      ev_sum += ev;
      trades++;
      if(ev > 0.0) wins++;
   }

   if(trades <= 0) return -1e9;
   trades_out = trades;

   double win_rate = (double)wins / (double)trades;
   double avg_ev = ev_sum / (double)trades;
   return (avg_ev * 8.0) + (win_rate * 2.0);
}

bool FX6_WarmupTrainAndTune(const string symbol)
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

   int H = PredictionTargetMinutes;
   if(H < 1) H = 1;
   if(H > 720) H = 720;

   double base_buy_thr = AI_BuyThreshold;
   double base_sell_thr = AI_SellThreshold;
   FX6_SanitizeThresholdPair(base_buy_thr, base_sell_thr);
   double evThresholdPoints = FX6_Clamp(AI_EVThresholdPoints, 0.0, 100.0);

   int needed = warmup_samples + H + FEATURE_LB;

   FX6DataSnapshot snapshot;
   if(!FX6_ExportDataSnapshot(symbol, AI_CommissionPerLotSide, AI_CostBufferPoints, snapshot))
      return false;

   MqlRates rates_m1[];
   MqlRates rates_m5[];
   MqlRates rates_m15[];
   MqlRates rates_h1[];
   MqlRates rates_ctx_tmp[];

   double close_arr[];
   datetime time_arr[];
   int spread_m1[];
   if(!FX6_LoadSeriesWithSpread(symbol, needed, rates_m1, close_arr, time_arr, spread_m1))
      return false;

   if(ArraySize(close_arr) < needed || ArraySize(time_arr) < needed)
      return false;

   int needed_m5 = (needed / 5) + 80;
   int needed_m15 = (needed / 15) + 80;
   int needed_h1 = (needed / 60) + 80;

   double close_m5[];
   datetime time_m5[];
   double close_m15[];
   datetime time_m15[];
   double close_h1[];
   datetime time_h1[];
   int map_m5[];
   int map_m15[];
   int map_h1[];

   FX6_LoadSeriesOptionalCached(symbol, PERIOD_M5, needed_m5, rates_m5, close_m5, time_m5);
   FX6_LoadSeriesOptionalCached(symbol, PERIOD_M15, needed_m15, rates_m15, close_m15, time_m15);
   FX6_LoadSeriesOptionalCached(symbol, PERIOD_H1, needed_h1, rates_h1, close_h1, time_h1);

   int lag_m5 = 2 * PeriodSeconds(PERIOD_M5);
   int lag_m15 = 2 * PeriodSeconds(PERIOD_M15);
   int lag_h1 = 2 * PeriodSeconds(PERIOD_H1);
   if(lag_m5 <= 0) lag_m5 = 600;
   if(lag_m15 <= 0) lag_m15 = 1800;
   if(lag_h1 <= 0) lag_h1 = 7200;

   FX6_BuildAlignedIndexMap(time_arr, time_m5, lag_m5, map_m5);
   FX6_BuildAlignedIndexMap(time_arr, time_m15, lag_m15, map_m15);
   FX6_BuildAlignedIndexMap(time_arr, time_h1, lag_h1, map_h1);

   int ctx_count = ArraySize(g_context_symbols);
   if(ctx_count > FX6_MAX_CONTEXT_SYMBOLS) ctx_count = FX6_MAX_CONTEXT_SYMBOLS;
   FX6ContextSeries ctx_series[];
   ArrayResize(ctx_series, ctx_count);
   for(int s=0; s<ctx_count; s++)
   {
      ctx_series[s].loaded = FX6_LoadSeriesOptionalCached(g_context_symbols[s],
                                                          PERIOD_M1,
                                                          needed,
                                                          rates_ctx_tmp,
                                                          ctx_series[s].close,
                                                          ctx_series[s].time);
   }

   int i_start = H;
   int i_end = H + warmup_samples - 1;
   int max_valid = needed - FEATURE_LB - 1;
   if(i_end > max_valid) i_end = max_valid;
   if(i_end <= i_start) return false;

   double ctx_mean_arr[];
   double ctx_std_arr[];
   double ctx_up_arr[];
   FX6_PrecomputeContextAggregates(time_arr,
                                   ctx_series,
                                   ctx_count,
                                   i_end,
                                   ctx_mean_arr,
                                   ctx_std_arr,
                                   ctx_up_arr);

   double cost_buffer_points = (AI_CostBufferPoints < 0.0 ? 0.0 : AI_CostBufferPoints);
   double commission_points = snapshot.commission_points;

   FX6PreparedSample samples[];
   FX6_PrecomputeTrainingSamples(i_start,
                                 i_end,
                                 H,
                                 commission_points,
                                 cost_buffer_points,
                                 evThresholdPoints,
                                 snapshot,
                                 spread_m1,
                                 time_arr,
                                 close_arr,
                                 time_m5,
                                 close_m5,
                                 map_m5,
                                 time_m15,
                                 close_m15,
                                 map_m15,
                                 time_h1,
                                 close_h1,
                                 map_h1,
                                 ctx_mean_arr,
                                 ctx_std_arr,
                                 ctx_up_arr,
                                 samples);

   int sample_span = i_end - i_start + 1;
   int fold_len = sample_span / (warmup_folds + 1);
   if(fold_len < 40) fold_len = 40;
   if(fold_len > (sample_span / 2)) fold_len = sample_span / 2;
   if(fold_len < 20) return false;

   FX6AIHyperParams base_hp;
   FX6_BuildHyperParams(base_hp);

   datetime bar_time = iTime(symbol, PERIOD_M1, 1);
   if(bar_time <= 0) bar_time = TimeCurrent();
   int seed = AI_WarmupSeed;
   if(seed < 0) seed = -seed;

   for(int ai_idx=0; ai_idx<FX6_AI_COUNT; ai_idx++)
   {
      MathSrand((uint)(seed + (ai_idx + 1) * 104729 + (int)(bar_time % 65521)));

      CFX6AIPlugin *fold_pool[];
      ArrayResize(fold_pool, warmup_folds);
      for(int f=0; f<warmup_folds; f++)
         fold_pool[f] = g_plugins.CreateInstance(ai_idx);

      double best_score = -1e18;
      FX6AIHyperParams best_hp = base_hp;
      double best_buy_thr = base_buy_thr;
      double best_sell_thr = base_sell_thr;

      for(int loop=0; loop<warmup_loops; loop++)
      {
         FX6AIHyperParams hp_trial;
         double buy_trial = base_buy_thr;
         double sell_trial = base_sell_thr;
         if(loop == 0)
            hp_trial = base_hp;
         else
         {
            FX6_SampleModelHyperParams(ai_idx, base_hp, hp_trial);
            FX6_SampleThresholdPair(base_buy_thr, base_sell_thr, buy_trial, sell_trial);
         }

         double score_sum = 0.0;
         int folds_used = 0;
         int trades_total = 0;

         for(int f=0; f<warmup_folds; f++)
         {
            int val_start = i_start + (f * fold_len);
            int val_end = val_start + fold_len - 1;
            if(val_start < i_start) val_start = i_start;
            if(val_end >= i_end) val_end = i_end - 1;
            if(val_end <= val_start) continue;

            int train_start = val_end + 1;
            int train_end = i_end;
            if(train_end - train_start < 100) continue;

            CFX6AIPlugin *trial = fold_pool[f];
            if(trial == NULL) continue;
            trial.Reset();
            trial.EnsureInitialized(hp_trial);

            for(int epoch=0; epoch<warmup_train_epochs; epoch++)
            {
               for(int i=train_end; i>=train_start; i--)
               {
                  if(i < 0 || i >= ArraySize(samples)) continue;
                  if(!samples[i].valid) continue;
                  FX6AISampleV2 s2;
                  s2.valid = samples[i].valid;
                  s2.label_class = samples[i].label_class;
                  s2.move_points = samples[i].move_points;
                  s2.min_move_points = samples[i].min_move_points;
                  s2.cost_points = samples[i].cost_points;
                  s2.sample_time = samples[i].sample_time;
                  for(int k=0; k<FX6_AI_WEIGHTS; k++)
                     s2.x[k] = samples[i].x[k];
                  trial.TrainV2(s2, hp_trial);
               }
            }

            int trades_fold = 0;
            double score_fold = FX6_ScoreWarmupTrial(*trial,
                                                     hp_trial,
                                                     val_start,
                                                     val_end,
                                                     buy_trial,
                                                     sell_trial,
                                                     samples,
                                                     trades_fold);

            if(score_fold <= -1e8 || trades_fold <= 0) continue;
            score_sum += score_fold;
            trades_total += trades_fold;
            folds_used++;
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
      }

      for(int f=0; f<warmup_folds; f++)
      {
         if(fold_pool[f] != NULL)
         {
            delete fold_pool[f];
            fold_pool[f] = NULL;
         }
      }

      if(best_score <= -1e17)
      {
         best_hp = base_hp;
         best_buy_thr = base_buy_thr;
         best_sell_thr = base_sell_thr;
      }

      g_model_hp[ai_idx] = best_hp;
      g_model_hp_ready[ai_idx] = true;
      g_model_buy_thr[ai_idx] = best_buy_thr;
      g_model_sell_thr[ai_idx] = best_sell_thr;
      g_model_thr_ready[ai_idx] = true;

      CFX6AIPlugin *runtime = g_plugins.Get(ai_idx);
      if(runtime == NULL) continue;

      runtime.Reset();
      runtime.EnsureInitialized(best_hp);
      FX6_ResetModelAuxState(ai_idx);

      for(int i=i_end; i>=i_start; i--)
      {
         if(i < 0 || i >= ArraySize(samples)) continue;
         FX6_ApplyPreparedSampleToModel(ai_idx, *runtime, samples[i], best_hp);
      }

      g_ai_trained[ai_idx] = true;
      g_ai_last_train_bar[ai_idx] = bar_time;
   }

   g_ai_warmup_done = true;
   Print("FX6 warmup completed: symbol=", symbol,
         ", samples=", warmup_samples,
         ", loops=", warmup_loops,
         ", folds=", warmup_folds);
   return true;
}

void FX6_PrecomputeTrainingSamples(const int i_start,
                                   const int i_end,
                                   const int H,
                                   const double commission_points,
                                   const double cost_buffer_points,
                                   const double ev_threshold_points,
                                   const FX6DataSnapshot &snapshot,
                                   const int &spread_m1[],
                                   const datetime &time_arr[],
                                   const double &close_arr[],
                                   const datetime &time_m5[],
                                   const double &close_m5[],
                                   const int &map_m5[],
                                   const datetime &time_m15[],
                                   const double &close_m15[],
                                   const int &map_m15[],
                                   const datetime &time_h1[],
                                   const double &close_h1[],
                                   const int &map_h1[],
                                   const double &ctx_mean_arr[],
                                   const double &ctx_std_arr[],
                                   const double &ctx_up_arr[],
                                   FX6PreparedSample &samples[])
{
   if(i_end < i_start) return;
   if(i_end < 0) return;

   int need_size = i_end + 1;
   if(ArraySize(samples) < need_size)
      ArrayResize(samples, need_size);

   for(int i=i_start; i<=i_end; i++)
   {
      if(i < 0 || i >= ArraySize(samples)) continue;
      FX6_PrepareTrainingSample(i,
                                H,
                                commission_points,
                                cost_buffer_points,
                                ev_threshold_points,
                                snapshot,
                                spread_m1,
                                time_arr,
                                close_arr,
                                time_m5,
                                close_m5,
                                map_m5,
                                time_m15,
                                close_m15,
                                map_m15,
                                time_h1,
                                close_h1,
                                map_h1,
                                ctx_mean_arr,
                                ctx_std_arr,
                                ctx_up_arr,
                                samples[i]);
   }
}

void FX6_ApplyPreparedSampleToModel(const int ai_idx,
                                    CFX6AIPlugin &plugin,
                                    const FX6PreparedSample &sample,
                                    const FX6AIHyperParams &hp)
{
   if(!sample.valid) return;

   FX6AISampleV2 s2;
   s2.valid = sample.valid;
   s2.label_class = sample.label_class;
   s2.move_points = sample.move_points;
   s2.min_move_points = sample.min_move_points;
   s2.cost_points = sample.cost_points;
   s2.sample_time = sample.sample_time;
   for(int k=0; k<FX6_AI_WEIGHTS; k++)
      s2.x[k] = sample.x[k];

   plugin.TrainV2(s2, hp);
   FX6_UpdateModelMoveStats(ai_idx, sample.move_points);
}

void FX6_TrainModelWindowPrepared(const int ai_idx,
                                  CFX6AIPlugin &plugin,
                                  const int i_start,
                                  const int i_end,
                                  const int epochs,
                                  const FX6AIHyperParams &hp,
                                  const FX6PreparedSample &samples[])
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
         FX6_ApplyPreparedSampleToModel(ai_idx, plugin, samples[i], hp);
   }
}

void FX6_BuildHyperParams(FX6AIHyperParams &hp)
{
   hp.lr = FX6_Clamp(AI_LearningRate, 0.001, 0.200);
   hp.l2 = FX6_Clamp(AI_L2, 0.0, 0.100);

   hp.ftrl_alpha = FX6_Clamp(FTRL_Alpha, 0.001, 1.000);
   hp.ftrl_beta  = FX6_Clamp(FTRL_Beta,  0.000, 5.000);
   hp.ftrl_l1    = FX6_Clamp(FTRL_L1,    0.000, 0.100);
   hp.ftrl_l2    = FX6_Clamp(FTRL_L2,    0.000, 1.000);

   hp.pa_c      = FX6_Clamp(PA_C,      0.010, 10.000);
   hp.pa_margin = FX6_Clamp(PA_Margin, 0.100, 2.000);

   hp.xgb_lr    = FX6_Clamp(XGB_FastLearningRate, 0.001, 0.200);
   hp.xgb_l2    = FX6_Clamp(XGB_FastL2,           0.000, 0.100);
   hp.xgb_split = FX6_Clamp(XGB_SplitThreshold,  -2.000, 2.000);

   hp.mlp_lr   = FX6_Clamp(MLP_LearningRate, 0.0005, 0.0500);
   hp.mlp_l2   = FX6_Clamp(MLP_L2,           0.0000, 0.0500);
   hp.mlp_init = FX6_Clamp(MLP_InitScale,    0.0100, 0.5000);

   hp.tcn_layers = (double)((int)FX6_Clamp((double)TCN_Layers, 2.0, 8.0));
   hp.tcn_kernel = (double)((int)FX6_Clamp((double)TCN_KernelSize, 2.0, 5.0));
   hp.tcn_dilation_base = (double)((int)FX6_Clamp((double)TCN_DilationBase, 1.0, 3.0));

   hp.quantile_lr = FX6_Clamp(Quantile_LearningRate, 0.0001, 0.1000);
   hp.quantile_l2 = FX6_Clamp(Quantile_L2,           0.0000, 0.1000);

   hp.enhash_lr = FX6_Clamp(ENHash_LearningRate, 0.0005, 0.1000);
   hp.enhash_l1 = FX6_Clamp(ENHash_L1,           0.0000, 0.1000);
   hp.enhash_l2 = FX6_Clamp(ENHash_L2,           0.0000, 0.1000);
}

double FX6_RandRange(const double lo, const double hi)
{
   if(hi <= lo) return lo;
   double u = (double)MathRand() / 32767.0;
   return lo + (hi - lo) * FX6_Clamp(u, 0.0, 1.0);
}

void FX6_ResetModelHyperParams()
{
   FX6AIHyperParams base;
   FX6_BuildHyperParams(base);
   double base_buy = AI_BuyThreshold;
   double base_sell = AI_SellThreshold;
   FX6_SanitizeThresholdPair(base_buy, base_sell);

   for(int i=0; i<FX6_AI_COUNT; i++)
   {
      g_model_hp[i] = base;
      g_model_hp_ready[i] = false;
      g_model_buy_thr[i] = base_buy;
      g_model_sell_thr[i] = base_sell;
      g_model_thr_ready[i] = false;
   }
}

void FX6_GetModelHyperParams(const int ai_idx, FX6AIHyperParams &hp)
{
   if(ai_idx >= 0 && ai_idx < FX6_AI_COUNT && g_model_hp_ready[ai_idx])
   {
      hp = g_model_hp[ai_idx];
      return;
   }
   FX6_BuildHyperParams(hp);
}

void FX6_GetModelThresholds(const int ai_idx,
                            const double base_buy,
                            const double base_sell,
                            double &buy_thr,
                            double &sell_thr)
{
   buy_thr = base_buy;
   sell_thr = base_sell;
   FX6_SanitizeThresholdPair(buy_thr, sell_thr);

   if(ai_idx < 0 || ai_idx >= FX6_AI_COUNT) return;
   if(!g_model_thr_ready[ai_idx]) return;

   buy_thr = g_model_buy_thr[ai_idx];
   sell_thr = g_model_sell_thr[ai_idx];
   FX6_SanitizeThresholdPair(buy_thr, sell_thr);
}

void FX6_SampleThresholdPair(const double base_buy,
                             const double base_sell,
                             double &buy_thr,
                             double &sell_thr)
{
   double b0 = base_buy;
   double s0 = base_sell;
   FX6_SanitizeThresholdPair(b0, s0);

   buy_thr = FX6_Clamp(FX6_RandRange(MathMax(0.52, b0 - 0.08), MathMin(0.90, b0 + 0.08)), 0.50, 0.95);
   sell_thr = FX6_Clamp(FX6_RandRange(MathMax(0.08, s0 - 0.08), MathMin(0.48, s0 + 0.08)), 0.05, 0.50);
   FX6_SanitizeThresholdPair(buy_thr, sell_thr);
}

void FX6_SampleModelHyperParams(const int ai_idx,
                                const FX6AIHyperParams &base,
                                FX6AIHyperParams &hp)
{
   hp = base;

   switch(ai_idx)
   {
      case (int)AI_TYPE_SGD_LOGIT:
      case (int)AI_TYPE_LSTM:
      case (int)AI_TYPE_LSTMG:
      case (int)AI_TYPE_S4:
      case (int)AI_TYPE_TFT:
      case (int)AI_TYPE_AUTOFORMER:
      case (int)AI_TYPE_STMN:
      case (int)AI_TYPE_TST:
      case (int)AI_TYPE_GEODESICATTENTION:
      case (int)AI_TYPE_PATCHTST:
      case (int)AI_TYPE_CHRONOS:
      case (int)AI_TYPE_TIMESFM:
         hp.lr = FX6_RandRange(0.0030, 0.0600);
         hp.l2 = FX6_RandRange(0.0000, 0.0300);
         break;

      case (int)AI_TYPE_TCN:
         hp.lr = FX6_RandRange(0.0030, 0.0500);
         hp.l2 = FX6_RandRange(0.0000, 0.0200);
         hp.tcn_layers = (double)((int)MathRound(FX6_RandRange(3.0, 6.0)));
         hp.tcn_kernel = (double)((int)MathRound(FX6_RandRange(2.0, 4.0)));
         hp.tcn_dilation_base = (double)((int)MathRound(FX6_RandRange(1.0, 3.0)));
         break;

      case (int)AI_TYPE_FTRL_LOGIT:
         hp.ftrl_alpha = FX6_RandRange(0.0100, 0.2500);
         hp.ftrl_beta = FX6_RandRange(0.1000, 2.5000);
         hp.ftrl_l1 = FX6_RandRange(0.0000, 0.0100);
         hp.ftrl_l2 = FX6_RandRange(0.0000, 0.1000);
         break;

      case (int)AI_TYPE_PA_LINEAR:
         hp.pa_c = FX6_RandRange(0.0500, 3.0000);
         hp.pa_margin = FX6_RandRange(0.3000, 1.5000);
         break;

      case (int)AI_TYPE_XGB_FAST:
      case (int)AI_TYPE_LIGHTGBM:
      case (int)AI_TYPE_XGBOOST:
      case (int)AI_TYPE_CATBOOST:
         hp.xgb_lr = FX6_RandRange(0.0050, 0.1200);
         hp.xgb_l2 = FX6_RandRange(0.0000, 0.0300);
         hp.xgb_split = FX6_RandRange(-0.8000, 0.8000);
         break;

      case (int)AI_TYPE_MLP_TINY:
         hp.mlp_lr = FX6_RandRange(0.0010, 0.0300);
         hp.mlp_l2 = FX6_RandRange(0.0000, 0.0200);
         hp.mlp_init = FX6_RandRange(0.0300, 0.2500);
         break;

      case (int)AI_TYPE_QUANTILE:
         hp.quantile_lr = FX6_RandRange(0.0010, 0.0500);
         hp.quantile_l2 = FX6_RandRange(0.0000, 0.0200);
         break;

      case (int)AI_TYPE_ENHASH:
         hp.enhash_lr = FX6_RandRange(0.0020, 0.0500);
         hp.enhash_l1 = FX6_RandRange(0.0000, 0.0100);
         hp.enhash_l2 = FX6_RandRange(0.0000, 0.0200);
         break;

      default:
         hp.lr = FX6_RandRange(0.0030, 0.0600);
         hp.l2 = FX6_RandRange(0.0000, 0.0300);
         break;
   }
}

void FX6_ParseContextSymbols(const string raw, string &symbols[])
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
      if(ArraySize(symbols) >= FX6_MAX_CONTEXT_SYMBOLS)
         break;
   }
}

void FX6_PrecomputeContextAggregates(const datetime &main_time[],
                                     FX6ContextSeries &ctx_series[],
                                     const int ctx_count,
                                     const int upto_index,
                                     double &ctx_mean_arr[],
                                     double &ctx_std_arr[],
                                     double &ctx_up_arr[])
{
   int n = ArraySize(main_time);
   ArrayResize(ctx_mean_arr, n);
   ArrayResize(ctx_std_arr, n);
   ArrayResize(ctx_up_arr, n);

   int lag_m1 = 2 * PeriodSeconds(PERIOD_M1);
   if(lag_m1 <= 0) lag_m1 = 120;

   int upto = upto_index;
   if(upto < 0) upto = 0;
   if(upto >= n) upto = n - 1;

   for(int s=0; s<ctx_count; s++)
   {
      if(!ctx_series[s].loaded)
      {
         ArrayResize(ctx_series[s].aligned_idx, 0);
         continue;
      }
      FX6_BuildAlignedIndexMapRange(main_time,
                                    ctx_series[s].time,
                                    lag_m1,
                                    upto,
                                    ctx_series[s].aligned_idx);
   }

   for(int i=0; i<=upto; i++)
   {
      ctx_mean_arr[i] = 0.0;
      ctx_std_arr[i] = 0.0;
      ctx_up_arr[i] = 0.5;

      datetime t_ref = main_time[i];
      if(t_ref <= 0 || ctx_count <= 0) continue;

      double sum = 0.0;
      double sum2 = 0.0;
      int valid = 0;
      int up = 0;

      for(int s=0; s<ctx_count; s++)
      {
         if(!ctx_series[s].loaded) continue;
         int idx = -1;
         if(i >= 0 && i < ArraySize(ctx_series[s].aligned_idx))
            idx = ctx_series[s].aligned_idx[i];
         if(idx < 0) continue;

         double r = FX6_SafeReturn(ctx_series[s].close, idx, idx + 1);
         sum += r;
         sum2 += r * r;
         valid++;
         if(r > 0.0) up++;
      }

      if(valid <= 0) continue;

      double mean = sum / (double)valid;
      double var = (sum2 / (double)valid) - (mean * mean);
      if(var < 0.0) var = 0.0;

      ctx_mean_arr[i] = mean;
      ctx_std_arr[i] = MathSqrt(var);
      ctx_up_arr[i] = (double)up / (double)valid;
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
   FX6_ResetModelHyperParams();
   FX6_ResetReliabilityPending();

   for(int i=0; i<FX6_AI_COUNT; i++)
   {
      g_ai_trained[i] = false;
      g_ai_last_train_bar[i] = 0;
      FX6_ResetModelAuxState(i);
   }

   if(g_plugins_ready)
      g_plugins.ResetAll();
}

int OnInit()
{
   MathSrand((uint)TimeLocal());

   double buy_init = AI_BuyThreshold;
   double sell_init = AI_SellThreshold;
   FX6_SanitizeThresholdPair(buy_init, sell_init);
   if(MathAbs(buy_init - AI_BuyThreshold) > 1e-12 || MathAbs(sell_init - AI_SellThreshold) > 1e-12)
   {
      // Optimizer-safe behavior: keep running and sanitize thresholds in runtime path.
      Print("FX6 warning: threshold inputs are outside recommended relation/range. ",
            "Runtime threshold sanitization will be applied.");
   }

   InitialEquity = AccountInfoDouble(ACCOUNT_EQUITY);
   EquiMax       = InitialEquity;
   TP_Value      = InitialEquity + TP_USD;

   g_plugins_ready = g_plugins.Initialize();
   if(!g_plugins_ready)
      return(INIT_FAILED);

   FX6_ParseContextSymbols(AI_ContextSymbols, g_context_symbols);

   ResetAIState(_Symbol);
   return(INIT_SUCCEEDED);
}

void OnDeinit(const int reason)
{
   g_plugins.Release();
   g_plugins_ready = false;
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

datetime FX6_GetOldestPositionTime()
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

double FX6_NormalizeLot(const string symbol, const double requested_lot)
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

   lot = FX6_Clamp(lot, vmin, vmax);
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

   Print("FX6 warning: CloseAll incomplete. Remaining positions=", PositionsTotal(),
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
      if(!FX6_IsInLiquidSession(symbol,
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

   double maxdd = FX6_Clamp(MaxDD, 0.0, 99.9);
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
      start_t = FX6_GetOldestPositionTime();
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

   int H = PredictionTargetMinutes;
   if(H < 1) H = 1;
   if(H > 720) H = 720;

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
   if(aiType < 0 || aiType >= FX6_AI_COUNT)
      aiType = (int)AI_TYPE_SGD_LOGIT;

   bool ensembleMode = (bool)AI_Ensemble;
   double agreePct = FX6_Clamp(Ensemble_AgreePct, 50.0, 100.0);

   double buyThr = AI_BuyThreshold;
   double sellThr = AI_SellThreshold;
   FX6_SanitizeThresholdPair(buyThr, sellThr);

   double evThresholdPoints = FX6_Clamp(AI_EVThresholdPoints, 0.0, 100.0);
   int evLookback = AI_EVLookbackSamples;
   if(evLookback < 20) evLookback = 20;
   if(evLookback > 400) evLookback = 400;

   if(g_ai_last_symbol != symbol)
      ResetAIState(symbol);

   if(AI_Warmup && !g_ai_warmup_done)
   {
      if(!FX6_WarmupTrainAndTune(symbol))
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

   FX6DataSnapshot snapshot;
   if(!FX6_ExportDataSnapshot(symbol, AI_CommissionPerLotSide, AI_CostBufferPoints, snapshot))
   {
      g_ai_last_reason = "snapshot_export_failed";
      return -1;
   }
   // Keep cache/training keyed to the same closed bar anchor.
   snapshot.bar_time = signal_bar;

   const int FEATURE_LB = 10;
   int needed = (K > base ? K : base) + H + FEATURE_LB;
   int init_start = H;
   int init_end = H + base - 1;
   int online_start = H;
   int online_end = H + K - 1;
   int max_valid = needed - FEATURE_LB - 1;
   if(init_end > max_valid) init_end = max_valid;
   if(online_end > max_valid) online_end = max_valid;
   bool have_init_window = (init_end >= init_start);
   bool have_online_window = (online_end >= online_start);

   int precompute_end = -1;
   if(have_init_window) precompute_end = init_end;
   if(have_online_window && online_end > precompute_end) precompute_end = online_end;
   int align_upto = (precompute_end > 0 ? precompute_end : 0);

   static MqlRates rates_m1[];
   static MqlRates rates_m5[];
   static MqlRates rates_m15[];
   static MqlRates rates_h1[];
   static string cache_symbol = "";
   static datetime last_bar_m1 = 0;
   static datetime last_bar_m5 = 0;
   static datetime last_bar_m15 = 0;
   static datetime last_bar_h1 = 0;

   static double close_arr[];
   static datetime time_arr[];
   static int spread_m1[];
   static FX6ContextSeries ctx_series[];
   static double ctx_mean_arr[];
   static double ctx_std_arr[];
   static double ctx_up_arr[];

   if(cache_symbol != symbol)
   {
      cache_symbol = symbol;
      last_bar_m1 = 0;
      last_bar_m5 = 0;
      last_bar_m15 = 0;
      last_bar_h1 = 0;
      ArrayResize(rates_m1, 0);
      ArrayResize(rates_m5, 0);
      ArrayResize(rates_m15, 0);
      ArrayResize(rates_h1, 0);
      ArrayResize(ctx_series, 0);
   }

   FX6_AdvanceReliabilityClock(signal_bar);
   int signal_seq = g_rel_clock_seq;

   if(!FX6_UpdateRatesRolling(symbol, PERIOD_M1, needed, last_bar_m1, rates_m1))
   {
      g_ai_last_reason = "m1_series_load_failed";
      return -1;
   }
   FX6_ExtractRatesCloseTimeSpread(rates_m1, close_arr, time_arr, spread_m1);
   if(ArraySize(close_arr) < needed || ArraySize(time_arr) < needed || ArraySize(spread_m1) < needed)
   {
      g_ai_last_reason = "m1_series_size_failed";
      return -1;
   }

   int needed_m5 = (needed / 5) + 80;
   int needed_m15 = (needed / 15) + 80;
   int needed_h1 = (needed / 60) + 80;

   static double close_m5[];
   static datetime time_m5[];
   static double close_m15[];
   static datetime time_m15[];
   static double close_h1[];
   static datetime time_h1[];
   static int map_m5[];
   static int map_m15[];
   static int map_h1[];
   if(FX6_UpdateRatesRolling(symbol, PERIOD_M5, needed_m5, last_bar_m5, rates_m5))
      FX6_ExtractRatesCloseTime(rates_m5, close_m5, time_m5);
   else
   {
      ArrayResize(close_m5, 0);
      ArrayResize(time_m5, 0);
      ArrayResize(map_m5, 0);
   }

   if(FX6_UpdateRatesRolling(symbol, PERIOD_M15, needed_m15, last_bar_m15, rates_m15))
      FX6_ExtractRatesCloseTime(rates_m15, close_m15, time_m15);
   else
   {
      ArrayResize(close_m15, 0);
      ArrayResize(time_m15, 0);
      ArrayResize(map_m15, 0);
   }

   if(FX6_UpdateRatesRolling(symbol, PERIOD_H1, needed_h1, last_bar_h1, rates_h1))
      FX6_ExtractRatesCloseTime(rates_h1, close_h1, time_h1);
   else
   {
      ArrayResize(close_h1, 0);
      ArrayResize(time_h1, 0);
      ArrayResize(map_h1, 0);
   }

   int lag_m5 = 2 * PeriodSeconds(PERIOD_M5);
   int lag_m15 = 2 * PeriodSeconds(PERIOD_M15);
   int lag_h1 = 2 * PeriodSeconds(PERIOD_H1);
   if(lag_m5 <= 0) lag_m5 = 600;
   if(lag_m15 <= 0) lag_m15 = 1800;
   if(lag_h1 <= 0) lag_h1 = 7200;

   FX6_BuildAlignedIndexMapRange(time_arr, time_m5, lag_m5, align_upto, map_m5);
   FX6_BuildAlignedIndexMapRange(time_arr, time_m15, lag_m15, align_upto, map_m15);
   FX6_BuildAlignedIndexMapRange(time_arr, time_h1, lag_h1, align_upto, map_h1);

   int ctx_count = ArraySize(g_context_symbols);
   if(ctx_count > FX6_MAX_CONTEXT_SYMBOLS) ctx_count = FX6_MAX_CONTEXT_SYMBOLS;
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

      ctx_series[s].loaded = FX6_UpdateRatesRolling(ctx_symbol,
                                                    PERIOD_M1,
                                                    needed,
                                                    ctx_series[s].last_bar_time,
                                                    ctx_series[s].rates);
      if(ctx_series[s].loaded)
      {
         FX6_ExtractRatesCloseTime(ctx_series[s].rates,
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

   FX6_PrecomputeContextAggregates(time_arr,
                                   ctx_series,
                                   ctx_count,
                                   align_upto,
                                   ctx_mean_arr,
                                   ctx_std_arr,
                                   ctx_up_arr);

   double cost_buffer_points = (AI_CostBufferPoints < 0.0 ? 0.0 : AI_CostBufferPoints);
   double commission_points = snapshot.commission_points;
   double spread_pred = FX6_GetSpreadAtIndex(0, spread_m1, snapshot.spread_points);
   double min_move_pred = spread_pred + commission_points + cost_buffer_points;
   if(min_move_pred < 0.0) min_move_pred = 0.0;
   double ctx_mean_pred = FX6_GetArrayValue(ctx_mean_arr, 0, 0.0);
   double ctx_std_pred = FX6_GetArrayValue(ctx_std_arr, 0, 0.0);
   double ctx_up_pred = FX6_GetArrayValue(ctx_up_arr, 0, 0.5);

   double feat_pred[FX6_AI_FEATURES];
   if(!FX6_ComputeFeatureVector(0,
                                spread_pred,
                                time_arr,
                                close_arr,
                                time_m5,
                                close_m5,
                                map_m5,
                                time_m15,
                                close_m15,
                                map_m15,
                                time_h1,
                                close_h1,
                                map_h1,
                                ctx_mean_pred,
                                ctx_std_pred,
                                ctx_up_pred,
                                feat_pred))
   {
      g_ai_last_signal_bar = signal_bar;
      g_ai_last_signal_key = decisionKey;
      g_ai_last_signal = -1;
      g_ai_last_reason = "predict_features_failed";
      return -1;
   }

   double x_pred[FX6_AI_WEIGHTS];
   FX6_BuildInputVector(feat_pred, x_pred);

   double fallback_expected_move = FX6_EstimateExpectedAbsMovePoints(close_arr,
                                                                      H,
                                                                      evLookback,
                                                                      snapshot.point);
   if(fallback_expected_move <= 0.0)
      fallback_expected_move = min_move_pred;

   static FX6PreparedSample samples[];
   if(precompute_end >= H)
   {
      FX6_PrecomputeTrainingSamples(H,
                                    precompute_end,
                                    H,
                                    commission_points,
                                    cost_buffer_points,
                                    evThresholdPoints,
                                    snapshot,
                                    spread_m1,
                                    time_arr,
                                    close_arr,
                                    time_m5,
                                    close_m5,
                                    map_m5,
                                    time_m15,
                                    close_m15,
                                    map_m15,
                                    time_h1,
                                    close_h1,
                                    map_h1,
                                    ctx_mean_arr,
                                    ctx_std_arr,
                                    ctx_up_arr,
                                    samples);
   }

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
      for(int ai_idx=0; ai_idx<FX6_AI_COUNT; ai_idx++)
      {
         if(g_plugins.Get(ai_idx) == NULL) continue;
         int sz = ArraySize(active_ai_ids);
         ArrayResize(active_ai_ids, sz + 1);
         active_ai_ids[sz] = ai_idx;
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

   int buyVotes = 0;
   int sellVotes = 0;
   int decisiveVotes = 0;
   double weightedBuyVotes = 0.0;
   double weightedSellVotes = 0.0;
   int singleSignal = -1;

   for(int m=0; m<ArraySize(active_ai_ids); m++)
   {
      int ai_idx = active_ai_ids[m];

      CFX6AIPlugin *plugin = g_plugins.Get(ai_idx);
      if(plugin == NULL)
         continue;

      FX6AIHyperParams hp_model;
      FX6_GetModelHyperParams(ai_idx, hp_model);
      plugin.EnsureInitialized(hp_model);

      // Reliability is updated only from matured, out-of-sample predictions.
      FX6_UpdateReliabilityFromPending(ai_idx,
                                       signal_seq,
                                       H,
                                       snapshot,
                                       spread_m1,
                                       close_arr,
                                       commission_points,
                                       cost_buffer_points,
                                       evThresholdPoints);

      if(!g_ai_trained[ai_idx])
      {
         if(have_init_window)
         {
            FX6_TrainModelWindowPrepared(ai_idx,
                                         *plugin,
                                         init_start,
                                         init_end,
                                         trainEpochs,
                                         hp_model,
                                         samples);
         }

         g_ai_trained[ai_idx] = true;
         g_ai_last_train_bar[ai_idx] = snapshot.bar_time;
      }
      else if(snapshot.bar_time != g_ai_last_train_bar[ai_idx])
      {
         if(have_online_window)
         {
            FX6_TrainModelWindowPrepared(ai_idx,
                                         *plugin,
                                         online_start,
                                         online_end,
                                         onlineEpochs,
                                         hp_model,
                                         samples);
         }

         g_ai_last_train_bar[ai_idx] = snapshot.bar_time;
      }

      FX6AIPredictV2 req;
      req.min_move_points = min_move_pred;
      req.cost_points = min_move_pred;
      req.sample_time = snapshot.bar_time;
      for(int k=0; k<FX6_AI_WEIGHTS; k++)
         req.x[k] = x_pred[k];

      FX6AIPredictionV2 pred;
      plugin.PredictV2(req, hp_model, pred);

      double class_probs_pred[3];
      class_probs_pred[0] = pred.class_probs[0];
      class_probs_pred[1] = pred.class_probs[1];
      class_probs_pred[2] = pred.class_probs[2];

      double expected_move = pred.expected_move_points;
      if(expected_move <= 0.0)
         expected_move = FX6_GetModelExpectedMove(ai_idx, fallback_expected_move);
      if(expected_move <= 0.0)
         expected_move = fallback_expected_move;

      FX6_EnqueueReliabilityPending(ai_idx, signal_seq, class_probs_pred);

      double modelBuyThr = buyThr;
      double modelSellThr = sellThr;
      FX6_GetModelThresholds(ai_idx, buyThr, sellThr, modelBuyThr, modelSellThr);

      double buyMinProb = modelBuyThr;
      double sellMinProb = 1.0 - modelSellThr;
      double skipMinProb = 0.55;
      FX6_DeriveAdaptiveThresholds(modelBuyThr,
                                   modelSellThr,
                                   min_move_pred,
                                   expected_move,
                                   feat_pred[5],
                                   buyMinProb,
                                   sellMinProb,
                                   skipMinProb);

      int signal = FX6_ClassSignalFromEV(class_probs_pred,
                                         buyMinProb,
                                         sellMinProb,
                                         skipMinProb,
                                         expected_move,
                                         min_move_pred,
                                         evThresholdPoints);

      if(ensembleMode == 0)
      {
         singleSignal = signal;
      }
      else
      {
         double voteWeight = FX6_GetModelVoteWeight(ai_idx);
         if(signal == 1)
         {
            buyVotes++;
            decisiveVotes++;
            weightedBuyVotes += voteWeight;
         }
         else if(signal == 0)
         {
            sellVotes++;
            decisiveVotes++;
            weightedSellVotes += voteWeight;
         }
      }
   }

   int decision = -1;
   if(ensembleMode == 0)
   {
      decision = singleSignal;
   }
   else
   {
      double decisiveWeight = weightedBuyVotes + weightedSellVotes;
      if(decisiveVotes > 0 && decisiveWeight > 0.0)
      {
         double buyPct = 100.0 * (weightedBuyVotes / decisiveWeight);
         double sellPct = 100.0 * (weightedSellVotes / decisiveWeight);

         if(buyPct >= agreePct && weightedBuyVotes > weightedSellVotes) decision = 1;
         else if(sellPct >= agreePct && weightedSellVotes > weightedBuyVotes) decision = 0;
      }
   }

   g_ai_last_signal_bar = signal_bar;
   g_ai_last_signal_key = decisionKey;
   g_ai_last_signal = decision;
   if(decision == 1) g_ai_last_reason = "buy";
   else if(decision == 0) g_ai_last_reason = "sell";
   else if(ensembleMode != 0 && decisiveVotes == 0) g_ai_last_reason = "no_decisive_votes";
   else g_ai_last_reason = "no_consensus_or_ev";

   return decision;
}

//-------------------------- SEND TRADE ------------------------------
void SendTrade()
{
   datetime bar_t = iTime(_Symbol, PERIOD_M1, 1);
   bool emit_debug = (AI_DebugFlow && bar_t > 0 && bar_t != g_last_debug_bar);
   if(emit_debug) g_last_debug_bar = bar_t;

   string trade_reason = "ok";
   if(TradePossible(_Symbol, trade_reason) != 1)
   {
      if(emit_debug)
         Print("FX6 debug: Trade blocked. reason=", trade_reason);
      return;
   }

   Calc_TP();

   int direction = SpecialDirectionAI(_Symbol);
   if(direction == -1)
   {
      if(emit_debug)
         Print("FX6 debug: AI no-trade. reason=", g_ai_last_reason);
      return;
   }

   double trade_lot = FX6_NormalizeLot(_Symbol, Lot);
   if(trade_lot <= 0.0) return;

   bool ok = false;
   if(direction == 1) ok = trade.Buy(trade_lot, _Symbol, 0, 0, 0, "Buy");
   else               ok = trade.Sell(trade_lot, _Symbol, 0, 0, 0, "Sell");

   if(!ok && emit_debug)
      Print("FX6 debug: Order send failed. retcode=", (int)trade.ResultRetcode(),
            " desc=", trade.ResultRetcodeDescription());
   else if(ok && emit_debug)
      Print("FX6 debug: Order sent. direction=", direction, " lot=", DoubleToString(trade_lot, 2));

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
   FX6_ProcessReliabilityBar(_Symbol);

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

   if(OrdersTotal() + PositionsTotal() == 0)
      SendTrade();
}
//+------------------------------------------------------------------+
