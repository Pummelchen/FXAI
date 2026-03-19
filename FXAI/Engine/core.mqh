#ifndef __FXAI_CORE_MQH__
#define __FXAI_CORE_MQH__

#define FXAI_CONTEXT_TOP_SYMBOLS 3
#define FXAI_MTF_STATE_FEATURES_PER_TF 4
#define FXAI_MAIN_MTF_TF_COUNT 4
#define FXAI_CONTEXT_MTF_TF_COUNT 5
#define FXAI_MAIN_MTF_FEATURE_OFFSET 84
#define FXAI_CONTEXT_MTF_FEATURE_OFFSET (FXAI_MAIN_MTF_FEATURE_OFFSET + FXAI_MAIN_MTF_TF_COUNT * FXAI_MTF_STATE_FEATURES_PER_TF)
#define FXAI_AI_FEATURES (FXAI_CONTEXT_MTF_FEATURE_OFFSET + FXAI_CONTEXT_TOP_SYMBOLS * FXAI_CONTEXT_MTF_TF_COUNT * FXAI_MTF_STATE_FEATURES_PER_TF)
#define FXAI_AI_WEIGHTS (FXAI_AI_FEATURES + 1)
#define FXAI_AI_MLP_HIDDEN 12
#define FXAI_AI_COUNT 32
#define FXAI_CONFORMAL_DEPTH 96
#define FXAI_NORM_METHOD_COUNT 15
#define FXAI_ENHASH_BUCKETS 128
#define FXAI_PLUGIN_CLASS_FEATURES 5
#define FXAI_PLUGIN_REGIME_BUCKETS 12
#define FXAI_PLUGIN_SESSION_BUCKETS 6
#define FXAI_PLUGIN_HORIZON_BUCKETS 8
#define FXAI_PLUGIN_REPLAY_CAPACITY 96
#define FXAI_PLUGIN_REPLAY_STEPS 2
#define FXAI_CONTEXT_BASE_SYMBOL_FEATS 4
#define FXAI_CONTEXT_SHARED_ADAPTER_FEATS 4
#define FXAI_CONTEXT_SHARED_OFFSET (FXAI_CONTEXT_TOP_SYMBOLS * FXAI_CONTEXT_BASE_SYMBOL_FEATS)
#define FXAI_CONTEXT_SLOT_MTF_FEATS (FXAI_CONTEXT_MTF_TF_COUNT * FXAI_MTF_STATE_FEATURES_PER_TF)
#define FXAI_CONTEXT_MTF_OFFSET (FXAI_CONTEXT_SHARED_OFFSET + FXAI_CONTEXT_SHARED_ADAPTER_FEATS)
#define FXAI_CONTEXT_EXTRA_FEATS (FXAI_CONTEXT_MTF_OFFSET + FXAI_CONTEXT_TOP_SYMBOLS * FXAI_CONTEXT_SLOT_MTF_FEATS)
#define FXAI_CONTEXT_DYNAMIC_POOL 12
#define FXAI_SHARED_TRANSFER_FEATURES 20
#define FXAI_SHARED_TRANSFER_LATENT 12
#define FXAI_SHARED_TRANSFER_DOMAIN_BUCKETS 8
#define FXAI_SHARED_TRANSFER_HORIZON_BUCKETS 8
#define FXAI_EXEC_TRACE_BARS 12
#define FXAI_API_VERSION_V4 4
#define FXAI_MAX_SEQUENCE_BARS 96

#ifndef FXAI_PATHFLAG_DUAL_HIT
#define FXAI_PATHFLAG_DUAL_HIT 1
#endif
#ifndef FXAI_PATHFLAG_KILLED_EARLY
#define FXAI_PATHFLAG_KILLED_EARLY 2
#endif
#ifndef FXAI_PATHFLAG_SPREAD_STRESS
#define FXAI_PATHFLAG_SPREAD_STRESS 4
#endif
#ifndef FXAI_PATHFLAG_SLOW_HIT
#define FXAI_PATHFLAG_SLOW_HIT 8
#endif


enum ENUM_AI_TYPE
{
   AI_AUTOFORMER = 0,
   AI_CATBOOST,
   AI_CHRONOS,
   AI_ENHASH,
   AI_FTRL_LOGIT,
   AI_GEODESICATTENTION,
   AI_LIGHTGBM,
   AI_LSTM,
   AI_LSTMG,
   AI_MLP_TINY,
   AI_PA_LINEAR,
   AI_PATCHTST,
   AI_QUANTILE,
   AI_S4,
   AI_SGD_LOGIT,
   AI_STMN,
   AI_TCN,
   AI_TFT,
   AI_TIMESFM,
   AI_TST,
   AI_XGB_FAST,
   AI_XGBOOST,
   AI_CFX_WORLD,
   AI_LOFFM,
   AI_TRR,
   AI_GRAPHWM,
   AI_MOE_CONFORMAL,
   AI_RETRDIFF,
   AI_M1SYNC,
   AI_BUY_ONLY,
   AI_SELL_ONLY,
   AI_RANDOM_NOSKIP
};


enum ENUM_FXAI_LABEL_CLASS
{
   FXAI_LABEL_SELL = 0,
   FXAI_LABEL_BUY  = 1,
   FXAI_LABEL_SKIP = 2
};

enum ENUM_FXAI_AI_FAMILY
{
   FXAI_FAMILY_LINEAR = 0,
   FXAI_FAMILY_TREE,
   FXAI_FAMILY_RECURRENT,
   FXAI_FAMILY_CONVOLUTIONAL,
   FXAI_FAMILY_TRANSFORMER,
   FXAI_FAMILY_STATE_SPACE,
   FXAI_FAMILY_DISTRIBUTIONAL,
   FXAI_FAMILY_MIXTURE,
   FXAI_FAMILY_RETRIEVAL,
   FXAI_FAMILY_WORLD_MODEL,
   FXAI_FAMILY_RULE_BASED,
   FXAI_FAMILY_OTHER
};

enum ENUM_FXAI_REFERENCE_TIER
{
   FXAI_REFERENCE_FULL_NATIVE = 0,
   FXAI_REFERENCE_COMPRESSED_NATIVE,
   FXAI_REFERENCE_SURROGATE,
   FXAI_REFERENCE_RULE_BASELINE
};

enum ENUM_FXAI_SEQUENCE_STYLE
{
   FXAI_SEQ_STYLE_GENERIC = 0,
   FXAI_SEQ_STYLE_RECURRENT,
   FXAI_SEQ_STYLE_CONVOLUTIONAL,
   FXAI_SEQ_STYLE_TRANSFORMER,
   FXAI_SEQ_STYLE_STATE_SPACE,
   FXAI_SEQ_STYLE_WORLD
};

enum ENUM_FXAI_EXECUTION_PROFILE
{
   FXAI_EXEC_DEFAULT = 0,
   FXAI_EXEC_TIGHT_FX,
   FXAI_EXEC_PRIME_ECN,
   FXAI_EXEC_RETAIL_FX,
   FXAI_EXEC_STRESS
};

enum ENUM_FXAI_PLUGIN_CAPABILITY
{
   FXAI_CAP_ONLINE_LEARNING    = 1,
   FXAI_CAP_REPLAY             = 2,
   FXAI_CAP_STATEFUL           = 4,
   FXAI_CAP_WINDOW_CONTEXT     = 8,
   FXAI_CAP_MULTI_HORIZON      = 16,
   FXAI_CAP_NATIVE_DISTRIBUTION= 32,
   FXAI_CAP_SELF_TEST          = 64
};

enum ENUM_FXAI_RESET_REASON
{
   FXAI_RESET_FULL = 0,
   FXAI_RESET_SYMBOL_CHANGE,
   FXAI_RESET_SESSION_CHANGE,
   FXAI_RESET_REGIME_CHANGE,
   FXAI_RESET_COMPLIANCE,
   FXAI_RESET_MANUAL
};

enum ENUM_FXAI_FEATURE_GROUP
{
   FXAI_FEAT_GROUP_PRICE = 0,
   FXAI_FEAT_GROUP_MULTI_TIMEFRAME,
   FXAI_FEAT_GROUP_VOLATILITY,
   FXAI_FEAT_GROUP_TIME,
   FXAI_FEAT_GROUP_CONTEXT,
   FXAI_FEAT_GROUP_COST,
   FXAI_FEAT_GROUP_MICROSTRUCTURE,
   FXAI_FEAT_GROUP_FILTERS
};

enum ENUM_FXAI_FEATURE_SCHEMA
{
   FXAI_SCHEMA_FULL = 1,
   FXAI_SCHEMA_SPARSE_STAT = 2,
   FXAI_SCHEMA_SEQUENCE = 3,
   FXAI_SCHEMA_RULE = 4,
   FXAI_SCHEMA_TREE = 5,
   FXAI_SCHEMA_CONTEXTUAL = 6
};

enum ENUM_FXAI_FEATURE_NORMALIZATION
{
   FXAI_NORM_EXISTING = 0,          // Existing project scaling: regime-normalized + clipped.
   FXAI_NORM_MINMAX_BUFFER5,        // Min/Max mapped to [0,1] with +/-5% buffer.
   FXAI_NORM_CHANGE_PERCENT,        // Percent change versus previous feature value.
   FXAI_NORM_BINARY_01,             // Binary step: higher than previous -> 1, else 0.
   FXAI_NORM_LOG_RETURN,            // Log-return on buffered min/max mapped values.
   FXAI_NORM_RELATIVE_CHANGE_PERCENT,// Symmetric relative change in percent.
   FXAI_NORM_CANDLE_GEOMETRY,       // Candle body/range/wicks normalized by prior close or bar range.
   FXAI_NORM_VOL_STD_RETURNS,       // Return features scaled by rolling past-only return std.
   FXAI_NORM_ATR_NATR_UNIT,         // Change features scaled by ATR/NATR volatility unit.
   FXAI_NORM_ZSCORE,                // Z-score standardization.
   FXAI_NORM_ROBUST_MEDIAN_IQR,     // Robust scaling by median/IQR.
   FXAI_NORM_QUANTILE_TO_NORMAL,    // Rank/quantile map (approx.) to normal space.
   FXAI_NORM_POWER_YEOJOHNSON,      // Yeo-Johnson power transform + standardize.
   FXAI_NORM_REVIN,                 // RevIN instance/window normalization.
   FXAI_NORM_DAIN                   // DAIN adaptive normalization (lightweight learnable).
};

enum ENUM_FXAI_MTF_STATE_METRIC
{
   FXAI_MTF_BODY_BIAS = 0,
   FXAI_MTF_CLOSE_LOCATION,
   FXAI_MTF_RANGE_PRESSURE,
   FXAI_MTF_SPREAD_PRESSURE
};

int FXAI_GetM1SyncBars(void);

struct FXAIAIHyperParams
{
   double lr;
   double l2;

   double ftrl_alpha;
   double ftrl_beta;
   double ftrl_l1;
   double ftrl_l2;

   double pa_c;
   double pa_margin;

   double xgb_lr;
   double xgb_l2;
   double xgb_split;

   double mlp_lr;
   double mlp_l2;
   double mlp_init;

   double quantile_lr;
   double quantile_l2;

   double enhash_lr;
   double enhash_l1;
   double enhash_l2;

   double tcn_layers;
   double tcn_kernel;
   double tcn_dilation_base;
};

struct FXAIDataSnapshot
{
   string symbol;
   datetime bar_time;
   double point;
   double spread_points;
   double commission_points;
   double min_move_points;
};

struct FXAIAIManifestV4
{
   int api_version;
   int ai_id;
   string ai_name;
   int family;
   int reference_tier;
   ulong capability_mask;
   int feature_schema_id;
   ulong feature_groups_mask;
   int min_horizon_minutes;
   int max_horizon_minutes;
   int min_sequence_bars;
   int max_sequence_bars;
};

struct FXAIAIContextV4
{
   int api_version;
   int regime_id;
   int session_bucket;
   int horizon_minutes;
   int feature_schema_id;
   int normalization_method_id;
   int sequence_bars;
   double cost_points;
   double min_move_points;
   double point_value;
   double domain_hash;
   datetime sample_time;
};

struct FXAIAITrainRequestV4
{
   bool valid;
   FXAIAIContextV4 ctx;
   int label_class;
   double move_points;
   double sample_weight;
   double mfe_points;
   double mae_points;
   double time_to_hit_frac;
   int    path_flags;
   double path_risk;
   double fill_risk;
   double masked_step_target;
   double next_vol_target;
   double regime_shift_target;
   double context_lead_target;
   int window_size;
   double x[FXAI_AI_WEIGHTS];
   double x_window[FXAI_MAX_SEQUENCE_BARS][FXAI_AI_WEIGHTS];
};

struct FXAIAIPredictRequestV4
{
   bool valid;
   FXAIAIContextV4 ctx;
   int window_size;
   double x[FXAI_AI_WEIGHTS];
   double x_window[FXAI_MAX_SEQUENCE_BARS][FXAI_AI_WEIGHTS];
};

struct FXAIAIPredictionV4
{
   double class_probs[3];
   double move_mean_points;
   double move_q25_points;
   double move_q50_points;
   double move_q75_points;
   double mfe_mean_points;
   double mae_mean_points;
   double hit_time_frac;
   double path_risk;
   double fill_risk;
   double confidence;
   double reliability;
};

struct FXAIExecutionProfile
{
   int    profile_id;
   double commission_per_lot_side;
   double cost_buffer_points;
   double slippage_points;
   double fill_penalty_points;
   double slippage_cost_weight;
   double slippage_stress_weight;
   double slippage_horizon_weight;
   double dual_hit_penalty;
   double slow_hit_penalty;
   double spread_shock_penalty;
   double partial_fill_penalty;
   double latency_penalty_points;
   double allowed_deviation_points;
};

struct FXAIExecutionReplayFrame
{
   double slippage_mult;
   double fill_mult;
   double latency_add_points;
   double reject_prob;
   double partial_fill_prob;
   double drift_penalty_points;
   int    event_flags;
};

struct FXAIExecutionTraceStats
{
   double spread_mean_ratio;
   double spread_peak_ratio;
   double range_mean_ratio;
   double body_efficiency;
   double gap_ratio;
   double reversal_ratio;
   double session_transition_exposure;
   double rollover_exposure;
};

struct FXAIAIModelOutputV4
{
   double class_probs[3];
   double move_mean_points;
   double move_q25_points;
   double move_q50_points;
   double move_q75_points;
   double mfe_mean_points;
   double mae_mean_points;
   double hit_time_frac;
   double path_risk;
   double fill_risk;
   double confidence;
   double reliability;
   bool   has_quantiles;
   bool   has_confidence;
   bool   has_path_quality;
};

void FXAI_ApplyConformalPredictionAdjustment(const int ai_idx,
                                             const int regime_id,
                                             const int horizon_minutes,
                                             const double min_move_points,
                                             FXAIAIPredictionV4 &pred);
void FXAI_ResetConformalState(void);
void FXAI_EnqueueConformalPending(const int ai_idx,
                                  const int signal_seq,
                                  const int regime_id,
                                  const int horizon_minutes,
                                  const FXAIAIPredictionV4 &pred);
void FXAI_UpdateConformalFromPending(const int ai_idx,
                                     const int current_signal_seq,
                                     const FXAIDataSnapshot &snapshot,
                                     const int &spread_m1[],
                                     const datetime &time_arr[],
                                     const double &high_arr[],
                                     const double &low_arr[],
                                     const double &close_arr[],
                                     const double commission_points,
                                     const double cost_buffer_points,
                                     const double ev_threshold_points);
void FXAI_ResolveExecutionProfile(FXAIExecutionProfile &profile);
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
                                   int &path_flags);
void FXAI_MarkRuntimeArtifactsDirty(void);
bool FXAI_SaveRuntimeArtifacts(const string symbol);
bool FXAI_LoadRuntimeArtifacts(const string symbol);
void FXAI_MaybeSaveRuntimeArtifacts(const string symbol,
                                    const datetime bar_time);
void FXAI_ClearExecutionTraceStats(FXAIExecutionTraceStats &trace);
void FXAI_BuildExecutionTraceStats(const int i,
                                   const int horizon_minutes,
                                   const double point_value,
                                   const datetime &time_arr[],
                                   const double &open_arr[],
                                   const double &high_arr[],
                                   const double &low_arr[],
                                   const double &close_arr[],
                                   const int &spread_arr[],
                                   FXAIExecutionTraceStats &trace);

string FXAI_ReferenceTierName(const int tier)
{
   switch(tier)
   {
      case FXAI_REFERENCE_FULL_NATIVE: return "full_native";
      case FXAI_REFERENCE_COMPRESSED_NATIVE: return "compressed_native";
      case FXAI_REFERENCE_SURROGATE: return "surrogate";
      case FXAI_REFERENCE_RULE_BASELINE: return "rule_baseline";
      default: return "unknown";
   }
}

int FXAI_DefaultReferenceTierForAI(const int ai_id)
{
   switch(ai_id)
   {
      case AI_CHRONOS:
      case AI_TIMESFM:
      case AI_CFX_WORLD:
      case AI_GRAPHWM:
         return (int)FXAI_REFERENCE_SURROGATE;

      case AI_S4:
      case AI_STMN:
      case AI_TRR:
      case AI_RETRDIFF:
      case AI_LOFFM:
      case AI_MOE_CONFORMAL:
      case AI_TST:
      case AI_AUTOFORMER:
      case AI_PATCHTST:
      case AI_GEODESICATTENTION:
      case AI_CATBOOST:
      case AI_LIGHTGBM:
      case AI_XGB_FAST:
      case AI_XGBOOST:
      case AI_QUANTILE:
      case AI_ENHASH:
      case AI_MLP_TINY:
      case AI_LSTM:
      case AI_LSTMG:
      case AI_TCN:
      case AI_TFT:
         return (int)FXAI_REFERENCE_COMPRESSED_NATIVE;

      case AI_FTRL_LOGIT:
      case AI_PA_LINEAR:
      case AI_SGD_LOGIT:
         return (int)FXAI_REFERENCE_FULL_NATIVE;

      case AI_M1SYNC:
      case AI_BUY_ONLY:
      case AI_SELL_ONLY:
      case AI_RANDOM_NOSKIP:
         return (int)FXAI_REFERENCE_RULE_BASELINE;

      default:
         return (int)FXAI_REFERENCE_FULL_NATIVE;
   }
}

bool FXAI_HasCapability(const ulong capability_mask,
                        const int capability)
{
   ulong cap = (ulong)capability;
   return ((capability_mask & cap) == cap);
}

int FXAI_DeriveSessionBucket(const datetime sample_time)
{
   datetime t = sample_time;
   if(t <= 0) t = TimeCurrent();

   MqlDateTime dt;
   TimeToStruct(t, dt);
   int hour = dt.hour;
   if(hour < 0) hour = 0;
   if(hour > 23) hour = 23;

   int bucket = hour / 4;
   if(bucket < 0) bucket = 0;
   if(bucket >= FXAI_PLUGIN_SESSION_BUCKETS) bucket = FXAI_PLUGIN_SESSION_BUCKETS - 1;
   return bucket;
}

double FXAI_Clamp(const double v, const double lo, const double hi)
{
   if(v < lo) return lo;
   if(v > hi) return hi;
   return v;
}

double FXAI_Sigmoid(const double z)
{
   if(z > 35.0) return 1.0;
   if(z < -35.0) return 0.0;
   return 1.0 / (1.0 + MathExp(-z));
}

double FXAI_Logit(const double p)
{
   double x = FXAI_Clamp(p, 1e-6, 1.0 - 1e-6);
   return MathLog(x / (1.0 - x));
}

double FXAI_SymbolHash01(const string symbol)
{
   string s = symbol;
   if(StringLen(s) <= 0)
      s = _Symbol;
   uint h = 2166136261U;
   int n = StringLen(s);
   for(int i=0; i<n; i++)
   {
      uint ch = (uint)StringGetCharacter(s, i);
      h ^= ch;
      h *= 16777619U;
   }
   return (double)(h % 100000U) / 100000.0;
}

double FXAI_Tanh(const double z)
{
   if(z > 18.0) return 1.0;
   if(z < -18.0) return -1.0;
   double e2 = MathExp(2.0 * z);
   return (e2 - 1.0) / (e2 + 1.0);
}

double FXAI_DotLinear(const double &w[], const double &x[])
{
   double z = 0.0;
   for(int i=0; i<FXAI_AI_WEIGHTS; i++)
      z += w[i] * x[i];
   return z;
}

double FXAI_Sign(const double v)
{
   if(v > 0.0) return 1.0;
   if(v < 0.0) return -1.0;
   return 0.0;
}

double FXAI_ClipSym(const double v, const double limit_abs)
{
   double lim = (limit_abs > 0.0 ? limit_abs : 0.0);
   if(lim <= 0.0) return v;
   if(v > lim) return lim;
   if(v < -lim) return -lim;
   return v;
}

double FXAI_MoveWeight(const double move_points)
{
   double a = MathAbs(move_points);
   // Keep weighting lightweight and bounded for stable online updates.
   return FXAI_Clamp(1.0 + (0.05 * a), 0.80, 3.00);
}

double FXAI_MoveEdgeWeight(const double move_points, const double cost_points)
{
   double mv = MathAbs(move_points);
   double c = (cost_points > 0.0 ? cost_points : 0.0);
   double edge = mv - c;
   double denom = MathMax(c, 1.0);
   return FXAI_Clamp(0.50 + (edge / denom), 0.25, 4.00);
}

double FXAI_PathRiskFromTargets(const double mfe_points,
                                const double mae_points,
                                const double min_move_points,
                                const double time_to_hit_frac,
                                const int path_flags)
{
   double mfe = MathMax(MathAbs(mfe_points), MathMax(min_move_points, 0.10));
   double mae = MathMax(MathAbs(mae_points), 0.0);
   double adverse_ratio = FXAI_Clamp(mae / mfe, 0.0, 3.0);
   double slow = FXAI_Clamp(time_to_hit_frac, 0.0, 1.0);
   double risk = 0.45 * adverse_ratio + 0.30 * slow;
   if((path_flags & 1) != 0)
      risk += 0.15;
   if((path_flags & 2) != 0)
      risk += 0.10;
   if((path_flags & 8) != 0)
      risk += 0.08;
   return FXAI_Clamp(risk, 0.0, 1.0);
}

double FXAI_FillRiskFromTargets(const double spread_stress_points,
                                const double min_move_points,
                                const double cost_points)
{
   double denom = MathMax(min_move_points + MathMax(cost_points, 0.0), 0.25);
   return FXAI_Clamp(MathAbs(spread_stress_points) / denom, 0.0, 1.0);
}

void FXAI_ClearExecutionProfile(FXAIExecutionProfile &profile)
{
   profile.profile_id = (int)FXAI_EXEC_DEFAULT;
   profile.commission_per_lot_side = 0.0;
   profile.cost_buffer_points = 2.0;
   profile.slippage_points = 0.0;
   profile.fill_penalty_points = 0.0;
   profile.slippage_cost_weight = 0.04;
   profile.slippage_stress_weight = 0.18;
   profile.slippage_horizon_weight = 0.02;
   profile.dual_hit_penalty = 0.12;
   profile.slow_hit_penalty = 0.10;
   profile.spread_shock_penalty = 0.25;
   profile.partial_fill_penalty = 0.0;
   profile.latency_penalty_points = 0.0;
   profile.allowed_deviation_points = 2.0;
}

void FXAI_ClearExecutionReplayFrame(FXAIExecutionReplayFrame &frame)
{
   frame.slippage_mult = 1.0;
   frame.fill_mult = 1.0;
   frame.latency_add_points = 0.0;
   frame.reject_prob = 0.0;
   frame.partial_fill_prob = 0.0;
   frame.drift_penalty_points = 0.0;
   frame.event_flags = 0;
}

void FXAI_ClearExecutionTraceStats(FXAIExecutionTraceStats &trace)
{
   trace.spread_mean_ratio = 1.0;
   trace.spread_peak_ratio = 1.0;
   trace.range_mean_ratio = 1.0;
   trace.body_efficiency = 0.5;
   trace.gap_ratio = 0.0;
   trace.reversal_ratio = 0.0;
   trace.session_transition_exposure = 0.0;
   trace.rollover_exposure = 0.0;
}

void FXAI_BuildExecutionTraceStats(const int i,
                                   const int horizon_minutes,
                                   const double point_value,
                                   const datetime &time_arr[],
                                   const double &open_arr[],
                                   const double &high_arr[],
                                   const double &low_arr[],
                                   const double &close_arr[],
                                   const int &spread_arr[],
                                   FXAIExecutionTraceStats &trace)
{
   FXAI_ClearExecutionTraceStats(trace);
   int n = ArraySize(close_arr);
   if(i < 0 || i >= n || point_value <= 0.0)
      return;

   int steps = horizon_minutes;
   if(steps < 1)
      steps = 1;
   if(steps > 1440)
      steps = 1440;
   if(steps > FXAI_EXEC_TRACE_BARS)
      steps = FXAI_EXEC_TRACE_BARS;
   if(i < steps)
      steps = i;
   if(steps < 1)
      steps = 1;

   double entry_spread = MathMax(FXAI_GetSpreadAtIndex(i, spread_arr, 1.0), 0.25);
   double entry_range = MathMax((high_arr[i] - low_arr[i]) / point_value, 0.25);
   double spread_sum = 0.0;
   double spread_peak = entry_spread;
   double range_sum = 0.0;
   double body_sum = 0.0;
   double gap_sum = 0.0;
   int reversal_count = 0;
   double prev_dir = 0.0;
   double session_sum = 0.0;
   double rollover_sum = 0.0;

   for(int step=0; step<=steps; step++)
   {
      int idx = i - step;
      if(idx < 0 || idx >= n)
         break;

      double spread = MathMax(FXAI_GetSpreadAtIndex(idx, spread_arr, entry_spread), 0.0);
      if(spread > spread_peak)
         spread_peak = spread;
      spread_sum += spread;

      double range_points = MathMax((high_arr[idx] - low_arr[idx]) / point_value, 0.0);
      range_sum += range_points;
      double body_eff = MathAbs(close_arr[idx] - open_arr[idx]) / MathMax(high_arr[idx] - low_arr[idx], point_value);
      body_sum += FXAI_Clamp(body_eff, 0.0, 1.0);

      if(idx + 1 < n)
      {
         double gap_points = MathAbs(open_arr[idx] - close_arr[idx + 1]) / point_value;
         gap_sum += gap_points;
      }

      double bar_dir = FXAI_Sign(close_arr[idx] - open_arr[idx]);
      if(step > 0 && MathAbs(bar_dir) > 1e-9 && MathAbs(prev_dir) > 1e-9 && bar_dir != prev_dir)
         reversal_count++;
      if(MathAbs(bar_dir) > 1e-9)
         prev_dir = bar_dir;

      datetime t = (idx < ArraySize(time_arr) ? time_arr[idx] : 0);
      MqlDateTime dt;
      TimeToStruct(t, dt);
      double hour_value = (double)dt.hour + (double)dt.min / 60.0;
      double asia_to_eu = FXAI_Clamp(1.0 - MathAbs(hour_value - 7.0) / 2.5, 0.0, 1.0);
      double eu_to_us = FXAI_Clamp(1.0 - MathAbs(hour_value - 13.0) / 2.5, 0.0, 1.0);
      double us_to_roll = FXAI_Clamp(1.0 - MathAbs(hour_value - 21.0) / 2.5, 0.0, 1.0);
      session_sum += FXAI_Clamp(0.60 * asia_to_eu + 0.80 * eu_to_us - 0.70 * us_to_roll, -1.0, 1.0);
      double roll_d = MathAbs(hour_value - 23.0);
      if(roll_d > 12.0)
         roll_d = 24.0 - roll_d;
      rollover_sum += FXAI_Clamp(1.0 - roll_d / 3.0, 0.0, 1.0);
   }

   double used = (double)(steps + 1);
   trace.spread_mean_ratio = FXAI_Clamp((spread_sum / used) / entry_spread, 0.5, 6.0);
   trace.spread_peak_ratio = FXAI_Clamp(spread_peak / entry_spread, 1.0, 10.0);
   trace.range_mean_ratio = FXAI_Clamp((range_sum / used) / entry_range, 0.25, 8.0);
   trace.body_efficiency = FXAI_Clamp(body_sum / used, 0.0, 1.0);
   trace.gap_ratio = FXAI_Clamp((gap_sum / used) / entry_spread, 0.0, 8.0);
   trace.reversal_ratio = FXAI_Clamp((double)reversal_count / MathMax((double)steps, 1.0), 0.0, 1.0);
   trace.session_transition_exposure = FXAI_Clamp(0.5 + 0.5 * (session_sum / used), 0.0, 1.0);
   trace.rollover_exposure = FXAI_Clamp(rollover_sum / used, 0.0, 1.0);
}

void FXAI_BuildExecutionReplayFrame(const FXAIExecutionProfile &profile,
                                    const datetime sample_time,
                                    const int horizon_minutes,
                                    const double spread_stress,
                                    const int path_flags,
                                    const int scenario_id,
                                    const FXAIExecutionTraceStats &trace,
                                    FXAIExecutionReplayFrame &frame)
{
   FXAI_ClearExecutionReplayFrame(frame);

   MqlDateTime dt;
   TimeToStruct((sample_time > 0 ? sample_time : TimeCurrent()), dt);
   double stress = FXAI_Clamp(spread_stress, 0.0, 4.0);
   double horizon_scale = MathSqrt((double)MathMax(horizon_minutes, 1));
   double mm_norm = ((double)dt.min + (double)dt.sec / 60.0) / 60.0;
   double session_edge = 1.0 - MathMin(MathAbs((double)dt.hour - 8.0), MathAbs((double)dt.hour - 16.0)) / 8.0;
   session_edge = FXAI_Clamp(session_edge, 0.0, 1.0);
   double rollover_edge = 1.0 - MathMin(MathAbs((double)dt.hour - 23.0), 6.0) / 6.0;
   rollover_edge = FXAI_Clamp(rollover_edge, 0.0, 1.0);
   double pulse = 0.5 + 0.5 * MathSin(6.283185307179586 * mm_norm);
   double trace_spread = FXAI_Clamp(0.55 * (trace.spread_mean_ratio - 1.0) +
                                    0.45 * (trace.spread_peak_ratio - 1.0),
                                    0.0,
                                    6.0);
   double trace_range = FXAI_Clamp(trace.range_mean_ratio - 1.0, -0.5, 4.0);
   double trace_gap = FXAI_Clamp(trace.gap_ratio, 0.0, 6.0);
   double trace_reversal = FXAI_Clamp(trace.reversal_ratio, 0.0, 1.0);
   double trace_session = FXAI_Clamp(0.55 * session_edge + 0.45 * trace.session_transition_exposure, 0.0, 1.0);
   double trace_roll = FXAI_Clamp(0.55 * rollover_edge + 0.45 * trace.rollover_exposure, 0.0, 1.0);
   double body_penalty = FXAI_Clamp(1.0 - trace.body_efficiency, 0.0, 1.0);

   frame.slippage_mult = 1.0 +
                         0.04 * stress +
                         0.05 * trace_spread +
                         0.03 * MathMax(trace_range, 0.0) +
                         0.04 * trace_gap +
                         0.04 * trace_session +
                         0.03 * trace_roll +
                         0.015 * horizon_scale +
                         0.02 * pulse;
   frame.fill_mult = 1.0 +
                     0.03 * stress +
                     0.05 * trace_reversal +
                     0.03 * body_penalty +
                     0.03 * trace_session +
                     0.02 * trace_roll;
   frame.latency_add_points = MathMax(profile.latency_penalty_points, 0.0) *
                              (0.32 + 0.22 * trace_spread + 0.18 * trace_gap + 0.16 * trace_session + 0.12 * trace_roll + 0.08 * stress);
   frame.reject_prob = FXAI_Clamp(0.002 +
                                  0.008 * stress +
                                  0.012 * trace_spread +
                                  0.010 * trace_gap +
                                  0.009 * trace_session +
                                  0.006 * trace_roll,
                                  0.0,
                                  0.35);
   frame.partial_fill_prob = FXAI_Clamp(0.01 +
                                        0.030 * stress +
                                        0.040 * trace_spread +
                                        0.030 * trace_reversal +
                                        0.020 * body_penalty +
                                        0.020 * trace_session,
                                        0.0,
                                        0.95);
   frame.drift_penalty_points = 0.05 * MathMax(profile.cost_buffer_points, 0.0) *
                                (trace_session + trace_roll + 0.35 * trace_spread);

   if((path_flags & FXAI_PATHFLAG_DUAL_HIT) != 0)
      frame.event_flags |= FXAI_PATHFLAG_DUAL_HIT;
   if((path_flags & FXAI_PATHFLAG_SPREAD_STRESS) != 0 || trace.spread_peak_ratio > 1.35)
      frame.event_flags |= FXAI_PATHFLAG_SPREAD_STRESS;

   if(scenario_id == 11) // market_session_edges
   {
      frame.slippage_mult += 0.10;
      frame.fill_mult += 0.08;
      frame.reject_prob = FXAI_Clamp(frame.reject_prob + 0.05, 0.0, 0.45);
      frame.partial_fill_prob = FXAI_Clamp(frame.partial_fill_prob + 0.10, 0.0, 0.98);
      frame.event_flags |= FXAI_PATHFLAG_SLOW_HIT;
   }
   else if(scenario_id == 12) // market_spread_shock
   {
      frame.slippage_mult += 0.12 + 0.03 * trace_spread;
      frame.fill_mult += 0.10 + 0.02 * trace_spread;
      frame.latency_add_points += 0.20 + 0.10 * stress + 0.06 * trace_gap;
      frame.reject_prob = FXAI_Clamp(frame.reject_prob + 0.08, 0.0, 0.50);
      frame.partial_fill_prob = FXAI_Clamp(frame.partial_fill_prob + 0.14, 0.0, 0.99);
      frame.event_flags |= FXAI_PATHFLAG_SPREAD_STRESS;
   }
   else if(scenario_id == 13) // market_walkforward
   {
      frame.slippage_mult += 0.04 + 0.02 * trace_reversal;
      frame.fill_mult += 0.03 + 0.02 * body_penalty;
      frame.reject_prob = FXAI_Clamp(frame.reject_prob + 0.03, 0.0, 0.40);
   }
}

void FXAI_BuildExecutionReplayFrame(const FXAIExecutionProfile &profile,
                                    const datetime sample_time,
                                    const int horizon_minutes,
                                    const double spread_stress,
                                    const int path_flags,
                                    const int scenario_id,
                                    FXAIExecutionReplayFrame &frame)
{
   FXAIExecutionTraceStats trace;
   FXAI_ClearExecutionTraceStats(trace);
   FXAI_BuildExecutionReplayFrame(profile,
                                  sample_time,
                                  horizon_minutes,
                                  spread_stress,
                                  path_flags,
                                  scenario_id,
                                  trace,
                                  frame);
}

void FXAI_SetExecutionProfilePreset(const int profile_id,
                                    FXAIExecutionProfile &profile)
{
   FXAI_ClearExecutionProfile(profile);
   profile.profile_id = profile_id;

   if(profile_id == (int)FXAI_EXEC_TIGHT_FX)
   {
      profile.cost_buffer_points = 1.5;
      profile.slippage_points = 0.10;
      profile.fill_penalty_points = 0.10;
      profile.allowed_deviation_points = 2.0;
      return;
   }
   if(profile_id == (int)FXAI_EXEC_PRIME_ECN)
   {
      profile.commission_per_lot_side = 3.5;
      profile.cost_buffer_points = 1.5;
      profile.slippage_points = 0.20;
      profile.fill_penalty_points = 0.15;
      profile.allowed_deviation_points = 2.5;
      return;
   }
   if(profile_id == (int)FXAI_EXEC_RETAIL_FX)
   {
      profile.cost_buffer_points = 2.5;
      profile.slippage_points = 0.40;
      profile.fill_penalty_points = 0.25;
      profile.slippage_cost_weight = 0.05;
      profile.allowed_deviation_points = 4.0;
      return;
   }
   if(profile_id == (int)FXAI_EXEC_STRESS)
   {
      profile.commission_per_lot_side = 5.0;
      profile.cost_buffer_points = 3.5;
      profile.slippage_points = 1.0;
      profile.fill_penalty_points = 0.50;
      profile.slippage_cost_weight = 0.06;
      profile.slippage_stress_weight = 0.25;
      profile.slippage_horizon_weight = 0.03;
      profile.dual_hit_penalty = 0.20;
      profile.slow_hit_penalty = 0.16;
      profile.spread_shock_penalty = 0.55;
      profile.partial_fill_penalty = 0.25;
      profile.latency_penalty_points = 0.20;
      profile.allowed_deviation_points = 8.0;
      return;
   }
}

double FXAI_ExecutionEntryCostPoints(const double spread_points,
                                     const double commission_points,
                                     const double base_cost_buffer_points,
                                     const FXAIExecutionProfile &profile)
{
   double cost = MathMax(spread_points, 0.0) +
                 MathMax(commission_points, 0.0) +
                 MathMax(base_cost_buffer_points, 0.0) +
                 MathMax(profile.cost_buffer_points, 0.0) +
                 MathMax(profile.slippage_points, 0.0) +
                 MathMax(profile.fill_penalty_points, 0.0) +
                 MathMax(profile.latency_penalty_points, 0.0);
   if(cost < 0.0) cost = 0.0;
   return cost;
}

double FXAI_ExecutionSlippagePoints(const FXAIExecutionProfile &profile,
                                    const double roundtrip_cost_points,
                                    const int horizon_minutes,
                                    const double spread_stress,
                                    const int path_flags)
{
   double stress = FXAI_Clamp(spread_stress, 0.0, 4.0);
   double slippage_points = MathMax(profile.slippage_points, 0.0) +
                            profile.slippage_cost_weight * MathMax(roundtrip_cost_points, 0.0) +
                            profile.slippage_stress_weight * stress +
                            profile.slippage_horizon_weight * MathSqrt((double)MathMax(horizon_minutes, 1)) +
                            MathMax(profile.latency_penalty_points, 0.0);
   if((path_flags & 1) != 0)
      slippage_points += MathMax(profile.dual_hit_penalty, 0.0) +
                         0.12 * MathMax(roundtrip_cost_points, 0.0) +
                         0.12 * stress;
   if((path_flags & 8) != 0)
      slippage_points += MathMax(profile.slow_hit_penalty, 0.0);
   if((path_flags & 4) != 0)
      slippage_points += MathMax(profile.spread_shock_penalty, 0.0);
   if(slippage_points > 12.0) slippage_points = 12.0;
   return slippage_points;
}

double FXAI_ExecutionSlippagePointsReplay(const FXAIExecutionProfile &profile,
                                          const FXAIExecutionReplayFrame &frame,
                                          const double roundtrip_cost_points,
                                          const int horizon_minutes,
                                          const double spread_stress,
                                          const int path_flags)
{
   double slippage_points = FXAI_ExecutionSlippagePoints(profile,
                                                         roundtrip_cost_points,
                                                         horizon_minutes,
                                                         spread_stress,
                                                         path_flags | frame.event_flags);
   slippage_points = slippage_points * FXAI_Clamp(frame.slippage_mult, 0.50, 3.00) +
                     MathMax(frame.latency_add_points, 0.0) +
                     MathMax(frame.drift_penalty_points, 0.0);
   if(slippage_points > 18.0) slippage_points = 18.0;
   return slippage_points;
}

double FXAI_ExecutionFillPenaltyPoints(const FXAIExecutionProfile &profile,
                                       const double roundtrip_cost_points,
                                       const double spread_stress,
                                       const int path_flags)
{
   double stress = FXAI_Clamp(spread_stress, 0.0, 4.0);
   double fill_penalty = MathMax(profile.fill_penalty_points, 0.0) +
                         MathMax(profile.partial_fill_penalty, 0.0) * (0.20 + 0.20 * stress);
   if((path_flags & 4) != 0)
      fill_penalty += 0.10 * MathMax(roundtrip_cost_points, 0.0) + 0.12 * stress;
   if((path_flags & 1) != 0)
      fill_penalty += 0.08 * MathMax(roundtrip_cost_points, 0.0);
   if(fill_penalty > 10.0) fill_penalty = 10.0;
   return fill_penalty;
}

double FXAI_ExecutionFillPenaltyPointsReplay(const FXAIExecutionProfile &profile,
                                             const FXAIExecutionReplayFrame &frame,
                                             const double roundtrip_cost_points,
                                             const double spread_stress,
                                             const int path_flags)
{
   double fill_penalty = FXAI_ExecutionFillPenaltyPoints(profile,
                                                         roundtrip_cost_points,
                                                         spread_stress,
                                                         path_flags | frame.event_flags);
   fill_penalty *= FXAI_Clamp(frame.fill_mult, 0.50, 3.00);
   fill_penalty += MathMax(profile.partial_fill_penalty, 0.0) * FXAI_Clamp(frame.partial_fill_prob, 0.0, 1.0);
   fill_penalty += 0.50 * MathMax(frame.drift_penalty_points, 0.0);
   if(fill_penalty > 15.0) fill_penalty = 15.0;
   return fill_penalty;
}

double FXAI_ExecutionAllowedDeviationPoints(const FXAIExecutionProfile &profile,
                                            const double path_risk,
                                            const double fill_risk)
{
   double dev = MathMax(profile.allowed_deviation_points, 0.0) +
                2.5 * FXAI_Clamp(path_risk, 0.0, 1.0) +
                3.0 * FXAI_Clamp(fill_risk, 0.0, 1.0);
   if(dev > 25.0) dev = 25.0;
   return dev;
}

int FXAI_CoreClampHorizon(const int horizon_minutes)
{
   int h = horizon_minutes;
   if(h < 1) h = 1;
   if(h > 1440) h = 1440;
   return h;
}

double FXAI_SymbolModelScale(const string symbol)
{
   string s = symbol;
   StringToUpper(s);
   if(StringFind(s, "XAU") >= 0 || StringFind(s, "GOLD") >= 0 ||
      StringFind(s, "XAG") >= 0 || StringFind(s, "SILVER") >= 0)
      return 1.18;
   if(StringFind(s, "US30") >= 0 || StringFind(s, "NAS") >= 0 ||
      StringFind(s, "SPX") >= 0 || StringFind(s, "DAX") >= 0 ||
      StringFind(s, "GER40") >= 0 || StringFind(s, "JP225") >= 0)
      return 1.15;
   if(StringFind(s, "OIL") >= 0 || StringFind(s, "WTI") >= 0 ||
      StringFind(s, "BRENT") >= 0 || StringFind(s, "NGAS") >= 0)
      return 1.12;
   if(StringFind(s, "JPY") >= 0 || StringFind(s, "GBP") >= 0)
      return 1.06;
   return 1.00;
}

double FXAI_HorizonModelScale(const int horizon_minutes)
{
   int h = FXAI_CoreClampHorizon(horizon_minutes);
   if(h <= 5) return 0.92;
   if(h <= 15) return 0.98;
   if(h <= 60) return 1.00;
   if(h <= 240) return 1.10;
   return 1.18;
}

double FXAI_ModelCapacityScale(const string symbol,
                               const int horizon_minutes)
{
   return FXAI_Clamp(FXAI_SymbolModelScale(symbol) * FXAI_HorizonModelScale(horizon_minutes), 0.85, 1.35);
}

int FXAI_ContextSequenceSpan(const int max_cap,
                             const int horizon_minutes,
                             const string symbol,
                             const int base_min = 8)
{
   int cap = MathMax(base_min, max_cap);
   double scale = FXAI_ModelCapacityScale(symbol, horizon_minutes);
   int span = (int)MathRound((double)cap * FXAI_Clamp(0.55 + 0.35 * scale, 0.45, 1.10));
   if(span < base_min) span = base_min;
   if(span > max_cap) span = max_cap;
   return span;
}

int FXAI_ContextBatchSpan(const int max_cap,
                          const int horizon_minutes,
                          const string symbol,
                          const int base_min = 4)
{
   int cap = MathMax(base_min, max_cap);
   double scale = FXAI_ModelCapacityScale(symbol, horizon_minutes);
   int span = (int)MathRound((double)cap * FXAI_Clamp(0.60 + 0.30 * scale, 0.45, 1.00));
   if(span < base_min) span = base_min;
   if(span > max_cap) span = max_cap;
   return span;
}

int FXAI_ContextTreeBudget(const int max_cap,
                           const int horizon_minutes,
                           const string symbol,
                           const int base_min)
{
   int cap = MathMax(base_min, max_cap);
   double scale = FXAI_ModelCapacityScale(symbol, horizon_minutes);
   int budget = (int)MathRound((double)cap * FXAI_Clamp(0.55 + 0.40 * scale, 0.50, 1.15));
   if(budget < base_min) budget = base_min;
   if(budget > max_cap) budget = max_cap;
   return budget;
}

void FXAI_UpdateMoveEMA(double &ema_abs_move,
                       bool &ready,
                       const double move_points,
                       const double alpha)
{
   double a = FXAI_Clamp(alpha, 0.001, 0.500);
   double v = MathAbs(move_points);
   if(!MathIsValidNumber(v)) return;

   if(!ready)
   {
      ema_abs_move = v;
      ready = true;
      return;
   }

   ema_abs_move = (1.0 - a) * ema_abs_move + a * v;
}

int FXAI_ThreeWayBranch(const double x, const double split)
{
   if(x < split - 0.50) return 0;
   if(x > split + 0.50) return 2;
   return 1;
}

void FXAI_BuildInputVector(const double &features[], double &x[])
{
   x[0] = 1.0;
   for(int i=0; i<FXAI_AI_FEATURES; i++)
      x[i + 1] = features[i];
}

void FXAI_ClearInputWindow(double &x_window[][FXAI_AI_WEIGHTS], int &window_size)
{
   window_size = 0;
   for(int b=0; b<FXAI_MAX_SEQUENCE_BARS; b++)
      for(int k=0; k<FXAI_AI_WEIGHTS; k++)
         x_window[b][k] = 0.0;
}

void FXAI_CopyInputVector(const double &src[], double &dst[])
{
   int n = MathMin(ArraySize(src), ArraySize(dst));
   for(int i=0; i<n; i++)
      dst[i] = src[i];
}

double FXAI_GetInputFeature(const double &x[], const int feature_idx)
{
   int idx = feature_idx + 1;
   if(idx >= 0 && idx < ArraySize(x)) return x[idx];
   return 0.0;
}

void FXAI_SetInputFeature(double &x[], const int feature_idx, const double value)
{
   int idx = feature_idx + 1;
   if(idx >= 0 && idx < ArraySize(x))
      x[idx] = value;
}

double FXAI_MeanInputFeatureRange(const double &x[],
                                  const int first_feature_idx,
                                  const int last_feature_idx)
{
   if(last_feature_idx < first_feature_idx) return 0.0;
   double sum = 0.0;
   int used = 0;
   for(int f=first_feature_idx; f<=last_feature_idx; f++)
   {
      sum += FXAI_GetInputFeature(x, f);
      used++;
   }
   if(used <= 0) return 0.0;
   return sum / (double)used;
}

double FXAI_QuantizeSignedFeature(const double value, const double step)
{
   double s = (step > 1e-9 ? step : 0.25);
   return MathRound(value / s) * s;
}

ulong FXAI_FeatureGroupBit(const int group_id)
{
   if(group_id < 0 || group_id > (int)FXAI_FEAT_GROUP_FILTERS)
      return 0;
   return ((ulong)1 << (ulong)group_id);
}

int FXAI_GetFeatureGroupForIndex(const int feature_idx)
{
   if(feature_idx < 0 || feature_idx >= FXAI_AI_FEATURES)
      return (int)FXAI_FEAT_GROUP_PRICE;

   if(feature_idx <= 5) return (int)FXAI_FEAT_GROUP_PRICE;
   if(feature_idx == 6) return (int)FXAI_FEAT_GROUP_COST;
   if(feature_idx <= 9) return (int)FXAI_FEAT_GROUP_MULTI_TIMEFRAME;
   if(feature_idx <= 12) return (int)FXAI_FEAT_GROUP_CONTEXT;
   if(feature_idx <= 14) return (int)FXAI_FEAT_GROUP_MULTI_TIMEFRAME;
   if(feature_idx <= 17) return (int)FXAI_FEAT_GROUP_TIME;
   if(feature_idx <= 21) return (int)FXAI_FEAT_GROUP_PRICE;
   if(feature_idx <= 37) return (int)FXAI_FEAT_GROUP_MULTI_TIMEFRAME;
   if(feature_idx <= 45) return (int)FXAI_FEAT_GROUP_VOLATILITY;
   if(feature_idx <= 49) return (int)FXAI_FEAT_GROUP_FILTERS;
   if(feature_idx <= 65) return (int)FXAI_FEAT_GROUP_CONTEXT;
   if(feature_idx <= 71) return (int)FXAI_FEAT_GROUP_MICROSTRUCTURE;
   if(feature_idx <= 73) return (int)FXAI_FEAT_GROUP_TIME;
   if(feature_idx <= 78) return (int)FXAI_FEAT_GROUP_COST;
   if(feature_idx == 79) return (int)FXAI_FEAT_GROUP_FILTERS;
   if(feature_idx <= 83) return (int)FXAI_FEAT_GROUP_COST;
   if(feature_idx < FXAI_CONTEXT_MTF_FEATURE_OFFSET) return (int)FXAI_FEAT_GROUP_MULTI_TIMEFRAME;
   return (int)FXAI_FEAT_GROUP_CONTEXT;
}

ulong FXAI_DefaultFeatureGroupsForFamily(const int family)
{
   ulong mask = 0;
   switch(family)
   {
      case FXAI_FAMILY_LINEAR:
         mask |= FXAI_FeatureGroupBit((int)FXAI_FEAT_GROUP_PRICE);
         mask |= FXAI_FeatureGroupBit((int)FXAI_FEAT_GROUP_MULTI_TIMEFRAME);
         mask |= FXAI_FeatureGroupBit((int)FXAI_FEAT_GROUP_VOLATILITY);
         mask |= FXAI_FeatureGroupBit((int)FXAI_FEAT_GROUP_TIME);
         mask |= FXAI_FeatureGroupBit((int)FXAI_FEAT_GROUP_CONTEXT);
         mask |= FXAI_FeatureGroupBit((int)FXAI_FEAT_GROUP_COST);
         mask |= FXAI_FeatureGroupBit((int)FXAI_FEAT_GROUP_MICROSTRUCTURE);
         break;
      case FXAI_FAMILY_TREE:
         mask |= FXAI_FeatureGroupBit((int)FXAI_FEAT_GROUP_PRICE);
         mask |= FXAI_FeatureGroupBit((int)FXAI_FEAT_GROUP_MULTI_TIMEFRAME);
         mask |= FXAI_FeatureGroupBit((int)FXAI_FEAT_GROUP_VOLATILITY);
         mask |= FXAI_FeatureGroupBit((int)FXAI_FEAT_GROUP_TIME);
         mask |= FXAI_FeatureGroupBit((int)FXAI_FEAT_GROUP_CONTEXT);
         mask |= FXAI_FeatureGroupBit((int)FXAI_FEAT_GROUP_COST);
         mask |= FXAI_FeatureGroupBit((int)FXAI_FEAT_GROUP_MICROSTRUCTURE);
         mask |= FXAI_FeatureGroupBit((int)FXAI_FEAT_GROUP_FILTERS);
         break;
      case FXAI_FAMILY_RULE_BASED:
         mask |= FXAI_FeatureGroupBit((int)FXAI_FEAT_GROUP_PRICE);
         mask |= FXAI_FeatureGroupBit((int)FXAI_FEAT_GROUP_TIME);
         mask |= FXAI_FeatureGroupBit((int)FXAI_FEAT_GROUP_COST);
         break;
      case FXAI_FAMILY_DISTRIBUTIONAL:
         mask |= FXAI_FeatureGroupBit((int)FXAI_FEAT_GROUP_PRICE);
         mask |= FXAI_FeatureGroupBit((int)FXAI_FEAT_GROUP_MULTI_TIMEFRAME);
         mask |= FXAI_FeatureGroupBit((int)FXAI_FEAT_GROUP_VOLATILITY);
         mask |= FXAI_FeatureGroupBit((int)FXAI_FEAT_GROUP_CONTEXT);
         mask |= FXAI_FeatureGroupBit((int)FXAI_FEAT_GROUP_COST);
         mask |= FXAI_FeatureGroupBit((int)FXAI_FEAT_GROUP_MICROSTRUCTURE);
         break;
      default:
         mask |= FXAI_FeatureGroupBit((int)FXAI_FEAT_GROUP_PRICE);
         mask |= FXAI_FeatureGroupBit((int)FXAI_FEAT_GROUP_MULTI_TIMEFRAME);
         mask |= FXAI_FeatureGroupBit((int)FXAI_FEAT_GROUP_VOLATILITY);
         mask |= FXAI_FeatureGroupBit((int)FXAI_FEAT_GROUP_TIME);
         mask |= FXAI_FeatureGroupBit((int)FXAI_FEAT_GROUP_CONTEXT);
         mask |= FXAI_FeatureGroupBit((int)FXAI_FEAT_GROUP_COST);
         mask |= FXAI_FeatureGroupBit((int)FXAI_FEAT_GROUP_MICROSTRUCTURE);
         mask |= FXAI_FeatureGroupBit((int)FXAI_FEAT_GROUP_FILTERS);
         break;
   }
   return mask;
}

int FXAI_DefaultFeatureSchemaForFamily(const int family)
{
   switch(family)
   {
      case FXAI_FAMILY_LINEAR:
      case FXAI_FAMILY_DISTRIBUTIONAL:
         return (int)FXAI_SCHEMA_SPARSE_STAT;
      case FXAI_FAMILY_RULE_BASED:
         return (int)FXAI_SCHEMA_RULE;
      case FXAI_FAMILY_TREE:
         return (int)FXAI_SCHEMA_TREE;
      case FXAI_FAMILY_RECURRENT:
      case FXAI_FAMILY_CONVOLUTIONAL:
      case FXAI_FAMILY_TRANSFORMER:
      case FXAI_FAMILY_STATE_SPACE:
         return (int)FXAI_SCHEMA_SEQUENCE;
      case FXAI_FAMILY_RETRIEVAL:
      case FXAI_FAMILY_MIXTURE:
      case FXAI_FAMILY_WORLD_MODEL:
         return (int)FXAI_SCHEMA_CONTEXTUAL;
      default:
         return (int)FXAI_SCHEMA_FULL;
   }
}

bool FXAI_IsFeatureEnabledForSchema(const int feature_idx,
                                    const int schema_id,
                                    const ulong groups_mask)
{
   if(feature_idx >= FXAI_MAIN_MTF_FEATURE_OFFSET && feature_idx < FXAI_AI_FEATURES)
      return true;

   int group_id = FXAI_GetFeatureGroupForIndex(feature_idx);
   ulong bit = FXAI_FeatureGroupBit(group_id);
   if(bit == 0 || (groups_mask & bit) == 0)
      return false;

   switch(schema_id)
   {
      case FXAI_SCHEMA_SPARSE_STAT:
         if(feature_idx >= 46 && feature_idx <= 49) return false;
         if(feature_idx >= 50 && feature_idx <= 71) return false;
         return true;
      case FXAI_SCHEMA_RULE:
         return (group_id == (int)FXAI_FEAT_GROUP_PRICE ||
                 group_id == (int)FXAI_FEAT_GROUP_TIME ||
                 group_id == (int)FXAI_FEAT_GROUP_COST);
      case FXAI_SCHEMA_CONTEXTUAL:
         if(group_id == (int)FXAI_FEAT_GROUP_TIME && feature_idx >= 15 && feature_idx <= 17)
            return false;
         return true;
      case FXAI_SCHEMA_TREE:
      case FXAI_SCHEMA_SEQUENCE:
      case FXAI_SCHEMA_FULL:
      default:
         return true;
   }
}

void FXAI_ApplyFeatureSchemaToInputEx(const int schema_id,
                                      const ulong groups_mask,
                                      const int sequence_bars,
                                      const double &x_window[][FXAI_AI_WEIGHTS],
                                      const int window_size,
                                      double &x[])
{
   if(ArraySize(x) < FXAI_AI_WEIGHTS)
      return;

   double raw_x[FXAI_AI_WEIGHTS];
   for(int k=0; k<FXAI_AI_WEIGHTS; k++)
      raw_x[k] = x[k];

   x[0] = 1.0;
   for(int f=0; f<FXAI_AI_FEATURES; f++)
   {
      if(!FXAI_IsFeatureEnabledForSchema(f, schema_id, groups_mask))
         x[f + 1] = 0.0;
   }

   int seq_n = sequence_bars;
   if(seq_n < 1) seq_n = 1;
   if(seq_n > FXAI_MAX_SEQUENCE_BARS) seq_n = FXAI_MAX_SEQUENCE_BARS;
   if(window_size > 0 && window_size < seq_n) seq_n = window_size;

   double seq_mean[FXAI_AI_WEIGHTS];
   double seq_delta[FXAI_AI_WEIGHTS];
   double seq_std[FXAI_AI_WEIGHTS];
   double seq_short_mean[FXAI_AI_WEIGHTS];
   double seq_mid_mean[FXAI_AI_WEIGHTS];
   double seq_long_mean[FXAI_AI_WEIGHTS];
   double seq_short_delta[FXAI_AI_WEIGHTS];
   double seq_mid_delta[FXAI_AI_WEIGHTS];
   for(int k=0; k<FXAI_AI_WEIGHTS; k++)
   {
      seq_mean[k] = raw_x[k];
      seq_delta[k] = 0.0;
      seq_std[k] = 0.0;
      seq_short_mean[k] = raw_x[k];
      seq_mid_mean[k] = raw_x[k];
      seq_long_mean[k] = raw_x[k];
      seq_short_delta[k] = 0.0;
      seq_mid_delta[k] = 0.0;
   }

   if(seq_n > 1)
   {
      int used = 0;
      for(int b=0; b<seq_n; b++)
      {
         if(b >= window_size) break;
         used++;
         for(int k=0; k<FXAI_AI_WEIGHTS; k++)
            seq_mean[k] += x_window[b][k];
      }
      if(used > 0)
      {
         double denom = (double)(used + 1);
         for(int k=0; k<FXAI_AI_WEIGHTS; k++)
            seq_mean[k] /= denom;
      }

      for(int b=0; b<seq_n && b<window_size; b++)
      {
         for(int k=0; k<FXAI_AI_WEIGHTS; k++)
         {
            double d = x_window[b][k] - seq_mean[k];
            seq_std[k] += d * d;
         }
      }
      for(int k=0; k<FXAI_AI_WEIGHTS; k++)
         seq_std[k] = MathSqrt(seq_std[k] / (double)MathMax(seq_n - 1, 1));

      int short_n = MathMax(seq_n / 4, 1);
      int mid_n = MathMax(seq_n / 2, 1);
      int long_n = seq_n;
      int short_used = 0;
      int mid_used = 0;
      int long_used = 0;
      for(int b=0; b<seq_n && b<window_size; b++)
      {
         bool use_short = (b < short_n);
         bool use_mid = (b < mid_n);
         bool use_long = (b < long_n);
         if(use_short) short_used++;
         if(use_mid) mid_used++;
         if(use_long) long_used++;
         for(int k=0; k<FXAI_AI_WEIGHTS; k++)
         {
            if(use_short) seq_short_mean[k] += x_window[b][k];
            if(use_mid) seq_mid_mean[k] += x_window[b][k];
            if(use_long) seq_long_mean[k] += x_window[b][k];
         }
      }
      for(int k=0; k<FXAI_AI_WEIGHTS; k++)
      {
         seq_short_mean[k] /= (double)MathMax(short_used + 1, 1);
         seq_mid_mean[k] /= (double)MathMax(mid_used + 1, 1);
         seq_long_mean[k] /= (double)MathMax(long_used + 1, 1);
      }

      int last_idx = seq_n - 2;
      if(last_idx >= 0 && last_idx < window_size)
      {
         for(int k=0; k<FXAI_AI_WEIGHTS; k++)
            seq_delta[k] = raw_x[k] - x_window[last_idx][k];
      }

      int short_last = short_n - 1;
      if(short_last >= 0 && short_last < window_size)
      {
         for(int k=0; k<FXAI_AI_WEIGHTS; k++)
            seq_short_delta[k] = raw_x[k] - x_window[short_last][k];
      }
      int mid_last = mid_n - 1;
      if(mid_last >= 0 && mid_last < window_size)
      {
         for(int k=0; k<FXAI_AI_WEIGHTS; k++)
            seq_mid_delta[k] = raw_x[k] - x_window[mid_last][k];
      }
   }

   // Schema-specific projection stage. This is intentionally stronger than
   // simple masking so each plugin family sees a representation aligned to its
   // inductive bias while staying on the shared feature contract.
   switch(schema_id)
   {
      case FXAI_SCHEMA_SPARSE_STAT:
      {
         double mtf_ret = FXAI_MeanInputFeatureRange(x, 7, 9);
         double mtf_slope = FXAI_MeanInputFeatureRange(x, 13, 14);
         double sma_fast = 0.25 * (FXAI_GetInputFeature(x, 22) + FXAI_GetInputFeature(x, 24) +
                                   FXAI_GetInputFeature(x, 26) + FXAI_GetInputFeature(x, 28));
         double sma_slow = 0.25 * (FXAI_GetInputFeature(x, 23) + FXAI_GetInputFeature(x, 25) +
                                   FXAI_GetInputFeature(x, 27) + FXAI_GetInputFeature(x, 29));
         double ema_fast = 0.25 * (FXAI_GetInputFeature(x, 30) + FXAI_GetInputFeature(x, 32) +
                                   FXAI_GetInputFeature(x, 34) + FXAI_GetInputFeature(x, 36));
         double ema_slow = 0.25 * (FXAI_GetInputFeature(x, 31) + FXAI_GetInputFeature(x, 33) +
                                   FXAI_GetInputFeature(x, 35) + FXAI_GetInputFeature(x, 37));
         double vol_pack = FXAI_MeanInputFeatureRange(x, 41, 45);
         double filt_pack = FXAI_MeanInputFeatureRange(x, 46, 49);
         double ctx_ret = FXAI_GetInputFeature(x, 10);
         double ctx_rel = FXAI_GetInputFeature(x, 12);
         double ctx_corr = 0.0;
         bool ctx_enabled = ((groups_mask & FXAI_FeatureGroupBit((int)FXAI_FEAT_GROUP_CONTEXT)) != 0);
         if(ctx_enabled)
         {
            ctx_ret = (FXAI_GetInputFeature(x, 10) + FXAI_GetInputFeature(raw_x, 50) +
                       FXAI_GetInputFeature(raw_x, 54) + FXAI_GetInputFeature(raw_x, 58)) / 4.0;
            ctx_rel = (FXAI_GetInputFeature(x, 12) + FXAI_GetInputFeature(raw_x, 52) +
                       FXAI_GetInputFeature(raw_x, 56) + FXAI_GetInputFeature(raw_x, 60)) / 4.0;
            ctx_corr = (FXAI_GetInputFeature(raw_x, 53) + FXAI_GetInputFeature(raw_x, 57) +
                        FXAI_GetInputFeature(raw_x, 61)) / 3.0;
         }

         FXAI_SetInputFeature(x, 7, mtf_ret);
         FXAI_SetInputFeature(x, 8, mtf_slope);
         FXAI_SetInputFeature(x, 9, 0.5 * (mtf_ret + mtf_slope));
         FXAI_SetInputFeature(x, 22, sma_fast);
         FXAI_SetInputFeature(x, 23, sma_slow);
         FXAI_SetInputFeature(x, 24, ema_fast);
         FXAI_SetInputFeature(x, 25, ema_slow);
         FXAI_SetInputFeature(x, 26, sma_fast - sma_slow);
         FXAI_SetInputFeature(x, 27, ema_fast - ema_slow);
         FXAI_SetInputFeature(x, 28, vol_pack);
         FXAI_SetInputFeature(x, 29, filt_pack);
         FXAI_SetInputFeature(x, 30, ctx_ret);
         FXAI_SetInputFeature(x, 31, ctx_rel);
         FXAI_SetInputFeature(x, 32, ctx_corr);
         for(int f=33; f<=39; f++)
            FXAI_SetInputFeature(x, f, 0.0);
         break;
      }

      case FXAI_SCHEMA_SEQUENCE:
      {
         double ret_short = FXAI_GetInputFeature(x, 0);
         double ret_mid = FXAI_GetInputFeature(x, 1);
         double ret_long = FXAI_GetInputFeature(x, 2);
         double mtf_fast = 0.5 * (FXAI_GetInputFeature(x, 7) + FXAI_GetInputFeature(x, 13));
         double mtf_slow = 0.5 * (FXAI_GetInputFeature(x, 9) + FXAI_GetInputFeature(x, 14));
         double seq_ret_short = FXAI_GetInputFeature(seq_short_mean, 0);
         double seq_ret_mid = FXAI_GetInputFeature(seq_mid_mean, 1);
         double seq_ret_long = FXAI_GetInputFeature(seq_long_mean, 2);
         double seq_ctx = 0.50 * FXAI_GetInputFeature(seq_short_mean, 10) +
                          0.30 * FXAI_GetInputFeature(seq_mid_mean, 10) +
                          0.20 * FXAI_GetInputFeature(seq_long_mean, 10);
         double seq_vol = 0.50 * FXAI_GetInputFeature(seq_short_mean, 41) +
                          0.30 * FXAI_GetInputFeature(seq_mid_mean, 41) +
                          0.20 * FXAI_GetInputFeature(seq_long_mean, 41);
         double seq_ret_accel = FXAI_GetInputFeature(seq_short_mean, 0) - FXAI_GetInputFeature(seq_mid_mean, 0);
         double seq_ctx_delta = FXAI_GetInputFeature(seq_short_mean, 10) - FXAI_GetInputFeature(seq_long_mean, 10);

         FXAI_SetInputFeature(x, 13, ret_short - ret_mid);
         FXAI_SetInputFeature(x, 14, ret_mid - ret_long);
         FXAI_SetInputFeature(x, 22, FXAI_GetInputFeature(x, 22) - FXAI_GetInputFeature(x, 23));
         FXAI_SetInputFeature(x, 23, FXAI_GetInputFeature(x, 24) - FXAI_GetInputFeature(x, 25));
         FXAI_SetInputFeature(x, 24, FXAI_GetInputFeature(x, 26) - FXAI_GetInputFeature(x, 27));
         FXAI_SetInputFeature(x, 25, FXAI_GetInputFeature(x, 28) - FXAI_GetInputFeature(x, 29));
         FXAI_SetInputFeature(x, 26, FXAI_GetInputFeature(x, 30) - FXAI_GetInputFeature(x, 31));
         FXAI_SetInputFeature(x, 27, FXAI_GetInputFeature(x, 32) - FXAI_GetInputFeature(x, 33));
         FXAI_SetInputFeature(x, 28, FXAI_GetInputFeature(x, 34) - FXAI_GetInputFeature(x, 35));
         FXAI_SetInputFeature(x, 29, FXAI_GetInputFeature(x, 36) - FXAI_GetInputFeature(x, 37));
         FXAI_SetInputFeature(x, 30, mtf_fast);
         FXAI_SetInputFeature(x, 31, mtf_slow);
         FXAI_SetInputFeature(x, 32, mtf_fast - mtf_slow);
         FXAI_SetInputFeature(x, 33, seq_ret_short - seq_ret_mid);
         FXAI_SetInputFeature(x, 34, seq_ret_mid - seq_ret_long);
         FXAI_SetInputFeature(x, 35, FXAI_GetInputFeature(seq_short_delta, 0));
         FXAI_SetInputFeature(x, 36, FXAI_GetInputFeature(seq_mid_delta, 1));
         FXAI_SetInputFeature(x, 37, seq_ret_accel);
         FXAI_SetInputFeature(x, 38, seq_ctx);
         FXAI_SetInputFeature(x, 39, seq_vol);
         // Preserve explicit lag/context blocks for sequence-capable plugins.
         FXAI_SetInputFeature(x, 50, FXAI_GetInputFeature(seq_short_mean, 50));
         FXAI_SetInputFeature(x, 51, FXAI_GetInputFeature(seq_short_mean, 51));
         FXAI_SetInputFeature(x, 52, FXAI_GetInputFeature(seq_short_mean, 52));
         FXAI_SetInputFeature(x, 53, FXAI_GetInputFeature(seq_short_mean, 53));
         FXAI_SetInputFeature(x, 54, FXAI_GetInputFeature(seq_mid_mean, 50));
         FXAI_SetInputFeature(x, 55, FXAI_GetInputFeature(seq_mid_mean, 51));
         FXAI_SetInputFeature(x, 56, FXAI_GetInputFeature(seq_mid_mean, 52));
         FXAI_SetInputFeature(x, 57, FXAI_GetInputFeature(seq_mid_mean, 53));
         FXAI_SetInputFeature(x, 58, FXAI_GetInputFeature(seq_short_delta, 50));
         FXAI_SetInputFeature(x, 59, seq_ctx_delta);
         FXAI_SetInputFeature(x, 60, FXAI_GetInputFeature(seq_std, 50));
         FXAI_SetInputFeature(x, 61, FXAI_GetInputFeature(seq_std, 51));
         break;
      }

      case FXAI_SCHEMA_RULE:
      {
         for(int f=0; f<FXAI_AI_FEATURES; f++)
         {
            int group_id = FXAI_GetFeatureGroupForIndex(f);
            if(group_id == (int)FXAI_FEAT_GROUP_TIME)
               continue;

            double v = FXAI_GetInputFeature(x, f);
            double out_v = 0.0;
            if(v > 0.15) out_v = 1.0;
            else if(v < -0.15) out_v = -1.0;
            FXAI_SetInputFeature(x, f, out_v);
         }
         break;
      }

      case FXAI_SCHEMA_TREE:
      {
         for(int f=0; f<FXAI_AI_FEATURES; f++)
         {
            double v = FXAI_GetInputFeature(x, f);
            FXAI_SetInputFeature(x, f, FXAI_QuantizeSignedFeature(v, 0.25));
         }
         break;
      }

      case FXAI_SCHEMA_CONTEXTUAL:
      {
         double ctx_ret = (FXAI_GetInputFeature(x, 50) + FXAI_GetInputFeature(x, 54) +
                           FXAI_GetInputFeature(x, 58)) / 3.0;
         double ctx_lag = (FXAI_GetInputFeature(x, 51) + FXAI_GetInputFeature(x, 55) +
                           FXAI_GetInputFeature(x, 59)) / 3.0;
         double ctx_rel = (FXAI_GetInputFeature(x, 52) + FXAI_GetInputFeature(x, 56) +
                           FXAI_GetInputFeature(x, 60)) / 3.0;
         double ctx_corr = (FXAI_GetInputFeature(x, 53) + FXAI_GetInputFeature(x, 57) +
                            FXAI_GetInputFeature(x, 61)) / 3.0;
         double ctx_ret_fast = (FXAI_GetInputFeature(seq_short_mean, 50) +
                                FXAI_GetInputFeature(seq_short_mean, 54) +
                                FXAI_GetInputFeature(seq_short_mean, 58)) / 3.0;
         double ctx_ret_slow = (FXAI_GetInputFeature(seq_long_mean, 50) +
                                FXAI_GetInputFeature(seq_long_mean, 54) +
                                FXAI_GetInputFeature(seq_long_mean, 58)) / 3.0;
         double ctx_corr_fast = (FXAI_GetInputFeature(seq_short_mean, 53) +
                                 FXAI_GetInputFeature(seq_short_mean, 57) +
                                 FXAI_GetInputFeature(seq_short_mean, 61)) / 3.0;
         double ctx_strength = 0.30 * MathAbs(ctx_ret) +
                               0.30 * MathAbs(ctx_rel) +
                               0.25 * MathAbs(ctx_corr) +
                               0.15 * MathAbs(ctx_lag);

         FXAI_SetInputFeature(x, 10, 0.35 * FXAI_GetInputFeature(x, 10) + 0.35 * ctx_ret + 0.30 * ctx_ret_fast);
         FXAI_SetInputFeature(x, 11, 0.40 * FXAI_GetInputFeature(x, 11) + 0.35 * MathAbs(ctx_rel) + 0.25 * MathAbs(ctx_ret_fast - ctx_ret_slow));
         FXAI_SetInputFeature(x, 12, 0.35 * FXAI_GetInputFeature(x, 12) + 0.35 * ctx_corr + 0.30 * ctx_corr_fast);
         FXAI_SetInputFeature(x, 13, 0.50 * ctx_lag + 0.50 * (ctx_ret_fast - ctx_ret_slow));
         FXAI_SetInputFeature(x, 14, ctx_strength + 0.20 * MathAbs(ctx_ret_fast - ctx_ret_slow));
         // Keep explicit per-symbol context blocks instead of collapsing them all.
         FXAI_SetInputFeature(x, 15, FXAI_GetInputFeature(x, 50));
         FXAI_SetInputFeature(x, 16, FXAI_GetInputFeature(x, 51));
         FXAI_SetInputFeature(x, 17, FXAI_GetInputFeature(x, 52));
         FXAI_SetInputFeature(x, 18, FXAI_GetInputFeature(x, 53));
         FXAI_SetInputFeature(x, 19, FXAI_GetInputFeature(x, 54));
         FXAI_SetInputFeature(x, 20, FXAI_GetInputFeature(x, 55));
         FXAI_SetInputFeature(x, 21, FXAI_GetInputFeature(x, 56));
         FXAI_SetInputFeature(x, 22, FXAI_GetInputFeature(x, 57));
         FXAI_SetInputFeature(x, 50, FXAI_GetInputFeature(x, 58));
         FXAI_SetInputFeature(x, 51, FXAI_GetInputFeature(x, 59));
         FXAI_SetInputFeature(x, 52, FXAI_GetInputFeature(x, 60));
         FXAI_SetInputFeature(x, 53, FXAI_GetInputFeature(x, 61));
         FXAI_SetInputFeature(x, 54, FXAI_GetInputFeature(seq_mean, 50));
         FXAI_SetInputFeature(x, 55, FXAI_GetInputFeature(seq_mean, 51));
         FXAI_SetInputFeature(x, 56, FXAI_GetInputFeature(seq_mean, 52));
         FXAI_SetInputFeature(x, 57, FXAI_GetInputFeature(seq_mean, 53));
         FXAI_SetInputFeature(x, 58, FXAI_GetInputFeature(seq_delta, 50));
         FXAI_SetInputFeature(x, 59, FXAI_GetInputFeature(seq_short_delta, 51));
         FXAI_SetInputFeature(x, 60, FXAI_GetInputFeature(seq_std, 50));
         FXAI_SetInputFeature(x, 61, FXAI_GetInputFeature(seq_std, 51));
         break;
      }

      case FXAI_SCHEMA_FULL:
      default:
         break;
   }
}

void FXAI_ApplyFeatureSchemaToPayloadEx(const int schema_id,
                                        const ulong groups_mask,
                                        const int sequence_bars,
                                        double &x_window[][FXAI_AI_WEIGHTS],
                                        const int window_size,
                                        double &x[])
{
   int ws = window_size;
   if(ws < 0) ws = 0;
   if(ws > FXAI_MAX_SEQUENCE_BARS) ws = FXAI_MAX_SEQUENCE_BARS;

   double raw_window[FXAI_MAX_SEQUENCE_BARS][FXAI_AI_WEIGHTS];
   for(int b=0; b<FXAI_MAX_SEQUENCE_BARS; b++)
   {
      for(int k=0; k<FXAI_AI_WEIGHTS; k++)
      {
         raw_window[b][k] = 0.0;
         if(b < ws)
            raw_window[b][k] = x_window[b][k];
      }
   }

   // Project each historical row using only older rows. This gives plugins a
   // schema-native rolling payload instead of a raw normalized shared vector.
   for(int b=0; b<ws; b++)
   {
      double row[FXAI_AI_WEIGHTS];
      for(int k=0; k<FXAI_AI_WEIGHTS; k++)
         row[k] = raw_window[b][k];

      double tail_window[FXAI_MAX_SEQUENCE_BARS][FXAI_AI_WEIGHTS];
      int tail_size = 0;
      for(int tb=b + 1; tb<ws && tail_size<FXAI_MAX_SEQUENCE_BARS; tb++, tail_size++)
      {
         for(int k=0; k<FXAI_AI_WEIGHTS; k++)
            tail_window[tail_size][k] = raw_window[tb][k];
      }

      FXAI_ApplyFeatureSchemaToInputEx(schema_id,
                                       groups_mask,
                                       sequence_bars,
                                       tail_window,
                                       tail_size,
                                       row);
      for(int k=0; k<FXAI_AI_WEIGHTS; k++)
         x_window[b][k] = row[k];
   }

   FXAI_ApplyFeatureSchemaToInputEx(schema_id,
                                    groups_mask,
                                    sequence_bars,
                                    x_window,
                                    ws,
                                    x);
}

int FXAI_ContextExtraIndex(const int sample_idx, const int feat_idx)
{
   if(sample_idx < 0) return -1;
   if(feat_idx < 0 || feat_idx >= FXAI_CONTEXT_EXTRA_FEATS) return -1;
   return sample_idx * FXAI_CONTEXT_EXTRA_FEATS + feat_idx;
}

int FXAI_MainMTFBarsForSlot(const int tf_slot)
{
   switch(tf_slot)
   {
      case 0: return 5;
      case 1: return 15;
      case 2: return 30;
      case 3: return 60;
      default: return 1;
   }
}

int FXAI_ContextMTFBarsForSlot(const int tf_slot)
{
   switch(tf_slot)
   {
      case 0: return 1;
      case 1: return 5;
      case 2: return 15;
      case 3: return 30;
      case 4: return 60;
      default: return 1;
   }
}

int FXAI_MainMTFFeatureIndex(const int tf_slot, const int metric)
{
   if(tf_slot < 0 || tf_slot >= FXAI_MAIN_MTF_TF_COUNT)
      return -1;
   if(metric < 0 || metric >= FXAI_MTF_STATE_FEATURES_PER_TF)
      return -1;
   return FXAI_MAIN_MTF_FEATURE_OFFSET + tf_slot * FXAI_MTF_STATE_FEATURES_PER_TF + metric;
}

int FXAI_ContextMTFFeatureIndex(const int slot,
                                const int tf_slot,
                                const int metric)
{
   if(slot < 0 || slot >= FXAI_CONTEXT_TOP_SYMBOLS)
      return -1;
   if(tf_slot < 0 || tf_slot >= FXAI_CONTEXT_MTF_TF_COUNT)
      return -1;
   if(metric < 0 || metric >= FXAI_MTF_STATE_FEATURES_PER_TF)
      return -1;
   return FXAI_CONTEXT_MTF_FEATURE_OFFSET +
          (slot * FXAI_CONTEXT_MTF_TF_COUNT + tf_slot) * FXAI_MTF_STATE_FEATURES_PER_TF +
          metric;
}

int FXAI_ContextSlotMTFExtraIndex(const int slot,
                                  const int tf_slot,
                                  const int metric)
{
   if(slot < 0 || slot >= FXAI_CONTEXT_TOP_SYMBOLS)
      return -1;
   if(tf_slot < 0 || tf_slot >= FXAI_CONTEXT_MTF_TF_COUNT)
      return -1;
   if(metric < 0 || metric >= FXAI_MTF_STATE_FEATURES_PER_TF)
      return -1;
   return FXAI_CONTEXT_MTF_OFFSET +
          slot * FXAI_CONTEXT_SLOT_MTF_FEATS +
          tf_slot * FXAI_MTF_STATE_FEATURES_PER_TF +
          metric;
}

bool FXAI_ComputeAggregatedCandleSpreadState(const int idx,
                                             const int window_bars,
                                             const double &open_arr[],
                                             const double &high_arr[],
                                             const double &low_arr[],
                                             const double &close_arr[],
                                             const int &spread_arr[],
                                             const double point_value,
                                             double &body_bias,
                                             double &close_loc,
                                             double &range_pressure,
                                             double &spread_pressure)
{
   body_bias = 0.0;
   close_loc = 0.0;
   range_pressure = 0.0;
   spread_pressure = 0.0;

   int n = ArraySize(close_arr);
   if(idx < 0 || window_bars < 1 || n <= 0)
      return false;
   if(ArraySize(open_arr) != n || ArraySize(high_arr) != n || ArraySize(low_arr) != n || ArraySize(spread_arr) != n)
      return false;

   int last = idx + window_bars - 1;
   if(last >= n)
      return false;

   double point = (point_value > 0.0 ? point_value : 1.0);
   double agg_open = open_arr[last];
   double agg_close = close_arr[idx];
   double agg_high = high_arr[idx];
   double agg_low = low_arr[idx];
   double spread_sum_cur = 0.0;
   int spread_n_cur = 0;

   for(int k=0; k<window_bars; k++)
   {
      int ik = idx + k;
      if(ik < 0 || ik >= n)
         break;
      if(high_arr[ik] > agg_high) agg_high = high_arr[ik];
      if(low_arr[ik] < agg_low) agg_low = low_arr[ik];
      spread_sum_cur += MathMax((double)spread_arr[ik], 0.0);
      spread_n_cur++;
   }

   double bar_range = MathMax(agg_high - agg_low, point);
   double bar_range_points = MathMax(0.0, (agg_high - agg_low) / point);
   double spread_cur = spread_sum_cur / (double)MathMax(spread_n_cur, 1);

   double avg_range_points = 0.0;
   double avg_spread = 0.0;
   int windows_used = 0;
   for(int w=0; w<20; w++)
   {
      int base = idx + w * window_bars;
      int base_last = base + window_bars - 1;
      if(base < 0 || base_last >= n)
         break;

      double win_high = high_arr[base];
      double win_low = low_arr[base];
      double win_spread_sum = 0.0;
      int win_spread_n = 0;
      for(int k=0; k<window_bars; k++)
      {
         int ik = base + k;
         if(ik < 0 || ik >= n)
            break;
         if(high_arr[ik] > win_high) win_high = high_arr[ik];
         if(low_arr[ik] < win_low) win_low = low_arr[ik];
         win_spread_sum += MathMax((double)spread_arr[ik], 0.0);
         win_spread_n++;
      }

      avg_range_points += MathMax(0.0, (win_high - win_low) / point);
      avg_spread += win_spread_sum / (double)MathMax(win_spread_n, 1);
      windows_used++;
   }

   if(windows_used <= 0)
   {
      avg_range_points = MathMax(bar_range_points, 0.25);
      avg_spread = MathMax(spread_cur, 0.25);
      windows_used = 1;
   }
   else
   {
      avg_range_points /= (double)windows_used;
      avg_spread /= (double)windows_used;
   }

   body_bias = FXAI_Clamp((agg_close - agg_open) / bar_range, -1.2, 1.2);
   close_loc = FXAI_Clamp(((agg_close - agg_low) - (agg_high - agg_close)) / bar_range, -1.2, 1.2);
   range_pressure = FXAI_ClipSym((bar_range_points / MathMax(avg_range_points, 0.25)) - 1.0, 6.0);
   spread_pressure = FXAI_ClipSym((spread_cur / MathMax(avg_spread, 0.25)) - 1.0, 8.0);
   return true;
}

double FXAI_GetContextExtraValue(const double &arr[],
                                 const int sample_idx,
                                 const int feat_idx,
                                 const double def_value)
{
   int idx = FXAI_ContextExtraIndex(sample_idx, feat_idx);
   if(idx >= 0 && idx < ArraySize(arr)) return arr[idx];
   return def_value;
}

void FXAI_SetContextExtraValue(double &arr[],
                               const int sample_idx,
                               const int feat_idx,
                               const double value)
{
   int idx = FXAI_ContextExtraIndex(sample_idx, feat_idx);
   if(idx >= 0 && idx < ArraySize(arr))
      arr[idx] = value;
}


void FXAI_ApplyFeatureSchemaToInput(const int schema_id,
                                    const ulong groups_mask,
                                    double &x[])
{
   double dummy_window[FXAI_MAX_SEQUENCE_BARS][FXAI_AI_WEIGHTS];
   for(int b=0; b<FXAI_MAX_SEQUENCE_BARS; b++)
      for(int k=0; k<FXAI_AI_WEIGHTS; k++)
         dummy_window[b][k] = 0.0;
   FXAI_ApplyFeatureSchemaToPayloadEx(schema_id, groups_mask, 1, dummy_window, 0, x);
}

void FXAI_ClearContextV4(FXAIAIContextV4 &ctx)
{
   ctx.api_version = FXAI_API_VERSION_V4;
   ctx.regime_id = 0;
   ctx.session_bucket = 0;
   ctx.horizon_minutes = 1;
   ctx.feature_schema_id = FXAI_SCHEMA_FULL;
   ctx.normalization_method_id = FXAI_NORM_EXISTING;
   ctx.sequence_bars = 1;
   ctx.cost_points = 0.0;
   ctx.min_move_points = 0.0;
   ctx.point_value = 0.0;
    ctx.domain_hash = FXAI_SymbolHash01(_Symbol);
   ctx.sample_time = 0;
}

bool FXAI_ValidateContextV4(const FXAIAIContextV4 &ctx,
                            string &reason)
{
   if(ctx.api_version != FXAI_API_VERSION_V4)
   {
      reason = "ctx.api_version";
      return false;
   }
   if(ctx.regime_id < 0 || ctx.regime_id >= FXAI_PLUGIN_REGIME_BUCKETS)
   {
      reason = "ctx.regime_id";
      return false;
   }
   if(ctx.session_bucket < 0 || ctx.session_bucket >= FXAI_PLUGIN_SESSION_BUCKETS)
   {
      reason = "ctx.session_bucket";
      return false;
   }
   if(ctx.horizon_minutes <= 0)
   {
      reason = "ctx.horizon_minutes";
      return false;
   }
   if(ctx.feature_schema_id < FXAI_SCHEMA_FULL || ctx.feature_schema_id > FXAI_SCHEMA_CONTEXTUAL)
   {
      reason = "ctx.feature_schema_id";
      return false;
   }
   if(ctx.normalization_method_id < 0 || ctx.normalization_method_id >= FXAI_NORM_METHOD_COUNT)
   {
      reason = "ctx.normalization_method_id";
      return false;
   }
   if(ctx.sequence_bars <= 0 || ctx.sequence_bars > FXAI_MAX_SEQUENCE_BARS)
   {
      reason = "ctx.sequence_bars";
      return false;
   }
   if(!MathIsValidNumber(ctx.cost_points) || ctx.cost_points < 0.0)
   {
      reason = "ctx.cost_points";
      return false;
   }
   if(!MathIsValidNumber(ctx.min_move_points) || ctx.min_move_points < 0.0)
   {
      reason = "ctx.min_move_points";
      return false;
   }
   if(!MathIsValidNumber(ctx.domain_hash) || ctx.domain_hash < 0.0 || ctx.domain_hash > 1.0)
   {
      reason = "ctx.domain_hash";
      return false;
   }
   if(!MathIsValidNumber(ctx.point_value) || ctx.point_value <= 0.0)
   {
      reason = "ctx.point_value";
      return false;
   }
   reason = "";
   return true;
}

void FXAI_ClearPredictRequest(FXAIAIPredictRequestV4 &req)
{
   req.valid = false;
   FXAI_ClearContextV4(req.ctx);
   req.window_size = 0;
   for(int k=0; k<FXAI_AI_WEIGHTS; k++)
      req.x[k] = 0.0;
   for(int b=0; b<FXAI_MAX_SEQUENCE_BARS; b++)
      for(int k=0; k<FXAI_AI_WEIGHTS; k++)
         req.x_window[b][k] = 0.0;
}

void FXAI_ClearTrainRequest(FXAIAITrainRequestV4 &req)
{
   req.valid = false;
   FXAI_ClearContextV4(req.ctx);
   req.label_class = (int)FXAI_LABEL_SKIP;
   req.move_points = 0.0;
   req.sample_weight = 0.0;
   req.mfe_points = 0.0;
   req.mae_points = 0.0;
   req.time_to_hit_frac = 1.0;
   req.path_flags = 0;
   req.path_risk = 0.0;
   req.fill_risk = 0.0;
   req.masked_step_target = 0.0;
   req.next_vol_target = 0.0;
   req.regime_shift_target = 0.0;
   req.context_lead_target = 0.5;
   req.window_size = 0;
   for(int k=0; k<FXAI_AI_WEIGHTS; k++)
      req.x[k] = 0.0;
   for(int b=0; b<FXAI_MAX_SEQUENCE_BARS; b++)
      for(int k=0; k<FXAI_AI_WEIGHTS; k++)
         req.x_window[b][k] = 0.0;
}

bool FXAI_ValidatePredictRequestV4(const FXAIAIPredictRequestV4 &req,
                                   string &reason)
{
   if(!req.valid)
   {
      reason = "req.valid";
      return false;
   }
   if(!FXAI_ValidateContextV4(req.ctx, reason))
      return false;
   if(req.window_size < 0 || req.window_size > FXAI_MAX_SEQUENCE_BARS)
   {
      reason = "req.window_size";
      return false;
   }
   if(req.ctx.sequence_bars > 1 && req.window_size <= 0)
   {
      reason = "req.window_payload";
      return false;
   }
   for(int k=0; k<FXAI_AI_WEIGHTS; k++)
   {
      if(!MathIsValidNumber(req.x[k]))
      {
         reason = "req.x";
         return false;
      }
   }
   reason = "";
   return true;
}

bool FXAI_ValidateTrainRequestV4(const FXAIAITrainRequestV4 &req,
                                 string &reason)
{
   if(!req.valid)
   {
      reason = "req.valid";
      return false;
   }
   if(!FXAI_ValidateContextV4(req.ctx, reason))
      return false;
   if(req.window_size < 0 || req.window_size > FXAI_MAX_SEQUENCE_BARS)
   {
      reason = "req.window_size";
      return false;
   }
   if(req.ctx.sequence_bars > 1 && req.window_size <= 0)
   {
      reason = "req.window_payload";
      return false;
   }
   for(int k=0; k<FXAI_AI_WEIGHTS; k++)
   {
      if(!MathIsValidNumber(req.x[k]))
      {
         reason = "req.x";
         return false;
      }
   }
   if(req.label_class < (int)FXAI_LABEL_SELL || req.label_class > (int)FXAI_LABEL_SKIP)
   {
      reason = "req.label_class";
      return false;
   }
   if(!MathIsValidNumber(req.move_points))
   {
      reason = "req.move_points";
      return false;
   }
   if(!MathIsValidNumber(req.sample_weight) || req.sample_weight < 0.0)
   {
      reason = "req.sample_weight";
      return false;
   }
   if(!MathIsValidNumber(req.mfe_points) || req.mfe_points < 0.0 ||
      !MathIsValidNumber(req.mae_points) || req.mae_points < 0.0)
   {
      reason = "req.path_excursions";
      return false;
   }
   if(!MathIsValidNumber(req.time_to_hit_frac) || req.time_to_hit_frac < 0.0 || req.time_to_hit_frac > 1.0)
   {
      reason = "req.time_to_hit_frac";
      return false;
   }
   if(!MathIsValidNumber(req.path_risk) || req.path_risk < 0.0 || req.path_risk > 1.0)
   {
      reason = "req.path_risk";
      return false;
   }
   if(!MathIsValidNumber(req.fill_risk) || req.fill_risk < 0.0 || req.fill_risk > 1.0)
   {
      reason = "req.fill_risk";
      return false;
   }
   if(!MathIsValidNumber(req.masked_step_target))
   {
      reason = "req.masked_step_target";
      return false;
   }
   if(!MathIsValidNumber(req.next_vol_target) || req.next_vol_target < 0.0)
   {
      reason = "req.next_vol_target";
      return false;
   }
   if(!MathIsValidNumber(req.regime_shift_target) || req.regime_shift_target < 0.0 || req.regime_shift_target > 1.0)
   {
      reason = "req.regime_shift_target";
      return false;
   }
   if(!MathIsValidNumber(req.context_lead_target) || req.context_lead_target < 0.0 || req.context_lead_target > 1.0)
   {
      reason = "req.context_lead_target";
      return false;
   }
   reason = "";
   return true;
}

void FXAI_SetTrainRequestPathTargets(FXAIAITrainRequestV4 &req,
                                     const double mfe_points,
                                     const double mae_points,
                                     const double time_to_hit_frac,
                                     const int path_flags,
                                     const double spread_stress_points)
{
   req.mfe_points = MathMax(MathAbs(mfe_points), 0.0);
   req.mae_points = MathMax(MathAbs(mae_points), 0.0);
   req.time_to_hit_frac = FXAI_Clamp(time_to_hit_frac, 0.0, 1.0);
   req.path_flags = path_flags;
   req.path_risk = FXAI_PathRiskFromTargets(req.mfe_points,
                                            req.mae_points,
                                            req.ctx.min_move_points,
                                            req.time_to_hit_frac,
                                            req.path_flags);
   req.fill_risk = FXAI_FillRiskFromTargets(spread_stress_points,
                                            req.ctx.min_move_points,
                                            req.ctx.cost_points);
}

void FXAI_SetTrainRequestAuxTargets(FXAIAITrainRequestV4 &req,
                                    const double masked_step_target,
                                    const double next_vol_target,
                                    const double regime_shift_target,
                                    const double context_lead_target)
{
   req.masked_step_target = masked_step_target;
   req.next_vol_target = MathMax(next_vol_target, 0.0);
   req.regime_shift_target = FXAI_Clamp(regime_shift_target, 0.0, 1.0);
   req.context_lead_target = FXAI_Clamp(context_lead_target, 0.0, 1.0);
}

void FXAI_CopyWindowPayload(const double &src[][FXAI_AI_WEIGHTS], const int src_size, double &dst[][FXAI_AI_WEIGHTS], int &dst_size)
{
   dst_size = src_size;
   if(dst_size < 0) dst_size = 0;
   if(dst_size > FXAI_MAX_SEQUENCE_BARS) dst_size = FXAI_MAX_SEQUENCE_BARS;
   for(int b=0; b<FXAI_MAX_SEQUENCE_BARS; b++)
      for(int k=0; k<FXAI_AI_WEIGHTS; k++)
         dst[b][k] = (b < dst_size ? src[b][k] : 0.0);
}

#endif // __FXAI_CORE_MQH__
