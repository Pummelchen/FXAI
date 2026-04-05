#ifndef __FXAI_CORE_MQH__
#define __FXAI_CORE_MQH__

#define FXAI_CONTEXT_TOP_SYMBOLS 3
#define FXAI_MTF_STATE_FEATURES_PER_TF 4
#define FXAI_MAIN_MTF_TF_COUNT 4
#define FXAI_CONTEXT_MTF_TF_COUNT 5
#define FXAI_MAIN_MTF_FEATURE_OFFSET 84
#define FXAI_CONTEXT_MTF_FEATURE_OFFSET (FXAI_MAIN_MTF_FEATURE_OFFSET + FXAI_MAIN_MTF_TF_COUNT * FXAI_MTF_STATE_FEATURES_PER_TF)
#define FXAI_MACRO_EVENT_FEATURE_OFFSET (FXAI_CONTEXT_MTF_FEATURE_OFFSET + FXAI_CONTEXT_TOP_SYMBOLS * FXAI_CONTEXT_MTF_TF_COUNT * FXAI_MTF_STATE_FEATURES_PER_TF)
#define FXAI_MACRO_EVENT_FEATURES 20
#define FXAI_AI_FEATURES (FXAI_MACRO_EVENT_FEATURE_OFFSET + FXAI_MACRO_EVENT_FEATURES)
#define FXAI_AI_WEIGHTS (FXAI_AI_FEATURES + 1)
#define FXAI_AI_MLP_HIDDEN 12
#define FXAI_AI_COUNT 36
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
#define FXAI_SHARED_TRANSFER_FEATURES 28
#define FXAI_SHARED_TRANSFER_LATENT 12
#define FXAI_SHARED_TRANSFER_SEQUENCE_TOKENS 16
#define FXAI_SHARED_TRANSFER_BAR_FEATURES 12
#define FXAI_SHARED_TRANSFER_STATE_FEATURES FXAI_SHARED_TRANSFER_BAR_FEATURES
#define FXAI_SHARED_TRANSFER_DOMAIN_BUCKETS 8
#define FXAI_SHARED_TRANSFER_HORIZON_BUCKETS 8
#define FXAI_EXEC_TRACE_BARS 12
#define FXAI_BROKER_EXEC_TRACE_CAP 192
#define FXAI_BROKER_EXEC_SYMBOL_BUCKETS 12
#define FXAI_BROKER_EXEC_SIDE_COUNT 3
#define FXAI_BROKER_EXEC_ORDER_TYPE_COUNT 5
#define FXAI_BROKER_EXEC_EVENT_KIND_COUNT 4
#define FXAI_BROKER_EXEC_LIBRARY_CELLS (FXAI_BROKER_EXEC_SYMBOL_BUCKETS * FXAI_PLUGIN_SESSION_BUCKETS * FXAI_SHARED_TRANSFER_HORIZON_BUCKETS * FXAI_BROKER_EXEC_SIDE_COUNT * FXAI_BROKER_EXEC_ORDER_TYPE_COUNT)
#define FXAI_BROKER_EXEC_LIBRARY_EVENT_CELLS (FXAI_BROKER_EXEC_LIBRARY_CELLS * FXAI_BROKER_EXEC_EVENT_KIND_COUNT)
#define FXAI_ANALOG_MEMORY_CAP 384
#define FXAI_ANALOG_MEMORY_FEATS 12
#define FXAI_ANALOG_MEMORY_MIN_MATCHES 3
#define FXAI_API_VERSION_V4 4
#define FXAI_MAX_SEQUENCE_BARS 96

#ifndef FXAI_REGIME_COUNT
#define FXAI_REGIME_COUNT FXAI_PLUGIN_REGIME_BUCKETS
#endif

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
   AI_RANDOM_NOSKIP,
   AI_QCEW,
   AI_FEWC,
   AI_GHA,
   AI_TESSERACT
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

struct FXAIBrokerExecutionStats
{
   double coverage;
   double slippage_points;
   double latency_points;
   double reject_prob;
   double partial_fill_prob;
   double trace_coverage;
   double library_coverage;
   double fill_ratio_mean;
   double event_burst_penalty;
};

struct FXAIFoundationSignals
{
   double masked_step;
   double next_vol;
   double regime_transition;
   double context_alignment;
   double direction_bias;
   double move_ratio;
   double tradability;
   double trust;
};

struct FXAIStudentSignals
{
   double class_probs[3];
   double move_ratio;
   double tradability;
   double horizon_fit;
   double trust;
};

struct FXAIAnalogMemoryQuery
{
   double similarity;
   double direction_agreement;
   double edge_norm;
   double quality;
   double path_safety;
   double execution_safety;
   double domain_alignment;
   int    matches;
};

struct FXAIHierarchicalSignals
{
   double tradability;
   double direction_confidence;
   double move_adequacy;
   double path_quality;
   double execution_viability;
   double horizon_fit;
   double consistency;
   double score;
};

struct FXAIRegimeGraphQuery
{
   double persistence;
   double transition_confidence;
   double instability;
   double edge_bias;
   double quality_bias;
   double macro_alignment;
   int    predicted_regime;
};

bool   g_broker_execution_ready = false;
double g_broker_execution_obs[FXAI_PLUGIN_SESSION_BUCKETS][FXAI_SHARED_TRANSFER_HORIZON_BUCKETS];
double g_broker_execution_slippage_ema[FXAI_PLUGIN_SESSION_BUCKETS][FXAI_SHARED_TRANSFER_HORIZON_BUCKETS];
double g_broker_execution_latency_ema[FXAI_PLUGIN_SESSION_BUCKETS][FXAI_SHARED_TRANSFER_HORIZON_BUCKETS];
double g_broker_execution_reject_ema[FXAI_PLUGIN_SESSION_BUCKETS][FXAI_SHARED_TRANSFER_HORIZON_BUCKETS];
double g_broker_execution_partial_ema[FXAI_PLUGIN_SESSION_BUCKETS][FXAI_SHARED_TRANSFER_HORIZON_BUCKETS];
double g_broker_execution_library_obs[FXAI_BROKER_EXEC_LIBRARY_CELLS];
double g_broker_execution_library_slippage[FXAI_BROKER_EXEC_LIBRARY_CELLS];
double g_broker_execution_library_latency[FXAI_BROKER_EXEC_LIBRARY_CELLS];
double g_broker_execution_library_reject[FXAI_BROKER_EXEC_LIBRARY_CELLS];
double g_broker_execution_library_partial[FXAI_BROKER_EXEC_LIBRARY_CELLS];
double g_broker_execution_library_fill_ratio[FXAI_BROKER_EXEC_LIBRARY_CELLS];
double g_broker_execution_library_event_mass[FXAI_BROKER_EXEC_LIBRARY_EVENT_CELLS];
int    g_broker_execution_trace_head = 0;
int    g_broker_execution_trace_size = 0;
datetime g_broker_execution_trace_time[FXAI_BROKER_EXEC_TRACE_CAP];
int    g_broker_execution_trace_session[FXAI_BROKER_EXEC_TRACE_CAP];
int    g_broker_execution_trace_horizon[FXAI_BROKER_EXEC_TRACE_CAP];
int    g_broker_execution_trace_symbol_bucket[FXAI_BROKER_EXEC_TRACE_CAP];
int    g_broker_execution_trace_side[FXAI_BROKER_EXEC_TRACE_CAP];
int    g_broker_execution_trace_order_type[FXAI_BROKER_EXEC_TRACE_CAP];
int    g_broker_execution_trace_event_kind[FXAI_BROKER_EXEC_TRACE_CAP];
double g_broker_execution_trace_slippage[FXAI_BROKER_EXEC_TRACE_CAP];
double g_broker_execution_trace_latency[FXAI_BROKER_EXEC_TRACE_CAP];
double g_broker_execution_trace_reject[FXAI_BROKER_EXEC_TRACE_CAP];
double g_broker_execution_trace_partial[FXAI_BROKER_EXEC_TRACE_CAP];
double g_broker_execution_trace_fill_ratio[FXAI_BROKER_EXEC_TRACE_CAP];

bool     g_analog_memory_ready = false;
int      g_analog_memory_head = 0;
int      g_analog_memory_size = 0;
datetime g_analog_memory_time[FXAI_ANALOG_MEMORY_CAP];
int      g_analog_memory_regime[FXAI_ANALOG_MEMORY_CAP];
int      g_analog_memory_session[FXAI_ANALOG_MEMORY_CAP];
int      g_analog_memory_horizon[FXAI_ANALOG_MEMORY_CAP];
double   g_analog_memory_domain_hash[FXAI_ANALOG_MEMORY_CAP];
double   g_analog_memory_vec[FXAI_ANALOG_MEMORY_CAP][FXAI_ANALOG_MEMORY_FEATS];
double   g_analog_memory_direction[FXAI_ANALOG_MEMORY_CAP];
double   g_analog_memory_edge_norm[FXAI_ANALOG_MEMORY_CAP];
double   g_analog_memory_quality[FXAI_ANALOG_MEMORY_CAP];
double   g_analog_memory_path_risk[FXAI_ANALOG_MEMORY_CAP];
double   g_analog_memory_fill_risk[FXAI_ANALOG_MEMORY_CAP];
double   g_analog_memory_weight[FXAI_ANALOG_MEMORY_CAP];

#include "Core\core_analog_memory.mqh"

bool     g_regime_graph_ready = false;
int      g_regime_graph_last_regime = -1;
datetime g_regime_graph_last_time = 0;
double   g_regime_graph_transition_obs[FXAI_REGIME_COUNT][FXAI_REGIME_COUNT];
double   g_regime_graph_transition_edge[FXAI_REGIME_COUNT][FXAI_REGIME_COUNT];
double   g_regime_graph_transition_quality[FXAI_REGIME_COUNT][FXAI_REGIME_COUNT];
double   g_regime_graph_macro_alignment[FXAI_REGIME_COUNT][FXAI_REGIME_COUNT];
double   g_regime_graph_dwell_ema[FXAI_REGIME_COUNT];
double   g_regime_graph_outbound_mass[FXAI_REGIME_COUNT];

#include "Core\core_regime_graph.mqh"

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
      case AI_QCEW:
      case AI_FEWC:
      case AI_GHA:
      case AI_TESSERACT:
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

#include "Core\core_broker_execution.mqh"

#include "Core\core_model_context.mqh"

#include "Core\core_feature_schema.mqh"

#include "Core\core_runtime_perf.mqh"

#include "Core\core_requests.mqh"

#endif // __FXAI_CORE_MQH__
