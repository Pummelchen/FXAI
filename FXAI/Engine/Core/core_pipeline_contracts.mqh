#ifndef __FXAI_CORE_PIPELINE_CONTRACTS_MQH__
#define __FXAI_CORE_PIPELINE_CONTRACTS_MQH__

#ifndef FXAI_MAX_CONTEXT_SYMBOLS
#define FXAI_MAX_CONTEXT_SYMBOLS 48
#endif

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
   int horizon_minutes;
   int fit_start;
   int fit_end;
   bool ready;
   FXAIPreparedSample samples[];
};

struct FXAINormInputCache
{
   int method_id;
   int horizon_minutes;
   bool ready;
   double x[FXAI_AI_WEIGHTS];
};

struct FXAIDataCoreRequest
{
   bool live_mode;
   string symbol;
   datetime signal_bar;
   int needed;
   int align_upto;
   double commission_per_lot_side;
   double buffer_points;
   int context_symbol_count;
   string context_symbols[FXAI_MAX_CONTEXT_SYMBOLS];
};

struct FXAIDataCoreBundle
{
   bool ready;
   bool live_mode;
   string symbol;
   datetime signal_bar;
   int needed;
   int align_upto;
   FXAIDataSnapshot snapshot;
   datetime last_bar_m1;
   datetime last_bar_m5;
   datetime last_bar_m15;
   datetime last_bar_m30;
   datetime last_bar_h1;
   MqlRates rates_m1[];
   MqlRates rates_m5[];
   MqlRates rates_m15[];
   MqlRates rates_m30[];
   MqlRates rates_h1[];
   double open_arr[];
   double high_arr[];
   double low_arr[];
   double close_arr[];
   datetime time_arr[];
   int spread_m1[];
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
   FXAIContextSeries ctx_series[];
   double ctx_mean_arr[];
   double ctx_std_arr[];
   double ctx_up_arr[];
   double ctx_extra_arr[];
};

struct FXAIFeatureCoreRequest
{
   int sample_idx;
   int horizon_minutes;
   ENUM_FXAI_FEATURE_NORMALIZATION norm_method;
};

struct FXAIFeatureCoreFrame
{
   bool valid;
   int sample_idx;
   int horizon_minutes;
   ENUM_FXAI_FEATURE_NORMALIZATION norm_method;
   datetime sample_time;
   double spread_points;
   bool has_previous;
   double raw[FXAI_AI_FEATURES];
   double previous[FXAI_AI_FEATURES];
};

struct FXAINormalizationCoreFrame
{
   bool valid;
   int horizon_minutes;
   ENUM_FXAI_FEATURE_NORMALIZATION norm_method;
   datetime sample_time;
   double normalized[FXAI_AI_FEATURES];
   double model_input[FXAI_AI_WEIGHTS];
};

struct FXAINormalizationPayloadRequest
{
   bool valid;
   int feature_schema_id;
   ulong feature_groups_mask;
   int normalization_method_id;
   int horizon_minutes;
   int sequence_bars;
   datetime sample_time;
   int window_size;
   double x[FXAI_AI_WEIGHTS];
   double x_window[FXAI_MAX_SEQUENCE_BARS][FXAI_AI_WEIGHTS];
};

struct FXAINormalizationPayloadFrame
{
   bool valid;
   int feature_schema_id;
   ulong feature_groups_mask;
   int normalization_method_id;
   int horizon_minutes;
   int sequence_bars;
   datetime sample_time;
   int window_size;
   double x[FXAI_AI_WEIGHTS];
   double x_window[FXAI_MAX_SEQUENCE_BARS][FXAI_AI_WEIGHTS];
};

#endif // __FXAI_CORE_PIPELINE_CONTRACTS_MQH__
