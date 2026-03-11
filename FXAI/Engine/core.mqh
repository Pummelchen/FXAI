#ifndef __FXAI_CORE_MQH__
#define __FXAI_CORE_MQH__

#define FXAI_AI_FEATURES 62
#define FXAI_AI_WEIGHTS (FXAI_AI_FEATURES + 1)
#define FXAI_AI_MLP_HIDDEN 8
#define FXAI_AI_COUNT 32
#define FXAI_NORM_METHOD_COUNT 15
#define FXAI_ENHASH_BUCKETS 128
#define FXAI_PLUGIN_CLASS_FEATURES 5
#define FXAI_PLUGIN_REGIME_BUCKETS 12
#define FXAI_PLUGIN_SESSION_BUCKETS 6
#define FXAI_PLUGIN_HORIZON_BUCKETS 8
#define FXAI_PLUGIN_REPLAY_CAPACITY 96
#define FXAI_PLUGIN_REPLAY_STEPS 2
#define FXAI_CONTEXT_TOP_SYMBOLS 3
#define FXAI_CONTEXT_EXTRA_FEATS (FXAI_CONTEXT_TOP_SYMBOLS * 4)
#define FXAI_CONTEXT_DYNAMIC_POOL 8
#define FXAI_API_VERSION_V4 4


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
   datetime sample_time;
};

struct FXAIAITrainRequestV4
{
   bool valid;
   FXAIAIContextV4 ctx;
   int label_class;
   double move_points;
   double sample_weight;
   double x[FXAI_AI_WEIGHTS];
};

struct FXAIAIPredictRequestV4
{
   bool valid;
   FXAIAIContextV4 ctx;
   double x[FXAI_AI_WEIGHTS];
};

struct FXAIAIPredictionV4
{
   double class_probs[3];
   double move_mean_points;
   double move_q25_points;
   double move_q50_points;
   double move_q75_points;
   double confidence;
   double reliability;
};

struct FXAIAIModelOutputV4
{
   double class_probs[3];
   double move_mean_points;
   double move_q25_points;
   double move_q50_points;
   double move_q75_points;
   double confidence;
   double reliability;
   bool   has_quantiles;
   bool   has_confidence;
};

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
         break;
      case FXAI_FAMILY_TREE:
         mask |= FXAI_FeatureGroupBit((int)FXAI_FEAT_GROUP_PRICE);
         mask |= FXAI_FeatureGroupBit((int)FXAI_FEAT_GROUP_MULTI_TIMEFRAME);
         mask |= FXAI_FeatureGroupBit((int)FXAI_FEAT_GROUP_VOLATILITY);
         mask |= FXAI_FeatureGroupBit((int)FXAI_FEAT_GROUP_TIME);
         mask |= FXAI_FeatureGroupBit((int)FXAI_FEAT_GROUP_CONTEXT);
         mask |= FXAI_FeatureGroupBit((int)FXAI_FEAT_GROUP_COST);
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
         break;
      default:
         mask |= FXAI_FeatureGroupBit((int)FXAI_FEAT_GROUP_PRICE);
         mask |= FXAI_FeatureGroupBit((int)FXAI_FEAT_GROUP_MULTI_TIMEFRAME);
         mask |= FXAI_FeatureGroupBit((int)FXAI_FEAT_GROUP_VOLATILITY);
         mask |= FXAI_FeatureGroupBit((int)FXAI_FEAT_GROUP_TIME);
         mask |= FXAI_FeatureGroupBit((int)FXAI_FEAT_GROUP_CONTEXT);
         mask |= FXAI_FeatureGroupBit((int)FXAI_FEAT_GROUP_COST);
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
   int group_id = FXAI_GetFeatureGroupForIndex(feature_idx);
   ulong bit = FXAI_FeatureGroupBit(group_id);
   if(bit == 0 || (groups_mask & bit) == 0)
      return false;

   switch(schema_id)
   {
      case FXAI_SCHEMA_SPARSE_STAT:
         if(feature_idx >= 46 && feature_idx <= 49) return false;
         if(feature_idx >= 50 && feature_idx <= 61) return false;
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

void FXAI_ApplyFeatureSchemaToInput(const int schema_id,
                                    const ulong groups_mask,
                                    double &x[])
{
   if(ArraySize(x) < FXAI_AI_WEIGHTS)
      return;

   x[0] = 1.0;
   for(int f=0; f<FXAI_AI_FEATURES; f++)
   {
      if(!FXAI_IsFeatureEnabledForSchema(f, schema_id, groups_mask))
         x[f + 1] = 0.0;
   }
}

int FXAI_ContextExtraIndex(const int sample_idx, const int feat_idx)
{
   if(sample_idx < 0) return -1;
   if(feat_idx < 0 || feat_idx >= FXAI_CONTEXT_EXTRA_FEATS) return -1;
   return sample_idx * FXAI_CONTEXT_EXTRA_FEATS + feat_idx;
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

#endif // __FXAI_CORE_MQH__
