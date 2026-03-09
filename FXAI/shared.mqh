// FXAI v2
// FXAI v1
#ifndef __FXAI_SHARED_MQH__
#define __FXAI_SHARED_MQH__

#define FXAI_AI_FEATURES 62
#define FXAI_AI_WEIGHTS (FXAI_AI_FEATURES + 1)
#define FXAI_AI_MLP_HIDDEN 8
#define FXAI_AI_COUNT 28
#define FXAI_ENHASH_BUCKETS 128
#define FXAI_PLUGIN_CLASS_FEATURES 5
#define FXAI_CONTEXT_TOP_SYMBOLS 3
#define FXAI_CONTEXT_EXTRA_FEATS (FXAI_CONTEXT_TOP_SYMBOLS * 4)


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
   AI_RETRDIFF
};


enum ENUM_FXAI_LABEL_CLASS
{
   FXAI_LABEL_SELL = 0,
   FXAI_LABEL_BUY  = 1,
   FXAI_LABEL_SKIP = 2
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

// V2 plugin API sample payload used for training.
struct FXAIAISampleV2
{
   bool valid;
   int label_class;
   double move_points;
   double min_move_points;
   double cost_points;
   datetime sample_time;
   double x[FXAI_AI_WEIGHTS];
};

// V2 plugin API payload used for inference.
struct FXAIAIPredictV2
{
   double min_move_points;
   double cost_points;
   datetime sample_time;
   double x[FXAI_AI_WEIGHTS];
};

// V2 plugin prediction output (native 3-class + move estimate).
struct FXAIAIPredictionV2
{
   double class_probs[3];
   double p_up;
   double expected_move_points;
};

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

#endif // __FXAI_SHARED_MQH__
