// FXAI v1
#ifndef __FX6_SHARED_MQH__
#define __FX6_SHARED_MQH__

#define FX6_AI_FEATURES 15
#define FX6_AI_WEIGHTS (FX6_AI_FEATURES + 1)
#define FX6_AI_MLP_HIDDEN 8
#define FX6_AI_COUNT 22
#define FX6_ENHASH_BUCKETS 128
#define FX6_PLUGIN_CLASS_FEATURES 5

enum ENUM_AI_TYPE
{
   AI_TYPE_SGD_LOGIT = 0,
   AI_TYPE_FTRL_LOGIT,
   AI_TYPE_PA_LINEAR,
   AI_TYPE_XGB_FAST,
   AI_TYPE_MLP_TINY,
   AI_TYPE_LSTM,
   AI_TYPE_LSTMG,
   AI_TYPE_LIGHTGBM,
   AI_TYPE_S4,
   AI_TYPE_TCN,
   AI_TYPE_TFT,
   AI_TYPE_XGBOOST,
   AI_TYPE_QUANTILE,
   AI_TYPE_ENHASH,
   AI_TYPE_AUTOFORMER,
   AI_TYPE_STMN,
   AI_TYPE_TST,
   AI_TYPE_GEODESICATTENTION,
   AI_TYPE_CATBOOST,
   AI_TYPE_PATCHTST,
   AI_TYPE_CHRONOS,
   AI_TYPE_TIMESFM
};

enum ENUM_FX6_LABEL_CLASS
{
   FX6_LABEL_SELL = 0,
   FX6_LABEL_BUY  = 1,
   FX6_LABEL_SKIP = 2
};

struct FX6AIHyperParams
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

struct FX6DataSnapshot
{
   string symbol;
   datetime bar_time;
   double point;
   double spread_points;
   double commission_points;
   double min_move_points;
};

// V2 plugin API sample payload used for training.
struct FX6AISampleV2
{
   bool valid;
   int label_class;
   double move_points;
   double min_move_points;
   double cost_points;
   datetime sample_time;
   double x[FX6_AI_WEIGHTS];
};

// V2 plugin API payload used for inference.
struct FX6AIPredictV2
{
   double min_move_points;
   double cost_points;
   datetime sample_time;
   double x[FX6_AI_WEIGHTS];
};

// V2 plugin prediction output (native 3-class + move estimate).
struct FX6AIPredictionV2
{
   double class_probs[3];
   double p_up;
   double expected_move_points;
};

double FX6_Clamp(const double v, const double lo, const double hi)
{
   if(v < lo) return lo;
   if(v > hi) return hi;
   return v;
}

double FX6_Sigmoid(const double z)
{
   if(z > 35.0) return 1.0;
   if(z < -35.0) return 0.0;
   return 1.0 / (1.0 + MathExp(-z));
}

double FX6_Logit(const double p)
{
   double x = FX6_Clamp(p, 1e-6, 1.0 - 1e-6);
   return MathLog(x / (1.0 - x));
}

double FX6_Tanh(const double z)
{
   if(z > 18.0) return 1.0;
   if(z < -18.0) return -1.0;
   double e2 = MathExp(2.0 * z);
   return (e2 - 1.0) / (e2 + 1.0);
}

double FX6_DotLinear(const double &w[], const double &x[])
{
   double z = 0.0;
   for(int i=0; i<FX6_AI_WEIGHTS; i++)
      z += w[i] * x[i];
   return z;
}

double FX6_Sign(const double v)
{
   if(v > 0.0) return 1.0;
   if(v < 0.0) return -1.0;
   return 0.0;
}

double FX6_ClipSym(const double v, const double limit_abs)
{
   double lim = (limit_abs > 0.0 ? limit_abs : 0.0);
   if(lim <= 0.0) return v;
   if(v > lim) return lim;
   if(v < -lim) return -lim;
   return v;
}

double FX6_MoveWeight(const double move_points)
{
   double a = MathAbs(move_points);
   // Keep weighting lightweight and bounded for stable online updates.
   return FX6_Clamp(1.0 + (0.05 * a), 0.80, 3.00);
}

double FX6_MoveEdgeWeight(const double move_points, const double cost_points)
{
   double mv = MathAbs(move_points);
   double c = (cost_points > 0.0 ? cost_points : 0.0);
   double edge = mv - c;
   double denom = MathMax(c, 1.0);
   return FX6_Clamp(0.50 + (edge / denom), 0.25, 4.00);
}

void FX6_UpdateMoveEMA(double &ema_abs_move,
                       bool &ready,
                       const double move_points,
                       const double alpha)
{
   double a = FX6_Clamp(alpha, 0.001, 0.500);
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

int FX6_ThreeWayBranch(const double x, const double split)
{
   if(x < split - 0.50) return 0;
   if(x > split + 0.50) return 2;
   return 1;
}

void FX6_BuildInputVector(const double &features[], double &x[])
{
   x[0] = 1.0;
   for(int i=0; i<FX6_AI_FEATURES; i++)
      x[i + 1] = features[i];
}

#endif // __FX6_SHARED_MQH__
