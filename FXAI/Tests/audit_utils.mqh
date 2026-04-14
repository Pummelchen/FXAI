#ifndef __FXAI_AUDIT_UTILS_MQH__
#define __FXAI_AUDIT_UTILS_MQH__

double FXAI_AuditGetArrayValue(const double &arr[],
                               const int idx,
                               const double def_value)
{
   if(idx >= 0 && idx < ArraySize(arr)) return arr[idx];
   return def_value;
}

double FXAI_GetArrayValue(const double &arr[],
                          const int idx,
                          const double def_value)
{
   return FXAI_AuditGetArrayValue(arr, idx, def_value);
}

double FXAI_AuditGetIntArrayMean(const int &arr[],
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

int FXAI_AuditClampHorizon(const int h_in)
{
   int h = h_in;
   if(h < 1) h = 1;
   if(h > 720) h = 720;
   return h;
}

int FXAI_ClampHorizon(const int h_in)
{
   return FXAI_AuditClampHorizon(h_in);
}

int FXAI_AuditGetHorizonSlot(const int horizon_minutes)
{
   int h = FXAI_AuditClampHorizon(horizon_minutes);
   if(h <= 3) return 0;
   if(h <= 5) return 1;
   if(h <= 8) return 2;
   if(h <= 13) return 3;
   if(h <= 21) return 4;
   if(h <= 34) return 5;
   if(h <= 55) return 6;
   return 7;
}

int FXAI_GetHorizonSlot(const int horizon_minutes)
{
   return FXAI_AuditGetHorizonSlot(horizon_minutes);
}

ENUM_FXAI_FEATURE_NORMALIZATION FXAI_SanitizeNormMethod(const int method_id)
{
   int v = method_id;
   if(v < (int)FXAI_NORM_EXISTING || v > (int)FXAI_NORM_MINMAX_BUFFER3)
      v = (int)FXAI_NORM_EXISTING;
   return (ENUM_FXAI_FEATURE_NORMALIZATION)v;
}

int FXAI_AuditGetStaticRegimeId(const datetime sample_time,
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
   if(regime >= FXAI_PLUGIN_REGIME_BUCKETS) regime = FXAI_PLUGIN_REGIME_BUCKETS - 1;
   return regime;
}

void FXAI_AuditDefaultHyperParams(const int ai_idx,
                                  FXAIAIHyperParams &hp)
{
   hp.lr = 0.0100;
   hp.l2 = 0.0030;
   hp.ftrl_alpha = 0.08;
   hp.ftrl_beta = 1.0;
   hp.ftrl_l1 = 0.0005;
   hp.ftrl_l2 = 0.0100;
   hp.pa_c = 4.0;
   hp.pa_margin = 1.2;
   hp.xgb_lr = 0.08;
   hp.xgb_l2 = 0.02;
   hp.xgb_split = 0.50;
   hp.mlp_lr = 0.0100;
   hp.mlp_l2 = 0.0030;
   hp.mlp_init = 0.10;
   hp.quantile_lr = 0.0100;
   hp.quantile_l2 = 0.0030;
   hp.enhash_lr = 0.0100;
   hp.enhash_l1 = 0.0000;
   hp.enhash_l2 = 0.0050;
   hp.tcn_layers = 4.0;
   hp.tcn_kernel = 3.0;
   hp.tcn_dilation_base = 2.0;

   switch(ai_idx)
   {
      case (int)AI_M1SYNC:
      case (int)AI_BUY_ONLY:
      case (int)AI_SELL_ONLY:
      case (int)AI_RANDOM_NOSKIP:
         hp.lr = 0.0;
         hp.l2 = 0.0;
         break;
      case (int)AI_FTRL_LOGIT:
         hp.ftrl_alpha = 0.08;
         hp.ftrl_beta = 1.0;
         hp.ftrl_l1 = 0.0000;
         hp.ftrl_l2 = 0.01;
         break;
      case (int)AI_PA_LINEAR:
         hp.lr = 0.06;
         hp.l2 = 0.003;
         hp.pa_c = 4.0;
         hp.pa_margin = 1.2;
         break;
      case (int)AI_TCN:
         hp.lr = 0.006;
         hp.l2 = 0.002;
         hp.tcn_layers = 4.0;
         hp.tcn_kernel = 3.0;
         hp.tcn_dilation_base = 2.0;
         break;
      case (int)AI_LSTM:
      case (int)AI_LSTMG:
      case (int)AI_TFT:
      case (int)AI_AUTOFORMER:
      case (int)AI_PATCHTST:
      case (int)AI_CHRONOS:
      case (int)AI_TIMESFM:
      case (int)AI_TST:
      case (int)AI_STMN:
      case (int)AI_S4:
      case (int)AI_GEODESICATTENTION:
      case (int)AI_QCEW:
      case (int)AI_FEWC:
      case (int)AI_GHA:
      case (int)AI_TESSERACT:
         hp.lr = 0.006;
         hp.l2 = 0.002;
         break;
      default:
         break;
   }
}


#endif // __FXAI_AUDIT_UTILS_MQH__
