#ifndef __FXAI_FEATURE_NORM_MQH__
#define __FXAI_FEATURE_NORM_MQH__

#define FXAI_NORM_QUANTILE_KNOTS 17

bool g_fxai_norm_hist_inited = false;
datetime g_fxai_norm_last_sample_time[FXAI_NORM_METHOD_COUNT];
int g_fxai_norm_last_cfg_version[FXAI_NORM_METHOD_COUNT];
double g_fxai_norm_hist[FXAI_NORM_METHOD_COUNT][FXAI_AI_FEATURES][FXAI_NORM_ROLL_WINDOW_MAX];
int g_fxai_norm_hist_count[FXAI_NORM_METHOD_COUNT][FXAI_AI_FEATURES];
int g_fxai_norm_hist_head[FXAI_NORM_METHOD_COUNT][FXAI_AI_FEATURES];

bool g_fxai_norm_fit_inited = false;
bool g_fxai_norm_fit_ready[FXAI_MAX_HORIZONS][FXAI_NORM_METHOD_COUNT];
int g_fxai_norm_fit_obs[FXAI_MAX_HORIZONS][FXAI_NORM_METHOD_COUNT];
double g_fxai_norm_fit_min[FXAI_MAX_HORIZONS][FXAI_NORM_METHOD_COUNT][FXAI_AI_FEATURES];
double g_fxai_norm_fit_max[FXAI_MAX_HORIZONS][FXAI_NORM_METHOD_COUNT][FXAI_AI_FEATURES];
double g_fxai_norm_fit_mean[FXAI_MAX_HORIZONS][FXAI_NORM_METHOD_COUNT][FXAI_AI_FEATURES];
double g_fxai_norm_fit_std[FXAI_MAX_HORIZONS][FXAI_NORM_METHOD_COUNT][FXAI_AI_FEATURES];
double g_fxai_norm_fit_median[FXAI_MAX_HORIZONS][FXAI_NORM_METHOD_COUNT][FXAI_AI_FEATURES];
double g_fxai_norm_fit_iqr[FXAI_MAX_HORIZONS][FXAI_NORM_METHOD_COUNT][FXAI_AI_FEATURES];
double g_fxai_norm_fit_yeojohnson_lambda[FXAI_MAX_HORIZONS][FXAI_NORM_METHOD_COUNT][FXAI_AI_FEATURES];
double g_fxai_norm_fit_yeojohnson_mean[FXAI_MAX_HORIZONS][FXAI_NORM_METHOD_COUNT][FXAI_AI_FEATURES];
double g_fxai_norm_fit_yeojohnson_std[FXAI_MAX_HORIZONS][FXAI_NORM_METHOD_COUNT][FXAI_AI_FEATURES];
double g_fxai_norm_fit_quantiles[FXAI_MAX_HORIZONS][FXAI_NORM_METHOD_COUNT][FXAI_AI_FEATURES][FXAI_NORM_QUANTILE_KNOTS];

bool FXAI_MethodUsesAdaptivePayloadNormalization(const ENUM_FXAI_FEATURE_NORMALIZATION method)
{
   return (method == FXAI_NORM_REVIN || method == FXAI_NORM_DAIN);
}

bool FXAI_MethodUsesFittedStats(const ENUM_FXAI_FEATURE_NORMALIZATION method)
{
   switch(method)
   {
      case FXAI_NORM_MINMAX_BUFFER5:
      case FXAI_NORM_MINMAX_BUFFER2:
      case FXAI_NORM_MINMAX_BUFFER3:
      case FXAI_NORM_ZSCORE:
      case FXAI_NORM_ROBUST_MEDIAN_IQR:
      case FXAI_NORM_QUANTILE_TO_NORMAL:
      case FXAI_NORM_POWER_YEOJOHNSON:
      case FXAI_NORM_REVIN:
      case FXAI_NORM_DAIN:
         return true;
      default:
         return false;
   }
}

bool FXAI_MethodUsesRollingNormalizationHistory(const ENUM_FXAI_FEATURE_NORMALIZATION method)
{
   return (method == FXAI_NORM_MINMAX_BUFFER5 ||
           method == FXAI_NORM_MINMAX_BUFFER2 ||
           method == FXAI_NORM_MINMAX_BUFFER3 ||
           method == FXAI_NORM_ZSCORE ||
           method == FXAI_NORM_ROBUST_MEDIAN_IQR ||
           method == FXAI_NORM_QUANTILE_TO_NORMAL);
}

int FXAI_NormalizationHorizonSlot(const int horizon_minutes)
{
   int hslot = FXAI_GetHorizonSlot(horizon_minutes);
   if(hslot >= 0 && hslot < FXAI_MAX_HORIZONS)
      return hslot;

   hslot = FXAI_GetHorizonSlot(FXAI_ClampHorizon(PredictionTargetMinutes));
   if(hslot >= 0 && hslot < FXAI_MAX_HORIZONS)
      return hslot;
   return 0;
}

double FXAI_MinMaxBufferMap(const double v,
                            const double lo,
                            const double hi,
                            const double buffer_frac)
{
   double span = hi - lo;
   if(span <= 1e-12)
      return 0.5;

   double frac = FXAI_Clamp(buffer_frac, 0.0, 0.50);
   double lo_b = lo - frac * span;
   double hi_b = hi + frac * span;
   double den = hi_b - lo_b;
   if(den <= 1e-12)
      return 0.5;

   return FXAI_ClampUnitOpen((v - lo_b) / den);
}

double FXAI_MinMaxBuffer5Map(const double v, const double lo, const double hi)
{
   return FXAI_MinMaxBufferMap(v, lo, hi, 0.05);
}

double FXAI_ZScoreFromBounds(const double v, const double lo, const double hi)
{
   double span = hi - lo;
   if(span <= 1e-12) return 0.0;
   double mean = 0.5 * (lo + hi);
   double std = span / 4.0;
   if(std <= 1e-12) return 0.0;
   return (v - mean) / std;
}

double FXAI_RobustScaleFromBounds(const double v, const double lo, const double hi)
{
   double span = hi - lo;
   if(span <= 1e-12) return 0.0;
   double median = 0.5 * (lo + hi);
   double iqr = span / 2.0;
   if(iqr <= 1e-12) return 0.0;
   return (v - median) / iqr;
}

double FXAI_InvNormCDF(const double p)
{
   double x = FXAI_Clamp(p, 1e-12, 1.0 - 1e-12);

   const double a1 = -3.969683028665376e+01;
   const double a2 =  2.209460984245205e+02;
   const double a3 = -2.759285104469687e+02;
   const double a4 =  1.383577518672690e+02;
   const double a5 = -3.066479806614716e+01;
   const double a6 =  2.506628277459239e+00;

   const double b1 = -5.447609879822406e+01;
   const double b2 =  1.615858368580409e+02;
   const double b3 = -1.556989798598866e+02;
   const double b4 =  6.680131188771972e+01;
   const double b5 = -1.328068155288572e+01;

   const double c1 = -7.784894002430293e-03;
   const double c2 = -3.223964580411365e-01;
   const double c3 = -2.400758277161838e+00;
   const double c4 = -2.549732539343734e+00;
   const double c5 =  4.374664141464968e+00;
   const double c6 =  2.938163982698783e+00;

   const double d1 =  7.784695709041462e-03;
   const double d2 =  3.224671290700398e-01;
   const double d3 =  2.445134137142996e+00;
   const double d4 =  3.754408661907416e+00;

   const double p_low  = 0.02425;
   const double p_high = 1.0 - p_low;

   if(x < p_low)
   {
      double q = MathSqrt(-2.0 * MathLog(x));
      return (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) /
             ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0);
   }

   if(x > p_high)
   {
      double q = MathSqrt(-2.0 * MathLog(1.0 - x));
      return -(((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) /
               ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0);
   }

   double q = x - 0.5;
   double r = q * q;
   return (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6) * q /
          (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1.0);
}

double FXAI_YeoJohnson(const double x, const double lambda)
{
   double l = lambda;
   if(x >= 0.0)
   {
      if(MathAbs(l) < 1e-8)
         return MathLog(1.0 + x);
      return (MathPow(1.0 + x, l) - 1.0) / l;
   }

   double lm2 = 2.0 - l;
   if(MathAbs(lm2) < 1e-8)
      return -MathLog(1.0 - x);
   return -((MathPow(1.0 - x, lm2) - 1.0) / lm2);
}

double FXAI_SignedLog1P(const double x)
{
   double ax = MathAbs(x);
   return (x < 0.0 ? -1.0 : 1.0) * MathLog(1.0 + ax);
}

void FXAI_VectorMeanStd(const double &v[], double &mean, double &std)
{
   double sum = 0.0;
   double sum2 = 0.0;
   int n = 0;
   for(int i=0; i<FXAI_AI_FEATURES; i++)
   {
      double x = v[i];
      if(!MathIsValidNumber(x)) continue;
      sum += x;
      sum2 += x * x;
      n++;
   }

   if(n <= 0)
   {
      mean = 0.0;
      std = 1.0;
      return;
   }

   mean = sum / (double)n;
   double var = (sum2 / (double)n) - (mean * mean);
   if(var < 1e-12) var = 1e-12;
   std = MathSqrt(var);
}

void FXAI_GetFeatureClipBounds(const int f, double &lo, double &hi)
{
   lo = -8.0;
   hi = 8.0;

   if(f == 5)
   {
      lo = 0.0;
      hi = 10.0;
   }
   else if(f == 6)
   {
      lo = 0.0;
      hi = 12.0;
   }
   else if(f == 12)
   {
      lo = -1.0;
      hi = 1.0;
   }
   else if(f >= 15 && f <= 17)
   {
      lo = -1.2;
      hi = 1.2;
   }
   else if(f == 40)
   {
      lo = -1.2;
      hi = 1.2;
   }
   else if(f >= 41 && f <= 43)
   {
      lo = 0.0;
      hi = 40.0;
   }
   else if(f >= 44 && f <= 45)
   {
      lo = 0.0;
      hi = 40.0;
   }
   else if(f == 47)
   {
      lo = -12.0;
      hi = 12.0;
   }
   else if(f >= 62 && f <= 67)
   {
      lo = -1.2;
      hi = 1.2;
   }
   else if(f >= 68 && f <= 69)
   {
      lo = -8.0;
      hi = 8.0;
   }
   else if(f == 70)
   {
      lo = 0.0;
      hi = 8.0;
   }
   else if(f == 71)
   {
      lo = -6.0;
      hi = 6.0;
   }
   else if(f >= 72 && f <= 75)
   {
      lo = -1.2;
      hi = 1.2;
   }
   else if(f == 76 || f == 77)
   {
      lo = -4.5;
      hi = 4.5;
   }
   else if(f == 78 || f == 79)
   {
      lo = -6.0;
      hi = 6.0;
   }
   else if(f == 80)
   {
      lo = 0.0;
      hi = 6.0;
   }
   else if(f == 81)
   {
      lo = -8.0;
      hi = 8.0;
   }
   else if(f == 82)
   {
      lo = 0.0;
      hi = 4.5;
   }
   else if(f == 83)
   {
      lo = -1.1;
      hi = 1.1;
   }
   else if(f >= FXAI_MAIN_MTF_FEATURE_OFFSET && f < FXAI_MACRO_EVENT_FEATURE_OFFSET)
   {
      int rel = 0;
      if(f >= FXAI_CONTEXT_MTF_FEATURE_OFFSET)
         rel = (f - FXAI_CONTEXT_MTF_FEATURE_OFFSET) % FXAI_MTF_STATE_FEATURES_PER_TF;
      else
         rel = (f - FXAI_MAIN_MTF_FEATURE_OFFSET) % FXAI_MTF_STATE_FEATURES_PER_TF;

      if(rel <= 1)
      {
         lo = -1.2;
         hi = 1.2;
      }
      else if(rel == 2)
      {
         lo = -6.0;
         hi = 6.0;
      }
      else
      {
         lo = -6.0;
         hi = 8.0;
      }
   }
   else if(f >= FXAI_MACRO_EVENT_FEATURE_OFFSET && f < FXAI_AI_FEATURES)
   {
      int rel = f - FXAI_MACRO_EVENT_FEATURE_OFFSET;
      if(rel <= 2)
      {
         lo = 0.0;
         hi = 1.2;
      }
      else if(rel == 3)
      {
         lo = -6.0;
         hi = 6.0;
      }
      else if(rel == 4)
      {
         lo = 0.0;
         hi = 6.0;
      }
      else
      {
         lo = -1.2;
         hi = 1.2;
      }
   }
   else if(f >= 50 && f < FXAI_AI_FEATURES)
   {
      int rel = (f - 50) % 4;
      if(rel == 3)
      {
         lo = -1.1;
         hi = 1.1;
      }
      else
      {
         lo = -8.0;
         hi = 8.0;
      }
   }
}

void FXAI_GetNormalizationFallbackStats(const int f,
                                        double &fit_min,
                                        double &fit_max,
                                        double &fit_mean,
                                        double &fit_std,
                                        double &fit_median,
                                        double &fit_iqr)
{
   double lo = -8.0;
   double hi = 8.0;
   FXAI_GetFeatureClipBounds(f, lo, hi);
   fit_min = lo;
   fit_max = hi;
   fit_mean = 0.5 * (lo + hi);
   fit_std = MathMax((hi - lo) / 4.0, 1e-6);
   fit_median = fit_mean;
   fit_iqr = MathMax((hi - lo) / 2.0, 1e-6);
}

double FXAI_SortedQuantile(const double &sorted_values[],
                           const int count,
                           const double q)
{
   if(count <= 0)
      return 0.0;
   if(count == 1)
      return sorted_values[0];

   double qq = FXAI_Clamp(q, 0.0, 1.0);
   double pos = qq * (double)(count - 1);
   int lo_idx = (int)MathFloor(pos);
   int hi_idx = (lo_idx + 1 < count ? lo_idx + 1 : lo_idx);
   double frac = pos - (double)lo_idx;
   return sorted_values[lo_idx] + frac * (sorted_values[hi_idx] - sorted_values[lo_idx]);
}

void FXAI_ClearNormalizationFitSlot(const int hslot,
                                    const int method_idx)
{
   if(hslot < 0 || hslot >= FXAI_MAX_HORIZONS)
      return;
   if(method_idx < 0 || method_idx >= FXAI_NORM_METHOD_COUNT)
      return;

   g_fxai_norm_fit_ready[hslot][method_idx] = false;
   g_fxai_norm_fit_obs[hslot][method_idx] = 0;
   for(int f=0; f<FXAI_AI_FEATURES; f++)
   {
      double fit_min = 0.0;
      double fit_max = 0.0;
      double fit_mean = 0.0;
      double fit_std = 1.0;
      double fit_median = 0.0;
      double fit_iqr = 1.0;
      FXAI_GetNormalizationFallbackStats(f,
                                         fit_min,
                                         fit_max,
                                         fit_mean,
                                         fit_std,
                                         fit_median,
                                         fit_iqr);
      g_fxai_norm_fit_min[hslot][method_idx][f] = fit_min;
      g_fxai_norm_fit_max[hslot][method_idx][f] = fit_max;
      g_fxai_norm_fit_mean[hslot][method_idx][f] = fit_mean;
      g_fxai_norm_fit_std[hslot][method_idx][f] = fit_std;
      g_fxai_norm_fit_median[hslot][method_idx][f] = fit_median;
      g_fxai_norm_fit_iqr[hslot][method_idx][f] = fit_iqr;
      g_fxai_norm_fit_yeojohnson_lambda[hslot][method_idx][f] = 1.0;
      g_fxai_norm_fit_yeojohnson_mean[hslot][method_idx][f] = fit_mean;
      g_fxai_norm_fit_yeojohnson_std[hslot][method_idx][f] = fit_std;
      for(int k=0; k<FXAI_NORM_QUANTILE_KNOTS; k++)
      {
         double p = (FXAI_NORM_QUANTILE_KNOTS > 1 ? (double)k / (double)(FXAI_NORM_QUANTILE_KNOTS - 1) : 0.0);
         g_fxai_norm_fit_quantiles[hslot][method_idx][f][k] =
            fit_min + p * (fit_max - fit_min);
      }
   }
}

void FXAI_ResetFeatureNormalizationFits()
{
   g_fxai_norm_fit_inited = true;
   for(int h=0; h<FXAI_MAX_HORIZONS; h++)
      for(int m=0; m<FXAI_NORM_METHOD_COUNT; m++)
         FXAI_ClearNormalizationFitSlot(h, m);
}

void FXAI_ResetFeatureNormalizationMethodState(const int method_id)
{
   int method_idx = method_id;
   if(method_idx < 0) method_idx = 0;
   if(method_idx >= FXAI_NORM_METHOD_COUNT) method_idx = FXAI_NORM_METHOD_COUNT - 1;

   g_fxai_norm_last_sample_time[method_idx] = 0;
   g_fxai_norm_last_cfg_version[method_idx] = -1;
   for(int f=0; f<FXAI_AI_FEATURES; f++)
   {
      g_fxai_norm_hist_count[method_idx][f] = 0;
      g_fxai_norm_hist_head[method_idx][f] = 0;
      for(int k=0; k<FXAI_NORM_ROLL_WINDOW_MAX; k++)
         g_fxai_norm_hist[method_idx][f][k] = 0.0;
   }
}

void FXAI_ResetFeatureNormalizationState()
{
   g_fxai_norm_hist_inited = true;
   for(int m=0; m<FXAI_NORM_METHOD_COUNT; m++)
      FXAI_ResetFeatureNormalizationMethodState(m);

   if(!g_fxai_norm_fit_inited)
      FXAI_ResetFeatureNormalizationFits();
}

bool FXAI_GetFeatureNormalizationFitStats(const ENUM_FXAI_FEATURE_NORMALIZATION method,
                                          const int horizon_minutes,
                                          const int feature_idx,
                                          double &fit_min,
                                          double &fit_max,
                                          double &fit_mean,
                                          double &fit_std,
                                          double &fit_median,
                                          double &fit_iqr,
                                          double &fit_lambda,
                                          double &fit_yj_mean,
                                          double &fit_yj_std,
                                          bool &fit_ready)
{
   FXAI_GetNormalizationFallbackStats(feature_idx,
                                      fit_min,
                                      fit_max,
                                      fit_mean,
                                      fit_std,
                                      fit_median,
                                      fit_iqr);
   fit_lambda = 1.0;
   fit_yj_mean = fit_mean;
   fit_yj_std = fit_std;
   fit_ready = false;

   if(!g_fxai_norm_fit_inited || !FXAI_MethodUsesFittedStats(method))
      return false;

   int hslot = FXAI_NormalizationHorizonSlot(horizon_minutes);
   int method_idx = (int)method;
   if(method_idx < 0 || method_idx >= FXAI_NORM_METHOD_COUNT)
      return false;

   fit_ready = g_fxai_norm_fit_ready[hslot][method_idx];
   fit_min = g_fxai_norm_fit_min[hslot][method_idx][feature_idx];
   fit_max = g_fxai_norm_fit_max[hslot][method_idx][feature_idx];
   fit_mean = g_fxai_norm_fit_mean[hslot][method_idx][feature_idx];
   fit_std = MathMax(g_fxai_norm_fit_std[hslot][method_idx][feature_idx], 1e-6);
   fit_median = g_fxai_norm_fit_median[hslot][method_idx][feature_idx];
   fit_iqr = MathMax(g_fxai_norm_fit_iqr[hslot][method_idx][feature_idx], 1e-6);
   fit_lambda = g_fxai_norm_fit_yeojohnson_lambda[hslot][method_idx][feature_idx];
   fit_yj_mean = g_fxai_norm_fit_yeojohnson_mean[hslot][method_idx][feature_idx];
   fit_yj_std = MathMax(g_fxai_norm_fit_yeojohnson_std[hslot][method_idx][feature_idx], 1e-6);
   return fit_ready;
}

double FXAI_QuantileToNormalFromFit(const int hslot,
                                    const int method_idx,
                                    const int feature_idx,
                                    const double cur)
{
   double first = g_fxai_norm_fit_quantiles[hslot][method_idx][feature_idx][0];
   double last = g_fxai_norm_fit_quantiles[hslot][method_idx][feature_idx][FXAI_NORM_QUANTILE_KNOTS - 1];
   if(cur <= first)
      return FXAI_Clamp(FXAI_InvNormCDF(1e-6), -6.0, 6.0);
   if(cur >= last)
      return FXAI_Clamp(FXAI_InvNormCDF(1.0 - 1e-6), -6.0, 6.0);

   double q = 0.5;
   for(int k=0; k<FXAI_NORM_QUANTILE_KNOTS - 1; k++)
   {
      double q0 = g_fxai_norm_fit_quantiles[hslot][method_idx][feature_idx][k];
      double q1 = g_fxai_norm_fit_quantiles[hslot][method_idx][feature_idx][k + 1];
      if(cur > q1)
         continue;

      double p0 = (double)k / (double)(FXAI_NORM_QUANTILE_KNOTS - 1);
      double p1 = (double)(k + 1) / (double)(FXAI_NORM_QUANTILE_KNOTS - 1);
      if(MathAbs(q1 - q0) < 1e-9)
         q = 0.5 * (p0 + p1);
      else
         q = p0 + (cur - q0) / (q1 - q0) * (p1 - p0);
      break;
   }

   return FXAI_Clamp(FXAI_InvNormCDF(FXAI_Clamp(q, 1e-6, 1.0 - 1e-6)), -6.0, 6.0);
}

double FXAI_RawNormalizationMatrixValue(const double &rows[],
                                        const int row_idx,
                                        const int feature_idx)
{
   int idx = row_idx * FXAI_AI_FEATURES + feature_idx;
   if(idx < 0 || idx >= ArraySize(rows))
      return 0.0;
   return rows[idx];
}

bool FXAI_BuildRawNormalizationMatrix(const int i_start,
                                      const int i_end,
                                      const int horizon_minutes,
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
                                      const ENUM_FXAI_FEATURE_NORMALIZATION norm_method,
                                      double &rows[],
                                      int &row_count)
{
   row_count = 0;
   ArrayResize(rows, 0);

   int n = ArraySize(close_arr);
   if(n <= 0)
      return false;

   int start = i_start;
   int end = i_end;
   if(start < 0) start = 0;
   if(end >= n) end = n - 1;
   if(end < start)
      return false;

   int cap = end - start + 1;
   ArrayResize(rows, cap * FXAI_AI_FEATURES);

   for(int i=end; i>=start; i--)
   {
      double spread_i = FXAI_GetSpreadAtIndex(i, spread_m1, snapshot.spread_points);
      double ctx_mean_i = FXAI_GetArrayValue(ctx_mean_arr, i, 0.0);
      double ctx_std_i = FXAI_GetArrayValue(ctx_std_arr, i, 0.0);
      double ctx_up_i = FXAI_GetArrayValue(ctx_up_arr, i, 0.5);

      double feat[FXAI_AI_FEATURES];
      if(!FXAI_ComputeFeatureVector(i,
                                    snapshot.symbol,
                                    spread_i,
                                    time_arr,
                                    open_arr,
                                    high_arr,
                                    low_arr,
                                    close_arr,
                                    spread_m1,
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
      {
         continue;
      }

      for(int f=0; f<FXAI_AI_FEATURES; f++)
         rows[row_count * FXAI_AI_FEATURES + f] = feat[f];
      row_count++;
   }

   ArrayResize(rows, row_count * FXAI_AI_FEATURES);
   return (row_count > 0);
}

void FXAI_FitYeoJohnsonStats(const double &values[],
                             const int count,
                             double &best_lambda,
                             double &best_mean,
                             double &best_std)
{
   best_lambda = 1.0;
   best_mean = 0.0;
   best_std = 1.0;
   if(count <= 0)
      return;

   double lambda_grid[11] = {-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0};
   double best_score = DBL_MAX;

   for(int li=0; li<ArraySize(lambda_grid); li++)
   {
      double lambda = lambda_grid[li];
      double sum = 0.0;
      double sum2 = 0.0;
      for(int i=0; i<count; i++)
      {
         double yt = FXAI_YeoJohnson(values[i], lambda);
         sum += yt;
         sum2 += yt * yt;
      }

      double mean = sum / (double)count;
      double var = (sum2 / (double)count) - (mean * mean);
      if(var < 1e-12) var = 1e-12;
      double std = MathSqrt(var);

      double m3 = 0.0;
      for(int i=0; i<count; i++)
      {
         double z = (FXAI_YeoJohnson(values[i], lambda) - mean) / std;
         m3 += z * z * z;
      }
      double skew = m3 / (double)count;
      double score = MathAbs(skew) + 0.025 * MathAbs(MathLog(std + 1e-6));
      if(score < best_score)
      {
         best_score = score;
         best_lambda = lambda;
         best_mean = mean;
         best_std = std;
      }
   }

   if(best_std < 1e-6)
      best_std = 1.0;
}

bool FXAI_FitFeatureNormalizationMethodForRange(const int method_id,
                                                const int horizon_minutes,
                                                const int fit_start,
                                                const int fit_end,
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
                                                const double &ctx_extra_arr[])
{
   ENUM_FXAI_FEATURE_NORMALIZATION method = FXAI_SanitizeNormMethod(method_id);
   if(!g_fxai_norm_fit_inited)
      FXAI_ResetFeatureNormalizationFits();

   int hslot = FXAI_NormalizationHorizonSlot(horizon_minutes);
   int method_idx = (int)method;
   FXAI_ClearNormalizationFitSlot(hslot, method_idx);
   if(!FXAI_MethodUsesFittedStats(method))
      return true;

   double rows[];
   int row_count = 0;
   if(!FXAI_BuildRawNormalizationMatrix(fit_start,
                                        fit_end,
                                        horizon_minutes,
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
                                        method,
                                        rows,
                                        row_count))
   {
      return false;
   }

   if(row_count < 8)
      return false;

   g_fxai_norm_fit_obs[hslot][method_idx] = row_count;

   double feature_vals[];
   ArrayResize(feature_vals, row_count);
   for(int f=0; f<FXAI_AI_FEATURES; f++)
   {
      double sum = 0.0;
      double sum2 = 0.0;
      double vmin = DBL_MAX;
      double vmax = -DBL_MAX;
      for(int r=0; r<row_count; r++)
      {
         double v = FXAI_RawNormalizationMatrixValue(rows, r, f);
         feature_vals[r] = v;
         sum += v;
         sum2 += v * v;
         if(v < vmin) vmin = v;
         if(v > vmax) vmax = v;
      }

      double mean = sum / (double)row_count;
      double var = (sum2 / (double)row_count) - (mean * mean);
      if(var < 1e-12) var = 1e-12;
      double std = MathSqrt(var);

      FXAI_SortSmall(feature_vals, row_count);
      double median = FXAI_SortedQuantile(feature_vals, row_count, 0.50);
      double q25 = FXAI_SortedQuantile(feature_vals, row_count, 0.25);
      double q75 = FXAI_SortedQuantile(feature_vals, row_count, 0.75);
      double iqr = q75 - q25;
      if(MathAbs(iqr) < 1e-9) iqr = 1.0;

      g_fxai_norm_fit_min[hslot][method_idx][f] = vmin;
      g_fxai_norm_fit_max[hslot][method_idx][f] = vmax;
      g_fxai_norm_fit_mean[hslot][method_idx][f] = mean;
      g_fxai_norm_fit_std[hslot][method_idx][f] = std;
      g_fxai_norm_fit_median[hslot][method_idx][f] = median;
      g_fxai_norm_fit_iqr[hslot][method_idx][f] = iqr;

      for(int k=0; k<FXAI_NORM_QUANTILE_KNOTS; k++)
      {
         double p = (FXAI_NORM_QUANTILE_KNOTS > 1 ? (double)k / (double)(FXAI_NORM_QUANTILE_KNOTS - 1) : 0.0);
         g_fxai_norm_fit_quantiles[hslot][method_idx][f][k] =
            FXAI_SortedQuantile(feature_vals, row_count, p);
      }

      double yj_lambda = 1.0;
      double yj_mean = mean;
      double yj_std = std;
      FXAI_FitYeoJohnsonStats(feature_vals, row_count, yj_lambda, yj_mean, yj_std);
      g_fxai_norm_fit_yeojohnson_lambda[hslot][method_idx][f] = yj_lambda;
      g_fxai_norm_fit_yeojohnson_mean[hslot][method_idx][f] = yj_mean;
      g_fxai_norm_fit_yeojohnson_std[hslot][method_idx][f] = MathMax(yj_std, 1e-6);
   }

   g_fxai_norm_fit_ready[hslot][method_idx] = true;
   return true;
}

bool FXAI_FeatureNormNeedsPrevious(const ENUM_FXAI_FEATURE_NORMALIZATION method)
{
   return (method == FXAI_NORM_CHANGE_PERCENT ||
           method == FXAI_NORM_BINARY_01 ||
           method == FXAI_NORM_LOG_RETURN ||
           method == FXAI_NORM_RELATIVE_CHANGE_PERCENT);
}

void FXAI_ApplyFeatureNormalizationEx(const ENUM_FXAI_FEATURE_NORMALIZATION method,
                                      const int horizon_minutes,
                                      const double &cur_features[],
                                      const double &prev_features[],
                                      const bool has_prev,
                                      const datetime sample_time,
                                      double &out_features[])
{
   if(!g_fxai_norm_window_inited)
      FXAI_ResetNormalizationWindows(FXAI_NORM_ROLL_WINDOW_DEFAULT);

   if(!g_fxai_norm_hist_inited)
      FXAI_ResetFeatureNormalizationState();
   if(!g_fxai_norm_fit_inited)
      FXAI_ResetFeatureNormalizationFits();

   int method_idx = (int)method;
   if(method_idx < 0) method_idx = 0;
   if(method_idx >= FXAI_NORM_METHOD_COUNT) method_idx = FXAI_NORM_METHOD_COUNT - 1;
   int hslot = FXAI_NormalizationHorizonSlot(horizon_minutes);

   bool use_full_hist = FXAI_MethodUsesRollingNormalizationHistory(method);
   if(use_full_hist)
   {
      bool rewind = (sample_time > 0 &&
                     g_fxai_norm_last_sample_time[method_idx] > 0 &&
                     sample_time <= g_fxai_norm_last_sample_time[method_idx]);
      bool cfg_changed = (g_fxai_norm_last_cfg_version[method_idx] != g_fxai_norm_window_cfg_version);
      if(rewind || cfg_changed)
      {
         for(int f=0; f<FXAI_AI_FEATURES; f++)
         {
            g_fxai_norm_hist_count[method_idx][f] = 0;
            g_fxai_norm_hist_head[method_idx][f] = 0;
         }
      }
      if(sample_time > 0)
         g_fxai_norm_last_sample_time[method_idx] = sample_time;
   }
   g_fxai_norm_last_cfg_version[method_idx] = g_fxai_norm_window_cfg_version;

   bool fit_ready_method = (FXAI_MethodUsesFittedStats(method) && g_fxai_norm_fit_ready[hslot][method_idx]);

   for(int f=0; f<FXAI_AI_FEATURES; f++)
   {
      double cur = cur_features[f];
      double prev = prev_features[f];
      if(!MathIsValidNumber(cur)) cur = 0.0;
      if(!MathIsValidNumber(prev)) prev = 0.0;

      double fit_min = 0.0;
      double fit_max = 0.0;
      double fit_mean = 0.0;
      double fit_std = 1.0;
      double fit_median = 0.0;
      double fit_iqr = 1.0;
      double fit_lambda = 1.0;
      double fit_yj_mean = 0.0;
      double fit_yj_std = 1.0;
      bool fit_ready = false;
      FXAI_GetFeatureNormalizationFitStats(method,
                                           horizon_minutes,
                                           f,
                                           fit_min,
                                           fit_max,
                                           fit_mean,
                                           fit_std,
                                           fit_median,
                                           fit_iqr,
                                           fit_lambda,
                                           fit_yj_mean,
                                           fit_yj_std,
                                           fit_ready);
      fit_ready = (fit_ready && fit_ready_method);

      int window_f = g_fxai_norm_default_window;
      if(f >= 0 && f < FXAI_AI_FEATURES && g_fxai_norm_feature_window[f] > 0)
         window_f = g_fxai_norm_feature_window[f];
      window_f = FXAI_NormalizationWindowClamp(window_f);

      int n_hist_total = g_fxai_norm_hist_count[method_idx][f];
      int n_hist = n_hist_total;
      if(n_hist > window_f) n_hist = window_f;

      double out_v = cur;
      switch(method)
      {
         case FXAI_NORM_MINMAX_BUFFER5:
         case FXAI_NORM_MINMAX_BUFFER2:
         case FXAI_NORM_MINMAX_BUFFER3:
         {
            double buffer_frac = 0.05;
            if(method == FXAI_NORM_MINMAX_BUFFER2) buffer_frac = 0.02;
            else if(method == FXAI_NORM_MINMAX_BUFFER3) buffer_frac = 0.03;

            if(fit_ready)
            {
               out_v = FXAI_MinMaxBufferMap(cur, fit_min, fit_max, buffer_frac);
            }
            else if(n_hist >= 2)
            {
               int idx0 = g_fxai_norm_hist_head[method_idx][f] - 1;
               if(idx0 < 0) idx0 += FXAI_NORM_ROLL_WINDOW_MAX;
               double vmin = g_fxai_norm_hist[method_idx][f][idx0];
               double vmax = g_fxai_norm_hist[method_idx][f][idx0];
               for(int k=1; k<n_hist; k++)
               {
                  int idx = g_fxai_norm_hist_head[method_idx][f] - 1 - k;
                  while(idx < 0) idx += FXAI_NORM_ROLL_WINDOW_MAX;
                  double v = g_fxai_norm_hist[method_idx][f][idx];
                  if(v < vmin) vmin = v;
                  if(v > vmax) vmax = v;
               }
               out_v = FXAI_MinMaxBufferMap(cur, vmin, vmax, buffer_frac);
            }
            else
            {
               out_v = FXAI_MinMaxBufferMap(cur, fit_min, fit_max, buffer_frac);
            }
            break;
         }

         case FXAI_NORM_CHANGE_PERCENT:
         {
            if(!has_prev)
               out_v = 0.0;
            else
            {
               double den = MathMax(MathAbs(prev), 1e-6);
               out_v = ((cur - prev) / den) * 100.0;
               out_v = FXAI_Clamp(out_v, -500.0, 500.0);
            }
            break;
         }

         case FXAI_NORM_BINARY_01:
            out_v = (has_prev && cur > prev ? FXAI_UNIT_RANGE_CEIL : FXAI_UNIT_RANGE_FLOOR);
            break;

         case FXAI_NORM_LOG_RETURN:
         {
            if(!has_prev)
               out_v = 0.0;
            else
            {
               out_v = FXAI_SignedLog1P(cur) - FXAI_SignedLog1P(prev);
               out_v = FXAI_Clamp(out_v, -8.0, 8.0);
            }
            break;
         }

         case FXAI_NORM_RELATIVE_CHANGE_PERCENT:
         {
            if(!has_prev)
               out_v = 0.0;
            else
            {
               double den = MathAbs(cur) + MathAbs(prev);
               if(den < 1e-6) out_v = 0.0;
               else out_v = (200.0 * (cur - prev)) / den;
               out_v = FXAI_Clamp(out_v, -200.0, 200.0);
            }
            break;
         }

         case FXAI_NORM_ZSCORE:
         {
            if(fit_ready)
            {
               out_v = (cur - fit_mean) / fit_std;
            }
            else if(n_hist >= 2)
            {
               double sum = 0.0;
               double sum2 = 0.0;
               for(int k=0; k<n_hist; k++)
               {
                  int idx = g_fxai_norm_hist_head[method_idx][f] - 1 - k;
                  while(idx < 0) idx += FXAI_NORM_ROLL_WINDOW_MAX;
                  double v = g_fxai_norm_hist[method_idx][f][idx];
                  sum += v;
                  sum2 += v * v;
               }
               double mean = sum / (double)n_hist;
               double var = (sum2 / (double)n_hist) - (mean * mean);
               if(var < 1e-12) var = 1e-12;
               out_v = (cur - mean) / MathSqrt(var);
            }
            else
            {
               out_v = (cur - fit_mean) / fit_std;
            }
            break;
         }

         case FXAI_NORM_ROBUST_MEDIAN_IQR:
         {
            if(fit_ready)
            {
               out_v = (cur - fit_median) / fit_iqr;
            }
            else if(n_hist >= 8)
            {
               double tmp[];
               ArrayResize(tmp, n_hist);
               for(int k=0; k<n_hist; k++)
               {
                  int idx = g_fxai_norm_hist_head[method_idx][f] - 1 - k;
                  while(idx < 0) idx += FXAI_NORM_ROLL_WINDOW_MAX;
                  tmp[k] = g_fxai_norm_hist[method_idx][f][idx];
               }
               FXAI_SortSmall(tmp, n_hist);
               double q50 = FXAI_SortedQuantile(tmp, n_hist, 0.50);
               double q25 = FXAI_SortedQuantile(tmp, n_hist, 0.25);
               double q75 = FXAI_SortedQuantile(tmp, n_hist, 0.75);
               double iqr = q75 - q25;
               if(MathAbs(iqr) < 1e-9) iqr = 1.0;
               out_v = (cur - q50) / iqr;
            }
            else
            {
               out_v = (cur - fit_median) / fit_iqr;
            }
            break;
         }

         case FXAI_NORM_QUANTILE_TO_NORMAL:
         {
            if(fit_ready)
            {
               out_v = FXAI_QuantileToNormalFromFit(hslot, method_idx, f, cur);
            }
            else if(n_hist >= 8)
            {
               double tmp[];
               ArrayResize(tmp, n_hist);
               for(int k=0; k<n_hist; k++)
               {
                  int idx = g_fxai_norm_hist_head[method_idx][f] - 1 - k;
                  while(idx < 0) idx += FXAI_NORM_ROLL_WINDOW_MAX;
                  tmp[k] = g_fxai_norm_hist[method_idx][f][idx];
               }
               FXAI_SortSmall(tmp, n_hist);

               int left = 0, right = n_hist;
               while(left < right)
               {
                  int mid = (left + right) / 2;
                  if(tmp[mid] < cur) left = mid + 1;
                  else right = mid;
               }
               int lower = left;

               left = 0;
               right = n_hist;
               while(left < right)
               {
                  int mid = (left + right) / 2;
                  if(tmp[mid] <= cur) left = mid + 1;
                  else right = mid;
               }
               int upper = left;

               double q = ((double)lower + 0.5 * (double)(upper - lower) + 0.5) /
                          ((double)n_hist + 1.0);
               out_v = FXAI_InvNormCDF(FXAI_Clamp(q, 1e-6, 1.0 - 1e-6));
               out_v = FXAI_Clamp(out_v, -6.0, 6.0);
            }
            else
            {
               out_v = FXAI_QuantileToNormalFromFit(hslot, method_idx, f, cur);
            }
            break;
         }

         case FXAI_NORM_POWER_YEOJOHNSON:
         {
            double yt = FXAI_YeoJohnson(cur, fit_lambda);
            out_v = (yt - fit_yj_mean) / fit_yj_std;
            break;
         }

         case FXAI_NORM_REVIN:
         case FXAI_NORM_DAIN:
            out_v = cur;
            break;

         case FXAI_NORM_CANDLE_GEOMETRY:
         case FXAI_NORM_VOL_STD_RETURNS:
         case FXAI_NORM_ATR_NATR_UNIT:
         case FXAI_NORM_EXISTING:
         default:
            out_v = cur;
            break;
      }

      if(!MathIsValidNumber(out_v)) out_v = 0.0;
      out_features[f] = out_v;

      if(use_full_hist)
      {
         int h = g_fxai_norm_hist_head[method_idx][f];
         g_fxai_norm_hist[method_idx][f][h] = cur;
         h++;
         if(h >= FXAI_NORM_ROLL_WINDOW_MAX) h = 0;
         g_fxai_norm_hist_head[method_idx][f] = h;
         if(n_hist_total < FXAI_NORM_ROLL_WINDOW_MAX)
            g_fxai_norm_hist_count[method_idx][f] = n_hist_total + 1;
      }
   }
}

void FXAI_ApplyFeatureNormalization(const ENUM_FXAI_FEATURE_NORMALIZATION method,
                                    const double &cur_features[],
                                    const double &prev_features[],
                                    const bool has_prev,
                                    const datetime sample_time,
                                    double &out_features[])
{
   FXAI_ApplyFeatureNormalizationEx(method,
                                    FXAI_ClampHorizon(PredictionTargetMinutes),
                                    cur_features,
                                    prev_features,
                                    has_prev,
                                    sample_time,
                                    out_features);
}

void FXAI_ComputePayloadFeatureWindowStats(const int feature_idx,
                                           const double &x_window[][FXAI_AI_WEIGHTS],
                                           const int window_size,
                                           const double &x[],
                                           double &local_mean,
                                           double &local_std,
                                           int &count)
{
   count = 0;
   double sum = 0.0;
   double sum2 = 0.0;

   double cur = FXAI_GetInputFeature(x, feature_idx);
   if(MathIsValidNumber(cur))
   {
      sum += cur;
      sum2 += cur * cur;
      count++;
   }

   for(int b=0; b<window_size && b<FXAI_MAX_SEQUENCE_BARS; b++)
   {
      int idx = feature_idx + 1;
      if(idx < 0 || idx >= FXAI_AI_WEIGHTS)
         continue;
      double v = x_window[b][idx];
      if(!MathIsValidNumber(v))
         continue;
      sum += v;
      sum2 += v * v;
      count++;
   }

   if(count <= 0)
   {
      local_mean = 0.0;
      local_std = 1.0;
      return;
   }

   local_mean = sum / (double)count;
   double var = (sum2 / (double)count) - (local_mean * local_mean);
   if(var < 1e-12) var = 1e-12;
   local_std = MathSqrt(var);
}

double FXAI_GetWindowInputFeature(const double &x_window[][FXAI_AI_WEIGHTS],
                                  const int row_idx,
                                  const int feature_idx)
{
   if(row_idx < 0 || row_idx >= FXAI_MAX_SEQUENCE_BARS)
      return 0.0;
   int idx = feature_idx + 1;
   if(idx < 0 || idx >= FXAI_AI_WEIGHTS)
      return 0.0;
   return x_window[row_idx][idx];
}

void FXAI_SetWindowInputFeature(double &x_window[][FXAI_AI_WEIGHTS],
                                const int row_idx,
                                const int feature_idx,
                                const double value)
{
   if(row_idx < 0 || row_idx >= FXAI_MAX_SEQUENCE_BARS)
      return;
   int idx = feature_idx + 1;
   if(idx < 0 || idx >= FXAI_AI_WEIGHTS)
      return;
   x_window[row_idx][idx] = value;
}

void FXAI_ApplyPayloadAdaptiveNormalization(const int method_id,
                                            const int horizon_minutes,
                                            double &x_window[][FXAI_AI_WEIGHTS],
                                            const int window_size,
                                            double &x[])
{
   ENUM_FXAI_FEATURE_NORMALIZATION method = FXAI_SanitizeNormMethod(method_id);
   if(!FXAI_MethodUsesAdaptivePayloadNormalization(method))
      return;

   int ws = window_size;
   if(ws < 0) ws = 0;
   if(ws > FXAI_MAX_SEQUENCE_BARS) ws = FXAI_MAX_SEQUENCE_BARS;
   int hslot = FXAI_NormalizationHorizonSlot(horizon_minutes);
   int method_idx = (int)method;
   bool fit_ready_method = (g_fxai_norm_fit_inited && g_fxai_norm_fit_ready[hslot][method_idx]);

   for(int f=0; f<FXAI_AI_FEATURES; f++)
   {
      double global_mean = 0.0;
      double global_std = 1.0;
      double local_mean = 0.0;
      double local_std = 1.0;
      int local_count = 0;
      FXAI_ComputePayloadFeatureWindowStats(f, x_window, ws, x, local_mean, local_std, local_count);

      double fit_min = 0.0;
      double fit_max = 0.0;
      double fit_median = 0.0;
      double fit_iqr = 1.0;
      double fit_lambda = 1.0;
      double fit_yj_mean = 0.0;
      double fit_yj_std = 1.0;
      bool fit_ready = false;
      FXAI_GetFeatureNormalizationFitStats(method,
                                           horizon_minutes,
                                           f,
                                           fit_min,
                                           fit_max,
                                           global_mean,
                                           global_std,
                                           fit_median,
                                           fit_iqr,
                                           fit_lambda,
                                           fit_yj_mean,
                                           fit_yj_std,
                                           fit_ready);
      fit_ready = (fit_ready && fit_ready_method);

      double center = global_mean;
      double scale = MathMax(global_std, 1e-6);
      if(method == FXAI_NORM_REVIN)
      {
         if(local_count >= 2)
         {
            center = local_mean;
            scale = MathMax(local_std, 1e-6);
         }
      }
      else
      {
         double mean_gap = MathAbs(local_mean - global_mean) / MathMax(global_std, 1e-6);
         double std_ratio = local_std / MathMax(global_std, 1e-6);
         double shift_mix = FXAI_Clamp(0.30 + 0.20 * mean_gap, 0.20, 0.85);
         double scale_mix = FXAI_Clamp(0.35 + 0.20 * MathAbs(MathLog(MathMax(std_ratio, 1e-6))), 0.25, 0.85);

         if(local_count >= 2)
         {
            center = (1.0 - shift_mix) * global_mean + shift_mix * local_mean;
            scale = (1.0 - scale_mix) * global_std + scale_mix * local_std;
         }

         double centered_probe = (FXAI_GetInputFeature(x, f) - center) / MathMax(scale, 1e-6);
         double gate = FXAI_Sigmoid(0.75 + 0.35 * MathAbs(centered_probe));
         scale = MathMax(scale / MathMax(gate, 0.25), 1e-6);
      }

      if(!fit_ready && local_count < 2)
      {
         center = global_mean;
         scale = MathMax(global_std, 1e-6);
      }

      FXAI_SetInputFeature(x, f, (FXAI_GetInputFeature(x, f) - center) / scale);
      for(int b=0; b<ws; b++)
      {
         double v = FXAI_GetWindowInputFeature(x_window, b, f);
         FXAI_SetWindowInputFeature(x_window, b, f, (v - center) / scale);
      }
   }
}

#endif // __FXAI_FEATURE_NORM_MQH__
