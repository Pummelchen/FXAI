#ifndef __FXAI_FEATURE_NORM_MQH__
#define __FXAI_FEATURE_NORM_MQH__

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

double FXAI_MinMaxBuffer5Map(const double v, const double lo, const double hi)
{
   double span = hi - lo;
   if(span <= 1e-12) return 0.5;

   double lo_b = lo - (0.05 * span);
   double hi_b = hi + (0.05 * span);
   double den = hi_b - lo_b;
   if(den <= 1e-12) return 0.5;

   return FXAI_Clamp((v - lo_b) / den, 0.0, 1.0);
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

double FXAI_DAINTransform(const int f,
                          const double cur,
                          const double vec_mean,
                          const double vec_std)
{
   // Stateless DAIN-style transform to avoid time-order leakage in rolling
   // backtests and warmup folds.
   double global = (cur - vec_mean) / MathMax(vec_std, 1e-6);
   global = FXAI_Clamp(global, -8.0, 8.0);

   double lo = -8.0;
   double hi = 8.0;
   FXAI_GetFeatureClipBounds(f, lo, hi);
   double span = MathAbs(hi - lo);
   if(span < 1e-6) span = 1.0;

   double gain = FXAI_Clamp(8.0 / span, 0.25, 4.0);
   double gate = FXAI_Sigmoid(1.2 * (MathAbs(global) - 0.5));
   return gain * global * gate;
}

bool FXAI_FeatureNormNeedsPrevious(const ENUM_FXAI_FEATURE_NORMALIZATION method)
{
   return (method == FXAI_NORM_CHANGE_PERCENT ||
           method == FXAI_NORM_BINARY_01 ||
           method == FXAI_NORM_LOG_RETURN ||
           method == FXAI_NORM_RELATIVE_CHANGE_PERCENT);
}

void FXAI_ApplyFeatureNormalization(const ENUM_FXAI_FEATURE_NORMALIZATION method,
                                    const double &cur_features[],
                                    const double &prev_features[],
                                    const bool has_prev,
                                    const datetime sample_time,
                                    double &out_features[])
{
   static bool hist_inited = false;
   static datetime last_sample_time[FXAI_NORM_METHOD_COUNT];
   static int last_cfg_version[FXAI_NORM_METHOD_COUNT];
   static double hist[FXAI_NORM_METHOD_COUNT][FXAI_AI_FEATURES][FXAI_NORM_ROLL_WINDOW_MAX];
   static int hist_count[FXAI_NORM_METHOD_COUNT][FXAI_AI_FEATURES];
   static int hist_head[FXAI_NORM_METHOD_COUNT][FXAI_AI_FEATURES];

   if(!g_fxai_norm_window_inited)
      FXAI_ResetNormalizationWindows(FXAI_NORM_ROLL_WINDOW_DEFAULT);

   if(!hist_inited)
   {
      for(int m=0; m<FXAI_NORM_METHOD_COUNT; m++)
      {
         last_sample_time[m] = 0;
         last_cfg_version[m] = -1;
         for(int f=0; f<FXAI_AI_FEATURES; f++)
         {
            hist_count[m][f] = 0;
            hist_head[m][f] = 0;
            for(int k=0; k<FXAI_NORM_ROLL_WINDOW_MAX; k++)
               hist[m][f][k] = 0.0;
         }
      }
      hist_inited = true;
   }

   int method_idx = (int)method;
   if(method_idx < 0) method_idx = 0;
   if(method_idx >= FXAI_NORM_METHOD_COUNT) method_idx = FXAI_NORM_METHOD_COUNT - 1;

   bool use_full_hist = (method == FXAI_NORM_MINMAX_BUFFER5 ||
                         method == FXAI_NORM_ZSCORE ||
                         method == FXAI_NORM_ROBUST_MEDIAN_IQR ||
                         method == FXAI_NORM_QUANTILE_TO_NORMAL);

   if(use_full_hist)
   {
      bool rewind = (sample_time > 0 && last_sample_time[method_idx] > 0 && sample_time <= last_sample_time[method_idx]);
      bool cfg_changed = (last_cfg_version[method_idx] != g_fxai_norm_window_cfg_version);
      if(rewind || cfg_changed)
      {
         for(int f=0; f<FXAI_AI_FEATURES; f++)
         {
            hist_count[method_idx][f] = 0;
            hist_head[method_idx][f] = 0;
         }
      }
      if(sample_time > 0) last_sample_time[method_idx] = sample_time;
   }
   last_cfg_version[method_idx] = g_fxai_norm_window_cfg_version;

   double vec_mean = 0.0;
   double vec_std = 1.0;
   if(method == FXAI_NORM_REVIN || method == FXAI_NORM_DAIN)
      FXAI_VectorMeanStd(cur_features, vec_mean, vec_std);

   for(int f=0; f<FXAI_AI_FEATURES; f++)
   {
      double cur = cur_features[f];
      double prev = prev_features[f];
      if(!MathIsValidNumber(cur)) cur = 0.0;
      if(!MathIsValidNumber(prev)) prev = 0.0;

      double lo = -8.0;
      double hi = 8.0;
      FXAI_GetFeatureClipBounds(f, lo, hi);

      int window_f = g_fxai_norm_default_window;
      if(f >= 0 && f < FXAI_AI_FEATURES && g_fxai_norm_feature_window[f] > 0)
         window_f = g_fxai_norm_feature_window[f];
      window_f = FXAI_NormalizationWindowClamp(window_f);

      int n_hist_total = hist_count[method_idx][f];
      int n_hist = n_hist_total;
      if(n_hist > window_f) n_hist = window_f;

      double out_v = cur;
      switch(method)
      {
         case FXAI_NORM_MINMAX_BUFFER5:
         {
            if(n_hist >= 2)
            {
               int idx0 = hist_head[method_idx][f] - 1;
               if(idx0 < 0) idx0 += FXAI_NORM_ROLL_WINDOW_MAX;
               double vmin = hist[method_idx][f][idx0];
               double vmax = hist[method_idx][f][idx0];
               for(int k=1; k<n_hist; k++)
               {
                  int idx = hist_head[method_idx][f] - 1 - k;
                  while(idx < 0) idx += FXAI_NORM_ROLL_WINDOW_MAX;
                  double v = hist[method_idx][f][idx];
                  if(v < vmin) vmin = v;
                  if(v > vmax) vmax = v;
               }
               out_v = FXAI_MinMaxBuffer5Map(cur, vmin, vmax);
            }
            else
            {
               out_v = FXAI_MinMaxBuffer5Map(cur, lo, hi);
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
            out_v = (has_prev && cur > prev ? 1.0 : 0.0);
            break;

         case FXAI_NORM_LOG_RETURN:
         {
            if(!has_prev)
               out_v = 0.0;
            else
            {
               double cur01 = FXAI_MinMaxBuffer5Map(cur, lo, hi);
               double prev01 = FXAI_MinMaxBuffer5Map(prev, lo, hi);
               out_v = MathLog((cur01 + 1e-6) / (prev01 + 1e-6));
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
            if(n_hist >= 2)
            {
               double sum = 0.0;
               double sum2 = 0.0;
               for(int k=0; k<n_hist; k++)
               {
                  int idx = hist_head[method_idx][f] - 1 - k;
                  while(idx < 0) idx += FXAI_NORM_ROLL_WINDOW_MAX;
                  double v = hist[method_idx][f][idx];
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
               out_v = 0.0;
            }
            break;
         }

         case FXAI_NORM_ROBUST_MEDIAN_IQR:
         {
            if(n_hist >= 8)
            {
               double tmp[];
               ArrayResize(tmp, n_hist);
               for(int k=0; k<n_hist; k++)
               {
                  int idx = hist_head[method_idx][f] - 1 - k;
                  while(idx < 0) idx += FXAI_NORM_ROLL_WINDOW_MAX;
                  tmp[k] = hist[method_idx][f][idx];
               }
               FXAI_SortSmall(tmp, n_hist);

               double p50 = 0.50 * (double)(n_hist - 1);
               double p25 = 0.25 * (double)(n_hist - 1);
               double p75 = 0.75 * (double)(n_hist - 1);

               int i50a = (int)MathFloor(p50);
               int i25a = (int)MathFloor(p25);
               int i75a = (int)MathFloor(p75);
               int i50b = (i50a + 1 < n_hist ? i50a + 1 : i50a);
               int i25b = (i25a + 1 < n_hist ? i25a + 1 : i25a);
               int i75b = (i75a + 1 < n_hist ? i75a + 1 : i75a);

               double q50 = tmp[i50a] + (p50 - (double)i50a) * (tmp[i50b] - tmp[i50a]);
               double q25 = tmp[i25a] + (p25 - (double)i25a) * (tmp[i25b] - tmp[i25a]);
               double q75 = tmp[i75a] + (p75 - (double)i75a) * (tmp[i75b] - tmp[i75a]);
               double iqr = q75 - q25;
               if(MathAbs(iqr) < 1e-9) iqr = 1.0;
               out_v = (cur - q50) / iqr;
            }
            else
            {
               out_v = FXAI_RobustScaleFromBounds(cur, lo, hi);
            }
            break;
         }

         case FXAI_NORM_QUANTILE_TO_NORMAL:
         {
            if(n_hist >= 8)
            {
               double tmp[];
               ArrayResize(tmp, n_hist);
               for(int k=0; k<n_hist; k++)
               {
                  int idx = hist_head[method_idx][f] - 1 - k;
                  while(idx < 0) idx += FXAI_NORM_ROLL_WINDOW_MAX;
                  tmp[k] = hist[method_idx][f][idx];
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
               double q0 = FXAI_MinMaxBuffer5Map(cur, lo, hi);
               out_v = FXAI_InvNormCDF(FXAI_Clamp(q0, 1e-6, 1.0 - 1e-6));
               out_v = FXAI_Clamp(out_v, -6.0, 6.0);
            }
            break;
         }

         case FXAI_NORM_POWER_YEOJOHNSON:
         {
            double lambda = 0.25;
            double yt = FXAI_YeoJohnson(cur, lambda);
            double ylo = FXAI_YeoJohnson(lo, lambda);
            double yhi = FXAI_YeoJohnson(hi, lambda);
            double ys = yhi - ylo;
            double ymu = 0.5 * (yhi + ylo);
            double ystd = (MathAbs(ys) > 1e-9 ? MathAbs(ys) / 4.0 : 1.0);
            out_v = (yt - ymu) / ystd;
            break;
         }

         case FXAI_NORM_REVIN:
            out_v = (cur - vec_mean) / MathMax(vec_std, 1e-6);
            break;

         case FXAI_NORM_DAIN:
            out_v = FXAI_DAINTransform(f, cur, vec_mean, vec_std);
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
         int h = hist_head[method_idx][f];
         hist[method_idx][f][h] = cur;
         h++;
         if(h >= FXAI_NORM_ROLL_WINDOW_MAX) h = 0;
         hist_head[method_idx][f] = h;
         if(n_hist_total < FXAI_NORM_ROLL_WINDOW_MAX) hist_count[method_idx][f] = n_hist_total + 1;
      }
   }
}


#endif // __FXAI_FEATURE_NORM_MQH__
