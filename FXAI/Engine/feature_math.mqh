#ifndef __FXAI_FEATURE_MATH_MQH__
#define __FXAI_FEATURE_MATH_MQH__

double FXAI_SafeReturn(const double &arr[],
                      const int idx_now,
                      const int idx_prev)
{
   int n = ArraySize(arr);
   if(n <= 0) return 0.0;
   if(idx_now < 0 || idx_prev < 0) return 0.0;
   if(idx_now >= n || idx_prev >= n) return 0.0;

   double a = arr[idx_now];
   double b = arr[idx_prev];
   if(b <= 0.0) return 0.0;

   return (a - b) / b;
}

double FXAI_NormalizedSlope(const double &arr[],
                           const int start_idx,
                           const int width)
{
   int n = ArraySize(arr);
   if(width < 2 || start_idx < 0 || start_idx + width > n) return 0.0;

   double sumx = 0.0;
   double sumx2 = 0.0;
   double sumy = 0.0;
   double sumxy = 0.0;

   for(int k=0; k<width; k++)
   {
      double x = (double)k;
      double y = arr[start_idx + k];
      sumx += x;
      sumx2 += x * x;
      sumy += y;
      sumxy += x * y;
   }

   double denom = ((double)width * sumx2 - sumx * sumx);
   if(denom == 0.0) return 0.0;

   double slope = (((double)width * sumxy) - (sumx * sumy)) / denom;
   double c = arr[start_idx];
   if(c == 0.0) return 0.0;

   // Arrays are stored as-series: start_idx is the most recent bar and larger
   // indices move backward in time. Reverse the regression sign so an upward
   // move into the present yields a positive slope.
   return (-slope) / c;
}

double FXAI_EstimateExpectedAbsMovePoints(const double &close_arr[],
                                         const int horizon_m1,
                                         const int sample_count,
                                         const double point)
{
   int n = ArraySize(close_arr);
   if(n <= horizon_m1 + 1 || point <= 0.0) return 0.0;

   int count = 0;
   double sum_abs = 0.0;

   int max_i = n - 1;
   int stop_i = horizon_m1 + sample_count;
   if(stop_i > max_i) stop_i = max_i;

   for(int i=horizon_m1; i<=stop_i; i++)
   {
      int f = i - horizon_m1;
      if(f < 0 || f >= n) continue;

      double mv = FXAI_MovePoints(close_arr[i], close_arr[f], point);
      sum_abs += MathAbs(mv);
      count++;
   }

   if(count <= 0) return 0.0;
   return sum_abs / (double)count;
}

double FXAI_EstimateExpectedAbsMovePointsAtIndex(const double &close_arr[],
                                                const int start_idx,
                                                const int horizon_m1,
                                                const int sample_count,
                                                const double point)
{
   int n = ArraySize(close_arr);
   if(n <= 0 || start_idx < 0 || start_idx >= n || horizon_m1 < 1 || point <= 0.0)
      return 0.0;

   int oldest_needed = start_idx + horizon_m1 + sample_count;
   if(oldest_needed >= n) oldest_needed = n - 1;

   int count = 0;
   double sum_abs = 0.0;
   for(int i=start_idx + horizon_m1; i<=oldest_needed; i++)
   {
      int f = i - horizon_m1;
      if(f < start_idx || f >= n) continue;
      double mv = FXAI_MovePoints(close_arr[i], close_arr[f], point);
      sum_abs += MathAbs(mv);
      count++;
   }

   if(count <= 0) return 0.0;
   return sum_abs / (double)count;
}

double FXAI_RollingAbsReturn(const double &arr[],
                            const int start_idx,
                            const int width)
{
   int n = ArraySize(arr);
   if(width < 2 || start_idx < 0 || start_idx >= n) return 0.0;

   int max_width = n - start_idx - 1;
   int w = width;
   if(w > max_width) w = max_width;
   if(w < 2) return 0.0;

   double sum_abs = 0.0;
   int count = 0;
   for(int k=0; k<w; k++)
   {
      double r = FXAI_SafeReturn(arr, start_idx + k, start_idx + k + 1);
      sum_abs += MathAbs(r);
      count++;
   }

   if(count <= 0) return 0.0;
   return sum_abs / (double)count;
}

double FXAI_RollingReturnStd(const double &arr[],
                            const int start_idx,
                            const int width)
{
   int n = ArraySize(arr);
   if(width < 2 || start_idx < 0 || start_idx >= n) return 0.0;

   int max_width = n - start_idx - 1;
   int w = width;
   if(w > max_width) w = max_width;
   if(w < 2) return 0.0;

   double sum = 0.0;
   double sum2 = 0.0;
   int count = 0;
   for(int k=0; k<w; k++)
   {
      double r = FXAI_SafeReturn(arr, start_idx + k, start_idx + k + 1);
      sum += r;
      sum2 += r * r;
      count++;
   }

   if(count < 2) return 0.0;
   double mean = sum / (double)count;
   double var = (sum2 / (double)count) - (mean * mean);
   if(var < 0.0) var = 0.0;
   return MathSqrt(var);
}

double FXAI_SMAAt(const double &arr[],
                 const int start_idx,
                 const int period)
{
   int n = ArraySize(arr);
   if(period <= 0 || start_idx < 0 || start_idx >= n) return 0.0;
   if(start_idx + period > n) return 0.0;

   double sum = 0.0;
   for(int k=0; k<period; k++)
      sum += arr[start_idx + k];
   return sum / (double)period;
}

double FXAI_EMAAt(const double &arr[],
                 const int start_idx,
                 const int period)
{
   int n = ArraySize(arr);
   if(period <= 1 || start_idx < 0 || start_idx >= n) return 0.0;
   if(start_idx + period > n) return 0.0;

   int oldest = start_idx + period - 1;
   double ema = arr[oldest];
   double alpha = 2.0 / ((double)period + 1.0);
   double one_minus_alpha = 1.0 - alpha;

   for(int j=oldest - 1; j>=start_idx; j--)
      ema = (alpha * arr[j]) + (one_minus_alpha * ema);

   return ema;
}

double FXAI_MAEdgeFeature(const double ref_price,
                         const double ma_value,
                         const double vol_unit)
{
   if(ref_price <= 0.0 || ma_value <= 0.0) return 0.0;
   if(vol_unit <= 0.0) return 0.0;
   return ((ref_price - ma_value) / ma_value) / vol_unit;
}

void FXAI_CandleGeometryNormalize(const double o,
                                  const double h,
                                  const double l,
                                  const double c,
                                  const double prev_close,
                                  const double eps,
                                  double &body_norm,
                                  double &upper_wick_norm,
                                  double &lower_wick_norm,
                                  double &range_norm)
{
   double e = (eps > 0.0 ? eps : 1e-8);
   double range = h - l;
   if(range < 0.0) range = -range;

   double den_close = MathMax(MathAbs(prev_close), e);
   double den_range = MathMax(range, e);

   double upper_wick = h - MathMax(o, c);
   double lower_wick = MathMin(o, c) - l;
   if(upper_wick < 0.0) upper_wick = 0.0;
   if(lower_wick < 0.0) lower_wick = 0.0;

   body_norm = (c - o) / den_close;
   upper_wick_norm = upper_wick / den_range;
   lower_wick_norm = lower_wick / den_range;
   range_norm = range / den_close;
}

double FXAI_QSDEMAAt(const double &arr[],
                    const int start_idx,
                    const int period)
{
   int n = ArraySize(arr);
   if(period <= 1 || start_idx < 0 || start_idx >= n) return 0.0;

   int warmup = period * 6;
   if(warmup < period + 20) warmup = period + 20;
   int oldest = start_idx + warmup - 1;
   if(oldest >= n) oldest = n - 1;
   if(oldest - start_idx + 1 < period) return 0.0;

   double alpha = 2.0 / ((double)period + 1.0);
   double one_minus_alpha = 1.0 - alpha;

   double ema1[4];
   double ema2[4];
   double dema[4];

   double seed = arr[oldest];
   for(int p=0; p<4; p++)
   {
      ema1[p] = seed;
      ema2[p] = seed;
      dema[p] = seed;
   }

   for(int j=oldest - 1; j>=start_idx; j--)
   {
      double val_in = arr[j];
      for(int p=0; p<4; p++)
      {
         ema1[p] = (alpha * val_in) + (one_minus_alpha * ema1[p]);
         ema2[p] = (alpha * ema1[p]) + (one_minus_alpha * ema2[p]);
         dema[p] = (2.0 * ema1[p]) - ema2[p];
         val_in = dema[p];
      }
   }

   return dema[3];
}

double FXAI_RSIAt(const double &arr[],
                 const int start_idx,
                 const int period)
{
   int n = ArraySize(arr);
   if(period <= 1 || start_idx < 0 || start_idx >= n) return 50.0;
   if(start_idx + period >= n) return 50.0;

   double gain = 0.0;
   double loss = 0.0;
   for(int k=period; k>=1; k--)
   {
      int older = start_idx + k;
      int newer = older - 1;
      double d = arr[newer] - arr[older];
      if(d > 0.0) gain += d;
      else        loss -= d;
   }

   double avg_gain = gain / (double)period;
   double avg_loss = loss / (double)period;
   if(avg_loss <= 1e-12 && avg_gain <= 1e-12) return 50.0;
   if(avg_loss <= 1e-12) return 100.0;

   double rs = avg_gain / avg_loss;
   return 100.0 - (100.0 / (1.0 + rs));
}

double FXAI_ATRAt(const double &high_arr[],
                 const double &low_arr[],
                 const double &close_arr[],
                 const int start_idx,
                 const int period)
{
   int nh = ArraySize(high_arr);
   int nl = ArraySize(low_arr);
   int nc = ArraySize(close_arr);
   if(period <= 1 || start_idx < 0) return 0.0;
   if(nh <= 0 || nl <= 0 || nc <= 0) return 0.0;
   if(nh != nl || nh != nc) return 0.0;
   if(start_idx + period >= nc) return 0.0; // +1 close needed

   double sum_tr = 0.0;
   int count = 0;
   int last = start_idx + period - 1;
   for(int j=start_idx; j<=last; j++)
   {
      double prev_close = close_arr[j + 1];
      double tr1 = high_arr[j] - low_arr[j];
      if(tr1 < 0.0) tr1 = -tr1;
      double tr2 = MathAbs(high_arr[j] - prev_close);
      double tr3 = MathAbs(low_arr[j] - prev_close);
      double tr = tr1;
      if(tr2 > tr) tr = tr2;
      if(tr3 > tr) tr = tr3;
      sum_tr += tr;
      count++;
   }

   if(count <= 0) return 0.0;
   return sum_tr / (double)count;
}

double FXAI_ParkinsonVolAt(const double &high_arr[],
                          const double &low_arr[],
                          const int start_idx,
                          const int period)
{
   int nh = ArraySize(high_arr);
   int nl = ArraySize(low_arr);
   if(period <= 1 || start_idx < 0 || nh <= 0 || nl <= 0) return 0.0;
   if(nh != nl) return 0.0;
   if(start_idx + period > nh) return 0.0;

   double sum_sq = 0.0;
   int count = 0;
   int last = start_idx + period - 1;
   for(int j=start_idx; j<=last; j++)
   {
      double h = high_arr[j];
      double l = low_arr[j];
      if(h <= 0.0 || l <= 0.0) continue;
      if(h < l)
      {
         double tmp = h;
         h = l;
         l = tmp;
      }
      if(h <= l) continue;

      double lr = MathLog(h / l);
      sum_sq += lr * lr;
      count++;
   }

   if(count <= 1) return 0.0;
   double denom = 4.0 * MathLog(2.0) * (double)count;
   if(denom <= 0.0) return 0.0;
   return MathSqrt(sum_sq / denom);
}

void FXAI_SortSmall(double &arr[], const int count)
{
   if(count <= 1) return;
   for(int i=1; i<count; i++)
   {
      double key = arr[i];
      int j = i - 1;
      while(j >= 0 && arr[j] > key)
      {
         arr[j + 1] = arr[j];
         j--;
      }
      arr[j + 1] = key;
   }
}

double FXAI_RollingMedianAt(const double &arr[],
                           const int start_idx,
                           const int period)
{
   int n = ArraySize(arr);
   if(period <= 1 || start_idx < 0 || start_idx >= n) return 0.0;
   if(start_idx + period > n) return 0.0;

   double tmp[];
   ArrayResize(tmp, period);
   for(int k=0; k<period; k++)
      tmp[k] = arr[start_idx + k];
   FXAI_SortSmall(tmp, period);

   if((period % 2) == 1) return tmp[period / 2];
   int m = period / 2;
   return 0.5 * (tmp[m - 1] + tmp[m]);
}

double FXAI_RollingMADAt(const double &arr[],
                        const int start_idx,
                        const int period,
                        const double median)
{
   int n = ArraySize(arr);
   if(period <= 1 || start_idx < 0 || start_idx >= n) return 0.0;
   if(start_idx + period > n) return 0.0;

   double dev[];
   ArrayResize(dev, period);
   for(int k=0; k<period; k++)
      dev[k] = MathAbs(arr[start_idx + k] - median);
   FXAI_SortSmall(dev, period);

   if((period % 2) == 1) return dev[period / 2];
   int m = period / 2;
   return 0.5 * (dev[m - 1] + dev[m]);
}

double FXAI_RogersSatchellVolAt(const double &open_arr[],
                               const double &high_arr[],
                               const double &low_arr[],
                               const double &close_arr[],
                               const int start_idx,
                               const int period)
{
   int n = ArraySize(close_arr);
   if(period <= 1 || start_idx < 0 || start_idx + period > n) return 0.0;
   if(ArraySize(open_arr) != n || ArraySize(high_arr) != n || ArraySize(low_arr) != n) return 0.0;

   double sum = 0.0;
   int count = 0;
   int last = start_idx + period - 1;
   for(int j=start_idx; j<=last; j++)
   {
      double o = open_arr[j];
      double h = high_arr[j];
      double l = low_arr[j];
      double c = close_arr[j];
      if(o <= 0.0 || h <= 0.0 || l <= 0.0 || c <= 0.0) continue;
      if(h < l)
      {
         double t = h;
         h = l;
         l = t;
      }
      if(h <= l) continue;

      double term = MathLog(h / c) * MathLog(h / o) + MathLog(l / c) * MathLog(l / o);
      if(term < 0.0) term = 0.0;
      sum += term;
      count++;
   }

   if(count <= 1) return 0.0;
   return MathSqrt(sum / (double)count);
}

double FXAI_GarmanKlassVolAt(const double &open_arr[],
                            const double &high_arr[],
                            const double &low_arr[],
                            const double &close_arr[],
                            const int start_idx,
                            const int period)
{
   int n = ArraySize(close_arr);
   if(period <= 1 || start_idx < 0 || start_idx + period > n) return 0.0;
   if(ArraySize(open_arr) != n || ArraySize(high_arr) != n || ArraySize(low_arr) != n) return 0.0;

   double cst = (2.0 * MathLog(2.0)) - 1.0;
   double sum = 0.0;
   int count = 0;
   int last = start_idx + period - 1;
   for(int j=start_idx; j<=last; j++)
   {
      double o = open_arr[j];
      double h = high_arr[j];
      double l = low_arr[j];
      double c = close_arr[j];
      if(o <= 0.0 || h <= 0.0 || l <= 0.0 || c <= 0.0) continue;
      if(h < l)
      {
         double t = h;
         h = l;
         l = t;
      }
      if(h <= l) continue;

      double hl = MathLog(h / l);
      double co = MathLog(c / o);
      double v = (0.5 * hl * hl) - (cst * co * co);
      if(v < 0.0) v = 0.0;
      sum += v;
      count++;
   }

   if(count <= 1) return 0.0;
   return MathSqrt(sum / (double)count);
}

double FXAI_KalmanEstimateAt(const double &arr[],
                            const int start_idx,
                            const int period)
{
   int n = ArraySize(arr);
   if(period <= 2 || start_idx < 0 || start_idx >= n) return 0.0;
   int oldest = start_idx + period - 1;
   if(oldest >= n) oldest = n - 1;
   if(oldest - start_idx + 1 < 3) return 0.0;

   double ret_var = 0.0;
   int ret_count = 0;
   for(int j=oldest; j>start_idx; j--)
   {
      double prev = arr[j];
      double cur = arr[j - 1];
      if(prev <= 0.0) continue;
      double r = (cur - prev) / prev;
      ret_var += r * r;
      ret_count++;
   }
   if(ret_count <= 0) ret_count = 1;
   ret_var /= (double)ret_count;
   if(ret_var < 1e-10) ret_var = 1e-10;

   double meas_var = ret_var * 4.0;
   double q = ret_var;
   double r = meas_var;

   double x = arr[oldest];
   double p = 1.0;
   for(int j=oldest - 1; j>=start_idx; j--)
   {
      p += q;
      double k = p / (p + r);
      x = x + k * (arr[j] - x);
      p = (1.0 - k) * p;
   }
   return x;
}

double FXAI_EhlersSuperSmootherAt(const double &arr[],
                                 const int start_idx,
                                 const int period)
{
   int n = ArraySize(arr);
   if(period <= 2 || start_idx < 0 || start_idx >= n) return 0.0;

   int warmup = period * 3;
   if(warmup < 12) warmup = 12;
   int oldest = start_idx + warmup - 1;
   if(oldest >= n) oldest = n - 1;
   if(oldest - start_idx + 1 < 3) return 0.0;

   double a1 = MathExp(-1.41421356237 * M_PI / (double)period);
   double b1 = 2.0 * a1 * MathCos(1.41421356237 * M_PI / (double)period);
   double c2 = b1;
   double c3 = -a1 * a1;
   double c1 = 1.0 - c2 - c3;

   double y2 = arr[oldest];
   double y1 = arr[oldest - 1];
   for(int j=oldest - 2; j>=start_idx; j--)
   {
      double x0 = arr[j];
      double x1 = arr[j + 1];
      double y = c1 * 0.5 * (x0 + x1) + c2 * y1 + c3 * y2;
      y2 = y1;
      y1 = y;
   }

   return y1;
}


#endif // __FXAI_FEATURE_MATH_MQH__
