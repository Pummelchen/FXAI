#ifndef __FX6_DATA_MQH__
#define __FX6_DATA_MQH__

#include "shared.mqh"

double FX6_GetCurrentSpreadPoints(const string symbol)
{
   long spread_raw = SymbolInfoInteger(symbol, SYMBOL_SPREAD);
   if(spread_raw > 0) return (double)spread_raw;

   double bid = SymbolInfoDouble(symbol, SYMBOL_BID);
   double ask = SymbolInfoDouble(symbol, SYMBOL_ASK);
   double point = SymbolInfoDouble(symbol, SYMBOL_POINT);

   if(point <= 0.0 || bid <= 0.0 || ask <= 0.0) return 0.0;
   return (ask - bid) / point;
}

double FX6_GetCommissionPointsRoundTripPerLot(const string symbol,
                                               const double commission_per_lot_side)
{
   if(commission_per_lot_side <= 0.0) return 0.0;

   double point = SymbolInfoDouble(symbol, SYMBOL_POINT);
   double tick_size = SymbolInfoDouble(symbol, SYMBOL_TRADE_TICK_SIZE);
   double tick_val = SymbolInfoDouble(symbol, SYMBOL_TRADE_TICK_VALUE);

   if(point <= 0.0 || tick_size <= 0.0 || tick_val <= 0.0) return 0.0;

   double money_per_point_per_lot = tick_val * (point / tick_size);
   if(money_per_point_per_lot <= 0.0) return 0.0;

   return (commission_per_lot_side * 2.0) / money_per_point_per_lot;
}

bool FX6_ExportDataSnapshot(const string symbol,
                            const double commission_per_lot_side,
                            const double buffer_points,
                            FX6DataSnapshot &snapshot)
{
   snapshot.symbol = symbol;
   snapshot.bar_time = iTime(symbol, PERIOD_M1, 1);
   snapshot.point = SymbolInfoDouble(symbol, SYMBOL_POINT);
   if(snapshot.bar_time == 0 || snapshot.point <= 0.0) return false;

   snapshot.spread_points = FX6_GetCurrentSpreadPoints(symbol);
   snapshot.commission_points = FX6_GetCommissionPointsRoundTripPerLot(symbol, commission_per_lot_side);

   double buffer = (buffer_points < 0.0 ? 0.0 : buffer_points);
   snapshot.min_move_points = snapshot.spread_points + snapshot.commission_points + buffer;
   if(snapshot.min_move_points < 0.0) snapshot.min_move_points = 0.0;

   return true;
}

bool FX6_IsInLiquidSession(const string symbol,
                           const datetime now,
                           const int min_after_open_minutes,
                           const int min_before_close_minutes)
{
   if(!SymbolSelect(symbol, true)) return false;

   MqlDateTime ts;
   TimeToStruct(now, ts);

   int sec_now = ts.hour * 3600 + ts.min * 60 + ts.sec;
   int pad_open = (min_after_open_minutes > 0 ? min_after_open_minutes : 0) * 60;
   int pad_close = (min_before_close_minutes > 0 ? min_before_close_minutes : 0) * 60;
   bool has_session = false;

   for(int session_idx=0; session_idx<10; session_idx++)
   {
      datetime from = 0;
      datetime to = 0;
      if(!SymbolInfoSessionTrade(symbol, (ENUM_DAY_OF_WEEK)ts.day_of_week, session_idx, from, to))
      {
         // Some symbols/brokers do not expose a session table.
         // Fail open to avoid accidentally disabling trading.
         if(session_idx == 0)
            return true;
         break;
      }

      has_session = true;

      int from_sec = (int)(from % 86400);
      int to_sec = (int)(to % 86400);

      if(from_sec == to_sec)
         continue;

      if(from_sec < to_sec)
      {
         int open_cut = from_sec + pad_open;
         int close_cut = to_sec - pad_close;
         if(open_cut <= close_cut && sec_now >= open_cut && sec_now <= close_cut)
            return true;
      }
      else
      {
         int w1_start = from_sec + pad_open;
         int w1_end = 86399;
         int w2_start = 0;
         int w2_end = to_sec - pad_close;

         bool in_w1 = (w1_start <= w1_end && sec_now >= w1_start && sec_now <= w1_end);
         bool in_w2 = (w2_start <= w2_end && sec_now >= w2_start && sec_now <= w2_end);
         if(in_w1 || in_w2)
            return true;
      }
   }

   if(!has_session) return true;
   return false;
}

bool FX6_LoadRatesOptional(const string symbol,
                           const ENUM_TIMEFRAMES tf,
                           const int needed,
                           MqlRates &rates_arr[])
{
   if(needed <= 0) return false;
   if(StringLen(symbol) <= 0) return false;
   if(!SymbolSelect(symbol, true)) return false;

   int cur = ArraySize(rates_arr);
   if(cur != needed)
      ArrayResize(rates_arr, needed);
   ArraySetAsSeries(rates_arr, true);

   int got = CopyRates(symbol, tf, 1, needed, rates_arr);
   if(got <= 0)
   {
      ArrayResize(rates_arr, 0);
      return false;
   }

   if(got < needed)
      ArrayResize(rates_arr, got);

   return true;
}

bool FX6_UpdateRatesRolling(const string symbol,
                            const ENUM_TIMEFRAMES tf,
                            const int needed,
                            datetime &last_bar_time,
                            MqlRates &rates_arr[])
{
   if(needed <= 0) return false;
   if(StringLen(symbol) <= 0) return false;
   if(!SymbolSelect(symbol, true)) return false;

   datetime cur_bar_time = iTime(symbol, tf, 1);
   if(cur_bar_time <= 0) return false;

   int cur = ArraySize(rates_arr);
   if(cur != needed)
   {
      ArrayResize(rates_arr, needed);
      ArraySetAsSeries(rates_arr, true);
      last_bar_time = 0;
   }
   else
   {
      ArraySetAsSeries(rates_arr, true);
   }

   if(last_bar_time == 0)
   {
      int got0 = CopyRates(symbol, tf, 1, needed, rates_arr);
      if(got0 < needed)
      {
         ArrayResize(rates_arr, 0);
         return false;
      }
      last_bar_time = cur_bar_time;
      return true;
   }

   if(cur_bar_time == last_bar_time)
      return true;

   int shift = iBarShift(symbol, tf, last_bar_time, true);
   if(shift < 1)
   {
      int got1 = CopyRates(symbol, tf, 1, needed, rates_arr);
      if(got1 < needed)
      {
         ArrayResize(rates_arr, 0);
         return false;
      }
      last_bar_time = cur_bar_time;
      return true;
   }

   int new_count = shift - 1;
   if(new_count <= 0)
   {
      last_bar_time = cur_bar_time;
      return true;
   }

   if(new_count >= needed)
   {
      int got1b = CopyRates(symbol, tf, 1, needed, rates_arr);
      if(got1b < needed)
      {
         ArrayResize(rates_arr, 0);
         return false;
      }
      last_bar_time = cur_bar_time;
      return true;
   }

   int copy_n = new_count;
   MqlRates fresh[];
   ArrayResize(fresh, copy_n);
   ArraySetAsSeries(fresh, true);

   int got_fresh = CopyRates(symbol, tf, 1, copy_n, fresh);
   if(got_fresh != copy_n)
   {
      int got2 = CopyRates(symbol, tf, 1, needed, rates_arr);
      if(got2 < needed)
      {
         ArrayResize(rates_arr, 0);
         return false;
      }
      last_bar_time = cur_bar_time;
      return true;
   }

   for(int i=needed - 1; i>=copy_n; i--)
      rates_arr[i] = rates_arr[i - new_count];
   for(int i=0; i<copy_n; i++)
      rates_arr[i] = fresh[i];

   last_bar_time = cur_bar_time;
   return true;
}

void FX6_ExtractRatesCloseTime(const MqlRates &rates_arr[],
                               double &close_arr[],
                               datetime &time_arr[])
{
   int n = ArraySize(rates_arr);
   int csz = ArraySize(close_arr);
   int tsz = ArraySize(time_arr);
   if(csz != n) ArrayResize(close_arr, n);
   if(tsz != n) ArrayResize(time_arr, n);
   ArraySetAsSeries(close_arr, true);
   ArraySetAsSeries(time_arr, true);

   for(int i=0; i<n; i++)
   {
      close_arr[i] = rates_arr[i].close;
      time_arr[i] = rates_arr[i].time;
   }
}

void FX6_ExtractRatesCloseTimeSpread(const MqlRates &rates_arr[],
                                     double &close_arr[],
                                     datetime &time_arr[],
                                     int &spread_arr[])
{
   int n = ArraySize(rates_arr);
   int csz = ArraySize(close_arr);
   int tsz = ArraySize(time_arr);
   int ssz = ArraySize(spread_arr);
   if(csz != n) ArrayResize(close_arr, n);
   if(tsz != n) ArrayResize(time_arr, n);
   if(ssz != n) ArrayResize(spread_arr, n);
   ArraySetAsSeries(close_arr, true);
   ArraySetAsSeries(time_arr, true);
   ArraySetAsSeries(spread_arr, true);

   for(int i=0; i<n; i++)
   {
      close_arr[i] = rates_arr[i].close;
      time_arr[i] = rates_arr[i].time;
      spread_arr[i] = (int)rates_arr[i].spread;
   }
}

bool FX6_LoadSeriesOptionalCached(const string symbol,
                                  const ENUM_TIMEFRAMES tf,
                                  const int needed,
                                  MqlRates &rates_arr[],
                                  double &close_arr[],
                                  datetime &time_arr[])
{
   if(!FX6_LoadRatesOptional(symbol, tf, needed, rates_arr))
   {
      ArrayResize(close_arr, 0);
      ArrayResize(time_arr, 0);
      return false;
   }

   FX6_ExtractRatesCloseTime(rates_arr, close_arr, time_arr);
   return (ArraySize(close_arr) > 0 && ArraySize(time_arr) > 0);
}

bool FX6_LoadSeriesWithSpread(const string symbol,
                              const int needed,
                              MqlRates &rates_arr[],
                              double &close_arr[],
                              datetime &time_arr[],
                              int &spread_arr[])
{
   if(!FX6_LoadRatesOptional(symbol, PERIOD_M1, needed, rates_arr))
   {
      ArrayResize(close_arr, 0);
      ArrayResize(time_arr, 0);
      ArrayResize(spread_arr, 0);
      return false;
   }

   FX6_ExtractRatesCloseTimeSpread(rates_arr, close_arr, time_arr, spread_arr);
   return (ArraySize(close_arr) >= needed &&
           ArraySize(time_arr) >= needed &&
           ArraySize(spread_arr) >= needed);
}

bool FX6_LoadSeriesOptional(const string symbol,
                            const ENUM_TIMEFRAMES tf,
                            const int needed,
                            double &close_arr[],
                            datetime &time_arr[])
{
   MqlRates rates_tmp[];
   return FX6_LoadSeriesOptionalCached(symbol, tf, needed, rates_tmp, close_arr, time_arr);
}

bool FX6_LoadSeries(const string symbol,
                    const int needed,
                    double &close_arr[],
                    datetime &time_arr[])
{
   if(!FX6_LoadSeriesOptional(symbol, PERIOD_M1, needed, close_arr, time_arr))
      return false;

   return (ArraySize(close_arr) >= needed && ArraySize(time_arr) >= needed);
}

bool FX6_LoadSpreadSeriesOptional(const string symbol,
                                  const ENUM_TIMEFRAMES tf,
                                  const int needed,
                                  int &spread_arr[])
{
   MqlRates rates_tmp[];
   if(!FX6_LoadRatesOptional(symbol, tf, needed, rates_tmp))
   {
      ArrayResize(spread_arr, 0);
      return false;
   }

   int got = ArraySize(rates_tmp);
   if(got <= 0)
   {
      ArrayResize(spread_arr, 0);
      return false;
   }

   int ssz = ArraySize(spread_arr);
   if(ssz != got) ArrayResize(spread_arr, got);
   ArraySetAsSeries(spread_arr, true);
   for(int i=0; i<got; i++)
      spread_arr[i] = (int)rates_tmp[i].spread;

   return true;
}

double FX6_GetSpreadAtIndex(const int i,
                            const int &spread_arr[],
                            const double fallback_spread_points)
{
   if(i >= 0 && i < ArraySize(spread_arr))
   {
      int spread_raw = spread_arr[i];
      if(spread_raw >= 0) return (double)spread_raw;
   }

   if(fallback_spread_points > 0.0) return fallback_spread_points;
   return 0.0;
}

double FX6_MovePoints(const double price_now,
                      const double price_future,
                      const double point)
{
   if(point <= 0.0) return 0.0;
   return (price_future - price_now) / point;
}

int FX6_BuildEVClassLabel(const double move_points,
                          const double roundtrip_cost_points,
                          const double ev_threshold_points)
{
   double ev_min = (ev_threshold_points < 0.0 ? 0.0 : ev_threshold_points);

   double buy_ev = move_points - roundtrip_cost_points;
   double sell_ev = -move_points - roundtrip_cost_points;

   if(buy_ev >= ev_min && buy_ev > sell_ev) return (int)FX6_LABEL_BUY;
   if(sell_ev >= ev_min && sell_ev > buy_ev) return (int)FX6_LABEL_SELL;
   return (int)FX6_LABEL_SKIP;
}

bool FX6_ClassToBinaryY(const int label_class,
                        int &y)
{
   if(label_class == (int)FX6_LABEL_BUY)
   {
      y = 1;
      return true;
   }

   if(label_class == (int)FX6_LABEL_SELL)
   {
      y = 0;
      return true;
   }

   y = -1;
   return false;
}

int FX6_FindAlignedIndex(const datetime &time_arr[],
                         const datetime ref_time,
                         const int max_lag_seconds)
{
   int n = ArraySize(time_arr);
   if(n <= 0) return -1;
   if(ref_time <= 0) return -1;

   int lo = 0;
   int hi = n - 1;
   int ans = -1;

   // descending timeseries: first index where bar_time <= ref_time
   while(lo <= hi)
   {
      int mid = (lo + hi) / 2;
      datetime t = time_arr[mid];
      if(t <= ref_time)
      {
         ans = mid;
         hi = mid - 1;
      }
      else
      {
         lo = mid + 1;
      }
   }

   if(ans < 0) return -1;
   if(max_lag_seconds <= 0) return ans;

   long lag = (long)(ref_time - time_arr[ans]);
   if(lag < 0 || lag > (long)max_lag_seconds)
      return -1;

   return ans;
}

void FX6_BuildAlignedIndexMap(const datetime &ref_time_arr[],
                              const datetime &target_time_arr[],
                              const int max_lag_seconds,
                              int &out_idx_arr[])
{
   int n_ref = ArraySize(ref_time_arr);
   FX6_BuildAlignedIndexMapRange(ref_time_arr, target_time_arr, max_lag_seconds, n_ref - 1, out_idx_arr);
}

void FX6_BuildAlignedIndexMapRange(const datetime &ref_time_arr[],
                                   const datetime &target_time_arr[],
                                   const int max_lag_seconds,
                                   const int upto_index,
                                   int &out_idx_arr[])
{
   int n_ref = ArraySize(ref_time_arr);
   int cur_sz = ArraySize(out_idx_arr);
   if(cur_sz != n_ref) ArrayResize(out_idx_arr, n_ref);
   ArraySetAsSeries(out_idx_arr, true);

   if(n_ref <= 0)
      return;

   int upto = upto_index;
   if(upto < 0) upto = 0;
   if(upto >= n_ref) upto = n_ref - 1;

   if(cur_sz != n_ref)
   {
      for(int i=0; i<n_ref; i++)
         out_idx_arr[i] = -1;
   }
   else
   {
      for(int i=0; i<=upto; i++)
         out_idx_arr[i] = -1;
   }

   int n_tgt = ArraySize(target_time_arr);
   if(n_tgt <= 0) return;

   int j = 0;
   for(int i=0; i<=upto; i++)
   {
      datetime t_ref = ref_time_arr[i];
      if(t_ref <= 0) continue;

      while(j < n_tgt && target_time_arr[j] > t_ref)
         j++;

      if(j >= n_tgt)
         break;

      datetime t_tgt = target_time_arr[j];
      long lag = (long)(t_ref - t_tgt);
      if(lag < 0) continue;
      if(max_lag_seconds > 0 && lag > (long)max_lag_seconds) continue;

      out_idx_arr[i] = j;
   }
}

double FX6_SafeReturn(const double &arr[],
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

double FX6_NormalizedSlope(const double &arr[],
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

   return slope / c;
}

double FX6_EstimateExpectedAbsMovePoints(const double &close_arr[],
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

      double mv = FX6_MovePoints(close_arr[i], close_arr[f], point);
      sum_abs += MathAbs(mv);
      count++;
   }

   if(count <= 0) return 0.0;
   return sum_abs / (double)count;
}

double FX6_RollingAbsReturn(const double &arr[],
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
      double r = FX6_SafeReturn(arr, start_idx + k, start_idx + k + 1);
      sum_abs += MathAbs(r);
      count++;
   }

   if(count <= 0) return 0.0;
   return sum_abs / (double)count;
}

bool FX6_ComputeFeatureVector(const int i,
                              const double spread_points,
                              const datetime &main_t1[],
                              const double &main_m1[],
                              const datetime &main_t5[],
                              const double &main_m5[],
                              const int &map_m5[],
                              const datetime &main_t15[],
                              const double &main_m15[],
                              const int &map_m15[],
                              const datetime &main_h1_t[],
                              const double &main_h1[],
                              const int &map_h1[],
                              const double ctx_ret_mean,
                              const double ctx_ret_std,
                              const double ctx_up_ratio,
                              double &features[])
{
   int n = ArraySize(main_m1);
   if(n < 40) return false;
   if(i < 0) return false;
   if(i + 10 >= n) return false;
   if(ArraySize(main_t1) != n) return false;
   if(i >= ArraySize(main_t1)) return false;

   datetime t_ref = main_t1[i];
   if(t_ref <= 0) return false;

   for(int f=0; f<FX6_AI_FEATURES; f++) features[f] = 0.0;

   double c = main_m1[i];
   double c1 = main_m1[i + 1];
   double c3 = main_m1[i + 3];
   double c5 = main_m1[i + 5];
   if(c1 <= 0.0 || c3 <= 0.0 || c5 <= 0.0) return false;

   // Use rolling return magnitude as lightweight regime normalization.
   double vol_unit = FX6_RollingAbsReturn(main_m1, i, 20);
   if(vol_unit < 1e-6) vol_unit = 1e-6;
   double spread_norm = 1.0 + (10000.0 * vol_unit);
   if(spread_norm < 1.0) spread_norm = 1.0;

   // M1 core features
   features[0] = ((c - c1) / c1) / vol_unit;
   features[1] = ((c - c3) / c3) / vol_unit;
   features[2] = ((c - c5) / c5) / vol_unit;
   features[3] = FX6_NormalizedSlope(main_m1, i, 10) / vol_unit;

   double sumy = 0.0;
   for(int k=0; k<10; k++) sumy += main_m1[i + k];
   double mean = sumy / 10.0;

   double var = 0.0;
   for(int k=0; k<10; k++)
   {
      double d = main_m1[i + k] - mean;
      var += d * d;
   }
   double std = MathSqrt(var / 10.0);
   features[4] = (std > 0.0 ? ((c - mean) / std) : 0.0);

   double rsum = 0.0;
   double rsum2 = 0.0;
   for(int k=0; k<10; k++)
   {
      double r = FX6_SafeReturn(main_m1, i + k, i + k + 1);
      rsum += r;
      rsum2 += r * r;
   }
   double rmean = rsum / 10.0;
   double rvar = (rsum2 / 10.0) - (rmean * rmean);
   features[5] = (rvar > 0.0 ? MathSqrt(rvar) / vol_unit : 0.0);

   features[6] = spread_points / spread_norm;

   int lag_m5 = 2 * PeriodSeconds(PERIOD_M5);
   int lag_m15 = 2 * PeriodSeconds(PERIOD_M15);
   int lag_h1 = 2 * PeriodSeconds(PERIOD_H1);
   if(lag_m5 <= 0) lag_m5 = 600;
   if(lag_m15 <= 0) lag_m15 = 1800;
   if(lag_h1 <= 0) lag_h1 = 7200;

   // Multi-timeframe trend/return context aligned by timestamp
   int i5 = -1;
   int i15 = -1;
   int i60 = -1;
   if(i >= 0 && i < ArraySize(map_m5)) i5 = map_m5[i];
   if(i >= 0 && i < ArraySize(map_m15)) i15 = map_m15[i];
   if(i >= 0 && i < ArraySize(map_h1)) i60 = map_h1[i];

   if(i5 < 0) i5 = FX6_FindAlignedIndex(main_t5, t_ref, lag_m5);
   if(i15 < 0) i15 = FX6_FindAlignedIndex(main_t15, t_ref, lag_m15);
   if(i60 < 0) i60 = FX6_FindAlignedIndex(main_h1_t, t_ref, lag_h1);

   features[7] = FX6_SafeReturn(main_m5, i5, i5 + 1) / vol_unit;
   features[8] = FX6_SafeReturn(main_m15, i15, i15 + 1) / vol_unit;
   features[9] = FX6_SafeReturn(main_h1, i60, i60 + 1) / vol_unit;

   // Cross-symbol context (dynamic list, pre-aggregated in caller)
   // [10] mean return, [11] return dispersion, [12] up-breadth in [-1, +1]
   features[10] = ctx_ret_mean / vol_unit;
   features[11] = ctx_ret_std / vol_unit;
   features[12] = FX6_Clamp((ctx_up_ratio - 0.5) * 2.0, -1.0, 1.0);

   // MTF slopes on aligned anchor bars
   features[13] = FX6_NormalizedSlope(main_m5, i5, 6) / vol_unit;
   features[14] = FX6_NormalizedSlope(main_h1, i60, 6) / vol_unit;

   for(int f=0; f<FX6_AI_FEATURES; f++)
   {
      double lo = -8.0;
      double hi = 8.0;
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
      features[f] = FX6_Clamp(features[f], lo, hi);
   }

   return true;
}

#endif // __FX6_DATA_MQH__
