#ifndef __FXAI_DATA_MQH__
#define __FXAI_DATA_MQH__

#include "shared.mqh"

#define FXAI_NORM_ROLL_WINDOW_DEFAULT 192
#define FXAI_NORM_ROLL_WINDOW_MAX 512

int g_fxai_norm_default_window = FXAI_NORM_ROLL_WINDOW_DEFAULT;
int g_fxai_norm_feature_window[FXAI_AI_FEATURES];
int g_fxai_norm_window_cfg_version = 0;
bool g_fxai_norm_window_inited = false;

int FXAI_NormalizationWindowClamp(const int w)
{
   int v = w;
   if(v < 16) v = 16;
   if(v > FXAI_NORM_ROLL_WINDOW_MAX) v = FXAI_NORM_ROLL_WINDOW_MAX;
   return v;
}

void FXAI_ResetNormalizationWindows(const int default_window = FXAI_NORM_ROLL_WINDOW_DEFAULT)
{
   g_fxai_norm_default_window = FXAI_NormalizationWindowClamp(default_window);
   for(int f=0; f<FXAI_AI_FEATURES; f++)
      g_fxai_norm_feature_window[f] = g_fxai_norm_default_window;
   g_fxai_norm_window_inited = true;
   g_fxai_norm_window_cfg_version++;
}

void FXAI_SetNormalizationWindows(const int &windows[], const int default_window = FXAI_NORM_ROLL_WINDOW_DEFAULT)
{
   int def_w = FXAI_NormalizationWindowClamp(default_window);
   if(!g_fxai_norm_window_inited)
      FXAI_ResetNormalizationWindows(def_w);
   else
      g_fxai_norm_default_window = def_w;

   int n = ArraySize(windows);
   for(int f=0; f<FXAI_AI_FEATURES; f++)
   {
      int w = def_w;
      if(f < n) w = FXAI_NormalizationWindowClamp(windows[f]);
      g_fxai_norm_feature_window[f] = w;
   }
   g_fxai_norm_window_cfg_version++;
}

void FXAI_GetNormalizationWindows(int &out_windows[], int &out_default_window)
{
   if(!g_fxai_norm_window_inited)
      FXAI_ResetNormalizationWindows(FXAI_NORM_ROLL_WINDOW_DEFAULT);

   if(ArraySize(out_windows) != FXAI_AI_FEATURES)
      ArrayResize(out_windows, FXAI_AI_FEATURES);
   for(int f=0; f<FXAI_AI_FEATURES; f++)
      out_windows[f] = g_fxai_norm_feature_window[f];
   out_default_window = g_fxai_norm_default_window;
}

double FXAI_GetCurrentSpreadPoints(const string symbol)
{
   long spread_raw = SymbolInfoInteger(symbol, SYMBOL_SPREAD);
   if(spread_raw > 0) return (double)spread_raw;

   double bid = SymbolInfoDouble(symbol, SYMBOL_BID);
   double ask = SymbolInfoDouble(symbol, SYMBOL_ASK);
   double point = SymbolInfoDouble(symbol, SYMBOL_POINT);

   if(point <= 0.0 || bid <= 0.0 || ask <= 0.0) return 0.0;
   return (ask - bid) / point;
}

double FXAI_GetCommissionPointsRoundTripPerLot(const string symbol,
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

bool FXAI_ExportDataSnapshot(const string symbol,
                            const double commission_per_lot_side,
                            const double buffer_points,
                            FXAIDataSnapshot &snapshot)
{
   snapshot.symbol = symbol;
   snapshot.bar_time = iTime(symbol, PERIOD_M1, 1);
   snapshot.point = SymbolInfoDouble(symbol, SYMBOL_POINT);
   if(snapshot.bar_time == 0 || snapshot.point <= 0.0) return false;

   snapshot.spread_points = FXAI_GetCurrentSpreadPoints(symbol);
   snapshot.commission_points = FXAI_GetCommissionPointsRoundTripPerLot(symbol, commission_per_lot_side);

   double buffer = (buffer_points < 0.0 ? 0.0 : buffer_points);
   snapshot.min_move_points = snapshot.spread_points + snapshot.commission_points + buffer;
   if(snapshot.min_move_points < 0.0) snapshot.min_move_points = 0.0;

   return true;
}

bool FXAI_IsInLiquidSession(const string symbol,
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

   // First check the current trading day.
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

   // Then check spillover from the previous trading day for sessions that cross midnight.
   int prev_dow = ts.day_of_week - 1;
   if(prev_dow < 0) prev_dow = 6;
   for(int session_idx=0; session_idx<10; session_idx++)
   {
      datetime from = 0;
      datetime to = 0;
      if(!SymbolInfoSessionTrade(symbol, (ENUM_DAY_OF_WEEK)prev_dow, session_idx, from, to))
         break;

      has_session = true;

      int from_sec = (int)(from % 86400);
      int to_sec = (int)(to % 86400);
      if(from_sec <= to_sec)
         continue;

      int close_cut = to_sec - pad_close;
      if(close_cut < 0) close_cut = -1;
      if(close_cut >= 0 && sec_now <= close_cut)
         return true;
   }

   if(!has_session) return true;
   return false;
}

bool FXAI_LoadRatesOptional(const string symbol,
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

bool FXAI_UpdateRatesRolling(const string symbol,
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

void FXAI_ExtractRatesCloseTime(const MqlRates &rates_arr[],
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

void FXAI_ExtractRatesCloseTimeSpread(const MqlRates &rates_arr[],
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

void FXAI_ExtractRatesOHLC(const MqlRates &rates_arr[],
                          double &open_arr[],
                          double &high_arr[],
                          double &low_arr[],
                          double &close_arr[])
{
   int n = ArraySize(rates_arr);
   if(ArraySize(open_arr) != n)  ArrayResize(open_arr, n);
   if(ArraySize(high_arr) != n)  ArrayResize(high_arr, n);
   if(ArraySize(low_arr) != n)   ArrayResize(low_arr, n);
   if(ArraySize(close_arr) != n) ArrayResize(close_arr, n);
   ArraySetAsSeries(open_arr, true);
   ArraySetAsSeries(high_arr, true);
   ArraySetAsSeries(low_arr, true);
   ArraySetAsSeries(close_arr, true);

   for(int i=0; i<n; i++)
   {
      open_arr[i] = rates_arr[i].open;
      high_arr[i] = rates_arr[i].high;
      low_arr[i] = rates_arr[i].low;
      close_arr[i] = rates_arr[i].close;
   }
}

bool FXAI_LoadSeriesOptionalCached(const string symbol,
                                  const ENUM_TIMEFRAMES tf,
                                  const int needed,
                                  MqlRates &rates_arr[],
                                  double &close_arr[],
                                  datetime &time_arr[])
{
   if(!FXAI_LoadRatesOptional(symbol, tf, needed, rates_arr))
   {
      ArrayResize(close_arr, 0);
      ArrayResize(time_arr, 0);
      return false;
   }

   FXAI_ExtractRatesCloseTime(rates_arr, close_arr, time_arr);
   return (ArraySize(close_arr) > 0 && ArraySize(time_arr) > 0);
}

bool FXAI_LoadSeriesWithSpread(const string symbol,
                              const int needed,
                              MqlRates &rates_arr[],
                              double &close_arr[],
                              datetime &time_arr[],
                              int &spread_arr[])
{
   if(!FXAI_LoadRatesOptional(symbol, PERIOD_M1, needed, rates_arr))
   {
      ArrayResize(close_arr, 0);
      ArrayResize(time_arr, 0);
      ArrayResize(spread_arr, 0);
      return false;
   }

   FXAI_ExtractRatesCloseTimeSpread(rates_arr, close_arr, time_arr, spread_arr);
   return (ArraySize(close_arr) >= needed &&
           ArraySize(time_arr) >= needed &&
           ArraySize(spread_arr) >= needed);
}

bool FXAI_LoadSeriesOptional(const string symbol,
                            const ENUM_TIMEFRAMES tf,
                            const int needed,
                            double &close_arr[],
                            datetime &time_arr[])
{
   MqlRates rates_tmp[];
   return FXAI_LoadSeriesOptionalCached(symbol, tf, needed, rates_tmp, close_arr, time_arr);
}

bool FXAI_LoadSeries(const string symbol,
                    const int needed,
                    double &close_arr[],
                    datetime &time_arr[])
{
   if(!FXAI_LoadSeriesOptional(symbol, PERIOD_M1, needed, close_arr, time_arr))
      return false;

   return (ArraySize(close_arr) >= needed && ArraySize(time_arr) >= needed);
}

double FXAI_GetSpreadAtIndex(const int i,
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

double FXAI_MovePoints(const double price_now,
                      const double price_future,
                      const double point)
{
   if(point <= 0.0) return 0.0;
   return (price_future - price_now) / point;
}

int FXAI_BuildEVClassLabel(const double move_points,
                          const double roundtrip_cost_points,
                          const double ev_threshold_points)
{
   double ev_min = (ev_threshold_points < 0.0 ? 0.0 : ev_threshold_points);

   double buy_ev = move_points - roundtrip_cost_points;
   double sell_ev = -move_points - roundtrip_cost_points;

   if(buy_ev >= ev_min && buy_ev > sell_ev) return (int)FXAI_LABEL_BUY;
   if(sell_ev >= ev_min && sell_ev > buy_ev) return (int)FXAI_LABEL_SELL;
   return (int)FXAI_LABEL_SKIP;
}

int FXAI_FindAlignedIndex(const datetime &time_arr[],
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

double FXAI_AlignedFreshnessWeight(const datetime &time_arr[],
                                  const int idx,
                                  const datetime ref_time,
                                  const int max_lag_seconds)
{
   int n = ArraySize(time_arr);
   if(n <= 0) return 0.0;
   if(idx < 0 || idx >= n) return 0.0;
   if(ref_time <= 0) return 0.0;
   if(max_lag_seconds <= 0) return 1.0;

   long lag = (long)(ref_time - time_arr[idx]);
   if(lag < 0 || lag > (long)max_lag_seconds) return 0.0;

   double w = 1.0 - ((double)lag / (double)max_lag_seconds);
   return FXAI_Clamp(w, 0.0, 1.0);
}

void FXAI_BuildAlignedIndexMap(const datetime &ref_time_arr[],
                              const datetime &target_time_arr[],
                              const int max_lag_seconds,
                              int &out_idx_arr[])
{
   int n_ref = ArraySize(ref_time_arr);
   FXAI_BuildAlignedIndexMapRange(ref_time_arr, target_time_arr, max_lag_seconds, n_ref - 1, out_idx_arr);
}

void FXAI_BuildAlignedIndexMapRange(const datetime &ref_time_arr[],
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

   // Always clear full map to prevent stale indices when caller changes
   // range size across calls.
   for(int i=0; i<n_ref; i++)
      out_idx_arr[i] = -1;

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

   return slope / c;
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
   static int last_method = -1;
   static datetime last_sample_time = 0;
   static int last_cfg_version = -1;
   static double hist[FXAI_AI_FEATURES][FXAI_NORM_ROLL_WINDOW_MAX];
   static int hist_count[FXAI_AI_FEATURES];
   static int hist_head[FXAI_AI_FEATURES];

   if(!g_fxai_norm_window_inited)
      FXAI_ResetNormalizationWindows(FXAI_NORM_ROLL_WINDOW_DEFAULT);

   if(!hist_inited)
   {
      for(int f=0; f<FXAI_AI_FEATURES; f++)
      {
         hist_count[f] = 0;
         hist_head[f] = 0;
         for(int k=0; k<FXAI_NORM_ROLL_WINDOW_MAX; k++)
            hist[f][k] = 0.0;
      }
      hist_inited = true;
   }

   bool use_full_hist = (method == FXAI_NORM_MINMAX_BUFFER5 ||
                         method == FXAI_NORM_ZSCORE ||
                         method == FXAI_NORM_ROBUST_MEDIAN_IQR ||
                         method == FXAI_NORM_QUANTILE_TO_NORMAL);

   if(use_full_hist)
   {
      bool rewind = (sample_time > 0 && last_sample_time > 0 && sample_time <= last_sample_time);
      bool method_changed = ((int)method != last_method);
      bool cfg_changed = (last_cfg_version != g_fxai_norm_window_cfg_version);
      if(rewind || method_changed || cfg_changed)
      {
         for(int f=0; f<FXAI_AI_FEATURES; f++)
         {
            hist_count[f] = 0;
            hist_head[f] = 0;
         }
      }
      if(sample_time > 0) last_sample_time = sample_time;
   }
   last_method = (int)method;
   last_cfg_version = g_fxai_norm_window_cfg_version;

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

      int n_hist_total = hist_count[f];
      int n_hist = n_hist_total;
      if(n_hist > window_f) n_hist = window_f;

      double out_v = cur;
      switch(method)
      {
         case FXAI_NORM_MINMAX_BUFFER5:
         {
            if(n_hist >= 2)
            {
               int idx0 = hist_head[f] - 1;
               if(idx0 < 0) idx0 += FXAI_NORM_ROLL_WINDOW_MAX;
               double vmin = hist[f][idx0];
               double vmax = hist[f][idx0];
               for(int k=1; k<n_hist; k++)
               {
                  int idx = hist_head[f] - 1 - k;
                  while(idx < 0) idx += FXAI_NORM_ROLL_WINDOW_MAX;
                  double v = hist[f][idx];
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
                  int idx = hist_head[f] - 1 - k;
                  while(idx < 0) idx += FXAI_NORM_ROLL_WINDOW_MAX;
                  double v = hist[f][idx];
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
                  int idx = hist_head[f] - 1 - k;
                  while(idx < 0) idx += FXAI_NORM_ROLL_WINDOW_MAX;
                  tmp[k] = hist[f][idx];
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
                  int idx = hist_head[f] - 1 - k;
                  while(idx < 0) idx += FXAI_NORM_ROLL_WINDOW_MAX;
                  tmp[k] = hist[f][idx];
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
         int h = hist_head[f];
         hist[f][h] = cur;
         h++;
         if(h >= FXAI_NORM_ROLL_WINDOW_MAX) h = 0;
         hist_head[f] = h;
         if(n_hist_total < FXAI_NORM_ROLL_WINDOW_MAX) hist_count[f] = n_hist_total + 1;
      }
   }
}

bool FXAI_ComputeFeatureVector(const int i,
                              const double spread_points,
                              const datetime &main_t1[],
                              const double &main_o1[],
                              const double &main_h1_ohlc[],
                              const double &main_l1[],
                              const double &main_m1[],
                              const datetime &main_t5[],
                              const double &main_m5[],
                              const int &map_m5[],
                              const datetime &main_t15[],
                              const double &main_m15[],
                              const int &map_m15[],
                              const datetime &main_t30[],
                              const double &main_m30[],
                              const int &map_m30[],
                              const datetime &main_h1_t[],
                              const double &main_h1[],
                              const int &map_h1[],
                              const double ctx_ret_mean,
                              const double ctx_ret_std,
                              const double ctx_up_ratio,
                              const double &ctx_extra_arr[],
                              const ENUM_FXAI_FEATURE_NORMALIZATION norm_method,
                              double &features[])
{
   int n = ArraySize(main_m1);
   if(n < 40) return false;
   if(i < 0) return false;
   if(i + 10 >= n) return false;
   if(ArraySize(main_t1) != n) return false;
   if(ArraySize(main_o1) != n) return false;
   if(ArraySize(main_h1_ohlc) != n) return false;
   if(ArraySize(main_l1) != n) return false;
   if(i >= ArraySize(main_t1)) return false;

   datetime t_ref = main_t1[i];
   if(t_ref <= 0) return false;

   for(int f=0; f<FXAI_AI_FEATURES; f++) features[f] = 0.0;

   double c = main_m1[i];
   double c1 = main_m1[i + 1];
   double c3 = main_m1[i + 3];
   double c5 = main_m1[i + 5];
   if(c1 <= 0.0 || c3 <= 0.0 || c5 <= 0.0) return false;

   // Default volatility unit: rolling absolute return (past-only for as-series arrays).
   double vol_unit = FXAI_RollingAbsReturn(main_m1, i, 20);
   if(norm_method == FXAI_NORM_VOL_STD_RETURNS)
   {
      double ru = FXAI_RollingReturnStd(main_m1, i, 20);
      if(ru > 1e-8) vol_unit = ru;
   }

   // ATR/NATR-based volatility-unit normalization option.
   double atr14 = FXAI_ATRAt(main_h1_ohlc, main_l1, main_m1, i, 14);
   if(norm_method == FXAI_NORM_ATR_NATR_UNIT && c > 0.0)
   {
      double atr_unit = atr14 / c;
      if(atr_unit > 1e-8) vol_unit = atr_unit;
   }
   if(vol_unit < 1e-6) vol_unit = 1e-6;
   double spread_norm = 1.0 + (10000.0 * vol_unit);
   if(spread_norm < 1.0) spread_norm = 1.0;

   // M1 core features
   features[0] = ((c - c1) / c1) / vol_unit;
   features[1] = ((c - c3) / c3) / vol_unit;
   features[2] = ((c - c5) / c5) / vol_unit;
   features[3] = FXAI_NormalizedSlope(main_m1, i, 10) / vol_unit;

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
      double r = FXAI_SafeReturn(main_m1, i + k, i + k + 1);
      rsum += r;
      rsum2 += r * r;
   }
   double rmean = rsum / 10.0;
   double rvar = (rsum2 / 10.0) - (rmean * rmean);
   features[5] = (rvar > 0.0 ? MathSqrt(rvar) / vol_unit : 0.0);

   features[6] = spread_points / spread_norm;

   int lag_m5 = 2 * PeriodSeconds(PERIOD_M5);
   int lag_m15 = 2 * PeriodSeconds(PERIOD_M15);
   int lag_m30 = 2 * PeriodSeconds(PERIOD_M30);
   int lag_h1 = 2 * PeriodSeconds(PERIOD_H1);
   if(lag_m5 <= 0) lag_m5 = 600;
   if(lag_m15 <= 0) lag_m15 = 1800;
   if(lag_m30 <= 0) lag_m30 = 3600;
   if(lag_h1 <= 0) lag_h1 = 7200;

   // Multi-timeframe trend/return context aligned by timestamp
   int i5 = -1;
   int i15 = -1;
   int i30 = -1;
   int i60 = -1;
   if(i >= 0 && i < ArraySize(map_m5)) i5 = map_m5[i];
   if(i >= 0 && i < ArraySize(map_m15)) i15 = map_m15[i];
   if(i >= 0 && i < ArraySize(map_m30)) i30 = map_m30[i];
   if(i >= 0 && i < ArraySize(map_h1)) i60 = map_h1[i];

   if(i5 < 0) i5 = FXAI_FindAlignedIndex(main_t5, t_ref, lag_m5);
   if(i15 < 0) i15 = FXAI_FindAlignedIndex(main_t15, t_ref, lag_m15);
   if(i30 < 0) i30 = FXAI_FindAlignedIndex(main_t30, t_ref, lag_m30);
   if(i60 < 0) i60 = FXAI_FindAlignedIndex(main_h1_t, t_ref, lag_h1);

   double w5 = FXAI_AlignedFreshnessWeight(main_t5, i5, t_ref, lag_m5);
   double w15 = FXAI_AlignedFreshnessWeight(main_t15, i15, t_ref, lag_m15);
   double w60 = FXAI_AlignedFreshnessWeight(main_h1_t, i60, t_ref, lag_h1);

   double miss_penalty = -0.25;
   double ret5 = FXAI_SafeReturn(main_m5, i5, i5 + 1) / vol_unit;
   double ret15 = FXAI_SafeReturn(main_m15, i15, i15 + 1) / vol_unit;
   double ret60 = FXAI_SafeReturn(main_h1, i60, i60 + 1) / vol_unit;

   features[7] = (w5 * ret5) + ((1.0 - w5) * miss_penalty);
   features[8] = (w15 * ret15) + ((1.0 - w15) * miss_penalty);
   features[9] = (w60 * ret60) + ((1.0 - w60) * miss_penalty);

   // Cross-symbol context (dynamic list, pre-aggregated in caller)
   // [10] mean return, [11] return dispersion, [12] up-breadth in [-1, +1]
   features[10] = ctx_ret_mean / vol_unit;
   features[11] = ctx_ret_std / vol_unit;
   features[12] = FXAI_Clamp((ctx_up_ratio - 0.5) * 2.0, -1.0, 1.0);

   // MTF slopes on aligned anchor bars
   double sl5 = FXAI_NormalizedSlope(main_m5, i5, 6) / vol_unit;
   double sl60 = FXAI_NormalizedSlope(main_h1, i60, 6) / vol_unit;
   features[13] = (w5 * sl5) + ((1.0 - w5) * miss_penalty);
   features[14] = (w60 * sl60) + ((1.0 - w60) * miss_penalty);

   MqlDateTime dt;
   TimeToStruct(t_ref, dt);
   int weekday = dt.day_of_week;
   if(weekday < 1) weekday = 1;
   if(weekday > 5) weekday = 5;
   int hh = dt.hour;
   int mm = dt.min;
   if(hh < 0) hh = 0;
   if(hh > 23) hh = 23;
   if(mm < 0) mm = 0;
   if(mm > 59) mm = 59;

   // Time features (normalized from requested MT5 ranges)
   features[15] = ((double)weekday - 3.0) / 2.0;
   features[16] = ((double)hh - 11.5) / 11.5;
   features[17] = ((double)mm - 29.5) / 29.5;

   // OHLC features from current M1 bar
   double o = main_o1[i];
   double h = main_h1_ohlc[i];
   double l = main_l1[i];
   if(norm_method == FXAI_NORM_CANDLE_GEOMETRY)
   {
      FXAI_CandleGeometryNormalize(o, h, l, c, c1, 1e-8,
                                   features[18], features[19], features[20], features[21]);
   }
   else
   {
      features[18] = FXAI_MAEdgeFeature(c, o, vol_unit);
      features[19] = FXAI_MAEdgeFeature(h, c, vol_unit);
      features[20] = FXAI_MAEdgeFeature(c, l, vol_unit);
      features[21] = (c > 0.0 ? ((h - l) / c) / vol_unit : 0.0);
   }

   // Multi-timeframe SMA distance features (100/200)
   double ma_m5_100 = FXAI_SMAAt(main_m5, i5, 100);
   double ma_m5_200 = FXAI_SMAAt(main_m5, i5, 200);
   double ma_m15_100 = FXAI_SMAAt(main_m15, i15, 100);
   double ma_m15_200 = FXAI_SMAAt(main_m15, i15, 200);
   double ma_m30_100 = FXAI_SMAAt(main_m30, i30, 100);
   double ma_m30_200 = FXAI_SMAAt(main_m30, i30, 200);
   double ma_h1_100 = FXAI_SMAAt(main_h1, i60, 100);
   double ma_h1_200 = FXAI_SMAAt(main_h1, i60, 200);

   features[22] = FXAI_MAEdgeFeature(c, ma_m5_100, vol_unit);
   features[23] = FXAI_MAEdgeFeature(c, ma_m5_200, vol_unit);
   features[24] = FXAI_MAEdgeFeature(c, ma_m15_100, vol_unit);
   features[25] = FXAI_MAEdgeFeature(c, ma_m15_200, vol_unit);
   features[26] = FXAI_MAEdgeFeature(c, ma_m30_100, vol_unit);
   features[27] = FXAI_MAEdgeFeature(c, ma_m30_200, vol_unit);
   features[28] = FXAI_MAEdgeFeature(c, ma_h1_100, vol_unit);
   features[29] = FXAI_MAEdgeFeature(c, ma_h1_200, vol_unit);

   // Multi-timeframe EMA distance features (100/200)
   double ema_m5_100 = FXAI_EMAAt(main_m5, i5, 100);
   double ema_m5_200 = FXAI_EMAAt(main_m5, i5, 200);
   double ema_m15_100 = FXAI_EMAAt(main_m15, i15, 100);
   double ema_m15_200 = FXAI_EMAAt(main_m15, i15, 200);
   double ema_m30_100 = FXAI_EMAAt(main_m30, i30, 100);
   double ema_m30_200 = FXAI_EMAAt(main_m30, i30, 200);
   double ema_h1_100 = FXAI_EMAAt(main_h1, i60, 100);
   double ema_h1_200 = FXAI_EMAAt(main_h1, i60, 200);

   features[30] = FXAI_MAEdgeFeature(c, ema_m5_100, vol_unit);
   features[31] = FXAI_MAEdgeFeature(c, ema_m5_200, vol_unit);
   features[32] = FXAI_MAEdgeFeature(c, ema_m15_100, vol_unit);
   features[33] = FXAI_MAEdgeFeature(c, ema_m15_200, vol_unit);
   features[34] = FXAI_MAEdgeFeature(c, ema_m30_100, vol_unit);
   features[35] = FXAI_MAEdgeFeature(c, ema_m30_200, vol_unit);
   features[36] = FXAI_MAEdgeFeature(c, ema_h1_100, vol_unit);
   features[37] = FXAI_MAEdgeFeature(c, ema_h1_200, vol_unit);

   // Additional volatility/momentum features
   double qsdema_100 = FXAI_QSDEMAAt(main_m1, i, 100);
   double qsdema_200 = FXAI_QSDEMAAt(main_m1, i, 200);
   features[38] = FXAI_MAEdgeFeature(c, qsdema_100, vol_unit);
   features[39] = FXAI_MAEdgeFeature(c, qsdema_200, vol_unit);

   double rsi14 = FXAI_RSIAt(main_m1, i, 14);
   features[40] = (rsi14 - 50.0) / 50.0;

   features[41] = (c > 0.0 ? ((atr14 / c) / vol_unit) : 0.0);

   double natr14 = (c > 0.0 ? (100.0 * atr14 / c) : 0.0);
   features[42] = natr14;

   double parkinson20 = FXAI_ParkinsonVolAt(main_h1_ohlc, main_l1, i, 20);
   features[43] = (vol_unit > 0.0 ? (parkinson20 / vol_unit) : 0.0);

   double rs20 = FXAI_RogersSatchellVolAt(main_o1, main_h1_ohlc, main_l1, main_m1, i, 20);
   features[44] = (vol_unit > 0.0 ? (rs20 / vol_unit) : 0.0);

   double gk20 = FXAI_GarmanKlassVolAt(main_o1, main_h1_ohlc, main_l1, main_m1, i, 20);
   features[45] = (vol_unit > 0.0 ? (gk20 / vol_unit) : 0.0);

   double med21 = FXAI_RollingMedianAt(main_m1, i, 21);
   features[46] = FXAI_MAEdgeFeature(c, med21, vol_unit);
   double mad21 = FXAI_RollingMADAt(main_m1, i, 21, med21);
   double hampel_denom = 1.4826 * mad21;
   if(hampel_denom < 1e-8) hampel_denom = 1e-8;
   features[47] = (c - med21) / hampel_denom;

   double kalman34 = FXAI_KalmanEstimateAt(main_m1, i, 34);
   features[48] = FXAI_MAEdgeFeature(c, kalman34, vol_unit);

   double ss20 = FXAI_EhlersSuperSmootherAt(main_m1, i, 20);
   features[49] = FXAI_MAEdgeFeature(c, ss20, vol_unit);

   // Detailed cross-symbol context: per-symbol aligned returns, lagged returns,
   // relative-strength residuals, and rolling correlation to the main symbol.
   for(int slot=0; slot<FXAI_CONTEXT_TOP_SYMBOLS; slot++)
   {
      int base_f = 50 + slot * 4;
      double ctx_ret = FXAI_GetContextExtraValue(ctx_extra_arr, i, slot * 4 + 0, 0.0);
      double ctx_lag = FXAI_GetContextExtraValue(ctx_extra_arr, i, slot * 4 + 1, 0.0);
      double ctx_rel = FXAI_GetContextExtraValue(ctx_extra_arr, i, slot * 4 + 2, 0.0);
      double ctx_corr = FXAI_GetContextExtraValue(ctx_extra_arr, i, slot * 4 + 3, 0.0);

      features[base_f + 0] = ctx_ret / vol_unit;
      features[base_f + 1] = ctx_lag / vol_unit;
      features[base_f + 2] = ctx_rel / vol_unit;
      features[base_f + 3] = ctx_corr;
   }

   for(int f=0; f<FXAI_AI_FEATURES; f++)
   {
      double lo = -8.0;
      double hi = 8.0;
      FXAI_GetFeatureClipBounds(f, lo, hi);
      features[f] = FXAI_Clamp(features[f], lo, hi);
   }

   return true;
}

#endif // __FXAI_DATA_MQH__
