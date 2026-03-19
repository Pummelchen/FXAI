#ifndef __FXAI_DATA_IO_MQH__
#define __FXAI_DATA_IO_MQH__

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
   FXAIExecutionProfile exec_profile;
   FXAI_ResolveExecutionProfile(exec_profile);
   double profile_commission_points = FXAI_GetCommissionPointsRoundTripPerLot(symbol,
                                                                              exec_profile.commission_per_lot_side);
   if(profile_commission_points > snapshot.commission_points)
      snapshot.commission_points = profile_commission_points;

   double buffer = (buffer_points < 0.0 ? 0.0 : buffer_points);
   snapshot.min_move_points = FXAI_ExecutionEntryCostPoints(snapshot.spread_points,
                                                            snapshot.commission_points,
                                                            buffer,
                                                            exec_profile);
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

bool FXAI_ValidateM1SeriesBundle(const datetime &time_arr[],
                                 const double &open_arr[],
                                 const double &high_arr[],
                                 const double &low_arr[],
                                 const double &close_arr[],
                                 const int &spread_arr[],
                                 const int required_bars)
{
   int n = ArraySize(close_arr);
   if(required_bars > 0 && n < required_bars)
      return false;
   if(ArraySize(time_arr) != n || ArraySize(open_arr) != n || ArraySize(high_arr) != n ||
      ArraySize(low_arr) != n || ArraySize(spread_arr) != n)
      return false;

   int check_n = n;
   if(required_bars > 0 && check_n > required_bars)
      check_n = required_bars;
   for(int i=0; i<check_n; i++)
   {
      if(time_arr[i] <= 0)
         return false;
      if(!MathIsValidNumber(open_arr[i]) || !MathIsValidNumber(high_arr[i]) ||
         !MathIsValidNumber(low_arr[i]) || !MathIsValidNumber(close_arr[i]))
         return false;
      double hi = high_arr[i];
      double lo = low_arr[i];
      double mx = MathMax(open_arr[i], close_arr[i]);
      double mn = MathMin(open_arr[i], close_arr[i]);
      if(hi + 1e-10 < mx || lo - 1e-10 > mn || hi < lo)
         return false;
      if(spread_arr[i] < 0)
         return false;
   }
   return true;
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


#endif // __FXAI_DATA_IO_MQH__
