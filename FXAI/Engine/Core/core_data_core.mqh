#ifndef __FXAI_CORE_DATA_CORE_MQH__
#define __FXAI_CORE_DATA_CORE_MQH__

void FXAI_DataCoreResetRequest(FXAIDataCoreRequest &request)
{
   request.live_mode = false;
   request.symbol = "";
   request.signal_bar = 0;
   request.needed = 0;
   request.align_upto = -1;
   request.commission_per_lot_side = 0.0;
   request.buffer_points = 0.0;
   request.context_symbol_count = 0;
   for(int i=0; i<FXAI_MAX_CONTEXT_SYMBOLS; i++)
      request.context_symbols[i] = "";
}

void FXAI_DataCoreInitRequest(FXAIDataCoreRequest &request,
                              const bool live_mode,
                              const string symbol,
                              const datetime signal_bar,
                              const int needed,
                              const int align_upto,
                              const double commission_per_lot_side,
                              const double buffer_points)
{
   FXAI_DataCoreResetRequest(request);
   request.live_mode = live_mode;
   request.symbol = symbol;
   request.signal_bar = signal_bar;
   request.needed = needed;
   request.align_upto = align_upto;
   request.commission_per_lot_side = commission_per_lot_side;
   request.buffer_points = buffer_points;
}

bool FXAI_DataCoreAddContextSymbol(FXAIDataCoreRequest &request,
                                   const string symbol)
{
   if(StringLen(symbol) <= 0)
      return false;

   for(int i=0; i<request.context_symbol_count; i++)
      if(request.context_symbols[i] == symbol)
         return false;

   if(request.context_symbol_count >= FXAI_MAX_CONTEXT_SYMBOLS)
      return false;

   request.context_symbols[request.context_symbol_count] = symbol;
   request.context_symbol_count++;
   return true;
}

void FXAI_DataCoreCaptureGlobalContextSymbols(FXAIDataCoreRequest &request)
{
#ifdef FXAI_DISABLE_DYNAMIC_CONTEXT_API
   request.context_symbol_count = 0;
#else
   for(int i=0; i<ArraySize(g_context_symbols); i++)
      FXAI_DataCoreAddContextSymbol(request, g_context_symbols[i]);
#endif
}

void FXAI_DataCoreResetContextSeries(FXAIContextSeries &series)
{
   series.loaded = false;
   series.symbol = "";
   series.last_bar_time = 0;
   ArrayResize(series.rates, 0);
   ArrayResize(series.open, 0);
   ArrayResize(series.high, 0);
   ArrayResize(series.low, 0);
   ArrayResize(series.close, 0);
   ArrayResize(series.time, 0);
   ArrayResize(series.spread, 0);
   ArrayResize(series.aligned_idx, 0);
}

void FXAI_DataCoreResetBundle(FXAIDataCoreBundle &bundle)
{
   bundle.ready = false;
   bundle.live_mode = false;
   bundle.symbol = "";
   bundle.signal_bar = 0;
   bundle.needed = 0;
   bundle.align_upto = -1;
   bundle.last_bar_m1 = 0;
   bundle.last_bar_m5 = 0;
   bundle.last_bar_m15 = 0;
   bundle.last_bar_m30 = 0;
   bundle.last_bar_h1 = 0;
   ArrayResize(bundle.rates_m1, 0);
   ArrayResize(bundle.rates_m5, 0);
   ArrayResize(bundle.rates_m15, 0);
   ArrayResize(bundle.rates_m30, 0);
   ArrayResize(bundle.rates_h1, 0);
   ArrayResize(bundle.open_arr, 0);
   ArrayResize(bundle.high_arr, 0);
   ArrayResize(bundle.low_arr, 0);
   ArrayResize(bundle.close_arr, 0);
   ArrayResize(bundle.time_arr, 0);
   ArrayResize(bundle.spread_m1, 0);
   ArrayResize(bundle.close_m5, 0);
   ArrayResize(bundle.time_m5, 0);
   ArrayResize(bundle.close_m15, 0);
   ArrayResize(bundle.time_m15, 0);
   ArrayResize(bundle.close_m30, 0);
   ArrayResize(bundle.time_m30, 0);
   ArrayResize(bundle.close_h1, 0);
   ArrayResize(bundle.time_h1, 0);
   ArrayResize(bundle.map_m5, 0);
   ArrayResize(bundle.map_m15, 0);
   ArrayResize(bundle.map_m30, 0);
   ArrayResize(bundle.map_h1, 0);
   int ctx_count = ArraySize(bundle.ctx_series);
   for(int i=0; i<ctx_count; i++)
      FXAI_DataCoreResetContextSeries(bundle.ctx_series[i]);
   ArrayResize(bundle.ctx_series, 0);
   ArrayResize(bundle.ctx_mean_arr, 0);
   ArrayResize(bundle.ctx_std_arr, 0);
   ArrayResize(bundle.ctx_up_arr, 0);
   ArrayResize(bundle.ctx_extra_arr, 0);
}

void FXAI_DataCoreClearContextAggregates(FXAIDataCoreBundle &bundle)
{
   int n = ArraySize(bundle.time_arr);
   if(n < 0)
      n = 0;

   ArrayResize(bundle.ctx_mean_arr, n);
   ArrayResize(bundle.ctx_std_arr, n);
   ArrayResize(bundle.ctx_up_arr, n);
   ArrayResize(bundle.ctx_extra_arr, n * FXAI_CONTEXT_EXTRA_FEATS);

   for(int i=0; i<n; i++)
   {
      bundle.ctx_mean_arr[i] = 0.0;
      bundle.ctx_std_arr[i] = 0.0;
      bundle.ctx_up_arr[i] = 0.5;
   }
   for(int i=0; i<ArraySize(bundle.ctx_extra_arr); i++)
      bundle.ctx_extra_arr[i] = 0.0;
}

void FXAI_DataCorePrepareTimeframeNeeds(const int needed,
                                        int &needed_m5,
                                        int &needed_m15,
                                        int &needed_m30,
                                        int &needed_h1)
{
   needed_m5 = (needed / 5) + 80;
   needed_m15 = (needed / 15) + 80;
   needed_m30 = (needed / 30) + 80;
   needed_h1 = (needed / 60) + 80;
   if(needed_m5 < 220) needed_m5 = 220;
   if(needed_m15 < 220) needed_m15 = 220;
   if(needed_m30 < 220) needed_m30 = 220;
   if(needed_h1 < 220) needed_h1 = 220;
}

void FXAI_DataCorePrepareTimeframeLags(int &lag_m5,
                                       int &lag_m15,
                                       int &lag_m30,
                                       int &lag_h1)
{
   lag_m5 = 2 * PeriodSeconds(PERIOD_M5);
   lag_m15 = 2 * PeriodSeconds(PERIOD_M15);
   lag_m30 = 2 * PeriodSeconds(PERIOD_M30);
   lag_h1 = 2 * PeriodSeconds(PERIOD_H1);
   if(lag_m5 <= 0) lag_m5 = 600;
   if(lag_m15 <= 0) lag_m15 = 1800;
   if(lag_m30 <= 0) lag_m30 = 3600;
   if(lag_h1 <= 0) lag_h1 = 7200;
}

void FXAI_DataCoreEnsureContextSlots(FXAIDataCoreBundle &bundle,
                                     const int ctx_count)
{
   int old_size = ArraySize(bundle.ctx_series);
   if(old_size != ctx_count)
   {
      ArrayResize(bundle.ctx_series, ctx_count);
      for(int i=old_size; i<ctx_count; i++)
         FXAI_DataCoreResetContextSeries(bundle.ctx_series[i]);
   }
}

bool FXAI_DataCoreBuildContextAggregates(FXAIDataCoreBundle &bundle,
                                         const int align_upto,
                                         string &reason)
{
#ifdef FXAI_DISABLE_DYNAMIC_CONTEXT_API
   FXAI_DataCoreEnsureContextSlots(bundle, 0);
   FXAI_DataCoreClearContextAggregates(bundle);
   reason = "";
   return true;
#else
   int ctx_count = ArraySize(bundle.ctx_series);
   if(ctx_count <= 0)
   {
      FXAI_DataCoreClearContextAggregates(bundle);
      reason = "";
      return true;
   }
   FXAI_PrecomputeDynamicContextAggregates(bundle.time_arr,
                                           bundle.close_arr,
                                           bundle.ctx_series,
                                           ctx_count,
                                           align_upto,
                                           bundle.ctx_mean_arr,
                                           bundle.ctx_std_arr,
                                           bundle.ctx_up_arr,
                                           bundle.ctx_extra_arr);
   reason = "";
   return true;
#endif
}

bool FXAI_DataCoreLoadHistoryHigherTimeframe(const string symbol,
                                             const ENUM_TIMEFRAMES tf,
                                             const int needed,
                                             MqlRates &rates[],
                                             double &close_arr[],
                                             datetime &time_arr[])
{
   return FXAI_LoadSeriesOptionalCached(symbol, tf, needed, rates, close_arr, time_arr);
}

bool FXAI_DataCoreRefreshLiveHigherTimeframe(const string symbol,
                                             const ENUM_TIMEFRAMES tf,
                                             const int needed,
                                             datetime &last_bar_time,
                                             MqlRates &rates[],
                                             double &close_arr[],
                                             datetime &time_arr[],
                                             int &map_arr[])
{
   if(FXAI_UpdateRatesRolling(symbol, tf, needed, last_bar_time, rates))
   {
      FXAI_ExtractRatesCloseTime(rates, close_arr, time_arr);
      return true;
   }

   ArrayResize(close_arr, 0);
   ArrayResize(time_arr, 0);
   ArrayResize(map_arr, 0);
   return false;
}

bool FXAI_DataCoreLoadHistoryContext(FXAIDataCoreBundle &bundle,
                                     const int needed,
                                     const FXAIDataCoreRequest &request)
{
#ifdef FXAI_DISABLE_DYNAMIC_CONTEXT_API
   FXAI_DataCoreEnsureContextSlots(bundle, 0);
   return true;
#else
   int ctx_count = request.context_symbol_count;
   FXAI_DataCoreEnsureContextSlots(bundle, ctx_count);
   MqlRates rates_ctx_tmp[];
   for(int s=0; s<ctx_count; s++)
   {
      FXAI_DataCoreResetContextSeries(bundle.ctx_series[s]);
      string ctx_symbol = request.context_symbols[s];
      bundle.ctx_series[s].symbol = ctx_symbol;
      bundle.ctx_series[s].loaded = FXAI_LoadRatesOptional(ctx_symbol,
                                                           PERIOD_M1,
                                                           needed,
                                                           rates_ctx_tmp);
      if(bundle.ctx_series[s].loaded)
      {
         FXAI_ExtractRatesCloseTimeSpread(rates_ctx_tmp,
                                          bundle.ctx_series[s].close,
                                          bundle.ctx_series[s].time,
                                          bundle.ctx_series[s].spread);
         FXAI_ExtractRatesOHLC(rates_ctx_tmp,
                               bundle.ctx_series[s].open,
                               bundle.ctx_series[s].high,
                               bundle.ctx_series[s].low,
                               bundle.ctx_series[s].close);
         bundle.ctx_series[s].loaded = FXAI_ValidateM1SeriesBundle(bundle.ctx_series[s].time,
                                                                   bundle.ctx_series[s].open,
                                                                   bundle.ctx_series[s].high,
                                                                   bundle.ctx_series[s].low,
                                                                   bundle.ctx_series[s].close,
                                                                   bundle.ctx_series[s].spread,
                                                                   needed);
      }
      if(!bundle.ctx_series[s].loaded)
         FXAI_DataCoreResetContextSeries(bundle.ctx_series[s]);
      else
         bundle.ctx_series[s].symbol = ctx_symbol;
   }
   return true;
#endif
}

bool FXAI_DataCoreRefreshLiveContext(FXAIDataCoreBundle &bundle,
                                     const int needed,
                                     const FXAIDataCoreRequest &request)
{
#ifdef FXAI_DISABLE_DYNAMIC_CONTEXT_API
   FXAI_DataCoreEnsureContextSlots(bundle, 0);
   return true;
#else
   int ctx_count = request.context_symbol_count;
   FXAI_DataCoreEnsureContextSlots(bundle, ctx_count);
   for(int s=0; s<ctx_count; s++)
   {
      string ctx_symbol = request.context_symbols[s];
      if(bundle.ctx_series[s].symbol != ctx_symbol)
      {
         FXAI_DataCoreResetContextSeries(bundle.ctx_series[s]);
         bundle.ctx_series[s].symbol = ctx_symbol;
      }

      bundle.ctx_series[s].loaded = FXAI_UpdateRatesRolling(ctx_symbol,
                                                            PERIOD_M1,
                                                            needed,
                                                            bundle.ctx_series[s].last_bar_time,
                                                            bundle.ctx_series[s].rates);
      if(bundle.ctx_series[s].loaded)
      {
         FXAI_ExtractRatesCloseTimeSpread(bundle.ctx_series[s].rates,
                                          bundle.ctx_series[s].close,
                                          bundle.ctx_series[s].time,
                                          bundle.ctx_series[s].spread);
         FXAI_ExtractRatesOHLC(bundle.ctx_series[s].rates,
                               bundle.ctx_series[s].open,
                               bundle.ctx_series[s].high,
                               bundle.ctx_series[s].low,
                               bundle.ctx_series[s].close);
         if(!FXAI_ValidateM1SeriesBundle(bundle.ctx_series[s].time,
                                         bundle.ctx_series[s].open,
                                         bundle.ctx_series[s].high,
                                         bundle.ctx_series[s].low,
                                         bundle.ctx_series[s].close,
                                         bundle.ctx_series[s].spread,
                                         needed))
            bundle.ctx_series[s].loaded = false;
      }
      if(!bundle.ctx_series[s].loaded)
      {
         string keep_symbol = bundle.ctx_series[s].symbol;
         FXAI_DataCoreResetContextSeries(bundle.ctx_series[s]);
         bundle.ctx_series[s].symbol = keep_symbol;
      }
   }
   return true;
#endif
}

bool FXAI_DataCoreLoadBundleFromRequest(const FXAIDataCoreRequest &request,
                                        FXAIDataCoreBundle &bundle,
                                        string &reason)
{
   if(bundle.symbol != request.symbol || bundle.live_mode != request.live_mode)
      FXAI_DataCoreResetBundle(bundle);

   bundle.live_mode = request.live_mode;
   bundle.symbol = request.symbol;
   bundle.signal_bar = request.signal_bar;
   bundle.needed = request.needed;
   bundle.align_upto = request.align_upto;

   if(!FXAI_ExportDataSnapshot(request.symbol,
                               request.commission_per_lot_side,
                               request.buffer_points,
                               bundle.snapshot))
   {
      reason = "snapshot_export_failed";
      return false;
   }
   if(request.live_mode && request.signal_bar > 0)
      bundle.snapshot.bar_time = request.signal_bar;
   bundle.signal_bar = bundle.snapshot.bar_time;

   if(request.live_mode)
   {
      if(!FXAI_UpdateRatesRolling(request.symbol,
                                  PERIOD_M1,
                                  request.needed,
                                  bundle.last_bar_m1,
                                  bundle.rates_m1))
      {
         reason = "m1_series_load_failed";
         return false;
      }
      FXAI_ExtractRatesCloseTimeSpread(bundle.rates_m1,
                                       bundle.close_arr,
                                       bundle.time_arr,
                                       bundle.spread_m1);
   }
   else
   {
      if(!FXAI_LoadSeriesWithSpread(request.symbol,
                                    request.needed,
                                    bundle.rates_m1,
                                    bundle.close_arr,
                                    bundle.time_arr,
                                    bundle.spread_m1))
      {
         reason = "m1_series_load_failed";
         return false;
      }
   }

   FXAI_ExtractRatesOHLC(bundle.rates_m1,
                         bundle.open_arr,
                         bundle.high_arr,
                         bundle.low_arr,
                         bundle.close_arr);
   if(ArraySize(bundle.close_arr) < request.needed ||
      ArraySize(bundle.time_arr) < request.needed ||
      ArraySize(bundle.spread_m1) < request.needed)
   {
      reason = "m1_series_size_failed";
      return false;
   }
   if(!FXAI_ValidateM1SeriesBundle(bundle.time_arr,
                                   bundle.open_arr,
                                   bundle.high_arr,
                                   bundle.low_arr,
                                   bundle.close_arr,
                                   bundle.spread_m1,
                                   request.needed))
   {
      reason = "m1_series_integrity_failed";
      return false;
   }

   int needed_m5 = 0;
   int needed_m15 = 0;
   int needed_m30 = 0;
   int needed_h1 = 0;
   FXAI_DataCorePrepareTimeframeNeeds(request.needed, needed_m5, needed_m15, needed_m30, needed_h1);

   if(request.live_mode)
   {
      FXAI_DataCoreRefreshLiveHigherTimeframe(request.symbol, PERIOD_M5, needed_m5, bundle.last_bar_m5, bundle.rates_m5, bundle.close_m5, bundle.time_m5, bundle.map_m5);
      FXAI_DataCoreRefreshLiveHigherTimeframe(request.symbol, PERIOD_M15, needed_m15, bundle.last_bar_m15, bundle.rates_m15, bundle.close_m15, bundle.time_m15, bundle.map_m15);
      FXAI_DataCoreRefreshLiveHigherTimeframe(request.symbol, PERIOD_M30, needed_m30, bundle.last_bar_m30, bundle.rates_m30, bundle.close_m30, bundle.time_m30, bundle.map_m30);
      FXAI_DataCoreRefreshLiveHigherTimeframe(request.symbol, PERIOD_H1, needed_h1, bundle.last_bar_h1, bundle.rates_h1, bundle.close_h1, bundle.time_h1, bundle.map_h1);
   }
   else
   {
      FXAI_DataCoreLoadHistoryHigherTimeframe(request.symbol, PERIOD_M5, needed_m5, bundle.rates_m5, bundle.close_m5, bundle.time_m5);
      FXAI_DataCoreLoadHistoryHigherTimeframe(request.symbol, PERIOD_M15, needed_m15, bundle.rates_m15, bundle.close_m15, bundle.time_m15);
      FXAI_DataCoreLoadHistoryHigherTimeframe(request.symbol, PERIOD_M30, needed_m30, bundle.rates_m30, bundle.close_m30, bundle.time_m30);
      FXAI_DataCoreLoadHistoryHigherTimeframe(request.symbol, PERIOD_H1, needed_h1, bundle.rates_h1, bundle.close_h1, bundle.time_h1);
   }

   int lag_m5 = 0;
   int lag_m15 = 0;
   int lag_m30 = 0;
   int lag_h1 = 0;
   FXAI_DataCorePrepareTimeframeLags(lag_m5, lag_m15, lag_m30, lag_h1);

   FXAI_BuildAlignedIndexMapRange(bundle.time_arr, bundle.time_m5, lag_m5, request.align_upto, bundle.map_m5);
   FXAI_BuildAlignedIndexMapRange(bundle.time_arr, bundle.time_m15, lag_m15, request.align_upto, bundle.map_m15);
   FXAI_BuildAlignedIndexMapRange(bundle.time_arr, bundle.time_m30, lag_m30, request.align_upto, bundle.map_m30);
   FXAI_BuildAlignedIndexMapRange(bundle.time_arr, bundle.time_h1, lag_h1, request.align_upto, bundle.map_h1);

   if(request.live_mode)
   {
      if(!FXAI_DataCoreRefreshLiveContext(bundle, request.needed, request))
      {
         reason = "context_load_failed";
         return false;
      }
   }
   else
   {
      if(!FXAI_DataCoreLoadHistoryContext(bundle, request.needed, request))
      {
         reason = "context_load_failed";
         return false;
      }
   }
   if(!FXAI_DataCoreBuildContextAggregates(bundle, request.align_upto, reason))
      return false;

   bundle.ready = true;
   reason = "";
   return true;
}

bool FXAI_DataCoreLoadHistoryBundle(const string symbol,
                                    const int needed,
                                    const int align_upto,
                                    const double commission_per_lot_side,
                                    const double buffer_points,
                                    FXAIDataCoreBundle &bundle,
                                    string &reason)
{
   FXAIDataCoreRequest request;
   FXAI_DataCoreInitRequest(request,
                            false,
                            symbol,
                            0,
                            needed,
                            align_upto,
                            commission_per_lot_side,
                            buffer_points);
   FXAI_DataCoreCaptureGlobalContextSymbols(request);
   return FXAI_DataCoreLoadBundleFromRequest(request, bundle, reason);
}

bool FXAI_DataCoreRefreshLiveBundle(FXAIDataCoreBundle &bundle,
                                    const string symbol,
                                    const datetime signal_bar,
                                    const int needed,
                                    const int align_upto,
                                    const double commission_per_lot_side,
                                    const double buffer_points,
                                    string &reason)
{
   FXAIDataCoreRequest request;
   FXAI_DataCoreInitRequest(request,
                            true,
                            symbol,
                            signal_bar,
                            needed,
                            align_upto,
                            commission_per_lot_side,
                            buffer_points);
   FXAI_DataCoreCaptureGlobalContextSymbols(request);
   return FXAI_DataCoreLoadBundleFromRequest(request, bundle, reason);
}

void FXAI_DataCoreBindArrayBundle(const FXAIDataSnapshot &snapshot,
                                  const int needed,
                                  const int align_upto,
                                  const double &open_arr[],
                                  const double &high_arr[],
                                  const double &low_arr[],
                                  const double &close_arr[],
                                  const datetime &time_arr[],
                                  const int &spread_m1[],
                                  const double &close_m5[],
                                  const datetime &time_m5[],
                                  const int &map_m5[],
                                  const double &close_m15[],
                                  const datetime &time_m15[],
                                  const int &map_m15[],
                                  const double &close_m30[],
                                  const datetime &time_m30[],
                                  const int &map_m30[],
                                  const double &close_h1[],
                                  const datetime &time_h1[],
                                  const int &map_h1[],
                                  const double &ctx_mean_arr[],
                                  const double &ctx_std_arr[],
                                  const double &ctx_up_arr[],
                                  const double &ctx_extra_arr[],
                                  FXAIDataCoreBundle &bundle)
{
   FXAI_DataCoreResetBundle(bundle);
   bundle.ready = true;
   bundle.live_mode = false;
   bundle.symbol = snapshot.symbol;
   bundle.signal_bar = snapshot.bar_time;
   bundle.needed = needed;
   bundle.align_upto = align_upto;
   bundle.snapshot = snapshot;

   ArrayCopy(bundle.open_arr, open_arr);
   ArrayCopy(bundle.high_arr, high_arr);
   ArrayCopy(bundle.low_arr, low_arr);
   ArrayCopy(bundle.close_arr, close_arr);
   ArrayCopy(bundle.time_arr, time_arr);
   ArrayCopy(bundle.spread_m1, spread_m1);
   ArrayCopy(bundle.close_m5, close_m5);
   ArrayCopy(bundle.time_m5, time_m5);
   ArrayCopy(bundle.map_m5, map_m5);
   ArrayCopy(bundle.close_m15, close_m15);
   ArrayCopy(bundle.time_m15, time_m15);
   ArrayCopy(bundle.map_m15, map_m15);
   ArrayCopy(bundle.close_m30, close_m30);
   ArrayCopy(bundle.time_m30, time_m30);
   ArrayCopy(bundle.map_m30, map_m30);
   ArrayCopy(bundle.close_h1, close_h1);
   ArrayCopy(bundle.time_h1, time_h1);
   ArrayCopy(bundle.map_h1, map_h1);
   ArrayCopy(bundle.ctx_mean_arr, ctx_mean_arr);
   ArrayCopy(bundle.ctx_std_arr, ctx_std_arr);
   ArrayCopy(bundle.ctx_up_arr, ctx_up_arr);
   ArrayCopy(bundle.ctx_extra_arr, ctx_extra_arr);
}

#endif // __FXAI_CORE_DATA_CORE_MQH__
