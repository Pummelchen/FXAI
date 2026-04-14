bool FXAI_WarmupTrainAndTune(const string symbol)
{
   const int FEATURE_LB = 10;

   int warmup_samples = AI_WarmupSamples;
   if(warmup_samples < 2000) warmup_samples = 2000;
   if(warmup_samples > 50000) warmup_samples = 50000;

   int warmup_loops = AI_WarmupLoops;
   if(warmup_loops < 10) warmup_loops = 10;
   if(warmup_loops > 500) warmup_loops = 500;

   int warmup_train_epochs = AI_Epochs;
   if(warmup_train_epochs < 1) warmup_train_epochs = 1;
   if(warmup_train_epochs > 5) warmup_train_epochs = 5;

   int warmup_folds = AI_WarmupFolds;
   if(warmup_folds < 2) warmup_folds = 2;
   if(warmup_folds > 5) warmup_folds = 5;

   int warmup_min_trades = AI_WarmupMinTrades;
   if(warmup_min_trades < 20) warmup_min_trades = 20;
   if(warmup_min_trades > 2000) warmup_min_trades = 2000;

   int base_h = FXAI_ClampHorizon(PredictionTargetMinutes);
   int horizons[];
   ArrayResize(horizons, 0);
   if(AI_MultiHorizon && ArraySize(g_horizon_minutes) > 0)
   {
      int hn = ArraySize(g_horizon_minutes);
      if(hn > FXAI_MAX_HORIZONS) hn = FXAI_MAX_HORIZONS;
      ArrayResize(horizons, hn);
      for(int i=0; i<hn; i++)
         horizons[i] = FXAI_ClampHorizon(g_horizon_minutes[i]);
   }
   if(ArraySize(horizons) <= 0)
   {
      ArrayResize(horizons, 1);
      horizons[0] = base_h;
   }
   bool have_primary = false;
   int max_h = base_h;
   for(int i=0; i<ArraySize(horizons); i++)
   {
      if(horizons[i] == base_h) have_primary = true;
      if(horizons[i] > max_h) max_h = horizons[i];
   }
   if(!have_primary && ArraySize(horizons) < FXAI_MAX_HORIZONS)
   {
      int hs = ArraySize(horizons);
      ArrayResize(horizons, hs + 1);
      horizons[hs] = base_h;
      if(base_h > max_h) max_h = base_h;
   }

   double base_buy_thr = AI_BuyThreshold;
   double base_sell_thr = AI_SellThreshold;
   FXAI_SanitizeThresholdPair(base_buy_thr, base_sell_thr);
   double evThresholdPoints = FXAI_Clamp(AI_EVThresholdPoints, 0.0, 100.0);

   int needed = warmup_samples + max_h + FEATURE_LB;

   FXAIDataSnapshot snapshot;
   if(!FXAI_ExportDataSnapshot(symbol, AI_CommissionPerLotSide, AI_CostBufferPoints, snapshot))
      return false;
   FXAI_ResetFeatureNormalizationState();
   FXAI_ResetFeatureNormalizationFits();

   MqlRates rates_m1[];
   MqlRates rates_m5[];
   MqlRates rates_m15[];
   MqlRates rates_m30[];
   MqlRates rates_h1[];
   MqlRates rates_ctx_tmp[];

   double open_arr[];
   double high_arr[];
   double low_arr[];
   double close_arr[];
   datetime time_arr[];
   int spread_m1[];
   if(!FXAI_LoadSeriesWithSpread(symbol, needed, rates_m1, close_arr, time_arr, spread_m1))
      return false;

   FXAI_ExtractRatesOHLC(rates_m1, open_arr, high_arr, low_arr, close_arr);

   if(ArraySize(close_arr) < needed || ArraySize(time_arr) < needed)
      return false;
   if(!FXAI_ValidateM1SeriesBundle(time_arr, open_arr, high_arr, low_arr, close_arr, spread_m1, needed))
      return false;

   int needed_m5 = (needed / 5) + 80;
   int needed_m15 = (needed / 15) + 80;
   int needed_m30 = (needed / 30) + 80;
   int needed_h1 = (needed / 60) + 80;
   if(needed_m5 < 220) needed_m5 = 220;
   if(needed_m15 < 220) needed_m15 = 220;
   if(needed_m30 < 220) needed_m30 = 220;
   if(needed_h1 < 220) needed_h1 = 220;

   double close_m5[];
   datetime time_m5[];
   double close_m15[];
   datetime time_m15[];
   double close_m30[];
   datetime time_m30[];
   double close_h1[];
   datetime time_h1[];
   int map_m5[];
   int map_m15[];
   int map_m30[];
   int map_h1[];

   FXAI_LoadSeriesOptionalCached(symbol, PERIOD_M5, needed_m5, rates_m5, close_m5, time_m5);
   FXAI_LoadSeriesOptionalCached(symbol, PERIOD_M15, needed_m15, rates_m15, close_m15, time_m15);
   FXAI_LoadSeriesOptionalCached(symbol, PERIOD_M30, needed_m30, rates_m30, close_m30, time_m30);
   FXAI_LoadSeriesOptionalCached(symbol, PERIOD_H1, needed_h1, rates_h1, close_h1, time_h1);

   int lag_m5 = 2 * PeriodSeconds(PERIOD_M5);
   int lag_m15 = 2 * PeriodSeconds(PERIOD_M15);
   int lag_m30 = 2 * PeriodSeconds(PERIOD_M30);
   int lag_h1 = 2 * PeriodSeconds(PERIOD_H1);
   if(lag_m5 <= 0) lag_m5 = 600;
   if(lag_m15 <= 0) lag_m15 = 1800;
   if(lag_m30 <= 0) lag_m30 = 3600;
   if(lag_h1 <= 0) lag_h1 = 7200;

   FXAI_BuildAlignedIndexMap(time_arr, time_m5, lag_m5, map_m5);
   FXAI_BuildAlignedIndexMap(time_arr, time_m15, lag_m15, map_m15);
   FXAI_BuildAlignedIndexMap(time_arr, time_m30, lag_m30, map_m30);
   FXAI_BuildAlignedIndexMap(time_arr, time_h1, lag_h1, map_h1);

   int ctx_count = ArraySize(g_context_symbols);
   if(ctx_count > FXAI_MAX_CONTEXT_SYMBOLS) ctx_count = FXAI_MAX_CONTEXT_SYMBOLS;
   FXAIContextSeries ctx_series[];
   ArrayResize(ctx_series, ctx_count);
   for(int s=0; s<ctx_count; s++)
   {
      ctx_series[s].symbol = g_context_symbols[s];
      ctx_series[s].loaded = FXAI_LoadRatesOptional(g_context_symbols[s],
                                                    PERIOD_M1,
                                                    needed,
                                                    rates_ctx_tmp);
      if(ctx_series[s].loaded)
      {
         FXAI_ExtractRatesCloseTimeSpread(rates_ctx_tmp,
                                          ctx_series[s].close,
                                          ctx_series[s].time,
                                          ctx_series[s].spread);
         FXAI_ExtractRatesOHLC(rates_ctx_tmp,
                               ctx_series[s].open,
                               ctx_series[s].high,
                               ctx_series[s].low,
                               ctx_series[s].close);
         ctx_series[s].loaded = FXAI_ValidateM1SeriesBundle(ctx_series[s].time,
                                                            ctx_series[s].open,
                                                            ctx_series[s].high,
                                                            ctx_series[s].low,
                                                            ctx_series[s].close,
                                                            ctx_series[s].spread,
                                                            needed);
      }
   }

   int i_start = max_h;
   int i_end = max_h + warmup_samples - 1;
   int max_valid = needed - FEATURE_LB - 1;
   if(i_end > max_valid) i_end = max_valid;
   if(i_end <= i_start) return false;

   double ctx_mean_arr[];
   double ctx_std_arr[];
   double ctx_up_arr[];
   double ctx_extra_arr[];
   FXAI_PrecomputeDynamicContextAggregates(time_arr,
                                           close_arr,
                                           ctx_series,
                                           ctx_count,
                                           i_end,
                                           ctx_mean_arr,
                                           ctx_std_arr,
                                           ctx_up_arr,
                                           ctx_extra_arr);

   double cost_buffer_points = (AI_CostBufferPoints < 0.0 ? 0.0 : AI_CostBufferPoints);
   double commission_points = snapshot.commission_points;
   FXAIAIHyperParams base_hp;
   FXAI_BuildHyperParams(base_hp);

   // Warmup-stage feature-adaptive normalization window search.
   FXAI_OptimizeNormalizationWindows(i_start,
                                     i_end,
                                     base_h,
                                     commission_points,
                                     cost_buffer_points,
                                     evThresholdPoints,
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
                                     base_hp,
                                     base_buy_thr,
                                     base_sell_thr);

   datetime bar_time = iTime(symbol, PERIOD_M1, 1);
   if(bar_time <= 0) bar_time = TimeCurrent();
   int seed = AI_WarmupSeed;
   if(seed < 0) seed = -seed;
   int evLookbackWarm = AI_EVLookbackSamples;
   if(evLookbackWarm < 20) evLookbackWarm = 20;
   if(evLookbackWarm > 400) evLookbackWarm = 400;
   int ai_hint = (AI_Ensemble ? -1 : (int)AI_Type);
   if(ai_hint < -1 || ai_hint >= FXAI_AI_COUNT) ai_hint = -1;
   FXAIPreparedSample primary_samples[];
   for(int hi=0; hi<ArraySize(horizons); hi++)
   {
      int H = FXAI_ClampHorizon(horizons[hi]);
      FXAIPreparedSample samples_h[];
      FXAI_PrecomputeTrainingSamples(i_start,
                                    i_end,
                                    H,
                                    commission_points,
                                    cost_buffer_points,
                                    evThresholdPoints,
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
                                    -1,
                                    samples_h);
      FXAI_WarmupTrainHorizonPolicyForSamples(H,
                                              base_h,
                                              evLookbackWarm,
                                              snapshot,
                                              close_arr,
                                              ai_hint,
                                              i_start,
                                              i_end,
                                              samples_h);

      FXAINormSampleCache norm_caches[];
      ArrayResize(norm_caches, 0);
      FXAI_WarmupSelectNormBanksForHorizon(H,
                                           H == base_h,
                                           warmup_train_epochs,
                                           i_start,
                                           i_end,
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
                                           base_hp,
                                           base_buy_thr,
                                           base_sell_thr,
                                           norm_caches);

      FXAI_WarmupSelectBanksForHorizon(H,
                                       H == base_h,
                                       warmup_loops,
                                       warmup_folds,
                                       warmup_train_epochs,
                                       warmup_min_trades,
                                       seed,
                                       bar_time,
                                       base_hp,
                                       base_buy_thr,
                                       base_sell_thr,
                                       i_start,
                                       i_end,
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
                                       samples_h,
                                       norm_caches);
      if(H == base_h)
         FXAI_CopyPreparedSamples(samples_h, primary_samples);
   }

   for(int hi=0; hi<ArraySize(horizons); hi++)
   {
      int H = FXAI_ClampHorizon(horizons[hi]);
      FXAIPreparedSample samples_h[];
      FXAI_PrecomputeTrainingSamples(i_start,
                                    i_end,
                                    H,
                                    commission_points,
                                    cost_buffer_points,
                                    evThresholdPoints,
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
                                    -1,
                                    samples_h);
      FXAINormSampleCache norm_caches[];
      ArrayResize(norm_caches, 0);
      for(int ai_idx=0; ai_idx<FXAI_AI_COUNT; ai_idx++)
      {
         FXAI_EnsureRoutedNormCachesForSamples(ai_idx,
                                               i_start,
                                               i_end,
                                               H,
                                               commission_points,
                                               cost_buffer_points,
                                               evThresholdPoints,
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
                                               samples_h,
                                               norm_caches);
      }
      FXAI_WarmupPretrainMetaForSamples(H,
                                        warmup_folds,
                                        warmup_train_epochs,
                                        i_start,
                                        i_end,
                                        base_buy_thr,
                                        base_sell_thr,
                                        samples_h,
                                        norm_caches);
      if(H == base_h && ArraySize(primary_samples) <= 0)
         FXAI_CopyPreparedSamples(samples_h, primary_samples);
   }

   if(ArraySize(primary_samples) <= 0) return false;
   for(int ai_idx=0; ai_idx<FXAI_AI_COUNT; ai_idx++)
   {
      CFXAIAIPlugin *runtime = g_plugins.Get(ai_idx);
      if(runtime == NULL) continue;

      FXAI_ResetModelAuxState(ai_idx);
      runtime.Reset();
      FXAIAIHyperParams hp_init;
      FXAI_GetModelHyperParamsRouted(ai_idx, 0, base_h, hp_init);
      runtime.EnsureInitialized(hp_init);
   }

   // Warm the runtime models across every configured horizon. The online path
   // uses a single runtime instance per model, so base-horizon-only warmup can
   // leave routed non-base horizons effectively cold on the first live bars.
   for(int hi=0; hi<ArraySize(horizons); hi++)
   {
      int H = FXAI_ClampHorizon(horizons[hi]);
      FXAIPreparedSample runtime_samples[];
      FXAINormSampleCache runtime_norm_caches[];
      ArrayResize(runtime_norm_caches, 0);
      if(H == base_h)
      {
         FXAI_CopyPreparedSamples(primary_samples, runtime_samples);
      }
      else
      {
         FXAI_PrecomputeTrainingSamples(i_start,
                                       i_end,
                                       H,
                                       commission_points,
                                       cost_buffer_points,
                                       evThresholdPoints,
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
                                       -1,
                                       runtime_samples);
      }

      FXAI_WarmupPrimeFeatureDriftBaseline(runtime_samples);
      FXAI_WarmupPrimeAnalogMemorySamples(runtime_samples, 1, ArraySize(runtime_samples) - 1);
      int transfer_cap = MathMin(640, MathMax(128, warmup_samples / 6));
      FXAI_WarmupPretrainGlobalTransferSamples(runtime_samples,
                                               MathMin(transfer_cap * 2, 1536),
                                               0.75,
                                               base_hp.lr);
      for(int ai_idx=0; ai_idx<FXAI_AI_COUNT; ai_idx++)
      {
         CFXAIAIPlugin *runtime = g_plugins.Get(ai_idx);
         if(runtime == NULL) continue;

         FXAIAIHyperParams hp_transfer;
         FXAI_GetModelHyperParamsRouted(ai_idx, 0, H, hp_transfer);
         FXAI_WarmupPretrainSharedTransferSamples(runtime_samples,
                                                  transfer_cap,
                                                  0.60,
                                                  *runtime,
                                                  hp_transfer);
      }

      string transfer_universe[];
      FXAI_WarmupBuildTransferUniverse(symbol, transfer_universe);
      for(int s=0; s<ArraySize(transfer_universe); s++)
      {
         FXAIPreparedSample transfer_samples[];
         if(transfer_universe[s] == symbol)
            continue;
         if(!FXAI_WarmupBuildTransferSymbolSamplesForHorizon(transfer_universe[s],
                                                             symbol,
                                                             needed,
                                                             max_h,
                                                             transfer_cap,
                                                             H,
                                                             AI_CommissionPerLotSide,
                                                             cost_buffer_points,
                                                             evThresholdPoints,
                                                             transfer_samples))
            continue;

         FXAI_WarmupPrimeAnalogMemorySamples(transfer_samples, 0, ArraySize(transfer_samples) - 1);
         FXAI_WarmupPretrainGlobalTransferSamples(transfer_samples,
                                                  MathMax(96, transfer_cap),
                                                  0.60,
                                                  base_hp.lr);
         for(int ai_idx=0; ai_idx<FXAI_AI_COUNT; ai_idx++)
         {
            CFXAIAIPlugin *runtime = g_plugins.Get(ai_idx);
            if(runtime == NULL) continue;

            FXAIAIHyperParams hp_transfer;
            FXAI_GetModelHyperParamsRouted(ai_idx, 0, H, hp_transfer);
            FXAI_WarmupPretrainSharedTransferSamples(transfer_samples,
                                                     MathMax(64, transfer_cap / 2),
                                                     0.45,
                                                     *runtime,
                                                     hp_transfer);
         }
      }

      for(int ai_idx=0; ai_idx<FXAI_AI_COUNT; ai_idx++)
      {
         FXAI_EnsureRoutedNormCachesForSamples(ai_idx,
                                               i_start,
                                               i_end,
                                               H,
                                               commission_points,
                                               cost_buffer_points,
                                               evThresholdPoints,
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
                                               runtime_samples,
                                               runtime_norm_caches);
      }

      for(int ai_idx=0; ai_idx<FXAI_AI_COUNT; ai_idx++)
      {
         CFXAIAIPlugin *runtime = g_plugins.Get(ai_idx);
         if(runtime == NULL) continue;

         FXAI_TrainModelWindowPreparedRoutedCached(ai_idx,
                                                   *runtime,
                                                   i_start,
                                                   i_end,
                                                   warmup_train_epochs,
                                                   runtime_samples,
                                                   runtime_norm_caches);
      }
   }

   int portfolio_eval_cap = MathMin(160, MathMax(64, warmup_samples / 12));
   FXAI_WarmupBuildPortfolioDiagnostics(symbol,
                                        needed,
                                        max_h,
                                        base_h,
                                        AI_CommissionPerLotSide,
                                        cost_buffer_points,
                                        evThresholdPoints,
                                        portfolio_eval_cap,
                                        primary_samples);

   for(int ai_idx=0; ai_idx<FXAI_AI_COUNT; ai_idx++)
   {
      if(g_plugins.Get(ai_idx) == NULL) continue;
      g_ai_trained[ai_idx] = true;
      g_ai_last_train_bar[ai_idx] = bar_time;
   }

   g_ai_warmup_done = true;
   FXAI_MarkMetaArtifactsDirty();
    FXAI_MarkRuntimeArtifactsDirty();
   FXAI_SaveMetaArtifacts(symbol);
   FXAI_SaveRuntimeArtifacts(symbol);
   Print("FXAI warmup completed: symbol=", symbol,
         ", samples=", warmup_samples,
         ", loops=", warmup_loops,
         ", folds=", warmup_folds,
         ", horizons=", ArraySize(horizons));
   return true;
}
