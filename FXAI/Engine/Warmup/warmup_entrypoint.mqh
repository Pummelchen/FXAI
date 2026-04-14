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

   FXAI_ResetFeatureNormalizationState();
   FXAI_ResetFeatureNormalizationFits();
   FXAIDataCoreBundle warmup_bundle;
   string warmup_data_reason = "";
   if(!FXAI_DataCoreLoadHistoryBundle(symbol,
                                      needed,
                                      needed - 1,
                                      AI_CommissionPerLotSide,
                                      AI_CostBufferPoints,
                                      warmup_bundle,
                                      warmup_data_reason))
   {
      Print("FXAI warmup data core load failed: ", warmup_data_reason);
      return false;
   }

   FXAIDataSnapshot snapshot = warmup_bundle.snapshot;
   double open_arr[];
   double high_arr[];
   double low_arr[];
   double close_arr[];
   datetime time_arr[];
   int spread_m1[];
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
   double ctx_mean_arr[];
   double ctx_std_arr[];
   double ctx_up_arr[];
   double ctx_extra_arr[];
   ArrayCopy(open_arr, warmup_bundle.open_arr);
   ArrayCopy(high_arr, warmup_bundle.high_arr);
   ArrayCopy(low_arr, warmup_bundle.low_arr);
   ArrayCopy(close_arr, warmup_bundle.close_arr);
   ArrayCopy(time_arr, warmup_bundle.time_arr);
   ArrayCopy(spread_m1, warmup_bundle.spread_m1);
   ArrayCopy(close_m5, warmup_bundle.close_m5);
   ArrayCopy(time_m5, warmup_bundle.time_m5);
   ArrayCopy(close_m15, warmup_bundle.close_m15);
   ArrayCopy(time_m15, warmup_bundle.time_m15);
   ArrayCopy(close_m30, warmup_bundle.close_m30);
   ArrayCopy(time_m30, warmup_bundle.time_m30);
   ArrayCopy(close_h1, warmup_bundle.close_h1);
   ArrayCopy(time_h1, warmup_bundle.time_h1);
   ArrayCopy(map_m5, warmup_bundle.map_m5);
   ArrayCopy(map_m15, warmup_bundle.map_m15);
   ArrayCopy(map_m30, warmup_bundle.map_m30);
   ArrayCopy(map_h1, warmup_bundle.map_h1);

   int i_start = max_h;
   int i_end = max_h + warmup_samples - 1;
   int max_valid = needed - FEATURE_LB - 1;
   if(i_end > max_valid) i_end = max_valid;
   if(i_end <= i_start) return false;
   ArrayCopy(ctx_mean_arr, warmup_bundle.ctx_mean_arr);
   ArrayCopy(ctx_std_arr, warmup_bundle.ctx_std_arr);
   ArrayCopy(ctx_up_arr, warmup_bundle.ctx_up_arr);
   ArrayCopy(ctx_extra_arr, warmup_bundle.ctx_extra_arr);

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
