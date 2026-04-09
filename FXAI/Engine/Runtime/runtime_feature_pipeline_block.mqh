   FXAIDataSnapshot snapshot;
   if(!FXAI_ExportDataSnapshot(symbol, AI_CommissionPerLotSide, AI_CostBufferPoints, snapshot))
   {
      g_ai_last_reason = "snapshot_export_failed";
      FXAI_RuntimePublishIdleSnapshot(symbol);
      return -1;
   }
   // Keep cache/training keyed to the same closed bar anchor.
   snapshot.bar_time = signal_bar;
   FXAI_ResetFeatureNormalizationState();

   const int FEATURE_LB = 10;
   int horizon_load_max = FXAI_GetMaxConfiguredHorizon(base_h);
   int needed = (K > base ? K : base) + horizon_load_max + FEATURE_LB;
   if(needed < 128) needed = 128;
   int align_upto = needed - 1;

   static MqlRates rates_m1[];
   static MqlRates rates_m5[];
   static MqlRates rates_m15[];
   static MqlRates rates_m30[];
   static MqlRates rates_h1[];
   static string cache_symbol = "";
   static datetime last_bar_m1 = 0;
   static datetime last_bar_m5 = 0;
   static datetime last_bar_m15 = 0;
   static datetime last_bar_m30 = 0;
   static datetime last_bar_h1 = 0;

   static double open_arr[];
   static double high_arr[];
   static double low_arr[];
   static double close_arr[];
   static datetime time_arr[];
   static int spread_m1[];
   static FXAIContextSeries ctx_series[];
   static double ctx_mean_arr[];
   static double ctx_std_arr[];
   static double ctx_up_arr[];
   static double ctx_extra_arr[];

   if(cache_symbol != symbol)
   {
      cache_symbol = symbol;
      last_bar_m1 = 0;
      last_bar_m5 = 0;
      last_bar_m15 = 0;
      last_bar_m30 = 0;
      last_bar_h1 = 0;
      ArrayResize(rates_m1, 0);
      ArrayResize(rates_m5, 0);
      ArrayResize(rates_m15, 0);
      ArrayResize(rates_m30, 0);
      ArrayResize(rates_h1, 0);
      ArrayResize(ctx_series, 0);
   }

   FXAI_AdvanceReliabilityClock(signal_bar);
   int signal_seq = g_rel_clock_seq;

   if(!FXAI_UpdateRatesRolling(symbol, PERIOD_M1, needed, last_bar_m1, rates_m1))
   {
      g_ai_last_reason = "m1_series_load_failed";
      FXAI_RuntimePublishIdleSnapshot(symbol);
      return -1;
   }
   FXAI_ExtractRatesCloseTimeSpread(rates_m1, close_arr, time_arr, spread_m1);
   FXAI_ExtractRatesOHLC(rates_m1, open_arr, high_arr, low_arr, close_arr);
   if(ArraySize(close_arr) < needed || ArraySize(time_arr) < needed || ArraySize(spread_m1) < needed)
   {
      g_ai_last_reason = "m1_series_size_failed";
      FXAI_RuntimePublishIdleSnapshot(symbol);
      return -1;
   }
   if(!FXAI_ValidateM1SeriesBundle(time_arr, open_arr, high_arr, low_arr, close_arr, spread_m1, needed))
   {
      g_ai_last_reason = "m1_series_integrity_failed";
      FXAI_RuntimePublishIdleSnapshot(symbol);
      return -1;
   }

   int needed_m5 = (needed / 5) + 80;
   int needed_m15 = (needed / 15) + 80;
   int needed_m30 = (needed / 30) + 80;
   int needed_h1 = (needed / 60) + 80;
   if(needed_m5 < 220) needed_m5 = 220;
   if(needed_m15 < 220) needed_m15 = 220;
   if(needed_m30 < 220) needed_m30 = 220;
   if(needed_h1 < 220) needed_h1 = 220;

   static double close_m5[];
   static datetime time_m5[];
   static double close_m15[];
   static datetime time_m15[];
   static double close_m30[];
   static datetime time_m30[];
   static double close_h1[];
   static datetime time_h1[];
   static int map_m5[];
   static int map_m15[];
   static int map_m30[];
   static int map_h1[];
   if(FXAI_UpdateRatesRolling(symbol, PERIOD_M5, needed_m5, last_bar_m5, rates_m5))
      FXAI_ExtractRatesCloseTime(rates_m5, close_m5, time_m5);
   else
   {
      ArrayResize(close_m5, 0);
      ArrayResize(time_m5, 0);
      ArrayResize(map_m5, 0);
   }

   if(FXAI_UpdateRatesRolling(symbol, PERIOD_M15, needed_m15, last_bar_m15, rates_m15))
      FXAI_ExtractRatesCloseTime(rates_m15, close_m15, time_m15);
   else
   {
      ArrayResize(close_m15, 0);
      ArrayResize(time_m15, 0);
      ArrayResize(map_m15, 0);
   }

   if(FXAI_UpdateRatesRolling(symbol, PERIOD_M30, needed_m30, last_bar_m30, rates_m30))
      FXAI_ExtractRatesCloseTime(rates_m30, close_m30, time_m30);
   else
   {
      ArrayResize(close_m30, 0);
      ArrayResize(time_m30, 0);
      ArrayResize(map_m30, 0);
   }

   if(FXAI_UpdateRatesRolling(symbol, PERIOD_H1, needed_h1, last_bar_h1, rates_h1))
      FXAI_ExtractRatesCloseTime(rates_h1, close_h1, time_h1);
   else
   {
      ArrayResize(close_h1, 0);
      ArrayResize(time_h1, 0);
      ArrayResize(map_h1, 0);
   }

   int lag_m5 = 2 * PeriodSeconds(PERIOD_M5);
   int lag_m15 = 2 * PeriodSeconds(PERIOD_M15);
   int lag_m30 = 2 * PeriodSeconds(PERIOD_M30);
   int lag_h1 = 2 * PeriodSeconds(PERIOD_H1);
   if(lag_m5 <= 0) lag_m5 = 600;
   if(lag_m15 <= 0) lag_m15 = 1800;
   if(lag_m30 <= 0) lag_m30 = 3600;
   if(lag_h1 <= 0) lag_h1 = 7200;

   FXAI_BuildAlignedIndexMapRange(time_arr, time_m5, lag_m5, align_upto, map_m5);
   FXAI_BuildAlignedIndexMapRange(time_arr, time_m15, lag_m15, align_upto, map_m15);
   FXAI_BuildAlignedIndexMapRange(time_arr, time_m30, lag_m30, align_upto, map_m30);
   FXAI_BuildAlignedIndexMapRange(time_arr, time_h1, lag_h1, align_upto, map_h1);

   int ctx_count = ArraySize(g_context_symbols);
   if(ctx_count > FXAI_MAX_CONTEXT_SYMBOLS) ctx_count = FXAI_MAX_CONTEXT_SYMBOLS;
   if(ArraySize(ctx_series) != ctx_count)
   {
      ArrayResize(ctx_series, ctx_count);
      for(int s=0; s<ctx_count; s++)
      {
         ctx_series[s].loaded = false;
         ctx_series[s].symbol = "";
         ctx_series[s].last_bar_time = 0;
         ArrayResize(ctx_series[s].rates, 0);
         ArrayResize(ctx_series[s].open, 0);
         ArrayResize(ctx_series[s].high, 0);
         ArrayResize(ctx_series[s].low, 0);
         ArrayResize(ctx_series[s].close, 0);
         ArrayResize(ctx_series[s].time, 0);
         ArrayResize(ctx_series[s].spread, 0);
         ArrayResize(ctx_series[s].aligned_idx, 0);
      }
   }
   for(int s=0; s<ctx_count; s++)
   {
      string ctx_symbol = g_context_symbols[s];
      if(ctx_series[s].symbol != ctx_symbol)
      {
         ctx_series[s].symbol = ctx_symbol;
         ctx_series[s].last_bar_time = 0;
         ArrayResize(ctx_series[s].rates, 0);
         ArrayResize(ctx_series[s].open, 0);
         ArrayResize(ctx_series[s].high, 0);
         ArrayResize(ctx_series[s].low, 0);
         ArrayResize(ctx_series[s].close, 0);
         ArrayResize(ctx_series[s].time, 0);
         ArrayResize(ctx_series[s].spread, 0);
         ArrayResize(ctx_series[s].aligned_idx, 0);
      }

      ctx_series[s].loaded = FXAI_UpdateRatesRolling(ctx_symbol,
                                                    PERIOD_M1,
                                                    needed,
                                                    ctx_series[s].last_bar_time,
                                                    ctx_series[s].rates);
      if(ctx_series[s].loaded)
      {
         FXAI_ExtractRatesCloseTimeSpread(ctx_series[s].rates,
                                         ctx_series[s].close,
                                         ctx_series[s].time,
                                         ctx_series[s].spread);
         FXAI_ExtractRatesOHLC(ctx_series[s].rates,
                               ctx_series[s].open,
                               ctx_series[s].high,
                               ctx_series[s].low,
                               ctx_series[s].close);
         if(!FXAI_ValidateM1SeriesBundle(ctx_series[s].time,
                                         ctx_series[s].open,
                                         ctx_series[s].high,
                                         ctx_series[s].low,
                                         ctx_series[s].close,
                                         ctx_series[s].spread,
                                         needed))
         {
            ctx_series[s].loaded = false;
         }
      }
      if(!ctx_series[s].loaded)
      {
         ArrayResize(ctx_series[s].open, 0);
         ArrayResize(ctx_series[s].high, 0);
         ArrayResize(ctx_series[s].low, 0);
         ArrayResize(ctx_series[s].close, 0);
         ArrayResize(ctx_series[s].time, 0);
         ArrayResize(ctx_series[s].spread, 0);
         ArrayResize(ctx_series[s].aligned_idx, 0);
      }
   }

   FXAI_PrecomputeDynamicContextAggregates(time_arr,
                                           close_arr,
                                           ctx_series,
                                           ctx_count,
                                           align_upto,
                                           ctx_mean_arr,
                                           ctx_std_arr,
                                           ctx_up_arr,
                                           ctx_extra_arr);

   FXAIExecutionProfile exec_profile;
   FXAI_ResolveExecutionProfile(exec_profile);
   double cost_buffer_points = (AI_CostBufferPoints < 0.0 ? 0.0 : AI_CostBufferPoints);
   double commission_points = snapshot.commission_points;
   double profile_commission_points = FXAI_GetCommissionPointsRoundTripPerLot(symbol,
                                                                              exec_profile.commission_per_lot_side);
   if(profile_commission_points > commission_points)
      commission_points = profile_commission_points;
   snapshot.commission_points = commission_points;
   double spread_pred = FXAI_GetSpreadAtIndex(0, spread_m1, snapshot.spread_points);
   double min_move_pred = FXAI_ExecutionEntryCostPoints(spread_pred,
                                                        commission_points,
                                                        cost_buffer_points,
                                                        exec_profile);
   if(min_move_pred < 0.0) min_move_pred = 0.0;
   snapshot.min_move_points = min_move_pred;
   FXAILiveDeploymentProfile deploy_profile;
   FXAI_LoadLiveDeploymentProfile(symbol, deploy_profile, false);
   FXAIStudentRouterProfile student_router;
   FXAI_LoadStudentRouterProfile(symbol, student_router, false);
   FXAIAdaptiveRouterProfile adaptive_router_profile;
   FXAI_LoadAdaptiveRouterProfile(symbol, adaptive_router_profile, false);
   FXAINewsPulsePairState adaptive_news_state;
   FXAI_ResetNewsPulsePairState(adaptive_news_state);
   if(NewsPulseEnabled)
      FXAI_ReadNewsPulsePairState(symbol, adaptive_news_state);
   FXAIRatesEnginePairState adaptive_rates_state;
   FXAI_ResetRatesEnginePairState(adaptive_rates_state);
   if(RatesEngineEnabled)
      FXAI_ReadRatesEnginePairState(symbol, adaptive_rates_state);
   FXAIMicrostructurePairState adaptive_micro_state;
   FXAI_ResetMicrostructurePairState(adaptive_micro_state);
   if(MicrostructureEnabled)
      FXAI_ReadMicrostructurePairState(symbol, adaptive_micro_state);
   double vol_hint = MathAbs(FXAI_SafeReturn(close_arr, 0, 1));
   int regime_hint = FXAI_GetRegimeId(snapshot.bar_time, spread_pred, vol_hint);
   int ai_hint = (ensembleMode ? -1 : aiType);
   double ctx_util = 0.0, ctx_stability = 0.0, ctx_lead = 0.0, ctx_coverage = 0.0;
   FXAI_GetDynamicContextState(ctx_util, ctx_stability, ctx_lead, ctx_coverage);
   double context_strength = FXAI_Clamp(MathAbs(FXAI_GetArrayValue(ctx_mean_arr, 0, 0.0)) +
                                        FXAI_GetArrayValue(ctx_std_arr, 0, 0.0) +
                                        MathAbs(FXAI_GetArrayValue(ctx_up_arr, 0, 0.5) - 0.5),
                                        0.0,
                                        4.0);
   double context_quality = FXAI_Clamp(0.45 * ctx_util +
                                       0.25 * ctx_stability +
                                       0.20 * ctx_lead +
                                       0.10 * ctx_coverage,
                                       -1.0,
                                       2.0);
   g_ai_last_context_strength = context_strength;
   g_ai_last_context_quality = context_quality;
   g_ai_last_min_move_points = min_move_pred;
   double model_reliability_hint = 0.50;
   if(ai_hint >= 0 && ai_hint < FXAI_AI_COUNT)
      model_reliability_hint = FXAI_Clamp(g_model_reliability[ai_hint], 0.0, 1.0);

   int H = FXAI_SelectRoutedHorizon(close_arr,
                                    snapshot,
                                    min_move_pred,
                                    evLookback,
                                    base_h,
                                    regime_hint,
                                    ai_hint,
                                    context_strength,
                                    context_quality,
                                    model_reliability_hint);
   int init_start = H;
   int init_end = H + base - 1;
   int online_start = H;
   int online_end = H + K - 1;
   int shadow_samples = Ensemble_ShadowSamples;
   if(shadow_samples < 8) shadow_samples = 8;
   if(shadow_samples > 200) shadow_samples = 200;
   int shadow_epochs = Ensemble_ShadowEpochs;
   if(shadow_epochs < 1) shadow_epochs = 1;
   if(shadow_epochs > 3) shadow_epochs = 3;
   int shadow_every = Ensemble_ShadowEveryBars;
   if(shadow_every < 1) shadow_every = 1;
   bool shadow_allowed = deploy_profile.shadow_enabled;
   bool run_shadow = (ensembleMode && shadow_allowed && FXAI_IsShadowBar(shadow_every, signal_seq));
   int shadow_start = H;
   int shadow_end = H + shadow_samples - 1;

   int max_valid = needed - FEATURE_LB - 1;
   if(init_end > max_valid) init_end = max_valid;
   if(online_end > max_valid) online_end = max_valid;
   if(shadow_end > max_valid) shadow_end = max_valid;
   bool have_init_window = (init_end >= init_start);
   bool have_online_window = (online_end >= online_start);
   bool have_shadow_window = (shadow_end >= shadow_start);

   int precompute_end = -1;
   if(have_init_window) precompute_end = init_end;
   if(have_online_window && online_end > precompute_end) precompute_end = online_end;
   if(run_shadow && have_shadow_window && shadow_end > precompute_end) precompute_end = shadow_end;

   double ctx_mean_pred = FXAI_GetArrayValue(ctx_mean_arr, 0, 0.0);
   double ctx_std_pred = FXAI_GetArrayValue(ctx_std_arr, 0, 0.0);
   double ctx_up_pred = FXAI_GetArrayValue(ctx_up_arr, 0, 0.5);

   ENUM_FXAI_FEATURE_NORMALIZATION norm_method = FXAI_GetFeatureNormalizationMethod();
   double feat_pred[FXAI_AI_FEATURES];
   if(!FXAI_ComputeFeatureVector(0,
                                _Symbol,
                                spread_pred,
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
                                ctx_mean_pred,
                                ctx_std_pred,
                                ctx_up_pred,
                                ctx_extra_arr,
                                norm_method,
                                feat_pred))
   {
      g_ai_last_signal_bar = signal_bar;
      g_ai_last_signal_key = decisionKey;
      g_ai_last_signal = -1;
      g_ai_last_reason = "predict_features_failed";
      FXAI_RuntimePublishIdleSnapshot(symbol);
      return -1;
   }

   double fallback_expected_move = FXAI_EstimateExpectedAbsMovePoints(close_arr,
                                                                      H,
                                                                      evLookback,
                                                                      snapshot.point);
   if(fallback_expected_move <= 0.0)
      fallback_expected_move = 0.0;
   double vol_proxy_abs = FXAI_RollingReturnStd(close_arr, 0, 10);
   if(vol_proxy_abs < 1e-6)
      vol_proxy_abs = FXAI_RollingAbsReturn(close_arr, 0, 10);
   if(vol_proxy_abs < 1e-6)
      vol_proxy_abs = vol_hint;
   FXAI_UpdateRegimeEMAs(spread_pred, vol_proxy_abs);
   int regime_id = FXAI_GetRegimeId(snapshot.bar_time, spread_pred, vol_proxy_abs);
   g_ai_last_horizon_minutes = H;
   g_ai_last_regime_id = regime_id;
   double hpolicy_feat[FXAI_HPOL_FEATS];
   FXAI_BuildHorizonPolicyFeatures(H,
                                   base_h,
                                   fallback_expected_move,
                                   min_move_pred,
                                   snapshot,
                                   MathAbs(FXAI_SafeReturn(close_arr, 0, 1)),
                                   regime_id,
                                   ai_hint,
                                   context_strength,
                                   context_quality,
                                   model_reliability_hint,
                                   hpolicy_feat);
   FXAI_EnqueueHorizonPolicyPending(signal_seq, regime_id, H, min_move_pred, hpolicy_feat);

   static FXAIPreparedSample samples[];
   if(precompute_end >= 1)
   {
      // Start at 1 (not H) so rolling normalizers see the full recent past for
      // prediction-time feature scaling, even when horizon H is large.
      FXAI_PrecomputeTrainingSamples(1,
                                    precompute_end,
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
                                    samples);
   }
   FXAI_RecordRuntimeStageMs(FXAI_RUNTIME_STAGE_FEATURE_PIPELINE,
                             (double)(GetMicrosecondCount() - feature_stage_t0) / 1000.0);

   if(have_online_window && online_start >= 0 && online_start < ArraySize(samples))
      FXAI_AddReplaySample(samples[online_start]);
