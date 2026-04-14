   FXAI_ResetFeatureNormalizationState();

   const int FEATURE_LB = 10;
   int horizon_load_max = FXAI_GetMaxConfiguredHorizon(base_h);
   int needed = (K > base ? K : base) + horizon_load_max + FEATURE_LB;
   if(needed < 128) needed = 128;
   int align_upto = needed - 1;

   FXAI_AdvanceReliabilityClock(signal_bar);
   int signal_seq = g_rel_clock_seq;

   static FXAIDataCoreBundle live_bundle;
   string data_reason = "";
   if(!FXAI_DataCoreRefreshLiveBundle(live_bundle,
                                      symbol,
                                      signal_bar,
                                      needed,
                                      align_upto,
                                      AI_CommissionPerLotSide,
                                      AI_CostBufferPoints,
                                      data_reason))
   {
      g_ai_last_reason = data_reason;
      FXAI_RuntimePublishIdleSnapshot(symbol);
      return -1;
   }

   FXAIDataSnapshot snapshot = live_bundle.snapshot;
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
   ArrayCopy(open_arr, live_bundle.open_arr);
   ArrayCopy(high_arr, live_bundle.high_arr);
   ArrayCopy(low_arr, live_bundle.low_arr);
   ArrayCopy(close_arr, live_bundle.close_arr);
   ArrayCopy(time_arr, live_bundle.time_arr);
   ArrayCopy(spread_m1, live_bundle.spread_m1);
   ArrayCopy(close_m5, live_bundle.close_m5);
   ArrayCopy(time_m5, live_bundle.time_m5);
   ArrayCopy(close_m15, live_bundle.close_m15);
   ArrayCopy(time_m15, live_bundle.time_m15);
   ArrayCopy(close_m30, live_bundle.close_m30);
   ArrayCopy(time_m30, live_bundle.time_m30);
   ArrayCopy(close_h1, live_bundle.close_h1);
   ArrayCopy(time_h1, live_bundle.time_h1);
   ArrayCopy(map_m5, live_bundle.map_m5);
   ArrayCopy(map_m15, live_bundle.map_m15);
   ArrayCopy(map_m30, live_bundle.map_m30);
   ArrayCopy(map_h1, live_bundle.map_h1);
   ArrayCopy(ctx_mean_arr, live_bundle.ctx_mean_arr);
   ArrayCopy(ctx_std_arr, live_bundle.ctx_std_arr);
   ArrayCopy(ctx_up_arr, live_bundle.ctx_up_arr);
   ArrayCopy(ctx_extra_arr, live_bundle.ctx_extra_arr);

   FXAIExecutionProfile exec_profile;
   FXAI_ResolveExecutionProfile(exec_profile);
   double cost_buffer_points = (AI_CostBufferPoints < 0.0 ? 0.0 : AI_CostBufferPoints);
   double commission_points = live_bundle.snapshot.commission_points;
   double profile_commission_points = FXAI_GetCommissionPointsRoundTripPerLot(symbol,
                                                                              exec_profile.commission_per_lot_side);
   if(profile_commission_points > commission_points)
      commission_points = profile_commission_points;
   live_bundle.snapshot.commission_points = commission_points;
   snapshot.commission_points = commission_points;
   double spread_pred = FXAI_GetSpreadAtIndex(0, live_bundle.spread_m1, live_bundle.snapshot.spread_points);
   double min_move_pred = FXAI_ExecutionEntryCostPoints(spread_pred,
                                                        commission_points,
                                                        cost_buffer_points,
                                                        exec_profile);
   if(min_move_pred < 0.0) min_move_pred = 0.0;
   live_bundle.snapshot.min_move_points = min_move_pred;
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
   FXAICrossAssetPairState adaptive_cross_asset_state;
   FXAI_ResetCrossAssetPairState(adaptive_cross_asset_state);
   if(CrossAssetEnabled)
      FXAI_ReadCrossAssetPairState(symbol, adaptive_cross_asset_state);
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

   ENUM_FXAI_FEATURE_NORMALIZATION norm_method = FXAI_GetFeatureNormalizationMethod();
   double feat_pred[FXAI_AI_FEATURES];
   FXAIFeatureCoreFrame predict_feature_frame;
   if(!FXAI_FeatureCoreBuildFrameFromBundle(live_bundle, 0, H, norm_method, predict_feature_frame))
   {
      g_ai_last_signal_bar = signal_bar;
      g_ai_last_signal_key = decisionKey;
      g_ai_last_signal = -1;
      g_ai_last_reason = "predict_features_failed";
      FXAI_RuntimePublishIdleSnapshot(symbol);
      return -1;
   }
   for(int f=0; f<FXAI_AI_FEATURES; f++)
      feat_pred[f] = predict_feature_frame.raw[f];

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
