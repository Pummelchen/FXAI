#ifndef __FXAI_ENGINE_RUNTIME_MQH__
#define __FXAI_ENGINE_RUNTIME_MQH__

int SpecialDirectionAI(const string symbol)
{
   g_ai_last_reason = "start";
   double cached_expected_move_points = g_ai_last_expected_move_points;
   double cached_trade_edge_points = g_ai_last_trade_edge_points;
   double cached_confidence = g_ai_last_confidence;
   double cached_reliability = g_ai_last_reliability;
   double cached_path_risk = g_ai_last_path_risk;
   double cached_fill_risk = g_ai_last_fill_risk;
   double cached_trade_gate = g_ai_last_trade_gate;
   double cached_context_quality = g_ai_last_context_quality;
   double cached_context_strength = g_ai_last_context_strength;
   double cached_min_move_points = g_ai_last_min_move_points;
   int cached_horizon_minutes = g_ai_last_horizon_minutes;
   int cached_regime_id = g_ai_last_regime_id;
   g_ai_last_expected_move_points = 0.0;
   g_ai_last_trade_edge_points = 0.0;
   g_ai_last_confidence = 0.0;
   g_ai_last_reliability = 0.0;
   g_ai_last_path_risk = 1.0;
   g_ai_last_fill_risk = 1.0;
   g_ai_last_trade_gate = 0.0;
   g_ai_last_context_quality = 0.0;
   g_ai_last_context_strength = 0.0;
   g_ai_last_min_move_points = 0.0;
   g_ai_last_horizon_minutes = 0;
   g_ai_last_regime_id = 0;
   if(!g_plugins_ready)
   {
      g_ai_last_reason = "plugins_not_ready";
      return -1;
   }

   int base_h = FXAI_ClampHorizon(PredictionTargetMinutes);

   int base = AI_Window;
   if(base < 50) base = 50;
   if(base > 500) base = 500;

   int K = AI_OnlineSamples;
   if(K < 5) K = 5;
   if(K > 200) K = 200;

   int onlineEpochs = AI_OnlineEpochs;
   if(onlineEpochs < 1) onlineEpochs = 1;
   if(onlineEpochs > 5) onlineEpochs = 5;

   int trainEpochs = AI_Epochs;
   if(trainEpochs < 1) trainEpochs = 1;
   if(trainEpochs > 20) trainEpochs = 20;

   int aiType = (int)AI_Type;
   if(aiType < 0 || aiType >= FXAI_AI_COUNT)
      aiType = (int)AI_SGD_LOGIT;

   bool ensembleMode = (bool)AI_Ensemble;
   double agreePct = FXAI_Clamp(Ensemble_AgreePct, 50.0, 100.0);

   double buyThr = AI_BuyThreshold;
   double sellThr = AI_SellThreshold;
   FXAI_SanitizeThresholdPair(buyThr, sellThr);

   double evThresholdPoints = FXAI_Clamp(AI_EVThresholdPoints, 0.0, 100.0);
   int evLookback = AI_EVLookbackSamples;
   if(evLookback < 20) evLookback = 20;
   if(evLookback > 400) evLookback = 400;

   if(g_ai_last_symbol != symbol)
      ResetAIState(symbol);

   if(AI_Warmup && !g_ai_warmup_done)
   {
      if(!FXAI_WarmupTrainAndTune(symbol))
      {
         g_ai_last_reason = "warmup_pending";
         return -1;
      }
   }

   datetime signal_bar = iTime(symbol, PERIOD_M1, 1);
   if(signal_bar == 0)
   {
      g_ai_last_reason = "bar_time_failed";
      return -1;
   }

   int pctKey = (int)MathRound(agreePct * 10.0);
   int decisionKey = (ensembleMode == 1 ? (100000 + pctKey) : aiType);
   if(g_ai_last_signal_bar == signal_bar && g_ai_last_signal_key == decisionKey)
   {
      g_ai_last_expected_move_points = cached_expected_move_points;
      g_ai_last_trade_edge_points = cached_trade_edge_points;
      g_ai_last_confidence = cached_confidence;
      g_ai_last_reliability = cached_reliability;
      g_ai_last_path_risk = cached_path_risk;
      g_ai_last_fill_risk = cached_fill_risk;
      g_ai_last_trade_gate = cached_trade_gate;
      g_ai_last_context_quality = cached_context_quality;
      g_ai_last_context_strength = cached_context_strength;
      g_ai_last_min_move_points = cached_min_move_points;
      g_ai_last_horizon_minutes = cached_horizon_minutes;
      g_ai_last_regime_id = cached_regime_id;
      g_ai_last_reason = "signal_cache_hit";
      return g_ai_last_signal;
   }

   FXAIDataSnapshot snapshot;
   if(!FXAI_ExportDataSnapshot(symbol, AI_CommissionPerLotSide, AI_CostBufferPoints, snapshot))
   {
      g_ai_last_reason = "snapshot_export_failed";
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
      return -1;
   }
   FXAI_ExtractRatesCloseTimeSpread(rates_m1, close_arr, time_arr, spread_m1);
   FXAI_ExtractRatesOHLC(rates_m1, open_arr, high_arr, low_arr, close_arr);
   if(ArraySize(close_arr) < needed || ArraySize(time_arr) < needed || ArraySize(spread_m1) < needed)
   {
      g_ai_last_reason = "m1_series_size_failed";
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
         ArrayResize(ctx_series[s].close, 0);
         ArrayResize(ctx_series[s].time, 0);
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
      }

      ctx_series[s].loaded = FXAI_UpdateRatesRolling(ctx_symbol,
                                                    PERIOD_M1,
                                                    needed,
                                                    ctx_series[s].last_bar_time,
                                                    ctx_series[s].rates);
      if(ctx_series[s].loaded)
      {
         FXAI_ExtractRatesCloseTime(ctx_series[s].rates,
                                   ctx_series[s].close,
                                   ctx_series[s].time);
      }
      else
      {
         ArrayResize(ctx_series[s].close, 0);
         ArrayResize(ctx_series[s].time, 0);
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
   bool run_shadow = (ensembleMode && FXAI_IsShadowBar(shadow_every, signal_seq));
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
      return -1;
   }

   double fallback_expected_move = FXAI_EstimateExpectedAbsMovePoints(close_arr,
                                                                      H,
                                                                      evLookback,
                                                                      snapshot.point);
   if(fallback_expected_move <= 0.0)
      fallback_expected_move = 0.0;
   double vol_proxy_abs = MathAbs(feat_pred[5]);
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

   if(have_online_window && online_start >= 0 && online_start < ArraySize(samples))
      FXAI_AddReplaySample(samples[online_start]);

   int active_ai_ids[];
   ArrayResize(active_ai_ids, 0);
   if(ensembleMode == 0)
   {
      if(g_plugins.Get(aiType) != NULL)
      {
         ArrayResize(active_ai_ids, 1);
         active_ai_ids[0] = aiType;
      }
   }
   else
   {
      int cand_ai_ids[];
      double cand_scores[];
      ArrayResize(cand_ai_ids, 0);
      ArrayResize(cand_scores, 0);

      for(int ai_idx=0; ai_idx<FXAI_AI_COUNT; ai_idx++)
      {
         if(g_plugins.Get(ai_idx) == NULL) continue;
         if(FXAI_IsModelPruned(ai_idx, regime_id)) continue;
         double score = FXAI_GetModelMetaScore(ai_idx, regime_id, min_move_pred);
         if(score <= 0.0) continue;
         int sz = ArraySize(cand_ai_ids);
         ArrayResize(cand_ai_ids, sz + 1);
         ArrayResize(cand_scores, sz + 1);
         cand_ai_ids[sz] = ai_idx;
         cand_scores[sz] = score;
      }

      int cand_n = ArraySize(cand_ai_ids);
      if(cand_n > 0)
      {
         int top_k = cand_n;
         if(top_k > 10) top_k = 10;
         if(top_k < 2 && cand_n >= 2) top_k = 2;

         bool used[];
         ArrayResize(used, cand_n);
         for(int j=0; j<cand_n; j++) used[j] = false;

         ArrayResize(active_ai_ids, top_k);
         int picked = 0;
         for(int pick=0; pick<top_k; pick++)
         {
            int best_j = -1;
            double best_sc = -1e18;
            for(int j=0; j<cand_n; j++)
            {
               if(used[j]) continue;
               if(cand_scores[j] > best_sc)
               {
                  best_sc = cand_scores[j];
                  best_j = j;
               }
            }
            if(best_j < 0) break;
            used[best_j] = true;
            active_ai_ids[picked] = cand_ai_ids[best_j];
            picked++;
         }
         if(picked < ArraySize(active_ai_ids))
            ArrayResize(active_ai_ids, picked);

         // Exploration arm: occasionally add one non-top model (cold-start biased)
         // to improve adaptation and avoid starvation under pruning.
         if(cand_n > picked && FXAI_ShouldSampleByPct(signal_bar, regime_id + 17, Ensemble_ExplorePct))
         {
            int explore_j = -1;
            double explore_sc = -1e18;
            for(int j=0; j<cand_n; j++)
            {
               if(used[j]) continue;
               int ai_id = cand_ai_ids[j];
               int obs = 0;
               if(regime_id >= 0 && regime_id < FXAI_REGIME_COUNT)
                  obs = g_model_regime_obs[ai_id][regime_id];
               double cold_bonus = 1.0 / MathSqrt(1.0 + (double)obs);
               double score = cand_scores[j] * (1.0 + (0.35 * cold_bonus));
               if(score > explore_sc)
               {
                  explore_sc = score;
                  explore_j = j;
               }
            }
            if(explore_j >= 0)
            {
               int add_id = cand_ai_ids[explore_j];
               if(!FXAI_IsModelInList(add_id, active_ai_ids))
               {
                  int sz = ArraySize(active_ai_ids);
                  ArrayResize(active_ai_ids, sz + 1);
                  active_ai_ids[sz] = add_id;
               }
            }
         }
      }
      else
      {
         // If all models are pruned for this regime, keep one conservative fallback.
         int fallback_id = -1;
         double fallback_rel = -1e9;
         for(int ai_idx=0; ai_idx<FXAI_AI_COUNT; ai_idx++)
         {
            if(g_plugins.Get(ai_idx) == NULL) continue;
            double rel = FXAI_GetModelVoteWeight(ai_idx);
            if(rel > fallback_rel)
            {
               fallback_rel = rel;
               fallback_id = ai_idx;
            }
         }
         if(fallback_id >= 0)
         {
            ArrayResize(active_ai_ids, 1);
            active_ai_ids[0] = fallback_id;
         }
      }
   }

   if(ArraySize(active_ai_ids) <= 0)
   {
      g_ai_last_signal_bar = signal_bar;
      g_ai_last_signal_key = decisionKey;
      g_ai_last_signal = -1;
      g_ai_last_reason = "no_active_models";
      return -1;
   }

   FXAINormSampleCache runtime_norm_caches[];
   ArrayResize(runtime_norm_caches, 0);
   FXAINormInputCache input_caches[];
   ArrayResize(input_caches, 0);

   for(int ai_pass=0; ai_pass<FXAI_AI_COUNT; ai_pass++)
   {
      bool needed_ai = false;
      if(FXAI_IsModelInList(ai_pass, active_ai_ids))
         needed_ai = true;
      else if(run_shadow && !FXAI_IsModelInList(ai_pass, active_ai_ids) && g_plugins.Get(ai_pass) != NULL)
         needed_ai = true;
      if(!needed_ai) continue;

      int method_id = (int)FXAI_GetModelNormMethodRouted(ai_pass, regime_id, H);
      if(precompute_end >= 1)
      {
         FXAI_EnsureRoutedNormCachesForSamples(ai_pass,
                                               1,
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
                                               samples,
                                               runtime_norm_caches);
      }

      if(FXAI_FindNormInputCache(method_id, input_caches) < 0)
      {
         FXAI_EnsureNormInputCache(method_id,
                                   spread_pred,
                                   spread_m1,
                                   snapshot,
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
                                   input_caches);
      }
   }

   if(run_shadow && have_shadow_window)
   {
      int warm_epochs = trainEpochs;
      if(warm_epochs > 4) warm_epochs = 4;
      if(warm_epochs < 1) warm_epochs = 1;

      for(int ai_idx=0; ai_idx<FXAI_AI_COUNT; ai_idx++)
      {
         if(FXAI_IsModelInList(ai_idx, active_ai_ids)) continue;
         CFXAIAIPlugin *plugin_shadow = g_plugins.Get(ai_idx);
         if(plugin_shadow == NULL) continue;

         FXAIAIHyperParams hp_shadow;
         FXAI_GetModelHyperParamsRouted(ai_idx, regime_id, H, hp_shadow);
         plugin_shadow.EnsureInitialized(hp_shadow);

         if(!g_ai_trained[ai_idx])
         {
            if(have_init_window)
               FXAI_TrainModelWindowPreparedRoutedCached(ai_idx,
                                                         *plugin_shadow,
                                                         init_start,
                                                         init_end,
                                                         warm_epochs,
                                                         samples,
                                                         runtime_norm_caches);
            FXAI_TrainModelReplay(ai_idx, *plugin_shadow, regime_id, H, 1);
            g_ai_trained[ai_idx] = true;
            g_ai_last_train_bar[ai_idx] = snapshot.bar_time;
         }

         if(snapshot.bar_time != g_ai_last_train_bar[ai_idx])
         {
            FXAI_TrainModelWindowPreparedRoutedCached(ai_idx,
                                                      *plugin_shadow,
                                                      shadow_start,
                                                      shadow_end,
                                                      shadow_epochs,
                                                      samples,
                                                      runtime_norm_caches);
            FXAI_TrainModelReplay(ai_idx, *plugin_shadow, regime_id, H, 1);
            g_ai_last_train_bar[ai_idx] = snapshot.bar_time;
         }
      }
   }

   int singleSignal = -1;
   string singleNoTradeReason = "";
   double ensemble_buy_ev_sum = 0.0;
   double ensemble_sell_ev_sum = 0.0;
   double ensemble_buy_support = 0.0;
   double ensemble_sell_support = 0.0;
   double ensemble_skip_support = 0.0;
   double ensemble_meta_total = 0.0;
   double ensemble_expected_sum = 0.0;
   double ensemble_expected_sq_sum = 0.0;
   double ensemble_conf_sum = 0.0;
   double ensemble_rel_sum = 0.0;
   double ensemble_margin_sum = 0.0;
   double ensemble_hit_time_sum = 0.0;
   double ensemble_path_risk_sum = 0.0;
   double ensemble_fill_risk_sum = 0.0;
   double ensemble_mfe_ratio_sum = 0.0;
   double ensemble_mae_ratio_sum = 0.0;
   double family_support[FXAI_FAMILY_OTHER + 1];
   for(int fam_i=0; fam_i<=FXAI_FAMILY_OTHER; fam_i++) family_support[fam_i] = 0.0;
   double ensemble_probs[3];
   ensemble_probs[0] = 0.3333;
   ensemble_probs[1] = 0.3333;
   ensemble_probs[2] = 0.3334;
   double stack_feat[FXAI_STACK_FEATS];
   for(int sf=0; sf<FXAI_STACK_FEATS; sf++) stack_feat[sf] = 0.0;

   for(int m=0; m<ArraySize(active_ai_ids); m++)
   {
      int ai_idx = active_ai_ids[m];

      CFXAIAIPlugin *plugin = g_plugins.Get(ai_idx);
      if(plugin == NULL)
         continue;

      FXAIAIManifestV4 manifest;
      FXAI_GetPluginManifest(*plugin, manifest);

      FXAIAIHyperParams hp_model;
      FXAI_GetModelHyperParamsRouted(ai_idx, regime_id, H, hp_model);
      plugin.EnsureInitialized(hp_model);

      if(!g_ai_trained[ai_idx])
      {
         if(have_init_window)
         {
            FXAI_TrainModelWindowPreparedRoutedCached(ai_idx,
                                                      *plugin,
                                                      init_start,
                                                      init_end,
                                                      trainEpochs,
                                                      samples,
                                                      runtime_norm_caches);
         }
         FXAI_TrainModelReplay(ai_idx, *plugin, regime_id, H, 1);

         g_ai_trained[ai_idx] = true;
         g_ai_last_train_bar[ai_idx] = snapshot.bar_time;
      }
      else if(snapshot.bar_time != g_ai_last_train_bar[ai_idx])
      {
         if(have_online_window)
         {
            FXAI_TrainModelWindowPreparedRoutedCached(ai_idx,
                                                      *plugin,
                                                      online_start,
                                                      online_end,
                                                      onlineEpochs,
                                                      samples,
                                                      runtime_norm_caches);
         }
         FXAI_TrainModelReplay(ai_idx, *plugin, regime_id, H, 1);

         g_ai_last_train_bar[ai_idx] = snapshot.bar_time;
      }

      FXAIAIPredictRequestV4 req;
      FXAI_ClearPredictRequest(req);
      req.valid = true;
      req.ctx.api_version = FXAI_API_VERSION_V4;
      req.ctx.regime_id = regime_id;
      req.ctx.session_bucket = FXAI_DeriveSessionBucket(snapshot.bar_time);
      req.ctx.horizon_minutes = H;
      req.ctx.feature_schema_id = manifest.feature_schema_id;
      int method_id = (int)FXAI_GetModelNormMethodRouted(ai_idx, regime_id, H);
      req.ctx.normalization_method_id = method_id;
      req.ctx.sequence_bars = FXAI_GetPluginSequenceBars(*plugin, H);
      req.ctx.min_move_points = min_move_pred;
      req.ctx.cost_points = min_move_pred;
      req.ctx.point_value = (_Point > 0.0 ? _Point : 1.0);
      req.ctx.sample_time = snapshot.bar_time;
      int input_idx = FXAI_FindNormInputCache(method_id, input_caches);
      if(input_idx < 0)
      {
         input_idx = FXAI_EnsureNormInputCache(method_id,
                                               spread_pred,
                                               spread_m1,
                                               snapshot,
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
                                               input_caches);
      }
      if(input_idx < 0)
         continue;
      for(int k=0; k<FXAI_AI_WEIGHTS; k++)
         req.x[k] = input_caches[input_idx].x[k];
      FXAI_BuildPreparedSampleWindowCached(ai_idx, samples, 0, runtime_norm_caches, req.ctx.sequence_bars, req.x_window, req.window_size);
      FXAI_ApplyFeatureSchemaToPayloadEx(manifest.feature_schema_id,
                                       manifest.feature_groups_mask,
                                       req.ctx.sequence_bars,
                                       req.x_window,
                                       req.window_size,
                                       req.x);

      FXAIAIPredictionV4 pred;
      FXAI_PredictViaV4(*plugin, req, hp_model, pred);

      double class_probs_pred[3];
      class_probs_pred[0] = pred.class_probs[0];
      class_probs_pred[1] = pred.class_probs[1];
      class_probs_pred[2] = pred.class_probs[2];
      FXAI_ApplyRegimeCalibration(ai_idx, regime_id, class_probs_pred);

      double expected_move = pred.move_mean_points;
      if(expected_move <= 0.0)
         expected_move = FXAI_GetModelExpectedMove(ai_idx, 0.0);
      if(expected_move <= 0.0)
         expected_move = 0.0;

      double modelBuyThr = buyThr;
      double modelSellThr = sellThr;
      FXAI_GetModelThresholds(ai_idx, regime_id, H, buyThr, sellThr, modelBuyThr, modelSellThr);

      double buyMinProb = modelBuyThr;
      double sellMinProb = 1.0 - modelSellThr;
      double skipMinProb = 0.55;
      FXAI_DeriveAdaptiveThresholds(modelBuyThr,
                                   modelSellThr,
                                   min_move_pred,
                                   expected_move,
                                   feat_pred[5],
                                   buyMinProb,
                                   sellMinProb,
                                   skipMinProb);

      int signal = FXAI_ClassSignalFromEV(class_probs_pred,
                                         buyMinProb,
                                         sellMinProb,
                                         skipMinProb,
                                         expected_move,
                                         min_move_pred,
                                         evThresholdPoints);
      if(ensembleMode == 0 && ai_idx == (int)AI_M1SYNC && signal == -1)
      {
         double p_buy = class_probs_pred[(int)FXAI_LABEL_BUY];
         double p_sell = class_probs_pred[(int)FXAI_LABEL_SELL];
         double p_skip = class_probs_pred[(int)FXAI_LABEL_SKIP];
         double buy_ev = ((2.0 * p_buy) - 1.0) * expected_move - min_move_pred;
         double sell_ev = ((2.0 * p_sell) - 1.0) * expected_move - min_move_pred;
         bool have_chain = (expected_move > 0.0 &&
                            (p_buy >= 0.50 || p_sell >= 0.50) &&
                            p_skip < 0.80);

         if(!have_chain)
            singleNoTradeReason = "m1sync_no_chain";
         else if(p_skip >= skipMinProb)
            singleNoTradeReason = "m1sync_chain_skip_block";
         else if((p_buy >= buyMinProb && buy_ev < evThresholdPoints) ||
                 (p_sell >= sellMinProb && sell_ev < evThresholdPoints))
            singleNoTradeReason = "m1sync_chain_ev_blocked";
         else
            singleNoTradeReason = "m1sync_chain_threshold_block";
      }
      FXAI_EnqueueReliabilityPending(ai_idx,
                                     signal_seq,
                                     signal,
                                     regime_id,
                                     expected_move,
                                     H,
                                     class_probs_pred);
      FXAI_EnqueueConformalPending(ai_idx,
                                   signal_seq,
                                   regime_id,
                                   H,
                                   pred);

      if(ensembleMode == 0)
      {
         double buy_ev = ((2.0 * class_probs_pred[(int)FXAI_LABEL_BUY]) - 1.0) * expected_move - min_move_pred;
         double sell_ev = ((2.0 * class_probs_pred[(int)FXAI_LABEL_SELL]) - 1.0) * expected_move - min_move_pred;
         double single_trade_gate = FXAI_Clamp(0.35 +
                                               0.25 * FXAI_Clamp(pred.confidence, 0.0, 1.0) +
                                               0.20 * FXAI_Clamp(pred.reliability, 0.0, 1.0) +
                                               0.10 * (1.0 - FXAI_Clamp(pred.path_risk, 0.0, 1.0)) +
                                               0.10 * (1.0 - FXAI_Clamp(pred.fill_risk, 0.0, 1.0)) +
                                               0.08 * FXAI_Clamp(context_quality, 0.0, 1.5) +
                                               0.07 * FXAI_Clamp(context_strength / 2.0, 0.0, 1.0) -
                                               0.10 * class_probs_pred[(int)FXAI_LABEL_SKIP],
                                               0.0,
                                               1.0);
         double chosen_edge = MathMax(buy_ev, sell_ev);
         if(signal == 1) chosen_edge = buy_ev;
         else if(signal == 0) chosen_edge = sell_ev;
         g_ai_last_expected_move_points = MathMax(expected_move, 0.0);
         g_ai_last_trade_edge_points = chosen_edge;
         g_ai_last_confidence = FXAI_Clamp(pred.confidence, 0.0, 1.0);
         g_ai_last_reliability = FXAI_Clamp(pred.reliability, 0.0, 1.0);
         g_ai_last_path_risk = FXAI_Clamp(pred.path_risk, 0.0, 1.0);
         g_ai_last_fill_risk = FXAI_Clamp(pred.fill_risk, 0.0, 1.0);
         g_ai_last_trade_gate = single_trade_gate;
         singleSignal = signal;
      }
      else
      {
         double meta_w = FXAI_GetModelMetaScore(ai_idx, regime_id, min_move_pred);
         if(meta_w <= 0.0) continue;

         double model_buy_ev = ((2.0 * class_probs_pred[(int)FXAI_LABEL_BUY]) - 1.0) * expected_move - min_move_pred;
         double model_sell_ev = ((2.0 * class_probs_pred[(int)FXAI_LABEL_SELL]) - 1.0) * expected_move - min_move_pred;
         model_buy_ev = FXAI_Clamp(model_buy_ev, -10.0 * min_move_pred, 10.0 * min_move_pred);
         model_sell_ev = FXAI_Clamp(model_sell_ev, -10.0 * min_move_pred, 10.0 * min_move_pred);

         ensemble_meta_total += meta_w;
         ensemble_buy_ev_sum += meta_w * model_buy_ev;
         ensemble_sell_ev_sum += meta_w * model_sell_ev;
         ensemble_expected_sum += meta_w * expected_move;
         ensemble_expected_sq_sum += meta_w * expected_move * expected_move;
         ensemble_conf_sum += meta_w * FXAI_Clamp(pred.confidence, 0.0, 1.0);
         ensemble_rel_sum += meta_w * FXAI_Clamp(pred.reliability, 0.0, 1.0);
         ensemble_margin_sum += meta_w * FXAI_Clamp(MathAbs(class_probs_pred[(int)FXAI_LABEL_BUY] - class_probs_pred[(int)FXAI_LABEL_SELL]), 0.0, 1.0);
         ensemble_hit_time_sum += meta_w * FXAI_Clamp(pred.hit_time_frac, 0.0, 1.0);
         ensemble_path_risk_sum += meta_w * FXAI_Clamp(pred.path_risk, 0.0, 1.0);
         ensemble_fill_risk_sum += meta_w * FXAI_Clamp(pred.fill_risk, 0.0, 1.0);
         ensemble_mfe_ratio_sum += meta_w * FXAI_Clamp(pred.mfe_mean_points / MathMax(expected_move, min_move_pred), 0.0, 4.0);
         ensemble_mae_ratio_sum += meta_w * FXAI_Clamp(pred.mae_mean_points / MathMax(pred.mfe_mean_points, min_move_pred), 0.0, 2.0);
         if(manifest.family >= 0 && manifest.family <= FXAI_FAMILY_OTHER)
            family_support[manifest.family] += meta_w;

         if(signal == 1) ensemble_buy_support += meta_w;
         else if(signal == 0) ensemble_sell_support += meta_w;
         else ensemble_skip_support += meta_w;
      }
   }

   int decision = -1;
   if(ensembleMode == 0)
   {
      decision = singleSignal;
   }
   else
   {
      if(ensemble_meta_total > 0.0)
      {
         double buyPct = 100.0 * (ensemble_buy_support / ensemble_meta_total);
         double sellPct = 100.0 * (ensemble_sell_support / ensemble_meta_total);
         double skipPct = 100.0 * (ensemble_skip_support / ensemble_meta_total);
         double avg_buy_ev = ensemble_buy_ev_sum / ensemble_meta_total;
         double avg_sell_ev = ensemble_sell_ev_sum / ensemble_meta_total;
         double avg_expected = ensemble_expected_sum / ensemble_meta_total;
         double avg_expected_sq = ensemble_expected_sq_sum / ensemble_meta_total;
         double avg_conf = ensemble_conf_sum / ensemble_meta_total;
         double avg_rel = ensemble_rel_sum / ensemble_meta_total;
         double avg_margin = ensemble_margin_sum / ensemble_meta_total;
         double avg_hit_time = ensemble_hit_time_sum / ensemble_meta_total;
         double avg_path_risk = ensemble_path_risk_sum / ensemble_meta_total;
         double avg_fill_risk = ensemble_fill_risk_sum / ensemble_meta_total;
         double avg_mfe_ratio = ensemble_mfe_ratio_sum / ensemble_meta_total;
         double avg_mae_ratio = ensemble_mae_ratio_sum / ensemble_meta_total;
         double move_dispersion = MathSqrt(MathMax(avg_expected_sq - avg_expected * avg_expected, 0.0));
         int active_family_count = 0;
         double dominant_family_support = 0.0;
         for(int fam_i=0; fam_i<=FXAI_FAMILY_OTHER; fam_i++)
         {
            if(family_support[fam_i] > 0.0)
            {
               active_family_count++;
               if(family_support[fam_i] > dominant_family_support)
                  dominant_family_support = family_support[fam_i];
            }
         }
         double active_family_ratio = (double)active_family_count / (double)MathMax(FXAI_FAMILY_OTHER + 1, 1);
         double dominant_family_ratio = dominant_family_support / MathMax(ensemble_meta_total, 1e-6);
         double vote_probs[3];
         vote_probs[(int)FXAI_LABEL_SELL] = FXAI_Clamp(ensemble_sell_support / ensemble_meta_total, 0.0, 1.0);
         vote_probs[(int)FXAI_LABEL_BUY] = FXAI_Clamp(ensemble_buy_support / ensemble_meta_total, 0.0, 1.0);
         vote_probs[(int)FXAI_LABEL_SKIP] = FXAI_Clamp(ensemble_skip_support / ensemble_meta_total, 0.0, 1.0);
         double vs = vote_probs[0] + vote_probs[1] + vote_probs[2];
         if(vs <= 0.0) vs = 1.0;
         vote_probs[0] /= vs; vote_probs[1] /= vs; vote_probs[2] /= vs;

         FXAI_StackBuildFeatures(buyPct,
                                 sellPct,
                                 skipPct,
                                 avg_buy_ev,
                                 avg_sell_ev,
                                 min_move_pred,
                                 avg_expected,
                                 vol_proxy_abs,
                                 H,
                                 avg_conf,
                                 avg_rel,
                                 move_dispersion,
                                 avg_margin,
                                 active_family_ratio,
                                 dominant_family_ratio,
                                 context_strength,
                                 context_quality,
                                 avg_hit_time,
                                 avg_path_risk,
                                 avg_fill_risk,
                                 avg_mfe_ratio,
                                 avg_mae_ratio,
                                 stack_feat);
         double stack_probs_dyn[];
         ArrayResize(stack_probs_dyn, 3);
         FXAI_StackPredict(regime_id, stack_feat, stack_probs_dyn);
         double trade_gate_prob = FXAI_TradeGatePredict(regime_id, H, stack_feat);
         double trade_gate_floor = FXAI_Clamp(0.34 +
                                              0.18 * avg_conf +
                                              0.16 * avg_rel +
                                              0.10 * dominant_family_ratio +
                                              0.08 * FXAI_Clamp(context_quality, 0.0, 1.5) +
                                              0.10 * (1.0 - avg_path_risk) +
                                              0.08 * (1.0 - avg_fill_risk) -
                                              0.10 * vote_probs[(int)FXAI_LABEL_SKIP],
                                              0.05,
                                              0.95);
         double trade_gate = FXAI_Clamp(0.65 * trade_gate_prob + 0.35 * trade_gate_floor, 0.0, 1.0);
         double trade_gate_thr = FXAI_Clamp(0.52 +
                                            0.06 * vote_probs[(int)FXAI_LABEL_SKIP] +
                                            0.05 * FXAI_Clamp(move_dispersion / MathMax(min_move_pred, 0.10), 0.0, 1.0) -
                                            0.05 * avg_conf -
                                            0.04 * avg_rel,
                                            0.42,
                                            0.68);
         double stack_blend = FXAI_Clamp(0.40 + 0.20 * avg_conf + 0.18 * avg_rel + 0.12 * dominant_family_ratio + 0.08 * FXAI_Clamp(context_quality, 0.0, 1.5) - 0.06 * FXAI_Clamp(move_dispersion / MathMax(min_move_pred, 0.10), 0.0, 1.0),
                                         0.45,
                                         0.85);
         ensemble_probs[0] = FXAI_Clamp(stack_blend * stack_probs_dyn[0] + (1.0 - stack_blend) * vote_probs[0], 0.0005, 0.9990);
         ensemble_probs[1] = FXAI_Clamp(stack_blend * stack_probs_dyn[1] + (1.0 - stack_blend) * vote_probs[1], 0.0005, 0.9990);
         ensemble_probs[2] = FXAI_Clamp(stack_blend * stack_probs_dyn[2] + (1.0 - stack_blend) * vote_probs[2], 0.0005, 0.9990);
         double ps = ensemble_probs[0] + ensemble_probs[1] + ensemble_probs[2];
         if(ps <= 0.0) ps = 1.0;
         ensemble_probs[0] /= ps; ensemble_probs[1] /= ps; ensemble_probs[2] /= ps;

         double stack_move = MathMax(avg_expected, 0.0);
         double stack_buy_ev = ((2.0 * ensemble_probs[(int)FXAI_LABEL_BUY]) - 1.0) * stack_move - min_move_pred;
         double stack_sell_ev = ((2.0 * ensemble_probs[(int)FXAI_LABEL_SELL]) - 1.0) * stack_move - min_move_pred;
         double chosen_edge = MathMax(stack_buy_ev, stack_sell_ev);
         g_ai_last_expected_move_points = stack_move;
         g_ai_last_confidence = FXAI_Clamp(avg_conf, 0.0, 1.0);
         g_ai_last_reliability = FXAI_Clamp(avg_rel, 0.0, 1.0);
         g_ai_last_path_risk = FXAI_Clamp(avg_path_risk, 0.0, 1.0);
         g_ai_last_fill_risk = FXAI_Clamp(avg_fill_risk, 0.0, 1.0);
         g_ai_last_trade_gate = trade_gate;

         if(trade_gate < trade_gate_thr)
            decision = -1;
         else if(ensemble_probs[(int)FXAI_LABEL_SKIP] >= 0.58 || skipPct >= 75.0)
            decision = -1;
         else if(ensemble_probs[(int)FXAI_LABEL_BUY] >= ensemble_probs[(int)FXAI_LABEL_SELL] &&
                 buyPct >= agreePct &&
                 stack_buy_ev >= evThresholdPoints &&
                 avg_buy_ev > avg_sell_ev)
            decision = 1;
         else if(ensemble_probs[(int)FXAI_LABEL_SELL] > ensemble_probs[(int)FXAI_LABEL_BUY] &&
                 sellPct >= agreePct &&
                 stack_sell_ev >= evThresholdPoints &&
                 avg_sell_ev > avg_buy_ev)
            decision = 0;
         else
         {
            // Conservative fallback if stack is uncertain.
            if(buyPct >= agreePct && avg_buy_ev >= evThresholdPoints && avg_buy_ev > avg_sell_ev)
               decision = 1;
            else if(sellPct >= agreePct && avg_sell_ev >= evThresholdPoints && avg_sell_ev > avg_buy_ev)
               decision = 0;
         }

         if(decision == 1) chosen_edge = stack_buy_ev;
         else if(decision == 0) chosen_edge = stack_sell_ev;
         g_ai_last_trade_edge_points = chosen_edge;
      }
   }

   if(ensembleMode != 0 && ensemble_meta_total > 0.0)
   {
      double ens_expected = MathMax(min_move_pred,
                                    (ensemble_expected_sum > 0.0 ? ensemble_expected_sum / ensemble_meta_total :
                                     (MathAbs(ensemble_buy_ev_sum) + MathAbs(ensemble_sell_ev_sum)) / MathMax(ensemble_meta_total, 1.0)));
      FXAI_EnqueueStackPending(signal_seq,
                               decision,
                               regime_id,
                               H,
                               ens_expected,
                               ensemble_probs,
                               stack_feat);
   }

   g_ai_last_signal_bar = signal_bar;
   g_ai_last_signal_key = decisionKey;
   g_ai_last_signal = decision;
   if(decision == 1) g_ai_last_reason = "buy";
   else if(decision == 0) g_ai_last_reason = "sell";
   else if(ensembleMode == 0 && aiType == (int)AI_M1SYNC && StringLen(singleNoTradeReason) > 0) g_ai_last_reason = singleNoTradeReason;
   else if(ensembleMode != 0 && ensemble_meta_total <= 0.0) g_ai_last_reason = "no_meta_weight";
   else g_ai_last_reason = "no_consensus_or_ev";

   return decision;
}


#endif // __FXAI_ENGINE_RUNTIME_MQH__
