   ulong router_stage_t0 = GetMicrosecondCount();
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
         double score = FXAI_GetModelMetaScore(ai_idx, regime_id, runtime_session_bucket, H, min_move_pred);
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

   FXAI_RuntimeApplyPerformanceModelCap(active_ai_ids, deploy_profile);
   FXAI_RecordRuntimeStageMs(FXAI_RUNTIME_STAGE_ROUTER,
                             (double)(GetMicrosecondCount() - router_stage_t0) / 1000.0);

   if(ArraySize(active_ai_ids) <= 0)
   {
      g_ai_last_signal_bar = signal_bar;
      g_ai_last_signal_key = decisionKey;
      g_ai_last_signal = -1;
      g_ai_last_reason = "no_active_models";
      FXAI_RuntimePublishIdleSnapshot(symbol);
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

      if(FXAI_FindNormInputCache(method_id, H, input_caches) < 0)
      {
         FXAI_EnsureNormInputCache(method_id,
                                   H,
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

   ulong shadow_stage_t0 = GetMicrosecondCount();
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
   if(run_shadow && have_shadow_window)
      FXAI_RecordRuntimeStageMs(FXAI_RUNTIME_STAGE_SHADOW,
                                (double)(GetMicrosecondCount() - shadow_stage_t0) / 1000.0);

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
   double ensemble_move_q25_sum = 0.0;
   double ensemble_move_q50_sum = 0.0;
   double ensemble_move_q75_sum = 0.0;
   double ensemble_conf_sum = 0.0;
   double ensemble_rel_sum = 0.0;
   double ensemble_margin_sum = 0.0;
   double ensemble_hit_time_sum = 0.0;
   double ensemble_path_risk_sum = 0.0;
   double ensemble_fill_risk_sum = 0.0;
   double ensemble_mfe_ratio_sum = 0.0;
   double ensemble_mae_ratio_sum = 0.0;
   double ensemble_ctx_edge_sum = 0.0;
   double ensemble_ctx_regret_sum = 0.0;
   double ensemble_global_edge_sum = 0.0;
   double ensemble_port_edge_sum = 0.0;
   double ensemble_port_stability_sum = 0.0;
   double ensemble_port_corr_sum = 0.0;
   double ensemble_port_div_sum = 0.0;
   double ensemble_ctx_trust_sum = 0.0;
   double best_model_signal_edge = -1e12;
   double best_model_meta_w = 0.0;
   double best_buy_edge = -1e12;
   double best_sell_edge = -1e12;
   double best_buy_meta_w = 0.0;
   double best_sell_meta_w = 0.0;
   double family_support[FXAI_FAMILY_OTHER + 1];
   for(int fam_i=0; fam_i<=FXAI_FAMILY_OTHER; fam_i++) family_support[fam_i] = 0.0;
   double ensemble_probs[3];
   ensemble_probs[0] = 0.3333;
   ensemble_probs[1] = 0.3333;
   ensemble_probs[2] = 0.3334;
   double stack_feat[FXAI_STACK_FEATS];
   for(int sf=0; sf<FXAI_STACK_FEATS; sf++) stack_feat[sf] = 0.0;
   bool feature_drift_updated = false;
   double live_feature_drift = FXAI_GetFeatureDriftPenalty();
   double drift_norm = FXAI_Clamp(live_feature_drift / 4.0, 0.0, 1.0);
   double routed_meta_weight[FXAI_AI_COUNT];
   bool routed_meta_selected[FXAI_AI_COUNT];
   double adaptive_router_suitability[FXAI_AI_COUNT];
   int adaptive_router_status[FXAI_AI_COUNT];
   FXAIDynamicEnsemblePluginRecord dynamic_records[];
   ArrayResize(dynamic_records, 0);
   FXAIDynamicEnsembleRuntimeState dynamic_ensemble_state;
   FXAI_ResetDynamicEnsembleRuntimeState(dynamic_ensemble_state);
   bool dynamic_ensemble_active = (ensembleMode != 0 && DynamicEnsembleEnabled);
   bool dynamic_ensemble_applied = false;
   for(int ai_i=0; ai_i<FXAI_AI_COUNT; ai_i++)
   {
      routed_meta_weight[ai_i] = -1.0;
      routed_meta_selected[ai_i] = true;
      adaptive_router_suitability[ai_i] = 0.0;
      adaptive_router_status[ai_i] = FXAI_ADAPTIVE_ROUTER_STATUS_ACTIVE;
   }
   bool adaptive_router_active = (AdaptiveRouterEnabled &&
                                  adaptive_router_profile.ready &&
                                  adaptive_router_profile.enabled &&
                                  (!adaptive_router_profile.fallback_to_student_router_only || student_router.ready));
   double route_cutoff = -1.0;
   if(ensembleMode != 0 && (student_router.ready || adaptive_router_active))
   {
      double top_scores[FXAI_AI_COUNT];
      for(int ti=0; ti<FXAI_AI_COUNT; ti++)
         top_scores[ti] = -1.0;
      int candidate_count = 0;
      for(int m=0; m<ArraySize(active_ai_ids); m++)
      {
         int ai_idx = active_ai_ids[m];
         CFXAIAIPlugin *plugin_scan = g_plugins.Get(ai_idx);
         if(plugin_scan == NULL)
            continue;
         FXAIAIManifestV4 scan_manifest;
         FXAI_GetPluginManifest(*plugin_scan, scan_manifest);
         if(student_router.ready &&
            !FXAI_StudentRouterAllowsPlugin(student_router, scan_manifest.ai_name, scan_manifest.family))
         {
            routed_meta_selected[ai_idx] = false;
            continue;
         }
         double base_meta_w = FXAI_GetModelMetaScore(ai_idx, regime_id, runtime_session_bucket, H, min_move_pred);
         double family_weight = (student_router.ready
                                 ? FXAI_StudentRouterFamilyWeight(student_router, scan_manifest.family)
                                 : 1.0);
         double plugin_weight = (student_router.ready
                                 ? FXAI_StudentRouterPluginWeight(student_router, scan_manifest.ai_name, scan_manifest.family)
                                 : 1.0);
         double adaptive_factor = (adaptive_router_active
                                   ? FXAI_AdaptiveRouterComputeSuitability(adaptive_router_profile,
                                                                          adaptive_regime_state,
                                                                          scan_manifest.ai_name)
                                   : 1.0);
         adaptive_router_suitability[ai_idx] = adaptive_factor;
         adaptive_router_status[ai_idx] = (adaptive_router_active
                                           ? FXAI_AdaptiveRouterSuitabilityStatus(adaptive_router_profile,
                                                                                  adaptive_factor)
                                           : FXAI_ADAPTIVE_ROUTER_STATUS_ACTIVE);
         if(adaptive_router_active &&
            adaptive_router_status[ai_idx] == FXAI_ADAPTIVE_ROUTER_STATUS_SUPPRESSED)
         {
            routed_meta_selected[ai_idx] = false;
            routed_meta_weight[ai_idx] = 0.0;
            continue;
         }
         double routed_w = base_meta_w * family_weight * plugin_weight * adaptive_factor;
         routed_meta_weight[ai_idx] = routed_w;
         double min_meta_weight = (student_router.ready
                                   ? student_router.min_meta_weight
                                   : 0.0);
         if(routed_w + 1e-12 < min_meta_weight)
         {
            routed_meta_selected[ai_idx] = false;
            continue;
         }
         candidate_count++;
         for(int ti=0; ti<FXAI_AI_COUNT; ti++)
         {
            if(routed_w > top_scores[ti])
            {
               for(int tj=FXAI_AI_COUNT - 1; tj>ti; tj--)
                  top_scores[tj] = top_scores[tj - 1];
               top_scores[ti] = routed_w;
               break;
            }
         }
      }
      int max_active_models = (student_router.ready ? student_router.max_active_models : ArraySize(active_ai_ids));
      if(max_active_models < 1)
         max_active_models = 1;
      if(candidate_count > max_active_models)
         route_cutoff = top_scores[max_active_models - 1];
      for(int ai_i=0; ai_i<FXAI_AI_COUNT; ai_i++)
      {
         double min_meta_weight = (student_router.ready ? student_router.min_meta_weight : 0.0);
         if(routed_meta_weight[ai_i] < min_meta_weight - 1e-12)
            routed_meta_selected[ai_i] = false;
         if(route_cutoff >= 0.0 &&
            routed_meta_weight[ai_i] >= 0.0 &&
            routed_meta_weight[ai_i] + 1e-12 < route_cutoff)
            routed_meta_selected[ai_i] = false;
      }
   }

   for(int m=0; m<ArraySize(active_ai_ids); m++)
   {
      int ai_idx = active_ai_ids[m];

      CFXAIAIPlugin *plugin = g_plugins.Get(ai_idx);
      if(plugin == NULL)
         continue;

      FXAIAIManifestV4 manifest;
      FXAI_GetPluginManifest(*plugin, manifest);
      if(student_router.ready &&
         !FXAI_StudentRouterAllowsPlugin(student_router, manifest.ai_name, manifest.family))
         continue;
      if(ensembleMode != 0 && (student_router.ready || adaptive_router_active) && !routed_meta_selected[ai_idx])
         continue;

      double adaptive_factor_live = (adaptive_router_active
                                     ? FXAI_AdaptiveRouterComputeSuitability(adaptive_router_profile,
                                                                            adaptive_regime_state,
                                                                            manifest.ai_name)
                                     : 1.0);
      adaptive_router_suitability[ai_idx] = adaptive_factor_live;
      adaptive_router_status[ai_idx] = (adaptive_router_active
                                        ? FXAI_AdaptiveRouterSuitabilityStatus(adaptive_router_profile,
                                                                               adaptive_factor_live)
                                        : FXAI_ADAPTIVE_ROUTER_STATUS_ACTIVE);
      if(adaptive_router_active &&
         adaptive_router_status[ai_idx] == FXAI_ADAPTIVE_ROUTER_STATUS_SUPPRESSED &&
         ensembleMode != 0)
         continue;

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
      req.ctx.point_value = (snapshot.point > 0.0 ? snapshot.point : (_Point > 0.0 ? _Point : 1.0));
      req.ctx.domain_hash = FXAI_SymbolHash01(snapshot.symbol);
      req.ctx.sample_time = snapshot.bar_time;
      int input_idx = FXAI_FindNormInputCache(method_id, H, input_caches);
      if(input_idx < 0)
      {
         input_idx = FXAI_EnsureNormInputCache(method_id,
                                               H,
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
      if(!feature_drift_updated)
      {
         FXAI_UpdateFeatureDriftLiveFromInput(input_caches[input_idx].x, snapshot.bar_time);
         FXAI_MarkRuntimeArtifactsDirty();
         live_feature_drift = FXAI_GetFeatureDriftPenalty();
         drift_norm = FXAI_Clamp(live_feature_drift / 4.0, 0.0, 1.0);
         feature_drift_updated = true;
      }
      for(int k=0; k<FXAI_AI_WEIGHTS; k++)
         req.x[k] = input_caches[input_idx].x[k];
      FXAI_BuildPreparedSampleWindowCached(ai_idx, samples, 0, runtime_norm_caches, req.ctx.sequence_bars, req.x_window, req.window_size);
      if(!FXAI_NormalizationCoreFinalizePredictRequest(manifest, req))
         continue;

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
                                     runtime_session_bucket,
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
                                               0.08 * drift_norm -
                                               0.10 * class_probs_pred[(int)FXAI_LABEL_SKIP],
                                               0.0,
                                               1.0);
         double chosen_edge = MathMax(buy_ev, sell_ev);
         if(signal == 1) chosen_edge = buy_ev;
         else if(signal == 0) chosen_edge = sell_ev;
         g_ai_last_expected_move_points = MathMax(expected_move, 0.0);
         g_ai_last_trade_edge_points = chosen_edge;
         g_ai_last_confidence = FXAI_Clamp(pred.confidence, 0.0, 1.0);
         g_ai_last_reliability = FXAI_Clamp(pred.reliability * (1.0 - 0.15 * drift_norm), 0.0, 1.0);
         g_ai_last_path_risk = FXAI_Clamp(pred.path_risk, 0.0, 1.0);
         g_ai_last_fill_risk = FXAI_Clamp(pred.fill_risk, 0.0, 1.0);
         g_ai_last_trade_gate = single_trade_gate;
         FXAIHierarchicalSignals single_hierarchy_sig;
         FXAI_BuildHierarchicalSignals(class_probs_pred,
                                       expected_move,
                                       min_move_pred,
                                       pred.confidence,
                                       pred.reliability,
                                       pred.path_risk,
                                       pred.fill_risk,
                                       pred.hit_time_frac,
                                       context_quality,
                                       H,
                                       current_foundation_sig,
                                       current_student_sig,
                                       current_analog_q,
                                       single_hierarchy_sig);
         g_ai_last_hierarchy_score = FXAI_Clamp(single_hierarchy_sig.score, 0.0, 1.0);
         g_ai_last_hierarchy_consistency = FXAI_Clamp(single_hierarchy_sig.consistency, 0.0, 1.0);
         g_ai_last_hierarchy_tradability = FXAI_Clamp(single_hierarchy_sig.tradability, 0.0, 1.0);
         g_ai_last_hierarchy_execution = FXAI_Clamp(single_hierarchy_sig.execution_viability, 0.0, 1.0);
         g_ai_last_hierarchy_horizon_fit = FXAI_Clamp(single_hierarchy_sig.horizon_fit, 0.0, 1.0);
         double single_stack_feat[];
         ArrayResize(single_stack_feat, FXAI_STACK_FEATS);
         for(int sf=0; sf<FXAI_STACK_FEATS; sf++)
            single_stack_feat[sf] = 0.0;
         single_stack_feat[1] = FXAI_Clamp(2.0 * class_probs_pred[(int)FXAI_LABEL_BUY] - 1.0, -1.0, 1.0);
         single_stack_feat[2] = FXAI_Clamp(2.0 * class_probs_pred[(int)FXAI_LABEL_SELL] - 1.0, -1.0, 1.0);
         single_stack_feat[3] = FXAI_Clamp(2.0 * class_probs_pred[(int)FXAI_LABEL_SKIP] - 1.0, -1.0, 1.0);
         single_stack_feat[20] = FXAI_Clamp(pred.confidence, 0.0, 1.0);
         single_stack_feat[21] = FXAI_Clamp(pred.reliability, 0.0, 1.0);
         single_stack_feat[49] = FXAI_Clamp(pred.path_risk, 0.0, 1.0);
         single_stack_feat[50] = FXAI_Clamp(pred.fill_risk, 0.0, 1.0);
         single_stack_feat[80] = g_ai_last_hierarchy_score;
         single_stack_feat[81] = g_ai_last_hierarchy_consistency;
         single_stack_feat[82] = g_ai_last_hierarchy_tradability;
         single_stack_feat[83] = g_ai_last_hierarchy_execution;
         double single_policy_feat[];
         double single_policy_pressure = (class_probs_pred[(int)FXAI_LABEL_BUY] >= class_probs_pred[(int)FXAI_LABEL_SELL]
                                          ? cp_buy.score : cp_sell.score);
         FXAI_BuildPolicyFeatures(single_stack_feat,
                                  single_trade_gate,
                                  chosen_edge,
                                  expected_move,
                                  min_move_pred,
                                  g_ai_last_macro_state_quality,
                                  context_quality,
                                  context_strength,
                                  current_foundation_sig.trust,
                                  current_foundation_sig.direction_bias,
                                  current_student_sig.trust,
                                  current_analog_q.similarity,
                                  current_analog_q.quality,
                                  current_regime_q,
                                  deploy_profile,
                                  single_policy_pressure,
                                  single_policy_feat);
         FXAIPolicyDecision single_policy;
         FXAI_PolicyPredict(regime_id, single_policy_feat, deploy_profile, single_policy);
         g_policy_last_trade_prob = single_policy.trade_prob;
         g_policy_last_no_trade_prob = single_policy.no_trade_prob;
         g_policy_last_enter_prob = single_policy.enter_prob;
         g_policy_last_exit_prob = single_policy.exit_prob;
         g_policy_last_direction_bias = single_policy.direction_bias;
         g_policy_last_size_mult = single_policy.size_mult;
         g_policy_last_hold_quality = single_policy.hold_quality;
         g_policy_last_expected_utility = single_policy.expected_utility;
         g_policy_last_confidence = single_policy.confidence;
         g_policy_last_portfolio_fit = single_policy.portfolio_fit;
         g_policy_last_capital_efficiency = single_policy.capital_efficiency;
         g_policy_last_add_prob = single_policy.add_prob;
         g_policy_last_reduce_prob = single_policy.reduce_prob;
         g_policy_last_tighten_prob = single_policy.tighten_prob;
         g_policy_last_timeout_prob = single_policy.timeout_prob;
         g_policy_last_action = single_policy.action_code;
         double single_transition_guard = FXAI_Clamp(1.0 - 0.35 * regime_transition_penalty -
                                                     0.40 * macro_profile_shortfall,
                                                     0.20,
                                                     1.0);
         g_ai_last_trade_gate = FXAI_Clamp((0.40 * single_trade_gate +
                                            0.34 * single_policy.trade_prob +
                                            0.14 * single_policy.enter_prob +
                                            0.12 * single_policy.portfolio_fit) *
                                           single_transition_guard,
                                           0.0,
                                           1.0);
         if(adaptive_router_active)
         {
            g_ai_last_trade_gate = FXAI_Clamp(g_ai_last_trade_gate * FXAI_Clamp(adaptive_factor_live, 0.25, 1.40), 0.0, 1.0);
            g_policy_last_size_mult = FXAI_Clamp(g_policy_last_size_mult * FXAI_Clamp(0.70 + 0.30 * adaptive_factor_live, 0.10, 1.60), 0.10, 1.60);
         }
         if(single_policy.action_code == FXAI_POLICY_ACTION_NO_TRADE ||
            single_policy.no_trade_prob > FXAI_Clamp(deploy_profile.policy_no_trade_cap, 0.25, 0.95))
            signal = -1;
         else if(adaptive_router_active &&
                 adaptive_router_status[ai_idx] == FXAI_ADAPTIVE_ROUTER_STATUS_SUPPRESSED)
         {
            signal = -1;
            singleNoTradeReason = "adaptive_router_suppressed";
         }
         else if(adaptive_router_active &&
                 adaptive_factor_live < adaptive_router_profile.abstain_threshold)
         {
            signal = -1;
            singleNoTradeReason = "adaptive_router_abstain_bias";
         }
         else if(FXAI_MacroEventLeakageSafe() &&
                 g_ai_last_macro_state_quality < FXAI_Clamp(deploy_profile.macro_quality_floor, 0.0, 1.0))
            signal = -1;
         else
         {
            double single_enter_floor = MathMax(FXAI_Clamp(deploy_profile.policy_trade_floor, 0.20, 0.90),
                                                single_policy.no_trade_prob);
            single_enter_floor = MathMax(single_enter_floor, 0.42);
            if(single_policy.enter_prob < single_enter_floor)
               signal = -1;
            else if(single_policy.direction_bias > 0.12 && buy_ev >= sell_ev && buy_ev >= evThresholdPoints)
               signal = 1;
            else if(single_policy.direction_bias < -0.12 && sell_ev >= buy_ev && sell_ev >= evThresholdPoints)
               signal = 0;
         }
         FXAI_EnqueuePolicyPending(signal_seq, regime_id, H, min_move_pred, single_policy_feat);
         singleSignal = signal;
      }
      else
      {
         double meta_w = routed_meta_weight[ai_idx];
         if(meta_w < 0.0)
         {
            meta_w = FXAI_GetModelMetaScore(ai_idx, regime_id, runtime_session_bucket, H, min_move_pred);
            if(student_router.ready)
               meta_w *= FXAI_StudentRouterFamilyWeight(student_router, manifest.family) *
                         FXAI_StudentRouterPluginWeight(student_router, manifest.ai_name, manifest.family);
            if(adaptive_router_active)
               meta_w *= adaptive_factor_live;
         }
         if(meta_w <= 0.0) continue;

         double model_buy_ev = ((2.0 * class_probs_pred[(int)FXAI_LABEL_BUY]) - 1.0) * expected_move - min_move_pred;
         double model_sell_ev = ((2.0 * class_probs_pred[(int)FXAI_LABEL_SELL]) - 1.0) * expected_move - min_move_pred;
         model_buy_ev = FXAI_Clamp(model_buy_ev, -10.0 * min_move_pred, 10.0 * min_move_pred);
         model_sell_ev = FXAI_Clamp(model_sell_ev, -10.0 * min_move_pred, 10.0 * min_move_pred);
         double mm_meta = MathMax(min_move_pred, 0.50);
         double ctx_edge_norm = FXAI_Clamp(FXAI_GetModelContextEdge(ai_idx, regime_id, H) / mm_meta, -4.0, 4.0) / 4.0;
         double ctx_regret = FXAI_Clamp(FXAI_GetModelContextRegret(ai_idx, regime_id, H), 0.0, 6.0) / 6.0;
         double global_edge_norm = FXAI_Clamp(FXAI_GetModelRegimeEdge(ai_idx, regime_id) / mm_meta, -4.0, 4.0) / 4.0;
         double port_edge_norm = FXAI_GetModelPortfolioEdgeNorm(ai_idx, mm_meta);
         double port_stability = FXAI_GetModelPortfolioStability(ai_idx);
         double port_corr = FXAI_GetModelPortfolioCorrPenalty(ai_idx);
         double port_div = FXAI_GetModelPortfolioDiversification(ai_idx);
         double ctx_trust = FXAI_GetModelContextTrust(ai_idx, regime_id, H);
         double model_best_edge = MathMax(model_buy_ev, model_sell_ev);

         if(dynamic_ensemble_active)
         {
            int rec_i = ArraySize(dynamic_records);
            ArrayResize(dynamic_records, rec_i + 1);
            FXAI_ResetDynamicEnsemblePluginRecord(dynamic_records[rec_i]);
            dynamic_records[rec_i].ready = true;
            dynamic_records[rec_i].ai_idx = ai_idx;
            dynamic_records[rec_i].ai_name = manifest.ai_name;
            dynamic_records[rec_i].family_id = manifest.family;
            dynamic_records[rec_i].signal = signal;
            dynamic_records[rec_i].buy_prob = FXAI_Clamp(class_probs_pred[(int)FXAI_LABEL_BUY], 0.0, 1.0);
            dynamic_records[rec_i].sell_prob = FXAI_Clamp(class_probs_pred[(int)FXAI_LABEL_SELL], 0.0, 1.0);
            dynamic_records[rec_i].skip_prob = FXAI_Clamp(class_probs_pred[(int)FXAI_LABEL_SKIP], 0.0, 1.0);
            dynamic_records[rec_i].expected_move = MathMax(expected_move, 0.0);
            dynamic_records[rec_i].move_q25 = MathMax(pred.move_q25_points, 0.0);
            dynamic_records[rec_i].move_q50 = MathMax(pred.move_q50_points, dynamic_records[rec_i].move_q25);
            dynamic_records[rec_i].move_q75 = MathMax(pred.move_q75_points, dynamic_records[rec_i].move_q50);
            dynamic_records[rec_i].confidence = FXAI_Clamp(pred.confidence, 0.0, 1.0);
            dynamic_records[rec_i].reliability = FXAI_Clamp(pred.reliability, 0.0, 1.0);
            dynamic_records[rec_i].margin = FXAI_Clamp(MathAbs(class_probs_pred[(int)FXAI_LABEL_BUY] - class_probs_pred[(int)FXAI_LABEL_SELL]), 0.0, 1.0);
            dynamic_records[rec_i].hit_time_frac = FXAI_Clamp(pred.hit_time_frac, 0.0, 1.0);
            dynamic_records[rec_i].path_risk = FXAI_Clamp(pred.path_risk, 0.0, 1.0);
            dynamic_records[rec_i].fill_risk = FXAI_Clamp(pred.fill_risk, 0.0, 1.0);
            dynamic_records[rec_i].mfe_ratio = FXAI_Clamp(pred.mfe_mean_points / MathMax(expected_move, min_move_pred), 0.0, 4.0);
            dynamic_records[rec_i].mae_ratio = FXAI_Clamp(pred.mae_mean_points / MathMax(pred.mfe_mean_points, min_move_pred), 0.0, 2.0);
            dynamic_records[rec_i].buy_ev = model_buy_ev;
            dynamic_records[rec_i].sell_ev = model_sell_ev;
            dynamic_records[rec_i].base_meta_weight = meta_w;
            dynamic_records[rec_i].adaptive_suitability = adaptive_factor_live;
            dynamic_records[rec_i].adaptive_status = adaptive_router_status[ai_idx];
            dynamic_records[rec_i].ctx_edge_norm = ctx_edge_norm;
            dynamic_records[rec_i].ctx_regret = ctx_regret;
            dynamic_records[rec_i].global_edge_norm = global_edge_norm;
            dynamic_records[rec_i].port_edge_norm = port_edge_norm;
            dynamic_records[rec_i].port_stability = port_stability;
            dynamic_records[rec_i].port_corr = port_corr;
            dynamic_records[rec_i].port_div = port_div;
            dynamic_records[rec_i].ctx_trust = ctx_trust;
         }

         ensemble_meta_total += meta_w;
         ensemble_buy_ev_sum += meta_w * model_buy_ev;
         ensemble_sell_ev_sum += meta_w * model_sell_ev;
         ensemble_expected_sum += meta_w * expected_move;
         ensemble_expected_sq_sum += meta_w * expected_move * expected_move;
         ensemble_move_q25_sum += meta_w * MathMax(pred.move_q25_points, 0.0);
         ensemble_move_q50_sum += meta_w * MathMax(pred.move_q50_points, MathMax(pred.move_q25_points, 0.0));
         ensemble_move_q75_sum += meta_w * MathMax(pred.move_q75_points, MathMax(pred.move_q50_points, pred.move_q25_points));
         ensemble_conf_sum += meta_w * FXAI_Clamp(pred.confidence, 0.0, 1.0);
         ensemble_rel_sum += meta_w * FXAI_Clamp(pred.reliability, 0.0, 1.0);
         ensemble_margin_sum += meta_w * FXAI_Clamp(MathAbs(class_probs_pred[(int)FXAI_LABEL_BUY] - class_probs_pred[(int)FXAI_LABEL_SELL]), 0.0, 1.0);
         ensemble_hit_time_sum += meta_w * FXAI_Clamp(pred.hit_time_frac, 0.0, 1.0);
         ensemble_path_risk_sum += meta_w * FXAI_Clamp(pred.path_risk, 0.0, 1.0);
         ensemble_fill_risk_sum += meta_w * FXAI_Clamp(pred.fill_risk, 0.0, 1.0);
         ensemble_mfe_ratio_sum += meta_w * FXAI_Clamp(pred.mfe_mean_points / MathMax(expected_move, min_move_pred), 0.0, 4.0);
         ensemble_mae_ratio_sum += meta_w * FXAI_Clamp(pred.mae_mean_points / MathMax(pred.mfe_mean_points, min_move_pred), 0.0, 2.0);
         ensemble_ctx_edge_sum += meta_w * ctx_edge_norm;
         ensemble_ctx_regret_sum += meta_w * ctx_regret;
         ensemble_global_edge_sum += meta_w * global_edge_norm;
         ensemble_port_edge_sum += meta_w * port_edge_norm;
         ensemble_port_stability_sum += meta_w * port_stability;
         ensemble_port_corr_sum += meta_w * port_corr;
         ensemble_port_div_sum += meta_w * port_div;
         ensemble_ctx_trust_sum += meta_w * ctx_trust;
         if(manifest.family >= 0 && manifest.family <= FXAI_FAMILY_OTHER)
            family_support[manifest.family] += meta_w;

         if(model_best_edge > best_model_signal_edge)
         {
            best_model_signal_edge = model_best_edge;
            best_model_meta_w = meta_w;
         }
         if(model_buy_ev > best_buy_edge)
         {
            best_buy_edge = model_buy_ev;
            best_buy_meta_w = meta_w;
         }
         if(model_sell_ev > best_sell_edge)
         {
            best_sell_edge = model_sell_ev;
            best_sell_meta_w = meta_w;
         }

         if(signal == 1) ensemble_buy_support += meta_w;
         else if(signal == 0) ensemble_sell_support += meta_w;
         else ensemble_skip_support += meta_w;
      }
   }

   if(ensembleMode != 0 && dynamic_ensemble_active && ArraySize(dynamic_records) > 0)
   {
      if(FXAI_DynamicEnsembleEvaluate(symbol,
                                      snapshot.bar_time,
                                      spread_pred,
                                      min_move_pred,
                                      drift_norm,
                                      adaptive_regime_state,
                                      adaptive_news_state,
                                      adaptive_rates_state,
                                      adaptive_cross_asset_state,
                                      adaptive_micro_state,
                                      dynamic_records,
                                      dynamic_ensemble_state))
      {
         ensemble_buy_ev_sum = 0.0;
         ensemble_sell_ev_sum = 0.0;
         ensemble_buy_support = 0.0;
         ensemble_sell_support = 0.0;
         ensemble_skip_support = 0.0;
         ensemble_meta_total = 0.0;
         ensemble_expected_sum = 0.0;
         ensemble_expected_sq_sum = 0.0;
         ensemble_move_q25_sum = 0.0;
         ensemble_move_q50_sum = 0.0;
         ensemble_move_q75_sum = 0.0;
         ensemble_conf_sum = 0.0;
         ensemble_rel_sum = 0.0;
         ensemble_margin_sum = 0.0;
         ensemble_hit_time_sum = 0.0;
         ensemble_path_risk_sum = 0.0;
         ensemble_fill_risk_sum = 0.0;
         ensemble_mfe_ratio_sum = 0.0;
         ensemble_mae_ratio_sum = 0.0;
         ensemble_ctx_edge_sum = 0.0;
         ensemble_ctx_regret_sum = 0.0;
         ensemble_global_edge_sum = 0.0;
         ensemble_port_edge_sum = 0.0;
         ensemble_port_stability_sum = 0.0;
         ensemble_port_corr_sum = 0.0;
         ensemble_port_div_sum = 0.0;
         ensemble_ctx_trust_sum = 0.0;
         best_model_signal_edge = -1e12;
         best_model_meta_w = 0.0;
         best_buy_edge = -1e12;
         best_sell_edge = -1e12;
         best_buy_meta_w = 0.0;
         best_sell_meta_w = 0.0;
         for(int fam_i=0; fam_i<=FXAI_FAMILY_OTHER; fam_i++)
            family_support[fam_i] = 0.0;

         for(int rec_i=0; rec_i<ArraySize(dynamic_records); rec_i++)
         {
            if(!dynamic_records[rec_i].ready || dynamic_records[rec_i].normalized_weight <= 0.0)
               continue;
            if(dynamic_records[rec_i].status < FXAI_DYNAMIC_ENSEMBLE_STATUS_DOWNWEIGHTED)
               continue;

            double meta_w = dynamic_records[rec_i].normalized_weight;
            double expected_move = MathMax(dynamic_records[rec_i].expected_move, 0.0);
            double model_buy_ev = dynamic_records[rec_i].buy_ev;
            double model_sell_ev = dynamic_records[rec_i].sell_ev;
            double model_best_edge = MathMax(model_buy_ev, model_sell_ev);

            ensemble_meta_total += meta_w;
            ensemble_buy_ev_sum += meta_w * model_buy_ev;
            ensemble_sell_ev_sum += meta_w * model_sell_ev;
            ensemble_expected_sum += meta_w * expected_move;
            ensemble_expected_sq_sum += meta_w * expected_move * expected_move;
            ensemble_move_q25_sum += meta_w * MathMax(dynamic_records[rec_i].move_q25, 0.0);
            ensemble_move_q50_sum += meta_w * MathMax(dynamic_records[rec_i].move_q50, dynamic_records[rec_i].move_q25);
            ensemble_move_q75_sum += meta_w * MathMax(dynamic_records[rec_i].move_q75, dynamic_records[rec_i].move_q50);
            ensemble_conf_sum += meta_w * dynamic_records[rec_i].confidence;
            ensemble_rel_sum += meta_w * dynamic_records[rec_i].reliability;
            ensemble_margin_sum += meta_w * dynamic_records[rec_i].margin;
            ensemble_hit_time_sum += meta_w * dynamic_records[rec_i].hit_time_frac;
            ensemble_path_risk_sum += meta_w * dynamic_records[rec_i].path_risk;
            ensemble_fill_risk_sum += meta_w * dynamic_records[rec_i].fill_risk;
            ensemble_mfe_ratio_sum += meta_w * dynamic_records[rec_i].mfe_ratio;
            ensemble_mae_ratio_sum += meta_w * dynamic_records[rec_i].mae_ratio;
            ensemble_ctx_edge_sum += meta_w * dynamic_records[rec_i].ctx_edge_norm;
            ensemble_ctx_regret_sum += meta_w * dynamic_records[rec_i].ctx_regret;
            ensemble_global_edge_sum += meta_w * dynamic_records[rec_i].global_edge_norm;
            ensemble_port_edge_sum += meta_w * dynamic_records[rec_i].port_edge_norm;
            ensemble_port_stability_sum += meta_w * dynamic_records[rec_i].port_stability;
            ensemble_port_corr_sum += meta_w * dynamic_records[rec_i].port_corr;
            ensemble_port_div_sum += meta_w * dynamic_records[rec_i].port_div;
            ensemble_ctx_trust_sum += meta_w * dynamic_records[rec_i].ctx_trust;

            if(dynamic_records[rec_i].family_id >= 0 && dynamic_records[rec_i].family_id <= FXAI_FAMILY_OTHER)
               family_support[dynamic_records[rec_i].family_id] += meta_w;

            if(model_best_edge > best_model_signal_edge)
            {
               best_model_signal_edge = model_best_edge;
               best_model_meta_w = meta_w;
            }
            if(model_buy_ev > best_buy_edge)
            {
               best_buy_edge = model_buy_ev;
               best_buy_meta_w = meta_w;
            }
            if(model_sell_ev > best_sell_edge)
            {
               best_sell_edge = model_sell_ev;
               best_sell_meta_w = meta_w;
            }

            if(dynamic_records[rec_i].signal == 1) ensemble_buy_support += meta_w;
            else if(dynamic_records[rec_i].signal == 0) ensemble_sell_support += meta_w;
            else ensemble_skip_support += meta_w;
         }
         dynamic_ensemble_applied = (ensemble_meta_total > 0.0);
         if(!dynamic_ensemble_applied)
            dynamic_ensemble_state.fallback_used = true;
      }
      else
      {
         dynamic_ensemble_state.fallback_used = true;
      }
   }
