bool FXAI_AuditRunScenario(CFXAIAIRegistry &registry,
                           const int ai_idx,
                           const FXAIAuditScenarioSpec &spec,
                           const int bars,
                           const int horizon_minutes,
                           const ulong seed,
                           const ENUM_FXAI_FEATURE_NORMALIZATION norm_method,
                           FXAIAuditScenarioMetrics &out)
{
   CFXAIAIPlugin *plugin = registry.CreateInstance(ai_idx);
   if(plugin == NULL) return false;

   FXAIAIManifestV4 manifest;
   FXAI_GetPluginManifest(*plugin, manifest);
   FXAI_AuditResetMetrics(out, ai_idx, manifest.ai_name, manifest.family, spec.name, bars);

   datetime time_arr[];
   double open_arr[];
   double high_arr[];
   double low_arr[];
   double close_arr[];
   int spread_arr[];
   datetime time_m5[];
   double close_m5[];
   int map_m5[];
   datetime time_m15[];
   double close_m15[];
   int map_m15[];
   datetime time_m30[];
   double close_m30[];
   int map_m30[];
   datetime time_h1[];
   double close_h1[];
   int map_h1[];
   double ctx_mean_arr[];
   double ctx_std_arr[];
   double ctx_up_arr[];
   double ctx_extra_arr[];
   double point = (_Point > 0.0 ? _Point : SymbolInfoDouble(_Symbol, SYMBOL_POINT));
   if(point <= 0.0) point = 0.0001;
   FXAIAIHyperParams hp;
   FXAI_AuditDefaultHyperParams(ai_idx, hp);

   bool generated = false;
   ulong scenario_seed = seed + ((ulong)(ai_idx + 1) * (ulong)1315423911);
   if(spec.name == "market_adversarial")
   {
      int adv_seq_bars = FXAI_ResolveManifestSequenceBars(manifest, horizon_minutes);
      int adv_seq_override = FXAI_AuditGetSequenceBarsOverride();
      if(adv_seq_override > 0)
         adv_seq_bars = adv_seq_override;
      int adv_schema_id = manifest.feature_schema_id;
      int adv_schema_override = FXAI_AuditGetSchemaOverride();
      if(adv_schema_override > 0)
         adv_schema_id = adv_schema_override;
      ulong adv_feature_groups_mask = manifest.feature_groups_mask;
      ulong adv_feature_groups_override = FXAI_AuditGetFeatureGroupsMaskOverride();
      if(adv_feature_groups_override != 0)
         adv_feature_groups_mask = adv_feature_groups_override;

      generated = FXAI_AuditGenerateAdversarialScenarioSeries(registry,
                                                              ai_idx,
                                                              bars,
                                                              horizon_minutes,
                                                              scenario_seed,
                                                              point,
                                                              norm_method,
                                                              manifest,
                                                              hp,
                                                              adv_seq_bars,
                                                              adv_schema_id,
                                                              adv_feature_groups_mask,
                                                              time_arr,
                                                              open_arr,
                                                              high_arr,
                                                              low_arr,
                                                              close_arr,
                                                              spread_arr,
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
                                                              ctx_extra_arr);
   }
   else
   {
      generated = FXAI_AuditGenerateScenarioSeries(spec,
                                                   bars,
                                                   scenario_seed,
                                                   point,
                                                   time_arr,
                                                   open_arr,
                                                   high_arr,
                                                   low_arr,
                                                   close_arr,
                                                   spread_arr,
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
                                                   ctx_extra_arr);
   }
   if(!generated)
   {
      delete plugin;
      return false;
   }

   FXAI_ResetNormalizationWindows(192);
   FXAI_ResetFeatureNormalizationState();
   if(plugin.SupportsSyntheticSeries())
      plugin.SetSyntheticSeries(time_arr, open_arr, high_arr, low_arr, close_arr);

   plugin.EnsureInitialized(hp);

   string runtime_scope = "AUDIT_" + _Symbol + "_" + manifest.ai_name + "_" + spec.name + "_" +
                          IntegerToString(ai_idx) + "_" + IntegerToString(bars) + "_" + IntegerToString(horizon_minutes);
   FXAI_AuditBindRuntimePlugin(plugin);
   FXAI_ResetConformalState();
   if(!FXAI_SaveRuntimeArtifacts(runtime_scope) || !FXAI_LoadRuntimeArtifacts(runtime_scope))
   {
      plugin.ClearSyntheticSeries();
      FXAI_AuditBindRuntimePlugin(NULL);
      delete plugin;
      return false;
   }

   int seq_bars = FXAI_ResolveManifestSequenceBars(manifest, horizon_minutes);
   int seq_override = FXAI_AuditGetSequenceBarsOverride();
   if(seq_override > 0) seq_bars = seq_override;
   int schema_id = manifest.feature_schema_id;
   int schema_override = FXAI_AuditGetSchemaOverride();
   if(schema_override > 0) schema_id = schema_override;
   ulong feature_groups_mask = manifest.feature_groups_mask;
   ulong feature_groups_override = FXAI_AuditGetFeatureGroupsMaskOverride();
   if(feature_groups_override != 0) feature_groups_mask = feature_groups_override;

   int n = ArraySize(close_arr);
   int start_idx = horizon_minutes + 1;
   int end_idx = n - 220;
   if(end_idx <= start_idx) end_idx = n - 32;
   if(end_idx <= start_idx) end_idx = n - 2;

   int wf_train_bars = FXAI_AuditGetWalkForwardTrainBars();
   int wf_test_bars = FXAI_AuditGetWalkForwardTestBars();
   int wf_purge_bars = FXAI_AuditGetWalkForwardPurgeBars();
   int wf_embargo_bars = FXAI_AuditGetWalkForwardEmbargoBars();
   int wf_folds = FXAI_AuditGetWalkForwardFolds();
   if(wf_train_bars < 96) wf_train_bars = 96;
   if(wf_test_bars < 24) wf_test_bars = 24;
   if(wf_purge_bars < horizon_minutes) wf_purge_bars = horizon_minutes;

   int wf_val_bars = MathMax(24, MathMin(wf_test_bars, wf_train_bars / 3));
   int wf_train_core_bars = wf_train_bars - wf_val_bars;
   if(wf_train_core_bars < 64)
   {
      wf_train_core_bars = 64;
      wf_val_bars = MathMax(16, wf_train_bars - wf_train_core_bars);
   }
   int wf_cycle = wf_train_core_bars + wf_val_bars + wf_purge_bars + wf_test_bars + wf_embargo_bars;

   FXAIAuditFoldMetrics wf_train_fold[];
   FXAIAuditFoldMetrics wf_test_fold[];
   if(spec.name == "market_walkforward")
   {
      ArrayResize(wf_train_fold, wf_folds);
      ArrayResize(wf_test_fold, wf_folds);
      for(int f=0; f<wf_folds; f++)
      {
         FXAI_AuditResetFoldMetrics(wf_train_fold[f]);
         FXAI_AuditResetFoldMetrics(wf_test_fold[f]);
      }
   }

   FXAIAIPredictionV4 held_pred_reset;
   FXAIAIPredictRequestV4 held_req;
   held_req.valid = false;
   bool held_req_ready = false;
   int held_req_idx = -1;
   int current_wf_fold = -1;

   for(int i=start_idx; i<end_idx; i++)
   {
      bool train_enabled = true;
      bool eval_enabled = true;
      bool track_overall_eval = true;
      int eval_bucket = 0;
      int fold_idx = -1;

      if(spec.name == "market_walkforward")
      {
         train_enabled = false;
         eval_enabled = false;
         track_overall_eval = false;

         int cycle = MathMax(wf_cycle, 1);
         int offset = i - start_idx;
         if(offset < 0)
            continue;

         fold_idx = offset / cycle;
         if(fold_idx < 0 || fold_idx >= wf_folds)
            continue;

         if(fold_idx != current_wf_fold)
         {
            current_wf_fold = fold_idx;
            plugin.Reset();
            plugin.EnsureInitialized(hp);
            plugin.ClearSyntheticSeries();
            if(plugin.SupportsSyntheticSeries())
               plugin.SetSyntheticSeries(time_arr, open_arr, high_arr, low_arr, close_arr);
            FXAI_ResetConformalState();
            if(!FXAI_SaveRuntimeArtifacts(runtime_scope))
            {
               plugin.ClearSyntheticSeries();
               FXAI_AuditBindRuntimePlugin(NULL);
               delete plugin;
               return false;
            }
         }

         int phase = offset % cycle;
         if(phase < wf_train_core_bars)
         {
            train_enabled = true;
         }
         else if(phase < wf_train_core_bars + wf_val_bars)
         {
            eval_enabled = true;
            eval_bucket = 1;
         }
         else if(phase < wf_train_core_bars + wf_val_bars + wf_purge_bars)
         {
         }
         else if(phase < wf_train_core_bars + wf_val_bars + wf_purge_bars + wf_test_bars)
         {
            eval_enabled = true;
            eval_bucket = 2;
            track_overall_eval = true;
         }
         else
         {
         }

         if(!train_enabled && !eval_enabled)
            continue;
      }

      FXAIAIContextV4 ctx;
      int label_class = (int)FXAI_LABEL_SKIP;
      double move_points = 0.0;
      double mfe_points = 0.0;
      double mae_points = 0.0;
      double time_to_hit_frac = 1.0;
      int path_flags = 0;
      double spread_stress = 0.0;
      FXAIExecutionTraceStats trace_stats;
      double sample_weight = 1.0;
      double x[];
      if(!FXAI_AuditBuildSample(i,
                                horizon_minutes,
                                point,
                                0.25,
                                norm_method,
                                time_arr,
                                open_arr,
                                high_arr,
                                low_arr,
                                close_arr,
                                spread_arr,
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
                                ctx,
                                label_class,
                                move_points,
                                mfe_points,
                                mae_points,
                                time_to_hit_frac,
                                path_flags,
                                spread_stress,
                                trace_stats,
                                sample_weight,
                                x))
         continue;

      ctx.sequence_bars = seq_bars;
      ctx.feature_schema_id = schema_id;

      if(track_overall_eval)
      {
        out.samples_total++;
        if(label_class == (int)FXAI_LABEL_BUY) out.true_buy_count++;
        else if(label_class == (int)FXAI_LABEL_SELL) out.true_sell_count++;
        else out.true_skip_count++;
      }

      FXAIAIPredictRequestV4 req;
      FXAI_ClearPredictRequest(req);
      req.valid = true;
      req.ctx = ctx;
      for(int k=0; k<FXAI_AI_WEIGHTS; k++)
         req.x[k] = x[k];
      FXAI_AuditBuildWindow(i,
                            req.ctx.sequence_bars,
                            horizon_minutes,
                            point,
                            0.25,
                            norm_method,
                            time_arr,
                            open_arr,
                            high_arr,
                            low_arr,
                            close_arr,
                            spread_arr,
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
                            req.x_window,
                            req.window_size);
      FXAI_ApplyFeatureSchemaToPayloadEx(schema_id,
                                         feature_groups_mask,
                                         req.ctx.sequence_bars,
                                         req.x_window,
                                         req.window_size,
                                         req.x);

      if(track_overall_eval)
      {
         double macro_pre = FXAI_GetInputFeature(req.x, FXAI_MACRO_EVENT_FEATURE_OFFSET + 0);
         double macro_post = FXAI_GetInputFeature(req.x, FXAI_MACRO_EVENT_FEATURE_OFFSET + 1);
         double macro_importance = FXAI_GetInputFeature(req.x, FXAI_MACRO_EVENT_FEATURE_OFFSET + 2);
         double macro_surprise_abs = FXAI_GetInputFeature(req.x, FXAI_MACRO_EVENT_FEATURE_OFFSET + 4);
         double macro_surprise_z_abs = MathAbs(FXAI_GetInputFeature(req.x, FXAI_MACRO_EVENT_FEATURE_OFFSET + 6));
         double macro_revision_abs = FXAI_GetInputFeature(req.x, FXAI_MACRO_EVENT_FEATURE_OFFSET + 7);
         double macro_currency_relevance = FXAI_GetInputFeature(req.x, FXAI_MACRO_EVENT_FEATURE_OFFSET + 8);
         double macro_provenance_trust = FXAI_GetInputFeature(req.x, FXAI_MACRO_EVENT_FEATURE_OFFSET + 9);
         double macro_activity = MathMax(macro_importance, MathMax(macro_pre, macro_post));
         out.macro_event_rate += (macro_activity > 0.05 ? 1.0 : 0.0);
         out.macro_pre_rate += (macro_pre > 0.05 ? 1.0 : 0.0);
         out.macro_post_rate += (macro_post > 0.05 ? 1.0 : 0.0);
         out.macro_importance_mean += macro_importance;
         out.macro_surprise_abs_mean += macro_surprise_abs;
         out.macro_data_coverage += FXAI_Clamp(macro_activity, 0.0, 1.0);
         out.macro_surprise_z_abs_mean += macro_surprise_z_abs;
         out.macro_revision_abs_mean += macro_revision_abs;
         out.macro_currency_relevance_mean += macro_currency_relevance;
         out.macro_provenance_trust_mean += macro_provenance_trust;
         out.macro_rates_rate += (FXAI_GetInputFeature(req.x, FXAI_MACRO_EVENT_FEATURE_OFFSET + 10) > 0.05 ? 1.0 : 0.0);
         out.macro_inflation_rate += (FXAI_GetInputFeature(req.x, FXAI_MACRO_EVENT_FEATURE_OFFSET + 11) > 0.05 ? 1.0 : 0.0);
         out.macro_labor_rate += (FXAI_GetInputFeature(req.x, FXAI_MACRO_EVENT_FEATURE_OFFSET + 12) > 0.05 ? 1.0 : 0.0);
         out.macro_growth_rate += (FXAI_GetInputFeature(req.x, FXAI_MACRO_EVENT_FEATURE_OFFSET + 13) > 0.05 ? 1.0 : 0.0);
      }

      if(eval_enabled)
      {
         FXAIAIPredictionV4 pred;
         bool ok = FXAI_PredictViaV4(*plugin, req, hp, pred);
         string pred_reason = "";
         bool pred_valid = FXAI_ValidatePredictionV4(pred, pred_reason);

         if(!ok || !pred_valid)
         {
            if(track_overall_eval)
               out.invalid_preds++;

            if(spec.name == "market_walkforward" && fold_idx >= 0)
            {
               if(eval_bucket == 1) FXAI_AuditFoldInvalid(wf_train_fold[fold_idx]);
               else if(eval_bucket == 2) FXAI_AuditFoldInvalid(wf_test_fold[fold_idx]);
            }
         }
         else
         {
            int decision = FXAI_AuditDecisionFromPred(pred);

            double target_probs[3] = {0.0, 0.0, 0.0};
            int cls_idx = label_class;
            if(cls_idx < (int)FXAI_LABEL_SELL || cls_idx > (int)FXAI_LABEL_SKIP)
               cls_idx = (move_points >= 0.0 ? (int)FXAI_LABEL_BUY : (int)FXAI_LABEL_SELL);
            target_probs[cls_idx] = 1.0;

            double brier = 0.0;
            for(int c=0; c<3; c++)
            {
               double d = pred.class_probs[c] - target_probs[c];
               brier += d * d;
            }

            bool directional_eval = (decision != (int)FXAI_LABEL_SKIP);
            double dir_conf = 0.0;
            bool dir_ok = false;
            double calibration_abs = 0.0;
            double path_quality = -1.0;
            double net_points = 0.0;
            if(directional_eval)
            {
               dir_conf = MathMax(pred.class_probs[(int)FXAI_LABEL_BUY], pred.class_probs[(int)FXAI_LABEL_SELL]);
               dir_ok = ((decision == (int)FXAI_LABEL_BUY && label_class == (int)FXAI_LABEL_BUY) ||
                         (decision == (int)FXAI_LABEL_SELL && label_class == (int)FXAI_LABEL_SELL));
               calibration_abs = MathAbs(dir_conf - (dir_ok ? 1.0 : 0.0));
               double move_scale = MathMax(MathAbs(move_points), MathMax(MathAbs(pred.move_mean_points), 0.50));
               path_quality = 0.25 * FXAI_Clamp(MathAbs(pred.mfe_mean_points - mfe_points) / move_scale, 0.0, 3.0) +
                              0.20 * FXAI_Clamp(MathAbs(pred.mae_mean_points - mae_points) / move_scale, 0.0, 3.0) +
                              0.20 * MathAbs(pred.hit_time_frac - time_to_hit_frac) +
                              0.20 * MathAbs(pred.path_risk - spread_stress) +
                              0.15 * MathAbs(pred.fill_risk - FXAI_Clamp(spread_stress + (((path_flags & FXAI_PATHFLAG_DUAL_HIT) != 0) ? 0.25 : 0.0), 0.0, 1.0));
               net_points = FXAI_AuditRealizedNetPointsForSignalReplayTrace(decision,
                                                                            move_points,
                                                                            ctx.min_move_points,
                                                                            horizon_minutes,
                                                                            spread_stress,
                                                                            path_flags,
                                                                            trace_stats,
                                                                            ctx.sample_time,
                                                                            spec.id);
            }

            if(track_overall_eval)
            {
               out.valid_preds++;
               if(decision == (int)FXAI_LABEL_BUY) out.buy_count++;
               else if(decision == (int)FXAI_LABEL_SELL) out.sell_count++;
               else out.skip_count++;

               if(decision == label_class) out.exact_match_count++;

               if(spec.name == "drift_up" || spec.name == "monotonic_up")
               {
                  if(decision == (int)FXAI_LABEL_BUY) out.trend_alignment_sum += 1.0;
                  else if(decision == (int)FXAI_LABEL_SELL) out.trend_alignment_sum -= 1.0;
                  out.trend_alignment_count++;
               }
               else if(spec.name == "drift_down" || spec.name == "monotonic_down")
               {
                  if(decision == (int)FXAI_LABEL_SELL) out.trend_alignment_sum += 1.0;
                  else if(decision == (int)FXAI_LABEL_BUY) out.trend_alignment_sum -= 1.0;
                  out.trend_alignment_count++;
               }
               else if(spec.name == "market_trend" || spec.name == "market_walkforward")
               {
                  if(label_class == (int)FXAI_LABEL_BUY)
                  {
                     if(decision == (int)FXAI_LABEL_BUY) out.trend_alignment_sum += 1.0;
                     else if(decision == (int)FXAI_LABEL_SELL) out.trend_alignment_sum -= 1.0;
                     out.trend_alignment_count++;
                  }
                  else if(label_class == (int)FXAI_LABEL_SELL)
                  {
                     if(decision == (int)FXAI_LABEL_SELL) out.trend_alignment_sum += 1.0;
                     else if(decision == (int)FXAI_LABEL_BUY) out.trend_alignment_sum -= 1.0;
                     out.trend_alignment_count++;
                  }
               }

               out.conf_sum += pred.confidence;
               out.rel_sum += pred.reliability;
               out.move_sum += pred.move_mean_points;
               out.brier_sum += brier;
               out.net_sum += net_points;

               if(directional_eval)
               {
                  out.directional_eval_count++;
                  out.dir_conf_sum += dir_conf;
                  if(dir_ok) out.directional_correct_count++;
                  out.dir_hit_sum += (dir_ok ? 1.0 : 0.0);
                  out.calibration_abs_sum += calibration_abs;
                  out.path_quality_abs_sum += path_quality;
                  out.path_quality_count++;
               }
            }

            if(spec.name == "market_walkforward" && fold_idx >= 0)
            {
               if(eval_bucket == 1)
                  FXAI_AuditFoldValid(wf_train_fold[fold_idx], decision, pred, brier, net_points, directional_eval, dir_ok, calibration_abs, path_quality);
               else if(eval_bucket == 2)
                  FXAI_AuditFoldValid(wf_test_fold[fold_idx], decision, pred, brier, net_points, directional_eval, dir_ok, calibration_abs, path_quality);
            }

            FXAI_EnqueueConformalPending(ai_idx,
                                         i,
                                         ctx.regime_id,
                                         horizon_minutes,
                                         pred);
            FXAI_UpdateConformalFromPending(ai_idx,
                                            (long)i,
                                            ctx.regime_id,
                                            horizon_minutes,
                                            label_class,
                                            move_points,
                                            mfe_points,
                                            mae_points,
                                            time_to_hit_frac,
                                            path_flags,
                                            spread_stress,
                                            FXAI_AuditGetCommissionPerLotSide(),
                                            FXAI_AuditGetCostBufferPoints(),
                                            ctx.min_move_points);
            FXAI_MaybeSaveRuntimeArtifacts(runtime_scope, ctx.sample_time);
         }
      }

      if(train_enabled)
      {
         FXAIAITrainRequestV4 train_req;
         FXAI_ClearTrainRequest(train_req);
         train_req.valid = true;
         train_req.ctx = ctx;
         train_req.label_class = label_class;
         train_req.move_points = move_points;
         train_req.sample_weight = sample_weight;
         FXAI_SetTrainRequestPathTargets(train_req,
                                         mfe_points,
                                         mae_points,
                                         time_to_hit_frac,
                                         path_flags,
                                         spread_stress);
         double masked_target = 0.0;
         double next_vol_target = MathAbs(move_points);
         double regime_shift_target = ((path_flags & FXAI_PATHFLAG_DUAL_HIT) != 0 ? 1.0 : 0.0);
         double context_lead_target = FXAI_Clamp(0.5 + 0.5 * FXAI_Sign(FXAI_GetInputFeature(x, 10)) * FXAI_Sign(move_points), 0.0, 1.0);
         FXAI_SetTrainRequestAuxTargets(train_req,
                                        masked_target,
                                        next_vol_target,
                                        regime_shift_target,
                                        context_lead_target);
         for(int k=0; k<FXAI_AI_WEIGHTS; k++)
            train_req.x[k] = x[k];
         FXAI_CopyWindowPayload(req.x_window, req.window_size, train_req.x_window, train_req.window_size);
         FXAI_ApplyFeatureSchemaToPayloadEx(schema_id,
                                            feature_groups_mask,
                                            train_req.ctx.sequence_bars,
                                            train_req.x_window,
                                            train_req.window_size,
                                            train_req.x);
         FXAI_TrainViaV4(*plugin, train_req, hp);
      }

      if(track_overall_eval)
      {
         held_req = req;
         held_req_ready = true;
         held_req_idx = i;
      }
   }

   if(spec.name == "market_walkforward")
      FXAI_AuditFinalizeWalkForward(out, wf_train_fold, wf_test_fold);

   if(held_req_ready)
   {
      bool held_ok = FXAI_PredictViaV4(*plugin, held_req, hp, held_pred_reset);
      string held_reason = "";
      bool held_valid = (held_ok && FXAI_ValidatePredictionV4(held_pred_reset, held_reason));
      plugin.ResetState((int)FXAI_RESET_MANUAL, held_req.ctx.sample_time);
      if(plugin.SupportsSyntheticSeries())
         plugin.SetSyntheticSeries(time_arr, open_arr, high_arr, low_arr, close_arr);
      FXAIAIPredictionV4 pred_after_reset;
      bool reset_ok = FXAI_PredictViaV4(*plugin, held_req, hp, pred_after_reset);
      string reset_reason = "";
      bool reset_valid = (reset_ok && FXAI_ValidatePredictionV4(pred_after_reset, reset_reason));
      if(held_valid && reset_valid)
         FXAI_AuditComparePredictions(held_pred_reset, pred_after_reset, out.reset_delta);
      else
         out.reset_delta = -1.0;

      if(FXAI_HasCapability(manifest.capability_mask, FXAI_CAP_WINDOW_CONTEXT) ||
         FXAI_HasCapability(manifest.capability_mask, FXAI_CAP_STATEFUL))
      {
         FXAIAIPredictRequestV4 seq_short = held_req;
         seq_short.ctx.sequence_bars = 1;
         seq_short.window_size = 0;
         FXAI_ClearInputWindow(seq_short.x_window, seq_short.window_size);
         FXAI_ApplyFeatureSchemaToPayloadEx(schema_id,
                                            feature_groups_mask,
                                            seq_short.ctx.sequence_bars,
                                            seq_short.x_window,
                                            seq_short.window_size,
                                            seq_short.x);

         FXAIAIPredictRequestV4 seq_long = held_req;
         seq_long.ctx.sequence_bars = seq_bars;
         if(held_req_idx >= 0)
         {
            FXAI_AuditBuildWindow(held_req_idx,
                                  seq_long.ctx.sequence_bars,
                                  horizon_minutes,
                                  point,
                                  0.25,
                                  norm_method,
                                  time_arr,
                                  open_arr,
                                  high_arr,
                                  low_arr,
                                  close_arr,
                                  spread_arr,
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
                                  seq_long.x_window,
                                  seq_long.window_size);
            FXAI_ApplyFeatureSchemaToPayloadEx(schema_id,
                                               feature_groups_mask,
                                               seq_long.ctx.sequence_bars,
                                               seq_long.x_window,
                                               seq_long.window_size,
                                               seq_long.x);
            plugin.ResetState((int)FXAI_RESET_MANUAL, held_req.ctx.sample_time);
            if(plugin.SupportsSyntheticSeries())
               plugin.SetSyntheticSeries(time_arr, open_arr, high_arr, low_arr, close_arr);
            FXAIAIPredictionV4 pred_short;
            bool short_ok = FXAI_PredictViaV4(*plugin, seq_short, hp, pred_short);
            string short_reason = "";
            bool short_valid = (short_ok && FXAI_ValidatePredictionV4(pred_short, short_reason));

            plugin.ResetState((int)FXAI_RESET_MANUAL, held_req.ctx.sample_time);
            if(plugin.SupportsSyntheticSeries())
               plugin.SetSyntheticSeries(time_arr, open_arr, high_arr, low_arr, close_arr);
            FXAIAIPredictionV4 pred_long;
            bool long_ok = FXAI_PredictViaV4(*plugin, seq_long, hp, pred_long);
            string long_reason = "";
            bool long_valid = (long_ok && FXAI_ValidatePredictionV4(pred_long, long_reason));

            if(short_valid && long_valid)
               FXAI_AuditComparePredictions(pred_short, pred_long, out.sequence_delta);
            else
               out.sequence_delta = -1.0;
         }
         else
         {
            out.sequence_delta = -1.0;
         }
      }
      else
      {
         out.sequence_delta = -1.0;
      }
   }
   else
   {
      out.reset_delta = -1.0;
      out.sequence_delta = -1.0;
   }

   FXAI_AuditFinalizeMetrics(out);
   if(!FXAI_SaveRuntimeArtifacts(runtime_scope))
   {
      plugin.ClearSyntheticSeries();
      FXAI_AuditBindRuntimePlugin(NULL);
      delete plugin;
      return false;
   }
   plugin.ClearSyntheticSeries();
   FXAI_AuditBindRuntimePlugin(NULL);
   delete plugin;
   return true;
}

