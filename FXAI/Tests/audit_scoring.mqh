#ifndef __FXAI_AUDIT_SCORING_MQH__
#define __FXAI_AUDIT_SCORING_MQH__

void FXAI_AuditComparePredictions(const FXAIAIPredictionV4 &a,
                                  const FXAIAIPredictionV4 &b,
                                  double &delta_out)
{
   delta_out = 0.0;
   for(int c=0; c<3; c++)
      delta_out += MathAbs(a.class_probs[c] - b.class_probs[c]);
   delta_out += 0.10 * MathAbs(a.move_mean_points - b.move_mean_points);
   delta_out += 0.04 * MathAbs(a.mfe_mean_points - b.mfe_mean_points);
   delta_out += 0.04 * MathAbs(a.mae_mean_points - b.mae_mean_points);
   delta_out += 0.05 * MathAbs(a.hit_time_frac - b.hit_time_frac);
   delta_out += 0.05 * MathAbs(a.path_risk - b.path_risk);
   delta_out += 0.05 * MathAbs(a.fill_risk - b.fill_risk);
   delta_out += 0.05 * MathAbs(a.confidence - b.confidence);
   delta_out += 0.05 * MathAbs(a.reliability - b.reliability);
}

void FXAI_AuditFinalizeMetrics(FXAIAuditScenarioMetrics &m)
{
   if(m.samples_total > 0)
   {
      m.skip_ratio = (double)m.skip_count / (double)m.samples_total;
      m.active_ratio = (double)(m.buy_count + m.sell_count) / (double)m.samples_total;
   }
   int active = m.buy_count + m.sell_count;
   if(active > 0)
      m.bias_abs = MathAbs((double)m.buy_count - (double)m.sell_count) / (double)active;
   if(m.directional_eval_count > 0)
   {
      double avg_conf = m.dir_conf_sum / (double)m.directional_eval_count;
      double avg_hit = m.dir_hit_sum / (double)m.directional_eval_count;
      m.conf_drift = MathAbs(avg_conf - avg_hit);
      m.calibration_error = m.calibration_abs_sum / (double)m.directional_eval_count;
   }
   if(m.valid_preds > 0)
      m.brier_score = m.brier_sum / (double)m.valid_preds;
   if(m.path_quality_count > 0)
      m.path_quality_error = m.path_quality_abs_sum / (double)m.path_quality_count;

   double score = 100.0;
   if(m.invalid_preds > 0) score -= 35.0;
   if(m.skip_ratio < 0.45 && (m.scenario == "random_walk" || m.scenario == "market_chop" || m.scenario == "market_spread_shock")) score -= 18.0;
   if(m.active_ratio > 0.80 && (m.scenario == "random_walk" || m.scenario == "market_chop" || m.scenario == "market_spread_shock")) score -= 12.0;
   if(m.trend_alignment_count > 0)
   {
      double align = m.trend_alignment_sum / (double)m.trend_alignment_count;
      if((m.scenario == "drift_up" || m.scenario == "drift_down" || m.scenario == "monotonic_up" || m.scenario == "monotonic_down" || m.scenario == "market_trend" || m.scenario == "market_walkforward") && align < 0.20)
         score -= 18.0;
   }
   if(m.scenario == "market_session_edges" && m.conf_drift > 0.18) score -= 8.0;
   if(m.conf_drift > 0.22) score -= 10.0;
   if(m.brier_score > 0.52) score -= 8.0;
   if(m.calibration_error > 0.28) score -= 8.0;
   if(m.path_quality_error > 0.55) score -= 8.0;
   if(m.reset_delta > 0.30) score -= 12.0;
   if(m.sequence_delta < 0.005 && m.sequence_delta >= 0.0) score -= 6.0;
   if(m.move_sum <= 0.0) score -= 8.0;
   if(score < 0.0) score = 0.0;
   m.score = score;

   if(m.invalid_preds > 0) m.issue_flags |= FXAI_AUDIT_ISSUE_INVALID_PRED;
   if((m.scenario == "random_walk" || m.scenario == "market_chop" || m.scenario == "market_spread_shock" || m.scenario == "market_session_edges") && (m.skip_ratio < 0.55 || m.active_ratio > 0.70))
      m.issue_flags |= FXAI_AUDIT_ISSUE_OVERTRADES_NOISE;
   if((m.scenario == "drift_up" || m.scenario == "drift_down" || m.scenario == "monotonic_up" || m.scenario == "monotonic_down" || m.scenario == "market_trend" || m.scenario == "market_walkforward") &&
      m.trend_alignment_count > 0 && (m.trend_alignment_sum / (double)m.trend_alignment_count) < 0.25)
      m.issue_flags |= FXAI_AUDIT_ISSUE_MISSES_TREND;
   if(m.conf_drift > 0.22) m.issue_flags |= FXAI_AUDIT_ISSUE_CALIBRATION_DRIFT;
   if(m.reset_delta > 0.30) m.issue_flags |= FXAI_AUDIT_ISSUE_RESET_DRIFT;
   if(m.sequence_delta >= 0.0 && m.sequence_delta < 0.005) m.issue_flags |= FXAI_AUDIT_ISSUE_SEQUENCE_WEAK;
   if(m.move_sum <= 0.0) m.issue_flags |= FXAI_AUDIT_ISSUE_DEAD_OUTPUT;
   if((m.scenario == "random_walk" || m.scenario == "market_chop" || m.scenario == "market_spread_shock") && m.bias_abs > 0.85 && active > 24)
      m.issue_flags |= FXAI_AUDIT_ISSUE_SIDE_COLLAPSE;
}

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
   plugin.Describe(manifest);
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

   if(!FXAI_AuditGenerateScenarioSeries(spec,
                                        bars,
                                        seed + ((ulong)(ai_idx + 1) * (ulong)1315423911),
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
                                        ctx_extra_arr))
   {
      delete plugin;
      return false;
   }

   FXAI_ResetNormalizationWindows(192);
   FXAI_ResetFeatureNormalizationState();
   if(plugin.SupportsSyntheticSeries())
      plugin.SetSyntheticSeries(time_arr, open_arr, high_arr, low_arr, close_arr);

   FXAIAIHyperParams hp;
   FXAI_AuditDefaultHyperParams(ai_idx, hp);
   plugin.EnsureInitialized(hp);
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
   if(wf_train_bars < 96) wf_train_bars = 96;
   if(wf_test_bars < 24) wf_test_bars = 24;

   FXAIAIPredictionV4 held_pred_reset;
   FXAIAIPredictRequestV4 held_req;
   held_req.valid = false;
   bool held_req_ready = false;
   int held_req_idx = -1;

   for(int i=start_idx; i<end_idx; i++)
   {
      FXAIAIContextV4 ctx;
      int label_class = (int)FXAI_LABEL_SKIP;
      double move_points = 0.0;
      double mfe_points = 0.0;
      double mae_points = 0.0;
      double time_to_hit_frac = 1.0;
      int path_flags = 0;
      double spread_stress = 0.0;
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
                                sample_weight,
                                x))
         continue;

      ctx.sequence_bars = seq_bars;
      ctx.feature_schema_id = schema_id;
      bool train_enabled = true;
      bool eval_enabled = true;
      if(spec.name == "market_walkforward")
      {
         int phase = (i - start_idx) % MathMax(wf_train_bars + wf_test_bars, 1);
         train_enabled = (phase >= 0 && phase < wf_train_bars);
         eval_enabled = !train_enabled;
      }

      if(eval_enabled)
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
      for(int k=0; k<FXAI_AI_WEIGHTS; k++) req.x[k] = x[k];
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
      FXAI_ApplyFeatureSchemaToPayloadEx(schema_id, feature_groups_mask, req.ctx.sequence_bars, req.x_window, req.window_size, req.x);

      if(eval_enabled)
      {
         FXAIAIPredictionV4 pred;
         bool ok = FXAI_PredictViaV4(*plugin, req, hp, pred);
         string pred_reason = "";
         bool pred_valid = FXAI_ValidatePredictionV4(pred, pred_reason);
         if(!ok || !pred_valid)
         {
            out.invalid_preds++;
         }
         else
         {
            out.valid_preds++;
            int decision = FXAI_AuditDecisionFromPred(pred);
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
            out.brier_sum += brier;

            if(decision != (int)FXAI_LABEL_SKIP)
            {
               out.directional_eval_count++;
               double dir_conf = MathMax(pred.class_probs[(int)FXAI_LABEL_BUY], pred.class_probs[(int)FXAI_LABEL_SELL]);
               out.dir_conf_sum += dir_conf;
               bool dir_ok = ((decision == (int)FXAI_LABEL_BUY && label_class == (int)FXAI_LABEL_BUY) ||
                              (decision == (int)FXAI_LABEL_SELL && label_class == (int)FXAI_LABEL_SELL));
               if(dir_ok) out.directional_correct_count++;
               out.dir_hit_sum += (dir_ok ? 1.0 : 0.0);
               out.calibration_abs_sum += MathAbs(dir_conf - (dir_ok ? 1.0 : 0.0));
               double move_scale = MathMax(MathAbs(move_points), MathMax(MathAbs(pred.move_mean_points), 0.50));
               double pq = 0.25 * FXAI_Clamp(MathAbs(pred.mfe_mean_points - mfe_points) / move_scale, 0.0, 3.0) +
                           0.20 * FXAI_Clamp(MathAbs(pred.mae_mean_points - mae_points) / move_scale, 0.0, 3.0) +
                           0.20 * MathAbs(pred.hit_time_frac - time_to_hit_frac) +
                           0.20 * MathAbs(pred.path_risk - spread_stress) +
                           0.15 * MathAbs(pred.fill_risk - FXAI_Clamp(spread_stress + (((path_flags & FXAI_PATHFLAG_DUAL_HIT) != 0) ? 0.25 : 0.0), 0.0, 1.0));
               out.path_quality_abs_sum += pq;
               out.path_quality_count++;
            }
         }
      }

      FXAIAITrainRequestV4 train_req;

      if(train_enabled)
      {
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
         for(int k=0; k<FXAI_AI_WEIGHTS; k++) train_req.x[k] = x[k];
         FXAI_CopyWindowPayload(req.x_window, req.window_size, train_req.x_window, train_req.window_size);
         FXAI_ApplyFeatureSchemaToPayloadEx(schema_id, feature_groups_mask, train_req.ctx.sequence_bars, train_req.x_window, train_req.window_size, train_req.x);
         FXAI_TrainViaV4(*plugin, train_req, hp);
      }

      if(eval_enabled && !held_req_ready && i > start_idx + 128)
      {
         held_req = req;
         held_req_ready = true;
         held_req_idx = i;
      }
   }

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
         FXAI_ApplyFeatureSchemaToPayloadEx(schema_id, feature_groups_mask, seq_short.ctx.sequence_bars, seq_short.x_window, seq_short.window_size, seq_short.x);
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
   plugin.ClearSyntheticSeries();
   delete plugin;
   return true;
}


#endif // __FXAI_AUDIT_SCORING_MQH__
