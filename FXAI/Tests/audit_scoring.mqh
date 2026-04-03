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

void FXAI_AuditResetFoldMetrics(FXAIAuditFoldMetrics &m)
{
   m.samples_total = 0;
   m.valid_preds = 0;
   m.invalid_preds = 0;
   m.buy_count = 0;
   m.sell_count = 0;
   m.skip_count = 0;
   m.directional_eval_count = 0;
   m.directional_correct_count = 0;
   m.conf_sum = 0.0;
   m.rel_sum = 0.0;
   m.move_sum = 0.0;
   m.brier_sum = 0.0;
   m.calibration_abs_sum = 0.0;
   m.path_quality_abs_sum = 0.0;
   m.path_quality_count = 0;
   m.net_sum = 0.0;
}

void FXAI_AuditFoldInvalid(FXAIAuditFoldMetrics &m)
{
   m.samples_total++;
   m.invalid_preds++;
}

void FXAI_AuditFoldValid(FXAIAuditFoldMetrics &m,
                         const int decision,
                         const FXAIAIPredictionV4 &pred,
                         const double brier,
                         const double net_points,
                         const bool directional_eval,
                         const bool directional_ok,
                         const double calibration_abs,
                         const double path_quality)
{
   m.samples_total++;
   m.valid_preds++;
   if(decision == (int)FXAI_LABEL_BUY) m.buy_count++;
   else if(decision == (int)FXAI_LABEL_SELL) m.sell_count++;
   else m.skip_count++;

   m.conf_sum += pred.confidence;
   m.rel_sum += pred.reliability;
   m.move_sum += pred.move_mean_points;
   m.brier_sum += brier;
   m.net_sum += net_points;
   if(directional_eval)
   {
      m.directional_eval_count++;
      if(directional_ok) m.directional_correct_count++;
      m.calibration_abs_sum += calibration_abs;
   }
   if(path_quality >= 0.0)
   {
      m.path_quality_abs_sum += path_quality;
      m.path_quality_count++;
   }
}

double FXAI_AuditSessionEdgePressure(const datetime sample_time)
{
   MqlDateTime dt;
   TimeToStruct(sample_time, dt);
   double hour = (double)dt.hour + ((double)dt.min / 60.0);
   double dist_tokyo = MathMin(MathAbs(hour - 0.0), MathAbs(hour - 24.0));
   double dist_london = MathAbs(hour - 8.0);
   double dist_newyork = MathAbs(hour - 16.0);
   double best = MathMin(dist_tokyo, MathMin(dist_london, dist_newyork));
   return FXAI_Clamp(1.0 - best / 4.0, 0.0, 1.0);
}

double FXAI_AuditAdversarialWeaknessScore(const int label_class,
                                          const double move_points,
                                          const double min_move_points,
                                          const double mfe_points,
                                          const double mae_points,
                                          const double time_to_hit_frac,
                                          const int path_flags,
                                          const double spread_stress,
                                          const double macro_activity,
                                          const datetime sample_time,
                                          const FXAIAIPredictionV4 &pred)
{
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

   int decision = FXAI_AuditDecisionFromPred(pred);
   bool directional_eval = (decision != (int)FXAI_LABEL_SKIP);
   bool dir_ok = ((decision == (int)FXAI_LABEL_BUY && cls_idx == (int)FXAI_LABEL_BUY) ||
                  (decision == (int)FXAI_LABEL_SELL && cls_idx == (int)FXAI_LABEL_SELL));
   double dir_conf = MathMax(pred.class_probs[(int)FXAI_LABEL_BUY], pred.class_probs[(int)FXAI_LABEL_SELL]);
   double calibration_abs = (directional_eval ? MathAbs(dir_conf - (dir_ok ? 1.0 : 0.0)) : MathAbs(pred.class_probs[(int)FXAI_LABEL_SKIP] - target_probs[(int)FXAI_LABEL_SKIP]));
   double move_scale = MathMax(MathAbs(move_points), MathMax(MathAbs(pred.move_mean_points), MathMax(min_move_points, 0.50)));
   double path_quality = 0.25 * FXAI_Clamp(MathAbs(pred.mfe_mean_points - mfe_points) / move_scale, 0.0, 3.0) +
                         0.20 * FXAI_Clamp(MathAbs(pred.mae_mean_points - mae_points) / move_scale, 0.0, 3.0) +
                         0.20 * MathAbs(pred.hit_time_frac - time_to_hit_frac) +
                         0.20 * MathAbs(pred.path_risk - spread_stress) +
                         0.15 * MathAbs(pred.fill_risk - FXAI_Clamp(spread_stress + (((path_flags & FXAI_PATHFLAG_DUAL_HIT) != 0) ? 0.25 : 0.0), 0.0, 1.0));

   double wrong_dir = (directional_eval && !dir_ok ? 1.0 : 0.0);
   double noise_overtrade = ((cls_idx == (int)FXAI_LABEL_SKIP) && directional_eval ? dir_conf : 0.0);
   double missed_trade = ((cls_idx != (int)FXAI_LABEL_SKIP) && decision == (int)FXAI_LABEL_SKIP ? 1.0 - pred.class_probs[(int)FXAI_LABEL_SKIP] : 0.0);
   double stress = 0.18 * FXAI_Clamp(spread_stress, 0.0, 4.0) +
                   0.10 * (((path_flags & FXAI_PATHFLAG_DUAL_HIT) != 0) ? 1.0 : 0.0) +
                   0.08 * (((path_flags & FXAI_PATHFLAG_SLOW_HIT) != 0) ? 1.0 : 0.0) +
                   0.08 * FXAI_Clamp(macro_activity, 0.0, 1.0) +
                   0.08 * FXAI_AuditSessionEdgePressure(sample_time) +
                   0.10 * FXAI_Clamp(MathAbs(move_points) / MathMax(min_move_points, 0.50), 0.0, 4.0);

   return 0.55 * brier +
          0.40 * calibration_abs +
          0.32 * path_quality +
          0.38 * wrong_dir * dir_conf +
          0.24 * noise_overtrade +
          0.18 * missed_trade +
          stress;
}

bool FXAI_AuditGenerateAdversarialScenarioSeries(CFXAIAIRegistry &registry,
                                                 const int ai_idx,
                                                 const int bars,
                                                 const int horizon_minutes,
                                                 const ulong seed,
                                                 const double point,
                                                 const ENUM_FXAI_FEATURE_NORMALIZATION norm_method,
                                                 const FXAIAIManifestV4 &manifest,
                                                 const FXAIAIHyperParams &hp,
                                                 const int seq_bars,
                                                 const int schema_id,
                                                 const ulong feature_groups_mask,
                                                 datetime &time_arr[],
                                                 double &open_arr[],
                                                 double &high_arr[],
                                                 double &low_arr[],
                                                 double &close_arr[],
                                                 int &spread_arr[],
                                                 datetime &time_m5[],
                                                 double &close_m5[],
                                                 int &map_m5[],
                                                 datetime &time_m15[],
                                                 double &close_m15[],
                                                 int &map_m15[],
                                                 datetime &time_m30[],
                                                 double &close_m30[],
                                                 int &map_m30[],
                                                 datetime &time_h1[],
                                                 double &close_h1[],
                                                 int &map_h1[],
                                                 double &ctx_mean_arr[],
                                                 double &ctx_std_arr[],
                                                 double &ctx_up_arr[],
                                                 double &ctx_extra_arr[])
{
   int search_bars = bars * 4;
   if(search_bars < bars + 512)
      search_bars = bars + 512;

   MqlRates rates_m1[];
   int got = FXAI_AuditCopyMarketRates(search_bars, rates_m1);
   if(got < bars + 64)
      return false;
   search_bars = got;

   datetime base_time[];
   double base_open[];
   double base_high[];
   double base_low[];
   double base_close[];
   int base_spread[];
   datetime base_time_m5[];
   double base_close_m5[];
   int base_map_m5[];
   datetime base_time_m15[];
   double base_close_m15[];
   int base_map_m15[];
   datetime base_time_m30[];
   double base_close_m30[];
   int base_map_m30[];
   datetime base_time_h1[];
   double base_close_h1[];
   int base_map_h1[];
   double base_ctx_mean[];
   double base_ctx_std[];
   double base_ctx_up[];
   double base_ctx_extra[];
   if(!FXAI_AuditBuildSeriesFromMarketRates(rates_m1,
                                            search_bars,
                                            point,
                                            base_time,
                                            base_open,
                                            base_high,
                                            base_low,
                                            base_close,
                                            base_spread,
                                            base_time_m5,
                                            base_close_m5,
                                            base_map_m5,
                                            base_time_m15,
                                            base_close_m15,
                                            base_map_m15,
                                            base_time_m30,
                                            base_close_m30,
                                            base_map_m30,
                                            base_time_h1,
                                            base_close_h1,
                                            base_map_h1,
                                            base_ctx_mean,
                                            base_ctx_std,
                                            base_ctx_up,
                                            base_ctx_extra))
      return false;

   CFXAIAIPlugin *miner = registry.CreateInstance(ai_idx);
   if(miner == NULL)
      return false;
   miner.EnsureInitialized(hp);
   if(miner.SupportsSyntheticSeries())
      miner.SetSyntheticSeries(base_time, base_open, base_high, base_low, base_close);

   int n = ArraySize(base_close);
   double weakness[];
   ArrayResize(weakness, n);
   ArrayInitialize(weakness, 0.0);

   int start_idx = horizon_minutes + 1;
   int end_idx = n - 220;
   if(end_idx <= start_idx)
      end_idx = n - 32;
   if(end_idx <= start_idx)
      end_idx = n - 2;

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
      FXAIExecutionTraceStats trace_stats;
      double sample_weight = 1.0;
      double x[];
      if(!FXAI_AuditBuildSample(i,
                                horizon_minutes,
                                point,
                                0.25,
                                norm_method,
                                base_time,
                                base_open,
                                base_high,
                                base_low,
                                base_close,
                                base_spread,
                                base_time_m5,
                                base_close_m5,
                                base_map_m5,
                                base_time_m15,
                                base_close_m15,
                                base_map_m15,
                                base_time_m30,
                                base_close_m30,
                                base_map_m30,
                                base_time_h1,
                                base_close_h1,
                                base_map_h1,
                                base_ctx_mean,
                                base_ctx_std,
                                base_ctx_up,
                                base_ctx_extra,
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
                            base_time,
                            base_open,
                            base_high,
                            base_low,
                            base_close,
                            base_spread,
                            base_time_m5,
                            base_close_m5,
                            base_map_m5,
                            base_time_m15,
                            base_close_m15,
                            base_map_m15,
                            base_time_m30,
                            base_close_m30,
                            base_map_m30,
                            base_time_h1,
                            base_close_h1,
                            base_map_h1,
                            base_ctx_mean,
                            base_ctx_std,
                            base_ctx_up,
                            base_ctx_extra,
                            req.x_window,
                            req.window_size);
      FXAI_ApplyFeatureSchemaToPayloadEx(schema_id,
                                         feature_groups_mask,
                                         req.ctx.sequence_bars,
                                         req.x_window,
                                         req.window_size,
                                         req.x);

      FXAIAIPredictionV4 pred;
      string pred_reason = "";
      bool pred_ok = FXAI_PredictViaV4(*miner, req, hp, pred);
      bool pred_valid = (pred_ok && FXAI_ValidatePredictionV4(pred, pred_reason));

      double macro_pre = FXAI_GetInputFeature(req.x, FXAI_MACRO_EVENT_FEATURE_OFFSET + 0);
      double macro_post = FXAI_GetInputFeature(req.x, FXAI_MACRO_EVENT_FEATURE_OFFSET + 1);
      double macro_importance = FXAI_GetInputFeature(req.x, FXAI_MACRO_EVENT_FEATURE_OFFSET + 2);
      double macro_activity = MathMax(macro_importance, MathMax(macro_pre, macro_post));

      if(!pred_valid)
      {
         weakness[i] = 3.5 +
                       0.25 * FXAI_Clamp(spread_stress, 0.0, 4.0) +
                       0.15 * FXAI_AuditSessionEdgePressure(ctx.sample_time) +
                       0.15 * FXAI_Clamp(macro_activity, 0.0, 1.0);
      }
      else
      {
         weakness[i] = FXAI_AuditAdversarialWeaknessScore(label_class,
                                                          move_points,
                                                          ctx.min_move_points,
                                                          mfe_points,
                                                          mae_points,
                                                          time_to_hit_frac,
                                                          path_flags,
                                                          spread_stress,
                                                          macro_activity,
                                                          ctx.sample_time,
                                                          pred);
      }

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
      FXAI_TrainViaV4(*miner, train_req, hp);
   }

   double pref_w[];
   double pref_tail_w[];
   int pref_valid[];
   int pref_tail_hits[];
   ArrayResize(pref_w, n + 1);
   ArrayResize(pref_tail_w, n + 1);
   ArrayResize(pref_valid, n + 1);
   ArrayResize(pref_tail_hits, n + 1);
   pref_w[0] = 0.0;
   pref_tail_w[0] = 0.0;
   pref_valid[0] = 0;
   pref_tail_hits[0] = 0;
   for(int i=0; i<n; i++)
   {
      double w = weakness[i];
      bool valid = (w > 0.0);
      bool tail = (w > 1.25);
      pref_w[i + 1] = pref_w[i] + w;
      pref_tail_w[i + 1] = pref_tail_w[i] + (tail ? w : 0.0);
      pref_valid[i + 1] = pref_valid[i] + (valid ? 1 : 0);
      pref_tail_hits[i + 1] = pref_tail_hits[i] + (tail ? 1 : 0);
   }

   int min_eval = MathMax(96, bars / 6);
   if(min_eval > bars)
      min_eval = bars;
   int best_start = 0;
   double best_score = -1e18;
   int max_start = n - bars;
   if(max_start < 0)
      max_start = 0;
   for(int start=0; start<=max_start; start++)
   {
      int end = start + bars;
      int valid = pref_valid[end] - pref_valid[start];
      if(valid < min_eval)
         continue;
      double sum_w = pref_w[end] - pref_w[start];
      double tail_w = pref_tail_w[end] - pref_tail_w[start];
      int tail_hits = pref_tail_hits[end] - pref_tail_hits[start];
      double mean_w = sum_w / (double)MathMax(valid, 1);
      double tail_density = (double)tail_hits / (double)MathMax(valid, 1);
      double tail_share = tail_w / MathMax(sum_w, 1e-6);
      double recent_bonus = 1.0 - ((double)start / (double)MathMax(max_start, 1));
      double score = 0.76 * mean_w +
                     0.16 * tail_density +
                     0.08 * tail_share +
                     0.03 * recent_bonus;
      if(score > best_score)
      {
         best_score = score;
         best_start = start;
      }
   }

   miner.ClearSyntheticSeries();
   delete miner;

   if(best_score <= -1e17)
   {
      FXAIAuditScenarioSpec fallback_spec;
      FXAI_AuditFillScenarioSpec(8, fallback_spec);
      return FXAI_AuditGenerateScenarioSeries(fallback_spec,
                                              bars,
                                              seed,
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

   MqlRates sel_m1[];
   ArrayResize(sel_m1, bars);
   ArraySetAsSeries(sel_m1, true);
   for(int i=0; i<bars; i++)
      sel_m1[i] = rates_m1[best_start + i];

   return FXAI_AuditBuildSeriesFromMarketRates(sel_m1,
                                               search_bars,
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

double FXAI_AuditApproxNormalCdf(const double x)
{
   return 1.0 / (1.0 + MathExp(-1.702 * x));
}

double FXAI_AuditMean(const double &values[])
{
   int n = ArraySize(values);
   if(n <= 0) return 0.0;
   double sum = 0.0;
   for(int i=0; i<n; i++)
      sum += values[i];
   return sum / (double)n;
}

double FXAI_AuditStd(const double &values[],
                     const double mean)
{
   int n = ArraySize(values);
   if(n <= 1) return 0.0;
   double var = 0.0;
   for(int i=0; i<n; i++)
   {
      double d = values[i] - mean;
      var += d * d;
   }
   var /= (double)(n - 1);
   if(var < 0.0) var = 0.0;
   return MathSqrt(var);
}

double FXAI_AuditScoreFold(const FXAIAuditFoldMetrics &m)
{
   if(m.samples_total < 12 || m.valid_preds <= 0)
      return -1e9;

   double invalid_rate = (double)m.invalid_preds / (double)m.samples_total;
   double active_ratio = (double)(m.buy_count + m.sell_count) / (double)m.samples_total;
   double skip_ratio = (double)m.skip_count / (double)m.samples_total;
   double hit_rate = (m.directional_eval_count > 0 ?
                      (double)m.directional_correct_count / (double)m.directional_eval_count :
                      0.50);
   double brier = m.brier_sum / (double)m.valid_preds;
   double cal = (m.directional_eval_count > 0 ?
                 m.calibration_abs_sum / (double)m.directional_eval_count :
                 0.35);
   double pq = (m.path_quality_count > 0 ?
                m.path_quality_abs_sum / (double)m.path_quality_count :
                0.50);
   double avg_conf = m.conf_sum / (double)m.valid_preds;
   double avg_rel = m.rel_sum / (double)m.valid_preds;
   double avg_move = m.move_sum / (double)m.valid_preds;
   double avg_net = m.net_sum / (double)m.valid_preds;

   double score = 100.0;
   score -= 42.0 * invalid_rate;
   score -= 28.0 * brier;
   score -= 16.0 * cal;
   score -= 12.0 * pq;
   if(active_ratio > 0.78)
      score -= 18.0 * FXAI_Clamp((active_ratio - 0.78) / 0.22, 0.0, 1.0);
   if(active_ratio < 0.05)
      score -= 10.0 * FXAI_Clamp((0.05 - active_ratio) / 0.05, 0.0, 1.0);
   if(skip_ratio > 0.92)
      score -= 6.0 * FXAI_Clamp((skip_ratio - 0.92) / 0.08, 0.0, 1.0);
   score += 24.0 * FXAI_Clamp(hit_rate - 0.50, -0.50, 0.50);
   score += 8.0 * FXAI_Clamp(avg_rel - 0.50, -0.50, 0.50);
   score += 4.0 * FXAI_Clamp(avg_conf - 0.50, -0.50, 0.50);
   score += 4.0 * FXAI_Clamp(avg_move / 8.0, 0.0, 1.0);
   score += 6.0 * FXAI_Clamp(avg_net / 8.0, -1.0, 1.0);
   if(score < 0.0) score = 0.0;
   if(score > 100.0) score = 100.0;
   return score;
}

double FXAI_AuditDeflatedSharpeProxy(const double &scores[],
                                     const double pbo)
{
   int n = ArraySize(scores);
   if(n <= 1) return 0.0;

   double returns[];
   ArrayResize(returns, n);
   for(int i=0; i<n; i++)
      returns[i] = (scores[i] - 60.0) / 20.0;

   double mean_ret = FXAI_AuditMean(returns);
   double std_ret = FXAI_AuditStd(returns, mean_ret);
   if(std_ret <= 1e-9)
      return (mean_ret > 0.0 ? 1.0 : 0.0);

   double sharpe = mean_ret / std_ret;
   double sample_deflator = MathSqrt((double)(n - 1) / (double)(n + 3));
   double selection_penalty = 0.35 + 0.65 * FXAI_Clamp(pbo, 0.0, 1.0);
   double z = sharpe * sample_deflator - selection_penalty - 0.08 * MathLog((double)n + 1.0);
   return FXAI_Clamp(FXAI_AuditApproxNormalCdf(z), 0.0, 1.0);
}

void FXAI_AuditFinalizeWalkForward(FXAIAuditScenarioMetrics &m,
                                   const FXAIAuditFoldMetrics &train_folds[],
                                   const FXAIAuditFoldMetrics &test_folds[])
{
   int count = MathMin(ArraySize(train_folds), ArraySize(test_folds));
   if(count <= 0)
      return;

   double train_scores[];
   double test_scores[];
   int pass_count = 0;
   int overfit_count = 0;

   for(int f=0; f<count; f++)
   {
      m.wf_train_samples += train_folds[f].samples_total;
      m.wf_test_samples += test_folds[f].samples_total;

      double train_score = FXAI_AuditScoreFold(train_folds[f]);
      double test_score = FXAI_AuditScoreFold(test_folds[f]);
      if(train_score <= -1e8 || test_score <= -1e8)
         continue;

      int train_sz = ArraySize(train_scores);
      ArrayResize(train_scores, train_sz + 1);
      train_scores[train_sz] = train_score;

      int test_sz = ArraySize(test_scores);
      ArrayResize(test_scores, test_sz + 1);
      test_scores[test_sz] = test_score;

      if(test_score + 6.0 < train_score)
         overfit_count++;
      if(test_score >= 68.0 && test_score + 8.0 >= train_score)
         pass_count++;
   }

   m.wf_folds = ArraySize(test_scores);
   if(m.wf_folds <= 0)
      return;

   m.wf_train_score = FXAI_AuditMean(train_scores);
   m.wf_test_score = FXAI_AuditMean(test_scores);
   m.wf_test_score_std = FXAI_AuditStd(test_scores, m.wf_test_score);
   m.wf_gap = m.wf_train_score - m.wf_test_score;
   m.wf_pbo = (double)overfit_count / (double)m.wf_folds;
   m.wf_pass_rate = (double)pass_count / (double)m.wf_folds;
   m.wf_dsr = FXAI_AuditDeflatedSharpeProxy(test_scores, m.wf_pbo);
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
   if(m.samples_total > 0)
   {
      double denom = (double)m.samples_total;
      m.macro_event_rate /= denom;
      m.macro_pre_rate /= denom;
      m.macro_post_rate /= denom;
      m.macro_importance_mean /= denom;
      m.macro_surprise_abs_mean /= denom;
      m.macro_data_coverage /= denom;
      m.macro_surprise_z_abs_mean /= denom;
      m.macro_revision_abs_mean /= denom;
      m.macro_currency_relevance_mean /= denom;
      m.macro_provenance_trust_mean /= denom;
      m.macro_rates_rate /= denom;
      m.macro_inflation_rate /= denom;
      m.macro_labor_rate /= denom;
      m.macro_growth_rate /= denom;
   }
   double avg_net = (m.valid_preds > 0 ? m.net_sum / (double)m.valid_preds : 0.0);
   double hit_rate = (m.directional_eval_count > 0 ?
                      (double)m.directional_correct_count / (double)m.directional_eval_count :
                      0.50);
   bool macro_dataset_active = FXAI_HasMacroEventDataset();
   bool macro_dataset_safe = (macro_dataset_active && FXAI_MacroEventLeakageSafe());

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
   if(m.scenario == "market_macro_event")
   {
      if(!macro_dataset_active)
      {
         // Keep the scenario neutral when no macro dataset is configured.
      }
      else if(!macro_dataset_safe)
      {
         score -= 22.0;
      }
      else
      {
         if(m.macro_data_coverage < 0.08) score -= 20.0;
         if(m.macro_event_rate < 0.06) score -= 16.0;
         if(m.macro_importance_mean < 0.08) score -= 10.0;
         if(m.macro_currency_relevance_mean < 0.40) score -= 8.0;
         if(m.macro_provenance_trust_mean < 0.45) score -= 8.0;
         if(m.active_ratio < 0.05 && m.macro_event_rate > 0.10) score -= 8.0;
         if(m.active_ratio > 0.88 && m.macro_surprise_abs_mean < 0.20) score -= 8.0;
         if(avg_net < 0.0) score -= 10.0 * FXAI_Clamp(-avg_net / 4.0, 0.0, 1.0);
      }
   }
   if((m.scenario == "market_session_edges" || m.scenario == "market_spread_shock" || m.scenario == "market_walkforward") && avg_net < 0.0)
      score -= 8.0 * FXAI_Clamp(-avg_net / 4.0, 0.0, 1.0);
   if(m.scenario == "market_adversarial")
   {
      if(hit_rate < 0.53) score -= 12.0 * FXAI_Clamp((0.53 - hit_rate) / 0.18, 0.0, 1.0);
      if(m.conf_drift > 0.20) score -= 8.0;
      if(m.calibration_error > 0.26) score -= 10.0;
      if(m.path_quality_error > 0.50) score -= 10.0;
      if(avg_net < 0.0) score -= 12.0 * FXAI_Clamp(-avg_net / 4.0, 0.0, 1.0);
      if(m.active_ratio < 0.03) score -= 6.0;
      if(m.active_ratio > 0.90 && m.brier_score > 0.42) score -= 8.0;
   }
   if(m.reset_delta > 0.30) score -= 12.0;
   if(m.sequence_delta < 0.005 && m.sequence_delta >= 0.0) score -= 6.0;
   if(m.move_sum <= 0.0) score -= 8.0;

   if(m.scenario == "market_walkforward")
   {
      if(m.wf_folds < 3) score -= 18.0;
      if(m.wf_gap > 12.0) score -= 10.0;
      if(m.wf_pbo > 0.45) score -= 12.0;
      if(m.wf_pass_rate < 0.55) score -= 12.0;
      if(m.wf_dsr < 0.35) score -= 10.0;
      if(m.wf_test_score > 0.0 && m.wf_test_score < 68.0) score -= 10.0;
      if(m.wf_test_score_std > 10.0) score -= 6.0;
   }

   if(score < 0.0) score = 0.0;
   m.score = score;

   if(m.invalid_preds > 0) m.issue_flags |= FXAI_AUDIT_ISSUE_INVALID_PRED;
   if((m.scenario == "random_walk" || m.scenario == "market_chop" || m.scenario == "market_spread_shock" || m.scenario == "market_session_edges") && (m.skip_ratio < 0.55 || m.active_ratio > 0.70))
      m.issue_flags |= FXAI_AUDIT_ISSUE_OVERTRADES_NOISE;
   if(m.scenario == "market_macro_event")
   {
      if(macro_dataset_active)
      {
         if(!macro_dataset_safe || m.macro_data_coverage < 0.05)
            m.issue_flags |= FXAI_AUDIT_ISSUE_MACRO_DATA_GAP;
         if(macro_dataset_safe && m.active_ratio < 0.05 && m.macro_event_rate > 0.10)
            m.issue_flags |= FXAI_AUDIT_ISSUE_MACRO_BLIND;
         if(macro_dataset_safe && m.active_ratio > 0.88 && m.macro_surprise_abs_mean < 0.20)
            m.issue_flags |= FXAI_AUDIT_ISSUE_MACRO_OVERREACT;
      }
   }
   if((m.scenario == "drift_up" || m.scenario == "drift_down" || m.scenario == "monotonic_up" || m.scenario == "monotonic_down" || m.scenario == "market_trend" || m.scenario == "market_walkforward") &&
      m.trend_alignment_count > 0 && (m.trend_alignment_sum / (double)m.trend_alignment_count) < 0.25)
      m.issue_flags |= FXAI_AUDIT_ISSUE_MISSES_TREND;
   if(m.conf_drift > 0.22) m.issue_flags |= FXAI_AUDIT_ISSUE_CALIBRATION_DRIFT;
   if(m.reset_delta > 0.30) m.issue_flags |= FXAI_AUDIT_ISSUE_RESET_DRIFT;
   if(m.sequence_delta >= 0.0 && m.sequence_delta < 0.005) m.issue_flags |= FXAI_AUDIT_ISSUE_SEQUENCE_WEAK;
   if(m.move_sum <= 0.0) m.issue_flags |= FXAI_AUDIT_ISSUE_DEAD_OUTPUT;
   if((m.scenario == "random_walk" || m.scenario == "market_chop" || m.scenario == "market_spread_shock") && m.bias_abs > 0.85 && active > 24)
      m.issue_flags |= FXAI_AUDIT_ISSUE_SIDE_COLLAPSE;
   if(m.scenario == "market_adversarial")
   {
      if(m.score < 68.0 || avg_net < 0.0 || hit_rate < 0.53 || m.calibration_error > 0.26 || m.path_quality_error > 0.50)
         m.issue_flags |= FXAI_AUDIT_ISSUE_ADVERSARIAL_WEAK;
   }

   if(m.scenario == "market_walkforward")
   {
      if(m.wf_pbo > 0.45 || m.wf_gap > 12.0)
         m.issue_flags |= FXAI_AUDIT_ISSUE_WF_OVERFIT;
      if(m.wf_folds < 3 || m.wf_pass_rate < 0.55 || m.wf_test_score_std > 10.0)
         m.issue_flags |= FXAI_AUDIT_ISSUE_WF_UNSTABLE;
      if(m.wf_dsr < 0.35 || (m.wf_test_score > 0.0 && m.wf_test_score < 68.0))
         m.issue_flags |= FXAI_AUDIT_ISSUE_WF_WEAK_EDGE;
   }
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


#endif // __FXAI_AUDIT_SCORING_MQH__
