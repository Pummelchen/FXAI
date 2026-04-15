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
      if(!FXAI_NormalizationCoreFinalizePredictPayload(schema_id, feature_groups_mask, req))
         continue;

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
      if(!FXAI_NormalizationCoreFinalizeTrainPayload(schema_id, feature_groups_mask, train_req))
         continue;
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
