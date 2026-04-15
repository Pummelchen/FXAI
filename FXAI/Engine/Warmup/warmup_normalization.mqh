double FXAI_WarmupPortfolioObjectiveProxy(const double total_score,
                                          const int total_trades,
                                          const double &regime_scores[],
                                          const int &regime_trades[])
{
   double mean = 0.0;
   double sq = 0.0;
   int used = 0;
   int covered = 0;
   for(int r=0; r<ArraySize(regime_scores) && r<ArraySize(regime_trades); r++)
   {
      if(regime_trades[r] <= 0)
         continue;
      double s = regime_scores[r];
      mean += s;
      sq += s * s;
      used++;
      if(regime_trades[r] >= 12)
         covered++;
   }
   if(used <= 0)
      return 0.0;

   mean /= (double)used;
   double var = MathMax(sq / (double)used - mean * mean, 0.0);
   double stdv = MathSqrt(var);
   double stability = 1.0 - FXAI_Clamp(stdv / MathMax(MathAbs(mean), 0.50), 0.0, 1.0);
   double diversification = FXAI_Clamp((double)covered / 4.0, 0.0, 1.0);
   double trade_cov = FXAI_Clamp((double)MathMax(total_trades, 0) / 64.0, 0.0, 1.0);
   double edge_norm = FXAI_Clamp(total_score / 100.0, -1.0, 1.0);
   double objective = 0.35 * stability +
                      0.25 * diversification +
                      0.20 * trade_cov +
                      0.20 * (0.5 + 0.5 * edge_norm);
   return FXAI_Clamp(1.20 * (objective - 0.50), -0.60, 0.60);
}

double FXAI_ScoreNormalizationSetup(const int i_start,
                                    const int i_end,
                                    const int H,
                                    const int target_ai_id,
                                    const double commission_points,
                                    const double cost_buffer_points,
                                    const double ev_threshold_points,
                                    const FXAIDataSnapshot &snapshot,
                                    const int &spread_m1[],
                                    const datetime &time_arr[],
                                    const double &open_arr[],
                                    const double &high_arr[],
                                    const double &low_arr[],
                                    const double &close_arr[],
                                    const datetime &time_m5[],
                                    const double &close_m5[],
                                    const int &map_m5[],
                                    const datetime &time_m15[],
                                    const double &close_m15[],
                                    const int &map_m15[],
                                    const datetime &time_m30[],
                                    const double &close_m30[],
                                    const int &map_m30[],
                                    const datetime &time_h1[],
                                    const double &close_h1[],
                                    const int &map_h1[],
                                    const double &ctx_mean_arr[],
                                    const double &ctx_std_arr[],
                                    const double &ctx_up_arr[],
                                    const double &ctx_extra_arr[],
                                    const FXAIAIHyperParams &hp,
                                    const double buy_thr,
                                    const double sell_thr)
{
   FXAIPreparedSample samples[];
   FXAI_PrecomputeTrainingSamples(i_start,
                                 i_end,
                                 H,
                                 commission_points,
                                 cost_buffer_points,
                                 ev_threshold_points,
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

   int span = i_end - i_start + 1;
   if(span < 240) return -1e9;

   int val_len = span / 3;
   if(val_len < 80) val_len = 80;
   if(val_len > 220) val_len = 220;
   int val_start = i_start;
   int val_end = val_start + val_len - 1;
   if(val_end >= i_end) val_end = i_end - 1;
   if(val_end <= val_start) return -1e9;

   int purge = H + 240;
   if(purge < H + 40) purge = H + 40;
   int train_start = val_end + purge + 1;
   int train_end = i_end;
   if(train_end - train_start < 100) return -1e9;

   CFXAIAIPlugin *trial = g_plugins.CreateInstance(target_ai_id);
   if(trial == NULL) return -1e9;

   trial.Reset();
   trial.EnsureInitialized(hp);
   FXAIAIManifestV4 trial_manifest;
   FXAI_GetPluginManifest(*trial, trial_manifest);
   for(int epoch=0; epoch<2; epoch++)
   {
      for(int i=train_end; i>=train_start; i--)
      {
         if(i < 0 || i >= ArraySize(samples)) continue;
         if(!samples[i].valid) continue;
         FXAIAITrainRequestV4 s3;
         FXAI_ClearTrainRequest(s3);
         s3.valid = samples[i].valid;
         s3.ctx.api_version = FXAI_API_VERSION_V4;
         s3.ctx.regime_id = samples[i].regime_id;
         s3.ctx.session_bucket = FXAI_DeriveSessionBucket(samples[i].sample_time);
         s3.ctx.horizon_minutes = samples[i].horizon_minutes;
         s3.ctx.feature_schema_id = trial_manifest.feature_schema_id;
         s3.ctx.normalization_method_id = (int)AI_FeatureNormalization;
         s3.ctx.sequence_bars = FXAI_GetPluginSequenceBars(*trial, samples[i].horizon_minutes);
         s3.ctx.cost_points = samples[i].cost_points;
         s3.ctx.min_move_points = samples[i].min_move_points;
         s3.ctx.point_value = (samples[i].point_value > 0.0 ? samples[i].point_value : (_Point > 0.0 ? _Point : 1.0));
         s3.ctx.domain_hash = samples[i].domain_hash;
         s3.ctx.sample_time = samples[i].sample_time;
         s3.label_class = samples[i].label_class;
         s3.move_points = samples[i].move_points;
         s3.sample_weight = 1.0;
         FXAI_SetTrainRequestPathTargets(s3,
                                         samples[i].mfe_points,
                                         samples[i].mae_points,
                                         samples[i].time_to_hit_frac,
                                         samples[i].path_flags,
                                         samples[i].spread_stress);
         FXAI_SetTrainRequestAuxTargets(s3,
                                        samples[i].masked_step_target,
                                        samples[i].next_vol_target,
                                        samples[i].regime_shift_target,
                                        samples[i].context_lead_target);
         for(int k=0; k<FXAI_AI_WEIGHTS; k++)
            s3.x[k] = samples[i].x[k];
         FXAI_BuildPreparedSampleWindow(samples, i, s3.ctx.sequence_bars, s3.x_window, s3.window_size);
         if(!FXAI_NormalizationCoreFinalizeTrainRequest(trial_manifest, s3))
            continue;
         FXAI_TrainViaV4(*trial, s3, hp);
      }
   }

   int trades = 0;
   double regime_scores[];
   int regime_trades[];
   double score = FXAI_ScoreWarmupTrial(*trial,
                                        hp,
                                        H,
                                        val_start,
                                        val_end,
                                        buy_thr,
                                        sell_thr,
                                        samples,
                                        trades,
                                        regime_scores,
                                        regime_trades);
   delete trial;
   if(trades < 20) score -= 1.5;
   score += FXAI_WarmupPortfolioObjectiveProxy(score, trades, regime_scores, regime_trades);
   return score;
}

void FXAI_BuildNormScoringModelList(const int primary_ai, int &model_ids[])
{
   ArrayResize(model_ids, 0);

   int p = primary_ai;
   if(p < 0 || p >= FXAI_AI_COUNT) p = (int)AI_SGD_LOGIT;

   int anchors[4];
   anchors[0] = p;
   anchors[1] = (int)AI_FTRL_LOGIT;
   anchors[2] = (int)AI_XGB_FAST;
   anchors[3] = (int)AI_LIGHTGBM;

   int max_models = (AI_Ensemble ? 4 : 1);
   if(max_models < 1) max_models = 1;
   if(max_models > 4) max_models = 4;

   for(int i=0; i<4; i++)
   {
      int id = anchors[i];
      bool exists = false;
      for(int j=0; j<ArraySize(model_ids); j++)
      {
         if(model_ids[j] == id)
         {
            exists = true;
            break;
         }
      }
      if(exists) continue;
      int sz = ArraySize(model_ids);
      ArrayResize(model_ids, sz + 1);
      model_ids[sz] = id;
      if(ArraySize(model_ids) >= max_models) break;
   }
}

double FXAI_ScoreNormalizationSetupMulti(const int i_start,
                                         const int i_end,
                                         const int H,
                                         const int &model_ids[],
                                         const double commission_points,
                                         const double cost_buffer_points,
                                         const double ev_threshold_points,
                                         const FXAIDataSnapshot &snapshot,
                                         const int &spread_m1[],
                                         const datetime &time_arr[],
                                         const double &open_arr[],
                                         const double &high_arr[],
                                         const double &low_arr[],
                                         const double &close_arr[],
                                         const datetime &time_m5[],
                                         const double &close_m5[],
                                         const int &map_m5[],
                                         const datetime &time_m15[],
                                         const double &close_m15[],
                                         const int &map_m15[],
                                         const datetime &time_m30[],
                                         const double &close_m30[],
                                         const int &map_m30[],
                                         const datetime &time_h1[],
                                         const double &close_h1[],
                                         const int &map_h1[],
                                         const double &ctx_mean_arr[],
                                         const double &ctx_std_arr[],
                                         const double &ctx_up_arr[],
                                         const double &ctx_extra_arr[],
                                         const FXAIAIHyperParams &hp,
                                         const double buy_thr,
                                         const double sell_thr)
{
   if(ArraySize(model_ids) <= 0) return -1e9;

   double sum = 0.0;
   int used = 0;
   for(int m=0; m<ArraySize(model_ids); m++)
   {
      int model_id = model_ids[m];
      double s = FXAI_ScoreNormalizationSetup(i_start,
                                              i_end,
                                              H,
                                              model_id,
                                              commission_points,
                                              cost_buffer_points,
                                              ev_threshold_points,
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
                                              hp,
                                              buy_thr,
                                              sell_thr);
      if(s <= -1e8) continue;
      sum += s;
      used++;
   }
   if(used <= 0) return -1e9;
   return sum / (double)used;
}

void FXAI_OptimizeNormalizationWindows(const int i_start,
                                       const int i_end,
                                       const int H,
                                       const double commission_points,
                                       const double cost_buffer_points,
                                       const double ev_threshold_points,
                                       const FXAIDataSnapshot &snapshot,
                                       const int &spread_m1[],
                                       const datetime &time_arr[],
                                       const double &open_arr[],
                                       const double &high_arr[],
                                       const double &low_arr[],
                                       const double &close_arr[],
                                       const datetime &time_m5[],
                                       const double &close_m5[],
                                       const int &map_m5[],
                                       const datetime &time_m15[],
                                       const double &close_m15[],
                                       const int &map_m15[],
                                       const datetime &time_m30[],
                                       const double &close_m30[],
                                       const int &map_m30[],
                                       const datetime &time_h1[],
                                       const double &close_h1[],
                                       const int &map_h1[],
                                       const double &ctx_mean_arr[],
                                       const double &ctx_std_arr[],
                                       const double &ctx_up_arr[],
                                       const double &ctx_extra_arr[],
                                       const FXAIAIHyperParams &base_hp,
                                       const double buy_thr,
                                       const double sell_thr)
{
   int default_window = FXAI_GetNormDefaultWindow();
   int w_fast = 96;
   int w_mid = default_window;
   int w_slow = default_window + 64;
   if(w_slow > FXAI_NORM_ROLL_WINDOW_MAX) w_slow = FXAI_NORM_ROLL_WINDOW_MAX;
   int w_regime = 128;

   int windows_tmp[];
   FXAI_BuildNormWindowsFromGroups(w_fast, w_mid, w_slow, w_regime, windows_tmp);
   FXAI_ApplyNormWindows(windows_tmp, default_window);
   int model_ids[];
   FXAI_BuildNormScoringModelList((int)AI_Type, model_ids);

   double best_score = FXAI_ScoreNormalizationSetupMulti(i_start,
                                                         i_end,
                                                         H,
                                                         model_ids,
                                                         commission_points,
                                                         cost_buffer_points,
                                                         ev_threshold_points,
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
                                                         buy_thr,
                                                         sell_thr);

   int candidates[6] = {64, 96, 128, 192, 256, 320};
   for(int group=0; group<4; group++)
   {
      int best_w = (group == 0 ? w_fast : (group == 1 ? w_mid : (group == 2 ? w_slow : w_regime)));
      for(int ci=0; ci<6; ci++)
      {
         int trial_w = FXAI_NormalizationWindowClamp(candidates[ci]);
         int tf = w_fast, tm = w_mid, ts = w_slow, tr = w_regime;
         if(group == 0) tf = trial_w;
         else if(group == 1) tm = trial_w;
         else if(group == 2) ts = trial_w;
         else tr = trial_w;

         FXAI_BuildNormWindowsFromGroups(tf, tm, ts, tr, windows_tmp);
         FXAI_ApplyNormWindows(windows_tmp, default_window);
         double score = FXAI_ScoreNormalizationSetupMulti(i_start,
                                                          i_end,
                                                          H,
                                                          model_ids,
                                                          commission_points,
                                                          cost_buffer_points,
                                                          ev_threshold_points,
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
                                                          buy_thr,
                                                          sell_thr);
         if(score > best_score)
         {
            best_score = score;
            best_w = trial_w;
         }
      }

      if(group == 0) w_fast = best_w;
      else if(group == 1) w_mid = best_w;
      else if(group == 2) w_slow = best_w;
      else w_regime = best_w;
   }

   FXAI_BuildNormWindowsFromGroups(w_fast, w_mid, w_slow, w_regime, windows_tmp);
   FXAI_ApplyNormWindows(windows_tmp, default_window);
}

bool FXAI_DeriveNormCandidateSplit(const int H,
                                   const int i_start,
                                   const int i_end,
                                   int &val_start,
                                   int &val_end,
                                   int &train_start,
                                   int &train_end);

double FXAI_ScoreNormMethodCandidate(const int ai_idx,
                                     const int method_id,
                                     const int H,
                                     const int warmup_train_epochs,
                                     const int i_start,
                                     const int i_end,
                                     const FXAIAIHyperParams &hp,
                                     const double buy_thr,
                                     const double sell_thr,
                                     const FXAIPreparedSample &samples[],
                                     double &regime_scores[],
                                     int &regime_trades[])
{
   int val_start = 0;
   int val_end = -1;
   int train_start = 0;
   int train_end = -1;
   if(!FXAI_DeriveNormCandidateSplit(H,
                                     i_start,
                                     i_end,
                                     val_start,
                                     val_end,
                                     train_start,
                                     train_end))
      return -1e9;

   CFXAIAIPlugin *trial = g_plugins.CreateInstance(ai_idx);
   if(trial == NULL) return -1e9;

   trial.Reset();
   trial.EnsureInitialized(hp);
   int train_epochs = FXAI_WarmupEpochBudget(ai_idx, H, warmup_train_epochs);
   if(train_epochs < 1) train_epochs = 1;
   if(train_epochs > 6) train_epochs = 6;
   FXAI_TrainModelWindowPrepared(ai_idx,
                                 *trial,
                                 train_start,
                                 train_end,
                                 train_epochs,
                                 hp,
                                 samples);

   int trades = 0;
   double score = FXAI_ScoreWarmupTrial(*trial,
                                        hp,
                                        H,
                                        val_start,
                                        val_end,
                                        buy_thr,
                                        sell_thr,
                                        samples,
                                        trades,
                                        regime_scores,
                                        regime_trades);
   delete trial;
   score += FXAI_WarmupPortfolioObjectiveProxy(score, trades, regime_scores, regime_trades);
   return score;
}

bool FXAI_DeriveNormCandidateSplit(const int H,
                                   const int i_start,
                                   const int i_end,
                                   int &val_start,
                                   int &val_end,
                                   int &train_start,
                                   int &train_end)
{
   val_start = 0;
   val_end = -1;
   train_start = 0;
   train_end = -1;

   int span = i_end - i_start + 1;
   if(span < 240)
      return false;

   int val_len = span / 3;
   if(val_len < 80) val_len = 80;
   if(val_len > 240) val_len = 240;
   val_start = i_start;
   val_end = val_start + val_len - 1;
   if(val_end >= i_end) val_end = i_end - 1;
   if(val_end <= val_start)
      return false;

   int purge = H + 240;
   if(purge < H + 40) purge = H + 40;
   train_start = val_end + purge + 1;
   train_end = i_end;
   return (train_end - train_start >= 100);
}

void FXAI_StoreNormBank(const int ai_idx,
                        const int regime_id,
                        const int horizon_minutes,
                        const int method_id)
{
   if(ai_idx < 0 || ai_idx >= FXAI_AI_COUNT) return;
   int hslot = FXAI_GetHorizonSlot(horizon_minutes);
   if(hslot < 0 || hslot >= FXAI_MAX_HORIZONS) return;

   int m = (int)FXAI_SanitizeNormMethod(method_id);
   if(regime_id < 0)
   {
      g_model_norm_method_horizon[ai_idx][hslot] = m;
      g_model_norm_horizon_ready[ai_idx][hslot] = true;
   }
   else if(regime_id >= 0 && regime_id < FXAI_REGIME_COUNT)
   {
      g_model_norm_method_bank[ai_idx][regime_id][hslot] = m;
      g_model_norm_bank_ready[ai_idx][regime_id][hslot] = true;
   }
}

void FXAI_WarmupSelectNormBanksForHorizon(const int H,
                                          const bool primary_horizon,
                                          const int warmup_train_epochs,
                                          const int i_start,
                                          const int i_end,
                                          const FXAIDataSnapshot &snapshot,
                                          const int &spread_m1[],
                                          const datetime &time_arr[],
                                          const double &open_arr[],
                                          const double &high_arr[],
                                          const double &low_arr[],
                                          const double &close_arr[],
                                          const datetime &time_m5[],
                                          const double &close_m5[],
                                          const int &map_m5[],
                                          const datetime &time_m15[],
                                          const double &close_m15[],
                                          const int &map_m15[],
                                          const datetime &time_m30[],
                                          const double &close_m30[],
                                          const int &map_m30[],
                                          const datetime &time_h1[],
                                          const double &close_h1[],
                                          const int &map_h1[],
                                          const double &ctx_mean_arr[],
                                          const double &ctx_std_arr[],
                                          const double &ctx_up_arr[],
                                          const double &ctx_extra_arr[],
                                          const FXAIAIHyperParams &base_hp,
                                          const double base_buy_thr,
                                          const double base_sell_thr,
                                          FXAINormSampleCache &norm_caches[])
{
   for(int ai_idx=0; ai_idx<FXAI_AI_COUNT; ai_idx++)
   {
      int methods[];
      FXAI_BuildNormMethodCandidateList(ai_idx, methods);
      if(ArraySize(methods) <= 0) continue;

      double best_score = -1e18;
      int best_method = methods[0];
      double best_regime_score[FXAI_REGIME_COUNT];
      int best_regime_method[FXAI_REGIME_COUNT];
      for(int r=0; r<FXAI_REGIME_COUNT; r++)
      {
         best_regime_score[r] = -1e18;
         best_regime_method[r] = best_method;
      }

      for(int m=0; m<ArraySize(methods); m++)
      {
         int method_id = methods[m];
         int val_start = 0;
         int val_end = -1;
         int train_start = 0;
         int train_end = -1;
         if(!FXAI_DeriveNormCandidateSplit(H,
                                           i_start,
                                           i_end,
                                           val_start,
                                           val_end,
                                           train_start,
                                           train_end))
            continue;

         int cache_idx = FXAI_EnsureNormSampleCache(method_id,
                                                    H,
                                                    i_start,
                                                    i_end,
                                                    train_start,
                                                    train_end,
                                                    H,
                                                    snapshot.commission_points,
                                                    (AI_CostBufferPoints < 0.0 ? 0.0 : AI_CostBufferPoints),
                                                    FXAI_Clamp(AI_EVThresholdPoints, 0.0, 100.0),
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
                                                    norm_caches);
         if(cache_idx < 0) continue;

         double regime_scores[];
         int regime_trades[];
         double score = FXAI_ScoreNormMethodCandidate(ai_idx,
                                                      method_id,
                                                      H,
                                                      warmup_train_epochs,
                                                      i_start,
                                                      i_end,
                                                      base_hp,
                                                      base_buy_thr,
                                                      base_sell_thr,
                                                      norm_caches[cache_idx].samples,
                                                      regime_scores,
                                                      regime_trades);
         if(score > best_score)
         {
            best_score = score;
            best_method = method_id;
         }

         for(int r=0; r<FXAI_REGIME_COUNT; r++)
         {
            if(r >= ArraySize(regime_scores) || r >= ArraySize(regime_trades)) continue;
            if(regime_trades[r] < 12) continue;
            if(regime_scores[r] > best_regime_score[r])
            {
               best_regime_score[r] = regime_scores[r];
               best_regime_method[r] = method_id;
            }
         }
      }

      FXAI_StoreNormBank(ai_idx, -1, H, best_method);
      if(primary_horizon)
      {
         g_model_norm_method[ai_idx] = best_method;
         g_model_norm_ready[ai_idx] = true;
      }
      for(int r=0; r<FXAI_REGIME_COUNT; r++)
      {
         int method_r = (best_regime_score[r] > -1e17 ? best_regime_method[r] : best_method);
         FXAI_StoreNormBank(ai_idx, r, H, method_r);
      }
   }
}

void FXAI_WarmupTrainHorizonPolicyForSamples(const int H,
                                             const int base_h,
                                             const int ev_lookback,
                                             const FXAIDataSnapshot &snapshot,
                                             const double &close_arr[],
                                             const int ai_hint,
                                             const int i_start,
                                             const int i_end,
                                             const FXAIPreparedSample &samples[])
{
   int n = ArraySize(samples);
   if(n <= 0) return;

   int start = i_start;
   int end = i_end;
   if(start < 0) start = 0;
   if(end >= n) end = n - 1;
   if(end < start) return;

   for(int i=end; i>=start; i--)
   {
      if(i < 0 || i >= ArraySize(samples)) continue;
      if(!samples[i].valid) continue;

      double exp_abs = FXAI_EstimateExpectedAbsMovePointsAtIndex(close_arr,
                                                                 i,
                                                                 H,
                                                                 ev_lookback,
                                                                 snapshot.point);
      if(exp_abs <= 0.0)
         exp_abs = MathMax(samples[i].min_move_points, MathAbs(samples[i].move_points));
      double current_vol = FXAI_RollingAbsReturn(close_arr, i, 20);
      if(current_vol < 1e-6) current_vol = MathAbs(samples[i].move_points) * snapshot.point;
      double feat[FXAI_HPOL_FEATS];
      FXAIDataSnapshot snap_i = snapshot;
      snap_i.bar_time = samples[i].sample_time;
      double warm_ctx_strength = FXAI_Clamp(MathAbs(FXAI_GetArrayValue(samples[i].x, 10, 0.0)) +
                                             FXAI_GetArrayValue(samples[i].x, 11, 0.0) +
                                             MathAbs(FXAI_GetArrayValue(samples[i].x, 12, 0.0)),
                                             0.0,
                                             4.0);
      double warm_ctx_quality = FXAI_Clamp(0.50 * FXAI_GetArrayValue(samples[i].x, 10, 0.0) +
                                            0.30 * FXAI_GetArrayValue(samples[i].x, 11, 0.0) +
                                            0.20 * FXAI_GetArrayValue(samples[i].x, 12, 0.0),
                                            -1.0,
                                            2.0);
      double warm_rel_hint = 0.50;
      if(ai_hint >= 0 && ai_hint < FXAI_AI_COUNT)
         warm_rel_hint = FXAI_Clamp(g_model_reliability[ai_hint], 0.0, 1.0);
      FXAI_BuildHorizonPolicyFeatures(H,
                                      base_h,
                                      exp_abs,
                                      samples[i].min_move_points,
                                      snap_i,
                                      current_vol,
                                      samples[i].regime_id,
                                      ai_hint,
                                      warm_ctx_strength,
                                      warm_ctx_quality,
                                      warm_rel_hint,
                                      feat);

      double edge = MathMax(MathAbs(samples[i].move_points) - samples[i].min_move_points, 0.0);
      double reward = (samples[i].label_class == (int)FXAI_LABEL_SKIP ? -0.25 :
                       samples[i].quality_score * edge / MathMax(samples[i].min_move_points, 0.50));
      reward = FXAI_Clamp(reward, -2.0, 6.0);
      FXAI_UpdateHorizonPolicy(samples[i].regime_id, feat, reward);
   }
}

struct FXAIWarmupBucketStats
{
   int trades;
   int wins;
   double net_sum;
   double gross_pos;
   double gross_neg;
   double eq;
   double eq_peak;
   double max_dd;
};
