#ifndef __FXAI_ENGINE_WARMUP_MQH__
#define __FXAI_ENGINE_WARMUP_MQH__

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
         s3.ctx.point_value = (_Point > 0.0 ? _Point : 1.0);
         s3.ctx.sample_time = samples[i].sample_time;
         s3.label_class = samples[i].label_class;
         s3.move_points = samples[i].move_points;
         s3.sample_weight = 1.0;
         for(int k=0; k<FXAI_AI_WEIGHTS; k++)
            s3.x[k] = samples[i].x[k];
         FXAI_BuildPreparedSampleWindow(samples, i, s3.ctx.sequence_bars, s3.x_window, s3.window_size);
         FXAI_ApplyFeatureSchemaToInputEx(trial_manifest.feature_schema_id,
                                          trial_manifest.feature_groups_mask,
                                          s3.ctx.sequence_bars,
                                          s3.x_window,
                                          s3.window_size,
                                          s3.x);
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

double FXAI_ScoreNormMethodCandidate(const int ai_idx,
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
   int span = i_end - i_start + 1;
   if(span < 240) return -1e9;

   int val_len = span / 3;
   if(val_len < 80) val_len = 80;
   if(val_len > 240) val_len = 240;
   int val_start = i_start;
   int val_end = val_start + val_len - 1;
   if(val_end >= i_end) val_end = i_end - 1;
   if(val_end <= val_start) return -1e9;

   int purge = H + 240;
   if(purge < H + 40) purge = H + 40;
   int train_start = val_end + purge + 1;
   int train_end = i_end;
   if(train_end - train_start < 100) return -1e9;

   CFXAIAIPlugin *trial = g_plugins.CreateInstance(ai_idx);
   if(trial == NULL) return -1e9;

   trial.Reset();
   trial.EnsureInitialized(hp);
   int train_epochs = warmup_train_epochs;
   if(train_epochs < 1) train_epochs = 1;
   if(train_epochs > 2) train_epochs = 2;
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
   return score;
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
         int cache_idx = FXAI_FindNormSampleCache(method_id, norm_caches);
         if(cache_idx < 0) continue;

         double regime_scores[];
         int regime_trades[];
         double score = FXAI_ScoreNormMethodCandidate(ai_idx,
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

void FXAI_ResetWarmupBucketStats(FXAIWarmupBucketStats &stats)
{
   stats.trades = 0;
   stats.wins = 0;
   stats.net_sum = 0.0;
   stats.gross_pos = 0.0;
   stats.gross_neg = 0.0;
   stats.eq = 0.0;
   stats.eq_peak = 0.0;
   stats.max_dd = 0.0;
}

void FXAI_UpdateWarmupBucketStats(FXAIWarmupBucketStats &stats, const double net_pts)
{
   stats.net_sum += net_pts;
   if(net_pts >= 0.0) stats.gross_pos += net_pts;
   else               stats.gross_neg += -net_pts;
   stats.eq += net_pts;
   if(stats.eq > stats.eq_peak) stats.eq_peak = stats.eq;
   double dd = stats.eq_peak - stats.eq;
   if(dd > stats.max_dd) stats.max_dd = dd;
   stats.trades++;
   if(net_pts > 0.0) stats.wins++;
}

double FXAI_ScoreWarmupBucketStats(const FXAIWarmupBucketStats &stats)
{
   if(stats.trades <= 0) return -1e9;
   double win_rate = (double)stats.wins / (double)stats.trades;
   double avg_net = stats.net_sum / (double)stats.trades;
   double profit_factor = stats.gross_pos / MathMax(stats.gross_neg, 1e-6);
   if(profit_factor > 8.0) profit_factor = 8.0;

   double dd_penalty = 0.0;
   if(stats.gross_pos > 0.0) dd_penalty = stats.max_dd / stats.gross_pos;
   else if(stats.max_dd > 0.0) dd_penalty = 2.0;

   return (avg_net * 5.0) + (win_rate * 1.75) + (0.80 * profit_factor) - (1.50 * dd_penalty);
}

double FXAI_ScoreWarmupTrial(CFXAIAIPlugin &plugin,
                            const FXAIAIHyperParams &hp,
                            const int horizon_minutes,
                            const int val_start,
                            const int val_end,
                            const double buyThr,
                            const double sellThr,
                            const FXAIPreparedSample &samples[],
                            int &trades_out,
                            double &regime_scores[],
                            int &regime_trades[])
{
   trades_out = 0;
   int n = ArraySize(samples);
   if(n <= 0 || val_end < val_start) return -1e9;

   if(ArraySize(regime_scores) != FXAI_REGIME_COUNT) ArrayResize(regime_scores, FXAI_REGIME_COUNT);
   if(ArraySize(regime_trades) != FXAI_REGIME_COUNT) ArrayResize(regime_trades, FXAI_REGIME_COUNT);
   FXAIWarmupBucketStats regime_stats[FXAI_REGIME_COUNT];
   for(int r=0; r<FXAI_REGIME_COUNT; r++)
   {
      regime_scores[r] = -1e9;
      regime_trades[r] = 0;
      FXAI_ResetWarmupBucketStats(regime_stats[r]);
   }

   int start = val_start;
   int end = val_end;
   if(start < 0) start = 0;
   if(end >= n) end = n - 1;
   if(end < start) return -1e9;

   FXAIWarmupBucketStats total_stats;
   FXAI_ResetWarmupBucketStats(total_stats);
   double fallback_move_ema = 0.0;
   bool fallback_move_ready = false;
   FXAIAIManifestV4 plugin_manifest;
   FXAI_GetPluginManifest(plugin, plugin_manifest);

   int score_h = FXAI_ClampHorizon(horizon_minutes);
   for(int i=end; i>=start; i--)
   {
      if(!samples[i].valid) continue;

      FXAIAIPredictRequestV4 req;
      FXAI_ClearPredictRequest(req);
      req.valid = samples[i].valid;
      req.ctx.api_version = FXAI_API_VERSION_V4;
      req.ctx.regime_id = samples[i].regime_id;
      req.ctx.session_bucket = FXAI_DeriveSessionBucket(samples[i].sample_time);
      req.ctx.horizon_minutes = score_h;
      req.ctx.feature_schema_id = plugin_manifest.feature_schema_id;
      req.ctx.normalization_method_id = (int)AI_FeatureNormalization;
      req.ctx.sequence_bars = FXAI_GetPluginSequenceBars(plugin, score_h);
      req.ctx.cost_points = samples[i].cost_points;
      req.ctx.min_move_points = samples[i].min_move_points;
      req.ctx.point_value = (_Point > 0.0 ? _Point : 1.0);
      req.ctx.sample_time = samples[i].sample_time;
      for(int k=0; k<FXAI_AI_WEIGHTS; k++)
         req.x[k] = samples[i].x[k];
      FXAI_BuildPreparedSampleWindow(samples, i, req.ctx.sequence_bars, req.x_window, req.window_size);
      FXAI_ApplyFeatureSchemaToInputEx(plugin_manifest.feature_schema_id,
                                       plugin_manifest.feature_groups_mask,
                                       req.ctx.sequence_bars,
                                       req.x_window,
                                       req.window_size,
                                       req.x);

      FXAIAIPredictionV4 pred;
      FXAI_PredictViaV4(plugin, req, hp, pred);

      double probs_eval[3];
      probs_eval[(int)FXAI_LABEL_SELL] = pred.class_probs[(int)FXAI_LABEL_SELL];
      probs_eval[(int)FXAI_LABEL_BUY] = pred.class_probs[(int)FXAI_LABEL_BUY];
      probs_eval[(int)FXAI_LABEL_SKIP] = pred.class_probs[(int)FXAI_LABEL_SKIP];

      double expected_move = pred.move_mean_points;
      if(expected_move <= 0.0 && fallback_move_ready)
         expected_move = MathMax(fallback_move_ema, samples[i].min_move_points);
      if(expected_move <= 0.0) expected_move = samples[i].min_move_points;
      if(expected_move <= 0.0) expected_move = 0.10;

      double buyMinProb = buyThr;
      double sellMinProb = 1.0 - sellThr;
      double skipMinProb = 0.55;
      double vol_proxy = 0.0;
      if(FXAI_AI_WEIGHTS > 6) vol_proxy = MathAbs(samples[i].x[6]);
      FXAI_DeriveAdaptiveThresholds(buyThr,
                                   sellThr,
                                   samples[i].min_move_points,
                                   expected_move,
                                   vol_proxy,
                                   buyMinProb,
                                   sellMinProb,
                                   skipMinProb);

      int signal = FXAI_ClassSignalFromEV(probs_eval,
                                         buyMinProb,
                                         sellMinProb,
                                         skipMinProb,
                                         expected_move,
                                         samples[i].min_move_points,
                                         FXAI_Clamp(AI_EVThresholdPoints, 0.0, 100.0));
      if(signal == -1) continue;

      double net_pts = FXAI_RealizedNetPointsForSignal(signal,
                                                       samples[i].move_points,
                                                       samples[i].min_move_points,
                                                       score_h);
      FXAI_UpdateWarmupBucketStats(total_stats, net_pts);
      int regime_id = samples[i].regime_id;
      if(regime_id < 0 || regime_id >= FXAI_REGIME_COUNT) regime_id = 0;
      FXAI_UpdateWarmupBucketStats(regime_stats[regime_id], net_pts);
      FXAI_UpdateMoveEMA(fallback_move_ema, fallback_move_ready, samples[i].move_points, 0.08);
   }

   if(total_stats.trades <= 0) return -1e9;
   trades_out = total_stats.trades;
   for(int r=0; r<FXAI_REGIME_COUNT; r++)
   {
      regime_trades[r] = regime_stats[r].trades;
      if(regime_stats[r].trades > 0)
         regime_scores[r] = FXAI_ScoreWarmupBucketStats(regime_stats[r]);
   }
   return FXAI_ScoreWarmupBucketStats(total_stats);
}

double FXAI_ScoreWarmupTrialRouted(const int ai_idx,
                                   CFXAIAIPlugin &plugin,
                                   const FXAIAIHyperParams &hp,
                                   const int horizon_minutes,
                                   const int val_start,
                                   const int val_end,
                                   const double buyThr,
                                   const double sellThr,
                                   const FXAIPreparedSample &samples[],
                                   FXAINormSampleCache &caches[],
                                   int &trades_out,
                                   double &regime_scores[],
                                   int &regime_trades[])
{
   trades_out = 0;
   int n = ArraySize(samples);
   if(n <= 0 || val_end < val_start) return -1e9;

   if(ArraySize(regime_scores) != FXAI_REGIME_COUNT) ArrayResize(regime_scores, FXAI_REGIME_COUNT);
   if(ArraySize(regime_trades) != FXAI_REGIME_COUNT) ArrayResize(regime_trades, FXAI_REGIME_COUNT);
   FXAIWarmupBucketStats regime_stats[FXAI_REGIME_COUNT];
   for(int r=0; r<FXAI_REGIME_COUNT; r++)
   {
      regime_scores[r] = -1e9;
      regime_trades[r] = 0;
      FXAI_ResetWarmupBucketStats(regime_stats[r]);
   }

   int start = val_start;
   int end = val_end;
   if(start < 0) start = 0;
   if(end >= n) end = n - 1;
   if(end < start) return -1e9;

   FXAIWarmupBucketStats total_stats;
   FXAI_ResetWarmupBucketStats(total_stats);
   double fallback_move_ema = 0.0;
   bool fallback_move_ready = false;
   FXAIAIManifestV4 plugin_manifest;
   FXAI_GetPluginManifest(plugin, plugin_manifest);

   int score_h = FXAI_ClampHorizon(horizon_minutes);
   for(int i=end; i>=start; i--)
   {
      if(i < 0 || i >= n) continue;
      if(!samples[i].valid) continue;

      FXAIPreparedSample eval_sample;
      FXAI_GetCachedPreparedSample(ai_idx, samples[i], i, caches, eval_sample);
      if(!eval_sample.valid) continue;

      FXAIAIPredictRequestV4 req;
      FXAI_ClearPredictRequest(req);
      req.valid = eval_sample.valid;
      req.ctx.api_version = FXAI_API_VERSION_V4;
      req.ctx.regime_id = eval_sample.regime_id;
      req.ctx.session_bucket = FXAI_DeriveSessionBucket(eval_sample.sample_time);
      req.ctx.horizon_minutes = score_h;
      req.ctx.feature_schema_id = plugin_manifest.feature_schema_id;
      req.ctx.normalization_method_id = (int)FXAI_GetModelNormMethodRouted(ai_idx,
                                                                           eval_sample.regime_id,
                                                                           score_h);
      req.ctx.sequence_bars = FXAI_GetPluginSequenceBars(plugin, score_h);
      req.ctx.cost_points = eval_sample.cost_points;
      req.ctx.min_move_points = eval_sample.min_move_points;
      req.ctx.point_value = (_Point > 0.0 ? _Point : 1.0);
      req.ctx.sample_time = eval_sample.sample_time;
      for(int k=0; k<FXAI_AI_WEIGHTS; k++)
         req.x[k] = eval_sample.x[k];
      FXAI_BuildPreparedSampleWindowCached(ai_idx, samples, i, caches, req.ctx.sequence_bars, req.x_window, req.window_size);
      FXAI_ApplyFeatureSchemaToInputEx(plugin_manifest.feature_schema_id,
                                       plugin_manifest.feature_groups_mask,
                                       req.ctx.sequence_bars,
                                       req.x_window,
                                       req.window_size,
                                       req.x);

      FXAIAIPredictionV4 pred;
      FXAI_PredictViaV4(plugin, req, hp, pred);

      double probs_eval[3];
      probs_eval[(int)FXAI_LABEL_SELL] = pred.class_probs[(int)FXAI_LABEL_SELL];
      probs_eval[(int)FXAI_LABEL_BUY] = pred.class_probs[(int)FXAI_LABEL_BUY];
      probs_eval[(int)FXAI_LABEL_SKIP] = pred.class_probs[(int)FXAI_LABEL_SKIP];

      double expected_move = pred.move_mean_points;
      if(expected_move <= 0.0 && fallback_move_ready)
         expected_move = MathMax(fallback_move_ema, eval_sample.min_move_points);
      if(expected_move <= 0.0) expected_move = eval_sample.min_move_points;
      if(expected_move <= 0.0) expected_move = 0.10;

      double buyMinProb = buyThr;
      double sellMinProb = 1.0 - sellThr;
      double skipMinProb = 0.55;
      double vol_proxy = 0.0;
      if(FXAI_AI_WEIGHTS > 6) vol_proxy = MathAbs(eval_sample.x[6]);
      FXAI_DeriveAdaptiveThresholds(buyThr,
                                    sellThr,
                                    eval_sample.min_move_points,
                                    expected_move,
                                    vol_proxy,
                                    buyMinProb,
                                    sellMinProb,
                                    skipMinProb);

      int signal = FXAI_ClassSignalFromEV(probs_eval,
                                          buyMinProb,
                                          sellMinProb,
                                          skipMinProb,
                                          expected_move,
                                          eval_sample.min_move_points,
                                          FXAI_Clamp(AI_EVThresholdPoints, 0.0, 100.0));
      if(signal == -1) continue;

      double net_pts = FXAI_RealizedNetPointsForSignal(signal,
                                                       eval_sample.move_points,
                                                       eval_sample.min_move_points,
                                                       score_h);
      FXAI_UpdateWarmupBucketStats(total_stats, net_pts);
      int regime_id = eval_sample.regime_id;
      if(regime_id < 0 || regime_id >= FXAI_REGIME_COUNT) regime_id = 0;
      FXAI_UpdateWarmupBucketStats(regime_stats[regime_id], net_pts);
      FXAI_UpdateMoveEMA(fallback_move_ema, fallback_move_ready, eval_sample.move_points, 0.08);
   }

   if(total_stats.trades <= 0) return -1e9;
   trades_out = total_stats.trades;
   for(int r=0; r<FXAI_REGIME_COUNT; r++)
   {
      regime_trades[r] = regime_stats[r].trades;
      if(regime_stats[r].trades > 0)
         regime_scores[r] = FXAI_ScoreWarmupBucketStats(regime_stats[r]);
   }
   return FXAI_ScoreWarmupBucketStats(total_stats);
}

void FXAI_StoreWarmupBank(const int ai_idx,
                          const int regime_id,
                          const int horizon_minutes,
                          const FXAIAIHyperParams &hp,
                          const double buy_thr,
                          const double sell_thr)
{
   if(ai_idx < 0 || ai_idx >= FXAI_AI_COUNT) return;
   int hslot = FXAI_GetHorizonSlot(horizon_minutes);
   if(hslot < 0 || hslot >= FXAI_MAX_HORIZONS) return;

   if(regime_id < 0)
   {
      g_model_hp_horizon[ai_idx][hslot] = hp;
      g_model_hp_horizon_ready[ai_idx][hslot] = true;
      g_model_buy_thr_horizon[ai_idx][hslot] = buy_thr;
      g_model_sell_thr_horizon[ai_idx][hslot] = sell_thr;
      FXAI_SanitizeThresholdPair(g_model_buy_thr_horizon[ai_idx][hslot],
                                 g_model_sell_thr_horizon[ai_idx][hslot]);
      g_model_thr_horizon_ready[ai_idx][hslot] = true;
   }
   else if(regime_id >= 0 && regime_id < FXAI_REGIME_COUNT)
   {
      g_model_hp_bank[ai_idx][regime_id][hslot] = hp;
      g_model_hp_bank_ready[ai_idx][regime_id][hslot] = true;
      g_model_buy_thr_bank[ai_idx][regime_id][hslot] = buy_thr;
      g_model_sell_thr_bank[ai_idx][regime_id][hslot] = sell_thr;
      FXAI_SanitizeThresholdPair(g_model_buy_thr_bank[ai_idx][regime_id][hslot],
                                 g_model_sell_thr_bank[ai_idx][regime_id][hslot]);
      g_model_thr_bank_ready[ai_idx][regime_id][hslot] = true;
   }
}

void FXAI_WarmupSelectBanksForHorizon(const int H,
                                      const bool primary_horizon,
                                      const int warmup_loops,
                                      const int warmup_folds,
                                      const int warmup_train_epochs,
                                      const int warmup_min_trades,
                                      const int seed,
                                      const datetime bar_time,
                                      const FXAIAIHyperParams &base_hp,
                                      const double base_buy_thr,
                                      const double base_sell_thr,
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
                                      const FXAIPreparedSample &samples[],
                                      FXAINormSampleCache &norm_caches[])
{
   int sample_span = i_end - i_start + 1;
   int fold_len = sample_span / (warmup_folds + 1);
   if(fold_len < 40) fold_len = 40;
   if(fold_len > (sample_span / 2)) fold_len = sample_span / 2;
   if(fold_len < 20) return;

   for(int ai_idx=0; ai_idx<FXAI_AI_COUNT; ai_idx++)
   {
      FXAI_EnsureRoutedNormCachesForSamples(ai_idx,
                                            i_start,
                                            i_end,
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
                                            samples,
                                            norm_caches);

      MathSrand((uint)(seed + (ai_idx + 1) * 104729 + (int)(bar_time % 65521) + H * 97));

      CFXAIAIPlugin *fold_pool[];
      ArrayResize(fold_pool, warmup_folds);
      for(int f=0; f<warmup_folds; f++)
         fold_pool[f] = g_plugins.CreateInstance(ai_idx);

      double best_score = -1e18;
      FXAIAIHyperParams best_hp = base_hp;
      double best_buy_thr = base_buy_thr;
      double best_sell_thr = base_sell_thr;

      double best_regime_score[FXAI_REGIME_COUNT];
      FXAIAIHyperParams best_regime_hp[FXAI_REGIME_COUNT];
      double best_regime_buy[FXAI_REGIME_COUNT];
      double best_regime_sell[FXAI_REGIME_COUNT];
      for(int r=0; r<FXAI_REGIME_COUNT; r++)
      {
         best_regime_score[r] = -1e18;
         best_regime_hp[r] = base_hp;
         best_regime_buy[r] = base_buy_thr;
         best_regime_sell[r] = base_sell_thr;
      }

      for(int loop=0; loop<warmup_loops; loop++)
      {
         FXAIAIHyperParams hp_trial;
         double buy_trial = base_buy_thr;
         double sell_trial = base_sell_thr;
         if(loop == 0)
            hp_trial = base_hp;
         else
         {
            FXAI_SampleModelHyperParams(ai_idx, base_hp, hp_trial);
            FXAI_SampleThresholdPair(base_buy_thr, base_sell_thr, buy_trial, sell_trial);
         }

         double score_sum = 0.0;
         int folds_used = 0;
         int trades_total = 0;
         double regime_score_sum[FXAI_REGIME_COUNT];
         int regime_score_used[FXAI_REGIME_COUNT];
         int regime_trade_total[FXAI_REGIME_COUNT];
         for(int r=0; r<FXAI_REGIME_COUNT; r++)
         {
            regime_score_sum[r] = 0.0;
            regime_score_used[r] = 0;
            regime_trade_total[r] = 0;
         }

         for(int f=0; f<warmup_folds; f++)
         {
            int val_start = i_start + (f * fold_len);
            int val_end = val_start + fold_len - 1;
            if(val_start < i_start) val_start = i_start;
            if(val_end >= i_end) val_end = i_end - 1;
            if(val_end <= val_start) continue;

            int purge = H + 240;
            if(purge < H + 40) purge = H + 40;
            int train_start = val_end + purge + 1;
            int train_end = i_end;
            if(train_end - train_start < 100) continue;

            CFXAIAIPlugin *trial = fold_pool[f];
            if(trial == NULL) continue;
            trial.Reset();
            trial.EnsureInitialized(hp_trial);

            for(int epoch=0; epoch<warmup_train_epochs; epoch++)
            {
               FXAI_TrainModelWindowPreparedRoutedCached(ai_idx,
                                                         *trial,
                                                         train_start,
                                                         train_end,
                                                         1,
                                                         samples,
                                                         norm_caches);
            }

            int trades_fold = 0;
            double regime_scores_fold[];
            int regime_trades_fold[];
            double score_fold = FXAI_ScoreWarmupTrialRouted(ai_idx,
                                                            *trial,
                                                            hp_trial,
                                                            H,
                                                            val_start,
                                                            val_end,
                                                            buy_trial,
                                                            sell_trial,
                                                            samples,
                                                            norm_caches,
                                                            trades_fold,
                                                            regime_scores_fold,
                                                            regime_trades_fold);

            if(score_fold <= -1e8 || trades_fold <= 0) continue;
            score_sum += score_fold;
            trades_total += trades_fold;
            folds_used++;
            for(int r=0; r<FXAI_REGIME_COUNT; r++)
            {
               if(r >= ArraySize(regime_scores_fold) || r >= ArraySize(regime_trades_fold)) continue;
               if(regime_trades_fold[r] <= 0 || regime_scores_fold[r] <= -1e8) continue;
               regime_score_sum[r] += regime_scores_fold[r];
               regime_score_used[r]++;
               regime_trade_total[r] += regime_trades_fold[r];
            }
         }

         if(folds_used <= 0) continue;

         double score = score_sum / (double)folds_used;
         if(trades_total < warmup_min_trades)
         {
            double miss = (double)(warmup_min_trades - trades_total) / (double)warmup_min_trades;
            score -= 1.5 * miss;
         }

         if(score > best_score)
         {
            best_score = score;
            best_hp = hp_trial;
            best_buy_thr = buy_trial;
            best_sell_thr = sell_trial;
         }

         for(int r=0; r<FXAI_REGIME_COUNT; r++)
         {
            if(regime_score_used[r] <= 0) continue;
            int min_regime_trades = warmup_min_trades / 8;
            if(min_regime_trades < 12) min_regime_trades = 12;
            if(regime_trade_total[r] < min_regime_trades) continue;
            double regime_score = regime_score_sum[r] / (double)regime_score_used[r];
            if(regime_score > best_regime_score[r])
            {
               best_regime_score[r] = regime_score;
               best_regime_hp[r] = hp_trial;
               best_regime_buy[r] = buy_trial;
               best_regime_sell[r] = sell_trial;
            }
         }
      }

      for(int f=0; f<warmup_folds; f++)
      {
         if(fold_pool[f] != NULL)
         {
            delete fold_pool[f];
            fold_pool[f] = NULL;
         }
      }

      FXAI_StoreWarmupBank(ai_idx, -1, H, best_hp, best_buy_thr, best_sell_thr);
      for(int r=0; r<FXAI_REGIME_COUNT; r++)
      {
         FXAIAIHyperParams hp_reg = (best_regime_score[r] > -1e17 ? best_regime_hp[r] : best_hp);
         double buy_reg = (best_regime_score[r] > -1e17 ? best_regime_buy[r] : best_buy_thr);
         double sell_reg = (best_regime_score[r] > -1e17 ? best_regime_sell[r] : best_sell_thr);
         FXAI_StoreWarmupBank(ai_idx, r, H, hp_reg, buy_reg, sell_reg);
         if(primary_horizon)
         {
            g_model_buy_thr_regime[ai_idx][r] = buy_reg;
            g_model_sell_thr_regime[ai_idx][r] = sell_reg;
            FXAI_SanitizeThresholdPair(g_model_buy_thr_regime[ai_idx][r],
                                       g_model_sell_thr_regime[ai_idx][r]);
            g_model_thr_regime_ready[ai_idx][r] = true;
         }
      }

      if(primary_horizon)
      {
         g_model_hp[ai_idx] = best_hp;
         g_model_hp_ready[ai_idx] = true;
         g_model_buy_thr[ai_idx] = best_buy_thr;
         g_model_sell_thr[ai_idx] = best_sell_thr;
         FXAI_SanitizeThresholdPair(g_model_buy_thr[ai_idx], g_model_sell_thr[ai_idx]);
         g_model_thr_ready[ai_idx] = true;
      }
   }
}

void FXAI_WarmupPretrainMetaForSamples(const int H,
                                       const int warmup_folds,
                                       const int warmup_train_epochs,
                                       const int i_start,
                                       const int i_end,
                                       const double base_buy_thr,
                                       const double base_sell_thr,
                                       const FXAIPreparedSample &samples[],
                                       FXAINormSampleCache &norm_caches[])
{
   int sample_span = i_end - i_start + 1;
   int fold_len = sample_span / (warmup_folds + 1);
   if(fold_len < 40) fold_len = 40;
   if(fold_len > (sample_span / 2)) fold_len = sample_span / 2;
   if(fold_len < 20) return;

   int warm_epochs = warmup_train_epochs;
   if(warm_epochs < 1) warm_epochs = 1;
   if(warm_epochs > 3) warm_epochs = 3;

   for(int f=0; f<warmup_folds; f++)
   {
      int val_start = i_start + (f * fold_len);
      int val_end = val_start + fold_len - 1;
      if(val_start < i_start) val_start = i_start;
      if(val_end >= i_end) val_end = i_end - 1;
      if(val_end <= val_start) continue;

      int purge = H + 240;
      if(purge < H + 40) purge = H + 40;
      int train_start = val_end + purge + 1;
      int train_end = i_end;
      if(train_end - train_start < 100) continue;

      CFXAIAIPlugin *pool[];
      ArrayResize(pool, FXAI_AI_COUNT);
      for(int ai_idx=0; ai_idx<FXAI_AI_COUNT; ai_idx++)
      {
         pool[ai_idx] = g_plugins.CreateInstance(ai_idx);
         if(pool[ai_idx] == NULL) continue;
         FXAIAIHyperParams hp_init;
         FXAI_GetModelHyperParamsRouted(ai_idx, 0, H, hp_init);
         pool[ai_idx].Reset();
         pool[ai_idx].EnsureInitialized(hp_init);
         FXAI_TrainModelWindowPreparedRoutedCached(ai_idx,
                                                   *pool[ai_idx],
                                                   train_start,
                                                   train_end,
                                                   warm_epochs,
                                                   samples,
                                                   norm_caches);
      }

      double fallback_move_ema = 0.0;
      bool fallback_move_ready = false;
      for(int i=val_end; i>=val_start; i--)
      {
         if(i < 0 || i >= ArraySize(samples)) continue;
         if(!samples[i].valid) continue;

         int regime_id = samples[i].regime_id;
         if(regime_id < 0 || regime_id >= FXAI_REGIME_COUNT) regime_id = 0;
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
         double family_support[FXAI_FAMILY_OTHER + 1];
         for(int fam_i=0; fam_i<=FXAI_FAMILY_OTHER; fam_i++) family_support[fam_i] = 0.0;

         for(int ai_idx=0; ai_idx<FXAI_AI_COUNT; ai_idx++)
         {
            CFXAIAIPlugin *plugin = pool[ai_idx];
            if(plugin == NULL) continue;

            FXAIAIHyperParams hp_model;
            FXAI_GetModelHyperParamsRouted(ai_idx, regime_id, H, hp_model);

            FXAIPreparedSample pred_sample;
            FXAI_GetCachedPreparedSample(ai_idx, samples[i], i, norm_caches, pred_sample);
            FXAIAIPredictRequestV4 req;
            FXAI_ClearPredictRequest(req);
            req.valid = pred_sample.valid;
            req.ctx.api_version = FXAI_API_VERSION_V4;
            req.ctx.regime_id = pred_sample.regime_id;
            req.ctx.session_bucket = FXAI_DeriveSessionBucket(pred_sample.sample_time);
            req.ctx.horizon_minutes = pred_sample.horizon_minutes;
            FXAIAIManifestV4 plugin_manifest;
            FXAI_GetPluginManifest(*plugin, plugin_manifest);
            req.ctx.feature_schema_id = plugin_manifest.feature_schema_id;
            req.ctx.normalization_method_id = (int)FXAI_GetModelNormMethodRouted(ai_idx,
                                                                                 pred_sample.regime_id,
                                                                                 pred_sample.horizon_minutes);
            req.ctx.sequence_bars = FXAI_GetPluginSequenceBars(*plugin, pred_sample.horizon_minutes);
            req.ctx.cost_points = pred_sample.cost_points;
            req.ctx.min_move_points = pred_sample.min_move_points;
            req.ctx.point_value = (_Point > 0.0 ? _Point : 1.0);
            req.ctx.sample_time = pred_sample.sample_time;
            for(int k=0; k<FXAI_AI_WEIGHTS; k++)
               req.x[k] = pred_sample.x[k];
            FXAI_BuildPreparedSampleWindowCached(ai_idx, samples, i, norm_caches, req.ctx.sequence_bars, req.x_window, req.window_size);
            FXAI_ApplyFeatureSchemaToInputEx(plugin_manifest.feature_schema_id,
                                             plugin_manifest.feature_groups_mask,
                                             req.ctx.sequence_bars,
                                             req.x_window,
                                             req.window_size,
                                             req.x);

            FXAIAIPredictionV4 pred;
            FXAI_PredictViaV4(*plugin, req, hp_model, pred);

            double probs_eval[3];
            probs_eval[0] = pred.class_probs[0];
            probs_eval[1] = pred.class_probs[1];
            probs_eval[2] = pred.class_probs[2];
            FXAI_ApplyRegimeCalibration(ai_idx, regime_id, probs_eval);

            double expected_move = pred.move_mean_points;
            if(expected_move <= 0.0 && fallback_move_ready)
               expected_move = MathMax(fallback_move_ema, samples[i].min_move_points);
            if(expected_move <= 0.0) expected_move = samples[i].min_move_points;
            if(expected_move <= 0.0) expected_move = 0.10;

            double modelBuyThr = base_buy_thr;
            double modelSellThr = base_sell_thr;
            FXAI_GetModelThresholds(ai_idx, regime_id, H, base_buy_thr, base_sell_thr, modelBuyThr, modelSellThr);

            double buyMinProb = modelBuyThr;
            double sellMinProb = 1.0 - modelSellThr;
            double skipMinProb = 0.55;
            double vol_proxy = 0.0;
            if(FXAI_AI_WEIGHTS > 6) vol_proxy = MathAbs(pred_sample.x[6]);
            FXAI_DeriveAdaptiveThresholds(modelBuyThr,
                                         modelSellThr,
                                         samples[i].min_move_points,
                                         expected_move,
                                         vol_proxy,
                                         buyMinProb,
                                         sellMinProb,
                                         skipMinProb);

            int signal = FXAI_ClassSignalFromEV(probs_eval,
                                               buyMinProb,
                                               sellMinProb,
                                               skipMinProb,
                                               expected_move,
                                               samples[i].min_move_points,
                                               FXAI_Clamp(AI_EVThresholdPoints, 0.0, 100.0));

            FXAI_UpdateModelReliability(ai_idx,
                                        samples[i].label_class,
                                        signal,
                                        samples[i].move_points,
                                        samples[i].min_move_points,
                                        expected_move,
                                        probs_eval);
            FXAI_UpdateRegimeCalibration(ai_idx, regime_id, samples[i].label_class, probs_eval);
            FXAI_UpdateModelPerformance(ai_idx,
                                        regime_id,
                                        samples[i].label_class,
                                        signal,
                                        samples[i].move_points,
                                        samples[i].min_move_points,
                                        H,
                                        expected_move,
                                        probs_eval);

            double meta_w = FXAI_GetModelMetaScore(ai_idx, regime_id, samples[i].min_move_points);
            if(meta_w <= 0.0) meta_w = 1.0;
            double model_buy_ev = ((2.0 * probs_eval[(int)FXAI_LABEL_BUY]) - 1.0) * expected_move - samples[i].min_move_points;
            double model_sell_ev = ((2.0 * probs_eval[(int)FXAI_LABEL_SELL]) - 1.0) * expected_move - samples[i].min_move_points;
            model_buy_ev = FXAI_Clamp(model_buy_ev, -10.0 * samples[i].min_move_points, 10.0 * samples[i].min_move_points);
            model_sell_ev = FXAI_Clamp(model_sell_ev, -10.0 * samples[i].min_move_points, 10.0 * samples[i].min_move_points);

            ensemble_meta_total += meta_w;
            ensemble_buy_ev_sum += meta_w * model_buy_ev;
            ensemble_sell_ev_sum += meta_w * model_sell_ev;
            ensemble_expected_sum += meta_w * expected_move;
            ensemble_expected_sq_sum += meta_w * expected_move * expected_move;
            ensemble_conf_sum += meta_w * FXAI_Clamp(pred.confidence, 0.0, 1.0);
            ensemble_rel_sum += meta_w * FXAI_Clamp(pred.reliability, 0.0, 1.0);
            ensemble_margin_sum += meta_w * FXAI_Clamp(MathAbs(probs_eval[(int)FXAI_LABEL_BUY] - probs_eval[(int)FXAI_LABEL_SELL]), 0.0, 1.0);
            if(plugin_manifest.family >= 0 && plugin_manifest.family <= FXAI_FAMILY_OTHER)
               family_support[plugin_manifest.family] += meta_w;
            if(signal == 1) ensemble_buy_support += meta_w;
            else if(signal == 0) ensemble_sell_support += meta_w;
            else ensemble_skip_support += meta_w;
         }

         FXAI_UpdateMoveEMA(fallback_move_ema, fallback_move_ready, samples[i].move_points, 0.08);

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
            double feat[FXAI_STACK_FEATS];
            FXAI_StackBuildFeatures(buyPct,
                                    sellPct,
                                    skipPct,
                                    avg_buy_ev,
                                    avg_sell_ev,
                                    samples[i].min_move_points,
                                    avg_expected,
                                    (FXAI_AI_WEIGHTS > 6 ? MathAbs(samples[i].x[6]) : 0.0),
                                    H,
                                    avg_conf,
                                    avg_rel,
                                    move_dispersion,
                                    avg_margin,
                                    active_family_ratio,
                                    dominant_family_ratio,
                                    warm_ctx_strength,
                                    warm_ctx_quality,
                                    feat);
            FXAI_StackUpdate(regime_id, samples[i].label_class, feat, samples[i].sample_weight);
         }
      }

      for(int ai_idx=0; ai_idx<FXAI_AI_COUNT; ai_idx++)
      {
         if(pool[ai_idx] != NULL)
         {
            delete pool[ai_idx];
            pool[ai_idx] = NULL;
         }
      }
   }
}

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

   FXAIDataSnapshot snapshot;
   if(!FXAI_ExportDataSnapshot(symbol, AI_CommissionPerLotSide, AI_CostBufferPoints, snapshot))
      return false;

   MqlRates rates_m1[];
   MqlRates rates_m5[];
   MqlRates rates_m15[];
   MqlRates rates_m30[];
   MqlRates rates_h1[];
   MqlRates rates_ctx_tmp[];

   double open_arr[];
   double high_arr[];
   double low_arr[];
   double close_arr[];
   datetime time_arr[];
   int spread_m1[];
   if(!FXAI_LoadSeriesWithSpread(symbol, needed, rates_m1, close_arr, time_arr, spread_m1))
      return false;

   FXAI_ExtractRatesOHLC(rates_m1, open_arr, high_arr, low_arr, close_arr);

   if(ArraySize(close_arr) < needed || ArraySize(time_arr) < needed)
      return false;

   int needed_m5 = (needed / 5) + 80;
   int needed_m15 = (needed / 15) + 80;
   int needed_m30 = (needed / 30) + 80;
   int needed_h1 = (needed / 60) + 80;
   if(needed_m5 < 220) needed_m5 = 220;
   if(needed_m15 < 220) needed_m15 = 220;
   if(needed_m30 < 220) needed_m30 = 220;
   if(needed_h1 < 220) needed_h1 = 220;

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

   FXAI_LoadSeriesOptionalCached(symbol, PERIOD_M5, needed_m5, rates_m5, close_m5, time_m5);
   FXAI_LoadSeriesOptionalCached(symbol, PERIOD_M15, needed_m15, rates_m15, close_m15, time_m15);
   FXAI_LoadSeriesOptionalCached(symbol, PERIOD_M30, needed_m30, rates_m30, close_m30, time_m30);
   FXAI_LoadSeriesOptionalCached(symbol, PERIOD_H1, needed_h1, rates_h1, close_h1, time_h1);

   int lag_m5 = 2 * PeriodSeconds(PERIOD_M5);
   int lag_m15 = 2 * PeriodSeconds(PERIOD_M15);
   int lag_m30 = 2 * PeriodSeconds(PERIOD_M30);
   int lag_h1 = 2 * PeriodSeconds(PERIOD_H1);
   if(lag_m5 <= 0) lag_m5 = 600;
   if(lag_m15 <= 0) lag_m15 = 1800;
   if(lag_m30 <= 0) lag_m30 = 3600;
   if(lag_h1 <= 0) lag_h1 = 7200;

   FXAI_BuildAlignedIndexMap(time_arr, time_m5, lag_m5, map_m5);
   FXAI_BuildAlignedIndexMap(time_arr, time_m15, lag_m15, map_m15);
   FXAI_BuildAlignedIndexMap(time_arr, time_m30, lag_m30, map_m30);
   FXAI_BuildAlignedIndexMap(time_arr, time_h1, lag_h1, map_h1);

   int ctx_count = ArraySize(g_context_symbols);
   if(ctx_count > FXAI_MAX_CONTEXT_SYMBOLS) ctx_count = FXAI_MAX_CONTEXT_SYMBOLS;
   FXAIContextSeries ctx_series[];
   ArrayResize(ctx_series, ctx_count);
   for(int s=0; s<ctx_count; s++)
   {
      ctx_series[s].loaded = FXAI_LoadSeriesOptionalCached(g_context_symbols[s],
                                                          PERIOD_M1,
                                                          needed,
                                                          rates_ctx_tmp,
                                                          ctx_series[s].close,
                                                          ctx_series[s].time);
   }

   int i_start = max_h;
   int i_end = max_h + warmup_samples - 1;
   int max_valid = needed - FEATURE_LB - 1;
   if(i_end > max_valid) i_end = max_valid;
   if(i_end <= i_start) return false;

   double ctx_mean_arr[];
   double ctx_std_arr[];
   double ctx_up_arr[];
   double ctx_extra_arr[];
   FXAI_PrecomputeDynamicContextAggregates(time_arr,
                                           close_arr,
                                           ctx_series,
                                           ctx_count,
                                           i_end,
                                           ctx_mean_arr,
                                           ctx_std_arr,
                                           ctx_up_arr,
                                           ctx_extra_arr);

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
      for(int ai_idx=0; ai_idx<FXAI_AI_COUNT; ai_idx++)
      {
         int methods[];
         FXAI_BuildNormMethodCandidateList(ai_idx, methods);
         for(int m=0; m<ArraySize(methods); m++)
         {
            if(FXAI_FindNormSampleCache(methods[m], norm_caches) >= 0) continue;
            FXAI_EnsureNormSampleCache(methods[m],
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
                                       norm_caches);
         }
      }

      FXAI_WarmupSelectNormBanksForHorizon(H,
                                           H == base_h,
                                           warmup_train_epochs,
                                           i_start,
                                           i_end,
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

   for(int ai_idx=0; ai_idx<FXAI_AI_COUNT; ai_idx++)
   {
      if(g_plugins.Get(ai_idx) == NULL) continue;
      g_ai_trained[ai_idx] = true;
      g_ai_last_train_bar[ai_idx] = bar_time;
   }

   g_ai_warmup_done = true;
   Print("FXAI warmup completed: symbol=", symbol,
         ", samples=", warmup_samples,
         ", loops=", warmup_loops,
         ", folds=", warmup_folds,
         ", horizons=", ArraySize(horizons));
   return true;
}


#endif // __FXAI_ENGINE_WARMUP_MQH__
