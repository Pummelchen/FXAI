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
      req.ctx.point_value = (samples[i].point_value > 0.0 ? samples[i].point_value : (_Point > 0.0 ? _Point : 1.0));
      req.ctx.domain_hash = samples[i].domain_hash;
      req.ctx.sample_time = samples[i].sample_time;
      for(int k=0; k<FXAI_AI_WEIGHTS; k++)
         req.x[k] = samples[i].x[k];
      FXAI_BuildPreparedSampleWindow(samples, i, req.ctx.sequence_bars, req.x_window, req.window_size);
      FXAI_ApplyPayloadTransformPipelineEx(plugin_manifest.feature_schema_id,
                                           plugin_manifest.feature_groups_mask,
                                           req.ctx.normalization_method_id,
                                           req.ctx.horizon_minutes,
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

      FXAIExecutionTraceStats trace;
      FXAI_LoadExecutionTraceFromSample(samples[i], trace);
      double net_pts = FXAI_RealizedNetPointsForSignalReplayTrace(signal,
                                                                  samples[i].move_points,
                                                                  samples[i].min_move_points,
                                                                  score_h,
                                                                  samples[i].spread_stress,
                                                                  samples[i].path_flags,
                                                                  trace,
                                                                  samples[i].sample_time,
                                                                  0);
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
      req.ctx.point_value = (eval_sample.point_value > 0.0 ? eval_sample.point_value : (_Point > 0.0 ? _Point : 1.0));
      req.ctx.domain_hash = eval_sample.domain_hash;
      req.ctx.sample_time = eval_sample.sample_time;
      for(int k=0; k<FXAI_AI_WEIGHTS; k++)
         req.x[k] = eval_sample.x[k];
      FXAI_BuildPreparedSampleWindowCached(ai_idx, samples, i, caches, req.ctx.sequence_bars, req.x_window, req.window_size);
      FXAI_ApplyPayloadTransformPipelineEx(plugin_manifest.feature_schema_id,
                                           plugin_manifest.feature_groups_mask,
                                           req.ctx.normalization_method_id,
                                           req.ctx.horizon_minutes,
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

      FXAIExecutionTraceStats trace;
      FXAI_LoadExecutionTraceFromSample(eval_sample, trace);
      double net_pts = FXAI_RealizedNetPointsForSignalReplayTrace(signal,
                                                                  eval_sample.move_points,
                                                                  eval_sample.min_move_points,
                                                                  score_h,
                                                                  eval_sample.spread_stress,
                                                                  eval_sample.path_flags,
                                                                  trace,
                                                                  eval_sample.sample_time,
                                                                  0);
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

      FXAI_SetRandomSeed((ulong)(seed + (ulong)((ai_idx + 1) * 104729) + (ulong)((int)(bar_time % 65521)) + (ulong)(H * 97) + 1));

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

            int epoch_budget = FXAI_WarmupEpochBudget(ai_idx, H, warmup_train_epochs);
            if(epoch_budget < 1) epoch_budget = 1;
            if(epoch_budget > 10) epoch_budget = 10;
            int patience = (FXAI_IsSeriousNativeAI(ai_idx) ? 3 : 1);
            int trades_fold = 0;
            double regime_scores_fold[];
            int regime_trades_fold[];
            double score_fold = -1e18;
            int stale_epochs = 0;
            for(int epoch=0; epoch<epoch_budget; epoch++)
            {
               FXAI_TrainModelWindowPreparedRoutedCached(ai_idx,
                                                         *trial,
                                                         train_start,
                                                         train_end,
                                                         1,
                                                         samples,
                                                         norm_caches);

               int trades_probe = 0;
               double regime_scores_probe[];
               int regime_trades_probe[];
               double score_probe = FXAI_ScoreWarmupTrialRouted(ai_idx,
                                                                *trial,
                                                                hp_trial,
                                                                H,
                                                                val_start,
                                                                val_end,
                                                                buy_trial,
                                                                sell_trial,
                                                                samples,
                                                                norm_caches,
                                                                trades_probe,
                                                                regime_scores_probe,
                                                                regime_trades_probe);
               if(score_probe > score_fold + 0.05 && trades_probe > 0)
               {
                  score_fold = score_probe;
                  trades_fold = trades_probe;
                  ArrayCopy(regime_scores_fold, regime_scores_probe);
                  ArrayCopy(regime_trades_fold, regime_trades_probe);
                  stale_epochs = 0;
               }
               else
               {
                  stale_epochs++;
               }
               if(stale_epochs >= patience && epoch + 1 >= MathMin(2, epoch_budget))
                  break;
            }

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
         double regime_scores_avg[FXAI_REGIME_COUNT];
         int regime_trades_agg[FXAI_REGIME_COUNT];
         for(int r=0; r<FXAI_REGIME_COUNT; r++)
         {
            regime_scores_avg[r] = (regime_score_used[r] > 0 ? regime_score_sum[r] / (double)regime_score_used[r] : 0.0);
            regime_trades_agg[r] = regime_trade_total[r];
         }
         score += FXAI_WarmupPortfolioObjectiveProxy(score, trades_total, regime_scores_avg, regime_trades_agg);

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
