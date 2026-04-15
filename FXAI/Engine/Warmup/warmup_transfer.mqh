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
         int ai_epochs = FXAI_WarmupEpochBudget(ai_idx, H, warm_epochs);
         if(ai_epochs < 1) ai_epochs = 1;
         if(ai_epochs > 6) ai_epochs = 6;
         FXAI_TrainModelWindowPreparedRoutedCached(ai_idx,
                                                   *pool[ai_idx],
                                                   train_start,
                                                   train_end,
                                                   ai_epochs,
                                                   samples,
                                                   norm_caches);
      }

      FXAI_ResetAnalogMemory();
      FXAI_WarmupPrimeAnalogMemorySamples(samples, train_start, train_end);

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
         double shared_window[FXAI_MAX_SEQUENCE_BARS][FXAI_AI_WEIGHTS];
         int shared_window_size = 0;
         int shared_span = FXAI_ContextSequenceSpan(24, H, _Symbol, 8);
         FXAI_BuildPreparedSampleWindow(samples, i, shared_span, shared_window, shared_window_size);
         double shared_transfer_a[];
         FXAI_BuildSharedTransferInputGlobal(samples[i].x,
                                             shared_window,
                                             shared_window_size,
                                             samples[i].domain_hash,
                                             H,
                                             shared_transfer_a);
         FXAIFoundationSignals foundation_sig;
         FXAI_GlobalFoundationPredict(shared_transfer_a,
                                      shared_window,
                                      shared_window_size,
                                      samples[i].domain_hash,
                                      H,
                                      FXAI_DeriveSessionBucket(samples[i].sample_time),
                                      foundation_sig);
         FXAIStudentSignals student_sig;
         FXAI_GlobalStudentPredict(shared_transfer_a,
                                   shared_window,
                                   shared_window_size,
                                   samples[i].domain_hash,
                                   H,
                                   FXAI_DeriveSessionBucket(samples[i].sample_time),
                                   student_sig);
         FXAIAnalogMemoryQuery analog_q;
         FXAI_QueryAnalogMemory(samples[i].x,
                                regime_id,
                                FXAI_DeriveSessionBucket(samples[i].sample_time),
                                H,
                                samples[i].domain_hash,
                                analog_q);

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
            req.ctx.point_value = (pred_sample.point_value > 0.0 ? pred_sample.point_value : (_Point > 0.0 ? _Point : 1.0));
            req.ctx.domain_hash = pred_sample.domain_hash;
            req.ctx.sample_time = pred_sample.sample_time;
            for(int k=0; k<FXAI_AI_WEIGHTS; k++)
               req.x[k] = pred_sample.x[k];
            FXAI_BuildPreparedSampleWindowCached(ai_idx, samples, i, norm_caches, req.ctx.sequence_bars, req.x_window, req.window_size);
            if(!FXAI_NormalizationCoreFinalizePredictRequest(plugin_manifest, req))
               continue;

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
                                        FXAI_DeriveSessionBucket(samples[i].sample_time),
                                        samples[i].label_class,
                                        signal,
                                        samples[i].move_points,
                                        samples[i].min_move_points,
                                        H,
                                        samples[i].spread_stress,
                                        samples[i].path_flags,
                                        expected_move,
                                        probs_eval);

            double meta_w = FXAI_GetModelMetaScore(ai_idx,
                                                   regime_id,
                                                   FXAI_DeriveSessionBucket(samples[i].sample_time),
                                                   H,
                                                   samples[i].min_move_points);
            if(meta_w <= 0.0) meta_w = 1.0;
            double model_buy_ev = ((2.0 * probs_eval[(int)FXAI_LABEL_BUY]) - 1.0) * expected_move - samples[i].min_move_points;
            double model_sell_ev = ((2.0 * probs_eval[(int)FXAI_LABEL_SELL]) - 1.0) * expected_move - samples[i].min_move_points;
            model_buy_ev = FXAI_Clamp(model_buy_ev, -10.0 * samples[i].min_move_points, 10.0 * samples[i].min_move_points);
            model_sell_ev = FXAI_Clamp(model_sell_ev, -10.0 * samples[i].min_move_points, 10.0 * samples[i].min_move_points);
            double mm_meta = MathMax(samples[i].min_move_points, 0.50);
            double ctx_edge_norm = FXAI_Clamp(FXAI_GetModelContextEdge(ai_idx, regime_id, H) / mm_meta, -4.0, 4.0) / 4.0;
            double ctx_regret = FXAI_Clamp(FXAI_GetModelContextRegret(ai_idx, regime_id, H), 0.0, 6.0) / 6.0;
            double global_edge_norm = FXAI_Clamp(FXAI_GetModelRegimeEdge(ai_idx, regime_id) / mm_meta, -4.0, 4.0) / 4.0;
            double port_edge_norm = FXAI_GetModelPortfolioEdgeNorm(ai_idx, mm_meta);
            double port_stability = FXAI_GetModelPortfolioStability(ai_idx);
            double port_corr = FXAI_GetModelPortfolioCorrPenalty(ai_idx);
            double port_div = FXAI_GetModelPortfolioDiversification(ai_idx);
            double ctx_trust = FXAI_GetModelContextTrust(ai_idx, regime_id, H);
            double model_best_edge = MathMax(model_buy_ev, model_sell_ev);

            ensemble_meta_total += meta_w;
            ensemble_buy_ev_sum += meta_w * model_buy_ev;
            ensemble_sell_ev_sum += meta_w * model_sell_ev;
            ensemble_expected_sum += meta_w * expected_move;
            ensemble_expected_sq_sum += meta_w * expected_move * expected_move;
            ensemble_conf_sum += meta_w * FXAI_Clamp(pred.confidence, 0.0, 1.0);
            ensemble_rel_sum += meta_w * FXAI_Clamp(pred.reliability, 0.0, 1.0);
            ensemble_margin_sum += meta_w * FXAI_Clamp(MathAbs(probs_eval[(int)FXAI_LABEL_BUY] - probs_eval[(int)FXAI_LABEL_SELL]), 0.0, 1.0);
            ensemble_hit_time_sum += meta_w * FXAI_Clamp(pred.hit_time_frac, 0.0, 1.0);
            ensemble_path_risk_sum += meta_w * FXAI_Clamp(pred.path_risk, 0.0, 1.0);
            ensemble_fill_risk_sum += meta_w * FXAI_Clamp(pred.fill_risk, 0.0, 1.0);
            ensemble_mfe_ratio_sum += meta_w * FXAI_Clamp(pred.mfe_mean_points / MathMax(expected_move, samples[i].min_move_points), 0.0, 4.0);
            ensemble_mae_ratio_sum += meta_w * FXAI_Clamp(pred.mae_mean_points / MathMax(pred.mfe_mean_points, samples[i].min_move_points), 0.0, 2.0);
            ensemble_ctx_edge_sum += meta_w * ctx_edge_norm;
            ensemble_ctx_regret_sum += meta_w * ctx_regret;
            ensemble_global_edge_sum += meta_w * global_edge_norm;
            ensemble_port_edge_sum += meta_w * port_edge_norm;
            ensemble_port_stability_sum += meta_w * port_stability;
            ensemble_port_corr_sum += meta_w * port_corr;
            ensemble_port_div_sum += meta_w * port_div;
            ensemble_ctx_trust_sum += meta_w * ctx_trust;
            if(plugin_manifest.family >= 0 && plugin_manifest.family <= FXAI_FAMILY_OTHER)
               family_support[plugin_manifest.family] += meta_w;
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
            double avg_hit_time = ensemble_hit_time_sum / ensemble_meta_total;
            double avg_path_risk = ensemble_path_risk_sum / ensemble_meta_total;
            double avg_fill_risk = ensemble_fill_risk_sum / ensemble_meta_total;
            double avg_mfe_ratio = ensemble_mfe_ratio_sum / ensemble_meta_total;
            double avg_mae_ratio = ensemble_mae_ratio_sum / ensemble_meta_total;
            double avg_ctx_edge_norm = ensemble_ctx_edge_sum / ensemble_meta_total;
            double avg_ctx_regret = ensemble_ctx_regret_sum / ensemble_meta_total;
            double avg_global_edge_norm = ensemble_global_edge_sum / ensemble_meta_total;
            double avg_port_edge_norm = ensemble_port_edge_sum / ensemble_meta_total;
            double avg_port_stability = ensemble_port_stability_sum / ensemble_meta_total;
            double avg_port_corr = ensemble_port_corr_sum / ensemble_meta_total;
            double avg_port_div = ensemble_port_div_sum / ensemble_meta_total;
            double avg_ctx_trust = ensemble_ctx_trust_sum / ensemble_meta_total;
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
            double best_counterfactual_edge_norm = 0.0;
            if(best_model_signal_edge > -1e11)
               best_counterfactual_edge_norm = FXAI_Clamp(best_model_signal_edge / MathMax(samples[i].min_move_points, 0.10), -4.0, 4.0) / 4.0;
            double ensemble_vs_best_gap_norm = 0.0;
            if(best_model_signal_edge > -1e11)
               ensemble_vs_best_gap_norm = FXAI_Clamp((MathMax(best_model_signal_edge, 0.0) - MathMax(avg_buy_ev, avg_sell_ev)) / MathMax(samples[i].min_move_points, 0.10), 0.0, 4.0) / 4.0;
            double best_model_share = FXAI_Clamp(best_model_meta_w / MathMax(ensemble_meta_total, 1e-6), 0.0, 1.0);
            double best_buy_share = FXAI_Clamp(best_buy_meta_w / MathMax(ensemble_meta_total, 1e-6), 0.0, 1.0);
            double best_sell_share = FXAI_Clamp(best_sell_meta_w / MathMax(ensemble_meta_total, 1e-6), 0.0, 1.0);
            double vote_probs[3];
            vote_probs[(int)FXAI_LABEL_SELL] = FXAI_Clamp(ensemble_sell_support / ensemble_meta_total, 0.0, 1.0);
            vote_probs[(int)FXAI_LABEL_BUY] = FXAI_Clamp(ensemble_buy_support / ensemble_meta_total, 0.0, 1.0);
            vote_probs[(int)FXAI_LABEL_SKIP] = FXAI_Clamp(ensemble_skip_support / ensemble_meta_total, 0.0, 1.0);
            double vote_sum = vote_probs[0] + vote_probs[1] + vote_probs[2];
            if(vote_sum <= 0.0) vote_sum = 1.0;
            vote_probs[0] /= vote_sum;
            vote_probs[1] /= vote_sum;
            vote_probs[2] /= vote_sum;
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
            FXAIHierarchicalSignals hierarchy_sig;
            FXAI_BuildHierarchicalSignals(vote_probs,
                                          avg_expected,
                                          samples[i].min_move_points,
                                          avg_conf,
                                          avg_rel,
                                          avg_path_risk,
                                          avg_fill_risk,
                                          avg_hit_time,
                                          warm_ctx_quality,
                                          H,
                                          foundation_sig,
                                          student_sig,
                                          analog_q,
                                          hierarchy_sig);
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
                                    avg_hit_time,
                                    avg_path_risk,
                                    avg_fill_risk,
                                    avg_mfe_ratio,
                                    avg_mae_ratio,
                                    avg_ctx_edge_norm,
                                    avg_ctx_regret,
                                    avg_global_edge_norm,
                                    best_counterfactual_edge_norm,
                                    ensemble_vs_best_gap_norm,
                                    avg_port_edge_norm,
                                    avg_port_stability,
                                    avg_port_corr,
                                    avg_port_div,
                                    best_model_share,
                                    best_buy_share,
                                    best_sell_share,
                                    avg_ctx_trust,
                                    foundation_sig.trust,
                                    foundation_sig.direction_bias,
                                    foundation_sig.move_ratio,
                                    student_sig.trust,
                                    student_sig.tradability,
                                    analog_q.similarity,
                                    analog_q.edge_norm,
                                    analog_q.quality,
                                    hierarchy_sig.consistency,
                                    hierarchy_sig.tradability,
                                    hierarchy_sig.execution_viability,
                                    hierarchy_sig.horizon_fit,
                                    feat);
            double stack_pred_probs[3];
            FXAI_StackPredict(regime_id, H, feat, stack_pred_probs);
            double teacher_probs[3];
            for(int c=0; c<3; c++)
               teacher_probs[c] = FXAI_Clamp(0.55 * stack_pred_probs[c] + 0.45 * vote_probs[c], 0.0005, 0.9990);
            double teacher_sum = teacher_probs[0] + teacher_probs[1] + teacher_probs[2];
            if(teacher_sum <= 0.0) teacher_sum = 1.0;
            teacher_probs[0] /= teacher_sum;
            teacher_probs[1] /= teacher_sum;
            teacher_probs[2] /= teacher_sum;
            FXAI_GlobalStudentUpdate(shared_transfer_a,
                                     shared_window,
                                     shared_window_size,
                                     samples[i].domain_hash,
                                     H,
                                     FXAI_DeriveSessionBucket(samples[i].sample_time),
                                     teacher_probs,
                                     FXAI_Clamp(avg_expected / MathMax(samples[i].min_move_points, 0.10), 0.05, 4.0),
                                     hierarchy_sig.tradability,
                                     hierarchy_sig.horizon_fit,
                                     samples[i].sample_weight,
                                     0.012);
            FXAI_StackUpdate(regime_id, samples[i].label_class, feat, samples[i].sample_weight);
            double realized_edge = 0.0;
            if(samples[i].label_class == (int)FXAI_LABEL_BUY)
               realized_edge = samples[i].move_points - samples[i].min_move_points;
            else if(samples[i].label_class == (int)FXAI_LABEL_SELL)
               realized_edge = -samples[i].move_points - samples[i].min_move_points;
            else
               realized_edge = -MathMax(MathAbs(samples[i].move_points) - samples[i].min_move_points, 0.0);
            FXAI_StackRouterObserve(regime_id,
                                    H,
                                    samples[i].label_class,
                                    realized_edge,
                                    samples[i].quality_score,
                                    feat,
                                    stack_pred_probs,
                                    samples[i].sample_weight);
            bool trade_target = (samples[i].label_class != (int)FXAI_LABEL_SKIP &&
                                 realized_edge > 0.0 &&
                                 samples[i].quality_score > 0.70 &&
                                 samples[i].time_to_hit_frac < 0.95);
            double edge_ratio = realized_edge / MathMax(samples[i].min_move_points, 0.50);
            double oof_score = (samples[i].label_class == (int)FXAI_LABEL_SKIP
                                ? -0.25
                                : samples[i].quality_score * edge_ratio);
            FXAI_UpdateOOFHorizonPriors(regime_id,
                                        H,
                                        oof_score,
                                        edge_ratio,
                                        samples[i].quality_score,
                                        trade_target);
            FXAI_TradeGateUpdate(regime_id, trade_target, feat, samples[i].sample_weight);
            FXAI_GlobalFoundationUpdate(shared_transfer_a,
                                        shared_window,
                                        shared_window_size,
                                        samples[i].domain_hash,
                                        H,
                                        FXAI_DeriveSessionBucket(samples[i].sample_time),
                                        samples[i].masked_step_target,
                                        samples[i].next_vol_target,
                                        samples[i].regime_shift_target,
                                        samples[i].context_lead_target,
                                        samples[i].sample_weight,
                                        0.012);
            FXAI_UpdateAnalogMemory(samples[i].x,
                                    regime_id,
                                    FXAI_DeriveSessionBucket(samples[i].sample_time),
                                    H,
                                    samples[i].domain_hash,
                                    samples[i].move_points,
                                    samples[i].min_move_points,
                                    samples[i].quality_score,
                                    FXAI_Clamp(0.50 * samples[i].trace_reversal_ratio + 0.50 * samples[i].trace_gap_ratio, 0.0, 1.0),
                                    FXAI_Clamp(samples[i].trace_spread_mean_ratio / 2.0, 0.0, 1.0),
                                    samples[i].sample_time,
                                    samples[i].sample_weight);
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

void FXAI_WarmupPrimeFeatureDriftBaseline(const FXAIPreparedSample &samples[])
{
   int n = ArraySize(samples);
   for(int i=0; i<n; i++)
   {
      if(!samples[i].valid)
         continue;
      FXAI_UpdateFeatureDriftBaselineFromInput(samples[i].x);
   }
}

void FXAI_WarmupPrimeAnalogMemorySamples(const FXAIPreparedSample &samples[],
                                         const int start_idx,
                                         const int end_idx)
{
   int n = ArraySize(samples);
   if(n <= 0)
      return;

   int start = start_idx;
   int stop = end_idx;
   if(start < 0) start = 0;
   if(stop >= n) stop = n - 1;
   if(stop < start)
      return;

   for(int i=stop; i>=start; i--)
   {
      if(!samples[i].valid)
         continue;
      FXAI_UpdateAnalogMemory(samples[i].x,
                              samples[i].regime_id,
                              FXAI_DeriveSessionBucket(samples[i].sample_time),
                              samples[i].horizon_minutes,
                              samples[i].domain_hash,
                              samples[i].move_points,
                              samples[i].min_move_points,
                              samples[i].quality_score,
                              FXAI_Clamp(0.50 * samples[i].trace_reversal_ratio + 0.50 * samples[i].trace_gap_ratio, 0.0, 1.0),
                              FXAI_Clamp(samples[i].trace_spread_mean_ratio / 2.0, 0.0, 1.0),
                              samples[i].sample_time,
                              samples[i].sample_weight);
   }
}

void FXAI_WarmupPretrainSharedTransferSamples(const FXAIPreparedSample &samples[],
                                              const int sample_cap,
                                              const double sample_scale,
                                              CFXAIAIPlugin &plugin,
                                              const FXAIAIHyperParams &hp)
{
   int n = ArraySize(samples);
   if(n <= 0 || sample_cap <= 0)
      return;

   int valid_total = 0;
   for(int i=0; i<n; i++)
      if(samples[i].valid)
         valid_total++;
   if(valid_total <= 0)
      return;

   int stride = 1;
   if(valid_total > sample_cap)
      stride = MathMax(1, valid_total / sample_cap);

   int emitted = 0;
   int seen = 0;
   for(int i=n - 1; i>=0; i--)
   {
      if(!samples[i].valid)
         continue;
      if((seen % stride) != 0)
      {
         seen++;
         continue;
      }
      seen++;

      FXAIAIContextV4 ctx;
      FXAI_ClearContextV4(ctx);
      double x_window[FXAI_MAX_SEQUENCE_BARS][FXAI_AI_WEIGHTS];
      int window_size = 0;
      int sequence_span = FXAI_ContextSequenceSpan(24, samples[i].horizon_minutes, _Symbol, 8);
      FXAI_BuildPreparedSampleWindow(samples, i, sequence_span, x_window, window_size);
      ctx.api_version = FXAI_API_VERSION_V4;
      ctx.regime_id = samples[i].regime_id;
      ctx.session_bucket = FXAI_DeriveSessionBucket(samples[i].sample_time);
      ctx.horizon_minutes = samples[i].horizon_minutes;
      ctx.feature_schema_id = (int)FXAI_SCHEMA_FULL;
      ctx.normalization_method_id = (int)FXAI_NORM_EXISTING;
      ctx.sequence_bars = MathMax(window_size, 1);
      ctx.cost_points = samples[i].cost_points;
      ctx.min_move_points = samples[i].min_move_points;
      ctx.point_value = (samples[i].point_value > 0.0 ? samples[i].point_value : (_Point > 0.0 ? _Point : 1.0));
      ctx.domain_hash = samples[i].domain_hash;
      ctx.sample_time = samples[i].sample_time;

      double sample_w = FXAI_Clamp(samples[i].sample_weight * sample_scale, 0.20, 4.00);
      plugin.TrainSharedTransfer(ctx,
                                 samples[i].x,
                                 x_window,
                                 window_size,
                                 samples[i].move_points,
                                 sample_w,
                                 hp.lr);
      emitted++;
      if(emitted >= sample_cap)
         break;
   }
}

void FXAI_WarmupPretrainGlobalTransferSamples(const FXAIPreparedSample &samples[],
                                              const int sample_cap,
                                              const double sample_scale,
                                              const double lr)
{
   int n = ArraySize(samples);
   if(n <= 0 || sample_cap <= 0)
      return;

   int valid_total = 0;
   for(int i=0; i<n; i++)
      if(samples[i].valid)
         valid_total++;
   if(valid_total <= 0)
      return;

   int stride = 1;
   if(valid_total > sample_cap)
      stride = MathMax(1, valid_total / sample_cap);

   int emitted = 0;
   int seen = 0;
   for(int i=n - 1; i>=0; i--)
   {
      if(!samples[i].valid)
         continue;
      if((seen % stride) != 0)
      {
         seen++;
         continue;
      }
      seen++;

      double a[];
      double x_window[FXAI_MAX_SEQUENCE_BARS][FXAI_AI_WEIGHTS];
      int window_size = 0;
      int sequence_span = FXAI_ContextSequenceSpan(24, samples[i].horizon_minutes, _Symbol, 8);
      FXAI_BuildPreparedSampleWindow(samples, i, sequence_span, x_window, window_size);
      FXAI_BuildSharedTransferInputGlobal(samples[i].x,
                                          x_window,
                                          window_size,
                                          samples[i].domain_hash,
                                          samples[i].horizon_minutes,
                                          a);
      double sample_w = FXAI_Clamp(samples[i].sample_weight * sample_scale, 0.20, 4.00);
      FXAI_GlobalSharedTransferUpdate(a,
                                      x_window,
                                      window_size,
                                      samples[i].domain_hash,
                                      samples[i].horizon_minutes,
                                      FXAI_DeriveSessionBucket(samples[i].sample_time),
                                      samples[i].label_class,
                                      samples[i].cost_points,
                                      samples[i].move_points,
                                      sample_w,
                                      lr);
      FXAI_GlobalFoundationUpdate(a,
                                  x_window,
                                  window_size,
                                  samples[i].domain_hash,
                                  samples[i].horizon_minutes,
                                  FXAI_DeriveSessionBucket(samples[i].sample_time),
                                  samples[i].masked_step_target,
                                  samples[i].next_vol_target,
                                  samples[i].regime_shift_target,
                                  samples[i].context_lead_target,
                                  sample_w,
                                  lr);
      emitted++;
      if(emitted >= sample_cap)
         break;
   }

   if(emitted > 0)
      FXAI_MarkRuntimeArtifactsDirty();
}

void FXAI_WarmupBuildTransferUniverse(const string main_symbol,
                                      string &symbols[])
{
   ArrayResize(symbols, 0);

   string seed_symbols[] = {"EURUSD", "USDJPY", "GBPUSD", "AUDUSD", "USDCAD", "USDCHF", "NZDUSD", "EURJPY", "EURGBP", "EURAUD"};
   int total = 0;
   if(StringLen(main_symbol) > 0)
   {
      ArrayResize(symbols, 1);
      symbols[0] = main_symbol;
      total = 1;
   }

   for(int i=0; i<ArraySize(g_context_symbols); i++)
   {
      string candidate = g_context_symbols[i];
      if(StringLen(candidate) <= 0)
         continue;
      bool exists = false;
      for(int j=0; j<total; j++)
      {
         if(symbols[j] == candidate)
         {
            exists = true;
            break;
         }
      }
      if(exists)
         continue;
      ArrayResize(symbols, total + 1);
      symbols[total] = candidate;
      total++;
   }

   for(int i=0; i<ArraySize(seed_symbols); i++)
   {
      string candidate = seed_symbols[i];
      if(StringLen(candidate) <= 0)
         continue;
      bool exists = false;
      for(int j=0; j<total; j++)
      {
         if(symbols[j] == candidate)
         {
            exists = true;
            break;
         }
      }
      if(exists)
         continue;
      ArrayResize(symbols, total + 1);
      symbols[total] = candidate;
      total++;
   }
}

bool FXAI_WarmupBuildTransferSymbolSamplesForHorizon(const string target_symbol,
                                                     const string main_symbol,
                                                     const int needed,
                                                     const int max_h,
                                                     const int sample_cap,
                                                     const int horizon_minutes,
                                                     const double commission_per_lot_side,
                                                     const double cost_buffer_points,
                                                     const double ev_threshold_points,
                                                     FXAIPreparedSample &out[])
{
   ArrayResize(out, 0);
   if(StringLen(target_symbol) <= 0 || sample_cap <= 0)
      return false;

   const int FEATURE_LB = 10;
   FXAIDataCoreRequest data_request;
   FXAI_DataCoreInitRequest(data_request,
                            false,
                            target_symbol,
                            0,
                            needed,
                            needed - 1,
                            commission_per_lot_side,
                            cost_buffer_points);
   if(StringLen(main_symbol) > 0 && main_symbol != target_symbol)
      FXAI_DataCoreAddContextSymbol(data_request, main_symbol);
   FXAI_DataCoreCaptureGlobalContextSymbols(data_request);

   FXAIDataCoreBundle transfer_bundle;
   string transfer_reason = "";
   if(!FXAI_DataCoreLoadBundleFromRequest(data_request, transfer_bundle, transfer_reason))
      return false;

   int i_start = max_h;
   int i_end = i_start + sample_cap - 1;
   int max_valid = needed - FEATURE_LB - 1;
   if(i_end > max_valid)
      i_end = max_valid;
   if(i_end <= i_start)
      return false;

   FXAI_PrecomputeTrainingSamples(i_start,
                                  i_end,
                                  horizon_minutes,
                                  transfer_bundle.snapshot.commission_points,
                                  cost_buffer_points,
                                  ev_threshold_points,
                                  transfer_bundle.snapshot,
                                  transfer_bundle.spread_m1,
                                  transfer_bundle.time_arr,
                                  transfer_bundle.open_arr,
                                  transfer_bundle.high_arr,
                                  transfer_bundle.low_arr,
                                  transfer_bundle.close_arr,
                                  transfer_bundle.time_m5,
                                  transfer_bundle.close_m5,
                                  transfer_bundle.map_m5,
                                  transfer_bundle.time_m15,
                                  transfer_bundle.close_m15,
                                  transfer_bundle.map_m15,
                                  transfer_bundle.time_m30,
                                  transfer_bundle.close_m30,
                                  transfer_bundle.map_m30,
                                  transfer_bundle.time_h1,
                                  transfer_bundle.close_h1,
                                  transfer_bundle.map_h1,
                                  transfer_bundle.ctx_mean_arr,
                                  transfer_bundle.ctx_std_arr,
                                  transfer_bundle.ctx_up_arr,
                                  transfer_bundle.ctx_extra_arr,
                                  -1,
                                  out);

   return (ArraySize(out) > 0);
}
