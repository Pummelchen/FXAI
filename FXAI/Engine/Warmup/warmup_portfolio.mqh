double FXAI_WarmupEstimatePortfolioSymbolCorrelation(const FXAIPreparedSample &samples[])
{
   int n = ArraySize(samples);
   double corr_sum = 0.0;
   int used = 0;
   for(int i=0; i<n; i++)
   {
      if(!samples[i].valid)
         continue;
      corr_sum += MathAbs(FXAI_GetInputFeature(samples[i].x, 53));
      used++;
   }
   if(used <= 0)
      return 0.0;
   return FXAI_Clamp(corr_sum / (double)used, 0.0, 1.0);
}

double FXAI_WarmupEvaluatePortfolioSymbol(CFXAIAIPlugin &plugin,
                                          const int ai_idx,
                                          const int horizon_minutes,
                                          const FXAIPreparedSample &samples[],
                                          const int sample_cap,
                                          double &trade_rate_out)
{
   trade_rate_out = 0.0;
   int n = ArraySize(samples);
   if(n <= 0 || sample_cap <= 0)
      return 0.0;

   FXAIAIManifestV4 manifest;
   FXAI_GetPluginManifest(plugin, manifest);
   FXAIAIHyperParams hp_model;
   FXAI_GetModelHyperParamsRouted(ai_idx, 0, horizon_minutes, hp_model);

   int valid_total = 0;
   for(int i=0; i<n; i++)
      if(samples[i].valid)
         valid_total++;
   if(valid_total <= 0)
      return 0.0;

   int stride = 1;
   if(valid_total > sample_cap)
      stride = MathMax(1, valid_total / sample_cap);

   double edge_sum = 0.0;
   double weight_sum = 0.0;
   int trades = 0;
   int evals = 0;
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

      FXAIAIPredictRequestV4 req;
      FXAI_ClearPredictRequest(req);
      req.valid = true;
      req.ctx.api_version = FXAI_API_VERSION_V4;
      req.ctx.regime_id = samples[i].regime_id;
      req.ctx.session_bucket = FXAI_DeriveSessionBucket(samples[i].sample_time);
      req.ctx.horizon_minutes = samples[i].horizon_minutes;
      req.ctx.feature_schema_id = manifest.feature_schema_id;
      req.ctx.normalization_method_id = (int)FXAI_NORM_EXISTING;
      req.ctx.sequence_bars = FXAI_GetPluginSequenceBars(plugin, samples[i].horizon_minutes);
      req.ctx.cost_points = samples[i].cost_points;
      req.ctx.min_move_points = samples[i].min_move_points;
      req.ctx.point_value = (samples[i].point_value > 0.0 ? samples[i].point_value : (_Point > 0.0 ? _Point : 1.0));
      req.ctx.domain_hash = samples[i].domain_hash;
      req.ctx.sample_time = samples[i].sample_time;
      for(int k=0; k<FXAI_AI_WEIGHTS; k++)
         req.x[k] = samples[i].x[k];
      FXAI_BuildPreparedSampleWindow(samples, i, req.ctx.sequence_bars, req.x_window, req.window_size);
      FXAI_ApplyFeatureSchemaToPayloadEx(manifest.feature_schema_id,
                                         manifest.feature_groups_mask,
                                         req.ctx.sequence_bars,
                                         req.x_window,
                                         req.window_size,
                                         req.x);

      FXAIAIPredictionV4 pred;
      if(!FXAI_PredictViaV4(plugin, req, hp_model, pred))
         continue;

      double probs_eval[3];
      probs_eval[0] = pred.class_probs[0];
      probs_eval[1] = pred.class_probs[1];
      probs_eval[2] = pred.class_probs[2];
      FXAI_ApplyRegimeCalibration(ai_idx, samples[i].regime_id, probs_eval);

      double expected_move = pred.move_mean_points;
      if(expected_move <= 0.0)
         expected_move = MathMax(samples[i].min_move_points, 0.10);

      double modelBuyThr = AI_BuyThreshold;
      double modelSellThr = AI_SellThreshold;
      FXAI_GetModelThresholds(ai_idx,
                              samples[i].regime_id,
                              samples[i].horizon_minutes,
                              AI_BuyThreshold,
                              AI_SellThreshold,
                              modelBuyThr,
                              modelSellThr);
      double buyMinProb = modelBuyThr;
      double sellMinProb = 1.0 - modelSellThr;
      double skipMinProb = 0.55;
      double vol_proxy = MathAbs(FXAI_GetInputFeature(samples[i].x, 5));
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

      double realized_edge = 0.0;
      if(signal == 1)
         realized_edge = samples[i].move_points - samples[i].min_move_points;
      else if(signal == 0)
         realized_edge = -samples[i].move_points - samples[i].min_move_points;
      else
         realized_edge = -0.25 * MathMax(MathAbs(samples[i].move_points) - samples[i].min_move_points, 0.0);

      double quality = 1.0 +
                       0.20 * (1.0 - FXAI_Clamp(samples[i].time_to_hit_frac, 0.0, 1.0)) -
                       0.08 * FXAI_Clamp(pred.path_risk, 0.0, 1.0) -
                       0.06 * FXAI_Clamp(pred.fill_risk, 0.0, 1.0);
      double sw = FXAI_Clamp(samples[i].sample_weight * quality, 0.20, 4.00);
      edge_sum += sw * realized_edge;
      weight_sum += sw;
      if(signal >= 0)
         trades++;
      evals++;
      if(evals >= sample_cap)
         break;
   }

   if(evals <= 0 || weight_sum <= 0.0)
      return 0.0;
   trade_rate_out = FXAI_Clamp((double)trades / (double)MathMax(evals, 1), 0.0, 1.0);
   return edge_sum / weight_sum;
}

void FXAI_WarmupBuildPortfolioDiagnostics(const string main_symbol,
                                          const int needed,
                                          const int max_h,
                                          const int horizon_minutes,
                                          const double commission_per_lot_side,
                                          const double cost_buffer_points,
                                          const double ev_threshold_points,
                                          const int eval_cap,
                                          const FXAIPreparedSample &primary_samples[])
{
   if(ArraySize(primary_samples) <= 0)
      return;

   double score_sum[FXAI_AI_COUNT];
   double score_sq_sum[FXAI_AI_COUNT];
   double weight_sum[FXAI_AI_COUNT];
   double corr_sum[FXAI_AI_COUNT];
   double div_sum[FXAI_AI_COUNT];
   int symbol_count[FXAI_AI_COUNT];
   for(int ai=0; ai<FXAI_AI_COUNT; ai++)
   {
      score_sum[ai] = 0.0;
      score_sq_sum[ai] = 0.0;
      weight_sum[ai] = 0.0;
      corr_sum[ai] = 0.0;
      div_sum[ai] = 0.0;
      symbol_count[ai] = 0;
   }

   for(int ai_idx=0; ai_idx<FXAI_AI_COUNT; ai_idx++)
   {
      CFXAIAIPlugin *runtime = g_plugins.Get(ai_idx);
      if(runtime == NULL)
         continue;
      double trade_rate = 0.0;
      double edge = FXAI_WarmupEvaluatePortfolioSymbol(*runtime,
                                                       ai_idx,
                                                       horizon_minutes,
                                                       primary_samples,
                                                       eval_cap,
                                                       trade_rate);
      double w = FXAI_Clamp(0.70 + 0.30 * trade_rate, 0.35, 1.20);
      score_sum[ai_idx] += w * edge;
      score_sq_sum[ai_idx] += w * edge * edge;
      weight_sum[ai_idx] += w;
      div_sum[ai_idx] += w;
      symbol_count[ai_idx]++;
   }

   string transfer_universe[];
   FXAI_WarmupBuildTransferUniverse(main_symbol, transfer_universe);
   for(int s=0; s<ArraySize(transfer_universe); s++)
   {
      string target_symbol = transfer_universe[s];
      if(StringLen(target_symbol) <= 0 || target_symbol == main_symbol)
         continue;

      FXAIPreparedSample transfer_samples[];
      if(!FXAI_WarmupBuildTransferSymbolSamplesForHorizon(target_symbol,
                                                          main_symbol,
                                                          needed,
                                                          max_h,
                                                          eval_cap,
                                                          horizon_minutes,
                                                          commission_per_lot_side,
                                                          cost_buffer_points,
                                                          ev_threshold_points,
                                                          transfer_samples))
         continue;

      double abs_corr = FXAI_WarmupEstimatePortfolioSymbolCorrelation(transfer_samples);
      double div_weight = FXAI_Clamp(1.0 - 0.60 * abs_corr, 0.25, 1.0);
      for(int ai_idx=0; ai_idx<FXAI_AI_COUNT; ai_idx++)
      {
         CFXAIAIPlugin *runtime = g_plugins.Get(ai_idx);
         if(runtime == NULL)
            continue;
         double trade_rate = 0.0;
         double edge = FXAI_WarmupEvaluatePortfolioSymbol(*runtime,
                                                          ai_idx,
                                                          horizon_minutes,
                                                          transfer_samples,
                                                          MathMax(48, eval_cap / 2),
                                                          trade_rate);
         double w = FXAI_Clamp(div_weight * (0.55 + 0.45 * trade_rate), 0.15, 1.10);
         score_sum[ai_idx] += w * edge;
         score_sq_sum[ai_idx] += w * edge * edge;
         weight_sum[ai_idx] += w;
         corr_sum[ai_idx] += w * abs_corr;
         div_sum[ai_idx] += w * div_weight;
         symbol_count[ai_idx]++;
      }
   }

   for(int ai_idx=0; ai_idx<FXAI_AI_COUNT; ai_idx++)
   {
      if(weight_sum[ai_idx] <= 0.0)
         continue;
      double mean_edge = score_sum[ai_idx] / weight_sum[ai_idx];
      double var_edge = MathMax(score_sq_sum[ai_idx] / weight_sum[ai_idx] - mean_edge * mean_edge, 0.0);
      double std_edge = MathSqrt(var_edge);
      double scale = MathMax(MathAbs(mean_edge), 0.50);
      double stability = 1.0 - FXAI_Clamp(std_edge / scale, 0.0, 1.0);
      double corr_penalty = (corr_sum[ai_idx] > 0.0 ? corr_sum[ai_idx] / weight_sum[ai_idx] : 0.0);
      double diversification = FXAI_Clamp(div_sum[ai_idx] / weight_sum[ai_idx], 0.0, 1.0);
      FXAI_SetModelPortfolioDiagnostics(ai_idx,
                                        mean_edge,
                                        stability,
                                        corr_penalty,
                                        diversification,
                                        symbol_count[ai_idx]);
   }
}

